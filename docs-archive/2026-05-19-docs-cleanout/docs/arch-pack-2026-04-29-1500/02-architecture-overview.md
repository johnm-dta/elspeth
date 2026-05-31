# Architecture Overview

The three architectural commitments that shape every part of ELSPETH.

1. The **4-layer model** — strictly downward import direction across
   `contracts` → `core` → `engine` → application surfaces.
2. The **three-tier trust model** — distinct handling rules for our
   data, pipeline data, and external data.
3. The **SDA execution pattern** — Sense (sources) → Decide (transforms,
   gates, aggregations) → Act (sinks), with audit-grade lineage at
   every boundary.

Each of these is **mechanically enforced**, not aspirational. CI gates,
context-manager invariants, and AST-scanning drift tests are the
discipline; the documentation describes what the mechanism guarantees.

---

## §1 The 4-layer model

ELSPETH partitions all production Python into four layers. Imports
must flow downward only.

```text
L0  contracts/     Leaf — imports nothing above. Shared types, enums, protocols.
L1  core/          Can import L0 only. Landscape, DAG, config, canonical JSON.
L2  engine/        Can import L0, L1. Orchestrator, processors, executors.
L3  plugins/       Can import L0, L1, L2. Sources, transforms, sinks, clients.
    mcp/ tui/ cli* telemetry/ testing/ web/ composer_mcp/   — also L3 (application layer)
```

### How it is enforced

- **Path-based layer assignment** in `scripts/cicd/enforce_tier_model.py:237–248`
  (`LAYER_HIERARCHY` 237–241 + `LAYER_NAMES` 243–248). Anything not
  named is implicitly L3.
- **CI gate** runs `enforce_tier_model.py check` on every commit.
  Upward imports (rule `L1`) are CI failures; TYPE_CHECKING upward
  imports (rule `TC`) are warnings.
- **Allowlist with justification** at `config/cicd/enforce_tier_model/`
  permits per-file and per-finding exemptions for legitimate
  exceptions, each carrying a rationale.

### Status (re-verified at this pack's HEAD)

```
$ enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
No bug-hiding patterns detected. Check passed.
```

The L0 leaf is empirically a leaf: zero outbound edges in the L3 import
oracle. The codebase is layer-conformant today.

### Resolving a new cross-layer need

When a higher layer needs something from a lower-layer module that
also depends upward, the four resolutions are:

1. **Move the code down.** Relocate the needed code to the lower layer
   if it has no upward dependencies of its own.
2. **Extract the primitive.** Pull the type or constant into
   `contracts/` and import from there.
3. **Restructure the caller.** Use dependency injection, callbacks, or
   protocols defined in `contracts/`.
4. **NEVER**: add a lazy import with an apologetic comment. This is the
   "Shifting the Burden" archetype — it defers the structural fix and
   the pattern recurs.

The single open instance of this anti-pattern in the L0 cluster is
`contracts/plugin_context.py:31`, captured as finding K2 / R12.

---

## §2 The three-tier trust model

ELSPETH treats data as belonging to one of three trust tiers, each
with distinct handling rules.

### Tier 1 — Our data (audit DB, checkpoints) · Full trust

Must be 100% pristine at all times. **Bad data crashes immediately.**
No coercion, no defaults, no silent recovery.

If we read garbage from our own database, something catastrophic
happened — a bug in our code, database corruption, or tampering.
Silently coercing the garbage into a valid-looking value would be
evidence tampering: an auditor asking "why did row 42 get routed
here?" would receive a confident wrong answer.

### Tier 2 — Pipeline data (post-source) · Elevated trust ("probably ok")

Type-valid but potentially operation-unsafe. Data that passed source
validation.

- Types are trustworthy (the source validated and/or coerced them).
- Values might still cause operation failures (division by zero,
  invalid date formats, etc.).
- Transforms and sinks **expect conformance** — if types are wrong,
  that is an upstream plugin bug.
- **No coercion** at transform or sink level — if a transform receives
  `"42"` when it expected `int`, that is a bug in the source or upstream
  transform.

### Tier 3 — External data (source input) · Zero trust

Can be literal trash. We do not control what external systems feed us.

- Malformed CSV rows, NULLs everywhere, wrong types, unexpected JSON
  structures.
- **Validate at the boundary, coerce where possible, record what we
  got.**
- **Record what we did not get.** If we expected data and the external
  system did not provide it, that absence is a fact worth recording,
  not a gap to fill with a fabricated default.
- Sources MAY coerce: `"42"` → `42`, `"true"` → `True`. This normalises
  external data while preserving meaning.
- **Coercion is meaning-preserving; fabrication is not.** Inferring a
  missing field from adjacent fields produces a synthetic datum the
  external system never asserted. The audit trail then contains a
  confident answer to a question the source never answered.
- Quarantine rows that cannot be coerced or validated. The audit trail
  records "row 42 was quarantined because field X was NULL" — that is a
  valid audit outcome.

### The trust flow

```text
EXTERNAL DATA              PIPELINE DATA              AUDIT TRAIL
(zero trust)               (elevated trust)           (full trust)

┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ External Source │        │ Transform/Sink  │        │ Landscape DB    │
│                 │        │                 │        │                 │
│ • Coerce OK     │───────►│ • No coercion   │───────►│ • Crash on      │
│ • Validate      │        │ • Expect types  │        │   any anomaly   │
│ • Quarantine    │        │ • Wrap ops on   │        │ • No coercion   │
│   failures      │        │   row values    │        │   ever          │
└─────────────────┘        └─────────────────┘        └─────────────────┘
         │                          │
    Source is the              Operations on row
    ONLY place                 values need wrapping
    coercion is                (values can still
    allowed                    fail at runtime)
```

### How it is enforced

- **CI scanner.** `scripts/cicd/enforce_tier_model.py` (the same script
  that enforces the 4-layer model) detects defensive patterns
  (`getattr` with default, `try/except` swallowing, `hasattr`) and
  flags them. Each finding can be allowlisted with a justification at
  the trust boundary it crosses.
- **Documented identically in every plugin module.** Every source
  repeats "ONLY place coercion is allowed"; every sink repeats "wrong
  types = upstream bug = crash". Repetition is the protocol that
  prevents drift.
- **Open work:** the discipline is honoured today by author discipline;
  CI does not currently catch a violator at runtime. Finding P2 / a
  recommended runtime-probe test suite would mechanise this.

---

## §3 The SDA execution pattern

ELSPETH pipelines compile to DAGs. Linear pipelines are degenerate
DAGs (a single `continue` path).

### The model

```text
SENSE (Sources)  →  DECIDE (Transforms / Gates / Aggregations)  →  ACT (Sinks)
   exactly 1            0 or more, ordered                          1 or more, named
```

| Stage | What it does |
|-------|-------------|
| **Source** | Loads data from an external system (CSV, API, database, message queue). Exactly one per run. The only place where Tier-3-to-Tier-2 coercion is permitted. |
| **Transform** (row) | Processes one row, emits one row. Stateless. |
| **Gate** (transform subtype) | Evaluates a row, decides destination(s) via `continue`, `route_to_sink`, or `fork_to_paths`. |
| **Aggregation** (transform subtype) | Collects N rows until a trigger fires, emits a result. Stateful. |
| **Coalesce** (transform subtype) | Merges results from parallel paths after a fork. |
| **Sink** | Writes results to an external destination. One or more, each named. |

### Token identity through the DAG

Every row carries identity that survives forks and joins:

| Field | Meaning |
|-------|---------|
| `row_id` | Stable source-row identity. Survives every transformation. |
| `token_id` | Instance of a row in a specific DAG path. Forks mint new tokens. |
| `parent_token_id` | Lineage for forks and joins. |

### The terminal-state-per-token invariant

Every row reaches **exactly one** terminal state. The seven terminal
states are: `COMPLETED`, `ROUTED`, `FORKED`, `CONSUMED_IN_BATCH`,
`COALESCED`, `QUARANTINED`, `FAILED`, `EXPANDED`. (`BUFFERED` is
non-terminal — it becomes `COMPLETED` on flush.) No silent drops.

This invariant is **structurally guaranteed**, not conventional. It is
implemented as a context-manager pattern in
`engine/executors/state_guard.py:NodeStateGuard` and locked by
`tests/unit/engine/test_state_guard_audit_evidence_discriminator.py`
and `tests/unit/engine/test_row_outcome.py`. The type system cooperates
with the runtime to make the invariant non-bypassable.

---

## §4 Audit-trail design

The audit trail is the **legal record**. Three principles govern it:

1. **Audit primacy.** Audit fires first (synchronous, crash-on-failure),
   then telemetry (asynchronous, best-effort), then logging (only if
   the audit and telemetry systems are broken).
2. **Hash integrity.** Hashes survive payload deletion. The hash chain
   remains verifiable even after large blobs are purged for retention.
3. **The attributability test.** For any output, the operation
   `explain(recorder, run_id, token_id)` must prove complete lineage
   back to source data, configuration, and code version.

### How the layers cooperate

| Layer | Role |
|-------|------|
| `contracts/` (L0) | Owns the audit DTO vocabulary: `AuditEvidenceBase` ABC, the `@tier_1_error` decorator with frozen registry, the `DeclarationContract` 4-site framework, secret-scrub last-line-of-defence. |
| `core/landscape/` (L1) | Persists the audit trail. The 4-repository facade (`DataFlowRepository`, `ExecutionRepository`, `QueryRepository`, `RunLifecycleRepository`) is the only path callers can take to the audit DB; repositories are not re-exported through `core/__init__.py`. |
| `engine/` (L2) | Encodes the terminal-state-per-token invariant via `NodeStateGuard`. Implements the ADR-010 dispatcher across 4 sites × 7 contract adopters with a single dispatcher and an AST-scanning drift test. |

### The ADR-010 declaration-trust framework

A 4-site dispatcher (`pre_emission_check`, `post_emission_check`,
`batch_flush_check`, `boundary_check`) drives 7 contract adopters
mapped 1:1 to ADRs 007 / 008 / 011 / 012 / 013 / 014 / 016 / 017. The
mapping is locked by the AST-scanning unit test
`tests/unit/engine/test_declaration_contract_bootstrap_drift.py`:
adding a new adopter without registering it fails CI.

The dispatcher's **audit-complete posture** — every contract's
violations are aggregated before raising, rather than first-fire
short-circuiting — is documented inline at
`src/elspeth/engine/executors/declaration_dispatch.py:1-26` and
verified at lines 120–172 with 1,923 LOC of dedicated test coverage
across unit, property, and integration tiers.

---

## §5 Plugin ownership: system code, not user code

All plugins (sources, transforms, aggregations, sinks) are
**system-owned code**, not user-provided extensions. Gates are
config-driven system operations, not plugins.

ELSPETH uses `pluggy` for clean architecture, **not** to accept
arbitrary user plugins. Plugins are developed, tested, and deployed as
part of ELSPETH with the same rigour as engine code.

### Implications for error handling

| Scenario | Correct response | Wrong response |
|----------|------------------|----------------|
| Plugin method throws an exception | **CRASH** — bug in our code | Catch and log silently |
| Plugin returns wrong type | **CRASH** — bug in our code | Coerce to expected type |
| Plugin missing expected attribute | **CRASH** — interface violation | `getattr(x, 'attr', default)` |
| User data has wrong type | Quarantine row, continue | Crash the pipeline |
| User data missing field | Quarantine row, continue | Crash the pipeline |

A defective plugin that silently produces wrong results is **worse
than a crash**. A crash stops the pipeline; an operator investigates;
the bug is fixed. A silent wrong result flows through, gets recorded
as "correct", auditors see garbage, trust is destroyed.

---

## §6 Configuration & dependency precedence

Configuration is loaded from multiple sources in defined priority
order:

1. Runtime overrides (CLI flags, environment variables).
2. Pipeline configuration (`settings.yaml`).
3. Profile configuration (`profiles/production.yaml`).
4. Plugin pack defaults (`packs/llm/defaults.yaml`).
5. System defaults.

The configuration loader is `core/config.py` (Dynaconf + Pydantic v2;
2,227 LOC — flagged for per-file deep-dive in finding C1 / R5).

The Settings → Runtime contract is verified by
`scripts/check_contracts:main`, which alignment-tests every Settings
field against its runtime consumer. The contract surface lives in
`contracts/config/`.
