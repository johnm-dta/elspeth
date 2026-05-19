# Improvement Roadmap

> Severity reflects **architectural impact**, not effort to remediate.
> A High-severity finding may be Small to fix; a Low may be Large. Both
> matter. Effort uses the standard t-shirt scale: **S** (≤4 hr), **M**
> (1–2 days), **L** (3–5 days), **XL** (>1 week).

---

## At-a-glance

| # | Item | Severity | Effort | Status | Dependency |
|---|------|----------|--------|--------|------------|
| R1 | ADR-010 dispatcher audit-completeness verification | — | — | **Resolved in prior pass** | — |
| R2 | Architecture-pack decision on SCC#4 (`web/*` 7-node SCC) | **High** | L | Open | Best informed by R5 |
| R3 | Re-run ≥1,500-LOC inventory; add missed entries to backlog | Medium | S | Open | None |
| R4 | Cross-cluster integration-tier audit | Medium | M | Open | None |
| R5 | Per-file deep-dive on `processor.py` and `core/config.py` | Medium | M each | Open | None |
| R6 | Frontend-aware archaeologist pass | Medium | L | Open | None |
| R7 | Test-architecture pass | Medium | L | Open | None |
| R8 | STRIDE threat model + audit-completeness verification | Medium | L | Open | R1 (closed) |
| R9 | Document the composer credential flow | Medium | S | Open | None |
| R10 | `ARCHITECTURE.md` doc-correctness pass | Low | S | Open | None |
| R11 | Resolve `errors.py` Tier-1 / Tier-2 split | Low | M | Open | Trigger on next material edit |
| R12 | Resolve `plugin_context.py:31` TYPE_CHECKING smell | Low | S | Open | None |

---

## Recommended sequence

This sequence prioritises **security posture (R8)** and **the two
scope gaps (R6, R7)** before structural-refactor decisions (R5, R2).
Mechanical hygiene items (R3, R4, R9, R10, R11, R12) can proceed in
parallel without blocking the strategic sequence.

```
R1 (closed) ──► R8 ──┐
                     ├──► R2 ──► (next major web/ work)
R5 ──────────────────┘
                     │
R6 ──────────────────┤  (parallel; no dependency on others)
R7 ──────────────────┘

Hygiene track (parallel, no dependencies):  R3 │ R4 │ R9 │ R10 │ R11 │ R12
```

---

## R1 — ADR-010 dispatcher audit-completeness · *Resolved*

The prior assessment closed this finding by direct read of
`src/elspeth/engine/executors/declaration_dispatch.py:120–172`. Both
`except DeclarationContractViolation` and `except PluginContractViolation`
branches append to the violations list; post-loop logic correctly
distinguishes 0 / 1 / N≥2 cases with reference-equality preservation at
N=1. Test coverage: 1,923 LOC across unit, property, and integration
tiers.

**Action:** none. The verification record stands.

Maps to: [`06-quality-assessment.md`](06-quality-assessment.md) §1 E1.

---

## R2 — SCC#4 decomposition decision (`web/*` 7-node SCC) · **High** · Effort: L

### Why

The 7-node strongly-connected component is structurally load-bearing
today. Adding any new sub-package to `web/` extends the cycle by
default. This is the single architectural item that should land before
the next major addition to `web/`.

### What

An architecture-pack decision document that:

1. Confirms or revises the proposed shape: extract `web/_core/`
   containing `WebSettings` and `run_sync_in_worker`; make `web/app.py`
   the only place that imports sub-package routers.
2. Identifies the per-sub-package migration cost and ordering.
3. Defines the post-decomposition import contract that future
   sub-packages must honour.

### Until then

**Freeze new sub-package additions to `web/` unless explicitly
architecture-reviewed.** Adding a new sub-package without the
decomposition decision extends the SCC by default and worsens the
remediation cost.

Maps to: [`06-quality-assessment.md`](06-quality-assessment.md) §4 W1.

---

## R3 — Re-run ≥1,500-LOC inventory; add missed entries · **Medium** · Effort: S

### Why

The prior 12-file deferral list missed `web/sessions/routes.py`, which
has since grown from 1,563 to 2,067 LOC (+32%). The synthesis's
deferred-deep-dive list under-counts the live ≥1,500-LOC population by
five files (`cli.py`, `execution_repository.py`, `azure_batch.py`,
`data_flow_repository.py`, `web/sessions/routes.py`).

### What

Re-run the scan, update the deep-dive backlog, schedule the
deep-dives. The mechanical cost is minutes; the documentation
correctness payoff is material.

### Live ≥1,500-LOC roster (at this pack's HEAD)

| File | LOC | Status |
|------|---:|--------|
| `web/composer/tools.py` | 3,860 | Backlog (R5) |
| `engine/orchestrator/core.py` | 3,281 | Backlog (R5) |
| `engine/processor.py` | 2,700 | Backlog priority-1 (R5) |
| `cli.py` | 2,357 | Backlog (newly captured) |
| `core/config.py` | 2,227 | Backlog priority-2 (R5) |
| `web/sessions/routes.py` | 2,067 | Backlog (newly captured; **was missed**) |
| `core/dag/graph.py` | 1,968 | Backlog |
| `core/landscape/execution_repository.py` | 1,750 | Backlog (newly captured) |
| `web/composer/state.py` | 1,710 | Backlog (paired with R2) |
| `engine/coalesce_executor.py` | 1,603 | Resolved (essential complexity) |
| `plugins/transforms/llm/azure_batch.py` | 1,592 | Backlog (newly captured) |
| `core/landscape/data_flow_repository.py` | 1,590 | Backlog (newly captured) |
| `contracts/errors.py` | 1,566 | Trigger-on-edit (R11) |

Maps to: [`06-quality-assessment.md`](06-quality-assessment.md) §4 W3.

---

## R4 — Cross-cluster integration-tier audit · **Medium** · Effort: M

### Why

There is no `tests/integration/engine/` directory. The
`CLAUDE.md`-mandated production-path rule (integration tests must use
`ExecutionGraph.from_plugin_instances()` and
`instantiate_plugins_from_config()`) is currently un-auditable from
inside any single cluster. The integration-test architecture is a
cross-cluster concern that no per-cluster pass can resolve.

### What

Locate the integration-tier coverage of engine paths, verify consistent
production-path use, document the test-architecture topology. This is
distinct from R7 (test-pyramid analysis) and from any per-cluster
deep-dive.

Maps to: [`06-quality-assessment.md`](06-quality-assessment.md) §1 E3.

---

## R5 — Per-file deep-dive on `processor.py` and `core/config.py` · **Medium** · Effort: M each

### Why

These are the two largest files where essential-vs-accidental cohesion
is open. Without per-file reads, the architecture pack cannot make
split-or-keep recommendations on them. The reads are bounded effort;
the verdicts unblock downstream decisions (notably R2's decomposition
shape for `web/composer/tools.py` benefits from the same methodology).

### What

A per-file architecture-archaeology pass that:

1. Reads the entire file at L3 depth.
2. Names each cohesive responsibility cluster.
3. Quantifies essential complexity versus accidental concentration.
4. Recommends split-or-keep with seam locations if split is recommended.

### Order

| Priority | File | Why first |
|---|---|---|
| 1 | `engine/processor.py` (2,700 LOC) | Highest blast-radius open question; the row-processor is on every pipeline's hot path |
| 2 | `core/config.py` (2,227 LOC) | Onboarding friction; load-bearing entry into runtime configuration |
| 3 | `web/composer/tools.py` (3,860 LOC) | Largest file; required input to R2 decomposition decision |
| 4 | `engine/orchestrator/core.py` (3,281 LOC) | Decomposition partially complete; quantify residual |
| 5 | `core/dag/graph.py` (1,968 LOC) | Cascade-prone (every executor consumes it) |
| 6 | `web/composer/state.py` (1,710 LOC) | Paired with R2 |
| 7 | `cli.py` (2,357 LOC) | Houses `TRANSFORM_PLUGINS` registry; coupling debt |

Maps to: [`06-quality-assessment.md`](06-quality-assessment.md) §1 E2,
§2 C1.

---

## R6 — Frontend-aware archaeologist pass · **Medium** · Effort: L

### Why

`src/elspeth/web/frontend/` (~13k LOC TypeScript / React) is materially
load-bearing for security, session state, and API-contract integrity
claims. A Python-lens archaeologist cannot cover it. Specific
consequences of leaving the gap open:

- The composer cluster's "0 inbound cross-cluster edges" finding is
  structurally true at the Python-import level but semantically
  incomplete — the frontend consumes the composer's HTTP/MCP surface,
  invisible to `enforce_tier_model.py`.
- Authentication / session-state flow on the SPA side is unanalysed.
- The frontend's coupling to backend API contracts is unverified;
  contract drift is a known source of production-grade SPA failures.

### What

Engage `lyra-site-designer` or a TypeScript/React-specialised codebase
explorer. Output: a parallel pack section (or peer document) covering
the frontend's component shape, state flow, API consumption, and
auth/session handling.

Maps to: [`08-known-gaps.md`](08-known-gaps.md) §1.

---

## R7 — Test-architecture pass · **Medium** · Effort: L

### Why

The `tests/` tree is 2.9× the size of `src/` (~351k LOC across ~851
files). Whether this represents remarkable test discipline (the
audit-grade nature of the system would warrant it) or an inverted
pyramid (a known cost-of-ownership trap) is unanswered. The two
hypotheses have wildly different implications.

### What

A `ordis-quality-engineering:analyze-pyramid` pass (or equivalent):

- Map test-tier distribution (unit vs integration vs property vs E2E).
- Identify fixture topology and any cross-tier coupling.
- Verify production-path conformance (overlaps with R4).
- Flag any sleepy-assertion or test-interdependence anti-patterns.

Maps to: [`08-known-gaps.md`](08-known-gaps.md) §2.

---

## R8 — STRIDE threat model + audit-completeness verification · **Medium** · Effort: L

### Why

The audit-trail and trust-tier surfaces are the highest-stakes security
territory because the audit trail is the legal record. R1 is now closed
and the inputs (trust-tier topology, audit-trail completeness, ADR-010
dispatcher state) are stable.

### What

A `ordis-security-architect:threat-model` pass scoped to:

- The trust-tier topology end-to-end (Tier 3 ingress through Tier 1
  audit DB).
- Audit-trail completeness across the engine → core → contracts join.
- The ADR-010 declaration-trust verification at the dispatcher's
  audit-complete boundary.
- The composer credential-flow question (overlaps with R9).

Maps to: [`05-cross-cutting-concerns.md`](05-cross-cutting-concerns.md) §1.

---

## R9 — Document the composer credential flow · **Medium** · Effort: S

### Why

`web/secrets/` has zero outbound edges to other clusters at
package-collapse granularity, yet composer/execution rely on
LLM-provider credentials. **The fact that the import graph cannot
answer this is itself a red flag** — credential flow should be visible
to architectural analysis, not hidden in DI plumbing.

### What

Either:

1. Confirm credentials flow via `WebSettings` injection at request
   time, and add the diagram to `ARCHITECTURE.md` and the `web` page in
   `subsystems/`.
2. Or, name the alternative mechanism explicitly and assess whether it
   should be made statically visible.

Maps to: [`05-cross-cutting-concerns.md`](05-cross-cutting-concerns.md) §1.

---

## R10 — `ARCHITECTURE.md` doc-correctness pass · **Low** · Effort: S

### Why

Several institutional-documentation drifts have material onboarding
friction in aggregate, even though each is individually small.

### Itemised backlog

| Defect | Source-of-truth value | Documented value |
|--------|----------------------|------------------|
| Plugin count | 29 (verified) | 25 (one statement), 46 (another) |
| Audit table count | 20 (observed in `core/landscape/`) | 21 |
| ADR index | 17 ADRs accepted (`docs/architecture/adr/001..017`) | Table covers 001..006 only |
| Schema-mode vocabulary | `observed` / `fixed` / `free` (runtime YAML) | `dynamic` / `strict` / `free` (prose table) |
| `testing/` subsystem description | Pytest plugin, 877 LOC, 2 files | "~9,500 LOC including ChaosLLM/ChaosWeb/ChaosEngine" — that describes `tests/`, not `src/elspeth/testing/` |
| Subsystem LOC figures | `contracts` 17.4k, `engine` 17.4k, `telemetry` 2.9k | `contracts` ~8.3k, `engine` ~12k, `telemetry` ~1.2k (stale) |

Each is a small edit; the aggregate is a material onboarding-friction
reduction.

See also [`reference/adr-index.md`](reference/adr-index.md) for the
authoritative ADR catalogue.

Maps to: [`06-quality-assessment.md`](06-quality-assessment.md) §2 C3,
§3 P3, §5 K4.

---

## R11 — Resolve `errors.py` Tier-1 / Tier-2 split · **Low** · Effort: M

### Why

`contracts/errors.py` (1,566 LOC) mixes Tier-1 raiseable exceptions,
Tier-2 frozen audit DTOs, structured-reason TypedDicts, and re-exported
`FrameworkBugError` in a single file. The Tier-1 / Tier-2 distinction
is encoded by inline comments today; a CI-enforced split would
mechanise it.

### Trigger

When `contracts/errors.py` next requires material edits, split into
`errors_tier1.py` (raiseable exceptions) and `errors_dtos.py` (frozen
audit DTOs). **Don't split-for-the-sake-of-splitting** — pair with the
next ADR that touches Tier-1 error definitions.

Maps to: [`06-quality-assessment.md`](06-quality-assessment.md) §5 K1.

---

## R12 — Resolve `plugin_context.py:31` TYPE_CHECKING smell · **Low** · Effort: S

### Why

`contracts/plugin_context.py:31` carries a TYPE_CHECKING import of
`core.rate_limit.RateLimitRegistry`. This is the cluster's only
cross-layer reference and is the canonical marker of a deferred
structural fix that ADR-006d's "never lazy-import" rule forbids.

### What

Extract `RateLimitRegistryProtocol` into `contracts.config.protocols`;
remove the TYPE_CHECKING block. Cleanly addresses ADR-006d Violation
#11.

Maps to: [`06-quality-assessment.md`](06-quality-assessment.md) §5 K2.
