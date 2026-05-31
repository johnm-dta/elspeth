# Quality Assessment

> **Posture:** Findings are rated by objective severity (Critical /
> High / Medium / Low). Strengths are reported only where evidence
> warrants them. Methodology and citation chains are in
> [`appendix/C-provenance.md`](appendix/C-provenance.md).

**Subsystem score totals:** Critical 0 · High 1 · Medium 12 · Low 7.
**Cross-cutting:** 2 Medium (1 resolved in-pass).

| Subsystem | Score | Critical | High | Medium | Low |
|-----------|:----:|:----:|:----:|:----:|:----:|
| `engine/` | **4 / 5** | 0 | 0 | 2 | 1 |
| `core/` | **4 / 5** | 0 | 0 | 3 | 1 |
| `plugins/` | **4 / 5** | 0 | 0 | 2 | 2 |
| `web/` + `composer_mcp/` | **3 / 5** | 0 | 1 | 3 | 1 |
| `contracts/` | **5 / 5** | 0 | 0 | 2 | 2 |

Score rubric: **5** = production-grade discipline, structurally
enforced. **4** = strong, mechanically verified, with named work in
flight. **3** = adequate but with material structural debt. **2** =
weak, requires architectural intervention. **1** = unsafe.

---

## §1 `engine/` — score 4 / 5

The SDA execution layer: orchestrator, row processor, executors
(transform / coalesce / pass-through), retry manager, artefact
pipeline, span factory, triggers. 36 files, 17,425 LOC.

### Findings

#### E1 — ADR-010 dispatcher audit-completeness · **Resolved**

The prior assessment closed this finding via direct read of
`src/elspeth/engine/executors/declaration_dispatch.py:120–172`. Both
`except DeclarationContractViolation` (lines 137–141) and
`except PluginContractViolation` (lines 142–150) branches append to the
violations list; neither swallows. Post-loop logic correctly distinguishes
0 / 1 / N≥2 cases with reference-equality preservation at N=1 (the
non-regression invariant). Test coverage: 1,923 LOC across
`tests/unit/engine/test_declaration_dispatch.py` (642 LOC),
`tests/property/engine/test_declaration_dispatch_properties.py` (1,183
LOC), and `tests/integration/pipeline/orchestrator/test_declaration_contract_aggregate.py`
(98 LOC). **No remediation needed.**

#### E2 — `processor.py` cohesion is unverified · **Medium**

`engine/processor.py` (2,700 LOC) carries a docstring claiming one
cohesive responsibility (`RowProcessor` end-to-end), but its imports
span six visible concerns: DAG navigation, retry classification,
terminal-state assignment, ADR-009b cross-check, batch error handling,
quarantine routing. Whether the LOC reflects essential complexity (one
responsibility honestly large) or accidental concentration (multiple
responsibilities accreted) cannot be answered without a per-file
deep-dive. **Impact:** maintenance burden and onboarding friction; not
a runtime risk. **Recommendation:** [R5](07-improvement-roadmap.md#r5).
**Effort:** Medium.

#### E3 — Engine integration tests have no in-cluster directory · **Medium**

There is no `tests/integration/engine/`. Engine integration coverage
exists somewhere in the integration tree but cannot be located within
the engine cluster's scope. `CLAUDE.md` requires that integration tests
use `ExecutionGraph.from_plugin_instances()` and
`instantiate_plugins_from_config()`; the rule is currently
un-auditable from inside engine. **Impact:** the production-path-rule
guarantee for engine integration testing is currently un-auditable.
**Recommendation:** [R4](07-improvement-roadmap.md#r4) — a cross-cluster
integration-tier audit. **Effort:** Medium.

### Strengths

- **Terminal-state-per-token invariant is structurally guaranteed.**
  `engine/executors/state_guard.py:NodeStateGuard` implements "every
  row reaches exactly one terminal state" as a context-manager pattern,
  locked by `tests/unit/engine/test_state_guard_audit_evidence_discriminator.py`
  and `tests/unit/engine/test_row_outcome.py`. Context-manager-as-invariant
  for safety properties is genuinely good architectural practice — the
  type system cooperates with the runtime to make the invariant
  non-bypassable.
- **ADR-010 dispatch surface is drift-resistant by construction.** The
  4-site × 7-adopter mapping is locked by an AST-scanning unit test
  (`tests/unit/engine/test_declaration_contract_bootstrap_drift.py`);
  adding a new adopter without registering it fails CI.
- **`orchestrator/core.py` (3,281 LOC) is in-progress decomposition,
  not stagnant debt.** The orchestrator was previously a single
  3,000-LOC module; it has been refactored into six focused siblings
  (`core.py`, `types.py`, `validation.py`, `export.py`, `aggregation.py`,
  `outcomes.py`) with stable public API. Active remediation visible in
  the tree.

For cluster-level details, see [`subsystems/engine.md`](subsystems/engine.md).

---

## §2 `core/` — score 4 / 5

The L1 foundation: Landscape audit DB (4 repositories), DAG
construction & validation, Dynaconf+Pydantic configuration,
canonical-JSON hashing, payload store, retention, rate limiting,
security primitives, expression parser. 49 files, 20,791 LOC across six
sub-packages.

### Findings

#### C1 — `core/config.py` cohesion is unverified · **Medium**

Single-file Pydantic settings holding 12+ child dataclasses across
checkpoint, concurrency, database, landscape, payload-store, rate-limit,
retry, secrets, sinks, sources, transforms (2,227 LOC). Pydantic
settings concentrate for cross-validation reasons; whether 2,227 LOC is
appropriately concentrated or accreted by addition is open. **Impact:**
onboarding friction; this is the load-bearing entry into runtime
configuration and the largest single config file.
**Recommendation:** [R5](07-improvement-roadmap.md#r5). **Effort:** Medium.

#### C2 — `core/dag/graph.py` blast radius · **Medium**

`ExecutionGraph` (1,968 LOC) is consumed by every executor in `engine/`,
by `web/composer/_semantic_validator.py`, by `web/execution/validation.py`,
by `core/checkpoint/{manager,compatibility}.py`, and indirectly by
every plugin via the schema-contract validation flow. Any change to
`ExecutionGraph` semantics is system-wide. The test files exist
(`test_graph.py`, `test_graph_validation.py`) but their assertion
density relative to the file's behavioural surface is unverified.
**Recommendation:** L3 deep-dive on the public-contract test surface;
consider a pinned snapshot of `ExecutionGraph` semantic invariants.
**Effort:** Medium–Large.

#### C3 — Audit table count divergence · **Medium**

`core/landscape/` contains 20 schema tables; the institutional
documentation (`KNOW-A24` in the analysis's knowledge map) records 21.
**Impact:** documentation correctness, not architecture. `ARCHITECTURE.md`
is one major iteration behind on the Landscape schema.
**Recommendation:** [R10](07-improvement-roadmap.md#r10). **Effort:** Small.

#### C4 — `core/secrets.py` placement · **Low**

`core/secrets.py` (124 LOC, runtime resolver) lives at the `core/`
root, while `core/security/{secret_loader,config_secrets}.py` (529 LOC
combined) live in the sub-package. Responsibility-cut question between
L0 contracts and L1 core post-ADR-006. **Impact:** organisational
hygiene only. **Recommendation:** consider relocating to
`core/security/` for namespace consistency, or document the rationale
for the split inline. **Effort:** Small.

### Strengths

- **The Landscape facade pattern is real, not aspirational.**
  `landscape/__init__.py` re-exports exactly `RecorderFactory` and the
  four named repositories (`DataFlowRepository`, `ExecutionRepository`,
  `QueryRepository`, `RunLifecycleRepository`); repositories are *not*
  re-exported through `core/__init__.py`. Callers can only reach the
  audit DB through `RecorderFactory`. The encapsulation is mechanically
  enforceable, not documentary.
- **Protocol-based no-op parity is a deliberate offensive-programming
  discipline.** `EventBus`/`NullEventBus`, `RateLimiter`/`NoOpLimiter`
  pairs ensure callers never branch on `is None`; absent functionality
  is represented by an active no-op object that satisfies the protocol.

See [`subsystems/core.md`](subsystems/core.md).

---

## §3 `plugins/` — score 4 / 5

The L3 plugin ecosystem: system-owned (not user-extensible) sources,
transforms, sinks, plus shared infrastructure (audited HTTP/LLM
clients, hookspecs, base classes). 98 files, 30,399 LOC across four
sub-packages. The largest single subsystem.

### Findings

#### P1 — `azure_batch.py` is unread at component depth · **Medium**

`plugins/transforms/llm/azure_batch.py` (1,592 LOC) is the largest
single plugin file. The LLM batch path is high-stakes (financial cost,
audit coverage, retry semantics) and its internal cohesion is
un-assessed at this pack's depth. **Recommendation:** L3 deep-dive
([R5](07-improvement-roadmap.md#r5)); align effort estimate with
`processor.py` and `web/composer/tools.py`. **Effort:** Medium.

#### P2 — Trust-tier discipline is structural but not runtime-enforced · **Medium**

Every source module repeats "ONLY place coercion is allowed"; every
sink module repeats "wrong types = upstream bug = crash"; the
discipline is encoded in the `allow_coercion` config flag. But
cross-cluster invariant tests — for example, a fixture that injects a
transform that coerces and asserts the run fails — do not exist.
**Impact:** the contract is honoured today by author discipline; CI
does not catch a violator. **Recommendation:** a property-based or
fixture-based runtime probe, paired with an architecture-pack decision
on whether to mechanise the discipline. **Effort:** Medium.

#### P3 — Plugin-count drift in `ARCHITECTURE.md` · **Low**

Three statements in the institutional documentation disagree: 25
plugins, 46 plugins, and 29 plugins (the live verified count). Four
post-doc plugins were added without a doc update. **Impact:**
documentation correctness. **Recommendation:** [R10](07-improvement-roadmap.md#r10).
**Effort:** Small.

#### P4 — Module-level cycle in `plugins/transforms/llm` · **Low**

`plugins/transforms/llm` ↔ `plugins/transforms/llm/providers` form a
2-node strongly-connected component (provider-registry pattern with
deferred runtime instantiation; the runtime decoupling is documented at
`transform.py:9-13`). Module-level cycle is visible to the import
system but runtime-decoupled. **Impact:** none at runtime; visible in
static analysis. **Recommendation:** compare cost of moving shared
types into `plugins/infrastructure/` versus leaving the cycle visible.
**Effort:** Small.

### Strengths

- **`plugins/infrastructure/` is the structural spine and it is honoured
  consistently.** All 23 intra-cluster edges flow toward `infrastructure/`;
  `plugins/sinks → plugins/infrastructure` weight 45 is the heaviest
  single L3 edge in the codebase. Sources, transforms, and sinks are
  clients of one another's infrastructure layer, not peers — the
  dependency shape matches the documented design.
- **Trust-tier discipline is documented identically in every leaf
  module.** Repetition is not a smell here; it is the protocol that
  prevents drift. New contributors writing a new source see the same
  "ONLY place coercion is allowed" notice as every existing source.

See [`subsystems/plugins.md`](subsystems/plugins.md).

---

## §4 `web/` + `composer_mcp/` — score 3 / 5

The composer cluster: FastAPI web UI server (8 backend sub-packages —
`auth/`, `blobs/`, `catalog/`, `composer/`, `execution/`, `middleware/`,
`secrets/`, `sessions/` — plus the deferred `frontend/`) and the
stateful MCP pipeline-construction server (`composer_mcp/`). 75 files,
~23,400 LOC of Python.

This cluster scores lowest not because the code is poor — by every
mechanical measure it is competent — but because of three structural
features that combine to make it the system's highest-risk
architectural surface.

### Findings

#### W1 — 7-node strongly-connected component spans every `web/*` sub-package · **High**

The cycle covers `web ↔ web/auth ↔ web/blobs ↔ web/composer ↔ web/execution ↔ web/secrets ↔ web/sessions`.
No acyclic decomposition is possible within `web/`. The cycle is
structurally load-bearing — it implements the FastAPI app-factory
pattern, where `web/app.py:create_app()` imports every sub-package's
router (the wiring leg) and sub-packages reach back via
`from elspeth.web.config import WebSettings` and `run_sync_in_worker`
(the shared-infrastructure leg). Both directions are intentional.

**Impact:** any architectural change in `web/` must reason about all
seven sub-packages simultaneously. Adding a new sub-package extends the
SCC by default.

**Recommendation:** [R2](07-improvement-roadmap.md#r2). The right shape
is probably (a) extract a `web/_core/` containing `WebSettings` and
`run_sync_in_worker` so sub-packages depend on `_core` rather than the
namespace root; (b) make `web/app.py` the only place that imports
sub-package routers. **Until then, freeze new sub-package additions to
`web/` unless they are explicitly architecture-reviewed.**

**Effort:** Large (5–8 hr architecture-pack pass + Large–Extra Large
implementation).

#### W2 — Largest concentration of composer logic · **Medium**

`web/composer/tools.py` (3,860 LOC) and `web/composer/state.py` (1,710
LOC) together carry 5,570 LOC — larger than any non-engine subsystem at
this pack's depth. The composer state machine is the most
architecturally-consequential surface in the system after the engine,
and any change has high blast radius.

**Recommendation:** decomposition is paired with the SCC#4 decision
(W1). Isolated decomposition without the SCC context risks producing a
worse cycle. **Effort:** Large.

#### W3 — `web/sessions/routes.py` was missed by the prior inventory · **Medium**

The prior inventory listed 12 files >1,500 LOC; `web/sessions/routes.py`
was not among them. At this pack's HEAD it is 2,067 LOC — a +504 LOC
growth (+32%) since the inventory pass. The file exists, is large, and
remains unread at component depth.

**Impact:** inventory-completeness defect.

**Recommendation:** [R3](07-improvement-roadmap.md#r3). Add to the
deep-dive backlog; re-run the ≥1,500-LOC scan as part of any future
inventory pass. **Effort:** Small (catalog) + Medium (eventual deep-dive).

#### W4 — `composer_mcp/` is structurally a sibling of `web/composer/`, not of `mcp/` · **Medium**

The institutional layout framed `mcp/` and `composer_mcp/` as siblings.
The L3 import oracle records zero edges between them and a
weight-13 edge from `composer_mcp → web/composer` (up from weight 12 at
the time of the prior assessment). `composer_mcp/` is the MCP transport
that the web composer uses; calling it a sibling of the audit-analyser
`mcp/` mis-frames the relationship.

**Impact:** the institutional mental model is wrong; the live code is
fine, but anyone reading the layout in isolation would be misled.

**Recommendation:** either move `composer_mcp/` under `web/composer/`,
or document the structural relationship in `ARCHITECTURE.md`.
**Effort:** Small (doc) or Medium (relocation).

#### W5 — `web/execution → .` (cli root) edge purpose unclear · **Low**

A weight-3 edge from `web/execution` into the cli-root namespace.
Could be a benign re-export of a public symbol from
`elspeth/__init__.py`, or a deferred-import hack to bypass an explicit
cluster dependency. Resolution takes minutes at L3 depth.
**Recommendation:** L3 inspection of the import line(s). **Effort:** Small.

### Strengths

- **The composer cluster is a structural import-graph leaf.** Zero
  inbound edges from any other cluster; only the two console-script
  entry points (`elspeth-web`, `elspeth-composer`) consume it.
  Architectural changes inside the cluster cannot break library callers
  elsewhere — a remarkably clean blast-radius property for a
  ~23k-LOC subsystem.

See [`subsystems/web-composer.md`](subsystems/web-composer.md).

---

## §5 `contracts/` — score 5 / 5

The L0 leaf: shared types, protocols, enums, errors, frozen-dataclass
primitives, audit DTOs, declaration-contract framework. 63 files,
17,403 LOC.

The L0 leaf is the system's strongest cluster. It is mechanically
verified to import nothing above (zero outbound edges); its
responsibility discipline is coherent; its CI gates are stable; and the
cross-cluster handshakes against engine, core, and plugins are aligned.
The score of 5 reflects the discipline visible in the cluster, not the
absence of improvements.

### Findings

#### K1 — `contracts/errors.py` mixes Tier-1 and Tier-2 surfaces · **Medium**

`contracts/errors.py` (1,566 LOC) holds Tier-1 raiseable exceptions,
Tier-2 frozen audit DTOs, structured-reason TypedDicts, and re-exported
`FrameworkBugError` in a single file. The Tier-1 / Tier-2 distinction
is currently encoded by inline comments, not by file split.

**Impact:** the discipline currently relies on convention; a CI-enforced
split (e.g., `errors_tier1.py` versus `errors_dtos.py`) would mechanise
it.

**Recommendation:** [R11](07-improvement-roadmap.md#r11). Split when
the file next needs material edits; don't split-for-the-sake-of-splitting.
**Effort:** Medium.

#### K2 — `plugin_context.py:31` TYPE_CHECKING smell · **Medium**

`contracts/plugin_context.py:31` is the only cross-layer reference in
the cluster — a TYPE_CHECKING import of `core.rate_limit.RateLimitRegistry`.
ADR-006d Violation #11 candidate; an extracted `RateLimitRegistryProtocol`
in `contracts.config.protocols` would eliminate the TYPE_CHECKING block.

**Impact:** annotation-only; the runtime is not coupled. But
TYPE_CHECKING imports are the canonical marker of a deferred structural
fix, and ADR-006d has a "never lazy-import" rule that this violates.

**Recommendation:** [R12](07-improvement-roadmap.md#r12). The pattern
is well-understood. **Effort:** Small.

#### K3 — `schema_contract` sub-package promotion · **Low**

The `schema_contract` cluster (8 files, ~3,500 LOC) has high internal
cohesion; promoting it to `contracts/schema_contracts/` would mirror the
`config/` partition. **Impact:** organisational hygiene only.
**Recommendation:** defer until a near-term ADR motivates it. **Effort:** Small.

#### K4 — Catalog citation editorial defect · **Low**

Ten KNOW-A* citations in the institutional knowledge map have IDs that
resolve but inline rationales that mismatch. **Impact:** documentation
correctness. **Recommendation:** [R10](07-improvement-roadmap.md#r10).
**Effort:** Small.

### Strengths

- **L0 leaf invariant is mechanically confirmed.** Zero outbound edges
  in the L3 import oracle; layer-conformance scan empty for both L1
  upward-import and TYPE_CHECKING findings. The leaf is a leaf,
  verifiably.
- **ADR-010 declaration-trust framework's L0 surface is complete.**
  `AuditEvidenceBase` ABC, `@tier_1_error` decorator + frozen registry,
  `DeclarationContract` 4-site framework with bundle types and
  payload-schema H5 enforcement, secret-scrub last-line-of-defence —
  all present, all consumed by engine via the contracts-defined
  protocols.

See [`subsystems/contracts.md`](subsystems/contracts.md).

---

## §6 Confidence ledger

| Section | Confidence | Why |
|---|---|---|
| §1 (engine) | High | Cluster-level analysis + live LOC + clean CI enforcer at this pack's HEAD |
| §2 (core) | High | Cluster-level analysis + live verification + institutional knowledge corroboration |
| §3 (plugins) | High | Cluster-level analysis + L3 oracle edge weights + live LOC |
| §4 (web/composer) | High | Cluster-level analysis + L3 oracle SCC topology + live LOC |
| §5 (contracts) | High | Cluster-level analysis + L3 oracle leaf-status confirmation |

For per-finding citation chains, see the linked subsystem documents in
[`subsystems/`](subsystems/) and the provenance ledger at
[`appendix/C-provenance.md`](appendix/C-provenance.md).
