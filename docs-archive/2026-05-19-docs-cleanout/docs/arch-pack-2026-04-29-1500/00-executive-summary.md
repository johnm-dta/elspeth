# Executive Summary

> **Verdict:** ELSPETH is a **structurally well-disciplined**, audit-grade
> Python codebase of ~122,500 production LOC. It is safe to continue
> building on. It carries **named, locatable, addressable structural debt**
> in two concentrations and one inventory defect, and ships with two
> material scope gaps (frontend and test architecture) that downstream
> packs must close before any "complete architecture" claim is honest.

---

## What ELSPETH is

A **domain-agnostic framework for auditable Sense / Decide / Act (SDA)
pipelines**: data flows from a single source through ordered transforms
(including gates and aggregations) to one or more named sinks. Every
decision — whether the "decide" step is an LLM, an ML model, a rules
engine, or a threshold check — must be traceable to its source data,
configuration, and code version. The audit trail is the legal record;
"I don't know what happened" is never an acceptable answer for any
output.

The system is structured as 11 top-level Python subsystems organised
into a 4-layer model (`contracts` → `core` → `engine` → application
surfaces), with three distinct trust tiers governing data handling at
each boundary. See [`02-architecture-overview.md`](02-architecture-overview.md).

---

## Headline strengths

These claims are evidence-anchored to deterministic oracles, code
`file:line` references, and the project's own institutional documents.
They are not diplomatic framing.

| # | Strength | Evidence |
|---|----------|----------|
| 1 | **The 4-layer model is mechanically clean.** Zero upward-import violations across all 11 subsystems. | CI-enforced by `scripts/cicd/enforce_tier_model.py`; re-verified at this pack's HEAD. |
| 2 | **Audit invariants are structurally guaranteed, not conventional.** Every row reaches exactly one terminal state (CONTEXT-MANAGER PATTERN), the ADR-010 dispatch surface is locked by an AST-scanning drift test, all frozen dataclass containers are deeply immutable. | `engine/executors/state_guard.py:NodeStateGuard`; `tests/unit/engine/test_declaration_contract_bootstrap_drift.py`; `contracts/freeze.py:freeze_fields`. |
| 3 | **The L0 leaf invariant is mechanically confirmed.** `contracts/` imports nothing above; the leaf is a leaf, verifiably. | Zero outbound edges in the L3 import oracle. |
| 4 | **The plugin spine pattern is honoured consistently.** All plugin sub-packages depend downward on `plugins/infrastructure/`; sources, transforms, and sinks are clients of one another's infrastructure layer, not peers. | Heaviest single L3 edge is `plugins/sinks → plugins/infrastructure` (weight 45). |
| 5 | **The composer cluster is a structural import-graph leaf.** Architectural changes inside `web/`+`composer_mcp/` cannot break library callers elsewhere — a remarkably clean blast-radius property for a ~23k-LOC subsystem. | Zero inbound cross-cluster edges to the composer cluster. |

This level of mechanical discipline — CI-enforced layer model,
AST-scanning drift tests, context-manager state guards, deep-frozen
dataclass primitives, allowlist-with-justification defensive-pattern
detection — is **above the median for codebases of this size and
complexity**.

---

## Structural debt (locatable, addressable)

The debt is not hidden. The two concentrations and the one inventory
defect below have specific files, specific severities, and specific
remediation paths in [`07-improvement-roadmap.md`](07-improvement-roadmap.md).

| # | Item | Severity | Effort | Roadmap |
|---|------|----------|--------|---------|
| 1 | **7-node strongly-connected component spans every `web/*` sub-package.** No acyclic decomposition is possible without an architecture-pack decision. Every new sub-package added to `web/` extends the cycle by default. | High | Large (5–8 hr architecture pass + L–XL implementation) | R2 |
| 2 | **13 files exceed 1,500 LOC**, concentrating ~23% of production Python in 0.6% of files (28,271 of 122,554 LOC). The largest are `web/composer/tools.py` (3,860), `engine/orchestrator/core.py` (3,281), `engine/processor.py` (2,700), `cli.py` (2,357), `core/config.py` (2,227). Per-file cohesion is unverified at this pack's depth. | Medium (concentration risk; verdicts open) | Medium per file | R5 |
| 3 | **`web/sessions/routes.py` (2,067 LOC) was missed by the prior inventory pass entirely**, and has grown ~32% since then. An inventory-completeness defect, not a design defect. | Medium | Small (catalog) + Medium (eventual deep-dive) | R3 |

**Recommendation:** the single architectural item that should land
before the next major addition to `web/` is the SCC#4 decomposition
decision. Until it is made, **freeze new sub-package additions to
`web/` unless explicitly architecture-reviewed**.

---

## Cross-cutting concerns

| Concern | Posture | Notes |
|---------|---------|-------|
| **Audit-trail completeness** | Strong | End-to-end mechanical guarantees: terminal-state-per-token (engine), 4-repository facade (core), L0 audit DTOs (contracts). One verification gap flagged by the prior assessment was closed during it; ADR-010 dispatcher behaviour confirmed correct with 1,923 LOC of dedicated test coverage. |
| **Trust-tier discipline** | Strong with one open question | Tier 3 (external) → sources coerce → Tier 2 (pipeline) pass through → Tier 1 (audit DB) crash on anomaly. Discipline is documented identically in every plugin module and partially CI-enforced; runtime-probe tests for the discipline are absent. |
| **Security (other than audit)** | **Open** | Composer credential flow is not visible to the import graph — either it threads through `WebSettings` injection at request time (and should be diagrammed) or via a mechanism the static analysis cannot see (which should be named). Frontend authentication / session-state flow is unanalysed (out of scope). |
| **Maintainability** | Mixed | Layer model honoured cleanly; per-file LOC discipline mixed (see debt item 2 above); documentation drift is material — `ARCHITECTURE.md` is one major iteration behind on plugin counts, audit-table counts, and the ADR index. |
| **Performance** | **Not assessed** | Out of scope. `engine/processor.py` is an architecturally-hot path candidate; whether it is in the actually-hot path requires profiling. |
| **Testability** | **Not assessed** | The `tests/` tree is 2.9× the size of `src/`. Whether that is remarkable test discipline or an inverted pyramid is unknown without a test-architecture pass. |

---

## Recommended next moves

In order of value-per-effort. Full descriptions in
[`07-improvement-roadmap.md`](07-improvement-roadmap.md).

1. **R8 — STRIDE threat model** on the trust-tier topology and audit-trail
   completeness. The architectural inputs are now stable; the security
   pack can begin.
2. **R6 — Frontend-aware archaeologist pass** on the ~13k-LOC TS/React
   SPA under `web/frontend/`. This is a prerequisite for any complete
   security claim about the FastAPI-plus-SPA system.
3. **R7 — Test-architecture pass** to resolve the inverted-pyramid
   question on the 2.9× src-to-tests ratio.
4. **R5 — Per-file deep-dives** on `processor.py` and `core/config.py`
   first; these are the largest files where essential-vs-accidental
   cohesion is open and remediation decisions are blocked on the answer.
5. **R2 — SCC#4 decomposition decision** for `web/`. Best informed by
   the file deep-dives (R5) on `web/composer/tools.py` and
   `web/composer/state.py`.

The mechanical hygiene items (R3 inventory refresh, R4 integration-test
audit, R9 credential-flow documentation, R10 ARCHITECTURE.md
doc-correctness, R11 errors.py split, R12 TYPE_CHECKING smell removal)
can proceed in parallel without blocking the strategic sequence.

---

## What this pack does not claim

The largest dishonest claim available about this codebase would be
"the architecture analysis is complete." It is not. The frontend, the
test architecture, and the per-file cohesion of the 13 large files are
genuinely outside this pack's coverage. **The honest claim is that this
pack is the strongest foundation available without those additional
passes — and names exactly where each one should land.**

See [`08-known-gaps.md`](08-known-gaps.md) for the full list of named
limitations and the downstream packs that resolve them.

---

## At-a-glance numbers

| Metric | Value | Source |
|--------|-------|--------|
| Production Python LOC | ~122,554 | `wc -l src/elspeth/**/*.py` at this pack's HEAD |
| Top-level subsystems | 11 | [`subsystems/`](subsystems/) |
| Composite (≥4 sub-pkgs OR ≥10k LOC OR ≥20 files) | 5 | engine, core, contracts, plugins, web |
| L3 import-graph nodes | 33 | `reference/l3-import-graph.json#stats.total_nodes` |
| L3 import-graph edges | 79 | `reference/l3-import-graph.json#stats.total_edges` |
| Strongly-connected components | 5 (largest 7 nodes) | `reference/l3-import-graph.json#stats.scc_count` |
| Files ≥1,500 LOC | 13 | live `wc -l` |
| Subsystem quality scores (1–5) | engine 4, core 4, plugins 4, web 3, contracts 5 | [`06-quality-assessment.md`](06-quality-assessment.md) |
| Findings: Critical / High / Medium / Low | 0 / 1 / 12 / 7 | [`06-quality-assessment.md`](06-quality-assessment.md) |
| Cross-cutting findings (security, frontend gap) | 2 Medium (1 resolved in-pass) | [`05-cross-cutting-concerns.md`](05-cross-cutting-concerns.md) |
