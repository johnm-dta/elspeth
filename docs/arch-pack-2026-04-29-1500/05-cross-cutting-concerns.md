# Cross-Cutting Concerns

Architectural concerns that span subsystem boundaries. Each section
states the posture, names the evidence, and flags the open work.

---

## §1 Security

### §1.1 The defence-in-depth posture

The audit-trail and trust-tier surfaces are the highest-stakes security
territory because the audit trail is the legal record. At the
architectural level, the defence-in-depth is real:

- **Trust-tier topology is structural, not aspirational.** Tier 3
  (external) → sources coerce → Tier 2 (pipeline) → transforms / sinks
  pass through → Tier 1 (audit DB) crash on anomaly. The
  `enforce_tier_model.py` CI scanner detects defensive patterns at
  trust boundaries and is honoured by the codebase today.
- **Audit-trail completeness is end-to-end.** Engine encodes
  terminal-state-per-token via `NodeStateGuard`; core's Landscape
  facade persists the seven terminal and one non-terminal state across
  20 schema tables; contracts owns the L0 audit DTO vocabulary.
- **Secret scrub is a last-line defence at the L0 boundary.** Encoded
  in `contracts/declaration_contracts.py` via the H5 payload-schema
  enforcement.

See [`02-architecture-overview.md#4-audit-trail-design`](02-architecture-overview.md#4-audit-trail-design)
and [`04-component-view.md#3-the-audit-trail-backbone`](04-component-view.md#3-the-audit-trail-backbone)
for the structural detail.

### §1.2 Open security concerns

| # | Severity | Concern |
|---|----------|---------|
| SEC1 | **Resolved in prior pass** | The synthesis-flagged audit-completeness verification gap in `engine/executors/declaration_dispatch.py:137,142` was closed by direct read — the pattern is correct aggregation, with 1,923 LOC of dedicated test coverage. See [`06-quality-assessment.md#1-engine--score-4--5`](06-quality-assessment.md#1-engine--score-4--5) finding E1. |
| SEC2 | **Medium** | Composer credential flow (`web/secrets/` has zero outbound edges to other clusters at package-collapse granularity, yet composer / execution rely on LLM-provider credentials). Whether credentials flow via `WebSettings` injection at request time, or via some other mechanism, is not visible to architectural analysis. **The fact that the import graph cannot answer this is itself a red flag** — credential flow should be visible to architectural analysis, not hidden in DI plumbing. Resolution: [R9](07-improvement-roadmap.md#r9). |
| SEC3 | **Medium** | The frontend (out of scope for this pack — see [`08-known-gaps.md#1`](08-known-gaps.md#1-frontend-webfrontend)) handles authentication tokens and session state. **No security analysis of the frontend has been performed.** A defence-in-depth claim about a browser-facing FastAPI-plus-SPA system is incomplete without a frontend-side review. Resolution: [R6](07-improvement-roadmap.md#r6). |

### §1.3 Recommended downstream

A `ordis-security-architect:threat-model` pass on the now-stable
trust-tier topology, audit-trail completeness across the engine → core
→ contracts join, and the composer credential-flow question. Mapped to
[R8](07-improvement-roadmap.md#r8).

---

## §2 Performance & operability

**Out of scope for this pack.** No profiling, no synthetic workload,
no production-trace analysis. Performance assertions cannot be made
from this pack.

What this pack can say from static analysis:

- `engine/processor.py` (2,700 LOC) handles per-row processing and is
  an **architecturally-hot** path candidate.
- Whether it is in the **actually-hot** path (versus being dominated by
  source I/O or LLM-provider latency) is a profiling question.

Resolution: a separate profile-driven pass with representative
workloads (e.g., `axiom-python-engineering:profile`). Not in this
pack's roadmap because performance work belongs to an operational
specialist, not an architecture review.

---

## §3 Maintainability

| Dimension | Assessment | Evidence |
|-----------|------------|----------|
| **Layer model honoured** | Strong | CI-enforced clean today |
| **Per-file LOC discipline** | Mixed | 13 files ≥1,500 LOC; ~23% of LOC in 0.6% of files |
| **Test coverage of core invariants** | Likely strong, but inverse-pyramid risk unknown | `tests/` is 2.9× src/; no test-architecture pass performed (R7) |
| **Documentation drift** | Material | Plugin-count drift (25 vs 46 vs verified 29), audit-table count (20 vs 21), ADR index (`docs/architecture/adr/007..017` exist but `ARCHITECTURE.md`'s table covers `001..006` only). See [`reference/adr-index.md`](reference/adr-index.md). |
| **Onboarding readiness** | Medium | `CLAUDE.md` is excellent (load-bearing institutional memory); `ARCHITECTURE.md` is one major iteration behind. |
| **Architectural decision records** | Good | 17 ADRs accepted (`docs/architecture/adr/`); index discipline has slipped. Resolution: [R10](07-improvement-roadmap.md#r10). |

The maintainability profile is the area where **structural strength**
and **documentation drift** diverge most. The code is well-disciplined;
the institutional documentation about the code has not kept pace and is
the main onboarding-friction surface today.

---

## §4 Testability

### §4.1 What this pack can say

- **Engine integration testing has no in-cluster directory.** The
  `CLAUDE.md`-mandated production-path rule (integration tests must use
  `ExecutionGraph.from_plugin_instances()` and
  `instantiate_plugins_from_config()`) cannot be verified from inside
  the engine cluster. Captured as finding E3 / [R4](07-improvement-roadmap.md#r4).
- **`tests/unit/engine/conftest.py:23` `MockCoalesceExecutor` carries
  an explicit "Tests bypass the DAG builder" comment.** This is fine at
  unit scope but should be verified at integration scope.

### §4.2 What this pack cannot say

The 2.9× src-to-tests ratio either represents remarkable test
discipline (the audit-grade nature of the system would warrant it) or
an inverted pyramid (a known cost-of-ownership trap). **The pack's
inputs cannot tell which.**

### §4.3 Resolution

A `ordis-quality-engineering:analyze-pyramid` pass — see
[R7](07-improvement-roadmap.md#r7). This is the highest-leverage
follow-up after the security pack.

---

## §5 The frontend gap

`src/elspeth/web/frontend/` (~13k LOC TS/React) is an architectural
component of a FastAPI-plus-SPA system. It is **outside this pack's
scope by design** — a Python-lens archaeologist cannot map TSX
usefully — but it is **inside the architectural perimeter of any
honest system description**.

### Specific consequences of leaving the gap open

- The composer cluster's "0 inbound cross-cluster edges" finding is
  structurally true at the Python-import level but **semantically
  incomplete** — the frontend consumes the composer's HTTP / MCP
  surface, and that consumption is invisible to the static analysis
  used here.
- Authentication and session-state flow on the SPA side is unanalysed.
- The frontend's coupling to backend API contracts is unverified;
  contract drift is a known source of production-grade SPA failures.

### Resolution

A frontend-aware archaeologist pass (`lyra-site-designer` or a
TypeScript / React–specialised codebase explorer) is a prerequisite for
any "complete architecture" claim. Mapped to
[R6](07-improvement-roadmap.md#r6).

---

## §6 Data integrity (audit trail)

Restated for clarity because it is the system's most consequential
property.

### The attributability test

> For any output, the operation `explain(recorder, run_id, token_id)`
> must prove complete lineage back to source data, configuration, and
> code version.

### How three subsystems jointly satisfy it

1. **`engine/`** verifies that the **terminal-state-per-token
   invariant is structurally guaranteed.**
   `engine/executors/state_guard.py:NodeStateGuard` implements "every
   row reaches exactly one terminal state" as a context-manager
   pattern, locked by tests.
2. **`core/`** verifies that the **Landscape facade pattern owns the
   audit write/read mechanics** through the four named repositories,
   and that 20 schema tables persist the row / token lifecycle.
3. **`contracts/`** verifies that the **L0 audit DTO surface is
   complete** and the L0 / L2 split is clean — engine consumes the L0
   vocabulary, core persists it.

The ADR-010 declaration-trust framework reinforces audit completeness
end-to-end. Contracts owns the L0 vocabulary; engine implements the
4-site × 7-adopter dispatcher with audit-complete (collect-then-raise)
semantics, locked by an AST-scanning unit test.

### Verification status

The prior synthesis flagged ADR-010 dispatcher audit-completeness as
the highest-stakes verification gap. The prior assessment **closed it
by direct read**. The verification record is unchanged at this pack's
HEAD; see [`06-quality-assessment.md`](06-quality-assessment.md) §1 E1.
