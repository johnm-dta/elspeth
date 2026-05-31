# Known Gaps

> **Posture:** named, not hidden. Any "complete architecture" claim
> about ELSPETH requires the gaps below to be closed by downstream
> packs. Each gap names what is missing, why it was deferred, and which
> recommendation in [`07-improvement-roadmap.md`](07-improvement-roadmap.md)
> resolves it.

---

## §1 Frontend (`web/frontend/`)

### What is missing

`src/elspeth/web/frontend/` (~13k LOC TypeScript / React) is **outside
this pack's coverage**. No component map, no state-flow analysis, no
API-consumption verification, no auth/session-handling review.

### Why deferred

A Python-lens archaeologist cannot map TSX usefully. The pack's
methodology — import-graph oracle, layer-conformance scanner, file-level
LOC counting — does not apply to a React SPA.

### Specific consequences

- The composer cluster's "0 inbound cross-cluster edges" finding is
  structurally true at the Python-import level but **semantically
  incomplete**: the frontend consumes the composer's HTTP/MCP surface,
  invisible to the static analysis used here.
- Authentication and session-state flow on the SPA side is unanalysed.
- Backend-API-contract drift is a known SPA failure mode and is
  unverified.
- Any defence-in-depth claim about the FastAPI-plus-SPA system is
  necessarily incomplete.

### Resolution

[R6](07-improvement-roadmap.md#r6) — frontend-aware archaeologist pass
(`lyra-site-designer` or a TypeScript/React-specialised explorer).

### Owner

Frontend specialist pack lead; not the architecture pack.

---

## §2 Test architecture (`tests/`)

### What is missing

The `tests/` tree (~351k LOC across ~851 files) is **outside this
pack's coverage**. No test-pyramid analysis, no fixture-topology map,
no production-path verification.

### Why deferred

Test architecture is a separate deliverable. The institutional
documentation set treats `tests/` as out of scope from the production
source tree, and the LOC ratio (2.9× src/) makes it a comparable-effort
deliverable in its own right.

### Specific consequences

- Whether the 2.9× src-to-tests ratio represents **remarkable test
  discipline** (the audit-grade nature of the system would warrant it)
  or an **inverted pyramid** (a known cost-of-ownership trap) is
  unanswered.
- The `CLAUDE.md`-mandated production-path rule (integration tests must
  use `ExecutionGraph.from_plugin_instances()` and
  `instantiate_plugins_from_config()`) is currently un-auditable from
  inside any single cluster (this is also recorded as finding E3 / R4
  for engine-specific scope).
- The audit guarantees the codebase makes are partially load-bearing on
  the test suite's design; without a test-architecture pass, the
  guarantees are only as strong as the unverified test discipline that
  upholds them.

### Resolution

[R7](07-improvement-roadmap.md#r7) — `ordis-quality-engineering:analyze-pyramid`
pass (or equivalent).

### Owner

Quality-engineering pack lead; not the architecture pack.

---

## §3 Per-file cohesion of 13 ≥1,500-LOC files

### What is missing

This pack flags 13 files that exceed 1,500 LOC and **does not assess
their internal cohesion**. The "essential complexity vs accidental
concentration" question on each file is open.

### Why deferred

Per-file deep-dives are a separate methodology (axiom-system-archaeologist
per-file pass) that requires reading the full file at component depth.
The cluster-level analysis that produced this pack stops at the file
boundary by design — its job is to map the system, not to dissect each
large file.

### Files

See [`07-improvement-roadmap.md#r3`](07-improvement-roadmap.md#r3) for
the full roster (13 files, ranging from 1,566 to 3,860 LOC).

### Resolution

[R5](07-improvement-roadmap.md#r5) — per-file deep-dives, prioritised
on `engine/processor.py` and `core/config.py` first because they have
the highest blast radius and gate other architectural decisions.

### Owner

Architecture-archaeologist per-file pass (multiple sessions).

---

## §4 Performance characteristics

### What is missing

No profiling, no synthetic workload analysis, no production-trace
review. Performance assertions cannot be made from this pack's input
set.

### Why deferred

Performance is a profile-driven discipline that requires representative
workloads. The architecture pack identifies architecturally-hot paths
(those visible in the import graph and call topology); whether they
are actually hot in production cannot be answered from static analysis.

### Specific consequences

- `engine/processor.py` (2,700 LOC) handles per-row processing. It is
  an architecturally-hot path candidate. Whether it dominates wall-clock
  time, or whether wall-clock is dominated by source I/O or LLM-provider
  latency, is unknown.
- The `coalesce_executor.py` policies (4 policies × 3 strategies × branch-loss × late-arrivals × checkpoint resume)
  are essential complexity but their performance characteristics under
  load are unverified.

### Resolution

A separate profiling-driven pass (`axiom-python-engineering:profile`)
with representative workloads. Not in this pack's roadmap because
performance work belongs to an operational specialist pack, not an
architecture review.

### Owner

Performance specialist pack; not the architecture pack.

---

## §5 Operational quality

### What is missing

CI/CD topology, deployment patterns, observability outside Landscape,
on-call ergonomics. Out of scope for this pack.

### Why deferred

Operational quality requires interviews with on-call engineers, review
of runbook completeness, and analysis of incident history — none of
which are static-analysis deliverables.

### Specific consequences

- The Landscape audit DB is well-characterised as the legal record;
  whether on-call engineers can navigate it under pressure is unknown.
- Telemetry primacy ordering (Landscape → telemetry → logging) is
  documented and partially CI-enforced; whether the operational
  consequences of the discipline (e.g., debugging without `slog.info`
  for pipeline activity) are net-positive in practice is unknown.

### Resolution

A separate operational-readiness pass; not in this pack's roadmap
because it is downstream of the security pack (R8).

---

## §6 The HEAD-drift caveat

This pack ships against codebase HEAD `5a5e05d7` (2026-04-29). The
underlying analysis was performed at HEAD `47d3dd82`. Structural claims
have been re-verified at this pack's HEAD; minor numerical drift is
documented.

### Drift summary

| Metric | Analysis HEAD | This pack's HEAD | Implication |
|--------|--------------|------------------|-------------|
| Total Python LOC | ~121,408 | ~122,554 (+1,146) | Cosmetic |
| L3 import edges | 77 | 79 (+2) | Two new edges in the composer/execution corner of SCC#4 |
| `web/sessions/routes.py` LOC | 1,563 | 2,067 (+504, +32%) | The file the prior inventory missed has grown materially — strengthens R3 |
| `web/composer/tools.py` LOC | 3,804 | 3,860 (+56) | Cosmetic |
| 4-layer model conformance | Clean | Clean | Re-verified |
| SCC count | 5 | 5 | Identical |
| Largest SCC size | 7 (web cluster) | 7 (same nodes) | Identical |
| Heaviest L3 edge | `plugins/sinks → plugins/infrastructure` w=45 | Same | Identical |

**Verdict:** all structural claims hold; numerical claims are
re-derived against this pack's HEAD throughout. To re-verify before
relying on a specific finding, follow [`reference/re-derive.md`](reference/re-derive.md).
