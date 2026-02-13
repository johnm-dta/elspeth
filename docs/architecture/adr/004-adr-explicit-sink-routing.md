# ADR: Replace default_sink with Explicit Per-Transform Sink Routing

**Status:** Approved (with conditions)
**Bead:** `elspeth-rapid-o639`
**Date:** 2026-02-09
**Decision Makers:** Architecture Review Board
**Review Board Verdict:** Unanimous Approve (3/3) with required changes — all incorporated below

## Context

ELSPETH's routing model has an asymmetry. Error routing is explicit (`on_error: sink_name`), but success routing is implicit — every row that completes the transform chain without being gate-routed falls into a global `default_sink`. This undermines the auditability model: "where does this row go?" has an answer of "whatever the default is" rather than an explicit, per-transform declaration.

### Current Routing Model

```
Source → T1 → T2 → T3 → ??? → default_sink
                    │
                    └─ on_error → quarantine_sink (explicit)
```

Transforms can explicitly route errors but have no mechanism to explicitly route successes. Only gates can do conditional routing to named sinks.

### Problems with default_sink

1. **Implicit routing contradicts the attributability standard.** The audit trail records that a row reached `default_sink`, but the _reason_ is "nothing else claimed it" — there's no positive declaration that this row _should_ go there.

2. **Four special-case wiring blocks in dag.py.** The DAG construction has separate code paths for: last-gate continue route (lines 772-778), no-gates final edge (784-785), and coalesce terminal (828-834, duplicated). Each is a special case because the system can't express "where does this transform's output go?"

3. **~40 references across 6 core files** with fallback chains like `result.token.branch_name or default_sink_name` and `if sink_name not in pending_tokens: sink_name = default_sink_name`.

4. **`branch_name` conflation.** The fallback `result.token.branch_name or default_sink_name` in `aggregation.py` confuses lineage identity ("which fork path did this token traverse?") with routing destination ("where should this token go?"). A fork branch named "analytics" silently routes to a sink named "analytics" if one exists — even if that wasn't intended. This is an implicit name-coupling bug.

5. **No per-transform output routing.** The only way to route to different sinks is via a gate. To fork rows into two transforms that each exit to a different sink, you need gates wrapping gates — the system forces you into conditional routing when you want unconditional routing.

### Archetype: Shifting the Burden

`default_sink` is a symptomatic solution that avoids the fundamental fix (explicit routing). It works "well enough" that explicit routing was never required — but over time it accumulated 40+ fallback chains that mask routing intent. Every new routing feature (aggregation output, coalesce output, deaggregation) requires threading `default_sink_name` through yet another code path. The 40 references will become 60, then 80.

## Decision

### Add `on_success: str | None` to TransformProtocol

Complement the existing `on_error` pattern:

```python
class TransformProtocol(Protocol):
    @property
    def on_error(self) -> str | None: ...   # existing — DIVERT edge (reactive)
    @property
    def on_success(self) -> str | None: ... # new — MOVE edge (declarative)
```

**Important asymmetry:** `on_error` and `on_success` are conceptually parallel (both declare "where does output go?") but mechanically different:
- `on_error` creates `RoutingMode.DIVERT` edges — structural markers bypassed during normal execution. Routing happens reactively in the executor when `TransformResult.error()` is returned.
- `on_success` creates `RoutingMode.MOVE` edges — part of the normal execution path. Routing is declared at DAG construction time. The `on_success` edge replaces the implicit `continue` edge to `default_sink`.

### Add `on_success: str` to SourceProtocol

Sources declare where their output goes, mirroring the existing `_on_validation_failure` pattern:

```python
class SourceProtocol(Protocol):
    @property
    def _on_validation_failure(self) -> str: ...  # existing — where bad rows go
    @property
    def on_success(self) -> str: ...              # new — where good rows go
```

This handles source-only pipelines (no transforms) without special cases.

### Remove `default_sink` from ElspethSettings

The `default_sink: str` field is deleted. Every row must reach a sink through explicit routing.

### Separate `branch_name` from routing

`branch_name` on `TokenInfo` is **lineage metadata only** — it answers "which fork path did this token traverse?" It is never used for sink routing. The fallback chains `result.token.branch_name or default_sink_name` in `aggregation.py` and `outcomes.py` are deleted entirely, not refactored.

- `on_success` answers: "where does this token's output go?" (routing)
- `branch_name` answers: "which fork path did this token traverse?" (audit/lineage)

### Validation Rules

**Every terminal node in every execution path must have an explicit sink destination.**

- **Terminal transforms** (no outgoing `continue` edge in the DAG) MUST declare `on_success`
- **Non-terminal transforms** MUST NOT declare `on_success` — it is a `GraphValidationError` if a transform has `on_success` and also has a downstream `continue` edge. (Mid-chain "circuit breaker" semantics deferred to a future ADR.)
- **Sources** MUST declare `on_success`
- **Gates** continue to use their `routes:` dict (which already maps to explicit sink names or `continue`)
- **Gate `continue` at terminal position** is a validation error — must route to a named sink. `continue` remains valid mid-chain ("proceed to next transform").
- **Aggregation** config gets `on_success: sink_name` when terminal, omits it when mid-chain (downstream transforms determine the sink)
- **Coalesce** config gets `on_success: sink_name` when terminal. Only valid at terminal position — mid-chain coalesce with `on_success` is a validation error.

**"Terminal" is defined structurally:** A node is terminal if and only if `ExecutionGraph` has no outgoing edge with label `continue` from that node to another transform/gate/aggregation node. This is a graph property computed after DAG construction, not a YAML list position.

### RowOutcome Semantics

`COMPLETED` and `ROUTED` remain semantically distinct. Both carry `sink_name` on `RowResult`:

- `COMPLETED` + `sink_name` = row finished all transforms in its path, explicitly routed to its declared `on_success` sink. Normal termination.
- `ROUTED` + `sink_name` = gate made a conditional routing decision to send the row to a specific sink.

This preserves audit semantics: an auditor can distinguish "this row completed normally at its declared destination" from "this row was conditionally routed by a gate evaluation."

**New invariant:** `RowOutcome.COMPLETED` implies `RowResult.sink_name is not None`. Enforced by property test.

### Pipeline YAML

**Before:**
```yaml
default_sink: output

sinks:
  output: { plugin: csv, options: { path: results.csv } }
  quarantine: { plugin: csv, options: { path: quarantine.csv } }

source:
  plugin: csv
  options: { path: input.csv }

transforms:
  - plugin: classifier
    on_error: quarantine
    # success goes to... default_sink (implicit)
```

**After:**
```yaml
sinks:
  output: { plugin: csv, options: { path: results.csv } }
  quarantine: { plugin: csv, options: { path: quarantine.csv } }

source:
  plugin: csv
  options:
    path: input.csv
    on_success: output         # explicit — source declares output sink

transforms:
  - plugin: classifier
    on_error: quarantine
    on_success: output         # explicit — terminal transform declares sink
```

**Fork to different sinks:**
```yaml
sinks:
  approved: { plugin: csv, options: { path: approved.csv } }
  rejected: { plugin: csv, options: { path: rejected.csv } }

source:
  plugin: csv
  options:
    path: input.csv
    on_success: approved       # default path if no transforms route elsewhere

gates:
  - name: threshold
    condition: "row['score'] > 0.8"
    routes:
      "true": path_a
      "false": path_b
    fork_to: [path_a, path_b]

transforms:
  - plugin: enricher_a
    path: path_a
    on_success: approved

  - plugin: enricher_b
    path: path_b
    on_success: rejected
```

## Impact Analysis

### Quantified Blast Radius

| Area | Files | `default_sink` Refs | Mechanical? |
|------|-------|---------------------|-------------|
| Source code | 6 | 63 | Mostly — `aggregation.py` (14 refs with branch_name fallback) needs manual review |
| Unit tests | 23 | ~260 | Mixed — `test_config.py` (83), `test_dag.py` (75) largely mechanical; `test_aggregation.py` (33) needs manual review |
| Property tests | 4 | 47 | Mechanical — fixture parameter changes |
| Integration tests | 14 | ~44 | Mostly mechanical — via `build_linear_pipeline()` fixture |
| E2E tests | 7 | ~18 | Mechanical — YAML config changes |
| Performance tests | 5 | ~19 | Mechanical — pipeline fixture updates |
| Fixtures | 1 | 11 | **Critical** — `tests/fixtures/pipeline.py` is the single leverage point |
| Example YAMLs | 30 | 31 | Mechanical — add `on_success:`, remove `default_sink:` |
| **TOTAL** | **50 test files + 30 examples** | **~472 refs** | **~70% mechanical, ~30% manual** |

### Protocol & Config Layer

| File | Change | Scope |
|------|--------|-------|
| `plugins/protocols.py` | Add `on_success` property to `TransformProtocol`, `BatchTransformProtocol`, `SourceProtocol` | 3 additions |
| `core/config.py` | Add `on_success` to `TransformSettings` and `SourceSettings`; remove `default_sink` from `ElspethSettings`; remove `validate_default_sink_exists` validator | ~15 lines |
| `contracts/config/` | Add `on_success` to config alignment; run `check_contracts` | Required |

### DAG Construction

| File | Change | Scope |
|------|--------|-------|
| `core/dag.py` | Remove `default_sink` parameter from `from_plugin_instances()`; replace 4 special-case wiring blocks with `on_success` MOVE edge creation; add terminal-node validation; validate `on_success` only on terminal nodes | ~50 lines |

### Engine (Processor + Executors)

| File | Change | Scope |
|------|--------|-------|
| `engine/processor.py` | Thread `sink_name` from transform's `on_success` into `RowResult(COMPLETED, sink_name=...)` at 5 construction sites | ~20 lines |
| `engine/executors.py` | No change — executor returns result, routing decision is structural | 0 |

### Orchestrator

| File | Change | Scope |
|------|--------|-------|
| `engine/orchestrator/outcomes.py` | Remove `default_sink_name` parameter from 3 functions; use `result.sink_name` exclusively; delete `branch_name` fallback chains | ~30 lines |
| `engine/orchestrator/aggregation.py` | Remove `default_sink_name` from 2 functions (14 refs); delete `branch_name or default_sink_name` fallback chains; use explicit sink from aggregation config or downstream transform's `on_success` | ~40 lines |
| `engine/orchestrator/core.py` | Remove `default_sink_name` variable; remove threading of default sink | ~15 lines |

### CLI & Config Loading

| File | Change | Scope |
|------|--------|-------|
| `cli.py` | Remove 4 `default_sink=` references in pipeline construction | ~4 lines |

## Risks

### R1: Blast radius (MEDIUM)

472 references across 50 test files + 30 example YAMLs. Mitigated by:
- ~70% mechanical transformation
- DAG validation catches missing `on_success` at construction time, not runtime
- No legacy code policy means no backwards compatibility needed
- `tests/fixtures/pipeline.py` is the single leverage point — updating it cascades to 20+ downstream test files

### R2: Aggregation complexity (HIGH)

`aggregation.py` has 14 `default_sink_name` references across two large near-duplicate functions (`check_aggregation_timeouts` and `flush_remaining_aggregation_buffers`) with multiple outcome branches and the most complex fallback chains. **Address this file first in implementation.** Consider extracting a shared `_route_aggregation_outcome()` helper BEFORE the migration so the migration only touches one place.

### R3: RowResult.sink_name responsibility shift (MEDIUM)

The processor comment at line 2042-2043 says: "Orchestrator knows branch→sink mapping, processor does not." This change reverses that — the processor must now resolve `sink_name` from the transform's `on_success`. This is a fundamental responsibility shift from orchestrator to processor.

### R4: Checkpoint/resume compatibility (LOW)

Adding `on_success` to transform config changes the config hash, which changes node IDs, which breaks checkpoint compatibility with runs started before the change. This is acceptable pre-release. Runs checkpointed before migration cannot be resumed after migration — they must be re-run. Document this in release notes.

### R5: Config contracts alignment (LOW if tested)

The P2-2026-01-21 `exponential_base` bug showed the risk: a Settings field added but never mapped to runtime. `on_success` must flow from `TransformSettings` → `TransformProtocol` → DAG construction → processor routing. Run `check_contracts` as a CI gate. Add to config alignment tests.

## Implementation

### Atomic Change (No Phased Rollout)

Per CLAUDE.md's No Legacy Code Policy: "WE HAVE NO USERS YET." A phased rollout triples implementation effort and introduces a dual-path routing period (Phase 2) that recreates the exact Shifting the Burden pattern this ADR eliminates. The phased rollout's fallback logic is the most bug-prone part of the entire change.

**Do it atomically in one commit.** Add `on_success`, remove `default_sink`, update all tests and examples.

### Implementation Order

1. **Extract `_route_aggregation_outcome()` helper** in `aggregation.py` — consolidate the two near-duplicate functions before the migration
2. **Add `on_success` to protocols, config, and config alignment tests** — run `check_contracts`
3. **Update DAG construction** — replace 4 special-case wiring blocks with `on_success` MOVE edge creation; add terminal-node validation
4. **Update processor** — thread `sink_name` into `RowResult(COMPLETED)` at 5 construction sites
5. **Update outcomes and aggregation** — delete `default_sink_name` parameters and `branch_name` fallback chains
6. **Update orchestrator/core** — remove `default_sink_name` variable
7. **Update CLI** — remove 4 `default_sink=` references
8. **Update `tests/fixtures/pipeline.py`** — single leverage point that cascades to integration/E2E tests
9. **Update remaining tests** — mechanical find-and-replace for unit/property tests
10. **Update 30 example YAMLs** — add `on_success:`, remove `default_sink:`
11. **Final verification** — grep codebase for `default_sink`; if any reference exists, migration is incomplete

### Required Safeguards

1. **Config contracts alignment test** — verify `on_success` flows end-to-end, just like `exponential_base` fix
2. **Property test** — `forall RowResult: outcome == COMPLETED => sink_name is not None`
3. **Property test** — `forall RowResult: sink_name in configured_sinks`
4. **DAG validation** — `on_success` references unknown sink → `GraphValidationError`
5. **DAG validation** — terminal node without `on_success` → `GraphValidationError`
6. **DAG validation** — non-terminal node with `on_success` → `GraphValidationError`
7. **Zero `default_sink` grep** — post-implementation verification that no references remain

### Required Tests (Priority Order)

| Test | Category | Priority |
|------|----------|----------|
| Terminal transform without `on_success` fails DAG validation | Unit (DAG) | P0 |
| Non-terminal transform with `on_success` fails DAG validation | Unit (DAG) | P0 |
| `on_success` references unknown sink fails DAG validation | Unit (DAG) | P0 |
| Gate `continue` at terminal position fails DAG validation | Unit (DAG) | P0 |
| `COMPLETED` RowResult always has `sink_name` set | Property | P0 |
| Every `sink_name` in results is in configured sinks | Property | P0 |
| `branch_name` never determines sink routing | Property | P0 |
| `default_sink` field in YAML raises validation error | Unit (Config) | P0 |
| Completed row carries explicit `sink_name` end-to-end | Integration | P0 |
| Fork branches with different `on_success` route correctly | Integration | P1 |
| Coalesce output routes to declared `on_success` sink | Integration | P1 |
| Aggregation flush routes to `on_success` sink | Integration | P1 |
| Checkpoint/resume with `on_success` routes identically | Integration | P1 |

## Future Extensions (Deferred)

- **Mid-chain `on_success` ("circuit breaker")** — allow a mid-chain transform to declare `on_success`, making it structurally terminal. Requires dead-code detection (unreachable downstream transforms) and reachability validation. Deferred to a separate ADR.
- **`elspeth scaffold` command** — generate YAML templates with `on_success` pre-wired to reduce configuration friction.

## Review Board Decisions

| Question | Decision | Vote |
|----------|----------|------|
| Q1: Source-only pipelines | Option (a): `on_success` on `SourceProtocol` | 2-1 (Architect + Systems vs QA) |
| Q2: Gate `continue` at terminal | Option (a): validation error | Unanimous |
| Q3: RowOutcome semantics | Keep `COMPLETED` + `sink_name` distinct from `ROUTED` + `sink_name` | Unanimous |
| Q4: Coalesce output routing | Option (a): `on_success` on coalesce config (terminal only) | Unanimous |
| Q5: Aggregation output routing | Option (a): `on_success` when terminal, omit when mid-chain | Unanimous |
| `branch_name` vs `on_success` | `on_success` is sole routing authority; `branch_name` is lineage metadata only | Unanimous |
| Phased rollout | Rejected — atomic change per No Legacy Code Policy | Unanimous |
| Mid-chain `on_success` | Deferred to future ADR — terminal-only for V1 | Unanimous |
| `on_error`/`on_success` symmetry | Conceptually parallel, mechanically different (DIVERT vs MOVE) — document, don't mirror | Unanimous |

## References

- `on_error` implementation: `plugins/protocols.py:207`, `engine/executors.py:448-491`, `core/dag.py:813-823`
- Gate routing: `engine/executors.py:504-700`, `core/config.py:292-441`
- Default sink wiring: `core/dag.py:772-785`, `core/dag.py:828-834`
- Outcome accumulation: `engine/orchestrator/outcomes.py:58-88`
- Aggregation fallback chains: `engine/orchestrator/aggregation.py:198,227,346,379`
- Config contracts pattern: `contracts/config/protocols.py`, `scripts/check_contracts`
- Prior field-orphaning bug: P2-2026-01-21 (`exponential_base`)
- Prior test-path divergence bug: BUG-LINEAGE-01
