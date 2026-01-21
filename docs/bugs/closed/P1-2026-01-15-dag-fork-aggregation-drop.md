# Bug Report: Forked/aggregated tokens dropped by orchestrator

## Summary

- ~~Forked and aggregated outcomes are returned by `RowProcessor` but ignored by `Orchestrator`, causing silent token drops and incomplete lineage.~~
- **UPDATE 2026-01-19:** Significant progress made. Orchestrator now handles fork outcomes and routes fork children to branch-named sinks. Needs verification that all fork children reach terminal states.

## Severity

- Severity: major (downgraded from critical - partial fix in place)
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-15
- Related run/issue ID: N/A

## Environment

- Commit/branch: not checked
- OS: not checked
- Python version: not checked
- Config profile / env vars: not checked
- Data set or fixture: not checked

## Agent Context (if relevant)

- Goal or task prompt: deep dive into codebase, identify a bug, do RCA, compare to architecture, propose fix
- Model/version: GPT-5 (Codex)
- Tooling and permissions (sandbox/approvals): sandbox read-only, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: inspected `src/elspeth/engine/processor.py`, `src/elspeth/engine/orchestrator.py`, `docs/design/architecture.md`

## Steps To Reproduce

1. Configure a **config gate** that routes to `fork`, e.g. `routes: {all: fork}` with `fork_to: [path_a, path_b]`, and define sinks for `path_a` and `path_b`.
2. Run a pipeline execution and observe sink outputs and counters for branch-named sinks.

(Alternative) Configure an aggregation (via `aggregations:`) that accepts rows and triggers a flush, then run a pipeline and observe downstream outputs.

## Expected Behavior

- Forked child tokens are enqueued for continued processing, and outputs reach the intended sinks.
- Aggregation outputs are processed downstream (or the system fails fast if DAG execution is unsupported).

## Actual Behavior

- ~~`RowProcessor` returns `outcome="forked"` or `outcome="consumed"`, but `Orchestrator` ignores those outcomes, dropping tokens silently.~~
- **UPDATE 2026-01-19:** Fork outcomes are now handled and fork children are routed to branch-named sinks. Aggregation flush outputs are still not propagated as downstream work (flush does not emit sink-bound tokens/results).

## Evidence

- Fork outcomes and aggregation consumption outcomes are produced by the processor:
  - Fork: `src/elspeth/engine/processor.py:328-349`
  - Aggregation consume: `src/elspeth/engine/processor.py:350-379`
- Orchestrator now handles fork outcomes and routes fork children to branch-named sinks: `src/elspeth/engine/orchestrator.py:581-637`
- Aggregation flush currently does not enqueue downstream work (flush has no return path into `Orchestrator`'s sink buffer): `src/elspeth/engine/processor.py:365-370`
- Invariant: no silent drops: `src/elspeth/plugins/results.py:19-32`

## Impact

- User-facing impact: forked/aggregated paths never reach sinks; outputs are missing without error.
- Data integrity / security impact: lineage is incomplete; routing events exist without corresponding downstream node states.
- Performance or cost impact: wasted compute on gate evaluation; potential reprocessing costs due to missing outputs.

## Root Cause Hypothesis

- Orchestrator executes a linear pipeline and lacks a DAG work queue for forked child tokens or aggregation outputs. The runtime partially supports routing at the executor level but never consumes those results at the orchestration level.

## Proposed Fix

- Code changes (modules/files):
  - Implement a DAG-aware work queue in `src/elspeth/engine/orchestrator.py` to enqueue child tokens and aggregation outputs.
  - Alternatively, fail fast when `RowProcessor` returns `"forked"` or `"consumed"` to prevent silent drops.
- Config or schema changes: none.
- Tests to add/update:
  - Add orchestrator tests for `fork_to_paths` routing and aggregation flush handling in `tests/engine/test_orchestrator.py`.
- Risks or migration steps:
  - Introducing a work queue changes execution order and checkpoint semantics; update checkpoint logic accordingly.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/architecture.md#L248` (routing decisions must be captured) and `docs/design/architecture.md#L166` (pipelines compile to DAGs).
- Observed divergence: forked/aggregated tokens are created but never processed; no sink output or lineage continuation.
- Reason (if known): linear execution path retained while DAG features were added incrementally.
- Alignment plan or decision needed: decide whether to implement DAG execution now or explicitly disable unsupported routing outcomes with a hard error.

## Acceptance Criteria

- Forked tokens are processed through downstream transforms and reach the intended sinks (or the system errors clearly when fork/aggregation is used).
- Aggregation outputs are emitted and processed downstream with complete audit trail.
- New tests cover fork and aggregation outcomes at the orchestrator level.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py`
- New tests required: yes, for forked and aggregation outcomes.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md`

## Triage Update (2026-01-19)

**Status:** Promoted from pending to open for verification.

**Evidence of partial fix:**
- `src/elspeth/engine/orchestrator.py:566`: Tracks `rows_forked` counter
- `src/elspeth/engine/orchestrator.py:631-634`: Handles `RowOutcome.FORKED`
- `src/elspeth/engine/orchestrator.py:602-609`: Fork children route to branch-named sinks
- `src/elspeth/engine/processor.py:322-338`: Processor generates fork results with proper child tokens

**What needs verification:**
1. Fork children actually reach terminal states (COMPLETED/ROUTED)
2. Aggregation flush outputs are processed downstream
3. Integration tests cover multi-branch fork scenarios

**Next steps:** Write integration test that verifies fork child tokens appear in sink outputs.

---

## Resolution

**Status:** CLOSED (VERIFIED FIXED)
**Date:** 2026-01-21
**Verified by:** Claude Opus 4.5

### Verification Summary

The bug was fully fixed by commits made after the original report:

**Key commits:**
- `8d1c54a fix(engine): flush aggregation buffers at end-of-source to prevent data loss`
- `1c8300a refactor(engine): extract aggregation flush to helper and fix silent data loss bug`

### Evidence of Fix

**1. Fork handling verified:**
- Test `test_full_pipeline_with_fork_writes_all_children_to_sink` passes ✓
- Orchestrator handles `RowOutcome.FORKED` and routes children to branch-named sinks
- Lines 780-781, 1315-1316 handle FORKED outcomes

**2. Aggregation flush to sink verified:**
- Method `_flush_remaining_aggregation_buffers()` (lines 1443-1565) properly routes flush outputs
- Line 1533: `pending_tokens[output_sink_name].append(output_token)` - single row output
- Lines 1550-1551: Multi-row outputs via `expand_token` added to `pending_tokens`
- `pending_tokens` are written to sinks in lines 850-851, 1357-1358

**3. All integration tests pass:**
- 25/25 integration tests pass including:
  - `test_full_pipeline_with_fork_writes_all_children_to_sink`
  - `test_fork_coalesce_writes_merged_to_sink`
  - `test_diamond_dag_fork_transform_coalesce`
  - `test_explain_for_aggregated_row`
- 42/42 aggregation tests pass
- 467/467 engine tests pass

### Acceptance Criteria Met

✅ Forked tokens are processed through downstream transforms and reach the intended sinks
✅ Aggregation outputs are emitted and processed downstream with complete audit trail
✅ Integration tests cover fork and aggregation scenarios
