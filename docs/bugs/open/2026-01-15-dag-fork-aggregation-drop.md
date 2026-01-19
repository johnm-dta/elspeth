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

1. Create a gate plugin that returns `RoutingAction.fork_to_paths(["path_a", "path_b"])`.
2. Configure a pipeline with that gate and sinks for each path.
3. Run a pipeline execution and observe sink outputs and counters.

(Alternative) Configure an aggregation plugin that accepts rows and triggers a flush, then run a pipeline and observe downstream outputs.

## Expected Behavior

- Forked child tokens are enqueued for continued processing, and outputs reach the intended sinks.
- Aggregation outputs are processed downstream (or the system fails fast if DAG execution is unsupported).

## Actual Behavior

- `RowProcessor` returns `outcome="forked"` or `outcome="consumed"`, but `Orchestrator._execute_run` ignores those outcomes, dropping tokens silently.

## Evidence

- `src/elspeth/engine/processor.py:108-183` returns `"forked"` and `"consumed"` outcomes with an explicit note that the caller must handle them.
- `src/elspeth/engine/orchestrator.py:349-381` only handles `"completed"`, `"routed"`, and `"failed"` outcomes; no handling for fork/aggregation outcomes.
- `src/elspeth/plugins/results.py:19-32` documents the invariant: “No silent drops.”

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
