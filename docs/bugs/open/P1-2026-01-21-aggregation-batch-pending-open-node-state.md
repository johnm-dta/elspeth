# Bug Report: BatchPendingError leaves aggregation node_state open and batch unlinked

## Summary

- `AggregationExecutor.execute_flush()` begins a node_state and marks the batch as `executing`, but if the transform raises `BatchPendingError` the code re-raises without completing the node_state or linking the batch to that state.
- Each retry will create additional open node_states, and the batch remains `executing` with `aggregation_state_id` unset.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (main)
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: batch-aware transform that raises `BatchPendingError`

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/executors.py` for bugs and create reports
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `src/elspeth/engine/executors.py`

## Steps To Reproduce

1. Configure an aggregation node using a batch-aware transform that raises `BatchPendingError` on flush (e.g., an async batch LLM transform).
2. Run a pipeline until the aggregation trigger fires.
3. Inspect `node_states` and `batches` after the `BatchPendingError` is raised.

## Expected Behavior

- The batch is linked to the in-progress node_state (aggregation_state_id set), and the node_state is completed or otherwise tracked so retries do not leave orphaned OPEN states.

## Actual Behavior

- The node_state remains OPEN with no completion record, and the batch stays `executing` with no aggregation_state_id. Subsequent retries can create multiple OPEN node_states.

## Evidence

- Node_state is opened before the transform call:
  - `src/elspeth/engine/executors.py:906`
  - `src/elspeth/engine/executors.py:909`
- Batch status is set to `executing` without state linkage:
  - `src/elspeth/engine/executors.py:899`
  - `src/elspeth/engine/executors.py:903`
- `BatchPendingError` is re-raised without completing node_state or updating the batch:
  - `src/elspeth/engine/executors.py:929`
  - `src/elspeth/engine/executors.py:935`

## Impact

- User-facing impact: repeated retries create accumulating OPEN node_states and incomplete audit trails.
- Data integrity / security impact: batch execution is not traceable to a state record, violating auditability.
- Performance or cost impact: retries accumulate audit noise and complicate recovery.

## Root Cause Hypothesis

- The BatchPendingError path exits before completing node_state or persisting state_id for batch linkage.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/executors.py`: when `BatchPendingError` is raised, either:
    - persist `state_id` in the batch (`aggregation_state_id`) and reuse the same state on retry, or
    - defer `begin_node_state` until the batch completes, or
    - introduce a `pending` terminal status to close the node_state with explicit semantics.
- Config or schema changes: possibly add a `pending` node_state status if needed.
- Tests to add/update:
  - Add a test covering BatchPendingError to ensure no orphaned OPEN node_states and proper batch linkage.
- Risks or migration steps: none.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md` auditability principles (transform boundaries must be recorded) and batch linkage expectations.
- Observed divergence: open node_state with no completion for pending batches.
- Reason (if known): control-flow exception bypasses completion logic.
- Alignment plan or decision needed: define how pending batch work is represented in node_states.

## Acceptance Criteria

- BatchPendingError does not leave OPEN node_states in the audit trail.
- Executing batches have an `aggregation_state_id` linked to the flush attempt.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_processor.py -k batch_pending`
- New tests required: yes (BatchPendingError audit invariants).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
