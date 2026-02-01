# Bug Report: Aggregation Flush Bypasses Downstream Gates And Aggregations

## Summary

- Batch-aware aggregation returns terminal results without enqueuing downstream steps, so config gates and later aggregations never execute when a batch flushes.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-4 @ 86357898ee109a1dbb8d60f3dc687983fa22c1f0
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of /home/john/elspeth-rapid/src/elspeth/engine/processor.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure a pipeline with at least one aggregation (any output_mode) and at least one config gate (or a second aggregation) downstream.
2. Run the pipeline until the aggregation flushes.
3. Inspect node_states/routing events or outputs for the aggregated token.

## Expected Behavior

- Aggregated output continues through remaining aggregations and config gates, producing routing events and node_state records.

## Actual Behavior

- Aggregated output returns RowOutcome.COMPLETED (or terminal results) immediately and skips downstream gates/aggregations.

## Evidence

- `src/elspeth/engine/processor.py:728`
- `src/elspeth/engine/processor.py:226`
- `src/elspeth/engine/processor.py:265`
- `src/elspeth/engine/processor.py:864`
- `src/elspeth/core/dag.py:416`

## Impact

- User-facing impact: routing decisions and downstream transforms are skipped for aggregated outputs.
- Data integrity / security impact: audit trail misses expected gate evaluations and node_states.
- Performance or cost impact: Unknown.

## Root Cause Hypothesis

- `_process_batch_aggregation_node` treats a flush as terminal and returns without scheduling continuation, while config gates are only processed after the transform loop.

## Proposed Fix

- Code changes (modules/files):
  - Update `_process_batch_aggregation_node` in `src/elspeth/engine/processor.py` to enqueue work items when downstream steps exist (remaining transforms or any config gates), including output_mode `single`.
- Config or schema changes: None.
- Tests to add/update:
  - Add tests in `tests/engine/test_processor.py` for aggregation followed by config gates and for multiple aggregations chained.
- Risks or migration steps:
  - Low risk; affects aggregation continuation and gate execution order.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/engine/processor.py:62`
- Observed divergence: pipeline order says transforms -> config gates -> sink, but aggregated tokens never reach config gates.
- Reason (if known): `_process_batch_aggregation_node` returns terminal results without checking downstream steps.
- Alignment plan or decision needed: ensure aggregation flush schedules downstream processing before marking outcomes terminal.

## Acceptance Criteria

- Aggregated outputs execute downstream aggregations and config gates as defined by the DAG.
- Routing events for config gates exist in the audit trail after aggregation flush.
- No premature RowOutcome.COMPLETED when gates remain.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/`
- New tests required: yes, aggregation + config-gate sequencing and multi-aggregation chaining.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `ARCHITECTURE.md`
---
# Bug Report: Batch Aggregation Masks Missing Output Row With Empty Dict

## Summary

- For output_mode `single` or `transform`, missing output rows are replaced with `{}`, masking plugin contract violations and corrupting output data.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-4 @ 86357898ee109a1dbb8d60f3dc687983fa22c1f0
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of /home/john/elspeth-rapid/src/elspeth/engine/processor.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Implement a batch-aware transform that returns `TransformResult.success_multi(...)` while aggregation output_mode is `single`, or otherwise returns a success without `row`.
2. Run a pipeline that triggers an aggregation flush.
3. Inspect the output row for the aggregated token.

## Expected Behavior

- The pipeline should raise an error for missing output data in `single`/`transform` modes.

## Actual Behavior

- The processor substitutes `{}` and continues, emitting an empty row.

## Evidence

- `src/elspeth/engine/processor.py:227`
- `src/elspeth/engine/processor.py:313`
- `src/elspeth/contracts/results.py:70`
- `src/elspeth/contracts/results.py:98`

## Impact

- User-facing impact: silent emission of empty rows instead of crashing on plugin bugs.
- Data integrity / security impact: audit trail records incorrect output data.
- Performance or cost impact: Unknown.

## Root Cause Hypothesis

- `_process_batch_aggregation_node` uses fallback empty dicts instead of enforcing TransformResult output contracts.

## Proposed Fix

- Code changes (modules/files):
  - In `src/elspeth/engine/processor.py`, enforce `result.row is not None` for `single` and `transform` single-row outputs and raise if missing; reject multi-row results in `single`.
- Config or schema changes: None.
- Tests to add/update:
  - Add tests in `tests/engine/test_processor.py` ensuring aggregation raises on missing row outputs for `single`/`transform`.
- Risks or migration steps:
  - Medium risk for pipelines relying on buggy transforms; intended to surface existing defects.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/contracts/results.py:70`
- Observed divergence: success results should provide row data, but processor tolerates missing output.
- Reason (if known): defensive fallback to `{}` in aggregation handling.
- Alignment plan or decision needed: enforce TransformResult contracts and fail fast.

## Acceptance Criteria

- Missing output data in batch transforms raises an error instead of emitting `{}`.
- Aggregation output conforms to TransformResult contract for all modes.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/`
- New tests required: yes, aggregation output contract enforcement.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `src/elspeth/contracts/results.py`
---
# Bug Report: Transform-Mode Aggregation Records CONSUMED_IN_BATCH Without Batch ID

## Summary

- The triggering token in output_mode `transform` is recorded as CONSUMED_IN_BATCH after the flush, but the batch_id has already been cleared, leaving a null batch_id in the audit trail.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-4 @ 86357898ee109a1dbb8d60f3dc687983fa22c1f0
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of /home/john/elspeth-rapid/src/elspeth/engine/processor.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure an aggregation with output_mode `transform` and a count trigger of 1.
2. Run a pipeline that flushes batches.
3. Inspect `token_outcomes` for the triggering token.

## Expected Behavior

- CONSUMED_IN_BATCH outcomes include a valid batch_id for all consumed tokens.

## Actual Behavior

- The triggering token’s CONSUMED_IN_BATCH outcome has batch_id = NULL.

## Evidence

- `src/elspeth/engine/processor.py:329`
- `src/elspeth/engine/executors.py:1049`
- `src/elspeth/engine/executors.py:1067`
- `src/elspeth/core/landscape/recorder.py:2233`

## Impact

- User-facing impact: explain/batch lineage for the triggering token is incomplete.
- Data integrity / security impact: audit trail missing batch linkage for one token per batch.
- Performance or cost impact: Unknown.

## Root Cause Hypothesis

- `_process_batch_aggregation_node` retrieves `batch_id` after `execute_flush()` resets batch state, so `get_batch_id()` returns None.

## Proposed Fix

- Code changes (modules/files):
  - Capture batch_id before `execute_flush()` in `src/elspeth/engine/processor.py` and use that value when recording CONSUMED_IN_BATCH for the triggering token, or return batch_id from `execute_flush()`.
- Config or schema changes: None.
- Tests to add/update:
  - Add test ensuring batch_id is populated for the triggering token in transform-mode aggregations.
- Risks or migration steps:
  - Low risk; improves audit linkage.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/core/landscape/recorder.py:2233`
- Observed divergence: batch_id is required for CONSUMED_IN_BATCH outcomes but is missing for the triggering token.
- Reason (if known): batch state reset before batch_id is read.
- Alignment plan or decision needed: record batch_id before state reset.

## Acceptance Criteria

- All CONSUMED_IN_BATCH outcomes (including the triggering token) have a non-null batch_id.
- Audit trail links every consumed token to its batch.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/`
- New tests required: yes, transform-mode aggregation batch_id coverage.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `src/elspeth/core/landscape/recorder.py`
---
# Bug Report: Aggregation Passthrough/Transform Drops Coalesce Metadata For Forked Tokens

## Summary

- When batch aggregation emits child work items (passthrough/transform modes), it does not propagate coalesce_name/coalesce_at_step, so forked tokens never reach coalesce.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-4 @ 86357898ee109a1dbb8d60f3dc687983fa22c1f0
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of /home/john/elspeth-rapid/src/elspeth/engine/processor.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Add a plugin gate before a batch aggregation that forks to branches mapped to a coalesce point.
2. Configure the aggregation output_mode as `passthrough` or `transform`.
3. Run a pipeline and observe coalesce behavior for forked tokens.

## Expected Behavior

- Forked tokens should carry coalesce metadata through aggregation and reach COALESCED outcomes.

## Actual Behavior

- Tokens bypass coalesce because coalesce_name/coalesce_at_step are dropped when new work items are created.

## Evidence

- `src/elspeth/engine/processor.py:42`
- `src/elspeth/engine/processor.py:688`
- `src/elspeth/engine/processor.py:270`
- `src/elspeth/engine/processor.py:348`
- `src/elspeth/engine/processor.py:945`

## Impact

- User-facing impact: join semantics break; forked branches do not merge.
- Data integrity / security impact: audit trail records incorrect lineage (no COALESCED state).
- Performance or cost impact: Unknown.

## Root Cause Hypothesis

- `_process_batch_aggregation_node` creates new `_WorkItem` instances without coalesce metadata, and there is no recomputation from branch_name.

## Proposed Fix

- Code changes (modules/files):
  - Propagate `coalesce_name` and `coalesce_at_step` into `_process_batch_aggregation_node` and include them in `_WorkItem` creation, or recompute from `token.branch_name` using `self._branch_to_coalesce`/`self._coalesce_step_map`.
- Config or schema changes: None.
- Tests to add/update:
  - Add a test covering fork -> aggregation -> coalesce to ensure COALESCED outcomes occur.
- Risks or migration steps:
  - Low risk; affects coalesce routing only when aggregations are in the forked path.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/engine/processor.py:945`
- Observed divergence: coalesce check requires coalesce metadata, but aggregation work items drop it.
- Reason (if known): child work items only include token/start_step in aggregation code paths.
- Alignment plan or decision needed: carry or recompute coalesce metadata for downstream processing.

## Acceptance Criteria

- Forked tokens passing through aggregation still reach coalesce when configured.
- COALESCED outcomes are recorded for such paths.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/`
- New tests required: yes, fork/aggregation/coalesce integration test.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `src/elspeth/core/dag.py`
