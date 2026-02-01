# Test Defect Report

## Summary

- Missing coverage for the branch that preserves pre-assigned transform node_ids (aggregation transforms), so regression could override or error without test failure

## Severity

- Severity: major
- Priority: P1

## Category

- [Missing Edge Cases]

## Evidence

- Implementation explicitly skips assignment when a transform already has a `node_id`, which is the aggregation path: `src/elspeth/engine/orchestrator.py:417`.
```python
for seq, transform in enumerate(transforms):
    if transform.node_id is not None:
        # Already has node_id (e.g., aggregation transform) - skip
        continue
```
- All tests set transform `node_id` to `None`, so the skip branch is never exercised: `tests/engine/test_node_id_assignment.py:61`, `tests/engine/test_node_id_assignment.py:153`.
```python
t1 = MagicMock()
t1.node_id = None
t2 = MagicMock()
t2.node_id = None
```
- Missing example: no test asserts that a transform with pre-set `node_id` keeps that value and does not require a `transform_id_map` entry.

## Impact

- A regression that removes or changes the skip branch could overwrite aggregation `node_id`s or raise `ValueError` when `transform_id_map` omits pre-assigned sequences.
- This would break DAG lineage and auditability for aggregation transforms while tests still pass, creating false confidence.

## Root Cause Hypothesis

- Tests focus on the default assignment path and error cases, but overlooked the aggregation-specific branch highlighted in the implementation comment.

## Recommended Fix

- Add a test in `tests/engine/test_node_id_assignment.py` that:
  - Creates a transform with `node_id` already set (e.g., `"agg-1"`).
  - Calls `_assign_plugin_node_ids` with no `transform_id_map` entry for that sequence.
  - Asserts the pre-set `node_id` remains unchanged and no `ValueError` is raised.
- Optional: include a second transform with `node_id = None` to ensure mixed behavior still assigns from `transform_id_map`.
- Priority justification: aggregation node IDs are core to lineage; missing coverage leaves a high-risk branch unprotected.
