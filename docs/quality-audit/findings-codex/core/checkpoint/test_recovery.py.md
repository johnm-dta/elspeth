# Test Defect Report

## Summary

- `test_get_resume_point` only checks that resume fields are non-null/positive instead of asserting the exact checkpoint-derived values, allowing incorrect resume points to pass

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/core/checkpoint/test_recovery.py:89` creates a checkpoint with explicit token/node/sequence values that should be validated.
- `tests/core/checkpoint/test_recovery.py:197` only asserts non-null/positive fields, not the expected values.

```python
# tests/core/checkpoint/test_recovery.py
checkpoint_manager.create_checkpoint(run_id, "tok-001", "node-001", 1, graph=mock_graph)

resume_point = recovery_manager.get_resume_point(failed_run_with_checkpoint, mock_graph)
assert resume_point is not None
assert resume_point.token_id is not None
assert resume_point.node_id is not None
assert resume_point.sequence_number > 0
```

- Missing: assertions like `resume_point.token_id == "tok-001"`, `resume_point.node_id == "node-001"`, `resume_point.sequence_number == 1`.

## Impact

- A regression returning the wrong token/node/sequence (but still non-null/positive) would pass, potentially resuming at the wrong point.
- This can mask bugs that skip or reprocess rows, undermining recovery correctness and audit integrity.
- Creates false confidence in the resume-point correctness.

## Root Cause Hypothesis

- Tests were written as minimal smoke checks and never tightened to validate exact expectations from the fixture setup.
- Likely focused on presence/shape rather than correctness of critical values.

## Recommended Fix

- Strengthen `test_get_resume_point` to assert exact values derived from the fixture checkpoint.
- Example:
  ```python
  resume_point = recovery_manager.get_resume_point(failed_run_with_checkpoint, mock_graph)
  assert resume_point is not None
  assert resume_point.token_id == "tok-001"
  assert resume_point.node_id == "node-001"
  assert resume_point.sequence_number == 1
  ```
- Priority is P1 because resume-point correctness is a critical recovery path and weak assertions can let serious regressions through.
