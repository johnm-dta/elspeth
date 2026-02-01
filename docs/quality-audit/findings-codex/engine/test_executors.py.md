# Test Defect Report

## Summary

- Transform error-path tests assert only `TransformResult`/`error_sink` and never verify that the audit trail records error details (node_states.error, transform_errors) even though executors write them.

## Severity

- Severity: major
- Priority: P1

## Category

- [Missing Audit Trail Verification]

## Evidence

- `tests/engine/test_executors.py:127` and `tests/engine/test_executors.py:134` show the error-path test stops at result-only assertions:
```python
result, _, _error_sink = executor.execute_transform(
    transform=as_transform(transform),
    token=token,
    ctx=ctx,
    step_in_pipeline=1,
)

assert result.status == "error"
assert result.reason == {"message": "validation failed"}
```
- `tests/engine/test_executors.py:369` and `tests/engine/test_executors.py:376` show the discard-path test only checks `error_sink`:
```python
result, _updated_token, error_sink = executor.execute_transform(
    transform=as_transform(transform),
    token=token,
    ctx=ctx,
    step_in_pipeline=1,
)

assert result.status == "error"
assert error_sink == "discard"
```
- `src/elspeth/engine/executors.py:241` and `src/elspeth/engine/executors.py:264` show audit writes that are not asserted in these tests:
```python
self._recorder.complete_node_state(
    state_id=state.state_id,
    status="failed",
    duration_ms=duration_ms,
    error=result.reason,
)
ctx.record_transform_error(
    token_id=token.token_id,
    transform_id=transform.node_id,
    row=token.row_data,
    error_details=result.reason or {},
    destination=on_error,
)
```

## Impact

- Regressions that drop or corrupt `node_states.error` or `transform_errors` on TransformResult.error would still pass tests.
- This directly undermines ELSPETH’s auditability guarantees for failure paths.

## Root Cause Hypothesis

- Test coverage focuses on functional return values and routing signals; audit-trail assertions are uneven across success vs. error paths.

## Recommended Fix

- Extend the error-path tests to query the Landscape recorder and assert error recording and transform_error rows.
- Example pattern:
```python
states = recorder.get_node_states_for_token(token.token_id)
assert len(states) == 1
state = states[0]
assert state.status == "failed"
assert state.error == {"message": "validation failed"}

errors = recorder.get_transform_errors_for_token(token.token_id)
assert len(errors) == 1
assert errors[0].transform_id == node.node_id
assert errors[0].destination == "discard"
assert errors[0].error_details == {"message": "validation failed"}
```
- Priority justification: audit trail completeness is a core contract; missing verification here risks silent audit loss.
---
# Test Defect Report

## Summary

- Tests use `hasattr` guards for `NodeState` fields (duration_ms/output_hash), which conflicts with the no-defensive-programming rule and weakens contract enforcement.

## Severity

- Severity: minor
- Priority: P3

## Category

- [Bug-Hiding Defensive Patterns]

## Evidence

- `tests/engine/test_executors.py:197` uses `hasattr` to guard access:
```python
state = states[0]
assert state.status == "failed"
assert hasattr(state, "duration_ms") and state.duration_ms is not None
```
- `tests/engine/test_executors.py:318` uses the same pattern for output hashes:
```python
assert state.input_hash is not None
assert hasattr(state, "output_hash") and state.output_hash is not None
```
- `tests/engine/test_executors.py:1211` repeats the pattern for gate failures.

## Impact

- Tests normalize defensive checks instead of enforcing the expected `NodeState` shape.
- If the wrong `NodeState` type is returned, the failure signal is weaker and less aligned with Tier 1 “crash on anomaly” expectations.

## Root Cause Hypothesis

- `hasattr` is used for “type narrowing” convenience rather than asserting the contract directly.

## Recommended Fix

- Remove `hasattr` and access fields directly so type/shape mismatches fail loudly.
- Example:
```python
assert state.duration_ms is not None
assert state.output_hash is not None
```
- This aligns tests with the prohibition on defensive patterns and strengthens contract enforcement.
