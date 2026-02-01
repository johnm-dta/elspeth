Using using-quality-engineering (test-maintenance-patterns) to guide the test-quality audit.
# Test Defect Report

## Summary

- Tests exercise `get_calls` but only assert ordering and the in-memory return object, leaving persisted audit fields (call_type/status enums, hashes, refs) unverified.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/core/landscape/test_recorder_calls.py:96` only checks ordering from `get_calls`, not the persisted call fields:
```python
calls = recorder.get_calls(state_id)
assert len(calls) == 2
assert calls[0].call_index == 0
assert calls[1].call_index == 1
```
- `tests/core/landscape/test_recorder_calls.py:57` validates only the `record_call` return value (no persisted-row verification):
```python
call = recorder.record_call(...)
assert call.request_hash is not None
```
- `src/elspeth/contracts/audit.py:228` defines a strict contract requiring enums, but tests do not assert this on data read back from storage:
```python
class Call:
    """An external call made during node processing.

    Strict contract - call_type and status must be enums.
    """
    call_type: CallType
    status: CallStatus
```

## Impact

- A regression that stores incorrect or NULL call fields (or returns raw DB strings) can pass these tests, so audit trail corruption could slip into production undetected.
- `get_calls` is a retrieval API; under-validation here undermines replay/verify and explainability guarantees.

## Root Cause Hypothesis

- Tests focus on the immediate `record_call` return object and treat persistence as implicit, so persisted audit data is not explicitly checked.
- No helper/assertion pattern exists in this file for validating stored call records.

## Recommended Fix

- Extend `test_multiple_calls_same_state` (or add a dedicated test) to assert persisted fields from `get_calls`:
  - `call_type`/`status` are enums and match expected values.
  - `request_hash`/`response_hash` match `stable_hash(request_data)` and `stable_hash(response_data)`.
  - `request_ref`/`response_ref` and `error_json` match expected values (`canonical_json` where applicable).
- Use `stable_hash`/`canonical_json` from `elspeth.core.canonical` to compute expected values consistently.
- Priority justification: audit trail correctness is core to ELSPETH; failing to validate persisted call records risks silent integrity regressions.
