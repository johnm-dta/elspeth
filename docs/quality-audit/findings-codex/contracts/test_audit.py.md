# Test Defect Report

## Summary

- Checkpoint contract test asserts `created_at` can be None, contradicting the audit contract that requires a non-null timestamp for Tier 1 data.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Tier 1 Corruption Tests

## Evidence

- `tests/contracts/test_audit.py:691` allows NULL created_at and asserts it:
```python
def test_checkpoint_created_at_optional(self) -> None:
    checkpoint = Checkpoint(
        ...
        created_at=None,
        ...
    )
    assert checkpoint.created_at is None
```
- `src/elspeth/contracts/audit.py:332` documents created_at as required/NOT NULL:
```python
created_at: datetime  # Required - schema enforces NOT NULL (Tier 1 audit data)
```

## Impact

- Permits Tier 1 audit records without timestamps, undermining recovery ordering and audit integrity.
- Encourages code changes that accept NULL in required audit fields, violating the Three-Tier Trust Model.
- Masks corruption bugs in checkpoint persistence or load paths.

## Root Cause Hypothesis

- Test author assumed optional timestamp because the dataclass lacks runtime validation.
- Contract comments and Tier 1 requirements were not reconciled with the test expectations.

## Recommended Fix

- Replace `test_checkpoint_created_at_optional` with a negative test asserting `created_at=None` raises (TypeError/ValueError) once validation is enforced.
- Keep only positive tests that pass a real `datetime` for `created_at`.
- Priority justification: NULL timestamps in Tier 1 audit data are corruption and must crash immediately.
---
# Test Defect Report

## Summary

- Enum “must be enum, not string” tests only assert happy paths, never verify that invalid enum values or strings are rejected.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Tier 1 Corruption Tests

## Evidence

- `tests/contracts/test_audit.py:55` claims enum enforcement but only passes a valid enum and checks `isinstance`:
```python
def test_run_status_must_be_enum(self) -> None:
    run = Run(..., status=RunStatus.COMPLETED)
    assert isinstance(run.status, RunStatus)
```
- `tests/contracts/test_audit.py:391` repeats the same pattern for Call enums:
```python
def test_call_type_must_be_enum(self) -> None:
    call = Call(..., call_type=CallType.HTTP, ...)
    assert isinstance(call.call_type, CallType)
```
- `src/elspeth/contracts/audit.py:1` states strict enum-only contracts and crash-on-garbage behavior:
```python
These are strict contracts - all enum fields use proper enum types.
Per Data Manifesto: ... crash immediately.
```
- No `pytest.raises` tests for invalid enum values exist in this file.

## Impact

- Corrupted enum values (strings, invalid enums, None) can slip through without test failures.
- Undermines Tier 1 audit integrity and the “crash on anomaly” requirement.
- Creates false confidence that enum strictness is enforced.

## Root Cause Hypothesis

- Tests rely on type hints and value checks instead of enforcing failure on invalid inputs.
- Pattern likely repeated across other contract tests for enum fields.

## Recommended Fix

- Add negative tests for each enum field (Run/Node/Edge/Call/RoutingEvent/Batch) that pass invalid values and assert a hard failure.
- Use parametrized `pytest.raises` with invalid values (e.g., `"completed"`, `None`, wrong enum) to enforce strictness.
- Priority justification: enum corruption in audit data is a Tier 1 integrity breach.
---
# Test Defect Report

## Summary

- NodeStatePending variant is untested; contract coverage only exercises Open/Completed/Failed.

## Severity

- Severity: minor
- Priority: P2

## Category

- Incomplete Contract Coverage

## Evidence

- `src/elspeth/contracts/audit.py:146` defines NodeStatePending with required completion fields:
```python
@dataclass(frozen=True)
class NodeStatePending:
    ...
    status: Literal[NodeStateStatus.PENDING]
    completed_at: datetime
    duration_ms: float
```
- `tests/contracts/test_audit.py:284` covers only Open/Completed/Failed variants:
```python
def test_open_state_has_literal_status(self) -> None: ...
def test_completed_state_requires_output(self) -> None: ...
def test_failed_state_has_error_fields(self) -> None: ...
```
- No test constructs NodeStatePending or verifies its invariants.

## Impact

- Regressions in async/pending behavior (missing completed_at/duration_ms) can slip through.
- Weakens contract confidence for batch/async operations that rely on PENDING state.

## Root Cause Hypothesis

- Tests focused on the most common variants and overlooked the pending state used in async workflows.

## Recommended Fix

- Add a `test_pending_state_requires_completion_fields` that instantiates `NodeStatePending` with valid fields and asserts status/invariants.
- If validation is implemented, add negative tests for missing `completed_at` or `duration_ms`.
- Priority justification: pending state is part of core audit contract and should be validated.
