# Test Defect Report

## Summary

- Tests only assert values from literal dicts and never validate the `ExecutionResult` contract (required/optional keys), so contract regressions go undetected

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/cli/test_execution_result.py:11` asserts only `run_id` even though `status` and `rows_processed` are required by the contract
```python
# tests/cli/test_execution_result.py:11
result: ExecutionResult = {
    "run_id": "run-123",
    "status": "completed",
    "rows_processed": 100,
}
assert result["run_id"] == "run-123"
```
- `tests/cli/test_execution_result.py:22` asserts only `rows_succeeded` while `rows_failed` and `duration_seconds` are set but never checked
```python
# tests/cli/test_execution_result.py:22
result: ExecutionResult = {
    "run_id": "run-456",
    "status": "completed",
    "rows_processed": 1000,
    "rows_succeeded": 990,
    "rows_failed": 10,
    "duration_seconds": 45.5,
}
assert result["rows_succeeded"] == 990
```
- `src/elspeth/contracts/cli.py:30` defines required vs optional keys, but tests never assert those sets
```python
# src/elspeth/contracts/cli.py:30
class ExecutionResult(TypedDict):
    run_id: str
    status: str
    rows_processed: int
    rows_succeeded: NotRequired[int]
    rows_failed: NotRequired[int]
    duration_seconds: NotRequired[float]
```

## Impact

- Required fields (`status`, `rows_processed`) or optional fields could be removed/renamed without any test failure
- Type contract drift in `ExecutionResult` can slip through, creating false confidence about CLI result shape

## Root Cause Hypothesis

- Tests were written to prove importability and basic usage, assuming type checkers enforce the contract, leaving runtime assertions minimal

## Recommended Fix

- Add contract-level assertions using `ExecutionResult.__required_keys__` and `ExecutionResult.__optional_keys__` to lock the schema
- Strengthen existing tests to assert all fields used in the literals (`status`, `rows_processed`, `rows_failed`, `duration_seconds`)
- Optional: use `typing.get_type_hints(ExecutionResult)` to validate field types in the test
