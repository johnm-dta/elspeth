# Test Defect Report

## Summary

- Using `using-quality-engineering` skill because this is a test quality audit.
- Contract tests are tautological and do not validate the TypedDict schema (required/optional keys); they only assert on locally constructed dicts and omit declared optional fields like `field` and `comparison`.

## Severity

- Severity: minor
- Priority: P2

## Category

- Incomplete Contract Coverage

## Evidence

- `tests/contracts/test_errors.py:29` and `tests/contracts/test_errors.py:35` only check key presence/value on a locally constructed dict; no assertions against TypedDict metadata, so schema changes would not fail:
```python
error: ExecutionError = {
    "exception": "KeyError: 'foo'",
    "type": "KeyError",
    "traceback": "Traceback (most recent call last):\n...",
}

assert "traceback" in error
```
- `tests/contracts/test_errors.py:45` and `tests/contracts/test_errors.py:50` assert only `rule` while `matched_value` is never asserted and other optional fields are never exercised:
```python
reason: RoutingReason = {
    "rule": "value > threshold",
    "matched_value": 42,
}

assert reason["rule"] == "value > threshold"
```
- `src/elspeth/contracts/errors.py:29` and `src/elspeth/contracts/errors.py:31` define optional `field` and `comparison`, but the tests never validate they exist in the contract:
```python
threshold: NotRequired[float]
field: NotRequired[str]
comparison: NotRequired[str]
```

## Impact

- Contract tests can pass even if `ExecutionError`, `RoutingReason`, or `TransformReason` required/optional keys change.
- Schema drift in audit payloads could ship undetected, weakening Landscape auditability.
- Creates false confidence that error/reason payload contracts are enforced by tests.

## Root Cause Hypothesis

- Tests were written as example usage of TypedDicts instead of asserting the TypedDict schema metadata.
- Contract updates (e.g., adding `field`/`comparison`) were not paired with test updates.

## Recommended Fix

- In `tests/contracts/test_errors.py`, assert TypedDict schema metadata (`__required_keys__`, `__optional_keys__`) for all three contracts, and include optional fields in assertions to prevent silent drift.
- Example pattern:
```python
from elspeth.contracts import ExecutionError, RoutingReason, TransformReason

def test_execution_error_schema_keys() -> None:
    assert ExecutionError.__required_keys__ == {"exception", "type"}
    assert ExecutionError.__optional_keys__ == {"traceback"}

def test_routing_reason_schema_keys() -> None:
    assert RoutingReason.__required_keys__ == {"rule", "matched_value"}
    assert RoutingReason.__optional_keys__ == {"threshold", "field", "comparison"}

def test_transform_reason_schema_keys() -> None:
    assert TransformReason.__required_keys__ == {"action"}
    assert TransformReason.__optional_keys__ == {"fields_modified", "validation_errors"}
```
- Priority justification: these are audit-contract types; catching schema drift early prevents invalid audit records from reaching production.
