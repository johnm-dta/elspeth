# Test Defect Report

## Summary

- RetryPolicy tests contain tautological/partial assertions that do not validate the TypedDict contract or all defaulted fields, leaving schema/default regressions undetected.

## Severity

- Severity: minor
- Priority: P3

## Category

- Weak Assertions

## Evidence

- `tests/engine/test_retry_policy.py:12` and `tests/engine/test_retry_policy.py:16` only assert a literal value from a dict created in the test; there is no contract validation despite the schema in `src/elspeth/contracts/engine.py:7`.
```python
policy: RetryPolicy = {
    "max_attempts": 3,
    "base_delay": 1.0,
}
assert policy["max_attempts"] == 3
```
- `tests/engine/test_retry_policy.py:42` and `tests/engine/test_retry_policy.py:46` only verify `base_delay` default for partial policies, leaving `max_delay`/`jitter` defaults in `src/elspeth/engine/retry.py:81` and `src/elspeth/engine/retry.py:83` untested.
```python
policy: RetryPolicy = {"max_attempts": 10}
config = RetryConfig.from_policy(policy)
assert config.max_attempts == 10
assert config.base_delay == 1.0
```

## Impact

- Contract changes to `RetryPolicy` (keys/optionality) could ship without detection.
- Default value regressions for `max_delay` or `jitter` in `RetryConfig.from_policy` could slip, altering retry behavior in production.
- Tests provide false confidence because they don't assert the contract or all defaults.

## Root Cause Hypothesis

- Tests were written as smoke checks and rely on static typing, resulting in minimal runtime assertions.
- Contract/schema validation for TypedDicts was not treated as a testable behavior.

## Recommended Fix

- Strengthen `test_retry_policy_importable` to assert the schema (keys and `total=False`) using TypedDict introspection.
- Extend `test_retry_policy_partial` to assert defaults for `max_delay` and `jitter` when omitted.
- Example pattern:
```python
assert RetryPolicy.__total__ is False
assert set(RetryPolicy.__annotations__) == {
    "max_attempts",
    "base_delay",
    "max_delay",
    "jitter",
}

assert config.max_delay == 60.0
assert config.jitter == 1.0
```
- Priority justification: Low risk but improves regression detection for retry-policy contracts and defaults.
