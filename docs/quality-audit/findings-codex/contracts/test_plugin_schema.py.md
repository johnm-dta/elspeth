# Test Defect Report

## Summary

- Missing assertion that `PluginSchema` is non-strict (`strict=False`), so a regression to strict mode would not be detected.

## Severity

- Severity: minor
- Priority: P2

## Category

- Incomplete Contract Coverage

## Evidence

- `tests/contracts/test_plugin_schema.py:13-14` only asserts `extra` and `frozen`, omitting `strict`:
```python
assert PluginSchema.model_config.get("extra") == "ignore"
assert PluginSchema.model_config.get("frozen") is False
```
- `src/elspeth/contracts/data.py:43-46` defines `strict=False` as part of the contract:
```python
model_config = ConfigDict(
    extra="ignore",
    strict=False,
    frozen=False,
)
```

## Impact

- Regression to `strict=True` (disabling coercion) would pass this test file undetected.
- Violates the Tier 3 boundary expectation that PluginSchema allows coercion for external data, potentially turning valid input into validation failures.
- Creates false confidence that the core schema contract is locked down.

## Root Cause Hypothesis

- Test focuses on import surface and partial config checks; strictness was overlooked when the contract expanded.

## Recommended Fix

- Add an explicit assertion in `tests/contracts/test_plugin_schema.py` for `PluginSchema.model_config["strict"] is False` alongside the existing config checks.
- Priority justification: strictness controls coercion at the data boundary; a silent regression would alter core ingestion behavior.
---
# Test Defect Report

## Summary

- `SchemaValidationError.value` is never asserted, leaving regressions in captured invalid input undetected.

## Severity

- Severity: trivial
- Priority: P3

## Category

- Weak Assertions

## Evidence

- `tests/contracts/test_plugin_schema.py:20-22` only checks `field` and `message`:
```python
error = SchemaValidationError("field", "message", "value")
assert error.field == "field"
assert error.message == "message"
```
- `src/elspeth/contracts/data.py:65-68` defines the `value` attribute:
```python
def __init__(self, field: str, message: str, value: Any = None) -> None:
    self.field = field
    self.message = message
    self.value = value
```

## Impact

- If `value` stops being stored (or is altered), tests still pass, reducing diagnostic fidelity for schema errors.
- Weakens confidence that validation errors retain the offending input for audit/debugging.

## Root Cause Hypothesis

- Minimal assertions added to prove importability; payload validation fields were not fully validated.

## Recommended Fix

- Add `assert error.value == "value"` in `tests/contracts/test_plugin_schema.py`.
- Priority justification: low risk but completes the contract for error payloads.
