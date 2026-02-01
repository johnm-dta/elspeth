# Test Defect Report

## Summary

- Missing test coverage for PluginSchema’s coercive validation (strict=False), so a regression to strict validation would pass unnoticed.

## Severity

- Severity: minor
- Priority: P2

## Category

- Missing Edge Cases

## Evidence

- `src/elspeth/contracts/data.py:37` documents coercive validation and `strict=False`, which is a core contract.
- `src/elspeth/contracts/data.py:43` sets `strict=False` but no test in this file verifies coercion behavior.
- `tests/contracts/test_data.py:18` and `tests/contracts/test_data.py:23` show only a valid int and a non-coercible string; there is no test for coercible inputs like `"42"`.

```python
# tests/contracts/test_data.py
schema = MySchema(name="test", value=42)
assert schema.name == "test"

with pytest.raises(ValidationError):
    MySchema(name="test", value="not_an_int")
```

## Impact

- A regression to `strict=True` or other behavior that disables coercion would still pass these tests.
- This would violate the “Their Data” trust boundary by rejecting coercible inputs and could increase quarantined rows without detection.
- Creates false confidence that PluginSchema’s permissive validation contract is enforced.

## Root Cause Hypothesis

- Tests focus on negative validation paths and omit positive coercion scenarios, likely assuming Pydantic’s default behavior without enforcing it.

## Recommended Fix

- Add a test in `tests/contracts/test_data.py` that passes a coercible string (e.g., `"42"`) and asserts the value is converted and stored correctly.
- Example pattern:
  - Create `MySchema(name="test", value="42")`
  - Assert `schema.value == 42` and optionally `isinstance(schema.value, int)` to lock in coercion.
- This directly verifies the strictness contract rather than only the failure path.
