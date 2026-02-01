# Test Defect Report

## Summary

- `test_passthrough_preserves_all_fields` only compares key sets, so value mutations can slip through undetected.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/contracts/transform_contracts/test_passthrough_contract.py:56` only asserts key equality, not value equality:
```python
assert set(result.row.keys()) == set(input_row.keys())
```

## Impact

- A regression that alters values (e.g., coercion, normalization, dropping nested values) would still pass this test.
- Creates false confidence that PassThrough preserves data content, not just schema shape.

## Root Cause Hypothesis

- Likely an oversight where the test name implies full preservation but the assertion only checks keys.

## Recommended Fix

- Assert full dict equality for this example input (and optionally object identity if desired):
```python
assert result.row == input_row
```
- Keep the deep-copy test for mutation independence; this change specifically validates value preservation.
---
# Test Defect Report

## Summary

- `test_passthrough_is_deterministic` never asserts success, so error results with `row=None` can pass.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/contracts/transform_contracts/test_passthrough_contract.py:190` compares rows without checking status/row presence:
```python
assert result1.row == result2.row
```

## Impact

- If `process` returns error results for valid inputs, this test can still pass (`None == None`), masking regressions.
- Reduces confidence in determinism guarantees on the success path.

## Root Cause Hypothesis

- Test focuses on equality only and omits basic success preconditions.

## Recommended Fix

- Assert success and non-None rows before comparing outputs:
```python
assert result1.status == "success"
assert result2.status == "success"
assert result1.row is not None
assert result2.row is not None
assert result1.row == result2.row
```
- This ensures the determinism property is verified on the correct execution path.
