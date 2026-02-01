# Test Defect Report

## Summary

- Weak assertion in negative-zero test allows incorrect non-zero negative outputs to pass

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/core/test_canonical_mutation_gaps.py:135` allows any negative value to satisfy the assertion, not just negative zero:
```python
result = _normalize_value(-0.0)
assert result == 0.0 or math.copysign(1, result) == -1  # -0.0 or 0.0 both acceptable
```

## Impact

- A regression that mistakenly converts `-0.0` into another negative number (e.g., `-1.0`) would still pass this test, masking incorrect normalization behavior.
- The test does not enforce the magnitude being zero, so it provides false confidence about correctness for `-0.0` handling.

## Root Cause Hypothesis

- Attempt to allow both `-0.0` and `0.0` resulted in an overly permissive `or` condition that doesn’t constrain the value to zero.

## Recommended Fix

- Tighten the assertion to require a zero value and optionally verify sign only if preservation matters:
```python
result = _normalize_value(-0.0)
assert result == 0.0
```
- If negative sign preservation is intended, enforce it explicitly:
```python
result = _normalize_value(-0.0)
assert result == 0.0 and math.copysign(1.0, result) == -1.0
```
- Priority justification: This improves correctness guarantees for numeric normalization with minimal change.
