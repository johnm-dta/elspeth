# Test Defect Report

## Summary

- Missing per-instance independence check for `validation_errors` default list allows shared-mutable-default regressions to slip.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Edge Cases

## Evidence

- `test_validation_errors_defaults_to_empty_list` only asserts emptiness and non-None; it never verifies per-instance list independence (tests/core/landscape/test_lineage_mutation_gaps.py:109).
- By contrast, `transform_errors` has an explicit independence test, showing the intended pattern (tests/core/landscape/test_lineage_mutation_gaps.py:65).
- `LineageResult.validation_errors` uses `default_factory=list`, which is exactly the mutable-default surface area that needs a per-instance test (src/elspeth/core/landscape/lineage.py:53).

```python
def test_validation_errors_defaults_to_empty_list(...):
    result = LineageResult(...)
    assert result.validation_errors == []
    assert result.validation_errors is not None
```

## Impact

- A regression to a shared mutable default for `validation_errors` would not be detected; validation errors could leak between lineages, corrupting audit explanations.
- Tests would still pass, giving false confidence in lineage isolation guarantees.

## Root Cause Hypothesis

- The mutation-gap focus prioritized `transform_errors` and `outcome`; the analogous per-instance check for `validation_errors` was overlooked.

## Recommended Fix

- Add a per-instance independence test for `validation_errors`, mirroring `transform_errors`:

```python
def test_validation_errors_default_is_independent_per_instance(...):
    result1 = LineageResult(...)
    result2 = LineageResult(...)
    assert result1.validation_errors is not result2.validation_errors
    assert result2.validation_errors == []
```

- This closes the shared-mutable-default hole and aligns coverage with the transform_errors pattern.
