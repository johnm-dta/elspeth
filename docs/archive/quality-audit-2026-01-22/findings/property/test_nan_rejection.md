# Test Quality Review: test_nan_rejection.py

## Summary

Well-structured property tests with excellent coverage of NaN/Infinity rejection invariants. Several critical gaps identified: inadequate shrinking verification, missing Decimal type coverage, no testing of error message quality, and absence of round-trip property tests that would validate the two-phase canonicalization contract.

## Poorly Constructed Tests

### Test: test_positive_and_negative_infinity_both_rejected (line 156)

**Issue**: Duplicate test with no added value

**Evidence**:
```python
@given(inf=infinity_values)
def test_positive_and_negative_infinity_both_rejected(self, inf: float) -> None:
    """Property: Both +Infinity and -Infinity are rejected."""
    # This test explicitly verifies both directions
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json(inf)
```

This is identical to `test_python_infinity_rejected_in_canonical_json` (line 114). The strategy `infinity_values` already generates both `float("inf")` and `float("-inf")`, so the earlier test already validates both directions. The comment "explicitly verifies both directions" is misleading - both tests have identical behavior.

**Fix**: Delete this test entirely. If the intention is to verify both directions in a single example, write a parametrized unit test instead:
```python
@pytest.mark.parametrize("inf", [float("inf"), float("-inf")])
def test_both_infinity_directions_rejected(inf: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json(inf)
```

**Priority**: P2

### Test: test_nan_not_equal_to_itself_doesnt_bypass_check (line 171)

**Issue**: This is a unit test masquerading as a property test, testing implementation details rather than properties

**Evidence**:
```python
def test_nan_not_equal_to_itself_doesnt_bypass_check(self) -> None:
    """Verify that NaN's self-inequality doesn't bypass our check.

    NaN has the property that nan != nan. Our check uses math.isnan()
    which correctly identifies NaN regardless of this quirk.
    """
    nan = float("nan")
    assert nan != nan  # Confirm NaN self-inequality

    # But our check should still catch it
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json(nan)
```

This tests that the implementation uses `math.isnan()` rather than equality comparison. Property tests should verify observable behavior (rejection), not implementation choices. The docstring even acknowledges this is about "our check uses math.isnan()" - that's implementation knowledge.

**Fix**: Either delete this test (already covered by `test_python_nan_rejected_in_canonical_json`), or move it to the unit test file `tests/core/test_canonical.py` if the implementation detail matters for maintainability.

**Priority**: P2

### Test: test_very_large_float_not_infinity (line 205) and test_very_small_float_not_zero (line 215)

**Issue**: These are unit tests with hardcoded edge cases, not property tests. They belong in `tests/core/test_canonical.py`, not a property test suite.

**Evidence**:
```python
def test_very_large_float_not_infinity(self) -> None:
    """Verify that very large (but finite) floats are accepted."""
    # Largest finite float
    large = 1.7976931348623157e308
    assert math.isfinite(large)

    # Should not raise
    result = canonical_json({"large": large})
    assert isinstance(result, str)
```

Property tests generate values systematically via strategies. Hardcoded boundary values are characteristic of example-based unit tests. The strategy `st.floats(allow_nan=False, allow_infinity=False)` in `test_finite_floats_accepted` (line 237) already generates values across the entire finite float range, including edge values near max/min.

**Fix**: Move these to `tests/core/test_canonical.py` as parametrized unit tests, or delete if coverage is redundant.

**Priority**: P2

## Misclassified Tests

### Class: TestValidFloatsAccepted (line 232)

**Issue**: Positive tests ("valid floats are accepted") dilute property test focus

**Evidence**: Property tests are most valuable for exploring adversarial inputs (NaN, Infinity, edge cases). Testing that "all finite floats work" is important, but it's not the primary concern for NaN rejection properties. The file docstring states:

> ELSPETH strictly rejects NaN and Infinity values in canonical JSON.

The file name is `test_nan_rejection.py`, not `test_canonical_floats.py`. Including 3 tests (200 examples each) for the positive case shifts focus from the critical security property (rejection) to general correctness (acceptance).

**Fix**: Move `TestValidFloatsAccepted` to `tests/core/test_canonical.py` or create a separate property test file `test_canonical_floats_property.py`. Keep this file focused exclusively on NaN/Infinity rejection invariants.

**Priority**: P1

## Property Test Quality Issues

### Missing: Shrinking verification

**Issue**: No evidence that Hypothesis shrinking works correctly when NaN/Infinity is embedded in complex structures

**Evidence**: Tests like `test_nan_deeply_nested_rejected` (line 97) verify rejection, but don't verify that Hypothesis can shrink the failing case to a minimal example. Poor shrinking makes debugging harder.

**Fix**: Add at least one test that deliberately triggers shrinking and verifies the minimal failing case:
```python
from hypothesis import find

def test_hypothesis_shrinks_nan_to_minimal_case() -> None:
    """Verify Hypothesis shrinking produces useful minimal examples for NaN."""
    minimal = find(
        st.recursive(
            st.none() | st.floats(allow_nan=True, allow_infinity=False),
            lambda children: st.lists(children, min_size=1) | st.dictionaries(st.text(), children, min_size=1),
        ),
        lambda obj: is_nan_present(obj) and not is_rejected_correctly(obj),
    )
    # Should shrink to something like {"x": nan} or [nan], not deeply nested structure
```

**Priority**: P2

### Missing: Decimal NaN/Infinity coverage

**Issue**: Implementation supports `Decimal` and rejects non-finite Decimal values (line 111 in canonical.py), but property tests only cover `float` and `numpy` types

**Evidence**:
```python
# From canonical.py line 111:
if isinstance(obj, Decimal):
    if not obj.is_finite():  # Rejects NaN, sNaN, Infinity, -Infinity
        raise ValueError(f"Cannot canonicalize non-finite Decimal: {obj}...")
```

No test exercises this path with `Decimal("NaN")`, `Decimal("Infinity")`, or `Decimal("-Infinity")`.

**Fix**: Add a test class:
```python
class TestDecimalNonFiniteRejection:
    """Property tests for Decimal NaN/Infinity rejection."""

    @given(decimal_str=st.sampled_from(["NaN", "sNaN", "Infinity", "-Infinity"]))
    @settings(max_examples=10)
    def test_decimal_non_finite_rejected(self, decimal_str: str) -> None:
        """Property: Non-finite Decimal values are rejected."""
        dec = Decimal(decimal_str)
        with pytest.raises(ValueError, match="non-finite Decimal"):
            canonical_json(dec)

    @given(decimal_str=st.sampled_from(["NaN", "Infinity"]))
    def test_decimal_non_finite_in_dict_rejected(self, decimal_str: str) -> None:
        dec = Decimal(decimal_str)
        with pytest.raises(ValueError, match="non-finite Decimal"):
            canonical_json({"value": dec})
```

**Priority**: P0 (implementation supports it, tests must cover it)

### Missing: Error message quality assertions

**Issue**: Tests only verify `match="non-finite"` but don't validate that error messages are actionable

**Evidence**: Implementation provides detailed error messages:
```python
# Line 60 in canonical.py:
raise ValueError(f"Cannot canonicalize non-finite float: {obj}. Use None for missing values, not NaN.")

# Line 82:
raise ValueError("NaN/Infinity found in NumPy array. Audit trail requires finite values only. Use None for missing values, not NaN.")
```

Tests use `match="non-finite"` which is too loose - it would pass even if the message was just "non-finite error" with no guidance.

**Fix**: Use more specific regex patterns that validate the actionable guidance:
```python
def test_nan_error_message_provides_guidance(self) -> None:
    """Error messages for NaN must guide users to use None instead."""
    with pytest.raises(ValueError, match=r"Use None for missing values, not NaN"):
        canonical_json(float("nan"))

def test_numpy_array_error_message_identifies_container(self) -> None:
    """Error messages for arrays must identify the container type."""
    with pytest.raises(ValueError, match=r"NaN/Infinity found in NumPy array"):
        canonical_json({"arr": np.array([1.0, float("nan")])})
```

**Priority**: P1

### Missing: Round-trip property for finite values

**Issue**: No test verifies the critical invariant: `hash(canonical_json(x)) == hash(canonical_json(x))` for finite values

**Evidence**: The entire purpose of canonical JSON is deterministic hashing. Tests verify rejection of bad values but don't verify that good values produce stable hashes.

**Fix**: Add round-trip property test:
```python
@given(value=st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_finite_float_hash_stability(self, value: float) -> None:
    """Property: Finite floats produce stable hashes across serialization."""
    from elspeth.core.canonical import stable_hash

    # Hash should be deterministic
    hash1 = stable_hash(value)
    hash2 = stable_hash(value)
    assert hash1 == hash2

    # Hash should survive embedding in structures
    hash_direct = stable_hash(value)
    hash_in_dict = stable_hash({"value": value})
    # Not equal (different structure), but both should be deterministic
    assert hash_direct == stable_hash(value)
    assert hash_in_dict == stable_hash({"value": value})
```

**Priority**: P1

### Missing: Mixed valid/invalid in complex structures

**Issue**: `test_mixed_valid_and_invalid_rejected` (line 189) only tests flat dicts with 2 keys, not realistic nested structures

**Evidence**:
```python
@given(
    valid_float=st.floats(allow_nan=False, allow_infinity=False),
    non_finite=all_non_finite,
)
@settings(max_examples=50)
def test_mixed_valid_and_invalid_rejected(self, valid_float: float, non_finite: float) -> None:
    """Property: A structure with both valid and invalid floats is rejected."""
    assume(math.isfinite(valid_float))

    data = {"valid": valid_float, "invalid": non_finite}  # Only tests flat dict
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json(data)
```

Real pipeline data has lists of dicts, nested arrays, etc. This test doesn't explore those scenarios.

**Fix**: Use Hypothesis recursive strategies:
```python
@given(
    data=st.recursive(
        st.none() | st.booleans() | st.integers() | st.text() | st.floats(allow_nan=False, allow_infinity=False),
        lambda children: st.lists(children, max_size=5) | st.dictionaries(st.text(), children, max_size=5),
        max_leaves=20,
    )
)
def test_deeply_nested_finite_floats_accepted(self, data: Any) -> None:
    """Property: Arbitrary nested structures with finite floats are accepted."""
    result = canonical_json(data)
    assert isinstance(result, str)

@given(
    data=st.recursive(
        st.none() | st.booleans() | st.integers() | st.text() | st.floats(allow_nan=True, allow_infinity=True),
        lambda children: st.lists(children, max_size=5) | st.dictionaries(st.text(), children, max_size=5),
        max_leaves=20,
    )
)
def test_deeply_nested_with_any_non_finite_rejected(self, data: Any) -> None:
    """Property: ANY non-finite float in nested structure causes rejection."""
    assume(contains_non_finite(data))  # Only test cases with NaN/Inf
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json(data)
```

**Priority**: P2

## Infrastructure Gaps

### Gap: No test markers for slow tests

**Issue**: Some tests use `max_examples=200`, which may slow down the test suite. No way to skip slow property tests during rapid iteration.

**Evidence**: `test_finite_floats_accepted` (line 237) and `test_finite_floats_in_dict_accepted` (line 245) both use `max_examples=200`, while most tests use 20-50.

**Fix**: Add pytest markers:
```python
# At top of file
import pytest

pytestmark = pytest.mark.property

# For expensive tests
@pytest.mark.slow
@given(value=st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_finite_floats_accepted(self, value: float) -> None:
    ...
```

Then enable `pytest -m "not slow"` for fast feedback loops.

**Priority**: P3

### Gap: No shared fixtures for common assertions

**Issue**: Repeated `with pytest.raises(ValueError, match="non-finite")` blocks throughout (appears 20+ times)

**Evidence**: Every rejection test has identical pytest.raises structure. If the error message format changes (e.g., to include the value type), all tests need updating.

**Fix**: Create a helper function:
```python
def assert_canonical_rejects_non_finite(obj: Any, expected_message: str = "non-finite") -> None:
    """Assert that canonical_json rejects obj with non-finite error."""
    with pytest.raises(ValueError, match=expected_message):
        canonical_json(obj)
    with pytest.raises(ValueError, match=expected_message):
        stable_hash(obj)

# Usage:
def test_python_nan_rejected(self, nan: float) -> None:
    assert_canonical_rejects_non_finite(nan)
```

**Priority**: P3

### Gap: No hypothesis settings profile

**Issue**: `@settings(max_examples=20)` is repeated 14 times with slightly different values (20, 30, 50, 200)

**Evidence**: Inconsistent example counts suggest no deliberate tuning - just arbitrary choices.

**Fix**: Define a profile in `conftest.py`:
```python
from hypothesis import settings, Phase

settings.register_profile("ci", max_examples=500, phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink])
settings.register_profile("dev", max_examples=50)
settings.register_profile("quick", max_examples=10, phases=[Phase.generate])
```

Then remove all `@settings(max_examples=X)` decorators from tests and use the profile via pytest option or environment variable.

**Priority**: P3

### Gap: No testing of repr_hash fallback path

**Issue**: Implementation provides `repr_hash()` as fallback for non-canonical data (line 263 in canonical.py), but no tests verify it works with NaN/Infinity

**Evidence**:
```python
# From canonical.py line 263:
def repr_hash(obj: Any) -> str:
    """Generate SHA-256 hash of repr() for non-canonical data.

    Used as fallback when canonical_json fails (NaN, Infinity, or other
    non-serializable types).
    """
```

Docstring explicitly mentions NaN/Infinity use case, but `test_nan_rejection.py` never calls `repr_hash()`.

**Fix**: Add tests for the intended fallback behavior:
```python
class TestReprHashFallback:
    """Verify repr_hash can handle non-canonical data that canonical_json rejects."""

    @given(non_finite=all_non_finite)
    def test_repr_hash_accepts_non_finite(self, non_finite: float) -> None:
        """Property: repr_hash should succeed where canonical_json fails."""
        from elspeth.core.canonical import repr_hash

        # canonical_json rejects
        with pytest.raises(ValueError):
            canonical_json(non_finite)

        # repr_hash accepts
        hash_result = repr_hash(non_finite)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA-256 hex digest

    def test_repr_hash_not_stable_across_structures(self) -> None:
        """Verify repr_hash is for quarantine only, not general use."""
        nan = float("nan")
        # Different repr() → different hash (unlike canonical_json for valid data)
        hash1 = repr_hash(nan)
        hash2 = repr_hash([nan])
        assert hash1 != hash2  # Different structures, different repr
```

**Priority**: P1 (implementation explicitly documents this use case)

## Positive Observations

- **Excellent strategy design**: Separates Python float, NumPy scalar, and NumPy array cases cleanly
- **Good docstring discipline**: Every test has a clear property statement in the docstring
- **Appropriate use of `assume()`**: `test_mixed_valid_and_invalid_rejected` correctly uses `assume(math.isfinite(valid_float))` to skip degenerate cases
- **Comprehensive nesting coverage**: Tests verify rejection at multiple nesting levels (top-level, dict value, list element, deeply nested)
- **Clear test organization**: Logical grouping into `TestNaNRejection`, `TestInfinityRejection`, `TestNonFiniteEdgeCases`, `TestValidFloatsAccepted`
