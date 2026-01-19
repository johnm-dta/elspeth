# tests/property/canonical/test_nan_rejection.py
"""Property-based tests for NaN and Infinity rejection.

ELSPETH strictly rejects NaN and Infinity values in canonical JSON.
This is defense-in-depth for audit integrity:

- NaN in audit data = undefined comparison behavior
- Infinity in audit data = potential for silent overflow
- Silent conversion = data corruption in the legal record

These values should be caught and rejected at ingestion (Tier 3),
not silently handled in the canonical layer.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.core.canonical import canonical_json, stable_hash

# =============================================================================
# Strategies for generating NaN/Infinity values
# =============================================================================

# Python float NaN and Infinity
nan_values = st.sampled_from([float("nan"), float("-nan")])
infinity_values = st.sampled_from([float("inf"), float("-inf")])
non_finite_floats = nan_values | infinity_values

# NumPy NaN and Infinity
numpy_nan_values = st.sampled_from([np.nan, np.float64("nan")])
numpy_infinity_values = st.sampled_from(
    [np.inf, -np.inf, np.float64("inf"), np.float64("-inf")]
)
numpy_non_finite = numpy_nan_values | numpy_infinity_values

# All non-finite values
all_non_finite = non_finite_floats | numpy_non_finite


# =============================================================================
# NaN Rejection Properties
# =============================================================================


class TestNaNRejection:
    """Property tests verifying NaN values are rejected."""

    @given(nan=nan_values)
    @settings(max_examples=20)
    def test_python_nan_rejected_in_canonical_json(self, nan: float) -> None:
        """Property: Python float NaN raises ValueError in canonical_json()."""
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json(nan)

    @given(nan=nan_values)
    @settings(max_examples=20)
    def test_python_nan_rejected_in_stable_hash(self, nan: float) -> None:
        """Property: Python float NaN raises ValueError in stable_hash()."""
        with pytest.raises(ValueError, match="non-finite"):
            stable_hash(nan)

    @given(nan=numpy_nan_values)
    @settings(max_examples=20)
    def test_numpy_nan_rejected_in_canonical_json(self, nan: Any) -> None:
        """Property: NumPy NaN raises ValueError in canonical_json()."""
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json(nan)

    @given(nan=numpy_nan_values)
    @settings(max_examples=20)
    def test_numpy_nan_rejected_in_stable_hash(self, nan: Any) -> None:
        """Property: NumPy NaN raises ValueError in stable_hash()."""
        with pytest.raises(ValueError, match="non-finite"):
            stable_hash(nan)

    @given(nan=nan_values)
    @settings(max_examples=20)
    def test_nan_in_dict_value_rejected(self, nan: float) -> None:
        """Property: NaN nested in dict value is rejected."""
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json({"value": nan})

    @given(nan=nan_values)
    @settings(max_examples=20)
    def test_nan_in_list_rejected(self, nan: float) -> None:
        """Property: NaN nested in list is rejected."""
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json([1, 2, nan, 4])

    @given(nan=nan_values)
    @settings(max_examples=20)
    def test_nan_deeply_nested_rejected(self, nan: float) -> None:
        """Property: NaN deeply nested is still rejected."""
        deeply_nested = {"level1": {"level2": {"level3": [{"value": nan}]}}}
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json(deeply_nested)


# =============================================================================
# Infinity Rejection Properties
# =============================================================================


class TestInfinityRejection:
    """Property tests verifying Infinity values are rejected."""

    @given(inf=infinity_values)
    @settings(max_examples=20)
    def test_python_infinity_rejected_in_canonical_json(self, inf: float) -> None:
        """Property: Python float Infinity raises ValueError in canonical_json()."""
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json(inf)

    @given(inf=infinity_values)
    @settings(max_examples=20)
    def test_python_infinity_rejected_in_stable_hash(self, inf: float) -> None:
        """Property: Python float Infinity raises ValueError in stable_hash()."""
        with pytest.raises(ValueError, match="non-finite"):
            stable_hash(inf)

    @given(inf=numpy_infinity_values)
    @settings(max_examples=20)
    def test_numpy_infinity_rejected_in_canonical_json(self, inf: Any) -> None:
        """Property: NumPy Infinity raises ValueError in canonical_json()."""
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json(inf)

    @given(inf=numpy_infinity_values)
    @settings(max_examples=20)
    def test_numpy_infinity_rejected_in_stable_hash(self, inf: Any) -> None:
        """Property: NumPy Infinity raises ValueError in stable_hash()."""
        with pytest.raises(ValueError, match="non-finite"):
            stable_hash(inf)

    @given(inf=infinity_values)
    @settings(max_examples=20)
    def test_infinity_in_dict_value_rejected(self, inf: float) -> None:
        """Property: Infinity nested in dict value is rejected."""
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json({"value": inf})

    @given(inf=infinity_values)
    @settings(max_examples=20)
    def test_infinity_in_list_rejected(self, inf: float) -> None:
        """Property: Infinity nested in list is rejected."""
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json([1, 2, inf, 4])

    @given(inf=infinity_values)
    @settings(max_examples=20)
    def test_positive_and_negative_infinity_both_rejected(self, inf: float) -> None:
        """Property: Both +Infinity and -Infinity are rejected."""
        # This test explicitly verifies both directions
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json(inf)


# =============================================================================
# Edge Cases and Mixed Scenarios
# =============================================================================


class TestNonFiniteEdgeCases:
    """Property tests for edge cases involving non-finite floats."""

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

    @given(
        valid_float=st.floats(allow_nan=False, allow_infinity=False),
        non_finite=all_non_finite,
    )
    @settings(max_examples=50)
    def test_mixed_valid_and_invalid_rejected(
        self, valid_float: float, non_finite: float
    ) -> None:
        """Property: A structure with both valid and invalid floats is rejected."""
        assume(math.isfinite(valid_float))

        data = {"valid": valid_float, "invalid": non_finite}
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json(data)

    @given(non_finite=all_non_finite)
    @settings(max_examples=30)
    def test_non_finite_in_numpy_array_rejected(self, non_finite: float) -> None:
        """Property: Non-finite values in numpy arrays are rejected."""
        arr = np.array([1.0, 2.0, non_finite, 4.0])
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json({"array": arr})

    def test_very_large_float_not_infinity(self) -> None:
        """Verify that very large (but finite) floats are accepted."""
        # Largest finite float
        large = 1.7976931348623157e308
        assert math.isfinite(large)

        # Should not raise
        result = canonical_json({"large": large})
        assert isinstance(result, str)

    def test_very_small_float_not_zero(self) -> None:
        """Verify that very small (but non-zero) floats are accepted."""
        # Smallest positive float
        tiny = 2.2250738585072014e-308
        assert tiny > 0
        assert math.isfinite(tiny)

        # Should not raise
        result = canonical_json({"tiny": tiny})
        assert isinstance(result, str)


# =============================================================================
# Positive Tests (Valid Floats)
# =============================================================================


class TestValidFloatsAccepted:
    """Property tests verifying valid floats are accepted."""

    @given(value=st.floats(allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_finite_floats_accepted(self, value: float) -> None:
        """Property: All finite floats are accepted."""
        # Should not raise
        result = canonical_json(value)
        assert isinstance(result, str)

    @given(value=st.floats(allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_finite_floats_in_dict_accepted(self, value: float) -> None:
        """Property: Finite floats in dicts are accepted."""
        result = canonical_json({"value": value})
        assert isinstance(result, str)

    def test_zero_accepted(self) -> None:
        """Verify that zero (including negative zero) is accepted."""
        assert canonical_json(0.0) is not None
        assert canonical_json(-0.0) is not None

    def test_special_float_values_accepted(self) -> None:
        """Verify specific special but finite float values are accepted."""
        special_values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,  # Can't be represented exactly
            1e-10,
            1e10,
            1.7976931348623157e308,  # Max float
            2.2250738585072014e-308,  # Min positive float
        ]
        for value in special_values:
            assert math.isfinite(value)
            result = canonical_json({"value": value})
            assert isinstance(result, str)
