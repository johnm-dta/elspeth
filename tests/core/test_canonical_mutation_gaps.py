"""Mutation gap tests for core/canonical.py.

Tests targeting specific mutation survivors:
- Line 47-49: NaN/Infinity rejection with 'or' logic (not 'and')
- Ensures both float and np.floating types are checked
"""

import math

import numpy as np
import pytest


class TestNanInfinityOrLogic:
    """Tests for NaN/Infinity rejection logic.

    Targets line 49: if math.isnan(obj) or math.isinf(obj):
    Mutant might change 'or' to 'and', which would only reject values
    that are BOTH NaN AND Infinity (impossible).

    These tests ensure each condition independently triggers rejection.
    """

    def test_nan_alone_is_rejected_not_requiring_infinity(self) -> None:
        """Line 49: NaN must be rejected even without Infinity.

        If mutation changes 'or' to 'and', NaN alone would pass through
        because a value cannot be both NaN and Infinity.
        """
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(float("nan"))

    def test_positive_infinity_alone_is_rejected_not_requiring_nan(self) -> None:
        """Line 49: +Infinity must be rejected even without NaN.

        If mutation changes 'or' to 'and', Infinity alone would pass through.
        """
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(float("inf"))

    def test_negative_infinity_alone_is_rejected_not_requiring_nan(self) -> None:
        """Line 49: -Infinity must be rejected even without NaN.

        Ensures negative infinity is also caught.
        """
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(float("-inf"))


class TestTypeCheckCoversBothFloatTypes:
    """Tests for type check coverage.

    Targets line 48: if isinstance(obj, float | np.floating):
    Mutant might remove one type from the union, allowing NaN/Infinity
    to slip through for that type.
    """

    def test_python_float_nan_is_rejected(self) -> None:
        """Line 48: Python float NaN must be caught by type check."""
        from elspeth.core.canonical import _normalize_value

        nan_value = float("nan")
        assert isinstance(nan_value, float)  # Verify it's a Python float

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(nan_value)

    def test_numpy_float64_nan_is_rejected(self) -> None:
        """Line 48: numpy.float64 NaN must be caught by type check."""
        from elspeth.core.canonical import _normalize_value

        nan_value = np.float64("nan")
        assert isinstance(nan_value, np.floating)  # Verify it's numpy floating

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(nan_value)

    def test_numpy_float32_nan_is_rejected(self) -> None:
        """Line 48: numpy.float32 NaN must also be caught.

        np.float32 is a subtype of np.floating, should be covered.
        """
        from elspeth.core.canonical import _normalize_value

        nan_value = np.float32("nan")
        assert isinstance(nan_value, np.floating)

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(nan_value)

    def test_python_float_infinity_is_rejected(self) -> None:
        """Line 48: Python float Infinity must be caught."""
        from elspeth.core.canonical import _normalize_value

        inf_value = float("inf")
        assert isinstance(inf_value, float)

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(inf_value)

    def test_numpy_float64_infinity_is_rejected(self) -> None:
        """Line 48: numpy.float64 Infinity must be caught."""
        from elspeth.core.canonical import _normalize_value

        inf_value = np.float64("inf")
        assert isinstance(inf_value, np.floating)

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(inf_value)


class TestNormalFloatsPassThrough:
    """Tests verifying normal floats are NOT rejected.

    Ensures the NaN/Infinity check doesn't accidentally reject valid floats.
    """

    def test_zero_float_passes(self) -> None:
        """Zero is a valid float, should pass through."""
        from elspeth.core.canonical import _normalize_value

        assert _normalize_value(0.0) == 0.0

    def test_negative_zero_passes(self) -> None:
        """Negative zero is valid, should pass through."""
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(-0.0)
        assert result == 0.0 or math.copysign(1, result) == -1  # -0.0 or 0.0 both acceptable

    def test_large_float_passes(self) -> None:
        """Large (but finite) floats should pass through."""
        from elspeth.core.canonical import _normalize_value

        large_value = 1e308  # Near max float
        assert _normalize_value(large_value) == large_value

    def test_small_float_passes(self) -> None:
        """Small (but non-zero) floats should pass through."""
        from elspeth.core.canonical import _normalize_value

        small_value = 1e-308  # Near min positive float
        assert _normalize_value(small_value) == small_value

    def test_normal_numpy_float_passes(self) -> None:
        """Normal numpy float should pass through and be converted to Python float."""
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(np.float64(3.14))
        assert result == 3.14
        assert type(result) is float  # Should be converted from np.float64
