# tests/core/test_canonical.py
"""Tests for canonical JSON serialization and hashing."""

import pytest


class TestNormalizeValue:
    """Test _normalize_value handles Python primitives."""

    def test_string_passthrough(self) -> None:
        from elspeth.core.canonical import _normalize_value

        assert _normalize_value("hello") == "hello"

    def test_int_passthrough(self) -> None:
        from elspeth.core.canonical import _normalize_value

        assert _normalize_value(42) == 42

    def test_float_passthrough(self) -> None:
        from elspeth.core.canonical import _normalize_value

        assert _normalize_value(3.14) == 3.14

    def test_none_passthrough(self) -> None:
        from elspeth.core.canonical import _normalize_value

        assert _normalize_value(None) is None

    def test_bool_passthrough(self) -> None:
        from elspeth.core.canonical import _normalize_value

        assert _normalize_value(True) is True
        assert _normalize_value(False) is False


class TestNanInfinityRejection:
    """NaN and Infinity must be rejected, not silently converted."""

    def test_nan_raises_value_error(self) -> None:
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(float("nan"))

    def test_positive_infinity_raises_value_error(self) -> None:
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(float("inf"))

    def test_negative_infinity_raises_value_error(self) -> None:
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(float("-inf"))

    def test_normal_float_allowed(self) -> None:
        from elspeth.core.canonical import _normalize_value

        # These should NOT raise
        assert _normalize_value(0.0) == 0.0
        assert _normalize_value(-0.0) == -0.0
        assert _normalize_value(1e308) == 1e308


import numpy as np


class TestNumpyTypeConversion:
    """NumPy types must be converted to Python primitives."""

    def test_numpy_int64_converts_to_int(self) -> None:
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(np.int64(42))
        assert result == 42
        assert type(result) is int

    def test_numpy_float64_converts_to_float(self) -> None:
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(np.float64(3.14))
        assert result == 3.14
        assert type(result) is float

    def test_numpy_float64_nan_raises(self) -> None:
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(np.float64("nan"))

    def test_numpy_float64_inf_raises(self) -> None:
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(np.float64("inf"))

    def test_numpy_bool_converts_to_bool(self) -> None:
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(np.bool_(True))
        assert result is True
        assert type(result) is bool

    def test_numpy_array_converts_to_list(self) -> None:
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert type(result) is list
        assert all(type(x) is int for x in result)
