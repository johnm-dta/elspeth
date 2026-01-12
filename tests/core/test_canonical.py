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
