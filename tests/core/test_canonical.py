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


import pandas as pd


class TestPandasTypeConversion:
    """Pandas types must be converted to JSON-safe primitives."""

    def test_pandas_timestamp_naive_to_utc_iso(self) -> None:
        from elspeth.core.canonical import _normalize_value

        ts = pd.Timestamp("2026-01-12 10:30:00")
        result = _normalize_value(ts)
        assert result == "2026-01-12T10:30:00+00:00"
        assert type(result) is str

    def test_pandas_timestamp_aware_to_utc_iso(self) -> None:
        from elspeth.core.canonical import _normalize_value

        ts = pd.Timestamp("2026-01-12 10:30:00", tz="US/Eastern")
        result = _normalize_value(ts)
        # Should be converted to UTC
        assert "+00:00" in result or "Z" in result
        assert type(result) is str

    def test_pandas_nat_to_none(self) -> None:
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(pd.NaT)
        assert result is None

    def test_pandas_na_to_none(self) -> None:
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(pd.NA)
        assert result is None


import base64
from datetime import datetime, timezone
from decimal import Decimal


class TestSpecialTypeConversion:
    """Special Python types must be converted consistently."""

    def test_datetime_naive_to_utc_iso(self) -> None:
        from elspeth.core.canonical import _normalize_value

        dt = datetime(2026, 1, 12, 10, 30, 0)
        result = _normalize_value(dt)
        assert result == "2026-01-12T10:30:00+00:00"

    def test_datetime_aware_to_utc_iso(self) -> None:
        from elspeth.core.canonical import _normalize_value

        dt = datetime(2026, 1, 12, 10, 30, 0, tzinfo=timezone.utc)
        result = _normalize_value(dt)
        assert result == "2026-01-12T10:30:00+00:00"

    def test_bytes_to_base64_wrapper(self) -> None:
        from elspeth.core.canonical import _normalize_value

        data = b"hello world"
        result = _normalize_value(data)
        assert result == {"__bytes__": base64.b64encode(data).decode("ascii")}

    def test_decimal_to_string(self) -> None:
        from elspeth.core.canonical import _normalize_value

        # Decimal preserves precision as string
        result = _normalize_value(Decimal("123.456789012345"))
        assert result == "123.456789012345"
        assert type(result) is str


class TestRecursiveNormalization:
    """Nested structures must be normalized recursively."""

    def test_dict_with_numpy_values(self) -> None:
        from elspeth.core.canonical import _normalize_for_canonical

        data = {"count": np.int64(42), "rate": np.float64(3.14)}
        result = _normalize_for_canonical(data)
        assert result == {"count": 42, "rate": 3.14}
        assert type(result["count"]) is int
        assert type(result["rate"]) is float

    def test_list_with_mixed_types(self) -> None:
        from elspeth.core.canonical import _normalize_for_canonical

        data = [np.int64(1), pd.Timestamp("2026-01-12"), None]
        result = _normalize_for_canonical(data)
        assert result[0] == 1
        assert "2026-01-12" in result[1]
        assert result[2] is None

    def test_nested_dict(self) -> None:
        from elspeth.core.canonical import _normalize_for_canonical

        data = {
            "outer": {
                "inner": np.int64(42),
                "list": [np.float64(1.0), np.float64(2.0)],
            }
        }
        result = _normalize_for_canonical(data)
        assert result["outer"]["inner"] == 42
        assert result["outer"]["list"] == [1.0, 2.0]

    def test_tuple_converts_to_list(self) -> None:
        from elspeth.core.canonical import _normalize_for_canonical

        data = (1, 2, 3)
        result = _normalize_for_canonical(data)
        assert result == [1, 2, 3]
        assert type(result) is list

    def test_nan_in_nested_raises(self) -> None:
        from elspeth.core.canonical import _normalize_for_canonical

        data = {"values": [1.0, float("nan"), 3.0]}
        with pytest.raises(ValueError, match="non-finite"):
            _normalize_for_canonical(data)
