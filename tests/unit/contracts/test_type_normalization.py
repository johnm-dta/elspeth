"""Tests for type normalization utility.

This module tests normalize_type_for_contract() which converts numpy/pandas
types to Python primitives for consistent contract storage and validation.
"""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from elspeth.contracts.type_normalization import (
    ALLOWED_CONTRACT_TYPES,
    CONTRACT_TYPE_MAP,
    UNSUPPORTED_CONTRACT_TYPE,
    normalize_type_for_contract,
    require_supported_contract_type,
)


class TestNormalizeTypeForContract:
    """Tests for normalize_type_for_contract function."""

    # -------------------------------------------------------------------------
    # Python primitives pass through
    # -------------------------------------------------------------------------

    def test_none_returns_nonetype(self) -> None:
        """None -> type(None)."""
        result = normalize_type_for_contract(None)
        assert result is type(None)

    def test_int_returns_int(self) -> None:
        """42 -> int."""
        result = normalize_type_for_contract(42)
        assert result is int

    def test_str_returns_str(self) -> None:
        """'hello' -> str."""
        result = normalize_type_for_contract("hello")
        assert result is str

    def test_float_returns_float(self) -> None:
        """3.14 -> float."""
        result = normalize_type_for_contract(3.14)
        assert result is float

    def test_bool_returns_bool(self) -> None:
        """True -> bool."""
        result = normalize_type_for_contract(True)
        assert result is bool

    # -------------------------------------------------------------------------
    # NumPy types normalize to primitives
    # -------------------------------------------------------------------------

    def test_numpy_int64_returns_int(self) -> None:
        """np.int64(42) -> int."""
        result = normalize_type_for_contract(np.int64(42))
        assert result is int

    def test_numpy_int32_returns_int(self) -> None:
        """np.int32(42) -> int."""
        result = normalize_type_for_contract(np.int32(42))
        assert result is int

    def test_numpy_float64_returns_float(self) -> None:
        """np.float64(3.14) -> float."""
        result = normalize_type_for_contract(np.float64(3.14))
        assert result is float

    def test_numpy_float32_returns_float(self) -> None:
        """np.float32(3.14) -> float."""
        result = normalize_type_for_contract(np.float32(3.14))
        assert result is float

    def test_numpy_bool_returns_bool(self) -> None:
        """np.bool_(True) -> bool."""
        result = normalize_type_for_contract(np.bool_(True))
        assert result is bool

    # -------------------------------------------------------------------------
    # Pandas types normalize to primitives
    # -------------------------------------------------------------------------

    def test_pandas_timestamp_returns_datetime(self) -> None:
        """pd.Timestamp('2024-01-01') -> datetime."""
        result = normalize_type_for_contract(pd.Timestamp("2024-01-01"))
        assert result is datetime

    def test_pandas_nat_returns_nonetype(self) -> None:
        """pd.NaT (Not a Time) normalizes to type(None) like None."""
        assert normalize_type_for_contract(pd.NaT) is type(None)

    def test_pandas_na_returns_nonetype(self) -> None:
        """pd.NA (missing scalar) normalizes to type(None)."""
        assert normalize_type_for_contract(pd.NA) is type(None)

    def test_numpy_datetime64_returns_datetime(self) -> None:
        """np.datetime64('2024-01-01') -> datetime."""
        result = normalize_type_for_contract(np.datetime64("2024-01-01"))
        assert result is datetime

    def test_numpy_datetime64_nat_returns_nonetype(self) -> None:
        """np.datetime64('NaT') normalizes to type(None)."""
        assert normalize_type_for_contract(np.datetime64("NaT")) is type(None)

    # -------------------------------------------------------------------------
    # NaN/Infinity rejection (Tier 1 audit integrity)
    # -------------------------------------------------------------------------

    def test_float_nan_raises_valueerror(self) -> None:
        """float('nan') raises ValueError with 'non-finite'."""
        with pytest.raises(ValueError, match="non-finite"):
            normalize_type_for_contract(float("nan"))

    def test_float_inf_raises_valueerror(self) -> None:
        """float('inf') raises ValueError with 'non-finite'."""
        with pytest.raises(ValueError, match="non-finite"):
            normalize_type_for_contract(float("inf"))

    def test_float_negative_inf_raises_valueerror(self) -> None:
        """float('-inf') raises ValueError with 'non-finite'."""
        with pytest.raises(ValueError, match="non-finite"):
            normalize_type_for_contract(float("-inf"))

    def test_numpy_nan_raises_valueerror(self) -> None:
        """np.nan raises ValueError."""
        with pytest.raises(ValueError, match="non-finite"):
            normalize_type_for_contract(np.nan)

    def test_numpy_inf_raises_valueerror(self) -> None:
        """np.inf raises ValueError."""
        with pytest.raises(ValueError, match="non-finite"):
            normalize_type_for_contract(np.inf)

    def test_numpy_float64_nan_raises_valueerror(self) -> None:
        """np.float64('nan') raises ValueError."""
        with pytest.raises(ValueError, match="non-finite"):
            normalize_type_for_contract(np.float64("nan"))

    def test_numpy_float64_inf_raises_valueerror(self) -> None:
        """np.float64('inf') raises ValueError."""
        with pytest.raises(ValueError, match="non-finite"):
            normalize_type_for_contract(np.float64("inf"))

    # -------------------------------------------------------------------------
    # Unknown types return an explicit signal
    # -------------------------------------------------------------------------

    def test_decimal_returns_unsupported_signal(self) -> None:
        """Decimal returns sentinel - not serializable in checkpoint."""
        assert normalize_type_for_contract(Decimal("100.50")) is UNSUPPORTED_CONTRACT_TYPE

    def test_list_returns_unsupported_signal(self) -> None:
        """list returns sentinel - use 'any' type for complex fields."""
        assert normalize_type_for_contract([1, 2, 3]) is UNSUPPORTED_CONTRACT_TYPE

    def test_dict_returns_unsupported_signal(self) -> None:
        """dict returns sentinel - use 'any' type for complex fields."""
        assert normalize_type_for_contract({"a": 1}) is UNSUPPORTED_CONTRACT_TYPE

    def test_uuid_returns_unsupported_signal(self) -> None:
        """UUID returns sentinel - not serializable in checkpoint."""
        assert normalize_type_for_contract(UUID("12345678-1234-5678-1234-567812345678")) is UNSUPPORTED_CONTRACT_TYPE

    def test_custom_class_returns_unsupported_signal(self) -> None:
        """Custom class returns sentinel - not serializable in checkpoint."""

        class CustomClass:
            pass

        assert normalize_type_for_contract(CustomClass()) is UNSUPPORTED_CONTRACT_TYPE

    def test_require_supported_contract_type_raises_typeerror(self) -> None:
        """Fail-fast callers can still request a TypeError for unsupported types."""
        with pytest.raises(TypeError, match=r"Unsupported type.*Decimal"):
            require_supported_contract_type(Decimal("100.50"))


class TestContractTypeMapConsistency:
    """Verify CONTRACT_TYPE_MAP and ALLOWED_CONTRACT_TYPES stay in sync."""

    def test_allowed_types_equals_map_values(self) -> None:
        """ALLOWED_CONTRACT_TYPES must be exactly frozenset(CONTRACT_TYPE_MAP.values())."""
        assert frozenset(CONTRACT_TYPE_MAP.values()) == ALLOWED_CONTRACT_TYPES

    def test_allowed_types_is_frozenset(self) -> None:
        assert type(ALLOWED_CONTRACT_TYPES) is frozenset

    def test_map_contains_core_python_types(self) -> None:
        """CONTRACT_TYPE_MAP must include at minimum the core serializable types."""
        expected_keys = {"int", "str", "float", "bool", "NoneType", "datetime", "object"}
        assert set(CONTRACT_TYPE_MAP.keys()) == expected_keys

    def test_map_values_are_types(self) -> None:
        for key, value in CONTRACT_TYPE_MAP.items():
            assert isinstance(value, type), f"CONTRACT_TYPE_MAP[{key!r}] is {value!r}, not a type"


class TestEdgeCases:
    """Edge cases for type normalization."""

    def test_numpy_str_returns_str(self) -> None:
        """np.str_('hello') -> str."""
        result = normalize_type_for_contract(np.str_("hello"))
        assert result is str

    def test_numpy_bytes_returns_unsupported_signal(self) -> None:
        """numpy.bytes_ is not silently coerced to str."""
        assert normalize_type_for_contract(np.bytes_(b"hello")) is UNSUPPORTED_CONTRACT_TYPE

    def test_zero_float_is_valid(self) -> None:
        """0.0 is a valid float (not NaN/Infinity)."""
        result = normalize_type_for_contract(0.0)
        assert result is float

    def test_negative_float_is_valid(self) -> None:
        """-3.14 is a valid float."""
        result = normalize_type_for_contract(-3.14)
        assert result is float

    def test_numpy_zero_is_valid(self) -> None:
        """np.float64(0.0) is a valid float."""
        result = normalize_type_for_contract(np.float64(0.0))
        assert result is float

    def test_bool_false_returns_bool(self) -> None:
        """False -> bool (not confused with 0)."""
        result = normalize_type_for_contract(False)
        assert result is bool

    def test_empty_string_returns_str(self) -> None:
        """'' -> str."""
        result = normalize_type_for_contract("")
        assert result is str

    def test_datetime_returns_datetime(self) -> None:
        """Native datetime passes through."""
        result = normalize_type_for_contract(datetime(2024, 1, 1, tzinfo=UTC))
        assert result is datetime
