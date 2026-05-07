# tests/core/test_canonical.py
"""Tests for canonical JSON serialization and hashing."""

import base64
import hashlib
from datetime import UTC, date, datetime, time
from decimal import Decimal
from uuid import UUID

import numpy as np
import pandas as pd
import pytest


class TestNanInfinityRejection:
    """NaN and Infinity must be rejected, not silently converted.

    NOTE: Basic float NaN/Infinity rejection tests are in test_canonical_mutation_gaps.py
    (TestNanInfinityOrLogic, TestTypeCheckCoversBothFloatTypes, TestNormalFloatsPassThrough)
    which have better documentation of mutation testing targets.

    This class focuses on numpy ARRAY handling which is not covered there.
    """

    def test_numpy_array_with_nan_rejected(self) -> None:
        """BUG-CANON-01: Multi-dimensional arrays with NaN must be rejected."""
        from elspeth.core.canonical import _normalize_value

        # 2D array with NaN
        array_with_nan = np.array([[1.0, float("nan")], [2.0, 3.0]])
        with pytest.raises(ValueError, match="NaN/Infinity found in NumPy array"):
            _normalize_value(array_with_nan)

    def test_numpy_array_with_inf_rejected(self) -> None:
        """BUG-CANON-01: Multi-dimensional arrays with Infinity must be rejected."""
        from elspeth.core.canonical import _normalize_value

        # 3D array with Inf
        array_with_inf = np.array([[[1.0, 2.0], [3.0, float("inf")]], [[5.0, 6.0], [7.0, 8.0]]])
        with pytest.raises(ValueError, match="NaN/Infinity found in NumPy array"):
            _normalize_value(array_with_inf)

    def test_numpy_array_all_finite_accepted(self) -> None:
        """BUG-CANON-01: Arrays with all finite values should pass validation."""
        from elspeth.core.canonical import _normalize_value

        # Valid 2D array
        valid_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _normalize_value(valid_array)
        assert result == [[1.0, 2.0], [3.0, 4.0]]

        # Valid 1D array
        valid_1d = np.array([1.0, 2.0, 3.0])
        result_1d = _normalize_value(valid_1d)
        assert result_1d == [1.0, 2.0, 3.0]

        # Empty array (edge case)
        empty_array = np.array([])
        result_empty = _normalize_value(empty_array)
        assert result_empty == []

    def test_numpy_string_array_passes(self) -> None:
        """BUG-CANON-01: Non-numeric arrays should not raise TypeError."""
        from elspeth.core.canonical import _normalize_value

        # String arrays don't have NaN/Inf, should pass
        string_array = np.array(["hello", "world"])
        result = _normalize_value(string_array)
        assert result == ["hello", "world"]


class TestDecimalNonFiniteRejection:
    """Decimal NaN and Infinity must be rejected like float NaN/Infinity.

    Per CLAUDE.md: "NaN and Infinity are strictly rejected, not silently converted."
    This applies to Decimal as well as float - both are numeric types that can
    represent non-finite values which would corrupt audit hash integrity.
    """

    @pytest.mark.parametrize(
        "value",
        [
            Decimal("NaN"),
            Decimal("sNaN"),  # Signaling NaN - also non-finite
            Decimal("Infinity"),
            Decimal("-Infinity"),
        ],
    )
    def test_decimal_non_finite_raises_value_error(self, value: Decimal) -> None:
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(value)

    def test_decimal_normal_values_allowed(self) -> None:
        from elspeth.core.canonical import _normalize_value

        # These should NOT raise - all are finite Decimal values
        assert _normalize_value(Decimal("0")) == "0"
        assert _normalize_value(Decimal("-0")) == "-0"  # Decimal preserves signed zero
        assert _normalize_value(Decimal("-123.456")) == "-123.456"
        assert _normalize_value(Decimal("1E+100")) == "1E+100"
        assert _normalize_value(Decimal("1E-100")) == "1E-100"
        assert _normalize_value(Decimal("123.456789012345678901234567890")) == "123.456789012345678901234567890"


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

    # NOTE: numpy float64 NaN/Infinity rejection tests are in test_canonical_mutation_gaps.py
    # (TestTypeCheckCoversBothFloatTypes) which tests float32, float64, and Python floats

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

    def test_numpy_zero_dim_array_converts_to_primitive(self) -> None:
        """BUG-CANON-02: Scalar arrays should normalize as scalar primitives."""
        from elspeth.core.canonical import _normalize_value

        int_result = _normalize_value(np.array(5))
        assert int_result == 5
        assert type(int_result) is int

        float_result = _normalize_value(np.array(3.5))
        assert float_result == 3.5
        assert type(float_result) is float


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


class TestSpecialTypeConversion:
    """Special Python types must be converted consistently."""

    def test_datetime_naive_to_utc_iso(self) -> None:
        from elspeth.core.canonical import _normalize_value

        # Naive datetime (no tzinfo) - test that _normalize_value treats it as UTC
        dt = datetime(2026, 1, 12, 10, 30, 0, tzinfo=None)  # noqa: DTZ001
        result = _normalize_value(dt)
        assert result == "2026-01-12T10:30:00+00:00"

    def test_datetime_aware_to_utc_iso(self) -> None:
        from elspeth.core.canonical import _normalize_value

        dt = datetime(2026, 1, 12, 10, 30, 0, tzinfo=UTC)
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

    def test_date_to_iso_string(self) -> None:
        """date objects convert to ISO 8601 date strings."""
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(date(2024, 1, 15))
        assert result == "2024-01-15"
        assert type(result) is str

    def test_time_to_iso_string(self) -> None:
        """time objects convert to ISO 8601 time strings."""
        from elspeth.core.canonical import _normalize_value

        # Without microseconds
        result = _normalize_value(time(14, 30, 0))
        assert result == "14:30:00"
        assert type(result) is str

        # With microseconds
        result_micro = _normalize_value(time(14, 30, 0, 123456))
        assert result_micro == "14:30:00.123456"

    def test_uuid_to_string(self) -> None:
        """UUID objects convert to their string representation."""
        from elspeth.core.canonical import _normalize_value

        u = UUID("12345678-1234-5678-1234-567812345678")
        result = _normalize_value(u)
        assert result == "12345678-1234-5678-1234-567812345678"
        assert type(result) is str


class TestDateTimeUuidCanonicalJson:
    """Regression tests for date, time, and UUID canonical JSON serialization.

    Bug fix: These types were not handled by _normalize_value, causing
    CanonicalizationError when they appeared in coalesce collision values.
    """

    def test_canonical_json_handles_date(self) -> None:
        """date objects serialize correctly in canonical JSON."""
        from elspeth.core.canonical import canonical_json

        result = canonical_json({"d": date(2024, 1, 15)})
        assert result == '{"d":"2024-01-15"}'

    def test_canonical_json_handles_time(self) -> None:
        """time objects serialize correctly in canonical JSON."""
        from elspeth.core.canonical import canonical_json

        result = canonical_json({"t": time(14, 30, 0)})
        assert '"t":"14:30:00"' in result

    def test_canonical_json_handles_time_with_microseconds(self) -> None:
        """time objects with microseconds serialize correctly."""
        from elspeth.core.canonical import canonical_json

        result = canonical_json({"t": time(14, 30, 0, 123456)})
        assert '"t":"14:30:00.123456"' in result

    def test_canonical_json_handles_uuid(self) -> None:
        """UUID objects serialize correctly in canonical JSON."""
        from elspeth.core.canonical import canonical_json

        u = UUID("12345678-1234-5678-1234-567812345678")
        result = canonical_json({"id": u})
        assert result == '{"id":"12345678-1234-5678-1234-567812345678"}'

    def test_stable_hash_date_deterministic(self) -> None:
        """date values produce deterministic hashes."""
        from elspeth.core.canonical import stable_hash

        hash1 = stable_hash({"d": date(2024, 1, 15)})
        hash2 = stable_hash({"d": date(2024, 1, 15)})
        assert hash1 == hash2

    def test_stable_hash_time_deterministic(self) -> None:
        """time values produce deterministic hashes."""
        from elspeth.core.canonical import stable_hash

        hash1 = stable_hash({"t": time(14, 30, 0)})
        hash2 = stable_hash({"t": time(14, 30, 0)})
        assert hash1 == hash2

    def test_stable_hash_uuid_deterministic(self) -> None:
        """UUID values produce deterministic hashes."""
        from elspeth.core.canonical import stable_hash

        u = UUID("12345678-1234-5678-1234-567812345678")
        hash1 = stable_hash({"id": u})
        hash2 = stable_hash({"id": u})
        assert hash1 == hash2


class TestUnsupportedTypeRejection:
    """Unsupported types must be rejected during canonical serialization.

    Edge case: Types not explicitly handled by _normalize_value() pass through
    normalization unchanged and fail at RFC 8785 serialization boundary with TypeError.
    This is defense-in-depth - external system data must be validated at ingress.
    """

    def test_uuid_now_serializable(self) -> None:
        """UUID objects are now normalized to strings and serialize correctly.

        NOTE: This test was updated when UUID support was added. Previously,
        UUIDs would pass through _normalize_value unchanged and fail at
        rfc8785.dumps(). Now they are converted to strings during normalization.
        """
        from elspeth.core.canonical import canonical_json

        obj = {"id": UUID("550e8400-e29b-41d4-a716-446655440000")}
        result = canonical_json(obj)
        assert result == '{"id":"550e8400-e29b-41d4-a716-446655440000"}'

    def test_custom_class_instance_rejected(self) -> None:
        """Custom class instances must fail canonical serialization.

        Documents that application-specific types must be converted to primitives
        (str, int, dict, etc.) before reaching canonical_json().
        """
        import rfc8785

        from elspeth.core.canonical import canonical_json

        class CustomData:
            def __init__(self, value: int) -> None:
                self.value = value

        obj = {"data": CustomData(42)}

        # Custom class passes through _normalize_value but fails at rfc8785.dumps()
        with pytest.raises(rfc8785.CanonicalizationError, match="unsupported type"):
            canonical_json(obj)

    def test_set_rejected_during_serialization(self) -> None:
        """Set objects are not JSON-serializable (order non-deterministic).

        Documents that collections must be converted to lists before canonicalization.
        """
        import rfc8785

        from elspeth.core.canonical import canonical_json

        obj = {"tags": {"red", "green", "blue"}}

        # set passes through _normalize_value but fails at rfc8785.dumps()
        with pytest.raises(rfc8785.CanonicalizationError, match="unsupported type"):
            canonical_json(obj)


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


class TestCanonicalJsonSerialization:
    """RFC 8785 canonical JSON serialization."""

    def test_canonical_json_sorts_keys(self) -> None:
        from elspeth.core.canonical import canonical_json

        # Keys should be sorted alphabetically
        data = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(data)
        assert result == '{"a":2,"m":3,"z":1}'

    def test_canonical_json_no_whitespace(self) -> None:
        from elspeth.core.canonical import canonical_json

        data = {"key": "value", "list": [1, 2, 3]}
        result = canonical_json(data)
        assert " " not in result
        assert "\n" not in result

    def test_canonical_json_handles_numpy(self) -> None:
        from elspeth.core.canonical import canonical_json

        data = {"count": np.int64(42)}
        result = canonical_json(data)
        assert result == '{"count":42}'

    def test_canonical_json_handles_zero_dim_numpy_array(self) -> None:
        from elspeth.core.canonical import canonical_json

        result = canonical_json({"value": np.array(5)})
        assert result == '{"value":5}'


class TestStableHash:
    """Stable hashing with versioned algorithm."""

    def test_stable_hash_returns_hex_string(self) -> None:
        from elspeth.core.canonical import stable_hash

        result = stable_hash({"key": "value"})
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex

    def test_stable_hash_deterministic(self) -> None:
        from elspeth.core.canonical import stable_hash

        data = {"a": 1, "b": [2, 3]}
        hash1 = stable_hash(data)
        hash2 = stable_hash(data)
        assert hash1 == hash2

    def test_stable_hash_key_order_independent(self) -> None:
        from elspeth.core.canonical import stable_hash

        # Different key order should produce same hash
        hash1 = stable_hash({"a": 1, "b": 2})
        hash2 = stable_hash({"b": 2, "a": 1})
        assert hash1 == hash2

    def test_stable_hash_verifiable(self) -> None:
        from elspeth.core.canonical import canonical_json, stable_hash

        data = {"test": "data"}
        hash_result = stable_hash(data)
        # Should match manual computation
        canonical = canonical_json(data)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert hash_result == expected

    def test_stable_hash_handles_zero_dim_numpy_array(self) -> None:
        from elspeth.core.canonical import stable_hash

        value_hash = stable_hash({"value": np.array(5)})
        expected_hash = stable_hash({"value": 5})
        assert value_hash == expected_hash


class TestCrossProcessStability:
    """Hash must be stable across processes and time."""

    def test_golden_hash_stability(self) -> None:
        """Verify hash matches known golden value.

        This test catches accidental changes to canonicalization.
        If it fails, audit trail integrity may be compromised.
        """
        from elspeth.core.canonical import stable_hash

        # Fixed test data with multiple types
        data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"a": 1},
        }

        # Golden hash computed once, verified forever
        golden_hash = "aed53055632a45e17618f46527c07dba463b2ae719e2f6832b2735308a3bf2e1"

        result = stable_hash(data)
        assert result == golden_hash, (
            f"Hash stability broken! Got {result}, expected {golden_hash}. This may indicate audit trail integrity issues."
        )

    def test_version_constant_exists(self) -> None:
        """CANONICAL_VERSION must be available for audit records."""
        from elspeth.core.canonical import CANONICAL_VERSION

        assert CANONICAL_VERSION == "sha256-rfc8785-v1"


class TestPublicAPI:
    """Public API must be importable from elspeth.core."""

    def test_import_from_core_module(self) -> None:
        """canonical_json and stable_hash importable from elspeth.core."""
        from elspeth.core import CANONICAL_VERSION, canonical_json, stable_hash

        # Verify they work
        assert callable(canonical_json)
        assert callable(stable_hash)
        assert isinstance(CANONICAL_VERSION, str)


class TestSanitizeForCanonical:
    """Regression: sanitize_for_canonical must handle tuples and numpy floats.

    The sanitizer is used at Tier-3 quarantine boundaries to replace non-finite
    floats with None so quarantined data can still be hashed.
    """

    def test_sanitize_tuple_with_nan(self) -> None:
        """Tuples containing NaN are sanitized (NaN replaced with None)."""
        from elspeth.core.canonical import sanitize_for_canonical

        result = sanitize_for_canonical({"x": (1.0, float("nan"), 3.0)})
        assert result == {"x": [1.0, None, 3.0]}

    def test_sanitize_tuple_with_infinity(self) -> None:
        """Tuples containing Infinity are sanitized."""
        from elspeth.core.canonical import sanitize_for_canonical

        result = sanitize_for_canonical({"x": (float("inf"),)})
        assert result == {"x": [None]}

    def test_sanitize_nested_tuple(self) -> None:
        """Nested tuples are recursively sanitized."""
        from elspeth.core.canonical import sanitize_for_canonical

        result = sanitize_for_canonical({"outer": (1.0, (float("-inf"), 2.0))})
        assert result == {"outer": [1.0, [None, 2.0]]}

    def test_sanitize_numpy_float_nan(self) -> None:
        """numpy float32/64 NaN values are replaced with None."""
        from elspeth.core.canonical import sanitize_for_canonical

        result = sanitize_for_canonical({"x": np.float32("nan")})
        assert result == {"x": None}

        result = sanitize_for_canonical({"x": np.float64("inf")})
        assert result == {"x": None}

    def test_sanitize_numpy_finite_unchanged(self) -> None:
        """Finite numpy floats are NOT replaced."""
        from elspeth.core.canonical import sanitize_for_canonical

        result = sanitize_for_canonical({"x": np.float64(3.14)})
        assert result == {"x": np.float64(3.14)}

    def test_sanitize_dict_with_mixed_nan(self) -> None:
        """Mixed dict values are sanitized correctly."""
        from elspeth.core.canonical import sanitize_for_canonical

        data = {"a": 1, "b": float("nan"), "c": "ok", "d": None}
        result = sanitize_for_canonical(data)
        assert result == {"a": 1, "b": None, "c": "ok", "d": None}

    def test_sanitize_list_with_nan(self) -> None:
        """Lists containing NaN are sanitized."""
        from elspeth.core.canonical import sanitize_for_canonical

        result = sanitize_for_canonical([1.0, float("nan"), 3.0])
        assert result == [1.0, None, 3.0]

    def test_sanitize_finite_floats_unchanged(self) -> None:
        """Finite floats pass through unchanged."""
        from elspeth.core.canonical import sanitize_for_canonical

        result = sanitize_for_canonical({"x": 3.14, "y": 0.0, "z": -1.5})
        assert result == {"x": 3.14, "y": 0.0, "z": -1.5}

    def test_sanitize_numpy_longdouble_nan(self) -> None:
        """np.longdouble NaN/Inf must be sanitized — name heuristic must catch all numpy floats."""
        from elspeth.core.canonical import sanitize_for_canonical

        result = sanitize_for_canonical({"x": np.longdouble("nan")})
        assert result == {"x": None}

        result = sanitize_for_canonical({"x": np.longdouble("inf")})
        assert result == {"x": None}

        result = sanitize_for_canonical({"x": np.longdouble("-inf")})
        assert result == {"x": None}

    def test_sanitize_numpy_longdouble_finite_unchanged(self) -> None:
        """Finite np.longdouble values pass through unchanged."""
        from elspeth.core.canonical import sanitize_for_canonical

        result = sanitize_for_canonical({"x": np.longdouble(3.14)})
        assert result == {"x": np.longdouble(3.14)}

    def test_sanitize_numpy_longdouble_large_finite_unchanged(self) -> None:
        """Finite np.longdouble values that overflow float64 must NOT be replaced with None.

        Bug fix: sanitize_for_canonical used math.isfinite(float(obj)) which
        falsely treats large-but-finite np.longdouble values as non-finite
        because float() overflows to inf for values outside IEEE 754 double range.
        """
        from elspeth.core.canonical import sanitize_for_canonical

        # 2^1024 is finite in np.longdouble (max_exp=16384) but overflows float64
        large_val = np.longdouble(2.0) ** np.longdouble(1024)
        assert np.isfinite(large_val), "Test precondition: value must be finite in longdouble"

        result = sanitize_for_canonical({"x": large_val})
        # Sanitizer only replaces truly non-finite values — this is finite
        assert result["x"] is not None
        assert np.isfinite(result["x"])


class TestSanitizeNumpyArrays:
    """Regression tests for elspeth-4122ad82e0: np.ndarray was missing from
    sanitize_for_canonical(), causing quarantine hashing to fail on arrays
    containing NaN/Inf.
    """

    def test_sanitize_ndarray_with_nan(self) -> None:
        """np.ndarray containing NaN is converted to list with None."""
        from elspeth.core.canonical import sanitize_for_canonical

        arr = np.array([1.0, float("nan"), 3.0])
        result = sanitize_for_canonical({"x": arr})
        assert result == {"x": [1.0, None, 3.0]}

    def test_sanitize_ndarray_with_inf(self) -> None:
        """np.ndarray containing Inf is converted to list with None."""
        from elspeth.core.canonical import sanitize_for_canonical

        arr = np.array([float("inf"), 2.0, float("-inf")])
        result = sanitize_for_canonical({"x": arr})
        assert result == {"x": [None, 2.0, None]}

    def test_sanitize_ndarray_finite_unchanged(self) -> None:
        """np.ndarray with all-finite values is converted to list, values preserved."""
        from elspeth.core.canonical import sanitize_for_canonical

        arr = np.array([1.0, 2.0, 3.0])
        result = sanitize_for_canonical({"x": arr})
        assert result == {"x": [1.0, 2.0, 3.0]}

    def test_sanitize_ndarray_2d(self) -> None:
        """Multi-dimensional np.ndarray is recursively sanitized."""
        from elspeth.core.canonical import sanitize_for_canonical

        arr = np.array([[1.0, float("nan")], [float("inf"), 4.0]])
        result = sanitize_for_canonical({"x": arr})
        assert result == {"x": [[1.0, None], [None, 4.0]]}


class TestNumpyLongdoubleCanonicalOverflow:
    """np.longdouble values that are finite but overflow IEEE 754 double.

    Bug fix for elspeth-7ab03de217: _normalize_value used math.isnan/isinf
    which implicitly downcast np.longdouble through float(), causing:
    - Finite values falsely rejected as non-finite
    - Distinct high-precision values collapsing onto the same hash
    """

    def test_normalize_longdouble_overflow_raises_clear_error(self) -> None:
        """Finite longdouble that overflows float64 must raise a clear error, not misreport as non-finite."""
        from elspeth.core.canonical import _normalize_value

        large_val = np.longdouble(2.0) ** np.longdouble(1024)
        assert np.isfinite(large_val), "Test precondition"

        with pytest.raises(ValueError, match="exceeds IEEE 754 double range"):
            _normalize_value(large_val)

    def test_normalize_longdouble_nan_rejected_correctly(self) -> None:
        """np.longdouble NaN must still be rejected as non-finite."""
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(np.longdouble("nan"))

    def test_normalize_longdouble_inf_rejected_correctly(self) -> None:
        """np.longdouble Infinity must still be rejected as non-finite."""
        from elspeth.core.canonical import _normalize_value

        with pytest.raises(ValueError, match="non-finite"):
            _normalize_value(np.longdouble("inf"))

    def test_normalize_longdouble_representable_value_works(self) -> None:
        """np.longdouble values within float64 range convert normally."""
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(np.longdouble(3.14))
        assert result == 3.14
        assert type(result) is float


class TestNumpyDatetime64Normalization:
    """np.datetime64 values must be normalized to ISO 8601 strings.

    BUG FIX: P1-2026-02-14 — np.datetime64 could pass schema validation
    (contract layer treats it as datetime) but crashed canonical hashing
    because _normalize_value had no branch for it.
    """

    def test_np_datetime64_converts_to_iso_string(self) -> None:
        """np.datetime64 with a valid date converts to UTC ISO 8601."""
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(np.datetime64("2024-01-01"))
        assert isinstance(result, str)
        assert "2024-01-01" in result
        assert "+00:00" in result  # UTC timezone

    def test_np_datetime64_with_time_converts_to_iso_string(self) -> None:
        """np.datetime64 with datetime precision converts correctly."""
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(np.datetime64("2024-06-15T10:30:00"))
        assert isinstance(result, str)
        assert "2024-06-15" in result
        assert "10:30:00" in result

    def test_np_datetime64_nat_converts_to_none(self) -> None:
        """np.datetime64('NaT') must convert to None (missing value)."""
        from elspeth.core.canonical import _normalize_value

        result = _normalize_value(np.datetime64("NaT"))
        assert result is None

    def test_np_datetime64_stable_hash_deterministic(self) -> None:
        """np.datetime64 values must produce stable hashes."""
        from elspeth.core.canonical import stable_hash

        data1 = {"event_time": np.datetime64("2024-01-01")}
        data2 = {"event_time": np.datetime64("2024-01-01")}

        assert stable_hash(data1) == stable_hash(data2)

    def test_np_datetime64_matches_pd_timestamp_hash(self) -> None:
        """np.datetime64 and pd.Timestamp for same date produce same hash."""
        from elspeth.core.canonical import stable_hash

        np_data = {"event_time": np.datetime64("2024-01-01")}
        pd_data = {"event_time": pd.Timestamp("2024-01-01")}

        assert stable_hash(np_data) == stable_hash(pd_data)

    def test_np_datetime64_canonical_json_succeeds(self) -> None:
        """canonical_json should handle np.datetime64 without error."""
        from elspeth.core.canonical import canonical_json

        result = canonical_json({"ts": np.datetime64("2024-03-15T08:00:00")})
        assert isinstance(result, str)
        assert "2024-03-15" in result

    def test_np_datetime64_nat_in_nested_structure(self) -> None:
        """NaT in nested structures normalizes to null in canonical JSON."""
        from elspeth.core.canonical import canonical_json

        result = canonical_json({"values": [np.datetime64("2024-01-01"), np.datetime64("NaT")]})
        assert "null" in result

    def test_np_datetime64_in_dict_with_other_types(self) -> None:
        """np.datetime64 works alongside other numpy/pandas types."""
        from elspeth.core.canonical import _normalize_for_canonical

        data = {
            "timestamp": np.datetime64("2024-01-01"),
            "count": np.int64(42),
            "rate": np.float64(3.14),
        }
        result = _normalize_for_canonical(data)
        assert isinstance(result["timestamp"], str)
        assert result["count"] == 42
        assert result["rate"] == 3.14


class TestCoreIntegration:
    """Core module integration - all Phase 1 components exportable."""

    def test_dag_importable_from_core(self) -> None:
        from elspeth.core import ExecutionGraph, GraphValidationError

        assert ExecutionGraph is not None
        assert GraphValidationError is not None

    def test_config_importable_from_core(self) -> None:
        from elspeth.core import ElspethSettings, load_settings

        assert ElspethSettings is not None
        assert callable(load_settings)

    def test_payload_store_importable_from_core(self) -> None:
        from elspeth.core import FilesystemPayloadStore, PayloadStore

        assert FilesystemPayloadStore is not None
        assert PayloadStore is not None
