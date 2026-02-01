# tests/property/canonical/test_hash_determinism.py
"""Property-based tests for canonical JSON determinism.

These tests verify the foundational property of ELSPETH's audit trail:
same input data MUST always produce the same hash. Non-deterministic
hashing would make the audit trail meaningless.

The tests use Hypothesis to generate thousands of random inputs and
verify that determinism holds for ALL of them, not just the specific
examples we think of.
"""

from __future__ import annotations

import string
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.core.canonical import (
    CANONICAL_VERSION,
    canonical_json,
    stable_hash,
)
from tests.property.conftest import (
    MAX_SAFE_INT,
    MIN_SAFE_INT,
    dict_keys,
    json_primitives,
    json_values,
    row_data,
)

# =============================================================================
# Core Determinism Properties
# =============================================================================


class TestCanonicalJsonDeterminism:
    """Property tests for canonical_json() determinism."""

    @given(data=json_values)
    @settings(max_examples=500)
    def test_canonical_json_is_deterministic(self, data: Any) -> None:
        """Property: canonical_json(x) == canonical_json(x) for all valid inputs.

        This is THE foundational property. If this fails, the audit trail
        is worthless because the same data could produce different hashes.
        """
        result1 = canonical_json(data)
        result2 = canonical_json(data)
        assert result1 == result2, f"Non-deterministic output for input: {data!r}"

    @given(data=json_values)
    @settings(max_examples=500)
    def test_canonical_json_returns_string(self, data: Any) -> None:
        """Property: canonical_json() always returns a string."""
        result = canonical_json(data)
        assert isinstance(result, str)

    @given(data=json_values)
    @settings(max_examples=200)
    def test_canonical_json_is_valid_json(self, data: Any) -> None:
        """Property: canonical_json() output is valid JSON."""
        import json

        result = canonical_json(data)
        # Should not raise
        parsed = json.loads(result)
        # Round-trip should preserve structure (though types may change)
        assert parsed is not None or data is None

    @given(
        data=st.dictionaries(
            st.text(min_size=1, max_size=8, alphabet=string.ascii_letters + string.digits),
            st.one_of(
                st.integers(min_value=MIN_SAFE_INT, max_value=MAX_SAFE_INT),
                st.text(max_size=20, alphabet=string.ascii_letters + string.digits),
                st.booleans(),
                st.none(),
            ),
            min_size=1,
            max_size=8,
        )
    )
    @settings(max_examples=300)
    def test_canonical_json_sorts_ascii_keys_and_compact(self, data: dict[str, Any]) -> None:
        """Property: ASCII-keyed dicts are sorted and compact per RFC 8785.

        For ASCII keys and primitive values, RFC 8785 ordering matches
        json.dumps(sort_keys=True) and uses compact separators.
        """
        import json

        result = canonical_json(data)
        expected = json.dumps(data, sort_keys=True, separators=(",", ":"))
        assert result == expected


class TestStableHashDeterminism:
    """Property tests for stable_hash() determinism."""

    @given(data=json_values)
    @settings(max_examples=500)
    def test_stable_hash_is_deterministic(self, data: Any) -> None:
        """Property: stable_hash(x) == stable_hash(x) for all valid inputs.

        If hashes aren't deterministic, then:
        - explain() queries would return inconsistent results
        - Payload deduplication would fail
        - Integrity verification would be impossible
        """
        hash1 = stable_hash(data)
        hash2 = stable_hash(data)
        assert hash1 == hash2, f"Non-deterministic hash for input: {data!r}"

    @given(data=json_values)
    @settings(max_examples=300)
    def test_stable_hash_returns_hex_string(self, data: Any) -> None:
        """Property: stable_hash() returns a valid SHA-256 hex string."""
        result = stable_hash(data)
        assert isinstance(result, str)
        assert len(result) == 64, f"Expected 64 hex chars, got {len(result)}"
        assert all(c in "0123456789abcdef" for c in result), f"Invalid hex: {result}"

    @given(data1=json_values, data2=json_values)
    @settings(max_examples=300)
    def test_different_data_different_hash(self, data1: Any, data2: Any) -> None:
        """Property: Different data should (almost always) produce different hashes.

        Note: Hash collisions are theoretically possible but astronomically
        unlikely for SHA-256. This test verifies we're not accidentally
        producing constant hashes.
        """
        assume(data1 != data2)
        hash1 = stable_hash(data1)
        hash2 = stable_hash(data2)
        # Not a strict requirement (collisions exist), but should hold
        # for any reasonable set of test cases
        assert hash1 != hash2, f"Collision: {data1!r} and {data2!r} have same hash"

    @given(data=row_data)
    @settings(max_examples=200)
    def test_hash_version_parameter_works(self, data: dict[str, Any]) -> None:
        """Property: Version parameter is accepted (for future compatibility)."""
        hash1 = stable_hash(data)
        hash2 = stable_hash(data, version=CANONICAL_VERSION)
        assert hash1 == hash2


# =============================================================================
# Pandas/NumPy Type Handling Properties
# =============================================================================


class TestPandasNumpyNormalization:
    """Property tests for pandas/numpy type normalization."""

    @given(value=st.integers(min_value=MIN_SAFE_INT, max_value=MAX_SAFE_INT))
    @settings(max_examples=200)
    def test_numpy_int64_deterministic(self, value: int) -> None:
        """Property: numpy.int64 values hash deterministically.

        Note: Uses JavaScript-safe integer range for RFC 8785 compatibility.
        """
        np_value = np.int64(value)
        hash1 = stable_hash({"value": np_value})
        hash2 = stable_hash({"value": np_value})
        assert hash1 == hash2

    @given(value=st.integers(min_value=MIN_SAFE_INT, max_value=MAX_SAFE_INT))
    @settings(max_examples=200)
    def test_numpy_int64_same_as_python_int(self, value: int) -> None:
        """Property: numpy.int64 produces same hash as equivalent Python int.

        Note: Uses JavaScript-safe integer range for RFC 8785 compatibility.
        """
        np_value = np.int64(value)
        hash_np = stable_hash({"value": np_value})
        hash_py = stable_hash({"value": value})
        assert hash_np == hash_py, f"numpy.int64({value}) != int({value})"

    @given(value=st.floats(allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_numpy_float64_deterministic(self, value: float) -> None:
        """Property: numpy.float64 values hash deterministically."""
        np_value = np.float64(value)
        hash1 = stable_hash({"value": np_value})
        hash2 = stable_hash({"value": np_value})
        assert hash1 == hash2

    @given(value=st.booleans())
    @settings(max_examples=50)
    def test_numpy_bool_same_as_python_bool(self, value: bool) -> None:
        """Property: numpy.bool_ produces same hash as equivalent Python bool."""
        np_value = np.bool_(value)
        hash_np = stable_hash({"value": np_value})
        hash_py = stable_hash({"value": value})
        assert hash_np == hash_py

    @given(
        year=st.integers(min_value=1970, max_value=2100),
        month=st.integers(min_value=1, max_value=12),
        day=st.integers(min_value=1, max_value=28),
        hour=st.integers(min_value=0, max_value=23),
        minute=st.integers(min_value=0, max_value=59),
    )
    @settings(max_examples=100)
    def test_pandas_timestamp_deterministic(self, year: int, month: int, day: int, hour: int, minute: int) -> None:
        """Property: pandas.Timestamp values hash deterministically."""
        ts = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, tz="UTC")
        hash1 = stable_hash({"timestamp": ts})
        hash2 = stable_hash({"timestamp": ts})
        assert hash1 == hash2

    def test_pandas_nat_deterministic(self) -> None:
        """Property: pd.NaT hashes deterministically to null."""
        hash1 = stable_hash({"value": pd.NaT})
        hash2 = stable_hash({"value": pd.NaT})
        assert hash1 == hash2

        # NaT should hash same as None
        hash_none = stable_hash({"value": None})
        assert hash1 == hash_none

    def test_pandas_na_deterministic(self) -> None:
        """Property: pd.NA hashes deterministically to null."""
        hash1 = stable_hash({"value": pd.NA})
        hash2 = stable_hash({"value": pd.NA})
        assert hash1 == hash2

        # NA should hash same as None
        hash_none = stable_hash({"value": None})
        assert hash1 == hash_none


# =============================================================================
# Special Type Properties
# =============================================================================


class TestSpecialTypes:
    """Property tests for special type handling (bytes, Decimal, datetime)."""

    @given(data=st.binary(max_size=100))
    @settings(max_examples=100)
    def test_bytes_deterministic(self, data: bytes) -> None:
        """Property: bytes values hash deterministically."""
        hash1 = stable_hash({"binary": data})
        hash2 = stable_hash({"binary": data})
        assert hash1 == hash2

    @given(
        int_part=st.integers(min_value=-(10**10), max_value=10**10),
        dec_places=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100)
    def test_decimal_deterministic(self, int_part: int, dec_places: int) -> None:
        """Property: Decimal values hash deterministically."""
        # Create a Decimal with specific precision
        value = Decimal(int_part) / Decimal(10**dec_places) if dec_places > 0 else Decimal(int_part)
        hash1 = stable_hash({"decimal": value})
        hash2 = stable_hash({"decimal": value})
        assert hash1 == hash2

    @given(
        year=st.integers(min_value=1970, max_value=2100),
        month=st.integers(min_value=1, max_value=12),
        day=st.integers(min_value=1, max_value=28),
        hour=st.integers(min_value=0, max_value=23),
        minute=st.integers(min_value=0, max_value=59),
        second=st.integers(min_value=0, max_value=59),
    )
    @settings(max_examples=100)
    def test_datetime_utc_deterministic(self, year: int, month: int, day: int, hour: int, minute: int, second: int) -> None:
        """Property: datetime values with UTC hash deterministically."""
        dt = datetime(year, month, day, hour, minute, second, tzinfo=UTC)
        hash1 = stable_hash({"datetime": dt})
        hash2 = stable_hash({"datetime": dt})
        assert hash1 == hash2

    @given(
        year=st.integers(min_value=1970, max_value=2100),
        month=st.integers(min_value=1, max_value=12),
        day=st.integers(min_value=1, max_value=28),
    )
    @settings(max_examples=50)
    def test_naive_datetime_treated_as_utc(self, year: int, month: int, day: int) -> None:
        """Property: Naive datetime is treated as UTC (consistent policy)."""
        # Create naive datetime by removing timezone from UTC datetime
        utc_dt = datetime(year, month, day, 12, 0, 0, tzinfo=UTC)
        naive_dt = utc_dt.replace(tzinfo=None)

        hash_naive = stable_hash({"datetime": naive_dt})
        hash_utc = stable_hash({"datetime": utc_dt})

        assert hash_naive == hash_utc, "Naive datetime should be treated as UTC"


# =============================================================================
# Structural Properties
# =============================================================================


class TestStructuralProperties:
    """Property tests for structural invariants."""

    @given(data=json_values)
    @settings(max_examples=100)
    def test_canonical_json_no_whitespace(self, data: Any) -> None:
        """Property: Canonical JSON is compact for simple structures."""
        result = canonical_json(data)
        assert "\n" not in result, "Newlines in canonical JSON"

    def test_canonical_json_compact_for_simple_object(self) -> None:
        """Canonical JSON uses compact separators for simple objects."""
        result = canonical_json({"b": 1, "a": 2})
        assert result == '{"a":2,"b":1}'

    @given(data=st.dictionaries(dict_keys, json_primitives, min_size=2, max_size=10))
    @settings(max_examples=100)
    def test_dict_key_order_independent(self, data: dict[str, Any]) -> None:
        """Property: Dictionary hash is independent of insertion order.

        This is crucial for determinism - dicts created with different
        insertion orders must produce the same hash.
        """
        items = list(data.items())
        if len(items) < 2:
            return
        reversed_items = list(reversed(items))
        if reversed_items == items:
            return

        dict_from_shuffled = dict(reversed_items)

        hash1 = stable_hash(data)
        hash2 = stable_hash(dict_from_shuffled)

        assert hash1 == hash2, "Dict hash should be independent of key order"

    @given(values=st.lists(json_primitives, min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_list_order_matters(self, values: list[Any]) -> None:
        """Property: List hash depends on element order.

        Unlike dicts, list order is semantically meaningful.
        """
        if len(values) < 2 or len({str(v) for v in values}) == 1:
            # Can't test order dependence with identical elements
            return

        reversed_values = list(reversed(values))
        if values == reversed_values:
            # Palindrome - skip
            return

        hash1 = stable_hash({"list": values})
        hash2 = stable_hash({"list": reversed_values})

        assert hash1 != hash2, "List hash should depend on element order"
