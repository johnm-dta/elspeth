# tests/property/core/test_formatters_properties.py
"""Property-based tests for landscape formatters (serialization utilities).

The formatters module provides serialize_datetime() and dataclass_to_dict()
used throughout the export pipeline. These enforce audit integrity by:
- Rejecting NaN/Infinity at any nesting depth
- Converting datetimes to ISO strings
- Converting dataclasses/Enums to JSON-serializable dicts

Properties tested:
- NaN/Infinity rejection at arbitrary depth (dict and list nesting)
- datetime → ISO string conversion and parseability
- Dataclass recursive conversion
- CSVFormatter flatten idempotency
- None → empty dict conversion
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.core.landscape.formatters import (
    CSVFormatter,
    JSONFormatter,
    dataclass_to_dict,
    serialize_datetime,
)

# =============================================================================
# Strategies
# =============================================================================

# Finite floats (no NaN/Infinity)
finite_floats = st.floats(allow_nan=False, allow_infinity=False)

# Datetimes for serialization tests
test_datetimes = st.datetimes(
    min_value=datetime(2000, 1, 1),  # noqa: DTZ001 — Hypothesis requires naive bounds
    max_value=datetime(2100, 1, 1),  # noqa: DTZ001
    timezones=st.just(UTC),
)


def _nest_in_dicts(value: Any, depth: int) -> dict[str, Any]:
    """Wrap a value in nested dicts to a given depth."""
    result: Any = value
    for i in range(depth):
        result = {f"level_{i}": result}
    out: dict[str, Any] = result
    return out


def _nest_in_lists(value: Any, depth: int) -> list[Any]:
    """Wrap a value in nested lists to a given depth."""
    result: Any = value
    for _ in range(depth):
        result = [result]
    out: list[Any] = result
    return out


# =============================================================================
# NaN/Infinity Rejection Properties
# =============================================================================


class TestNaNInfinityRejection:
    """NaN and Infinity must be rejected at any nesting depth."""

    @given(depth=st.integers(min_value=0, max_value=8))
    @settings(max_examples=50)
    def test_nan_in_nested_dicts_rejected(self, depth: int) -> None:
        """Property: NaN at any dict nesting depth raises ValueError."""
        data = _nest_in_dicts(float("nan"), depth)
        with pytest.raises(ValueError, match="NaN"):
            serialize_datetime(data)

    @given(depth=st.integers(min_value=0, max_value=8))
    @settings(max_examples=50)
    def test_infinity_in_nested_dicts_rejected(self, depth: int) -> None:
        """Property: Infinity at any dict nesting depth raises ValueError."""
        data = _nest_in_dicts(float("inf"), depth)
        with pytest.raises(ValueError, match="Infinity"):
            serialize_datetime(data)

    @given(depth=st.integers(min_value=0, max_value=8))
    @settings(max_examples=50)
    def test_neg_infinity_in_nested_dicts_rejected(self, depth: int) -> None:
        """Property: -Infinity at any dict nesting depth raises ValueError."""
        data = _nest_in_dicts(float("-inf"), depth)
        with pytest.raises(ValueError, match="Infinity"):
            serialize_datetime(data)

    @given(depth=st.integers(min_value=1, max_value=8))
    @settings(max_examples=50)
    def test_nan_in_nested_lists_rejected(self, depth: int) -> None:
        """Property: NaN at any list nesting depth raises ValueError."""
        data = _nest_in_lists(float("nan"), depth)
        with pytest.raises(ValueError, match="NaN"):
            serialize_datetime(data)

    @given(depth=st.integers(min_value=1, max_value=8))
    @settings(max_examples=50)
    def test_infinity_in_nested_lists_rejected(self, depth: int) -> None:
        """Property: Infinity at any list nesting depth raises ValueError."""
        data = _nest_in_lists(float("inf"), depth)
        with pytest.raises(ValueError, match="Infinity"):
            serialize_datetime(data)

    @given(
        n_valid=st.integers(min_value=1, max_value=5),
        nan_position=st.integers(min_value=0),
    )
    @settings(max_examples=100)
    def test_nan_among_valid_values_still_rejected(self, n_valid: int, nan_position: int) -> None:
        """Property: A single NaN among valid values still raises."""
        items: list[Any] = [i * 1.0 for i in range(n_valid)]
        insert_pos = nan_position % (n_valid + 1)
        items.insert(insert_pos, float("nan"))
        with pytest.raises(ValueError, match="NaN"):
            serialize_datetime(items)


# =============================================================================
# Datetime Serialization Properties
# =============================================================================


class TestDatetimeSerialization:
    """Datetimes must convert to parseable ISO strings."""

    @given(dt=test_datetimes)
    @settings(max_examples=200)
    def test_datetime_becomes_iso_string(self, dt: datetime) -> None:
        """Property: Datetimes are converted to ISO format strings."""
        result = serialize_datetime(dt)
        assert isinstance(result, str)
        assert dt.isoformat() == result

    @given(dt=test_datetimes)
    @settings(max_examples=200)
    def test_serialized_datetime_is_parseable(self, dt: datetime) -> None:
        """Property: Serialized datetime can be parsed back to datetime."""
        result = serialize_datetime(dt)
        parsed = datetime.fromisoformat(result)
        assert parsed == dt

    @given(dt=test_datetimes)
    @settings(max_examples=100)
    def test_datetime_in_dict_converted(self, dt: datetime) -> None:
        """Property: Datetimes nested in dicts are converted."""
        data = {"timestamp": dt, "name": "test"}
        result = serialize_datetime(data)
        assert isinstance(result["timestamp"], str)
        assert result["name"] == "test"

    @given(dt=test_datetimes)
    @settings(max_examples=100)
    def test_datetime_in_list_converted(self, dt: datetime) -> None:
        """Property: Datetimes nested in lists are converted."""
        data = [dt, "other"]
        result = serialize_datetime(data)
        assert isinstance(result[0], str)
        assert result[1] == "other"

    @given(value=st.one_of(st.integers(), st.text(max_size=50), st.booleans(), st.none()))
    @settings(max_examples=200)
    def test_non_datetime_non_float_passthrough(self, value) -> None:
        """Property: Non-datetime, non-float values pass through unchanged."""
        result = serialize_datetime(value)
        assert result == value

    @given(f=finite_floats)
    @settings(max_examples=200)
    def test_finite_floats_passthrough(self, f: float) -> None:
        """Property: Finite floats pass through unchanged."""
        result = serialize_datetime(f)
        assert result == f


# =============================================================================
# dataclass_to_dict Properties
# =============================================================================


class StatusEnum(Enum):
    """Test enum for dataclass conversion tests."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


@dataclass
class InnerData:
    """Test nested dataclass."""

    value: int
    label: str


@dataclass
class OuterData:
    """Test dataclass with nested dataclass and enum."""

    name: str
    status: StatusEnum
    inner: InnerData
    items: list[int]
    timestamp: datetime | None = None


class TestDataclassToDict:
    """dataclass_to_dict must recursively convert all types."""

    def test_none_returns_empty_dict(self) -> None:
        """Property: None input returns empty dict."""
        assert dataclass_to_dict(None) == {}

    @given(
        name=st.text(min_size=1, max_size=20),
        status=st.sampled_from(list(StatusEnum)),
        value=st.integers(),
        label=st.text(max_size=20),
        items=st.lists(st.integers(), max_size=5),
    )
    @settings(max_examples=100)
    def test_dataclass_converts_to_dict(self, name: str, status: StatusEnum, value: int, label: str, items: list[int]) -> None:
        """Property: Dataclass converts to dict with all fields."""
        obj = OuterData(
            name=name,
            status=status,
            inner=InnerData(value=value, label=label),
            items=items,
        )
        result = dataclass_to_dict(obj)
        assert isinstance(result, dict)
        assert result["name"] == name
        assert result["status"] == status.value  # Enum → .value
        assert result["inner"]["value"] == value
        assert result["inner"]["label"] == label
        assert result["items"] == items

    @given(status=st.sampled_from(list(StatusEnum)))
    @settings(max_examples=20)
    def test_enum_converts_to_value(self, status: StatusEnum) -> None:
        """Property: Enum fields become their .value."""
        obj = OuterData(
            name="test",
            status=status,
            inner=InnerData(value=1, label="x"),
            items=[],
        )
        result = dataclass_to_dict(obj)
        assert result["status"] == status.value
        assert isinstance(result["status"], str)

    @given(dt=test_datetimes)
    @settings(max_examples=50)
    def test_datetime_field_serialized(self, dt: datetime) -> None:
        """Property: Datetime fields in dataclasses are serialized to ISO strings."""
        obj = OuterData(
            name="test",
            status=StatusEnum.ACTIVE,
            inner=InnerData(value=1, label="x"),
            items=[],
            timestamp=dt,
        )
        result = dataclass_to_dict(obj)
        assert isinstance(result["timestamp"], str)
        assert datetime.fromisoformat(result["timestamp"]) == dt

    @given(values=st.lists(st.integers(), min_size=0, max_size=10))
    @settings(max_examples=50)
    def test_list_of_primitives_passthrough(self, values: list[int]) -> None:
        """Property: Lists of primitives in dataclasses pass through."""
        obj = OuterData(
            name="test",
            status=StatusEnum.ACTIVE,
            inner=InnerData(value=0, label=""),
            items=values,
        )
        result = dataclass_to_dict(obj)
        assert result["items"] == values

    def test_list_of_dataclasses_converted(self) -> None:
        """Property: Lists of dataclasses are all converted."""
        items = [InnerData(value=i, label=f"item_{i}") for i in range(3)]
        result = dataclass_to_dict(items)
        assert len(result) == 3
        for i, item in enumerate(result):
            assert item["value"] == i
            assert item["label"] == f"item_{i}"

    @given(value=st.one_of(st.integers(), st.text(max_size=20), st.floats(allow_nan=False, allow_infinity=False)))
    @settings(max_examples=100)
    def test_plain_values_passthrough(self, value) -> None:
        """Property: Non-dataclass values pass through unchanged."""
        result = dataclass_to_dict(value)
        assert result == value


# =============================================================================
# CSVFormatter Flatten Properties
# =============================================================================


class TestCSVFormatterFlatten:
    """CSVFormatter.flatten must produce flat dicts with dot-notation keys."""

    @given(
        keys=st.lists(
            st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            min_size=1,
            max_size=5,
            unique=True,
        ),
        values=st.lists(st.text(max_size=20), min_size=1, max_size=5),
    )
    @settings(max_examples=100)
    def test_flat_dict_unchanged(self, keys: list[str], values: list[str]) -> None:
        """Property: Already-flat dicts are returned as-is."""
        padded = (values * ((len(keys) // len(values)) + 1))[: len(keys)]
        record = dict(zip(keys, padded, strict=False))
        formatter = CSVFormatter()
        result = formatter.flatten(record)
        assert result == record

    def test_nested_dict_flattened_with_dots(self) -> None:
        """Property: Nested dicts become dot-notation keys."""
        record = {"outer": {"inner": "value"}}
        formatter = CSVFormatter()
        result = formatter.flatten(record)
        assert result == {"outer.inner": "value"}

    def test_deeply_nested_dict_flattened(self) -> None:
        """Property: Multiple nesting levels produce chained dot keys."""
        record = {"a": {"b": {"c": "deep"}}}
        formatter = CSVFormatter()
        result = formatter.flatten(record)
        assert result == {"a.b.c": "deep"}

    def test_flatten_is_idempotent_on_flat(self) -> None:
        """Property: Flattening an already-flat dict returns same result."""
        record = {"a.b.c": "value", "x": "y"}
        formatter = CSVFormatter()
        result1 = formatter.flatten(record)
        result2 = formatter.flatten(result1)
        assert result1 == result2

    def test_lists_become_json_strings(self) -> None:
        """Property: List values in records are serialized as JSON strings."""
        record = {"tags": ["a", "b", "c"]}
        formatter = CSVFormatter()
        result = formatter.flatten(record)
        import json

        assert result["tags"] == json.dumps(["a", "b", "c"])

    def test_format_returns_flattened(self) -> None:
        """Property: format() delegates to flatten()."""
        record = {"outer": {"inner": "value"}}
        formatter = CSVFormatter()
        assert formatter.format(record) == formatter.flatten(record)


# =============================================================================
# JSONFormatter Properties
# =============================================================================


class TestJSONFormatterProperties:
    """JSONFormatter.format must produce valid JSON strings."""

    @given(
        keys=st.lists(
            st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            min_size=1,
            max_size=3,
            unique=True,
        ),
        values=st.lists(st.text(max_size=20), min_size=1, max_size=3),
    )
    @settings(max_examples=100)
    def test_format_produces_valid_json(self, keys: list[str], values: list[str]) -> None:
        """Property: format() always produces parseable JSON."""
        import json

        padded = (values * ((len(keys) // len(values)) + 1))[: len(keys)]
        record = dict(zip(keys, padded, strict=False))
        formatter = JSONFormatter()
        result = formatter.format(record)
        parsed = json.loads(result)
        assert parsed == record

    def test_format_handles_datetime_as_iso_string(self) -> None:
        """Property: Datetimes are normalized to ISO-8601 strings."""
        import json

        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        record = {"ts": dt}
        formatter = JSONFormatter()
        result = formatter.format(record)
        parsed = json.loads(result)
        assert parsed["ts"] == "2024-06-15T12:00:00+00:00"
