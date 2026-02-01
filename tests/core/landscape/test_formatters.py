# tests/core/landscape/test_formatters.py
"""Tests for export formatters."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

import pytest

from elspeth.core.landscape.formatters import (
    CSVFormatter,
    JSONFormatter,
    dataclass_to_dict,
    serialize_datetime,
)


class TestSerializeDatetime:
    """Tests for serialize_datetime utility."""

    def test_converts_datetime_to_iso(self) -> None:
        """datetime objects become ISO strings."""
        dt = datetime(2026, 1, 29, 12, 30, 45, tzinfo=UTC)
        result = serialize_datetime(dt)
        assert result == "2026-01-29T12:30:45+00:00"

    def test_preserves_non_datetime(self) -> None:
        """Non-datetime values pass through unchanged."""
        assert serialize_datetime("hello") == "hello"
        assert serialize_datetime(42) == 42
        assert serialize_datetime(None) is None

    def test_recursively_handles_dict(self) -> None:
        """Dicts have datetime values converted recursively."""
        dt = datetime(2026, 1, 29, 12, 0, 0, tzinfo=UTC)
        data = {"created_at": dt, "name": "test", "nested": {"time": dt}}
        result = serialize_datetime(data)

        assert result["created_at"] == "2026-01-29T12:00:00+00:00"
        assert result["name"] == "test"
        assert result["nested"]["time"] == "2026-01-29T12:00:00+00:00"

    def test_recursively_handles_list(self) -> None:
        """Lists have datetime values converted recursively."""
        dt = datetime(2026, 1, 29, 12, 0, 0, tzinfo=UTC)
        data = [dt, "string", {"time": dt}]
        result = serialize_datetime(data)

        assert result[0] == "2026-01-29T12:00:00+00:00"
        assert result[1] == "string"
        assert result[2]["time"] == "2026-01-29T12:00:00+00:00"

    def test_rejects_nan(self) -> None:
        """NaN values are rejected per CLAUDE.md audit integrity requirements."""
        with pytest.raises(ValueError, match="NaN"):
            serialize_datetime(float("nan"))

    def test_rejects_infinity(self) -> None:
        """Infinity values are rejected per CLAUDE.md audit integrity requirements."""
        with pytest.raises(ValueError, match="Infinity"):
            serialize_datetime(float("inf"))

        with pytest.raises(ValueError, match="Infinity"):
            serialize_datetime(float("-inf"))

    def test_rejects_nan_in_nested_structure(self) -> None:
        """NaN in nested structures is also rejected."""
        with pytest.raises(ValueError, match="NaN"):
            serialize_datetime({"nested": {"value": float("nan")}})

        with pytest.raises(ValueError, match="NaN"):
            serialize_datetime([1, 2, float("nan")])


class TestDataclassToDict:
    """Tests for dataclass_to_dict utility."""

    def test_converts_simple_dataclass(self) -> None:
        """Simple dataclass becomes dict."""

        @dataclass
        class Simple:
            name: str
            value: int

        obj = Simple(name="test", value=42)
        result = dataclass_to_dict(obj)

        assert result == {"name": "test", "value": 42}

    def test_converts_nested_dataclass(self) -> None:
        """Nested dataclasses are recursively converted."""

        @dataclass
        class Inner:
            x: int

        @dataclass
        class Outer:
            inner: Inner
            y: str

        obj = Outer(inner=Inner(x=1), y="hello")
        result = dataclass_to_dict(obj)

        assert result == {"inner": {"x": 1}, "y": "hello"}

    def test_handles_enum_values(self) -> None:
        """Enum values are converted to their string value."""

        class Status(Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        @dataclass
        class WithEnum:
            status: Status

        obj = WithEnum(status=Status.ACTIVE)
        result = dataclass_to_dict(obj)

        assert result == {"status": "active"}

    def test_handles_datetime_in_dataclass(self) -> None:
        """Datetime fields are converted to ISO strings."""

        @dataclass
        class WithTime:
            created_at: datetime

        dt = datetime(2026, 1, 29, 12, 0, 0, tzinfo=UTC)
        obj = WithTime(created_at=dt)
        result = dataclass_to_dict(obj)

        assert result == {"created_at": "2026-01-29T12:00:00+00:00"}

    def test_handles_list_of_dataclasses(self) -> None:
        """Lists of dataclasses are recursively converted."""

        @dataclass
        class Item:
            id: int

        @dataclass
        class Container:
            items: list[Item]

        obj = Container(items=[Item(id=1), Item(id=2)])
        result = dataclass_to_dict(obj)

        assert result == {"items": [{"id": 1}, {"id": 2}]}

    def test_handles_none(self) -> None:
        """None returns empty dict."""
        result = dataclass_to_dict(None)
        assert result == {}

    def test_handles_plain_dict(self) -> None:
        """Plain dict passes through (not a dataclass)."""
        result = dataclass_to_dict({"a": 1})
        assert result == {"a": 1}


class TestCSVFormatter:
    """CSVFormatter flattens nested structures for CSV output."""

    def test_csv_formatter_flattens_nested_fields(self) -> None:
        """CSV formatter should flatten nested dicts to dot notation."""
        formatter = CSVFormatter()

        record = {
            "record_type": "node_state",
            "metadata": {"attempt": 1, "reason": "retry"},
        }

        flat = formatter.flatten(record)

        assert flat["metadata.attempt"] == 1
        assert flat["metadata.reason"] == "retry"

    def test_csv_formatter_preserves_flat_fields(self) -> None:
        """CSV formatter should preserve already-flat fields unchanged."""
        formatter = CSVFormatter()

        record = {
            "record_type": "row",
            "row_id": "abc123",
            "row_index": 42,
        }

        flat = formatter.flatten(record)

        assert flat["record_type"] == "row"
        assert flat["row_id"] == "abc123"
        assert flat["row_index"] == 42

    def test_csv_formatter_handles_deeply_nested(self) -> None:
        """CSV formatter should flatten deeply nested dicts."""
        formatter = CSVFormatter()

        record = {
            "outer": {
                "middle": {
                    "inner": "value",
                }
            }
        }

        flat = formatter.flatten(record)

        assert flat["outer.middle.inner"] == "value"

    def test_csv_formatter_converts_lists_to_json(self) -> None:
        """CSV formatter should convert lists to JSON strings."""
        formatter = CSVFormatter()

        record = {
            "tags": ["a", "b", "c"],
        }

        flat = formatter.flatten(record)

        assert flat["tags"] == '["a", "b", "c"]'

    def test_csv_formatter_format_returns_flat_dict(self) -> None:
        """CSVFormatter.format() should return flattened dict."""
        formatter = CSVFormatter()

        record = {
            "record_type": "node_state",
            "metadata": {"attempt": 1},
        }

        result = formatter.format(record)

        assert isinstance(result, dict)
        assert result["metadata.attempt"] == 1

    def test_csv_formatter_handles_none_values(self) -> None:
        """CSV formatter should preserve None values."""
        formatter = CSVFormatter()

        record = {
            "field": None,
            "nested": {"also_none": None},
        }

        flat = formatter.flatten(record)

        assert flat["field"] is None
        assert flat["nested.also_none"] is None

    def test_csv_formatter_handles_empty_dict(self) -> None:
        """CSV formatter should handle empty nested dicts."""
        formatter = CSVFormatter()

        record = {
            "record_type": "test",
            "empty": {},
        }

        flat = formatter.flatten(record)

        assert flat["record_type"] == "test"
        # Empty dict produces no keys for that prefix
        assert "empty" not in flat

    def test_csv_formatter_rejects_nan_in_list(self) -> None:
        """CSV formatter must reject NaN in lists per CLAUDE.md audit integrity.

        Bug: P2-2026-01-31-csv-formatter-nan-in-lists

        Lists are serialized to JSON strings for CSV. If NaN slips through,
        the resulting JSON is non-standard and may fail in downstream systems.
        """
        import math

        formatter = CSVFormatter()

        record = {
            "scores": [0.9, 0.8, math.nan],
        }

        with pytest.raises(ValueError, match="NaN"):
            formatter.flatten(record)

    def test_csv_formatter_rejects_infinity_in_list(self) -> None:
        """CSV formatter must reject Infinity in lists."""
        import math

        formatter = CSVFormatter()

        record = {
            "values": [1.0, 2.0, math.inf],
        }

        with pytest.raises(ValueError, match="Infinity"):
            formatter.flatten(record)

    def test_csv_formatter_rejects_nested_nan_in_list(self) -> None:
        """CSV formatter must reject NaN in nested list structures."""
        import math

        formatter = CSVFormatter()

        record = {
            "events": [{"score": math.nan}],
        }

        with pytest.raises(ValueError, match="NaN"):
            formatter.flatten(record)

    def test_csv_formatter_handles_valid_list_with_floats(self) -> None:
        """CSV formatter should accept lists with valid float values."""
        formatter = CSVFormatter()

        record = {
            "scores": [0.1, 0.5, 0.9],
        }

        flat = formatter.flatten(record)

        assert flat["scores"] == "[0.1, 0.5, 0.9]"


class TestJSONFormatter:
    """JSONFormatter preserves nested structure for JSON output."""

    def test_json_formatter_preserves_structure(self) -> None:
        """JSON formatter should preserve nested structure."""
        formatter = JSONFormatter()

        record = {
            "record_type": "node_state",
            "metadata": {"attempt": 1},
        }

        output = formatter.format(record)

        parsed = json.loads(output)
        assert parsed["metadata"]["attempt"] == 1

    def test_json_formatter_outputs_string(self) -> None:
        """JSON formatter should output a string."""
        formatter = JSONFormatter()

        record = {"key": "value"}

        output = formatter.format(record)

        assert isinstance(output, str)

    def test_json_formatter_handles_datetime_via_default(self) -> None:
        """JSON formatter should handle datetime via default=str."""
        formatter = JSONFormatter()

        record = {
            "timestamp": datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        }

        output = formatter.format(record)

        parsed = json.loads(output)
        assert "2024-01-15" in parsed["timestamp"]

    def test_json_formatter_handles_lists(self) -> None:
        """JSON formatter should preserve lists."""
        formatter = JSONFormatter()

        record = {
            "items": [1, 2, 3],
        }

        output = formatter.format(record)

        parsed = json.loads(output)
        assert parsed["items"] == [1, 2, 3]

    def test_json_formatter_handles_nested_lists_of_dicts(self) -> None:
        """JSON formatter should handle complex nested structures."""
        formatter = JSONFormatter()

        record = {
            "events": [
                {"type": "click", "target": "button"},
                {"type": "scroll", "position": 100},
            ]
        }

        output = formatter.format(record)

        parsed = json.loads(output)
        assert len(parsed["events"]) == 2
        assert parsed["events"][0]["type"] == "click"


class TestLineageTextFormatter:
    """Tests for LineageTextFormatter."""

    def test_formats_basic_lineage(self) -> None:
        """Formats LineageResult as human-readable text."""
        from elspeth.contracts import RowLineage, Token
        from elspeth.core.landscape.formatters import LineageTextFormatter
        from elspeth.core.landscape.lineage import LineageResult

        now = datetime(2026, 1, 29, 12, 0, 0, tzinfo=UTC)
        result = LineageResult(
            token=Token(token_id="tok-123", row_id="row-456", created_at=now),
            source_row=RowLineage(
                row_id="row-456",
                run_id="run-789",
                source_node_id="src-node",
                row_index=0,
                source_data_hash="abc123",
                created_at=now,
                source_data={"id": 1, "name": "test"},
                payload_available=True,
            ),
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
        )

        formatter = LineageTextFormatter()
        text = formatter.format(result)

        assert "Token: tok-123" in text
        assert "Row: row-456" in text
        assert "Source Data Hash: abc123" in text

    def test_formats_with_outcome(self) -> None:
        """Includes outcome when present."""
        from elspeth.contracts import RowLineage, RowOutcome, Token, TokenOutcome
        from elspeth.core.landscape.formatters import LineageTextFormatter
        from elspeth.core.landscape.lineage import LineageResult

        now = datetime(2026, 1, 29, 12, 0, 0, tzinfo=UTC)
        result = LineageResult(
            token=Token(token_id="tok-123", row_id="row-456", created_at=now),
            source_row=RowLineage(
                row_id="row-456",
                run_id="run-789",
                source_node_id="src-node",
                row_index=0,
                source_data_hash="abc123",
                created_at=now,
                source_data={"id": 1},
                payload_available=True,
            ),
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
            outcome=TokenOutcome(
                outcome_id="out-1",
                token_id="tok-123",
                run_id="run-789",
                outcome=RowOutcome.COMPLETED,
                sink_name="output",
                is_terminal=True,
                recorded_at=now,
            ),
        )

        formatter = LineageTextFormatter()
        text = formatter.format(result)

        assert "Outcome: COMPLETED" in text
        assert "Sink: output" in text

    def test_formats_missing_latency_as_na(self) -> None:
        """Missing call latency should render as N/A."""
        from elspeth.contracts import Call, CallStatus, CallType, RowLineage, Token
        from elspeth.core.landscape.formatters import LineageTextFormatter
        from elspeth.core.landscape.lineage import LineageResult

        now = datetime(2026, 1, 29, 12, 0, 0, tzinfo=UTC)
        result = LineageResult(
            token=Token(token_id="tok-123", row_id="row-456", created_at=now),
            source_row=RowLineage(
                row_id="row-456",
                run_id="run-789",
                source_node_id="src-node",
                row_index=0,
                source_data_hash="abc123",
                created_at=now,
                source_data={"id": 1},
                payload_available=True,
            ),
            node_states=[],
            routing_events=[],
            calls=[
                Call(
                    call_id="call-1",
                    call_index=0,
                    call_type=CallType.HTTP,
                    status=CallStatus.SUCCESS,
                    request_hash="req-hash",
                    created_at=now,
                    latency_ms=None,
                )
            ],
            parent_tokens=[],
        )

        formatter = LineageTextFormatter()
        text = formatter.format(result)

        assert "http: success (N/A)" in text

    def test_formats_none_gracefully(self) -> None:
        """Returns message for None result."""
        from elspeth.core.landscape.formatters import LineageTextFormatter

        formatter = LineageTextFormatter()
        text = formatter.format(None)

        assert "not found" in text.lower() or "no lineage" in text.lower()
