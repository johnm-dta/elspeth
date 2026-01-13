# tests/core/landscape/test_formatters.py
"""Tests for export formatters."""

import json
from datetime import UTC, datetime

from elspeth.core.landscape.formatters import CSVFormatter, JSONFormatter


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
