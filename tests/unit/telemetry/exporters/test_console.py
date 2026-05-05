# tests/telemetry/exporters/test_console.py
"""Tests for Console telemetry exporter.

Tests cover:
- Configuration validation (format and output enum values)
- Type validation for configuration options
- Error handling
- Export behavior (JSON/pretty formats, stream selection)
- Lifecycle operations (flush, close)
- Datetime and enum serialization
"""

import json
import sys
from datetime import UTC, datetime
from io import StringIO

import pytest

from elspeth.contracts.enums import NodeStateStatus, RoutingMode, TerminalOutcome, TerminalPath
from elspeth.contracts.events import (
    GateEvaluated,
    TokenCompleted,
    TransformCompleted,
)
from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.exporters.console import ConsoleExporter


class TestConsoleExporterConfiguration:
    """Tests for ConsoleExporter configuration."""

    def test_name_property(self) -> None:
        """Exporter name is 'console'."""
        exporter = ConsoleExporter()
        assert exporter.name == "console"

    def test_default_configuration(self) -> None:
        """Default configuration uses json format and stdout."""
        exporter = ConsoleExporter()
        exporter.configure({})
        assert exporter._format == "json"
        assert exporter._output == "stdout"

    def test_pretty_format(self) -> None:
        """Pretty format is valid."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "pretty"})
        assert exporter._format == "pretty"

    def test_stderr_output(self) -> None:
        """stderr output is valid."""
        exporter = ConsoleExporter()
        exporter.configure({"output": "stderr"})
        assert exporter._output == "stderr"

    def test_invalid_format_raises(self) -> None:
        """Invalid format value raises TelemetryExporterError."""
        exporter = ConsoleExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"format": "xml"})
        assert "Invalid format" in str(exc_info.value)

    def test_invalid_output_raises(self) -> None:
        """Invalid output value raises TelemetryExporterError."""
        exporter = ConsoleExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"output": "file"})
        assert "Invalid output" in str(exc_info.value)

    def test_format_wrong_type_raises(self) -> None:
        """Non-string format raises TelemetryExporterError with clear message."""
        exporter = ConsoleExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"format": 123})
        assert "'format' must be a string" in str(exc_info.value)
        assert "int" in str(exc_info.value)

    def test_output_wrong_type_raises(self) -> None:
        """Non-string output raises TelemetryExporterError with clear message."""
        exporter = ConsoleExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"output": ["stdout", "stderr"]})
        assert "'output' must be a string" in str(exc_info.value)
        assert "list" in str(exc_info.value)


class TestConsoleExporterExportBehavior:
    """Tests for export() method behavior with different formats and events."""

    def test_json_format_produces_valid_json(self) -> None:
        """JSON format exports valid JSON with event_type and run_id."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "json", "output": "stdout"})

        # Capture output
        captured = StringIO()
        exporter._stream = captured

        # Create test event
        event = TransformCompleted(
            timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-123",
            row_id="row-1",
            token_id="token-1",
            node_id="node-1",
            plugin_name="test_transform",
            status=NodeStateStatus.COMPLETED,
            duration_ms=42.5,
            input_hash="hash-in",
            output_hash="hash-out",
        )

        exporter.export(event)

        # Parse and validate JSON
        output = captured.getvalue().strip()
        parsed = json.loads(output)

        assert parsed["event_type"] == "TransformCompleted"
        assert parsed["run_id"] == "run-123"
        assert parsed["row_id"] == "row-1"
        assert parsed["token_id"] == "token-1"
        assert parsed["node_id"] == "node-1"
        assert parsed["plugin_name"] == "test_transform"
        assert parsed["status"] == "completed"  # Enum serialized to value
        assert parsed["duration_ms"] == 42.5
        assert parsed["input_hash"] == "hash-in"
        assert parsed["output_hash"] == "hash-out"
        assert parsed["timestamp"] == "2026-01-15T10:30:00+00:00"  # ISO 8601

    def test_json_format_with_gate_event(self) -> None:
        """JSON format handles GateEvaluated events with routing mode enum."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "json"})

        captured = StringIO()
        exporter._stream = captured

        event = GateEvaluated(
            timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-456",
            row_id="row-2",
            token_id="token-2",
            node_id="gate-1",
            plugin_name="routing_gate",
            routing_mode=RoutingMode.COPY,
            destinations=("sink-a", "sink-b"),
        )

        exporter.export(event)

        output = captured.getvalue().strip()
        parsed = json.loads(output)

        assert parsed["event_type"] == "GateEvaluated"
        assert parsed["routing_mode"] == "copy"  # Enum serialized
        assert parsed["destinations"] == ["sink-a", "sink-b"]  # Tuple serialized

    def test_json_format_with_token_completed_event(self) -> None:
        """JSON format handles TokenCompleted events with outcome enum."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "json"})

        captured = StringIO()
        exporter._stream = captured

        event = TokenCompleted(
            timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-789",
            row_id="row-3",
            token_id="token-3",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output_sink",
        )

        exporter.export(event)

        output = captured.getvalue().strip()
        parsed = json.loads(output)

        assert parsed["event_type"] == "TokenCompleted"
        assert parsed["outcome"] == "success"  # Enum serialized
        assert parsed["path"] == "default_flow"
        assert parsed["sink_name"] == "output_sink"

    def test_pretty_format_includes_timestamp_and_event_type(self) -> None:
        """Pretty format includes ISO timestamp and event type."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "pretty"})

        captured = StringIO()
        exporter._stream = captured

        event = TransformCompleted(
            timestamp=datetime(2026, 1, 15, 10, 30, 45, tzinfo=UTC),
            run_id="run-pretty",
            row_id="row-1",
            token_id="token-1",
            node_id="node-1",
            plugin_name="test_transform",
            status=NodeStateStatus.COMPLETED,
            duration_ms=100.0,
            input_hash="hash-in",
            output_hash="hash-out",
        )

        exporter.export(event)

        output = captured.getvalue().strip()

        # Format: [TIMESTAMP] EventType: run_id (details)
        assert output.startswith("[2026-01-15T10:30:45+00:00]")
        assert "TransformCompleted" in output
        assert "run-pretty" in output
        assert "duration_ms=100.0" in output
        assert "status=completed" in output  # Enum value used

    def test_pretty_format_with_minimal_event(self) -> None:
        """Pretty format handles events with only base fields."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "pretty"})

        captured = StringIO()
        exporter._stream = captured

        # TokenCompleted with None sink_name (minimal details)
        event = TokenCompleted(
            timestamp=datetime(2026, 1, 15, 11, 0, 0, tzinfo=UTC),
            run_id="run-minimal",
            row_id="row-x",
            token_id="token-x",
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.QUARANTINED_AT_SOURCE,
            sink_name=None,
        )

        exporter.export(event)

        output = captured.getvalue().strip()

        assert "[2026-01-15T11:00:00+00:00]" in output
        assert "TokenCompleted" in output
        assert "run-minimal" in output
        assert "outcome=failure" in output
        assert "path=quarantined_at_source" in output
        # sink_name=None should not appear in details
        assert "sink_name=None" not in output

    def test_stream_selection_stdout(self) -> None:
        """Output stream selection: stdout is used correctly."""
        exporter = ConsoleExporter()
        exporter.configure({"output": "stdout"})

        assert exporter._stream is sys.stdout

    def test_stream_selection_stderr(self) -> None:
        """Output stream selection: stderr is used correctly."""
        exporter = ConsoleExporter()
        exporter.configure({"output": "stderr"})

        assert exporter._stream is sys.stderr

    def test_export_does_not_raise_on_serialization_error(self) -> None:
        """Export must not raise exceptions - telemetry failures should not crash."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "json"})

        # Mock a broken stream that raises on write
        class BrokenStream:
            def write(self, *args: object, **kwargs: object) -> int:
                raise OSError("Simulated I/O error")

            def flush(self) -> None:
                pass

        exporter._stream = BrokenStream()  # type: ignore[assignment]  # Test intentionally uses mock

        event = TransformCompleted(
            timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-error",
            row_id="row-1",
            token_id="token-1",
            node_id="node-1",
            plugin_name="test",
            status=NodeStateStatus.COMPLETED,
            duration_ms=1.0,
            input_hash="h1",
            output_hash="h2",
        )

        # Should not raise
        exporter.export(event)


class TestConsoleExporterLifecycle:
    """Tests for flush() and close() lifecycle methods."""

    def test_flush_flushes_stream(self) -> None:
        """Flush calls flush() on the underlying stream."""
        exporter = ConsoleExporter()
        exporter.configure({})

        captured = StringIO()
        exporter._stream = captured

        # Write some data
        event = TransformCompleted(
            timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-flush",
            row_id="row-1",
            token_id="token-1",
            node_id="node-1",
            plugin_name="test",
            status=NodeStateStatus.COMPLETED,
            duration_ms=1.0,
            input_hash="h1",
            output_hash="h2",
        )
        exporter.export(event)

        # Flush should succeed
        exporter.flush()

        # Data should be available
        assert captured.getvalue()

    def test_flush_does_not_raise_on_error(self) -> None:
        """Flush must not raise exceptions."""

        class BrokenStream:
            def flush(self) -> None:
                raise OSError("Simulated flush error")

        exporter = ConsoleExporter()
        exporter.configure({})
        exporter._stream = BrokenStream()  # type: ignore[assignment]  # Test intentionally uses mock

        # Should not raise
        exporter.flush()

    def test_close_is_noop(self) -> None:
        """Close is a no-op for console exporter (does not own streams)."""
        exporter = ConsoleExporter()
        exporter.configure({})

        captured = StringIO()
        exporter._stream = captured

        # Close should not close the stream
        exporter.close()

        # Stream should still be usable
        assert not captured.closed


class TestConsoleExporterDatetimeAndEnumSerialization:
    """Tests for datetime and enum serialization in JSON format."""

    def test_datetime_serialization_to_iso8601(self) -> None:
        """Datetime fields are serialized to ISO 8601 format."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "json"})

        captured = StringIO()
        exporter._stream = captured

        event = TransformCompleted(
            timestamp=datetime(2026, 1, 15, 14, 30, 45, 123456, tzinfo=UTC),
            run_id="run-dt",
            row_id="row-1",
            token_id="token-1",
            node_id="node-1",
            plugin_name="test",
            status=NodeStateStatus.COMPLETED,
            duration_ms=1.0,
            input_hash="h1",
            output_hash="h2",
        )

        exporter.export(event)

        output = captured.getvalue().strip()
        parsed = json.loads(output)

        # ISO 8601 with microseconds and timezone
        assert parsed["timestamp"] == "2026-01-15T14:30:45.123456+00:00"

    def test_enum_serialization_to_value(self) -> None:
        """Enum fields are serialized to their string values."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "json"})

        captured = StringIO()
        exporter._stream = captured

        event = TransformCompleted(
            timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-enum",
            row_id="row-1",
            token_id="token-1",
            node_id="node-1",
            plugin_name="test",
            status=NodeStateStatus.FAILED,
            duration_ms=1.0,
            input_hash="h1",
            output_hash=None,
        )

        exporter.export(event)

        output = captured.getvalue().strip()
        parsed = json.loads(output)

        # NodeStateStatus.FAILED -> "failed" (enum value, not name)
        assert parsed["status"] == "failed"
        assert isinstance(parsed["status"], str)

    def test_multiple_enum_types_serialized_correctly(self) -> None:
        """Different enum types are all serialized to their values."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "json"})

        captured = StringIO()
        exporter._stream = captured

        event = GateEvaluated(
            timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-enums",
            row_id="row-1",
            token_id="token-1",
            node_id="gate-1",
            plugin_name="gate",
            routing_mode=RoutingMode.MOVE,
            destinations=("error_sink",),
        )

        exporter.export(event)

        output = captured.getvalue().strip()
        parsed = json.loads(output)

        # RoutingMode.MOVE -> "move"
        assert parsed["routing_mode"] == "move"
        assert isinstance(parsed["routing_mode"], str)


class TestConsoleExporterEdgeCases:
    """Tests for edge cases and additional export behaviors."""

    def test_json_format_with_none_values(self) -> None:
        """JSON format correctly serializes None values for optional fields."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "json"})

        captured = StringIO()
        exporter._stream = captured

        # TransformCompleted with None output_hash (failed transform case)
        event = TransformCompleted(
            timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-none",
            row_id="row-1",
            token_id="token-1",
            node_id="node-1",
            plugin_name="test_transform",
            status=NodeStateStatus.FAILED,
            duration_ms=10.0,
            input_hash="hash-in",
            output_hash=None,  # Failed transform has no output
        )

        exporter.export(event)

        output = captured.getvalue().strip()
        parsed = json.loads(output)

        assert parsed["output_hash"] is None
        assert parsed["input_hash"] == "hash-in"
        assert parsed["status"] == "failed"

    def test_multiple_events_are_separate_lines(self) -> None:
        """Multiple exports produce separate JSON lines (JSONL format)."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "json"})

        captured = StringIO()
        exporter._stream = captured

        event1 = TransformCompleted(
            timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-multi",
            row_id="row-1",
            token_id="token-1",
            node_id="node-1",
            plugin_name="transform-a",
            status=NodeStateStatus.COMPLETED,
            duration_ms=5.0,
            input_hash="h1",
            output_hash="h2",
        )

        event2 = TransformCompleted(
            timestamp=datetime(2026, 1, 15, 10, 30, 1, tzinfo=UTC),
            run_id="run-multi",
            row_id="row-2",
            token_id="token-2",
            node_id="node-1",
            plugin_name="transform-a",
            status=NodeStateStatus.COMPLETED,
            duration_ms=6.0,
            input_hash="h3",
            output_hash="h4",
        )

        exporter.export(event1)
        exporter.export(event2)

        lines = captured.getvalue().strip().split("\n")
        assert len(lines) == 2

        parsed1 = json.loads(lines[0])
        parsed2 = json.loads(lines[1])

        assert parsed1["row_id"] == "row-1"
        assert parsed2["row_id"] == "row-2"

    def test_pretty_format_with_tuple_destinations(self) -> None:
        """Pretty format renders tuple destinations as list."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "pretty"})

        captured = StringIO()
        exporter._stream = captured

        event = GateEvaluated(
            timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-tuple",
            row_id="row-1",
            token_id="token-1",
            node_id="gate-1",
            plugin_name="fork_gate",
            routing_mode=RoutingMode.COPY,
            destinations=("sink-a", "sink-b", "sink-c"),
        )

        exporter.export(event)

        output = captured.getvalue().strip()

        # Verify destinations tuple is rendered as list
        assert "['sink-a', 'sink-b', 'sink-c']" in output
        assert "routing_mode=copy" in output

    def test_export_works_with_default_configuration(self) -> None:
        """Export works without explicit configure() call (uses defaults)."""
        exporter = ConsoleExporter()
        # Deliberately NOT calling configure()

        captured = StringIO()
        exporter._stream = captured

        event = TokenCompleted(
            timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-default",
            row_id="row-1",
            token_id="token-1",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )

        exporter.export(event)

        # Should use default json format
        output = captured.getvalue().strip()
        parsed = json.loads(output)  # Valid JSON

        assert parsed["event_type"] == "TokenCompleted"
        assert parsed["run_id"] == "run-default"

    def test_json_format_with_empty_destinations(self) -> None:
        """JSON format handles empty tuple for destinations."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "json"})

        captured = StringIO()
        exporter._stream = captured

        # Edge case: gate with no destinations (unusual but valid)
        event = GateEvaluated(
            timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-empty-dest",
            row_id="row-1",
            token_id="token-1",
            node_id="gate-1",
            plugin_name="drop_gate",
            routing_mode=RoutingMode.MOVE,
            destinations=(),  # Empty tuple
        )

        exporter.export(event)

        output = captured.getvalue().strip()
        parsed = json.loads(output)

        assert parsed["destinations"] == []  # Empty tuple becomes empty list

    def test_pretty_format_with_all_none_optional_fields(self) -> None:
        """Pretty format omits all None optional fields."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "pretty"})

        captured = StringIO()
        exporter._stream = captured

        # TransformCompleted with both hashes as None
        event = TransformCompleted(
            timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-all-none",
            row_id="row-1",
            token_id="token-1",
            node_id="node-1",
            plugin_name="test",
            status=NodeStateStatus.FAILED,
            duration_ms=0.5,
            input_hash=None,
            output_hash=None,
        )

        exporter.export(event)

        output = captured.getvalue().strip()

        # None fields should not appear in pretty format
        assert "input_hash=None" not in output
        assert "output_hash=None" not in output
        # Non-None fields should appear
        assert "duration_ms=0.5" in output
        assert "status=failed" in output


class TestConsoleExporterRegistration:
    """Tests for plugin registration."""

    def test_exporter_in_builtin_plugin(self) -> None:
        """ConsoleExporter is registered in BuiltinExportersPlugin."""
        from elspeth.telemetry.exporters import BuiltinExportersPlugin, ConsoleExporter

        plugin = BuiltinExportersPlugin()
        exporters = plugin.elspeth_get_exporters()
        assert ConsoleExporter in exporters

    def test_exporter_in_package_all(self) -> None:
        """ConsoleExporter is exported from package __all__."""
        from elspeth.telemetry import exporters

        assert "ConsoleExporter" in exporters.__all__
