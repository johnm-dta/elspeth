# tests/unit/telemetry/test_console_exporter.py
"""Unit tests for ConsoleExporter.

Tests cover:
- Configuration validation (valid/invalid format and output values)
- JSON output format with proper type serialization
- Pretty output format with human-readable strings
- Error handling (export must not raise)
- Protocol compliance (name property, flush, close)
"""

import json
import sys
from datetime import UTC, datetime
from io import StringIO
from unittest.mock import patch

import pytest

from elspeth.contracts.enums import NodeStateStatus, RowOutcome, RunStatus
from elspeth.contracts.events import (
    PhaseAction,
    PipelinePhase,
    TelemetryEvent,
    TokenCompleted,
    TransformCompleted,
)
from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.events import (
    PhaseChanged,
    RunFinished,
    RunStarted,
)
from elspeth.telemetry.exporters.console import ConsoleExporter
from elspeth.telemetry.protocols import ExporterProtocol

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_timestamp() -> datetime:
    """Fixed timestamp for deterministic tests."""
    return datetime(2026, 1, 30, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def base_run_id() -> str:
    """Fixed run ID for tests."""
    return "run-console-test"


@pytest.fixture
def exporter() -> ConsoleExporter:
    """Create an unconfigured ConsoleExporter."""
    return ConsoleExporter()


@pytest.fixture
def json_exporter() -> ConsoleExporter:
    """Create a ConsoleExporter configured for JSON output to stdout."""
    exp = ConsoleExporter()
    exp.configure({"format": "json", "output": "stdout"})
    return exp


@pytest.fixture
def pretty_exporter() -> ConsoleExporter:
    """Create a ConsoleExporter configured for pretty output to stdout."""
    exp = ConsoleExporter()
    exp.configure({"format": "pretty", "output": "stdout"})
    return exp


def make_base_event(run_id: str, timestamp: datetime) -> TelemetryEvent:
    """Create a basic TelemetryEvent."""
    return TelemetryEvent(timestamp=timestamp, run_id=run_id)


def make_run_started(run_id: str, timestamp: datetime) -> RunStarted:
    """Create a RunStarted event."""
    return RunStarted(
        timestamp=timestamp,
        run_id=run_id,
        config_hash="abc123",
        source_plugin="csv",
    )


def make_run_finished(run_id: str, timestamp: datetime) -> RunFinished:
    """Create a RunFinished event."""
    return RunFinished(
        timestamp=timestamp,
        run_id=run_id,
        status=RunStatus.COMPLETED,
        row_count=100,
        duration_ms=5000.0,
    )


def make_transform_completed(run_id: str, timestamp: datetime) -> TransformCompleted:
    """Create a TransformCompleted event."""
    return TransformCompleted(
        timestamp=timestamp,
        run_id=run_id,
        row_id="row-1",
        token_id="token-1",
        node_id="node-1",
        plugin_name="field_mapper",
        status=NodeStateStatus.COMPLETED,
        duration_ms=50.0,
        input_hash="input-hash",
        output_hash="output-hash",
    )


def make_token_completed(run_id: str, timestamp: datetime) -> TokenCompleted:
    """Create a TokenCompleted event."""
    return TokenCompleted(
        timestamp=timestamp,
        run_id=run_id,
        row_id="row-1",
        token_id="token-1",
        outcome=RowOutcome.COMPLETED,
        sink_name="output",
    )


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Tests for ExporterProtocol compliance."""

    def test_implements_exporter_protocol(self, exporter: ConsoleExporter) -> None:
        """ConsoleExporter implements ExporterProtocol."""
        assert isinstance(exporter, ExporterProtocol)

    def test_name_property_returns_console(self, exporter: ConsoleExporter) -> None:
        """name property returns 'console'."""
        assert exporter.name == "console"

    def test_name_property_matches_class_attribute(self, exporter: ConsoleExporter) -> None:
        """name property matches _name class attribute."""
        assert exporter.name == ConsoleExporter._name

    def test_flush_is_callable(self, json_exporter: ConsoleExporter) -> None:
        """flush() method exists and is callable."""
        json_exporter.flush()  # Should not raise

    def test_close_is_callable(self, json_exporter: ConsoleExporter) -> None:
        """close() method exists and is callable."""
        json_exporter.close()  # Should not raise

    def test_close_is_idempotent(self, json_exporter: ConsoleExporter) -> None:
        """close() can be called multiple times safely."""
        json_exporter.close()
        json_exporter.close()
        json_exporter.close()  # Should not raise


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for configure() method."""

    def test_default_configuration(self, exporter: ConsoleExporter) -> None:
        """Default configuration uses json format and stdout."""
        exporter.configure({})
        assert exporter._format == "json"
        assert exporter._output == "stdout"
        assert exporter._stream is sys.stdout

    def test_configure_json_format(self, exporter: ConsoleExporter) -> None:
        """format='json' is accepted."""
        exporter.configure({"format": "json"})
        assert exporter._format == "json"

    def test_configure_pretty_format(self, exporter: ConsoleExporter) -> None:
        """format='pretty' is accepted."""
        exporter.configure({"format": "pretty"})
        assert exporter._format == "pretty"

    def test_configure_stdout_output(self, exporter: ConsoleExporter) -> None:
        """output='stdout' sets stream to sys.stdout."""
        exporter.configure({"output": "stdout"})
        assert exporter._output == "stdout"
        assert exporter._stream is sys.stdout

    def test_configure_stderr_output(self, exporter: ConsoleExporter) -> None:
        """output='stderr' sets stream to sys.stderr."""
        exporter.configure({"output": "stderr"})
        assert exporter._output == "stderr"
        assert exporter._stream is sys.stderr

    def test_configure_combined_options(self, exporter: ConsoleExporter) -> None:
        """Multiple options can be configured together."""
        exporter.configure({"format": "pretty", "output": "stderr"})
        assert exporter._format == "pretty"
        assert exporter._output == "stderr"
        assert exporter._stream is sys.stderr

    def test_invalid_format_raises_error(self, exporter: ConsoleExporter) -> None:
        """Invalid format value raises TelemetryExporterError."""
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"format": "xml"})

        assert exc_info.value.exporter_name == "console"
        assert "Invalid format 'xml'" in exc_info.value.message
        assert "json" in exc_info.value.message
        assert "pretty" in exc_info.value.message

    def test_invalid_output_raises_error(self, exporter: ConsoleExporter) -> None:
        """Invalid output value raises TelemetryExporterError."""
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"output": "file"})

        assert exc_info.value.exporter_name == "console"
        assert "Invalid output 'file'" in exc_info.value.message
        assert "stdout" in exc_info.value.message
        assert "stderr" in exc_info.value.message

    def test_unknown_config_keys_ignored(self, exporter: ConsoleExporter) -> None:
        """Unknown configuration keys are silently ignored."""
        exporter.configure({"format": "json", "output": "stdout", "unknown_key": "some_value"})
        assert exporter._format == "json"
        assert exporter._output == "stdout"


# =============================================================================
# JSON Output Tests
# =============================================================================


class TestJsonOutput:
    """Tests for JSON output format."""

    def test_json_output_is_valid_json(
        self,
        json_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """JSON output can be parsed as valid JSON."""
        event = make_run_started(base_run_id, base_timestamp)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            json_exporter._stream = mock_stdout
            json_exporter.export(event)
            output = mock_stdout.getvalue().strip()

        # Should not raise
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_json_includes_event_type(
        self,
        json_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """JSON output includes event_type field."""
        event = make_run_started(base_run_id, base_timestamp)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            json_exporter._stream = mock_stdout
            json_exporter.export(event)
            output = mock_stdout.getvalue().strip()

        parsed = json.loads(output)
        assert parsed["event_type"] == "RunStarted"

    def test_json_serializes_datetime_to_iso(
        self,
        json_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """JSON output serializes datetime to ISO 8601 string."""
        event = make_run_started(base_run_id, base_timestamp)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            json_exporter._stream = mock_stdout
            json_exporter.export(event)
            output = mock_stdout.getvalue().strip()

        parsed = json.loads(output)
        assert parsed["timestamp"] == base_timestamp.isoformat()

    def test_json_serializes_enum_to_value(
        self,
        json_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """JSON output serializes Enum to its value."""
        event = make_run_finished(base_run_id, base_timestamp)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            json_exporter._stream = mock_stdout
            json_exporter.export(event)
            output = mock_stdout.getvalue().strip()

        parsed = json.loads(output)
        assert parsed["status"] == RunStatus.COMPLETED.value

    def test_json_includes_all_event_fields(
        self,
        json_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """JSON output includes all event-specific fields."""
        event = make_transform_completed(base_run_id, base_timestamp)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            json_exporter._stream = mock_stdout
            json_exporter.export(event)
            output = mock_stdout.getvalue().strip()

        parsed = json.loads(output)
        assert parsed["run_id"] == base_run_id
        assert parsed["row_id"] == "row-1"
        assert parsed["token_id"] == "token-1"
        assert parsed["node_id"] == "node-1"
        assert parsed["plugin_name"] == "field_mapper"
        assert parsed["status"] == NodeStateStatus.COMPLETED.value
        assert parsed["duration_ms"] == 50.0
        assert parsed["input_hash"] == "input-hash"
        assert parsed["output_hash"] == "output-hash"

    def test_json_handles_none_values(
        self,
        json_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """JSON output handles None values correctly."""
        event = TokenCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            outcome=RowOutcome.FAILED,
            sink_name=None,
        )

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            json_exporter._stream = mock_stdout
            json_exporter.export(event)
            output = mock_stdout.getvalue().strip()

        parsed = json.loads(output)
        assert parsed["sink_name"] is None

    def test_json_one_line_per_event(
        self,
        json_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """Each JSON event is on a single line."""
        event = make_run_started(base_run_id, base_timestamp)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            json_exporter._stream = mock_stdout
            json_exporter.export(event)
            json_exporter.export(event)
            output = mock_stdout.getvalue()

        lines = output.strip().split("\n")
        assert len(lines) == 2
        # Each line should be valid JSON
        for line in lines:
            json.loads(line)


# =============================================================================
# Pretty Output Tests
# =============================================================================


class TestPrettyOutput:
    """Tests for pretty (human-readable) output format."""

    def test_pretty_includes_timestamp(
        self,
        pretty_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """Pretty output includes ISO timestamp in brackets."""
        event = make_run_started(base_run_id, base_timestamp)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            pretty_exporter._stream = mock_stdout
            pretty_exporter.export(event)
            output = mock_stdout.getvalue().strip()

        assert f"[{base_timestamp.isoformat()}]" in output

    def test_pretty_includes_event_type(
        self,
        pretty_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """Pretty output includes event type name."""
        event = make_run_started(base_run_id, base_timestamp)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            pretty_exporter._stream = mock_stdout
            pretty_exporter.export(event)
            output = mock_stdout.getvalue().strip()

        assert "RunStarted:" in output

    def test_pretty_includes_run_id(
        self,
        pretty_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """Pretty output includes run_id."""
        event = make_run_started(base_run_id, base_timestamp)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            pretty_exporter._stream = mock_stdout
            pretty_exporter.export(event)
            output = mock_stdout.getvalue().strip()

        assert base_run_id in output

    def test_pretty_includes_event_details(
        self,
        pretty_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """Pretty output includes key event details in parentheses."""
        event = make_run_started(base_run_id, base_timestamp)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            pretty_exporter._stream = mock_stdout
            pretty_exporter.export(event)
            output = mock_stdout.getvalue().strip()

        # Should include config_hash and source_plugin in parentheses
        assert "config_hash=abc123" in output
        assert "source_plugin=csv" in output

    def test_pretty_serializes_enum_to_value(
        self,
        pretty_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """Pretty output shows enum values, not enum names."""
        event = make_run_finished(base_run_id, base_timestamp)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            pretty_exporter._stream = mock_stdout
            pretty_exporter.export(event)
            output = mock_stdout.getvalue().strip()

        assert f"status={RunStatus.COMPLETED.value}" in output

    def test_pretty_base_event_no_extra_details(
        self,
        pretty_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """Base TelemetryEvent has no extra details (no parentheses)."""
        event = make_base_event(base_run_id, base_timestamp)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            pretty_exporter._stream = mock_stdout
            pretty_exporter.export(event)
            output = mock_stdout.getvalue().strip()

        # Should end with run_id, no parentheses
        assert output.endswith(base_run_id)


# =============================================================================
# Output Stream Tests
# =============================================================================


class TestOutputStream:
    """Tests for output stream selection."""

    def test_stdout_output(
        self,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """Events are written to stdout when configured."""
        exporter = ConsoleExporter()
        exporter.configure({"output": "stdout"})
        event = make_run_started(base_run_id, base_timestamp)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            exporter._stream = mock_stdout
            exporter.export(event)
            assert mock_stdout.getvalue() != ""

    def test_stderr_output(
        self,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """Events are written to stderr when configured."""
        exporter = ConsoleExporter()
        exporter.configure({"output": "stderr"})
        event = make_run_started(base_run_id, base_timestamp)

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            exporter._stream = mock_stderr
            exporter.export(event)
            assert mock_stderr.getvalue() != ""

    def test_flush_flushes_stream(
        self,
        json_exporter: ConsoleExporter,
    ) -> None:
        """flush() calls flush on the underlying stream."""
        mock_stream = StringIO()
        json_exporter._stream = mock_stream

        # StringIO.flush() is a no-op, but we can verify it's called
        with patch.object(mock_stream, "flush") as mock_flush:
            json_exporter.flush()
            mock_flush.assert_called_once()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling (export must not raise)."""

    def test_export_does_not_raise_on_stream_error(
        self,
        json_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """export() does not raise when stream write fails."""
        event = make_run_started(base_run_id, base_timestamp)

        # Create a mock stream that raises on write
        class BrokenStream:
            def write(self, s: str) -> int:
                raise OSError("Stream is broken")

        json_exporter._stream = BrokenStream()  # type: ignore[assignment]

        # Should not raise - logs warning instead
        with patch("elspeth.telemetry.exporters.console.logger") as mock_logger:
            json_exporter.export(event)
            mock_logger.warning.assert_called_once()

    def test_export_logs_warning_on_error(
        self,
        json_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """export() logs warning with details when an error occurs."""
        event = make_run_started(base_run_id, base_timestamp)

        class BrokenStream:
            def write(self, s: str) -> int:
                raise OSError("Stream is broken")

        json_exporter._stream = BrokenStream()  # type: ignore[assignment]

        with patch("elspeth.telemetry.exporters.console.logger") as mock_logger:
            json_exporter.export(event)

            call_args = mock_logger.warning.call_args
            assert call_args[0][0] == "Failed to export telemetry event"
            assert call_args[1]["exporter"] == "console"
            assert call_args[1]["event_type"] == "RunStarted"
            assert "Stream is broken" in call_args[1]["error"]

    def test_flush_does_not_raise_on_error(
        self,
        json_exporter: ConsoleExporter,
    ) -> None:
        """flush() does not raise when stream flush fails."""

        class BrokenStream:
            def flush(self) -> None:
                raise OSError("Flush failed")

        json_exporter._stream = BrokenStream()  # type: ignore[assignment]

        # Should not raise - logs warning instead
        with patch("elspeth.telemetry.exporters.console.logger") as mock_logger:
            json_exporter.flush()
            mock_logger.warning.assert_called_once()


# =============================================================================
# Serialization Edge Cases
# =============================================================================


class TestSerializationEdgeCases:
    """Tests for edge cases in event serialization."""

    def test_serialize_phase_changed_event(
        self,
        json_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """PhaseChanged event with enums serializes correctly."""
        event = PhaseChanged(
            timestamp=base_timestamp,
            run_id=base_run_id,
            phase=PipelinePhase.PROCESS,
            action=PhaseAction.PROCESSING,
        )

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            json_exporter._stream = mock_stdout
            json_exporter.export(event)
            output = mock_stdout.getvalue().strip()

        parsed = json.loads(output)
        assert parsed["phase"] == PipelinePhase.PROCESS.value
        assert parsed["action"] == PhaseAction.PROCESSING.value

    def test_serialize_tuple_to_list_in_pretty_format(
        self,
        pretty_exporter: ConsoleExporter,
        base_run_id: str,
        base_timestamp: datetime,
    ) -> None:
        """Tuples are converted to lists in pretty format details."""
        # GateEvaluated has a tuple field (destinations)
        from elspeth.contracts import GateEvaluated
        from elspeth.contracts.enums import RoutingMode

        event = GateEvaluated(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            node_id="gate-1",
            plugin_name="threshold_gate",
            routing_mode=RoutingMode.MOVE,
            destinations=("sink-a", "sink-b"),
        )

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            pretty_exporter._stream = mock_stdout
            pretty_exporter.export(event)
            output = mock_stdout.getvalue().strip()

        # Tuple should be converted to list representation
        assert "destinations=['sink-a', 'sink-b']" in output


# =============================================================================
# Plugin Registration Tests
# =============================================================================


class TestPluginRegistration:
    """Tests for pluggy plugin registration."""

    def test_builtin_exporters_plugin_returns_console_exporter(self) -> None:
        """BuiltinExportersPlugin returns ConsoleExporter in hook."""
        from elspeth.telemetry.exporters import BuiltinExportersPlugin

        plugin = BuiltinExportersPlugin()
        exporters = plugin.elspeth_get_exporters()

        assert ConsoleExporter in exporters

    def test_console_exporter_in_telemetry_package_exports(self) -> None:
        """ConsoleExporter is exported from elspeth.telemetry package."""
        from elspeth.telemetry import ConsoleExporter as ExportedConsoleExporter

        assert ExportedConsoleExporter is ConsoleExporter
