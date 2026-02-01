# tests/telemetry/exporters/test_otlp.py
"""Tests for OTLP telemetry exporter.

Tests cover:
- Configuration validation (endpoint required, batch_size > 0)
- Event buffering and batch export
- Event-to-span conversion (trace_id, span_id derivation)
- Attribute serialization (datetime, enum, dict, tuple handling)
- Flush and close lifecycle
- Error handling (export failures don't crash pipeline)
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts import TokenCompleted
from elspeth.contracts.enums import RowOutcome, RunStatus
from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.events import (
    RunFinished,
    RunStarted,
)
from elspeth.telemetry.exporters.otlp import (
    OTLPExporter,
    _derive_span_id,
    _derive_trace_id,
)

# Path to patch OTLPSpanExporter - must be at the source location
OTLP_EXPORTER_PATCH = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"


class TestTraceIdDerivation:
    """Tests for trace_id derivation from run_id."""

    def test_same_run_id_produces_same_trace_id(self) -> None:
        """Consistent run_id produces consistent trace_id."""
        run_id = "run-12345"
        trace_id_1 = _derive_trace_id(run_id)
        trace_id_2 = _derive_trace_id(run_id)
        assert trace_id_1 == trace_id_2

    def test_different_run_ids_produce_different_trace_ids(self) -> None:
        """Different run_ids produce different trace_ids."""
        trace_id_1 = _derive_trace_id("run-12345")
        trace_id_2 = _derive_trace_id("run-67890")
        assert trace_id_1 != trace_id_2

    def test_trace_id_is_128_bit_integer(self) -> None:
        """Trace ID is a positive 128-bit integer."""
        trace_id = _derive_trace_id("run-12345")
        assert isinstance(trace_id, int)
        assert trace_id > 0
        assert trace_id < 2**128


class TestSpanIdDerivation:
    """Tests for span_id derivation from events."""

    def test_same_event_produces_same_span_id(self) -> None:
        """Identical events produce identical span_ids."""
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        event1 = RunStarted(
            timestamp=ts,
            run_id="run-123",
            config_hash="abc123",
            source_plugin="csv",
        )
        event2 = RunStarted(
            timestamp=ts,
            run_id="run-123",
            config_hash="abc123",
            source_plugin="csv",
        )
        span_id_1 = _derive_span_id(event1)
        span_id_2 = _derive_span_id(event2)
        assert span_id_1 == span_id_2

    def test_different_timestamps_produce_different_span_ids(self) -> None:
        """Events with different timestamps have different span_ids."""
        event1 = RunStarted(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-123",
            config_hash="abc123",
            source_plugin="csv",
        )
        event2 = RunStarted(
            timestamp=datetime(2024, 1, 15, 10, 30, 1, tzinfo=UTC),
            run_id="run-123",
            config_hash="abc123",
            source_plugin="csv",
        )
        span_id_1 = _derive_span_id(event1)
        span_id_2 = _derive_span_id(event2)
        assert span_id_1 != span_id_2

    def test_events_with_token_id_use_it_for_span_id(self) -> None:
        """Events with token_id incorporate it in span_id."""
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        event1 = TokenCompleted(
            timestamp=ts,
            run_id="run-123",
            row_id="row-1",
            token_id="token-aaa",
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )
        event2 = TokenCompleted(
            timestamp=ts,
            run_id="run-123",
            row_id="row-1",
            token_id="token-bbb",
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )
        span_id_1 = _derive_span_id(event1)
        span_id_2 = _derive_span_id(event2)
        assert span_id_1 != span_id_2

    def test_span_id_is_64_bit_integer(self) -> None:
        """Span ID is a positive 64-bit integer."""
        event = RunStarted(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-123",
            config_hash="abc123",
            source_plugin="csv",
        )
        span_id = _derive_span_id(event)
        assert isinstance(span_id, int)
        assert span_id > 0
        assert span_id < 2**64


class TestOTLPExporterConfiguration:
    """Tests for OTLPExporter configuration."""

    def test_name_property(self) -> None:
        """Exporter name is 'otlp'."""
        exporter = OTLPExporter()
        assert exporter.name == "otlp"

    def test_missing_endpoint_raises(self) -> None:
        """Configuration without endpoint raises TelemetryExporterError."""
        exporter = OTLPExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({})
        assert "endpoint" in str(exc_info.value)

    def test_empty_endpoint_in_config_accepted(self) -> None:
        """Empty string endpoint is technically accepted (SDK will fail later)."""
        exporter = OTLPExporter()
        # This will fail at SDK level, but our config validation accepts it
        # since OpenTelemetry SDK should validate URLs
        with patch(OTLP_EXPORTER_PATCH) as mock_exporter_class:
            mock_exporter_class.return_value = MagicMock()
            exporter.configure({"endpoint": ""})
            mock_exporter_class.assert_called_once()

    def test_invalid_batch_size_raises(self) -> None:
        """batch_size < 1 raises TelemetryExporterError."""
        exporter = OTLPExporter()
        with patch(OTLP_EXPORTER_PATCH) as mock_exporter_class:
            mock_exporter_class.return_value = MagicMock()
            with pytest.raises(TelemetryExporterError) as exc_info:
                exporter.configure({"endpoint": "http://localhost:4317", "batch_size": 0})
            assert "batch_size" in str(exc_info.value)

    def test_negative_batch_size_raises(self) -> None:
        """Negative batch_size raises TelemetryExporterError."""
        exporter = OTLPExporter()
        with patch(OTLP_EXPORTER_PATCH) as mock_exporter_class:
            mock_exporter_class.return_value = MagicMock()
            with pytest.raises(TelemetryExporterError) as exc_info:
                exporter.configure({"endpoint": "http://localhost:4317", "batch_size": -5})
            assert "batch_size" in str(exc_info.value)

    def test_valid_configuration(self) -> None:
        """Valid configuration initializes exporter."""
        exporter = OTLPExporter()
        with patch(OTLP_EXPORTER_PATCH) as mock_exporter_class:
            mock_instance = MagicMock()
            mock_exporter_class.return_value = mock_instance

            exporter.configure(
                {
                    "endpoint": "http://localhost:4317",
                    "headers": {"Authorization": "Bearer token123"},
                    "batch_size": 50,
                }
            )

            mock_exporter_class.assert_called_once_with(
                endpoint="http://localhost:4317",
                headers=(("Authorization", "Bearer token123"),),
            )

    def test_default_batch_size(self) -> None:
        """Default batch_size is 100."""
        exporter = OTLPExporter()
        with patch(OTLP_EXPORTER_PATCH) as mock_exporter_class:
            mock_exporter_class.return_value = MagicMock()
            exporter.configure({"endpoint": "http://localhost:4317"})
            assert exporter._batch_size == 100

    def test_no_headers_configuration(self) -> None:
        """Configuration without headers passes None to SDK."""
        exporter = OTLPExporter()
        with patch(OTLP_EXPORTER_PATCH) as mock_exporter_class:
            mock_exporter_class.return_value = MagicMock()
            exporter.configure({"endpoint": "http://localhost:4317"})
            mock_exporter_class.assert_called_once_with(
                endpoint="http://localhost:4317",
                headers=None,
            )


class TestOTLPExporterBuffering:
    """Tests for event buffering and batch export."""

    def _create_configured_exporter(self, batch_size: int = 100) -> tuple[OTLPExporter, MagicMock]:
        """Create a configured exporter with mocked OTLP SDK."""
        exporter = OTLPExporter()
        with patch(OTLP_EXPORTER_PATCH) as mock_exporter_class:
            mock_instance = MagicMock()
            mock_exporter_class.return_value = mock_instance
            exporter.configure(
                {
                    "endpoint": "http://localhost:4317",
                    "batch_size": batch_size,
                }
            )
        return exporter, mock_instance

    def test_events_buffered_until_batch_size(self) -> None:
        """Events are buffered until batch_size is reached."""
        exporter, mock_sdk = self._create_configured_exporter(batch_size=3)

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )

        # First two events: buffered, not exported
        exporter.export(event)
        exporter.export(event)
        mock_sdk.export.assert_not_called()
        assert len(exporter._buffer) == 2

        # Third event: triggers batch export
        exporter.export(event)
        mock_sdk.export.assert_called_once()
        assert len(exporter._buffer) == 0

    def test_flush_exports_partial_batch(self) -> None:
        """flush() exports buffered events even if batch_size not reached."""
        exporter, mock_sdk = self._create_configured_exporter(batch_size=100)

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )

        exporter.export(event)
        exporter.export(event)
        mock_sdk.export.assert_not_called()

        exporter.flush()
        mock_sdk.export.assert_called_once()
        # Verify 2 spans were exported
        exported_spans = mock_sdk.export.call_args[0][0]
        assert len(exported_spans) == 2

    def test_flush_is_noop_when_buffer_empty(self) -> None:
        """flush() is a no-op when buffer is empty."""
        exporter, mock_sdk = self._create_configured_exporter()
        exporter.flush()
        mock_sdk.export.assert_not_called()

    def test_export_without_configure_logs_warning(self) -> None:
        """export() without configure() logs warning and continues."""
        exporter = OTLPExporter()
        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )

        # Should not raise
        exporter.export(event)
        # Buffer should be empty (not configured)
        assert len(exporter._buffer) == 0


class TestOTLPExporterSpanConversion:
    """Tests for event-to-span conversion."""

    def _create_configured_exporter(self) -> tuple[OTLPExporter, MagicMock]:
        """Create a configured exporter with mocked OTLP SDK."""
        exporter = OTLPExporter()
        with patch(OTLP_EXPORTER_PATCH) as mock_exporter_class:
            mock_instance = MagicMock()
            mock_exporter_class.return_value = mock_instance
            exporter.configure(
                {
                    "endpoint": "http://localhost:4317",
                    "batch_size": 1,  # Export immediately for testing
                }
            )
        return exporter, mock_instance

    def test_span_name_is_event_class_name(self) -> None:
        """Span name is the event class name."""
        exporter, mock_sdk = self._create_configured_exporter()

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )
        exporter.export(event)

        exported_spans = mock_sdk.export.call_args[0][0]
        assert exported_spans[0].name == "RunStarted"

    def test_span_trace_id_derived_from_run_id(self) -> None:
        """Span trace_id is derived from run_id."""
        exporter, mock_sdk = self._create_configured_exporter()

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )
        exporter.export(event)

        exported_spans = mock_sdk.export.call_args[0][0]
        expected_trace_id = _derive_trace_id("run-123")
        assert exported_spans[0].context.trace_id == expected_trace_id

    def test_span_attributes_contain_event_fields(self) -> None:
        """Span attributes contain all event fields."""
        exporter, mock_sdk = self._create_configured_exporter()

        event = RunStarted(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-123",
            config_hash="abc123",
            source_plugin="csv_source",
        )
        exporter.export(event)

        exported_spans = mock_sdk.export.call_args[0][0]
        attrs = exported_spans[0].attributes

        assert attrs["run_id"] == "run-123"
        assert attrs["config_hash"] == "abc123"
        assert attrs["source_plugin"] == "csv_source"
        assert attrs["event_type"] == "RunStarted"

    def test_datetime_serialized_as_iso8601(self) -> None:
        """datetime fields are serialized as ISO 8601 strings."""
        exporter, mock_sdk = self._create_configured_exporter()

        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        event = RunStarted(
            timestamp=ts,
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )
        exporter.export(event)

        exported_spans = mock_sdk.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        assert attrs["timestamp"] == "2024-01-15T10:30:00+00:00"

    def test_enum_serialized_as_value(self) -> None:
        """Enum fields are serialized as their string values."""
        exporter, mock_sdk = self._create_configured_exporter()

        event = RunFinished(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            status=RunStatus.COMPLETED,
            row_count=100,
            duration_ms=5000.0,
        )
        exporter.export(event)

        exported_spans = mock_sdk.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        assert attrs["status"] == "completed"

    def test_tuple_serialized_as_list(self) -> None:
        """Tuple fields are serialized as lists."""
        exporter, mock_sdk = self._create_configured_exporter()

        # GateEvaluated has destinations as tuple
        from elspeth.contracts import GateEvaluated
        from elspeth.contracts.enums import RoutingMode

        event = GateEvaluated(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            row_id="row-1",
            token_id="token-1",
            node_id="gate-1",
            plugin_name="threshold_gate",
            routing_mode=RoutingMode.MOVE,
            destinations=("sink_a", "sink_b"),
        )
        exporter.export(event)

        exported_spans = mock_sdk.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        assert attrs["destinations"] == ["sink_a", "sink_b"]

    def test_none_fields_omitted(self) -> None:
        """None fields are omitted from attributes."""
        exporter, mock_sdk = self._create_configured_exporter()

        event = TokenCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            row_id="row-1",
            token_id="token-1",
            outcome=RowOutcome.FAILED,
            sink_name=None,
        )
        exporter.export(event)

        exported_spans = mock_sdk.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        assert "sink_name" not in attrs

    def test_dict_serialized_as_json(self) -> None:
        """Dict fields are serialized as JSON strings (OTLP limitation)."""
        exporter, mock_sdk = self._create_configured_exporter()

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.telemetry.events import ExternalCallCompleted

        event = ExternalCallCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            state_id="state-1",
            call_type=CallType.LLM,
            provider="azure-openai",
            status=CallStatus.SUCCESS,
            latency_ms=150.0,
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        exporter.export(event)

        exported_spans = mock_sdk.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        # Dict should be JSON string
        import json

        token_usage = json.loads(attrs["token_usage"])
        assert token_usage == {"prompt_tokens": 100, "completion_tokens": 50}


class TestOTLPExporterLifecycle:
    """Tests for exporter lifecycle (close, error handling)."""

    def _create_configured_exporter(self) -> tuple[OTLPExporter, MagicMock]:
        """Create a configured exporter with mocked OTLP SDK."""
        exporter = OTLPExporter()
        with patch(OTLP_EXPORTER_PATCH) as mock_exporter_class:
            mock_instance = MagicMock()
            mock_exporter_class.return_value = mock_instance
            exporter.configure(
                {
                    "endpoint": "http://localhost:4317",
                    "batch_size": 100,
                }
            )
        return exporter, mock_instance

    def test_close_flushes_buffer(self) -> None:
        """close() flushes any remaining buffered events."""
        exporter, mock_sdk = self._create_configured_exporter()

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )
        exporter.export(event)

        exporter.close()
        mock_sdk.export.assert_called_once()

    def test_close_shuts_down_sdk_exporter(self) -> None:
        """close() calls shutdown on the underlying SDK exporter."""
        exporter, mock_sdk = self._create_configured_exporter()
        exporter.close()
        mock_sdk.shutdown.assert_called_once()

    def test_close_is_idempotent(self) -> None:
        """close() can be called multiple times safely."""
        exporter, mock_sdk = self._create_configured_exporter()
        exporter.close()
        exporter.close()  # Should not raise
        # shutdown called only once (exporter set to None after first close)
        mock_sdk.shutdown.assert_called_once()

    def test_export_failure_does_not_raise(self) -> None:
        """Export failures are logged but don't raise exceptions."""
        exporter, mock_sdk = self._create_configured_exporter()
        mock_sdk.export.side_effect = Exception("Network error")

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )

        # Should not raise even with batch_size=1 (immediate export)
        exporter._batch_size = 1
        exporter.export(event)  # Triggers export which fails
        # Buffer should be cleared despite failure
        assert len(exporter._buffer) == 0

    def test_flush_failure_does_not_raise(self) -> None:
        """flush() failures are logged but don't raise exceptions."""
        exporter, mock_sdk = self._create_configured_exporter()
        mock_sdk.export.side_effect = Exception("Network error")

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )
        exporter.export(event)

        # Should not raise
        exporter.flush()

    def test_close_failure_does_not_raise(self) -> None:
        """close() shutdown failures are logged but don't raise."""
        exporter, mock_sdk = self._create_configured_exporter()
        mock_sdk.shutdown.side_effect = Exception("Shutdown error")

        # Should not raise
        exporter.close()


class TestOTLPExporterRegistration:
    """Tests for plugin registration."""

    def test_exporter_in_builtin_plugin(self) -> None:
        """OTLPExporter is registered in BuiltinExportersPlugin."""
        from elspeth.telemetry.exporters import BuiltinExportersPlugin, OTLPExporter

        plugin = BuiltinExportersPlugin()
        exporters = plugin.elspeth_get_exporters()
        assert OTLPExporter in exporters

    def test_exporter_in_package_all(self) -> None:
        """OTLPExporter is exported from package __all__."""
        from elspeth.telemetry import exporters

        assert "OTLPExporter" in exporters.__all__


class TestSyntheticSpanSDKCompatibility:
    """Tests that verify _SyntheticReadableSpan works with actual SDK encoder.

    These tests catch SDK compatibility issues that mocked tests would miss.
    """

    def test_synthetic_span_encodes_with_sdk(self) -> None:
        """Verify _SyntheticReadableSpan works with actual OTLP encoder.

        This test catches SDK compatibility issues that mocked tests would miss.
        The SDK encoder accesses properties like dropped_attributes, dropped_events,
        and dropped_links that a mocked test would not exercise.
        """
        from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
        from opentelemetry.trace import SpanContext, SpanKind, TraceFlags

        from elspeth.telemetry.exporters.otlp import _SyntheticReadableSpan

        # Create a span context
        span_context = SpanContext(
            trace_id=0x12345678901234567890123456789012,
            span_id=0x1234567890123456,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )

        # Create a synthetic span
        span = _SyntheticReadableSpan(
            name="TestEvent",
            context=span_context,
            attributes={"test_key": "test_value", "count": 42},
            start_time=1700000000000000000,  # nanoseconds
            end_time=1700000000000000000,
            kind=SpanKind.INTERNAL,
        )

        # Verify encoding doesn't raise - this exercises all properties
        # including dropped_attributes, dropped_events, dropped_links
        result = encode_spans([span])
        assert result is not None
        # Verify we got a proper protobuf message
        assert hasattr(result, "resource_spans")

    def test_synthetic_span_has_required_dropped_properties(self) -> None:
        """Verify _SyntheticReadableSpan exposes dropped_* properties."""
        from opentelemetry.trace import SpanContext, SpanKind, TraceFlags

        from elspeth.telemetry.exporters.otlp import _SyntheticReadableSpan

        span_context = SpanContext(
            trace_id=0x12345678901234567890123456789012,
            span_id=0x1234567890123456,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )

        span = _SyntheticReadableSpan(
            name="TestEvent",
            context=span_context,
            attributes={},
            start_time=1700000000000000000,
            end_time=1700000000000000000,
            kind=SpanKind.INTERNAL,
        )

        # These properties must exist and return 0
        assert span.dropped_attributes == 0
        assert span.dropped_events == 0
        assert span.dropped_links == 0
