# tests/telemetry/exporters/test_otlp_integration.py
"""Integration tests for OTLP telemetry export.

These tests verify the complete export path:
1. TelemetryEvent → OpenTelemetry Span conversion
2. Span → OTLP protobuf format
3. Correct attributes, trace context, and span structure

Tests use the real OpenTelemetry SDK and verify spans can be encoded
to the OTLP wire format (protobuf).
"""

from datetime import UTC, datetime

import pytest
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace.export import SpanExportResult

from elspeth.contracts import TokenCompleted
from elspeth.contracts.enums import RowOutcome, RunStatus
from elspeth.telemetry.events import RunFinished, RunStarted
from elspeth.telemetry.exporters.otlp import OTLPExporter


class TestOTLPIntegration:
    """Integration tests verifying telemetry exports correctly via OTLP."""

    @pytest.fixture
    def captured_spans(self):
        """Capture spans and verify they encode to valid OTLP protobuf."""
        captured = []

        def capture_export(spans):
            # Verify spans can be encoded to OTLP protobuf format
            # This catches any SDK compatibility issues
            proto = encode_spans(list(spans))
            assert proto is not None
            assert hasattr(proto, "resource_spans")

            captured.extend(spans)
            return SpanExportResult.SUCCESS

        yield captured, capture_export

    @pytest.fixture
    def configured_exporter(self, captured_spans):
        """Create a real OTLP exporter with mocked transport."""
        captured, capture_export = captured_spans

        exporter = OTLPExporter()
        exporter.configure(
            {
                "endpoint": "http://localhost:4317",
                "batch_size": 1,  # Export immediately for testing
            }
        )

        # Replace export method to capture instead of sending over network
        exporter._span_exporter.export = capture_export  # type: ignore[method-assign]

        return exporter, captured

    def test_run_started_exports_valid_otlp(self, configured_exporter) -> None:
        """RunStarted event exports as valid OTLP span."""
        exporter, captured = configured_exporter

        event = RunStarted(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-otlp-test-123",
            config_hash="sha256:abc123",
            source_plugin="csv",
        )

        exporter.export(event)

        # Verify span was captured and is valid
        assert len(captured) == 1
        span = captured[0]

        assert span.name == "RunStarted"
        assert span.attributes.get("run_id") == "run-otlp-test-123"
        assert span.attributes.get("config_hash") == "sha256:abc123"
        assert span.attributes.get("source_plugin") == "csv"
        assert span.attributes.get("event_type") == "RunStarted"

    def test_run_finished_exports_with_enum_as_value(self, configured_exporter) -> None:
        """RunFinished event exports with enum status as string value."""
        exporter, captured = configured_exporter

        event = RunFinished(
            timestamp=datetime.now(UTC),
            run_id="run-456",
            status=RunStatus.COMPLETED,
            row_count=1000,
            duration_ms=5500.0,
        )

        exporter.export(event)

        assert len(captured) == 1
        span = captured[0]

        # Enum should be serialized as its value (string)
        assert span.attributes.get("status") == "completed"
        assert span.attributes.get("row_count") == 1000
        assert span.attributes.get("duration_ms") == 5500.0

    def test_token_completed_exports_correctly(self, configured_exporter) -> None:
        """TokenCompleted event exports with all fields."""
        exporter, captured = configured_exporter

        event = TokenCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-789",
            token_id="token-abc",
            row_id="row-def",
            outcome=RowOutcome.COMPLETED,
            sink_name="output_sink",
        )

        exporter.export(event)

        assert len(captured) == 1
        span = captured[0]

        assert span.name == "TokenCompleted"
        assert span.attributes.get("token_id") == "token-abc"
        assert span.attributes.get("row_id") == "row-def"
        assert span.attributes.get("outcome") == "completed"
        assert span.attributes.get("sink_name") == "output_sink"

    def test_trace_context_correlates_events(self, configured_exporter) -> None:
        """Events from same run share trace_id for distributed tracing correlation."""
        exporter, captured = configured_exporter

        run_id = "run-correlation-test"

        event1 = RunStarted(
            timestamp=datetime.now(UTC),
            run_id=run_id,
            config_hash="hash1",
            source_plugin="csv",
        )

        event2 = RunFinished(
            timestamp=datetime.now(UTC),
            run_id=run_id,
            status=RunStatus.COMPLETED,
            row_count=100,
            duration_ms=1000.0,
        )

        exporter.export(event1)
        exporter.export(event2)

        assert len(captured) == 2

        # Both spans should share the same trace_id
        trace_id_1 = captured[0].context.trace_id
        trace_id_2 = captured[1].context.trace_id

        assert trace_id_1 == trace_id_2, "Events from same run should share trace_id"
        assert trace_id_1 != 0, "trace_id should be non-zero"

        # But they should have different span_ids
        span_id_1 = captured[0].context.span_id
        span_id_2 = captured[1].context.span_id

        assert span_id_1 != span_id_2, "Different events should have different span_ids"


class TestOTLPSpanFormat:
    """Tests verifying spans conform to OTLP/OpenTelemetry expectations."""

    @pytest.fixture
    def exporter_with_capture(self):
        """Create exporter that captures spans for inspection."""
        captured = []

        def capture_export(spans):
            # Verify OTLP encoding works
            encode_spans(list(spans))
            captured.extend(spans)
            return SpanExportResult.SUCCESS

        exporter = OTLPExporter()
        exporter.configure(
            {
                "endpoint": "http://localhost:4317",
                "batch_size": 1,
            }
        )
        exporter._span_exporter.export = capture_export  # type: ignore[method-assign]

        return exporter, captured

    def test_span_timestamps_are_nanoseconds(self, exporter_with_capture) -> None:
        """Span timestamps are in nanoseconds (OpenTelemetry format)."""
        exporter, captured = exporter_with_capture

        timestamp = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        event = RunStarted(
            timestamp=timestamp,
            run_id="timestamp-test",
            config_hash="hash",
            source_plugin="csv",
        )
        exporter.export(event)

        span = captured[0]

        # OpenTelemetry uses nanoseconds since epoch
        expected_ns = int(timestamp.timestamp() * 1_000_000_000)
        assert span.start_time == expected_ns
        assert span.end_time == expected_ns  # Instant span

    def test_span_context_is_valid(self, exporter_with_capture) -> None:
        """Spans have valid trace context."""
        exporter, captured = exporter_with_capture

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="context-test",
            config_hash="hash",
            source_plugin="csv",
        )
        exporter.export(event)

        span = captured[0]

        assert span.context is not None
        assert span.context.trace_id != 0
        assert span.context.span_id != 0
        assert span.context.is_valid

    def test_datetime_serialized_as_iso8601_attribute(self, exporter_with_capture) -> None:
        """Datetime fields are serialized as ISO 8601 strings in attributes."""
        exporter, captured = exporter_with_capture

        timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        event = RunStarted(
            timestamp=timestamp,
            run_id="iso-test",
            config_hash="hash",
            source_plugin="csv",
        )
        exporter.export(event)

        span = captured[0]
        assert span.attributes.get("timestamp") == "2024-01-15T10:30:00+00:00"

    def test_dict_serialized_as_json_string(self, exporter_with_capture) -> None:
        """Dict fields are serialized as JSON strings (OTLP limitation)."""
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.telemetry.events import ExternalCallCompleted

        exporter, captured = exporter_with_capture

        event = ExternalCallCompleted(
            timestamp=datetime.now(UTC),
            run_id="dict-test",
            state_id="state-1",
            call_type=CallType.LLM,
            provider="openai",
            status=CallStatus.SUCCESS,
            latency_ms=100.0,
            token_usage={"prompt_tokens": 50, "completion_tokens": 25},
        )
        exporter.export(event)

        span = captured[0]
        import json

        token_usage = json.loads(span.attributes.get("token_usage"))
        assert token_usage == {"prompt_tokens": 50, "completion_tokens": 25}

    def test_tuple_serialized_as_list(self, exporter_with_capture) -> None:
        """Tuple fields are serialized as lists."""
        from elspeth.contracts import GateEvaluated
        from elspeth.contracts.enums import RoutingMode

        exporter, captured = exporter_with_capture

        event = GateEvaluated(
            timestamp=datetime.now(UTC),
            run_id="tuple-test",
            row_id="row-1",
            token_id="token-1",
            node_id="gate-1",
            plugin_name="threshold_gate",
            routing_mode=RoutingMode.COPY,
            destinations=("sink_a", "sink_b", "sink_c"),
        )
        exporter.export(event)

        span = captured[0]
        assert span.attributes.get("destinations") == ["sink_a", "sink_b", "sink_c"]

    def test_none_values_excluded_from_attributes(self, exporter_with_capture) -> None:
        """None values are not included as attributes."""
        exporter, captured = exporter_with_capture

        event = TokenCompleted(
            timestamp=datetime.now(UTC),
            run_id="none-test",
            token_id="token-1",
            row_id="row-1",
            outcome=RowOutcome.FAILED,
            sink_name=None,  # Explicitly None
        )
        exporter.export(event)

        span = captured[0]
        assert "sink_name" not in span.attributes


class TestOTLPBatching:
    """Tests for OTLP batching behavior with real SDK."""

    def test_multiple_events_batch_and_encode(self) -> None:
        """Multiple events batch correctly and encode to valid OTLP."""
        captured_batches = []

        def capture_export(spans):
            # Verify batch encodes correctly
            proto = encode_spans(list(spans))
            assert proto is not None
            captured_batches.append(list(spans))
            return SpanExportResult.SUCCESS

        exporter = OTLPExporter()
        exporter.configure(
            {
                "endpoint": "http://localhost:4317",
                "batch_size": 5,
            }
        )
        exporter._span_exporter.export = capture_export  # type: ignore[method-assign,union-attr]

        # Send 5 events (should trigger batch)
        for i in range(5):
            event = RunStarted(
                timestamp=datetime.now(UTC),
                run_id=f"run-{i}",
                config_hash=f"hash-{i}",
                source_plugin="csv",
            )
            exporter.export(event)

        # Should have one batch of 5 spans
        assert len(captured_batches) == 1
        assert len(captured_batches[0]) == 5

    def test_flush_sends_and_encodes_partial_batch(self) -> None:
        """flush() sends remaining events and they encode correctly."""
        captured_batches = []

        def capture_export(spans):
            proto = encode_spans(list(spans))
            assert proto is not None
            captured_batches.append(list(spans))
            return SpanExportResult.SUCCESS

        exporter = OTLPExporter()
        exporter.configure(
            {
                "endpoint": "http://localhost:4317",
                "batch_size": 100,  # Large batch size
            }
        )
        exporter._span_exporter.export = capture_export  # type: ignore[method-assign,union-attr]

        # Send 3 events (less than batch size)
        for i in range(3):
            event = RunStarted(
                timestamp=datetime.now(UTC),
                run_id=f"run-{i}",
                config_hash=f"hash-{i}",
                source_plugin="csv",
            )
            exporter.export(event)

        # Not sent yet
        assert len(captured_batches) == 0

        # Flush forces send
        exporter.flush()

        assert len(captured_batches) == 1
        assert len(captured_batches[0]) == 3
