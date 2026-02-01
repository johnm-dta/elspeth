# tests/telemetry/exporters/test_datadog_integration.py
"""Integration tests for Datadog telemetry export.

These tests verify the complete export path using the real ddtrace library:
1. TelemetryEvent → DatadogExporter.export()
2. Real span creation via ddtrace tracer
3. Correct span properties, tags, and timestamps

Tests use the real ddtrace library but don't require a running Datadog agent
(spans fail to send but that's expected and logged).
"""

from datetime import UTC, datetime

import pytest

# Skip entire module if ddtrace not installed
ddtrace = pytest.importorskip(
    "ddtrace",
    reason="ddtrace not installed",
)

from elspeth.contracts import TokenCompleted  # noqa: E402
from elspeth.contracts.enums import RowOutcome, RunStatus  # noqa: E402
from elspeth.telemetry.events import RunFinished, RunStarted  # noqa: E402
from elspeth.telemetry.exporters.datadog import DatadogExporter  # noqa: E402


class TestDatadogIntegration:
    """Integration tests verifying telemetry exports correctly via ddtrace."""

    @pytest.fixture
    def captured_spans(self):
        """Capture spans created by the exporter.

        Patches the tracer's start_span to capture spans while still
        creating real ddtrace span objects.
        """
        captured = []
        original_start_span = ddtrace.tracer.start_span

        def capturing_start_span(*args, **kwargs):
            span = original_start_span(*args, **kwargs)
            captured.append(span)
            return span

        ddtrace.tracer.start_span = capturing_start_span
        yield captured
        ddtrace.tracer.start_span = original_start_span

    @pytest.fixture
    def configured_exporter(self, captured_spans):
        """Create a configured Datadog exporter."""
        exporter = DatadogExporter()
        exporter.configure(
            {
                "service_name": "elspeth-integration-test",
                "env": "test",
                "agent_host": "localhost",
                "agent_port": 8126,
            }
        )
        return exporter, captured_spans

    def test_run_started_creates_real_span(self, configured_exporter) -> None:
        """RunStarted event creates a real ddtrace span."""
        exporter, captured = configured_exporter

        event = RunStarted(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-integration-test-123",
            config_hash="sha256:abc123",
            source_plugin="csv",
        )

        exporter.export(event)

        assert len(captured) == 1
        span = captured[0]

        assert span.name == "RunStarted"
        assert span.service == "elspeth-integration-test"
        assert span.resource == "RunStarted"

    def test_span_has_correct_tags(self, configured_exporter) -> None:
        """Span has all expected ELSPETH tags."""
        exporter, captured = configured_exporter

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-with-tags",
            config_hash="hash123",
            source_plugin="api",
        )

        exporter.export(event)

        span = captured[0]
        tags = dict(span._meta)

        # Standard Datadog tags
        assert tags.get("env") == "test"

        # ELSPETH-specific tags
        assert tags.get("elspeth.run_id") == "run-with-tags"
        assert tags.get("elspeth.event_type") == "RunStarted"
        assert tags.get("elspeth.config_hash") == "hash123"
        assert tags.get("elspeth.source_plugin") == "api"

    def test_span_timestamp_from_event(self, configured_exporter) -> None:
        """Span start time comes from event.timestamp, not export time."""
        exporter, captured = configured_exporter

        # Use a timestamp in the past
        event_timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        expected_ns = int(event_timestamp.timestamp() * 1_000_000_000)

        event = RunStarted(
            timestamp=event_timestamp,
            run_id="timestamp-test",
            config_hash="hash",
            source_plugin="csv",
        )

        exporter.export(event)

        span = captured[0]
        assert span.start_ns == expected_ns, (
            f"Span start_ns should be from event.timestamp ({expected_ns}), not auto-generated. Got: {span.start_ns}"
        )

    def test_run_finished_with_enum_status(self, configured_exporter) -> None:
        """RunFinished event correctly serializes enum status."""
        exporter, captured = configured_exporter

        event = RunFinished(
            timestamp=datetime.now(UTC),
            run_id="run-456",
            status=RunStatus.COMPLETED,
            row_count=1000,
            duration_ms=5500.0,
        )

        exporter.export(event)

        span = captured[0]
        tags = dict(span._meta)

        # Enum should be serialized as its value
        assert tags.get("elspeth.status") == "completed"

    def test_token_completed_with_outcome(self, configured_exporter) -> None:
        """TokenCompleted event exports with outcome enum."""
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

        span = captured[0]
        tags = dict(span._meta)

        assert span.name == "TokenCompleted"
        assert tags.get("elspeth.token_id") == "token-abc"
        assert tags.get("elspeth.row_id") == "row-def"
        assert tags.get("elspeth.outcome") == "completed"
        assert tags.get("elspeth.sink_name") == "output_sink"

    def test_datetime_serialized_as_iso8601(self, configured_exporter) -> None:
        """Datetime fields are serialized as ISO 8601 strings."""
        exporter, captured = configured_exporter

        timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        event = RunStarted(
            timestamp=timestamp,
            run_id="iso-test",
            config_hash="hash",
            source_plugin="csv",
        )

        exporter.export(event)

        span = captured[0]
        tags = dict(span._meta)
        assert tags.get("elspeth.timestamp") == "2024-01-15T10:30:00+00:00"


class TestDatadogSpanFormat:
    """Tests verifying spans conform to Datadog expectations."""

    @pytest.fixture
    def exporter_with_capture(self):
        """Create exporter that captures spans for inspection."""
        captured = []
        original_start_span = ddtrace.tracer.start_span

        def capturing_start_span(*args, **kwargs):
            span = original_start_span(*args, **kwargs)
            captured.append(span)
            return span

        ddtrace.tracer.start_span = capturing_start_span

        exporter = DatadogExporter()
        exporter.configure(
            {
                "service_name": "test-service",
                "env": "test",
                "version": "1.2.3",
            }
        )

        yield exporter, captured

        ddtrace.tracer.start_span = original_start_span

    def test_version_tag_included(self, exporter_with_capture) -> None:
        """Version tag is set when configured."""
        exporter, captured = exporter_with_capture

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="version-test",
            config_hash="hash",
            source_plugin="csv",
        )
        exporter.export(event)

        span = captured[0]
        tags = dict(span._meta)
        assert tags.get("version") == "1.2.3"

    def test_span_has_valid_ids(self, exporter_with_capture) -> None:
        """Spans have valid trace and span IDs."""
        exporter, captured = exporter_with_capture

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="id-test",
            config_hash="hash",
            source_plugin="csv",
        )
        exporter.export(event)

        span = captured[0]

        assert span.trace_id != 0
        assert span.span_id != 0

    def test_instant_span_has_zero_duration(self, exporter_with_capture) -> None:
        """Telemetry spans are instant (start == finish)."""
        exporter, captured = exporter_with_capture

        event_timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        event = RunStarted(
            timestamp=event_timestamp,
            run_id="duration-test",
            config_hash="hash",
            source_plugin="csv",
        )
        exporter.export(event)

        span = captured[0]
        # Duration should be 0 for instant spans
        assert span.duration_ns == 0, f"Expected 0 duration, got {span.duration_ns}"

    def test_none_values_not_in_tags(self, exporter_with_capture) -> None:
        """None values are not included as tags."""
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
        tag_keys = list(span._meta.keys())
        assert "elspeth.sink_name" not in tag_keys

    def test_dict_flattened_to_dotted_tags(self, exporter_with_capture) -> None:
        """Dict fields are flattened to dotted tag keys.

        Note: ddtrace stores numeric values in _metrics, strings in _meta.
        """
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
        # ddtrace stores numeric values in _metrics, not _meta
        metrics = dict(span._metrics)

        # Dict should be flattened with numeric values in metrics
        assert metrics.get("elspeth.token_usage.prompt_tokens") == 50
        assert metrics.get("elspeth.token_usage.completion_tokens") == 25

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
        tags = dict(span._meta)

        # Tuples become lists (ddtrace converts to string representation)
        destinations = tags.get("elspeth.destinations")
        assert destinations is not None
        # ddtrace may serialize list as string
        assert "sink_a" in str(destinations)
        assert "sink_b" in str(destinations)
        assert "sink_c" in str(destinations)


class TestDatadogMultipleEvents:
    """Tests for multiple event handling."""

    @pytest.fixture
    def exporter_with_capture(self):
        """Create exporter that captures spans."""
        captured = []
        original_start_span = ddtrace.tracer.start_span

        def capturing_start_span(*args, **kwargs):
            span = original_start_span(*args, **kwargs)
            captured.append(span)
            return span

        ddtrace.tracer.start_span = capturing_start_span

        exporter = DatadogExporter()
        exporter.configure({"service_name": "test-service"})

        yield exporter, captured

        ddtrace.tracer.start_span = original_start_span

    def test_multiple_events_create_separate_spans(self, exporter_with_capture) -> None:
        """Each event creates a separate span."""
        exporter, captured = exporter_with_capture

        for i in range(5):
            event = RunStarted(
                timestamp=datetime.now(UTC),
                run_id=f"run-{i}",
                config_hash=f"hash-{i}",
                source_plugin="csv",
            )
            exporter.export(event)

        assert len(captured) == 5

        # Each span should have unique run_id
        run_ids = [dict(span._meta).get("elspeth.run_id") for span in captured]
        assert run_ids == ["run-0", "run-1", "run-2", "run-3", "run-4"]

    def test_different_event_types_create_correctly_named_spans(self, exporter_with_capture) -> None:
        """Different event types create spans with correct names."""
        exporter, captured = exporter_with_capture

        run_id = "multi-event-run"

        exporter.export(
            RunStarted(
                timestamp=datetime.now(UTC),
                run_id=run_id,
                config_hash="hash",
                source_plugin="csv",
            )
        )

        exporter.export(
            TokenCompleted(
                timestamp=datetime.now(UTC),
                run_id=run_id,
                token_id="token-1",
                row_id="row-1",
                outcome=RowOutcome.COMPLETED,
                sink_name="output",
            )
        )

        exporter.export(
            RunFinished(
                timestamp=datetime.now(UTC),
                run_id=run_id,
                status=RunStatus.COMPLETED,
                row_count=1,
                duration_ms=100.0,
            )
        )

        assert len(captured) == 3
        names = [span.name for span in captured]
        assert names == ["RunStarted", "TokenCompleted", "RunFinished"]
