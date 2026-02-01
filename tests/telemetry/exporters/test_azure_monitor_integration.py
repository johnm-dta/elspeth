# tests/telemetry/exporters/test_azure_monitor_integration.py
"""Integration tests for Azure Monitor telemetry export to Application Insights.

These tests verify the complete export path:
1. TelemetryEvent → OpenTelemetry Span conversion
2. Span → Application Insights format
3. Correct attributes, trace context, and span structure

Tests use the real Azure Monitor SDK but mock the HTTP transport layer
to capture what would be sent to Application Insights.
"""

from datetime import UTC, datetime

import pytest

# Skip entire module if Azure Monitor SDK not installed
azure_monitor = pytest.importorskip(
    "azure.monitor.opentelemetry.exporter",
    reason="azure-monitor-opentelemetry-exporter not installed",
)

from elspeth.contracts.enums import RowOutcome, RunStatus  # noqa: E402
from elspeth.telemetry.events import RunFinished, RunStarted  # noqa: E402
from elspeth.telemetry.exporters.azure_monitor import AzureMonitorExporter  # noqa: E402


class TestAzureMonitorIntegration:
    """Integration tests verifying telemetry reaches App Insights correctly."""

    @pytest.fixture
    def captured_spans(self):
        """Capture spans sent to the Azure Monitor exporter.

        Patches the underlying SDK's export method to capture spans
        without making actual HTTP calls.
        """
        captured = []

        def capture_export(spans):
            captured.extend(spans)
            # Return success
            from opentelemetry.sdk.trace.export import SpanExportResult

            return SpanExportResult.SUCCESS

        yield captured, capture_export

    @pytest.fixture
    def configured_exporter(self, captured_spans):
        """Create a real Azure Monitor exporter with mocked transport."""
        captured, capture_export = captured_spans

        exporter = AzureMonitorExporter()

        # Configure with a dummy connection string
        # The SDK validates format but we mock before it sends
        connection_string = (
            "InstrumentationKey=00000000-0000-0000-0000-000000000000;IngestionEndpoint=https://test.in.applicationinsights.azure.com/"
        )
        exporter.configure({"connection_string": connection_string, "batch_size": 1})

        # Patch the SDK's export to capture instead of send
        exporter._azure_exporter.export = capture_export  # type: ignore[method-assign,union-attr]

        return exporter, captured

    def test_run_started_exports_to_appinsights(self, configured_exporter) -> None:
        """RunStarted event exports with correct span structure."""
        exporter, captured = configured_exporter

        event = RunStarted(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            run_id="run-integration-test-123",
            config_hash="sha256:abc123",
            source_plugin="csv",
        )

        exporter.export(event)

        # With batch_size=1, should immediately export
        assert len(captured) == 1

        span = captured[0]
        assert span.name == "RunStarted"

        # Verify Azure-specific attributes are present
        assert span.attributes.get("cloud.provider") == "azure"
        assert span.attributes.get("elspeth.exporter") == "azure_monitor"

        # Verify event data is preserved
        assert span.attributes.get("run_id") == "run-integration-test-123"
        assert span.attributes.get("config_hash") == "sha256:abc123"
        assert span.attributes.get("source_plugin") == "csv"
        assert span.attributes.get("event_type") == "RunStarted"

        # Verify timestamp is ISO 8601
        assert span.attributes.get("timestamp") == "2024-01-15T10:30:00+00:00"

    def test_run_finished_exports_with_status(self, configured_exporter) -> None:
        """RunFinished event exports with enum status as value."""
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

        # Enum should be serialized as its value
        assert span.attributes.get("status") == "completed"
        assert span.attributes.get("row_count") == 1000
        assert span.attributes.get("duration_ms") == 5500.0

    def test_token_completed_exports_with_outcome(self, configured_exporter) -> None:
        """TokenCompleted event exports with outcome enum as value."""
        from elspeth.contracts import TokenCompleted

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

    def test_trace_context_is_consistent(self, configured_exporter) -> None:
        """Events from same run share trace_id for correlation."""
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

        # Both spans should have the same trace_id (derived from run_id)
        trace_id_1 = captured[0].context.trace_id
        trace_id_2 = captured[1].context.trace_id

        assert trace_id_1 == trace_id_2, "Events from same run should share trace_id"
        assert trace_id_1 != 0, "trace_id should be non-zero"

    def test_multiple_events_batch_correctly(self) -> None:
        """Multiple events batch together when batch_size > 1."""
        captured = []

        def capture_export(spans):
            captured.append(list(spans))  # Capture as batch
            from opentelemetry.sdk.trace.export import SpanExportResult

            return SpanExportResult.SUCCESS

        exporter = AzureMonitorExporter()
        connection_string = (
            "InstrumentationKey=00000000-0000-0000-0000-000000000000;IngestionEndpoint=https://test.in.applicationinsights.azure.com/"
        )
        exporter.configure({"connection_string": connection_string, "batch_size": 3})
        exporter._azure_exporter.export = capture_export  # type: ignore[method-assign,union-attr]  # type: ignore[method-assign,union-attr]

        # Send 3 events (should trigger batch)
        for i in range(3):
            event = RunStarted(
                timestamp=datetime.now(UTC),
                run_id=f"run-{i}",
                config_hash=f"hash-{i}",
                source_plugin="csv",
            )
            exporter.export(event)

        # Should have one batch of 3 spans
        assert len(captured) == 1
        assert len(captured[0]) == 3

    def test_flush_sends_partial_batch(self) -> None:
        """flush() sends remaining events even if batch not full."""
        captured = []

        def capture_export(spans):
            captured.append(list(spans))
            from opentelemetry.sdk.trace.export import SpanExportResult

            return SpanExportResult.SUCCESS

        exporter = AzureMonitorExporter()
        connection_string = (
            "InstrumentationKey=00000000-0000-0000-0000-000000000000;IngestionEndpoint=https://test.in.applicationinsights.azure.com/"
        )
        exporter.configure({"connection_string": connection_string, "batch_size": 10})
        exporter._azure_exporter.export = capture_export  # type: ignore[method-assign,union-attr]  # type: ignore[method-assign,union-attr]

        # Send 2 events (less than batch_size of 10)
        for i in range(2):
            event = RunStarted(
                timestamp=datetime.now(UTC),
                run_id=f"run-{i}",
                config_hash=f"hash-{i}",
                source_plugin="csv",
            )
            exporter.export(event)

        # Not sent yet
        assert len(captured) == 0

        # Flush forces send
        exporter.flush()

        assert len(captured) == 1
        assert len(captured[0]) == 2


class TestAzureMonitorSpanFormat:
    """Tests verifying spans conform to Application Insights expectations."""

    @pytest.fixture
    def exporter_with_capture(self):
        """Create exporter that captures spans for inspection."""
        captured = []

        def capture_export(spans):
            captured.extend(spans)
            from opentelemetry.sdk.trace.export import SpanExportResult

            return SpanExportResult.SUCCESS

        exporter = AzureMonitorExporter()
        connection_string = (
            "InstrumentationKey=00000000-0000-0000-0000-000000000000;IngestionEndpoint=https://test.in.applicationinsights.azure.com/"
        )
        exporter.configure({"connection_string": connection_string, "batch_size": 1})
        exporter._azure_exporter.export = capture_export  # type: ignore[method-assign,union-attr]

        return exporter, captured

    def test_span_has_valid_context(self, exporter_with_capture) -> None:
        """Spans have valid trace context for App Insights correlation."""
        exporter, captured = exporter_with_capture

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="context-test-run",
            config_hash="hash",
            source_plugin="csv",
        )
        exporter.export(event)

        span = captured[0]

        # Verify span context exists and has valid IDs
        assert span.context is not None
        assert span.context.trace_id != 0
        assert span.context.span_id != 0
        assert span.context.is_valid  # Property, not method

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

    def test_none_values_are_excluded(self, exporter_with_capture) -> None:
        """None values are not included as attributes."""
        from elspeth.contracts import TokenCompleted

        exporter, captured = exporter_with_capture

        event = TokenCompleted(
            timestamp=datetime.now(UTC),
            run_id="none-test",
            token_id="token-1",
            row_id="row-1",
            outcome=RowOutcome.COMPLETED,
            sink_name=None,  # Explicitly None
        )
        exporter.export(event)

        span = captured[0]

        # sink_name should not be in attributes since it's None
        assert "sink_name" not in span.attributes or span.attributes.get("sink_name") is None
