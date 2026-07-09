# tests/telemetry/exporters/test_azure_monitor.py
"""Tests for Azure Monitor telemetry exporter.

Tests cover:
- Configuration validation (connection_string required, batch_size > 0)
- Event buffering and batch export
- Event-to-span conversion with Azure-specific attributes
- Attribute serialization (datetime, enum, dict, tuple handling)
- Flush and close lifecycle
- Error handling (export failures don't crash pipeline)

Note: The Azure Monitor SDK is an optional dependency. These tests mock the SDK
import inside configure() to allow running without installing the package.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult

from elspeth.contracts.enums import RunStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.events import (
    RunFinished,
    RunStarted,
)
from elspeth.telemetry.errors import TelemetryExporterError

# Import the exporter class - it doesn't import Azure SDK at module level
from elspeth.telemetry.exporters.azure_monitor import AzureMonitorExporter


@dataclass
class AzureMonitorTraceExporterStub:
    """Small Azure SDK exporter fake with explicit call recording."""

    export_result: SpanExportResult | None = None
    export_error: BaseException | None = None
    shutdown_error: BaseException | None = None
    export_calls: list[list[Any]] = field(default_factory=list)
    shutdown_call_count: int = 0

    def export(self, spans: list[Any]) -> SpanExportResult | None:
        self.export_calls.append(spans)
        if self.export_error is not None:
            raise self.export_error
        return self.export_result

    def shutdown(self) -> None:
        self.shutdown_call_count += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


@dataclass
class AzureMonitorTraceExporterFactory:
    """Callable replacement for AzureMonitorTraceExporter."""

    instance: AzureMonitorTraceExporterStub = field(default_factory=AzureMonitorTraceExporterStub)
    constructor_kwargs: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, **kwargs: Any) -> AzureMonitorTraceExporterStub:
        self.constructor_kwargs.append(kwargs)
        return self.instance


@pytest.fixture
def azure_monitor_trace_exporter() -> AzureMonitorTraceExporterFactory:
    """Fixture that mocks the Azure Monitor SDK import inside configure().

    Uses patch on the specific import location rather than polluting sys.modules.
    This is the correct pattern for mocking optional dependencies that are
    imported lazily inside methods.

    Note: We don't mock the OpenTelemetry SDK (Resource, TracerProvider) because
    they are required to test the proper creation of resource attributes.
    """
    exporter_factory = AzureMonitorTraceExporterFactory()
    exporter_module = ModuleType("azure.monitor.opentelemetry.exporter")
    exporter_module.AzureMonitorTraceExporter = exporter_factory

    # Patch the import inside configure() - this is where the SDK is actually imported
    with patch.dict(
        "sys.modules",
        {"azure.monitor.opentelemetry.exporter": exporter_module},
    ):
        yield exporter_factory


_VALID_CONFIG: dict[str, object] = {
    "connection_string": "InstrumentationKey=test-key",
    "service_name": "test-service",
}


@pytest.fixture
def configured_exporter(azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory) -> AzureMonitorExporter:
    """Create a configured exporter with mocked Azure SDK."""
    exporter = AzureMonitorExporter()
    exporter.configure(dict(_VALID_CONFIG))
    return exporter


def make_run_started(run_id: str = "run-123") -> RunStarted:
    """Create a RunStarted event for testing."""
    return RunStarted(
        timestamp=datetime.now(UTC),
        run_id=run_id,
        config_hash="abc123",
        source_plugin="csv",
    )


def make_run_finished(run_id: str = "run-123") -> RunFinished:
    """Create a RunFinished event for testing."""
    return RunFinished(
        timestamp=datetime.now(UTC),
        run_id=run_id,
        status=RunStatus.COMPLETED,
        row_count=10,
        duration_ms=1500.0,
    )


class TestAzureMonitorExporterConfiguration:
    """Tests for AzureMonitorExporter configuration."""

    def test_name_property(self) -> None:
        """Exporter name is 'azure_monitor'."""
        exporter = AzureMonitorExporter()
        assert exporter.name == "azure_monitor"

    def test_configure_requires_connection_string(self) -> None:
        """Configuration fails without connection_string."""
        exporter = AzureMonitorExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({})
        assert "connection_string" in str(exc_info.value)

    def test_configure_validates_connection_string_type(self) -> None:
        """Configuration fails if connection_string is not a string."""
        exporter = AzureMonitorExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"connection_string": 123})
        assert "must be a string" in str(exc_info.value)

    def test_configure_validates_batch_size_type(self) -> None:
        """Configuration fails if batch_size is not an integer."""
        exporter = AzureMonitorExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({**_VALID_CONFIG, "connection_string": "test", "batch_size": "100"})
        assert "must be an integer" in str(exc_info.value)

    def test_configure_validates_batch_size_positive(self) -> None:
        """Configuration fails if batch_size < 1."""
        exporter = AzureMonitorExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({**_VALID_CONFIG, "connection_string": "test", "batch_size": 0})
        assert "must be >= 1" in str(exc_info.value)

    def test_configure_success_with_valid_config(self, azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory) -> None:
        """Configuration succeeds with valid connection_string."""
        exporter = AzureMonitorExporter()
        exporter.configure(dict(_VALID_CONFIG))

        assert len(azure_monitor_trace_exporter.constructor_kwargs) == 1
        assert exporter._configured is True

    def test_configure_passes_connection_string_to_sdk(self, azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory) -> None:
        """Connection string and tracer_provider are passed to Azure SDK."""
        exporter = AzureMonitorExporter()
        exporter.configure({**_VALID_CONFIG, "connection_string": "InstrumentationKey=my-key-123"})

        # Verify connection_string was passed
        call_kwargs = azure_monitor_trace_exporter.constructor_kwargs[0]
        assert call_kwargs["connection_string"] == "InstrumentationKey=my-key-123"
        # Verify tracer_provider was passed (fixes ProxyTracerProvider bug)
        assert "tracer_provider" in call_kwargs
        assert call_kwargs["tracer_provider"] is not None

    def test_configure_validates_service_name_type(self, azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory) -> None:
        """service_name must be a string if provided."""
        exporter = AzureMonitorExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({**_VALID_CONFIG, "connection_string": "test", "service_name": 123})
        assert "'service_name' must be a string" in str(exc_info.value)

    def test_configure_validates_service_version_type(self, azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory) -> None:
        """service_version must be a string or None if provided."""
        exporter = AzureMonitorExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({**_VALID_CONFIG, "connection_string": "test", "service_version": 123})
        assert "'service_version' must be a string or null" in str(exc_info.value)

    def test_configure_validates_deployment_environment_type(self, azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory) -> None:
        """deployment_environment must be a string or None if provided."""
        exporter = AzureMonitorExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({**_VALID_CONFIG, "connection_string": "test", "deployment_environment": 123})
        assert "'deployment_environment' must be a string or null" in str(exc_info.value)

    def test_configure_with_service_metadata(self, azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory) -> None:
        """Service metadata is passed to TracerProvider resource."""
        from opentelemetry.sdk.resources import SERVICE_NAME
        from opentelemetry.sdk.trace import TracerProvider

        exporter = AzureMonitorExporter()
        exporter.configure(
            {
                "connection_string": "InstrumentationKey=test-key",
                "service_name": "my-pipeline",
                "service_version": "2.0.0",
                "deployment_environment": "staging",
            }
        )

        # Verify tracer_provider was passed with correct resource
        call_kwargs = azure_monitor_trace_exporter.constructor_kwargs[0]
        tracer_provider = call_kwargs["tracer_provider"]
        assert isinstance(tracer_provider, TracerProvider)

        # Verify resource attributes
        resource_attrs = tracer_provider.resource.attributes
        assert resource_attrs[SERVICE_NAME] == "my-pipeline"
        assert resource_attrs["service.version"] == "2.0.0"
        assert resource_attrs["deployment.environment"] == "staging"

    def test_missing_service_name_raises(self) -> None:
        """Missing service_name raises TelemetryExporterError."""
        exporter = AzureMonitorExporter()
        with pytest.raises(TelemetryExporterError, match="service_name"):
            exporter.configure({"connection_string": "InstrumentationKey=test-key"})


class TestAzureMonitorExporterBuffering:
    """Tests for event buffering behavior."""

    def test_export_buffers_events(
        self,
        configured_exporter: AzureMonitorExporter,
        azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory,
    ) -> None:
        """Events are buffered until batch_size is reached."""
        event = make_run_started()
        configured_exporter.export(event)

        # Should be buffered, not exported yet
        assert azure_monitor_trace_exporter.instance.export_calls == []
        assert len(configured_exporter._buffer) == 1

    def test_export_flushes_at_batch_size(self, azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory) -> None:
        """Buffer is flushed when batch_size is reached."""
        exporter = AzureMonitorExporter()
        exporter.configure(
            {
                **_VALID_CONFIG,
                "batch_size": 2,
            }
        )

        event1 = make_run_started()
        event2 = make_run_finished()

        exporter.export(event1)
        assert azure_monitor_trace_exporter.instance.export_calls == []

        exporter.export(event2)
        assert len(azure_monitor_trace_exporter.instance.export_calls) == 1
        assert len(exporter._buffer) == 0


class TestAzureMonitorExporterSpanConversion:
    """Tests for event-to-span conversion."""

    def test_span_includes_azure_attributes(
        self,
        configured_exporter: AzureMonitorExporter,
        azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory,
    ) -> None:
        """Spans include Azure-specific attributes."""
        event = make_run_started()
        configured_exporter._buffer.append(event)
        configured_exporter._flush_batch()

        assert len(azure_monitor_trace_exporter.instance.export_calls) == 1
        spans = azure_monitor_trace_exporter.instance.export_calls[0]
        assert len(spans) == 1

        # Check Azure-specific attributes
        span = spans[0]
        assert span.attributes.get("cloud.provider") == "azure"
        assert span.attributes.get("elspeth.exporter") == "azure_monitor"

    def test_datetime_serialized_as_iso8601(
        self,
        configured_exporter: AzureMonitorExporter,
        azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory,
    ) -> None:
        """Datetime fields are serialized as ISO 8601 strings."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        event = RunStarted(
            timestamp=timestamp,
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )

        configured_exporter._buffer.append(event)
        configured_exporter._flush_batch()

        spans = azure_monitor_trace_exporter.instance.export_calls[0]
        assert spans[0].attributes.get("timestamp") == "2024-01-15T10:30:00+00:00"

    def test_enum_serialized_as_value(
        self,
        configured_exporter: AzureMonitorExporter,
        azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory,
    ) -> None:
        """Enum fields are serialized as their values."""
        event = make_run_finished()
        configured_exporter._buffer.append(event)
        configured_exporter._flush_batch()

        spans = azure_monitor_trace_exporter.instance.export_calls[0]
        assert spans[0].attributes.get("status") == "completed"


class TestAzureMonitorExporterLifecycle:
    """Tests for flush and close lifecycle."""

    def test_flush_exports_remaining_buffer(
        self,
        configured_exporter: AzureMonitorExporter,
        azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory,
    ) -> None:
        """flush() exports any buffered events."""
        event = make_run_started()
        configured_exporter.export(event)
        assert azure_monitor_trace_exporter.instance.export_calls == []

        configured_exporter.flush()
        assert len(azure_monitor_trace_exporter.instance.export_calls) == 1

    def test_sdk_failure_status_reports_handled_failure(
        self,
        configured_exporter: AzureMonitorExporter,
        azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory,
    ) -> None:
        """Azure SDK failure statuses are reported to the telemetry manager."""
        azure_monitor_trace_exporter.instance.export_result = SpanExportResult.FAILURE

        event = make_run_started()
        configured_exporter.export(event)

        result = configured_exporter.flush()
        assert result is False

    def test_flush_logs_ack_when_buffer_empty(
        self,
        configured_exporter: AzureMonitorExporter,
        azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory,
    ) -> None:
        """flush() logs acknowledgment when there are no buffered events."""
        with patch("elspeth.telemetry.exporters.azure_monitor.logger.debug") as mock_debug:
            configured_exporter.flush()

        assert azure_monitor_trace_exporter.instance.export_calls == []
        mock_debug.assert_called_once_with(
            "Azure Monitor flush requested with empty buffer",
            exporter="azure_monitor",
        )

    def test_close_flushes_and_shuts_down(
        self,
        configured_exporter: AzureMonitorExporter,
        azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory,
    ) -> None:
        """close() flushes buffer and shuts down SDK."""
        event = make_run_started()
        configured_exporter.export(event)
        configured_exporter.close()

        assert len(azure_monitor_trace_exporter.instance.export_calls) == 1
        assert azure_monitor_trace_exporter.instance.shutdown_call_count == 1

    def test_close_is_idempotent(
        self,
        configured_exporter: AzureMonitorExporter,
        azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory,
    ) -> None:
        """close() can be called multiple times safely."""
        configured_exporter.close()
        configured_exporter.close()

        # shutdown only called once (second close has no exporter)
        assert azure_monitor_trace_exporter.instance.shutdown_call_count == 1


class TestAzureMonitorExporterErrorHandling:
    """Tests for error handling - export failures should not crash pipeline."""

    def test_export_without_configure_logs_warning(self) -> None:
        """Export before configure() logs warning but doesn't crash."""
        exporter = AzureMonitorExporter()
        event = make_run_started()

        # Should not raise
        exporter.export(event)

        # Buffer should still be empty (event dropped)
        assert len(exporter._buffer) == 0

    def test_sdk_export_failure_reports_handled_failure(
        self,
        configured_exporter: AzureMonitorExporter,
        azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory,
    ) -> None:
        """SDK export transport failures are reported without raising."""
        azure_monitor_trace_exporter.instance.export_error = ConnectionError("SDK transport error")

        event = make_run_started()
        configured_exporter._buffer.append(event)

        result = configured_exporter._flush_batch()
        assert result is False

        # Buffer should be cleared even on failure
        assert len(configured_exporter._buffer) == 0

    def test_sdk_shutdown_failure_does_not_raise(
        self,
        configured_exporter: AzureMonitorExporter,
        azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory,
    ) -> None:
        """SDK shutdown failure is logged but doesn't raise."""
        azure_monitor_trace_exporter.instance.shutdown_error = ConnectionError("Shutdown transport error")

        # Should not raise
        configured_exporter.close()

    def test_programming_error_in_export_crashes(
        self,
        configured_exporter: AzureMonitorExporter,
        azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory,
    ) -> None:
        """Programming errors (non-transport) must crash — not be swallowed."""
        azure_monitor_trace_exporter.instance.export_error = ValueError("Bad payload construction")

        event = make_run_started()
        configured_exporter._buffer.append(event)

        with pytest.raises(ValueError, match="Bad payload construction"):
            configured_exporter._flush_batch()


class TestAzureMonitorExporterTokenCompleted:
    """Tests specifically for TokenCompleted event handling."""

    def test_token_completed_converted_to_span(
        self,
        configured_exporter: AzureMonitorExporter,
        azure_monitor_trace_exporter: AzureMonitorTraceExporterFactory,
    ) -> None:
        """TokenCompleted events are properly converted to spans."""
        from elspeth.contracts import TokenCompleted

        event = TokenCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            token_id="token-456",
            row_id="row-789",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )

        configured_exporter._buffer.append(event)
        configured_exporter._flush_batch()

        assert len(azure_monitor_trace_exporter.instance.export_calls) == 1
        spans = azure_monitor_trace_exporter.instance.export_calls[0]
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "TokenCompleted"
        assert span.attributes.get("token_id") == "token-456"
        assert span.attributes.get("outcome") == "success"
        assert span.attributes.get("path") == "default_flow"
