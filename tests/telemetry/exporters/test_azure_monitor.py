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

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.enums import RowOutcome, RunStatus
from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.events import (
    RunFinished,
    RunStarted,
)

# Import the exporter class - it doesn't import Azure SDK at module level
from elspeth.telemetry.exporters.azure_monitor import AzureMonitorExporter


@pytest.fixture
def mock_azure_exporter():
    """Fixture that mocks the Azure Monitor SDK import inside configure().

    Uses patch on the specific import location rather than polluting sys.modules.
    This is the correct pattern for mocking optional dependencies that are
    imported lazily inside methods.

    Note: We don't mock the OpenTelemetry SDK (Resource, TracerProvider) because
    they are required to test the proper creation of resource attributes.
    """
    mock_exporter_class = MagicMock()
    mock_instance = MagicMock()
    mock_exporter_class.return_value = mock_instance

    # Patch the import inside configure() - this is where the SDK is actually imported
    with patch.dict(
        "sys.modules",
        {"azure.monitor.opentelemetry.exporter": MagicMock(AzureMonitorTraceExporter=mock_exporter_class)},
    ):
        yield {
            "class": mock_exporter_class,
            "instance": mock_instance,
        }


@pytest.fixture
def configured_exporter(mock_azure_exporter):
    """Create a configured exporter with mocked Azure SDK."""
    exporter = AzureMonitorExporter()
    exporter.configure({"connection_string": "InstrumentationKey=test-key"})
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
            exporter.configure({"connection_string": "test", "batch_size": "100"})
        assert "must be an integer" in str(exc_info.value)

    def test_configure_validates_batch_size_positive(self) -> None:
        """Configuration fails if batch_size < 1."""
        exporter = AzureMonitorExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"connection_string": "test", "batch_size": 0})
        assert "must be >= 1" in str(exc_info.value)

    def test_configure_success_with_valid_config(self, mock_azure_exporter) -> None:
        """Configuration succeeds with valid connection_string."""
        exporter = AzureMonitorExporter()
        exporter.configure({"connection_string": "InstrumentationKey=test-key"})

        mock_azure_exporter["class"].assert_called_once()
        assert exporter._configured is True

    def test_configure_passes_connection_string_to_sdk(self, mock_azure_exporter) -> None:
        """Connection string and tracer_provider are passed to Azure SDK."""
        exporter = AzureMonitorExporter()
        exporter.configure({"connection_string": "InstrumentationKey=my-key-123"})

        # Verify connection_string was passed
        call_kwargs = mock_azure_exporter["class"].call_args.kwargs
        assert call_kwargs["connection_string"] == "InstrumentationKey=my-key-123"
        # Verify tracer_provider was passed (fixes ProxyTracerProvider bug)
        assert "tracer_provider" in call_kwargs
        assert call_kwargs["tracer_provider"] is not None

    def test_configure_validates_service_name_type(self, mock_azure_exporter) -> None:
        """service_name must be a string if provided."""
        exporter = AzureMonitorExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"connection_string": "test", "service_name": 123})
        assert "'service_name' must be a string" in str(exc_info.value)

    def test_configure_validates_service_version_type(self, mock_azure_exporter) -> None:
        """service_version must be a string or None if provided."""
        exporter = AzureMonitorExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"connection_string": "test", "service_version": 123})
        assert "'service_version' must be a string or null" in str(exc_info.value)

    def test_configure_validates_deployment_environment_type(self, mock_azure_exporter) -> None:
        """deployment_environment must be a string or None if provided."""
        exporter = AzureMonitorExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"connection_string": "test", "deployment_environment": 123})
        assert "'deployment_environment' must be a string or null" in str(exc_info.value)

    def test_configure_with_service_metadata(self, mock_azure_exporter) -> None:
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
        call_kwargs = mock_azure_exporter["class"].call_args.kwargs
        tracer_provider = call_kwargs["tracer_provider"]
        assert isinstance(tracer_provider, TracerProvider)

        # Verify resource attributes
        resource_attrs = tracer_provider.resource.attributes
        assert resource_attrs[SERVICE_NAME] == "my-pipeline"
        assert resource_attrs["service.version"] == "2.0.0"
        assert resource_attrs["deployment.environment"] == "staging"

    def test_configure_default_service_name(self, mock_azure_exporter) -> None:
        """Default service name is 'elspeth' when not specified."""
        from opentelemetry.sdk.resources import SERVICE_NAME

        exporter = AzureMonitorExporter()
        exporter.configure({"connection_string": "InstrumentationKey=test-key"})

        call_kwargs = mock_azure_exporter["class"].call_args.kwargs
        tracer_provider = call_kwargs["tracer_provider"]
        assert tracer_provider.resource.attributes[SERVICE_NAME] == "elspeth"


class TestAzureMonitorExporterBuffering:
    """Tests for event buffering behavior."""

    def test_export_buffers_events(self, configured_exporter, mock_azure_exporter) -> None:
        """Events are buffered until batch_size is reached."""
        event = make_run_started()
        configured_exporter.export(event)

        # Should be buffered, not exported yet
        mock_azure_exporter["instance"].export.assert_not_called()
        assert len(configured_exporter._buffer) == 1

    def test_export_flushes_at_batch_size(self, mock_azure_exporter) -> None:
        """Buffer is flushed when batch_size is reached."""
        exporter = AzureMonitorExporter()
        exporter.configure(
            {
                "connection_string": "InstrumentationKey=test-key",
                "batch_size": 2,
            }
        )

        event1 = make_run_started()
        event2 = make_run_finished()

        exporter.export(event1)
        mock_azure_exporter["instance"].export.assert_not_called()

        exporter.export(event2)
        mock_azure_exporter["instance"].export.assert_called_once()
        assert len(exporter._buffer) == 0


class TestAzureMonitorExporterSpanConversion:
    """Tests for event-to-span conversion."""

    def test_span_includes_azure_attributes(self, configured_exporter, mock_azure_exporter) -> None:
        """Spans include Azure-specific attributes."""
        event = make_run_started()
        configured_exporter._buffer.append(event)
        configured_exporter._flush_batch()

        mock_azure_exporter["instance"].export.assert_called_once()
        spans = mock_azure_exporter["instance"].export.call_args[0][0]
        assert len(spans) == 1

        # Check Azure-specific attributes
        span = spans[0]
        assert span.attributes.get("cloud.provider") == "azure"
        assert span.attributes.get("elspeth.exporter") == "azure_monitor"

    def test_datetime_serialized_as_iso8601(self, configured_exporter, mock_azure_exporter) -> None:
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

        spans = mock_azure_exporter["instance"].export.call_args[0][0]
        assert spans[0].attributes.get("timestamp") == "2024-01-15T10:30:00+00:00"

    def test_enum_serialized_as_value(self, configured_exporter, mock_azure_exporter) -> None:
        """Enum fields are serialized as their values."""
        event = make_run_finished()
        configured_exporter._buffer.append(event)
        configured_exporter._flush_batch()

        spans = mock_azure_exporter["instance"].export.call_args[0][0]
        assert spans[0].attributes.get("status") == "completed"


class TestAzureMonitorExporterLifecycle:
    """Tests for flush and close lifecycle."""

    def test_flush_exports_remaining_buffer(self, configured_exporter, mock_azure_exporter) -> None:
        """flush() exports any buffered events."""
        event = make_run_started()
        configured_exporter.export(event)
        mock_azure_exporter["instance"].export.assert_not_called()

        configured_exporter.flush()
        mock_azure_exporter["instance"].export.assert_called_once()

    def test_close_flushes_and_shuts_down(self, configured_exporter, mock_azure_exporter) -> None:
        """close() flushes buffer and shuts down SDK."""
        event = make_run_started()
        configured_exporter.export(event)
        configured_exporter.close()

        mock_azure_exporter["instance"].export.assert_called_once()
        mock_azure_exporter["instance"].shutdown.assert_called_once()

    def test_close_is_idempotent(self, configured_exporter, mock_azure_exporter) -> None:
        """close() can be called multiple times safely."""
        configured_exporter.close()
        configured_exporter.close()

        # shutdown only called once (second close has no exporter)
        mock_azure_exporter["instance"].shutdown.assert_called_once()


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

    def test_sdk_export_failure_does_not_raise(self, configured_exporter, mock_azure_exporter) -> None:
        """SDK export failure is logged but doesn't raise."""
        mock_azure_exporter["instance"].export.side_effect = Exception("SDK error")

        event = make_run_started()
        configured_exporter._buffer.append(event)

        # Should not raise
        configured_exporter._flush_batch()

        # Buffer should be cleared even on failure
        assert len(configured_exporter._buffer) == 0

    def test_sdk_shutdown_failure_does_not_raise(self, configured_exporter, mock_azure_exporter) -> None:
        """SDK shutdown failure is logged but doesn't raise."""
        mock_azure_exporter["instance"].shutdown.side_effect = Exception("Shutdown error")

        # Should not raise
        configured_exporter.close()


class TestAzureMonitorExporterTokenCompleted:
    """Tests specifically for TokenCompleted event handling."""

    def test_token_completed_converted_to_span(self, configured_exporter, mock_azure_exporter) -> None:
        """TokenCompleted events are properly converted to spans."""
        from elspeth.contracts import TokenCompleted

        event = TokenCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            token_id="token-456",
            row_id="row-789",
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )

        configured_exporter._buffer.append(event)
        configured_exporter._flush_batch()

        mock_azure_exporter["instance"].export.assert_called_once()
        spans = mock_azure_exporter["instance"].export.call_args[0][0]
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "TokenCompleted"
        assert span.attributes.get("token_id") == "token-456"
        assert span.attributes.get("outcome") == "completed"
