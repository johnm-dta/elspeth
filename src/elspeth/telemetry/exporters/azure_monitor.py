# src/elspeth/telemetry/exporters/azure_monitor.py
"""Azure Monitor exporter for telemetry events.

Exports telemetry events to Azure Monitor / Application Insights using
the azure-monitor-opentelemetry-exporter package.

Converts ELSPETH TelemetryEvents to OpenTelemetry Spans and ships them
to Application Insights for distributed tracing, monitoring, and alerting.
"""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING, Any

import structlog

from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.exporters.otlp import (
    _derive_span_id,
    _derive_trace_id,
    _SyntheticReadableSpan,
)

if TYPE_CHECKING:
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

    from elspeth.contracts.events import TelemetryEvent

logger = structlog.get_logger(__name__)


class AzureMonitorExporter:
    """Export telemetry events to Azure Monitor / Application Insights.

    Uses azure-monitor-opentelemetry-exporter for native integration with
    Azure observability stack. Spans appear in Application Insights under
    the "Distributed Tracing" blade.

    Configuration options:
        connection_string: Application Insights connection string (required).
            Typically from APPLICATIONINSIGHTS_CONNECTION_STRING env var.
        batch_size: Number of events to buffer before export (default: 100)
        service_name: Service name for resource attributes (default: "elspeth")
        service_version: Service version (optional)
        deployment_environment: Deployment environment (optional, e.g. "production")

    Example configuration:
        telemetry:
          exporters:
            - name: azure_monitor
              connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
              batch_size: 100
              service_name: "my-pipeline"
              service_version: "1.0.0"
              deployment_environment: "production"

    Thread safety:
        Assumes single-threaded access. Buffer is not thread-safe.

    Azure-specific attributes:
        All spans include cloud.provider="azure" for filtering in
        Application Insights queries.

    Resource attributes:
        This exporter creates its own TracerProvider with proper Resource
        attributes. This avoids the ProxyTracerProvider issue where the
        Azure Monitor SDK tries to access `get_tracer_provider().resource`
        but gets a ProxyTracerProvider with no resource attribute.
    """

    _name = "azure_monitor"

    def __init__(self) -> None:
        """Initialize unconfigured exporter."""
        self._connection_string: str | None = None
        self._batch_size: int = 100
        self._service_name: str = "elspeth"
        self._service_version: str | None = None
        self._deployment_environment: str | None = None
        self._azure_exporter: AzureMonitorTraceExporter | None = None
        self._resource: Any | None = None  # Resource - stored for span creation
        self._buffer: list[TelemetryEvent] = []
        self._configured: bool = False

    @property
    def name(self) -> str:
        """Exporter name for configuration reference."""
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the exporter with settings from pipeline configuration.

        Args:
            config: Exporter-specific configuration dict containing:
                - connection_string (required): Application Insights connection string
                - batch_size (optional): Buffer size before auto-flush (default: 100)
                - service_name (optional): Service name for resource attributes (default: "elspeth")
                - service_version (optional): Service version for resource attributes
                - deployment_environment (optional): Deployment environment (e.g. "production")

        Raises:
            TelemetryExporterError: If connection_string is missing, wrong types provided,
                or Azure Monitor packages are not installed
        """
        if "connection_string" not in config:
            raise TelemetryExporterError(
                self._name,
                "Azure Monitor exporter requires 'connection_string' in config",
            )

        # Validate connection_string type
        connection_string = config["connection_string"]
        if not isinstance(connection_string, str):
            raise TelemetryExporterError(
                self._name,
                f"'connection_string' must be a string, got {type(connection_string).__name__}",
            )
        self._connection_string = connection_string

        # Validate batch_size type and value
        batch_size = config.get("batch_size", 100)
        if not isinstance(batch_size, int):
            raise TelemetryExporterError(
                self._name,
                f"'batch_size' must be an integer, got {type(batch_size).__name__}",
            )
        if batch_size < 1:
            raise TelemetryExporterError(
                self._name,
                f"batch_size must be >= 1, got {batch_size}",
            )
        self._batch_size = batch_size

        # Validate and extract service_name
        service_name = config.get("service_name", "elspeth")
        if not isinstance(service_name, str):
            raise TelemetryExporterError(
                self._name,
                f"'service_name' must be a string, got {type(service_name).__name__}",
            )
        self._service_name = service_name

        # Validate and extract service_version (optional)
        service_version = config.get("service_version")
        if service_version is not None and not isinstance(service_version, str):
            raise TelemetryExporterError(
                self._name,
                f"'service_version' must be a string or null, got {type(service_version).__name__}",
            )
        self._service_version = service_version

        # Validate and extract deployment_environment (optional)
        deployment_environment = config.get("deployment_environment")
        if deployment_environment is not None and not isinstance(deployment_environment, str):
            raise TelemetryExporterError(
                self._name,
                f"'deployment_environment' must be a string or null, got {type(deployment_environment).__name__}",
            )
        self._deployment_environment = deployment_environment

        # Import and initialize the Azure Monitor exporter
        try:
            from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
            from opentelemetry.sdk.resources import SERVICE_NAME, Resource
            from opentelemetry.sdk.trace import TracerProvider

            # Build resource attributes for proper service identification in App Insights
            # This fixes the ProxyTracerProvider issue: without an explicit TracerProvider,
            # the Azure SDK calls get_tracer_provider() which returns a ProxyTracerProvider
            # that has no .resource attribute, causing AttributeError during export.
            resource_attributes: dict[str, str] = {
                SERVICE_NAME: self._service_name,
            }
            if self._service_version:
                resource_attributes["service.version"] = self._service_version
            if self._deployment_environment:
                resource_attributes["deployment.environment"] = self._deployment_environment

            # Create and store resource for both TracerProvider and span creation
            self._resource = Resource.create(resource_attributes)
            tracer_provider = TracerProvider(resource=self._resource)

            # Pass our TracerProvider to the exporter so it doesn't fall back to
            # get_tracer_provider() which would return ProxyTracerProvider
            self._azure_exporter = AzureMonitorTraceExporter(
                connection_string=self._connection_string,
                tracer_provider=tracer_provider,
            )
        except ImportError as e:
            raise TelemetryExporterError(
                self._name,
                f"Azure Monitor exporter not installed: {e}. Install with: uv pip install azure-monitor-opentelemetry-exporter",
            ) from e

        self._configured = True
        self._buffer = []

        logger.debug(
            "Azure Monitor exporter configured",
            batch_size=self._batch_size,
            service_name=self._service_name,
            service_version=self._service_version,
            deployment_environment=self._deployment_environment,
        )

    def export(self, event: TelemetryEvent) -> None:
        """Export a single telemetry event.

        Events are buffered until batch_size is reached, then flushed.
        This method MUST NOT raise exceptions - telemetry failures should
        not crash the pipeline.

        Args:
            event: The telemetry event to export
        """
        if not self._configured:
            logger.warning(
                "Azure Monitor exporter not configured, dropping event",
                event_type=type(event).__name__,
            )
            return

        try:
            self._buffer.append(event)
            if len(self._buffer) >= self._batch_size:
                self._flush_batch()
        except Exception as e:
            # Export MUST NOT raise - log and continue
            logger.warning(
                "Failed to buffer telemetry event",
                exporter=self._name,
                event_type=type(event).__name__,
                error=str(e),
            )

    def _flush_batch(self) -> None:
        """Convert buffered events to spans and export to Azure Monitor.

        Called internally when buffer reaches batch_size, and
        externally via flush().
        """
        if not self._buffer:
            return

        if not self._azure_exporter:
            logger.warning("Azure Monitor exporter not initialized, dropping batch")
            self._buffer.clear()
            return

        try:
            spans = [self._event_to_span(e) for e in self._buffer]
            self._azure_exporter.export(spans)
            logger.debug(
                "Azure Monitor batch exported",
                span_count=len(spans),
            )
        except Exception as e:
            logger.warning(
                "Failed to export Azure Monitor batch",
                exporter=self._name,
                span_count=len(self._buffer),
                error=str(e),
            )
        finally:
            self._buffer.clear()

    def _event_to_span(self, event: TelemetryEvent) -> _SyntheticReadableSpan:
        """Convert TelemetryEvent to OpenTelemetry ReadableSpan.

        Reuses the OTLP exporter's _SyntheticReadableSpan for consistency.
        Adds Azure-specific attributes for better filtering in Application Insights.

        Args:
            event: The telemetry event to convert

        Returns:
            ReadableSpan-compatible object suitable for Azure Monitor export
        """
        from opentelemetry.trace import SpanContext, SpanKind, TraceFlags

        # Derive IDs using shared functions from OTLP exporter
        trace_id = _derive_trace_id(event.run_id)
        span_id = _derive_span_id(event)

        # Convert timestamp to nanoseconds since epoch
        if event.timestamp.tzinfo is None:
            ts = event.timestamp.replace(tzinfo=UTC)
        else:
            ts = event.timestamp
        timestamp_ns = int(ts.timestamp() * 1_000_000_000)

        # Build attributes from event fields with Azure-specific additions
        attributes = self._serialize_event_attributes(event)

        # Add Azure-specific attributes for filtering in Application Insights
        attributes["cloud.provider"] = "azure"
        attributes["elspeth.exporter"] = "azure_monitor"

        # Create span context
        span_context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )

        # Create a ReadableSpan with resource attributes for proper service identification
        span = _SyntheticReadableSpan(
            name=type(event).__name__,
            context=span_context,
            attributes=attributes,
            start_time=timestamp_ns,
            end_time=timestamp_ns,  # Instant span
            kind=SpanKind.INTERNAL,
            resource=self._resource,  # Pass resource for service.name, etc.
        )

        return span

    def _serialize_event_attributes(self, event: TelemetryEvent) -> dict[str, Any]:
        """Serialize event fields as span attributes.

        Handles type conversions for OpenTelemetry compatibility:
        - datetime -> ISO 8601 string
        - Enum -> value
        - dict -> JSON string (Azure Monitor doesn't support nested attributes)
        - tuple -> list

        Args:
            event: The telemetry event

        Returns:
            Dictionary of attribute key-value pairs
        """
        import json
        from dataclasses import asdict
        from datetime import datetime
        from enum import Enum

        data = asdict(event)
        data["event_type"] = type(event).__name__

        result: dict[str, Any] = {}
        for key, value in data.items():
            if value is None:
                continue  # Skip None values
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, dict):
                # Azure Monitor doesn't support nested attributes, serialize as JSON
                result[key] = json.dumps(value)
            elif isinstance(value, tuple):
                result[key] = list(value)
            else:
                result[key] = value

        return result

    def flush(self) -> None:
        """Flush any buffered events to Azure Monitor.

        Called periodically and at pipeline shutdown to ensure events
        are delivered.
        """
        try:
            self._flush_batch()
        except Exception as e:
            logger.warning(
                "Failed to flush Azure Monitor exporter",
                exporter=self._name,
                error=str(e),
            )

    def close(self) -> None:
        """Release resources held by the exporter.

        Flushes any remaining buffered events and shuts down the
        underlying Azure Monitor exporter. Idempotent - safe to call
        multiple times.
        """
        self.flush()
        if self._azure_exporter:
            try:
                self._azure_exporter.shutdown()
            except Exception as e:
                logger.warning(
                    "Failed to shutdown Azure Monitor exporter",
                    exporter=self._name,
                    error=str(e),
                )
            self._azure_exporter = None
        self._configured = False
