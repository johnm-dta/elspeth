"""OTLP exporter for telemetry events.

Exports telemetry events via OpenTelemetry Protocol (OTLP) to any compatible
backend: Jaeger, Tempo, Datadog, Honeycomb, etc.

Converts ELSPETH TelemetryEvents to OpenTelemetry Spans and ships them via gRPC.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC
from time import time_ns
from typing import TYPE_CHECKING, Any, TypedDict
from urllib.parse import urlsplit

import structlog
from opentelemetry.sdk.trace.export import SpanExportResult

from elspeth.telemetry.errors import TELEMETRY_TRANSPORT_ERRORS, TelemetryExporterError
from elspeth.telemetry.serialization import (
    SyntheticReadableSpan,
    derive_trace_id,
    generate_span_id,
    serialize_otlp_event_attributes,
)

if TYPE_CHECKING:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    from elspeth.contracts.events import TelemetryEvent

logger = structlog.get_logger(__name__)

_MAX_RESOURCE_IDENTITY_CHARS = 128
_MAX_BATCH_SIZE = 10_000


class OTLPDeliveryMetrics(TypedDict):
    """Bounded operational delivery facts for one exporter instance."""

    attempted: int
    delivered: int
    failed: int
    dropped: int
    pending: int
    consecutive_failures: int
    last_success_unix_nano: int | None
    lifecycle_failures: int


def _configuration_error(field: str, check: str) -> TelemetryExporterError:
    """Build a static validation error that cannot echo untrusted config."""
    return TelemetryExporterError("otlp", f"'{field}' {check}")


def _validate_endpoint(value: object) -> str:
    if type(value) is not str:
        raise _configuration_error("endpoint", "must be an HTTP(S) URL")
    if not value or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise _configuration_error("endpoint", "must be an HTTP(S) URL without control characters")
    try:
        parsed = urlsplit(value)
        # Accessing port performs urllib's range/shape validation.
        _ = parsed.port
    except ValueError as exc:
        raise _configuration_error("endpoint", "must be a well-formed HTTP(S) URL") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise _configuration_error("endpoint", "must be a credential-free HTTP(S) URL without query or fragment")
    return value


def _validate_resource_identity(field: str, value: object, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if type(value) is not str:
        raise _configuration_error(field, "must be a bounded string")
    if (
        not value.strip()
        or value != value.strip()
        or len(value) > _MAX_RESOURCE_IDENTITY_CHARS
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)
    ):
        raise _configuration_error(field, "must be a non-blank bounded string without control characters")
    return value


class OTLPExporter:
    """Export telemetry events via OpenTelemetry Protocol.

    Converts ELSPETH TelemetryEvents to OTLP spans and ships to any
    OTLP-compatible backend (Jaeger, Tempo, Datadog, Honeycomb, etc.).

    Configuration options:
        endpoint: OTLP endpoint URL (required). For gRPC, typically port 4317.
        headers: Optional dict of headers (e.g., Authorization)
        batch_size: Number of events to buffer before export (default: 100)

    Flushing occurs on batch_size threshold or explicit flush() call.

    Example configuration:
        telemetry:
          exporters:
            - name: otlp
              endpoint: http://localhost:4317
              headers:
                Authorization: Bearer ${OTEL_TOKEN}
              batch_size: 100

    Thread safety:
        Assumes single-threaded access. Buffer is not thread-safe.
    """

    _name = "otlp"

    def __init__(self) -> None:
        """Initialize unconfigured exporter."""
        self._endpoint: str | None = None
        self._headers: dict[str, str] = {}
        self._batch_size: int = 100
        self._span_exporter: OTLPSpanExporter | None = None
        self._buffer: list[TelemetryEvent] = []
        self._configured: bool = False
        self._resource: Any = None
        self._attempted = 0
        self._delivered = 0
        self._failed = 0
        self._dropped = 0
        self._consecutive_failures = 0
        self._last_success_unix_nano: int | None = None
        self._lifecycle_failures = 0

    @property
    def name(self) -> str:
        """Exporter name for configuration reference."""
        return self._name

    @property
    def resource(self) -> Any:
        """Immutable OpenTelemetry resource applied to every synthetic span."""
        return self._resource

    @property
    def delivery_metrics(self) -> OTLPDeliveryMetrics:
        """Return a copy of delivery accounting; buffered is never delivered."""
        return {
            "attempted": self._attempted,
            "delivered": self._delivered,
            "failed": self._failed,
            "dropped": self._dropped,
            "pending": len(self._buffer),
            "consecutive_failures": self._consecutive_failures,
            "last_success_unix_nano": self._last_success_unix_nano,
            "lifecycle_failures": self._lifecycle_failures,
        }

    def configure(self, config: Mapping[str, Any]) -> None:
        """Configure the exporter with settings from pipeline configuration.

        Args:
            config: Exporter-specific configuration dict containing:
                - endpoint (required): OTLP gRPC endpoint URL
                - headers (optional): Dict of header key-value pairs
                - batch_size (optional): Buffer size before auto-flush (default: 100)

        Raises:
            TelemetryExporterError: If endpoint is missing or OpenTelemetry
                packages are not installed
        """
        if "endpoint" not in config:
            raise TelemetryExporterError(
                self._name,
                "OTLP exporter requires 'endpoint' in config",
            )

        endpoint = _validate_endpoint(config["endpoint"])

        raw_headers = config.get("headers", {})
        if raw_headers is None:
            headers: dict[str, str] = {}
        elif not isinstance(raw_headers, dict):
            raise TelemetryExporterError(
                self._name,
                f"'headers' must be a dictionary or null, got {type(raw_headers).__name__}",
            )
        else:
            headers = {}
            for key, value in raw_headers.items():
                if not isinstance(key, str):
                    raise TelemetryExporterError(
                        self._name,
                        f"'headers' keys must be strings, got {type(key).__name__}",
                    )
                if not isinstance(value, str):
                    raise TelemetryExporterError(
                        self._name,
                        f"'headers[{key}]' must be a string, got {type(value).__name__}",
                    )
                headers[key] = value

        batch_size = config.get("batch_size", 100)
        if type(batch_size) is not int or not 1 <= batch_size <= _MAX_BATCH_SIZE:
            raise _configuration_error("batch_size", "must be an integer between 1 and 10000")

        service_name = _validate_resource_identity("service_name", config.get("service_name", "elspeth"), required=True)
        assert service_name is not None
        service_version = _validate_resource_identity("service_version", config.get("service_version"), required=False)
        deployment_environment = _validate_resource_identity("deployment_environment", config.get("deployment_environment"), required=False)
        cloud_provider = _validate_resource_identity("cloud_provider", config.get("cloud_provider"), required=False)

        from opentelemetry.sdk.resources import Resource

        resource_attributes = {"service.name": service_name}
        if service_version is not None:
            resource_attributes["service.version"] = service_version
        if deployment_environment is not None:
            resource_attributes["deployment.environment"] = deployment_environment
        if cloud_provider is not None:
            resource_attributes["cloud.provider"] = cloud_provider

        self._endpoint = endpoint
        self._headers = headers
        self._batch_size = batch_size
        self._resource = Resource(resource_attributes)

        # Import and initialize the OTLP exporter
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            self._span_exporter = OTLPSpanExporter(
                endpoint=self._endpoint,
                headers=tuple(self._headers.items()) if self._headers else None,
            )
        except ImportError as e:
            raise TelemetryExporterError(
                self._name,
                f"OpenTelemetry OTLP exporter not installed: {e}. Install with: uv pip install opentelemetry-exporter-otlp-proto-grpc",
            ) from e

        self._configured = True
        self._buffer = []
        self._attempted = 0
        self._delivered = 0
        self._failed = 0
        self._dropped = 0
        self._consecutive_failures = 0
        self._last_success_unix_nano = None
        self._lifecycle_failures = 0

    def export(self, event: TelemetryEvent) -> bool | None:
        """Export a single telemetry event.

        Events are buffered until batch_size is reached, then flushed.
        Handled transport failures return False so TelemetryManager can account
        for them without crashing the pipeline.

        Args:
            event: The telemetry event to export
        """
        if not self._configured:
            self._attempted += 1
            self._failed += 1
            self._dropped += 1
            self._consecutive_failures += 1
            logger.warning(
                "OTLP exporter not configured, dropping event",
                event_type=type(event).__name__,
            )
            return False

        try:
            self._buffer.append(event)
            self._attempted += 1
            if len(self._buffer) >= self._batch_size:
                return self._flush_batch()
        except Exception as e:
            if not isinstance(e, TELEMETRY_TRANSPORT_ERRORS):
                raise  # Programming error — must crash
            logger.warning(
                "Failed to buffer telemetry event",
                exporter=self._name,
                event_type=type(event).__name__,
                error_type=type(e).__name__,
            )
            return False
        return None

    def _flush_batch(self) -> bool | None:
        """Convert buffered events to spans and export via OTLP.

        Called internally when buffer reaches batch_size, and
        externally via flush(). Returns False when a handled transport failure
        prevents delivery.
        """
        if not self._buffer:
            logger.debug(
                "OTLP flush requested with empty buffer",
                exporter=self._name,
            )
            return None

        if not self._span_exporter:
            batch_count = len(self._buffer)
            logger.warning("OTLP exporter not initialized, dropping batch")
            self._failed += batch_count
            self._dropped += batch_count
            self._consecutive_failures += 1
            self._buffer.clear()
            return False

        batch_count = len(self._buffer)
        try:
            spans = [self._event_to_span(e) for e in self._buffer]
            result = self._span_exporter.export(spans)
            if result == SpanExportResult.FAILURE:
                self._failed += batch_count
                self._dropped += batch_count
                self._consecutive_failures += 1
                logger.warning(
                    "OTLP exporter reported failed status",
                    exporter=self._name,
                    span_count=len(spans),
                )
                return False
            self._delivered += batch_count
            self._consecutive_failures = 0
            self._last_success_unix_nano = time_ns()
        except Exception as e:
            if not isinstance(e, TELEMETRY_TRANSPORT_ERRORS):
                raise  # Programming error — must crash
            logger.warning(
                "Failed to export OTLP batch",
                exporter=self._name,
                span_count=len(self._buffer),
                error_type=type(e).__name__,
            )
            self._failed += batch_count
            self._dropped += batch_count
            self._consecutive_failures += 1
            return False
        finally:
            self._buffer.clear()
        return None

    def _event_to_span(self, event: TelemetryEvent) -> SyntheticReadableSpan:
        """Convert TelemetryEvent to OpenTelemetry ReadableSpan.

        Mapping:
        - span.name = event class name (e.g., "TransformCompleted")
        - span.start_time = event.timestamp (converted to nanoseconds)
        - span.end_time = start_time (instant span - events are points in time)
        - span.attributes = all event fields as attributes
        - span.trace_id = derived from run_id (consistent within run)
        - span.span_id = derived from event-specific IDs

        Args:
            event: The telemetry event to convert

        Returns:
            ReadableSpan-compatible object suitable for OTLP export
        """
        from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags

        # Derive IDs
        trace_id = derive_trace_id(event.run_id)
        span_id = generate_span_id()

        # Convert timestamp to nanoseconds since epoch
        # OpenTelemetry expects timestamps in nanoseconds
        if event.timestamp.tzinfo is None:
            # Assume UTC for naive timestamps
            ts = event.timestamp.replace(tzinfo=UTC)
        else:
            ts = event.timestamp
        timestamp_ns = int(ts.timestamp() * 1_000_000_000)

        # Build attributes from event fields
        attributes = self._serialize_event_attributes(event)
        status = Status(StatusCode.ERROR if _event_failed(event) else StatusCode.OK)

        # Create span context
        span_context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )

        # Create a ReadableSpan directly
        # Note: ReadableSpan is typically created by the SDK during tracing,
        # but we can construct one for export purposes
        span = SyntheticReadableSpan(
            name=type(event).__name__,
            context=span_context,
            attributes=attributes,
            start_time=timestamp_ns,
            end_time=timestamp_ns,  # Instant span
            kind=SpanKind.INTERNAL,
            resource=self._resource,
            status=status,
        )

        return span

    @staticmethod
    def _serialize_event_attributes(event: TelemetryEvent) -> dict[str, Any]:
        """Serialize event fields as span attributes."""
        return serialize_otlp_event_attributes(event)

    def flush(self) -> bool | None:
        """Flush any buffered events to the OTLP endpoint.

        Called periodically and at pipeline shutdown to ensure events
        are delivered. Returns False for a handled transport failure.
        """
        try:
            return self._flush_batch()
        except Exception as e:
            if not isinstance(e, TELEMETRY_TRANSPORT_ERRORS):
                raise  # Programming error — must crash
            logger.warning(
                "Failed to flush OTLP exporter",
                exporter=self._name,
                error_type=type(e).__name__,
            )
            return False

    def close(self) -> None:
        """Release resources held by the exporter.

        Flushes any remaining buffered events and shuts down the
        underlying OTLP exporter. Idempotent - safe to call multiple times.
        """
        self.flush()
        if self._span_exporter:
            try:
                self._span_exporter.shutdown()
            except Exception as e:
                if not isinstance(e, TELEMETRY_TRANSPORT_ERRORS):
                    raise  # Programming error — must crash
                logger.warning(
                    "Failed to shutdown OTLP exporter",
                    exporter=self._name,
                    error_type=type(e).__name__,
                )
                self._lifecycle_failures += 1
            self._span_exporter = None
        self._configured = False


def _event_failed(event: TelemetryEvent) -> bool:
    """Map closed result enums to OTel status without copying error text."""
    from elspeth.contracts.enums import CallStatus, RunStatus
    from elspeth.contracts.events import ExternalCallCompleted, RunFinished

    if isinstance(event, RunFinished):
        return event.status is not RunStatus.COMPLETED
    if isinstance(event, ExternalCallCompleted):
        return event.status is not CallStatus.SUCCESS
    return False
