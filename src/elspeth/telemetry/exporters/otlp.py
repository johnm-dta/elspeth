# src/elspeth/telemetry/exporters/otlp.py
"""OTLP exporter for telemetry events.

Exports telemetry events via OpenTelemetry Protocol (OTLP) to any compatible
backend: Jaeger, Tempo, Datadog, Honeycomb, etc.

Converts ELSPETH TelemetryEvents to OpenTelemetry Spans and ships them via gRPC.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

from elspeth import __version__ as _elspeth_version
from elspeth.telemetry.errors import TelemetryExporterError

if TYPE_CHECKING:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    from elspeth.contracts.events import TelemetryEvent

logger = structlog.get_logger(__name__)


def _derive_trace_id(run_id: str) -> int:
    """Derive a consistent 128-bit trace ID from run_id.

    All events from the same run share the same trace ID, enabling
    distributed tracing correlation within a pipeline run.

    Args:
        run_id: Pipeline run identifier

    Returns:
        128-bit integer suitable for OpenTelemetry trace_id
    """
    hash_bytes = hashlib.sha256(run_id.encode()).digest()[:16]
    return int.from_bytes(hash_bytes, byteorder="big")


def _generate_span_id() -> int:
    """Generate a random 64-bit span ID per OpenTelemetry convention.

    Uses cryptographically secure random bits, matching the standard
    OpenTelemetry SDK behavior. The previous deterministic derivation
    (hash of event type + timestamp + context fields) had collision risk
    for events without token_id/row_id at the same microsecond.

    Returns:
        Non-zero 64-bit integer suitable for OpenTelemetry span_id
    """
    span_id = secrets.randbits(64)
    # OpenTelemetry requires non-zero span_id
    return span_id if span_id != 0 else 1


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

    @property
    def name(self) -> str:
        """Exporter name for configuration reference."""
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
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

        endpoint = config["endpoint"]
        if not isinstance(endpoint, str):
            raise TelemetryExporterError(
                self._name,
                f"'endpoint' must be a string, got {type(endpoint).__name__}",
            )

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

        self._endpoint = endpoint
        self._headers = headers
        self._batch_size = batch_size

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

        logger.debug(
            "OTLP exporter configured",
            endpoint=self._endpoint,
            batch_size=self._batch_size,
            headers_count=len(self._headers),
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
                "OTLP exporter not configured, dropping event",
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
        """Convert buffered events to spans and export via OTLP.

        Called internally when buffer reaches batch_size, and
        externally via flush().
        """
        if not self._buffer:
            logger.debug(
                "OTLP flush requested with empty buffer",
                exporter=self._name,
            )
            return

        if not self._span_exporter:
            logger.warning("OTLP exporter not initialized, dropping batch")
            self._buffer.clear()
            return

        try:
            spans = [self._event_to_span(e) for e in self._buffer]
            self._span_exporter.export(spans)  # type: ignore[arg-type]  # _SyntheticReadableSpan duck-types ReadableSpan
            logger.debug(
                "OTLP batch exported",
                span_count=len(spans),
            )
        except Exception as e:
            logger.warning(
                "Failed to export OTLP batch",
                exporter=self._name,
                span_count=len(self._buffer),
                error=str(e),
            )
        finally:
            self._buffer.clear()

    def _event_to_span(self, event: TelemetryEvent) -> _SyntheticReadableSpan:
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
        from opentelemetry.trace import SpanContext, SpanKind, TraceFlags

        # Derive IDs
        trace_id = _derive_trace_id(event.run_id)
        span_id = _generate_span_id()

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
        span = _SyntheticReadableSpan(
            name=type(event).__name__,
            context=span_context,
            attributes=attributes,
            start_time=timestamp_ns,
            end_time=timestamp_ns,  # Instant span
            kind=SpanKind.INTERNAL,
        )

        return span

    def _serialize_event_attributes(self, event: TelemetryEvent) -> dict[str, Any]:
        """Serialize event fields as span attributes.

        Handles type conversions for OpenTelemetry compatibility:
        - datetime -> ISO 8601 string
        - Enum -> value
        - dict -> JSON string (OTLP doesn't support nested attributes)
        - tuple -> list

        Args:
            event: The telemetry event

        Returns:
            Dictionary of attribute key-value pairs
        """
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
                # OTLP doesn't support nested attributes, serialize as JSON
                result[key] = json.dumps(value)
            elif isinstance(value, tuple):
                result[key] = list(value)
            else:
                result[key] = value

        return result

    def flush(self) -> None:
        """Flush any buffered events to the OTLP endpoint.

        Called periodically and at pipeline shutdown to ensure events
        are delivered.
        """
        try:
            self._flush_batch()
        except Exception as e:
            logger.warning(
                "Failed to flush OTLP exporter",
                exporter=self._name,
                error=str(e),
            )

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
                logger.warning(
                    "Failed to shutdown OTLP exporter",
                    exporter=self._name,
                    error=str(e),
                )
            self._span_exporter = None
        self._configured = False


# Conditional import for proper inheritance - opentelemetry is optional
try:
    from opentelemetry.sdk.trace import ReadableSpan as _ReadableSpanBase
except ImportError:
    _ReadableSpanBase = object  # type: ignore[misc,assignment]  # fallback when opentelemetry is not installed


class _SyntheticReadableSpan(_ReadableSpanBase):
    """A ReadableSpan subclass for direct export of ELSPETH telemetry events.

    OpenTelemetry's ReadableSpan is typically created by the SDK during
    normal tracing operations. Since we're converting ELSPETH events to
    spans post-hoc, we create ReadableSpan instances directly.

    Inherits from ReadableSpan to ensure type compatibility with exporters.
    """

    def __init__(
        self,
        name: str,
        context: Any,  # SpanContext
        attributes: dict[str, Any],
        start_time: int,
        end_time: int,
        kind: Any,  # SpanKind
        resource: Any | None = None,  # Resource - optional, defaults to empty
    ) -> None:
        # Import SDK types for parent __init__
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.util.instrumentation import InstrumentationScope
        from opentelemetry.trace import Status, StatusCode

        # Use provided resource or create empty default
        span_resource = resource if resource is not None else Resource.create({})

        # Call parent __init__ with appropriate defaults
        super().__init__(
            name=name,
            context=context,
            parent=None,  # Synthetic spans have no parent
            resource=span_resource,
            attributes=attributes,
            events=(),
            links=(),
            kind=kind,
            instrumentation_scope=InstrumentationScope(
                name="elspeth.telemetry",
                version=_elspeth_version,
            ),
            status=Status(StatusCode.OK),
            start_time=start_time,
            end_time=end_time,
        )
