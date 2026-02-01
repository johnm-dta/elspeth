# src/elspeth/telemetry/exporters/datadog.py
"""Datadog exporter for telemetry events.

Exports telemetry events to Datadog via the ddtrace library. Uses native
Datadog APM integration with full tracing features.

Unlike OTLP which converts events to spans post-hoc, ddtrace creates real
Datadog spans that are automatically batched and exported to the Datadog agent.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

from elspeth.telemetry.errors import TelemetryExporterError

if TYPE_CHECKING:
    from ddtrace._trace.tracer import Tracer

    from elspeth.contracts.events import TelemetryEvent

logger = structlog.get_logger(__name__)


class DatadogExporter:
    """Export telemetry events to Datadog via native API.

    Uses ddtrace for native Datadog integration with full APM features.
    Each telemetry event is exported as a span with appropriate tags.

    Configuration options:
        api_key: Datadog API key (optional if using local agent)
        service_name: Service name for Datadog APM (default: "elspeth")
        env: Environment tag (default: "production")
        agent_host: Datadog agent hostname (default: "localhost")
        agent_port: Datadog agent port (default: 8126)
        version: Service version tag (optional)

    Example configuration:
        telemetry:
          exporters:
            - name: datadog
              api_key: ${DD_API_KEY}  # Optional if local agent
              service_name: "elspeth-pipeline"
              env: "production"
              agent_host: "localhost"
              agent_port: 8126

    Note:
        Config values come pre-resolved by Dynaconf (${DD_API_KEY} -> actual value).
        The api_key is optional when using a local Datadog agent, which handles
        authentication itself.
    """

    _name = "datadog"

    def __init__(self) -> None:
        """Initialize unconfigured exporter."""
        self._tracer: Tracer | None = None
        self._service_name: str = "elspeth"
        self._env: str = "production"
        self._version: str | None = None
        self._configured: bool = False

    @property
    def name(self) -> str:
        """Exporter name for configuration reference."""
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the exporter with settings from pipeline configuration.

        Args:
            config: Exporter-specific configuration dict containing:
                - api_key (optional): Datadog API key (not needed with local agent)
                - service_name (optional): Service name for APM (default: "elspeth")
                - env (optional): Environment tag (default: "production")
                - agent_host (optional): Agent hostname (default: "localhost")
                - agent_port (optional): Agent port (default: 8126)
                - version (optional): Service version tag

        Raises:
            TelemetryExporterError: If ddtrace is not installed or config types are wrong
        """
        # Validate and extract service_name
        service_name = config.get("service_name", "elspeth")
        if not isinstance(service_name, str):
            raise TelemetryExporterError(
                self._name,
                f"'service_name' must be a string, got {type(service_name).__name__}",
            )
        self._service_name = service_name

        # Validate and extract env
        env = config.get("env", "production")
        if not isinstance(env, str):
            raise TelemetryExporterError(
                self._name,
                f"'env' must be a string, got {type(env).__name__}",
            )
        self._env = env

        # Validate and extract version (optional)
        version = config.get("version")
        if version is not None and not isinstance(version, str):
            raise TelemetryExporterError(
                self._name,
                f"'version' must be a string or null, got {type(version).__name__}",
            )
        self._version = version

        # Validate and extract agent_host
        agent_host = config.get("agent_host", "localhost")
        if not isinstance(agent_host, str):
            raise TelemetryExporterError(
                self._name,
                f"'agent_host' must be a string, got {type(agent_host).__name__}",
            )

        # Validate agent_port type and value
        agent_port = config.get("agent_port", 8126)
        if not isinstance(agent_port, int):
            raise TelemetryExporterError(
                self._name,
                f"'agent_port' must be an integer, got {type(agent_port).__name__}",
            )
        if agent_port < 1 or agent_port > 65535:
            raise TelemetryExporterError(
                self._name,
                f"agent_port must be a valid port number (1-65535), got {agent_port}",
            )

        # Import and configure the tracer
        # ddtrace 4.x uses environment variables for agent configuration
        try:
            import os

            from ddtrace import tracer  # type: ignore[attr-defined]

            # Set agent connection via environment variables (ddtrace 4.x API)
            # These must be set before the tracer sends any spans
            os.environ["DD_AGENT_HOST"] = agent_host
            os.environ["DD_TRACE_AGENT_PORT"] = str(agent_port)

            self._tracer = tracer
        except ImportError as e:
            raise TelemetryExporterError(
                self._name,
                f"ddtrace not installed: {e}. Install with: uv pip install ddtrace",
            ) from e

        self._configured = True

        logger.debug(
            "Datadog exporter configured",
            service_name=self._service_name,
            env=self._env,
            agent_host=agent_host,
            agent_port=agent_port,
        )

    def export(self, event: TelemetryEvent) -> None:
        """Export a single telemetry event as a Datadog span.

        Creates a span with the event class name as the operation name.
        All event fields are added as span tags.

        This method MUST NOT raise exceptions - telemetry failures should
        not crash the pipeline. Errors are logged internally.

        Args:
            event: The telemetry event to export
        """
        if not self._configured or self._tracer is None:
            logger.warning(
                "Datadog exporter not configured, dropping event",
                event_type=type(event).__name__,
            )
            return

        try:
            self._create_span_for_event(event)
        except Exception as e:
            # Export MUST NOT raise - log and continue
            logger.warning(
                "Failed to export telemetry event to Datadog",
                exporter=self._name,
                event_type=type(event).__name__,
                error=str(e),
            )

    def _create_span_for_event(self, event: TelemetryEvent) -> None:
        """Create a Datadog span for the given event.

        The span is created and immediately finished since telemetry events
        represent completed operations (points in time, not durations).

        IMPORTANT: Uses explicit timestamps from event.timestamp rather than
        letting ddtrace auto-timestamp at export time. This ensures spans
        reflect when events actually occurred, not when they were exported
        (critical for buffered/async export scenarios).

        Args:
            event: The telemetry event to convert to a span
        """
        if self._tracer is None:
            return

        event_type = type(event).__name__

        # Convert event timestamp to Unix seconds and nanoseconds
        # ddtrace finish() expects seconds, but start_ns is in nanoseconds
        event_unix_seconds = event.timestamp.timestamp()
        event_ns = int(event_unix_seconds * 1_000_000_000)

        # Create span (ddtrace 4.x removed 'start' parameter from start_span)
        span = self._tracer.start_span(
            name=event_type,
            service=self._service_name,
            resource=event_type,
        )

        # Set explicit start time from event.timestamp (ddtrace 4.x API)
        # This ensures span timing reflects when the event occurred, not export time
        span.start_ns = event_ns

        try:
            # Set standard Datadog tags
            span.set_tag("env", self._env)
            if self._version:
                span.set_tag("version", self._version)

            # Set ELSPETH-specific tags
            span.set_tag("elspeth.run_id", event.run_id)
            span.set_tag("elspeth.event_type", event_type)

            # Set all event fields as tags
            self._set_event_tags(span, event)
        finally:
            # Finish span with explicit timestamp (instant span - start == finish)
            # This ensures proper cleanup even if tag setting fails
            span.finish(finish_time=event_unix_seconds)

    def _set_event_tags(self, span: Any, event: TelemetryEvent) -> None:
        """Set event fields as span tags.

        Handles type conversions for Datadog compatibility:
        - datetime -> ISO 8601 string
        - Enum -> value
        - dict -> individual tags with dotted keys
        - tuple -> list (JSON-compatible)
        - None values are skipped

        Args:
            span: The Datadog span to add tags to
            event: The telemetry event with fields to convert
        """
        data = asdict(event)

        for key, value in data.items():
            if value is None:
                continue
            tag_key = f"elspeth.{key}"
            self._set_tag_value(span, tag_key, value)

    def _set_tag_value(self, span: Any, key: str, value: Any) -> None:
        """Set a single tag value with appropriate type conversion.

        Args:
            span: The Datadog span
            key: Tag key
            value: Tag value to convert and set
        """
        if isinstance(value, datetime):
            span.set_tag(key, value.isoformat())
        elif isinstance(value, Enum):
            span.set_tag(key, value.value)
        elif isinstance(value, dict):
            # Flatten dict to dotted keys for Datadog
            for sub_key, sub_value in value.items():
                self._set_tag_value(span, f"{key}.{sub_key}", sub_value)
        elif isinstance(value, tuple):
            # Convert tuple to list for JSON compatibility
            span.set_tag(key, list(value))
        else:
            span.set_tag(key, value)

    def flush(self) -> None:
        """Flush any buffered spans to the Datadog agent.

        The ddtrace tracer handles its own batching and flushing.
        This method triggers an immediate flush of any pending spans.
        """
        if not self._tracer:
            return

        try:
            # ddtrace tracer has a flush method that sends pending spans
            self._tracer.flush()  # type: ignore[no-untyped-call]
        except Exception as e:
            logger.warning(
                "Failed to flush Datadog exporter",
                exporter=self._name,
                error=str(e),
            )

    def close(self) -> None:
        """Release resources held by the exporter.

        Flushes any remaining buffered spans and shuts down the tracer.
        Idempotent - safe to call multiple times.
        """
        self.flush()

        if self._tracer:
            try:
                # Shutdown the tracer
                self._tracer.shutdown()  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.warning(
                    "Failed to shutdown Datadog tracer",
                    exporter=self._name,
                    error=str(e),
                )
            self._tracer = None

        self._configured = False
