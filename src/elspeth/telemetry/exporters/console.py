# src/elspeth/telemetry/exporters/console.py
"""Console exporter for telemetry events.

Writes telemetry events to stdout or stderr in JSON or human-readable format.
Primarily used for testing and local debugging.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, fields
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TextIO, TypeGuard

import structlog

from elspeth.telemetry.errors import TelemetryExporterError

if TYPE_CHECKING:
    from elspeth.contracts.events import TelemetryEvent

logger = structlog.get_logger(__name__)


def _is_valid_format(v: str) -> TypeGuard[Literal["json", "pretty"]]:
    """TypeGuard for format validation - enables mypy type narrowing."""
    return v in {"json", "pretty"}


def _is_valid_output(v: str) -> TypeGuard[Literal["stdout", "stderr"]]:
    """TypeGuard for output validation - enables mypy type narrowing."""
    return v in {"stdout", "stderr"}


class ConsoleExporter:
    """Export telemetry events to stdout/stderr for testing and debugging.

    Supports two output formats:
    - json: One JSON object per line (for machine processing)
    - pretty: Human-readable format with timestamp and event type

    Configuration options:
        format: Output format - "json" (default) or "pretty"
        output: Output stream - "stdout" (default) or "stderr"

    Example configuration:
        telemetry:
          exporters:
            - name: console
              format: pretty
              output: stderr
    """

    _name = "console"

    # Valid configuration values (kept for error messages)
    _VALID_FORMATS: frozenset[str] = frozenset({"json", "pretty"})
    _VALID_OUTPUTS: frozenset[str] = frozenset({"stdout", "stderr"})

    def __init__(self) -> None:
        """Initialize unconfigured exporter."""
        self._format: Literal["json", "pretty"] = "json"
        self._output: Literal["stdout", "stderr"] = "stdout"
        self._stream: TextIO = sys.stdout

    @property
    def name(self) -> str:
        """Exporter name for configuration reference."""
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the exporter with settings from pipeline configuration.

        Args:
            config: Exporter-specific configuration dict

        Raises:
            TelemetryExporterError: If configuration values are invalid
        """
        # Validate type and value for format
        format_value = config.get("format", "json")
        if not isinstance(format_value, str):
            raise TelemetryExporterError(
                self._name,
                f"'format' must be a string, got {type(format_value).__name__}",
            )
        if _is_valid_format(format_value):
            self._format = format_value  # TypeGuard narrows type in this branch
        else:
            raise TelemetryExporterError(
                self._name,
                f"Invalid format '{format_value}'. Must be one of: {', '.join(sorted(self._VALID_FORMATS))}",
            )

        # Validate type and value for output stream
        output_value = config.get("output", "stdout")
        if not isinstance(output_value, str):
            raise TelemetryExporterError(
                self._name,
                f"'output' must be a string, got {type(output_value).__name__}",
            )
        if _is_valid_output(output_value):
            self._output = output_value  # TypeGuard narrows type in this branch
        else:
            raise TelemetryExporterError(
                self._name,
                f"Invalid output '{output_value}'. Must be one of: {', '.join(sorted(self._VALID_OUTPUTS))}",
            )
        self._stream = sys.stdout if self._output == "stdout" else sys.stderr

        logger.debug(
            "Console exporter configured",
            format=self._format,
            output=self._output,
        )

    def export(self, event: TelemetryEvent) -> None:
        """Export a single telemetry event to the console.

        This method MUST NOT raise exceptions - telemetry failures should not
        crash the pipeline. Errors are logged internally.

        Args:
            event: The telemetry event to export
        """
        try:
            if self._format == "json":
                line = json.dumps(self._serialize_event(event))
            else:
                line = self._format_pretty(event)

            print(line, file=self._stream)
        except Exception as e:
            # Export MUST NOT raise - log and continue
            logger.warning(
                "Failed to export telemetry event",
                exporter=self._name,
                event_type=type(event).__name__,
                error=str(e),
            )

    def _serialize_event(self, event: TelemetryEvent) -> dict[str, Any]:
        """Serialize event for JSON output with proper type handling.

        Handles:
        - datetime -> ISO 8601 string
        - Enum -> value
        - Other types passed through

        Args:
            event: The telemetry event to serialize

        Returns:
            Dictionary suitable for JSON serialization
        """
        data = asdict(event)
        data["event_type"] = type(event).__name__

        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Enum):
                data[key] = value.value

        return data

    def _format_pretty(self, event: TelemetryEvent) -> str:
        """Format event in human-readable form.

        Format: [TIMESTAMP] EventType: run_id (key details)

        Args:
            event: The telemetry event to format

        Returns:
            Human-readable string representation
        """
        event_type = type(event).__name__
        timestamp_str = event.timestamp.isoformat()

        # Extract additional details based on event type
        details = self._extract_pretty_details(event)
        if details:
            return f"[{timestamp_str}] {event_type}: {event.run_id} ({details})"
        return f"[{timestamp_str}] {event_type}: {event.run_id}"

    def _extract_pretty_details(self, event: TelemetryEvent) -> str:
        """Extract key details for pretty-print format.

        Args:
            event: The telemetry event

        Returns:
            String with key details, or empty string if no extra details
        """
        # Get field names for this event type (excluding base TelemetryEvent fields)
        base_fields = {"timestamp", "run_id"}
        event_fields = {f.name for f in fields(event)} - base_fields

        if not event_fields:
            return ""

        # Format key fields as key=value pairs
        details = []
        for field_name in sorted(event_fields):
            value = getattr(event, field_name)
            if value is not None:
                if isinstance(value, Enum):
                    value = value.value
                elif isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, tuple):
                    value = list(value)
                details.append(f"{field_name}={value}")

        return ", ".join(details)

    def flush(self) -> None:
        """Flush any buffered output.

        For console output, this flushes the underlying stream to ensure
        all output is visible immediately.
        """
        try:
            self._stream.flush()
        except Exception as e:
            logger.warning(
                "Failed to flush console stream",
                exporter=self._name,
                error=str(e),
            )

    def close(self) -> None:
        """Release resources (no-op for console exporter).

        The console exporter does not own the stdout/stderr streams,
        so close() is intentionally a no-op.
        """
        pass
