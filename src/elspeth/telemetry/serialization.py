"""Shared serialization utilities for telemetry exporters.

This module provides common serialization functions used by multiple exporters
(OTLP, Azure Monitor) to convert ELSPETH TelemetryEvents into formats suitable
for OpenTelemetry-compatible backends.

Public API:
- derive_trace_id: Generate consistent trace ID from run_id
- generate_span_id: Generate random span ID per OTel convention
- serialize_event_attributes: Convert event fields to span attributes
- SyntheticReadableSpan: ReadableSpan subclass for direct export
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from elspeth import __version__ as _elspeth_version

if TYPE_CHECKING:
    from elspeth.contracts.events import TelemetryEvent


def derive_trace_id(run_id: str) -> int:
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


def generate_span_id() -> int:
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


def serialize_event_attributes(event: TelemetryEvent) -> dict[str, Any]:
    """Serialize event fields as span attributes.

    Shared by OTLP and Azure Monitor exporters. Handles type conversions
    for OpenTelemetry compatibility:
    - datetime -> ISO 8601 string
    - Enum -> value
    - dict -> JSON string (OTLP doesn't support nested attributes)
    - tuple -> list

    Args:
        event: The telemetry event

    Returns:
        Dictionary of attribute key-value pairs
    """
    data = event.to_dict()
    data["event_type"] = type(event).__name__

    result: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        elif isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, Enum):
            result[key] = value.value
        elif isinstance(value, dict):
            result[key] = json.dumps(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value

    return result


# Conditional import for proper inheritance - opentelemetry is optional
try:
    from opentelemetry.sdk.trace import ReadableSpan as _ReadableSpanBase
except ImportError:
    _ReadableSpanBase = object  # type: ignore[misc,assignment]  # fallback when opentelemetry is not installed


class SyntheticReadableSpan(_ReadableSpanBase):
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


__all__ = [
    "SyntheticReadableSpan",
    "derive_trace_id",
    "generate_span_id",
    "serialize_event_attributes",
]
