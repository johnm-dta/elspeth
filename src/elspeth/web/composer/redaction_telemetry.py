"""OTel surface for the redaction walker (spec §4.2.4).

Rev-2 M_telemetry_implementation: uses module-level create_counter() objects
and .add() calls per the project's established pattern (service.py:135, 148,
824, 868, 1172, 1182). No _increment_counter helper — that does not exist.
"""

from __future__ import annotations

from typing import Protocol

from opentelemetry import metrics

_meter = metrics.get_meter(__name__)

_UNKNOWN_RESPONSE_KEY_COUNTER = _meter.create_counter(
    "composer.redaction.unknown_response_key_redacted",
    description="Count of unknown response keys substituted with the fixed sentinel.",
)

_MANIFEST_DISPATCH_COUNTER = _meter.create_counter(
    "composer.redaction.manifest_dispatch",
    description="Count of tool calls dispatched through the redaction manifest.",
)

_SUMMARIZER_ERROR_COUNTER = _meter.create_counter(
    "composer.redaction.summarizer_errors_total",
    description="Count of summarizer failures (exception OR non-string return) immediately before AuditIntegrityError raise.",
)


class RedactionTelemetry(Protocol):
    def unknown_response_key_redacted(self, *, tool_name: str) -> None: ...
    def manifest_dispatch(self, *, tool_name: str, shape: str) -> None: ...
    def summarizer_error(self, *, tool_name: str) -> None:
        """Incremented immediately before AuditIntegrityError raise on
        summarizer exception or non-str return. Wired in Task 7's walker code
        path. (Rev-2 M_telemetry_implementation / M.8)"""
        ...


class NoopRedactionTelemetry:
    """In-memory test impl. Records every call; assertable."""

    def __init__(self) -> None:
        self.unknown_response_key_calls: list[dict[str, str]] = []
        self.manifest_dispatch_calls: list[dict[str, str]] = []
        self.summarizer_error_calls: list[dict[str, str]] = []

    def unknown_response_key_redacted(self, *, tool_name: str) -> None:
        self.unknown_response_key_calls.append({"tool_name": tool_name})

    def manifest_dispatch(self, *, tool_name: str, shape: str) -> None:
        self.manifest_dispatch_calls.append({"tool_name": tool_name, "shape": shape})

    def summarizer_error(self, *, tool_name: str) -> None:
        self.summarizer_error_calls.append({"tool_name": tool_name})


class OtelRedactionTelemetry:
    """Production impl. Emits via module-level OTel counter objects."""

    def unknown_response_key_redacted(self, *, tool_name: str) -> None:
        _UNKNOWN_RESPONSE_KEY_COUNTER.add(1, {"tool_name": tool_name})

    def manifest_dispatch(self, *, tool_name: str, shape: str) -> None:
        _MANIFEST_DISPATCH_COUNTER.add(1, {"tool_name": tool_name, "shape": shape})

    def summarizer_error(self, *, tool_name: str) -> None:
        _SUMMARIZER_ERROR_COUNTER.add(1, {"tool_name": tool_name})
