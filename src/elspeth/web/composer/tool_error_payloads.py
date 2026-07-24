"""Tool-error payload contracts shared by composer dispatch paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from elspeth.web.composer.audit import canonicalize_pydantic_cause
from elspeth.web.composer.protocol import ToolArgumentError

INVALID_TOOL_ARGUMENTS_REDACTION_STATUS: Final[str] = "invalid_tool_arguments"
UNKNOWN_TOOL_REDACTION_STATUS: Final[str] = "unknown_tool"

if TYPE_CHECKING:
    from elspeth.web.composer.redaction_telemetry import RedactionTelemetry


def unknown_tool_arguments_redaction(*, telemetry: RedactionTelemetry) -> Mapping[str, Any]:
    """Return the fixed value-free argument shape for an unknown tool."""
    telemetry.unknown_tool_redacted()
    return {"_redaction_status": UNKNOWN_TOOL_REDACTION_STATUS}


def unknown_tool_response_redaction() -> Mapping[str, Any]:
    """Preserve the semantic failure without retaining an unredactable payload."""
    return {
        "_redaction_status": UNKNOWN_TOOL_REDACTION_STATUS,
        "success": False,
        "data": {"error": "Unknown tool"},
    }


def arg_error_payload(exc: ToolArgumentError, tool_name: str) -> Mapping[str, Any]:
    """Build the structured payload for an ARG_ERROR audit record and LLM tool message."""
    safe_message = exc.args[0] if exc.args else "tool argument error"
    payload: dict[str, Any] = {"error": f"Tool '{tool_name}' failed: {safe_message}"}
    validation_errors = canonicalize_pydantic_cause(exc.__cause__)
    if validation_errors is not None:
        payload["validation_errors"] = validation_errors
    return payload
