"""Tool-error payload contracts shared by composer dispatch paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from elspeth.web.composer.audit import canonicalize_pydantic_cause
from elspeth.web.composer.protocol import ToolArgumentError

INVALID_TOOL_ARGUMENTS_REDACTION_STATUS: Final[str] = "invalid_tool_arguments"


def arg_error_payload(exc: ToolArgumentError, tool_name: str) -> Mapping[str, Any]:
    """Build the structured payload for an ARG_ERROR audit record and LLM tool message."""
    safe_message = exc.args[0] if exc.args else "tool argument error"
    payload: dict[str, Any] = {"error": f"Tool '{tool_name}' failed: {safe_message}"}
    validation_errors = canonicalize_pydantic_cause(exc.__cause__)
    if validation_errors is not None:
        payload["validation_errors"] = validation_errors
    return payload
