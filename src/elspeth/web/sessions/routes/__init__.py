"""Session API routes -- /api/sessions/* with IDOR protection.

All endpoints require authentication via Depends(get_current_user).
Session-scoped endpoints verify ownership before any business logic.
"""

from __future__ import annotations

from fastapi import APIRouter

from . import _helpers as _helpers_module
from ._helpers import (
    _COMPOSER_AUTHORING_VALIDATION_COUNTER,
    _COMPOSER_REQUEST_TERMINAL_COUNTER,
    _COMPOSER_REQUESTS_INFLIGHT,
    _COMPOSER_RUNTIME_PREFLIGHT_COUNTER,
    _RUNTIME_PREFLIGHT_FAILED,
    _capture_runtime_preflight_failure,
    _composer_chat_history,
    _composer_history_content,
    _composer_persisted_validation,
    _dispatch_guided_respond,
    _handle_convergence_error,
    _handle_plugin_crash,
    _handle_runtime_preflight_failure,
    _initial_composition_state_with_guided_session,
    _litellm_error_detail,
    _persist_tool_invocations,
    _record_composer_authoring_validation_telemetry,
    _record_composer_runtime_preflight_telemetry,
    _reject_hidden_field_submissions,
    _runtime_preflight_for_state,
    _RuntimePreflightFailed,
    _state_data_from_composer_state,
    _summarize_guided_response,
    load_run_accounting_for_settings,
    slog,
    solve_step_chat_with_auto_drop,
    step_advance,
)
from .composer import register_composer_routes
from .interpretation import register_interpretation_routes
from .messages import register_message_routes
from .runs import register_run_routes
from .sessions import register_session_routes


def create_session_router() -> APIRouter:
    """Create the session router with /api/sessions prefix."""
    router = APIRouter(prefix="/api/sessions", tags=["sessions"])
    register_session_routes(router)
    register_composer_routes(router)
    register_message_routes(router)
    register_run_routes(router)
    register_interpretation_routes(router)
    return router


__all__ = [
    "_COMPOSER_AUTHORING_VALIDATION_COUNTER",
    "_COMPOSER_REQUESTS_INFLIGHT",
    "_COMPOSER_REQUEST_TERMINAL_COUNTER",
    "_COMPOSER_RUNTIME_PREFLIGHT_COUNTER",
    "_RUNTIME_PREFLIGHT_FAILED",
    "_RuntimePreflightFailed",
    "_capture_runtime_preflight_failure",
    "_composer_chat_history",
    "_composer_history_content",
    "_composer_persisted_validation",
    "_dispatch_guided_respond",
    "_handle_convergence_error",
    "_handle_plugin_crash",
    "_handle_runtime_preflight_failure",
    "_helpers_module",
    "_initial_composition_state_with_guided_session",
    "_litellm_error_detail",
    "_persist_tool_invocations",
    "_record_composer_authoring_validation_telemetry",
    "_record_composer_runtime_preflight_telemetry",
    "_reject_hidden_field_submissions",
    "_runtime_preflight_for_state",
    "_state_data_from_composer_state",
    "_summarize_guided_response",
    "create_session_router",
    "load_run_accounting_for_settings",
    "slog",
    "solve_step_chat_with_auto_drop",
    "step_advance",
]
