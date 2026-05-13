"""Session API routes -- /api/sessions/* with IDOR protection.

All endpoints require authentication via Depends(get_current_user).
Session-scoped endpoints verify ownership before any business logic.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace as _replace
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from opentelemetry import metrics
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
from elspeth.contracts.composer_llm_audit import (
    ComposerChatInitiator,
    ComposerChatTurn,
    ComposerChatTurnStatus,
    ComposerLLMCall,
)
from elspeth.contracts.errors import AuditIntegrityError, FailedTurnMetadata
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.secret_scrub import scrub_text_for_audit
from elspeth.core.canonical import stable_hash
from elspeth.core.dag.models import GraphValidationError
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.infrastructure.manager import PluginNotFoundError
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.protocol import BlobQuotaExceededError, BlobServiceProtocol
from elspeth.web.catalog.protocol import CatalogService as CatalogServiceProtocol
from elspeth.web.composer import yaml_generator
from elspeth.web.composer.audit import (
    BufferingRecorder,
    audit_envelope,
    chat_turn_audit_envelope,
    llm_call_audit_envelope,
)
from elspeth.web.composer.guided.audit import (
    emit_dropped_to_freeform,
    emit_step_advanced,
    emit_turn_answered,
    emit_turn_emitted,
)
from elspeth.web.composer.guided.emitters import (
    build_initial_step_1_turn,
    build_step_1_inspect_and_confirm_turn_from_intent,
    build_step_1_schema_form_turn,
    build_step_2_5_recipe_offer_turn,
    build_step_2_multi_select_turn,
    build_step_2_schema_form_turn,
    build_step_2_single_select_turn,
    build_step_3_propose_chain_turn,
)
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import ChatRole, ChatTurn, ControlSignal, GuidedStep, TurnResponse, TurnType
from elspeth.web.composer.guided.recipe_match import match_recipe
from elspeth.web.composer.guided.state_machine import (
    GuidedSession,
    SinkIntent,
    SourceResolved,
    TerminalKind,
    TerminalReason,
    TerminalState,
    TurnRecord,
    mark_solver_exhausted,
    step_advance,
)
from elspeth.web.composer.guided.steps import (
    handle_step_1_source,
    handle_step_2_5_recipe_apply,
    handle_step_2_sink,
    handle_step_3_chain_accept,
)
from elspeth.web.composer.progress import (
    ComposerProgressEvent,
    ComposerProgressRegistry,
    ComposerProgressSink,
    ComposerProgressSnapshot,
    client_cancelled_progress_event,
    convergence_progress_event,
)
from elspeth.web.composer.protocol import (
    ComposerConvergenceError,
    ComposerPluginCrashError,
    ComposerRuntimePreflightError,
    ComposerService,
    ComposerServiceError,
)
from elspeth.web.composer.redaction import redact_source_storage_path
from elspeth.web.composer.state import CompositionState, PipelineMetadata, ValidationEntry, ValidationSummary
from elspeth.web.composer.tools import _DATA_ERROR_KEY
from elspeth.web.composer.yaml_generator import generate_yaml
from elspeth.web.execution.accounting import load_run_accounting_for_settings
from elspeth.web.execution.schemas import RunAccounting, RunStatusResponse, ValidationResult
from elspeth.web.execution.validation import validate_pipeline
from elspeth.web.middleware.rate_limit import ComposerRateLimiter, get_rate_limiter
from elspeth.web.sessions._guided_solve_chain import solve_chain_with_auto_drop
from elspeth.web.sessions._guided_step_chat import solve_step_chat_with_auto_drop
from elspeth.web.sessions.converters import state_from_record as _state_from_record
from elspeth.web.sessions.protocol import (
    AUDIT_GRADE_VIEW_QUERY_ARG_ALLOWLIST,
    SESSION_TERMINAL_RUN_STATUS_VALUES,
    ChatMessageRecord,
    ChatMessageRole,
    CompositionStateData,
    CompositionStateRecord,
    InvalidForkTargetError,
    RunRecord,
    SessionRecord,
    SessionServiceProtocol,
)
from elspeth.web.sessions.schemas import (
    ChatMessageResponse,
    ChatTurnResponse,
    CompositionStateResponse,
    CreateSessionRequest,
    ForkSessionRequest,
    ForkSessionResponse,
    GetGuidedResponse,
    GuidedChatRequest,
    GuidedChatResponse,
    GuidedRespondRequest,
    GuidedRespondResponse,
    GuidedSessionResponse,
    MessageWithStateResponse,
    RevertStateRequest,
    RunResponse,
    SendMessageRequest,
    SessionResponse,
    TerminalStateResponse,
    TurnPayloadResponse,
    TurnRecordResponse,
    ValidationEntryResponse,
)

slog = structlog.get_logger()

_REDACTED_SECRET_DETAIL = "<redacted-secret>"
_PROVIDER_DETAIL_REDACTED = "Provider detail redacted because it may contain secrets."
_MAX_PROVIDER_DETAIL_CHARS = 1_000


class _SessionComposeLockRegistry:
    """Per-session compose/recompose locks.

    Lazily creates asyncio.Lock instances under a running event loop so the
    registry can live on app.state without needing sync-time initialization in
    create_app().
    """

    def __init__(self) -> None:
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._locks_lock: asyncio.Lock | None = None

    def _ensure_locks_lock(self) -> asyncio.Lock:
        if self._locks_lock is None:
            self._locks_lock = asyncio.Lock()
        return self._locks_lock

    async def get_lock(self, session_id: str) -> asyncio.Lock:
        async with self._ensure_locks_lock():
            if session_id not in self._session_locks:
                self._session_locks[session_id] = asyncio.Lock()
            return self._session_locks[session_id]

    async def cleanup_session_lock(self, session_id: str) -> None:
        async with self._ensure_locks_lock():
            if session_id in self._session_locks:
                self._session_locks.pop(session_id)


def _get_session_compose_lock_registry(request: Request) -> _SessionComposeLockRegistry:
    """Return the app-scoped compose lock registry, creating it on first use."""
    registry = getattr(request.app.state, "session_compose_lock_registry", None)
    if registry is None:
        registry = _SessionComposeLockRegistry()
        request.app.state.session_compose_lock_registry = registry
    return registry


def _get_composer_progress_registry(request: Request) -> ComposerProgressRegistry:
    """Return the app-scoped composer progress registry."""
    return cast(ComposerProgressRegistry, request.app.state.composer_progress_registry)


def _composer_progress_sink(
    registry: ComposerProgressRegistry,
    *,
    session_id: str,
    request_id: str | None,
    user_id: str,
) -> ComposerProgressSink:
    """Bind a registry sink to one session/request/user.

    ``user_id`` flows into the registry's internal user index so the
    /_active in-flight enumeration endpoint can scope cross-session
    visibility to the authenticated principal that initiated the
    composer request.
    """

    async def _publish(event: ComposerProgressEvent) -> None:
        await registry.publish(
            session_id=session_id,
            request_id=request_id,
            user_id=user_id,
            event=event,
        )

    return _publish


async def _publish_progress(
    registry: ComposerProgressRegistry,
    *,
    session_id: str,
    request_id: str | None,
    user_id: str,
    event: ComposerProgressEvent,
) -> None:
    await registry.publish(
        session_id=session_id,
        request_id=request_id,
        user_id=user_id,
        event=event,
    )


def _session_response(session: SessionRecord) -> SessionResponse:
    """Convert a SessionRecord to a SessionResponse."""
    return SessionResponse(
        id=str(session.id),
        user_id=session.user_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        forked_from_session_id=str(session.forked_from_session_id) if session.forked_from_session_id else None,
        forked_from_message_id=str(session.forked_from_message_id) if session.forked_from_message_id else None,
    )


def _message_response(msg: ChatMessageRecord, *, include_raw_content: bool = False) -> ChatMessageResponse:
    """Convert a ChatMessageRecord to a ChatMessageResponse.

    ``include_raw_content`` opt-in surfaces the model's pre-synthesis prose
    for assistant turns intercepted by the empty-state synthesizer. Default
    False keeps the conversation channel free of audit-only data; eval
    tooling sets True via the ``?include_raw_content=true`` query param.
    """
    return ChatMessageResponse(
        id=str(msg.id),
        session_id=str(msg.session_id),
        role=msg.role,
        content=msg.content,
        raw_content=msg.raw_content if include_raw_content else None,
        tool_calls=deep_thaw(msg.tool_calls) if msg.tool_calls is not None else None,
        created_at=msg.created_at,
        composition_state_id=str(msg.composition_state_id) if msg.composition_state_id else None,
        tool_call_id=msg.tool_call_id,
        parent_assistant_id=str(msg.parent_assistant_id) if msg.parent_assistant_id else None,
        sequence_no=msg.sequence_no,
    )


def _run_accounting_integrity_http(
    message: str,
    *,
    run_id: UUID | None = None,
    landscape_run_id: str | None = None,
    landscape_run_ids: Sequence[str] | None = None,
    validation_errors: Sequence[Any] | None = None,
    error: str | None = None,
) -> HTTPException:
    """Return the structured 500 used for public run-accounting contract failures."""
    detail: dict[str, object] = {
        "code": "run_integrity_error",
        "message": message,
    }
    if run_id is not None:
        detail["run_id"] = str(run_id)
    if landscape_run_id is not None:
        detail["landscape_run_id"] = landscape_run_id
    if landscape_run_ids is not None:
        detail["landscape_run_ids"] = list(landscape_run_ids)
    if validation_errors is not None:
        detail["validation_errors"] = list(validation_errors)
    if error is not None:
        detail["error"] = error
    return HTTPException(status_code=500, detail=detail)


def _validate_run_status_accounting_for_list(run: RunRecord, accounting: RunAccounting | None) -> None:
    """Apply the canonical run status/accounting contract before list serialization."""
    try:
        RunStatusResponse(
            run_id=str(run.id),
            status=run.status,
            started_at=run.started_at,
            finished_at=run.finished_at,
            accounting=accounting,
            error=run.error,
            landscape_run_id=run.landscape_run_id,
        )
    except PydanticValidationError as exc:
        raise _run_accounting_integrity_http(
            "Session run failed internal accounting validation.",
            run_id=run.id,
            landscape_run_id=run.landscape_run_id,
            validation_errors=exc.errors(include_url=False, include_context=False, include_input=False),
        ) from exc


def _litellm_error_detail(
    error_type: str,
    exc: Exception,
    *,
    expose_provider_error: bool,
) -> dict[str, object]:
    """Build the HTTP error payload for LiteLLM failures.

    ``detail`` remains class-name-only for the stable redaction contract. When
    staging/debug mode is explicitly enabled, ``provider_detail`` carries a
    bounded, scrubbed provider message for operator triage.
    """
    detail: dict[str, object] = {
        "error_type": error_type,
        "detail": type(exc).__name__,
    }
    if not expose_provider_error:
        return detail

    raw_provider_detail = str(exc).strip()
    if raw_provider_detail:
        scrubbed = scrub_text_for_audit(raw_provider_detail).strip()
        detail["provider_detail"] = (
            _PROVIDER_DETAIL_REDACTED if scrubbed == _REDACTED_SECRET_DETAIL else scrubbed[:_MAX_PROVIDER_DETAIL_CHARS]
        )

    status_code = getattr(exc, "status_code", None)
    if type(status_code) is int:
        detail["provider_status_code"] = status_code
    return detail


def _state_response(
    state: CompositionStateRecord,
    live_validation: ValidationSummary | None = None,
) -> CompositionStateResponse:
    """Convert a CompositionStateRecord to a CompositionStateResponse.

    Unfreezes container fields (MappingProxyType, tuple) into plain
    dicts/lists so redact_source_storage_path can return redacted copies.

    When live_validation is provided (from a just-computed validate() call),
    transient warnings and suggestions are included in the response.
    Historical loads pass None, producing null for these fields.
    """
    # B4: Redact internal storage paths from blob-backed sources
    source_data = deep_thaw(state.source)
    if source_data is not None:
        redacted = redact_source_storage_path({"source": source_data})
        source_data = redacted.get("source", source_data)

    return CompositionStateResponse(
        id=str(state.id),
        session_id=str(state.session_id),
        version=state.version,
        source=source_data,
        nodes=deep_thaw(state.nodes),
        edges=deep_thaw(state.edges),
        outputs=deep_thaw(state.outputs),
        metadata=deep_thaw(state.metadata_),
        is_valid=state.is_valid,
        validation_errors=deep_thaw(state.validation_errors),
        validation_warnings=[
            ValidationEntryResponse(component=e.component, message=e.message, severity=e.severity) for e in live_validation.warnings
        ]
        if live_validation is not None
        else None,
        validation_suggestions=[
            ValidationEntryResponse(component=e.component, message=e.message, severity=e.severity) for e in live_validation.suggestions
        ]
        if live_validation is not None
        else None,
        derived_from_state_id=str(state.derived_from_state_id) if state.derived_from_state_id is not None else None,
        created_at=state.created_at,
        composer_meta=deep_thaw(state.composer_meta) if state.composer_meta is not None else None,
    )


_PreflightExceptionPolicy = Literal["raise", "persist_invalid"]
_ComposerPreflightTelemetryResult = Literal["passed", "failed", "exception"]
_ComposerPreflightTelemetrySource = Literal[
    "compose",
    "recompose",
    "convergence",
    "plugin_crash",
    "runtime_preflight",
    "yaml_export",
    # Path-1 cached re-raise — composer.compose() re-raised a previously-
    # cached runtime-preflight failure via web/composer/service.py:
    # _raise_cached_runtime_preflight_failure. Distinct from "compose" to
    # let operators separate first-time preflight failures (source=compose,
    # emitted by _state_data_from_composer_state's raise arm) from
    # cache-hit re-raises (source=cached_preflight, emitted at the route
    # outer catch). See elspeth-0891e8da73.
    "cached_preflight",
]


@dataclass(frozen=True, slots=True)
class _RuntimePreflightFailed:
    """Sentinel for internal preflight failure during composer state persistence.

    Carries structured diagnostic fields so the persisted state's
    ``validation_errors`` row attributes the failure (exception class,
    first line of the message, ``file:line:function`` frames) instead of
    the legacy opaque ``["runtime_preflight_failed"]`` sentinel — which
    forced operators to ``/state/revert`` because the audit row had
    nowhere to record triage data (slog deliberately omits ``exc_info``
    to avoid leaking secret-bearing ``__cause__`` chains; the OTel
    counter only carries the bounded exception_class bucket).

    Frame strings are ``file:line:function`` only — never source-line
    text or local-variable repr — so secret-bearing values that may
    live in plugin config dicts, DB connection strings, or bound SQL
    parameters do not enter the audit row. See
    :func:`_safe_frame_strings` for the capture rule.

    The fields are optional (with empty defaults) so the ``with no
    diagnostics captured`` legacy path — preserved for the
    ``_RUNTIME_PREFLIGHT_FAILED`` module-level constant referenced by
    older tests — continues to produce the bare sentinel through
    :func:`_composer_persisted_validation`.
    """

    exception_class: str | None = None
    exception_message_first_line: str = ""
    frames: tuple[str, ...] = ()


_RUNTIME_PREFLIGHT_FAILED = _RuntimePreflightFailed()
_RuntimePreflightOutcome = ValidationResult | _RuntimePreflightFailed | None

# Bound on diagnostic detail captured for runtime-preflight failures. Keeps
# the audit row, structured response body, and slog event size predictable
# regardless of how deep the failing call stack is.
_RUNTIME_PREFLIGHT_FRAME_LIMIT = 6
_RUNTIME_PREFLIGHT_MESSAGE_LIMIT = 240


def _first_message_line(text: str) -> str:
    """Return the first non-empty line of an exception's str(), trimmed.

    ``str(exc)`` may contain newlines, embedded SQL fragments, or
    Pydantic-style multi-line dumps. The audit row only retains the
    leading message line so a bounded, scannable signal lands in
    ``validation_errors`` even when the underlying exception's repr is
    sprawling.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _safe_frame_strings(
    exc: BaseException,
    *,
    max_frames: int = _RUNTIME_PREFLIGHT_FRAME_LIMIT,
) -> tuple[str, ...]:
    """Format ``exc.__traceback__`` (and ``__cause__`` chain) frames.

    Returns a bounded tuple of ``frame=<file>:<line>:<function>`` strings
    drawn from the live traceback, walking the ``__cause__`` chain so a
    ``raise ... from ...`` boundary inside the failing helper does not
    cut diagnostics short.

    Includes only file path, line number, and function name. **Never**
    captures source-line text, local-variable repr, or exception values
    that ``traceback.format_exception`` would render — secret-bearing
    plugin config values, DB connection strings, and bound SQL
    parameters can flow through those surfaces and the audit row /
    structured server log must not retain them. File paths reference
    the on-disk source tree, which the operator already has access to
    under ELSPETH's single-operator deployment model (see the
    ``_compose_preflight_failure_message`` comment in
    ``web/composer/service.py`` for the deployment-shape caveat).

    Frames are emitted in walked order — outermost frame of the original
    exception first, then deeper frames, then the next ``__cause__``.
    The ``max_frames`` cap is shared across the entire chain so the
    audit row size stays predictable regardless of stack depth.
    """
    frames: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and len(frames) < max_frames:
        if id(current) in seen:
            break
        seen.add(id(current))
        tb = current.__traceback__
        while tb is not None and len(frames) < max_frames:
            f = tb.tb_frame
            frames.append(f"frame={f.f_code.co_filename}:{tb.tb_lineno}:{f.f_code.co_name}")
            tb = tb.tb_next
        current = current.__cause__
    return tuple(frames)


def _runtime_preflight_failure_errors(
    exception_class: str,
    exception_message_first_line: str,
    frames: Sequence[str],
) -> list[str]:
    """Build the structured ``validation_errors`` list for a runtime-preflight crash.

    Replaces the legacy opaque ``["runtime_preflight_failed"]`` sentinel
    with a self-describing list. The first entry remains
    ``"runtime_preflight_failed"`` so existing parsers/UIs that key on
    the sentinel continue to work; subsequent entries are advisory and
    may be empty when frame capture is impossible (e.g. when only
    ``exception_class`` is available without a live traceback).

    Schema is preserved: each entry is a string, so
    :class:`CompositionStateData.validation_errors` (typed
    ``Sequence[str]``) does not need a migration. Operators / the LLM
    parsing the audit row receive ``"key=value"`` shaped strings.

    Bounded length: ``exception_message_first_line`` is clipped to
    :data:`_RUNTIME_PREFLIGHT_MESSAGE_LIMIT` characters. ``frames`` is
    limited at the caller (see :func:`_safe_frame_strings`).
    """
    truncated_msg = exception_message_first_line[:_RUNTIME_PREFLIGHT_MESSAGE_LIMIT]
    return [
        "runtime_preflight_failed",
        f"exception_class={exception_class}",
        f"exception_message={truncated_msg}",
        *frames,
    ]


def _capture_runtime_preflight_failure(exc: BaseException) -> _RuntimePreflightFailed:
    """Build a :class:`_RuntimePreflightFailed` with structured diagnostics.

    Single source of truth for the capture: any new raise site that
    converts an unexpected runtime-preflight exception to a
    persist-invalid sentinel must use this helper so the audit-row
    attribution is uniform.
    """
    return _RuntimePreflightFailed(
        exception_class=type(exc).__name__,
        exception_message_first_line=_first_message_line(str(exc)),
        frames=_safe_frame_strings(exc),
    )


_COMPOSER_RUNTIME_PREFLIGHT_COUNTER = metrics.get_meter(__name__).create_counter(
    "composer.runtime_preflight.total",
    unit="1",
    description="Count of composer runtime preflight outcomes by route and result",
)
_COMPOSER_AUTHORING_VALIDATION_COUNTER = metrics.get_meter(__name__).create_counter(
    "composer.authoring_validation.total",
    unit="1",
    description="Count of composer authoring-state validation outcomes by route and result",
)

# Composer-request lifecycle telemetry — answers the operator question
# "what's running on this server right now?" without requiring a per-session
# /composer-progress poll. UpDownCounter is a gauge: incremented when a
# request enters the compose-engaged window (after rate limit + ownership
# check + initial user-message persistence) and decremented in a finally
# clause regardless of outcome. The terminal counter records the outcome
# distribution, with one terminal event per request matching the in-flight
# decrement. session_id is intentionally NOT a metric attribute (cardinality
# explosion) — per-session attribution lives on the progress snapshot.
_ComposerRequestEndpoint = Literal["send_message", "recompose"]
_ComposerRequestTerminalStatus = Literal["completed", "failed", "timed_out", "cancelled"]
_COMPOSER_REQUESTS_INFLIGHT = metrics.get_meter(__name__).create_up_down_counter(
    "composer.requests.inflight",
    unit="1",
    description="Current count of in-flight composer message requests by endpoint",
)
_COMPOSER_REQUEST_TERMINAL_COUNTER = metrics.get_meter(__name__).create_counter(
    "composer.request.terminal.total",
    unit="1",
    description="Count of completed composer message requests by endpoint and terminal status",
)

# Audit-primacy counters for the route-side persistence helpers
# (``_persist_tool_invocations``, ``_persist_llm_calls``). These mirror the
# SessionServiceImpl-side counters in ``web/sessions/telemetry.py`` by
# OTel metric name so dashboards aggregate; the route helpers use module-
# level access (consistent with the other route counters above) rather
# than threading the SessionsTelemetry container through every helper.
# The ``helper`` attribute discriminates which row family failed
# (``tool_invocations`` vs ``llm_calls``).
_COMPOSER_TIER1_VIOLATION_COUNTER = metrics.get_meter(__name__).create_counter(
    "composer.audit.tool_row_tier1_violation_total",
    unit="1",
    description=(
        "Count of Tier-1 audit-row persist failures on the success path "
        "(assistant row already written; tool/LLM-call audit row failed). "
        "Each increment is paired with an AuditIntegrityError raise."
    ),
)
_COMPOSER_PERSIST_FAILED_DURING_UNWIND_COUNTER = metrics.get_meter(__name__).create_counter(
    "composer.audit.tool_row_persist_failed_during_unwind_total",
    unit="1",
    description=(
        "Count of audit-row persist failures on the unwind path (a primary "
        "failure was already in flight; this row failure is recorded but "
        "does not raise so it cannot mask the primary exception)."
    ),
)


def _record_composer_request_terminal(
    status: _ComposerRequestTerminalStatus,
    *,
    endpoint: _ComposerRequestEndpoint,
) -> None:
    _COMPOSER_REQUEST_TERMINAL_COUNTER.add(1, {"endpoint": endpoint, "status": status})


_COMPOSER_EXCEPTION_CLASS_BUCKETS = frozenset(
    {
        "AttributeError",
        "ComposerRuntimePreflightError",
        "FileNotFoundError",
        "GraphValidationError",
        "ImportError",
        "OSError",
        "PermissionError",
        "PluginConfigError",
        "PluginNotFoundError",
        "RuntimeError",
        "TimeoutError",
        "TypeError",
        "ValueError",
    }
)
_OTHER_COMPOSER_EXCEPTION_CLASS = "other"


def _bounded_composer_exception_class(exception_class: str | None) -> str | None:
    if exception_class is None:
        return None
    if exception_class in _COMPOSER_EXCEPTION_CLASS_BUCKETS:
        return exception_class
    return _OTHER_COMPOSER_EXCEPTION_CLASS


def _record_composer_runtime_preflight_telemetry(
    result: _ComposerPreflightTelemetryResult,
    *,
    source: _ComposerPreflightTelemetrySource,
    exception_class: str | None = None,
) -> None:
    attrs: dict[str, str] = {"result": result, "source": source}
    bounded_exception_class = _bounded_composer_exception_class(exception_class)
    if bounded_exception_class is not None:
        attrs["exception_class"] = bounded_exception_class
    _COMPOSER_RUNTIME_PREFLIGHT_COUNTER.add(1, attrs)


def _record_composer_authoring_validation_telemetry(
    result: _ComposerPreflightTelemetryResult,
    *,
    source: _ComposerPreflightTelemetrySource,
    exception_class: str | None = None,
) -> None:
    attrs: dict[str, str] = {"result": result, "source": source}
    bounded_exception_class = _bounded_composer_exception_class(exception_class)
    if bounded_exception_class is not None:
        attrs["exception_class"] = bounded_exception_class
    _COMPOSER_AUTHORING_VALIDATION_COUNTER.add(1, attrs)


def _composer_history_content(message: ChatMessageRecord) -> str:
    """Return the content sent back to the composer LLM for a stored message.

    All composer synthesis shapes are augmentations (see
    service._finalize_no_tool_response). The model's prose is preserved
    verbatim with an operator-facing suffix appended; ``content`` starts
    with ``raw_content``. The LLM should see its own prose unmodified
    on subsequent turns; the operator-facing suffix stays out of the
    prompt history.

    Equality (``content == raw_content``) is augmentation with an empty
    suffix and returns ``raw_content`` unchanged. Empty ``raw_content``
    (the no-mutation, preflight-invalid empty-state, and
    preflight-invalid non-empty-state augmentation branches in
    ``service._finalize_no_tool_response`` when the model emitted no
    prose) is augmentation by the structural rule: the model produced
    no prose and the operator-facing suffix replaces nothing — so the
    LLM sees an empty prior turn and the operator-facing suffix stays
    out of the prompt history.

    The structural rule's correctness depends on the producer never
    emitting a non-augmentation shape (``content`` not starting with
    ``raw_content``). This is mechanically enforced at construction by
    ``service._enforce_augmentation_prefix_invariant``. A row reaching
    this discriminator that violates the contract is an audit-integrity
    violation and crashes here rather than silently misroute. The
    field-level overloading that makes the structural discriminator
    necessary at all is tracked as architectural debt at
    ``elspeth-7ae1732ab2`` (introduce a producer-side discriminator
    field on the record).
    """
    if message.role == "assistant" and message.raw_content is not None:
        if not message.content.startswith(message.raw_content):
            raise AuditIntegrityError(
                "Tier 1: composer chat row violates augmentation prefix "
                "invariant on the read path. content does not start with "
                "raw_content. All synthesis shapes are augmentations "
                "post-elspeth-9cfbad6901; a row reaching this discriminator "
                "that breaks the contract is an audit-integrity violation. "
                "Fix the producer (service._finalize_no_tool_response) "
                "or the row (drop the audit DB if the row predates the "
                "augmentation-only migration; per project_db_migration_policy, "
                "operator deletes the old DB on schema-shape changes)."
            )
        # Augmentation (incl. equality and empty raw_content): emit
        # unmodified model prose. The operator-facing suffix stays
        # out of LLM history.
        return message.raw_content
    return message.content


def _is_composer_audit_tool_message(message: ChatMessageRecord) -> bool:
    """Return true when a persisted chat row is an audit-only composer row.

    Rev-4: ``role="audit"`` rows are unconditionally audit-only — they exist
    precisely because there is no parent assistant to carry a real
    ``role="tool"`` row. The legacy ``role="tool"`` audit-envelope path is
    preserved for the dispatch trail of in-loop tool calls produced by the
    compose loop; those rows are excluded from prompt history because
    replaying them without the assistant's tool-call request would create
    orphan OpenAI tool messages.
    """
    if message.role == "audit":
        return True
    if message.role != "tool":
        return False
    if message.tool_calls is None:
        return True
    for tool_call in message.tool_calls:
        if "_kind" not in tool_call:
            continue
        kind = tool_call["_kind"]
        if kind in {"audit", "llm_call_audit"}:
            return True
        if kind is not None:
            return True
    return True


def _is_composer_llm_audit_tool_message(message: ChatMessageRecord) -> bool:
    """Return true only for persisted composer LLM-call audit sidecars.

    Rev-4: LLM-call audit rows are persisted with ``role="audit"`` (they
    have no real OpenAI tool-call identity, so they cannot be ``role="tool"``
    after the parent-CHECK biconditional landed in Task 1).
    """
    if message.role != "audit" or message.tool_calls is None:
        return False
    return any("_kind" in tool_call and tool_call["_kind"] == "llm_call_audit" for tool_call in message.tool_calls)


def _composer_conversation_messages(messages: Sequence[ChatMessageRecord]) -> list[ChatMessageRecord]:
    """Return persisted messages that are part of the LLM conversation."""
    return [message for message in messages if not _is_composer_audit_tool_message(message)]


def _composer_conversation_or_llm_audit_messages(messages: Sequence[ChatMessageRecord]) -> list[ChatMessageRecord]:
    """Return user-visible conversation plus safe per-LLM-call audit sidecars."""
    return [message for message in messages if not _is_composer_audit_tool_message(message) or _is_composer_llm_audit_tool_message(message)]


def _composer_conversation_or_tool_messages(messages: Sequence[ChatMessageRecord]) -> list[ChatMessageRecord]:
    """Return user-visible conversation plus real assistant-linked tool rows."""

    return [message for message in messages if message.role == "tool" or not _is_composer_audit_tool_message(message)]


def _composer_conversation_tool_or_llm_audit_messages(messages: Sequence[ChatMessageRecord]) -> list[ChatMessageRecord]:
    """Return conversation, tool rows, and safe per-LLM-call audit sidecars."""

    return [
        message
        for message in messages
        if message.role == "tool" or not _is_composer_audit_tool_message(message) or _is_composer_llm_audit_tool_message(message)
    ]


def _composer_chat_history(messages: Sequence[ChatMessageRecord]) -> list[dict[str, str]]:
    """Convert persisted session messages to LLM chat history.

    ``raw_content`` is attribution/audit data and feeds the LLM-context
    decision in ``_composer_history_content``. All assistant-synthesis
    shapes are augmentations (see ``service._finalize_no_tool_response``
    and ``_composer_history_content`` for the policy): the model's
    prose is preserved verbatim and an operator-facing suffix is
    appended. The LLM sees its own prose unmodified on subsequent
    turns; the suffix stays out of the prompt.

    Composer tool-call audit rows are persisted as ``role="tool"`` messages so
    the session record retains the dispatch trail. They are not prior LLM
    dialogue turns: replaying them without the in-loop assistant tool-call
    request that produced them creates orphan OpenAI tool messages. Keep them
    in storage, but exclude them from prompt history and normal chat responses.
    """
    return [{"role": message.role, "content": _composer_history_content(message)} for message in _composer_conversation_messages(messages)]


def _composer_persisted_validation(
    authoring: ValidationSummary,
    runtime_preflight: _RuntimePreflightOutcome,
) -> tuple[bool, list[str] | None]:
    """Return persisted validity/errors for a composer-produced state.

    When the runtime preflight crashed unexpectedly, emit a structured
    diagnostic list (sentinel + ``exception_class=...`` +
    ``exception_message=...`` + ``frame=...`` entries) so the persisted
    audit row carries the attribution the previous opaque
    ``["runtime_preflight_failed"]`` sentinel withheld. The first entry
    remains the legacy sentinel so external parsers keying on it (the
    SPA / LLM recovery loop) continue to detect the failure class.

    The bare-sentinel path is preserved for the
    :data:`_RUNTIME_PREFLIGHT_FAILED` zero-arg constant — older tests
    construct it directly and assert the legacy single-string output
    to lock in the contract that authoring-valid + opaque-runtime-fail
    persists as ``is_valid=False``.
    """
    if isinstance(runtime_preflight, _RuntimePreflightFailed):
        if runtime_preflight.exception_class is None:
            return False, ["runtime_preflight_failed"]
        return False, _runtime_preflight_failure_errors(
            runtime_preflight.exception_class,
            runtime_preflight.exception_message_first_line,
            runtime_preflight.frames,
        )
    if runtime_preflight is not None:
        messages = [error.message for error in runtime_preflight.errors]
        return runtime_preflight.is_valid, messages or None
    if authoring.is_valid:
        raise ValueError("Composer persistence for authoring-valid state requires runtime preflight outcome")
    messages = [error.message for error in authoring.errors]
    return authoring.is_valid, messages or None


async def _runtime_preflight_for_state(
    state: CompositionState,
    *,
    settings: Any,
    secret_service: Any | None,
    user_id: str | None,
) -> ValidationResult:
    return await asyncio.wait_for(
        run_sync_in_worker(
            validate_pipeline,
            state,
            settings,
            yaml_generator,
            secret_service=secret_service,
            user_id=user_id,
        ),
        timeout=settings.composer_runtime_preflight_timeout_seconds,
    )


async def _persist_tool_invocations(
    service: SessionServiceProtocol,
    session_id: UUID,
    tool_invocations: tuple[ComposerToolInvocation, ...],
    composition_state_id: UUID | None,
    *,
    parent_assistant_id: UUID | None = None,
    plugin_crash_pending: bool,
) -> None:
    """Persist per-tool-call audit records, splitting role by parent presence.

    Rev-4: when ``parent_assistant_id`` is supplied (success-path callers
    after the assistant row was persisted), the row uses ``role="tool"``
    with the OpenAI-shaped tool-call linkage. When the caller has no
    assistant row (failure / convergence / preflight paths), the row uses
    ``role="audit"`` — an internal-only role for audit breadcrumbs that
    cannot satisfy the ``ck_chat_messages_parent_role`` /
    ``ck_chat_messages_tool_call_id_role`` biconditional CHECKs.

    Each :class:`ComposerToolInvocation` still lands as one chat row whose
    ``tool_calls`` JSON column carries the audit envelope under a ``_kind``
    discriminator (see :func:`elspeth.web.composer.audit.audit_envelope`).

    ``content`` shape per status:

    - SUCCESS or ARG_ERROR with payload: ``invocation.result_canonical`` —
      this is what the LLM saw on the tool path. Auditors can replay
      verbatim.
    - PLUGIN_CRASH (no payload): synthetic JSON with ``error_class`` only.
      The LLM never saw a result on this path; the tool row records that
      a crash occurred at this position in the dispatch sequence.

    ``composition_state_id`` is the post-compose state id when this turn
    advanced state, else ``None``. All tool rows from one turn share this
    id; the per-call audit envelope's ``version_before``/``version_after``
    captures the per-call state delta.

    Audit primacy disposition (caller-driven via ``plugin_crash_pending``).
    Mirrors :meth:`SessionServiceImpl.persist_compose_turn` exactly —
    same name, same semantics:

    - ``plugin_crash_pending=False`` (success path): the assistant row
      was already written and we are about to return success. A
      SQLAlchemyError here means the assistant message exists in the
      audit trail but the tool rows that prove what the LLM saw are
      missing. That is a Tier-1 audit corruption (CLAUDE.md: "I don't
      know what happened" is never an acceptable answer). Increment the
      Tier-1 counter and raise :class:`AuditIntegrityError` chained
      through the SQLAlchemyError. The request will 500 with the chained
      cause visible to the operator.

    - ``plugin_crash_pending=True`` (unwind / recovery path): the
      caller already has a primary failure in hand
      (ConvergenceError / PluginCrashError / RuntimePreflightError /
      LLM provider error / CancelledError) and is calling this helper to
      record audit detail of what happened. Raising
      AuditIntegrityError here would mask the original failure, which is
      what the operator needs to see. Increment the
      "persist failed during unwind" counter, slog the audit-system
      failure (the slog is permitted under CLAUDE.md primacy because the
      audit system itself failed — telemetry has nowhere to write the
      structured event), and continue. The unwind disposition is
      observable via the counter increment + slog event; the partial
      tool-trail is observable on read-back via per-message ``tool_calls``
      count vs. ``ComposerResult.tool_invocations`` length.
    """
    for invocation in tool_invocations:
        if invocation.status == ComposerToolStatus.PLUGIN_CRASH:
            content = json.dumps(
                {
                    "error_class": invocation.error_class,
                    "error_message": invocation.error_message,
                }
            )
        elif invocation.result_canonical is not None:
            content = invocation.result_canonical
        else:
            # ARG_ERROR with no captured payload — fall back to error class.
            content = json.dumps(
                {
                    "error_class": invocation.error_class,
                    "error_message": invocation.error_message,
                }
            )
        role: ChatMessageRole = "tool" if parent_assistant_id is not None else "audit"
        try:
            await service.add_message(
                session_id,
                role,
                content,
                tool_calls=[audit_envelope(invocation)],
                composition_state_id=composition_state_id,
                writer_principal="compose_loop",
                tool_call_id=invocation.tool_call_id if role == "tool" else None,
                parent_assistant_id=parent_assistant_id if role == "tool" else None,
            )
        except SQLAlchemyError as save_err:
            if plugin_crash_pending:
                # Unwind path: a primary failure is already in flight.
                # Counting + slog preserves audibility without masking
                # the original exception.
                _COMPOSER_PERSIST_FAILED_DURING_UNWIND_COUNTER.add(
                    1,
                    {"helper": "tool_invocations"},
                )
                slog.error(
                    "composer_tool_invocation_persist_failed_during_unwind",
                    session_id=str(session_id),
                    tool_call_id=invocation.tool_call_id,
                    tool_name=invocation.tool_name,
                    exc_class=type(save_err).__name__,
                )
                continue
            # Success-path Tier-1 violation: the assistant row succeeded
            # but the audit-companion tool row failed. The audit trail
            # would assert "this tool was called" without the row that
            # proves what it returned. Crash with the chained cause so
            # the operator sees the full diagnostic.
            _COMPOSER_TIER1_VIOLATION_COUNTER.add(
                1,
                {"helper": "tool_invocations"},
            )
            raise AuditIntegrityError(
                f"composer_tool_invocation_persist_failed: audit insert "
                f"failed for session_id={session_id!r} after assistant row "
                f"was persisted — Tier-1 audit corruption (no recovery)"
            ) from save_err


def _llm_calls_from_exception(exc: BaseException) -> tuple[ComposerLLMCall, ...]:
    exc_dict = exc.__dict__
    if "llm_calls" not in exc_dict:
        return ()
    calls = exc_dict["llm_calls"]
    if type(calls) is not tuple:
        return ()
    return cast(tuple[ComposerLLMCall, ...], calls)


async def _persist_llm_calls(
    service: SessionServiceProtocol,
    session_id: UUID,
    llm_calls: tuple[ComposerLLMCall, ...],
    composition_state_id: UUID | None,
    *,
    plugin_crash_pending: bool,
) -> None:
    """Persist per-LLM-call audit records as audit-only ``role=audit`` rows.

    Audit-primacy disposition mirrors :func:`_persist_tool_invocations`
    via the ``plugin_crash_pending`` flag — see that helper's docstring
    for the full rationale. The shape is the same: success-path failure
    is a Tier-1 audit corruption that must crash; unwind-path failure
    is recorded via counter + slog so it cannot mask the primary error.
    """
    for call in llm_calls:
        content = json.dumps(
            {
                "_kind": "llm_call_audit",
                "status": call.status.value,
                "model_requested": call.model_requested,
                "model_returned": call.model_returned,
                "total_tokens": call.total_tokens,
                "reasoning_tokens": call.reasoning_tokens,
                "provider_cost": call.provider_cost,
            }
        )
        try:
            await service.add_message(
                session_id,
                "audit",
                content,
                tool_calls=[llm_call_audit_envelope(call)],
                composition_state_id=composition_state_id,
                writer_principal="compose_loop",
            )
        except SQLAlchemyError as save_err:
            if plugin_crash_pending:
                _COMPOSER_PERSIST_FAILED_DURING_UNWIND_COUNTER.add(
                    1,
                    {"helper": "llm_calls"},
                )
                slog.error(
                    "composer_llm_call_persist_failed_during_unwind",
                    session_id=str(session_id),
                    model_requested=call.model_requested,
                    status=call.status.value,
                    exc_class=type(save_err).__name__,
                )
                continue
            _COMPOSER_TIER1_VIOLATION_COUNTER.add(
                1,
                {"helper": "llm_calls"},
            )
            raise AuditIntegrityError(
                f"composer_llm_call_persist_failed: audit insert failed for "
                f"session_id={session_id!r} on success path — Tier-1 audit "
                f"corruption (no recovery)"
            ) from save_err


async def _persist_chat_turns(
    service: SessionServiceProtocol,
    session_id: UUID,
    chat_turns: tuple[ComposerChatTurn, ...],
    composition_state_id: UUID | None,
    *,
    request_unwinding: bool,
) -> None:
    """Persist per-chat-turn audit records as audit-only ``role=audit`` rows.

    Sibling of :func:`_persist_llm_calls`.  Each ComposerChatTurn produces
    one ``role=audit`` row tagged ``_kind=chat_turn_audit``; auditors query
    by ``json_extract(content, '$._kind')='chat_turn_audit'`` and by
    ``composition_state_id`` to scope to a particular composition snapshot.

    SQLAlchemy failures propagate on the success path: otherwise the
    guided-session ``chat_history`` state write can commit while the
    corresponding audit-only row disappears.  During exception unwinds,
    failures are logged instead so an audit cleanup problem does not mask
    the primary HTTPException.
    """
    for turn in chat_turns:
        content = json.dumps(
            {
                "_kind": "chat_turn_audit",
                "status": turn.status.value,
                "step": turn.step,
                "initiator": turn.initiator.value,
                "chat_turn_seq": turn.chat_turn_seq,
                "model": turn.model,
                "latency_ms": turn.latency_ms,
                "error_class": turn.error_class,
            }
        )
        try:
            await service.add_message(
                session_id,
                "audit",
                content,
                tool_calls=[chat_turn_audit_envelope(turn)],
                composition_state_id=composition_state_id,
                writer_principal="compose_loop",
            )
        except SQLAlchemyError as save_err:
            if request_unwinding:
                slog.error(
                    "composer_chat_turn_persist_failed_during_unwind",
                    session_id=str(session_id),
                    step=turn.step,
                    status=turn.status.value,
                    exc_class=type(save_err).__name__,
                )
                continue
            _COMPOSER_TIER1_VIOLATION_COUNTER.add(
                1,
                {"helper": "chat_turns"},
            )
            raise AuditIntegrityError(
                f"composer_chat_turn_persist_failed: audit insert failed for "
                f"session_id={session_id!r} on success path — Tier-1 audit "
                f"corruption (no recovery)"
            ) from save_err
        except Exception as save_err:
            if not request_unwinding:
                raise
            slog.error(
                "composer_chat_turn_persist_failed_during_unwind",
                session_id=str(session_id),
                step=turn.step,
                status=turn.status.value,
                exc_class=type(save_err).__name__,
            )


async def _state_data_from_composer_state(
    state: CompositionState,
    *,
    settings: Any,
    secret_service: Any | None,
    user_id: str | None,
    runtime_preflight: _RuntimePreflightOutcome,
    preflight_exception_policy: _PreflightExceptionPolicy,
    initial_version: int | None,
    telemetry_source: _ComposerPreflightTelemetrySource,
    composer_meta: Mapping[str, Any] | None = None,
) -> tuple[CompositionStateData, ValidationSummary]:
    try:
        authoring = state.validate()
    except (ValueError, TypeError, KeyError) as val_err:
        _record_composer_authoring_validation_telemetry(
            "exception",
            source=telemetry_source,
            exception_class=type(val_err).__name__,
        )
        authoring = ValidationSummary(
            is_valid=False,
            errors=(ValidationEntry("validation", "validation_failed", "high"),),
        )
    else:
        _record_composer_authoring_validation_telemetry(
            "passed" if authoring.is_valid else "failed",
            source=telemetry_source,
        )

    runtime = runtime_preflight
    if runtime is None and authoring.is_valid:
        try:
            runtime = await _runtime_preflight_for_state(
                state,
                settings=settings,
                secret_service=secret_service,
                user_id=user_id,
            )
        except Exception as exc:
            # Telemetry MUST fire on both policy branches. Emitting before
            # the policy split preserves originating-route attribution
            # (source=compose|recompose|...) on the raise path, which would
            # otherwise be lost — the recovery handler's re-call relabels its
            # own emission as source=runtime_preflight, so without this
            # primary emission the dashboards cannot distinguish failures
            # by originating route. The dual emission per failure (primary +
            # recovery) is intentional: each event represents a distinct
            # observable — primary failure occurred, then recovery handled
            # it. Operators query them with different source filters.
            _record_composer_runtime_preflight_telemetry(
                "exception",
                source=telemetry_source,
                exception_class=type(exc).__name__,
            )
            if preflight_exception_policy == "raise":
                raise ComposerRuntimePreflightError.capture(
                    exc,
                    state=state,
                    initial_version=initial_version if initial_version is not None else state.version,
                ) from exc
            # persist_invalid path: capture structured diagnostics
            # (exception class + first message line + bounded frames)
            # so the persisted state's validation_errors row is
            # self-describing instead of the opaque legacy sentinel.
            # Frames are file:line:function only — no source text, no
            # local-variable repr — so secrets in plugin config dicts
            # / DB URLs / bound SQL params do not enter the audit row.
            runtime = _capture_runtime_preflight_failure(exc)
    if isinstance(runtime, ValidationResult):
        _record_composer_runtime_preflight_telemetry(
            "passed" if runtime.is_valid else "failed",
            source=telemetry_source,
        )
    persisted_is_valid, persisted_errors = _composer_persisted_validation(
        authoring,
        runtime,
    )
    state_d = state.to_dict()
    return (
        CompositionStateData(
            source=state_d["source"],
            nodes=state_d["nodes"],
            edges=state_d["edges"],
            outputs=state_d["outputs"],
            metadata_=state_d["metadata"],
            is_valid=persisted_is_valid,
            validation_errors=persisted_errors,
            composer_meta=composer_meta,
        ),
        authoring,
    )


async def _verify_session_ownership(
    session_id: UUID,
    user: UserIdentity,
    request: Request,
) -> SessionRecord:
    """Verify the session exists and belongs to the current user.

    Returns 404 (not 403) to avoid leaking session existence (IDOR, W5).
    """
    service: SessionServiceProtocol = request.app.state.session_service
    try:
        session = await service.get_session(session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None

    settings = request.app.state.settings
    if session.user_id != user.user_id or session.auth_provider_type != settings.auth_provider:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


async def _failed_turn_response_body(
    service: SessionServiceProtocol,
    session_id: UUID,
    failed_turn: FailedTurnMetadata,
) -> dict[str, object]:
    """Return the stable response fragment for a persisted failed compose turn."""

    tool_responses_persisted = failed_turn.tool_responses_persisted
    if tool_responses_persisted is None:
        tool_responses_persisted = await service.count_tool_responses_for_assistant_async(
            session_id=str(session_id),
            assistant_message_id=failed_turn.assistant_message_id,
        )
    return {
        "assistant_message_id": failed_turn.assistant_message_id,
        "tool_calls_attempted": failed_turn.tool_calls_attempted,
        "tool_responses_persisted": tool_responses_persisted,
        "transcript_url": None,
    }


async def _handle_convergence_error(
    exc: ComposerConvergenceError,
    service: SessionServiceProtocol,
    session_id: UUID,
    user_id: str,
    log_prefix: str,
    llm_composition_state_id: UUID | None,
    settings: Any,
    secret_service: Any | None,
) -> dict[str, object]:
    """Build 422 response body and persist partial state for convergence errors.

    Shared by send_message and recompose — only the structlog event prefix
    differs between callers.

    Symmetric with :func:`_handle_plugin_crash` and
    :func:`_handle_runtime_preflight_failure` — the same recovery shape
    (``preflight_exception_policy="persist_invalid"``, partial-state
    persistence, SQLAlchemyError fail-soft) and the same signature
    placement of ``user_id`` between ``session_id`` and ``log_prefix``.

    Args:
        exc: The convergence error with optional partial_state.
        service: Session service for DB persistence.
        session_id: Session to persist partial state to.
        user_id: Authenticated user id. Forwarded to
            :func:`_state_data_from_composer_state` so the partial-state
            runtime preflight can resolve user-scoped secret refs.
            Without this, validation.py:248 skips the secret-ref
            resolution block, unresolved ``{secret_ref: ...}`` dicts
            flow into plugin instantiation, and typical plugin code
            (``config["api_key"].lower()`` etc.) raises AttributeError —
            a programmer-bug class that escapes ``validate_pipeline``'s
            typed catches and reduces the persisted audit row to the
            uninformative ``["runtime_preflight_failed"]`` sentinel.
        log_prefix: Prefix for structlog event names (e.g. "convergence" or "recompose_convergence").
        settings: App settings (forwarded to _state_data_from_composer_state).
        secret_service: Scoped secret resolver (forwarded to runtime preflight).

    Returns:
        Response body dict for HTTPException(status_code=422).
    """
    # Build the discriminated progress event ONCE so the 422 body and the
    # /composer-progress snapshot share a single canonical taxonomy. Without
    # this parity, the chat-side error UX (driven by detail + recovery_text)
    # could drift from the polling-side reason — exactly the failure mode
    # elspeth-5030f7373d called out, but at a different layer.
    progress = convergence_progress_event(budget_exhausted=exc.budget_exhausted)
    response_body: dict[str, object] = {
        "error_type": "convergence",
        "detail": str(exc),
        "turns_used": exc.max_turns,
        "budget_exhausted": exc.budget_exhausted,
        # ``reason`` is the public taxonomy the SPA / LLM recovery loop branch
        # on. Distinct from ``budget_exhausted`` (engine-internal triple) so
        # the two enums can evolve independently — e.g. a future
        # ``convergence_provider_quota_timeout`` would land here without
        # widening ``budget_exhausted``.
        "reason": progress.reason,
        # ``recovery_text`` mirrors the progress ``likely_next`` so the chat
        # error message can name the next practical action without parsing
        # the headline. Required by the issue's "user-facing recovery text
        # names the next practical action for each class" criterion.
        "recovery_text": progress.likely_next,
    }
    if exc.failed_turn is not None:
        response_body["failed_turn"] = await _failed_turn_response_body(service, session_id, exc.failed_turn)
    persisted_state_id: UUID | None = None
    if exc.partial_state is not None:
        # Persistence guard: DB write failure should not upgrade the
        # response from 422 (convergence error) to 500 (internal).
        #
        # SQLAlchemyError ONLY — narrowed per CLAUDE.md Tier 1 semantics.
        # _state_data_from_composer_state's internal validate() guard catches
        # (ValueError, TypeError, KeyError) from structurally damaged partial
        # state — those are acceptable there. A TypeError/KeyError from
        # state.to_dict() or CompositionStateData(...) is a Tier 1 invariant
        # bug and must propagate. This catch is the SQLAlchemy persistence
        # layer only.
        try:
            state_data, _validation = await _state_data_from_composer_state(
                exc.partial_state,
                settings=settings,
                secret_service=secret_service,
                user_id=user_id,
                runtime_preflight=None,
                preflight_exception_policy="persist_invalid",
                initial_version=None,
                telemetry_source="convergence",
            )
            partial_record = await service.save_composition_state(
                session_id,
                state_data,
                provenance="convergence_persist",
            )
            persisted_state_id = partial_record.id
            state_d = exc.partial_state.to_dict()
            response_body["partial_state"] = redact_source_storage_path(state_d)
        except SQLAlchemyError as save_err:
            # Full SQLAlchemyError family — ``IntegrityError`` alone would
            # let ``OperationalError`` (lock timeout / pool disconnect /
            # deadlock), ``ProgrammingError`` (schema drift), and siblings
            # escape, upgrading 422 → unstructured 500.
            # exc_info deliberately omitted: SQLAlchemyError __cause__
            # chains can carry DB connection strings, schema introspection
            # detail, or operational secrets that structured server logs
            # must not retain.
            slog.error(
                f"{log_prefix}_partial_state_save_failed",
                session_id=str(session_id),
                exc_class=type(save_err).__name__,
            )
            response_body["partial_state_save_failed"] = True
            # Class name only. ``str(save_err)`` on SQLAlchemyError
            # subclasses expands to ``[SQL: ...]`` + ``[parameters: ...]``
            # (the bound composition-state payload, which may reference
            # secrets) and appends ``__cause__`` text that can carry DB
            # URLs or credentials on ``OperationalError``. The slog above
            # is the triage surface; the HTTP body must not re-expose the
            # same material the ``exc_info`` omission was protecting.
            response_body["partial_state_save_error"] = type(save_err).__name__

    # Persist the per-tool-call audit trail regardless of whether
    # partial_state was set. tool_invocations is populated even when the
    # convergence happened before any state mutation (e.g. budget hit on
    # discovery-only turns), so failed runs without state changes still
    # leave a record of what the LLM tried.
    # Compose-loop carriers with failed_turn were already committed by
    # persist_compose_turn_async; only pre-cutover/non-loop carriers drain here.
    if exc.tool_invocations and exc.failed_turn is None:
        await _persist_tool_invocations(
            service,
            session_id,
            exc.tool_invocations,
            persisted_state_id,
            plugin_crash_pending=True,
        )
    if exc.llm_calls:
        await _persist_llm_calls(
            service,
            session_id,
            exc.llm_calls,
            llm_composition_state_id,
            plugin_crash_pending=True,
        )
    return response_body


async def _handle_plugin_crash(
    exc: ComposerPluginCrashError,
    service: SessionServiceProtocol,
    session_id: UUID,
    user_id: str,
    log_prefix: str,
    llm_composition_state_id: UUID | None,
    settings: Any,
    secret_service: Any | None,
) -> dict[str, object]:
    """Build 500 response body and persist partial state for plugin crashes.

    Symmetric with :func:`_handle_convergence_error` — same validation
    guard, same persistence guard, same response-body shape. The only
    differences are the exception class and the HTTP status (500 vs 422):
    a plugin crash is a server-side bug, not a user-driven failure.

    Args:
        exc: The plugin-crash wrapper with optional ``partial_state``.
        service: Session service for DB persistence of partial state.
        session_id: Session to persist partial state to.
        user_id: Authenticated user id (logged for triage).
        log_prefix: Prefix for structlog event names
            (e.g. "compose" or "recompose").
        settings: App settings (forwarded to _state_data_from_composer_state)
        secret_service: Scoped secret resolver (forwarded to runtime preflight)

    Returns:
        Response body dict for ``HTTPException(status_code=500, ...)``.
    """
    response_body: dict[str, object] = {
        "error_type": "composer_plugin_error",
        # Honest detail (elspeth-2c3d63037c): the prior wording promised
        # "see server logs for the traceback" but the slog event below
        # deliberately omits ``exc_info`` to keep secret-bearing
        # ``__cause__`` chains out of journald. Operators following the
        # claim found nothing to read.
        #
        # Stage attribution: this helper handles plugin crashes inside
        # the compose loop's ``execute_tool()`` (per
        # ``ComposerPluginCrashError``'s docstring). It is NOT the
        # runtime-preflight path — that's
        # :func:`_handle_runtime_preflight_failure`. The diagnostic
        # surface for the primary plugin crash is the structured slog
        # event ``{log_prefix}_plugin_crash`` with ``exc_class``
        # (existing tech debt being migrated under elspeth-940bfe3a0d).
        # If the partial-state runtime preflight ALSO crashes
        # downstream, that secondary failure's structured frames land
        # in the persisted state's ``validation_errors`` via the
        # post-elspeth-2c3d63037c capture helper — but we don't promise
        # frames in the response detail because the dominant case is
        # a single plugin crash with only the slog ``exc_class`` to
        # show for it.
        "detail": (
            "A composer plugin crashed during a composer tool call. "
            "The exception class is recorded in the structured server "
            "log event for triage. This is not a user-retryable error."
        ),
    }
    if exc.failed_turn is not None:
        response_body["failed_turn"] = await _failed_turn_response_body(service, session_id, exc.failed_turn)

    persisted_state_id_pc: UUID | None = None
    if exc.partial_state is not None:
        # Persistence guard: DB write failure MUST NOT mask the original
        # plugin crash (response stays as the 500 below, the save failure
        # is recorded as a separate audit-system-failure slog event).
        #
        # SQLAlchemyError ONLY — narrowed per CLAUDE.md Tier 1 semantics.
        # _state_data_from_composer_state's validate() guard catches
        # (ValueError, TypeError, KeyError) from structurally damaged partial
        # state — those are acceptable there. TypeError/KeyError from
        # state.to_dict() or CompositionStateData(...) is a Tier 1 invariant
        # bug and must propagate. Symmetric with _handle_convergence_error;
        # see the comment there for the full rationale.
        try:
            state_data, _validation = await _state_data_from_composer_state(
                exc.partial_state,
                settings=settings,
                secret_service=secret_service,
                user_id=user_id,
                runtime_preflight=None,
                preflight_exception_policy="persist_invalid",
                initial_version=None,
                telemetry_source="plugin_crash",
            )
            partial_record = await service.save_composition_state(
                session_id,
                state_data,
                provenance="plugin_crash_persist",
            )
            persisted_state_id_pc = partial_record.id
        except SQLAlchemyError as save_err:
            # Full SQLAlchemyError family — a narrow ``IntegrityError``
            # catch would let ``OperationalError`` / ``ProgrammingError`` /
            # siblings escape and mask the primary plugin-crash response.
            # exc_info deliberately omitted: ``str(save_err)`` and
            # ``__cause__`` text can carry SQL + bound parameters (the
            # composition-state payload, which may reference secrets) and
            # DB connection strings on ``OperationalError``.
            slog.error(
                f"{log_prefix}_plugin_crash_partial_state_save_failed",
                session_id=str(session_id),
                exc_class=type(save_err).__name__,
            )
            # Symmetry with _handle_convergence_error: frontend recovery UX
            # needs the same ``partial_state_save_failed`` signal on the
            # 500 path it already branches on for the 422 path. Without
            # this, a plugin crash whose partial-state persist also failed
            # looks identical to a plugin crash that succeeded in
            # persisting — the UI can't distinguish "state is captured,
            # safe to retry later" from "state is lost, start over."
            # Class name only; see the save_error comment in
            # _handle_convergence_error for the leak rationale.
            response_body["partial_state_save_failed"] = True
            response_body["partial_state_save_error"] = type(save_err).__name__

    # exc_info deliberately omitted: exc.original_exc / its __cause__ chain
    # may carry DB URLs, filesystem paths, or secret fragments. The
    # structured exc_class + session_id correlation is the complete
    # triage surface. The broader slog-for-run-events migration is
    # tracked in elspeth-940bfe3a0d.
    slog.error(
        f"{log_prefix}_plugin_crash",
        session_id=str(session_id),
        user_id=user_id,
        exc_class=exc.exc_class,
    )

    # Persist the per-tool-call audit trail. The crashing call itself
    # is the LAST entry in tool_invocations with status=PLUGIN_CRASH;
    # earlier entries are the calls that successfully ran before the
    # plugin bug fired.
    # Compose-loop plugin-crash rows commit before the carrier is raised.
    # Retain this drain only for older/non-loop carriers with no failed_turn.
    if exc.tool_invocations and exc.failed_turn is None:
        await _persist_tool_invocations(
            service,
            session_id,
            exc.tool_invocations,
            persisted_state_id_pc,
            plugin_crash_pending=True,
        )
    if exc.llm_calls:
        await _persist_llm_calls(
            service,
            session_id,
            exc.llm_calls,
            llm_composition_state_id,
            plugin_crash_pending=True,
        )
    return response_body


async def _handle_runtime_preflight_failure(
    exc: ComposerRuntimePreflightError,
    service: SessionServiceProtocol,
    session_id: UUID,
    user_id: str,
    log_prefix: str,
    llm_composition_state_id: UUID | None,
    settings: Any,
    secret_service: Any | None,
) -> dict[str, object]:
    """Build 500 response body and persist partial state for runtime-preflight failures.

    Symmetric with :func:`_handle_plugin_crash` — same partial-state
    persistence guard, same response-body shape, same SQLAlchemyError
    fail-soft contract. The exception class is the only structural
    difference: ``ComposerRuntimePreflightError`` carries
    ``original_exc`` instead of the plugin-crash wrapper's ``exc_class``
    string, but the recovery semantics are identical.

    Two raise sites converge here:

    1. **Cached path** — :func:`elspeth.web.composer.service._raise_cached_runtime_preflight_failure`
       inside ``composer.compose()`` re-raises a previously-cached
       runtime-preflight failure when the LLM's final state matches a
       cached failure key. Caught at the route's top-level compose-time
       except clause.

    2. **Post-compose path** — :func:`_state_data_from_composer_state`
       called with ``preflight_exception_policy="raise"`` on the
       post-compose state-save block. This raise site sits *after* the
       compose-time try/except, so the caller wraps the post-compose
       block in its own try/except that delegates here.

    Telemetry attribution
    ---------------------
    Every invocation of this helper produces a recovery emission, paired
    with a primary emission attributed to the originating raise site:

    * Primary site (path-2) — ``_state_data_from_composer_state`` lifts
      its exception telemetry above the policy split, so the raise arm
      fires
      ``composer.runtime_preflight.total{result=exception,
      source=<originating>, exception_class=...}`` (where
      ``<originating>`` is ``"compose"`` or ``"recompose"``) before
      propagating the ``ComposerRuntimePreflightError``. This preserves
      originating-route attribution that would otherwise be lost.

    * Primary site (path-1, elspeth-0891e8da73) — the route catch around
      ``composer.compose()`` fires
      ``composer.runtime_preflight.total{result=exception,
      source=cached_preflight, exception_class=...}`` before delegating
      here. The cached raise site lives in the composer service rather
      than ``_state_data_from_composer_state``, so the primary emission
      cannot be lifted into the helper itself; the route catch is the
      first place that knows the failure mode is a runtime-preflight
      cache re-raise.

    * Recovery site (both paths) — this helper emits
      ``source=runtime_preflight`` exactly once per invocation. When
      ``exc.partial_state is not None``, the emission is fired
      automatically by the persist_invalid arm of the re-call to
      ``_state_data_from_composer_state``. When ``exc.partial_state is
      None`` (the path-1 cached re-raise with no LLM mutation case), the
      persist_invalid re-call is skipped and the helper instead fires
      the recovery emission inline. Either way, the recovery counter
      increments once per handler invocation.

    Operators querying primary failure rate filter on
    ``source ∈ {compose, recompose, cached_preflight, plugin_crash,
    convergence, yaml_export}``; querying recovery handler invocations
    filters on ``source = runtime_preflight``. The two sources represent
    distinct observable events (failure occurred; recovery ran), not
    double-counting.

    Caveat: the persist_invalid arm only re-runs the runtime preflight
    when the partial state is *authoring-valid* (the
    ``runtime is None and authoring.is_valid`` branch in
    :func:`_state_data_from_composer_state`). For a partial state captured
    immediately after a runtime-preflight raise — the dominant case for
    both raise sites — authoring is valid by construction, so the counter
    increments. If a future caller passes a partial that fails authoring
    validation, the runtime-preflight counter will not increment for that
    invocation; only the authoring-validation counter will. That is the
    correct attribution: a state that can't be authoring-validated cannot
    have a meaningful runtime-preflight outcome to attribute.

    Logging policy
    --------------
    Unlike :func:`_handle_plugin_crash` and
    :func:`_handle_convergence_error`, this helper does **not** emit a
    ``slog.error`` triage event. Per CLAUDE.md telemetry/logging primacy,
    the OpenTelemetry counter above is the canonical triage surface for
    runtime-preflight outcomes. The slog calls in the sibling helpers
    are pre-existing tech debt being migrated under elspeth-940bfe3a0d;
    new code does not adopt the pattern.

    The persistence-failure ``slog.error`` inside the SQLAlchemyError
    catch is the audit-system-failure exemption — when DB persistence
    itself breaks, slog is the appropriate last-resort channel — and is
    retained here for parity with the sibling helpers' fallback path.

    Args:
        exc: The runtime-preflight wrapper with ``original_exc`` and
            optional ``partial_state``.
        service: Session service for DB persistence of partial state.
        session_id: Session to persist partial state to.
        user_id: Authenticated user id (logged on the SQLAlchemyError
            audit-failure path for triage correlation).
        log_prefix: Prefix for the SQLAlchemyError audit-failure event
            name (e.g. ``"compose"`` or ``"recompose"``).
        settings: App settings (forwarded to
            :func:`_state_data_from_composer_state`).
        secret_service: Scoped secret resolver (forwarded to runtime
            preflight on the recovery re-call).

    Returns:
        Response body dict for ``HTTPException(status_code=500, ...)``.
    """
    response_body: dict[str, object] = {
        "error_type": "composer_plugin_error",
        # Honest detail (elspeth-2c3d63037c): the prior wording promised
        # "see server logs for the traceback" but this helper deliberately
        # emits no slog.error triage event (per the "Logging policy"
        # section above) and the sibling helpers' slog calls omit
        # exc_info to avoid leaking secret-bearing __cause__ chains. The
        # diagnostic surface is now the persisted state's
        # validation_errors row (see :func:`_runtime_preflight_failure_errors`
        # — exception_class + first-line message + bounded
        # file:line:function frames) plus the OTel
        # ``composer.runtime_preflight.total`` counter, bucketed by
        # ``exception_class``.
        "detail": (
            "A composer plugin crashed during runtime preflight. "
            "Diagnostic frames are recorded in the persisted state's "
            "validation_errors when a partial state was captured. "
            "This is not a user-retryable error."
        ),
    }
    if exc.failed_turn is not None:
        response_body["failed_turn"] = await _failed_turn_response_body(service, session_id, exc.failed_turn)

    persisted_state_id_rpf: UUID | None = None
    if exc.partial_state is not None:
        # Persistence guard: DB write failure MUST NOT mask the original
        # runtime-preflight failure (response stays as the 500 below; the
        # save failure is recorded as a separate audit-system-failure
        # slog event — that slog event is the persistence-fallback
        # exemption, NOT a normal-flow log).
        #
        # SQLAlchemyError ONLY — narrowed per CLAUDE.md Tier 1 semantics,
        # symmetric with the sibling _handle_plugin_crash /
        # _handle_convergence_error helpers. See the comment in
        # _handle_convergence_error for the full rationale on the
        # SQLAlchemyError width.
        try:
            state_data, _validation = await _state_data_from_composer_state(
                exc.partial_state,
                settings=settings,
                secret_service=secret_service,
                user_id=user_id,
                runtime_preflight=None,
                preflight_exception_policy="persist_invalid",
                initial_version=None,
                telemetry_source="runtime_preflight",
            )
            partial_record = await service.save_composition_state(
                session_id,
                state_data,
                provenance="preflight_persist",
            )
            persisted_state_id_rpf = partial_record.id
        except SQLAlchemyError as save_err:
            # See sibling helpers for redaction rationale (exc_info
            # omitted; class name only on the response body).
            slog.error(
                f"{log_prefix}_runtime_preflight_partial_state_save_failed",
                session_id=str(session_id),
                user_id=user_id,
                exc_class=type(save_err).__name__,
            )
            response_body["partial_state_save_failed"] = True
            response_body["partial_state_save_error"] = type(save_err).__name__
    else:
        # Recovery acknowledgment (elspeth-0891e8da73). When no partial
        # state is captured (the path-1 cached re-raise with no LLM
        # mutation case), the persist_invalid re-call above is skipped,
        # so its source=runtime_preflight emission inside
        # _state_data_from_composer_state never fires. CLAUDE.md
        # telemetry primacy ("every telemetry emission point must send
        # or explicitly acknowledge 'nothing to send.'") requires the
        # recovery handler to count its own invocation regardless of
        # whether persistence work occurred — otherwise dashboards
        # filtering composer.runtime_preflight.total{source=
        # runtime_preflight} silently under-count handler invocations
        # by exactly the count of "no partial state" recoveries.
        _record_composer_runtime_preflight_telemetry(
            "exception",
            source="runtime_preflight",
            exception_class=exc.exc_class,
        )

    # Persist the per-tool-call audit trail. Preview-path runtime
    # preflight failures now record the preview_pipeline tool invocation
    # before raising; other runtime-preflight failures may still carry an
    # empty tuple. The unconditional call handles both via the empty-tuple
    # early-return inside _persist_tool_invocations.
    # Runtime-preflight carriers from the compose loop use the committed
    # failed_turn row; post-compose/non-loop carriers still drain here.
    if exc.tool_invocations and exc.failed_turn is None:
        await _persist_tool_invocations(
            service,
            session_id,
            exc.tool_invocations,
            persisted_state_id_rpf,
            plugin_crash_pending=True,
        )
    if exc.llm_calls:
        await _persist_llm_calls(
            service,
            session_id,
            exc.llm_calls,
            llm_composition_state_id,
            plugin_crash_pending=True,
        )
    return response_body


def _initial_composition_state_with_guided_session() -> CompositionState:
    """Construct a fresh CompositionState with a guided-mode session attached.

    Per spec §5.2 / errata C7: new sessions default to guided. Every lazy-
    create branch in this module must use this helper rather than building
    CompositionState directly, so the default-guided invariant holds
    uniformly across endpoints (send_message, recompose, the future
    guided/respond endpoint added in Task 3.5).
    """
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
        guided_session=GuidedSession.initial(),
    )


def _validate_control_signal(raw: str | None) -> ControlSignal | None:
    """Validate ``control_signal`` against the closed :class:`ControlSignal` enum.

    Returns the parsed :class:`ControlSignal` (or ``None`` if *raw* is ``None``).
    Raises :class:`HTTPException` (400) when *raw* is non-None and not a
    recognised enum value.  Null passes through — omitting the signal is
    the normal (non-control) path.

    This is the Tier-3 -> Tier-2 coercion site: ``GuidedRespondRequest`` keeps
    ``control_signal`` as ``str | None`` so stale clients carrying an unknown
    signal value receive a clear protocol error from this guard rather than a
    Pydantic deserialization crash. After this function returns, the parsed
    :class:`ControlSignal` is the typed value flowing into ``TurnResponse``,
    where the field is declared ``ControlSignal | None``.
    """
    if raw is None:
        return None
    try:
        return ControlSignal(raw)
    except ValueError as exc:
        valid = [e.value for e in ControlSignal]
        raise HTTPException(
            status_code=400,
            detail=(f"Unknown control_signal: {raw!r}. Valid values: {valid}"),
        ) from exc


def _validate_step_indices(
    turn_type: TurnType,
    accepted_step_index: int | None,
    edit_step_index: int | None,
    guided: GuidedSession,
) -> None:
    """Validate ``accepted_step_index`` / ``edit_step_index`` against the current step.

    Raises :class:`HTTPException` (400) when an index field is non-None for a
    turn type that carries no step-index semantics, or when the index is out of
    range for the proposal that is actually staged.

    Step-index semantics exist **only** for ``PROPOSE_CHAIN`` turns: they
    reference positions within ``guided.step_3_proposal.steps``.  Every other
    turn type must send both fields as ``None`` — a non-None value on any other
    turn type indicates a stale or mis-shaped client payload.

    Matrix (exhaustive):

    +---------------------------------+--------------------------------------+
    | Turn type                       | Accepted / edit index               |
    +=================================+======================================+
    | PROPOSE_CHAIN                   | Must be None or                     |
    |                                 | 0 <= idx < len(proposal.steps)      |
    +---------------------------------+--------------------------------------+
    | All other turn types            | Must be None                         |
    +---------------------------------+--------------------------------------+
    """
    if turn_type is not TurnType.PROPOSE_CHAIN:
        if accepted_step_index is not None or edit_step_index is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"accepted_step_index and edit_step_index must be null for turn type "
                    f"{turn_type.value!r}; got accepted_step_index={accepted_step_index!r}, "
                    f"edit_step_index={edit_step_index!r}."
                ),
            )
        return

    # PROPOSE_CHAIN: validate against the staged proposal.
    # If step_3_proposal is None but an index was supplied, the client is
    # sending a step-index against a step that has not been reached yet —
    # treat as a client-fault 400 (not a server 500) because the proposal
    # absence is not necessarily an invariant violation at this point (the
    # route handler checks later for the accept path).
    proposal = guided.step_3_proposal
    if proposal is None:
        if accepted_step_index is not None or edit_step_index is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "accepted_step_index / edit_step_index sent but no chain proposal is staged for this session (step_3_proposal is None)."
                ),
            )
        return

    step_count = len(proposal.steps)
    for field_name, idx in (
        ("accepted_step_index", accepted_step_index),
        ("edit_step_index", edit_step_index),
    ):
        if idx is None:
            continue
        if not (0 <= idx < step_count):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{field_name}={idx!r} is out of range for the current proposal "
                    f"(proposal has {step_count} step(s); valid range is 0-{step_count - 1})."
                ),
            )


async def _dispatch_guided_respond(
    *,
    state: CompositionState,
    guided: GuidedSession,
    current_step: GuidedStep,
    current_turn_type: TurnType,
    turn_response: Mapping[str, Any],
    catalog: CatalogServiceProtocol,
    recorder: BufferingRecorder,
    user_id: str,
    data_dir: str | None,
    session_engine: Any,
    session_id: str,
    model: str,
) -> tuple[CompositionState, GuidedSession, Any | None]:
    """Dispatch a guided respond to the correct step handler and next-turn emitter.

    Pure routing logic: identifies which branch to take based on
    ``current_step`` and ``current_turn_type``, calls the appropriate
    side-effect step handler, advances the session pointer, and emits
    the next turn.

    Returns ``(updated_state, updated_session, next_turn_or_None)``.

    The dispatcher is called only when ``guided.terminal is None``.  The
    caller checks terminality before and after; the dispatcher never
    terminates a session (that is ``step_advance``'s responsibility for
    exit_to_freeform and the step-2.5 recipe-accept path).  The caller
    also bypasses the dispatcher entirely on the exit-from-COMPLETED path
    (see ``post_guided_respond``): that path transitions terminal kind
    COMPLETED -> EXITED_TO_FREEFORM directly, without running any
    step-handler dispatch logic.

    Decision table:

    +-------------------+---------------------------+---------------------------+
    | current_step      | guided.step (after adv.)  | action                    |
    +-------------------+---------------------------+---------------------------+
    | STEP_1_SOURCE     | STEP_1_SOURCE             | intra-step; turn_type     |
    |   (intra)         | (no advance fired)        | decides next turn:        |
    |                   |                           | SINGLE_SELECT →           |
    |                   |                           |   emit SCHEMA_FORM        |
    |                   |                           | SCHEMA_FORM →             |
    |                   |                           |   handle_step_1_source;   |
    |                   |                           |   advance to STEP_2;      |
    |                   |                           |   emit SINGLE_SELECT      |
    | STEP_1_SOURCE     | STEP_2_SINK               | INSPECT_AND_CONFIRM path; |
    |   (advancing)     | (step_advance fired)      | handle_step_1_source;     |
    |                   |                           | emit SINGLE_SELECT (sink) |
    | STEP_2_SINK       | STEP_2_SINK               | intra-step; turn_type     |
    |   (intra)         | (no advance fired)        | decides next turn:        |
    |                   |                           | SINGLE_SELECT →           |
    |                   |                           |   emit SCHEMA_FORM        |
    |                   |                           | SCHEMA_FORM →             |
    |                   |                           |   emit MULTI_SELECT       |
    |                   |                           | MULTI_SELECT →            |
    |                   |                           |   unreachable here        |
    |                   |                           |   (_advance_step_2 always |
    |                   |                           |   fires for this type)    |
    | STEP_2_SINK       | STEP_2_5_RECIPE_MATCH     | MULTI_SELECT path;        |
    |   (advancing)     | (step_advance fired)      | handle_step_2_sink;       |
    |                   |                           | match_recipe;             |
    |                   |                           | emit RECIPE_OFFER or None |
    | STEP_2_5_RECIPE   | STEP_2_5_RECIPE_MATCH     | RECIPE_OFFER chosen=      |
    |   (intra/term.)   | (accept: no advance)      | accept → recipe apply;    |
    |                   |                           | terminal=COMPLETED        |
    | STEP_2_5_RECIPE   | STEP_3_TRANSFORMS         | RECIPE_OFFER chosen=      |
    |   (advancing)     | (step_advance fired)      | build_manually → step 3   |
    +-------------------+---------------------------+---------------------------+

    ``step_advance`` has already run; ``guided.step`` may already point to the
    next step (when step_advance fired a step transition).  The dispatcher uses
    ``current_step`` (before advance) and ``guided.step`` (after advance) to
    detect transitions.
    """
    from dataclasses import replace as _replace

    next_turn: Any | None = None

    # --- STEP_1_SOURCE intra-step turns ----------------------------------
    if current_step is GuidedStep.STEP_1_SOURCE and guided.step is GuidedStep.STEP_1_SOURCE:
        # step_advance did NOT advance the step — intra-step turn.
        if current_turn_type is TurnType.SINGLE_SELECT:
            # User picked a source plugin. Emit schema_form for the plugin options.
            chosen = turn_response["chosen"] or []
            if not chosen:
                raise HTTPException(
                    status_code=400,
                    detail="single_select response at step 1 must include chosen plugin name.",
                )
            plugin_name = str(chosen[0])
            next_turn = build_step_1_schema_form_turn(plugin_name, catalog)
            new_record = TurnRecord(
                step=current_step,
                turn_type=TurnType(next_turn["type"]),
                payload_hash=stable_hash(next_turn["payload"]),
                response_hash=None,
                emitter="server",
            )
            emit_turn_emitted(
                recorder,
                step=current_step,
                turn_type=TurnType(next_turn["type"]),
                payload_hash=stable_hash(next_turn["payload"]),
                payload_payload_id="",
                emitter="server",
                composition_version=state.version,
                actor=user_id,
            )
            guided = _replace(guided, history=(*guided.history, new_record))
            return state, guided, next_turn

        if current_turn_type is TurnType.SCHEMA_FORM:
            # User submitted source options. Call handle_step_1_source to commit.
            # ``edited_values`` is the wire contract for the Step-1 SCHEMA_FORM
            # response: the SchemaFormTurn widget always sends
            # ``{"plugin": str, "options": Mapping, "observed_columns": list,
            # "sample_rows": list}``. A missing key, a null payload, a non-string
            # ``plugin``, or non-list-shaped ``observed_columns`` / ``sample_rows``
            # is a protocol violation, not a Tier-3 "user typo" we paper over
            # with ``.get()`` defaults — silent defaults here would surface
            # far downstream as inference from adjacent fields (forbidden per
            # CLAUDE.md §Three-Tier Trust Model). The HTTP boundary is a trust
            # boundary, so we raise 400 with a contract-citing message.
            edited = turn_response["edited_values"]
            if edited is None:
                raise HTTPException(
                    status_code=400,
                    detail="schema_form response at step 1 requires edited_values; received null.",
                )
            missing = {"plugin", "options", "observed_columns", "sample_rows"} - edited.keys()
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"schema_form response at step 1 edited_values missing required keys: {sorted(missing)}; got keys: {sorted(edited.keys())}"
                    ),
                )
            plugin_name = edited["plugin"]
            if not isinstance(plugin_name, str) or not plugin_name:
                raise HTTPException(
                    status_code=400,
                    detail=(f"schema_form response at step 1 edited_values['plugin'] must be a non-empty string; got {plugin_name!r}"),
                )
            options_raw = edited["options"]
            if not isinstance(options_raw, Mapping):
                raise HTTPException(
                    status_code=400,
                    detail=(f"schema_form response at step 1 edited_values['options'] must be an object; got {type(options_raw).__name__}"),
                )
            observed_columns_raw = edited["observed_columns"]
            if not isinstance(observed_columns_raw, list):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"schema_form response at step 1 edited_values['observed_columns'] must be a list; got {type(observed_columns_raw).__name__}"
                    ),
                )
            sample_rows_raw = edited["sample_rows"]
            if not isinstance(sample_rows_raw, list):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"schema_form response at step 1 edited_values['sample_rows'] must be a list; got {type(sample_rows_raw).__name__}"
                    ),
                )
            # Codex #16: validate each element is a Mapping before calling
            # dict(r).  A non-Mapping element (e.g. int, str, list) triggers
            # TypeError from dict() — uncontrolled 500 instead of a clear 400.
            # The HTTP boundary is a trust boundary: external data may contain
            # arbitrary JSON values at any list position.
            for _sr_idx, _sr_elem in enumerate(sample_rows_raw):
                if not isinstance(_sr_elem, Mapping):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"schema_form response at step 1 edited_values['sample_rows'][{_sr_idx}] "
                            f"must be an object; got {type(_sr_elem).__name__}"
                        ),
                    )
            resolved = SourceResolved(
                plugin=plugin_name,
                options=dict(options_raw),
                observed_columns=tuple(observed_columns_raw),
                sample_rows=tuple(dict(r) for r in sample_rows_raw),
            )
            handler_result = handle_step_1_source(
                state=state,
                session=guided,
                resolved=resolved,
                catalog=catalog,
                data_dir=data_dir,
                session_engine=session_engine,
                session_id=session_id,
            )
            if not handler_result.tool_result.success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Step 1 source commit failed: {handler_result.tool_result}",
                )
            state = handler_result.state
            # Advance step pointer to STEP_2.
            guided = _replace(
                handler_result.session,
                step=GuidedStep.STEP_2_SINK,
            )
            # Emit Step 2 initial turn.
            next_turn = build_step_2_single_select_turn(catalog)
            new_record = TurnRecord(
                step=GuidedStep.STEP_2_SINK,
                turn_type=TurnType(next_turn["type"]),
                payload_hash=stable_hash(next_turn["payload"]),
                response_hash=None,
                emitter="server",
            )
            emit_step_advanced(
                recorder,
                prev=GuidedStep.STEP_1_SOURCE,
                next_=GuidedStep.STEP_2_SINK,
                reason="user_advanced",
                composition_version=state.version,
                actor=user_id,
            )
            emit_turn_emitted(
                recorder,
                step=GuidedStep.STEP_2_SINK,
                turn_type=TurnType(next_turn["type"]),
                payload_hash=stable_hash(next_turn["payload"]),
                payload_payload_id="",
                emitter="server",
                composition_version=state.version,
                actor=user_id,
            )
            guided = _replace(guided, history=(*guided.history, new_record))
            return state, guided, next_turn

        # INSPECT_AND_CONFIRM at STEP_1: step_advance already advanced to STEP_2.
        # But in this branch guided.step is still STEP_1 — step_advance
        # must have already advanced it.  This case is unreachable (step_advance
        # advances for INSPECT_AND_CONFIRM → guided.step becomes STEP_2).
        # Fall through to the post-advance branch below.

    # --- STEP_1_SOURCE → STEP_2_SINK (step_advance fired for INSPECT_AND_CONFIRM)
    if current_step is GuidedStep.STEP_1_SOURCE and guided.step is GuidedStep.STEP_2_SINK:
        # step_advance already ran and advanced the step.  _advance_step_1 built
        # SourceResolved from step_1_source_intent (plugin/options/samples) +
        # edited_values["columns"] (the only field the widget can edit) and stored
        # the result in guided.step_1_result.  It also cleared step_1_source_intent.
        #
        # The new wire contract for INSPECT_AND_CONFIRM edited_values is narrow:
        #   ``{"columns": list[str]}``
        # Plugin, options, observed_columns (samples), and warnings are now
        # server-side knowledge held in step_1_source_intent before the turn is
        # emitted; the widget never sees them in the response body.
        #
        # Shadowing note: ``_advance_step_1`` raises ValueError (client-fault)
        # on null ``edited_values`` and KeyError on missing ``columns`` — those
        # propagate as HTTP 500 before reaching here. The null guard and missing-key guard
        # below are defense-in-depth (locally complete; unreachable as 400 in
        # normal flow). The isinstance(columns_raw, list) guard IS HTTP-reachable
        # as 400: _advance_step_1's ``tuple(str(c) for c in columns_raw)`` coercion
        # silently accepts a scalar string (iterating its characters), so the
        # dispatcher catches non-list here before that coercion runs.
        #
        # EMIT-SITE NOTE: step_1_source_intent must be set before emitting the
        # INSPECT_AND_CONFIRM turn.  Today no production code path emits this turn
        # (``_build_get_guided_turn`` always passes blob_inspection=None); the only
        # emitter is _seed_inspect_and_confirm_history in the integration test suite.
        # When a real emitter is added, it must
        #   ``replace(session, step_1_source_intent=SourceIntent(...))``
        # before returning.
        edited = turn_response["edited_values"]
        if edited is None:
            raise HTTPException(
                status_code=400,
                detail="inspect_and_confirm response at step 1 requires edited_values; received null.",
            )
        if "columns" not in edited:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"inspect_and_confirm response at step 1 edited_values missing required key: 'columns'; got keys: {sorted(edited.keys())}"
                ),
            )
        columns_raw = edited["columns"]
        if not isinstance(columns_raw, list):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"inspect_and_confirm response at step 1 edited_values['columns'] must be a list; got {type(columns_raw).__name__}"
                ),
            )
        # SourceResolved was built by _advance_step_1 and stored in guided.step_1_result.
        # Unreachable None: _advance_step_1 always sets step_1_result on advance (the
        # branch is only reached when guided.step is STEP_2_SINK, which only happens
        # after _advance_step_1 sets step_1_result).  Assert to keep mypy happy and
        # provide a clear crash message if this invariant is ever violated.
        if guided.step_1_result is None:
            raise InvariantError(
                "inspect_and_confirm post-advance: guided.step_1_result is None after "
                "_advance_step_1 ran — this is an invariant violation in the state machine."
            )
        resolved = guided.step_1_result
        handler_result = handle_step_1_source(
            state=state,
            session=guided,
            resolved=resolved,
            catalog=catalog,
            data_dir=data_dir,
            session_engine=session_engine,
            session_id=session_id,
        )
        if not handler_result.tool_result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Step 1 source commit failed: {handler_result.tool_result}",
            )
        state = handler_result.state
        guided = handler_result.session
        next_turn = build_step_2_single_select_turn(catalog)
        new_record = TurnRecord(
            step=GuidedStep.STEP_2_SINK,
            turn_type=TurnType(next_turn["type"]),
            payload_hash=stable_hash(next_turn["payload"]),
            response_hash=None,
            emitter="server",
        )
        emit_turn_emitted(
            recorder,
            step=GuidedStep.STEP_2_SINK,
            turn_type=TurnType(next_turn["type"]),
            payload_hash=stable_hash(next_turn["payload"]),
            payload_payload_id="",
            emitter="server",
            composition_version=state.version,
            actor=user_id,
        )
        guided = _replace(guided, history=(*guided.history, new_record))
        return state, guided, next_turn

    # --- STEP_2_SINK intra-step turns ------------------------------------
    if current_step is GuidedStep.STEP_2_SINK and guided.step is GuidedStep.STEP_2_SINK:
        if current_turn_type is TurnType.SINGLE_SELECT:
            # User picked a sink plugin. Emit schema_form for the plugin options.
            chosen = turn_response["chosen"] or []
            if not chosen:
                raise HTTPException(
                    status_code=400,
                    detail="single_select response at step 2 must include chosen plugin name.",
                )
            plugin_name = str(chosen[0])
            next_turn = build_step_2_schema_form_turn(plugin_name, catalog)
            new_record = TurnRecord(
                step=current_step,
                turn_type=TurnType(next_turn["type"]),
                payload_hash=stable_hash(next_turn["payload"]),
                response_hash=None,
                emitter="server",
            )
            emit_turn_emitted(
                recorder,
                step=current_step,
                turn_type=TurnType(next_turn["type"]),
                payload_hash=stable_hash(next_turn["payload"]),
                payload_payload_id="",
                emitter="server",
                composition_version=state.version,
                actor=user_id,
            )
            # Persist the chosen plugin name so GET /guided can rebuild the SCHEMA_FORM
            # on refresh (rebuild requires the plugin name to fetch the schema from the
            # catalog).  Cleared atomically when step_2_sink_intent is set below.
            guided = _replace(
                guided,
                history=(*guided.history, new_record),
                step_2_chosen_plugin=plugin_name,
            )
            return state, guided, next_turn

        if current_turn_type is TurnType.SCHEMA_FORM:
            # User filled in sink options. Persist the chosen plugin + options into
            # step_2_sink_intent so _advance_step_2 can reconstruct the full
            # SinkOutputResolved when the subsequent MULTI_SELECT_WITH_CUSTOM
            # response arrives (which only carries required_fields, not the plugin).
            # ``edited_values`` is the wire contract for the Step-2 SCHEMA_FORM
            # response: the SchemaFormTurn widget always sends
            # ``{"plugin": str, "options": Mapping, "observed_columns": list,
            # "sample_rows": list}``. Step 2 only consumes ``plugin`` + ``options``
            # for the sink_intent, but the wire shape is fully validated to keep
            # the contract identical with Step 1 (no silent-tolerance drift).
            # A missing key, a null payload, or a wrong-typed value is a
            # protocol violation — silent ``.get()`` defaults would fabricate
            # fields the client never supplied (forbidden per CLAUDE.md
            # §Three-Tier Trust Model). The HTTP boundary is a trust boundary,
            # so we raise 400 with a contract-citing message.
            edited = turn_response["edited_values"]
            if edited is None:
                raise HTTPException(
                    status_code=400,
                    detail="schema_form response at step 2 requires edited_values; received null.",
                )
            missing = {"plugin", "options"} - edited.keys()
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"schema_form response at step 2 edited_values missing required keys: {sorted(missing)}; got keys: {sorted(edited.keys())}"
                    ),
                )
            plugin_name = edited["plugin"]
            if not isinstance(plugin_name, str) or not plugin_name:
                raise HTTPException(
                    status_code=400,
                    detail=(f"schema_form response at step 2 edited_values['plugin'] must be a non-empty string; got {plugin_name!r}"),
                )
            options_raw = edited["options"]
            if not isinstance(options_raw, Mapping):
                raise HTTPException(
                    status_code=400,
                    detail=(f"schema_form response at step 2 edited_values['options'] must be an object; got {type(options_raw).__name__}"),
                )
            sink_intent = SinkIntent(
                plugin=plugin_name,
                options=dict(options_raw),
            )
            # Set step_2_sink_intent and clear step_2_chosen_plugin atomically.
            # Once step_2_sink_intent is set, the full plugin+options tuple is
            # held there; step_2_chosen_plugin is only needed in the window
            # between SINGLE_SELECT and SCHEMA_FORM (to rebuild SCHEMA_FORM on
            # GET /guided refresh).  Having both set simultaneously would be
            # redundant and could mislead rebuild logic about the intra-step phase.
            guided = _replace(guided, step_2_sink_intent=sink_intent, step_2_chosen_plugin=None)

            # Emit multi_select_with_custom for required fields.
            # Observed columns come from step_1_result if available.
            observed_columns: tuple[str, ...] = ()
            if guided.step_1_result is not None:
                observed_columns = tuple(guided.step_1_result.observed_columns)
            next_turn = build_step_2_multi_select_turn(observed_columns)
            new_record = TurnRecord(
                step=current_step,
                turn_type=TurnType(next_turn["type"]),
                payload_hash=stable_hash(next_turn["payload"]),
                response_hash=None,
                emitter="server",
            )
            emit_turn_emitted(
                recorder,
                step=current_step,
                turn_type=TurnType(next_turn["type"]),
                payload_hash=stable_hash(next_turn["payload"]),
                payload_payload_id="",
                emitter="server",
                composition_version=state.version,
                actor=user_id,
            )
            guided = _replace(guided, history=(*guided.history, new_record))
            return state, guided, next_turn

    # --- STEP_2_SINK → STEP_2_5 (step_advance fired for MULTI_SELECT_WITH_CUSTOM)
    # step_advance sets guided.step=STEP_2_5_RECIPE_MATCH and populates
    # guided.step_2_result for every MULTI_SELECT_WITH_CUSTOM response, so the
    # intra-step branch (current_step==guided.step==STEP_2_SINK) for this turn
    # type is structurally unreachable — _advance_step_2 always fires.
    # This is the ONLY reachable MULTI_SELECT_WITH_CUSTOM dispatch point.
    if current_step is GuidedStep.STEP_2_SINK and guided.step is GuidedStep.STEP_2_5_RECIPE_MATCH:
        # step_advance already set guided.step_2_result and advanced to STEP_2_5.
        if guided.step_1_result is None or guided.step_2_result is None:
            raise HTTPException(
                status_code=500,
                detail="Dispatcher invariant: step_1_result and step_2_result must be set.",
            )
        source = guided.step_1_result
        sink = guided.step_2_result

        # Commit the sink to CompositionState.outputs via handle_step_2_sink.
        # step_advance (pure) encoded the sink in guided.step_2_result but did
        # NOT call _execute_set_output — that side-effect is our responsibility.
        sink_handler_result = handle_step_2_sink(
            state=state,
            session=guided,
            resolved=sink,
            catalog=catalog,
            data_dir=data_dir,
        )
        if not sink_handler_result.tool_result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Step 2 sink commit failed: {sink_handler_result.tool_result}",
            )
        state = sink_handler_result.state
        guided = sink_handler_result.session

        recipe_match = match_recipe(source, sink)
        if recipe_match is not None:
            next_turn = build_step_2_5_recipe_offer_turn(recipe_match)
            new_record = TurnRecord(
                step=GuidedStep.STEP_2_5_RECIPE_MATCH,
                turn_type=TurnType(next_turn["type"]),
                payload_hash=stable_hash(next_turn["payload"]),
                response_hash=None,
                emitter="server",
            )
            emit_turn_emitted(
                recorder,
                step=GuidedStep.STEP_2_5_RECIPE_MATCH,
                turn_type=TurnType(next_turn["type"]),
                payload_hash=stable_hash(next_turn["payload"]),
                payload_payload_id="",
                emitter="server",
                composition_version=state.version,
                actor=user_id,
            )
            # Stage the offered RecipeMatch so the accept branch can verify
            # the client-supplied recipe_name matches what was actually offered.
            # Cleared (set to None) atomically when the acceptance is consumed.
            guided = _replace(guided, history=(*guided.history, new_record), step_2_5_recipe_offer=recipe_match)
            return state, guided, next_turn

        # No recipe match — solve the chain via the LLM and emit propose_chain.
        # I2: wrap transient LLM failures (LiteLLM API/auth/bad-request,
        # timeouts, malformed-response shape) so they auto-drop to freeform
        # via mark_solver_exhausted rather than escape as 500s.  See
        # ``solve_chain_with_auto_drop`` for the contract.
        proposal, guided = await solve_chain_with_auto_drop(
            site="step_2_sink_initial_solve",
            session=guided,
            session_id=session_id,
            user_id=user_id,
            composition_version=state.version,
            recorder=recorder,
            model=model,
            source=source,
            sink=sink,
            recipe_match=None,
        )
        if proposal is None:
            return state, guided, None
        guided = _replace(guided, step=GuidedStep.STEP_3_TRANSFORMS, step_3_proposal=proposal)
        next_turn = build_step_3_propose_chain_turn(proposal)
        new_record = TurnRecord(
            step=GuidedStep.STEP_3_TRANSFORMS,
            turn_type=TurnType.PROPOSE_CHAIN,
            payload_hash=stable_hash(next_turn["payload"]),
            response_hash=None,
            emitter="server",
        )
        emit_step_advanced(
            recorder,
            prev=GuidedStep.STEP_2_5_RECIPE_MATCH,
            next_=GuidedStep.STEP_3_TRANSFORMS,
            # System-driven advance: no recipe matched the (source, sink) topology,
            # so the dispatcher hops STEP_2_5 → STEP_3 without operator input.
            reason="auto_advanced",
            composition_version=state.version,
            actor=user_id,
        )
        emit_turn_emitted(
            recorder,
            step=GuidedStep.STEP_3_TRANSFORMS,
            turn_type=TurnType.PROPOSE_CHAIN,
            payload_hash=stable_hash(next_turn["payload"]),
            payload_payload_id="",
            emitter="server",
            composition_version=state.version,
            actor=user_id,
        )
        guided = _replace(guided, history=(*guided.history, new_record))
        return state, guided, next_turn

    # --- STEP_2_5_RECIPE_MATCH turns ------------------------------------
    if current_step is GuidedStep.STEP_2_5_RECIPE_MATCH:
        chosen = list(turn_response["chosen"] or [])
        if chosen == ["accept"]:
            # User accepted the recipe. Extract the slots from edited_values
            # (user may have filled in required slots).
            # ``edited_values`` is the wire contract for recipe_offer ['accept']:
            # the RecipeOfferTurn widget always sends ``{"recipe_name": str,
            # "slots": {...}}``. A missing key, a null payload, or a non-string
            # ``recipe_name`` is a protocol violation, not a Tier-3 "user typo"
            # we paper over with empty-string defaults — silent defaults here
            # would surface far downstream as a misleading "unknown recipe ''"
            # error, masking the actual contract drift. The HTTP boundary is a
            # trust boundary, so we raise 400 with a contract-citing message.
            edited = turn_response["edited_values"]
            if edited is None:
                raise HTTPException(
                    status_code=400,
                    detail="recipe_offer ['accept'] requires edited_values; received null.",
                )
            missing = {"recipe_name", "slots"} - edited.keys()
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"recipe_offer ['accept'] edited_values missing required keys: {sorted(missing)}; got keys: {sorted(edited.keys())}"
                    ),
                )
            recipe_name = edited["recipe_name"]
            if not isinstance(recipe_name, str) or not recipe_name:
                raise HTTPException(
                    status_code=400,
                    detail=(f"recipe_offer ['accept'] edited_values['recipe_name'] must be a non-empty string; got {recipe_name!r}"),
                )
            slots_raw = edited["slots"]
            if not isinstance(slots_raw, Mapping):
                raise HTTPException(
                    status_code=400,
                    detail=(f"recipe_offer ['accept'] edited_values['slots'] must be an object; got {type(slots_raw).__name__}"),
                )

            # Binding check: the client-supplied recipe_name must match the
            # recipe that was actually offered for this step.  ``step_2_5_recipe_offer``
            # is written by the server immediately before emitting the RECIPE_OFFER
            # turn (Option A staging pattern, mirrors step_2_sink_intent).  A missing
            # offer means the session was never in the recipe-offer state — the client
            # is sending an accept without a prior offer (protocol violation or crafted
            # request).  A mismatched recipe_name means the client is trying to accept
            # a different recipe than the one offered — also rejected with 400.
            #
            # Slots are operator-editable by design (the operator fills unsatisfied
            # required slots and may rubber-stamp / override pre-fills); the binding
            # check does NOT compare slot values, only recipe_name.
            offered = guided.step_2_5_recipe_offer
            if offered is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "recipe_offer ['accept'] received but no recipe was staged for this session "
                        "(step_2_5_recipe_offer is None — the session has not reached the recipe-offer state "
                        "or the offer was already consumed)."
                    ),
                )
            if recipe_name != offered.recipe_name:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"recipe_offer ['accept'] recipe_name mismatch: "
                        f"client sent {recipe_name!r} but the server offered {offered.recipe_name!r}. "
                        "Accept must reference the recipe that was offered."
                    ),
                )

            # The slots must be passed as edited_values from the client.
            # The recipe_offer turn pre-populates partial slots; the user fills
            # the rest via the recipe_offer form and sends them back in edited_values.
            from elspeth.web.composer.guided.recipe_match import RecipeMatch as _RecipeMatch

            slots = dict(slots_raw)
            # The "accept" reconstruction is post-acceptance: the operator has
            # supplied the slot values (merged into ``edited.slots`` by the
            # widget).  ``unsatisfied_slots`` was a property of the *offer*;
            # at apply time ``handle_step_2_5_recipe_apply`` consumes only
            # ``recipe_name`` and ``slots``, so an empty mapping is correct.
            match = _RecipeMatch(recipe_name=recipe_name, slots=slots, unsatisfied_slots={})
            # Clear the staged offer atomically before passing to the handler
            # so it cannot be re-read by a later step.
            guided = _replace(guided, step_2_5_recipe_offer=None)

            handler_result = handle_step_2_5_recipe_apply(
                state=state,
                session=guided,
                match=match,
                catalog=catalog,
                data_dir=data_dir,
                session_engine=session_engine,
                session_id=session_id,
            )
            if not handler_result.tool_result.success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Recipe application failed: {handler_result.tool_result}",
                )
            state = handler_result.state
            guided = handler_result.session
            # terminal is now set on guided.terminal (TerminalKind.COMPLETED).
            return state, guided, None

        if chosen == ["build_manually"]:
            # step_advance already advanced to STEP_3.  Solve the chain via the LLM.
            if guided.step_1_result is None or guided.step_2_result is None:
                raise HTTPException(
                    status_code=500,
                    detail="Dispatcher invariant: step_1_result and step_2_result must be set at build_manually.",
                )
            source = guided.step_1_result
            sink = guided.step_2_result
            # I2: same auto-drop wrap as the no-recipe-match branch above.
            proposal, guided = await solve_chain_with_auto_drop(
                site="step_2_5_build_manually_solve",
                session=guided,
                session_id=session_id,
                user_id=user_id,
                composition_version=state.version,
                recorder=recorder,
                model=model,
                source=source,
                sink=sink,
                recipe_match=None,
            )
            if proposal is None:
                return state, guided, None
            guided = _replace(guided, step_3_proposal=proposal)
            next_turn = build_step_3_propose_chain_turn(proposal)
            new_record = TurnRecord(
                step=GuidedStep.STEP_3_TRANSFORMS,
                turn_type=TurnType.PROPOSE_CHAIN,
                payload_hash=stable_hash(next_turn["payload"]),
                response_hash=None,
                emitter="server",
            )
            emit_turn_emitted(
                recorder,
                step=GuidedStep.STEP_3_TRANSFORMS,
                turn_type=TurnType.PROPOSE_CHAIN,
                payload_hash=stable_hash(next_turn["payload"]),
                payload_payload_id="",
                emitter="server",
                composition_version=state.version,
                actor=user_id,
            )
            guided = _replace(guided, history=(*guided.history, new_record))
            return state, guided, next_turn

        raise HTTPException(
            status_code=400,
            detail=f"recipe_offer response must have chosen=['accept'] or chosen=['build_manually'], got {chosen!r}.",
        )

    # --- STEP_3_TRANSFORMS turns ----------------------------------------
    # The only response shape we handle in Phase 4 is ACCEPT on a
    # propose_chain turn.  Reject and clarifying SINGLE_SELECT responses are
    # deferred to Phase 5 (which adds re-solve, repair, and advisor flows).
    if current_step is GuidedStep.STEP_3_TRANSFORMS:
        if current_turn_type is TurnType.PROPOSE_CHAIN:
            chosen = list(turn_response["chosen"] or [])
            if chosen == ["accept"]:
                if guided.step_3_proposal is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Dispatcher invariant: step_3_proposal must be set when accepting a propose_chain turn.",
                    )
                handler_result = handle_step_3_chain_accept(
                    state=state,
                    session=guided,
                    proposal=guided.step_3_proposal,
                    catalog=catalog,
                    data_dir=data_dir,
                    session_engine=session_engine,
                    session_id=session_id,
                )
                if not handler_result.tool_result.success:
                    # Initial commit failed. Attempt one LLM repair: feed the
                    # validation errors back as a system-prompt addendum and
                    # ask the LLM to produce a corrected chain.
                    #
                    # Validation error text is Tier 1 audit data — taken verbatim
                    # from the ToolResult; no paraphrasing or fabrication.
                    failed_result = handler_result.tool_result
                    repair_context_lines = [e.message for e in failed_result.validation.errors]
                    if not repair_context_lines and failed_result.data is not None:
                        # No validation errors but a data-layer error message;
                        # use it verbatim so the LLM sees the actual fault.
                        raw_data = dict(failed_result.data)
                        if _DATA_ERROR_KEY in raw_data:
                            repair_context_lines = [str(raw_data[_DATA_ERROR_KEY])]
                    repair_context = "\n".join(repair_context_lines) or str(failed_result.validation)

                    # Obtain source/sink from the guided session for the repair call.
                    # Both must be non-None because Step 3 can only be reached after
                    # Steps 1 and 2 commit successfully (dispatcher invariant).
                    if guided.step_1_result is None:
                        raise InvariantError("repair: step_1_result is None — STEP_3 unreachable without Step 1 commit")
                    if guided.step_2_result is None:
                        raise InvariantError("repair: step_2_result is None — STEP_3 unreachable without Step 2 commit")
                    # I2: a transient failure during repair auto-drops
                    # IMMEDIATELY — repair is already attempt #2, so a
                    # transient on repair means we are done trying. The
                    # early ``return`` below short-circuits BEFORE the
                    # downstream ``mark_solver_exhausted`` path so we never
                    # double-emit ``guided_dropped_to_freeform``.
                    repair_proposal, guided = await solve_chain_with_auto_drop(
                        site="step_3_repair_solve",
                        session=guided,
                        session_id=session_id,
                        user_id=user_id,
                        composition_version=state.version,
                        recorder=recorder,
                        model=model,
                        source=guided.step_1_result,
                        sink=guided.step_2_result,
                        recipe_match=None,
                        repair_context=repair_context,
                    )
                    if repair_proposal is None:
                        return state, guided, None
                    repair_result = handle_step_3_chain_accept(
                        # state is the original pre-attempt state — _execute_set_pipeline
                        # is validate-then-mutate: on failure the state is untouched,
                        # so handler_result.state is the same object as `state`.
                        state=state,
                        session=guided,
                        proposal=repair_proposal,
                        catalog=catalog,
                        data_dir=data_dir,
                        session_engine=session_engine,
                        session_id=session_id,
                    )
                    if repair_result.tool_result.success:
                        # Repair succeeded: wizard completes normally.
                        return repair_result.state, repair_result.session, None

                    # Repair also failed. Mark solver exhausted and auto-drop to freeform.
                    # Build the validation_result dict from the repair failure (Tier 1
                    # data — only real fields from ToolResult, no fabrication).
                    # The repair always runs on the auto-drop path; validation_result is
                    # always present per spec §9.1 (set when drop_reason="solver_exhausted").
                    # An empty errors list is real audit data (the validator rejected without
                    # producing structured errors), not fabrication.
                    repair_validation = repair_result.tool_result.validation
                    validation_result_payload: dict[str, Any] = {
                        "is_valid": repair_validation.is_valid,
                        "errors": [e.to_dict() for e in repair_validation.errors],
                    }

                    new_guided, _terminal, directives = mark_solver_exhausted(
                        session=guided,
                        validation_result=validation_result_payload,
                    )
                    for directive in directives:
                        if directive.tool_name == "guided_dropped_to_freeform":
                            args = dict(directive.arguments)
                            emit_dropped_to_freeform(
                                recorder,
                                prev=GuidedStep(args["prev_step"]),
                                drop_reason=TerminalReason(args["drop_reason"]),
                                validation_result=args["validation_result"],
                                composition_version=state.version,
                                actor=user_id,
                            )
                    # Return 200 with terminal — auto-drop is a clean wizard outcome.
                    return state, new_guided, None

                # handler_result.session.terminal is COMPLETED on success.
                return handler_result.state, handler_result.session, None
            if chosen == ["reject"]:
                raise HTTPException(
                    status_code=501,
                    detail=(
                        "Step 3 chain rejection is not yet implemented — Phase 5 will add "
                        "re-solve and repair flows. Use exit-to-freeform to drop to freeform mode."
                    ),
                )
            raise HTTPException(
                status_code=400,
                detail=f"propose_chain response must have chosen=['accept'] or chosen=['reject'], got {chosen!r}.",
            )
        # SINGLE_SELECT clarifying-question response at STEP_3 — Phase 5.
        raise HTTPException(
            status_code=501,
            detail="Step 3 clarifying question handling is not yet implemented — Phase 5.",
        )

    # Unhandled branch — this is a dispatcher gap, not a user error.
    raise InvariantError(
        f"_dispatch_guided_respond: unhandled branch "
        f"current_step={current_step!r}, current_turn_type={current_turn_type!r}, "
        f"guided.step={guided.step!r}"
    )


def create_session_router() -> APIRouter:
    """Create the session router with /api/sessions prefix."""
    router = APIRouter(prefix="/api/sessions", tags=["sessions"])

    @router.post("", status_code=201, response_model=SessionResponse)
    async def create_session(
        body: CreateSessionRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> SessionResponse:
        """Create a new session for the authenticated user."""
        service = request.app.state.session_service
        settings = request.app.state.settings
        session = await service.create_session(
            user.user_id,
            body.title,
            settings.auth_provider,
        )
        return _session_response(session)

    @router.get("", response_model=list[SessionResponse])
    async def list_sessions(
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> list[SessionResponse]:
        """List sessions for the authenticated user."""
        service = request.app.state.session_service
        settings = request.app.state.settings
        sessions = await service.list_sessions(
            user.user_id,
            settings.auth_provider,
            limit=limit,
            offset=offset,
        )
        return [_session_response(s) for s in sessions]

    # NOTE: Registered before "/{session_id}" so FastAPI matches "_active"
    # against this exact-path route rather than attempting to parse "_active"
    # as a UUID (which would 422). The leading underscore also guarantees
    # the path can never collide with a real session id (UUIDs only contain
    # hex digits and hyphens).
    @router.get("/_active", response_model=list[ComposerProgressSnapshot])
    async def list_active_composer_requests(
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> list[ComposerProgressSnapshot]:
        """List in-flight composer requests for the authenticated user.

        Closes the operator-visibility gap captured in the source report:
        Uvicorn's access log only writes the POST line when the response
        completes, so an in-flight or client-cancelled composer request
        was previously invisible to operators unless they polled
        ``/composer-progress`` for a specific session id.

        Returns snapshots whose phase is in NON_TERMINAL_PROGRESS_PHASES
        (starting / calling_model / using_tools / validating / saving),
        ordered by ``updated_at`` ascending so the longest-running request
        is at the top — typical triage starting point. Filtered by the
        authenticated user's id against the registry's internal user
        index, so a caller cannot see other users' active sessions even
        when they share a server.

        ``cancelled``, ``failed``, ``complete``, and ``idle`` snapshots
        are intentionally excluded from this view: those requests are no
        longer in flight and the per-session ``/composer-progress`` GET
        is the right surface for inspecting a terminal outcome.
        """
        registry = _get_composer_progress_registry(request)
        snapshots = await registry.list_active(user_id=str(user.user_id))
        return list(snapshots)

    @router.get("/{session_id}", response_model=SessionResponse)
    async def get_session(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> SessionResponse:
        """Get a single session. IDOR-protected."""
        session = await _verify_session_ownership(session_id, user, request)
        return _session_response(session)

    @router.get(
        "/{session_id}/composer-progress",
        response_model=ComposerProgressSnapshot,
    )
    async def get_composer_progress(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> ComposerProgressSnapshot:
        """Return the latest provider-safe composer progress for a session."""
        session = await _verify_session_ownership(session_id, user, request)
        registry = _get_composer_progress_registry(request)
        return await registry.get_latest(str(session.id))

    @router.delete("/{session_id}", status_code=204)
    async def delete_session(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> None:
        """Archive (delete) a session and all associated data.

        Rejects deletion while a pipeline run is active — archive_session()
        would delete run rows and blob directories out from under the
        background worker, causing status update failures and data loss.
        """
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service

        active_run = await service.get_active_run(session.id)
        if active_run is not None:
            raise HTTPException(
                status_code=409,
                detail="Cannot delete session while a pipeline run is active. Cancel the run first.",
            )

        try:
            await service.archive_session(session.id)
        finally:
            # Clean up ephemeral per-session state regardless of archive outcome.
            # If archive fails, the session still exists and a retry will re-enter
            # this path. The lock cleanup is idempotent.
            execution_service = request.app.state.execution_service
            execution_service.cleanup_session_lock(str(session.id))
            compose_lock_registry = _get_session_compose_lock_registry(request)
            await compose_lock_registry.cleanup_session_lock(str(session.id))
            progress_registry = _get_composer_progress_registry(request)
            await progress_registry.clear(str(session.id))

    @router.post(
        "/{session_id}/messages",
        response_model=MessageWithStateResponse,
    )
    async def send_message(
        session_id: UUID,
        body: SendMessageRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        rate_limiter: ComposerRateLimiter = Depends(get_rate_limiter),  # noqa: B008
    ) -> MessageWithStateResponse:
        """Send a user message, run the LLM composer, persist results.

        1. Rate limit check (per-user).
        2. Load or create the current CompositionState (pre-send provenance).
        3. Persist the user message with pre-send state_id.
        4. Pre-fetch chat history for the composer.
        5. Run the LLM composition loop.
        6. Save state if version changed (post-compose provenance).
        7. Persist the assistant response message with post-compose state_id.
        8. Return the assistant message and (optionally) the new state.
        """
        # 0. Rate limit check — before any work
        await rate_limiter.check(user.user_id)

        session = await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        settings = request.app.state.settings
        compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session.id))
        async with compose_lock:
            # 1. Load or create CompositionState — needed before user message
            #    for pre-send provenance (AD-7: user msg records what user saw).
            state_record = await service.get_current_state(session.id)
            if state_record is None:
                state = _initial_composition_state_with_guided_session()
                pre_send_state_id: UUID | None = None
            else:
                state = _state_from_record(state_record)
                # If client provided a state_id, verify it belongs to this session.
                # Use client-asserted state for provenance (AD-2) — it reflects
                # what the user was looking at, which may differ from current if
                # another tab mutated state.
                if body.state_id is not None:
                    client_state_id = body.state_id
                    # Two 404 paths below return byte-identical bodies by
                    # design. The commit that introduced this validation
                    # called the RuntimeError/ValueError mapping
                    # "load-bearing ... to avoid leaking other sessions'
                    # state existence" — but distinguishable 404 *details*
                    # would leak exactly that (an attacker could observe
                    # "State not found" vs "State not found for this
                    # session" and conclude the UUID exists in some OTHER
                    # user's session, which is the IDOR information leak
                    # the check exists to prevent). Keep both details
                    # identical; if a future refactor needs diagnostic
                    # precision, route it through structured audit/
                    # telemetry (server-side only), not through the HTTP
                    # response body.
                    try:
                        client_state = await service.get_state(client_state_id)
                    except ValueError:
                        raise HTTPException(
                            status_code=404,
                            detail="State not found",
                        ) from None
                    if client_state.session_id != session.id:
                        raise HTTPException(
                            status_code=404,
                            detail="State not found",
                        )
                    pre_send_state_id = client_state_id
                else:
                    pre_send_state_id = state_record.id

            # 1b. Detect guided→freeform mode transition (spec §8.2).
            # The first freeform chat turn after guided_session.terminal is set
            # uses a layered system prompt. ``transition_consumed`` guards against
            # re-firing on subsequent turns.
            _guided = state.guided_session
            _guided_terminal_for_compose = (
                _guided.terminal if (_guided is not None and _guided.terminal is not None and not _guided.transition_consumed) else None
            )

            # 2. Persist user message with pre-send provenance.
            # Keep the inserted row so the subsequent snapshot can prove
            # it is composing against the transcript that actually ends
            # at this request's user turn.
            user_msg = await service.add_message(
                session.id,
                "user",
                body.content,
                composition_state_id=pre_send_state_id,
                writer_principal="route_user_message",
            )
            progress_registry = _get_composer_progress_registry(request)
            progress_sink = _composer_progress_sink(
                progress_registry,
                session_id=str(session.id),
                request_id=str(user_msg.id),
                user_id=str(user.user_id),
            )
            await _publish_progress(
                progress_registry,
                session_id=str(session.id),
                request_id=str(user_msg.id),
                user_id=str(user.user_id),
                event=ComposerProgressEvent(
                    phase="starting",
                    headline="I'm reading your request and current pipeline.",
                    evidence=("The request was accepted for this session.",),
                    likely_next="ELSPETH will prepare the composer prompt with the current pipeline.",
                ),
            )

            _COMPOSER_REQUESTS_INFLIGHT.add(1, {"endpoint": "send_message"})
            terminal_status: _ComposerRequestTerminalStatus = "failed"
            try:
                # 3. Pre-fetch chat history as plain dicts (seam contract B)
                # Pass limit=None to fetch the full conversation — the default
                # limit=100 would silently drop recent context once a session
                # exceeds 100 turns, causing the LLM to lose conversation state.
                # Exclude the just-persisted user message — the composer receives
                # it separately via body.content and appends it in _build_messages.
                records = await service.get_messages(session.id, limit=None)
                if not records or records[-1].id != user_msg.id:
                    raise AuditIntegrityError(
                        "Tier 1 audit anomaly: send_message transcript snapshot "
                        f"for session {session.id} does not end at inserted user "
                        f"message {user_msg.id}. Refusing to compose against "
                        "interleaved session history."
                    )
                chat_messages = _composer_chat_history(records[:-1])

                # 4. Run the LLM composition loop
                composer: ComposerService = request.app.state.composer_service
                from litellm.exceptions import APIError as LiteLLMAPIError
                from litellm.exceptions import AuthenticationError as LiteLLMAuthError

                try:
                    result = await composer.compose(
                        body.content,
                        chat_messages,
                        state,
                        session_id=str(session_id),
                        user_id=str(user.user_id),
                        progress=progress_sink,
                        guided_terminal=_guided_terminal_for_compose,
                    )
                except ComposerConvergenceError as exc:
                    terminal_status = "timed_out" if exc.budget_exhausted == "timeout" else "failed"
                    # Discriminate the three sub-causes (composition / discovery /
                    # wall-clock timeout) using exc.budget_exhausted. Without this
                    # dispatch the three failure modes would collapse into a single
                    # generic event — the original bug filed as elspeth-5030f7373d.
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=str(user_msg.id),
                        user_id=str(user.user_id),
                        event=convergence_progress_event(budget_exhausted=exc.budget_exhausted),
                    )
                    response_body = await _handle_convergence_error(
                        exc,
                        service,
                        session.id,
                        str(user.user_id),
                        "convergence",
                        pre_send_state_id,
                        settings=settings,
                        secret_service=request.app.state.scoped_secret_resolver,
                    )
                    raise HTTPException(status_code=422, detail=response_body) from exc
                except LiteLLMAuthError as exc:
                    # ``str(exc)`` on LiteLLM exceptions can embed the provider
                    # name, model ID, request payload fragments, and — on
                    # certain provider code paths — the upstream HTTP response
                    # body, which has been observed to echo the Authorization
                    # header.  Redact the HTTP ``detail`` field to the class
                    # name only; route the full exception to structured
                    # server-side logging via ``slog.error`` with session
                    # correlation.  Mirrors the ``partial_state_save_error``
                    # contract on the SQLAlchemy 422 path in
                    # ``_handle_convergence_error`` above.
                    # exc_info deliberately omitted for the same reason
                    # SQLAlchemy ``exc_info`` is dropped in the canonical
                    # narrow-catch sites: ``__cause__`` chains on these
                    # exception classes can carry upstream provider detail
                    # that must not be retained in structured logs either.
                    slog.error(
                        "compose_llm_auth_error",
                        session_id=str(session_id),
                        exc_class=type(exc).__name__,
                    )
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=str(user_msg.id),
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="failed",
                            headline="The composer model is not available.",
                            evidence=("The model provider rejected the composer request.",),
                            likely_next="Check the composer provider configuration before retrying.",
                            reason="provider_auth_failed",
                        ),
                    )
                    llm_calls = _llm_calls_from_exception(exc)
                    if llm_calls:
                        await _persist_llm_calls(service, session.id, llm_calls, pre_send_state_id, plugin_crash_pending=True)
                    raise HTTPException(
                        status_code=502,
                        detail=_litellm_error_detail(
                            "llm_auth_error",
                            exc,
                            expose_provider_error=settings.composer_expose_provider_errors,
                        ),
                    ) from exc
                except LiteLLMAPIError as exc:
                    # Same redaction rationale as the auth-error block above.
                    # ``LiteLLMAPIError`` message shape varies by provider (OpenAI,
                    # Azure OpenAI, Anthropic, Bedrock) and can include
                    # rate-limit window details, account/tenant identifiers,
                    # and upstream request IDs that are operator-only material.
                    slog.error(
                        "compose_llm_unavailable",
                        session_id=str(session_id),
                        exc_class=type(exc).__name__,
                    )
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=str(user_msg.id),
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="failed",
                            headline="The composer model is temporarily unavailable.",
                            evidence=("The model provider did not complete the request.",),
                            likely_next="Retry when the provider is available.",
                            reason="provider_unavailable",
                        ),
                    )
                    llm_calls = _llm_calls_from_exception(exc)
                    if llm_calls:
                        await _persist_llm_calls(service, session.id, llm_calls, pre_send_state_id, plugin_crash_pending=True)
                    raise HTTPException(
                        status_code=502,
                        detail=_litellm_error_detail(
                            "llm_unavailable",
                            exc,
                            expose_provider_error=settings.composer_expose_provider_errors,
                        ),
                    ) from exc
                except ComposerPluginCrashError as crash:
                    # Plugin-crash path: _compose_loop wraps any non-ToolArgumentError
                    # escape from execute_tool into ComposerPluginCrashError carrying
                    # partial_state — the accumulated mutations from earlier successful
                    # tool calls within the same request. _handle_plugin_crash persists
                    # that state into composition_states symmetrically with the
                    # convergence-error path, so recompose does not lose those
                    # mutations. The HTTP response body is fully redacted; the cause
                    # chain is preserved via `from crash.original_exc` for the ASGI /
                    # server-level error machinery only.
                    #
                    # MUST be caught BEFORE the generic `except ComposerServiceError`
                    # below — ComposerPluginCrashError inherits from
                    # ComposerServiceError (so it isn't caught by a later bare
                    # Exception or mistakenly promoted by the route's convergence
                    # handler), and Python evaluates except clauses top-to-bottom.
                    # Inverting this order routes plugin crashes into the 502
                    # composer_error branch, re-introducing the silent-laundering
                    # behaviour this plan exists to eliminate.
                    #
                    # The same precedence rule applies to the
                    # `except ComposerRuntimePreflightError` clause below:
                    # ComposerRuntimePreflightError also inherits from
                    # ComposerServiceError, and demoting it past the generic
                    # 502 catch would convert a recoverable preflight failure
                    # (with persisted partial_state) into an opaque 502 with
                    # no audit trail. Both narrow catches MUST precede the
                    # generic ComposerServiceError catch.
                    response_body = await _handle_plugin_crash(
                        crash,
                        service,
                        session.id,
                        str(user.user_id),
                        "compose",
                        pre_send_state_id,
                        settings=settings,
                        secret_service=request.app.state.scoped_secret_resolver,
                    )
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=str(user_msg.id),
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="failed",
                            headline="The composer could not safely finish this request.",
                            evidence=("A pipeline tool failed on the server side.",),
                            likely_next="Review the visible error message, then retry after the issue is resolved.",
                            reason="plugin_crash",
                        ),
                    )
                    raise HTTPException(status_code=500, detail=response_body) from crash.original_exc
                except ComposerRuntimePreflightError as rpf_exc:
                    # Path 1 (cached preflight): composer.compose() re-raised a
                    # previously-cached runtime-preflight failure via
                    # _raise_cached_runtime_preflight_failure in
                    # web/composer/service.py. The shared
                    # _handle_runtime_preflight_failure helper persists
                    # rpf_exc.partial_state symmetrically with
                    # _handle_plugin_crash so accumulated tool-call mutations
                    # are not silently dropped from the audit trail. The
                    # SAME helper is invoked from the post-compose
                    # state-save catch below for path 2.
                    #
                    # Telemetry primacy (elspeth-0891e8da73): the cached
                    # raise site does NOT have a paired primary emission
                    # inside _state_data_from_composer_state (path-2's
                    # raise arm) because the failure originates inside
                    # composer.compose() before that helper is reached.
                    # Emit cached_preflight here so dashboards filtering
                    # composer.runtime_preflight.total{result=exception}
                    # do not silently under-count cache-hit re-raises —
                    # particularly the "no LLM mutation before re-raise"
                    # case, where the recovery handler also short-circuits
                    # past its persist_invalid emission. The outer catch
                    # is unambiguously the cached path: path-2 is caught
                    # inline around the post-compose
                    # _state_data_from_composer_state call below.
                    _record_composer_runtime_preflight_telemetry(
                        "exception",
                        source="cached_preflight",
                        exception_class=rpf_exc.exc_class,
                    )
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=str(user_msg.id),
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="failed",
                            headline="The composer could not safely finish this request.",
                            evidence=("Runtime preflight failed before the compose loop returned.",),
                            likely_next="Review the visible error message, then retry after the issue is resolved.",
                            reason="runtime_preflight_failed",
                        ),
                    )
                    response_body = await _handle_runtime_preflight_failure(
                        rpf_exc,
                        service,
                        session.id,
                        str(user.user_id),
                        "compose",
                        pre_send_state_id,
                        settings=settings,
                        secret_service=request.app.state.scoped_secret_resolver,
                    )
                    raise HTTPException(status_code=500, detail=response_body) from rpf_exc.original_exc
                except ComposerServiceError as exc:
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=str(user_msg.id),
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="failed",
                            headline="The composer could not finish this request.",
                            evidence=("Prompt preparation or composer service setup failed.",),
                            likely_next="Retry once the composer service is available.",
                            reason="service_setup_failed",
                        ),
                    )
                    llm_calls = _llm_calls_from_exception(exc)
                    if llm_calls:
                        await _persist_llm_calls(service, session.id, llm_calls, pre_send_state_id, plugin_crash_pending=True)
                    raise HTTPException(
                        status_code=502,
                        detail={"error_type": "composer_error", "detail": str(exc)},
                    ) from exc

                # 5. Save state if version changed — post-compose provenance.
                #
                # Path 2 (post-compose runtime preflight): the call to
                # _state_data_from_composer_state below uses
                # preflight_exception_policy="raise", which raises
                # ComposerRuntimePreflightError when the post-compose
                # preflight crashes. That raise site sits OUTSIDE the
                # compose-time try/except above, so the post-compose block is
                # wrapped in its own try/except that delegates to the same
                # shared helper. Without this wrapper, a path-2 raise escapes
                # as Starlette's bare 500 — partial_state is dropped from
                # the audit trail and the frontend receives an opaque error.
                #
                # 5a. Compute the post-compose guided_session.
                # If the transition prompt fired this turn, flip transition_consumed
                # on the guided_session so subsequent turns use the freeform-only
                # prompt.  The updated guided_session is included in composer_meta
                # for both the version-changed save path and the version-unchanged
                # standalone save below.
                _post_compose_guided: GuidedSession | None = result.state.guided_session
                if _guided_terminal_for_compose is not None:
                    # transition_consumed flip — _guided is non-None because
                    # _guided_terminal_for_compose was derived from _guided.terminal.
                    # The explicit RuntimeError defends against an impossible state
                    # (the gate above ensures _guided is not None when this fires)
                    # and satisfies the type checker without defensive get() calls.
                    if _guided is None:
                        raise InvariantError(
                            "guided_terminal_for_compose is set but guided_session is None — "
                            "impossible state: transition gate should have blocked this path"
                        )
                    from dataclasses import replace as _replace_dc

                    _post_compose_guided = _replace_dc(
                        _guided,
                        transition_consumed=True,
                    )

                # 5b. Compute the composer_meta that carries both repair_turns_used
                # and guided_session.  Guided_session rides in composer_meta (not a
                # first-class column) so any save must propagate it forward — failing
                # to include it would silently drop the guided session from the DB on
                # every freeform compose turn that mutates state.
                _post_compose_meta: dict[str, Any] = {"repair_turns_used": result.repair_turns_used}
                if _post_compose_guided is not None:
                    _post_compose_meta["guided_session"] = _post_compose_guided.to_dict()

                state_response: CompositionStateResponse | None = None
                post_compose_state_id: UUID | None = pre_send_state_id
                if result.state.version != state.version:
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=str(user_msg.id),
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="validating",
                            headline="The composer has updated the pipeline and is validating the result.",
                            evidence=("The updated pipeline state is being checked before persistence.",),
                            likely_next="ELSPETH will save the validated pipeline snapshot.",
                        ),
                    )
                    try:
                        state_data, validation = await _state_data_from_composer_state(
                            result.state,
                            settings=settings,
                            secret_service=request.app.state.scoped_secret_resolver,
                            user_id=str(user.user_id),
                            runtime_preflight=result.runtime_preflight,
                            preflight_exception_policy="raise",
                            initial_version=state.version,
                            telemetry_source="compose",
                            composer_meta=_post_compose_meta,
                        )
                    except ComposerRuntimePreflightError as rpf_exc:
                        rpf_exc = ComposerRuntimePreflightError(
                            original_exc=rpf_exc.original_exc,
                            partial_state=rpf_exc.partial_state,
                            tool_invocations=result.tool_invocations,
                            llm_calls=result.llm_calls,
                        )
                        # UX-distinct from path 1 above: the LLM call already
                        # succeeded, so the failure messaging frames this as
                        # a validation/persistence-stage failure rather than
                        # a compose-stage failure.
                        await _publish_progress(
                            progress_registry,
                            session_id=str(session.id),
                            request_id=str(user_msg.id),
                            user_id=str(user.user_id),
                            event=ComposerProgressEvent(
                                phase="failed",
                                headline="The composer could not safely validate the pipeline update.",
                                evidence=("Runtime preflight failed during state persistence.",),
                                likely_next="Review the visible error message, then retry after the issue is resolved.",
                                reason="runtime_preflight_failed",
                            ),
                        )
                        response_body = await _handle_runtime_preflight_failure(
                            rpf_exc,
                            service,
                            session.id,
                            str(user.user_id),
                            "compose",
                            pre_send_state_id,
                            settings=settings,
                            secret_service=request.app.state.scoped_secret_resolver,
                        )
                        raise HTTPException(status_code=500, detail=response_body) from rpf_exc.original_exc
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=str(user_msg.id),
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="saving",
                            headline="ELSPETH is saving the pipeline update.",
                            evidence=("A new composition state version is being stored for this session.",),
                            likely_next="The assistant response will appear after the save completes.",
                        ),
                    )
                    new_state_record = await service.save_composition_state(
                        session.id,
                        state_data,
                        # Preserves pre-fix labelling. This call site (post-
                        # compose path 1, send_message) wrote ``session_seed``
                        # under the previous hardcoded label and continues to
                        # do so here. The mismatch between this label and the
                        # actual writer category (post-compose state advance,
                        # not session create / branch reseed) is a SEPARATE
                        # mis-attribution from the three handler sites that
                        # commit elspeth-obs-f217c634aa addresses; widening
                        # this commit to relabel post-compose paths would
                        # require its own spec amendment + observation.
                        provenance="session_seed",
                    )
                    state_response = _state_response(new_state_record, live_validation=validation)
                    post_compose_state_id = new_state_record.id
                elif _guided_terminal_for_compose is not None and _post_compose_guided is not None:
                    # Version unchanged but transition_consumed must be flipped.
                    # Persist the updated guided_session in a new state row so
                    # subsequent turns pick up transition_consumed=True.
                    _existing_meta: dict[str, Any] = {}
                    if state_record is not None and state_record.composer_meta is not None:
                        _existing_meta = dict(deep_thaw(state_record.composer_meta))
                    _transition_meta = {**_existing_meta, **_post_compose_meta}
                    _transition_state = result.state
                    _transition_state_d = _transition_state.to_dict()
                    _transition_state_data = CompositionStateData(
                        source=_transition_state_d["source"],
                        nodes=_transition_state_d["nodes"],
                        edges=_transition_state_d["edges"],
                        outputs=_transition_state_d["outputs"],
                        metadata_=_transition_state_d["metadata"],
                        is_valid=False,
                        validation_errors=None,
                        composer_meta=_transition_meta,
                    )
                    _transition_record = await service.save_composition_state(
                        session.id,
                        _transition_state_data,
                        # Mirrors the paired session_seed-labelled site immediately
                        # above (post-compose path 1, transition_consumed flip).
                        # Same known mis-attribution as that paired site — see the
                        # comment block at the earlier save call for the
                        # elspeth-obs-f217c634aa relabelling history.
                        provenance="session_seed",
                    )
                    post_compose_state_id = _transition_record.id

                # 6. Persist assistant message with post-compose provenance
                assistant_msg = await service.add_message(
                    session.id,
                    "assistant",
                    result.message,
                    composition_state_id=post_compose_state_id,
                    raw_content=result.raw_assistant_content,
                    writer_principal="compose_loop",
                )
                # 6b. Persist per-tool-call audit trail. Each ComposerToolInvocation
                # lands as one role=tool chat message linked to the post-compose
                # state id (when version advanced) so the audit trail records
                # which tool calls produced this state.
                if result.tool_invocations and not result.persisted_tool_call_turn:
                    await _persist_tool_invocations(
                        service,
                        session.id,
                        result.tool_invocations,
                        post_compose_state_id,
                        parent_assistant_id=assistant_msg.id,
                        plugin_crash_pending=False,
                    )
                if result.llm_calls:
                    await _persist_llm_calls(
                        service,
                        session.id,
                        result.llm_calls,
                        pre_send_state_id,
                        plugin_crash_pending=False,
                    )
                await _publish_progress(
                    progress_registry,
                    session_id=str(session.id),
                    request_id=str(user_msg.id),
                    user_id=str(user.user_id),
                    event=ComposerProgressEvent(
                        phase="complete",
                        headline="The composer has updated the pipeline."
                        if result.state.version != state.version
                        else "The composer response is ready.",
                        evidence=("The assistant response has been saved for this session.",),
                        likely_next="Review the response and current pipeline.",
                        reason="composer_complete",
                    ),
                )

                # 7. Return response.
                #
                # Build the response object FIRST so a Pydantic packaging
                # failure (very unlikely — the inputs are already shaped
                # UUID/string fields, but contractually possible) does not
                # poison the terminal counter with a misleading
                # ``status=completed`` for a request that ultimately 500'd.
                # The flag flips only once the response is fully
                # constructed and ready to hand back to FastAPI.
                response = MessageWithStateResponse(
                    message=_message_response(assistant_msg),
                    state=state_response,
                )
                terminal_status = "completed"
                return response
            except InvariantError as exc:
                # Same B1-sanitization rationale as the /guided/respond
                # step_advance and dispatch handlers: server-invariant
                # violations route through a static 500 detail and a
                # structured slog event so on-call dashboards can filter on
                # ``guided.invariant_violated``.  Without this handler an
                # InvariantError raised from the post-compose transition_consumed
                # impossible-state guard would land at FastAPI's default 500
                # ({"detail": "Internal Server Error"}) with no structured log.
                slog.error(
                    "guided.invariant_violated",
                    session_id=str(session_id),
                    user_id=user.user_id,
                    exc_class=type(exc).__name__,
                    site="send_message",
                    frames=_safe_frame_strings(exc),
                )
                raise HTTPException(
                    status_code=500,
                    detail="Server invariant violated. See application audit log for diagnostic detail.",
                ) from exc
            except asyncio.CancelledError as exc:
                # Client-disconnect or operator cancel during the
                # composer-engaged window. Publish a discriminated
                # ``cancelled`` snapshot under ``asyncio.shield`` so the
                # registry update reaches /_active and per-session pollers
                # even though the outer task is being torn down. The
                # nested except absorbs the CancelledError that ``await
                # asyncio.shield`` re-raises on the cancelling task — the
                # shielded coroutine itself runs to completion in the
                # background.
                llm_calls = _llm_calls_from_exception(exc)
                if llm_calls:
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(
                            _persist_llm_calls(
                                service,
                                session.id,
                                llm_calls,
                                pre_send_state_id,
                                plugin_crash_pending=True,
                            )
                        )
                with contextlib.suppress(asyncio.CancelledError):
                    # The shielded publish runs to completion in the
                    # background; the outer await re-raises CancelledError
                    # on the cancelling task, which we deliberately swallow
                    # because we already know we're being cancelled and
                    # ``raise`` two lines below restores the cancel chain.
                    await asyncio.shield(
                        _publish_progress(
                            progress_registry,
                            session_id=str(session.id),
                            request_id=str(user_msg.id),
                            user_id=str(user.user_id),
                            event=client_cancelled_progress_event(),
                        )
                    )
                terminal_status = "cancelled"
                raise
            finally:
                _COMPOSER_REQUESTS_INFLIGHT.add(-1, {"endpoint": "send_message"})
                _record_composer_request_terminal(terminal_status, endpoint="send_message")

    @router.post(
        "/{session_id}/recompose",
        response_model=MessageWithStateResponse,
    )
    async def recompose(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        rate_limiter: ComposerRateLimiter = Depends(get_rate_limiter),  # noqa: B008
    ) -> MessageWithStateResponse:
        """Re-run the composer without inserting a new user message.

        Used by the frontend retry flow when the original send_message
        persisted the user message but the composer failed. Skips step 2
        of send_message (user message insertion) and uses the existing
        conversation history.
        """
        await rate_limiter.check(user.user_id)
        session = await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        settings = request.app.state.settings
        compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session.id))
        async with compose_lock:
            # Load current state
            state_record = await service.get_current_state(session.id)
            if state_record is None:
                state = _initial_composition_state_with_guided_session()
                pre_send_state_id: UUID | None = None
            else:
                state = _state_from_record(state_record)
                pre_send_state_id = state_record.id

            # Fetch full chat history. Audit-only tool rows can trail a failed
            # user turn, so the recompose precondition is the last
            # conversational message rather than the last persisted row.
            # Reject if the conversation does not end at a user turn; blindly
            # dropping the final conversational row would corrupt the transcript.
            records = await service.get_messages(session.id, limit=None)
            conversation_records = _composer_conversation_messages(records)
            if not conversation_records:
                raise HTTPException(status_code=400, detail="No messages to recompose from")
            if conversation_records[-1].role != "user":
                raise HTTPException(
                    status_code=409,
                    detail="Cannot recompose: the last message is not a user message. "
                    "Recompose is only valid when the most recent message is the "
                    "user turn whose composition failed.",
                )

            last_user_content = conversation_records[-1].content
            request_id = str(conversation_records[-1].id)
            progress_registry = _get_composer_progress_registry(request)
            progress_sink = _composer_progress_sink(
                progress_registry,
                session_id=str(session.id),
                request_id=request_id,
                user_id=str(user.user_id),
            )
            await _publish_progress(
                progress_registry,
                session_id=str(session.id),
                request_id=request_id,
                user_id=str(user.user_id),
                event=ComposerProgressEvent(
                    phase="starting",
                    headline="I'm rereading your request and current pipeline.",
                    evidence=("The retry was accepted for this session.",),
                    likely_next="ELSPETH will prepare the composer prompt with the current pipeline.",
                ),
            )
            # Detect guided→freeform mode transition (spec §8.2).
            # Recompose is a retried freeform chat call — progressive disclosure
            # fires here on the same semantics as send_message (first freeform
            # turn after guided_session.terminal is set uses the layered prompt).
            _guided = state.guided_session
            _guided_terminal_for_compose = (
                _guided.terminal if (_guided is not None and _guided.terminal is not None and not _guided.transition_consumed) else None
            )

            _COMPOSER_REQUESTS_INFLIGHT.add(1, {"endpoint": "recompose"})
            terminal_status: _ComposerRequestTerminalStatus = "failed"
            try:
                # Exclude the last user message — the composer receives it
                # separately via the message arg and appends it in _build_messages.
                chat_messages = _composer_chat_history(conversation_records[:-1])

                # Run the LLM composition loop
                composer: ComposerService = request.app.state.composer_service
                from litellm.exceptions import APIError as LiteLLMAPIError
                from litellm.exceptions import AuthenticationError as LiteLLMAuthError

                try:
                    result = await composer.compose(
                        last_user_content,
                        chat_messages,
                        state,
                        session_id=str(session_id),
                        user_id=str(user.user_id),
                        progress=progress_sink,
                        guided_terminal=_guided_terminal_for_compose,
                    )
                except ComposerConvergenceError as exc:
                    terminal_status = "timed_out" if exc.budget_exhausted == "timeout" else "failed"
                    # Same three-sub-cause discriminator as the /messages catch
                    # above — recompose mirrors send_message exactly so the two
                    # routes cannot drift on failure UX.
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=request_id,
                        user_id=str(user.user_id),
                        event=convergence_progress_event(budget_exhausted=exc.budget_exhausted),
                    )
                    response_body = await _handle_convergence_error(
                        exc,
                        service,
                        session.id,
                        str(user.user_id),
                        "recompose_convergence",
                        pre_send_state_id,
                        settings=settings,
                        secret_service=request.app.state.scoped_secret_resolver,
                    )
                    raise HTTPException(status_code=422, detail=response_body) from exc
                except LiteLLMAuthError as exc:
                    # Recompose mirror of the redaction contract in send_message
                    # (see block comment there for full rationale).  The two
                    # paths MUST carry byte-identical response shapes and
                    # redaction granularity — any future divergence becomes a
                    # selective leak surface (attacker picks whichever endpoint
                    # still echoes str(exc)).
                    slog.error(
                        "recompose_llm_auth_error",
                        session_id=str(session_id),
                        exc_class=type(exc).__name__,
                    )
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=request_id,
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="failed",
                            headline="The composer model is not available.",
                            evidence=("The model provider rejected the composer request.",),
                            likely_next="Check the composer provider configuration before retrying.",
                            reason="provider_auth_failed",
                        ),
                    )
                    llm_calls = _llm_calls_from_exception(exc)
                    if llm_calls:
                        await _persist_llm_calls(service, session.id, llm_calls, pre_send_state_id, plugin_crash_pending=True)
                    raise HTTPException(
                        status_code=502,
                        detail=_litellm_error_detail(
                            "llm_auth_error",
                            exc,
                            expose_provider_error=settings.composer_expose_provider_errors,
                        ),
                    ) from exc
                except LiteLLMAPIError as exc:
                    slog.error(
                        "recompose_llm_unavailable",
                        session_id=str(session_id),
                        exc_class=type(exc).__name__,
                    )
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=request_id,
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="failed",
                            headline="The composer model is temporarily unavailable.",
                            evidence=("The model provider did not complete the request.",),
                            likely_next="Retry when the provider is available.",
                            reason="provider_unavailable",
                        ),
                    )
                    llm_calls = _llm_calls_from_exception(exc)
                    if llm_calls:
                        await _persist_llm_calls(service, session.id, llm_calls, pre_send_state_id, plugin_crash_pending=True)
                    raise HTTPException(
                        status_code=502,
                        detail=_litellm_error_detail(
                            "llm_unavailable",
                            exc,
                            expose_provider_error=settings.composer_expose_provider_errors,
                        ),
                    ) from exc
                except ComposerPluginCrashError as crash:
                    # Plugin-crash path: mirror /messages handler. See the send_message
                    # block comment for full rationale on why the response body is
                    # redacted but partial_state is still persisted, AND for why this
                    # catch MUST precede `except ComposerServiceError` below.
                    response_body = await _handle_plugin_crash(
                        crash,
                        service,
                        session.id,
                        str(user.user_id),
                        "recompose",
                        pre_send_state_id,
                        settings=settings,
                        secret_service=request.app.state.scoped_secret_resolver,
                    )
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=request_id,
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="failed",
                            headline="The composer could not safely finish this retry.",
                            evidence=("A pipeline tool failed on the server side.",),
                            likely_next="Review the visible error message, then retry after the issue is resolved.",
                            reason="plugin_crash",
                        ),
                    )
                    raise HTTPException(status_code=500, detail=response_body) from crash.original_exc
                except ComposerRuntimePreflightError as rpf_exc:
                    # Path 1 (cached preflight) mirror of the send_message catch;
                    # see the send_message handler for the full rationale on
                    # why both raise sites delegate to the shared
                    # _handle_runtime_preflight_failure helper, on the
                    # path-1 vs path-2 distinction, and on the
                    # cached_preflight telemetry attribution
                    # (elspeth-0891e8da73).
                    _record_composer_runtime_preflight_telemetry(
                        "exception",
                        source="cached_preflight",
                        exception_class=rpf_exc.exc_class,
                    )
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=request_id,
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="failed",
                            headline="The composer could not safely finish this retry.",
                            evidence=("Runtime preflight failed before the compose loop returned.",),
                            likely_next="Review the visible error message, then retry after the issue is resolved.",
                            reason="runtime_preflight_failed",
                        ),
                    )
                    response_body = await _handle_runtime_preflight_failure(
                        rpf_exc,
                        service,
                        session.id,
                        str(user.user_id),
                        "recompose",
                        pre_send_state_id,
                        settings=settings,
                        secret_service=request.app.state.scoped_secret_resolver,
                    )
                    raise HTTPException(status_code=500, detail=response_body) from rpf_exc.original_exc
                except ComposerServiceError as exc:
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=request_id,
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="failed",
                            headline="The composer could not finish this retry.",
                            evidence=("Prompt preparation or composer service setup failed.",),
                            likely_next="Retry once the composer service is available.",
                            reason="service_setup_failed",
                        ),
                    )
                    llm_calls = _llm_calls_from_exception(exc)
                    if llm_calls:
                        await _persist_llm_calls(service, session.id, llm_calls, pre_send_state_id, plugin_crash_pending=True)
                    raise HTTPException(
                        status_code=502,
                        detail={"error_type": "composer_error", "detail": str(exc)},
                    ) from exc

                # Compute the post-compose guided_session and composer_meta.
                # Mirror of send_message §5a-§5b: if the transition prompt fired
                # this turn, flip transition_consumed so subsequent turns use the
                # freeform-only prompt.  guided_session rides in composer_meta (not
                # a first-class column) — any save must propagate it forward.
                _post_compose_guided: GuidedSession | None = result.state.guided_session
                if _guided_terminal_for_compose is not None:
                    # transition_consumed flip — _guided is non-None because
                    # _guided_terminal_for_compose was derived from _guided.terminal.
                    if _guided is None:
                        raise InvariantError(
                            "guided_terminal_for_compose is set but guided_session is None — "
                            "impossible state: transition gate should have blocked this path"
                        )
                    from dataclasses import replace as _replace_dc

                    _post_compose_guided = _replace_dc(
                        _guided,
                        transition_consumed=True,
                    )

                _post_compose_meta: dict[str, Any] = {"repair_turns_used": result.repair_turns_used}
                if _post_compose_guided is not None:
                    _post_compose_meta["guided_session"] = _post_compose_guided.to_dict()

                # Save state if version changed.
                # Path 2 (post-compose runtime preflight): mirror of the
                # send_message post-compose try/except — see the send_message
                # block for the full rationale on the structural fix.
                state_response: CompositionStateResponse | None = None
                post_compose_state_id: UUID | None = pre_send_state_id
                if result.state.version != state.version:
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=request_id,
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="validating",
                            headline="The composer has updated the pipeline and is validating the result.",
                            evidence=("The updated pipeline state is being checked before persistence.",),
                            likely_next="ELSPETH will save the validated pipeline snapshot.",
                        ),
                    )
                    try:
                        state_data, validation = await _state_data_from_composer_state(
                            result.state,
                            settings=settings,
                            secret_service=request.app.state.scoped_secret_resolver,
                            user_id=str(user.user_id),
                            runtime_preflight=result.runtime_preflight,
                            preflight_exception_policy="raise",
                            initial_version=state.version,
                            telemetry_source="recompose",
                            composer_meta=_post_compose_meta,
                        )
                    except ComposerRuntimePreflightError as rpf_exc:
                        rpf_exc = ComposerRuntimePreflightError(
                            original_exc=rpf_exc.original_exc,
                            partial_state=rpf_exc.partial_state,
                            tool_invocations=result.tool_invocations,
                            llm_calls=result.llm_calls,
                        )
                        await _publish_progress(
                            progress_registry,
                            session_id=str(session.id),
                            request_id=request_id,
                            user_id=str(user.user_id),
                            event=ComposerProgressEvent(
                                phase="failed",
                                headline="The composer could not safely validate the pipeline update.",
                                evidence=("Runtime preflight failed during state persistence.",),
                                likely_next="Review the visible error message, then retry after the issue is resolved.",
                                reason="runtime_preflight_failed",
                            ),
                        )
                        response_body = await _handle_runtime_preflight_failure(
                            rpf_exc,
                            service,
                            session.id,
                            str(user.user_id),
                            "recompose",
                            pre_send_state_id,
                            settings=settings,
                            secret_service=request.app.state.scoped_secret_resolver,
                        )
                        raise HTTPException(status_code=500, detail=response_body) from rpf_exc.original_exc
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session.id),
                        request_id=request_id,
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="saving",
                            headline="ELSPETH is saving the pipeline update.",
                            evidence=("A new composition state version is being stored for this session.",),
                            likely_next="The assistant response will appear after the save completes.",
                        ),
                    )
                    new_state_record = await service.save_composition_state(
                        session.id,
                        state_data,
                        # Preserves pre-fix labelling — see the parallel
                        # comment on the post-compose path 1 site above.
                        # Symmetric mis-attribution; out of scope for the
                        # f217c634aa handler-site fix.
                        provenance="session_seed",
                    )
                    state_response = _state_response(new_state_record, live_validation=validation)
                    post_compose_state_id = new_state_record.id
                elif _guided_terminal_for_compose is not None and _post_compose_guided is not None:
                    # Version unchanged but transition_consumed must be flipped.
                    # Persist the updated guided_session in a new state row so
                    # subsequent turns pick up transition_consumed=True.
                    _existing_meta: dict[str, Any] = {}
                    if state_record is not None and state_record.composer_meta is not None:
                        _existing_meta = dict(deep_thaw(state_record.composer_meta))
                    _transition_meta = {**_existing_meta, **_post_compose_meta}
                    _transition_state = result.state
                    _transition_state_d = _transition_state.to_dict()
                    _transition_state_data = CompositionStateData(
                        source=_transition_state_d["source"],
                        nodes=_transition_state_d["nodes"],
                        edges=_transition_state_d["edges"],
                        outputs=_transition_state_d["outputs"],
                        metadata_=_transition_state_d["metadata"],
                        is_valid=False,
                        validation_errors=None,
                        composer_meta=_transition_meta,
                    )
                    _transition_record = await service.save_composition_state(
                        session.id,
                        _transition_state_data,
                        # Mirrors the paired session_seed-labelled site immediately
                        # above (post-compose path 1, transition_consumed flip).
                        # Same known mis-attribution as that paired site — see the
                        # comment block at the earlier save call for the
                        # elspeth-obs-f217c634aa relabelling history.
                        provenance="session_seed",
                    )
                    post_compose_state_id = _transition_record.id

                # Persist assistant message
                assistant_msg = await service.add_message(
                    session.id,
                    "assistant",
                    result.message,
                    composition_state_id=post_compose_state_id,
                    raw_content=result.raw_assistant_content,
                    writer_principal="compose_loop",
                )
                # Per-tool-call audit trail (recompose path; symmetric with send_message).
                if result.tool_invocations and not result.persisted_tool_call_turn:
                    await _persist_tool_invocations(
                        service,
                        session.id,
                        result.tool_invocations,
                        post_compose_state_id,
                        parent_assistant_id=assistant_msg.id,
                        plugin_crash_pending=False,
                    )
                if result.llm_calls:
                    await _persist_llm_calls(
                        service,
                        session.id,
                        result.llm_calls,
                        pre_send_state_id,
                        plugin_crash_pending=False,
                    )
                await _publish_progress(
                    progress_registry,
                    session_id=str(session.id),
                    request_id=request_id,
                    user_id=str(user.user_id),
                    event=ComposerProgressEvent(
                        phase="complete",
                        headline="The composer has updated the pipeline."
                        if result.state.version != state.version
                        else "The composer response is ready.",
                        evidence=("The assistant response has been saved for this session.",),
                        likely_next="Review the response and current pipeline.",
                        reason="composer_complete",
                    ),
                )

                # See send_message return-flow comment for why response
                # construction precedes the terminal_status flip.
                response = MessageWithStateResponse(
                    message=_message_response(assistant_msg),
                    state=state_response,
                )
                terminal_status = "completed"
                return response
            except InvariantError as exc:
                # Mirror of send_message InvariantError handler — same
                # B1-sanitization rationale. Static 500 detail; slog carries
                # exc_class + frames only.
                slog.error(
                    "guided.invariant_violated",
                    session_id=str(session_id),
                    user_id=user.user_id,
                    exc_class=type(exc).__name__,
                    site="recompose",
                    frames=_safe_frame_strings(exc),
                )
                raise HTTPException(
                    status_code=500,
                    detail="Server invariant violated. See application audit log for diagnostic detail.",
                ) from exc
            except asyncio.CancelledError as exc:
                # Mirror of send_message cancellation path. See block
                # comment there for the shielded-publish rationale.
                llm_calls = _llm_calls_from_exception(exc)
                if llm_calls:
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(
                            _persist_llm_calls(
                                service,
                                session.id,
                                llm_calls,
                                pre_send_state_id,
                                plugin_crash_pending=True,
                            )
                        )
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(
                        _publish_progress(
                            progress_registry,
                            session_id=str(session.id),
                            request_id=request_id,
                            user_id=str(user.user_id),
                            event=client_cancelled_progress_event(),
                        )
                    )
                terminal_status = "cancelled"
                raise
            finally:
                _COMPOSER_REQUESTS_INFLIGHT.add(-1, {"endpoint": "recompose"})
                _record_composer_request_terminal(terminal_status, endpoint="recompose")

    @router.get(
        "/{session_id}/messages",
        response_model=list[ChatMessageResponse],
    )
    async def get_messages(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
        include_llm_audit: bool = Query(False),
        include_raw_content: bool = Query(False),
        include_tool_rows: bool = Query(False),
    ) -> list[ChatMessageResponse]:
        """Get conversation history for a session.

        ``include_raw_content`` opts in to the assistant message's
        pre-synthesis prose (the model's actual final text when the
        empty-state synthesizer replaced the visible content). Default
        omits it — the SPA conversation channel does not need audit data.
        Eval/diagnosis tooling enables it to verify whether the model
        converged on useful output that the synthesizer hid.
        """
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        if include_tool_rows:
            audit_query_args = {key: value for key, value in request.query_params.items() if key in AUDIT_GRADE_VIEW_QUERY_ARG_ALLOWLIST}
            await service.record_audit_grade_view_async(
                session_id=str(session.id),
                requesting_principal=user.user_id,
                request_path=request.url.path,
                query_args=audit_query_args,
                ip_address=request.client.host if request.client else None,
            )
        # Fetch before slicing so hidden audit rows cannot skew normal-chat
        # pagination. The service remains the durable audit store; this route
        # is the user-facing conversation channel. The eval harness can opt in
        # to LLM-call sidecars, which contain model/usage/cost metadata but not
        # raw prompts, tool arguments, or tool results.
        messages = await service.get_messages(session.id, limit=None)
        if include_tool_rows and include_llm_audit:
            conversation_messages = _composer_conversation_tool_or_llm_audit_messages(messages)
        elif include_tool_rows:
            conversation_messages = _composer_conversation_or_tool_messages(messages)
        elif include_llm_audit:
            conversation_messages = _composer_conversation_or_llm_audit_messages(messages)
        else:
            conversation_messages = _composer_conversation_messages(messages)
        paged_messages = conversation_messages[offset : offset + limit]
        return [_message_response(m, include_raw_content=include_raw_content) for m in paged_messages]

    @router.get(
        "/{session_id}/runs",
        response_model=list[RunResponse],
    )
    async def list_session_runs(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> list[RunResponse]:
        """List all runs for a session, newest first."""
        session = await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        runs = await service.list_runs_for_session(session.id)
        from elspeth.web.execution.discard_summary import load_discard_summaries_for_settings

        terminal_landscape_run_ids = tuple(
            run.landscape_run_id for run in runs if run.status in SESSION_TERMINAL_RUN_STATUS_VALUES and run.landscape_run_id is not None
        )
        accounting_by_run_id = {}
        if terminal_landscape_run_ids:
            try:
                accounting_by_run_id = await run_sync_in_worker(
                    load_run_accounting_for_settings,
                    request.app.state.settings,
                    terminal_landscape_run_ids,
                )
            except ValueError as exc:
                raise _run_accounting_integrity_http(
                    "Session run accounting projection failed.",
                    landscape_run_ids=terminal_landscape_run_ids,
                    error=str(exc),
                ) from exc
        discard_summaries = {}
        if terminal_landscape_run_ids:
            discard_summaries = await run_sync_in_worker(
                load_discard_summaries_for_settings,
                request.app.state.settings,
                terminal_landscape_run_ids,
            )

        # Resolve composition_version from each run's state_id.
        # A missing state is Tier 1 data corruption — crash, don't hide.
        # Scope the read to the current session: the current-schema
        # composite FK prevents cross-session state refs at the schema
        # layer. ``get_state_in_session`` raises
        # ``AuditIntegrityError`` on session mismatch, surfacing Tier 1
        # corruption rather than silently returning the wrong state's
        # version number in another session's listing.
        responses: list[RunResponse] = []
        for run in runs:
            state = await service.get_state_in_session(run.state_id, session.id)
            version = state.version
            discard_summary = None
            if run.landscape_run_id is not None and run.landscape_run_id in discard_summaries:
                discard_summary = discard_summaries[run.landscape_run_id]
            accounting = None
            if run.landscape_run_id is not None and run.landscape_run_id in accounting_by_run_id:
                accounting = accounting_by_run_id[run.landscape_run_id]
            _validate_run_status_accounting_for_list(run, accounting)
            responses.append(
                RunResponse(
                    id=str(run.id),
                    session_id=str(run.session_id),
                    status=run.status,
                    accounting=accounting,
                    error=run.error,
                    started_at=run.started_at,
                    finished_at=run.finished_at,
                    composition_version=version,
                    discard_summary=discard_summary,
                )
            )
        return responses

    @router.get("/{session_id}/state")
    async def get_current_state(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> CompositionStateResponse | None:
        """Get the current (highest-version) composition state."""
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        state = await service.get_current_state(session.id)
        if state is None:
            return None
        return _state_response(state)

    @router.get(
        "/{session_id}/state/versions",
        response_model=list[CompositionStateResponse],
    )
    async def get_state_versions(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> list[CompositionStateResponse]:
        """Get composition state versions for a session."""
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        versions = await service.get_state_versions(session.id, limit=limit, offset=offset)
        return [_state_response(v) for v in versions]

    @router.post(
        "/{session_id}/state/revert",
        response_model=CompositionStateResponse,
    )
    async def revert_state(
        session_id: UUID,
        body: RevertStateRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> CompositionStateResponse:
        """Revert the pipeline to a prior composition state version (R1).

        Creates a new version that is a copy of the specified prior state.
        Injects a system message recording the revert.
        """
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service

        try:
            new_state = await service.set_active_state(
                session.id,
                body.state_id,
            )
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail="State not found",
            ) from None

        # Look up the original version number for the system message
        original_state = await service.get_state(body.state_id)
        await service.add_message(
            session.id,
            role="system",
            content=f"Pipeline reverted to version {original_state.version}.",
            writer_principal="route_system_message",
        )

        return _state_response(new_state)

    @router.get("/{session_id}/state/yaml")
    async def get_state_yaml(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> dict[str, str]:
        """Get YAML representation of the current composition state (M1).

        Runs runtime preflight on the exact CompositionState reconstructed
        from the persisted record, then generates deterministic YAML via
        generate_yaml() against that same snapshot. The two operations see
        the same Python object — there is no re-fetch between preflight
        and serialization, so a state that passes the gate is byte-
        identical to the state that gets serialized.
        """
        session = await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        state_record = await service.get_current_state(session.id)
        if state_record is None:
            raise HTTPException(status_code=404, detail="No composition state exists")
        state = _state_from_record(state_record)
        try:
            runtime_validation = await _runtime_preflight_for_state(
                state,
                settings=request.app.state.settings,
                secret_service=request.app.state.scoped_secret_resolver,
                user_id=str(user.user_id),
            )
        except (
            TimeoutError,
            OSError,
            PluginConfigError,
            PluginNotFoundError,
            GraphValidationError,
        ) as exc:
            # Narrowed per CLAUDE.md offensive-programming policy. This
            # tuple covers the user-fixable preflight failure modes:
            #
            # * TimeoutError — asyncio.wait_for exceeded
            #   composer_runtime_preflight_timeout_seconds. Operator
            #   action: increase timeout or fix the slow plugin.
            # * OSError — filesystem error during plugin instantiation
            #   (file not found, permission denied, broken pipe, etc.).
            #   Operator action: fix the file/permissions.
            # * PluginConfigError / PluginNotFoundError — the user's
            #   pipeline references a misconfigured or missing plugin.
            #   Operator action: fix the pipeline config.
            # * GraphValidationError — the pipeline graph is structurally
            #   invalid (validate_pipeline normally absorbs this, but
            #   it's listed here for defense-in-depth in case a future
            #   refactor lets it escape).
            #
            # Programmer-bug classes (AttributeError, TypeError,
            # KeyError, RuntimeError, ImportError, etc.) are deliberately
            # NOT caught — they propagate to FastAPI's default 500
            # handler so operators see real tracebacks rather than the
            # misleading "fix your pipeline" 409 message. The
            # exception-counter is reserved for the user-fixable bucket
            # so dashboards measure real preflight failure rate, not
            # bugs we introduced ourselves.
            _record_composer_runtime_preflight_telemetry(
                "exception",
                source="yaml_export",
                exception_class=type(exc).__name__,
            )
            raise HTTPException(
                status_code=409,
                detail="Runtime preflight could not complete; YAML export aborted.",
            ) from exc
        _record_composer_runtime_preflight_telemetry(
            "passed" if runtime_validation.is_valid else "failed",
            source="yaml_export",
        )
        if not runtime_validation.is_valid:
            detail = "Current composition state failed runtime preflight. Fix validation errors before exporting YAML."
            if runtime_validation.errors:
                detail = f"{detail} First error: {runtime_validation.errors[0].message}"
            raise HTTPException(status_code=409, detail=detail)
        yaml_str = generate_yaml(state)
        response = {"yaml": yaml_str}
        if state.source is not None and "blob_ref" in state.source.options:
            response["source_blob_id"] = str(state.source.options["blob_ref"])
        return response

    @router.post(
        "/{session_id}/fork",
        status_code=201,
        response_model=ForkSessionResponse,
    )
    async def fork_from_message(
        session_id: UUID,
        body: ForkSessionRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> ForkSessionResponse:
        """Fork a session from a specific user message.

        Creates a new session inheriting history and composition state up to
        the fork point, with the edited message replacing the original.
        The original session is never mutated.
        """
        await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        settings = request.app.state.settings

        try:
            new_session, new_messages, copied_state = await service.fork_session(
                source_session_id=session_id,
                fork_message_id=body.from_message_id,
                new_message_content=body.new_message_content,
                user_id=user.user_id,
                auth_provider_type=settings.auth_provider,
            )
        except InvalidForkTargetError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        # Everything after fork_session() is a compensatable post-commit
        # phase.  If ANY step fails, archive the fork to avoid orphaned
        # sessions/blobs/state.  BlobQuotaExceededError gets a specific
        # 413; all other failures re-raise after cleanup.
        blob_service: BlobServiceProtocol = request.app.state.blob_service
        try:
            source_blobs = await blob_service.list_blobs(session_id)
            # Copy blobs from source session into the forked session.
            # Returns old_id → new_blob mapping for source reference rewriting.
            blob_map = await blob_service.copy_blobs_for_fork(session_id, new_session.id)
            source_blob_path_map = {blob.storage_path: blob_map[blob.id] for blob in source_blobs if blob.id in blob_map}

            # Rewrite source references in the forked state so the fork is
            # self-contained.  Without this, blob_ref and path in the source
            # options still point at the original session's assets.
            if copied_state is not None and copied_state.source is not None and blob_map:
                source_dict = deep_thaw(copied_state.source) if copied_state.source else None
                if not isinstance(source_dict, dict):
                    raise AuditIntegrityError(
                        f"Tier 1 audit anomaly: composition_state {copied_state.id} "
                        f"has source type {type(source_dict).__name__}, expected dict "
                        f"before fork blob rewrite for session {new_session.id}."
                    )

                options = source_dict.get("options")
                if options is None:
                    rewritten = False
                else:
                    if not isinstance(options, dict):
                        raise AuditIntegrityError(
                            f"Tier 1 audit anomaly: composition_state {copied_state.id} "
                            f"has source.options type {type(options).__name__}, expected "
                            f"dict before fork blob rewrite for session {new_session.id}."
                        )

                    rewritten = False
                    rewrite_target = None
                    # Remap blob_ref to the new blob's ID.
                    # composition_states.source is Tier 1 ("our data") — the
                    # composer writes blob_ref as the blob's UUID string
                    # (composer/tools.py _execute_set_source_from_blob).  A
                    # non-UUID value here means a write-path bug, DB
                    # corruption, or tampering — crash with a diagnostic
                    # rather than silently skipping the remap.  Silent skip
                    # would leave the forked state's blob_ref pointing at
                    # the source session's blob, which is the cross-session
                    # reference class closed at the FK layer by the
                    # current-schema composite FK and is audit-contradictory
                    # on its face.  The enclosing ``except Exception``
                    # block archives the partially-created fork (see the
                    # cleanup-rollback site below), so this crash does
                    # not leak artifacts.
                    old_ref = options.get("blob_ref")
                    if old_ref is not None:
                        try:
                            old_uuid = UUID(old_ref) if isinstance(old_ref, str) else old_ref
                        except ValueError as exc:
                            raise AuditIntegrityError(
                                f"Tier 1 audit anomaly: composition_state "
                                f"{copied_state.id} has non-UUID blob_ref "
                                f"{old_ref!r} in source.options (expected a "
                                f"UUID string written by composer/tools.py). "
                                f"Fork aborted to prevent cross-session blob "
                                f"reference in forked session {new_session.id}."
                            ) from exc
                        rewrite_target = blob_map.get(old_uuid)

                    if rewrite_target is None:
                        for path_key in ("path", "file"):
                            path_value = options.get(path_key)
                            if isinstance(path_value, str) and path_value in source_blob_path_map:
                                rewrite_target = source_blob_path_map[path_value]
                                break

                    if rewrite_target is not None:
                        options["blob_ref"] = str(rewrite_target.id)
                        if "path" in options or "file" not in options:
                            options["path"] = rewrite_target.storage_path
                        if "file" in options:
                            options["file"] = rewrite_target.storage_path
                        rewritten = True

                if rewritten:
                    source_dict["options"] = options
                    # Save updated state with remapped source. Preserve the
                    # source state's composer_meta — fork inherits the
                    # operational provenance of the parent compose.
                    state_data = CompositionStateData(
                        source=source_dict,
                        nodes=deep_thaw(copied_state.nodes),
                        edges=deep_thaw(copied_state.edges),
                        outputs=deep_thaw(copied_state.outputs),
                        metadata_=deep_thaw(copied_state.metadata_),
                        is_valid=copied_state.is_valid,
                        validation_errors=list(copied_state.validation_errors) if copied_state.validation_errors else None,
                        composer_meta=deep_thaw(copied_state.composer_meta) if copied_state.composer_meta is not None else None,
                    )
                    copied_state = await service.save_composition_state(
                        new_session.id,
                        state_data,
                        # Preserves pre-fix labelling. The fork-time source-
                        # storage rewrite previously wrote ``session_seed``
                        # under the hardcoded label and continues to do so.
                        # Whether this row should carry ``session_fork``
                        # (the rewrite is part of the fork operation) or a
                        # new ``fork_storage_rewrite`` discriminator is a
                        # separate audit-attribution question outside the
                        # scope of elspeth-obs-f217c634aa.
                        provenance="session_seed",
                    )

                    # The edited user message (last in list) still references
                    # the pre-rewrite state.  Re-point it at the replacement
                    # state so message-state lineage is self-contained.
                    user_msg = new_messages[-1]
                    await service.update_message_composition_state(
                        user_msg.id,
                        copied_state.id,
                    )
                    new_messages[-1] = ChatMessageRecord(
                        id=user_msg.id,
                        session_id=user_msg.session_id,
                        role=user_msg.role,
                        content=user_msg.content,
                        raw_content=user_msg.raw_content,
                        tool_calls=user_msg.tool_calls,
                        created_at=user_msg.created_at,
                        sequence_no=user_msg.sequence_no,
                        composition_state_id=copied_state.id,
                        writer_principal=user_msg.writer_principal,
                        tool_call_id=user_msg.tool_call_id,
                        parent_assistant_id=user_msg.parent_assistant_id,
                    )
        except BlobQuotaExceededError:
            # Build the HTTPException up-front so cleanup failures can be
            # attached as a note on the object that actually propagates —
            # the inner BlobQuotaExceededError is suppressed by `from None`
            # and any note attached to it would never reach operator logs.
            # Cleanup catch is narrowed to recoverable IO/DB failures so
            # programmer bugs (AttributeError, TypeError) still crash.
            quota_exc = HTTPException(
                status_code=413,
                detail="Blob quota exceeded during fork — unable to copy files",
            )
            try:
                await service.archive_session(new_session.id)
            except (SQLAlchemyError, OSError) as cleanup_exc:
                quota_exc.add_note(
                    f"RecoveryFailed[{type(cleanup_exc).__name__}]: "
                    f"could not archive forked session {new_session.id} "
                    f"after blob quota rollback ({cleanup_exc}). "
                    f"Manual cleanup of sessions.id={new_session.id} required."
                )
            raise quota_exc from None
        except Exception as primary_exc:
            # Mirror the RecoveryFailed[...] convention from
            # ``BlobServiceImpl.copy_blobs_for_fork`` and
            # ``BlobServiceImpl.finalize_run_output_blobs`` (web/blobs/service.py):
            # cleanup failures must NOT mask the original error.  Narrow the
            # catch to (SQLAlchemyError, OSError) — programmer bugs in
            # archive_session must propagate — and attach the cleanup
            # failure as a note so the orphan session row is visible to
            # operators reading the traceback.  Bare `raise` preserves
            # primary_exc and its original traceback as the headline.
            try:
                await service.archive_session(new_session.id)
            except (SQLAlchemyError, OSError) as cleanup_exc:
                primary_exc.add_note(
                    f"RecoveryFailed[{type(cleanup_exc).__name__}]: "
                    f"could not archive forked session {new_session.id} "
                    f"during fork rollback ({cleanup_exc}). "
                    f"Manual cleanup of sessions.id={new_session.id} required."
                )
            raise

        return ForkSessionResponse(
            session=_session_response(new_session),
            messages=[_message_response(m) for m in new_messages],
            composition_state=_state_response(copied_state) if copied_state else None,
        )

    def _build_get_guided_turn(
        state: Any,
        guided: Any,
        *,
        catalog: Any,
    ) -> Any | None:
        """Build the turn payload to return from GET /guided for the current step.

        Called exclusively by ``get_guided`` to determine ``next_turn`` in the
        response.  Rebuilds the correct turn deterministically from
        ``(state, guided, catalog)`` alone, including intra-step turns for
        steps that maintain staging fields on the session.

        Per-step rebuild rules:

        - STEP_1_SOURCE: three sub-cases in priority order:
          1. ``step_1_source_intent`` is set → the SCHEMA_FORM response was
             submitted; the session is waiting for INSPECT_AND_CONFIRM
             confirmation.  Emit ``inspect_and_confirm`` from the intent's
             observed columns (warnings are not stored on SourceIntent;
             the rebuild emits an empty warnings list).
          2. ``step_1_source_intent`` is None → initial state or SINGLE_SELECT
             window.  Fall through to ``build_initial_step_1_turn``.
             Note: the window between SINGLE_SELECT and SCHEMA_FORM does not
             persist the chosen source plugin name; a GET /guided in that
             window will re-emit the SINGLE_SELECT.  This is an observation-
             not-fix gap (parallel to Finding 2, addressed by
             step_2_chosen_plugin) and is tracked as obs-STEP1-chosen-plugin.

        - STEP_2_SINK: three sub-cases in priority order:
          1. ``step_2_sink_intent`` is set → the SCHEMA_FORM response was
             submitted; the session is waiting for MULTI_SELECT_WITH_CUSTOM
             confirmation.  Emit ``multi_select_with_custom`` from step_1_result.
          2. ``step_2_chosen_plugin`` is set → the SINGLE_SELECT response was
             submitted; the session is waiting for SCHEMA_FORM submission.
             Emit ``schema_form`` for the chosen plugin.
          3. Neither is set → initial state.  Emit ``single_select`` listing
             all registered sink plugins.

        - STEP_2_5_RECIPE_MATCH: deterministically re-run ``match_recipe``; if a
          match is found, emit ``recipe_offer``.  Returns ``None`` when no
          recipe matched (session stays at this step but no turn exists).

        - STEP_3_TRANSFORMS: if ``step_3_proposal`` is set, emit
          ``propose_chain`` from the staged proposal (this is the normal
          path — step_3_proposal is always set when the session reaches
          STEP_3_TRANSFORMS via the LLM chain solver).  Returns ``None``
          if the proposal is absent (LLM call has not completed; should
          not occur in practice — guarded defensively to avoid a crash).

        Returns:
            A ``Turn`` TypedDict, or ``None`` when the step has no rebuildable
            turn (no-recipe STEP_2_5 path, or STEP_3 without a proposal).
        """
        step = guided.step
        if step is GuidedStep.STEP_1_SOURCE:
            # Finding 3 (Codex #14): if step_1_source_intent is set, the SCHEMA_FORM
            # was already submitted and the session is waiting for INSPECT_AND_CONFIRM.
            # Rebuild from the staged intent (observed_columns; warnings default to empty
            # as they are not stored on SourceIntent).
            if guided.step_1_source_intent is not None:
                return build_step_1_inspect_and_confirm_turn_from_intent(guided.step_1_source_intent)
            return build_initial_step_1_turn(state, blob_inspection=None, catalog=catalog)
        if step is GuidedStep.STEP_2_SINK:
            # Finding 2 (Codex #10): determine intra-step position and rebuild
            # the correct turn, not always the initial SINGLE_SELECT.
            if guided.step_2_sink_intent is not None:
                # SCHEMA_FORM was submitted; session is waiting for MULTI_SELECT_WITH_CUSTOM.
                observed_columns: tuple[str, ...] = ()
                if guided.step_1_result is not None:
                    observed_columns = tuple(guided.step_1_result.observed_columns)
                return build_step_2_multi_select_turn(observed_columns)
            if guided.step_2_chosen_plugin is not None:
                # SINGLE_SELECT was submitted; session is waiting for SCHEMA_FORM submission.
                return build_step_2_schema_form_turn(guided.step_2_chosen_plugin, catalog)
            # Initial state: no plugin chosen yet.  Emit the sink plugin list.
            return build_step_2_single_select_turn(catalog)
        if step is GuidedStep.STEP_2_5_RECIPE_MATCH:
            # A recipe_offer TurnRecord exists iff a recipe was matched.
            # Reconstruct by re-running match_recipe (deterministic).
            if guided.step_1_result is not None and guided.step_2_result is not None:
                recipe_match = match_recipe(guided.step_1_result, guided.step_2_result)
                if recipe_match is not None:
                    return build_step_2_5_recipe_offer_turn(recipe_match)
            # No recipe matched — no initial turn for this step.
            return None
        if step is GuidedStep.STEP_3_TRANSFORMS:
            # Finding 1 (Codex #5): rebuild propose_chain from the staged proposal.
            # step_3_proposal is always set when the session reaches STEP_3_TRANSFORMS
            # via the LLM chain solver (set atomically with the step pointer in the
            # dispatcher at routes.py:2051 and routes.py:4063).  A None here would
            # mean the proposal was never computed — guarded defensively to avoid
            # crashing on a corrupt session rather than returning a misleading null.
            if guided.step_3_proposal is not None:
                return build_step_3_propose_chain_turn(guided.step_3_proposal)
            # No proposal yet — LLM call has not completed; return None and let the
            # idempotency machinery handle it (no TurnRecord emitted; client retries).
            return None
        return None

    @router.get("/{session_id}/guided", response_model=GetGuidedResponse)
    async def get_guided(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> GetGuidedResponse:
        """Return the current guided-mode state for a session.

        **Mutating on first visit:** if the current step has no emitted
        TurnRecord in the guided session history, a turn is built and
        persisted.  Subsequent fetches are idempotent — the existing
        TurnRecord's payload_hash is returned verbatim.

        If the session has no existing CompositionState, one is created
        with ``GuidedSession.initial()`` attached (spec §5.2 default-guided
        invariant).

        Returns 404 if the session does not exist or does not belong to
        the requesting user.
        Returns 400 if the session's composition state has no guided_session
        attached (freeform session — use /api/sessions/{id}/messages instead).
        """
        await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        catalog: CatalogServiceProtocol = request.app.state.catalog_service
        recorder = BufferingRecorder()

        compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))
        async with compose_lock:
            # PR-review B3: drain the recorder on every exit path, including
            # ``raise HTTPException`` rejections.  Without this, any audit
            # event emitted before a mid-body raise would be discarded — a
            # CLAUDE.md auditability violation ("rejected requests are facts
            # worth recording").  ``state_record_out`` is hoisted so the
            # finally block can pass its id (or None) regardless of where
            # control left the try.
            state_record_out: CompositionStateRecord | None = None
            try:
                # Load or create CompositionState.
                state_record = await service.get_current_state(session_id)
                if state_record is None:
                    state = _initial_composition_state_with_guided_session()
                else:
                    state = _state_from_record(state_record)
                    state_record_out = state_record

                # Reject freeform sessions.
                if state.guided_session is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Session is not in guided mode. Use /api/sessions/{id}/messages.",
                    )

                guided = state.guided_session
                current_step = guided.step

                # Idempotency check: if this step already has an emitted TurnRecord,
                # return the existing payload without re-emitting.
                existing_record_for_step: TurnRecord | None = next(
                    (r for r in reversed(guided.history) if r.step == current_step),
                    None,
                )

                # Build the initial turn for the current step (deterministic from
                # state + catalog).  Returns None for steps with no rebuildable
                # initial turn (STEP_3 / no-recipe STEP_2_5 path) or when the
                # session is already terminal.
                turn = _build_get_guided_turn(state, guided, catalog=catalog) if guided.terminal is None else None
                turn_type: TurnType | None = TurnType(turn["type"]) if turn is not None else None
                payload_hash: str | None = stable_hash(turn["payload"]) if turn is not None else None

                if existing_record_for_step is None and turn is not None:
                    # First fetch for this step AND a turn exists: record TurnRecord,
                    # persist, emit audit.  When turn is None (terminal state, STEP_3,
                    # or no-recipe STEP_2_5 path) there is no turn to record.
                    # Guaranteed by the conditional assignments above: turn is not
                    # None on this branch, so both turn_type and payload_hash were
                    # populated from turn["type"] / stable_hash(turn["payload"]).
                    # Use InvariantError (not bare assert) so python -O does not
                    # strip the gate and silently feed None to TurnRecord.
                    if turn_type is None:
                        raise InvariantError(
                            "GET guided: turn is not None but turn_type is None — TurnType derivation skipped despite turn being present."
                        )
                    if payload_hash is None:
                        raise InvariantError(
                            "GET guided: turn is not None but payload_hash is None — stable_hash derivation skipped despite turn being present."
                        )
                    new_record = TurnRecord(
                        step=current_step,
                        turn_type=turn_type,
                        payload_hash=payload_hash,
                        response_hash=None,
                        emitter="server",
                    )
                    from dataclasses import replace as _replace

                    # If this is the STEP_2_5_RECIPE_MATCH step, populate
                    # step_2_5_recipe_offer so the POST /respond accept branch
                    # can verify the client-supplied recipe_name against the offer.
                    # This covers the GET-before-POST flow (user reloads the page
                    # while the session is at the recipe-offer step).
                    if (
                        current_step is GuidedStep.STEP_2_5_RECIPE_MATCH
                        and guided.step_1_result is not None
                        and guided.step_2_result is not None
                    ):
                        staged_offer = match_recipe(guided.step_1_result, guided.step_2_result)
                    else:
                        staged_offer = None

                    new_guided = _replace(guided, history=(*guided.history, new_record))
                    if staged_offer is not None:
                        new_guided = _replace(new_guided, step_2_5_recipe_offer=staged_offer)
                    new_state = _replace(state, guided_session=new_guided)

                    # Persist state with updated guided_session in composer_meta.
                    # Preserve any existing composer_meta keys (e.g. repair_turns_used).
                    existing_meta: dict[str, Any] = {}
                    if state_record is not None and state_record.composer_meta is not None:
                        existing_meta = dict(deep_thaw(state_record.composer_meta))
                    new_composer_meta = {**existing_meta, "guided_session": new_guided.to_dict()}

                    state_d = new_state.to_dict()
                    state_data = CompositionStateData(
                        source=state_d["source"],
                        nodes=state_d["nodes"],
                        edges=state_d["edges"],
                        outputs=state_d["outputs"],
                        metadata_=state_d["metadata"],
                        is_valid=False,
                        validation_errors=None,
                        composer_meta=new_composer_meta,
                    )
                    state_record_out = await service.save_composition_state(
                        session_id,
                        state_data,
                        # Guided-mode server-emitted turn: the LLM converged on a
                        # guided step transition and the resulting state is being
                        # persisted. ``convergence_persist`` is the closest existing
                        # provenance category (Phase 1A enum does not have a
                        # ``guided_persist`` value; widening the enum mid-merge
                        # would require a Phase 1A spec amendment and is out of
                        # scope here — see merge commit message).
                        provenance="convergence_persist",
                    )

                    # Emit audit event.  Persistence of the buffered invocations
                    # is handled by the finally block below — that way both
                    # success and rejection paths drain identically.
                    emit_turn_emitted(
                        recorder,
                        step=current_step,
                        turn_type=turn_type,
                        payload_hash=payload_hash,
                        payload_payload_id="",  # No payload store for server-emitted turns yet.
                        emitter="server",
                        composition_version=new_state.version,
                        actor=user.user_id,
                    )

                    guided = new_guided

                # Idempotency-path repair: if the session is at STEP_2_5_RECIPE_MATCH
                # and the staged offer is absent (persisted before the step_2_5_recipe_offer
                # field was introduced), re-populate it and persist the repaired state so
                # the POST /respond accept branch can verify the recipe_name.  Without this
                # repair, sessions that reached STEP_2_5 before this field was added would
                # always fail the accept binding check with 400.  The offer is
                # deterministically reconstructable from (step_1_result, step_2_result).
                if (
                    existing_record_for_step is not None
                    and current_step is GuidedStep.STEP_2_5_RECIPE_MATCH
                    and guided.step_2_5_recipe_offer is None
                    and guided.step_1_result is not None
                    and guided.step_2_result is not None
                ):
                    recovered_offer = match_recipe(guided.step_1_result, guided.step_2_result)
                    if recovered_offer is not None:
                        from dataclasses import replace as _replace_repair

                        repaired_guided = _replace_repair(guided, step_2_5_recipe_offer=recovered_offer)
                        repaired_state = _replace_repair(state, guided_session=repaired_guided)
                        existing_meta_repair: dict[str, Any] = {}
                        if state_record is not None and state_record.composer_meta is not None:
                            existing_meta_repair = dict(deep_thaw(state_record.composer_meta))
                        repair_meta = {**existing_meta_repair, "guided_session": repaired_guided.to_dict()}
                        repaired_state_d = repaired_state.to_dict()
                        repair_data = CompositionStateData(
                            source=repaired_state_d["source"],
                            nodes=repaired_state_d["nodes"],
                            edges=repaired_state_d["edges"],
                            outputs=repaired_state_d["outputs"],
                            metadata_=repaired_state_d["metadata"],
                            is_valid=False,
                            validation_errors=None,
                            composer_meta=repair_meta,
                        )
                        state_record_out = await service.save_composition_state(
                            session_id,
                            repair_data,
                            # Idempotency repair: re-populating ``step_2_5_recipe_offer``
                            # that was missing on a session created before that field
                            # was introduced. The repair restores a derived seed value,
                            # not a new convergence — ``session_seed`` is the closest
                            # existing provenance category. See merge commit message
                            # for the guided-vs-enum scope decision.
                            provenance="session_seed",
                        )
                        guided = repaired_guided

                # Build response.  On re-fetch the same turn is returned (deterministic
                # rebuild) and the payload_hash matches what was recorded on first visit.
                terminal = guided.terminal
                return GetGuidedResponse(
                    guided_session=GuidedSessionResponse(
                        step=guided.step.value,
                        history=[
                            TurnRecordResponse(
                                step=r.step.value,
                                turn_type=r.turn_type.value,
                                payload_hash=r.payload_hash,
                                response_hash=r.response_hash,
                                emitter=r.emitter,
                            )
                            for r in guided.history
                        ],
                        terminal=TerminalStateResponse(
                            kind=terminal.kind.value,
                            reason=terminal.reason.value if terminal.reason is not None else None,
                            pipeline_yaml=terminal.pipeline_yaml,
                        )
                        if terminal is not None
                        else None,
                        chat_history=[
                            ChatTurnResponse(
                                role=t.role.value,
                                content=t.content,
                                seq=t.seq,
                                step=t.step.value,
                                ts_iso=t.ts_iso,
                            )
                            for t in guided.chat_history
                        ],
                        chat_turn_seq=guided.chat_turn_seq,
                    ),
                    next_turn=TurnPayloadResponse(
                        type=turn["type"],
                        step_index=turn["step_index"],
                        payload=dict(turn["payload"]),
                    )
                    if turn is not None
                    else None,
                    terminal=TerminalStateResponse(
                        kind=terminal.kind.value,
                        reason=terminal.reason.value if terminal.reason is not None else None,
                        pipeline_yaml=terminal.pipeline_yaml,
                    )
                    if terminal is not None
                    else None,
                    composition_state=_state_response(state_record_out) if state_record_out is not None else None,
                )
            finally:
                # PR-review B3: drain the recorder unconditionally — success
                # paths and ``raise HTTPException`` rejection paths take the
                # same exit.  Empty drains are a no-op (BufferingRecorder
                # starts with an empty invocations list and
                # ``_persist_tool_invocations`` iterates an empty tuple).
                #
                # The suppress-and-log path is only for exception unwinds:
                # Python's default behaviour is to let a ``finally``-block
                # exception replace the original, which would surface a generic
                # 500 instead of the intended 400/409.  On a successful return,
                # audit persist failures must propagate — otherwise a state
                # write can succeed while the guided audit row silently
                # disappears.  Per CLAUDE.md telemetry/logging primacy,
                # audit-system failures during exception handling are the one
                # exemption where ``slog`` is the correct channel.  The log
                # payload follows the B1 convention: ``exc_class`` + ``frames``
                # only, never ``str(exc)`` or ``exc_info`` (frames are bounded
                # and value-free; the exception message can carry Tier-bearing
                # strings).
                #
                # The two recorder channels (tool invocations and LLM calls)
                # drain through TWO separate try blocks so that a failure
                # persisting one does not skip the other.  ``_persist_llm_calls``
                # covers the :class:`ComposerLLMCall` rows that ``solve_chain``
                # buffers during guided Step 3 (chain solver) invocations.
                # Without the second drain the LLM-call audit would be
                # garbage-collected with the recorder at function exit.
                primary_exc = sys.exception()
                if primary_exc is None:
                    await _persist_tool_invocations(
                        service,
                        session_id,
                        recorder.invocations,
                        state_record_out.id if state_record_out is not None else None,
                        # Guided endpoints don't dispatch tools that can plugin-crash;
                        # auto-drop on solver exhaustion is a separate channel that
                        # doesn't surface through this audit path. Phase 1B/rev-4
                        # made this keyword-only with no default, exposing the gap.
                        plugin_crash_pending=False,
                    )
                    await _persist_llm_calls(
                        service,
                        session_id,
                        recorder.llm_calls,
                        state_record_out.id if state_record_out is not None else None,
                        plugin_crash_pending=False,
                    )
                else:
                    try:
                        await _persist_tool_invocations(
                            service,
                            session_id,
                            recorder.invocations,
                            state_record_out.id if state_record_out is not None else None,
                            # Guided endpoints don't dispatch tools that can plugin-crash;
                            # auto-drop on solver exhaustion is a separate channel that
                            # doesn't surface through this audit path. Phase 1B/rev-4
                            # made this keyword-only with no default, exposing the gap.
                            plugin_crash_pending=False,
                        )
                    except Exception as persist_exc:
                        # Terminal logger-of-last-resort: no safer channel exists if structlog itself raises here.
                        with contextlib.suppress(Exception):
                            slog.error(
                                "guided.audit_persist_failed_during_exception_handling",
                                session_id=str(session_id),
                                user_id=user.user_id,
                                site="get_guided",
                                channel="tool_invocations",
                                exc_class=type(persist_exc).__name__,
                                frames=_safe_frame_strings(persist_exc),
                            )
                    try:
                        await _persist_llm_calls(
                            service,
                            session_id,
                            recorder.llm_calls,
                            state_record_out.id if state_record_out is not None else None,
                            plugin_crash_pending=False,
                        )
                    except Exception as persist_exc:
                        with contextlib.suppress(Exception):
                            slog.error(
                                "guided.audit_persist_failed_during_exception_handling",
                                session_id=str(session_id),
                                user_id=user.user_id,
                                site="get_guided",
                                channel="llm_calls",
                                exc_class=type(persist_exc).__name__,
                                frames=_safe_frame_strings(persist_exc),
                            )

    @router.post("/{session_id}/guided/respond", response_model=GuidedRespondResponse)
    async def post_guided_respond(
        session_id: UUID,
        body: GuidedRespondRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> GuidedRespondResponse:
        """Submit a user response to the current guided-mode turn.

        **Dispatcher:** Identifies the current turn type from the last
        ``TurnRecord`` in the session history, applies the response via
        ``step_advance`` (pure), runs any required side-effect step handler,
        persists the updated state, and emits audit events.

        Returns the updated ``guided_session``, the next ``next_turn`` (or
        ``None`` if the session has reached a terminal state), and the
        ``terminal`` payload (or ``None`` while still active).

        Raises 400 if the session has no ``guided_session`` attached.
        Raises 409 if the guided session is already in a terminal state,
            EXCEPT for the wizard-teardown signal
            ``control_signal=exit_to_freeform`` against a ``kind=completed``
            terminal -- that path transitions the terminal to
            ``exited_to_freeform`` so the chat surface can return.  Already-
            exited sessions and non-exit payloads against terminal sessions
            still 409.
        Raises 404 if the session does not exist or belong to the requesting user.
        """
        await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        catalog: CatalogServiceProtocol = request.app.state.catalog_service
        recorder = BufferingRecorder()

        compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))
        async with compose_lock:
            # PR-review B3: drain the recorder on every exit path, including
            # ``raise HTTPException`` rejections.  This handler emits
            # ``guided_turn_answered`` (line below the ``_validate_*`` calls)
            # BEFORE the dispatcher's try-block can raise 400, so without an
            # unconditional drain the turn-answered event for a rejected
            # advance attempt is silently dropped — a CLAUDE.md auditability
            # violation ("rejected requests are facts worth recording").
            #
            # ``state_record_out`` is hoisted above the try so the finally
            # block can reference it regardless of where control left.  On
            # rejection paths it may legitimately remain ``None`` — the
            # ``_persist_tool_invocations`` signature accepts that.
            state_record_out: CompositionStateRecord | None = None
            try:
                # Load state.
                state_record = await service.get_current_state(session_id)
                if state_record is None:
                    state = _initial_composition_state_with_guided_session()
                else:
                    state = _state_from_record(state_record)
                    state_record_out = state_record

                if state.guided_session is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Session is not in guided mode. Use /api/sessions/{id}/messages.",
                    )

                guided = state.guided_session

                # Parse control_signal (Tier-3 -> Tier-2 coercion) BEFORE the
                # terminal-rejection guard so we can recognise the
                # exit-from-COMPLETED meta-control signal. Other call sites
                # below reuse this parsed value rather than re-parsing.
                control_signal = _validate_control_signal(body.control_signal)

                # Exit-from-COMPLETED is a wizard-teardown signal, NOT a turn
                # response.  When the user clicks "Save and exit" or "Drop to
                # freeform to keep editing" on the CompletionSummary surface
                # (frontend: CompletionSummary.tsx), the buttons fire
                # control_signal=exit_to_freeform.  Without this branch, those
                # requests hit the generic 409 below and the user is locked
                # into the summary -- the ChatPanel discriminator
                # (ChatPanel.tsx:182) keeps the completed surface visible (and
                # the chat input hidden) until terminal.kind transitions to
                # exited_to_freeform.  This branch performs that transition.
                #
                # Scope of the exemption (intentionally narrow):
                #   * Only kind=COMPLETED is exempt.  Already-exited sessions
                #     (kind=EXITED_TO_FREEFORM) still 409 -- exiting an
                #     already-exited session is a no-op.
                #   * Only control_signal=EXIT_TO_FREEFORM is exempt.  Any
                #     other payload (chosen=..., edited_values=..., etc.) sent
                #     to a terminal session still 409s -- no turn is live to
                #     answer.
                #
                # Audit shape:
                #   * The turn-answering scaffolding (_validate_step_indices,
                #     existing_record lookup, response_hash computation,
                #     emit_turn_answered) is bypassed -- no turn is being
                #     answered.  The wizard had no live turn from a terminal
                #     state, so claiming one was answered would be a fabricated
                #     audit record.
                #   * The wizard-teardown directive
                #     ``guided_dropped_to_freeform`` is emitted directly, with
                #     prev_step capturing the step at which the wizard had
                #     completed.  This is the same directive shape the state
                #     machine emits for mid-wizard exit -- ``prev_step`` lets
                #     downstream consumers reconstruct the trajectory.
                #
                # Terminal shape:
                #   * The new terminal has ``pipeline_yaml=None`` because
                #     TerminalState (state_machine.py:53-54) restricts that
                #     field to kind=COMPLETED.  No information is lost: the
                #     yaml is recoverable from composition_state at any time,
                #     and the COMPLETED transition that produced the yaml was
                #     already audit-recorded by the preceding
                #     handle_step_*_accept call.
                #   * reason=USER_PRESSED_EXIT matches the
                #     state-machine-driven mid-wizard exit (state_machine.py:549).
                if (
                    guided.terminal is not None
                    and guided.terminal.kind is TerminalKind.COMPLETED
                    and control_signal is ControlSignal.EXIT_TO_FREEFORM
                ):
                    from dataclasses import replace as _replace

                    new_terminal = TerminalState(
                        kind=TerminalKind.EXITED_TO_FREEFORM,
                        reason=TerminalReason.USER_PRESSED_EXIT,
                        pipeline_yaml=None,
                    )
                    new_guided = _replace(guided, terminal=new_terminal)

                    emit_dropped_to_freeform(
                        recorder,
                        prev=new_guided.step,
                        drop_reason=TerminalReason.USER_PRESSED_EXIT,
                        validation_result=None,
                        composition_version=state.version,
                        actor=user.user_id,
                    )

                    new_state = _replace(state, guided_session=new_guided)
                    # ``existing_meta_exit`` distinguishes this scope from the
                    # later sibling block (~line 5031) that uses ``existing_meta``
                    # for the normal-turn persistence path; mypy treats same-name
                    # locals in the same function as redefinitions even when
                    # control flow makes them disjoint.
                    existing_meta_exit: dict[str, Any] = {}
                    if state_record is not None and state_record.composer_meta is not None:
                        existing_meta_exit = dict(deep_thaw(state_record.composer_meta))
                    new_composer_meta = {**existing_meta_exit, "guided_session": new_guided.to_dict()}

                    state_d = new_state.to_dict()
                    state_data = CompositionStateData(
                        source=state_d["source"],
                        nodes=state_d["nodes"],
                        edges=state_d["edges"],
                        outputs=state_d["outputs"],
                        metadata_=state_d["metadata"],
                        is_valid=False,
                        validation_errors=None,
                        composer_meta=new_composer_meta,
                    )
                    state_record_out = await service.save_composition_state(
                        session_id,
                        state_data,
                        # Exit-from-COMPLETED handler: user transitions a completed
                        # guided session to ``kind=exited_to_freeform`` via the
                        # control_signal=exit_to_freeform path. Same disposition as
                        # other guided convergence writes — the user-supplied
                        # exit signal converged on a new terminal state and the
                        # resulting state is being persisted.
                        provenance="convergence_persist",
                    )

                    new_terminal_response = TerminalStateResponse(
                        kind=new_terminal.kind.value,
                        reason=new_terminal.reason.value if new_terminal.reason is not None else None,
                        pipeline_yaml=new_terminal.pipeline_yaml,
                    )
                    return GuidedRespondResponse(
                        guided_session=GuidedSessionResponse(
                            step=new_guided.step.value,
                            history=[
                                TurnRecordResponse(
                                    step=r.step.value,
                                    turn_type=r.turn_type.value,
                                    payload_hash=r.payload_hash,
                                    response_hash=r.response_hash,
                                    emitter=r.emitter,
                                )
                                for r in new_guided.history
                            ],
                            terminal=new_terminal_response,
                            chat_history=[
                                ChatTurnResponse(
                                    role=t.role.value,
                                    content=t.content,
                                    seq=t.seq,
                                    step=t.step.value,
                                    ts_iso=t.ts_iso,
                                )
                                for t in new_guided.chat_history
                            ],
                            chat_turn_seq=new_guided.chat_turn_seq,
                        ),
                        next_turn=None,
                        terminal=new_terminal_response,
                        composition_state=_state_response(state_record_out),
                    )

                # Reject if session already terminal (any case not handled by
                # the exit-from-COMPLETED branch above).
                if guided.terminal is not None:
                    raise HTTPException(
                        status_code=409,
                        detail="Guided session is already in a terminal state. No further responses accepted.",
                    )

                # Derive the current turn type from the last TurnRecord for the
                # current step.  Crash if history is empty — the caller must have
                # fetched GET /guided first (which seeds the initial TurnRecord).
                current_step = guided.step
                existing_record: TurnRecord | None = next(
                    (r for r in reversed(guided.history) if r.step == current_step),
                    None,
                )
                if existing_record is None:
                    raise HTTPException(
                        status_code=400,
                        detail=("No turn has been emitted for the current step. Fetch GET /api/sessions/{id}/guided first."),
                    )

                current_turn_type = existing_record.turn_type

                # --- Wire-boundary validation (Codex #7, #12) -------------------
                # ``control_signal`` was parsed earlier (above the
                # exit-from-COMPLETED branch and the generic 409 guard) so
                # that the meta-control signal could be recognised before the
                # terminal-rejection short-circuit fires.  The parsed
                # ``ControlSignal | None`` flows into the typed ``TurnResponse``
                # below.

                # Codex #7: validate step-index fields against the current proposal.
                _validate_step_indices(
                    current_turn_type,
                    body.accepted_step_index,
                    body.edit_step_index,
                    guided,
                )

                # Build the TurnResponse dict from the request body.
                from dataclasses import replace as _replace

                turn_response: TurnResponse = {
                    "chosen": body.chosen,
                    "edited_values": body.edited_values,
                    "custom_inputs": body.custom_inputs,
                    "accepted_step_index": body.accepted_step_index,
                    "edit_step_index": body.edit_step_index,
                    "control_signal": control_signal,
                }

                # Record the response_hash on the existing TurnRecord.
                response_hash = stable_hash(turn_response)
                updated_record = _replace(existing_record, response_hash=response_hash)
                # Rebuild history tuple with response_hash stamped on this record.
                updated_history = tuple(updated_record if r is existing_record else r for r in guided.history)
                guided = _replace(guided, history=updated_history)

                # Emit guided_turn_answered audit event.  This writes to the
                # recorder BEFORE the dispatcher's try-block — if the
                # dispatcher then raises 400, this event must still reach
                # the audit DB (see PR-review B3 finally-drain rationale).
                emit_turn_answered(
                    recorder,
                    step=current_step,
                    turn_type=current_turn_type,
                    response_hash=response_hash,
                    response_payload_id="",
                    control_signal=body.control_signal,
                    composition_version=state.version,
                    actor=user.user_id,
                )

                # Run step_advance (pure — no I/O).
                # InvariantError indicates a server-side bug (e.g. stamped an
                # invalid turn type on a history record) — propagate as HTTP 500
                # with a static message so the response body never carries
                # ``str(exc)``. ``InvariantError`` raised from ``from_dict`` call
                # sites embeds ``{d!r}`` of the corrupted Tier-1 record, which
                # includes Tier-3 ``sample_rows`` source data. Interpolating that
                # into the HTTP detail field would have leaked user/PII content
                # into the JSON 500 body returned to the browser (PR #37 review
                # finding B1).
                #
                # Diagnostic trace is preserved via ``slog.error`` under the
                # audit-system-failure exemption (CLAUDE.md primacy order: when
                # ``from_dict`` itself fails, the audit-derivation path can't be
                # trusted to record the failure consistently). The slog event
                # carries ``exc_class`` + ``frames`` only — never the message
                # string, since for ``{d!r}`` -bearing InvariantErrors that text
                # is the leak vector. Sibling helpers in this file follow the
                # same "class + frames, no message" pattern when the exception
                # carries Tier-bearing strings (see ``_safe_frame_strings`` docs).
                #
                # ValueError indicates a client-supplied payload violated the
                # guided-mode protocol contract (e.g. unexpected chosen value on a
                # recipe_offer turn, or null edited_values on inspect_and_confirm)
                # — propagate as HTTP 400 so the caller can correct the request.
                # ValueError is not raised by ``from_dict`` and does not embed
                # ``{d!r}`` of Tier-1 records, so interpolating ``str(exc)`` here
                # is safe.
                try:
                    new_guided, _next_turn_from_advance, terminal_from_advance, directives = step_advance(
                        guided,
                        turn_response,
                        current_turn_type=current_turn_type,
                    )
                except InvariantError as exc:
                    slog.error(
                        "guided.invariant_violated",
                        session_id=str(session_id),
                        user_id=user.user_id,
                        exc_class=type(exc).__name__,
                        site="step_advance",
                        frames=_safe_frame_strings(exc),
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="Server invariant violated. See application audit log for diagnostic detail.",
                    ) from exc
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Guided-mode protocol error: {exc}",
                    ) from exc

                # Fan directives to emit_* helpers.
                for directive in directives:
                    if directive.tool_name == "guided_step_advanced":
                        args = dict(directive.arguments)
                        emit_step_advanced(
                            recorder,
                            prev=GuidedStep(args["prev_step"]),
                            next_=GuidedStep(args["next_step"]),
                            reason=args["reason"],
                            composition_version=state.version,
                            actor=user.user_id,
                        )
                    elif directive.tool_name == "guided_dropped_to_freeform":
                        args = dict(directive.arguments)
                        emit_dropped_to_freeform(
                            recorder,
                            prev=GuidedStep(args["prev_step"]),
                            drop_reason=TerminalReason(args["drop_reason"]),
                            validation_result=args["validation_result"],
                            composition_version=state.version,
                            actor=user.user_id,
                        )

                guided = new_guided
                terminal = terminal_from_advance

                # Run side-effect dispatcher if the session is not yet terminal.
                # The dispatcher calls step handlers (handle_step_1_source,
                # handle_step_2_sink, handle_step_2_5_recipe_apply) and emits
                # the next turn based on the updated step + turn type.
                next_turn: Any | None = None
                settings = request.app.state.settings
                data_dir: str | None = str(settings.data_dir) if settings.data_dir else None
                session_engine = request.app.state.session_engine

                if terminal is None:
                    try:
                        state, guided, next_turn = await _dispatch_guided_respond(
                            state=state,
                            guided=guided,
                            current_step=current_step,
                            current_turn_type=current_turn_type,
                            turn_response=turn_response,
                            catalog=catalog,
                            recorder=recorder,
                            user_id=user.user_id,
                            data_dir=data_dir,
                            session_engine=session_engine,
                            session_id=str(session_id),
                            model=settings.composer_model,
                        )
                    except InvariantError as exc:
                        # Same B1-sanitization rationale as the step_advance
                        # catch above: ``str(exc)`` from a ``from_dict`` site
                        # embeds ``{d!r}`` of the corrupted Tier-1 record
                        # including Tier-3 ``sample_rows``. Static detail; slog
                        # carries exc_class + frames only.
                        slog.error(
                            "guided.invariant_violated",
                            session_id=str(session_id),
                            user_id=user.user_id,
                            exc_class=type(exc).__name__,
                            site="dispatch_guided_respond",
                            frames=_safe_frame_strings(exc),
                        )
                        raise HTTPException(
                            status_code=500,
                            detail="Server invariant violated. See application audit log for diagnostic detail.",
                        ) from exc
                    except ValueError as exc:
                        # ValueError from inside the dispatcher indicates a
                        # client-supplied payload violated the guided-mode protocol
                        # contract (e.g. unknown plugin name from chosen, null
                        # edited_values) — propagate as HTTP 400. See Codex #8,
                        # #11, #15 and commit message for full context.
                        raise HTTPException(
                            status_code=400,
                            detail=f"Guided-mode protocol error: {exc}",
                        ) from exc
                    terminal = guided.terminal

                # Persist updated state.
                new_state = _replace(state, guided_session=guided)
                existing_meta: dict[str, Any] = {}
                if state_record is not None and state_record.composer_meta is not None:
                    existing_meta = dict(deep_thaw(state_record.composer_meta))
                new_composer_meta = {**existing_meta, "guided_session": guided.to_dict()}

                state_d = new_state.to_dict()
                state_data = CompositionStateData(
                    source=state_d["source"],
                    nodes=state_d["nodes"],
                    edges=state_d["edges"],
                    outputs=state_d["outputs"],
                    metadata_=state_d["metadata"],
                    is_valid=False,
                    validation_errors=None,
                    composer_meta=new_composer_meta,
                )
                state_record_out = await service.save_composition_state(
                    session_id,
                    state_data,
                    # Guided POST /respond: the user-supplied turn converged on a
                    # guided step transition and the resulting state is being
                    # persisted. Same provenance choice as the GET /guided server-
                    # emitted path (the Phase 1A enum predates guided mode; widening
                    # is out of scope here — see merge commit message).
                    provenance="convergence_persist",
                )

                # Recorder persistence happens in the finally block below so
                # rejection paths drain identically to the success path.

                return GuidedRespondResponse(
                    guided_session=GuidedSessionResponse(
                        step=guided.step.value,
                        history=[
                            TurnRecordResponse(
                                step=r.step.value,
                                turn_type=r.turn_type.value,
                                payload_hash=r.payload_hash,
                                response_hash=r.response_hash,
                                emitter=r.emitter,
                            )
                            for r in guided.history
                        ],
                        terminal=TerminalStateResponse(
                            kind=terminal.kind.value,
                            reason=terminal.reason.value if terminal.reason is not None else None,
                            pipeline_yaml=terminal.pipeline_yaml,
                        )
                        if terminal is not None
                        else None,
                        chat_history=[
                            ChatTurnResponse(
                                role=t.role.value,
                                content=t.content,
                                seq=t.seq,
                                step=t.step.value,
                                ts_iso=t.ts_iso,
                            )
                            for t in guided.chat_history
                        ],
                        chat_turn_seq=guided.chat_turn_seq,
                    ),
                    next_turn=TurnPayloadResponse(
                        type=next_turn["type"],
                        step_index=next_turn["step_index"],
                        payload=dict(next_turn["payload"]),
                    )
                    if next_turn is not None
                    else None,
                    terminal=TerminalStateResponse(
                        kind=terminal.kind.value,
                        reason=terminal.reason.value if terminal.reason is not None else None,
                        pipeline_yaml=terminal.pipeline_yaml,
                    )
                    if terminal is not None
                    else None,
                    composition_state=_state_response(state_record_out),
                )
            finally:
                # PR-review B3: drain the recorder unconditionally — success
                # paths and ``raise HTTPException`` rejection paths take the
                # same exit.  Empty drains are a no-op (BufferingRecorder
                # starts with an empty invocations list and
                # ``_persist_tool_invocations`` iterates an empty tuple).
                #
                # The suppress-and-log path is only for exception unwinds:
                # Python's default behaviour is to let a ``finally``-block
                # exception replace the original, which would surface a generic
                # 500 instead of the intended 400/409.  On a successful return,
                # audit persist failures must propagate — otherwise a state
                # write can succeed while the guided audit row silently
                # disappears.  Per CLAUDE.md telemetry/logging primacy,
                # audit-system failures during exception handling are the one
                # exemption where ``slog`` is the correct channel.  The log
                # payload follows the B1 convention: ``exc_class`` + ``frames``
                # only, never ``str(exc)`` or ``exc_info`` (frames are bounded
                # and value-free; the exception message can carry Tier-bearing
                # strings).
                #
                # The two recorder channels (tool invocations and LLM calls)
                # drain through TWO separate try blocks so that a failure
                # persisting one does not skip the other.  ``_persist_llm_calls``
                # covers the :class:`ComposerLLMCall` rows that ``solve_chain``
                # buffers during guided Step 3 (chain solver) invocations.
                # Without the second drain the LLM-call audit would be
                # garbage-collected with the recorder at function exit.
                primary_exc = sys.exception()
                if primary_exc is None:
                    await _persist_tool_invocations(
                        service,
                        session_id,
                        recorder.invocations,
                        state_record_out.id if state_record_out is not None else None,
                        # Guided endpoints don't dispatch tools that can plugin-crash;
                        # auto-drop on solver exhaustion is a separate channel that
                        # doesn't surface through this audit path. Phase 1B/rev-4
                        # made this keyword-only with no default, exposing the gap.
                        plugin_crash_pending=False,
                    )
                    await _persist_llm_calls(
                        service,
                        session_id,
                        recorder.llm_calls,
                        state_record_out.id if state_record_out is not None else None,
                        plugin_crash_pending=False,
                    )
                else:
                    try:
                        await _persist_tool_invocations(
                            service,
                            session_id,
                            recorder.invocations,
                            state_record_out.id if state_record_out is not None else None,
                            plugin_crash_pending=False,
                        )
                    except Exception as persist_exc:
                        with contextlib.suppress(Exception):
                            slog.error(
                                "guided.audit_persist_failed_during_exception_handling",
                                session_id=str(session_id),
                                user_id=user.user_id,
                                site="post_guided_respond",
                                channel="tool_invocations",
                                exc_class=type(persist_exc).__name__,
                                frames=_safe_frame_strings(persist_exc),
                            )
                    try:
                        await _persist_llm_calls(
                            service,
                            session_id,
                            recorder.llm_calls,
                            state_record_out.id if state_record_out is not None else None,
                            plugin_crash_pending=False,
                        )
                    except Exception as persist_exc:
                        with contextlib.suppress(Exception):
                            slog.error(
                                "guided.audit_persist_failed_during_exception_handling",
                                session_id=str(session_id),
                                user_id=user.user_id,
                                site="post_guided_respond",
                                channel="llm_calls",
                                exc_class=type(persist_exc).__name__,
                                frames=_safe_frame_strings(persist_exc),
                            )

    @router.post("/{session_id}/guided/chat", response_model=GuidedChatResponse)
    async def post_guided_chat(
        session_id: UUID,
        body: GuidedChatRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> GuidedChatResponse:
        """Submit a free-text chat message scoped to the user's current wizard step.

        **Not** a turn-answer — chat does not advance step state. The
        backend resolves the per-step skill briefing via
        :func:`elspeth.web.composer.guided.chat_solver.solve_step_chat`
        and returns the LLM's advisory reply. The frontend renders the
        reply inline in the guided history (slice 6's
        ``GuidedChatHistory`` component).

        Phase A is advisory-only; no tool palette, no state mutation.
        Slice 5 introduces ``ComposerChatTurn`` audit + ``chat_history``
        persistence on the ``GuidedSession``.

        Raises 400 if the session has no ``guided_session`` attached.
        Raises 400 if ``step_index`` is not a known ``GuidedStep`` value.
        Raises 409 if the guided session is already in a terminal state.
        Raises 409 if ``step_index`` does not match the session's current
        step (the wizard advanced under the user — client must re-fetch
        ``GET /guided`` and retry).
        Raises 404 if the session does not exist or belong to the user.

        Empty / oversize messages are rejected at the Pydantic boundary
        (HTTP 422). The route never reaches ``solve_step_chat`` with an
        invalid message; the solver's empty-string guard is a redundant
        defense-in-depth invariant, not the boundary check.

        Transient LLM failures (LiteLLM API/auth/bad-request, asyncio
        timeout, malformed response shape) return 200 with a synthetic
        unavailable message; the session is **not** terminated. This is
        intentional: chat is a non-load-bearing helper. Wizard widgets
        remain functional even when chat is offline.
        """
        await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service

        # Tier-3 → Tier-2 coercion at the step_index boundary. A stale
        # client sending an unknown value gets a 400 with a clear message
        # rather than a Pydantic 422; the typed ``GuidedStep`` then flows
        # into the equality check and the solver call site.
        try:
            requested_step = GuidedStep(body.step_index)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown step_index {body.step_index!r}. Valid values: {sorted(s.value for s in GuidedStep)}.",
            ) from exc

        compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))
        async with compose_lock:
            # Audit drain (slice 5.1): every chat turn lands as a
            # role=audit row tagged ``_kind=chat_turn_audit``.  The
            # recorder buffers in-memory during the request body; the
            # finally block drains it via _persist_chat_turns regardless
            # of exit path (success, 409, 400, or unexpected).  This is
            # the CLAUDE.md "no silent telemetry drop" contract: a
            # ComposerChatTurn that was constructed but never persisted
            # would be evidence tampering.  ``state_record_out`` is
            # captured to thread the persisted composition_state.id
            # into the audit envelope so an auditor can correlate the
            # chat turn to the state version it ran against.
            recorder = BufferingRecorder()
            state_record_out: CompositionStateRecord | None = None
            try:
                state_record = await service.get_current_state(session_id)
                if state_record is None:
                    state = _initial_composition_state_with_guided_session()
                else:
                    state = _state_from_record(state_record)
                    state_record_out = state_record

                if state.guided_session is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Session is not in guided mode. Use /api/sessions/{id}/messages.",
                    )

                guided = state.guided_session

                if guided.terminal is not None:
                    raise HTTPException(
                        status_code=409,
                        detail="Guided session is already in a terminal state. No further chat accepted.",
                    )

                # Step-mismatch is a state-conflict (the wizard advanced under
                # the user between client read and write), not a malformed
                # request — 409 mirrors the ``terminal`` case. The detail
                # carries both values so the client can re-fetch the right
                # step without a separate round-trip.
                if requested_step is not guided.step:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"step_index {requested_step.value!r} does not match the session's current step "
                            f"{guided.step.value!r}. Re-fetch GET /api/sessions/{{id}}/guided and retry."
                        ),
                    )

                settings = request.app.state.settings
                started_at = datetime.now(UTC)
                from time import perf_counter as _perf_counter

                started_perf = _perf_counter()
                # InvariantError from solve_step_chat (empty / whitespace LLM
                # content) indicates a defective model response we cannot
                # recover from.  Mirror of the post_guided_respond pattern at
                # the step_advance call site (line ~5044): sanitize to a
                # static 500 detail, emit slog with safe frame strings only
                # (no str(exc) since the InvariantError message embeds the
                # model name and step value — class + frames only, B1
                # convention), and re-raise so the audit-drain finally still
                # fires.  The chat handler being inconsistent with
                # post_guided_respond's InvariantError discipline was the
                # original gap surfaced by elspeth-obs-ac603d4e03.
                try:
                    chat_result = await solve_step_chat_with_auto_drop(
                        site="post_guided_chat",
                        session_id=str(session_id),
                        user_id=user.user_id,
                        model=settings.composer_model,
                        step=guided.step,
                        user_message=body.message,
                    )
                except InvariantError as exc:
                    finished_at = datetime.now(UTC)
                    latency_ms = int((_perf_counter() - started_perf) * 1000)
                    user_turn = ChatTurn(
                        role=ChatRole.USER,
                        content=body.message,
                        seq=guided.chat_turn_seq,
                        step=guided.step,
                        ts_iso=finished_at.isoformat(),
                    )
                    recorder.record_chat_turn(
                        ComposerChatTurn(
                            step=guided.step.value,
                            initiator=ComposerChatInitiator.USER,
                            chat_turn_seq=user_turn.seq,
                            user_message_hash=stable_hash(body.message),
                            assistant_message_hash=stable_hash(""),
                            latency_ms=latency_ms,
                            model=settings.composer_model,
                            status=ComposerChatTurnStatus.INVARIANT_VIOLATED,
                            started_at=started_at,
                            finished_at=finished_at,
                            error_class=type(exc).__name__,
                        )
                    )
                    slog.error(
                        "guided.invariant_violated",
                        session_id=str(session_id),
                        user_id=user.user_id,
                        exc_class=type(exc).__name__,
                        site="solve_step_chat",
                        frames=_safe_frame_strings(exc),
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="Server invariant violated. See application audit log for diagnostic detail.",
                    ) from exc
                finished_at = datetime.now(UTC)

                # Append both turns (user + assistant) to chat_history with
                # consecutive seq values, then bump chat_turn_seq past the pair.
                # Phase A keeps user and assistant turns in the same atomic
                # state update — a half-applied history (user without assistant)
                # would surface mid-flight on a concurrent /guided read.
                ts_iso = finished_at.isoformat()
                user_turn = ChatTurn(
                    role=ChatRole.USER,
                    content=body.message,
                    seq=guided.chat_turn_seq,
                    step=guided.step,
                    ts_iso=ts_iso,
                )
                assistant_turn = ChatTurn(
                    role=ChatRole.ASSISTANT,
                    content=chat_result.assistant_message,
                    seq=guided.chat_turn_seq + 1,
                    step=guided.step,
                    ts_iso=ts_iso,
                )
                new_guided = _replace(
                    guided,
                    chat_history=(*guided.chat_history, user_turn, assistant_turn),
                    chat_turn_seq=guided.chat_turn_seq + 2,
                )

                # Emit the ComposerChatTurn audit record.  Hashes use the
                # project canonical ``stable_hash`` over the literal message
                # strings — never the raw text into the audit row.  The
                # ``initiator`` is hard-coded to USER for Phase A; Phase A.5
                # will set STEP_ENTRY_OPENER for proactive turns through the
                # same record.
                user_message_hash = stable_hash(body.message)
                assistant_message_hash = stable_hash(chat_result.assistant_message)
                recorder.record_chat_turn(
                    ComposerChatTurn(
                        step=guided.step.value,
                        initiator=ComposerChatInitiator.USER,
                        chat_turn_seq=user_turn.seq,
                        user_message_hash=user_message_hash,
                        assistant_message_hash=assistant_message_hash,
                        latency_ms=chat_result.latency_ms,
                        model=settings.composer_model,
                        status=chat_result.status,
                        started_at=started_at,
                        finished_at=finished_at,
                        error_class=chat_result.error_class,
                    )
                )

                # Persist the updated GuidedSession.  Mirrors the persistence
                # pattern in ``post_guided_respond``: replace state with the
                # new guided_session, round-trip composer_meta through
                # ``to_dict()`` so the field carries the new chat_history /
                # chat_turn_seq values.
                new_state = _replace(state, guided_session=new_guided)
                existing_meta: dict[str, Any] = {}
                if state_record is not None and state_record.composer_meta is not None:
                    existing_meta = dict(deep_thaw(state_record.composer_meta))
                new_composer_meta = {**existing_meta, "guided_session": new_guided.to_dict()}

                state_d = new_state.to_dict()
                state_data = CompositionStateData(
                    source=state_d["source"],
                    nodes=state_d["nodes"],
                    edges=state_d["edges"],
                    outputs=state_d["outputs"],
                    metadata_=state_d["metadata"],
                    is_valid=False,
                    validation_errors=None,
                    composer_meta=new_composer_meta,
                )
                state_record_out = await service.save_composition_state(
                    session_id,
                    state_data,
                    # Per-step chat persists guided-session metadata
                    # (chat_history/chat_turn_seq) after the LLM response has
                    # converged. The Phase 1A provenance enum predates guided
                    # chat, so use the same closest category as sibling guided
                    # state writes rather than widening the closed list mid-merge.
                    provenance="convergence_persist",
                )

                return GuidedChatResponse(
                    assistant_message=chat_result.assistant_message,
                    guided_session=GuidedSessionResponse(
                        step=new_guided.step.value,
                        history=[
                            TurnRecordResponse(
                                step=r.step.value,
                                turn_type=r.turn_type.value,
                                payload_hash=r.payload_hash,
                                response_hash=r.response_hash,
                                emitter=r.emitter,
                            )
                            for r in new_guided.history
                        ],
                        terminal=None,
                        chat_history=[
                            ChatTurnResponse(
                                role=t.role.value,
                                content=t.content,
                                seq=t.seq,
                                step=t.step.value,
                                ts_iso=t.ts_iso,
                            )
                            for t in new_guided.chat_history
                        ],
                        chat_turn_seq=new_guided.chat_turn_seq,
                    ),
                )
            finally:
                # Drain the recorder unconditionally — same B3 pattern as
                # post_guided_respond.  Success-path audit persist failures
                # propagate: the state write above may already have stored
                # chat_history/chat_turn_seq, and returning 200 without the
                # corresponding audit-only row would create an evidence gap.
                #
                # During exception unwinds, audit-system failures are logged
                # rather than masking the primary HTTPException.  This mirrors
                # the guided/respond split and keeps logging as the channel of
                # last resort only when no safer audit channel remains.
                primary_exc = sys.exception()
                if primary_exc is None:
                    await _persist_chat_turns(
                        service,
                        session_id,
                        recorder.chat_turns,
                        state_record_out.id if state_record_out is not None else None,
                        request_unwinding=False,
                    )
                else:
                    try:
                        await _persist_chat_turns(
                            service,
                            session_id,
                            recorder.chat_turns,
                            state_record_out.id if state_record_out is not None else None,
                            request_unwinding=True,
                        )
                    except Exception as persist_exc:
                        with contextlib.suppress(Exception):
                            slog.error(
                                "guided.chat_turn_persist_failed_during_exception_handling",
                                session_id=str(session_id),
                                user_id=user.user_id,
                                site="post_guided_chat",
                                exc_class=type(persist_exc).__name__,
                                frames=_safe_frame_strings(persist_exc),
                            )

    return router


def _guided_step_index(step: Any) -> int:
    """Map a GuidedStep to its 0-based integer index.

    Mirrors ``emitters._step_index``; defined here so the route handler
    can reconstruct a re-fetch Turn without importing the emitters module
    just for this utility.
    """
    from elspeth.web.composer.guided.protocol import GuidedStep

    _ORDER: tuple[GuidedStep, ...] = (
        GuidedStep.STEP_1_SOURCE,
        GuidedStep.STEP_2_SINK,
        GuidedStep.STEP_2_5_RECIPE_MATCH,
        GuidedStep.STEP_3_TRANSFORMS,
    )
    return _ORDER.index(step)
