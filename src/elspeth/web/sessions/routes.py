"""Session API routes -- /api/sessions/* with IDOR protection.

All endpoints require authentication via Depends(get_current_user).
Session-scoped endpoints verify ownership before any business logic.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from opentelemetry import metrics
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
from elspeth.contracts.composer_llm_audit import ComposerLLMCall
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.secret_scrub import scrub_text_for_audit
from elspeth.core.dag.models import GraphValidationError
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.infrastructure.manager import PluginNotFoundError
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.protocol import BlobQuotaExceededError, BlobServiceProtocol
from elspeth.web.composer import yaml_generator
from elspeth.web.composer.audit import audit_envelope, llm_call_audit_envelope
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
from elspeth.web.composer.yaml_generator import generate_yaml
from elspeth.web.execution.accounting import load_run_accounting_for_settings
from elspeth.web.execution.schemas import RunAccounting, RunStatusResponse, ValidationResult
from elspeth.web.execution.validation import validate_pipeline
from elspeth.web.middleware.rate_limit import ComposerRateLimiter, get_rate_limiter
from elspeth.web.sessions.converters import state_from_record as _state_from_record
from elspeth.web.sessions.protocol import (
    SESSION_TERMINAL_RUN_STATUS_VALUES,
    ChatMessageRecord,
    CompositionStateData,
    CompositionStateRecord,
    InvalidForkTargetError,
    RunRecord,
    SessionRecord,
    SessionServiceProtocol,
)
from elspeth.web.sessions.schemas import (
    ChatMessageResponse,
    CompositionStateResponse,
    CreateSessionRequest,
    ForkSessionRequest,
    ForkSessionResponse,
    MessageWithStateResponse,
    RevertStateRequest,
    RunResponse,
    SendMessageRequest,
    SessionResponse,
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
    """Return true when a persisted chat row is an audit-only composer tool row."""
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
    """Return true only for persisted composer LLM-call audit sidecars."""
    if message.role != "tool" or message.tool_calls is None:
        return False
    return any("_kind" in tool_call and tool_call["_kind"] == "llm_call_audit" for tool_call in message.tool_calls)


def _composer_conversation_messages(messages: Sequence[ChatMessageRecord]) -> list[ChatMessageRecord]:
    """Return persisted messages that are part of the LLM conversation."""
    return [message for message in messages if not _is_composer_audit_tool_message(message)]


def _composer_conversation_or_llm_audit_messages(messages: Sequence[ChatMessageRecord]) -> list[ChatMessageRecord]:
    """Return user-visible conversation plus safe per-LLM-call audit sidecars."""
    return [message for message in messages if not _is_composer_audit_tool_message(message) or _is_composer_llm_audit_tool_message(message)]


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
) -> None:
    """Persist per-tool-call audit records as ``role=tool`` chat messages.

    Each :class:`ComposerToolInvocation` lands as one ``role=tool`` row whose
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

    Audit primacy: a SQLAlchemy failure here MUST NOT mask the calling
    handler's primary outcome (the assistant message has already been
    written, the partial-state row has already been written). The narrow
    catch matches the discipline used in ``_handle_*_error`` helpers —
    log with class name only and continue. The tool-row gap is observable
    via the per-message ``tool_calls`` count vs.
    ``ComposerResult.tool_invocations`` length on read-back.
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
        try:
            await service.add_message(
                session_id,
                "tool",
                content,
                tool_calls=[audit_envelope(invocation)],
                composition_state_id=composition_state_id,
            )
        except SQLAlchemyError as save_err:
            slog.error(
                "composer_tool_invocation_persist_failed",
                session_id=str(session_id),
                tool_call_id=invocation.tool_call_id,
                tool_name=invocation.tool_name,
                exc_class=type(save_err).__name__,
            )
            # Continue — the goal is to preserve as much of the audit
            # trail as possible. Logging this as a structured event lets
            # operators detect partial-trail persistence by comparing
            # event counts to the assistant's reported tool_invocations.


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
) -> None:
    """Persist per-LLM-call audit records as audit-only ``role=tool`` rows."""
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
                "tool",
                content,
                tool_calls=[llm_call_audit_envelope(call)],
                composition_state_id=composition_state_id,
            )
        except SQLAlchemyError as save_err:
            slog.error(
                "composer_llm_call_persist_failed",
                session_id=str(session_id),
                model_requested=call.model_requested,
                status=call.status.value,
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
            partial_record = await service.save_composition_state(session_id, state_data)
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
    if exc.tool_invocations:
        await _persist_tool_invocations(
            service,
            session_id,
            exc.tool_invocations,
            persisted_state_id,
        )
    if exc.llm_calls:
        await _persist_llm_calls(
            service,
            session_id,
            exc.llm_calls,
            llm_composition_state_id,
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
            partial_record = await service.save_composition_state(session_id, state_data)
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
    if exc.tool_invocations:
        await _persist_tool_invocations(
            service,
            session_id,
            exc.tool_invocations,
            persisted_state_id_pc,
        )
    if exc.llm_calls:
        await _persist_llm_calls(
            service,
            session_id,
            exc.llm_calls,
            llm_composition_state_id,
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
            partial_record = await service.save_composition_state(session_id, state_data)
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
    if exc.tool_invocations:
        await _persist_tool_invocations(
            service,
            session_id,
            exc.tool_invocations,
            persisted_state_id_rpf,
        )
    if exc.llm_calls:
        await _persist_llm_calls(
            service,
            session_id,
            exc.llm_calls,
            llm_composition_state_id,
        )
    return response_body


session_router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@session_router.post("", status_code=201, response_model=SessionResponse)
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


@session_router.get("", response_model=list[SessionResponse])
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
@session_router.get("/_active", response_model=list[ComposerProgressSnapshot])
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


@session_router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> SessionResponse:
    """Get a single session. IDOR-protected."""
    session = await _verify_session_ownership(session_id, user, request)
    return _session_response(session)


@session_router.get(
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


@session_router.delete("/{session_id}", status_code=204)
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


@session_router.post(
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
            state = CompositionState(
                source=None,
                nodes=(),
                edges=(),
                outputs=(),
                metadata=PipelineMetadata(),
                version=1,
            )
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

        # 2. Persist user message with pre-send provenance.
        # Keep the inserted row so the subsequent snapshot can prove
        # it is composing against the transcript that actually ends
        # at this request's user turn.
        user_msg = await service.add_message(
            session.id,
            "user",
            body.content,
            composition_state_id=pre_send_state_id,
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
                    await _persist_llm_calls(service, session.id, llm_calls, pre_send_state_id)
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
                    await _persist_llm_calls(service, session.id, llm_calls, pre_send_state_id)
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
                    await _persist_llm_calls(service, session.id, llm_calls, pre_send_state_id)
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
                        composer_meta={"repair_turns_used": result.repair_turns_used},
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
                )
                state_response = _state_response(new_state_record, live_validation=validation)
                post_compose_state_id = new_state_record.id

            # 6. Persist assistant message with post-compose provenance
            assistant_msg = await service.add_message(
                session.id,
                "assistant",
                result.message,
                composition_state_id=post_compose_state_id,
                raw_content=result.raw_assistant_content,
            )
            # 6b. Persist per-tool-call audit trail. Each ComposerToolInvocation
            # lands as one role=tool chat message linked to the post-compose
            # state id (when version advanced) so the audit trail records
            # which tool calls produced this state.
            if result.tool_invocations:
                await _persist_tool_invocations(
                    service,
                    session.id,
                    result.tool_invocations,
                    post_compose_state_id,
                )
            if result.llm_calls:
                await _persist_llm_calls(
                    service,
                    session.id,
                    result.llm_calls,
                    pre_send_state_id,
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
                    await asyncio.shield(_persist_llm_calls(service, session.id, llm_calls, pre_send_state_id))
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


@session_router.post(
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
            state = CompositionState(
                source=None,
                nodes=(),
                edges=(),
                outputs=(),
                metadata=PipelineMetadata(),
                version=1,
            )
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
                    await _persist_llm_calls(service, session.id, llm_calls, pre_send_state_id)
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
                    await _persist_llm_calls(service, session.id, llm_calls, pre_send_state_id)
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
                    await _persist_llm_calls(service, session.id, llm_calls, pre_send_state_id)
                raise HTTPException(
                    status_code=502,
                    detail={"error_type": "composer_error", "detail": str(exc)},
                ) from exc

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
                        composer_meta={"repair_turns_used": result.repair_turns_used},
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
                )
                state_response = _state_response(new_state_record, live_validation=validation)
                post_compose_state_id = new_state_record.id

            # Persist assistant message
            assistant_msg = await service.add_message(
                session.id,
                "assistant",
                result.message,
                composition_state_id=post_compose_state_id,
                raw_content=result.raw_assistant_content,
            )
            # Per-tool-call audit trail (recompose path; symmetric with send_message).
            if result.tool_invocations:
                await _persist_tool_invocations(
                    service,
                    session.id,
                    result.tool_invocations,
                    post_compose_state_id,
                )
            if result.llm_calls:
                await _persist_llm_calls(
                    service,
                    session.id,
                    result.llm_calls,
                    pre_send_state_id,
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
        except asyncio.CancelledError as exc:
            # Mirror of send_message cancellation path. See block
            # comment there for the shielded-publish rationale.
            llm_calls = _llm_calls_from_exception(exc)
            if llm_calls:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(_persist_llm_calls(service, session.id, llm_calls, pre_send_state_id))
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


@session_router.get(
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
    # Fetch before slicing so hidden audit rows cannot skew normal-chat
    # pagination. The service remains the durable audit store; this route
    # is the user-facing conversation channel. The eval harness can opt in
    # to LLM-call sidecars, which contain model/usage/cost metadata but not
    # raw prompts, tool arguments, or tool results.
    messages = await service.get_messages(session.id, limit=None)
    conversation_messages = (
        _composer_conversation_or_llm_audit_messages(messages) if include_llm_audit else _composer_conversation_messages(messages)
    )
    paged_messages = conversation_messages[offset : offset + limit]
    return [_message_response(m, include_raw_content=include_raw_content) for m in paged_messages]


@session_router.get(
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


@session_router.get("/{session_id}/state")
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


@session_router.get(
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


@session_router.post(
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
    )

    return _state_response(new_state)


@session_router.get("/{session_id}/state/yaml")
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


@session_router.post(
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
                    composition_state_id=copied_state.id,
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


def create_session_router() -> APIRouter:
    """Create the session router with /api/sessions prefix."""
    return session_router
