"""Session API routes -- /api/sessions/* with IDOR protection.

All endpoints require authentication via Depends(get_current_user).
Session-scoped endpoints verify ownership before any business logic.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace as _replace
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from opentelemetry import metrics
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import insert
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
from elspeth.contracts.composer_interpretation import (
    InterpretationChoice,
    InterpretationEventRecord,
    InterpretationSource,
)
from elspeth.contracts.composer_llm_audit import (
    ComposerChatInitiator,
    ComposerChatTurn,
    ComposerChatTurnStatus,
    ComposerLLMCall,
)
from elspeth.contracts.composer_progress import ComposerProgressEvent, ComposerProgressSink
from elspeth.contracts.errors import AuditIntegrityError, FailedTurnMetadata
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.secret_scrub import scrub_text_for_audit
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.dag.models import GraphValidationError
from elspeth.core.landscape.database import LandscapeDB
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.infrastructure.manager import PluginNotFoundError
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.protocol import BlobQuotaExceededError, BlobServiceProtocol
from elspeth.web.catalog.knob_schema import KnobField, KnobSchema
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
    emit_hidden_field_rejected,
    emit_step_advanced,
    emit_turn_answered,
    emit_turn_emitted,
)
from elspeth.web.composer.guided.chat_solver import maybe_resolve_step_1_source_chat
from elspeth.web.composer.guided.emitters import (
    build_initial_step_1_turn,
    build_step_1_inspect_and_confirm_turn_from_intent,
    build_step_1_schema_form_turn,
    build_step_2_5_recipe_offer_turn,
    build_step_2_multi_select_turn,
    build_step_2_schema_form_turn,
    build_step_2_single_select_turn,
    build_step_3_propose_chain_turn,
    build_step_3_schema_form_turn,
)
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.profile import EMPTY_PROFILE, WorkflowProfile
from elspeth.web.composer.guided.protocol import ChatRole, ChatTurn, ControlSignal, GuidedStep, TurnResponse, TurnType
from elspeth.web.composer.guided.recipe_match import match_recipe
from elspeth.web.composer.guided.state_machine import (
    ChainProposal,
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
from elspeth.web.composer.implicit_decisions import merge_implicit_decisions_meta
from elspeth.web.composer.progress import (
    ComposerProgressRegistry,
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
from elspeth.web.composer.redaction import (
    MANIFEST,
    redact_source_storage_path,
    redact_tool_call_arguments,
    redact_tool_call_response,
)
from elspeth.web.composer.redaction_telemetry import OtelRedactionTelemetry
from elspeth.web.composer.service import _BadRequestLLMError
from elspeth.web.composer.source_inspection import SourceInspectionFacts, inspect_blob_content
from elspeth.web.composer.state import CompositionState, PipelineMetadata, ValidationEntry, ValidationSummary
from elspeth.web.composer.telemetry_phase8 import (
    SessionsTelemetry,
    record_session_completed,
    record_session_switched,
)
from elspeth.web.composer.tools import _DATA_ERROR_KEY, execute_tool
from elspeth.web.composer.yaml_generator import generate_yaml
from elspeth.web.execution.accounting import load_run_accounting_for_settings
from elspeth.web.execution.schemas import RunAccounting, RunStatusResponse, ValidationResult
from elspeth.web.execution.validation import validate_pipeline
from elspeth.web.middleware.rate_limit import ComposerRateLimiter, get_rate_limiter
from elspeth.web.sessions._auto_title import maybe_auto_title_session
from elspeth.web.sessions._guided_solve_chain import solve_chain_with_auto_drop
from elspeth.web.sessions._guided_step_chat import resolve_step_1_source_chat_with_auto_drop, solve_step_chat_with_auto_drop
from elspeth.web.sessions.audit_story_models import RunAuditStoryResponse
from elspeth.web.sessions.audit_story_service import AuditStoryIntegrityError, AuditStoryService
from elspeth.web.sessions.converters import state_from_record as _state_from_record
from elspeth.web.sessions.models import composer_completion_events_table
from elspeth.web.sessions.protocol import (
    AUDIT_GRADE_VIEW_QUERY_ARG_ALLOWLIST,
    SESSION_TERMINAL_RUN_STATUS_VALUES,
    ChatMessageRecord,
    ChatMessageRole,
    ComposerSessionPreferencesRecord,
    CompositionProposalRecord,
    CompositionStateData,
    CompositionStateRecord,
    InvalidForkTargetError,
    ProposalEventRecord,
    ProposalLifecycleStatus,
    RunRecord,
    SessionRecord,
    SessionServiceProtocol,
)
from elspeth.web.sessions.schemas import (
    ChatMessageResponse,
    ChatTurnResponse,
    ComposerPreferencesResponse,
    CompositionProposalResponse,
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
    InterpretationEventResponse,
    InterpretationOptOutResponse,
    InterpretationResolveRequest,
    InterpretationResolveResponse,
    ListInterpretationEventsResponse,
    MessageWithStateResponse,
    OptOutSummaryResponse,
    ProposalEventResponse,
    RejectProposalRequest,
    RevertStateRequest,
    RunResponse,
    SendMessageRequest,
    SessionResponse,
    TerminalStateResponse,
    TurnPayloadResponse,
    TurnRecordResponse,
    UpdateComposerPreferencesRequest,
    UpdateSessionRequest,
    ValidationEntryResponse,
)
from elspeth.web.sessions.service import (
    InterpretationEventAlreadyResolvedError,
    InterpretationEventNotFoundError,
    InterpretationNodeMissingError,
    InterpretationNodePluginMutatedError,
    InterpretationPlaceholderConsumedError,
    InterpretationUnsupportedChoiceError,
)

slog = structlog.get_logger()

_REDACTED_SECRET_DETAIL = "<redacted-secret>"
_PROVIDER_DETAIL_REDACTED = "Provider detail redacted because it may contain secrets."
_MAX_PROVIDER_DETAIL_CHARS = 1_000
_INVALID_TOOL_ARGUMENTS_REDACTION_STATUS = "invalid_tool_arguments"


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
    """Return the app-scoped compose lock registry, creating it on first use.

    The registry is lazily attached to ``app.state`` on first use (see the
    class docstring for why it is not created in ``create_app()``). We probe
    presence with an explicit membership test on our own state container
    rather than a ``getattr`` default: absence is a designed first-use
    condition, and once the key is present a direct attribute read crashes if
    it is somehow the wrong type — no default papers over a contract bug.
    """
    if "session_compose_lock_registry" not in request.app.state:
        request.app.state.session_compose_lock_registry = _SessionComposeLockRegistry()
    return cast(_SessionComposeLockRegistry, request.app.state.session_compose_lock_registry)


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
        archived=session.archived_at is not None,
        forked_from_session_id=str(session.forked_from_session_id) if session.forked_from_session_id else None,
        forked_from_message_id=str(session.forked_from_message_id) if session.forked_from_message_id else None,
    )


def _composer_preferences_response(record: ComposerSessionPreferencesRecord) -> ComposerPreferencesResponse:
    return ComposerPreferencesResponse(
        session_id=str(record.session_id),
        trust_mode=record.trust_mode,
        density_default=record.density_default,
        interpretation_review_disabled=record.interpretation_review_disabled,
        updated_at=record.updated_at,
    )


def _composition_proposal_response(record: CompositionProposalRecord) -> CompositionProposalResponse:
    return CompositionProposalResponse(
        id=str(record.id),
        session_id=str(record.session_id),
        tool_call_id=record.tool_call_id,
        tool_name=record.tool_name,
        status=record.status,
        summary=record.summary,
        rationale=record.rationale,
        affects=list(record.affects),
        arguments_redacted_json=deep_thaw(record.arguments_redacted_json),
        base_state_id=str(record.base_state_id) if record.base_state_id else None,
        committed_state_id=str(record.committed_state_id) if record.committed_state_id else None,
        audit_event_id=str(record.audit_event_id) if record.audit_event_id else None,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _proposal_event_response(record: ProposalEventRecord) -> ProposalEventResponse:
    return ProposalEventResponse(
        id=str(record.id),
        session_id=str(record.session_id),
        proposal_id=str(record.proposal_id) if record.proposal_id else None,
        event_type=record.event_type,
        actor=record.actor,
        payload=deep_thaw(record.payload),
        created_at=record.created_at,
    )


async def _pending_proposal_responses(
    service: SessionServiceProtocol,
    session_id: UUID,
) -> list[CompositionProposalResponse]:
    proposals = await service.list_composition_proposals(session_id, status="pending")
    return [_composition_proposal_response(proposal) for proposal in proposals]


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

    For composer's ``_BadRequestLLMError`` carrier, the raw provider text and
    HTTP status code are read from the dedicated ``provider_detail`` /
    ``provider_status_code`` attributes — ``str(exc)`` on that class returns
    only the redacted wrap message. For LiteLLM-native exceptions the raw
    message and ``.status_code`` attribute are the right surface.
    """
    detail: dict[str, object] = {
        "error_type": error_type,
        "detail": type(exc).__name__,
    }
    if not expose_provider_error:
        return detail

    # Exact-type dispatch on our own ``_BadRequestLLMError`` carrier (raised
    # directly, never subclassed): it exposes scrubbed ``provider_detail`` /
    # ``provider_status_code`` attributes, whereas a LiteLLM-native exception
    # in the ``else`` branch is a Tier-3 SDK object we probe defensively.
    if type(exc) is _BadRequestLLMError:
        raw_provider_detail: str | None = exc.provider_detail
        raw_status_code: int | None = exc.provider_status_code
    else:
        raw_provider_detail = str(exc).strip() or None
        natural_status_code = getattr(exc, "status_code", None)
        raw_status_code = natural_status_code if type(natural_status_code) is int else None

    if raw_provider_detail:
        scrubbed = scrub_text_for_audit(raw_provider_detail).strip()
        detail["provider_detail"] = (
            _PROVIDER_DETAIL_REDACTED if scrubbed == _REDACTED_SECRET_DETAIL else scrubbed[:_MAX_PROVIDER_DETAIL_CHARS]
        )

    if raw_status_code is not None:
        detail["provider_status_code"] = raw_status_code
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
    # B4: Redact internal storage paths from blob-backed sources.
    # ``redact_source_storage_path`` guarantees the ``"sources"`` key is
    # preserved in the returned mapping. Index directly so any
    # future contract violation surfaces as ``KeyError`` rather than being
    # masked by a silent fallback — silent-failure-hunter I6 review finding,
    # 2026-05-24.
    sources_data = deep_thaw(state.sources)
    if sources_data is not None:
        redacted = redact_source_storage_path({"sources": sources_data})
        sources_data = redacted["sources"]

    return CompositionStateResponse(
        id=str(state.id),
        session_id=str(state.session_id),
        version=state.version,
        sources=sources_data,
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


def _recovery_partial_state_response(state: CompositionStateRecord) -> dict[str, Any]:
    """Serialize the persisted recovery state, including its server identity."""

    return _state_response(state).model_dump(mode="json")


def _interpretation_event_response(event: InterpretationEventRecord) -> InterpretationEventResponse:
    """Project an :class:`InterpretationEventRecord` to its wire shape.

    Field set is 1:1 with :class:`InterpretationEventResponse`. The
    response model declares ``id``/``session_id``/``composition_state_id``
    as ``UUID``, matching the record; Pydantic passes UUID values through
    untouched (no str↔UUID coercion).
    """
    return InterpretationEventResponse(
        id=event.id,
        session_id=event.session_id,
        composition_state_id=event.composition_state_id,
        affected_node_id=event.affected_node_id,
        tool_call_id=event.tool_call_id,
        user_term=event.user_term,
        kind=event.kind.value if event.kind is not None else None,
        llm_draft=event.llm_draft,
        accepted_value=event.accepted_value,
        choice=event.choice.value,
        created_at=event.created_at,
        resolved_at=event.resolved_at,
        actor=event.actor,
        interpretation_source=event.interpretation_source.value,
        model_identifier=event.model_identifier,
        model_version=event.model_version,
        provider=event.provider,
        composer_skill_hash=event.composer_skill_hash,
        arguments_hash=event.arguments_hash,
        hash_domain_version=event.hash_domain_version,
        runtime_model_identifier_at_resolve=event.runtime_model_identifier_at_resolve,
        runtime_model_version_at_resolve=event.runtime_model_version_at_resolve,
        resolved_prompt_template_hash=event.resolved_prompt_template_hash,
    )


def _extract_runtime_model_snapshot(
    state: CompositionStateRecord,
    node_id: str | None,
) -> tuple[str | None, str | None]:
    """Extract (model_identifier, model_version) for the affected LLM node (F-19).

    ``state.nodes`` is typed ``Sequence[Mapping[str, Any]] | None`` so the
    Mapping shape is guaranteed; ``id`` and ``options`` are structurally
    required keys (Tier-1 read — KeyError on absence is correct behaviour).
    ``options.model`` and ``options.model_version`` are *optional* keys by
    design — an LLM transform without an explicit pin uses an LLM-pack
    default at runtime. The audit row's columns are nullable; recording
    ``None`` accurately reflects "no runtime model pinned in composition
    state" without fabricating a default.

    A non-string value at one of the optional keys is a Tier-1 anomaly
    (the composition_state JSON came from our own writer). It is raised
    as :class:`AuditIntegrityError` rather than coerced or returned as
    NULL — a coerce would put garbage into the audit row, a NULL would
    hide the writer-side bug. ``type(value) is str`` is used rather
    than ``isinstance`` so callers (and the tier-model gate) can see
    the offensive check is exact-type, not duck-typed.
    """
    if node_id is None or state.nodes is None:
        return None, None
    for node in state.nodes:
        if node["id"] != node_id:
            continue
        options = node["options"]
        if not isinstance(options, Mapping):
            raise AuditIntegrityError(
                f"Tier 1 audit anomaly: node {node_id!r}.options is "
                f"{type(options).__name__!r}, expected mapping. The composition_state "
                f"writer guarantees an options mapping for LLM nodes."
            )
        identifier = options.get("model")
        version = options.get("model_version")
        if identifier is not None and type(identifier) is not str:
            raise AuditIntegrityError(
                f"Tier 1 audit anomaly: node {node_id!r}.options.model is "
                f"{type(identifier).__name__!r}, expected str. The composition_state "
                f"writer guarantees a str-or-absent shape for this key."
            )
        if version is not None and type(version) is not str:
            raise AuditIntegrityError(
                f"Tier 1 audit anomaly: node {node_id!r}.options.model_version is "
                f"{type(version).__name__!r}, expected str. The composition_state "
                f"writer guarantees a str-or-absent shape for this key."
            )
        return identifier, version
    return None, None


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
    after the parent-CHECK biconditional landed).
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
    # Exact-type dispatch on our own ``_RuntimePreflightOutcome`` union
    # (``ValidationResult | _RuntimePreflightFailed | None``). ``type() is``
    # rather than ``isinstance`` because these are first-party, non-subclassed
    # types and exact dispatch is the intended contract. Each member is matched
    # by a positive ``type() is`` test (which narrows for the type checker)
    # rather than relying on negative narrowing of a single branch.
    if type(runtime_preflight) is _RuntimePreflightFailed:
        if runtime_preflight.exception_class is None:
            return False, ["runtime_preflight_failed"]
        return False, _runtime_preflight_failure_errors(
            runtime_preflight.exception_class,
            runtime_preflight.exception_message_first_line,
            runtime_preflight.frames,
        )
    if type(runtime_preflight) is ValidationResult:
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


def _hash_canonical_payload(canonical_payload: str) -> str:
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def _load_canonical_mapping(canonical_payload: str | None) -> dict[str, object] | None:
    """Decode an audit canonical-JSON payload back into its mapping.

    ``canonical_payload`` is ``ComposerToolInvocation.arguments_canonical``
    / ``.result_canonical`` — RFC 8785 canonical JSON *we* authored at the
    dispatch boundary (see :mod:`elspeth.web.composer.audit`, which always
    emits ``canonical_json`` of a mapping, wrapping even unparseable LLM
    arguments in a sentinel object). It is therefore Tier-1 audit data on
    read-back: ``None`` input is the only honest absence (``result_canonical``
    is nullable for ARG_ERROR / PLUGIN_CRASH), but a *non-None* payload that
    fails to decode, or decodes to a non-mapping, is corruption of our own
    audit serialization — we crash rather than return ``None``.

    Returning ``None`` on a decode failure would be doubly wrong: it hides a
    Tier-1 anomaly, and it makes the redaction callers
    (``_redacted_{argument,result}_canonical_for_chat_message``) fall back to
    emitting the *raw, unredacted* canonical payload, defeating the
    redaction manifest for any row whose canonical text was corrupted.
    """
    if canonical_payload is None:
        return None
    try:
        decoded = json.loads(canonical_payload)
    except json.JSONDecodeError as exc:
        raise AuditIntegrityError(
            "Tier 1 audit anomaly: composer tool-invocation canonical payload "
            "is not valid JSON. arguments_canonical / result_canonical are "
            "RFC 8785 canonical JSON written by our own dispatch boundary; a "
            "decode failure is corruption of our audit serialization, not a "
            "recoverable data-quality issue."
        ) from exc
    # ``json.loads`` returns a plain ``dict`` (never a subclass) for a JSON
    # object, so the exact-type check is correct and lets the tier-model gate
    # see this is an offensive Tier-1 invariant assertion, not duck-typing.
    if type(decoded) is not dict:
        raise AuditIntegrityError(
            f"Tier 1 audit anomaly: composer tool-invocation canonical payload "
            f"decoded to {type(decoded).__name__!r}, expected a JSON object. "
            f"The dispatch boundary always emits canonical_json of a mapping; "
            f"a non-object payload is corruption of our audit serialization."
        )
    return cast(dict[str, object], decoded)


def _redacted_argument_canonical_for_chat_message(
    invocation: ComposerToolInvocation,
    *,
    telemetry: OtelRedactionTelemetry,
) -> str:
    arguments = _load_canonical_mapping(invocation.arguments_canonical)
    if invocation.tool_name not in MANIFEST or arguments is None:
        return invocation.arguments_canonical
    try:
        redacted = redact_tool_call_arguments(
            invocation.tool_name,
            arguments,
            telemetry=telemetry,
        )
    except PydanticValidationError:
        if invocation.status != ComposerToolStatus.ARG_ERROR:
            raise
        redacted = {
            "_redaction_status": _INVALID_TOOL_ARGUMENTS_REDACTION_STATUS,
            "error_class": invocation.error_class,
        }
    return canonical_json(redacted)


def _redacted_result_canonical_for_chat_message(
    invocation: ComposerToolInvocation,
    *,
    telemetry: OtelRedactionTelemetry,
) -> str | None:
    result = _load_canonical_mapping(invocation.result_canonical)
    if invocation.tool_name not in MANIFEST or result is None:
        return invocation.result_canonical
    if invocation.status == ComposerToolStatus.ARG_ERROR:
        return invocation.result_canonical
    redacted = redact_tool_call_response(
        invocation.tool_name,
        result,
        telemetry=telemetry,
    )
    return canonical_json(redacted)


def _redacted_tool_invocation_content_and_envelope(invocation: ComposerToolInvocation) -> tuple[str, dict[str, object]]:
    """Return chat-message content and tool-call envelope for the legacy drain.

    ``ComposerToolInvocation`` carries the exact arguments/results exchanged
    inside the compose loop. Some tools intentionally keep raw values there so
    in-memory convergence logic and dedicated audit buffers can reason about
    what happened. The legacy route drain writes to ``chat_messages`` instead;
    that is a persistence boundary, so it stores the redaction-manifest
    projection rather than mirroring pre-redaction canonical payloads.
    """

    telemetry = OtelRedactionTelemetry()
    arguments_canonical = _redacted_argument_canonical_for_chat_message(
        invocation,
        telemetry=telemetry,
    )
    result_canonical = _redacted_result_canonical_for_chat_message(
        invocation,
        telemetry=telemetry,
    )
    invocation_payload = invocation.to_dict()
    invocation_payload["arguments_canonical"] = arguments_canonical
    invocation_payload["arguments_hash"] = _hash_canonical_payload(arguments_canonical)
    invocation_payload["result_canonical"] = result_canonical
    invocation_payload["result_hash"] = _hash_canonical_payload(result_canonical) if result_canonical is not None else None

    if invocation.status == ComposerToolStatus.PLUGIN_CRASH:
        content = json.dumps(
            {
                "error_class": invocation.error_class,
                "error_message": invocation.error_message,
            }
        )
    elif result_canonical is not None:
        content = result_canonical
    else:
        # ARG_ERROR with no captured payload — fall back to error class.
        content = json.dumps(
            {
                "error_class": invocation.error_class,
                "error_message": invocation.error_message,
            }
        )
    return content, {"_kind": "audit", "invocation": invocation_payload}


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
    ``tool_calls`` JSON column carries an ``_kind="audit"`` envelope. Legacy
    drains pass through the redaction manifest before writing this row so
    ``chat_messages`` does not double-mirror sensitive tool arguments/results.

    ``content`` shape per status:

    - SUCCESS or ARG_ERROR with payload: the redacted projection of
      ``invocation.result_canonical`` for manifest-declared tools, or the
      original canonical payload for tools with no redaction surface.
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
        content, envelope = _redacted_tool_invocation_content_and_envelope(invocation)
        role: ChatMessageRole = "tool" if parent_assistant_id is not None else "audit"
        try:
            await service.add_message(
                session_id,
                role,
                content,
                tool_calls=[envelope],
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
    # Exact-type dispatch on our own ``_RuntimePreflightOutcome`` union:
    # ``runtime`` is a passing/failing ``ValidationResult`` only when the
    # preflight actually ran (not a captured ``_RuntimePreflightFailed`` and
    # not ``None``). ``type() is`` because ``ValidationResult`` is a
    # first-party, non-subclassed Pydantic model.
    if type(runtime) is ValidationResult:
        _record_composer_runtime_preflight_telemetry(
            "passed" if runtime.is_valid else "failed",
            source=telemetry_source,
        )
    persisted_is_valid, persisted_errors = _composer_persisted_validation(
        authoring,
        runtime,
    )
    state_d = state.to_dict()
    persisted_composer_meta = merge_implicit_decisions_meta(composer_meta, state)
    return (
        CompositionStateData(
            sources=state_d["sources"],
            nodes=state_d["nodes"],
            edges=state_d["edges"],
            outputs=state_d["outputs"],
            metadata_=state_d["metadata"],
            is_valid=persisted_is_valid,
            validation_errors=persisted_errors,
            composer_meta=persisted_composer_meta,
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
            response_body["partial_state"] = _recovery_partial_state_response(partial_record)
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
            response_body["partial_state"] = _recovery_partial_state_response(partial_record)
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
            response_body["partial_state"] = _recovery_partial_state_response(partial_record)
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


def _initial_composition_state_with_guided_session(*, profile: WorkflowProfile = EMPTY_PROFILE) -> CompositionState:
    """Construct a fresh CompositionState with a latent guided-mode session attached.

    Originally added under spec §5.2 / errata C7 ("new sessions default to
    guided"), and still pre-attaches :func:`GuidedSession.initial` so every
    server-side lazy-create branch (send_message, recompose, /guided
    endpoints) reaches a uniformly-shaped state.

    The user-visible default is now **freeform**: the frontend stopped
    auto-fetching ``GET /guided`` on session selection / creation, so the
    latent guided session is invisible until the operator clicks "Switch
    to guided" in the freeform chat header. That click hits ``GET /guided``,
    which surfaces (and persists, on first visit) the same wizard state
    this helper installs in-memory. The contract is unchanged — only the
    activation gesture moved client-side. The spec doc has not been
    re-issued; treat the title here as descriptive, not authoritative.
    """
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
        guided_session=GuidedSession.initial(profile=profile),
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


def _summarize_guided_response(
    turn_type: TurnType,
    response: TurnResponse,
) -> str | None:
    """Return a UI-safe summary for a completed guided turn.

    Summaries are denormalized display text for GuidedHistory. They may include
    user-chosen option identifiers and column names, but never raw schema-form
    options or recipe slot values because those may carry secret references or
    operator-provided sensitive text.
    """
    control_signal = response["control_signal"]
    if control_signal is ControlSignal.REJECT:
        return "Rejected proposed chain"
    if control_signal is ControlSignal.REQUEST_ADVISOR:
        return "Asked advisor to review proposal"
    if control_signal is ControlSignal.EXIT_TO_FREEFORM:
        return "Exited guided mode"
    if control_signal is ControlSignal.BACK:
        return "Went back to revise"

    if turn_type is TurnType.SINGLE_SELECT:
        chosen = response["chosen"] or []
        if chosen:
            return f"Selected: {', '.join(str(item) for item in chosen)}"
        return None

    if turn_type is TurnType.INSPECT_AND_CONFIRM:
        edited = response["edited_values"]
        if edited is None:
            return None
        if "columns" not in edited:
            return None
        columns = edited["columns"]
        if type(columns) is list:
            return f"Confirmed columns: {', '.join(str(c) for c in columns)}"
        return None

    if turn_type is TurnType.MULTI_SELECT_WITH_CUSTOM:
        chosen = tuple(str(item) for item in (response["chosen"] or ()))
        custom = tuple(str(item) for item in (response["custom_inputs"] or ()))
        fields = (*chosen, *custom)
        if fields:
            return f"Required fields: {', '.join(fields)}"
        return "No required fields selected"

    if turn_type is TurnType.SCHEMA_FORM:
        edited = response["edited_values"]
        if edited is None:
            return None
        if "plugin" not in edited:
            return None
        plugin = edited["plugin"]
        return f"Configured: {plugin}" if type(plugin) is str and plugin else None

    if turn_type is TurnType.RECIPE_OFFER:
        chosen = response["chosen"] or []
        if chosen == ["accept"]:
            edited = response["edited_values"]
            if edited is not None and "recipe_name" in edited and type(edited["recipe_name"]) is str:
                return f"Accepted recipe: {edited['recipe_name']}"
            return "Accepted recipe"
        if chosen == ["build_manually"]:
            return "Build manually"
        return None

    if turn_type is TurnType.PROPOSE_CHAIN:
        if response["edit_step_index"] is not None:
            return f"Editing proposed step {int(response['edit_step_index']) + 1}"
        chosen = response["chosen"] or []
        if chosen == ["accept"]:
            return "Accepted proposed chain"
        return None

    raise InvariantError(f"_summarize_guided_response: unhandled turn_type {turn_type!r}")


async def _inspect_latest_ready_session_blob(
    blob_service: BlobServiceProtocol,
    session_id: UUID,
) -> SourceInspectionFacts | None:
    """Inspect the newest ready blob for Step-1 schema prefill.

    Blob bytes are Tier 3 and ``inspect_blob_content`` is the source-boundary
    validation/coercion point. If the session has no ready blob, the caller
    falls back to the existing observed-schema prefill.
    """
    records = await blob_service.list_blobs(session_id, limit=None)
    for record in records:
        if record.status != "ready":
            continue
        content = await blob_service.read_blob_content(record.id)
        return inspect_blob_content(
            content=content,
            filename=record.filename,
            mime_type=record.mime_type,
            blob_id=record.id,
            content_hash=record.content_hash,
        )
    return None


def _prefilled_recipe_slot_mismatches(
    *,
    offered_slots: Mapping[str, Any],
    submitted_slots: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return server-prefilled recipe slots changed or omitted by the client."""
    mismatches: list[str] = []
    for slot_name, offered_value in offered_slots.items():
        if slot_name not in submitted_slots:
            mismatches.append(slot_name)
            continue
        if stable_hash(deep_thaw(submitted_slots[slot_name])) != stable_hash(deep_thaw(offered_value)):
            mismatches.append(slot_name)
    return tuple(sorted(mismatches))


def _reject_hidden_field_submissions(
    knobs: KnobSchema,
    submitted_options: Mapping[str, Any],
    *,
    recorder: BufferingRecorder,
    composition_version: int,
    actor: str,
    session_id: str,
    plugin_kind: str,
    plugin_name: str,
) -> None:
    """Reject submitted values for fields hidden by ``visible_when``.

    A field name may appear more than once in a discriminated KnobSchema when
    provider variants share a logical field. The submission is accepted if any
    matching field definition is currently visible; it is rejected only when
    every matching definition is hidden under the submitted discriminator state.
    """
    fields_by_name: dict[str, list[KnobField]] = {}
    for field in knobs["fields"]:
        field_name = field["name"]
        if field_name not in fields_by_name:
            fields_by_name[field_name] = []
        fields_by_name[field_name].append(field)

    for opt_name, opt_value in submitted_options.items():
        if opt_name not in fields_by_name:
            continue
        candidates = fields_by_name[opt_name]
        visible_candidates = [field for field in candidates if "visible_when" not in field]
        if visible_candidates:
            _validate_blob_ref_submission(visible_candidates, opt_name, opt_value)
            continue

        matched = False
        missing_predicate: Mapping[str, Any] | None = None
        hidden_predicate: Mapping[str, Any] | None = None
        hidden_actual_state: dict[str, Any] = {}
        for field in candidates:
            pred = field["visible_when"]
            discriminator = pred["field"]
            if discriminator not in submitted_options:
                missing_predicate = pred
                continue
            target_val = submitted_options[discriminator]
            if target_val == pred["equals"]:
                matched = True
                _validate_blob_ref_submission((field,), opt_name, opt_value)
                break
            hidden_predicate = pred
            hidden_actual_state = {discriminator: target_val}
        if matched:
            continue

        predicate = hidden_predicate if hidden_predicate is not None else missing_predicate
        if predicate is None:
            raise InvariantError(f"hidden-field rejection found no predicate for {plugin_kind}/{plugin_name}:{opt_name}")
        actual_state = hidden_actual_state
        if not actual_state:
            actual_state = {}
        emit_hidden_field_rejected(
            recorder,
            session_id=session_id,
            plugin_kind=plugin_kind,
            plugin_name=plugin_name,
            field=opt_name,
            predicate=predicate,
            actual_state=actual_state,
            composition_version=composition_version,
            actor=actor,
        )
        if not actual_state:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "hidden_field_submitted",
                    "field": opt_name,
                    "predicate": dict(predicate),
                    "message": f"Visibility discriminator {predicate['field']!r} is missing from submitted options.",
                },
            )
        raise HTTPException(
            status_code=400,
            detail={
                "code": "hidden_field_submitted",
                "field": opt_name,
                "predicate": dict(predicate),
                "actual_state": actual_state,
                "message": (
                    f"Field {opt_name!r} is hidden under current form state "
                    f"({predicate['field']}={actual_state[predicate['field']]!r}, predicate expects "
                    f"{predicate['field']}={predicate['equals']!r}). Hidden fields must not appear in edited_values.options."
                ),
            },
        )


def _validate_blob_ref_submission(fields: Sequence[KnobField], field_name: str, value: Any) -> None:
    if not any(field["kind"] == "blob-ref" for field in fields):
        return
    if value is None:
        return
    if type(value) is not str:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_blob_ref",
                "field": field_name,
                "message": f"Field {field_name!r} must be a UUID string when provided.",
            },
        )
    try:
        UUID(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_blob_ref",
                "field": field_name,
                "message": f"Field {field_name!r} must be a UUID string when provided.",
            },
        ) from exc


def _store_guided_audit_payload(payload_store: Any, payload: Mapping[str, Any]) -> str:
    """Persist a guided turn/response payload and return its content hash."""
    if payload_store is None:
        raise InvariantError("Guided audit payload store is not configured.")
    payload_ref = payload_store.store(canonical_json(payload).encode("utf-8"))
    if not isinstance(payload_ref, str) or not payload_ref:
        raise InvariantError("Guided audit payload store returned an invalid content hash.")
    return payload_ref


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
    blob_service: BlobServiceProtocol,
    payload_store: Any,
    model: str,
    temperature: float | None,
    seed: int | None,
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
            inspection_facts = guided.step_1_inspection_facts
            if inspection_facts is None:
                inspection_facts = await _inspect_latest_ready_session_blob(blob_service, UUID(session_id))
            next_turn = build_step_1_schema_form_turn(plugin_name, catalog, inspection_facts=inspection_facts)
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
                payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
                emitter="server",
                composition_version=state.version,
                actor=user_id,
            )
            guided = _replace(
                guided,
                history=(*guided.history, new_record),
                step_1_chosen_plugin=plugin_name,
                step_1_inspection_facts=inspection_facts,
            )
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
            if type(plugin_name) is not str or not plugin_name:
                raise HTTPException(
                    status_code=400,
                    detail=(f"schema_form response at step 1 edited_values['plugin'] must be a non-empty string; got {plugin_name!r}"),
                )
            options_raw = edited["options"]
            if type(options_raw) is not dict:
                raise HTTPException(
                    status_code=400,
                    detail=(f"schema_form response at step 1 edited_values['options'] must be an object; got {type(options_raw).__name__}"),
                )
            options_dict = dict(options_raw)
            schema_info = catalog.get_schema("source", plugin_name)
            _reject_hidden_field_submissions(
                cast(KnobSchema, schema_info.knob_schema),
                options_dict,
                recorder=recorder,
                composition_version=state.version,
                actor=user_id,
                session_id=session_id,
                plugin_kind="source",
                plugin_name=plugin_name,
            )
            observed_columns_raw = edited["observed_columns"]
            if type(observed_columns_raw) is not list:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"schema_form response at step 1 edited_values['observed_columns'] must be a list; got {type(observed_columns_raw).__name__}"
                    ),
                )
            sample_rows_raw = edited["sample_rows"]
            if type(sample_rows_raw) is not list:
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
                if type(_sr_elem) is not dict:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"schema_form response at step 1 edited_values['sample_rows'][{_sr_idx}] "
                            f"must be an object; got {type(_sr_elem).__name__}"
                        ),
                    )
            resolved = SourceResolved(
                plugin=plugin_name,
                options=options_dict,
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
                step_1_chosen_plugin=None,
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
                payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
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
        if type(columns_raw) is not list:
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
            payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
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
                payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
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
            if type(plugin_name) is not str or not plugin_name:
                raise HTTPException(
                    status_code=400,
                    detail=(f"schema_form response at step 2 edited_values['plugin'] must be a non-empty string; got {plugin_name!r}"),
                )
            options_raw = edited["options"]
            if type(options_raw) is not dict:
                raise HTTPException(
                    status_code=400,
                    detail=(f"schema_form response at step 2 edited_values['options'] must be an object; got {type(options_raw).__name__}"),
                )
            options_dict = dict(options_raw)
            schema_info = catalog.get_schema("sink", plugin_name)
            _reject_hidden_field_submissions(
                cast(KnobSchema, schema_info.knob_schema),
                options_dict,
                recorder=recorder,
                composition_version=state.version,
                actor=user_id,
                session_id=session_id,
                plugin_kind="sink",
                plugin_name=plugin_name,
            )
            sink_intent = SinkIntent(
                plugin=plugin_name,
                options=options_dict,
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
                payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
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
                payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
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
            temperature=temperature,
            seed=seed,
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
            payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
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
            if type(recipe_name) is not str or not recipe_name:
                raise HTTPException(
                    status_code=400,
                    detail=(f"recipe_offer ['accept'] edited_values['recipe_name'] must be a non-empty string; got {recipe_name!r}"),
                )
            slots_raw = edited["slots"]
            if type(slots_raw) is not dict:
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
            # The offered prefilled slots are server-authored and read-only in the UI.
            # Compare their stable hashes against the client-submitted subset so a
            # crafted API client cannot silently rewrite inspected facts.  Unsatisfied
            # slots remain operator-editable and are intentionally not bound here.
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
            prefilled_mismatches = _prefilled_recipe_slot_mismatches(
                offered_slots=offered.slots,
                submitted_slots=slots_raw,
            )
            if prefilled_mismatches:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "recipe_offer ['accept'] prefilled slot mismatch: "
                        f"client changed or omitted server-prefilled slot(s) {list(prefilled_mismatches)}. "
                        "Prefilled recipe slots are read-only; accept must echo the offered values."
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
                temperature=temperature,
                seed=seed,
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
                payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
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
    # The only response shape handled here is ACCEPT on a propose_chain
    # turn. Reject and clarifying SINGLE_SELECT responses remain explicit
    # gaps for future re-solve, repair, and advisor flows.
    if current_step is GuidedStep.STEP_3_TRANSFORMS:
        if current_turn_type is TurnType.PROPOSE_CHAIN:
            control = turn_response["control_signal"]
            if control in (ControlSignal.REJECT, ControlSignal.REQUEST_ADVISOR):
                if guided.step_1_result is None:
                    raise InvariantError("step 3 regenerate: step_1_result is None — STEP_3 unreachable without Step 1 commit")
                if guided.step_2_result is None:
                    raise InvariantError("step 3 regenerate: step_2_result is None — STEP_3 unreachable without Step 2 commit")
                repair_context = (
                    "The operator rejected the proposed transform chain. Generate a materially different valid proposal."
                    if control is ControlSignal.REJECT
                    else "The operator requested an advisor review. Reconsider the proposal carefully and generate the strongest valid chain."
                )
                proposal, guided = await solve_chain_with_auto_drop(
                    site=f"step_3_{control.value}_solve",
                    session=guided,
                    session_id=session_id,
                    user_id=user_id,
                    composition_version=state.version,
                    recorder=recorder,
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    source=guided.step_1_result,
                    sink=guided.step_2_result,
                    recipe_match=None,
                    repair_context=repair_context,
                )
                if proposal is None:
                    return state, guided, None
                guided = _replace(guided, step_3_proposal=proposal, step_3_edit_index=None)
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
                    payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
                    emitter="server",
                    composition_version=state.version,
                    actor=user_id,
                )
                guided = _replace(guided, history=(*guided.history, new_record))
                return state, guided, next_turn

            edit_step_index = turn_response["edit_step_index"]
            if edit_step_index is not None:
                if guided.step_3_proposal is None:
                    raise HTTPException(
                        status_code=400,
                        detail="propose_chain edit_step_index sent but no chain proposal is staged.",
                    )
                step = dict(guided.step_3_proposal.steps[edit_step_index])
                # Type-narrowing on our own staged ``ChainProposal`` steps. A
                # non-str plugin / non-mapping options here is a server-side
                # staging defect, not a client request fault, so this raises
                # InvariantError (HTTP 500) — mirroring the sibling check in
                # routes/composer.py. ``plugin`` uses exact-type (never a str
                # subclass). ``options`` MUST stay an ``isinstance(_, Mapping)``
                # check: the staged step value is frozen at construction
                # (``MappingProxyType``), so an exact ``type() is dict`` would
                # wrongly reject our own valid frozen mapping.
                plugin = step["plugin"]
                if type(plugin) is not str or not plugin:
                    raise InvariantError("step_3_proposal step plugin must be a non-empty string")
                options = step["options"]
                if not isinstance(options, Mapping):
                    raise InvariantError("step_3_proposal step options must be a mapping")
                next_turn = build_step_3_schema_form_turn(
                    plugin=plugin,
                    options=options,
                    catalog=catalog,
                )
                new_record = TurnRecord(
                    step=GuidedStep.STEP_3_TRANSFORMS,
                    turn_type=TurnType.SCHEMA_FORM,
                    payload_hash=stable_hash(next_turn["payload"]),
                    response_hash=None,
                    emitter="server",
                )
                emit_turn_emitted(
                    recorder,
                    step=GuidedStep.STEP_3_TRANSFORMS,
                    turn_type=TurnType.SCHEMA_FORM,
                    payload_hash=stable_hash(next_turn["payload"]),
                    payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
                    emitter="server",
                    composition_version=state.version,
                    actor=user_id,
                )
                guided = _replace(
                    guided,
                    history=(*guided.history, new_record),
                    step_3_edit_index=edit_step_index,
                )
                return state, guided, next_turn

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
                        temperature=temperature,
                        seed=seed,
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
                    detail=("Step 3 chain rejection is not yet implemented. Use exit-to-freeform to drop to freeform mode."),
                )
            raise HTTPException(
                status_code=400,
                detail=f"propose_chain response must have chosen=['accept'] or chosen=['reject'], got {chosen!r}.",
            )
        if current_turn_type is TurnType.SCHEMA_FORM:
            if guided.step_3_proposal is None:
                raise HTTPException(
                    status_code=400,
                    detail="schema_form response at step 3 received but no chain proposal is staged.",
                )
            edit_index = guided.step_3_edit_index
            if edit_index is None:
                raise HTTPException(
                    status_code=400,
                    detail="schema_form response at step 3 received but no edit_step_index is staged.",
                )
            edited = turn_response["edited_values"]
            if edited is None:
                raise HTTPException(
                    status_code=400,
                    detail="schema_form response at step 3 requires edited_values; received null.",
                )
            missing = {"plugin", "options"} - edited.keys()
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"schema_form response at step 3 edited_values missing required keys: {sorted(missing)}; got keys: {sorted(edited.keys())}"
                    ),
                )
            plugin_name = edited["plugin"]
            if type(plugin_name) is not str or not plugin_name:
                raise HTTPException(
                    status_code=400,
                    detail=(f"schema_form response at step 3 edited_values['plugin'] must be a non-empty string; got {plugin_name!r}"),
                )
            options_raw = edited["options"]
            if type(options_raw) is not dict:
                raise HTTPException(
                    status_code=400,
                    detail=(f"schema_form response at step 3 edited_values['options'] must be an object; got {type(options_raw).__name__}"),
                )
            schema_info = catalog.get_schema("transform", plugin_name)
            _reject_hidden_field_submissions(
                cast(KnobSchema, schema_info.knob_schema),
                options_raw,
                recorder=recorder,
                composition_version=state.version,
                actor=user_id,
                session_id=session_id,
                plugin_kind="transform",
                plugin_name=plugin_name,
            )
            existing_step = dict(guided.step_3_proposal.steps[edit_index])
            if plugin_name != existing_step["plugin"]:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"schema_form response at step 3 plugin mismatch: editing {existing_step['plugin']!r} but received {plugin_name!r}."
                    ),
                )
            updated_steps = [dict(step) for step in guided.step_3_proposal.steps]
            updated_steps[edit_index] = {
                **existing_step,
                "options": options_raw,
            }
            proposal = ChainProposal(
                steps=tuple(updated_steps),
                why=guided.step_3_proposal.why,
            )
            guided = _replace(guided, step_3_proposal=proposal, step_3_edit_index=None)
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
                payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
                emitter="server",
                composition_version=state.version,
                actor=user_id,
            )
            guided = _replace(guided, history=(*guided.history, new_record))
            return state, guided, next_turn
        # SINGLE_SELECT clarifying-question response at STEP_3.
        raise HTTPException(
            status_code=501,
            detail="Step 3 clarifying question handling is not yet implemented.",
        )

    # Unhandled branch — this is a dispatcher gap, not a user error.
    raise InvariantError(
        f"_dispatch_guided_respond: unhandled branch "
        f"current_step={current_step!r}, current_turn_type={current_turn_type!r}, "
        f"guided.step={guided.step!r}"
    )


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


__all__ = [
    "AUDIT_GRADE_VIEW_QUERY_ARG_ALLOWLIST",
    "SESSION_TERMINAL_RUN_STATUS_VALUES",
    "UTC",
    "UUID",
    "_COMPOSER_AUTHORING_VALIDATION_COUNTER",
    "_COMPOSER_EXCEPTION_CLASS_BUCKETS",
    "_COMPOSER_PERSIST_FAILED_DURING_UNWIND_COUNTER",
    "_COMPOSER_REQUESTS_INFLIGHT",
    "_COMPOSER_REQUEST_TERMINAL_COUNTER",
    "_COMPOSER_RUNTIME_PREFLIGHT_COUNTER",
    "_COMPOSER_TIER1_VIOLATION_COUNTER",
    "_DATA_ERROR_KEY",
    "_MAX_PROVIDER_DETAIL_CHARS",
    "_OTHER_COMPOSER_EXCEPTION_CLASS",
    "_PROVIDER_DETAIL_REDACTED",
    "_REDACTED_SECRET_DETAIL",
    "_RUNTIME_PREFLIGHT_FAILED",
    "_RUNTIME_PREFLIGHT_FRAME_LIMIT",
    "_RUNTIME_PREFLIGHT_MESSAGE_LIMIT",
    "APIRouter",
    "Any",
    "AuditIntegrityError",
    "AuditStoryIntegrityError",
    "AuditStoryService",
    "BlobQuotaExceededError",
    "BlobServiceProtocol",
    "BufferingRecorder",
    "CatalogServiceProtocol",
    "ChainProposal",
    "ChatMessageRecord",
    "ChatMessageResponse",
    "ChatMessageRole",
    "ChatRole",
    "ChatTurn",
    "ChatTurnResponse",
    "ComposerChatInitiator",
    "ComposerChatTurn",
    "ComposerChatTurnStatus",
    "ComposerConvergenceError",
    "ComposerLLMCall",
    "ComposerPluginCrashError",
    "ComposerPreferencesResponse",
    "ComposerProgressEvent",
    "ComposerProgressRegistry",
    "ComposerProgressSink",
    "ComposerProgressSnapshot",
    "ComposerRateLimiter",
    "ComposerRuntimePreflightError",
    "ComposerService",
    "ComposerServiceError",
    "ComposerSessionPreferencesRecord",
    "ComposerToolInvocation",
    "ComposerToolStatus",
    "CompositionProposalRecord",
    "CompositionProposalResponse",
    "CompositionState",
    "CompositionStateData",
    "CompositionStateRecord",
    "CompositionStateResponse",
    "ControlSignal",
    "CreateSessionRequest",
    "Depends",
    "FailedTurnMetadata",
    "ForkSessionRequest",
    "ForkSessionResponse",
    "GetGuidedResponse",
    "GraphValidationError",
    "GuidedChatRequest",
    "GuidedChatResponse",
    "GuidedRespondRequest",
    "GuidedRespondResponse",
    "GuidedSession",
    "GuidedSessionResponse",
    "GuidedStep",
    "HTTPException",
    "InterpretationChoice",
    "InterpretationEventAlreadyResolvedError",
    "InterpretationEventNotFoundError",
    "InterpretationEventRecord",
    "InterpretationEventResponse",
    "InterpretationNodeMissingError",
    "InterpretationNodePluginMutatedError",
    "InterpretationOptOutResponse",
    "InterpretationPlaceholderConsumedError",
    "InterpretationResolveRequest",
    "InterpretationResolveResponse",
    "InterpretationSource",
    "InterpretationUnsupportedChoiceError",
    "InvalidForkTargetError",
    "InvariantError",
    "KnobField",
    "KnobSchema",
    "LandscapeDB",
    "ListInterpretationEventsResponse",
    "Literal",
    "Mapping",
    "MessageWithStateResponse",
    "OptOutSummaryResponse",
    "PipelineMetadata",
    "PluginConfigError",
    "PluginNotFoundError",
    "ProposalEventRecord",
    "ProposalEventResponse",
    "ProposalLifecycleStatus",
    "PydanticValidationError",
    "Query",
    "RejectProposalRequest",
    "Request",
    "RevertStateRequest",
    "RunAccounting",
    "RunAuditStoryResponse",
    "RunRecord",
    "RunResponse",
    "RunStatusResponse",
    "SQLAlchemyError",
    "SendMessageRequest",
    "Sequence",
    "SessionRecord",
    "SessionResponse",
    "SessionServiceProtocol",
    "SessionsTelemetry",
    "SinkIntent",
    "SourceInspectionFacts",
    "SourceResolved",
    "TerminalKind",
    "TerminalReason",
    "TerminalState",
    "TerminalStateResponse",
    "TurnPayloadResponse",
    "TurnRecord",
    "TurnRecordResponse",
    "TurnResponse",
    "TurnType",
    "UpdateComposerPreferencesRequest",
    "UpdateSessionRequest",
    "UserIdentity",
    "ValidationEntry",
    "ValidationEntryResponse",
    "ValidationResult",
    "ValidationSummary",
    "_BadRequestLLMError",
    "_ComposerPreflightTelemetryResult",
    "_ComposerPreflightTelemetrySource",
    "_ComposerRequestEndpoint",
    "_ComposerRequestTerminalStatus",
    "_PreflightExceptionPolicy",
    "_RuntimePreflightFailed",
    "_RuntimePreflightOutcome",
    "_SessionComposeLockRegistry",
    "_bounded_composer_exception_class",
    "_capture_runtime_preflight_failure",
    "_composer_chat_history",
    "_composer_conversation_messages",
    "_composer_conversation_or_llm_audit_messages",
    "_composer_conversation_or_tool_messages",
    "_composer_conversation_tool_or_llm_audit_messages",
    "_composer_history_content",
    "_composer_persisted_validation",
    "_composer_preferences_response",
    "_composer_progress_sink",
    "_composition_proposal_response",
    "_dispatch_guided_respond",
    "_extract_runtime_model_snapshot",
    "_failed_turn_response_body",
    "_first_message_line",
    "_get_composer_progress_registry",
    "_get_session_compose_lock_registry",
    "_guided_step_index",
    "_handle_convergence_error",
    "_handle_plugin_crash",
    "_handle_runtime_preflight_failure",
    "_initial_composition_state_with_guided_session",
    "_inspect_latest_ready_session_blob",
    "_interpretation_event_response",
    "_is_composer_audit_tool_message",
    "_is_composer_llm_audit_tool_message",
    "_litellm_error_detail",
    "_llm_calls_from_exception",
    "_message_response",
    "_pending_proposal_responses",
    "_persist_chat_turns",
    "_persist_llm_calls",
    "_persist_tool_invocations",
    "_proposal_event_response",
    "_publish_progress",
    "_record_composer_authoring_validation_telemetry",
    "_record_composer_request_terminal",
    "_record_composer_runtime_preflight_telemetry",
    "_reject_hidden_field_submissions",
    "_replace",
    "_run_accounting_integrity_http",
    "_runtime_preflight_failure_errors",
    "_runtime_preflight_for_state",
    "_safe_frame_strings",
    "_session_response",
    "_state_data_from_composer_state",
    "_state_from_record",
    "_state_response",
    "_store_guided_audit_payload",
    "_summarize_guided_response",
    "_validate_blob_ref_submission",
    "_validate_control_signal",
    "_validate_run_status_accounting_for_list",
    "_validate_step_indices",
    "_verify_session_ownership",
    "annotations",
    "asyncio",
    "audit_envelope",
    "build_initial_step_1_turn",
    "build_step_1_inspect_and_confirm_turn_from_intent",
    "build_step_1_schema_form_turn",
    "build_step_2_5_recipe_offer_turn",
    "build_step_2_multi_select_turn",
    "build_step_2_schema_form_turn",
    "build_step_2_single_select_turn",
    "build_step_3_propose_chain_turn",
    "build_step_3_schema_form_turn",
    "cast",
    "chat_turn_audit_envelope",
    "client_cancelled_progress_event",
    "composer_completion_events_table",
    "contextlib",
    "convergence_progress_event",
    "dataclass",
    "datetime",
    "deep_thaw",
    "emit_dropped_to_freeform",
    "emit_hidden_field_rejected",
    "emit_step_advanced",
    "emit_turn_answered",
    "emit_turn_emitted",
    "execute_tool",
    "generate_yaml",
    "get_current_user",
    "get_rate_limiter",
    "handle_step_1_source",
    "handle_step_2_5_recipe_apply",
    "handle_step_2_sink",
    "handle_step_3_chain_accept",
    "insert",
    "inspect_blob_content",
    "json",
    "llm_call_audit_envelope",
    "load_run_accounting_for_settings",
    "mark_solver_exhausted",
    "match_recipe",
    "maybe_auto_title_session",
    "maybe_resolve_step_1_source_chat",
    "merge_implicit_decisions_meta",
    "metrics",
    "record_session_completed",
    "record_session_switched",
    "redact_source_storage_path",
    "resolve_step_1_source_chat_with_auto_drop",
    "run_sync_in_worker",
    "scrub_text_for_audit",
    "slog",
    "solve_chain_with_auto_drop",
    "solve_step_chat_with_auto_drop",
    "stable_hash",
    "step_advance",
    "structlog",
    "sys",
    "uuid4",
    "validate_pipeline",
    "yaml_generator",
]
