"""Session API routes -- /api/sessions/* with IDOR protection.

All endpoints require authentication via Depends(get_current_user).
Session-scoped endpoints verify ownership before any business logic.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace as _replace
from datetime import UTC, datetime
from typing import Any, Final, Literal, cast
from uuid import UUID, uuid4
from weakref import WeakValueDictionary

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
from elspeth.core.canonical import stable_hash
from elspeth.core.dag.models import GraphValidationError
from elspeth.core.landscape.database import LandscapeDB
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.infrastructure.manager import PluginNotFoundError
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.protocol import BlobQuotaExceededError, BlobServiceProtocol
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.protocol import CatalogService as CatalogServiceProtocol
from elspeth.web.composer import yaml_generator
from elspeth.web.composer.audit import (
    BufferingRecorder,
    audit_envelope,
    chat_turn_audit_envelope,
    llm_call_audit_envelope,
)
from elspeth.web.composer.audit_storage import redacted_tool_invocation_content_and_envelope
from elspeth.web.composer.guided.audit import (
    emit_dropped_to_freeform,
    emit_step_advanced,
    emit_turn_answered,
    emit_turn_emitted,
)
from elspeth.web.composer.guided.chat_solver import maybe_resolve_step_1_source_chat
from elspeth.web.composer.guided.emitters import (
    _inspection_matches_source_plugin,
    build_initial_step_1_turn,
    build_step_1_inspect_and_confirm_turn_from_intent,
    build_step_1_schema_form_turn,
    build_step_1_schema_form_turn_from_resolved,
    build_step_1_source_prefill,
    build_step_2_multi_select_turn,
    build_step_2_schema_form_turn,
    build_step_2_schema_form_turn_from_resolved,
    build_step_2_single_select_turn,
    build_step_4_wire_turn,
)
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.profile import EMPTY_PROFILE, WorkflowProfile
from elspeth.web.composer.guided.protocol import ChatRole, ChatTurn, ControlSignal, GuidedStep, TurnType
from elspeth.web.composer.guided.state_machine import (
    GuidedSession,
    SinkIntent,
    SourceResolved,
    TerminalKind,
    TerminalReason,
    TerminalState,
    TurnRecord,
)
from elspeth.web.composer.implicit_decisions import merge_implicit_decisions_meta
from elspeth.web.composer.pipeline_commit import PipelineDispatchAuditBinding
from elspeth.web.composer.pipeline_planner import PipelinePlannerError
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
from elspeth.web.composer.redaction import redact_guided_snapshot_storage_paths, redact_source_storage_path
from elspeth.web.composer.service import _BadRequestLLMError
from elspeth.web.composer.source_inspection import SourceInspectionFacts, inspect_blob_content
from elspeth.web.composer.state import CompositionState, PipelineMetadata, ValidationEntry, ValidationSummary
from elspeth.web.composer.telemetry_phase8 import (
    SessionsTelemetry,
    record_session_completed,
    record_session_switched,
)
from elspeth.web.composer.tools import _DATA_ERROR_KEY, ToolResult, execute_tool
from elspeth.web.composer.yaml_generator import generate_public_yaml
from elspeth.web.execution.accounting import load_run_accounting_for_settings
from elspeth.web.execution.schemas import RunAccounting, RunStatusResponse, ValidationResult
from elspeth.web.execution.validation import validate_pipeline
from elspeth.web.middleware.rate_limit import ComposerRateLimiter, get_rate_limiter
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry
from elspeth.web.plugin_policy.validation import validate_authored_composition_state
from elspeth.web.sessions._auto_title import maybe_auto_title_session
from elspeth.web.sessions._guided_step_chat import (
    _COMMIT_REJECTED_MESSAGE,
    _SYNTHETIC_UNAVAILABLE_MESSAGE,
    Step2SinkChatResult,
    StepChatResult,
    resolve_step_1_source_chat_with_auto_drop,
    resolve_step_2_sink_chat_with_auto_drop,
    solve_step_chat_with_auto_drop,
)
from elspeth.web.sessions.audit_story_models import RunAuditStoryResponse
from elspeth.web.sessions.audit_story_service import AuditStoryIntegrityError, AuditStoryService
from elspeth.web.sessions.converters import state_from_record as _state_from_record
from elspeth.web.sessions.guided_replay import project_composition_proposal, validation_errors_for_composer_surface
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
    AcceptProposalRequest,
    ChatMessageResponse,
    ChatTurnResponse,
    ComposerPreferencesResponse,
    CompositionObject,
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
    PluginPolicyFindingResponse,
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
    WorkflowProfileResponse,
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
_GUIDED_SOURCE_PATH_ALLOWLIST_DETAIL = (
    "Source path is outside the allowed upload area. "
    "Upload the file through the composer or use a path under the configured blobs directory."
)


def _guided_source_commit_failure_detail(tool_result: object) -> str:
    if type(tool_result) is not ToolResult:
        raise TypeError(f"guided source commit failure detail requires ToolResult, got {type(tool_result).__name__}")
    raw_data = tool_result.data
    if isinstance(raw_data, Mapping):
        error = raw_data.get(_DATA_ERROR_KEY)
        if isinstance(error, str) and error.startswith("Path violation (S2):") and "Source file paths" in error:
            return _GUIDED_SOURCE_PATH_ALLOWLIST_DETAIL
    return "Step 1 source commit failed"


_MAX_PROVIDER_DETAIL_CHARS = 1_000


class _SessionComposeLockRegistry:
    """Per-session compose/recompose locks.

    Lazily creates asyncio.Lock instances under a running event loop so the
    registry can live on app.state without needing sync-time initialization in
    create_app().
    """

    def __init__(self) -> None:
        # Weak custody is deliberate: every borrower holds the lock strongly
        # from lookup through acquisition/release, so concurrent borrowers
        # share one identity. Once no holder, waiter, or pre-acquire borrower
        # remains, the entry is reclaimed without deletion ever evicting a
        # still-live lock and splitting a session across two lock objects.
        self._session_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._locks_lock: asyncio.Lock | None = None

    def _ensure_locks_lock(self) -> asyncio.Lock:
        if self._locks_lock is None:
            self._locks_lock = asyncio.Lock()
        return self._locks_lock

    async def get_lock(self, session_id: str) -> asyncio.Lock:
        async with self._ensure_locks_lock():
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_id] = lock
            return lock

    async def cleanup_session_lock(self, session_id: str) -> None:
        """Allow weak reclamation without evicting a lock borrowed by a request."""
        async with self._ensure_locks_lock():
            # Iteration commits pending WeakValueDictionary removals while the
            # registry mutex excludes a simultaneous creator.
            tuple(self._session_locks)


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


def _request_plugin_policy_context(
    request: Request,
    user: UserIdentity,
) -> tuple[PolicyCatalogView, PluginAvailabilitySnapshot]:
    """Build the one immutable plugin-policy context for this HTTP request."""
    catalog: CatalogServiceProtocol = request.app.state.catalog_service
    snapshot: PluginAvailabilitySnapshot = request.app.state.plugin_snapshot_factory(user)
    return PolicyCatalogView(catalog, snapshot, request.app.state.operator_profile_registry), snapshot


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
    return project_composition_proposal(record)


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
        # "detail" (not "message"): parseResponse (frontend/src/api/client.ts)
        # reads nestedDetail.detail as the human-readable string, with no
        # "message" fallback, for ANY non-2xx response (not just 400s).
        "detail": message,
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
        try:
            exc_attrs = vars(exc)
        except TypeError:
            exc_attrs = {}
        natural_status_code = exc_attrs.get("status_code")
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
    *,
    policy_catalog: PolicyCatalogView | None = None,
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

    # B4 (guided): a guided blob-backed source is committed via the manual
    # set_source path, which strips ``blob_ref`` (it cannot prove
    # ``path == storage_path``). So the committed source AND the persisted
    # GuidedSession snapshot in ``composer_meta`` both carry the absolute
    # storage_path with no ``blob_ref`` for the source-keyed redaction above to
    # key off, and the snapshot is serialised here unredacted. Cross-reference
    # the snapshot's RETAINED ``blob_ref`` (a no-DB-lookup signal that the source
    # is blob-backed) to mask the storage_path in both the snapshot and the
    # committed source before either reaches the wire.
    composer_meta_data = deep_thaw(state.composer_meta) if state.composer_meta is not None else None
    sources_data, composer_meta_data = redact_guided_snapshot_storage_paths(sources_data, composer_meta_data)

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
            ValidationEntryResponse(component=e.component, message=e.message, severity=e.severity, error_code=e.error_code)
            for e in live_validation.warnings
        ]
        if live_validation is not None
        else None,
        validation_suggestions=[
            ValidationEntryResponse(component=e.component, message=e.message, severity=e.severity, error_code=e.error_code)
            for e in live_validation.suggestions
        ]
        if live_validation is not None
        else None,
        derived_from_state_id=str(state.derived_from_state_id) if state.derived_from_state_id is not None else None,
        created_at=state.created_at,
        composer_meta=composer_meta_data,
        plugin_policy_findings=_plugin_policy_findings(state, policy_catalog),
    )


def _plugin_policy_findings(
    state: CompositionStateRecord,
    policy_catalog: PolicyCatalogView | None,
) -> list[PluginPolicyFindingResponse]:
    """Describe persisted components unavailable in the current snapshot."""
    if policy_catalog is None:
        return []
    components: list[tuple[str, PluginId]] = []
    for source_name, source in (state.sources or {}).items():
        plugin_name = source.get("plugin")
        if isinstance(plugin_name, str):
            components.append((source_name, PluginId("source", plugin_name)))
    for node in state.nodes or ():
        plugin_name = node.get("plugin")
        component_id = node.get("id")
        if isinstance(plugin_name, str) and isinstance(component_id, str):
            components.append((component_id, PluginId("transform", plugin_name)))
    for output in state.outputs or ():
        plugin_name = output.get("plugin")
        component_id = output.get("name", output.get("sink_name"))
        if isinstance(plugin_name, str) and isinstance(component_id, str):
            components.append((component_id, PluginId("sink", plugin_name)))
    findings: list[PluginPolicyFindingResponse] = []
    for component_id, plugin_id in components:
        reason = policy_catalog.unavailable_reason(plugin_id)
        if reason is not None:
            findings.append(
                PluginPolicyFindingResponse(
                    component_id=component_id,
                    plugin_id=str(plugin_id),
                    reason_code=reason.value,
                    snapshot_fingerprint=policy_catalog.snapshot.snapshot_hash,
                )
            )
    return findings


def merge_composer_meta_updates(
    existing_meta: Mapping[str, Any] | None,
    updates: Mapping[str, Any],
) -> CompositionObject:
    """Merge route-owned updates without dropping opaque lifecycle metadata.

    ``composer_meta`` is a shared persistence envelope.  Version-changing
    freeform writes must carry forward keys owned by guided mode and other
    subsystems rather than rebuilding the envelope from only the keys they
    understand.
    """
    merged = cast(CompositionObject, dict(deep_thaw(existing_meta))) if existing_meta is not None else {}
    merged.update(updates)
    return merged


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
    "state_seed",
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
    session_id: str | UUID,
    plugin_snapshot: PluginAvailabilitySnapshot,
    profile_registry: OperatorProfileRegistry,
    catalog: CatalogServiceProtocol,
) -> ValidationResult:
    return await asyncio.wait_for(
        run_sync_in_worker(
            validate_pipeline,
            state,
            settings,
            yaml_generator,
            secret_service=secret_service,
            user_id=user_id,
            session_id=str(session_id),
            plugin_snapshot=plugin_snapshot,
            profile_registry=profile_registry,
            catalog=catalog,
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
) -> tuple[PipelineDispatchAuditBinding, ...]:
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
    persisted_pipeline_bindings: list[PipelineDispatchAuditBinding] = []
    for invocation in tool_invocations:
        content, envelope = redacted_tool_invocation_content_and_envelope(invocation)
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
            if invocation.tool_name == "set_pipeline" and invocation.status is ComposerToolStatus.SUCCESS:
                persisted_pipeline_bindings.append(PipelineDispatchAuditBinding.from_persisted_envelope(envelope))
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
    return tuple(persisted_pipeline_bindings)


def _llm_calls_from_exception(exc: BaseException) -> tuple[ComposerLLMCall, ...]:
    exc_dict = exc.__dict__
    if "llm_calls_durable" in exc_dict and exc_dict["llm_calls_durable"] is True:
        return ()
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


_CLIENT_DISCONNECT_CANCEL_MARKER = "elspeth_client_disconnected"


def _is_client_disconnect_cancel(exc: asyncio.CancelledError) -> bool:
    """True when ``exc`` was delivered by :func:`_cancel_on_client_disconnect`."""
    return bool(getattr(exc, _CLIENT_DISCONNECT_CANCEL_MARKER, False))


@contextlib.asynccontextmanager
async def _cancel_on_client_disconnect(request: Request) -> AsyncIterator[None]:
    """Cancel the enclosing route task when the HTTP client disconnects.

    The server stack does not do this on its own: uvicorn's
    ``connection_lost`` only flags the request cycle as disconnected (it
    never cancels the ASGI task), and Starlette's ``request_response``
    has no disconnect watcher, so a client abort (Stop button, SPA
    compose timeout, closed tab) leaves the route running to completion
    as a zombie — burning the LLM budget, holding the per-session
    compose lock for minutes, and mutating composition state the client
    will never see (elspeth-e08063c3a5). The composer stack is already
    built for cancellation (``attach_llm_calls`` rides on the
    CancelledError instance; the routes have cancelled-path
    bookkeeping); this watcher supplies the missing trigger.

    Semantics:

    * The guarded block MUST be awaited inline in the route task —
      running it in a child task would launder the CancelledError
      instance at the task boundary and drop the attached llm_calls
      audit records.
    * On disconnect, the route task is cancelled; the CancelledError is
      marked (see :func:`_is_client_disconnect_cancel`) and the task's
      cancellation count is restored via ``uncancel()`` so the route can
      convert it into a quiet HTTP response after its cancelled-path
      bookkeeping (uvicorn discards writes on a disconnected connection,
      but a CancelledError escaping the app is logged as "Exception in
      ASGI application").
    * If the disconnect races the guarded block's completion, the
      pending cancellation is flushed and absorbed here so it cannot
      detonate mid-way through the route's post-compose persist tail.
    * Both test transports (Starlette TestClient, httpx ASGITransport)
      block their ``receive()`` until the response completes, so the
      watcher stays dormant under tests unless a disconnect is
      explicitly simulated.
    """
    task = asyncio.current_task()
    if task is None:  # pure-sync dispatch (unit-test seams); nothing to watch
        yield
        return
    triggered = False

    async def _watch_disconnect() -> None:
        nonlocal triggered
        while True:
            try:
                message = await request.receive()
            except Exception:
                # A broken receive channel means we cannot observe the
                # client any more — stop watching rather than risk
                # cancelling a healthy compose on a transport quirk.
                return
            if message["type"] == "http.disconnect":
                triggered = True
                task.cancel()
                return

    watcher = asyncio.create_task(_watch_disconnect())
    try:
        yield
    except asyncio.CancelledError as exc:
        # Consume ONLY the watcher's own cancellation request.
        # ``triggered`` proves a disconnect happened, not that the
        # delivered CancelledError belongs to the watcher alone: an
        # external cancel (server shutdown, operator) can race the
        # disconnect, leaving two requests on the task. Mark the
        # exception as disconnect-initiated — licensing the route to
        # convert it into a quiet 499 — only when no other request
        # remains after ours is uncancelled (short-circuit keeps the
        # uncancel from running for a purely external cancel); otherwise
        # leave it unmarked so the route's cancelled-path re-raises and
        # the task keeps unwinding as genuinely cancelled (the mirror of
        # the else-branch's ``cancelling()`` re-check below).
        if triggered and task.uncancel() == 0:
            setattr(exc, _CLIENT_DISCONNECT_CANCEL_MARKER, True)
        raise
    else:
        # Normal exit: resolve completion races BEFORE the route resumes
        # its persist tail, so a disconnect-cancel that landed in the
        # same tick the guarded block completed cannot detonate mid-way
        # through the post-compose persists.
        if not watcher.done():
            watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            # A pending EXTERNAL cancel can also deliver at this await;
            # it is re-raised below via the cancelling() re-check rather
            # than silently swallowed by the suppress.
            await watcher
        if task.cancelling() > 0:
            if triggered:
                # task.cancel() was called but the CancelledError has
                # not been delivered yet (awaiting a done future does
                # not yield to the loop). Flush it at a controlled
                # suspension point and absorb it — the compose finished,
                # its results persist normally, and the response is
                # simply discarded by the disconnected transport.
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(0)
                task.uncancel()
            if task.cancelling() > 0:
                # An external cancel (e.g. server shutdown) raced the
                # guarded block's completion — keep unwinding exactly as
                # if it had landed inside the block.
                raise asyncio.CancelledError()
    finally:
        # Exception paths (compose errors, cancellation) reach here
        # without the else-branch teardown: detach the watcher without
        # awaiting it — awaiting would add a suspension point on the
        # unwind path. A cancelled task is collected by the loop without
        # "exception was never retrieved" noise.
        if not watcher.done():
            watcher.cancel()


async def _track_compose_inflight(
    session_id: UUID,
    request: Request,
) -> AsyncIterator[None]:
    """Count this request in the session's in-flight compose tally.

    FastAPI yield-dependency wired into ``send_message`` and ``/recompose``.
    The count spans the ENTIRE request — including the wait on the
    per-session compose lock, before any progress snapshot is published —
    and is decremented only when the request's exit stack closes, i.e.
    after the route has fully unwound (success, HTTP error, or the
    disconnect-cancel path).

    The SPA's post-abort reconciliation treats a zero count on the
    ``/composer-progress`` snapshot as its ONLY settlement signal
    (elspeth-06a23adfcc): phase-based inference races requests that have
    not yet published progress (queued on the lock, immediate Stop), where
    the registry still holds the previous turn's terminal snapshot.

    Keyed by the raw path ``session_id`` (pre-ownership-check): a request
    rejected by the ownership guard still transits the counter briefly,
    which is harmless — the counter only ever delays a resync while
    non-zero, and rejected requests decrement within the same request
    lifecycle.
    """
    registry = _get_composer_progress_registry(request)
    sid = str(session_id)
    registry.begin_request(sid)
    try:
        yield
    finally:
        registry.end_request(sid)


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
    session_id: str | UUID,
    plugin_snapshot: PluginAvailabilitySnapshot,
    profile_registry: OperatorProfileRegistry,
    catalog: CatalogServiceProtocol,
    runtime_preflight: _RuntimePreflightOutcome,
    preflight_exception_policy: _PreflightExceptionPolicy,
    initial_version: int | None,
    telemetry_source: _ComposerPreflightTelemetrySource,
    composer_meta: Mapping[str, Any] | None = None,
) -> tuple[CompositionStateData, ValidationSummary]:
    try:
        authoring = validate_authored_composition_state(
            state,
            snapshot=plugin_snapshot,
            profile_registry=profile_registry,
            catalog=catalog,
        ).validation
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
                session_id=session_id,
                plugin_snapshot=plugin_snapshot,
                profile_registry=profile_registry,
                catalog=catalog,
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
    surface_meta = dict(deep_thaw(composer_meta)) if composer_meta is not None else {}
    if state.guided_session is not None and "guided_session" not in surface_meta:
        surface_meta["guided_session"] = state.guided_session.to_dict()
    persisted_composer_meta = merge_implicit_decisions_meta(surface_meta, state)
    normalized_persisted_errors = validation_errors_for_composer_surface(
        composer_meta=persisted_composer_meta,
        is_valid=persisted_is_valid,
        validation_errors=persisted_errors,
    )
    return (
        CompositionStateData(
            sources=state_d["sources"],
            nodes=state_d["nodes"],
            edges=state_d["edges"],
            outputs=state_d["outputs"],
            metadata_=state_d["metadata"],
            is_valid=persisted_is_valid,
            validation_errors=normalized_persisted_errors,
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
    if session.archived_at is not None or session.user_id != user.user_id or session.auth_provider_type != settings.auth_provider:
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


# Freeform planner-failure taxonomy. Kept in lockstep with the guided path's
# ``PipelinePlannerError`` sub-mapping in
# ``routes/composer/guided_plan.py::_guided_full_failure_code`` and the
# ``_SAFE_FAILURES`` status table in ``routes/guided_operations.py`` so a given
# ``PipelinePlannerError.code`` yields the same closed failure code and HTTP
# status on both surfaces. The freeform surface has no ``guided_operations``
# lease to terminalize, so ``_handle_planner_failure`` writes an equivalent
# durable disposition audit row instead of calling
# ``fail_guided_operation_with_audit``.
# Byte-identical to the guided set — do NOT add codes here without adding them
# to guided's ``_guided_full_failure_code`` in the same change, or the two
# surfaces return different closed codes (and HTTP statuses) for the same
# ``PipelinePlannerError.code``, which is exactly the divergence Task 0 exists to
# prevent. ``COST_CAP_EXCEEDED`` and ``REQUEST_BYTES_EXHAUSTED`` are deliberately
# absent (they fall through to ``operation_failed`` on both surfaces), matching
# guided.
_FREEFORM_PLANNER_INVALID_PROVIDER_CODES: Final[frozenset[str]] = frozenset(
    {
        "COMPLETION_TOKENS_EXCEEDED",
        "COMPOSITION_EXHAUSTED",
        "COST_UNAVAILABLE",
        "DISCOVERY_CYCLE",
        "DISCOVERY_EXHAUSTED",
        "DISCOVERY_ONLY",
        "MALFORMED_RESPONSE",
        "PROVIDER_CALLS_EXHAUSTED",
        "REPAIR_EXHAUSTED",
        "RESPONSE_TRUNCATED",
        "TOOL_CALLS_EXHAUSTED",
        "VALIDATION_FAILED",
    }
)
# ``failure_code -> (http_status, safe static detail)``. Mirrors the subset of
# ``_SAFE_FAILURES`` the freeform planner can reach; the detail text is
# provider-safe (no exception message, no provider content).
_FREEFORM_PLANNER_FAILURE_HTTP: Final[dict[str, tuple[int, str]]] = {
    "provider_timeout": (504, "The composer model timed out before producing a pipeline. Retry the request."),
    "provider_unavailable": (503, "The composer model is unavailable. Retry the request."),
    "invalid_provider_response": (502, "The composer model returned an unusable pipeline plan. Retry the request."),
    "operation_failed": (500, "The composer could not build a pipeline for this request."),
}


def _freeform_planner_failure_code(exc: PipelinePlannerError) -> str:
    """Map a ``PipelinePlannerError.code`` to a closed freeform failure code.

    Byte-parity with the ``isinstance(exc, PipelinePlannerError)`` branch of the
    guided ``_guided_full_failure_code``; kept as a separate function so the
    guided path stays untouched.
    """
    if exc.code == "TIMEOUT":
        return "provider_timeout"
    if exc.code == "PROVIDER_ERROR":
        return "provider_unavailable"
    if exc.code in _FREEFORM_PLANNER_INVALID_PROVIDER_CODES:
        return "invalid_provider_response"
    return "operation_failed"


async def _handle_planner_failure(
    exc: PipelinePlannerError,
    service: SessionServiceProtocol,
    session_id: UUID,
    llm_composition_state_id: UUID | None,
) -> tuple[int, dict[str, object]]:
    """Translate a freeform ``PipelinePlannerError`` into a safe HTTP outcome.

    Mirrors the guided path's ``fail_guided_operation_with_audit``
    terminalization on the freeform surface, which has no guided-operation lease
    to close: persists one durable, redacted terminal failure-disposition audit
    row carrying the mapped closed failure code, then returns the
    ``(status, body)`` the route raises. Shared by ``send_message`` and
    ``recompose`` so the two freeform routes cannot drift on planner-failure UX.

    The planner's LLM-call audit evidence is already durable — ``plan_pipeline``
    stamps it onto the escaping exception via ``attach_llm_calls`` and
    ``_plan_and_stage_empty_pipeline`` persists it before re-raise (which also
    sets ``llm_calls_durable``) — so this MUST NOT re-persist it. The
    disposition row is a distinct ``_kind`` and carries only the closed code, no
    provider content, usage, or model metadata.
    """
    failure_code = _freeform_planner_failure_code(exc)
    status_code, detail = _FREEFORM_PLANNER_FAILURE_HTTP[failure_code]
    envelope: dict[str, object] = {
        "_kind": "planner_failure_disposition",
        "surface": "freeform",
        "failure_code": failure_code,
        # The raw PipelinePlannerError code: failure_code buckets many codes
        # (e.g. DISCOVERY_CYCLE and MALFORMED_RESPONSE both read
        # invalid_provider_response) and the code is the closed, leak-safe
        # discriminant a live 5xx investigation actually needs.
        "planner_code": exc.code,
    }
    # The closed validation codes of the last candidate rejection when the
    # failure is an exhaustion — names the wall a live 5xx hit without a temp
    # diagnostic (empty for non-rejection failures).
    if exc.detail_codes:
        envelope["rejection_codes"] = sorted(set(exc.detail_codes))
    # role="audit" keeps this out of the user-visible conversation channel
    # (``_is_composer_audit_tool_message`` excludes every audit row) exactly as
    # the per-LLM-call audit sidecars are; the distinct ``_kind`` keeps it out
    # of the ``llm_call_audit`` opt-in view as well.
    await service.add_message(
        session_id,
        "audit",
        json.dumps(envelope),
        tool_calls=[envelope],
        composition_state_id=llm_composition_state_id,
        writer_principal="compose_loop",
    )
    return status_code, {
        "error_type": "composer_planner_failure",
        "failure_code": failure_code,
        "detail": detail,
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
    plugin_snapshot: PluginAvailabilitySnapshot,
    profile_registry: OperatorProfileRegistry,
    catalog: CatalogServiceProtocol,
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
                session_id=session_id,
                plugin_snapshot=plugin_snapshot,
                profile_registry=profile_registry,
                catalog=catalog,
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
    plugin_snapshot: PluginAvailabilitySnapshot,
    profile_registry: OperatorProfileRegistry,
    catalog: CatalogServiceProtocol,
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
                session_id=session_id,
                plugin_snapshot=plugin_snapshot,
                profile_registry=profile_registry,
                catalog=catalog,
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
    plugin_snapshot: PluginAvailabilitySnapshot,
    profile_registry: OperatorProfileRegistry,
    catalog: CatalogServiceProtocol,
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
                session_id=session_id,
                plugin_snapshot=plugin_snapshot,
                profile_registry=profile_registry,
                catalog=catalog,
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


def _workflow_profile_response(guided: GuidedSession) -> WorkflowProfileResponse | None:
    """Project a GuidedSession's server-owned profile onto the wire subset.

    Returns ``None`` for the empty/live-guided profile (== ``EMPTY_PROFILE``).
    """
    if guided.profile == EMPTY_PROFILE:
        return None
    return WorkflowProfileResponse(
        coaching=guided.profile.coaching,
        bookends=guided.profile.bookends,
    )


async def _inspect_latest_ready_session_blob(
    blob_service: BlobServiceProtocol,
    session_id: UUID,
    *,
    filename: str | None = None,
    source_plugin: str | None = None,
) -> SourceInspectionFacts | None:
    """Inspect the newest matching ready blob for Step-1 schema prefill.

    Blob bytes are Tier 3 and ``inspect_blob_content`` is the source-boundary
    validation/coercion point. If the session has no ready blob, the caller
    falls back to the existing observed-schema prefill. When ``filename`` is
    provided, only ready blobs whose stored filename exactly matches it are
    eligible. When ``source_plugin`` is provided, inspection continues past
    newer ready blobs of other source kinds and returns the newest ready blob
    whose inspected content safely prefills that plugin.
    """
    records = await blob_service.list_blobs(session_id, limit=None)
    for record in records:
        if record.status != "ready":
            continue
        if filename is not None and record.filename != filename:
            continue
        content = await blob_service.read_blob_content(record.id)
        facts = inspect_blob_content(
            content=content,
            filename=record.filename,
            mime_type=record.mime_type,
            blob_id=record.id,
            content_hash=record.content_hash,
        )
        if source_plugin is not None and not _inspection_matches_source_plugin(source_plugin, facts):
            continue
        return facts
    return None


__all__ = [
    "AUDIT_GRADE_VIEW_QUERY_ARG_ALLOWLIST",
    "SESSION_TERMINAL_RUN_STATUS_VALUES",
    "UTC",
    "UUID",
    "_COMMIT_REJECTED_MESSAGE",
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
    "_SYNTHETIC_UNAVAILABLE_MESSAGE",
    "APIRouter",
    "AcceptProposalRequest",
    "Any",
    "AuditIntegrityError",
    "AuditStoryIntegrityError",
    "AuditStoryService",
    "BlobQuotaExceededError",
    "BlobServiceProtocol",
    "BufferingRecorder",
    "CatalogServiceProtocol",
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
    "LandscapeDB",
    "ListInterpretationEventsResponse",
    "Literal",
    "Mapping",
    "MessageWithStateResponse",
    "OptOutSummaryResponse",
    "PipelineMetadata",
    "PipelinePlannerError",
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
    "Step2SinkChatResult",
    "StepChatResult",
    "TerminalKind",
    "TerminalReason",
    "TerminalState",
    "TerminalStateResponse",
    "TurnPayloadResponse",
    "TurnRecord",
    "TurnRecordResponse",
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
    "_cancel_on_client_disconnect",
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
    "_extract_runtime_model_snapshot",
    "_failed_turn_response_body",
    "_first_message_line",
    "_get_composer_progress_registry",
    "_get_session_compose_lock_registry",
    "_handle_convergence_error",
    "_handle_planner_failure",
    "_handle_plugin_crash",
    "_handle_runtime_preflight_failure",
    "_initial_composition_state_with_guided_session",
    "_inspect_latest_ready_session_blob",
    "_interpretation_event_response",
    "_is_client_disconnect_cancel",
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
    "_replace",
    "_run_accounting_integrity_http",
    "_runtime_preflight_failure_errors",
    "_runtime_preflight_for_state",
    "_safe_frame_strings",
    "_session_response",
    "_state_data_from_composer_state",
    "_state_from_record",
    "_state_response",
    "_track_compose_inflight",
    "_validate_run_status_accounting_for_list",
    "_verify_session_ownership",
    "_workflow_profile_response",
    "annotations",
    "asyncio",
    "audit_envelope",
    "build_initial_step_1_turn",
    "build_step_1_inspect_and_confirm_turn_from_intent",
    "build_step_1_schema_form_turn",
    "build_step_1_schema_form_turn_from_resolved",
    "build_step_1_source_prefill",
    "build_step_2_multi_select_turn",
    "build_step_2_schema_form_turn",
    "build_step_2_schema_form_turn_from_resolved",
    "build_step_2_single_select_turn",
    "build_step_4_wire_turn",
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
    "emit_step_advanced",
    "emit_turn_answered",
    "emit_turn_emitted",
    "execute_tool",
    "generate_public_yaml",
    "get_current_user",
    "get_rate_limiter",
    "insert",
    "inspect_blob_content",
    "json",
    "llm_call_audit_envelope",
    "load_run_accounting_for_settings",
    "maybe_auto_title_session",
    "maybe_resolve_step_1_source_chat",
    "merge_composer_meta_updates",
    "merge_implicit_decisions_meta",
    "metrics",
    "record_session_completed",
    "record_session_switched",
    "redact_source_storage_path",
    "resolve_step_1_source_chat_with_auto_drop",
    "resolve_step_2_sink_chat_with_auto_drop",
    "run_sync_in_worker",
    "scrub_text_for_audit",
    "slog",
    "solve_step_chat_with_auto_drop",
    "stable_hash",
    "structlog",
    "sys",
    "uuid4",
    "validate_pipeline",
    "validation_errors_for_composer_surface",
    "yaml_generator",
]
