from __future__ import annotations

import asyncio
import json  # noqa: F401  # Preserve signed module statement positions.
from typing import TYPE_CHECKING, Literal, cast
from uuid import uuid4

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.plugin_capabilities import PluginCapability
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.infrastructure.validation import get_sink_config_model, get_source_config_model
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.guided.chat_solver import (
    build_step_chat_context_block,  # noqa: F401  # Preserve signed module statement positions.
)
from elspeth.web.composer.guided.emitters import build_component_review_turn
from elspeth.web.composer.guided.profile import TUTORIAL_PROFILE, WorkflowProfileKind, profile_for_kind
from elspeth.web.composer.guided.protocol import Turn, validate_current_turn
from elspeth.web.composer.guided.resolved import SinkResolved
from elspeth.web.composer.guided.stage_transitions import (
    AnsweredTurn,
    FieldSelectionResponse,
    InspectionResponse,
    PluginSelectionResponse,
    SchemaFormAuthority,
    SchemaFormResponse,
    add_component_intent,
    begin_component_edit,
    finish_component_review,
    remove_reviewed_component,
    reorder_reviewed_components,
    transition_sink_field_review,
    transition_sink_plugin_selection,
    transition_sink_schema_form,
    transition_source_inspection_review,
    transition_source_plugin_selection,
    transition_source_schema_form,
)
from elspeth.web.composer.guided.state_machine import ComponentTarget
from elspeth.web.composer.pipeline_proposal import composition_content_hash
from elspeth.web.composer.source_inspection import SourceInspectionFacts, inspect_blob_content
from elspeth.web.composer.tutorial_sample import (
    resolve_tutorial_sample_urls,
    tutorial_sample_base_url,
)
from elspeth.web.interpretation_state import refine_prompt_shield_warnings_for_availability
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.sessions._guided_step_chat import Step1SourceChatResult  # noqa: F401  # Preserve signed module statement positions.
from elspeth.web.sessions.guided_payloads import prepare_guided_json_payload
from elspeth.web.sessions.guided_replay import (
    guided_turn_token,
    load_guided_json_payload,
    parse_guided_response_descriptor,
    project_guided_response,
)
from elspeth.web.sessions.protocol import (
    GuidedAuditEvidence,
    GuidedCompositionStateResult,
    GuidedOperationActive,
    GuidedOperationCompleted,
    GuidedOperationConflictError,
    GuidedOperationFailed,
    GuidedOriginatingUserMessageDraft,
    GuidedReplayTurn,
    GuidedResponseDescriptor,
    GuidedStateOperationCommand,
    PreparedGuidedJsonPayload,
    guided_json_payload_id,
)
from elspeth.web.sessions.schemas import (
    AddComponentAction,
    ConvertGuidedRequest,
    EditComponentAction,
    FinishComponentsAction,
    GuidedStartOperationAbsentResponse,
    GuidedStartOperationCompletedResponse,
    GuidedStartOperationFailedResponse,
    GuidedStartOperationInProgressResponse,
    GuidedStartOperationReconciliationResponse,
    ReenterGuidedRequest,
    RemoveComponentAction,
    ReorderComponentsAction,
    StartGuidedRequest,
    TutorialSampleResponse,
)

from .._helpers import (
    UUID,
    Any,
    APIRouter,
    BlobServiceProtocol,
    BufferingRecorder,
    ChatTurnResponse,
    ComposerChatTurnStatus,
    CompositionState,
    CompositionStateData,
    CompositionStateRecord,
    ControlSignal,
    Depends,
    GetGuidedResponse,
    GuidedChatRequest,
    GuidedChatResponse,
    GuidedRespondRequest,
    GuidedRespondResponse,
    GuidedSession,
    GuidedSessionResponse,
    GuidedStep,
    HTTPException,
    InvariantError,
    Mapping,
    Request,
    SessionServiceProtocol,
    SourceResolved,
    TerminalKind,
    TerminalReason,
    TerminalState,
    TerminalStateResponse,
    TurnPayloadResponse,
    TurnRecord,
    TurnRecordResponse,
    TurnType,
    UserIdentity,
    _get_session_compose_lock_registry,
    _initial_composition_state_with_guided_session,
    _inspect_latest_ready_session_blob,
    _persist_llm_calls,
    _persist_tool_invocations,
    _replace,
    _request_plugin_policy_context,
    _safe_frame_strings,
    _state_from_record,
    _state_response,
    _track_compose_inflight,
    _verify_session_ownership,
    _workflow_profile_response,
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
    contextlib,
    deep_thaw,
    emit_dropped_to_freeform,
    emit_step_advanced,
    emit_turn_answered,
    emit_turn_emitted,
    generate_public_yaml,
    get_current_user,
    slog,
    sys,
)
from .pipeline_settlement import (
    _GUIDED_ATOMIC_SETTLEMENT_COMPLETED,
    _GUIDED_ATOMIC_SETTLEMENT_FAILURE,
    _await_guided_atomic_settlement,
    _await_with_deferred_cancellation,
)

if TYPE_CHECKING:
    from .guided_chat_atomic import GuidedChatProviderOutcome

_COMPLETED_TERMINAL_BEFORE_EXIT_META_KEY = "guided_completed_terminal_before_user_exit"
_MISSING_COMPLETED_TERMINAL_MARKER = object()


def _resolve_shield_available(snapshot: PluginAvailabilitySnapshot) -> bool:
    """Resolve whether the authorized prompt-injection shield is available for this user.

    Uses the same principal snapshot as every other policy surface.  Missing
    selection is the fail-safe State C result.
    """
    return dict(snapshot.selected).get(PluginCapability.PROMPT_SHIELD) is not None


def _guided_chat_wire_kind(status: ComposerChatTurnStatus) -> Literal["assistant", "synthetic_failure"]:
    """Map a ``StepChatResult``'s status to the wire discriminator (fp-review C-2)."""
    return "assistant" if status is ComposerChatTurnStatus.SUCCESS else "synthetic_failure"


def _chat_turn_synthetic_failure_reason(
    status: ComposerChatTurnStatus,
    error_class: str | None,
) -> Literal["quality_guard", "unavailable"] | None:
    """Classify a persisted ``ChatTurn``'s synthetic-failure cause (fp-review C-2).

    ``None`` on success. Otherwise ``"quality_guard"`` when a scaffold-leak
    guard rejected the reply, or ``"unavailable"`` for transient provider /
    solver failures. STEP_1/STEP_2 commit-seam rejection branches
    (``error_class="StepHandlerRejected"``) deliberately return ``None``:
    they are neither quality-guard nor availability events, and the audit row
    carries the redaction-safe classifier. ``error_class`` is compared by the
    literal class name string (``_guided_step_chat.py`` sets it via
    ``type(exc).__name__``); ``"AssistantScaffoldLeakError"`` is the ONLY
    class the dedicated scaffold-leak branches ever record. Persisted-only:
    the live ``GuidedChatResponse`` deliberately carries kind alone.
    """
    if status is ComposerChatTurnStatus.SUCCESS:
        return None
    if error_class == "StepHandlerRejected":
        return None
    return "quality_guard" if error_class == "AssistantScaffoldLeakError" else "unavailable"


def _turn_payload_response(
    turn: Mapping[str, Any] | None,
    *,
    guided: GuidedSession,
    shield_available: bool,
) -> TurnPayloadResponse | None:
    """Project the exact already-finalized payload bound by the occurrence.

    Shield availability is deliberately ignored here. ``confirm_wiring`` must
    be finalized before its hash, CAS payload, and ``TurnRecord`` are created;
    mutating it during response projection would make the wire body differ from
    its durable authority and make replay depend on mutable policy state.
    """
    del shield_available
    if turn is None:
        return None
    if not guided.history:
        raise AuditIntegrityError("Guided turn response has no persisted occurrence")
    record = guided.history[-1]
    expected_step_index = {
        GuidedStep.STEP_1_SOURCE: 0,
        GuidedStep.STEP_2_SINK: 1,
        GuidedStep.STEP_3_TRANSFORMS: 2,
        GuidedStep.STEP_4_WIRE: 3,
    }[record.step]
    if (
        record.turn_type.value != turn["type"]
        or expected_step_index != turn["step_index"]
        or record.payload_hash != guided_json_payload_id("turn", turn["payload"])
    ):
        raise AuditIntegrityError("Guided turn response differs from its persisted occurrence")
    return TurnPayloadResponse(
        type=turn["type"],
        step_index=turn["step_index"],
        turn_token=guided_turn_token(guided),
        payload=dict(turn["payload"]),
    )


def _finalize_guided_turn(turn: Mapping[str, Any], *, shield_available: bool) -> Turn:
    """Freeze request-scoped wire facts before occurrence custody is assigned."""

    payload = dict(deep_thaw(turn["payload"]))
    if turn["type"] == TurnType.CONFIRM_WIRING.value:
        payload["warnings"] = refine_prompt_shield_warnings_for_availability(
            payload["warnings"],
            shield_available=shield_available,
        )
    return Turn(
        type=turn["type"],
        step_index=turn["step_index"],
        payload=payload,
    )


def _load_durable_current_turn(
    guided: GuidedSession,
    *,
    payload_store: Any,
) -> tuple[Turn, PreparedGuidedJsonPayload]:
    """Load the exact current unanswered occurrence without live rebuilding."""

    guided_turn_token(guided)
    record = guided.history[-1]
    prepared = load_guided_json_payload(
        payload_store,
        payload_id=record.payload_hash,
        purpose="turn",
    )
    step_index = {
        GuidedStep.STEP_1_SOURCE: 0,
        GuidedStep.STEP_2_SINK: 1,
        GuidedStep.STEP_3_TRANSFORMS: 2,
        GuidedStep.STEP_4_WIRE: 3,
    }[record.step]
    turn = Turn(
        type=record.turn_type.value,
        step_index=step_index,
        payload=dict(deep_thaw(prepared.payload)),
    )
    try:
        validate_current_turn(record.step, turn)
    except ValueError as exc:
        raise AuditIntegrityError(f"Persisted current-schema turn is invalid: {exc}") from exc
    if record.turn_type in {TurnType.PROPOSE_PIPELINE, TurnType.CONFIRM_WIRING}:
        active_proposal = guided.active_proposal
        if active_proposal is None:
            raise AuditIntegrityError("Persisted guided proposal turn has no active proposal authority")
        if (
            turn["payload"]["proposal_id"] != str(active_proposal.proposal_id)
            or turn["payload"]["draft_hash"] != active_proposal.draft_hash
        ):
            raise AuditIntegrityError("Persisted guided proposal turn does not match active proposal authority")
    return turn, prepared


def _prepare_server_turn_occurrence(
    guided: GuidedSession,
    *,
    current_step: GuidedStep,
    turn: Turn,
    payload_store: Any,
) -> tuple[GuidedSession, TurnRecord, TurnType, PreparedGuidedJsonPayload]:
    """Validate first, then prepare CAS and append its exact occurrence."""

    try:
        validate_current_turn(current_step, turn)
    except ValueError as exc:
        raise InvariantError(f"Constructed current-schema turn is invalid: {exc}") from exc

    prepared = prepare_guided_json_payload(
        payload_store,
        purpose="turn",
        payload=turn["payload"],
    )
    new_guided, record, turn_type, payload_hash = _append_server_turn_record(
        guided,
        current_step=current_step,
        turn=turn,
    )
    if payload_hash != prepared.payload_id:
        raise AuditIntegrityError("Guided turn CAS differs from its history record")
    return new_guided, record, turn_type, prepared


def _turn_emission_evidence(
    *,
    step: GuidedStep,
    turn_type: TurnType,
    prepared: PreparedGuidedJsonPayload,
    composition_version: int,
    actor: str,
) -> GuidedAuditEvidence:
    recorder = BufferingRecorder()
    emit_turn_emitted(
        recorder,
        step=step,
        turn_type=turn_type,
        payload_hash=prepared.payload_id,
        payload_payload_id=prepared.payload_id,
        emitter="server",
        composition_version=composition_version,
        actor=actor,
    )
    return GuidedAuditEvidence(invocations=recorder.invocations)


router = APIRouter()


def _build_get_guided_turn(
    state: Any,
    guided: Any,
    *,
    catalog: Any,
) -> Any | None:
    """Deterministically rebuild a GET/reentry turn from schema-8 custody.

    Source and output workflows are selected by stable-id order. Pending
    intent phases identify the exact intra-step turn; an active edit target
    renders the corresponding reviewed component. Proposal payloads are held
    by the proposal service rather than in ``GuidedSession``, so STEP_3 has no
    synchronous checkpoint-only reconstruction here.
    """
    step = guided.step
    if step is GuidedStep.STEP_1_SOURCE:
        target = guided.active_edit_target
        if target is not None and target.kind == "source":
            edit_intent = guided.pending_source_intents.get(target.stable_id)
            if edit_intent is not None:
                if edit_intent.phase != "inspection_review":
                    raise InvariantError("active source edit has unsupported pending review custody")
                return build_step_1_inspect_and_confirm_turn_from_intent(edit_intent)
            return build_step_1_schema_form_turn_from_resolved(guided.reviewed_sources[target.stable_id], catalog)
        pending = next(
            (guided.pending_source_intents[stable_id] for stable_id in guided.source_order if stable_id in guided.pending_source_intents),
            None,
        )
        if pending is None:
            if guided.reviewed_sources:
                return build_component_review_turn(guided, "source")
            return build_initial_step_1_turn(state, blob_inspection=None, catalog=catalog)
        if pending.phase == "plugin_selection":
            return build_initial_step_1_turn(state, blob_inspection=None, catalog=catalog)
        if pending.phase == "plugin_options":
            if pending.plugin is None:  # pragma: no cover - guarded by SourceIntent
                raise InvariantError("STEP_1 plugin_options intent requires a plugin")
            return build_step_1_schema_form_turn(
                pending.plugin,
                catalog,
                inspection_facts=pending.inspection_facts,
            )
        if pending.phase == "inspection_review":
            return build_step_1_inspect_and_confirm_turn_from_intent(pending)
        raise InvariantError("STEP_1 pending source has an unsupported phase")
    if step is GuidedStep.STEP_2_SINK:
        target = guided.active_edit_target
        if target is not None and target.kind == "output":
            edit_intent = guided.pending_output_intents.get(target.stable_id)
            if edit_intent is not None:
                if edit_intent.phase != "field_review":
                    raise InvariantError("active output edit has unsupported pending review custody")
                observed_columns = tuple(
                    dict.fromkeys(
                        column
                        for stable_id in guided.source_order
                        if stable_id in guided.reviewed_sources
                        for column in guided.reviewed_sources[stable_id].observed_columns
                    )
                )
                return build_step_2_multi_select_turn(observed_columns)
            sink = SinkResolved(outputs=(guided.reviewed_outputs[target.stable_id],))
            return build_step_2_schema_form_turn_from_resolved(sink, catalog)
        pending = next(
            (guided.pending_output_intents[stable_id] for stable_id in guided.output_order if stable_id in guided.pending_output_intents),
            None,
        )
        if pending is None:
            if guided.reviewed_outputs:
                return build_component_review_turn(guided, "output")
            return build_step_2_single_select_turn(catalog)
        if pending.phase == "plugin_selection":
            return build_step_2_single_select_turn(catalog)
        if pending.phase == "plugin_options":
            if pending.plugin is None:  # pragma: no cover - guarded by SinkIntent
                raise InvariantError("STEP_2 plugin_options intent requires a plugin")
            return build_step_2_schema_form_turn(pending.plugin, catalog)
        if pending.phase == "field_review":
            observed_columns = tuple(
                dict.fromkeys(
                    column
                    for stable_id in guided.source_order
                    if stable_id in guided.reviewed_sources
                    for column in guided.reviewed_sources[stable_id].observed_columns
                )
            )
            return build_step_2_multi_select_turn(observed_columns)
        raise InvariantError("STEP_2 pending output has an unsupported phase")
    if step is GuidedStep.STEP_3_TRANSFORMS:
        return None
    if step is GuidedStep.STEP_4_WIRE:
        active = guided.active_proposal
        if active is None:
            raise InvariantError("STEP_4 wire review requires an active proposal binding")
        policy_validation = catalog.validate_composition_state(state)
        validation_state = state if policy_validation.validation.errors else policy_validation.executable_state
        return build_step_4_wire_turn(
            state,
            proposal_id=str(active.proposal_id),
            draft_hash=active.draft_hash,
            catalog=catalog,
            validation_state=validation_state,
            validation_summary=policy_validation.validation,
        )
    return None


def _step_1_plugin_hint(guided: GuidedSession) -> str | None:
    """Derive the Step-1 chat plugin hint from schema-8 structured state."""
    target = guided.active_edit_target
    if target is not None and target.kind == "source":
        return guided.reviewed_sources[target.stable_id].plugin
    pending = next(
        (guided.pending_source_intents[stable_id] for stable_id in guided.source_order if stable_id in guided.pending_source_intents),
        None,
    )
    if pending is not None:
        return pending.plugin
    return None


def _step_1_uploaded_input_filename(message: str) -> str | None:
    """Return the filename from the upload helper's Step-1 bind request."""
    stripped = message.strip()
    prefix = "I've uploaded \""
    suffix = '"; please use it as the pipeline input.'
    if not stripped.startswith(prefix) or not stripped.endswith(suffix):
        return None
    filename = stripped[len(prefix) : -len(suffix)]
    if not filename or '"' in filename or "\n" in filename or "\r" in filename:
        return None
    return filename


async def _source_from_latest_uploaded_blob_for_step_1_chat(
    *,
    message: str,
    plugin_hint: str | None,
    blob_service: BlobServiceProtocol,
    session_id: UUID,
) -> SourceResolved | None:
    """Build a source resolution from the newest uploaded blob for upload-hint chat.

    The frontend upload helper currently appends text like "I've uploaded
    <file>; please use it as the pipeline input." to the chat box. That text
    carries no blob id, so letting the LLM resolve it invites invented schema.
    When the session is already on a Step-1 schema form with a concrete plugin,
    bind the newest ready session blob through the same inspection prefill used
    by the visible form. The proposal custody boundary later resolves the
    masked ``blob:<id>`` sentinel authoritatively.
    """
    uploaded_filename = _step_1_uploaded_input_filename(message)
    if plugin_hint is None or uploaded_filename is None:
        return None
    inspection_facts = await _inspect_latest_ready_session_blob(
        blob_service,
        session_id,
        filename=uploaded_filename,
    )
    if inspection_facts is None:
        return None
    prefilled = build_step_1_source_prefill(plugin_hint, inspection_facts=inspection_facts)
    path = prefilled.get("path")
    if not isinstance(path, str):
        return None
    schema = prefilled.get("schema")
    options: dict[str, Any] = {"path": path}
    if isinstance(schema, Mapping):
        options["schema"] = dict(deep_thaw(schema))
    on_validation_failure = prefilled.get("on_validation_failure")
    if not isinstance(on_validation_failure, str) or not on_validation_failure:
        on_validation_failure = "discard"
    return SourceResolved(
        name="source",
        plugin=plugin_hint,
        options=options,
        observed_columns=tuple(inspection_facts.observed_headers or ()),
        sample_rows=(),
        on_validation_failure=on_validation_failure,
    )


def _guided_persisted_validity(
    state: CompositionState,
    *,
    catalog: PolicyCatalogView,
) -> tuple[bool, list[str] | None]:
    """Derive is_valid/validation_errors for a guided persist site.

    Mirrors the freeform persist convention's authoring-only fallback
    (``_composer_persisted_validation``'s last branch in ``_helpers.py``):
    Operator-profile aliases are the audit-safe persisted form, but contract
    probes require their executable binding. Validate policy and lower into an
    in-memory copy first; the authored state remains unchanged.

    No runtime preflight here (unlike ``_state_data_from_composer_state``):
    guided intermediate persists happen at points that do not uniformly carry
    a ``secret_service``/``settings`` runtime-preflight context, and runtime
    preflight is the freeform *commit* path's concern — the run-time gate in
    ``execution/service.py`` stays the hard backstop regardless of what this
    Stage-1-only check reports.
    """
    summary = catalog.validate_composition_state(state).validation
    if summary.is_valid:
        return True, None
    # Validator prose may contain paths, provider diagnostics, or operator
    # input. Guided checkpoints persist a closed status instead: this retains
    # truthful validity without widening the replay egress surface.
    return False, ["guided_composition_invalid"]


def _append_server_turn_record(
    guided: GuidedSession,
    *,
    current_step: GuidedStep,
    turn: Mapping[str, Any],
) -> tuple[GuidedSession, TurnRecord, TurnType, str]:
    """Append the server-emitted turn record for a guided step.

    The caller decides whether to persist the returned guided session. This
    lets the fresh-session GET /guided path return a deterministic initial
    turn without allocating an empty composition-state version.
    """
    if any(record.response_hash is None for record in guided.history):
        raise InvariantError("Cannot append a guided turn while an unanswered occurrence remains")
    try:
        turn_type = validate_current_turn(current_step, turn)
    except ValueError as exc:
        raise InvariantError(f"Constructed current-schema turn is invalid: {exc}") from exc
    payload_hash = guided_json_payload_id("turn", turn["payload"])
    new_record = TurnRecord(
        step=current_step,
        turn_type=turn_type,
        payload_hash=payload_hash,
        response_hash=None,
        emitter="server",
    )

    new_guided = _replace(guided, history=(*guided.history, new_record))
    return new_guided, new_record, turn_type, payload_hash


@router.get("/{session_id}/guided", response_model=GetGuidedResponse)
async def get_guided(
    session_id: UUID,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> GetGuidedResponse:
    """Return the current guided-mode state for a session.

    Fresh sessions are non-mutating on first visit: if there is no
    existing CompositionState, the initial GuidedSession and first turn are
    built in memory and returned with ``composition_state=None``. This keeps
    the version history from starting with an empty graph solely because
    the frontend auto-loaded guided mode.

    A missing occurrence is projected prospectively for both fresh and
    persisted sessions. GET never writes half of the state/evidence pair; the
    first fenced mutation materializes the occurrence, CAS, and audit evidence.

    Returns 404 if the session does not exist or does not belong to
    the requesting user.
    Returns 400 if the session's composition state has no guided_session
    attached (freeform session — use /api/sessions/{id}/messages instead).
    """
    await _verify_session_ownership(session_id, user, request)
    from elspeth.web.composer.guided.planning import (
        guided_private_reviewed_facts,
        verify_guided_proposal_projection,
    )
    from elspeth.web.composer.pipeline_proposal import PresentBase

    service: SessionServiceProtocol = request.app.state.session_service
    catalog, plugin_snapshot = _request_plugin_policy_context(request, user)
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

            active_authority = None
            if current_step is GuidedStep.STEP_3_TRANSFORMS and guided.active_proposal is not None:
                if state_record_out is None:
                    raise AuditIntegrityError("guided active proposal has no persisted checkpoint")
                reviewed_facts = guided_private_reviewed_facts(guided)
                try:
                    active_authority = await service.get_authoritative_pipeline_proposal(
                        session_id=session_id,
                        proposal_id=guided.active_proposal.proposal_id,
                        reviewed_facts=reviewed_facts,
                    )
                except (KeyError, ValueError) as exc:
                    raise AuditIntegrityError("guided proposal authority is missing or cross-session") from exc
                active = guided.active_proposal
                proposal = active_authority.proposal
                if (
                    active.draft_hash != proposal.draft_hash
                    or active.base != proposal.base
                    or active.reviewed_anchor_hash != proposal.reviewed_anchor_hash
                    or active.covered_deferred_intent_ids != proposal.covered_deferred_intent_ids
                    or active.creation_event_schema != "pipeline_proposal_created.v1"
                    or active.supersedes_proposal_id != active_authority.supersedes_proposal_id
                    or active.supersedes_draft_hash != proposal.supersedes_draft_hash
                ):
                    raise AuditIntegrityError("guided proposal reference differs from private authority")
                if type(proposal.base) is not PresentBase:
                    raise AuditIntegrityError("guided proposal authority has a non-present base")
                if proposal.base.state_id != state_record_out.id or proposal.base.composition_content_hash != composition_content_hash(
                    state
                ):
                    raise AuditIntegrityError("guided proposal base differs from current checkpoint")
                if active_authority.row.status == "rejected":
                    state_record_out = await service.reconcile_rejected_guided_pipeline_proposal(
                        session_id=session_id,
                        expected_current_state_id=state_record_out.id,
                        proposal_id=active.proposal_id,
                        draft_hash=active.draft_hash,
                        reviewed_facts=reviewed_facts,
                    )
                    state = _state_from_record(state_record_out)
                    if state.guided_session is None:  # pragma: no cover - service contract
                        raise AuditIntegrityError("guided proposal reconciliation removed guided checkpoint")
                    guided = state.guided_session
                    current_step = guided.step
                    active_authority = None
                elif active_authority.row.status != "pending":
                    raise AuditIntegrityError("guided active proposal is unexpectedly terminal")

            existing_record_for_step = (
                guided.history[-1]
                if guided.history and guided.history[-1].step is current_step and guided.history[-1].response_hash is None
                else None
            )

            # A persisted unanswered occurrence is immutable replay authority:
            # load its purpose-bound CAS payload exactly, without consulting
            # the live catalog or current plugin availability. Only a missing
            # occurrence is projected prospectively from live state.
            turn: Turn | None
            if guided.terminal is None:
                if existing_record_for_step is not None:
                    turn, _prepared = _load_durable_current_turn(
                        guided,
                        payload_store=request.app.state.payload_store,
                    )
                    if current_step is GuidedStep.STEP_3_TRANSFORMS:
                        if active_authority is None or guided.active_proposal is None:
                            raise AuditIntegrityError("guided proposal occurrence has no private authority")
                        catalog_ids = {
                            "source": frozenset(item.name for item in catalog.list_sources()),
                            "transform": frozenset(item.name for item in catalog.list_transforms()),
                            "sink": frozenset(item.name for item in catalog.list_sinks()),
                        }
                        verify_guided_proposal_projection(
                            payload=turn["payload"],
                            proposal_id=guided.active_proposal.proposal_id,
                            proposal=active_authority.proposal,
                            guided=guided,
                            catalog_plugin_ids=catalog_ids,
                        )
                else:
                    try:
                        turn = _build_get_guided_turn(state, guided, catalog=catalog)
                    except InvariantError as exc:
                        # Same B1-sanitization rationale as the POST /respond
                        # dispatcher's InvariantError catch: ``str(exc)`` can embed
                        # ``{d!r}`` of a corrupted Tier-1 record including Tier-3
                        # sample_rows. Static detail; slog carries exc_class +
                        # frames only.
                        slog.error(
                            "guided.invariant_violated",
                            session_id=str(session_id),
                            user_id=user.user_id,
                            exc_class=type(exc).__name__,
                            site="get_guided._build_get_guided_turn",
                            frames=_safe_frame_strings(exc),
                        )
                        raise HTTPException(
                            status_code=500,
                            detail="Server invariant violated. See application audit log for diagnostic detail.",
                        ) from exc
                    if turn is not None:
                        turn = _finalize_guided_turn(
                            turn,
                            shield_available=_resolve_shield_available(plugin_snapshot),
                        )
            else:
                turn = None
            turn_type: TurnType | None = TurnType(turn["type"]) if turn is not None else None
            payload_hash: str | None = guided_json_payload_id("turn", turn["payload"]) if turn is not None else None

            if existing_record_for_step is None and turn is not None:
                # First fetch for this step AND a turn exists: record TurnRecord,
                # persist, emit audit. When turn is None (terminal state or
                # STEP_3) there is no turn to record.
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
                new_guided, _new_record, turn_type, payload_hash = _append_server_turn_record(
                    guided,
                    current_step=current_step,
                    turn=turn,
                )
                guided = new_guided

            # Build response.  On re-fetch the same turn is returned (deterministic
            # rebuild) and the payload_hash matches what was recorded on first visit.
            terminal = guided.terminal
            shield_available = _resolve_shield_available(plugin_snapshot)
            return GetGuidedResponse(
                guided_session=GuidedSessionResponse(
                    step=guided.step.value,
                    history=[
                        TurnRecordResponse(
                            step=r.step.value,
                            turn_type=r.turn_type.value,
                            payload_hash=r.payload_hash,
                            response_hash=r.response_hash,
                            summary=r.summary,
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
                            assistant_message_kind=t.assistant_message_kind,
                            synthetic_failure_reason=t.synthetic_failure_reason,
                        )
                        for t in guided.chat_history
                    ],
                    chat_turn_seq=guided.chat_turn_seq,
                    profile=_workflow_profile_response(guided),
                ),
                next_turn=_turn_payload_response(turn, guided=guided, shield_available=shield_available),
                terminal=TerminalStateResponse(
                    kind=terminal.kind.value,
                    reason=terminal.reason.value if terminal.reason is not None else None,
                    pipeline_yaml=terminal.pipeline_yaml,
                )
                if terminal is not None
                else None,
                composition_state=_state_response(state_record_out, policy_catalog=catalog) if state_record_out is not None else None,
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
            # covers any :class:`ComposerLLMCall` rows buffered during guided
            # model invocations.
            # Without the second drain the LLM-call audit would be
            # garbage-collected with the recorder at function exit.
            primary_exc = sys.exception()
            if primary_exc is None:
                await _persist_tool_invocations(
                    service,
                    session_id,
                    recorder.invocations,
                    state_record_out.id if state_record_out is not None else None,
                    # Success path: no primary exception is in flight, so the
                    # success disposition applies. ``plugin_crash_pending``
                    # means "are we unwinding from a primary failure?", NOT
                    # "did a plugin crash" — here no, so a persist failure is
                    # a Tier-1 audit corruption that must raise (False). The
                    # unwind (else) branch below passes True.
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
                # Unwind path: a primary exception is in flight (this is the
                # ``finally`` block). ``plugin_crash_pending`` asks "are we
                # unwinding from a primary failure?", NOT "did a plugin
                # crash" — here the answer is yes. True selects the helper's
                # record-and-continue disposition (unwind counter + slog) so
                # an audit-persist failure does NOT raise AuditIntegrityError
                # and mask the primary failure the operator needs to see.
                try:
                    await _persist_tool_invocations(
                        service,
                        session_id,
                        recorder.invocations,
                        state_record_out.id if state_record_out is not None else None,
                        plugin_crash_pending=True,
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
                        plugin_crash_pending=True,
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


@router.get("/{session_id}/guided/tutorial-sample", response_model=TutorialSampleResponse)
async def get_guided_tutorial_sample(
    session_id: UUID,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> TutorialSampleResponse:
    """Return the runtime-derived synthetic-scrape inputs for a TUTORIAL session.

    Exposes the 3 synthetic sample-page URLs for the active tutorial session.
    The base is resolved via ``tutorial_sample_base_url`` (a configured
    ``WebSettings.tutorial_sample_base_url`` wins, else the canonical public
    GitHub Pages copy). The URLs are a runtime-derived payload computed by the
    server seam. The pages are publicly hosted, so the tutorial's ``web_scrape``
    node needs no server-injected SSRF allowlist (it uses the plugin default
    ``allowed_hosts="public_only"``).

    Read-only: this route never mutates state. Returns 404 if the session does
    not exist or does not belong to the requesting user. Returns 400 if the
    session has no guided session, or is not a tutorial session (a
    live/freeform session has no tutorial sample surface).
    """
    await _verify_session_ownership(session_id, user, request)
    service: SessionServiceProtocol = request.app.state.session_service

    state_record = await service.get_current_state(session_id)
    if state_record is None:
        raise HTTPException(
            status_code=400,
            detail="No active guided session for this session; start a tutorial session first.",
        )
    state = _state_from_record(state_record)
    guided = state.guided_session
    if guided is None:
        raise HTTPException(
            status_code=400,
            detail="Session is not in guided mode; the tutorial sample surface is guided-only.",
        )
    if guided.profile != TUTORIAL_PROFILE:
        raise HTTPException(
            status_code=400,
            detail="Session is not a tutorial session; no tutorial sample surface is available.",
        )

    settings = request.app.state.settings
    base_url = tutorial_sample_base_url(settings=settings)
    sample_urls = resolve_tutorial_sample_urls(base_url=base_url)
    return TutorialSampleResponse(sample_urls=list(sample_urls))


@router.post("/{session_id}/guided/reenter", response_model=GetGuidedResponse)
async def post_guided_reenter(
    session_id: UUID,
    body: ReenterGuidedRequest,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> GetGuidedResponse:
    """Re-enter guided mode after a deliberate user exit.

    This is a mode transition, not a turn answer. It is intentionally
    narrower than clearing any terminal state: solver-exhausted and
    protocol-violation terminals represent failed guided runs, while
    ``exited_to_freeform/user_pressed_exit`` records a reversible operator
    mode switch.
    """
    await _verify_session_ownership(session_id, user, request)
    service: SessionServiceProtocol = request.app.state.session_service
    catalog, plugin_snapshot = _request_plugin_policy_context(request, user)

    compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))

    def _response_from_record(record: CompositionStateRecord) -> GetGuidedResponse:
        descriptor = parse_guided_response_descriptor(record)
        if descriptor.kind != "guided_reenter":
            raise AuditIntegrityError("Guided re-entry result has the wrong replay descriptor")
        payloads: tuple[PreparedGuidedJsonPayload, ...] = ()
        if descriptor.next_turn is not None:
            payloads = (
                load_guided_json_payload(
                    request.app.state.payload_store,
                    payload_id=descriptor.next_turn.payload_id,
                    purpose="turn",
                ),
            )
        response = project_guided_response(record, payloads=payloads)
        if type(response) is not GetGuidedResponse:
            raise AuditIntegrityError("Guided re-entry projection returned the wrong response type")
        return response

    from elspeth.contracts.errors import AuditIntegrityError
    from elspeth.web.sessions.protocol import GuidedCompositionStateResult, GuidedOperationSettlementConflictError

    from ..guided_operations import (
        GuidedOperationLease,
        raise_guided_operation_failure,
        reserve_or_replay_guided_operation,
    )

    async def _replay(result: object) -> GetGuidedResponse:
        if type(result) is not GuidedCompositionStateResult:
            raise AuditIntegrityError("Guided re-entry replay has a non-state result locator")
        replay_record = await service.get_state_in_session(result.state_id, session_id)
        return _response_from_record(replay_record)

    reserved = await reserve_or_replay_guided_operation(
        service=service,
        session_id=session_id,
        kind="guided_reenter",
        request=body,
        replay=_replay,
        reserve_if_absent=False,
    )
    if reserved is None:
        # Reject invalid mode transitions before allocating an operation row.
        # The immutable checkpoint is re-read under the lock after a claim, so
        # this is classification rather than the write authority boundary.
        async with compose_lock:
            candidate_record = await service.get_current_state(session_id)
            if candidate_record is None:
                raise HTTPException(status_code=400, detail="Session has no guided state to re-enter.")
            candidate_state = _state_from_record(candidate_record)
            candidate_guided = candidate_state.guided_session
            if candidate_guided is None:
                raise HTTPException(
                    status_code=400,
                    detail="Session is not in guided mode. Use /api/sessions/{id}/messages.",
                )
            candidate_terminal = candidate_guided.terminal
            if candidate_terminal is None:
                raise HTTPException(status_code=409, detail="Guided session is already active.")
            if (
                candidate_terminal.kind is not TerminalKind.EXITED_TO_FREEFORM
                or candidate_terminal.reason is not TerminalReason.USER_PRESSED_EXIT
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Only a guided session ended by a user exit can be re-entered.",
                )
            if next((r for r in reversed(candidate_guided.history) if r.step == candidate_guided.step), None) is None:
                raise HTTPException(
                    status_code=409,
                    detail="Guided session cannot be re-entered because no current turn record exists.",
                )
        reserved = await reserve_or_replay_guided_operation(
            service=service,
            session_id=session_id,
            kind="guided_reenter",
            request=body,
            replay=_replay,
        )
    if reserved is None:  # pragma: no cover - reserve_if_absent defaults true
        raise AuditIntegrityError("Guided re-entry operation was not reserved")
    if not isinstance(reserved, GuidedOperationLease):
        return reserved

    async with compose_lock:
        state_record = await service.get_current_state(session_id)
        if state_record is None:
            raise HTTPException(
                status_code=400,
                detail="Session has no guided state to re-enter.",
            )

        state = _state_from_record(state_record)
        if state.guided_session is None:
            raise HTTPException(
                status_code=400,
                detail="Session is not in guided mode. Use /api/sessions/{id}/messages.",
            )

        guided = state.guided_session
        terminal = guided.terminal
        if terminal is None:
            raise HTTPException(
                status_code=409,
                detail="Guided session is already active.",
            )
        if terminal.kind is not TerminalKind.EXITED_TO_FREEFORM or terminal.reason is not TerminalReason.USER_PRESSED_EXIT:
            raise HTTPException(
                status_code=409,
                detail="Only a guided session ended by a user exit can be re-entered.",
            )

        current_record = next(
            (r for r in reversed(guided.history) if r.step == guided.step),
            None,
        )
        if current_record is None:
            raise HTTPException(
                status_code=409,
                detail="Guided session cannot be re-entered because no current turn record exists.",
            )
        existing_meta: dict[str, Any] = {}
        if state_record.composer_meta is not None:
            existing_meta = dict(deep_thaw(state_record.composer_meta))
        completed_terminal_raw = existing_meta.get(
            _COMPLETED_TERMINAL_BEFORE_EXIT_META_KEY,
            _MISSING_COMPLETED_TERMINAL_MARKER,
        )

        restored_terminal: TerminalState | None = None
        if completed_terminal_raw is not _MISSING_COMPLETED_TERMINAL_MARKER:
            try:
                if not isinstance(completed_terminal_raw, Mapping):
                    raise InvariantError("completed terminal re-entry marker must be a mapping")
                if set(completed_terminal_raw) != {"composition_hash"}:
                    raise InvariantError("completed terminal re-entry marker has an invalid shape")
                composition_hash = completed_terminal_raw["composition_hash"]
                if type(composition_hash) is not str:
                    raise InvariantError("completed terminal re-entry marker composition_hash must be a string")

                if composition_hash == composition_content_hash(state):
                    validation = catalog.validate_composition_state(state).validation
                    if not validation.is_valid:
                        raise InvariantError("unchanged completed pipeline failed validation during guided re-entry")
                    restored_terminal = TerminalState(
                        kind=TerminalKind.COMPLETED,
                        reason=None,
                        pipeline_yaml=generate_public_yaml(state),
                    )
            except InvariantError as exc:
                slog.error(
                    "guided.invariant_violated",
                    session_id=str(session_id),
                    user_id=user.user_id,
                    exc_class=type(exc).__name__,
                    site="guided_reenter_completed_terminal_marker",
                    frames=_safe_frame_strings(exc),
                )
                raise HTTPException(
                    status_code=500,
                    detail="Server invariant violated. See application audit log for diagnostic detail.",
                ) from exc

            existing_meta.pop(_COMPLETED_TERMINAL_BEFORE_EXIT_META_KEY)

        if restored_terminal is not None:
            new_guided = _replace(guided, terminal=restored_terminal)
            new_state = _replace(state, guided_session=new_guided)
            turn = None
            prepared_turn = None
            audit_evidence = GuidedAuditEvidence()
        else:
            active_guided = _replace(guided, terminal=None)
            active_state = _replace(state, guided_session=active_guided)
            rebuilt_turn = _build_get_guided_turn(active_state, active_guided, catalog=catalog)
            if rebuilt_turn is None:
                raise HTTPException(
                    status_code=409,
                    detail="Guided session cannot be re-entered because the current turn cannot be rebuilt.",
                )
            turn = _finalize_guided_turn(
                rebuilt_turn,
                shield_available=_resolve_shield_available(plugin_snapshot),
            )
            new_guided, _new_record, turn_type, prepared_turn = _prepare_server_turn_occurrence(
                active_guided,
                current_step=active_guided.step,
                turn=turn,
                payload_store=request.app.state.payload_store,
            )
            new_state = _replace(state, guided_session=new_guided)
            audit_evidence = _turn_emission_evidence(
                step=new_guided.step,
                turn_type=turn_type,
                prepared=prepared_turn,
                composition_version=new_state.version,
                actor=user.user_id,
            )

        new_composer_meta = {**existing_meta, "guided_session": new_guided.to_dict()}
        state_d = new_state.to_dict()
        persisted_is_valid, persisted_errors = _guided_persisted_validity(new_state, catalog=catalog)
        state_data = CompositionStateData(
            sources=state_d["sources"],
            nodes=state_d["nodes"],
            edges=state_d["edges"],
            outputs=state_d["outputs"],
            metadata_=state_d["metadata"],
            is_valid=persisted_is_valid,
            validation_errors=persisted_errors,
            composer_meta=new_composer_meta,
        )
        next_turn_descriptor = (
            GuidedReplayTurn(
                turn_type=TurnType(turn["type"]),
                step_index=turn["step_index"],
                payload_id=prepared_turn.payload_id,
            )
            if turn is not None and prepared_turn is not None
            else None
        )
        try:
            settlement = await service.settle_guided_state_operation(
                GuidedStateOperationCommand(
                    fence=reserved.fence,
                    expected_current_state_id=state_record.id,
                    expected_current_state_version=state_record.version,
                    expected_current_content_hash=composition_content_hash(state),
                    state_id=uuid4(),
                    state=state_data,
                    provenance="convergence_persist",
                    actor="composer_route",
                    response=GuidedResponseDescriptor(
                        kind="guided_reenter",
                        next_turn=next_turn_descriptor,
                        assistant_turn_seq=None,
                    ),
                    payloads=(prepared_turn,) if prepared_turn is not None else (),
                    audit_evidence=audit_evidence,
                ),
                payload_store=request.app.state.payload_store,
            )
        except GuidedOperationSettlementConflictError:
            failed = await service.fail_guided_operation(
                reserved.fence,
                failure_code="stale_conflict",
                actor="composer_route",
            )
            raise_guided_operation_failure(failed)
        return _response_from_record(settlement.result_state)


@router.post(
    "/{session_id}/guided/start/{operation_id}/reconcile",
    response_model=GuidedStartOperationReconciliationResponse,
)
async def reconcile_guided_start_operation(
    session_id: UUID,
    operation_id: UUID,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> GuidedStartOperationReconciliationResponse:
    """Return authoritative cold-start custody without replaying request content."""
    await _verify_session_ownership(session_id, user, request)
    service: SessionServiceProtocol = request.app.state.session_service
    compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))
    try:
        async with compose_lock:
            outcome = await service.reconcile_guided_start_operation(
                session_id=session_id,
                operation_id=str(operation_id),
                actor="composer_route",
            )
    except GuidedOperationConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail="Operation id is already bound to a different guided action.",
        ) from exc

    if outcome is None:
        return GuidedStartOperationAbsentResponse(status="absent")
    if type(outcome) is GuidedOperationActive:
        return GuidedStartOperationInProgressResponse(status="in_progress")
    if type(outcome) is GuidedOperationFailed:
        return GuidedStartOperationFailedResponse(status="failed", failure_code=outcome.failure_code)
    if type(outcome) is GuidedOperationCompleted:
        if type(outcome.result) is not GuidedCompositionStateResult or outcome.result.proposal_id is not None:
            raise AuditIntegrityError("guided-start reconciliation found an invalid completed result locator")
        return GuidedStartOperationCompletedResponse(
            status="completed",
            composition_state_id=outcome.result.state_id,
        )
    raise AuditIntegrityError("guided-start reconciliation returned an unsupported outcome")


@router.post("/{session_id}/guided/start", response_model=GetGuidedResponse)
async def post_guided_start(
    session_id: UUID,
    body: StartGuidedRequest,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> GetGuidedResponse:
    """Seed or replay one profile-owned guided session under operation custody."""
    await _verify_session_ownership(session_id, user, request)
    service: SessionServiceProtocol = request.app.state.session_service
    catalog, plugin_snapshot = _request_plugin_policy_context(request, user)

    # Bound the raw profile before canonical request hashing. Object-shaped,
    # oversized, or otherwise non-string inputs are outside the valid request
    # domain and must not reach RFC 8785 normalization.
    if not isinstance(body.profile, str) or len(body.profile) > 32:
        raise HTTPException(
            status_code=400,
            detail=(f"Invalid profile discriminator. Valid values: {sorted(k.value for k in WorkflowProfileKind)}."),
        )
    try:
        body.profile.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=(f"Invalid profile discriminator. Valid values: {sorted(k.value for k in WorkflowProfileKind)}."),
        ) from exc

    from elspeth.web.sessions.guided_operations import guided_operation_request_hash
    from elspeth.web.sessions.protocol import GuidedOperationConflictError

    try:
        await service.get_guided_operation(
            session_id=session_id,
            operation_id=body.operation_id,
            kind="guided_start",
            request_hash=guided_operation_request_hash(
                session_id=session_id,
                kind="guided_start",
                request=body,
            ),
        )
    except GuidedOperationConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail="Operation id is already bound to a different request.",
        ) from exc

    # Tier-3 -> Tier-2 coercion at the profile-kind boundary. A stale client
    # sending an unknown discriminator gets a 400 with a generic message
    # rather than a Pydantic 422; the typed kind then selects a SERVER-owned
    # constant — the client never supplies the profile object. Do not echo
    # the raw value: it may be a long string or an attempted profile object
    # carrying attacker-controlled profile fields.
    try:
        profile_kind = WorkflowProfileKind(body.profile)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=(f"Unknown profile discriminator. Valid values: {sorted(k.value for k in WorkflowProfileKind)}."),
        ) from exc
    profile = profile_for_kind(profile_kind)
    if profile_kind is WorkflowProfileKind.LIVE:
        if body.intent is None:
            raise HTTPException(status_code=400, detail="Live guided start requires a visible intent.")
    elif body.intent is not None:
        raise HTTPException(status_code=400, detail="Tutorial guided start forbids a client intent.")

    from elspeth.contracts.errors import AuditIntegrityError
    from elspeth.web.sessions.protocol import (
        GuidedCompositionStateResult,
        GuidedOperationFailureCode,
        GuidedOperationFenceLostError,
        GuidedOperationSettlementConflictError,
    )

    from ..guided_operations import (
        GuidedOperationLease,
        guided_response_hash,
        raise_guided_operation_failure,
        reserve_or_replay_guided_operation,
    )

    def _response_from_record(record: CompositionStateRecord) -> GetGuidedResponse:
        state = _state_from_record(record)
        guided = state.guided_session
        if guided is None:
            raise AuditIntegrityError("Guided start result state has no guided checkpoint")
        terminal = guided.terminal
        turn = None
        if terminal is None:
            turn, _prepared = _load_durable_current_turn(
                guided,
                payload_store=request.app.state.payload_store,
            )
        terminal_response = (
            TerminalStateResponse(
                kind=terminal.kind.value,
                reason=terminal.reason.value if terminal.reason is not None else None,
                pipeline_yaml=terminal.pipeline_yaml,
            )
            if terminal is not None
            else None
        )
        return GetGuidedResponse(
            guided_session=GuidedSessionResponse(
                step=guided.step.value,
                history=[
                    TurnRecordResponse(
                        step=turn_record.step.value,
                        turn_type=turn_record.turn_type.value,
                        payload_hash=turn_record.payload_hash,
                        response_hash=turn_record.response_hash,
                        summary=turn_record.summary,
                        emitter=turn_record.emitter,
                    )
                    for turn_record in guided.history
                ],
                terminal=terminal_response,
                chat_history=[
                    ChatTurnResponse(
                        role=chat_turn.role.value,
                        content=chat_turn.content,
                        seq=chat_turn.seq,
                        step=chat_turn.step.value,
                        ts_iso=chat_turn.ts_iso,
                        assistant_message_kind=chat_turn.assistant_message_kind,
                        synthetic_failure_reason=chat_turn.synthetic_failure_reason,
                    )
                    for chat_turn in guided.chat_history
                ],
                chat_turn_seq=guided.chat_turn_seq,
                profile=_workflow_profile_response(guided),
            ),
            next_turn=_turn_payload_response(
                turn,
                guided=guided,
                shield_available=_resolve_shield_available(plugin_snapshot),
            ),
            terminal=terminal_response,
            composition_state=_state_response(record, policy_catalog=catalog),
        )

    async def _verify_start_root(record: CompositionStateRecord) -> None:
        guided = _state_from_record(record).guided_session
        if guided is None:
            raise AuditIntegrityError("Guided start result state has no guided checkpoint")
        if guided.profile != profile:
            raise GuidedOperationSettlementConflictError()
        if profile_kind is WorkflowProfileKind.TUTORIAL:
            if guided.root_intent_message_id is not None:
                raise AuditIntegrityError("Tutorial guided start unexpectedly owns a client root intent")
            return
        if guided.root_intent_message_id is None:
            raise AuditIntegrityError("Live guided start is missing its durable root intent")
        matches = [
            message for message in await service.get_messages(session_id, limit=None) if str(message.id) == guided.root_intent_message_id
        ]
        if len(matches) != 1 or matches[0].role != "user" or matches[0].writer_principal != "route_user_message":
            raise AuditIntegrityError("Live guided start root intent failed session/role/content custody")
        if matches[0].content != body.intent:
            raise GuidedOperationSettlementConflictError()

    async def _replay(result: object) -> GetGuidedResponse:
        if type(result) is not GuidedCompositionStateResult:
            raise AuditIntegrityError("Guided start replay has a non-state result locator")
        replay_record = await service.get_state_in_session(result.state_id, session_id)
        await _verify_start_root(replay_record)
        return _response_from_record(replay_record)

    # Existing operations are immutable protocol facts. Resolve them before
    # classifying the mutable current head so a committed retry still replays
    # its located result after later session work changes modes.
    pending = await reserve_or_replay_guided_operation(
        service=service,
        session_id=session_id,
        kind="guided_start",
        request=body,
        replay=_replay,
        reserve_if_absent=False,
    )
    if pending is not None and not isinstance(pending, GuidedOperationLease):
        return pending

    compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))
    # Classify the exact head before allocating an operation row. Invalid
    # freeform starts remain ordinary 409s and leave no retry artefact behind.
    async with compose_lock:
        observed_record = await service.get_current_state(session_id)
        if observed_record is not None:
            observed_state = _state_from_record(observed_record)
            if observed_state.guided_session is None and pending is None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Cannot start guided on a session that already has existing "
                        "freeform composition state. Create a new session or fork before "
                        "starting the tutorial profile."
                    ),
                )
            observed_head: tuple[UUID, int] | None = (observed_record.id, observed_record.version)
        else:
            observed_head = None

    while True:
        if pending is None:
            reserved = await reserve_or_replay_guided_operation(
                service=service,
                session_id=session_id,
                kind="guided_start",
                request=body,
                replay=_replay,
            )
        else:
            reserved = pending
            pending = None
        if reserved is None:  # pragma: no cover - reserve_if_absent defaults true
            raise AuditIntegrityError("Guided start operation was not reserved")
        if not isinstance(reserved, GuidedOperationLease):
            return reserved

        try:
            async with compose_lock:
                # Waiting for the compose lock may consume most of a lease.
                # Renewal under the lock establishes fresh write authority
                # before the exact head is inspected or settled.
                renewed_fence = await service.renew_guided_operation(
                    reserved.fence,
                    actor="composer_route",
                    lease_seconds=300,
                )
                current_record = await service.get_current_state(session_id)
                if current_record is not None:
                    current_state = _state_from_record(current_record)
                    if current_state.guided_session is None:
                        raise GuidedOperationSettlementConflictError()
                    await _verify_start_root(current_record)
                    # A distinct start may have won after both requests
                    # preflighted an empty head. Settle the exact current guided
                    # checkpoint as an idempotent no-op; the service CAS prevents
                    # it from blessing a stale head.
                    settled_record = await _await_guided_atomic_settlement(
                        service.complete_existing_state_guided_operation(
                            renewed_fence,
                            state_id=current_record.id,
                            expected_current_state_id=current_record.id,
                            expected_current_state_version=current_record.version,
                            actor="composer_route",
                            response_hash_factory=lambda record: guided_response_hash(_response_from_record(record)),
                        )
                    )
                    return _response_from_record(settled_record)

                if observed_head is not None:
                    raise AuditIntegrityError("Guided start head disappeared after preflight")

                root_message = (
                    GuidedOriginatingUserMessageDraft(message_id=uuid4(), content=body.intent)
                    if profile_kind is WorkflowProfileKind.LIVE and body.intent is not None
                    else None
                )
                new_state = _initial_composition_state_with_guided_session(profile=profile)
                seeded_guided = new_state.guided_session
                if seeded_guided is None:  # pragma: no cover - helper contract
                    raise InvariantError("post_guided_start: initial state has no guided_session")
                if root_message is not None:
                    seeded_guided = _replace(
                        seeded_guided,
                        root_intent_message_id=str(root_message.message_id),
                    )
                seed_turn = _build_get_guided_turn(new_state, seeded_guided, catalog=catalog)
                if seed_turn is None:  # pragma: no cover - initial STEP_1 always emits
                    raise InvariantError("post_guided_start: initial guided session has no first turn")
                seed_turn = _finalize_guided_turn(
                    seed_turn,
                    shield_available=_resolve_shield_available(plugin_snapshot),
                )
                seeded_guided, _record, seed_turn_type, prepared_seed_turn = _prepare_server_turn_occurrence(
                    seeded_guided,
                    current_step=seeded_guided.step,
                    turn=seed_turn,
                    payload_store=request.app.state.payload_store,
                )
                seed_evidence = _turn_emission_evidence(
                    step=seeded_guided.step,
                    turn_type=seed_turn_type,
                    prepared=prepared_seed_turn,
                    composition_version=new_state.version,
                    actor=user.user_id,
                )
                new_state = _replace(new_state, guided_session=seeded_guided)
                state_d = new_state.to_dict()
                persisted_is_valid, persisted_errors = _guided_persisted_validity(new_state, catalog=catalog)
                state_data = CompositionStateData(
                    sources=state_d["sources"],
                    nodes=state_d["nodes"],
                    edges=state_d["edges"],
                    outputs=state_d["outputs"],
                    metadata_=state_d["metadata"],
                    is_valid=persisted_is_valid,
                    validation_errors=persisted_errors,
                    composer_meta={"guided_session": seeded_guided.to_dict()},
                )
                seed_outcome = await _await_guided_atomic_settlement(
                    service.seed_or_complete_guided_start_operation(
                        renewed_fence,
                        state=state_data,
                        provenance="session_seed",
                        actor="composer_route",
                        response_hash_factory=lambda record: guided_response_hash(_response_from_record(record)),
                        payloads=(prepared_seed_turn,),
                        audit_evidence=seed_evidence,
                        originating_message=root_message,
                        payload_store=request.app.state.payload_store,
                    )
                )
                return _response_from_record(seed_outcome.state)
        except GuidedOperationFenceLostError:
            # Never poll while holding the compose lock. Rejoin outside it;
            # reserve either observes the winner or performs the sole takeover.
            continue
        except asyncio.CancelledError as exc:
            if exc.__dict__.get(_GUIDED_ATOMIC_SETTLEMENT_COMPLETED) is True:
                raise
            settlement_failure = exc.__dict__.get(_GUIDED_ATOMIC_SETTLEMENT_FAILURE)
            if isinstance(settlement_failure, GuidedOperationFenceLostError):
                raise
            if settlement_failure is not None:
                cancel_failure_code: GuidedOperationFailureCode = (
                    "stale_conflict"
                    if isinstance(settlement_failure, GuidedOperationSettlementConflictError)
                    else "integrity_error"
                    if isinstance(settlement_failure, AuditIntegrityError)
                    else "operation_failed"
                )
            else:
                caller_task = asyncio.current_task()
                cancel_failure_code = (
                    "request_cancelled" if caller_task is not None and caller_task.cancelling() > 0 else "operation_failed"
                )
            try:
                await _await_with_deferred_cancellation(
                    service.fail_guided_operation(
                        reserved.fence,
                        failure_code=cancel_failure_code,
                        actor="composer_route",
                    )
                )
            except GuidedOperationFenceLostError as fence_lost:
                raise exc from fence_lost
            except Exception as failure_exc:
                raise exc from failure_exc
            if settlement_failure is not None:
                raise exc from settlement_failure
            raise
        except Exception as exc:
            failure_code: GuidedOperationFailureCode = (
                "stale_conflict"
                if isinstance(exc, GuidedOperationSettlementConflictError)
                else "integrity_error"
                if isinstance(exc, AuditIntegrityError)
                else "operation_failed"
            )
            if isinstance(exc, AuditIntegrityError):
                slog.error(
                    "guided.operation_terminal_failure",
                    session_id=str(session_id),
                    user_id=user.user_id,
                    exc_class=type(exc).__name__,
                    site="post_guided_start",
                    frames=_safe_frame_strings(exc),
                )
            try:
                failed = await service.fail_guided_operation(
                    reserved.fence,
                    failure_code=failure_code,
                    actor="composer_route",
                )
            except GuidedOperationFenceLostError:
                continue
            raise_guided_operation_failure(failed)


def _schema8_unsupported_stage(step: GuidedStep) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "code": "guided_respond_stage_unsupported",
            "detail": f"Schema-8 RESPOND is not available for {step.value}.",
        },
    )


def _verify_schema8_proposal_binding(guided: GuidedSession, body: GuidedRespondRequest) -> None:
    """Fail closed when a response names anything but the active proposal."""
    if body.control_signal == ControlSignal.EXIT_TO_FREEFORM.value:
        return
    if body.proposal_id is None:
        if guided.active_proposal is not None:
            raise HTTPException(
                status_code=409,
                detail="the active guided proposal requires proposal_id and draft_hash",
            )
        return
    active = guided.active_proposal
    if active is None or body.proposal_id != str(active.proposal_id) or body.draft_hash != active.draft_hash:
        raise HTTPException(
            status_code=409,
            detail="proposal_id and draft_hash do not identify the active guided proposal",
        )


def _schema8_prospective_occurrence(
    state: CompositionState,
    guided: GuidedSession,
    *,
    catalog: PolicyCatalogView,
    shield_available: bool,
    payload_store: Any,
) -> tuple[GuidedSession, Turn, PreparedGuidedJsonPayload]:
    if guided.history and guided.history[-1].response_hash is None:
        turn, prepared = _load_durable_current_turn(guided, payload_store=payload_store)
        return guided, turn, prepared
    prospective_turn = _build_get_guided_turn(state, guided, catalog=catalog)
    if prospective_turn is None:
        raise _schema8_unsupported_stage(guided.step)
    turn = cast("Turn", prospective_turn)
    finalized = _finalize_guided_turn(turn, shield_available=shield_available)
    prospective, _record, _turn_type, payload_id = _append_server_turn_record(
        guided,
        current_step=guided.step,
        turn=finalized,
    )
    return (
        prospective,
        finalized,
        PreparedGuidedJsonPayload(payload_id=payload_id, purpose="turn", payload=finalized["payload"]),
    )


def _schema8_only_response_fields(body: GuidedRespondRequest, *allowed: str) -> None:
    values = {
        "chosen": body.chosen,
        "edited_values": body.edited_values,
        "custom_inputs": body.custom_inputs,
        "control_signal": body.control_signal,
    }
    present = {name for name, value in values.items() if value is not None}
    if present - set(allowed):
        raise ValueError("response fields are not legal for the current turn type")


def _schema8_permitted_plugins(turn: Turn) -> tuple[str, ...]:
    options = turn["payload"].get("options")
    if type(options) is not list:
        raise InvariantError("single-select turn has no server-held option list")
    plugins: list[str] = []
    for option in options:
        if not isinstance(option, Mapping) or type(option.get("id")) is not str:
            raise InvariantError("single-select turn contains a malformed option")
        plugins.append(option["id"])
    return tuple(plugins)


def _schema8_pending_target(guided: GuidedSession, *, source: bool, phase: str) -> str:
    intents = guided.pending_source_intents if source else guided.pending_output_intents
    matches = [stable_id for stable_id, intent in intents.items() if intent.phase == phase]
    if len(matches) != 1:
        raise InvariantError("guided turn does not have exactly one server-held target")
    return matches[0]


def _schema8_form_target(guided: GuidedSession, *, source: bool) -> tuple[str, str]:
    """Return the server-held schema-form target and plugin.

    New components hold that authority in a pending intent. An edit instead
    holds it in the reviewed component named by ``active_edit_target`` until
    the edited form needs a follow-up inspection/field-review turn.
    """

    intents = guided.pending_source_intents if source else guided.pending_output_intents
    reviewed = guided.reviewed_sources if source else guided.reviewed_outputs
    matches = [stable_id for stable_id, intent in intents.items() if intent.phase == "plugin_options"]
    if len(matches) == 1:
        target = matches[0]
        plugin = intents[target].plugin
    elif not matches:
        active = guided.active_edit_target
        expected_kind = "source" if source else "output"
        if active is None or active.kind != expected_kind or active.stable_id not in reviewed:
            raise InvariantError("guided schema-form turn does not have exactly one server-held target")
        target = active.stable_id
        plugin = reviewed[target].plugin
    else:
        raise InvariantError("guided schema-form turn does not have exactly one server-held target")
    if type(plugin) is not str:
        raise InvariantError("guided schema-form target has no server-held plugin")
    return target, plugin


def _schema8_active_source_edit_blob_id(guided: GuidedSession) -> UUID | None:
    """Resolve exact blob custody for an active reviewed-source edit."""

    active = guided.active_edit_target
    if active is None or active.kind != "source":
        return None
    source = guided.reviewed_sources.get(active.stable_id)
    if source is None:
        raise InvariantError("active source edit target is not reviewed")
    raw_blob_id = source.options.get("blob_ref")
    if raw_blob_id is None:
        path = source.options.get("path")
        raw_blob_id = path.removeprefix("blob:") if type(path) is str and path.startswith("blob:") else None
    if raw_blob_id is None:
        return None
    try:
        blob_id = UUID(str(raw_blob_id))
    except (TypeError, ValueError) as exc:
        raise InvariantError("active source edit has a malformed blob custody id") from exc
    if str(blob_id) != str(raw_blob_id):
        raise InvariantError("active source edit blob custody id is not canonical")
    return blob_id


async def _schema8_active_source_edit_inspection(
    blob_service: BlobServiceProtocol,
    session_id: UUID,
    guided: GuidedSession,
) -> SourceInspectionFacts | None:
    """Re-inspect the exact blob owned by the active source edit target."""

    blob_id = _schema8_active_source_edit_blob_id(guided)
    if blob_id is None:
        return None
    record = await blob_service.get_blob(blob_id)
    if record.session_id != session_id or record.status != "ready":
        raise InvariantError("active source edit blob is not a ready blob owned by this session")
    content = await blob_service.read_blob_content(blob_id)
    return inspect_blob_content(
        content=content,
        filename=record.filename,
        mime_type=record.mime_type,
        blob_id=record.id,
        content_hash=record.content_hash,
    )


def _schema8_server_options(prefilled: Mapping[str, Any]) -> dict[str, object]:
    return {
        name: deep_thaw(value)
        for name, value in prefilled.items()
        if "blob" in name or (name in {"path", "file"} and type(value) is str and value.startswith("blob:"))
    }


def _schema8_schema_authority(
    *,
    turn: Turn,
    plugin: str,
    options: Mapping[str, Any],
    source: bool,
) -> SchemaFormAuthority:
    payload = turn["payload"]
    knobs = payload.get("knobs")
    prefilled = payload.get("prefilled")
    if not isinstance(knobs, Mapping) or not isinstance(prefilled, Mapping):
        raise InvariantError("schema-form turn is missing server-held form authority")
    server_options = _schema8_server_options(prefilled)
    merged = dict(deep_thaw(options))
    merged.update(server_options)
    config_model = get_source_config_model(plugin) if source else get_sink_config_model(plugin)
    model_validated = merged
    if config_model is not None:
        # Node failure policies are server-owned structural fields rather than
        # plugin config. Keep them in transition authority, but do not feed
        # them into strict plugin models that correctly reject extra keys.
        # Source config models own ``on_validation_failure`` directly. Sink
        # ``on_write_failure`` belongs to the node wrapper, not the plugin.
        plugin_options = merged if source else {name: value for name, value in merged.items() if name != "on_write_failure"}
        config = config_model.from_dict(plugin_options, plugin_name=plugin)
        model_validated = config.model_dump(mode="json", by_alias=True)
        # Pydantic expands nested semantic values (notably
        # ``schema: {mode: observed}``) with nullable defaults.  The pure
        # transition requires every submitted value to survive validation
        # exactly, so retain those validated wire values while still carrying
        # model defaults for fields the client omitted.
        model_validated.update(deep_thaw(merged))
    return SchemaFormAuthority(knobs=knobs, model_validated_options=model_validated, server_options=server_options)


def _schema8_transition(
    guided: GuidedSession,
    turn: Turn,
    body: GuidedRespondRequest,
    *,
    new_stable_id: UUID,
    source_inspection_facts: SourceInspectionFacts | None = None,
) -> tuple[GuidedSession, Mapping[str, Any]]:
    if body.proposal_id is not None or body.draft_hash is not None or body.edit_target is not None:
        raise _schema8_unsupported_stage(guided.step)
    answered = AnsweredTurn(history_index=len(guided.history) - 1)
    turn_type = TurnType(turn["type"])

    if body.component_action is not None:
        if turn_type is not TurnType.REVIEW_COMPONENTS:
            raise ValueError("component_action is legal only for a component review turn")
        review_kind = turn["payload"].get("component_kind")
        if type(review_kind) is not str or review_kind not in {"source", "output"}:
            raise InvariantError("component review turn has no valid server-held component kind")
        action = body.component_action
        action_kind = action.target.kind if isinstance(action, (EditComponentAction, RemoveComponentAction)) else action.component_kind
        if action_kind != review_kind:
            raise ValueError("component action kind does not match the current review stage")
        if isinstance(action, AddComponentAction):
            updated = add_component_intent(guided, action.component_kind, new_stable_id)
        elif isinstance(action, EditComponentAction):
            updated = begin_component_edit(
                guided,
                ComponentTarget(kind=action.target.kind, stable_id=str(action.target.stable_id)),
            )
        elif isinstance(action, RemoveComponentAction):
            updated = remove_reviewed_component(
                guided,
                ComponentTarget(kind=action.target.kind, stable_id=str(action.target.stable_id)),
            )
        elif isinstance(action, ReorderComponentsAction):
            updated = reorder_reviewed_components(guided, action.component_kind, tuple(action.stable_ids))
        elif isinstance(action, FinishComponentsAction):
            updated = finish_component_review(guided, action.component_kind)
        else:  # pragma: no cover - closed discriminated request union
            raise InvariantError("component review received an unsupported action model")
        return updated, {"component_action": action.model_dump(mode="json")}
    if turn_type is TurnType.REVIEW_COMPONENTS:
        raise ValueError("component review turns require component_action")

    if turn_type is TurnType.SINGLE_SELECT:
        _schema8_only_response_fields(body, "chosen")
        if body.chosen is None:
            raise ValueError("single_select requires chosen")
        plugin_response = PluginSelectionResponse(chosen=body.chosen)
        if guided.step is GuidedStep.STEP_1_SOURCE:
            selection_targets = [
                stable_id for stable_id, intent in guided.pending_source_intents.items() if intent.phase == "plugin_selection"
            ]
            updated = transition_source_plugin_selection(
                guided,
                turn=answered,
                response=plugin_response,
                permitted_plugins=_schema8_permitted_plugins(turn),
                inspection_facts=source_inspection_facts,
                new_stable_id=new_stable_id if not selection_targets else None,
                target_id=selection_targets[0] if len(selection_targets) == 1 else None,
            )
        elif guided.step is GuidedStep.STEP_2_SINK:
            selection_targets = [
                stable_id for stable_id, intent in guided.pending_output_intents.items() if intent.phase == "plugin_selection"
            ]
            updated = transition_sink_plugin_selection(
                guided,
                turn=answered,
                response=plugin_response,
                permitted_plugins=_schema8_permitted_plugins(turn),
                new_stable_id=new_stable_id if not selection_targets else None,
                target_id=selection_targets[0] if len(selection_targets) == 1 else None,
            )
        else:
            raise _schema8_unsupported_stage(guided.step)
        return updated, {"chosen": list(body.chosen)}

    if turn_type is TurnType.SCHEMA_FORM:
        _schema8_only_response_fields(body, "edited_values")
        edited = body.edited_values
        if type(edited) is not dict or set(edited) != {"plugin", "options"}:
            raise ValueError("schema_form edited_values must contain exactly plugin and options")
        plugin = edited["plugin"]
        options = edited["options"]
        if type(plugin) is not str or not isinstance(options, Mapping):
            raise ValueError("schema_form plugin and options have invalid types")
        form_response = SchemaFormResponse(plugin=plugin, options=options)
        if guided.step is GuidedStep.STEP_1_SOURCE:
            target, held_plugin = _schema8_form_target(guided, source=True)
            if plugin != held_plugin:
                raise ValueError("schema-form plugin does not echo the server-held source plugin")
            is_edit = (
                guided.active_edit_target is not None
                and guided.active_edit_target.kind == "source"
                and guided.active_edit_target.stable_id == target
            )
            updated = transition_source_schema_form(
                guided,
                target_id=target,
                turn=answered,
                response=form_response,
                authority=_schema8_schema_authority(turn=turn, plugin=held_plugin, options=options, source=True),
                edit_inspection_facts=source_inspection_facts if is_edit else None,
            )
        elif guided.step is GuidedStep.STEP_2_SINK:
            target, held_plugin = _schema8_form_target(guided, source=False)
            if plugin != held_plugin:
                raise ValueError("schema-form plugin does not echo the server-held sink plugin")
            updated = transition_sink_schema_form(
                guided,
                target_id=target,
                turn=answered,
                response=form_response,
                authority=_schema8_schema_authority(turn=turn, plugin=held_plugin, options=options, source=False),
            )
        else:
            raise _schema8_unsupported_stage(guided.step)
        return updated, {"edited_values": {"plugin": plugin, "options": deep_thaw(options)}}

    if turn_type is TurnType.INSPECT_AND_CONFIRM and guided.step is GuidedStep.STEP_1_SOURCE:
        _schema8_only_response_fields(body, "edited_values")
        edited = body.edited_values
        if type(edited) is not dict or set(edited) != {"columns"} or type(edited["columns"]) is not list:
            raise ValueError("inspect_and_confirm edited_values must contain exactly columns")
        target = _schema8_pending_target(guided, source=True, phase="inspection_review")
        updated = transition_source_inspection_review(
            guided,
            target_id=target,
            turn=answered,
            response=InspectionResponse(columns=edited["columns"]),
        )
        return updated, {"edited_values": {"columns": list(edited["columns"])}}

    if turn_type is TurnType.MULTI_SELECT_WITH_CUSTOM and guided.step is GuidedStep.STEP_2_SINK:
        _schema8_only_response_fields(body, "chosen", "custom_inputs", "control_signal")
        signal = ControlSignal(body.control_signal) if body.control_signal is not None else None
        target = _schema8_pending_target(guided, source=False, phase="field_review")
        updated = transition_sink_field_review(
            guided,
            target_id=target,
            turn=answered,
            response=FieldSelectionResponse(
                chosen=body.chosen or (),
                custom_inputs=body.custom_inputs or (),
                control_signal=signal,
            ),
        )
        return updated, {
            "chosen": list(body.chosen or ()),
            "custom_inputs": list(body.custom_inputs or ()),
            "control_signal": signal.value if signal is not None else None,
        }
    raise _schema8_unsupported_stage(guided.step)


def _schema8_answer_and_project_next(
    state: CompositionState,
    guided: GuidedSession,
    current_turn: Turn,
    body: GuidedRespondRequest,
    *,
    catalog: PolicyCatalogView,
    shield_available: bool,
    new_stable_id: UUID,
    source_inspection_facts: SourceInspectionFacts | None = None,
) -> tuple[CompositionState, PreparedGuidedJsonPayload, Turn | None, PreparedGuidedJsonPayload | None]:
    if body.control_signal == ControlSignal.EXIT_TO_FREEFORM.value:
        _schema8_only_response_fields(body, "control_signal")
        response_payload: Mapping[str, Any] = {"control_signal": ControlSignal.EXIT_TO_FREEFORM.value}
        transitioned = _replace(
            guided,
            terminal=TerminalState(
                kind=TerminalKind.EXITED_TO_FREEFORM,
                reason=TerminalReason.USER_PRESSED_EXIT,
                pipeline_yaml=None,
            ),
            transition_consumed=True,
        )
    else:
        transitioned, response_payload = _schema8_transition(
            guided,
            current_turn,
            body,
            new_stable_id=new_stable_id,
            source_inspection_facts=source_inspection_facts,
        )
    response_id = guided_json_payload_id("turn_response", response_payload)
    answered_record = _replace(
        transitioned.history[-1],
        response_hash=response_id,
        summary="Structured guided response accepted.",
    )
    transitioned = _replace(transitioned, history=(*transitioned.history[:-1], answered_record))
    next_turn = (
        None
        if transitioned.terminal is not None
        else _build_get_guided_turn(_replace(state, guided_session=transitioned), transitioned, catalog=catalog)
    )
    prepared_next: PreparedGuidedJsonPayload | None = None
    if next_turn is not None:
        next_turn = _finalize_guided_turn(next_turn, shield_available=shield_available)
        transitioned, _record, _turn_type, payload_id = _append_server_turn_record(
            transitioned,
            current_step=transitioned.step,
            turn=next_turn,
        )
        prepared_next = PreparedGuidedJsonPayload(payload_id=payload_id, purpose="turn", payload=next_turn["payload"])
    return (
        _replace(state, guided_session=transitioned),
        PreparedGuidedJsonPayload(payload_id=response_id, purpose="turn_response", payload=response_payload),
        next_turn,
        prepared_next,
    )


@router.post("/{session_id}/guided/respond", response_model=GuidedRespondResponse)
async def post_guided_respond(
    session_id: UUID,
    body: GuidedRespondRequest,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> GuidedRespondResponse:
    """Settle one schema-8 guided response as a fenced atomic cohort."""
    await _verify_session_ownership(session_id, user, request)
    if body.proposal_id is not None:
        try:
            parsed_proposal = UUID(body.proposal_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="proposal_id must be a canonical UUID") from exc
        if str(parsed_proposal) != body.proposal_id:
            raise HTTPException(status_code=400, detail="proposal_id must be a canonical UUID")
    if body.draft_hash is not None and (len(body.draft_hash) != 64 or any(char not in "0123456789abcdef" for char in body.draft_hash)):
        raise HTTPException(status_code=400, detail="draft_hash must be 64 lowercase hexadecimal characters")
    if body.edit_target is not None:
        try:
            parsed_target = UUID(body.edit_target.stable_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="edit_target.stable_id must be a canonical UUID") from exc
        if str(parsed_target) != body.edit_target.stable_id:
            raise HTTPException(status_code=400, detail="edit_target.stable_id must be a canonical UUID")

    from elspeth.core.canonical import stable_hash as _message_content_hash
    from elspeth.web.composer.guided.planning import (
        build_guided_proposal_projection,
        guided_private_reviewed_facts,
        verified_remaining_deferred_intents,
    )
    from elspeth.web.composer.guided.protocol import PROPOSAL_RATIONALE_TEMPLATE, PROPOSAL_SUMMARY_TEMPLATE
    from elspeth.web.composer.guided.state_machine import GuidedProposalRef
    from elspeth.web.composer.pipeline_commit import (
        PipelineCommitConfig,
        PreparedPipelineCommit,
        prepare_pipeline_proposal_commit,
    )
    from elspeth.web.composer.pipeline_planner import PlannerOriginatingMessage
    from elspeth.web.composer.pipeline_proposal import PresentBase
    from elspeth.web.composer.redaction import redact_tool_call_arguments
    from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry
    from elspeth.web.sessions.protocol import (
        GuidedCompositionStateResult,
        GuidedOperationFailureCode,
        GuidedOperationFailureCommand,
        GuidedOperationFenceLostError,
        GuidedOperationSettlementConflictError,
        GuidedPipelineProposalAcceptCommand,
        GuidedPipelineProposalBackEditCommand,
        GuidedPipelineProposalRejectCommand,
        GuidedPipelineProposalStageCommand,
    )

    from ..guided_operations import (
        GuidedOperationExpired,
        GuidedOperationLease,
        raise_guided_operation_failure,
        reserve_or_replay_guided_operation,
    )

    service: SessionServiceProtocol = request.app.state.session_service
    composer = request.app.state.composer_service
    payload_store = request.app.state.payload_store
    compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))

    def _response_from_record(record: CompositionStateRecord) -> GuidedRespondResponse:
        descriptor = parse_guided_response_descriptor(record)
        if descriptor.kind != "guided_respond":
            raise AuditIntegrityError("Guided RESPOND result has the wrong replay descriptor")
        payloads: tuple[PreparedGuidedJsonPayload, ...] = ()
        if descriptor.next_turn is not None:
            payloads = (load_guided_json_payload(payload_store, payload_id=descriptor.next_turn.payload_id, purpose="turn"),)
        response = project_guided_response(record, payloads=payloads)
        if type(response) is not GuidedRespondResponse:
            raise AuditIntegrityError("Guided RESPOND projection returned the wrong response type")
        return response

    async def _replay(result: object) -> GuidedRespondResponse:
        if type(result) is not GuidedCompositionStateResult:
            raise AuditIntegrityError("Guided RESPOND replay has a non-state result locator")
        return _response_from_record(await service.get_state_in_session(result.state_id, session_id))

    def _require_bound_revision_target(current_turn: Turn, *, public_error: bool) -> None:
        """Require the exact stable target advertised by the pending proposal."""

        if body.edit_target is None:
            raise AuditIntegrityError("guided proposal revision is missing its target")
        raw_targets = current_turn["payload"].get("edit_targets")
        requested = {
            "kind": body.edit_target.kind,
            "stable_id": body.edit_target.stable_id,
        }
        if type(raw_targets) is not list or sum(target == requested for target in raw_targets) != 1:
            if public_error:
                raise HTTPException(status_code=409, detail="edit_target does not identify a current proposal component")
            raise AuditIntegrityError("guided proposal revision target changed after reservation")

    async def _preflight_attempt(attempt_stable_id: UUID) -> SourceInspectionFacts | None:
        observed = await service.get_current_state(session_id)
        observed_state = _state_from_record(observed) if observed is not None else _initial_composition_state_with_guided_session()
        observed_guided = observed_state.guided_session
        if observed_guided is None:
            raise HTTPException(status_code=400, detail="Session is not in guided mode. Use /api/sessions/{id}/messages.")
        if observed_guided.terminal is not None:
            if not (
                observed_guided.terminal.kind is TerminalKind.COMPLETED
                and body.turn_token is None
                and body.control_signal == ControlSignal.EXIT_TO_FREEFORM.value
            ):
                raise HTTPException(status_code=409, detail="Guided session is already terminal.")
            return None
        _verify_schema8_proposal_binding(observed_guided, body)
        is_active_exit = body.control_signal == ControlSignal.EXIT_TO_FREEFORM.value
        if not is_active_exit and observed_guided.step is GuidedStep.STEP_3_TRANSFORMS:
            prospective, current_turn, _prepared_current = _schema8_prospective_occurrence(
                observed_state,
                observed_guided,
                catalog=catalog,
                shield_available=shield_available,
                payload_store=payload_store,
            )
            if body.turn_token != guided_turn_token(prospective):
                raise HTTPException(status_code=409, detail="turn_token does not identify the current unanswered turn.")
            if current_turn["type"] != TurnType.PROPOSE_PIPELINE.value:
                raise AuditIntegrityError("guided Step 3 active turn is not a pipeline proposal")
            is_accept = (
                body.chosen == ["accept"]
                and body.control_signal is None
                and body.edited_values is None
                and body.custom_inputs is None
                and body.edit_target is None
            )
            is_reject = (
                body.control_signal == "reject"
                and body.chosen is None
                and body.edited_values is None
                and body.custom_inputs is None
                and body.edit_target is None
            )
            is_revise = (
                body.edit_target is not None
                and body.edited_values is None
                and body.chosen is None
                and body.custom_inputs is None
                and body.control_signal is None
            )
            if sum((is_accept, is_reject, is_revise)) != 1:
                raise HTTPException(status_code=400, detail="Guided proposal action has an invalid closed shape.")
            if is_revise:
                _require_bound_revision_target(current_turn, public_error=True)
            return None
        if not is_active_exit and observed_guided.step not in {GuidedStep.STEP_1_SOURCE, GuidedStep.STEP_2_SINK}:
            raise _schema8_unsupported_stage(observed_guided.step)
        prospective, current_turn, _prepared_current = _schema8_prospective_occurrence(
            observed_state,
            observed_guided,
            catalog=catalog,
            shield_available=shield_available,
            payload_store=payload_store,
        )
        if body.turn_token != guided_turn_token(prospective):
            raise HTTPException(status_code=409, detail="turn_token does not identify the current unanswered turn.")
        inspection_facts: SourceInspectionFacts | None = None
        if observed_guided.step is GuidedStep.STEP_1_SOURCE:
            if current_turn["type"] == TurnType.SINGLE_SELECT.value:
                inspection_facts = await _inspect_latest_ready_session_blob(request.app.state.blob_service, session_id)
            elif current_turn["type"] == TurnType.SCHEMA_FORM.value:
                inspection_facts = await _schema8_active_source_edit_inspection(
                    request.app.state.blob_service,
                    session_id,
                    observed_guided,
                )
        try:
            _schema8_answer_and_project_next(
                observed_state,
                prospective,
                current_turn,
                body,
                catalog=catalog,
                shield_available=shield_available,
                new_stable_id=attempt_stable_id,
                source_inspection_facts=inspection_facts,
            )
        except (PluginConfigError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Guided response does not satisfy the current turn contract.",
            ) from exc
        return inspection_facts

    async def _preflight_or_sanitize(attempt_stable_id: UUID) -> SourceInspectionFacts | None:
        try:
            return await _preflight_attempt(attempt_stable_id)
        except InvariantError as exc:
            with contextlib.suppress(Exception):
                slog.error(
                    "guided.invariant_violated",
                    session_id=str(session_id),
                    user_id=user.user_id,
                    exc_class=type(exc).__name__,
                    site="post_guided_respond.preflight",
                    frames=_safe_frame_strings(exc),
                )
            raise HTTPException(
                status_code=500,
                detail="Server invariant violated. See application audit log for diagnostic detail.",
            ) from exc

    pending = await reserve_or_replay_guided_operation(
        service=service,
        session_id=session_id,
        kind="guided_respond",
        request=body,
        replay=_replay,
        reserve_if_absent=False,
        takeover_expired=False,
    )
    if pending is not None and not isinstance(pending, (GuidedOperationLease, GuidedOperationExpired)):
        return pending

    # Mutable policy/catalog state is consulted only after immutable operation
    # replay had the chance to return its committed projection.
    catalog, plugin_snapshot = _request_plugin_policy_context(request, user)
    shield_available = _resolve_shield_available(plugin_snapshot)

    admission_lock = await _get_session_compose_lock_registry(request).get_lock(f"{session_id}:guided-respond-admission")
    while True:
        rejoin_after_lock = False
        attempt_stable_id = uuid4()
        attempt_inspection_facts: SourceInspectionFacts | None = None
        bypass_admission = isinstance(pending, GuidedOperationExpired)
        if bypass_admission:
            # An expired same-operation retry must preflight before takeover,
            # but cannot queue behind the stale local worker it is fencing out.
            # This is a read plus a discarded pure transition. The fenced
            # settlement rechecks the exact head under compose before writing.
            attempt_inspection_facts = await _preflight_or_sanitize(attempt_stable_id)
            pending = await reserve_or_replay_guided_operation(
                service=service,
                session_id=session_id,
                kind="guided_respond",
                request=body,
                replay=_replay,
            )
            if pending is None:  # pragma: no cover
                raise AuditIntegrityError("Guided RESPOND takeover was not reserved")
            if not isinstance(pending, GuidedOperationLease):
                return pending

        attempt_guard = contextlib.nullcontext() if bypass_admission else admission_lock
        async with attempt_guard:
            # A duplicate may have settled while this request waited for local
            # admission. Rejoin outside the state lock: the helper may poll an
            # active owner and must never block that owner's compose work.
            if pending is None:
                rechecked = await reserve_or_replay_guided_operation(
                    service=service,
                    session_id=session_id,
                    kind="guided_respond",
                    request=body,
                    replay=_replay,
                    reserve_if_absent=False,
                    takeover_expired=False,
                )
                if isinstance(rechecked, GuidedOperationExpired):
                    pending = rechecked
                    continue
                if rechecked is not None and not isinstance(rechecked, GuidedOperationLease):
                    return rechecked
                pending = rechecked

            if not bypass_admission:
                # Ordinary operations remain locally ordered through full
                # preflight and settlement, so stale competing ids never mint
                # a loser operation row.
                async with compose_lock:
                    attempt_inspection_facts = await _preflight_or_sanitize(attempt_stable_id)

            # Reservation and any active-operation joining happen with the
            # compose lock released. The admission lock keeps local competing
            # operation IDs ordered so stale losers fail preflight without a
            # retry artefact.
            reserved = pending or await reserve_or_replay_guided_operation(
                service=service,
                session_id=session_id,
                kind="guided_respond",
                request=body,
                replay=_replay,
            )
            pending = None
            if reserved is None:  # pragma: no cover
                raise AuditIntegrityError("Guided RESPOND operation was not reserved")
            if isinstance(reserved, GuidedOperationExpired):  # pragma: no cover
                raise AuditIntegrityError("Guided RESPOND expired marker reached settlement")
            if not isinstance(reserved, GuidedOperationLease):
                return reserved

            recorder = BufferingRecorder()
            planner_recorder = BufferingRecorder()
            try:
                async with compose_lock:
                    fence = await service.renew_guided_operation(reserved.fence, actor="composer_route", lease_seconds=300)
                    state_record = await service.get_current_state(session_id)
                    state = (
                        _state_from_record(state_record) if state_record is not None else _initial_composition_state_with_guided_session()
                    )
                    guided = state.guided_session
                    if guided is None:
                        raise AuditIntegrityError("Guided RESPOND head lost its guided checkpoint")
                    prepared_payloads: list[PreparedGuidedJsonPayload] = []
                    next_turn: Turn | None = None
                    prepared_next: PreparedGuidedJsonPayload | None = None
                    existing_meta = dict(deep_thaw(state_record.composer_meta)) if state_record and state_record.composer_meta else {}

                    if guided.terminal is not None:
                        if not (
                            guided.terminal.kind is TerminalKind.COMPLETED
                            and body.turn_token is None
                            and body.control_signal == ControlSignal.EXIT_TO_FREEFORM.value
                        ):
                            raise AuditIntegrityError("Guided RESPOND terminal changed after reservation")
                        guided = _replace(
                            guided,
                            terminal=TerminalState(
                                kind=TerminalKind.EXITED_TO_FREEFORM,
                                reason=TerminalReason.USER_PRESSED_EXIT,
                                pipeline_yaml=None,
                            ),
                            transition_consumed=True,
                        )
                        emit_dropped_to_freeform(
                            recorder,
                            prev=guided.step,
                            drop_reason=TerminalReason.USER_PRESSED_EXIT,
                            composition_version=state.version,
                            actor=user.user_id,
                        )
                        existing_meta[_COMPLETED_TERMINAL_BEFORE_EXIT_META_KEY] = {
                            "composition_hash": composition_content_hash(state),
                        }
                        new_state = _replace(state, guided_session=guided)
                    elif guided.step is GuidedStep.STEP_3_TRANSFORMS:
                        if state_record is None or guided.active_proposal is None:
                            raise AuditIntegrityError("guided proposal action requires a persisted active proposal")
                        prospective, current_turn, _planned_current = _schema8_prospective_occurrence(
                            state,
                            guided,
                            catalog=catalog,
                            shield_available=shield_available,
                            payload_store=payload_store,
                        )
                        if body.turn_token != guided_turn_token(prospective):
                            raise AuditIntegrityError("guided proposal turn custody changed after reservation")
                        reviewed_facts = guided_private_reviewed_facts(guided)
                        authority = await service.get_authoritative_pipeline_proposal(
                            session_id=session_id,
                            proposal_id=guided.active_proposal.proposal_id,
                            reviewed_facts=reviewed_facts,
                        )
                        if (
                            authority.proposal.draft_hash != guided.active_proposal.draft_hash
                            or body.proposal_id != str(guided.active_proposal.proposal_id)
                            or body.draft_hash != guided.active_proposal.draft_hash
                        ):
                            raise AuditIntegrityError("guided proposal action authority changed after reservation")

                        if body.control_signal == "reject":
                            rejected = await service.reject_guided_pipeline_proposal(
                                GuidedPipelineProposalRejectCommand(
                                    fence=fence,
                                    expected_current_state_id=state_record.id,
                                    expected_current_state_version=state_record.version,
                                    proposal_id=guided.active_proposal.proposal_id,
                                    draft_hash=guided.active_proposal.draft_hash,
                                    reviewed_facts=reviewed_facts,
                                    actor="composer_route",
                                    response=GuidedResponseDescriptor(
                                        kind="guided_respond",
                                        next_turn=None,
                                        assistant_turn_seq=None,
                                    ),
                                )
                            )
                            return _response_from_record(rejected.result_state)

                        if body.edit_target is not None:
                            _require_bound_revision_target(current_turn, public_error=False)
                            target = {
                                "kind": body.edit_target.kind,
                                "stable_id": body.edit_target.stable_id,
                            }
                            response_payload = {
                                "action": "revise",
                                "proposal_id": str(authority.row.id),
                                "draft_hash": authority.proposal.draft_hash,
                                "edit_target": target,
                            }
                            prepared_response = prepare_guided_json_payload(
                                payload_store,
                                purpose="turn_response",
                                payload=response_payload,
                            )
                            answered = _replace(
                                guided.history[-1],
                                response_hash=prepared_response.payload_id,
                                summary="Guided pipeline proposal revision requested.",
                            )
                            if body.edit_target.kind in {"source", "output"}:
                                component_target = ComponentTarget(
                                    kind=body.edit_target.kind,
                                    stable_id=body.edit_target.stable_id,
                                )
                                target_step = GuidedStep.STEP_1_SOURCE if body.edit_target.kind == "source" else GuidedStep.STEP_2_SINK
                                rewound_guided = _replace(
                                    guided,
                                    step=target_step,
                                    history=(*guided.history[:-1], answered),
                                    active_proposal=None,
                                    active_edit_target=component_target,
                                )
                                rewound_state = _replace(state, guided_session=rewound_guided)
                                edit_turn = _build_get_guided_turn(rewound_state, rewound_guided, catalog=catalog)
                                if edit_turn is None:
                                    raise AuditIntegrityError("guided proposal component back-edit did not produce an edit form")
                                edit_turn = _finalize_guided_turn(edit_turn, shield_available=shield_available)
                                rewound_guided, _edit_record, edit_turn_type, prepared_edit = _prepare_server_turn_occurrence(
                                    rewound_guided,
                                    current_step=target_step,
                                    turn=edit_turn,
                                    payload_store=payload_store,
                                )
                                if edit_turn_type is not TurnType.SCHEMA_FORM:
                                    raise AuditIntegrityError("guided proposal component back-edit must produce a schema form")
                                rewound_state = _replace(state, guided_session=rewound_guided)
                                state_dict = rewound_state.to_dict()
                                rewind_state_data = CompositionStateData(
                                    sources=state_dict["sources"],
                                    nodes=state_dict["nodes"],
                                    edges=state_dict["edges"],
                                    outputs=state_dict["outputs"],
                                    metadata_=state_dict["metadata"],
                                    is_valid=state_record.is_valid,
                                    validation_errors=state_record.validation_errors,
                                    composer_meta={"guided_session": rewound_guided.to_dict()},
                                )
                                emit_turn_answered(
                                    recorder,
                                    step=GuidedStep.STEP_3_TRANSFORMS,
                                    turn_type=TurnType.PROPOSE_PIPELINE,
                                    response_hash=prepared_response.payload_id,
                                    response_payload_id=prepared_response.payload_id,
                                    control_signal=None,
                                    composition_version=state.version,
                                    actor=user.user_id,
                                )
                                emit_turn_emitted(
                                    recorder,
                                    step=target_step,
                                    turn_type=TurnType.SCHEMA_FORM,
                                    payload_hash=prepared_edit.payload_id,
                                    payload_payload_id=prepared_edit.payload_id,
                                    emitter="server",
                                    composition_version=state.version,
                                    actor=user.user_id,
                                )
                                rewind_response = GuidedResponseDescriptor(
                                    kind="guided_respond",
                                    next_turn=GuidedReplayTurn(
                                        turn_type=TurnType.SCHEMA_FORM,
                                        step_index=0 if body.edit_target.kind == "source" else 1,
                                        payload_id=prepared_edit.payload_id,
                                    ),
                                    assistant_turn_seq=None,
                                )
                                rewound = await service.back_edit_guided_pipeline_proposal(
                                    GuidedPipelineProposalBackEditCommand(
                                        fence=fence,
                                        expected_current_state_id=state_record.id,
                                        expected_current_state_version=state_record.version,
                                        expected_current_content_hash=composition_content_hash(state),
                                        proposal_id=guided.active_proposal.proposal_id,
                                        draft_hash=guided.active_proposal.draft_hash,
                                        reviewed_facts=reviewed_facts,
                                        edit_target=component_target,
                                        state=rewind_state_data,
                                        actor="composer_route",
                                        response=rewind_response,
                                        payloads=(prepared_response, prepared_edit),
                                        audit_evidence=GuidedAuditEvidence(invocations=recorder.invocations),
                                    ),
                                    payload_store=payload_store,
                                )
                                return _response_from_record(rewound.result_state)

                            planning_guided = _replace(
                                guided,
                                history=(*guided.history[:-1], answered),
                                active_proposal=None,
                                active_edit_target=None,
                            )

                            expected_originating_message_id = (
                                UUID(planning_guided.root_intent_message_id) if planning_guided.root_intent_message_id is not None else None
                            )
                            if authority.row.user_message_id != expected_originating_message_id:
                                raise AuditIntegrityError("guided proposal revision user-message lineage drifted")
                            message_ids = {
                                *(intent.originating_message_id for intent in planning_guided.deferred_intents),
                            }
                            messages_by_id: dict[str, Any] = {}
                            if message_ids:
                                for message in await service.get_messages(session_id, limit=None):
                                    if str(message.id) in message_ids:
                                        messages_by_id[str(message.id)] = message
                                if set(messages_by_id) != message_ids:
                                    raise AuditIntegrityError("guided planner lineage message is missing or cross-session")
                                if any(message.role != "user" for message in messages_by_id.values()):
                                    raise AuditIntegrityError("guided planner lineage must identify user messages")
                                for deferred in planning_guided.deferred_intents:
                                    if (
                                        _message_content_hash(messages_by_id[deferred.originating_message_id].content)
                                        != deferred.message_content_hash
                                    ):
                                        raise AuditIntegrityError("guided deferred intent message content hash mismatch")

                            root_message = (
                                await service.get_verified_guided_root_intent(
                                    session_id=session_id,
                                    root_message_id=UUID(planning_guided.root_intent_message_id),
                                )
                                if planning_guided.root_intent_message_id is not None
                                else None
                            )
                            revision_intents = {
                                "source": "Regenerate the complete pipeline while revising the selected source component.",
                                "node": "Regenerate the complete pipeline while revising the selected node component.",
                                "edge": "Regenerate the complete pipeline while revising the selected edge component.",
                                "output": "Regenerate the complete pipeline while revising the selected output component.",
                            }
                            planner_intent = revision_intents[body.edit_target.kind]
                            originating_message = PlannerOriginatingMessage(
                                session_id=str(session_id),
                                message_id=str(root_message.id) if root_message is not None else None,
                                content=root_message.content if root_message is not None else planner_intent,
                                user_id=user.user_id,
                            )
                            checkpoint_id = uuid4()
                            successor_proposal_id = uuid4()
                            predecessor_hash = composition_content_hash(state)
                            plan, catalog_ids = await composer.plan_guided_pipeline(
                                intent=planner_intent,
                                current_state=state,
                                guided=planning_guided,
                                originating_message=originating_message,
                                base=PresentBase(
                                    state_id=checkpoint_id,
                                    composition_content_hash=predecessor_hash,
                                ),
                                user_id=user.user_id,
                                supersedes_draft_hash=authority.proposal.draft_hash,
                                recorder=planner_recorder,
                                operation_fence=fence,
                            )
                            projection = build_guided_proposal_projection(
                                proposal_id=successor_proposal_id,
                                proposal=plan.proposal,
                                guided=planning_guided,
                                catalog_plugin_ids=catalog_ids,
                            )
                            proposal_turn = Turn(
                                type=TurnType.PROPOSE_PIPELINE.value,
                                step_index=2,
                                payload=projection,
                            )
                            successor_guided, _proposal_record, _proposal_turn_type, prepared_proposal = _prepare_server_turn_occurrence(
                                planning_guided,
                                current_step=GuidedStep.STEP_3_TRANSFORMS,
                                turn=proposal_turn,
                                payload_store=payload_store,
                            )
                            successor_guided = _replace(
                                successor_guided,
                                active_proposal=GuidedProposalRef(
                                    proposal_id=successor_proposal_id,
                                    draft_hash=plan.proposal.draft_hash,
                                    base=plan.proposal.base,
                                    reviewed_anchor_hash=plan.proposal.reviewed_anchor_hash,
                                    covered_deferred_intent_ids=plan.proposal.covered_deferred_intent_ids,
                                    creation_event_schema="pipeline_proposal_created.v1",
                                    supersedes_proposal_id=authority.row.id,
                                    supersedes_draft_hash=authority.proposal.draft_hash,
                                ),
                            )
                            successor_state = _replace(state, guided_session=successor_guided)
                            state_dict = successor_state.to_dict()
                            is_valid, validation_errors = _guided_persisted_validity(successor_state, catalog=catalog)
                            stage_state = CompositionStateData(
                                sources=state_dict["sources"],
                                nodes=state_dict["nodes"],
                                edges=state_dict["edges"],
                                outputs=state_dict["outputs"],
                                metadata_=state_dict["metadata"],
                                is_valid=is_valid,
                                validation_errors=validation_errors,
                                composer_meta={"guided_session": successor_guided.to_dict()},
                            )
                            emit_turn_answered(
                                recorder,
                                step=GuidedStep.STEP_3_TRANSFORMS,
                                turn_type=TurnType.PROPOSE_PIPELINE,
                                response_hash=prepared_response.payload_id,
                                response_payload_id=prepared_response.payload_id,
                                control_signal=None,
                                composition_version=state.version,
                                actor=user.user_id,
                            )
                            emit_turn_emitted(
                                recorder,
                                step=GuidedStep.STEP_3_TRANSFORMS,
                                turn_type=TurnType.PROPOSE_PIPELINE,
                                payload_hash=prepared_proposal.payload_id,
                                payload_payload_id=prepared_proposal.payload_id,
                                emitter="server",
                                composition_version=state.version,
                                actor=user.user_id,
                            )
                            stage_response = GuidedResponseDescriptor(
                                kind="guided_respond",
                                next_turn=GuidedReplayTurn(
                                    turn_type=TurnType.PROPOSE_PIPELINE,
                                    step_index=2,
                                    payload_id=prepared_proposal.payload_id,
                                ),
                                assistant_turn_seq=None,
                            )
                            stage_settlement = await _await_guided_atomic_settlement(
                                service.stage_guided_pipeline_proposal(
                                    GuidedPipelineProposalStageCommand(
                                        fence=fence,
                                        expected_current_state_id=state_record.id,
                                        expected_current_state_version=state_record.version,
                                        expected_current_content_hash=predecessor_hash,
                                        checkpoint_state_id=checkpoint_id,
                                        proposal_id=successor_proposal_id,
                                        state=stage_state,
                                        plan=plan,
                                        summary=PROPOSAL_SUMMARY_TEMPLATE,
                                        rationale=PROPOSAL_RATIONALE_TEMPLATE,
                                        affects=("pipeline",),
                                        arguments_redacted_json=redact_tool_call_arguments(
                                            "set_pipeline",
                                            deep_thaw(plan.proposal.pipeline),
                                            telemetry=NoopRedactionTelemetry(),
                                        ),
                                        catalog_plugin_ids=catalog_ids,
                                        actor="composer_route",
                                        user_message_id=root_message.id if root_message is not None else None,
                                        user_message_content_hash=(
                                            _message_content_hash(root_message.content) if root_message is not None else None
                                        ),
                                        supersedes_proposal_id=authority.row.id,
                                        response=stage_response,
                                        payloads=(prepared_response, prepared_proposal),
                                        audit_evidence=GuidedAuditEvidence(
                                            invocations=(*planner_recorder.invocations, *recorder.invocations),
                                            llm_calls=planner_recorder.llm_calls,
                                        ),
                                    ),
                                    payload_store=payload_store,
                                )
                            )
                            return _response_from_record(stage_settlement.result_state)

                        user_message_content: str | None = None
                        if authority.row.user_message_id is not None:
                            matches = [
                                message
                                for message in await service.get_messages(session_id, limit=None)
                                if message.id == authority.row.user_message_id
                            ]
                            if len(matches) != 1 or matches[0].role != "user":
                                raise AuditIntegrityError("guided proposal user-message lineage is missing")
                            user_message_content = matches[0].content
                        commit_recorder = BufferingRecorder()
                        prepared = await prepare_pipeline_proposal_commit(
                            authority=authority,
                            reviewed_facts=reviewed_facts,
                            current_state=state,
                            current_state_id=state_record.id,
                            policy_catalog=catalog,
                            plugin_snapshot=plugin_snapshot,
                            config=PipelineCommitConfig(
                                data_dir=str(request.app.state.settings.data_dir),
                                session_engine=request.app.state.session_engine,
                                secret_service=request.app.state.scoped_secret_resolver,
                                user_id=user.user_id,
                                user_message_content=user_message_content,
                                max_blob_storage_per_session_bytes=(request.app.state.settings.max_blob_storage_per_session_bytes),
                                runtime_preflight=None,
                                timeout_seconds=request.app.state.settings.composer_timeout_seconds,
                            ),
                            recorder=commit_recorder,
                            actor=f"user:{user.user_id}",
                            settlement_surface="guided",
                        )
                        if type(prepared) is not PreparedPipelineCommit:
                            raise AuditIntegrityError("guided proposal acceptance requires one fresh exact dispatch")

                        remaining = verified_remaining_deferred_intents(
                            guided=guided,
                            proposal=authority.proposal,
                        )
                        response_payload = {
                            "action": "accept",
                            "proposal_id": str(authority.row.id),
                            "draft_hash": authority.proposal.draft_hash,
                        }
                        prepared_response = prepare_guided_json_payload(
                            payload_store,
                            purpose="turn_response",
                            payload=response_payload,
                        )
                        answered = _replace(
                            guided.history[-1],
                            response_hash=prepared_response.payload_id,
                            summary="Guided pipeline proposal accepted.",
                        )
                        accepted_state = _replace(prepared.result.updated_state, guided_session=guided)
                        policy_validation = catalog.validate_composition_state(accepted_state)
                        validation_state = accepted_state if policy_validation.validation.errors else policy_validation.executable_state
                        wire_turn = build_step_4_wire_turn(
                            accepted_state,
                            proposal_id=str(authority.row.id),
                            draft_hash=authority.proposal.draft_hash,
                            catalog=catalog,
                            validation_state=validation_state,
                            validation_summary=policy_validation.validation,
                        )
                        wire_turn = _finalize_guided_turn(wire_turn, shield_available=shield_available)
                        try:
                            wire_type = validate_current_turn(GuidedStep.STEP_4_WIRE, wire_turn)
                        except ValueError as exc:
                            raise AuditIntegrityError("guided accepted proposal produced an invalid wire review") from exc
                        prepared_wire = prepare_guided_json_payload(
                            payload_store,
                            purpose="turn",
                            payload=wire_turn["payload"],
                        )
                        wire_record = TurnRecord(
                            step=GuidedStep.STEP_4_WIRE,
                            turn_type=wire_type,
                            payload_hash=prepared_wire.payload_id,
                            response_hash=None,
                            emitter="server",
                        )
                        final_guided = _replace(
                            guided,
                            step=GuidedStep.STEP_4_WIRE,
                            history=(*guided.history[:-1], answered, wire_record),
                            deferred_intents=remaining,
                            active_proposal=guided.active_proposal,
                            active_edit_target=None,
                        )
                        accepted_state = _replace(accepted_state, guided_session=final_guided)
                        emit_turn_answered(
                            recorder,
                            step=GuidedStep.STEP_3_TRANSFORMS,
                            turn_type=TurnType.PROPOSE_PIPELINE,
                            response_hash=prepared_response.payload_id,
                            response_payload_id=prepared_response.payload_id,
                            control_signal=None,
                            composition_version=state.version,
                            actor=user.user_id,
                        )
                        emit_step_advanced(
                            recorder,
                            prev=GuidedStep.STEP_3_TRANSFORMS,
                            next_=GuidedStep.STEP_4_WIRE,
                            reason="user_advanced",
                            composition_version=state.version,
                            actor=user.user_id,
                        )
                        emit_turn_emitted(
                            recorder,
                            step=GuidedStep.STEP_4_WIRE,
                            turn_type=wire_type,
                            payload_hash=prepared_wire.payload_id,
                            payload_payload_id=prepared_wire.payload_id,
                            emitter="server",
                            composition_version=state.version,
                            actor=user.user_id,
                        )
                        from .._helpers import _state_data_from_composer_state

                        state_data, _validation = await _state_data_from_composer_state(
                            accepted_state,
                            settings=request.app.state.settings,
                            secret_service=request.app.state.scoped_secret_resolver,
                            user_id=user.user_id,
                            session_id=session_id,
                            plugin_snapshot=plugin_snapshot,
                            profile_registry=request.app.state.operator_profile_registry,
                            catalog=request.app.state.catalog_service,
                            runtime_preflight=prepared.result.runtime_preflight,
                            preflight_exception_policy="raise",
                            initial_version=state.version,
                            telemetry_source="convergence",
                            composer_meta={"guided_session": final_guided.to_dict()},
                        )
                        accepted = await service.accept_guided_pipeline_proposal(
                            GuidedPipelineProposalAcceptCommand(
                                fence=fence,
                                expected_current_state_id=state_record.id,
                                expected_current_state_version=state_record.version,
                                proposal_id=authority.row.id,
                                draft_hash=authority.proposal.draft_hash,
                                reviewed_facts=reviewed_facts,
                                state=state_data,
                                candidate_content_hash=prepared.candidate_content_hash,
                                executor_content_hash=prepared.executor_content_hash,
                                invocation=prepared.invocation,
                                actor="composer_route",
                                response=GuidedResponseDescriptor(
                                    kind="guided_respond",
                                    next_turn=GuidedReplayTurn(
                                        turn_type=TurnType(wire_turn["type"]),
                                        step_index=wire_turn["step_index"],
                                        payload_id=prepared_wire.payload_id,
                                    ),
                                    assistant_turn_seq=None,
                                ),
                                payloads=(prepared_response, prepared_wire),
                                audit_evidence=GuidedAuditEvidence(invocations=recorder.invocations),
                            ),
                            payload_store=payload_store,
                        )
                        return _response_from_record(accepted.result_state)
                    else:
                        occurrence_was_prospective = not (guided.history and guided.history[-1].response_hash is None)
                        prospective, current_turn, planned_current = _schema8_prospective_occurrence(
                            state,
                            guided,
                            catalog=catalog,
                            shield_available=shield_available,
                            payload_store=payload_store,
                        )
                        if body.turn_token != guided_turn_token(prospective):
                            raise AuditIntegrityError("Guided RESPOND turn custody changed after reservation")
                        prior_step = prospective.step
                        try:
                            new_state, planned_response, next_turn, prepared_next = _schema8_answer_and_project_next(
                                state,
                                prospective,
                                current_turn,
                                body,
                                catalog=catalog,
                                shield_available=shield_available,
                                new_stable_id=attempt_stable_id,
                                source_inspection_facts=attempt_inspection_facts,
                            )
                        except (PluginConfigError, TypeError, ValueError) as exc:
                            raise AuditIntegrityError("Guided RESPOND contract changed after reservation") from exc
                        for planned in (planned_current, planned_response, prepared_next):
                            if planned is None:
                                continue
                            prepared_payload = prepare_guided_json_payload(
                                payload_store,
                                purpose=planned.purpose,
                                payload=planned.payload,
                            )
                            if prepared_payload.payload_id != planned.payload_id:
                                raise AuditIntegrityError("Guided RESPOND prospective payload changed before settlement")
                            prepared_payloads.append(prepared_payload)
                        if occurrence_was_prospective:
                            emit_turn_emitted(
                                recorder,
                                step=prior_step,
                                turn_type=TurnType(current_turn["type"]),
                                payload_hash=planned_current.payload_id,
                                payload_payload_id=planned_current.payload_id,
                                emitter="server",
                                composition_version=state.version,
                                actor=user.user_id,
                            )
                        emit_turn_answered(
                            recorder,
                            step=prior_step,
                            turn_type=TurnType(current_turn["type"]),
                            response_hash=planned_response.payload_id,
                            response_payload_id=planned_response.payload_id,
                            control_signal=body.control_signal,
                            composition_version=state.version,
                            actor=user.user_id,
                        )
                        resulting_guided = new_state.guided_session
                        if resulting_guided is None:  # pragma: no cover
                            raise AuditIntegrityError("Guided RESPOND transition removed its checkpoint")
                        if prior_step is not resulting_guided.step:
                            emit_step_advanced(
                                recorder,
                                prev=prior_step,
                                next_=resulting_guided.step,
                                reason="user_advanced",
                                composition_version=state.version,
                                actor=user.user_id,
                            )
                        if resulting_guided.terminal is not None:
                            emit_dropped_to_freeform(
                                recorder,
                                prev=prior_step,
                                drop_reason=TerminalReason.USER_PRESSED_EXIT,
                                composition_version=state.version,
                                actor=user.user_id,
                            )
                        if prepared_next is not None and next_turn is not None:
                            emit_turn_emitted(
                                recorder,
                                step=resulting_guided.step,
                                turn_type=TurnType(next_turn["type"]),
                                payload_hash=prepared_next.payload_id,
                                payload_payload_id=prepared_next.payload_id,
                                emitter="server",
                                composition_version=state.version,
                                actor=user.user_id,
                            )

                        if (
                            prior_step is GuidedStep.STEP_2_SINK
                            and resulting_guided.step is GuidedStep.STEP_3_TRANSFORMS
                            and resulting_guided.terminal is None
                        ):
                            if state_record is None:
                                raise AuditIntegrityError("guided proposal staging requires a persisted predecessor")
                            if resulting_guided.active_proposal is not None:
                                raise AuditIntegrityError("guided proposal transition already has an active proposal")
                            checkpoint_id = uuid4()
                            proposal_id = uuid4()
                            predecessor_hash = composition_content_hash(state)
                            if composition_content_hash(new_state) != predecessor_hash:
                                raise AuditIntegrityError("guided topology planning transition changed authored composition content")

                            message_ids = {
                                *(intent.originating_message_id for intent in resulting_guided.deferred_intents),
                            }
                            planner_messages_by_id: dict[str, Any] = {}
                            if message_ids:
                                for message in await service.get_messages(session_id, limit=None):
                                    if str(message.id) in message_ids:
                                        planner_messages_by_id[str(message.id)] = message
                                if set(planner_messages_by_id) != message_ids:
                                    raise AuditIntegrityError("guided planner lineage message is missing or cross-session")
                                if any(message.role != "user" for message in planner_messages_by_id.values()):
                                    raise AuditIntegrityError("guided planner lineage must identify user messages")
                                for deferred in resulting_guided.deferred_intents:
                                    if (
                                        _message_content_hash(planner_messages_by_id[deferred.originating_message_id].content)
                                        != deferred.message_content_hash
                                    ):
                                        raise AuditIntegrityError("guided deferred intent message content hash mismatch")

                            root_message = (
                                await service.get_verified_guided_root_intent(
                                    session_id=session_id,
                                    root_message_id=UUID(resulting_guided.root_intent_message_id),
                                )
                                if resulting_guided.root_intent_message_id is not None
                                else None
                            )
                            planner_intent = (
                                root_message.content
                                if root_message is not None
                                else "Build the complete pipeline from the reviewed guided components and deferred constraints."
                            )
                            originating_message = PlannerOriginatingMessage(
                                session_id=str(session_id),
                                message_id=str(root_message.id) if root_message is not None else None,
                                content=planner_intent,
                                user_id=user.user_id,
                            )
                            plan, catalog_ids = await composer.plan_guided_pipeline(
                                intent=planner_intent,
                                current_state=state,
                                guided=resulting_guided,
                                originating_message=originating_message,
                                base=PresentBase(
                                    state_id=checkpoint_id,
                                    composition_content_hash=predecessor_hash,
                                ),
                                user_id=user.user_id,
                                supersedes_draft_hash=None,
                                recorder=planner_recorder,
                                operation_fence=fence,
                            )
                            projection = build_guided_proposal_projection(
                                proposal_id=proposal_id,
                                proposal=plan.proposal,
                                guided=resulting_guided,
                                catalog_plugin_ids=catalog_ids,
                            )
                            proposal_turn = Turn(
                                type=TurnType.PROPOSE_PIPELINE.value,
                                step_index=2,
                                payload=projection,
                            )
                            resulting_guided, _proposal_record, _proposal_turn_type, prepared_proposal = _prepare_server_turn_occurrence(
                                resulting_guided,
                                current_step=GuidedStep.STEP_3_TRANSFORMS,
                                turn=proposal_turn,
                                payload_store=payload_store,
                            )
                            resulting_guided = _replace(
                                resulting_guided,
                                active_proposal=GuidedProposalRef(
                                    proposal_id=proposal_id,
                                    draft_hash=plan.proposal.draft_hash,
                                    base=plan.proposal.base,
                                    reviewed_anchor_hash=plan.proposal.reviewed_anchor_hash,
                                    covered_deferred_intent_ids=plan.proposal.covered_deferred_intent_ids,
                                    creation_event_schema="pipeline_proposal_created.v1",
                                ),
                            )
                            new_state = _replace(new_state, guided_session=resulting_guided)
                            prepared_payloads.append(prepared_proposal)
                            emit_turn_emitted(
                                recorder,
                                step=GuidedStep.STEP_3_TRANSFORMS,
                                turn_type=TurnType.PROPOSE_PIPELINE,
                                payload_hash=prepared_proposal.payload_id,
                                payload_payload_id=prepared_proposal.payload_id,
                                emitter="server",
                                composition_version=state.version,
                                actor=user.user_id,
                            )
                            state_dict = new_state.to_dict()
                            is_valid, validation_errors = _guided_persisted_validity(new_state, catalog=catalog)
                            stage_state = CompositionStateData(
                                sources=state_dict["sources"],
                                nodes=state_dict["nodes"],
                                edges=state_dict["edges"],
                                outputs=state_dict["outputs"],
                                metadata_=state_dict["metadata"],
                                is_valid=is_valid,
                                validation_errors=validation_errors,
                                composer_meta={"guided_session": resulting_guided.to_dict()},
                            )
                            stage_response = GuidedResponseDescriptor(
                                kind="guided_respond",
                                next_turn=GuidedReplayTurn(
                                    turn_type=TurnType.PROPOSE_PIPELINE,
                                    step_index=2,
                                    payload_id=prepared_proposal.payload_id,
                                ),
                                assistant_turn_seq=None,
                            )
                            stage_settlement = await _await_guided_atomic_settlement(
                                service.stage_guided_pipeline_proposal(
                                    GuidedPipelineProposalStageCommand(
                                        fence=fence,
                                        expected_current_state_id=state_record.id,
                                        expected_current_state_version=state_record.version,
                                        expected_current_content_hash=predecessor_hash,
                                        checkpoint_state_id=checkpoint_id,
                                        proposal_id=proposal_id,
                                        state=stage_state,
                                        plan=plan,
                                        summary=PROPOSAL_SUMMARY_TEMPLATE,
                                        rationale=PROPOSAL_RATIONALE_TEMPLATE,
                                        affects=("pipeline",),
                                        arguments_redacted_json=redact_tool_call_arguments(
                                            "set_pipeline",
                                            deep_thaw(plan.proposal.pipeline),
                                            telemetry=NoopRedactionTelemetry(),
                                        ),
                                        catalog_plugin_ids=catalog_ids,
                                        actor="composer_route",
                                        user_message_id=root_message.id if root_message is not None else None,
                                        user_message_content_hash=(
                                            _message_content_hash(root_message.content) if root_message is not None else None
                                        ),
                                        supersedes_proposal_id=None,
                                        response=stage_response,
                                        payloads=tuple(prepared_payloads),
                                        audit_evidence=GuidedAuditEvidence(
                                            invocations=(*planner_recorder.invocations, *recorder.invocations),
                                            llm_calls=planner_recorder.llm_calls,
                                        ),
                                    ),
                                    payload_store=payload_store,
                                )
                            )
                            return _response_from_record(stage_settlement.result_state)

                    settlement_guided = new_state.guided_session
                    if settlement_guided is None:  # pragma: no cover
                        raise AuditIntegrityError("Guided RESPOND settlement has no checkpoint")
                    existing_meta["guided_session"] = settlement_guided.to_dict()
                    state_dict = new_state.to_dict()
                    is_valid, validation_errors = _guided_persisted_validity(new_state, catalog=catalog)
                    state_data = CompositionStateData(
                        sources=state_dict["sources"],
                        nodes=state_dict["nodes"],
                        edges=state_dict["edges"],
                        outputs=state_dict["outputs"],
                        metadata_=state_dict["metadata"],
                        is_valid=is_valid,
                        validation_errors=validation_errors,
                        composer_meta=existing_meta,
                    )
                    replay_turn = (
                        GuidedReplayTurn(
                            turn_type=TurnType(next_turn["type"]),
                            step_index=next_turn["step_index"],
                            payload_id=prepared_next.payload_id,
                        )
                        if next_turn is not None and prepared_next is not None
                        else None
                    )
                    settlement_command = GuidedStateOperationCommand(
                        fence=fence,
                        expected_current_state_id=state_record.id if state_record is not None else None,
                        expected_current_state_version=state_record.version if state_record is not None else None,
                        expected_current_content_hash=(composition_content_hash(state) if state_record is not None else None),
                        state_id=uuid4(),
                        state=state_data,
                        provenance="convergence_persist",
                        actor="composer_route",
                        response=GuidedResponseDescriptor(
                            kind="guided_respond",
                            next_turn=replay_turn,
                            assistant_turn_seq=None,
                        ),
                        payloads=tuple(prepared_payloads),
                        audit_evidence=GuidedAuditEvidence(invocations=recorder.invocations),
                    )
                    # CHAT and other current writers do not all carry an
                    # expected-head CAS yet, so settlement remains mutually
                    # exclusive with them under the session compose lock.
                    settlement = await service.settle_guided_state_operation(
                        settlement_command,
                        payload_store=payload_store,
                    )
                    return _response_from_record(settlement.result_state)
            except GuidedOperationFenceLostError:
                rejoin_after_lock = True
            except asyncio.CancelledError as exc:
                exc_dict = exc.__dict__
                if exc_dict.get(_GUIDED_ATOMIC_SETTLEMENT_COMPLETED) is True:
                    raise
                settlement_failure = exc_dict.get(_GUIDED_ATOMIC_SETTLEMENT_FAILURE)
                if isinstance(settlement_failure, GuidedOperationFenceLostError):
                    raise
                caller_task = asyncio.current_task()
                request_cancelled = caller_task is not None and caller_task.cancelling() > 0
                if request_cancelled or "llm_calls" not in exc_dict:
                    cancel_failure_code: GuidedOperationFailureCode = (
                        "stale_conflict"
                        if isinstance(settlement_failure, GuidedOperationSettlementConflictError)
                        else "integrity_error"
                        if isinstance(settlement_failure, (AuditIntegrityError, InvariantError))
                        else "operation_failed"
                        if settlement_failure is not None or not request_cancelled
                        else "request_cancelled"
                    )
                    try:
                        await _await_with_deferred_cancellation(
                            service.fail_guided_operation_with_audit(
                                GuidedOperationFailureCommand(
                                    fence=reserved.fence,
                                    failure_code=cancel_failure_code,
                                    actor="composer_route",
                                    audit_evidence=GuidedAuditEvidence(
                                        invocations=planner_recorder.invocations,
                                        llm_calls=planner_recorder.llm_calls,
                                        chat_turns=planner_recorder.chat_turns,
                                    ),
                                ),
                            )
                        )
                    except GuidedOperationFenceLostError as fence_lost:
                        raise exc from fence_lost
                    except Exception as failure_exc:
                        raise exc from failure_exc
                    if settlement_failure is not None:
                        raise exc from settlement_failure
                    raise
                # Only planner terminal exceptions carry this evidence marker.
                attached_calls = exc_dict["llm_calls"]
                carrier_error: AuditIntegrityError | None = None
                if type(attached_calls) is not tuple or attached_calls != planner_recorder.llm_calls:
                    carrier_error = AuditIntegrityError("guided planner cancellation carried malformed or unrelated LLM audit evidence")
                    attached_calls = planner_recorder.llm_calls
                try:
                    await _await_with_deferred_cancellation(
                        service.fail_guided_operation_with_audit(
                            GuidedOperationFailureCommand(
                                fence=reserved.fence,
                                failure_code="operation_failed",
                                actor="composer_route",
                                audit_evidence=GuidedAuditEvidence(
                                    invocations=planner_recorder.invocations,
                                    llm_calls=attached_calls,
                                    chat_turns=planner_recorder.chat_turns,
                                ),
                            ),
                        )
                    )
                except GuidedOperationFenceLostError as fence_lost:
                    raise exc from fence_lost
                except Exception as failure_exc:
                    raise exc from failure_exc
                if carrier_error is not None:
                    raise exc from carrier_error
                raise
            except Exception as exc:
                failure_code: GuidedOperationFailureCode = (
                    "stale_conflict"
                    if isinstance(exc, GuidedOperationSettlementConflictError)
                    else "integrity_error"
                    if isinstance(exc, (AuditIntegrityError, InvariantError))
                    else "operation_failed"
                )
                with contextlib.suppress(Exception):
                    slog.error(
                        "guided.operation_terminal_failure",
                        session_id=str(session_id),
                        user_id=user.user_id,
                        exc_class=type(exc).__name__,
                        site="post_guided_respond",
                        frames=_safe_frame_strings(exc),
                    )
                try:
                    failed = await service.fail_guided_operation_with_audit(
                        GuidedOperationFailureCommand(
                            fence=reserved.fence,
                            failure_code=failure_code,
                            actor="composer_route",
                            audit_evidence=GuidedAuditEvidence(
                                invocations=planner_recorder.invocations,
                                llm_calls=planner_recorder.llm_calls,
                                chat_turns=planner_recorder.chat_turns,
                            ),
                        )
                    )
                except GuidedOperationFenceLostError:
                    rejoin_after_lock = True
                except Exception as failure_exc:
                    with contextlib.suppress(Exception):
                        slog.error(
                            "guided.operation_failure_record_failed",
                            session_id=str(session_id),
                            user_id=user.user_id,
                            exc_class=type(failure_exc).__name__,
                            site="post_guided_respond.fail_guided_operation",
                            frames=_safe_frame_strings(failure_exc),
                        )
                    raise AuditIntegrityError("Guided RESPOND could not record its terminal failure") from None
                else:
                    raise_guided_operation_failure(failed)

        if rejoin_after_lock:
            joined = await reserve_or_replay_guided_operation(
                service=service,
                session_id=session_id,
                kind="guided_respond",
                request=body,
                replay=_replay,
                reserve_if_absent=False,
            )
            if joined is None:
                raise AuditIntegrityError("Guided RESPOND fence was lost without a joinable winner")
            if isinstance(joined, GuidedOperationLease):
                pending = joined
                continue
            return joined
        raise AuditIntegrityError("Guided RESPOND settlement loop exited without a result")


async def _run_guided_chat_provider_attempt(**kwargs: Any) -> GuidedChatProviderOutcome:
    """Delegate provider work without importing the atomic route at module load."""

    from .guided_chat_atomic import run_guided_chat_provider_attempt

    return await run_guided_chat_provider_attempt(**kwargs)


@router.post("/{session_id}/guided/chat", response_model=GuidedChatResponse)
async def post_guided_chat(
    session_id: UUID,
    body: GuidedChatRequest,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
    _inflight_tally: None = Depends(_track_compose_inflight),
) -> GuidedChatResponse:
    """Settle one current schema-8 Step-1/Step-2 chat operation atomically."""

    await _verify_session_ownership(session_id, user, request)
    from .guided_chat_atomic import post_guided_chat_schema8

    return await post_guided_chat_schema8(
        session_id=session_id,
        body=body,
        request=request,
        user=user,
        provider_runner=_run_guided_chat_provider_attempt,
    )


# PLACEMENT IS LOAD-BEARING (TEMPORARY WORKAROUND — see elspeth-b8ea8a35cb).
# post_guided_convert logically belongs with the other guided routes (next to
# post_guided_start / post_guided_respond), but it is pinned LAST here for now.
# The trust_tier.tier_model raw fingerprint is sha256(rule_id | ast_path | dump)
# and ast_path begins with the enclosing function's module-level body[N] index.
# The route handlers above carry HMAC-SIGNED tier_model suppressions
# (config/cicd/enforce_tier_model/web.yaml) keyed by that fingerprint. Inserting
# a def ABOVE them renumbers their body[N], rotates every downstream fingerprint,
# and breaks 33 operator-held signatures (regen needs ELSPETH_JUDGE_METADATA_HMAC_KEY,
# which agents must never hold). Appending here shifts no existing index, so the
# branch stays green with no re-sign. elspeth-b8ea8a35cb moves it to its logical
# home at merge, where the standard merge re-sign absorbs the rotation. Until
# then: do not move this above another handler.
@router.post("/{session_id}/guided/convert", response_model=GetGuidedResponse)
async def post_guided_convert(
    session_id: UUID,
    body: ConvertGuidedRequest,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> GetGuidedResponse:
    """Move a freeform session into guided mode.

    "Switch to guided" on a session that has already done freeform composition
    work cannot lazily read guided state: its persisted CompositionState carries
    no ``guided_session``, so GET /guided 400s by design — and MUST keep doing
    so, because ``fetchGuidedStateForSelect`` probes GET on every session select
    and reads the 400 as "this session is freeform-only". A mutating GET would
    flip every worked freeform session into the guided surface on load. This
    explicit POST is the conversion; GET stays a pure reader.

    Per the "fresh wizard + consent" product decision (elspeth-e2c3dba6b5) the
    conversion does NOT walk the retained freeform graph through the wizard:
    ``GuidedSession.initial()`` starts at STEP_1_SOURCE and the step handlers
    rebuild source/sink/transform state from scratch, so proceeding over a
    pre-built graph would clobber it. Instead it seeds a FRESH wizard as a NEW
    composition-state version. The prior freeform pipeline stays recoverable via
    GET /state/versions + POST /state/revert (revert copies ``composer_meta``
    verbatim, so restoring the pre-conversion version lands the session back in
    freeform with the graph intact) — the same recoverability contract as YAML
    import. A system message records the switch and names the recoverable
    version.

    Idempotent and safe for every entry state, so "Switch to guided" can route
    through it unconditionally:
      * no persisted state (empty session) -> persist a fresh wizard checkpoint
        so the operation has an immutable replay locator.
      * ``guided_session`` already present -> return it UNCHANGED, including any
        terminal (so a completed / solver-exhausted / protocol-violation surface
        still renders — enterGuided routes those non-exit terminals here).
      * persisted state with ``guided_session is None`` -> the conversion.

    Raises 404 if the session does not exist or belong to the requesting user.
    """
    await _verify_session_ownership(session_id, user, request)
    service: SessionServiceProtocol = request.app.state.session_service
    catalog, plugin_snapshot = _request_plugin_policy_context(request, user)

    from elspeth.contracts.errors import AuditIntegrityError
    from elspeth.web.sessions.protocol import (
        GuidedCompositionStateResult,
        GuidedOperationFailureCode,
        GuidedOperationSettlementConflictError,
    )

    from ..guided_operations import (
        GuidedOperationLease,
        guided_response_hash,
        raise_guided_operation_failure,
        reserve_or_replay_guided_operation,
    )

    def _response_from_record(record: CompositionStateRecord) -> GetGuidedResponse:
        state = _state_from_record(record)
        guided = state.guided_session
        if guided is None:
            raise AuditIntegrityError("Guided conversion result state has no guided checkpoint")
        terminal = guided.terminal
        turn = None
        if terminal is None:
            turn, _prepared = _load_durable_current_turn(
                guided,
                payload_store=request.app.state.payload_store,
            )
        terminal_response = (
            TerminalStateResponse(
                kind=terminal.kind.value,
                reason=terminal.reason.value if terminal.reason is not None else None,
                pipeline_yaml=terminal.pipeline_yaml,
            )
            if terminal is not None
            else None
        )
        return GetGuidedResponse(
            guided_session=GuidedSessionResponse(
                step=guided.step.value,
                history=[
                    TurnRecordResponse(
                        step=turn_record.step.value,
                        turn_type=turn_record.turn_type.value,
                        payload_hash=turn_record.payload_hash,
                        response_hash=turn_record.response_hash,
                        summary=turn_record.summary,
                        emitter=turn_record.emitter,
                    )
                    for turn_record in guided.history
                ],
                terminal=terminal_response,
                chat_history=[
                    ChatTurnResponse(
                        role=chat_turn.role.value,
                        content=chat_turn.content,
                        seq=chat_turn.seq,
                        step=chat_turn.step.value,
                        ts_iso=chat_turn.ts_iso,
                        assistant_message_kind=chat_turn.assistant_message_kind,
                        synthetic_failure_reason=chat_turn.synthetic_failure_reason,
                    )
                    for chat_turn in guided.chat_history
                ],
                chat_turn_seq=guided.chat_turn_seq,
                profile=_workflow_profile_response(guided),
            ),
            next_turn=_turn_payload_response(
                turn,
                guided=guided,
                shield_available=_resolve_shield_available(plugin_snapshot),
            ),
            terminal=terminal_response,
            composition_state=_state_response(record, policy_catalog=catalog),
        )

    async def _replay(result: object) -> GetGuidedResponse:
        if type(result) is not GuidedCompositionStateResult:
            raise AuditIntegrityError("Guided conversion replay has a non-state result locator")
        replay_record = await service.get_state_in_session(result.state_id, session_id)
        return _response_from_record(replay_record)

    reserved = await reserve_or_replay_guided_operation(
        service=service,
        session_id=session_id,
        kind="guided_convert",
        request=body,
        replay=_replay,
    )
    if reserved is None:  # pragma: no cover - reserve_if_absent defaults true
        raise AuditIntegrityError("Guided conversion operation was not reserved")
    if not isinstance(reserved, GuidedOperationLease):
        return reserved

    compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))
    try:
        async with compose_lock:
            state_record = await service.get_current_state(session_id)

            # Branch 2: already guided (idempotent double-click, cross-tab race,
            # or a terminal session reached via enterGuided's non-exit branch).
            # Return the existing session unchanged and settle its exact head.
            if state_record is not None and _state_from_record(state_record).guided_session is not None:
                settled_record = await service.complete_existing_state_guided_operation(
                    reserved.fence,
                    state_id=state_record.id,
                    expected_current_state_id=state_record.id,
                    expected_current_state_version=state_record.version,
                    actor="composer_route",
                    response_hash_factory=lambda record: guided_response_hash(_response_from_record(record)),
                )
                return _response_from_record(settled_record)

            # Branches 1 & 3: seed a fresh guided wizard.
            new_state = _initial_composition_state_with_guided_session()
            seeded_guided = new_state.guided_session
            if seeded_guided is None:  # pragma: no cover — helper always attaches a guided session
                raise InvariantError("post_guided_convert: initial state has no guided_session")
            seed_turn = _build_get_guided_turn(new_state, seeded_guided, catalog=catalog)
            if seed_turn is None:  # pragma: no cover - initial STEP_1 always emits
                raise InvariantError("post_guided_convert: initial guided session has no first turn")
            seed_turn = _finalize_guided_turn(
                seed_turn,
                shield_available=_resolve_shield_available(plugin_snapshot),
            )
            seeded_guided, _record, seed_turn_type, prepared_seed_turn = _prepare_server_turn_occurrence(
                seeded_guided,
                current_step=seeded_guided.step,
                turn=seed_turn,
                payload_store=request.app.state.payload_store,
            )
            seed_evidence = _turn_emission_evidence(
                step=seeded_guided.step,
                turn_type=seed_turn_type,
                prepared=prepared_seed_turn,
                composition_version=new_state.version,
                actor=user.user_id,
            )
            new_state = _replace(new_state, guided_session=seeded_guided)
            new_composer_meta = {"guided_session": seeded_guided.to_dict()}
            state_d = new_state.to_dict()
            persisted_is_valid, persisted_errors = _guided_persisted_validity(new_state, catalog=catalog)
            state_data = CompositionStateData(
                sources=state_d["sources"],
                nodes=state_d["nodes"],
                edges=state_d["edges"],
                outputs=state_d["outputs"],
                metadata_=state_d["metadata"],
                is_valid=persisted_is_valid,
                validation_errors=persisted_errors,
                composer_meta=new_composer_meta,
            )
            system_message = None
            if state_record is not None:
                system_message = (
                    "Switched to guided mode with a fresh wizard. Your previous "
                    f"freeform pipeline is saved as version {state_record.version} and can "
                    "be restored from version history."
                )
            state_record_out = await service.save_state_for_guided_operation(
                reserved.fence,
                expected_current_state_id=state_record.id if state_record is not None else None,
                expected_current_state_version=state_record.version if state_record is not None else None,
                state=state_data,
                provenance="session_seed",
                actor="composer_route",
                response_hash_factory=lambda record: guided_response_hash(_response_from_record(record)),
                system_message=system_message,
                payloads=(prepared_seed_turn,),
                audit_evidence=seed_evidence,
                payload_store=request.app.state.payload_store,
            )
            return _response_from_record(state_record_out)
    except Exception as exc:
        failure_code: GuidedOperationFailureCode = (
            "stale_conflict"
            if isinstance(exc, GuidedOperationSettlementConflictError)
            else "integrity_error"
            if isinstance(exc, AuditIntegrityError)
            else "operation_failed"
        )
        if isinstance(exc, AuditIntegrityError):
            slog.error(
                "guided.operation_terminal_failure",
                session_id=str(session_id),
                user_id=user.user_id,
                exc_class=type(exc).__name__,
                site="post_guided_convert",
                frames=_safe_frame_strings(exc),
            )
        failed = await service.fail_guided_operation(
            reserved.fence,
            failure_code=failure_code,
            actor="composer_route",
        )
        raise_guided_operation_failure(failed)


from .guided_plan import router as guided_plan_router  # noqa: E402

router.include_router(guided_plan_router)
