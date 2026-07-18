from __future__ import annotations

import asyncio  # noqa: F401  # Preserve signed module statement positions.
import json  # noqa: F401  # Preserve signed module statement positions.
from typing import Literal, cast
from uuid import uuid4

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.plugin_capabilities import PluginCapability
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.infrastructure.validation import get_sink_config_model, get_source_config_model
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.guided.chat_solver import (
    build_step_chat_context_block,  # noqa: F401  # Preserve signed module statement positions.
)
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
    transition_sink_field_review,
    transition_sink_plugin_selection,
    transition_sink_schema_form,
    transition_source_inspection_review,
    transition_source_plugin_selection,
    transition_source_schema_form,
)
from elspeth.web.composer.pipeline_proposal import composition_content_hash
from elspeth.web.composer.source_inspection import SourceInspectionFacts
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
    GuidedReplayTurn,
    GuidedResponseDescriptor,
    GuidedStateOperationCommand,
    PreparedGuidedJsonPayload,
    guided_json_payload_id,
)
from elspeth.web.sessions.schemas import ConvertGuidedRequest, ReenterGuidedRequest, StartGuidedRequest, TutorialSampleResponse

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
    StepChatResult,
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
            return build_step_1_schema_form_turn_from_resolved(guided.reviewed_sources[target.stable_id], catalog)
        pending = next(
            (guided.pending_source_intents[stable_id] for stable_id in guided.source_order if stable_id in guided.pending_source_intents),
            None,
        )
        if pending is None:
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
            sink = SinkResolved(outputs=(guided.reviewed_outputs[target.stable_id],))
            return build_step_2_schema_form_turn_from_resolved(sink, catalog)
        pending = next(
            (guided.pending_output_intents[stable_id] for stable_id in guided.output_order if stable_id in guided.pending_output_intents),
            None,
        )
        if pending is None or pending.phase == "plugin_selection":
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
        policy_validation = catalog.validate_composition_state(state)
        validation_state = state if policy_validation.validation.errors else policy_validation.executable_state
        return build_step_4_wire_turn(
            state,
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
    from elspeth.web.sessions.protocol import GuidedCompositionStateResult

    from ..guided_operations import (
        GuidedOperationLease,
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
        return _response_from_record(settlement.result_state)


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

    from elspeth.contracts.errors import AuditIntegrityError
    from elspeth.web.sessions.protocol import (
        GuidedCompositionStateResult,
        GuidedOperationFailureCode,
        GuidedOperationFenceLostError,
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

    async def _replay(result: object) -> GetGuidedResponse:
        if type(result) is not GuidedCompositionStateResult:
            raise AuditIntegrityError("Guided start replay has a non-state result locator")
        replay_record = await service.get_state_in_session(result.state_id, session_id)
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
                        raise AuditIntegrityError("Guided start head is freeform after reservation")
                    # A distinct start may have won after both requests
                    # preflighted an empty head. Settle the exact current guided
                    # checkpoint as an idempotent no-op; the service CAS prevents
                    # it from blessing a stale head.
                    settled_record = await service.complete_existing_state_guided_operation(
                        renewed_fence,
                        state_id=current_record.id,
                        expected_current_state_id=current_record.id,
                        expected_current_state_version=current_record.version,
                        actor="composer_route",
                        response_hash_factory=lambda record: guided_response_hash(_response_from_record(record)),
                    )
                    return _response_from_record(settled_record)

                if observed_head is not None:
                    raise AuditIntegrityError("Guided start head disappeared after preflight")

                new_state = _initial_composition_state_with_guided_session(profile=profile)
                seeded_guided = new_state.guided_session
                if seeded_guided is None:  # pragma: no cover - helper contract
                    raise InvariantError("post_guided_start: initial state has no guided_session")
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
                seed_outcome = await service.seed_or_complete_guided_start_operation(
                    renewed_fence,
                    state=state_data,
                    provenance="session_seed",
                    actor="composer_route",
                    response_hash_factory=lambda record: guided_response_hash(_response_from_record(record)),
                    payloads=(prepared_seed_turn,),
                    audit_evidence=seed_evidence,
                    payload_store=request.app.state.payload_store,
                )
                return _response_from_record(seed_outcome.state)
        except GuidedOperationFenceLostError:
            # Never poll while holding the compose lock. Rejoin outside it;
            # reserve either observes the winner or performs the sole takeover.
            continue
        except Exception as exc:
            failure_code: GuidedOperationFailureCode = "integrity_error" if isinstance(exc, AuditIntegrityError) else "operation_failed"
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
        config = config_model.from_dict(merged, plugin_name=plugin)
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
            target = _schema8_pending_target(guided, source=True, phase="plugin_options")
            held_plugin = guided.pending_source_intents[target].plugin
            if type(held_plugin) is not str:
                raise InvariantError("source schema-form target has no server-held plugin")
            if plugin != held_plugin:
                raise ValueError("schema-form plugin does not echo the server-held source plugin")
            updated = transition_source_schema_form(
                guided,
                target_id=target,
                turn=answered,
                response=form_response,
                authority=_schema8_schema_authority(turn=turn, plugin=held_plugin, options=options, source=True),
            )
        elif guided.step is GuidedStep.STEP_2_SINK:
            target = _schema8_pending_target(guided, source=False, phase="plugin_options")
            held_plugin = guided.pending_output_intents[target].plugin
            if type(held_plugin) is not str:
                raise InvariantError("sink schema-form target has no server-held plugin")
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

    from elspeth.web.sessions.protocol import (
        GuidedCompositionStateResult,
        GuidedOperationFailureCode,
        GuidedOperationFenceLostError,
    )

    from ..guided_operations import (
        GuidedOperationExpired,
        GuidedOperationLease,
        raise_guided_operation_failure,
        reserve_or_replay_guided_operation,
    )

    service: SessionServiceProtocol = request.app.state.session_service
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
        if not is_active_exit and observed_guided.step not in {GuidedStep.STEP_1_SOURCE, GuidedStep.STEP_2_SINK}:
            raise _schema8_unsupported_stage(observed_guided.step)
        if not is_active_exit and observed_guided.active_edit_target is not None:
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
        inspection_facts = (
            await _inspect_latest_ready_session_blob(request.app.state.blob_service, session_id)
            if observed_guided.step is GuidedStep.STEP_1_SOURCE and current_turn["type"] == TurnType.SINGLE_SELECT.value
            else None
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
                    recorder = BufferingRecorder()
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
                            prepared = prepare_guided_json_payload(
                                payload_store,
                                purpose=planned.purpose,
                                payload=planned.payload,
                            )
                            if prepared.payload_id != planned.payload_id:
                                raise AuditIntegrityError("Guided RESPOND prospective payload changed before settlement")
                            prepared_payloads.append(prepared)
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

                    final_guided = new_state.guided_session
                    if final_guided is None:  # pragma: no cover
                        raise AuditIntegrityError("Guided RESPOND settlement has no checkpoint")
                    existing_meta["guided_session"] = final_guided.to_dict()
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
            except Exception as exc:
                failure_code: GuidedOperationFailureCode = (
                    "integrity_error" if isinstance(exc, (AuditIntegrityError, InvariantError)) else "operation_failed"
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
                    failed = await service.fail_guided_operation(
                        reserved.fence,
                        failure_code=failure_code,
                        actor="composer_route",
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


async def _run_guided_chat_provider_attempt(**kwargs: Any) -> tuple[StepChatResult, Any | None, SinkResolved | None]:
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
    from elspeth.web.sessions.protocol import GuidedCompositionStateResult, GuidedOperationFailureCode

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
        failure_code: GuidedOperationFailureCode = "integrity_error" if isinstance(exc, AuditIntegrityError) else "operation_failed"
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
