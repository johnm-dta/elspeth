from __future__ import annotations

import asyncio
import json
from typing import Literal

from elspeth.web.composer.guided.chat_solver import build_step_chat_context_block
from elspeth.web.composer.guided.errors import WireConfirmRejectedError
from elspeth.web.composer.guided.profile import TUTORIAL_PROFILE, WorkflowProfileKind, profile_for_kind
from elspeth.web.composer.guided.protocol import Turn
from elspeth.web.composer.protocol import ComposerService
from elspeth.web.composer.tools._common import ToolContext
from elspeth.web.composer.tools._shield_availability import azure_prompt_shield_available
from elspeth.web.composer.tutorial_sample import (
    resolve_tutorial_sample_urls,
    tutorial_sample_base_url,
)
from elspeth.web.interpretation_state import refine_prompt_shield_warnings_for_availability
from elspeth.web.sessions._guided_step_chat import Step1SourceChatResult
from elspeth.web.sessions.schemas import StartGuidedRequest, TutorialSampleResponse

from .._helpers import (
    _COMMIT_REJECTED_MESSAGE,
    UTC,
    UUID,
    Any,
    APIRouter,
    BlobQuotaExceededError,
    BlobServiceProtocol,
    BufferingRecorder,
    CatalogServiceProtocol,
    ChatRole,
    ChatTurn,
    ChatTurnResponse,
    ComposerChatInitiator,
    ComposerChatTurn,
    ComposerChatTurnStatus,
    ComposerProgressEvent,
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
    TurnResponse,
    TurnType,
    UserIdentity,
    _composer_progress_sink,
    _dispatch_guided_respond,
    _get_composer_progress_registry,
    _get_session_compose_lock_registry,
    _initial_composition_state_with_guided_session,
    _inspect_latest_ready_session_blob,
    _persist_chat_turns,
    _persist_llm_calls,
    _persist_tool_invocations,
    _publish_progress,
    _replace,
    _safe_frame_strings,
    _state_from_record,
    _state_response,
    _store_guided_audit_payload,
    _summarize_guided_response,
    _validate_control_signal,
    _validate_step_indices,
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
    build_step_3_propose_chain_turn,
    build_step_3_schema_form_turn,
    build_step_4_wire_turn,
    client_cancelled_progress_event,
    contextlib,
    datetime,
    deep_thaw,
    emit_dropped_to_freeform,
    emit_step_advanced,
    emit_turn_answered,
    emit_turn_emitted,
    get_current_user,
    handle_step_1_source,
    handle_step_2_sink,
    resolve_step_1_source_chat_with_auto_drop,
    resolve_step_2_sink_chat_with_auto_drop,
    slog,
    solve_step_chat_with_auto_drop,
    stable_hash,
    step_advance,
    sys,
)


def _resolve_shield_available(request: Request, user_id: str) -> bool:
    """Resolve whether the authorized prompt-injection shield is available for this user.

    Delegates to :func:`azure_prompt_shield_available` via the request's
    configured ``scoped_secret_resolver``.  A configured ``None`` resolver still
    returns ``False`` (State C); a missing app-state dependency is a wiring
    error — see
    :func:`elspeth.web.composer.tools._shield_availability.azure_prompt_shield_available`.
    """
    return azure_prompt_shield_available(
        ToolContext(
            catalog=request.app.state.catalog_service,
            secret_service=request.app.state.scoped_secret_resolver,
            user_id=user_id,
        )
    )


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
    shield_available: bool,
) -> TurnPayloadResponse | None:
    """Build a ``TurnPayloadResponse``, refining shield warnings for B-vs-C state.

    For ``confirm_wiring`` turns the ``payload["warnings"]`` list is
    post-processed by :func:`refine_prompt_shield_warnings_for_availability` so
    the caller receives State-B wording when the authorized shield is present in
    this deployment.  All other turn types pass through unchanged.

    Returns ``None`` when ``turn`` is ``None`` (terminal or no-turn step).
    """
    if turn is None:
        return None
    payload = dict(turn["payload"])
    if turn["type"] == TurnType.CONFIRM_WIRING.value:
        payload["warnings"] = refine_prompt_shield_warnings_for_availability(
            payload["warnings"],
            shield_available=shield_available,
        )
    return TurnPayloadResponse(
        type=turn["type"],
        step_index=turn["step_index"],
        payload=payload,
    )


router = APIRouter()


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
      2. ``step_1_chosen_plugin`` is set → the SINGLE_SELECT response was
         submitted; the session is waiting for SCHEMA_FORM submission.
         Emit ``schema_form`` for the chosen plugin with persisted
         inspection-fact prefill.
      3. Neither staging field is set → initial state. Fall through to
         ``build_initial_step_1_turn``.

    - STEP_2_SINK: three sub-cases in priority order:
      1. ``step_2_sink_intent`` is set → the SCHEMA_FORM response was
         submitted; the session is waiting for MULTI_SELECT_WITH_CUSTOM
         confirmation.  Emit ``multi_select_with_custom`` from step_1_result.
      2. ``step_2_chosen_plugin`` is set → the SINGLE_SELECT response was
         submitted; the session is waiting for SCHEMA_FORM submission.
         Emit ``schema_form`` for the chosen plugin.
      3. Neither is set → initial state.  Emit ``single_select`` listing
         all registered sink plugins.

    - STEP_2_5_RECIPE_MATCH: vestigial step (the recipe-offer deviation was
      removed); always returns ``None`` — no live session reaches it.

    - STEP_3_TRANSFORMS: if ``step_3_proposal`` is set, emit
      ``propose_chain`` from the staged proposal, unless
      ``step_3_edit_index`` is set AND in range for the current proposal, in
      which case emit the transform ``schema_form`` for the proposed step
      being revised.  A stale/out-of-range ``step_3_edit_index`` degrades to
      the no-edit ``propose_chain`` rebuild rather than raising.  Returns
      ``None`` if the proposal is absent (LLM call has not completed; should
      not occur in practice — guarded defensively to avoid a crash).

    - STEP_4_WIRE: rebuild the skeleton ``confirm_wiring`` turn from current
      validation without mutating history.

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
        if guided.step_1_chosen_plugin is not None:
            return build_step_1_schema_form_turn(
                guided.step_1_chosen_plugin,
                catalog,
                inspection_facts=guided.step_1_inspection_facts,
            )
        if guided.step_1_result is not None:
            # In-place applied source (chat apply): re-render the populated form.
            # Reaches here only when the chat-apply branch (Task 3) cleared the
            # staging fields after committing — so a manual in-progress plugin
            # switch (chosen_plugin set) still wins above.
            return build_step_1_schema_form_turn_from_resolved(guided.step_1_result, catalog)
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
        if guided.step_2_result is not None:
            # In-place applied sink (chat apply): re-render the populated form.
            return build_step_2_schema_form_turn_from_resolved(guided.step_2_result, catalog)
        # Initial state: no plugin chosen yet.  Emit the sink plugin list.
        return build_step_2_single_select_turn(catalog)
    if step is GuidedStep.STEP_2_5_RECIPE_MATCH:
        # Vestigial step — the recipe-offer deviation was removed; the sink
        # commit now hops straight to STEP_3. No live session reaches this step,
        # so there is no rebuildable turn here.
        return None
    if step is GuidedStep.STEP_3_TRANSFORMS:
        if guided.step_3_proposal is not None:
            edit_index = guided.step_3_edit_index
            # A stale edit_index can outlive the proposal it was staged
            # against (e.g. a chat-revise replaces step_3_proposal with a
            # shorter chain) if some future call site installs a new
            # proposal without clearing the index. Degrade to no-edit-in-
            # progress rather than raising IndexError — GET /guided must
            # stay reconstructible from whatever was last persisted.
            if edit_index is not None and 0 <= edit_index < len(guided.step_3_proposal.steps):
                step_record = dict(guided.step_3_proposal.steps[edit_index])
                plugin = step_record["plugin"]
                options = step_record["options"]
                # ChainProposal.__post_init__ deep-freezes ``steps``, so a
                # live in-memory or from_dict-reconstructed proposal's
                # ``options`` is a MappingProxyType, never a plain dict —
                # isinstance(Mapping) accepts both; build_step_3_schema_form_turn
                # deep_thaws before use.
                if type(plugin) is not str or not isinstance(options, Mapping):
                    raise InvariantError("STEP_3 edit rebuild requires proposal step plugin str and options mapping")
                return build_step_3_schema_form_turn(
                    plugin=plugin,
                    options=options,
                    catalog=catalog,
                )
            return build_step_3_propose_chain_turn(guided.step_3_proposal)
        # No proposal yet — LLM call has not completed; return None and let the
        # idempotency machinery handle it (no TurnRecord emitted; client retries).
        return None
    if step is GuidedStep.STEP_4_WIRE:
        return build_step_4_wire_turn(state)
    return None


def _step_1_plugin_hint(guided: GuidedSession) -> str | None:
    """Derive the Step-1 chat resolver's plugin hint from structured state.

    Priority mirrors ``_build_get_guided_turn``'s STEP_1_SOURCE rebuild order:
    ``step_1_source_intent`` (awaiting INSPECT_AND_CONFIRM) -> ``step_1_chosen_plugin``
    (awaiting SCHEMA_FORM) -> ``step_1_result`` (already committed, chat-apply
    re-render). Never parses ``TurnRecord.summary`` — that is denormalised
    display copy for the client, not structured state, and a copy change to
    its "Selected: " prefix must not silently break chat resolution.
    """
    if guided.step_1_source_intent is not None:
        return guided.step_1_source_intent.plugin
    if guided.step_1_chosen_plugin is not None:
        return guided.step_1_chosen_plugin
    if guided.step_1_result is not None:
        return guided.step_1_result.plugin
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
    by the visible form, then let ``handle_step_1_source`` resolve the masked
    ``blob:<id>`` sentinel authoritatively.
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
        plugin=plugin_hint,
        options=options,
        observed_columns=tuple(inspection_facts.observed_headers or ()),
        sample_rows=(),
        on_validation_failure=on_validation_failure,
    )


def _guided_persisted_validity(state: CompositionState) -> tuple[bool, list[str] | None]:
    """Derive is_valid/validation_errors for a guided persist site.

    Mirrors the freeform persist convention's authoring-only fallback
    (``_composer_persisted_validation``'s last branch in ``_helpers.py``):
    ``CompositionState.validate()`` is a pure function of the graph alone, so
    it is the correct, uniform check at every guided persist site regardless
    of how far through the wizard the session has progressed — a genuinely
    incomplete mid-flow graph (no source yet, no sinks yet) validates with
    real errors, never the previous permanent ``is_valid=False,
    validation_errors=None`` stamp that combination-never-produced-by-freeform.

    No runtime preflight here (unlike ``_state_data_from_composer_state``):
    guided intermediate persists happen at points that do not uniformly carry
    a ``secret_service``/``settings`` runtime-preflight context, and runtime
    preflight is the freeform *commit* path's concern — the run-time gate in
    ``execution/service.py`` stays the hard backstop regardless of what this
    Stage-1-only check reports.
    """
    summary = state.validate()
    messages = [error.message for error in summary.errors]
    return summary.is_valid, messages or None


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
    turn_type = TurnType(turn["type"])
    payload_hash = stable_hash(turn["payload"])
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

    For sessions that already have a persisted CompositionState, if the
    current step has no emitted TurnRecord in the guided session history, a
    turn is built and persisted. Subsequent fetches are idempotent — the
    existing TurnRecord's payload_hash is returned verbatim.

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

            # Orphaned pre-change sessions: STEP_2_5_RECIPE_MATCH was retired as a
            # live step (the recipe-offer interstitial is gone — sink commit now
            # hops straight to STEP_3). A session persisted at this step before
            # that change has no rebuildable turn (``_build_get_guided_turn``
            # returns None, so GET would otherwise render a blank turn) and no POST
            # dispatch branch (a non-exit response hits ``step_advance``'s
            # unhandled-step InvariantError → 500). Reject with a clear, structured
            # 409 that points at the salvage path rather than silently returning no
            # turn. A session that has already exited (terminal set) is left alone.
            if guided.terminal is None and current_step is GuidedStep.STEP_2_5_RECIPE_MATCH:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "This guided session was started before a composer update and can no "
                        "longer continue from its current step (step_2_5_recipe_match). POST "
                        "control_signal=exit_to_freeform to keep your work in freeform, or "
                        "start a new guided session."
                    ),
                )

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
            if guided.terminal is None:
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
            else:
                turn = None
            turn_type: TurnType | None = TurnType(turn["type"]) if turn is not None else None
            payload_hash: str | None = stable_hash(turn["payload"]) if turn is not None else None

            if existing_record_for_step is None and turn is not None and state_record is not None:
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
                payload_payload_id = _store_guided_audit_payload(request.app.state.payload_store, turn["payload"])
                new_guided, _new_record, turn_type, payload_hash = _append_server_turn_record(
                    guided,
                    current_step=current_step,
                    turn=turn,
                )
                new_state = _replace(state, guided_session=new_guided)

                # Persist state with updated guided_session in composer_meta.
                # Preserve any existing composer_meta keys (e.g. repair_turns_used).
                existing_meta: dict[str, Any] = {}
                if state_record is not None and state_record.composer_meta is not None:
                    existing_meta = dict(deep_thaw(state_record.composer_meta))
                new_composer_meta = {**existing_meta, "guided_session": new_guided.to_dict()}

                state_d = new_state.to_dict()
                persisted_is_valid, persisted_errors = _guided_persisted_validity(new_state)
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
                state_record_out = await service.save_composition_state(
                    session_id,
                    state_data,
                    # Guided-mode server-emitted turn: the LLM converged on a
                    # guided step transition and the resulting state is being
                    # persisted. ``convergence_persist`` is the closest existing
                    # provenance category. The closed enum does not have a
                    # ``guided_persist`` value; widening it is out of
                    # scope here — see merge commit message.
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
                    payload_payload_id=payload_payload_id,
                    emitter="server",
                    composition_version=new_state.version,
                    actor=user.user_id,
                )

                guided = new_guided

            # Build response.  On re-fetch the same turn is returned (deterministic
            # rebuild) and the payload_hash matches what was recorded on first visit.
            terminal = guided.terminal
            shield_available = _resolve_shield_available(request, user.user_id)
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
                next_turn=_turn_payload_response(turn, shield_available=shield_available),
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
    catalog: CatalogServiceProtocol = request.app.state.catalog_service

    compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))
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
        reopened_record = _replace(current_record, response_hash=None, summary=None)
        reopened_history = tuple(reopened_record if r is current_record else r for r in guided.history)
        new_guided = _replace(guided, history=reopened_history, terminal=None)
        new_state = _replace(state, guided_session=new_guided)
        turn = _build_get_guided_turn(new_state, new_guided, catalog=catalog)
        if turn is None:
            raise HTTPException(
                status_code=409,
                detail="Guided session cannot be re-entered because the current turn cannot be rebuilt.",
            )

        existing_meta: dict[str, Any] = {}
        if state_record.composer_meta is not None:
            existing_meta = dict(deep_thaw(state_record.composer_meta))
        new_composer_meta = {**existing_meta, "guided_session": new_guided.to_dict()}
        state_d = new_state.to_dict()
        persisted_is_valid, persisted_errors = _guided_persisted_validity(new_state)
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
        state_record_out = await service.save_composition_state(
            session_id,
            state_data,
            provenance="convergence_persist",
        )

        shield_available = _resolve_shield_available(request, user.user_id)
        return GetGuidedResponse(
            guided_session=GuidedSessionResponse(
                step=new_guided.step.value,
                history=[
                    TurnRecordResponse(
                        step=r.step.value,
                        turn_type=r.turn_type.value,
                        payload_hash=r.payload_hash,
                        response_hash=r.response_hash,
                        summary=r.summary,
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
                        assistant_message_kind=t.assistant_message_kind,
                        synthetic_failure_reason=t.synthetic_failure_reason,
                    )
                    for t in new_guided.chat_history
                ],
                chat_turn_seq=new_guided.chat_turn_seq,
                profile=_workflow_profile_response(new_guided),
            ),
            next_turn=_turn_payload_response(turn, shield_available=shield_available),
            terminal=None,
            composition_state=_state_response(state_record_out),
        )


@router.post("/{session_id}/guided/start", response_model=GetGuidedResponse)
async def post_guided_start(
    session_id: UUID,
    body: StartGuidedRequest,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> GetGuidedResponse:
    """Seed a guided session with a server-owned WorkflowProfile.

    The client supplies a closed-enum ``profile`` discriminator
    (``WorkflowProfileKind``); the SERVER maps it to the concrete profile
    object and persists the resulting GuidedSession, so a client cannot
    inject an arbitrary profile or weaken the advisor gate (D13/§4.3).

    **Idempotent (D16):** a second start for a session that ALREADY has a
    persisted GuidedSession returns the existing session unchanged — it
    never re-initialises or double-creates.
    GET /api/sessions/{session_id}/guided then reads the
    persisted ``GuidedSession.profile``; the lazy no-arg GET default path
    stays for live guided (empty profile).

    Decision D: ``start`` persists the SERVER-owned profile only (the behavior
    flags) and does not fabricate any source/topology into the CompositionState;
    the concrete pipeline is built downstream by the guided wizard + web-scrape
    recipe match.

    Raises 404 if the session does not exist or belong to the requester.
    Raises 409 if the session already has a freeform composition state with
    no GuidedSession; this route does not convert or discard freeform state.
    Raises 400 if ``profile`` is not a recognised WorkflowProfileKind or if
    a client sends anything other than a short discriminator string.
    """
    await _verify_session_ownership(session_id, user, request)
    service: SessionServiceProtocol = request.app.state.session_service
    catalog: CatalogServiceProtocol = request.app.state.catalog_service

    # Tier-3 -> Tier-2 coercion at the profile-kind boundary. A stale client
    # sending an unknown discriminator gets a 400 with a generic message
    # rather than a Pydantic 422; the typed kind then selects a SERVER-owned
    # constant — the client never supplies the profile object. Do not echo
    # the raw value: it may be a long string or an attempted profile object
    # carrying attacker-controlled profile fields.
    if not isinstance(body.profile, str) or len(body.profile) > 32:
        raise HTTPException(
            status_code=400,
            detail=(f"Invalid profile discriminator. Valid values: {sorted(k.value for k in WorkflowProfileKind)}."),
        )
    try:
        profile_kind = WorkflowProfileKind(body.profile)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=(f"Unknown profile discriminator. Valid values: {sorted(k.value for k in WorkflowProfileKind)}."),
        ) from exc
    # Map the validated kind to its SERVER-owned profile constant via the closed
    # mapper (profile.py): a future third kind raises InvariantError here instead
    # of silently mapping to EMPTY.
    profile = profile_for_kind(profile_kind)

    compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))
    async with compose_lock:
        # Idempotency (D16): if a guided session is already persisted, return
        # it UNCHANGED — never re-init (a second start must not clobber the
        # learner's in-progress wizard or re-seed a fresh profile).
        existing_record = await service.get_current_state(session_id)
        if existing_record is not None:
            existing_state = _state_from_record(existing_record)
            if existing_state.guided_session is not None:
                guided = existing_state.guided_session
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
                    # next_turn=None is safe HERE (unlike post_guided_convert's
                    # idempotency branch, elspeth-e2c3dba6b5 review P2): the start
                    # response is always followed by a GET /guided that rebuilds
                    # the live turn, whereas convert feeds the store directly via
                    # enterGuided. If start ever stops being GET-followed, this
                    # needs the same non-terminal rebuild convert now does.
                    next_turn=None,
                    terminal=TerminalStateResponse(
                        kind=terminal.kind.value,
                        reason=terminal.reason.value if terminal.reason is not None else None,
                        pipeline_yaml=terminal.pipeline_yaml,
                    )
                    if terminal is not None
                    else None,
                    composition_state=_state_response(existing_record),
                )
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot start guided on a session that already has existing "
                    "freeform composition state. Create a new session or fork before "
                    "starting the tutorial profile."
                ),
            )

        # No persisted guided session yet: attach the server-owned profile to a
        # fresh guided state and PERSIST it (so GET /api/sessions/{session_id}/guided
        # reads the profile back). Decision D: this attaches the profile only — it
        # does not fabricate any source/topology into the CompositionState; the
        # concrete pipeline is wizard/recipe-built downstream. For the live profile
        # this is the existing empty guided state.
        new_state = _initial_composition_state_with_guided_session(profile=profile)
        seeded_guided = new_state.guided_session
        if seeded_guided is None:  # pragma: no cover — helper always attaches a guided session
            raise InvariantError("post_guided_start: initial state has no guided_session")
        turn = _build_get_guided_turn(new_state, seeded_guided, catalog=catalog)

        new_composer_meta = {"guided_session": seeded_guided.to_dict()}
        state_d = new_state.to_dict()
        persisted_is_valid, persisted_errors = _guided_persisted_validity(new_state)
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
        state_record_out = await service.save_composition_state(
            session_id,
            state_data,
            # Start endpoint seeds the canonical guided session for a profile;
            # ``session_seed`` is the closest existing provenance category for
            # a fresh server-authored seed state (the closed enum has no
            # guided-specific value — see merge commit message).
            provenance="session_seed",
        )

        shield_available = _resolve_shield_available(request, user.user_id)
        return GetGuidedResponse(
            guided_session=GuidedSessionResponse(
                step=seeded_guided.step.value,
                history=[
                    TurnRecordResponse(
                        step=r.step.value,
                        turn_type=r.turn_type.value,
                        payload_hash=r.payload_hash,
                        response_hash=r.response_hash,
                        summary=r.summary,
                        emitter=r.emitter,
                    )
                    for r in seeded_guided.history
                ],
                terminal=None,
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
                    for t in seeded_guided.chat_history
                ],
                chat_turn_seq=seeded_guided.chat_turn_seq,
                profile=_workflow_profile_response(seeded_guided),
            ),
            next_turn=_turn_payload_response(turn, shield_available=shield_available),
            terminal=None,
            composition_state=_state_response(state_record_out),
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
    Raises 409 with a structured ``wire_confirm_rejected`` detail when a
        STEP_4_WIRE confirm targets an invalid pipeline — the rejection names
        each blocking validation issue and, deliberately, persists NO new
        composition-state version (elspeth-3b35abf148 variant 3).
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
            #     already audit-recorded by the preceding STEP_4_WIRE confirm
            #     dispatch.
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
                new_guided = _replace(guided, terminal=new_terminal, transition_consumed=True)

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
                persisted_is_valid, persisted_errors = _guided_persisted_validity(new_state)
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
                                summary=r.summary,
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
                                assistant_message_kind=t.assistant_message_kind,
                                synthetic_failure_reason=t.synthetic_failure_reason,
                            )
                            for t in new_guided.chat_history
                        ],
                        chat_turn_seq=new_guided.chat_turn_seq,
                        profile=_workflow_profile_response(new_guided),
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

            # Orphaned pre-change sessions parked at the retired
            # STEP_2_5_RECIPE_MATCH interstitial: a non-exit response has no
            # dispatch branch (``step_advance`` raises "unhandled step" → HTTP
            # 500). Reject with the same clear 409 GET emits. ``exit_to_freeform``
            # is intentionally exempt — ``step_advance`` handles it
            # step-independently (state_machine.py:529), so the user can still
            # salvage their source/sink work to freeform. Placed after the
            # terminal-rejection above (so ``guided`` is non-terminal here) and
            # before the turn-answering machinery, since no live turn at this
            # retired step can legitimately be answered.
            if guided.step is GuidedStep.STEP_2_5_RECIPE_MATCH and control_signal is not ControlSignal.EXIT_TO_FREEFORM:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "This guided session was started before a composer update and can no "
                        "longer continue from its current step (step_2_5_recipe_match). POST "
                        "control_signal=exit_to_freeform to keep your work in freeform, or "
                        "start a new guided session."
                    ),
                )

            # Optimistic-concurrency guard (D16): if the client carried an
            # expected step, reject a mismatch with 409 (the wizard advanced
            # under the client between read and write) — the same guard
            # ``POST /api/sessions/{session_id}/guided/chat`` already has
            # (guided.py:~1394). A stale client
            # sending an unknown value gets a 400, not a Pydantic 422,
            # mirroring control_signal. ``None`` (field absent) skips the
            # guard for turns that do not carry an expected step.
            if body.step_index is not None:
                try:
                    expected_step = GuidedStep(body.step_index)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=(f"Unknown step_index {body.step_index!r}. Valid values: {sorted(s.value for s in GuidedStep)}."),
                    ) from exc
                if expected_step is not guided.step:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"step_index {expected_step.value!r} does not match the session's "
                            f"current step {guided.step.value!r}. Re-fetch GET "
                            f"/api/sessions/{{id}}/guided and retry."
                        ),
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
                from dataclasses import replace as _replace

                turn = _build_get_guided_turn(state, guided, catalog=catalog)
                if turn is None:
                    # Structured detail — same envelope shape as the
                    # ``wire_confirm_rejected`` 409 below (``code``/``detail``/
                    # extra fields; the human-readable text is the nested
                    # ``detail`` key, matching ``parseResponse``'s
                    # ``nestedDetail.detail`` read), not the raw protocol
                    # string: the frontend was rendering the old "Fetch GET
                    # /api/sessions/{id}/guided first" instruction verbatim in
                    # the user's alert banner (C-3(c)). ``code`` lets a later
                    # frontend wave self-heal (re-fetch guided state) instead
                    # of instructing the user to call an API. ``step`` mirrors
                    # the ``step_index`` mismatch 409 above so the client can
                    # correlate without a second round-trip.
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "code": "turn_not_emitted",
                            "detail": (
                                "Your session's step is out of sync with the server. Refreshing the session will resync this automatically."
                            ),
                            "step": current_step.value,
                        },
                    )
                guided, existing_record, current_turn_type, emitted_payload_hash = _append_server_turn_record(
                    guided,
                    current_step=current_step,
                    turn=turn,
                )
                state = _replace(state, guided_session=guided)
                emit_turn_emitted(
                    recorder,
                    step=current_step,
                    turn_type=current_turn_type,
                    payload_hash=emitted_payload_hash,
                    payload_payload_id=_store_guided_audit_payload(request.app.state.payload_store, turn["payload"]),
                    emitter="server",
                    composition_version=state.version,
                    actor=user.user_id,
                )
            else:
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
            updated_record = _replace(
                existing_record,
                response_hash=response_hash,
                summary=_summarize_guided_response(current_turn_type, turn_response),
            )
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
                response_payload_id=_store_guided_audit_payload(request.app.state.payload_store, turn_response),
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
            # Why 500 (Tier-1 crash class) and not 4xx/quarantine: the
            # persisted ``guided_session`` is a Tier-1 checkpoint — our
            # ``to_dict`` envelope, unbroken chain of custody, no external
            # writer of ``composer_meta["guided_session"]`` — so a
            # ``from_dict`` failure is a Tier-1 anomaly, and the 500 is the
            # web expression of "crash on a Tier-1 anomaly", not a Tier-3
            # quarantine. Full rationale: ``guided/errors.py`` InvariantError
            # docstring.
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
            if (
                terminal is not None
                and terminal.kind is TerminalKind.EXITED_TO_FREEFORM
                and terminal.reason is TerminalReason.USER_PRESSED_EXIT
            ):
                guided = _replace(guided, transition_consumed=True)
                state = _replace(state, guided_session=guided)

            # Run side-effect dispatcher if the session is not yet terminal.
            # The dispatcher calls step handlers (handle_step_1_source,
            # handle_step_2_sink, handle_step_3_chain_accept) and emits
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
                        blob_service=request.app.state.blob_service,
                        payload_store=request.app.state.payload_store,
                        model=settings.composer_model,
                        temperature=settings.composer_temperature,
                        seed=settings.composer_seed,
                        composer_service=request.app.state.composer_service,
                        advisor_checkpoint_max_passes=settings.composer_advisor_checkpoint_max_passes,
                        settings=settings,
                    )
                except WireConfirmRejectedError as exc:
                    # Wire-stage confirm against an invalid pipeline: a
                    # structured, actionable 409 — NOT a silent 200 and NOT a
                    # new composition-state version (pre-fix, every failed
                    # confirm minted one; elspeth-3b35abf148 variant 3). No
                    # persistence here on purpose: the composition is
                    # unchanged by a rejected confirm, and the
                    # ``guided_turn_answered`` audit event still drains via
                    # the finally block, so the rejected attempt remains on
                    # the audit record without a version bump. The nested
                    # detail shape (``detail``/``code``/``validation_errors``)
                    # is the envelope the frontend parseResponse already
                    # understands. ``validation_errors`` carries the same
                    # ValidationEntry payloads already egressed via the wire
                    # turn — no new egress surface.
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "wire_confirm_rejected",
                            "detail": (
                                "The pipeline can't be confirmed yet - validation found "
                                f"{len(exc.issues)} blocking issue(s) at the wiring step. "
                                "Fix the issues below, then confirm again."
                            ),
                            "step": exc.step,
                            "validation_errors": list(exc.issues),
                        },
                    ) from exc
                except HTTPException:
                    # Dispatcher-level 400s happen after the turn has been
                    # answered and audited. Persist the response_hash on that
                    # TurnRecord so a rejected confirm remains a durable part of
                    # the guided transcript, then re-raise the original HTTP
                    # response.
                    new_state = _replace(state, guided_session=guided)
                    existing_meta_error: dict[str, Any] = {}
                    if state_record is not None and state_record.composer_meta is not None:
                        existing_meta_error = dict(deep_thaw(state_record.composer_meta))
                    error_meta = {**existing_meta_error, "guided_session": guided.to_dict()}
                    state_d_error = new_state.to_dict()
                    persisted_is_valid_error, persisted_errors_error = _guided_persisted_validity(new_state)
                    state_data_error = CompositionStateData(
                        sources=state_d_error["sources"],
                        nodes=state_d_error["nodes"],
                        edges=state_d_error["edges"],
                        outputs=state_d_error["outputs"],
                        metadata_=state_d_error["metadata"],
                        is_valid=persisted_is_valid_error,
                        validation_errors=persisted_errors_error,
                        composer_meta=error_meta,
                    )
                    state_record_out = await service.save_composition_state(
                        session_id,
                        state_data_error,
                        provenance="convergence_persist",
                    )
                    raise
                except InvariantError as exc:
                    # Same B1-sanitization rationale as the step_advance
                    # catch above: ``str(exc)`` from a ``from_dict`` site
                    # embeds ``{d!r}`` of the corrupted Tier-1 record
                    # including Tier-3 ``sample_rows``. Static detail; slog
                    # carries exc_class + frames only. Same Tier-1-checkpoint
                    # classification as the step_advance catch (500 = crash on
                    # a Tier-1 anomaly, not a Tier-3 quarantine); full
                    # rationale in ``guided/errors.py`` InvariantError docstring.
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
            persisted_is_valid, persisted_errors = _guided_persisted_validity(new_state)
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
            state_record_out = await service.save_composition_state(
                session_id,
                state_data,
                # Guided POST /respond: the user-supplied turn converged on a
                # guided step transition and the resulting state is being
                # persisted. Same provenance choice as the GET /guided server-
                # emitted path (the closed enum has no guided-specific value;
                # widening is out of scope here — see merge commit message).
                provenance="convergence_persist",
            )

            # B1 (spec §5): the guided dispatch path never reaches the
            # freeform fail-closed orphan gate, so a committed source /
            # transform / recipe-apply that creates interpretation sites
            # would orphan and only fail at run time. Surface every
            # resolvable pending review against the freshly-persisted state
            # so the guided UI can project + block on it (D12). Advisory
            # polarity: the run-time gate (execution/service.py:515-529)
            # stays the hard backstop, so a None composer (no impl wired in
            # this app) safely skips surfacing.
            composer: ComposerService = request.app.state.composer_service
            if composer is not None:
                await composer.surface_pending_interpretation_reviews(
                    new_state,
                    session_id=str(session_id),
                    current_state_id=str(state_record_out.id),
                )

            # Recorder persistence happens in the finally block below so
            # rejection paths drain identically to the success path.

            shield_available = _resolve_shield_available(request, user.user_id)
            return GuidedRespondResponse(
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
                next_turn=_turn_payload_response(next_turn, shield_available=shield_available),
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
                        plugin_crash_pending=True,
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


async def _build_guided_chat_apply_response(
    *,
    guided: GuidedSession,
    state: CompositionState,
    next_turn: Turn,
    assistant_message: str,
    service: SessionServiceProtocol,
    session_id: UUID,
    state_record: CompositionStateRecord | None,
    shield_available: bool,
) -> tuple[GuidedChatResponse, CompositionStateRecord]:
    """Persist the in-place-applied state and build the chat-apply response.

    Shared tail for the STEP_1/STEP_2/STEP_3 /guided/chat apply branches: build
    CompositionStateData from ``state`` + the committed ``guided``, persist via
    ``save_composition_state(provenance="convergence_persist")``, and assemble the
    GuidedChatResponse with the populated ``next_turn``. ``guided`` and ``state``
    must already carry the committed result, the appended history/chat_history,
    and cleared staging fields — this helper does NOT mutate them.

    Returns the response AND the freshly-saved ``CompositionStateRecord`` so the
    caller can re-bind its outer ``state_record_out`` before returning. That
    binding is load-bearing: the route's ``finally`` audit-drain threads
    ``state_record_out.id`` into every persisted ComposerChatTurn/LLMCall/Tool
    row, so a stale record here would mis-correlate the audit trail.
    """
    new_state = _replace(state, guided_session=guided)
    existing_meta: dict[str, Any] = {}
    if state_record is not None and state_record.composer_meta is not None:
        existing_meta = dict(deep_thaw(state_record.composer_meta))
    new_composer_meta = {**existing_meta, "guided_session": guided.to_dict()}
    state_d = new_state.to_dict()
    persisted_is_valid, persisted_errors = _guided_persisted_validity(new_state)
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
    state_record_out = await service.save_composition_state(session_id, state_data, provenance="convergence_persist")
    response = GuidedChatResponse(
        assistant_message=assistant_message,
        # Always a real LLM reply: every caller of this helper reached it via
        # a resolve/commit SUCCESS (STEP_1/STEP_2/STEP_3 apply branches), not
        # a fallback.
        assistant_message_kind="assistant",
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
            terminal=None,
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
        next_turn=_turn_payload_response(next_turn, shield_available=shield_available),
        terminal=None,
        composition_state=_state_response(state_record_out),
    )
    return response, state_record_out


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
    catalog: CatalogServiceProtocol = request.app.state.catalog_service
    shield_available = _resolve_shield_available(request, user.user_id)

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
        # Guards the finally block's terminal progress publish below: an
        # early 400/409 guard-clause rejection (not-guided / terminal /
        # step-mismatch) must not emit an orphan "complete"/"failed" event
        # for a compose that never started.
        progress_started = False
        progress_registry = _get_composer_progress_registry(request)
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

            # Uniform across every guided step (STEP_1/STEP_2/advisory-only
            # steps alike) — never forked on "is tutorial". STEP_2_SINK's
            # resolver additionally emits calling_model/using_tools hops
            # (see resolve_step_2_sink_chat_with_auto_drop below); other
            # steps get this single starting->complete bracket, which is
            # coarse but real — not a timer-based fake progression.
            progress_sink = _composer_progress_sink(
                progress_registry,
                session_id=str(session_id),
                request_id=None,
                user_id=str(user.user_id),
            )
            await _publish_progress(
                progress_registry,
                session_id=str(session_id),
                request_id=None,
                user_id=str(user.user_id),
                event=ComposerProgressEvent(
                    phase="starting",
                    headline="I'm reading your message for this step.",
                    evidence=("The chat message was accepted for this guided step.",),
                    likely_next="ELSPETH will ask the model how to respond.",
                ),
            )
            progress_started = True

            settings = request.app.state.settings
            started_at = datetime.now(UTC)
            from time import perf_counter as _perf_counter

            started_perf = _perf_counter()
            existing_record_for_chat: TurnRecord | None = next(
                (r for r in reversed(guided.history) if r.step == guided.step),
                None,
            )
            current_turn_type = existing_record_for_chat.turn_type if existing_record_for_chat is not None else None
            chat_result = None
            # Computed once, up front: reused by the STEP_1/STEP_2 resolve
            # calls below (so a declined-to-prose reply is grounded in the
            # same context a second, tool-less call would otherwise have
            # supplied) AND the final advisory fallback. Safe to hoist —
            # ``state``/``guided.step_1_result``/``guided.step_2_result`` are
            # only ever reassigned on a branch that returns immediately, so
            # nothing here changes before a later read of the same value.
            chat_context_block = build_step_chat_context_block(
                step=guided.step,
                current_source=guided.step_1_result,
                current_sink=guided.step_2_result,
                state=state,
            )

            if (
                existing_record_for_chat is not None
                and guided.step is GuidedStep.STEP_1_SOURCE
                and current_turn_type in (TurnType.SINGLE_SELECT, TurnType.SCHEMA_FORM, TurnType.INSPECT_AND_CONFIRM)
            ):
                plugin_hint = _step_1_plugin_hint(guided)
                source_chat_result: Step1SourceChatResult | None = None
                if current_turn_type is TurnType.SCHEMA_FORM:
                    uploaded_source = await _source_from_latest_uploaded_blob_for_step_1_chat(
                        message=body.message,
                        plugin_hint=plugin_hint,
                        blob_service=request.app.state.blob_service,
                        session_id=session_id,
                    )
                    if uploaded_source is not None:
                        finished_at = datetime.now(UTC)
                        latency_ms = int((_perf_counter() - started_perf) * 1000)
                        uploaded_data_dir: str | None = str(settings.data_dir) if settings.data_dir else None
                        handler_result = handle_step_1_source(
                            state=state,
                            session=guided,
                            resolved=uploaded_source,
                            catalog=catalog,
                            data_dir=uploaded_data_dir,
                            session_engine=request.app.state.session_engine,
                            session_id=str(session_id),
                        )
                        if not handler_result.tool_result.success:
                            source_chat_result = Step1SourceChatResult(
                                source_resolution=None,
                                fallback_chat=StepChatResult(
                                    assistant_message=_COMMIT_REJECTED_MESSAGE,
                                    status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
                                    latency_ms=latency_ms,
                                    error_class="StepHandlerRejected",
                                ),
                                prose_chat=None,
                            )
                        else:
                            state = handler_result.state
                            uploaded_turn_response: TurnResponse = {
                                "chosen": None,
                                "edited_values": {
                                    "plugin": uploaded_source.plugin,
                                    "options": dict(uploaded_source.options),
                                    "observed_columns": list(uploaded_source.observed_columns),
                                    "sample_rows": [dict(row) for row in uploaded_source.sample_rows],
                                },
                                "custom_inputs": None,
                                "accepted_step_index": None,
                                "edit_step_index": None,
                                "control_signal": None,
                            }
                            response_hash = stable_hash(uploaded_turn_response)
                            answered_record = _replace(
                                existing_record_for_chat,
                                response_hash=response_hash,
                                summary=_summarize_guided_response(TurnType.SCHEMA_FORM, uploaded_turn_response),
                            )
                            answered_history = tuple(answered_record if r is existing_record_for_chat else r for r in guided.history)
                            guided = _replace(
                                handler_result.session,
                                history=answered_history,
                                step_1_chosen_plugin=None,
                                step_1_source_intent=None,
                            )
                            emit_turn_answered(
                                recorder,
                                step=GuidedStep.STEP_1_SOURCE,
                                turn_type=TurnType.SCHEMA_FORM,
                                response_hash=response_hash,
                                response_payload_id=_store_guided_audit_payload(request.app.state.payload_store, uploaded_turn_response),
                                control_signal=None,
                                composition_version=state.version,
                                actor=user.user_id,
                            )
                            applied_step_1_result = handler_result.session.step_1_result
                            if applied_step_1_result is None:
                                raise InvariantError(
                                    "step_1_result is None after successful uploaded-blob source commit — "
                                    "handler set tool_result.success=True but did not set step_1_result"
                                )
                            next_turn = build_step_1_schema_form_turn_from_resolved(applied_step_1_result, catalog)
                            next_turn_type = TurnType(next_turn["type"])
                            next_payload_hash = stable_hash(next_turn["payload"])
                            new_record = TurnRecord(
                                step=GuidedStep.STEP_1_SOURCE,
                                turn_type=next_turn_type,
                                payload_hash=next_payload_hash,
                                response_hash=None,
                                emitter="server",
                            )
                            emit_turn_emitted(
                                recorder,
                                step=GuidedStep.STEP_1_SOURCE,
                                turn_type=next_turn_type,
                                payload_hash=next_payload_hash,
                                payload_payload_id=_store_guided_audit_payload(request.app.state.payload_store, next_turn["payload"]),
                                emitter="server",
                                composition_version=state.version,
                                actor=user.user_id,
                            )
                            ts_iso = finished_at.isoformat()
                            user_turn = ChatTurn(
                                role=ChatRole.USER,
                                content=body.message,
                                seq=guided.chat_turn_seq,
                                step=GuidedStep.STEP_1_SOURCE,
                                ts_iso=ts_iso,
                            )
                            assistant_message = "I found the uploaded file and applied it as the pipeline input."
                            assistant_turn = ChatTurn(
                                role=ChatRole.ASSISTANT,
                                content=assistant_message,
                                seq=guided.chat_turn_seq + 1,
                                step=GuidedStep.STEP_1_SOURCE,
                                ts_iso=ts_iso,
                                assistant_message_kind="assistant",
                            )
                            guided = _replace(
                                guided,
                                history=(*guided.history, new_record),
                                chat_history=(*guided.chat_history, user_turn, assistant_turn),
                                chat_turn_seq=guided.chat_turn_seq + 2,
                            )
                            recorder.record_chat_turn(
                                ComposerChatTurn(
                                    step=GuidedStep.STEP_1_SOURCE.value,
                                    initiator=ComposerChatInitiator.USER,
                                    chat_turn_seq=user_turn.seq,
                                    user_message_hash=stable_hash(body.message),
                                    assistant_message_hash=stable_hash(assistant_message),
                                    latency_ms=latency_ms,
                                    model=settings.composer_model,
                                    status=ComposerChatTurnStatus.SUCCESS,
                                    started_at=started_at,
                                    finished_at=finished_at,
                                    error_class=None,
                                )
                            )
                            response, state_record_out = await _build_guided_chat_apply_response(
                                guided=guided,
                                state=state,
                                next_turn=next_turn,
                                assistant_message=assistant_message,
                                service=service,
                                session_id=session_id,
                                state_record=state_record,
                                shield_available=shield_available,
                            )
                            return response

                if source_chat_result is None:
                    source_chat_result = await resolve_step_1_source_chat_with_auto_drop(
                        site="post_guided_chat",
                        session_id=str(session_id),
                        user_id=user.user_id,
                        model=settings.composer_model,
                        user_message=body.message,
                        plugin_hint=plugin_hint,
                        current_source=guided.step_1_result,
                        temperature=settings.composer_temperature,
                        seed=settings.composer_seed,
                        recorder=recorder,
                        # Server-side bound, consistent with freeform compose
                        # (elspeth-fb4464cdf0): the guided LLM call may not run
                        # past the composer budget.
                        timeout_seconds=settings.composer_timeout_seconds,
                        context_block=chat_context_block,
                    )
                # ``prose_chat`` (a declined-to-prose SUCCESS) and
                # ``fallback_chat`` (an error) are mutually exclusive; either
                # one here means "use this directly, skip the second call".
                chat_result = source_chat_result.fallback_chat or source_chat_result.prose_chat
                source_resolution = source_chat_result.source_resolution
                if source_resolution is not None:
                    finished_at = datetime.now(UTC)
                    latency_ms = int((_perf_counter() - started_perf) * 1000)

                    blob_service: BlobServiceProtocol = request.app.state.blob_service
                    try:
                        source_blob = await blob_service.create_blob(
                            session_id,
                            source_resolution.filename,
                            source_resolution.content.encode("utf-8"),
                            source_resolution.mime_type,
                            created_by="assistant",
                            source_description="Generated from guided Step 1 chat",
                        )
                    except BlobQuotaExceededError as exc:
                        raise HTTPException(
                            status_code=413,
                            detail="Blob storage quota exceeded for this session.",
                        ) from exc

                    resolved = SourceResolved(
                        plugin=source_resolution.plugin,
                        # Inline chat-resolved sources (json/csv) infer column types
                        # from the data they carry, so default `schema: {mode:
                        # observed}` when the resolver omitted it — otherwise the
                        # commit fails validation with "schema: Field required". An
                        # explicit schema from the resolver still wins (it is spread
                        # after this default).
                        options={
                            "schema": {"mode": "observed"},
                            **dict(source_resolution.options),
                            "path": source_blob.storage_path,
                        },
                        # observed_columns is backfilled from the data at the
                        # commit convergence point (handle_step_1_source) when the
                        # LLM left it empty — see that handler. Doing it here would
                        # miss the schema_form re-submit path, which also commits
                        # through handle_step_1_source.
                        observed_columns=source_resolution.observed_columns,
                        sample_rows=source_resolution.sample_rows,
                        # The composer chose the source NODE's invalid-row routing
                        # while resolving the source from chat; carry it through so
                        # the commit and the re-rendered schema_form agree.
                        on_validation_failure=source_resolution.on_validation_failure,
                    )
                    data_dir: str | None = str(settings.data_dir) if settings.data_dir else None
                    handler_result = handle_step_1_source(
                        state=state,
                        session=guided,
                        resolved=resolved,
                        catalog=catalog,
                        data_dir=data_dir,
                        session_engine=request.app.state.session_engine,
                        session_id=str(session_id),
                    )
                    if not handler_result.tool_result.success:
                        # Symmetric with the STEP_2_SINK reject path below: the
                        # strict commit seam rejected the re-resolved source.
                        # Degrade to advisory (no mutation) rather than a fatal
                        # 400 — chat is a non-load-bearing helper and a second
                        # Send must not be terminal. The raw tool_result message
                        # can embed Tier-3 row data, so it never reaches the
                        # response; ``error_class`` is a redaction-safe CLASSIFIER
                        # that keeps the rejection diagnosable in the audit trail
                        # (the prior 400 raised before any chat-turn audit, so the
                        # reason vanished entirely).
                        chat_result = StepChatResult(
                            assistant_message=_COMMIT_REJECTED_MESSAGE,
                            status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
                            latency_ms=int((_perf_counter() - started_perf) * 1000),
                            error_class="StepHandlerRejected",
                        )
                        source_resolution = None

                if source_resolution is not None:
                    # Adopt the post-commit state FIRST, BEFORE either leg — the
                    # answered-record emit below reads `state.version`, and
                    # CompositionState bumps version in-memory on every edit
                    # (state.py: each edit returns a new instance; set_source =>
                    # version+1). If this assignment stayed after the if/else, the
                    # answered record would stamp the STALE pre-commit version — a
                    # silent audit-correctness regression.
                    state = handler_result.state
                    if current_turn_type is TurnType.SCHEMA_FORM:
                        turn_response: TurnResponse = {
                            "chosen": None,
                            "edited_values": {
                                "plugin": source_resolution.plugin,
                                "options": dict(source_resolution.options),
                                "observed_columns": list(source_resolution.observed_columns),
                                "sample_rows": [dict(row) for row in source_resolution.sample_rows],
                            },
                            "custom_inputs": None,
                            "accepted_step_index": None,
                            "edit_step_index": None,
                            "control_signal": None,
                        }
                        response_hash = stable_hash(turn_response)
                        answered_record = _replace(
                            existing_record_for_chat,
                            response_hash=response_hash,
                            summary=_summarize_guided_response(TurnType.SCHEMA_FORM, turn_response),
                        )
                        answered_history = tuple(answered_record if r is existing_record_for_chat else r for r in guided.history)
                        guided = _replace(
                            handler_result.session,
                            history=answered_history,
                            step_1_chosen_plugin=None,
                            step_1_source_intent=None,
                        )
                        emit_turn_answered(
                            recorder,
                            step=GuidedStep.STEP_1_SOURCE,
                            turn_type=TurnType.SCHEMA_FORM,
                            response_hash=response_hash,
                            response_payload_id=_store_guided_audit_payload(request.app.state.payload_store, turn_response),
                            control_signal=None,
                            composition_version=state.version,
                            actor=user.user_id,
                        )
                    else:
                        # SINGLE_SELECT / INSPECT_AND_CONFIRM entry resolved via chat:
                        # there is no widget answer to stamp (response_hash stays None —
                        # the chat IS the answer, recorded as a ChatTurn + emit events).
                        # But denormalize a DISPLAY summary onto the entry record so
                        # "Decisions so far" reads "Configured: <plugin>" instead of a
                        # bare "Decided". Display-only; NOT an answered-via-widget claim.
                        # existing_record_for_chat is non-None here (guarded at the top
                        # of this STEP_1 chat block).
                        summarized_entry = _replace(
                            existing_record_for_chat,
                            summary=f"Configured: {source_resolution.plugin}",
                        )
                        entry_history = tuple(summarized_entry if r is existing_record_for_chat else r for r in guided.history)
                        guided = _replace(
                            handler_result.session,
                            history=entry_history,
                            step_1_chosen_plugin=None,
                            step_1_source_intent=None,
                        )

                    # Apply-in-place: the source is committed (step_1_result set by
                    # handle_step_1_source); the phase STAYS STEP_1 so the user can
                    # revise by typing again. NO step advance, NO emit_step_advanced.
                    # Re-render the source schema_form POPULATED from the committed
                    # source (Task 2.5 builder) — the same turn GET /guided now emits
                    # for this state, so apply and refresh agree.
                    applied_step_1_result = handler_result.session.step_1_result
                    if applied_step_1_result is None:
                        raise InvariantError(
                            "step_1_result is None after successful handle_step_1_source — "
                            "handler set tool_result.success=True but did not set step_1_result"
                        )
                    next_turn = build_step_1_schema_form_turn_from_resolved(applied_step_1_result, catalog)
                    next_turn_type = TurnType(next_turn["type"])
                    next_payload_hash = stable_hash(next_turn["payload"])
                    new_record = TurnRecord(
                        step=GuidedStep.STEP_1_SOURCE,
                        turn_type=next_turn_type,
                        payload_hash=next_payload_hash,
                        response_hash=None,
                        emitter="server",
                    )
                    emit_turn_emitted(
                        recorder,
                        step=GuidedStep.STEP_1_SOURCE,
                        turn_type=next_turn_type,
                        payload_hash=next_payload_hash,
                        payload_payload_id=_store_guided_audit_payload(request.app.state.payload_store, next_turn["payload"]),
                        emitter="server",
                        composition_version=state.version,
                        actor=user.user_id,
                    )

                    ts_iso = finished_at.isoformat()
                    user_turn = ChatTurn(
                        role=ChatRole.USER,
                        content=body.message,
                        seq=guided.chat_turn_seq,
                        step=GuidedStep.STEP_1_SOURCE,
                        ts_iso=ts_iso,
                    )
                    assistant_turn = ChatTurn(
                        role=ChatRole.ASSISTANT,
                        content=source_resolution.assistant_message,
                        seq=guided.chat_turn_seq + 1,
                        step=GuidedStep.STEP_1_SOURCE,
                        ts_iso=ts_iso,
                        # Always a real reply: this is the resolve-and-commit
                        # success path.
                        assistant_message_kind="assistant",
                    )
                    guided = _replace(
                        guided,
                        history=(*guided.history, new_record),
                        chat_history=(*guided.chat_history, user_turn, assistant_turn),
                        chat_turn_seq=guided.chat_turn_seq + 2,
                    )

                    recorder.record_chat_turn(
                        ComposerChatTurn(
                            step=GuidedStep.STEP_1_SOURCE.value,
                            initiator=ComposerChatInitiator.USER,
                            chat_turn_seq=user_turn.seq,
                            user_message_hash=stable_hash(body.message),
                            assistant_message_hash=stable_hash(source_resolution.assistant_message),
                            latency_ms=latency_ms,
                            model=settings.composer_model,
                            status=ComposerChatTurnStatus.SUCCESS,
                            started_at=started_at,
                            finished_at=finished_at,
                            error_class=None,
                        )
                    )

                    response, state_record_out = await _build_guided_chat_apply_response(
                        guided=guided,
                        state=state,
                        next_turn=next_turn,
                        assistant_message=source_resolution.assistant_message,
                        service=service,
                        session_id=session_id,
                        state_record=state_record,
                        shield_available=shield_available,
                    )
                    return response

            elif guided.step is GuidedStep.STEP_2_SINK:
                sink_chat_result = await resolve_step_2_sink_chat_with_auto_drop(
                    site="post_guided_chat",
                    session_id=str(session_id),
                    user_id=user.user_id,
                    model=settings.composer_model,
                    user_message=body.message,
                    current_sink=guided.step_2_result,
                    temperature=settings.composer_temperature,
                    seed=settings.composer_seed,
                    recorder=recorder,
                    # Activate the sink discovery loop: the composer model can
                    # list_sinks / get_plugin_schema before it resolves.
                    state=state,
                    catalog=catalog,
                    secret_service=request.app.state.scoped_secret_resolver,
                    max_discovery_iters=settings.composer_max_discovery_turns,
                    timeout_seconds=settings.composer_timeout_seconds,
                    context_block=chat_context_block,
                    progress=progress_sink,
                )
                # ``prose_chat`` (a declined-to-prose SUCCESS) and
                # ``fallback_chat`` (an error) are mutually exclusive; either
                # one here means "use this directly, skip the second call".
                chat_result = sink_chat_result.fallback_chat or sink_chat_result.prose_chat
                sink_resolution = sink_chat_result.sink_resolution
                if sink_resolution is not None:
                    # The resolver only ever publishes calling_model/using_tools
                    # (both map to tutorial substep 1 — "Choose sink shape").
                    # Without a distinct event here the registry stays pinned
                    # on the resolver's last phase through the commit below AND
                    # through _build_guided_chat_apply_response's real DB write,
                    # so substep 2 ("Prepare JSON file") is unreachable while
                    # the pending strip is mounted — the finally block's
                    # "complete" publish lands after guidedChatPending has
                    # already flipped false and unmounted the strip. "saving"
                    # is a real observable signal (the sink IS being committed
                    # to session state right now), not a synthetic tick.
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session_id),
                        request_id=None,
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="saving",
                            headline="ELSPETH is saving the output configuration.",
                            evidence=("The resolved sink is being committed to the pipeline.",),
                            likely_next="ELSPETH will confirm the updated pipeline.",
                        ),
                    )
                    finished_at = datetime.now(UTC)
                    latency_ms = int((_perf_counter() - started_perf) * 1000)
                    data_dir = str(settings.data_dir) if settings.data_dir else None
                    handler_result = handle_step_2_sink(
                        state=state,
                        session=guided,
                        resolved=sink_resolution,
                        catalog=catalog,
                        data_dir=data_dir,
                    )
                    if not handler_result.tool_result.success:
                        # Non-actionable: the strict commit seam rejected the
                        # proposal. Fall back to advisory (no mutation).
                        chat_result = StepChatResult(
                            assistant_message=_COMMIT_REJECTED_MESSAGE,
                            status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
                            latency_ms=latency_ms,
                            error_class="StepHandlerRejected",
                        )
                        sink_resolution = None
                if sink_resolution is not None:
                    # Clear the STEP_2 staging fields on commit so the next GET
                    # hits the from-resolved sub-case (Task 2.5), not the empty
                    # chosen-plugin form. handle_step_2_sink sets step_2_result
                    # but does NOT clear these (verified steps.py:214). Mirrors
                    # the STEP_1 staging clear (Task 3 2c). No epoch bump.
                    guided = _replace(
                        handler_result.session,
                        step_2_chosen_plugin=None,
                        step_2_sink_intent=None,
                    )
                    state = handler_result.state
                    # Audit parity with STEP_1 (guided.py:1843-1861): if the prior
                    # STEP_2 turn was an answerable SCHEMA_FORM (the user was
                    # editing the sink form), stamp it answered so its TurnRecord
                    # does not stay response_hash=None forever. Gated on the turn
                    # type for the same reason STEP_1's leg is (a SINGLE_SELECT/
                    # MULTI_SELECT entry record has no schema-form answer to stamp).
                    if existing_record_for_chat is not None and current_turn_type is TurnType.SCHEMA_FORM:
                        sink_turn_response: TurnResponse = {
                            "chosen": None,
                            "edited_values": {
                                "plugin": sink_resolution.outputs[0].plugin,
                                "options": dict(sink_resolution.outputs[0].options),
                                "observed_columns": [],
                                "sample_rows": [],
                            },
                            "custom_inputs": None,
                            "accepted_step_index": None,
                            "edit_step_index": None,
                            "control_signal": None,
                        }
                        sink_response_hash = stable_hash(sink_turn_response)
                        sink_answered_record = _replace(
                            existing_record_for_chat,
                            response_hash=sink_response_hash,
                            summary=_summarize_guided_response(TurnType.SCHEMA_FORM, sink_turn_response),
                        )
                        guided = _replace(
                            guided,
                            history=tuple(sink_answered_record if r is existing_record_for_chat else r for r in guided.history),
                        )
                        emit_turn_answered(
                            recorder,
                            step=GuidedStep.STEP_2_SINK,
                            turn_type=TurnType.SCHEMA_FORM,
                            response_hash=sink_response_hash,
                            response_payload_id=_store_guided_audit_payload(request.app.state.payload_store, sink_turn_response),
                            control_signal=None,
                            composition_version=state.version,
                            actor=user.user_id,
                        )
                    elif existing_record_for_chat is not None:
                        # SINGLE_SELECT entry resolved via chat — display-only summary
                        # (response_hash stays None; the chat is the answer). Mirrors
                        # STEP_1's else leg so "Decisions so far" reads
                        # "Configured: <plugin>" instead of a bare "Decided".
                        summarized_entry = _replace(
                            existing_record_for_chat,
                            summary=f"Configured: {sink_resolution.outputs[0].plugin}",
                        )
                        guided = _replace(
                            guided,
                            history=tuple(summarized_entry if r is existing_record_for_chat else r for r in guided.history),
                        )
                    ts_iso = finished_at.isoformat()
                    # POPULATED re-render from the committed sink (Task 2.5
                    # builder) — same turn GET /guided emits for this state.
                    applied_step_2_result = handler_result.session.step_2_result
                    if applied_step_2_result is None:
                        raise InvariantError(
                            "step_2_result is None after successful handle_step_2_sink — "
                            "handler set tool_result.success=True but did not set step_2_result"
                        )
                    next_turn = build_step_2_schema_form_turn_from_resolved(applied_step_2_result, catalog)
                    next_turn_type = TurnType(next_turn["type"])
                    next_payload_hash = stable_hash(next_turn["payload"])
                    new_record = TurnRecord(
                        step=GuidedStep.STEP_2_SINK,
                        turn_type=next_turn_type,
                        payload_hash=next_payload_hash,
                        response_hash=None,
                        emitter="server",
                    )
                    emit_turn_emitted(
                        recorder,
                        step=GuidedStep.STEP_2_SINK,
                        turn_type=next_turn_type,
                        payload_hash=next_payload_hash,
                        payload_payload_id=_store_guided_audit_payload(request.app.state.payload_store, next_turn["payload"]),
                        emitter="server",
                        composition_version=state.version,
                        actor=user.user_id,
                    )
                    user_turn = ChatTurn(
                        role=ChatRole.USER,
                        content=body.message,
                        seq=guided.chat_turn_seq,
                        step=GuidedStep.STEP_2_SINK,
                        ts_iso=ts_iso,
                    )
                    assistant_message = sink_chat_result.assistant_message or "Output configured."
                    assistant_turn = ChatTurn(
                        role=ChatRole.ASSISTANT,
                        content=assistant_message,
                        seq=guided.chat_turn_seq + 1,
                        step=GuidedStep.STEP_2_SINK,
                        ts_iso=ts_iso,
                        # Always a real reply: this is the resolve-and-commit
                        # success path.
                        assistant_message_kind="assistant",
                    )
                    guided = _replace(
                        guided,
                        history=(*guided.history, new_record),
                        chat_history=(*guided.chat_history, user_turn, assistant_turn),
                        chat_turn_seq=guided.chat_turn_seq + 2,
                    )
                    recorder.record_chat_turn(
                        ComposerChatTurn(
                            step=GuidedStep.STEP_2_SINK.value,
                            initiator=ComposerChatInitiator.USER,
                            chat_turn_seq=user_turn.seq,
                            user_message_hash=stable_hash(body.message),
                            assistant_message_hash=stable_hash(assistant_message),
                            latency_ms=latency_ms,
                            model=settings.composer_model,
                            status=ComposerChatTurnStatus.SUCCESS,
                            started_at=started_at,
                            finished_at=finished_at,
                            error_class=None,
                        )
                    )
                    response, state_record_out = await _build_guided_chat_apply_response(
                        guided=guided,
                        state=state,
                        next_turn=next_turn,
                        assistant_message=assistant_message,
                        service=service,
                        session_id=session_id,
                        state_record=state_record,
                        shield_available=shield_available,
                    )
                    return response

            elif guided.step is GuidedStep.STEP_3_TRANSFORMS:
                if guided.step_1_result is None or guided.step_2_result is None:
                    # Cannot propose a chain without a committed source + sink;
                    # fall through to advisory prose (no mutation).
                    chat_result = None
                else:
                    from elspeth.web.composer.guided.chain_solver import solve_chain
                    from elspeth.web.composer.guided.errors import ChainSolverResponseShapeError

                    try:
                        # The transient set MUST be byte-identical to the
                        # auto-drop wrapper's except at
                        # `_guided_solve_chain.py:170-183` — that wrapper wraps a
                        # direct `solve_chain` call, so its set is the proven set
                        # of what `solve_chain` propagates. All 13 (8 typed +
                        # IndexError/AttributeError/json.JSONDecodeError +
                        # ChainSolverResponseShapeError). A GuardrailRaisedException
                        # (expected on a Tier-3 user free-text revise) or
                        # OpenAIError/BudgetExceededError that escaped the tuple
                        # would 500 and brick the phase — violating the
                        # advisory-never-blocks contract this branch exists to honour.
                        from litellm.exceptions import APIError as _LLMAPIError
                        from litellm.exceptions import AuthenticationError as _LLMAuthError
                        from litellm.exceptions import BadRequestError as _LLMBadReq
                        from litellm.exceptions import (
                            BlockedPiiEntityError as _LLMBlockedPii,
                        )
                        from litellm.exceptions import (
                            BudgetExceededError as _LLMBudgetExceeded,
                        )
                        from litellm.exceptions import (
                            GuardrailInterventionNormalStringError as _LLMGuardrailNormalString,
                        )
                        from litellm.exceptions import (
                            GuardrailRaisedException as _LLMGuardrailRaised,
                        )
                        from litellm.exceptions import OpenAIError as _LLMOpenAIError

                        # revise_context flips solve_chain's prompt to the
                        # REVISE addendum (build_revise_addendum) which frames the
                        # user's text as "update the current proposal", NOT as a
                        # validation-failure repair. Pass the user's revise
                        # instruction as revise_context, never repair_context
                        # (repair_context stays reserved for the genuine
                        # validation-repair loop on /guided/respond). On the FIRST
                        # STEP_3 chat (no prior proposal) there is nothing to
                        # revise yet, so the user's message rides as ``intent`` —
                        # a cold start is a plain initial propose built FROM the
                        # request (the freeform mirror), not a revision.
                        if guided.step_3_proposal is not None:
                            revise_context = body.message
                            intent = None
                        else:
                            revise_context = None
                            intent = body.message
                        proposal = await solve_chain(
                            model=settings.composer_model,
                            source=guided.step_1_result,
                            sink=guided.step_2_result,
                            intent=intent,
                            revise_context=revise_context,
                            recorder=recorder,
                            temperature=settings.composer_temperature,
                            seed=settings.composer_seed,
                            # Discovery loop: the STEP_3 chat revise can look up real
                            # transform plugins + schemas before re-proposing.
                            state=state,
                            catalog=catalog,
                            secret_service=request.app.state.scoped_secret_resolver,
                            user_id=user.user_id,
                            max_discovery_iters=settings.composer_max_discovery_turns,
                            timeout_seconds=settings.composer_timeout_seconds,
                        )
                    except (
                        _LLMAPIError,
                        _LLMAuthError,
                        _LLMBadReq,
                        _LLMBudgetExceeded,
                        _LLMBlockedPii,
                        _LLMGuardrailRaised,
                        _LLMGuardrailNormalString,
                        _LLMOpenAIError,
                        TimeoutError,
                        IndexError,
                        AttributeError,
                        json.JSONDecodeError,
                        ChainSolverResponseShapeError,
                    ):
                        # asyncio.CancelledError is deliberately NOT in the set
                        # (client-disconnect cancellation must propagate), matching
                        # the auto-drop wrapper's docstring at _guided_solve_chain.py.
                        # Non-load-bearing: a transient solve failure on the
                        # STEP_3 chat path must NOT terminate the session (that
                        # is what solve_chain_with_auto_drop would do). Fall back
                        # to advisory prose with NO mutation.
                        proposal = None
                    if proposal is None:
                        chat_result = None  # advisory fall-through
                    else:
                        finished_at = datetime.now(UTC)
                        latency_ms = int((_perf_counter() - started_perf) * 1000)
                        # Record the transient proposal for in-place re-render.
                        # NOT committed: handle_step_3_chain_accept (commit +
                        # advance to WIRE) stays on /guided/respond.
                        # Clear step_3_edit_index in the SAME atomic replace that
                        # installs the replacement proposal: an edit staged
                        # against the prior (possibly longer) proposal can point
                        # past the end of a shorter revised chain, and a stale
                        # index left dangling makes GET /guided's rebuild index
                        # out of range for the new proposal's steps.
                        guided = _replace(guided, step_3_proposal=proposal, step_3_edit_index=None)
                        state = _replace(state, guided_session=guided)
                        next_turn = build_step_3_propose_chain_turn(proposal)
                        next_turn_type = TurnType(next_turn["type"])
                        next_payload_hash = stable_hash(next_turn["payload"])
                        new_record = TurnRecord(
                            step=GuidedStep.STEP_3_TRANSFORMS,
                            turn_type=next_turn_type,
                            payload_hash=next_payload_hash,
                            response_hash=None,
                            emitter="server",
                        )
                        emit_turn_emitted(
                            recorder,
                            step=GuidedStep.STEP_3_TRANSFORMS,
                            turn_type=next_turn_type,
                            payload_hash=next_payload_hash,
                            payload_payload_id=_store_guided_audit_payload(request.app.state.payload_store, next_turn["payload"]),
                            emitter="server",
                            composition_version=state.version,
                            actor=user.user_id,
                        )
                        ts_iso = finished_at.isoformat()
                        assistant_message = "Here is an updated proposal."
                        user_turn = ChatTurn(
                            role=ChatRole.USER,
                            content=body.message,
                            seq=guided.chat_turn_seq,
                            step=GuidedStep.STEP_3_TRANSFORMS,
                            ts_iso=ts_iso,
                        )
                        assistant_turn = ChatTurn(
                            role=ChatRole.ASSISTANT,
                            content=assistant_message,
                            seq=guided.chat_turn_seq + 1,
                            step=GuidedStep.STEP_3_TRANSFORMS,
                            ts_iso=ts_iso,
                            # Always a real reply: this is the propose_chain
                            # success path.
                            assistant_message_kind="assistant",
                        )
                        guided = _replace(
                            guided,
                            history=(*guided.history, new_record),
                            chat_history=(*guided.chat_history, user_turn, assistant_turn),
                            chat_turn_seq=guided.chat_turn_seq + 2,
                        )
                        recorder.record_chat_turn(
                            ComposerChatTurn(
                                step=GuidedStep.STEP_3_TRANSFORMS.value,
                                initiator=ComposerChatInitiator.USER,
                                chat_turn_seq=user_turn.seq,
                                user_message_hash=stable_hash(body.message),
                                assistant_message_hash=stable_hash(assistant_message),
                                latency_ms=latency_ms,
                                model=settings.composer_model,
                                status=ComposerChatTurnStatus.SUCCESS,
                                started_at=started_at,
                                finished_at=finished_at,
                                error_class=None,
                            )
                        )
                        response, state_record_out = await _build_guided_chat_apply_response(
                            guided=guided,
                            state=state,
                            next_turn=next_turn,
                            assistant_message=assistant_message,
                            service=service,
                            session_id=session_id,
                            state_record=state_record,
                            shield_available=shield_available,
                        )
                        return response

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
            if chat_result is None:
                try:
                    chat_result = await solve_step_chat_with_auto_drop(
                        site="post_guided_chat",
                        session_id=str(session_id),
                        user_id=user.user_id,
                        model=settings.composer_model,
                        step=guided.step,
                        user_message=body.message,
                        temperature=settings.composer_temperature,
                        seed=settings.composer_seed,
                        recorder=recorder,
                        timeout_seconds=settings.composer_timeout_seconds,
                        # LLM-safe current-build context so "explain what I'm
                        # seeing / why" gets a grounded answer (the decision
                        # card's Explain affordance rides this same path) —
                        # the SAME block already computed above for the
                        # STEP_1/STEP_2 resolve calls.
                        context_block=chat_context_block,
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
                # Real reply (advisory SUCCESS or a salvaged declined-prose
                # reply) or a synthetic failure — derive both from the same
                # status/error_class the wire response's kind uses.
                assistant_message_kind=_guided_chat_wire_kind(chat_result.status),
                synthetic_failure_reason=_chat_turn_synthetic_failure_reason(chat_result.status, chat_result.error_class),
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
            persisted_is_valid, persisted_errors = _guided_persisted_validity(new_state)
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
            state_record_out = await service.save_composition_state(
                session_id,
                state_data,
                # Per-step chat persists guided-session metadata
                # (chat_history/chat_turn_seq) after the LLM response has
                # converged. The closed provenance enum has no guided-chat
                # category, so use the same closest category as sibling guided
                # state writes rather than widening the closed list mid-merge.
                provenance="convergence_persist",
            )

            return GuidedChatResponse(
                assistant_message=chat_result.assistant_message,
                assistant_message_kind=_guided_chat_wire_kind(chat_result.status),
                guided_session=GuidedSessionResponse(
                    step=new_guided.step.value,
                    history=[
                        TurnRecordResponse(
                            step=r.step.value,
                            turn_type=r.turn_type.value,
                            payload_hash=r.payload_hash,
                            response_hash=r.response_hash,
                            summary=r.summary,
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
                            assistant_message_kind=t.assistant_message_kind,
                            synthetic_failure_reason=t.synthetic_failure_reason,
                        )
                        for t in new_guided.chat_history
                    ],
                    chat_turn_seq=new_guided.chat_turn_seq,
                    profile=_workflow_profile_response(new_guided),
                ),
                next_turn=None,
                terminal=None,
                composition_state=_state_response(state_record_out),
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
            if progress_started:
                # Published FIRST, before the audit-drain below — deliberately
                # NOT the same ordering as freeform's send_message
                # (messages.py:731 publishes its "complete" event AFTER
                # _persist_tool_invocations/_persist_llm_calls, because
                # messages.py builds and returns its response after those
                # persists too). Here every step branch above already built
                # and returned its response inline; only the audit drain
                # remains in this finally block, so an AuditIntegrityError
                # raised out of _persist_chat_turns on the success path must
                # not strand the frontend poller on a stale non-terminal
                # phase — hence publishing ahead of the drain, not after it.
                if primary_exc is None:
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session_id),
                        request_id=None,
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="complete",
                            headline="ELSPETH finished responding to this chat message.",
                            evidence=("The guided chat turn finished.",),
                            likely_next="Review the reply and continue the wizard.",
                            reason="composer_complete",
                        ),
                    )
                elif isinstance(primary_exc, asyncio.CancelledError):
                    # asyncio.shield, exactly like messages.py's
                    # client_cancelled publish (messages.py:797-821): the
                    # task is being torn down, so a bare await here risks
                    # the registry update never landing; shield lets the
                    # publish run to completion in the background while the
                    # suppress absorbs the CancelledError shield() re-raises
                    # on the cancelling task.
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(
                            _publish_progress(
                                progress_registry,
                                session_id=str(session_id),
                                request_id=None,
                                user_id=str(user.user_id),
                                event=client_cancelled_progress_event(),
                            )
                        )
                else:
                    # Guided chat degrades almost every failure mode to a
                    # 200 synthetic-unavailable reply inside the resolver
                    # (see post_guided_chat's docstring); reaching here means
                    # an unexpected exception genuinely escaped, so this is a
                    # rare defensive backstop, not the expected path.
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session_id),
                        request_id=None,
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="failed",
                            headline="The guided chat could not finish this request.",
                            evidence=("An unexpected error interrupted this chat turn.",),
                            likely_next="Review the visible error message, then retry.",
                            reason="service_setup_failed",
                        ),
                    )
            if primary_exc is None:
                await _persist_tool_invocations(
                    service,
                    session_id,
                    recorder.invocations,
                    state_record_out.id if state_record_out is not None else None,
                    plugin_crash_pending=False,
                )
                await _persist_llm_calls(
                    service,
                    session_id,
                    recorder.llm_calls,
                    state_record_out.id if state_record_out is not None else None,
                    plugin_crash_pending=False,
                )
                await _persist_chat_turns(
                    service,
                    session_id,
                    recorder.chat_turns,
                    state_record_out.id if state_record_out is not None else None,
                    request_unwinding=False,
                )
            else:
                # Unwind path: a primary exception is in flight (this is the
                # ``finally`` block). ``plugin_crash_pending`` asks "are we
                # unwinding from a primary failure?", NOT "did a plugin
                # crash" — here the answer is yes. True selects the helper's
                # record-and-continue disposition (unwind counter + slog) so
                # an audit-persist failure does NOT raise AuditIntegrityError
                # and mask the primary failure. Mirrors the
                # ``request_unwinding=True`` passed to _persist_chat_turns
                # below in this same branch.
                try:
                    await _persist_tool_invocations(
                        service,
                        session_id,
                        recorder.invocations,
                        state_record_out.id if state_record_out is not None else None,
                        plugin_crash_pending=True,
                    )
                except Exception as persist_exc:
                    with contextlib.suppress(Exception):
                        slog.error(
                            "guided.audit_persist_failed_during_exception_handling",
                            session_id=str(session_id),
                            user_id=user.user_id,
                            site="post_guided_chat",
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
                            site="post_guided_chat",
                            channel="llm_calls",
                            exc_class=type(persist_exc).__name__,
                            frames=_safe_frame_strings(persist_exc),
                        )
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
      * no persisted state (empty session) -> lazy fresh wizard, NON-persisting
        (identical to GET /guided's lazy path; an empty graph must not open the
        version history).
      * ``guided_session`` already present -> return it UNCHANGED, including any
        terminal (so a completed / solver-exhausted / protocol-violation surface
        still renders — enterGuided routes those non-exit terminals here).
      * persisted state with ``guided_session is None`` -> the conversion.

    Raises 404 if the session does not exist or belong to the requesting user.
    """
    await _verify_session_ownership(session_id, user, request)
    service: SessionServiceProtocol = request.app.state.session_service
    catalog: CatalogServiceProtocol = request.app.state.catalog_service

    compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))
    async with compose_lock:
        state_record = await service.get_current_state(session_id)

        # Branch 2: already guided (idempotent double-click, cross-tab race, or a
        # terminal session reached via enterGuided's non-exit else-branch).
        # Return the existing session UNCHANGED — never reseed. Mirrors
        # post_guided_start's idempotency block, including the terminal payload
        # and the ``next_turn=None`` snapshot semantics.
        if state_record is not None:
            existing_state = _state_from_record(state_record)
            if existing_state.guided_session is not None:
                guided = existing_state.guided_session
                terminal = guided.terminal
                # Rebuild the live turn for a non-terminal step so a double-click
                # / cross-tab "Switch to guided" that lands here returns the SAME
                # active turn GET /guided would (elspeth-e2c3dba6b5 review P2).
                # Reserve next_turn=None only for terminal sessions and steps with
                # no rebuildable initial turn (STEP_3, no-recipe STEP_2_5) — for
                # those _build_get_guided_turn returns None, matching GET's
                # turn/None contract exactly. Without this the frontend keeps
                # guidedSession but drops guidedNextTurn, isGuidedBuildActive goes
                # false, and ChatPanel falls back to the freeform surface.
                if terminal is None:
                    try:
                        turn = _build_get_guided_turn(existing_state, guided, catalog=catalog)
                    except InvariantError as exc:
                        # Same B1-sanitization rationale as get_guided: str(exc)
                        # can embed {d!r} of a corrupted Tier-1 record including
                        # Tier-3 sample_rows. Static detail; slog carries the
                        # exc_class + frames only.
                        slog.error(
                            "guided.invariant_violated",
                            session_id=str(session_id),
                            user_id=user.user_id,
                            exc_class=type(exc).__name__,
                            site="post_guided_convert._build_get_guided_turn",
                            frames=_safe_frame_strings(exc),
                        )
                        raise HTTPException(
                            status_code=500,
                            detail="Server invariant violated. See application audit log for diagnostic detail.",
                        ) from exc
                else:
                    turn = None
                shield_available = _resolve_shield_available(request, user.user_id)
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
                    next_turn=_turn_payload_response(turn, shield_available=shield_available),
                    terminal=TerminalStateResponse(
                        kind=terminal.kind.value,
                        reason=terminal.reason.value if terminal.reason is not None else None,
                        pipeline_yaml=terminal.pipeline_yaml,
                    )
                    if terminal is not None
                    else None,
                    composition_state=_state_response(state_record),
                )

        # Branches 1 & 3: seed a FRESH guided wizard.
        new_state = _initial_composition_state_with_guided_session()
        seeded_guided = new_state.guided_session
        if seeded_guided is None:  # pragma: no cover — helper always attaches a guided session
            raise InvariantError("post_guided_convert: initial state has no guided_session")
        turn = _build_get_guided_turn(new_state, seeded_guided, catalog=catalog)

        state_record_out: CompositionStateRecord | None
        if state_record is None:
            # Branch 1: empty session — lazy, non-persisting (mirror GET /guided).
            # The fresh wizard is returned in memory; the first real answer
            # persists it. An empty graph must not open the version history.
            state_record_out = None
        else:
            # Branch 3: THE CONVERSION. Reseed a fresh wizard as a NEW version,
            # setting the freeform graph aside. This is user-consented (two-step
            # confirm) and fully recoverable: the prior freeform version stays in
            # state/versions and revert restores it (composer_meta copied
            # verbatim -> back to freeform). ``session_seed`` provenance is reused
            # deliberately — the closed CompositionStateProvenance enum has no
            # guided-convert value and widening it is a governance action
            # (protocol.py), out of scope for this fix; the audit trail for the
            # graph-discard is the system message emitted just below.
            prior_version = state_record.version
            new_composer_meta = {"guided_session": seeded_guided.to_dict()}
            state_d = new_state.to_dict()
            persisted_is_valid, persisted_errors = _guided_persisted_validity(new_state)
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
            state_record_out = await service.save_composition_state(
                session_id,
                state_data,
                provenance="session_seed",
            )
            await service.add_message(
                session_id,
                role="system",
                content=(
                    "Switched to guided mode with a fresh wizard. Your previous "
                    f"freeform pipeline is saved as version {prior_version} and can "
                    "be restored from version history."
                ),
                writer_principal="route_system_message",
            )

        shield_available = _resolve_shield_available(request, user.user_id)
        return GetGuidedResponse(
            guided_session=GuidedSessionResponse(
                step=seeded_guided.step.value,
                history=[],
                terminal=None,
                chat_history=[],
                chat_turn_seq=seeded_guided.chat_turn_seq,
                profile=_workflow_profile_response(seeded_guided),
            ),
            next_turn=_turn_payload_response(turn, shield_available=shield_available),
            terminal=None,
            composition_state=_state_response(state_record_out) if state_record_out is not None else None,
        )
