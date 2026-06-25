from __future__ import annotations

from elspeth.web.composer.guided.profile import WorkflowProfileKind, profile_for_kind
from elspeth.web.composer.protocol import ComposerService
from elspeth.web.sessions.schemas import StartGuidedRequest

from .._helpers import (
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
    TurnResponse,
    TurnType,
    UserIdentity,
    _dispatch_guided_respond,
    _get_session_compose_lock_registry,
    _initial_composition_state_with_guided_session,
    _materialize_profile_entry_seed_state,
    _persist_chat_turns,
    _persist_llm_calls,
    _persist_tool_invocations,
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
    build_step_2_5_recipe_offer_turn,
    build_step_2_multi_select_turn,
    build_step_2_schema_form_turn,
    build_step_2_schema_form_turn_from_resolved,
    build_step_2_single_select_turn,
    build_step_3_propose_chain_turn,
    build_step_3_schema_form_turn,
    build_step_4_wire_turn,
    contextlib,
    datetime,
    deep_thaw,
    emit_dropped_to_freeform,
    emit_step_advanced,
    emit_turn_answered,
    emit_turn_emitted,
    get_current_user,
    handle_step_1_source,
    match_recipe,
    resolve_step_1_source_chat_with_auto_drop,
    slog,
    solve_step_chat_with_auto_drop,
    stable_hash,
    step_advance,
    sys,
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

    - STEP_2_5_RECIPE_MATCH: deterministically re-run ``match_recipe``; if a
      match is found, emit ``recipe_offer``.  Returns ``None`` when no
      recipe matched (session stays at this step but no turn exists).

    - STEP_3_TRANSFORMS: if ``step_3_proposal`` is set, emit
      ``propose_chain`` from the staged proposal, unless
      ``step_3_edit_index`` is set, in which case emit the transform
      ``schema_form`` for the proposed step being revised.  Returns ``None``
      if the proposal is absent (LLM call has not completed; should
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
        # A recipe_offer TurnRecord exists iff a recipe was matched.
        # Reconstruct by re-running match_recipe (deterministic).
        if guided.step_1_result is not None and guided.step_2_result is not None:
            recipe_match = match_recipe(guided.step_1_result, guided.step_2_result)
            if recipe_match is not None:
                return build_step_2_5_recipe_offer_turn(recipe_match)
        # No recipe matched — no initial turn for this step.
        return None
    if step is GuidedStep.STEP_3_TRANSFORMS:
        if guided.step_3_proposal is not None:
            if guided.step_3_edit_index is not None:
                step_record = dict(guided.step_3_proposal.steps[guided.step_3_edit_index])
                plugin = step_record["plugin"]
                options = step_record["options"]
                if type(plugin) is not str or type(options) is not dict:
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

    if current_step is GuidedStep.STEP_2_5_RECIPE_MATCH and guided.step_1_result is not None and guided.step_2_result is not None:
        staged_offer = match_recipe(guided.step_1_result, guided.step_2_result)
    else:
        staged_offer = None

    new_guided = _replace(guided, history=(*guided.history, new_record))
    if staged_offer is not None:
        new_guided = _replace(new_guided, step_2_5_recipe_offer=staged_offer)
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
                payload_payload_id = _store_guided_audit_payload(getattr(request.app.state, "payload_store", None), turn["payload"])
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
                state_data = CompositionStateData(
                    sources=state_d["sources"],
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
                        sources=repaired_state_d["sources"],
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
                        )
                        for t in guided.chat_history
                    ],
                    chat_turn_seq=guided.chat_turn_seq,
                    profile=_workflow_profile_response(guided),
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
        state_data = CompositionStateData(
            sources=state_d["sources"],
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
            provenance="convergence_persist",
        )

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
                    )
                    for t in new_guided.chat_history
                ],
                chat_turn_seq=new_guided.chat_turn_seq,
                profile=_workflow_profile_response(new_guided),
            ),
            next_turn=TurnPayloadResponse(
                type=turn["type"],
                step_index=turn["step_index"],
                payload=dict(turn["payload"]),
            ),
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

    Decision D: ``start`` persists the profile only — it does NOT materialize
    ``profile.entry_seed`` into the CompositionState. ``entry_seed`` is a
    server-side ``str`` framing prompt that stays on the persisted profile for
    the P7.5 welcome bookend to render; the concrete pipeline is built downstream
    by the guided wizard + web-scrape recipe match. ``entry_seed`` never rides
    the request or response wire.

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
    # carrying attacker-controlled fields such as entry_seed.
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
                            )
                            for t in guided.chat_history
                        ],
                        chat_turn_seq=guided.chat_turn_seq,
                        profile=_workflow_profile_response(guided),
                    ),
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

        # No persisted guided session yet: construct the profile-seeded
        # initial state and PERSIST it (so GET /api/sessions/{session_id}/guided
        # reads the profile back). Decision D: this attaches the server-owned
        # profile only — it does NOT materialize profile.entry_seed into the
        # CompositionState (entry_seed is a server-side str framing prompt that
        # stays on the persisted profile for the P7.5 welcome bookend to render;
        # the concrete pipeline is wizard/recipe-built downstream). For the live
        # profile this is the existing empty guided state.
        new_state = _materialize_profile_entry_seed_state(profile)
        seeded_guided = new_state.guided_session
        if seeded_guided is None:  # pragma: no cover — helper always attaches a guided session
            raise InvariantError("post_guided_start: initial state has no guided_session")
        turn = _build_get_guided_turn(new_state, seeded_guided, catalog=catalog)

        new_composer_meta = {"guided_session": seeded_guided.to_dict()}
        state_d = new_state.to_dict()
        state_data = CompositionStateData(
            sources=state_d["sources"],
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
            # Start endpoint seeds the canonical guided session for a profile;
            # ``session_seed`` is the closest existing provenance category for
            # a fresh server-authored seed state (the closed enum has no
            # guided-specific value — see merge commit message).
            provenance="session_seed",
        )

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
                    )
                    for t in seeded_guided.chat_history
                ],
                chat_turn_seq=seeded_guided.chat_turn_seq,
                profile=_workflow_profile_response(seeded_guided),
            ),
            next_turn=TurnPayloadResponse(
                type=turn["type"],
                step_index=turn["step_index"],
                payload=dict(turn["payload"]),
            )
            if turn is not None
            else None,
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
                state_data = CompositionStateData(
                    sources=state_d["sources"],
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
                    raise HTTPException(
                        status_code=400,
                        detail=("No turn has been emitted for the current step. Fetch GET /api/sessions/{id}/guided first."),
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
                    payload_payload_id=_store_guided_audit_payload(getattr(request.app.state, "payload_store", None), turn["payload"]),
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
                response_payload_id=_store_guided_audit_payload(getattr(request.app.state, "payload_store", None), turn_response),
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
                        blob_service=request.app.state.blob_service,
                        payload_store=getattr(request.app.state, "payload_store", None),
                        model=settings.composer_model,
                        temperature=settings.composer_temperature,
                        seed=settings.composer_seed,
                        composer_service=request.app.state.composer_service,
                        advisor_checkpoint_max_passes=settings.composer_advisor_checkpoint_max_passes,
                    )
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
                    state_data_error = CompositionStateData(
                        sources=state_d_error["sources"],
                        nodes=state_d_error["nodes"],
                        edges=state_d_error["edges"],
                        outputs=state_d_error["outputs"],
                        metadata_=state_d_error["metadata"],
                        is_valid=False,
                        validation_errors=None,
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
            state_data = CompositionStateData(
                sources=state_d["sources"],
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
                        )
                        for t in guided.chat_history
                    ],
                    chat_turn_seq=guided.chat_turn_seq,
                    profile=_workflow_profile_response(guided),
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
            existing_record_for_chat: TurnRecord | None = next(
                (r for r in reversed(guided.history) if r.step == guided.step),
                None,
            )
            current_turn_type = existing_record_for_chat.turn_type if existing_record_for_chat is not None else None
            chat_result = None

            if (
                existing_record_for_chat is not None
                and guided.step is GuidedStep.STEP_1_SOURCE
                and current_turn_type in (TurnType.SINGLE_SELECT, TurnType.SCHEMA_FORM, TurnType.INSPECT_AND_CONFIRM)
            ):
                plugin_hint: str | None = None
                selected_record = next(
                    (
                        r
                        for r in reversed(guided.history)
                        if r.step is GuidedStep.STEP_1_SOURCE
                        and r.turn_type is TurnType.SINGLE_SELECT
                        and r.summary is not None
                        and r.summary.startswith("Selected: ")
                    ),
                    None,
                )
                if selected_record is not None and selected_record.summary is not None:
                    plugin_hint = selected_record.summary.removeprefix("Selected: ").split(", ", 1)[0]

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
                )
                chat_result = source_chat_result.fallback_chat
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
                        options={**dict(source_resolution.options), "path": source_blob.storage_path},
                        observed_columns=source_resolution.observed_columns,
                        sample_rows=source_resolution.sample_rows,
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
                        raise HTTPException(
                            status_code=400,
                            detail="Step 1 source commit failed",
                        )

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
                            response_payload_id=_store_guided_audit_payload(
                                getattr(request.app.state, "payload_store", None), turn_response
                            ),
                            control_signal=None,
                            composition_version=state.version,
                            actor=user.user_id,
                        )
                    else:
                        # SINGLE_SELECT / INSPECT_AND_CONFIRM entry: no prior
                        # schema-form answer to stamp. Just adopt the committed session
                        # and clear the staging fields.
                        guided = _replace(
                            handler_result.session,
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
                        payload_payload_id=_store_guided_audit_payload(
                            getattr(request.app.state, "payload_store", None), next_turn["payload"]
                        ),
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

                    new_state = _replace(state, guided_session=guided)
                    source_existing_meta: dict[str, Any] = {}
                    if state_record is not None and state_record.composer_meta is not None:
                        source_existing_meta = dict(deep_thaw(state_record.composer_meta))
                    new_composer_meta = {**source_existing_meta, "guided_session": guided.to_dict()}

                    state_d = new_state.to_dict()
                    state_data = CompositionStateData(
                        sources=state_d["sources"],
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
                        provenance="convergence_persist",
                    )

                    return GuidedChatResponse(
                        assistant_message=source_resolution.assistant_message,
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
                                )
                                for t in guided.chat_history
                            ],
                            chat_turn_seq=guided.chat_turn_seq,
                            profile=_workflow_profile_response(guided),
                        ),
                        next_turn=TurnPayloadResponse(
                            type=next_turn["type"],
                            step_index=next_turn["step_index"],
                            payload=dict(next_turn["payload"]),
                        ),
                        terminal=None,
                        composition_state=_state_response(state_record_out),
                    )

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
                sources=state_d["sources"],
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
                # converged. The closed provenance enum has no guided-chat
                # category, so use the same closest category as sibling guided
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
