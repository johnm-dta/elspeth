from __future__ import annotations

from typing import TypedDict

from ._helpers import (
    _COMPOSER_REQUESTS_INFLIGHT,
    _DATA_ERROR_KEY,
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
    ComposerConvergenceError,
    ComposerPluginCrashError,
    ComposerPreferencesResponse,
    ComposerProgressEvent,
    ComposerProgressSnapshot,
    ComposerRateLimiter,
    ComposerRuntimePreflightError,
    ComposerService,
    ComposerServiceError,
    CompositionProposalRecord,
    CompositionProposalResponse,
    CompositionStateData,
    CompositionStateRecord,
    CompositionStateResponse,
    ControlSignal,
    Depends,
    GetGuidedResponse,
    GraphValidationError,
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
    MessageWithStateResponse,
    PluginConfigError,
    PluginNotFoundError,
    ProposalEventResponse,
    ProposalLifecycleStatus,
    Query,
    RejectProposalRequest,
    Request,
    RevertStateRequest,
    SessionServiceProtocol,
    SessionsTelemetry,
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
    UpdateComposerPreferencesRequest,
    UserIdentity,
    _BadRequestLLMError,
    _composer_chat_history,
    _composer_conversation_messages,
    _composer_preferences_response,
    _composer_progress_sink,
    _ComposerRequestTerminalStatus,
    _composition_proposal_response,
    _dispatch_guided_respond,
    _get_composer_progress_registry,
    _get_session_compose_lock_registry,
    _handle_convergence_error,
    _handle_plugin_crash,
    _handle_runtime_preflight_failure,
    _initial_composition_state_with_guided_session,
    _litellm_error_detail,
    _llm_calls_from_exception,
    _message_response,
    _pending_proposal_responses,
    _persist_chat_turns,
    _persist_llm_calls,
    _persist_tool_invocations,
    _proposal_event_response,
    _publish_progress,
    _record_composer_request_terminal,
    _record_composer_runtime_preflight_telemetry,
    _replace,
    _runtime_preflight_for_state,
    _safe_frame_strings,
    _state_data_from_composer_state,
    _state_from_record,
    _state_response,
    _summarize_guided_response,
    _validate_control_signal,
    _validate_step_indices,
    _verify_session_ownership,
    asyncio,
    build_initial_step_1_turn,
    build_step_1_inspect_and_confirm_turn_from_intent,
    build_step_1_schema_form_turn,
    build_step_2_5_recipe_offer_turn,
    build_step_2_multi_select_turn,
    build_step_2_schema_form_turn,
    build_step_2_single_select_turn,
    build_step_3_propose_chain_turn,
    build_step_3_schema_form_turn,
    cast,
    client_cancelled_progress_event,
    composer_completion_events_table,
    contextlib,
    convergence_progress_event,
    datetime,
    deep_thaw,
    emit_dropped_to_freeform,
    emit_step_advanced,
    emit_turn_answered,
    emit_turn_emitted,
    execute_tool,
    generate_yaml,
    get_current_user,
    get_rate_limiter,
    handle_step_1_source,
    insert,
    match_recipe,
    maybe_resolve_step_1_source_chat,
    record_session_completed,
    record_session_switched,
    run_sync_in_worker,
    slog,
    solve_step_chat_with_auto_drop,
    stable_hash,
    step_advance,
    sys,
    uuid4,
)

_PROPOSAL_COMPOSER_CONTEXT_FIELDS: tuple[str, ...] = (
    "composer_model_identifier",
    "composer_model_version",
    "composer_provider",
    "composer_skill_hash",
    "tool_arguments_hash",
)


def _inline_blob_content_for_proposal(
    proposal: CompositionProposalRecord,
    arguments: Mapping[str, Any],
) -> str | None:
    """Return inline blob content that accept replay would persist, if any."""
    if proposal.tool_name != "set_pipeline":
        return None
    source = arguments.get("source")
    if not isinstance(source, Mapping):
        return None
    inline_blob = source.get("inline_blob")
    if not isinstance(inline_blob, Mapping):
        return None
    content = inline_blob.get("content")
    return content if isinstance(content, str) else None


async def _proposal_user_message_content(
    service: SessionServiceProtocol,
    proposal: CompositionProposalRecord,
) -> str | None:
    """Recover the immutable originating user-message body for accept replay."""
    if proposal.user_message_id is None:
        return None
    messages = await service.get_messages(proposal.session_id, limit=None)
    for message in messages:
        if message.id != proposal.user_message_id:
            continue
        if message.role != "user":
            raise HTTPException(
                status_code=409,
                detail="Stored proposal references a non-user originating message; ask ELSPETH to regenerate the proposal.",
            )
        return message.content
    raise HTTPException(
        status_code=409,
        detail="Stored proposal references an originating message that could not be recovered; ask ELSPETH to regenerate the proposal.",
    )


def _missing_proposal_composer_context(
    proposal: CompositionProposalRecord,
    *,
    user_message_content: str | None,
) -> tuple[str, ...]:
    missing = [name for name in _PROPOSAL_COMPOSER_CONTEXT_FIELDS if getattr(proposal, name) is None]
    if user_message_content is None:
        missing.insert(0, "user_message_content")
    return tuple(missing)


def _ensure_inline_blob_proposal_context(
    proposal: CompositionProposalRecord,
    arguments: Mapping[str, Any],
    *,
    user_message_content: str | None,
) -> None:
    inline_blob_content = _inline_blob_content_for_proposal(proposal, arguments)
    if inline_blob_content is None:
        return
    if user_message_content is not None and inline_blob_content and inline_blob_content in user_message_content:
        return
    missing = _missing_proposal_composer_context(proposal, user_message_content=user_message_content)
    if not missing:
        return
    raise HTTPException(
        status_code=409,
        detail=(
            "Accepted proposal is missing composer provenance required for inline-blob source writes "
            f"({', '.join(missing)}). Ask ELSPETH to regenerate the proposal."
        ),
    )


class StateYamlResponse(TypedDict, total=False):
    yaml: str
    source_blob_ids: dict[str, str]


def register_composer_routes(router: APIRouter) -> None:

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

    @router.get(
        "/{session_id}/composer/preferences",
        response_model=ComposerPreferencesResponse,
    )
    async def get_composer_preferences(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> ComposerPreferencesResponse:
        session = await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        prefs = await service.get_composer_preferences(session.id)
        return _composer_preferences_response(prefs)

    @router.patch(
        "/{session_id}/composer/preferences",
        response_model=ComposerPreferencesResponse,
    )
    async def update_composer_preferences(
        session_id: UUID,
        body: UpdateComposerPreferencesRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> ComposerPreferencesResponse:
        session = await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        # B2 (load-bearing): the service returns ``(prior, current)`` so
        # Phase 8b's per-session ``composer.session.switched_total``
        # counter can read ``from_mode=transition.prior.trust_mode``
        # without a route-handler read-before-write (which would open a
        # TOCTOU window — see plan §"Option not taken — read-before-write
        # from the route handler"). The PATCH response shape is unchanged;
        # we only project ``current`` into the response model.
        transition = await service.update_composer_preferences(
            session.id,
            trust_mode=body.trust_mode,
            density_default=body.density_default,
            actor=f"user:{user.user_id}",
        )

        # Phase 8 Task 2 Step 3 — per-session ``trust_mode`` switch emit.
        #
        # Guarded on actual change (transition-rate semantic, distinct
        # from the account-level set-rate at preferences/routes.py).
        # The service's ``trust_mode.changed`` audit row at
        # ``sessions/service.py:1605-1619`` fires unconditionally on
        # every PATCH including no-ops; emitting the counter
        # unconditionally would over-count by the no-op rate. Guarding
        # on ``prior != current`` also gives the Q4 contract: a
        # combined PATCH that changes both ``trust_mode`` AND
        # ``density_default`` fires the counter exactly once,
        # attributed to the trust_mode change only.
        #
        # B1 (audit-primacy superset rule): the emit runs AFTER the
        # audit row commits (the service ``_run_sync`` returned),
        # which carries ``prior_trust_mode`` in its payload (B1
        # extension at sessions/service.py:1614). Telemetry attributes
        # are a strict subset of audit-recorded reality.
        #
        # Vocabulary (B1-r2): both attributes come from the per-session
        # ``trust_mode`` CHECK-constraint vocabulary
        # (``explicit_approve`` / ``auto_commit``), NOT the account-
        # level ``default_composer_mode`` vocabulary.
        if transition.prior.trust_mode != transition.current.trust_mode:
            telemetry: SessionsTelemetry = request.app.state.sessions_telemetry
            record_session_switched(
                telemetry,
                from_mode=transition.prior.trust_mode,
                to_mode=transition.current.trust_mode,
            )

        return _composer_preferences_response(transition.current)

    @router.get(
        "/{session_id}/proposals",
        response_model=list[CompositionProposalResponse],
    )
    async def list_composition_proposals(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        status: ProposalLifecycleStatus | None = Query(None),  # noqa: B008
    ) -> list[CompositionProposalResponse]:
        session = await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        proposals = await service.list_composition_proposals(session.id, status=status)
        return [_composition_proposal_response(proposal) for proposal in proposals]

    @router.get(
        "/{session_id}/proposal-events",
        response_model=list[ProposalEventResponse],
    )
    async def list_proposal_events(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> list[ProposalEventResponse]:
        session = await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        events = await service.list_proposal_events(session.id)
        return [_proposal_event_response(event) for event in events]

    @router.post(
        "/{session_id}/proposals/{proposal_id}/accept",
        response_model=CompositionProposalResponse,
    )
    async def accept_composition_proposal(
        session_id: UUID,
        proposal_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> CompositionProposalResponse:
        session = await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        proposals = await service.list_composition_proposals(session.id)
        proposal = next((item for item in proposals if item.id == proposal_id), None)
        if proposal is None:
            raise HTTPException(status_code=404, detail="Proposal not found")
        if proposal.status != "pending":
            raise HTTPException(
                status_code=409,
                detail="Only pending proposals can be accepted.",
            )

        current_record = await service.get_current_state(session.id)
        if proposal.base_state_id is not None and (current_record is None or current_record.id != proposal.base_state_id):
            raise HTTPException(
                status_code=409,
                detail="The session state changed after this proposal was created. Ask ELSPETH to rebase the proposal.",
            )
        current_state = (
            _state_from_record(current_record) if current_record is not None else _initial_composition_state_with_guided_session()
        )
        arguments = cast(dict[str, Any], deep_thaw(proposal.arguments_json))
        user_message_content = await _proposal_user_message_content(service, proposal)
        _ensure_inline_blob_proposal_context(
            proposal,
            arguments,
            user_message_content=user_message_content,
        )
        result = await run_sync_in_worker(
            execute_tool,
            proposal.tool_name,
            arguments,
            current_state,
            request.app.state.catalog_service,
            data_dir=str(request.app.state.settings.data_dir),
            session_engine=request.app.state.session_engine,
            session_id=str(session.id),
            secret_service=request.app.state.scoped_secret_resolver,
            user_id=str(user.user_id),
            user_message_id=str(proposal.user_message_id) if proposal.user_message_id is not None else None,
            user_message_content=user_message_content,
            composer_model_identifier=proposal.composer_model_identifier,
            composer_model_version=proposal.composer_model_version,
            composer_provider=proposal.composer_provider,
            composer_skill_hash=proposal.composer_skill_hash,
            tool_arguments_hash=proposal.tool_arguments_hash,
        )
        if result.updated_state.version == current_state.version:
            # The tool ran but did not advance composition state. The route
            # used to return a single uninformative 409 here regardless of
            # whether the tool succeeded with no-op or failed semantically.
            # Distinguish the cases so the operator sees the actual reason
            # — a generic "did not change composition state" leaves a user
            # with no path forward (session f613306b-… 2026-05-14: the LLM
            # emitted a set_pipeline with no `options` blocks on any node;
            # the validator rejected it; the 409 said only that state
            # didn't change, marking the proposal as "stale" in the
            # frontend without revealing the validation errors that were
            # the actual blocker).
            if not result.success:
                # ``result`` is our own ``execute_tool`` output. Every
                # ``success=False`` ToolResult is built by one of the two
                # failure factories in ``web/composer/tools/_common.py``
                # (``_failure_result`` and the credential-repair factory),
                # both of which set ``data`` to a Mapping carrying
                # ``_DATA_ERROR_KEY``. That is a first-party contract — read
                # it directly and let a contract violation (a future tool
                # building ``success=False`` without the error key) crash
                # loudly rather than degrade to a generic message.
                error_summary = result.data[_DATA_ERROR_KEY] or "Composer proposal failed validation."
                validation_errors_payload: list[dict[str, Any]] = []
                if result.validation is not None:
                    for entry in result.validation.errors:
                        validation_errors_payload.append(
                            {
                                "component": entry.component,
                                "message": entry.message,
                            }
                        )
                # Auto-reject the proposal. It is structurally unacceptable
                # in its current form — the LLM emitted invalid arguments
                # that the runtime validator rejects, and the proposal
                # cannot become acceptable without the composer producing
                # a fresh, corrected proposal. Leaving it pending causes
                # the "refresh asks me to reapprove" friction reported by
                # the operator on 2026-05-14: in-memory frontend stale
                # marking is lost on page reload, so the broken proposal
                # re-surfaces in the pending banner. Marking as rejected
                # server-side keeps the audit trail honest (the user
                # clicked Accept; the server recorded why it could not
                # apply; the proposal is terminal). Audit attribution
                # records the system as the rejecting actor so the trail
                # distinguishes operator-driven rejection from this
                # automatic-on-validation-failure path.
                # Control-flow sentinel, not a swallowed error. TOCTOU: this
                # handler does not hold a session write lock spanning the
                # proposal load (above) and this auto-reject write, so a
                # concurrent Accept/reject for the same proposal can transition
                # it out of "pending" in between.
                # ``reject_composition_proposal`` raises ``ValueError`` for
                # exactly that case (status != "pending"). When that fires, the
                # desired end state — the proposal is no longer pending — is
                # already satisfied, AND the concurrent request that won the
                # race already recorded its own ``proposal.rejected`` event in
                # ``proposal_events``, so no audit fact is lost here (do not
                # "fix" this by adding logging — there is nothing to record).
                # Fall through to the 422 below, which surfaces the real
                # validation failure to the operator. We suppress ONLY
                # ``ValueError`` (the benign status-race signal); we deliberately
                # do NOT suppress ``KeyError`` (proposal row missing): the row
                # was loaded successfully above and proposals are never
                # hard-deleted, so a missing row is corruption of our own data
                # and must crash rather than be swallowed.
                with contextlib.suppress(ValueError):
                    await service.reject_composition_proposal(
                        session_id=session.id,
                        proposal_id=proposal.id,
                        actor=f"system:auto_reject_validation_failed:user:{user.user_id}",
                    )
                raise HTTPException(
                    status_code=422,
                    detail={
                        "detail": (
                            f"The composer's proposed change could not be applied: {error_summary} "
                            "The proposal has been automatically rejected. "
                            "Ask the composer to revise and resubmit."
                        ),
                        "error_type": "proposal_validation_failed",
                        "tool_name": proposal.tool_name,
                        "validation_errors": validation_errors_payload,
                    },
                )
            # Tool succeeded but produced no state change. This is rare
            # post the blob-store-only-mutation interception fix
            # (web/composer/service.py — create_blob/update_blob/delete_blob
            # no longer reach this branch as proposals), but keep the
            # surface in case a future tool ends up here.
            raise HTTPException(
                status_code=409,
                detail="Accepted proposal did not change composition state.",
            )

        state_data, _validation = await _state_data_from_composer_state(
            result.updated_state,
            settings=request.app.state.settings,
            secret_service=request.app.state.scoped_secret_resolver,
            user_id=str(user.user_id),
            runtime_preflight=result.runtime_preflight,
            preflight_exception_policy="raise",
            initial_version=current_state.version,
            telemetry_source="compose",
        )
        state_record = await service.save_composition_state(
            session.id,
            state_data,
            provenance="tool_call",
        )
        try:
            committed = await service.mark_composition_proposal_committed(
                session_id=session.id,
                proposal_id=proposal.id,
                committed_state_id=state_record.id,
                actor=f"user:{user.user_id}",
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Proposal not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _composition_proposal_response(committed)

    @router.post(
        "/{session_id}/proposals/{proposal_id}/reject",
        response_model=CompositionProposalResponse,
    )
    async def reject_composition_proposal(
        session_id: UUID,
        proposal_id: UUID,
        body: RejectProposalRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> CompositionProposalResponse:
        session = await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        proposal = await service.reject_composition_proposal(
            session_id=session.id,
            proposal_id=proposal_id,
            actor=f"user:{user.user_id}",
        )
        _ = body
        return _composition_proposal_response(proposal)

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
                        current_state_id=str(pre_send_state_id) if pre_send_state_id is not None else None,
                        user_id=str(user.user_id),
                        progress=progress_sink,
                        guided_terminal=_guided_terminal_for_compose,
                        user_message_id=request_id,
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
                except _BadRequestLLMError as exc:
                    slog.error(
                        "recompose_llm_bad_request",
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
                            headline="The composer model rejected this retry.",
                            evidence=("The model provider rejected the composer request as invalid.",),
                            likely_next="Check the composer provider configuration and request options before retrying.",
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
                        sources=_transition_state_d["sources"],
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
                    state_response = _state_response(_transition_record)

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
                proposals = await _pending_proposal_responses(service, session.id)
                response = MessageWithStateResponse(
                    message=_message_response(assistant_msg),
                    state=state_response,
                    proposals=proposals,
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
    ) -> StateYamlResponse:
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

        # Phase 6A B3 — sessions-DB audit event for YAML export.
        #
        # Two Tier-1 audit events ship in Phase 6 (mark_ready_for_review and
        # export_yaml). This is the export_yaml site. Sync, crash-on-failure
        # per CLAUDE.md audit primacy — if this write fails the request
        # fails, no YAML is returned, no carve-out is permitted. The write
        # MUST land before the response is returned: the audit row is the
        # legal record that the YAML was exported on the user's behalf.
        #
        # The state record was just read via ``service.get_current_state``
        # above; ``state_record.id`` is the composition_state_id this
        # export is bound to.
        with request.app.state.session_engine.begin() as conn:
            conn.execute(
                insert(composer_completion_events_table).values(
                    id=str(uuid4()),
                    session_id=str(session_id),
                    composition_state_id=str(state_record.id),
                    event_type="export_yaml",
                    actor=str(user.user_id),
                    created_at=datetime.now(UTC),
                    payload_digest=None,
                    expires_at=None,
                )
            )

        # Phase 8 Sub-task 7c (telemetry-backfill: phase-6).
        # composer.session.completed_total — fires AFTER the audit
        # engine.begin() block has exited cleanly. If the audit INSERT
        # raises, the with-block exits via exception, FastAPI converts
        # it to a 5xx, and control never reaches this line — the counter
        # stays at zero and the superset invariant (counter aggregates
        # over committed audit rows) is structurally enforced.
        record_session_completed(
            request.app.state.sessions_telemetry,
            completion_verb="export_yaml",
        )

        response: StateYamlResponse = {"yaml": yaml_str}
        source_blob_ids = {
            source_name: str(source.options["blob_ref"]) for source_name, source in state.sources.items() if "blob_ref" in source.options
        }
        if source_blob_ids:
            response["source_blob_ids"] = source_blob_ids
        return response

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
                ),
                next_turn=TurnPayloadResponse(
                    type=turn["type"],
                    step_index=turn["step_index"],
                    payload=dict(turn["payload"]),
                ),
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
                        payload_payload_id="",
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
                            model=settings.composer_model,
                            temperature=settings.composer_temperature,
                            seed=settings.composer_seed,
                        )
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

                if (
                    existing_record_for_chat is not None
                    and guided.step is GuidedStep.STEP_1_SOURCE
                    and current_turn_type is TurnType.SCHEMA_FORM
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

                    source_resolution = await maybe_resolve_step_1_source_chat(
                        model=settings.composer_model,
                        user_message=body.message,
                        plugin_hint=plugin_hint,
                        temperature=settings.composer_temperature,
                        seed=settings.composer_seed,
                    )
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
                                detail=f"Step 1 source commit failed: {handler_result.tool_result}",
                            )

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
                        guided = _replace(handler_result.session, history=answered_history)
                        state = handler_result.state

                        emit_turn_answered(
                            recorder,
                            step=GuidedStep.STEP_1_SOURCE,
                            turn_type=TurnType.SCHEMA_FORM,
                            response_hash=response_hash,
                            response_payload_id="",
                            control_signal=None,
                            composition_version=state.version,
                            actor=user.user_id,
                        )

                        guided = _replace(guided, step=GuidedStep.STEP_2_SINK)
                        next_turn = build_step_2_single_select_turn(catalog)
                        next_turn_type = TurnType(next_turn["type"])
                        next_payload_hash = stable_hash(next_turn["payload"])
                        new_record = TurnRecord(
                            step=GuidedStep.STEP_2_SINK,
                            turn_type=next_turn_type,
                            payload_hash=next_payload_hash,
                            response_hash=None,
                            emitter="server",
                        )
                        emit_step_advanced(
                            recorder,
                            prev=GuidedStep.STEP_1_SOURCE,
                            next_=GuidedStep.STEP_2_SINK,
                            reason="user_advanced",
                            composition_version=state.version,
                            actor=user.user_id,
                        )
                        emit_turn_emitted(
                            recorder,
                            step=GuidedStep.STEP_2_SINK,
                            turn_type=next_turn_type,
                            payload_hash=next_payload_hash,
                            payload_payload_id="",
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
