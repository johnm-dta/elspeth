from __future__ import annotations

from .._helpers import (
    _COMPOSER_REQUESTS_INFLIGHT,
    UUID,
    Any,
    APIRouter,
    ChatMessageRecord,
    ComposerConvergenceError,
    ComposerPluginCrashError,
    ComposerProgressEvent,
    ComposerRateLimiter,
    ComposerRuntimePreflightError,
    ComposerService,
    ComposerServiceError,
    CompositionStateData,
    CompositionStateResponse,
    Depends,
    GuidedSession,
    HTTPException,
    InvariantError,
    MessageWithStateResponse,
    PipelinePlannerError,
    Request,
    SessionServiceProtocol,
    TransitionAssistantDraft,
    UserIdentity,
    _BadRequestLLMError,
    _cancel_on_client_disconnect,
    _composer_chat_history,
    _composer_conversation_messages,
    _composer_progress_sink,
    _ComposerRequestTerminalStatus,
    _get_composer_progress_registry,
    _get_session_compose_lock_registry,
    _handle_convergence_error,
    _handle_planner_failure,
    _handle_plugin_crash,
    _handle_runtime_preflight_failure,
    _initial_composition_state_with_guided_session,
    _is_client_disconnect_cancel,
    _litellm_error_detail,
    _llm_calls_from_exception,
    _message_response,
    _pending_proposal_responses,
    _persist_llm_calls,
    _persist_tool_invocations,
    _publish_progress,
    _record_composer_request_terminal,
    _record_composer_runtime_preflight_telemetry,
    _request_plugin_policy_context,
    _safe_frame_strings,
    _state_data_from_composer_state,
    _state_from_record,
    _state_response,
    _track_compose_inflight,
    _verify_session_ownership,
    asyncio,
    client_cancelled_progress_event,
    contextlib,
    convergence_progress_event,
    get_current_user,
    get_rate_limiter,
    merge_composer_meta_updates,
    slog,
    validation_errors_for_composer_surface,
)
from .pipeline_settlement import settle_pipeline_proposal_under_compose_lock

router = APIRouter()


@router.post(
    "/{session_id}/recompose",
    response_model=MessageWithStateResponse,
)
async def recompose(
    session_id: UUID,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
    rate_limiter: ComposerRateLimiter = Depends(get_rate_limiter),  # noqa: B008
    # In-flight compose tally for the SPA's post-abort settlement signal
    # (elspeth-06a23adfcc); decrements only after the route fully unwinds.
    _inflight_tally: None = Depends(_track_compose_inflight),
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
    _policy_catalog, plugin_snapshot = _request_plugin_policy_context(request, user)
    profile_registry = request.app.state.operator_profile_registry
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
                # Same disconnect watcher as send_message: cancel the
                # zombie turn when the client aborts the retry
                # (elspeth-e08063c3a5). Compose stays awaited inline —
                # see the send_message block comment.
                async with _cancel_on_client_disconnect(request):
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
                    plugin_snapshot=plugin_snapshot,
                    profile_registry=profile_registry,
                    catalog=request.app.state.catalog_service,
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
                    plugin_snapshot=plugin_snapshot,
                    profile_registry=profile_registry,
                    catalog=request.app.state.catalog_service,
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
                    plugin_snapshot=plugin_snapshot,
                    profile_registry=profile_registry,
                    catalog=request.app.state.catalog_service,
                )
                raise HTTPException(status_code=500, detail=response_body) from rpf_exc.original_exc
            except PipelinePlannerError as exc:
                # Freeform empty-pipeline planner failure (elspeth-54c11243a3).
                # Mirror of the send_message handler — recompose is a retried
                # freeform compose, so it must translate PipelinePlannerError
                # identically or the two routes drift on failure UX. See the
                # send_message clause for the full rationale; _handle_planner_failure
                # writes the durable disposition row and MUST NOT re-persist the
                # already-durable planner LLM-call audit evidence.
                await _publish_progress(
                    progress_registry,
                    session_id=str(session.id),
                    request_id=request_id,
                    user_id=str(user.user_id),
                    event=ComposerProgressEvent(
                        phase="failed",
                        headline="The composer could not build a pipeline for this retry.",
                        evidence=("The composer model did not return a usable pipeline plan.",),
                        likely_next="Retry the request; if it keeps failing, simplify it or check the composer provider.",
                        reason="provider_unavailable",
                    ),
                )
                status_code, planner_response_body = await _handle_planner_failure(
                    exc,
                    service,
                    session.id,
                    pre_send_state_id,
                )
                raise HTTPException(status_code=status_code, detail=planner_response_body) from exc
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

            _post_compose_updates: dict[str, Any] = {"repair_turns_used": result.repair_turns_used}
            if _post_compose_guided is not None:
                _post_compose_updates["guided_session"] = _post_compose_guided.to_dict()
            _post_compose_meta = merge_composer_meta_updates(
                state_record.composer_meta if state_record is not None else None,
                _post_compose_updates,
            )

            # Save state if version changed.
            # Path 2 (post-compose runtime preflight): mirror of the
            # send_message post-compose try/except — see the send_message
            # block for the full rationale on the structural fix.
            state_response: CompositionStateResponse | None = None
            post_compose_state_id: UUID | None = pre_send_state_id
            assistant_msg: ChatMessageRecord | None = None
            if result.pipeline_commit_intent is not None:
                authority = await service.get_authoritative_pipeline_proposal(
                    session_id=session.id,
                    proposal_id=result.pipeline_commit_intent.proposal_id,
                    reviewed_facts={},
                )
                route_settlement = await settle_pipeline_proposal_under_compose_lock(
                    request=request,
                    user=user,
                    authority=authority,
                    draft_hash=result.pipeline_commit_intent.draft_hash,
                    composer_meta=_post_compose_meta,
                    telemetry_source="recompose",
                    transition_assistant=TransitionAssistantDraft(
                        content=result.message,
                        raw_content=result.raw_assistant_content,
                    )
                    if _guided_terminal_for_compose is not None
                    else None,
                )
                state_response = _state_response(
                    route_settlement.settlement.state,
                    live_validation=route_settlement.validation,
                )
                post_compose_state_id = route_settlement.settlement.state.id
                assistant_msg = route_settlement.settlement.transition_message
            elif result.state.version != state.version:
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
                        session_id=session.id,
                        plugin_snapshot=plugin_snapshot,
                        profile_registry=profile_registry,
                        catalog=request.app.state.catalog_service,
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
                        plugin_snapshot=plugin_snapshot,
                        profile_registry=profile_registry,
                        catalog=request.app.state.catalog_service,
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
                if _guided_terminal_for_compose is not None:
                    transition_settlement = await service.commit_transition_response(
                        session_id=session.id,
                        expected_current_state_id=pre_send_state_id,
                        state=state_data,
                        assistant_content=result.message,
                        raw_content=result.raw_assistant_content,
                    )
                    new_state_record = transition_settlement.state
                    assistant_msg = transition_settlement.message
                else:
                    new_state_record = await service.save_composition_state(
                        session.id,
                        state_data,
                        # Successful recompose state advance after the LLM
                        # composer returns a newer state version.
                        provenance="post_compose",
                    )
                state_response = _state_response(new_state_record, live_validation=validation)
                post_compose_state_id = new_state_record.id
            elif _guided_terminal_for_compose is not None and _post_compose_guided is not None:
                # Version unchanged but transition_consumed must be flipped.
                # Persist the updated guided_session in a new state row so
                # subsequent turns pick up transition_consumed=True.
                _transition_state = result.state
                _transition_state_d = _transition_state.to_dict()
                _transition_state_data = CompositionStateData(
                    sources=_transition_state_d["sources"],
                    nodes=_transition_state_d["nodes"],
                    edges=_transition_state_d["edges"],
                    outputs=_transition_state_d["outputs"],
                    metadata_=_transition_state_d["metadata"],
                    is_valid=False,
                    validation_errors=validation_errors_for_composer_surface(
                        composer_meta=_post_compose_meta,
                        is_valid=False,
                        validation_errors=None,
                    ),
                    composer_meta=_post_compose_meta,
                )
                transition_settlement = await service.commit_transition_response(
                    session_id=session.id,
                    expected_current_state_id=pre_send_state_id,
                    state=_transition_state_data,
                    assistant_content=result.message,
                    raw_content=result.raw_assistant_content,
                )
                _transition_record = transition_settlement.state
                assistant_msg = transition_settlement.message
                post_compose_state_id = _transition_record.id
                state_response = _state_response(_transition_record)

            # Persist assistant message
            if assistant_msg is None:
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
            if _is_client_disconnect_cancel(exc):
                # Disconnect-initiated cancel — see the send_message
                # mirror for the 499-conversion rationale.
                raise HTTPException(
                    status_code=499,
                    detail="Client disconnected while the compose turn was running.",
                ) from exc
            raise
        finally:
            _COMPOSER_REQUESTS_INFLIGHT.add(-1, {"endpoint": "recompose"})
            _record_composer_request_terminal(terminal_status, endpoint="recompose")
