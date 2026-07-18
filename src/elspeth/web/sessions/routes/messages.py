from __future__ import annotations

from elspeth.web.sessions.titles import is_default_session_title

from ._helpers import (
    _COMPOSER_REQUESTS_INFLIGHT,
    AUDIT_GRADE_VIEW_QUERY_ARG_ALLOWLIST,
    UUID,
    Any,
    APIRouter,
    AuditIntegrityError,
    ChatMessageResponse,
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
    Query,
    Request,
    SendMessageRequest,
    SessionServiceProtocol,
    UserIdentity,
    _BadRequestLLMError,
    _cancel_on_client_disconnect,
    _composer_chat_history,
    _composer_conversation_messages,
    _composer_conversation_or_llm_audit_messages,
    _composer_conversation_or_tool_messages,
    _composer_conversation_tool_or_llm_audit_messages,
    _composer_progress_sink,
    _ComposerRequestTerminalStatus,
    _get_composer_progress_registry,
    _get_session_compose_lock_registry,
    _handle_convergence_error,
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
    maybe_auto_title_session,
    merge_composer_meta_updates,
    slog,
    validation_errors_for_composer_surface,
)
from .composer.pipeline_settlement import settle_pipeline_proposal_under_compose_lock


def _requests_audit_grade_messages_view(
    *,
    include_tool_rows: bool,
    include_llm_audit: bool,
    include_raw_content: bool,
) -> bool:
    return include_tool_rows or include_llm_audit or include_raw_content


def register_message_routes(router: APIRouter) -> None:

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
        # In-flight compose tally for the SPA's post-abort settlement signal
        # (elspeth-06a23adfcc); decrements only after the route fully unwinds.
        _inflight_tally: None = Depends(_track_compose_inflight),
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

            # Optimistic-concurrency baseline for the compose loop: the
            # ACTUAL head this request composes against (loaded above,
            # under the compose lock) — NOT the client-asserted
            # ``body.state_id``. ``pre_send_state_id`` records what the
            # user was looking at (AD-2/AD-7 provenance) and may
            # legitimately lag the head: a client-aborted turn keeps
            # mutating state server-side, and the SPA never sees the new
            # version. Seeding the loop's ``expected_current_state_id``
            # from the client id turned that legitimate lag into an
            # unrecoverable 409 stale_compose_state on every follow-up
            # send (elspeth-e08063c3a5). ``/recompose`` already seeds
            # from the head; this keeps the two routes symmetric.
            compose_base_state_id = state_record.id if state_record is not None else None
            _policy_catalog, plugin_snapshot = _request_plugin_policy_context(request, user)
            profile_registry = request.app.state.operator_profile_registry

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
            # Initialized here so the finally block can reference it even
            # if an exception fires before the first-message branch is
            # reached. Assigned below only when first-message conditions
            # hold.
            auto_title_task: asyncio.Task[None] | None = None
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

                # 3b. First-message auto-titling.
                #
                # When the just-persisted user message is the only message
                # in the session AND the title is still a mint-time default
                # ("Session — 2 Jul 2026" or a legacy "New session" /
                # "Untitled" — see sessions/titles.py), spawn a background
                # task that asks the composer model for a 3-6 word session
                # title and writes it via update_session_title. The task is
                # awaited (with timeout) in the finally block below so
                # the title is in the DB by the time the route returns —
                # the frontend's post-send loadSessions() picks it up.
                #
                # The "default title" guard is a demo guard, not a
                # contract: a user who manually rename-then-sends still
                # gets re-titled. Tighten to a separate auto_titled_at
                # column if this becomes annoying.
                if len(records) == 1 and is_default_session_title(session.title):
                    auto_title_task = asyncio.create_task(
                        maybe_auto_title_session(
                            service=service,
                            session_id=session.id,
                            user_message=body.content,
                            model=settings.composer_model,
                            temperature=settings.composer_temperature,
                            seed=settings.composer_seed,
                        )
                    )

                # 4. Run the LLM composition loop
                composer: ComposerService = request.app.state.composer_service
                from litellm.exceptions import APIError as LiteLLMAPIError
                from litellm.exceptions import AuthenticationError as LiteLLMAuthError

                try:
                    # The watcher cancels this task if the client
                    # disconnects mid-compose (Stop button, SPA compose
                    # timeout, closed tab) — the server stack does not do
                    # that on its own, so without it the turn runs to
                    # completion as a zombie (elspeth-e08063c3a5). The
                    # compose MUST stay awaited inline (no child task):
                    # ``attach_llm_calls`` rides on the CancelledError
                    # instance and a task boundary would drop it.
                    async with _cancel_on_client_disconnect(request):
                        result = await composer.compose(
                            body.content,
                            chat_messages,
                            state,
                            session_id=str(session_id),
                            current_state_id=str(compose_base_state_id) if compose_base_state_id is not None else None,
                            user_id=str(user.user_id),
                            progress=progress_sink,
                            guided_terminal=_guided_terminal_for_compose,
                            # Bind the freshly persisted user message id so any
                            # inline_blob created by
                            # this turn's tool calls can record provenance
                            # back to it.  The composite FK on
                            # ``(created_from_message_id, session_id)`` in
                            # ``blobs_table`` rejects cross-session lineage,
                            # so a stale or wrong id from a prior request
                            # would surface as an IntegrityError, not as a
                            # silent provenance corruption.
                            user_message_id=str(user_msg.id),
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
                        compose_base_state_id,
                        settings=settings,
                        secret_service=request.app.state.scoped_secret_resolver,
                        plugin_snapshot=plugin_snapshot,
                        profile_registry=profile_registry,
                        catalog=request.app.state.catalog_service,
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
                        await _persist_llm_calls(service, session.id, llm_calls, compose_base_state_id, plugin_crash_pending=True)
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
                        await _persist_llm_calls(service, session.id, llm_calls, compose_base_state_id, plugin_crash_pending=True)
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
                        "compose_llm_bad_request",
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
                            headline="The composer model rejected this request.",
                            evidence=("The model provider rejected the composer request as invalid.",),
                            likely_next="Check the composer provider configuration and request options before retrying.",
                            reason="provider_unavailable",
                        ),
                    )
                    llm_calls = _llm_calls_from_exception(exc)
                    if llm_calls:
                        await _persist_llm_calls(service, session.id, llm_calls, compose_base_state_id, plugin_crash_pending=True)
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
                        compose_base_state_id,
                        settings=settings,
                        secret_service=request.app.state.scoped_secret_resolver,
                        plugin_snapshot=plugin_snapshot,
                        profile_registry=profile_registry,
                        catalog=request.app.state.catalog_service,
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
                        compose_base_state_id,
                        settings=settings,
                        secret_service=request.app.state.scoped_secret_resolver,
                        plugin_snapshot=plugin_snapshot,
                        profile_registry=profile_registry,
                        catalog=request.app.state.catalog_service,
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
                        await _persist_llm_calls(service, session.id, llm_calls, compose_base_state_id, plugin_crash_pending=True)
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
                _post_compose_updates: dict[str, Any] = {"repair_turns_used": result.repair_turns_used}
                if _post_compose_guided is not None:
                    _post_compose_updates["guided_session"] = _post_compose_guided.to_dict()
                _post_compose_meta = merge_composer_meta_updates(
                    state_record.composer_meta if state_record is not None else None,
                    _post_compose_updates,
                )

                state_response: CompositionStateResponse | None = None
                post_compose_state_id: UUID | None = compose_base_state_id
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
                        telemetry_source="compose",
                    )
                    state_response = _state_response(
                        route_settlement.settlement.state,
                        live_validation=route_settlement.validation,
                    )
                    post_compose_state_id = route_settlement.settlement.state.id
                elif result.state.version != state.version:
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
                            session_id=session.id,
                            plugin_snapshot=plugin_snapshot,
                            profile_registry=profile_registry,
                            catalog=request.app.state.catalog_service,
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
                            compose_base_state_id,
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
                        # Successful send-message state advance after the LLM
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
                    _transition_record = await service.save_composition_state(
                        session.id,
                        _transition_state_data,
                        # Metadata-only post-compose advance: the LLM result
                        # did not change graph version, but the guided-session
                        # transition was consumed and must be audited separately
                        # from session seeding.
                        provenance="post_compose",
                    )
                    post_compose_state_id = _transition_record.id
                    state_response = _state_response(_transition_record)

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
                        compose_base_state_id,
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
                proposals = await _pending_proposal_responses(service, session.id)
                response = MessageWithStateResponse(
                    message=_message_response(assistant_msg),
                    state=state_response,
                    proposals=proposals,
                )
                terminal_status = "completed"
                return response
            except InvariantError as exc:
                # Same B1-sanitization rationale as the /guided/respond
                # transition and settlement handlers: server-invariant
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
                                compose_base_state_id,
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
                if _is_client_disconnect_cancel(exc):
                    # Disconnect-initiated cancellation (our
                    # _cancel_on_client_disconnect watcher): the client is
                    # gone, so the response body is discarded by the
                    # transport either way — but converting to an
                    # HTTPException here lets the task unwind as a normal
                    # handled request instead of escaping as a
                    # CancelledError, which uvicorn logs as "Exception in
                    # ASGI application" on every Stop click / client
                    # timeout. 499 is the de-facto "client closed request"
                    # status. A real external cancel (server shutdown)
                    # takes the bare ``raise`` below and keeps unwinding.
                    raise HTTPException(
                        status_code=499,
                        detail="Client disconnected while the compose turn was running.",
                    ) from exc
                raise
            finally:
                _COMPOSER_REQUESTS_INFLIGHT.add(-1, {"endpoint": "send_message"})
                _record_composer_request_terminal(terminal_status, endpoint="send_message")
                # Bounded await of the auto-title task so its result is in
                # the DB before the route returns and the strong reference
                # outlives the event-loop's weak-ref GC window. Expected
                # provider/timeout failures are recorded inside the task;
                # programmer bugs and DB write failures propagate here
                # instead of being silently swallowed.
                #
                # ``asyncio.wait`` reports the scheduling timeout as an
                # explicit ``(done, pending)`` partition rather than as a
                # raised ``TimeoutError`` — the timeout is a control-flow
                # signal, not an error to catch. On the done path we call
                # ``.result()`` to re-raise any task exception (programmer
                # bug / DB write failure) so it propagates exactly as the
                # comment above promises; on the not-done (timed-out) path
                # we cancel the runaway task and let the route return.
                if auto_title_task is not None:
                    done, _ = await asyncio.wait({auto_title_task}, timeout=2.0)
                    if auto_title_task in done:
                        auto_title_task.result()
                    else:
                        auto_title_task.cancel()

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
        if _requests_audit_grade_messages_view(
            include_tool_rows=include_tool_rows,
            include_llm_audit=include_llm_audit,
            include_raw_content=include_raw_content,
        ):
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
        # raw prompts, provider reasoning artifacts, tool arguments, or tool results.
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
