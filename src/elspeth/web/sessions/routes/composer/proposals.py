from __future__ import annotations

from collections.abc import Awaitable

from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.pipeline_commit import (
    PipelineCommitConfig,
    PipelineCommitError,
    RecoveredPipelineCommit,
    prepare_pipeline_proposal_commit,
)
from elspeth.web.composer.pipeline_proposal import reviewed_anchor_hash
from elspeth.web.composer.tools import is_approval_required_blob_store_only_mutation_tool
from elspeth.web.sessions.protocol import PipelineProposalRejectionReason

from .._helpers import (
    _DATA_ERROR_KEY,
    UUID,
    AcceptProposalRequest,
    Any,
    APIRouter,
    CompositionProposalRecord,
    CompositionProposalResponse,
    Depends,
    HTTPException,
    Mapping,
    ProposalEventResponse,
    ProposalLifecycleStatus,
    Query,
    RejectProposalRequest,
    Request,
    SessionServiceProtocol,
    UserIdentity,
    _composition_proposal_response,
    _get_session_compose_lock_registry,
    _initial_composition_state_with_guided_session,
    _persist_tool_invocations,
    _proposal_event_response,
    _state_data_from_composer_state,
    _state_from_record,
    _verify_session_ownership,
    asyncio,
    cast,
    deep_thaw,
    execute_tool,
    get_current_user,
    run_sync_in_worker,
)

router = APIRouter()


_PROPOSAL_COMPOSER_CONTEXT_FIELDS: tuple[str, ...] = (
    "composer_model_identifier",
    "composer_model_version",
    "composer_provider",
    "composer_skill_hash",
    "tool_arguments_hash",
)


@trust_boundary(
    tier=3,
    source="persisted LLM tool-call arguments of a stored CompositionProposalRecord (Tier-3 on read-back)",
    source_param="arguments",
    suppresses=("R5",),
    invariant="returns None on any absent/wrong-typed branch of arguments.source.inline_blob.content; never raises on arguments",
    non_raising=True,
)
def _inline_blob_content_for_proposal(
    proposal: CompositionProposalRecord,
    arguments: Mapping[str, Any],
) -> str | None:
    """Return inline blob content that accept replay would persist, if any."""
    if proposal.tool_name != "set_pipeline":
        return None
    source = arguments["source"] if "source" in arguments else None
    if not isinstance(source, Mapping):
        return None
    inline_blob = source["inline_blob"] if "inline_blob" in source else None
    if not isinstance(inline_blob, Mapping):
        return None
    content = inline_blob["content"] if "content" in inline_blob else None
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


async def _await_with_deferred_cancellation[T](awaitable: Awaitable[T]) -> tuple[T, bool]:
    """Await critical proposal work to completion while recording cancellation.

    ``run_sync_in_worker`` deliberately leaves its worker running if the
    request task is cancelled. Once an approved tool execution starts, the
    route must keep the session lock until the side effect is paired with a
    terminal proposal transition.
    """
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while True:
        try:
            return await asyncio.shield(task), cancelled
        except asyncio.CancelledError:
            cancelled = True
            if task.done():
                return task.result(), cancelled


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
    body: AcceptProposalRequest | None = None,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> CompositionProposalResponse:
    session = await _verify_session_ownership(session_id, user, request)
    compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session.id))
    async with compose_lock:
        service: SessionServiceProtocol = request.app.state.session_service
        try:
            proposal_authority = await service.get_authoritative_composition_proposal(
                session_id=session.id,
                proposal_id=proposal_id,
                reviewed_facts=None,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Proposal not found") from None
        proposal = proposal_authority.row
        pipeline_authority = proposal_authority.pipeline
        if pipeline_authority is not None:
            if body is None or body.draft_hash is None:
                raise HTTPException(status_code=422, detail="Canonical pipeline proposal acceptance requires draft_hash.")
            if body.draft_hash != pipeline_authority.proposal.draft_hash:
                raise HTTPException(status_code=409, detail="The pipeline proposal draft hash is stale or mismatched.")
            if pipeline_authority.proposal.surface.value in {"guided_staged", "tutorial_profile"}:
                raise HTTPException(
                    status_code=409,
                    detail="This pipeline proposal must be accepted through its guided workflow.",
                )
            if pipeline_authority.proposal.reviewed_anchor_hash != reviewed_anchor_hash({}):
                raise HTTPException(status_code=409, detail="The pipeline proposal reviewed anchor is stale or mismatched.")
            if proposal.status == "committed":
                return _composition_proposal_response(proposal)
            if proposal.status != "pending":
                raise HTTPException(status_code=409, detail="Only pending proposals can be accepted.")

            current_record = await service.get_current_state(session.id)
            current_state = (
                _state_from_record(current_record) if current_record is not None else _initial_composition_state_with_guided_session()
            )
            user_message_content = await _proposal_user_message_content(service, proposal)
            plugin_snapshot = request.app.state.plugin_snapshot_factory(user)
            policy_catalog = PolicyCatalogView(
                request.app.state.catalog_service,
                plugin_snapshot,
                request.app.state.operator_profile_registry,
            )
            recorder = BufferingRecorder()
            recovery = await service.get_pipeline_dispatch_recovery(authority=pipeline_authority)
            try:
                prepared, cancellation_deferred = await _await_with_deferred_cancellation(
                    prepare_pipeline_proposal_commit(
                        authority=pipeline_authority,
                        current_state=current_state,
                        current_state_id=current_record.id if current_record is not None else None,
                        policy_catalog=policy_catalog,
                        plugin_snapshot=plugin_snapshot,
                        config=PipelineCommitConfig(
                            data_dir=str(request.app.state.settings.data_dir),
                            session_engine=request.app.state.session_engine,
                            secret_service=request.app.state.scoped_secret_resolver,
                            user_id=str(user.user_id),
                            user_message_content=user_message_content,
                            max_blob_storage_per_session_bytes=request.app.state.settings.max_blob_storage_per_session_bytes,
                            runtime_preflight=None,
                            timeout_seconds=request.app.state.settings.composer_timeout_seconds,
                        ),
                        recorder=recorder,
                        actor=f"user:{user.user_id}",
                        recovery_dispatch=recovery.binding if recovery is not None else None,
                        recovery_executor_content_hash=recovery.executor_content_hash if recovery is not None else None,
                    )
                )
            except PipelineCommitError as exc:
                persisted_dispatch = recovery.binding if recovery is not None else exc.dispatch if exc.invocation is None else None
                captured = tuple(recorder.invocations)
                if captured:
                    bindings, was_cancelled = await _await_with_deferred_cancellation(
                        _persist_tool_invocations(
                            service,
                            session.id,
                            captured,
                            None,
                            plugin_crash_pending=True,
                        )
                    )
                    if exc.dispatch is not None:
                        if len(bindings) != 1:
                            raise RuntimeError("pipeline failure dispatch did not persist exactly one binding") from exc
                        persisted_dispatch = bindings[0]
                    if was_cancelled:
                        raise asyncio.CancelledError from exc
                reason_by_code: dict[str, PipelineProposalRejectionReason] = {
                    "CANDIDATE_EXECUTOR_MISMATCH": "candidate_executor_mismatch",
                    "VALIDATION_FAILED": "validation_failed",
                    "BASE_CONFLICT": "base_conflict",
                }
                reason = reason_by_code.get(exc.code)
                if reason is not None:
                    _rejected, rejection_cancelled = await _await_with_deferred_cancellation(
                        service.reject_pipeline_composition_proposal(
                            session_id=session.id,
                            proposal_id=proposal.id,
                            draft_hash=pipeline_authority.proposal.draft_hash,
                            reviewed_facts={},
                            reason=reason,
                            dispatch=persisted_dispatch,
                            actor=f"system:pipeline_commit:user:{user.user_id}",
                        )
                    )
                    if rejection_cancelled:
                        raise asyncio.CancelledError from exc
                status_code = 409 if exc.code in {"BASE_CONFLICT", "NOT_PENDING"} else 422
                raise HTTPException(status_code=status_code, detail=str(exc)) from exc
            except BaseException:
                captured = tuple(recorder.invocations)
                if captured:
                    await _persist_tool_invocations(
                        service,
                        session.id,
                        captured,
                        None,
                        plugin_crash_pending=True,
                    )
                raise

            if isinstance(prepared, RecoveredPipelineCommit):
                bindings = (prepared.dispatch,)
            else:
                bindings, was_cancelled = await _await_with_deferred_cancellation(
                    _persist_tool_invocations(
                        service,
                        session.id,
                        (prepared.invocation,),
                        None,
                        plugin_crash_pending=False,
                    )
                )
                cancellation_deferred = cancellation_deferred or was_cancelled
            if len(bindings) != 1:
                raise RuntimeError("pipeline acceptance dispatch did not persist exactly one binding")
            (state_data, _validation), was_cancelled = await _await_with_deferred_cancellation(
                _state_data_from_composer_state(
                    prepared.result.updated_state,
                    settings=request.app.state.settings,
                    secret_service=request.app.state.scoped_secret_resolver,
                    user_id=str(user.user_id),
                    session_id=session.id,
                    plugin_snapshot=plugin_snapshot,
                    profile_registry=request.app.state.operator_profile_registry,
                    catalog=request.app.state.catalog_service,
                    runtime_preflight=prepared.result.runtime_preflight,
                    preflight_exception_policy="raise",
                    initial_version=current_state.version,
                    telemetry_source="compose",
                )
            )
            cancellation_deferred = cancellation_deferred or was_cancelled
            settled, was_cancelled = await _await_with_deferred_cancellation(
                service.settle_pipeline_composition_proposal(
                    session_id=session.id,
                    proposal_id=proposal.id,
                    draft_hash=body.draft_hash,
                    reviewed_facts={},
                    state=state_data,
                    candidate_content_hash=prepared.candidate_content_hash,
                    executor_content_hash=prepared.executor_content_hash,
                    final_composer_metadata=state_data.composer_meta,
                    dispatch=bindings[0],
                    actor=f"user:{user.user_id}",
                )
            )
            cancellation_deferred = cancellation_deferred or was_cancelled
            response = _composition_proposal_response(settled.proposal)
            if cancellation_deferred:
                raise asyncio.CancelledError
            return response

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
        cancellation_deferred = False
        plugin_snapshot = request.app.state.plugin_snapshot_factory(user)
        policy_catalog = PolicyCatalogView(
            request.app.state.catalog_service,
            plugin_snapshot,
            request.app.state.operator_profile_registry,
        )
        result, was_cancelled = await _await_with_deferred_cancellation(
            run_sync_in_worker(
                execute_tool,
                proposal.tool_name,
                arguments,
                current_state,
                policy_catalog,
                plugin_snapshot=plugin_snapshot,
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
        )
        cancellation_deferred = cancellation_deferred or was_cancelled
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
                # Control-flow sentinel, not a swallowed error. The
                # route-level session compose lock serializes normal
                # Accept/reject HTTP handlers, but non-route callers can
                # still transition the proposal between our load above and
                # this defensive auto-reject write. ``reject_composition_proposal``
                # raises ``ValueError`` for exactly that terminal-state case.
                # When that fires, the desired end state — the proposal is no
                # longer pending — is already satisfied, and the winning
                # transition recorded its own lifecycle event.
                # Fall through to the 422 below, which surfaces the real
                # validation failure to the operator. We suppress ONLY
                # ``ValueError`` (the benign status-race signal); we deliberately
                # do NOT suppress ``KeyError`` (proposal row missing): the row
                # was loaded successfully above and proposals are never
                # hard-deleted, so a missing row is corruption of our own data
                # and must crash rather than be swallowed.
                try:
                    _rejected, was_cancelled = await _await_with_deferred_cancellation(
                        service.reject_composition_proposal(
                            session_id=session.id,
                            proposal_id=proposal.id,
                            actor=f"system:auto_reject_validation_failed:user:{user.user_id}",
                        )
                    )
                    cancellation_deferred = cancellation_deferred or was_cancelled
                except ValueError:
                    pass
                if cancellation_deferred:
                    raise asyncio.CancelledError
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
            if is_approval_required_blob_store_only_mutation_tool(proposal.tool_name):
                # Blob-only approvals intentionally produce no CompositionState
                # delta. The accepted proposal points at the current state snapshot
                # so the proposal lifecycle remains forward-only without inventing
                # a fake pipeline edit.
                if current_record is None:
                    (state_data, _validation), was_cancelled = await _await_with_deferred_cancellation(
                        _state_data_from_composer_state(
                            current_state,
                            settings=request.app.state.settings,
                            secret_service=request.app.state.scoped_secret_resolver,
                            user_id=str(user.user_id),
                            session_id=session.id,
                            plugin_snapshot=plugin_snapshot,
                            profile_registry=request.app.state.operator_profile_registry,
                            catalog=request.app.state.catalog_service,
                            runtime_preflight=result.runtime_preflight,
                            preflight_exception_policy="raise",
                            initial_version=current_state.version,
                            telemetry_source="compose",
                        )
                    )
                    cancellation_deferred = cancellation_deferred or was_cancelled
                    current_record, was_cancelled = await _await_with_deferred_cancellation(
                        service.save_composition_state(
                            session.id,
                            state_data,
                            provenance="tool_call",
                        )
                    )
                    cancellation_deferred = cancellation_deferred or was_cancelled
                assert current_record is not None
                try:
                    committed, was_cancelled = await _await_with_deferred_cancellation(
                        service.mark_composition_proposal_committed(
                            session_id=session.id,
                            proposal_id=proposal.id,
                            committed_state_id=current_record.id,
                            actor=f"user:{user.user_id}",
                        )
                    )
                    cancellation_deferred = cancellation_deferred or was_cancelled
                except KeyError:
                    raise HTTPException(status_code=404, detail="Proposal not found") from None
                except ValueError as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                response = _composition_proposal_response(committed)
                if cancellation_deferred:
                    raise asyncio.CancelledError
                return response

            # Tool succeeded but produced no state change. Non-blob proposal tools
            # are composition mutations and must advance state on success.
            raise HTTPException(
                status_code=409,
                detail="Accepted proposal did not change composition state.",
            )

        (state_data, _validation), was_cancelled = await _await_with_deferred_cancellation(
            _state_data_from_composer_state(
                result.updated_state,
                settings=request.app.state.settings,
                secret_service=request.app.state.scoped_secret_resolver,
                user_id=str(user.user_id),
                session_id=session.id,
                plugin_snapshot=plugin_snapshot,
                profile_registry=request.app.state.operator_profile_registry,
                catalog=request.app.state.catalog_service,
                runtime_preflight=result.runtime_preflight,
                preflight_exception_policy="raise",
                initial_version=current_state.version,
                telemetry_source="compose",
            )
        )
        cancellation_deferred = cancellation_deferred or was_cancelled
        state_record, was_cancelled = await _await_with_deferred_cancellation(
            service.save_composition_state(
                session.id,
                state_data,
                provenance="tool_call",
            )
        )
        cancellation_deferred = cancellation_deferred or was_cancelled
        try:
            committed, was_cancelled = await _await_with_deferred_cancellation(
                service.mark_composition_proposal_committed(
                    session_id=session.id,
                    proposal_id=proposal.id,
                    committed_state_id=state_record.id,
                    actor=f"user:{user.user_id}",
                )
            )
            cancellation_deferred = cancellation_deferred or was_cancelled
        except KeyError:
            raise HTTPException(status_code=404, detail="Proposal not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        response = _composition_proposal_response(committed)
        if cancellation_deferred:
            raise asyncio.CancelledError
        return response


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
    compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session.id))
    async with compose_lock:
        service: SessionServiceProtocol = request.app.state.session_service
        try:
            authority = await service.get_authoritative_composition_proposal(
                session_id=session.id,
                proposal_id=proposal_id,
                reviewed_facts=None,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Proposal not found") from None
        try:
            if authority.pipeline is None:
                proposal = await service.reject_composition_proposal(
                    session_id=session.id,
                    proposal_id=proposal_id,
                    actor=f"user:{user.user_id}",
                )
            else:
                proposal = await service.reject_pipeline_composition_proposal(
                    session_id=session.id,
                    proposal_id=proposal_id,
                    draft_hash=authority.pipeline.proposal.draft_hash,
                    reviewed_facts=None,
                    reason="operator_rejected",
                    dispatch=None,
                    actor=f"user:{user.user_id}",
                )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _ = body
        return _composition_proposal_response(proposal)
