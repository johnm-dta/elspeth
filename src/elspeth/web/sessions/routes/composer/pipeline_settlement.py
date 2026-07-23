"""Canonical pipeline settlement shared by manual and automatic approval.

Callers must already hold the session compose lock. Trust mode changes who
invokes this coordinator, never the prepare/audit/atomic-settlement sequence.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from typing import Literal

from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.pipeline_commit import (
    PipelineCommitConfig,
    PipelineCommitError,
    RecoveredPipelineCommit,
    prepare_pipeline_proposal_commit,
)
from elspeth.web.composer.pipeline_proposal import reviewed_anchor_hash
from elspeth.web.composer.state import ValidationSummary
from elspeth.web.sessions.protocol import (
    AuthoritativePipelineProposal,
    CompositionProposalRecord,
    PipelineProposalRejectionReason,
    PipelineProposalSettlementResult,
    TransitionAssistantDraft,
)

from .._helpers import (
    HTTPException,
    Request,
    SessionServiceProtocol,
    UserIdentity,
    _initial_composition_state_with_guided_session,
    _persist_tool_invocations,
    _state_data_from_composer_state,
    _state_from_record,
    asyncio,
)

_GUIDED_ATOMIC_SETTLEMENT_COMPLETED = "_elspeth_guided_atomic_settlement_completed"
_GUIDED_ATOMIC_SETTLEMENT_FAILURE = "_elspeth_guided_atomic_settlement_failure"


@dataclass(slots=True)
class _DeferredCancellationState:
    requested: bool = False


@dataclass(frozen=True, slots=True)
class PipelineRouteSettlement:
    settlement: PipelineProposalSettlementResult
    validation: ValidationSummary | None


async def _await_guided_atomic_settlement[T](awaitable: Awaitable[T]) -> T:
    """Drain a submitted guided settlement before preserving request cancellation."""

    settlement_task = asyncio.ensure_future(awaitable)
    caller_task = asyncio.current_task()
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(settlement_task)
        except asyncio.CancelledError as exc:
            if settlement_task.done() and settlement_task.cancelled():
                if cancellation is not None:
                    cancellation.__dict__[_GUIDED_ATOMIC_SETTLEMENT_FAILURE] = exc
                    raise cancellation from exc
                raise
            if caller_task is None or caller_task.cancelling() == 0:
                raise
            if cancellation is None:
                cancellation = exc
            if not settlement_task.done():
                continue
            try:
                result = settlement_task.result()
            except BaseException as failure:
                cancellation.__dict__[_GUIDED_ATOMIC_SETTLEMENT_FAILURE] = failure
                raise cancellation from failure
        except Exception as failure:
            if cancellation is None:
                raise
            cancellation.__dict__[_GUIDED_ATOMIC_SETTLEMENT_FAILURE] = failure
            raise cancellation from failure
        if cancellation is not None:
            cancellation.__dict__[_GUIDED_ATOMIC_SETTLEMENT_COMPLETED] = True
            raise cancellation
        return result


async def _await_with_deferred_cancellation[T](
    awaitable: Awaitable[T],
    *,
    state: _DeferredCancellationState | None = None,
) -> tuple[T, bool]:
    """Finish one dispatch+settlement critical section before cancelling.

    ``run_sync_in_worker`` leaves its worker running if the request task is
    cancelled. Once an approved execution starts, the route must retain the
    session lock until the side effect is paired with a terminal proposal
    transition.
    """
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    cancellation_state = state if state is not None else _DeferredCancellationState()
    while True:
        try:
            return await asyncio.shield(task), cancelled
        except asyncio.CancelledError:
            cancelled = True
            cancellation_state.requested = True
            if task.done():
                return task.result(), cancelled


async def _proposal_user_message_content(
    service: SessionServiceProtocol,
    proposal: CompositionProposalRecord,
) -> str | None:
    """Recover the immutable originating user-message body for replay."""
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


async def settle_pipeline_proposal_under_compose_lock(
    *,
    request: Request,
    user: UserIdentity,
    authority: AuthoritativePipelineProposal,
    draft_hash: str,
    composer_meta: Mapping[str, object] | None = None,
    telemetry_source: Literal["compose", "recompose"] = "compose",
    transition_assistant: TransitionAssistantDraft | None = None,
) -> PipelineRouteSettlement:
    """Settle one exact canonical proposal while the caller holds the lock."""
    service: SessionServiceProtocol = request.app.state.session_service
    proposal = authority.row
    if draft_hash != authority.proposal.draft_hash:
        raise HTTPException(status_code=409, detail="The pipeline proposal draft hash is stale or mismatched.")
    if authority.proposal.surface.value in {"guided_staged", "tutorial_profile"}:
        raise HTTPException(status_code=409, detail="This pipeline proposal must be accepted through its guided workflow.")
    if authority.proposal.reviewed_anchor_hash != reviewed_anchor_hash({}):
        raise HTTPException(status_code=409, detail="The pipeline proposal reviewed anchor is stale or mismatched.")
    if proposal.status == "committed":
        if proposal.committed_state_id is None:
            raise RuntimeError("committed pipeline proposal has no committed state id")
        state = await service.get_state(proposal.committed_state_id)
        return PipelineRouteSettlement(
            settlement=PipelineProposalSettlementResult(proposal=proposal, state=state),
            validation=None,
        )
    if proposal.status != "pending":
        raise HTTPException(status_code=409, detail="Only pending proposals can be accepted.")

    current_record = await service.get_current_state(proposal.session_id)
    current_state = _state_from_record(current_record) if current_record is not None else _initial_composition_state_with_guided_session()
    user_message_content = await _proposal_user_message_content(service, proposal)
    plugin_snapshot = request.app.state.plugin_snapshot_factory(user)
    policy_catalog = PolicyCatalogView(
        request.app.state.catalog_service,
        plugin_snapshot,
        request.app.state.operator_profile_registry,
    )
    recorder = BufferingRecorder()
    recovery = await service.get_pipeline_dispatch_recovery(authority=authority)
    cancellation_state = _DeferredCancellationState()
    try:
        prepared, _ = await _await_with_deferred_cancellation(
            prepare_pipeline_proposal_commit(
                authority=authority,
                reviewed_facts={},
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
                settlement_surface="generic",
                recovery_dispatch=recovery.binding if recovery is not None else None,
                recovery_executor_content_hash=recovery.executor_content_hash if recovery is not None else None,
            ),
            state=cancellation_state,
        )
    except PipelineCommitError as exc:
        try:
            persisted_dispatch = recovery.binding if recovery is not None else exc.dispatch if exc.invocation is None else None
            captured = (exc.invocation,) if exc.invocation is not None else tuple(recorder.invocations)
            if captured:
                bindings, _ = await _await_with_deferred_cancellation(
                    _persist_tool_invocations(
                        service,
                        proposal.session_id,
                        captured,
                        None,
                        plugin_crash_pending=True,
                    ),
                    state=cancellation_state,
                )
                if exc.invocation is not None:
                    if exc.dispatch is None or len(bindings) != 1:
                        raise RuntimeError("pipeline failure dispatch did not persist exactly one rebound binding") from exc
                    persisted_dispatch = bindings[0]
            reason_by_code: dict[str, PipelineProposalRejectionReason] = {
                "CANDIDATE_EXECUTOR_MISMATCH": "candidate_executor_mismatch",
                "VALIDATION_FAILED": "validation_failed",
                "BASE_CONFLICT": "base_conflict",
            }
            reason = reason_by_code.get(exc.code)
            if reason is not None:
                await _await_with_deferred_cancellation(
                    service.reject_pipeline_composition_proposal(
                        session_id=proposal.session_id,
                        proposal_id=proposal.id,
                        draft_hash=authority.proposal.draft_hash,
                        reviewed_facts={},
                        reason=reason,
                        dispatch=persisted_dispatch,
                        actor=f"system:pipeline_commit:user:{user.user_id}",
                    ),
                    state=cancellation_state,
                )
        except BaseException as cleanup_exc:
            if cancellation_state.requested:
                raise asyncio.CancelledError from cleanup_exc
            raise
        if cancellation_state.requested:
            raise asyncio.CancelledError from exc
        status_code = 409 if exc.code in {"BASE_CONFLICT", "NOT_PENDING"} else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except BaseException as exc:
        captured = tuple(recorder.invocations)
        if captured:
            await _await_with_deferred_cancellation(
                _persist_tool_invocations(
                    service,
                    proposal.session_id,
                    captured,
                    None,
                    plugin_crash_pending=True,
                ),
                state=cancellation_state,
            )
        if cancellation_state.requested:
            raise asyncio.CancelledError from exc
        raise

    try:
        if isinstance(prepared, RecoveredPipelineCommit):
            bindings = (prepared.dispatch,)
        else:
            bindings, _ = await _await_with_deferred_cancellation(
                _persist_tool_invocations(
                    service,
                    proposal.session_id,
                    (prepared.invocation,),
                    None,
                    plugin_crash_pending=False,
                ),
                state=cancellation_state,
            )
        if len(bindings) != 1:
            raise RuntimeError("pipeline acceptance dispatch did not persist exactly one binding")
        (state_data, validation), _ = await _await_with_deferred_cancellation(
            _state_data_from_composer_state(
                prepared.result.updated_state,
                settings=request.app.state.settings,
                secret_service=request.app.state.scoped_secret_resolver,
                user_id=str(user.user_id),
                session_id=proposal.session_id,
                plugin_snapshot=plugin_snapshot,
                profile_registry=request.app.state.operator_profile_registry,
                catalog=request.app.state.catalog_service,
                runtime_preflight=prepared.result.runtime_preflight,
                preflight_exception_policy="raise",
                initial_version=current_state.version,
                telemetry_source=telemetry_source,
                composer_meta=composer_meta,
            ),
            state=cancellation_state,
        )
        settled, _ = await _await_with_deferred_cancellation(
            service.settle_pipeline_composition_proposal(
                session_id=proposal.session_id,
                proposal_id=proposal.id,
                draft_hash=draft_hash,
                reviewed_facts={},
                state=state_data,
                candidate_content_hash=prepared.candidate_content_hash,
                executor_content_hash=prepared.executor_content_hash,
                final_composer_metadata=state_data.composer_meta,
                dispatch=bindings[0],
                actor=f"user:{user.user_id}",
                transition_assistant=transition_assistant,
            ),
            state=cancellation_state,
        )
    except BaseException as exc:
        if cancellation_state.requested:
            raise asyncio.CancelledError from exc
        raise
    if cancellation_state.requested:
        raise asyncio.CancelledError
    # Surface resolvable interpretation-review EVENTS for every site the
    # committed pipeline created (llm prompt templates etc.). The planner
    # path mints proposals without the compose loop's
    # request_interpretation_review dispatch, so without this pass the
    # committed state carries pending interpretation_requirements with no
    # event row — the run gate then fails closed
    # (interpretation_placeholder_unresolved) with nothing the user can
    # resolve. Mirrors the guided dispatcher's post-commit surfacing pass;
    # runs after settlement so events bind to the durable state id.
    composer = request.app.state.composer_service
    await composer.surface_pending_interpretation_reviews(
        prepared.result.updated_state,
        session_id=str(proposal.session_id),
        current_state_id=str(settled.state.id),
    )
    return PipelineRouteSettlement(settlement=settled, validation=validation)
