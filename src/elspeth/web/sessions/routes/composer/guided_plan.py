"""Retry-safe guided-full pipeline planning controller."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import UUID, uuid4

from elspeth.contracts.blobs import (
    BlobContentMissingError,
    BlobError,
    BlobGuidedOperationFenceLostError,
    BlobIntegrityError,
    BlobQuotaExceededError,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.pipeline_planner import PipelinePlannerError, PlannerOriginatingMessage
from elspeth.web.composer.pipeline_proposal import PlannerSurface, PresentBase, composition_content_hash
from elspeth.web.composer.proposals import build_tool_proposal_summary
from elspeth.web.composer.protocol import ComposerServiceError
from elspeth.web.composer.redaction import redact_tool_call_arguments
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.sessions.guided_replay import project_composition_proposal
from elspeth.web.sessions.protocol import (
    CompositionStateData,
    GuidedAuditEvidence,
    GuidedFullPipelineProposalStageCommand,
    GuidedOperationFailureCode,
    GuidedOperationFailureCommand,
    GuidedOperationFenceLostError,
    GuidedOperationSettlementConflictError,
    GuidedOriginatingUserMessageDraft,
    GuidedPipelineProposalResult,
)
from elspeth.web.sessions.schemas import CompositionProposalResponse, GuidedPlanRequest

from .._helpers import (
    APIRouter,
    ComposerRateLimiter,
    Depends,
    HTTPException,
    Request,
    SessionServiceProtocol,
    UserIdentity,
    _cancel_on_client_disconnect,
    _composer_progress_sink,
    _get_composer_progress_registry,
    _get_session_compose_lock_registry,
    _is_client_disconnect_cancel,
    _request_plugin_policy_context,
    _state_from_record,
    _track_compose_inflight,
    _verify_session_ownership,
    get_current_user,
    get_rate_limiter,
)
from ..guided_operations import (
    GuidedOperationExpired,
    GuidedOperationLease,
    raise_guided_operation_failure,
    reserve_or_replay_guided_operation,
)
from .pipeline_settlement import (
    _GUIDED_ATOMIC_SETTLEMENT_COMPLETED,
    _GUIDED_ATOMIC_SETTLEMENT_FAILURE,
    _await_guided_atomic_settlement,
    _await_with_deferred_cancellation,
)

router = APIRouter()


def _guided_full_failure_code(exc: BaseException) -> GuidedOperationFailureCode:
    if isinstance(exc, GuidedOperationSettlementConflictError):
        return "stale_conflict"
    if isinstance(exc, AuditIntegrityError):
        return "integrity_error"
    if isinstance(exc, BlobIntegrityError | BlobContentMissingError):
        return "integrity_error"
    if isinstance(exc, BlobQuotaExceededError):
        return "quota_exceeded"
    if isinstance(exc, BlobError):
        return "custody_error"
    if isinstance(exc, ComposerServiceError):
        return "provider_unavailable"
    if isinstance(exc, PipelinePlannerError):
        if exc.code == "TIMEOUT":
            return "provider_timeout"
        if exc.code == "PROVIDER_ERROR":
            return "provider_unavailable"
        if exc.code in {
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
        }:
            return "invalid_provider_response"
    return "operation_failed"


@router.post("/{session_id}/guided/plan", response_model=CompositionProposalResponse)
async def post_guided_plan(
    session_id: UUID,
    body: GuidedPlanRequest,
    request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
    rate_limiter: ComposerRateLimiter = Depends(get_rate_limiter),  # noqa: B008
    _inflight_tally: None = Depends(_track_compose_inflight),
) -> CompositionProposalResponse:
    """Plan and atomically stage one full guided proposal."""

    await rate_limiter.check(user.user_id)
    await _verify_session_ownership(session_id, user, request)
    service: SessionServiceProtocol = request.app.state.session_service

    async def replay(result: object) -> CompositionProposalResponse:
        if type(result) is not GuidedPipelineProposalResult:
            raise AuditIntegrityError("guided-full replay has a non-proposal result locator")
        checkpoint = await service.get_state_in_session(result.checkpoint_state_id, session_id)
        authority = await service.get_authoritative_pipeline_proposal(
            session_id=session_id,
            proposal_id=result.proposal_id,
            reviewed_facts={},
        )
        if (
            authority.proposal.surface is not PlannerSurface.GUIDED_FULL
            or type(authority.proposal.base) is not PresentBase
            or authority.proposal.base.state_id != result.checkpoint_state_id
            or authority.proposal.base.composition_content_hash != composition_content_hash(_state_from_record(checkpoint))
            or authority.row.base_state_id != result.checkpoint_state_id
            or authority.row.user_message_id is None
        ):
            raise AuditIntegrityError("guided-full replay authority differs from its operation locator")
        origin_rows = [
            message for message in await service.get_messages(session_id, limit=None) if message.id == authority.row.user_message_id
        ]
        if (
            len(origin_rows) != 1
            or origin_rows[0].session_id != session_id
            or origin_rows[0].role != "user"
            or origin_rows[0].writer_principal != "route_user_message"
            or origin_rows[0].content != body.intent
            or origin_rows[0].composition_state_id != checkpoint.id
        ):
            raise AuditIntegrityError("guided-full replay originating message authority changed")
        if authority.row.pipeline_metadata is None or authority.row.pipeline_metadata.draft_hash != authority.proposal.draft_hash:
            raise AuditIntegrityError("guided-full replay proposal metadata changed")
        creation_record = replace(
            authority.row,
            status="pending",
            committed_state_id=None,
            audit_event_id=authority.creation_event_id,
            updated_at=authority.row.created_at,
        )
        return project_composition_proposal(creation_record)

    reserved = await reserve_or_replay_guided_operation(
        service=service,
        session_id=session_id,
        kind="guided_plan",
        request=body,
        replay=replay,
    )
    if reserved is None:  # pragma: no cover - reserve defaults to true
        raise AuditIntegrityError("guided-full operation was not reserved")
    if not isinstance(reserved, GuidedOperationLease):
        return reserved

    recorder = BufferingRecorder()
    try:
        catalog, plugin_snapshot = _request_plugin_policy_context(request, user)
        compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))
        progress = _composer_progress_sink(
            _get_composer_progress_registry(request),
            session_id=str(session_id),
            request_id=body.operation_id,
            user_id=user.user_id,
        )
        async with compose_lock:
            observed_record = await service.get_current_state(session_id)
            observed_state = (
                _state_from_record(observed_record)
                if observed_record is not None
                else CompositionState(
                    source=None,
                    nodes=(),
                    edges=(),
                    outputs=(),
                    metadata=PipelineMetadata(),
                    version=1,
                )
            )
            expected_state_id = observed_record.id if observed_record is not None else None
            expected_state_version = observed_record.version if observed_record is not None else None
            expected_content_hash = composition_content_hash(observed_state) if observed_record is not None else None
            checkpoint_id = uuid4()
            proposal_id = uuid4()
            origin = GuidedOriginatingUserMessageDraft(message_id=uuid4(), content=body.intent)
            fence = await service.renew_guided_operation(
                reserved.fence,
                actor="composer_route",
                lease_seconds=300,
            )

        async with _cancel_on_client_disconnect(request):
            plan, _catalog_ids = await request.app.state.composer_service.plan_guided_full_pipeline(
                intent=body.intent,
                current_state=observed_state,
                originating_message=PlannerOriginatingMessage(
                    session_id=str(session_id),
                    message_id=str(origin.message_id),
                    content=origin.content,
                    user_id=user.user_id,
                ),
                base=PresentBase(
                    state_id=checkpoint_id,
                    composition_content_hash=composition_content_hash(observed_state),
                ),
                policy_catalog=catalog,
                plugin_snapshot=plugin_snapshot,
                recorder=recorder,
                operation_fence=fence,
                progress=progress,
            )

        redacted = redact_tool_call_arguments(
            "set_pipeline",
            deep_thaw(plan.proposal.pipeline),
            telemetry=NoopRedactionTelemetry(),
        )
        summary = build_tool_proposal_summary(
            tool_name="set_pipeline",
            arguments=deep_thaw(plan.proposal.pipeline),
            redacted_arguments=redacted,
        )
        state_dict = observed_state.to_dict()
        checkpoint_data = CompositionStateData(
            sources=state_dict["sources"],
            nodes=state_dict["nodes"],
            edges=state_dict["edges"],
            outputs=state_dict["outputs"],
            metadata_=state_dict["metadata"],
            is_valid=observed_record.is_valid if observed_record is not None else False,
            validation_errors=observed_record.validation_errors if observed_record is not None else None,
            composer_meta=observed_record.composer_meta if observed_record is not None else None,
        )
        async with compose_lock:
            renewed_fence = await service.renew_guided_operation(
                fence,
                actor="composer_route",
                lease_seconds=300,
            )
            settlement = await _await_guided_atomic_settlement(
                service.stage_guided_full_pipeline_proposal(
                    GuidedFullPipelineProposalStageCommand(
                        fence=renewed_fence,
                        expected_current_state_id=expected_state_id,
                        expected_current_state_version=expected_state_version,
                        expected_current_content_hash=expected_content_hash,
                        checkpoint_state_id=checkpoint_id,
                        proposal_id=proposal_id,
                        state=checkpoint_data,
                        plan=plan,
                        summary=summary.summary,
                        rationale=summary.rationale,
                        affects=summary.affects,
                        arguments_redacted_json=summary.arguments_redacted_json,
                        actor="composer_route",
                        originating_message=origin,
                        audit_evidence=GuidedAuditEvidence(
                            invocations=recorder.invocations,
                            llm_calls=recorder.llm_calls,
                            chat_turns=recorder.chat_turns,
                        ),
                    )
                )
            )
        return project_composition_proposal(settlement.proposal)
    except (GuidedOperationFenceLostError, BlobGuidedOperationFenceLostError) as exc:
        joined = await reserve_or_replay_guided_operation(
            service=service,
            session_id=session_id,
            kind="guided_plan",
            request=body,
            replay=replay,
            reserve_if_absent=False,
            takeover_expired=False,
        )
        if joined is None or isinstance(joined, (GuidedOperationLease, GuidedOperationExpired)):
            raise AuditIntegrityError("guided-full fence was lost without a replayable winner") from exc
        return joined
    except asyncio.CancelledError as exc:
        if exc.__dict__.get(_GUIDED_ATOMIC_SETTLEMENT_COMPLETED) is True:
            raise
        settlement_failure = exc.__dict__.get(_GUIDED_ATOMIC_SETTLEMENT_FAILURE)
        caller_task = asyncio.current_task()
        caller_cancelled = caller_task is not None and caller_task.cancelling() > 0
        disconnected = _is_client_disconnect_cancel(exc)
        cancel_failure_code: GuidedOperationFailureCode = (
            _guided_full_failure_code(settlement_failure)
            if settlement_failure is not None
            else "request_cancelled"
            if disconnected or caller_cancelled
            else "operation_failed"
        )
        try:
            (failed, _cancelled_during_failure_settlement) = await _await_with_deferred_cancellation(
                service.fail_guided_operation_with_audit(
                    GuidedOperationFailureCommand(
                        fence=reserved.fence,
                        failure_code=cancel_failure_code,
                        actor="composer_route",
                        audit_evidence=GuidedAuditEvidence(
                            invocations=recorder.invocations,
                            llm_calls=recorder.llm_calls,
                            chat_turns=recorder.chat_turns,
                        ),
                    )
                )
            )
        except GuidedOperationFenceLostError as fence_lost:
            raise exc from fence_lost
        if settlement_failure is not None:
            raise exc from settlement_failure
        if disconnected:
            raise HTTPException(
                status_code=499,
                detail="Client disconnected while guided full planning was running.",
            ) from exc
        if caller_cancelled:
            raise
        raise_guided_operation_failure(failed)
    except Exception as exc:
        failure_code = _guided_full_failure_code(exc)
        try:
            failed = await service.fail_guided_operation_with_audit(
                GuidedOperationFailureCommand(
                    fence=reserved.fence,
                    failure_code=failure_code,
                    actor="composer_route",
                    audit_evidence=GuidedAuditEvidence(
                        invocations=recorder.invocations,
                        llm_calls=recorder.llm_calls,
                        chat_turns=recorder.chat_turns,
                    ),
                )
            )
        except GuidedOperationFenceLostError:
            joined = await reserve_or_replay_guided_operation(
                service=service,
                session_id=session_id,
                kind="guided_plan",
                request=body,
                replay=replay,
                reserve_if_absent=False,
                takeover_expired=False,
            )
            if joined is None or isinstance(joined, (GuidedOperationLease, GuidedOperationExpired)):
                raise AuditIntegrityError("guided-full failure lost its fence without a winner") from exc
            return joined
        raise_guided_operation_failure(failed)


__all__ = ["router"]
