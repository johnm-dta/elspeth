"""Atomic schema-8 implementation for the guided Chat mutation surface."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import HTTPException, Request

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import stable_hash
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import ControlSignal, GuidedStep, Turn, TurnType
from elspeth.web.composer.guided.resolved import SinkResolved
from elspeth.web.composer.pipeline_proposal import composition_content_hash
from elspeth.web.sessions._guided_step_chat import (
    StepChatResult,
    resolve_step_1_source_chat_with_auto_drop,
    resolve_step_2_sink_chat_with_auto_drop,
    solve_step_chat_with_auto_drop,
)
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
    GuidedOperationFailureCode,
    GuidedOperationFenceLostError,
    GuidedOperationSettlementConflictError,
    GuidedOriginatingUserMessageDraft,
    GuidedReplayTurn,
    GuidedResponseDescriptor,
    GuidedStateOperationCommand,
    PreparedGuidedJsonPayload,
    SessionServiceProtocol,
)
from elspeth.web.sessions.schemas import GuidedChatRequest, GuidedChatResponse, GuidedRespondRequest

from .._helpers import (
    BufferingRecorder,
    ChatRole,
    ChatTurn,
    ComposerChatInitiator,
    ComposerChatTurn,
    ComposerChatTurnStatus,
    ComposerProgressEvent,
    CompositionState,
    CompositionStateData,
    CompositionStateRecord,
    UserIdentity,
    _cancel_on_client_disconnect,
    _composer_progress_sink,
    _get_composer_progress_registry,
    _get_session_compose_lock_registry,
    _initial_composition_state_with_guided_session,
    _is_client_disconnect_cancel,
    _publish_progress,
    _replace,
    _request_plugin_policy_context,
    _safe_frame_strings,
    _state_from_record,
    client_cancelled_progress_event,
    deep_thaw,
    emit_step_advanced,
    emit_turn_answered,
    emit_turn_emitted,
    slog,
)
from ..guided_operations import (
    GuidedOperationExpired,
    GuidedOperationLease,
    raise_guided_operation_failure,
    reserve_or_replay_guided_operation,
)

ProviderRunner = Callable[..., Awaitable[tuple[StepChatResult, Any | None, SinkResolved | None]]]


@dataclass(frozen=True, slots=True)
class _ChatPreflight:
    """Frozen authority observed before reservation/provider work."""

    state_record: CompositionStateRecord | None
    state: CompositionState
    guided: Any
    current_turn: Turn
    current_payload: PreparedGuidedJsonPayload


def _unsupported_stage(step: GuidedStep) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "code": "guided_chat_stage_unsupported",
            "detail": f"Schema-8 CHAT is not available for {step.value}.",
        },
    )


def _current_source(guided: Any) -> Any | None:
    target = guided.active_edit_target
    if target is not None and target.kind == "source":
        return guided.reviewed_sources[target.stable_id]
    return next((guided.reviewed_sources[item] for item in guided.source_order if item in guided.reviewed_sources), None)


def _current_sink(guided: Any) -> SinkResolved | None:
    outputs = tuple(guided.reviewed_outputs[item] for item in guided.output_order if item in guided.reviewed_outputs)
    return SinkResolved(outputs=outputs) if outputs else None


async def run_guided_chat_provider_attempt(
    *,
    session_id: UUID,
    user: UserIdentity,
    step: GuidedStep,
    guided: Any,
    state: CompositionState,
    message: str,
    settings: Any,
    catalog: Any,
    plugin_snapshot: Any,
    secret_service: Any,
    recorder: BufferingRecorder,
    progress: Any,
) -> tuple[StepChatResult, Any | None, SinkResolved | None]:
    """Run the only provider-bearing phase, with no compose lock held."""

    from elspeth.web.composer.guided.chat_solver import build_step_chat_context_block

    source = _current_source(guided)
    sink = _current_sink(guided)
    context_block = build_step_chat_context_block(step=step, current_source=source, current_sink=sink, state=state)
    started = perf_counter()

    if step is GuidedStep.STEP_1_SOURCE:
        plugin_hint = None
        target = guided.active_edit_target
        if target is not None and target.kind == "source":
            plugin_hint = guided.reviewed_sources[target.stable_id].plugin
        if plugin_hint is None:
            pending = next(
                (guided.pending_source_intents[item] for item in guided.source_order if item in guided.pending_source_intents),
                None,
            )
            plugin_hint = pending.plugin if pending is not None else None
        source_outcome = await resolve_step_1_source_chat_with_auto_drop(
            site="post_guided_chat",
            session_id=str(session_id),
            user_id=user.user_id,
            model=settings.composer_model,
            user_message=message,
            plugin_hint=plugin_hint,
            current_source=source,
            temperature=settings.composer_temperature,
            seed=settings.composer_seed,
            recorder=recorder,
            timeout_seconds=settings.composer_timeout_seconds,
            context_block=context_block,
        )
        if source_outcome.source_resolution is not None:
            return (
                StepChatResult(
                    assistant_message=source_outcome.source_resolution.assistant_message,
                    status=ComposerChatTurnStatus.SUCCESS,
                    latency_ms=int((perf_counter() - started) * 1000),
                    error_class=None,
                ),
                source_outcome.source_resolution,
                None,
            )
        direct = source_outcome.fallback_chat or source_outcome.prose_chat
        if direct is not None:
            return direct, None, None

    elif step is GuidedStep.STEP_2_SINK:
        sink_outcome = await resolve_step_2_sink_chat_with_auto_drop(
            site="post_guided_chat",
            session_id=str(session_id),
            user_id=user.user_id,
            model=settings.composer_model,
            user_message=message,
            current_sink=sink,
            temperature=settings.composer_temperature,
            seed=settings.composer_seed,
            recorder=recorder,
            state=state,
            catalog=catalog,
            plugin_snapshot=plugin_snapshot,
            secret_service=secret_service,
            max_discovery_iters=settings.composer_max_discovery_turns,
            timeout_seconds=settings.composer_timeout_seconds,
            context_block=context_block,
            progress=progress,
        )
        if sink_outcome.sink_resolution is not None:
            return (
                StepChatResult(
                    assistant_message=sink_outcome.assistant_message or "Output configuration prepared.",
                    status=ComposerChatTurnStatus.SUCCESS,
                    latency_ms=int((perf_counter() - started) * 1000),
                    error_class=None,
                ),
                None,
                sink_outcome.sink_resolution,
            )
        direct = sink_outcome.fallback_chat or sink_outcome.prose_chat
        if direct is not None:
            return direct, None, None
    else:  # pragma: no cover - preflight owns the closed stage gate
        raise AuditIntegrityError("Guided Chat provider received an unsupported schema-8 stage")

    advisory = await solve_step_chat_with_auto_drop(
        site="post_guided_chat",
        session_id=str(session_id),
        user_id=user.user_id,
        model=settings.composer_model,
        step=step,
        user_message=message,
        temperature=settings.composer_temperature,
        seed=settings.composer_seed,
        recorder=recorder,
        timeout_seconds=settings.composer_timeout_seconds,
        context_block=context_block,
    )
    return advisory, None, None


def _transition_request(
    *,
    body: GuidedChatRequest,
    guided: Any,
    current_turn: Turn,
    source_resolution: Any | None,
    sink_resolution: SinkResolved | None,
) -> GuidedRespondRequest | None:
    turn_type = TurnType(current_turn["type"])
    common = {"operation_id": body.operation_id, "turn_token": body.turn_token}
    if source_resolution is not None and guided.step is GuidedStep.STEP_1_SOURCE:
        if turn_type is TurnType.SINGLE_SELECT:
            return GuidedRespondRequest.model_validate({**common, "chosen": [source_resolution.plugin]}, strict=True)
        if turn_type is TurnType.SCHEMA_FORM:
            # Applying generated bytes requires the blob row, originating
            # message, state and operation result to share one transaction.
            # Until that custody participant exists, schema-form Chat is
            # deliberately advisory and the wizard remains authoritative.
            return None
        if turn_type is TurnType.INSPECT_AND_CONFIRM:
            return GuidedRespondRequest.model_validate(
                {**common, "edited_values": {"columns": list(source_resolution.observed_columns)}},
                strict=True,
            )
        return None

    if sink_resolution is not None and guided.step is GuidedStep.STEP_2_SINK:
        (output,) = sink_resolution.outputs
        if turn_type is TurnType.SINGLE_SELECT:
            return GuidedRespondRequest.model_validate({**common, "chosen": [output.plugin]}, strict=True)
        if turn_type is TurnType.SCHEMA_FORM:
            options = dict(deep_thaw(output.options))
            options["on_write_failure"] = output.on_write_failure
            return GuidedRespondRequest.model_validate(
                {**common, "edited_values": {"plugin": output.plugin, "options": options}},
                strict=True,
            )
        if turn_type is TurnType.MULTI_SELECT_WITH_CUSTOM:
            candidates = {
                column
                for stable_id in guided.source_order
                if stable_id in guided.reviewed_sources
                for column in guided.reviewed_sources[stable_id].observed_columns
            }
            chosen = [field for field in output.required_fields if field in candidates]
            custom = [field for field in output.required_fields if field not in candidates]
            payload: dict[str, object] = {**common, "chosen": chosen, "custom_inputs": custom}
            if not chosen and not custom:
                payload["control_signal"] = ControlSignal.PASSTHROUGH.value
            return GuidedRespondRequest.model_validate(payload, strict=True)
    return None


async def post_guided_chat_schema8(
    *,
    session_id: UUID,
    body: GuidedChatRequest,
    request: Request,
    user: UserIdentity,
    provider_runner: ProviderRunner,
) -> GuidedChatResponse:
    """Reserve, run, and atomically settle one schema-8 guided Chat turn."""

    from . import guided as guided_route

    service: SessionServiceProtocol = request.app.state.session_service
    payload_store = request.app.state.payload_store
    compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))

    def response_from_record(record: CompositionStateRecord) -> GuidedChatResponse:
        descriptor = parse_guided_response_descriptor(record)
        if descriptor.kind != "guided_chat":
            raise AuditIntegrityError("Guided Chat result has the wrong replay descriptor")
        payloads: tuple[PreparedGuidedJsonPayload, ...] = ()
        if descriptor.next_turn is not None:
            payloads = (load_guided_json_payload(payload_store, payload_id=descriptor.next_turn.payload_id, purpose="turn"),)
        response = project_guided_response(record, payloads=payloads)
        if type(response) is not GuidedChatResponse:
            raise AuditIntegrityError("Guided Chat projection returned the wrong response type")
        return response

    async def replay(result: object) -> GuidedChatResponse:
        if type(result) is not GuidedCompositionStateResult:
            raise AuditIntegrityError("Guided Chat replay has a non-state result locator")
        return response_from_record(await service.get_state_in_session(result.state_id, session_id))

    pending = await reserve_or_replay_guided_operation(
        service=service,
        session_id=session_id,
        kind="guided_chat",
        request=body,
        replay=replay,
        reserve_if_absent=False,
        takeover_expired=False,
    )
    if pending is not None and not isinstance(pending, (GuidedOperationLease, GuidedOperationExpired)):
        return pending

    catalog, plugin_snapshot = _request_plugin_policy_context(request, user)
    shield_available = guided_route._resolve_shield_available(plugin_snapshot)

    async def preflight() -> _ChatPreflight:
        state_record = await service.get_current_state(session_id)
        state = _state_from_record(state_record) if state_record is not None else _initial_composition_state_with_guided_session()
        guided = state.guided_session
        if guided is None:
            raise HTTPException(status_code=400, detail="Session is not in guided mode. Use /api/sessions/{id}/messages.")
        if guided.terminal is not None:
            raise HTTPException(status_code=409, detail="Guided session is already terminal.")
        if guided.step not in {GuidedStep.STEP_1_SOURCE, GuidedStep.STEP_2_SINK}:
            raise _unsupported_stage(guided.step)
        prospective, current_turn, current_payload = guided_route._schema8_prospective_occurrence(
            state,
            guided,
            catalog=catalog,
            shield_available=shield_available,
            payload_store=payload_store,
        )
        if body.turn_token != guided_turn_token(prospective):
            raise HTTPException(status_code=409, detail="turn_token does not identify the current unanswered turn.")
        return _ChatPreflight(
            state_record=state_record,
            state=state,
            guided=prospective,
            current_turn=current_turn,
            current_payload=current_payload,
        )

    admission_lock = await _get_session_compose_lock_registry(request).get_lock(f"{session_id}:guided-chat-admission")
    while True:
        rejoin_after_lock = False
        frozen: _ChatPreflight | None = None
        bypass_admission = isinstance(pending, GuidedOperationExpired)
        if bypass_admission:
            frozen = await preflight()
            pending = await reserve_or_replay_guided_operation(
                service=service,
                session_id=session_id,
                kind="guided_chat",
                request=body,
                replay=replay,
            )
            if pending is None:  # pragma: no cover
                raise AuditIntegrityError("Guided Chat takeover was not reserved")
            if not isinstance(pending, GuidedOperationLease):
                return pending

        attempt_guard = contextlib.nullcontext() if bypass_admission else admission_lock
        async with attempt_guard:
            if pending is None:
                rechecked = await reserve_or_replay_guided_operation(
                    service=service,
                    session_id=session_id,
                    kind="guided_chat",
                    request=body,
                    replay=replay,
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
                async with compose_lock:
                    frozen = await preflight()
            if frozen is None:  # pragma: no cover
                raise AuditIntegrityError("Guided Chat has no frozen preflight authority")

            reserved = pending or await reserve_or_replay_guided_operation(
                service=service,
                session_id=session_id,
                kind="guided_chat",
                request=body,
                replay=replay,
            )
            pending = None
            if not isinstance(reserved, GuidedOperationLease):
                if reserved is None or isinstance(reserved, GuidedOperationExpired):  # pragma: no cover
                    raise AuditIntegrityError("Guided Chat reservation returned no lease")
                return reserved

            recorder = BufferingRecorder()
            attempt_message_id = uuid4()
            progress_started = False
            progress_registry = _get_composer_progress_registry(request)
            try:
                progress_sink = _composer_progress_sink(
                    progress_registry,
                    session_id=str(session_id),
                    request_id=body.operation_id,
                    user_id=str(user.user_id),
                )
                await _publish_progress(
                    progress_registry,
                    session_id=str(session_id),
                    request_id=body.operation_id,
                    user_id=str(user.user_id),
                    event=ComposerProgressEvent(
                        phase="starting",
                        headline="I'm reading your message for this guided turn.",
                        evidence=("The message and current turn token were accepted.",),
                        likely_next="ELSPETH will prepare a bounded guided response.",
                    ),
                )
                progress_started = True
                settings = request.app.state.settings
                started_at = datetime.now(UTC)
                async with _cancel_on_client_disconnect(request):
                    chat_result, source_resolution, sink_resolution = await provider_runner(
                        session_id=session_id,
                        user=user,
                        step=frozen.guided.step,
                        guided=frozen.guided,
                        state=frozen.state,
                        message=body.message,
                        settings=settings,
                        catalog=catalog,
                        plugin_snapshot=plugin_snapshot,
                        secret_service=request.app.state.scoped_secret_resolver,
                        recorder=recorder,
                        progress=progress_sink,
                    )
                    if source_resolution is not None and TurnType(frozen.current_turn["type"]) is TurnType.SCHEMA_FORM:
                        source_resolution = None
                        chat_result = StepChatResult(
                            assistant_message=(
                                "I did not apply generated source content. Review the current source form and "
                                "submit it through the wizard controls."
                            ),
                            status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
                            latency_ms=chat_result.latency_ms,
                            error_class="InlineSourceNotApplied",
                        )

                async with compose_lock:
                    fence = await service.renew_guided_operation(reserved.fence, actor="composer_route", lease_seconds=300)
                    current_record = await service.get_current_state(session_id)
                    if (current_record is None) != (frozen.state_record is None):
                        raise GuidedOperationSettlementConflictError()
                    if (
                        current_record is not None
                        and frozen.state_record is not None
                        and (current_record.id != frozen.state_record.id or current_record.version != frozen.state_record.version)
                    ):
                        raise GuidedOperationSettlementConflictError()
                    current_state = (
                        _state_from_record(current_record)
                        if current_record is not None
                        else _initial_composition_state_with_guided_session()
                    )
                    if current_record is not None and composition_content_hash(current_state) != composition_content_hash(frozen.state):
                        raise AuditIntegrityError("Guided Chat current state content changed after provider work")
                    current_guided = current_state.guided_session
                    if current_guided is None:
                        raise AuditIntegrityError("Guided Chat head lost its guided checkpoint")
                    prospective, current_turn, planned_current = guided_route._schema8_prospective_occurrence(
                        current_state,
                        current_guided,
                        catalog=catalog,
                        shield_available=shield_available,
                        payload_store=payload_store,
                    )
                    if body.turn_token != guided_turn_token(prospective) or planned_current.payload_id != frozen.current_payload.payload_id:
                        raise AuditIntegrityError("Guided Chat turn custody changed after provider work")

                    occurrence_was_prospective = not (current_guided.history and current_guided.history[-1].response_hash is None)
                    transition_body = _transition_request(
                        body=body,
                        guided=prospective,
                        current_turn=current_turn,
                        source_resolution=source_resolution,
                        sink_resolution=sink_resolution,
                    )
                    resulting_state = _replace(current_state, guided_session=prospective)
                    planned_response: PreparedGuidedJsonPayload | None = None
                    next_turn: Turn | None = current_turn
                    prepared_next: PreparedGuidedJsonPayload | None = planned_current
                    transition_succeeded = False
                    if transition_body is not None:
                        try:
                            resulting_state, planned_response, next_turn, prepared_next = guided_route._schema8_answer_and_project_next(
                                current_state,
                                prospective,
                                current_turn,
                                transition_body,
                                catalog=catalog,
                                shield_available=shield_available,
                                new_stable_id=uuid4(),
                            )
                            transition_succeeded = True
                        except (PluginConfigError, InvariantError, TypeError, ValueError):
                            chat_result = StepChatResult(
                                assistant_message=(
                                    "I couldn't apply that configuration, so I didn't change your pipeline. "
                                    "Review the wizard fields and try again, or keep going with the wizard controls."
                                ),
                                status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
                                latency_ms=chat_result.latency_ms,
                                error_class="StepTransitionRejected",
                            )
                            next_turn = current_turn
                            prepared_next = planned_current

                    resulting_guided = resulting_state.guided_session
                    if resulting_guided is None:  # pragma: no cover
                        raise AuditIntegrityError("Guided Chat transition removed its checkpoint")
                    finished_at = datetime.now(UTC)
                    user_turn = ChatTurn(
                        role=ChatRole.USER,
                        content=body.message,
                        seq=resulting_guided.chat_turn_seq,
                        step=prospective.step,
                        ts_iso=finished_at.isoformat(),
                    )
                    assistant_kind: Literal["assistant", "synthetic_failure"] = (
                        "assistant" if chat_result.status is ComposerChatTurnStatus.SUCCESS else "synthetic_failure"
                    )
                    assistant_turn = ChatTurn(
                        role=ChatRole.ASSISTANT,
                        content=chat_result.assistant_message,
                        seq=resulting_guided.chat_turn_seq + 1,
                        step=prospective.step,
                        ts_iso=finished_at.isoformat(),
                        assistant_message_kind=assistant_kind,
                        synthetic_failure_reason=(
                            None
                            if assistant_kind == "assistant"
                            else "not_applied"
                            if chat_result.error_class in {"StepTransitionRejected", "InlineSourceNotApplied"}
                            else "quality_guard"
                            if chat_result.error_class == "AssistantScaffoldLeakError"
                            else "unavailable"
                        ),
                    )
                    resulting_guided = _replace(
                        resulting_guided,
                        chat_history=(*resulting_guided.chat_history, user_turn, assistant_turn),
                        chat_turn_seq=resulting_guided.chat_turn_seq + 2,
                    )
                    resulting_state = _replace(resulting_state, guided_session=resulting_guided)
                    recorder.record_chat_turn(
                        ComposerChatTurn(
                            step=prospective.step.value,
                            initiator=ComposerChatInitiator.USER,
                            chat_turn_seq=user_turn.seq,
                            user_message_hash=stable_hash(body.message),
                            assistant_message_hash=stable_hash(chat_result.assistant_message),
                            latency_ms=chat_result.latency_ms,
                            model=settings.composer_model,
                            status=chat_result.status,
                            started_at=started_at,
                            finished_at=finished_at,
                            error_class=chat_result.error_class,
                        )
                    )

                    audit = BufferingRecorder()
                    if occurrence_was_prospective:
                        emit_turn_emitted(
                            audit,
                            step=prospective.step,
                            turn_type=TurnType(current_turn["type"]),
                            payload_hash=planned_current.payload_id,
                            payload_payload_id=planned_current.payload_id,
                            emitter="server",
                            composition_version=current_state.version,
                            actor=user.user_id,
                        )
                    if transition_succeeded and planned_response is not None:
                        emit_turn_answered(
                            audit,
                            step=prospective.step,
                            turn_type=TurnType(current_turn["type"]),
                            response_hash=planned_response.payload_id,
                            response_payload_id=planned_response.payload_id,
                            control_signal=None,
                            composition_version=current_state.version,
                            actor=user.user_id,
                        )
                        if resulting_guided.step is not prospective.step:
                            emit_step_advanced(
                                audit,
                                prev=prospective.step,
                                next_=resulting_guided.step,
                                reason="user_advanced",
                                composition_version=current_state.version,
                                actor=user.user_id,
                            )
                        if prepared_next is not None and next_turn is not None:
                            emit_turn_emitted(
                                audit,
                                step=resulting_guided.step,
                                turn_type=TurnType(next_turn["type"]),
                                payload_hash=prepared_next.payload_id,
                                payload_payload_id=prepared_next.payload_id,
                                emitter="server",
                                composition_version=current_state.version,
                                actor=user.user_id,
                            )

                    prepared_payloads: list[PreparedGuidedJsonPayload] = []
                    for planned in (planned_current, planned_response, prepared_next):
                        if planned is None or planned.payload_id in {item.payload_id for item in prepared_payloads}:
                            continue
                        prepared = prepare_guided_json_payload(
                            payload_store,
                            purpose=planned.purpose,
                            payload=planned.payload,
                        )
                        if prepared.payload_id != planned.payload_id:
                            raise AuditIntegrityError("Guided Chat payload changed before settlement")
                        prepared_payloads.append(prepared)

                    existing_meta = (
                        dict(deep_thaw(current_record.composer_meta))
                        if current_record is not None and current_record.composer_meta is not None
                        else {}
                    )
                    existing_meta["guided_session"] = resulting_guided.to_dict()
                    state_dict = resulting_state.to_dict()
                    is_valid, validation_errors = guided_route._guided_persisted_validity(resulting_state, catalog=catalog)
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
                    evidence = GuidedAuditEvidence(
                        invocations=(*audit.invocations, *recorder.invocations),
                        llm_calls=recorder.llm_calls,
                        chat_turns=recorder.chat_turns,
                    )
                    await _publish_progress(
                        progress_registry,
                        session_id=str(session_id),
                        request_id=body.operation_id,
                        user_id=str(user.user_id),
                        event=ComposerProgressEvent(
                            phase="saving",
                            headline="I'm saving this guided turn.",
                            evidence=("The provider response passed the guided transition checks.",),
                            likely_next="ELSPETH will finish the atomic state and audit settlement.",
                        ),
                    )
                    settlement = await service.settle_guided_state_operation(
                        GuidedStateOperationCommand(
                            fence=fence,
                            expected_current_state_id=current_record.id if current_record is not None else None,
                            expected_current_state_version=current_record.version if current_record is not None else None,
                            expected_current_content_hash=(composition_content_hash(current_state) if current_record is not None else None),
                            state_id=uuid4(),
                            state=state_data,
                            provenance="convergence_persist",
                            actor="composer_route",
                            response=GuidedResponseDescriptor(
                                kind="guided_chat",
                                next_turn=replay_turn,
                                assistant_turn_seq=assistant_turn.seq,
                            ),
                            payloads=tuple(prepared_payloads),
                            audit_evidence=evidence,
                            originating_message=GuidedOriginatingUserMessageDraft(
                                message_id=attempt_message_id,
                                content=body.message,
                            ),
                        ),
                        payload_store=payload_store,
                    )
                    response = response_from_record(settlement.result_state)

                await _publish_progress(
                    progress_registry,
                    session_id=str(session_id),
                    request_id=body.operation_id,
                    user_id=str(user.user_id),
                    event=ComposerProgressEvent(
                        phase="complete",
                        headline="ELSPETH finished responding to this guided message.",
                        evidence=("The guided chat turn was settled atomically.",),
                        likely_next="Review the reply and continue the wizard.",
                        reason="composer_complete",
                    ),
                )
                return response
            except GuidedOperationFenceLostError:
                rejoin_after_lock = True
            except asyncio.CancelledError as exc:
                with contextlib.suppress(Exception):
                    await asyncio.shield(
                        service.fail_guided_operation(
                            reserved.fence,
                            failure_code="operation_failed",
                            actor="composer_route",
                        )
                    )
                if progress_started:
                    with contextlib.suppress(Exception):
                        await asyncio.shield(
                            _publish_progress(
                                progress_registry,
                                session_id=str(session_id),
                                request_id=body.operation_id,
                                user_id=str(user.user_id),
                                event=client_cancelled_progress_event(),
                            )
                        )
                if _is_client_disconnect_cancel(exc):
                    raise HTTPException(status_code=499, detail="Client disconnected while the guided chat turn was running.") from exc
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
                        site="post_guided_chat",
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
                else:
                    raise_guided_operation_failure(failed)
        if rejoin_after_lock:
            joined = await reserve_or_replay_guided_operation(
                service=service,
                session_id=session_id,
                kind="guided_chat",
                request=body,
                replay=replay,
                reserve_if_absent=False,
            )
            if joined is None:
                raise AuditIntegrityError("Guided Chat fence was lost without a joinable winner")
            if isinstance(joined, GuidedOperationLease):
                pending = joined
                continue
            return joined
        raise AuditIntegrityError("Guided Chat settlement loop exited without a result")


__all__ = ["post_guided_chat_schema8", "run_guided_chat_provider_attempt"]
