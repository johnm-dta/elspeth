"""Direct dispatcher coverage for the STEP_4_WIRE skeleton route path."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any
from uuid import uuid4

import pytest
from fastapi import HTTPException

from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.profile import TUTORIAL_PROFILE
from elspeth.web.composer.guided.protocol import GuidedStep, TurnResponse, TurnType
from elspeth.web.composer.guided.state_machine import (
    ChainProposal,
    GuidedSession,
    SinkOutputResolved,
    SinkResolved,
    SourceResolved,
    TerminalKind,
)
from elspeth.web.composer.guided.steps import (
    handle_step_1_source,
    handle_step_2_sink,
    handle_step_3_chain_accept,
    handle_step_4_wire_confirm,
)
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.sessions.routes import _helpers as guided_route_helpers
from elspeth.web.sessions.routes._helpers import _dispatch_guided_respond, _summarize_guided_response
from tests.fixtures.stores import MockPayloadStore


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _valid_confirm_body() -> TurnResponse:
    return {
        "chosen": ["confirm"],
        "edited_values": None,
        "custom_inputs": None,
        "accepted_step_index": None,
        "edit_step_index": None,
        "control_signal": None,
    }


def _proposal() -> ChainProposal:
    return ChainProposal(
        steps=(
            {
                "plugin": "passthrough",
                "options": {"schema": {"mode": "observed"}},
                "rationale": "identity transform",
            },
        ),
        why="source rows already match the sink schema",
    )


def _step3_ready_session() -> tuple[CompositionState, GuidedSession, Any, MockPayloadStore]:
    state = _empty_state()
    session = GuidedSession.initial(profile=TUTORIAL_PROFILE)
    catalog = create_catalog_service()

    step_1 = handle_step_1_source(
        state=state,
        session=session,
        catalog=catalog,
        resolved=SourceResolved(
            plugin="csv",
            options={"path": "x.csv", "schema": {"mode": "observed"}},
            observed_columns=("price",),
            sample_rows=({"price": "1.99"},),
        ),
    )
    step_2 = handle_step_2_sink(
        state=step_1.state,
        session=step_1.session,
        catalog=catalog,
        resolved=SinkResolved(
            outputs=(
                SinkOutputResolved(
                    plugin="json",
                    options={"path": "out.jsonl", "schema": {"mode": "observed"}},
                    required_fields=("price",),
                    schema_mode="observed",
                ),
            ),
        ),
    )
    guided = replace(
        step_2.session,
        step=GuidedStep.STEP_3_TRANSFORMS,
        step_3_proposal=_proposal(),
    )
    return step_2.state, guided, catalog, MockPayloadStore()


def _wire_ready_session(*, valid: bool = True) -> tuple[CompositionState, GuidedSession, Any, MockPayloadStore]:
    if not valid:
        return (
            _empty_state(),
            replace(GuidedSession.initial(), step=GuidedStep.STEP_4_WIRE),
            create_catalog_service(),
            MockPayloadStore(),
        )

    state, step3_session, catalog, payload_store = _step3_ready_session()
    result = handle_step_3_chain_accept(
        state=state,
        session=step3_session,
        proposal=step3_session.step_3_proposal,
        catalog=catalog,
    )
    assert result.tool_result.success is True
    assert result.session.step is GuidedStep.STEP_4_WIRE
    assert result.session.terminal is None
    return result.state, result.session, catalog, payload_store


async def _dispatch(
    state: CompositionState,
    guided: GuidedSession,
    catalog: Any,
    *,
    payload_store: MockPayloadStore,
    current_step: GuidedStep,
    current_turn_type: TurnType,
    turn_response: TurnResponse,
    recorder: BufferingRecorder | None = None,
) -> tuple[CompositionState, GuidedSession, Any | None, BufferingRecorder]:
    recorder = recorder or BufferingRecorder()
    state2, guided2, next_turn = await _dispatch_guided_respond(
        state=state,
        guided=guided,
        current_step=current_step,
        current_turn_type=current_turn_type,
        turn_response=turn_response,
        catalog=catalog,
        recorder=recorder,
        user_id="test-user",
        data_dir=None,
        session_engine=None,
        session_id=str(uuid4()),
        blob_service=None,
        payload_store=payload_store,
        model="test-model",
        temperature=None,
        seed=None,
    )
    return state2, guided2, next_turn, recorder


def _audit_args(recorder: BufferingRecorder, tool_name: str) -> list[dict[str, Any]]:
    return [json.loads(invocation.arguments_canonical) for invocation in recorder.invocations if invocation.tool_name == tool_name]


def _assert_final_wire_payload_stored(
    *,
    next_turn: Any,
    emitted_args: dict[str, Any],
    payload_store: MockPayloadStore,
) -> None:
    payload = next_turn["payload"]

    assert set(payload) == {
        "topology",
        "edge_contracts",
        "semantic_contracts",
        "warnings",
    }
    assert set(payload["topology"]) == {"sources", "nodes", "outputs"}
    stored = json.loads(payload_store.retrieve(emitted_args["payload_payload_id"]).decode("utf-8"))
    assert stored == payload


def test_summarize_confirm_wiring_exact_body_returns_summary() -> None:
    assert _summarize_guided_response(TurnType.CONFIRM_WIRING, _valid_confirm_body()) == "Confirmed wiring"


@pytest.mark.parametrize(
    "body",
    [
        {
            "chosen": ["confirm"],
            "edited_values": {},
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        },
        {
            "chosen": ["accept"],
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        },
    ],
)
def test_summarize_confirm_wiring_malformed_body_returns_none(body: TurnResponse) -> None:
    assert _summarize_guided_response(TurnType.CONFIRM_WIRING, body) is None


@pytest.mark.asyncio
async def test_chain_accept_returns_confirm_wiring_turn() -> None:
    state, step3_session, catalog, payload_store = _step3_ready_session()

    _state2, guided2, next_turn, recorder = await _dispatch(
        state,
        step3_session,
        catalog,
        payload_store=payload_store,
        current_step=GuidedStep.STEP_3_TRANSFORMS,
        current_turn_type=TurnType.PROPOSE_CHAIN,
        turn_response={
            "chosen": ["accept"],
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        },
    )

    assert guided2.step is GuidedStep.STEP_4_WIRE
    assert guided2.terminal is None
    assert next_turn is not None
    assert next_turn["type"] == TurnType.CONFIRM_WIRING.value
    advanced = _audit_args(recorder, "guided_step_advanced")
    assert advanced[-1]["prev_step"] == GuidedStep.STEP_3_TRANSFORMS.value
    assert advanced[-1]["next_step"] == GuidedStep.STEP_4_WIRE.value
    assert advanced[-1]["reason"] == "user_advanced"
    emitted = _audit_args(recorder, "guided_turn_emitted")
    assert emitted[-1]["turn_type"] == TurnType.CONFIRM_WIRING.value
    _assert_final_wire_payload_stored(next_turn=next_turn, emitted_args=emitted[-1], payload_store=payload_store)


@pytest.mark.asyncio
async def test_confirm_wiring_stamps_completed_terminal() -> None:
    state, wire_session, catalog, payload_store = _wire_ready_session()

    _state2, guided2, next_turn, _recorder = await _dispatch(
        state,
        wire_session,
        catalog,
        payload_store=payload_store,
        current_step=GuidedStep.STEP_4_WIRE,
        current_turn_type=TurnType.CONFIRM_WIRING,
        turn_response=_valid_confirm_body(),
    )

    assert next_turn is None
    assert guided2.terminal is not None
    assert guided2.terminal.kind is TerminalKind.COMPLETED


def test_confirm_wiring_invalid_pipeline_returns_failed_tool_result_without_terminal() -> None:
    state, wire_session, _catalog, _payload_store = _wire_ready_session(valid=False)

    result = handle_step_4_wire_confirm(state=state, session=wire_session)

    assert result.tool_result.success is False
    assert result.session.terminal is None
    assert result.session.step is GuidedStep.STEP_4_WIRE


@pytest.mark.asyncio
async def test_confirm_wiring_invalid_pipeline_reemits_wire_turn_without_terminal() -> None:
    state, wire_session, catalog, payload_store = _wire_ready_session(valid=False)

    _state2, guided2, next_turn, recorder = await _dispatch(
        state,
        wire_session,
        catalog,
        payload_store=payload_store,
        current_step=GuidedStep.STEP_4_WIRE,
        current_turn_type=TurnType.CONFIRM_WIRING,
        turn_response=_valid_confirm_body(),
    )

    assert guided2.terminal is None
    assert next_turn is not None
    assert next_turn["type"] == TurnType.CONFIRM_WIRING.value
    assert _audit_args(recorder, "guided_step_advanced") == []
    confirm_records = [r for r in guided2.history if r.turn_type is TurnType.CONFIRM_WIRING]
    assert len(confirm_records) == 1
    emitted = _audit_args(recorder, "guided_turn_emitted")
    assert len(emitted) == 1
    _assert_final_wire_payload_stored(next_turn=next_turn, emitted_args=emitted[0], payload_store=payload_store)


@pytest.mark.asyncio
async def test_confirm_wiring_invalid_pipeline_rebuild_path_is_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    from elspeth.web.composer.guided.emitters import rebuild_wire_turn_after_reconciliation as real_rebuild

    state, wire_session, catalog, payload_store = _wire_ready_session(valid=False)
    seen_versions: list[int] = []

    def recording_rebuild(state: CompositionState, *, resurface: Any) -> tuple[Any, bool]:
        seen_versions.append(state.version)
        return real_rebuild(state, resurface=resurface)

    monkeypatch.setattr(guided_route_helpers, "rebuild_wire_turn_after_reconciliation", recording_rebuild)

    _state2, _guided2, next_turn, _recorder = await _dispatch(
        state,
        wire_session,
        catalog,
        payload_store=payload_store,
        current_step=GuidedStep.STEP_4_WIRE,
        current_turn_type=TurnType.CONFIRM_WIRING,
        turn_response=_valid_confirm_body(),
    )

    assert seen_versions == [state.version]
    assert next_turn is not None
    assert next_turn["type"] == TurnType.CONFIRM_WIRING.value


@pytest.mark.parametrize(
    "body",
    [
        {
            "chosen": None,
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        },
        {
            "chosen": ["accept"],
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        },
        {
            "chosen": ["confirm"],
            "edited_values": {},
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        },
        # NOTE (P5.6): the former `custom_inputs=[]` case was REMOVED. Under the
        # profile-gated wire terminal, custom_inputs is arbitrary Tier-3 text that
        # the narrowed response guard no longer rejects and that can never
        # acknowledge the unavailable-escape (the escape gate requires
        # custom_inputs is None). See test_wire_stage_signoff_gate.py
        # ::test_custom_inputs_never_acknowledge_unavailable_escape.
    ],
)
@pytest.mark.asyncio
async def test_confirm_wiring_rejects_malformed_response_body(body: TurnResponse) -> None:
    state, wire_session, catalog, payload_store = _wire_ready_session()

    with pytest.raises(HTTPException) as exc:
        await _dispatch(
            state,
            wire_session,
            catalog,
            payload_store=payload_store,
            current_step=GuidedStep.STEP_4_WIRE,
            current_turn_type=TurnType.CONFIRM_WIRING,
            turn_response=body,
        )

    assert exc.value.status_code == 400
