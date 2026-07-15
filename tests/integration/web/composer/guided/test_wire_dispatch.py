"""Direct dispatcher coverage for the STEP_4_WIRE skeleton route path."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from elspeth.web.catalog.policy_view import PolicyCatalogView
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
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.sessions.routes._helpers import _dispatch_guided_respond, _summarize_guided_response
from elspeth.web.sessions.routes.composer.guided import _build_get_guided_turn, _guided_persisted_validity
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
                "options": {"schema": {"mode": "observed", "guaranteed_fields": ["price"]}},
                "rationale": "identity transform",
            },
        ),
        why="source rows already match the sink schema",
    )


def _trained_catalog() -> PolicyCatalogView:
    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    return PolicyCatalogView.for_trained_operator(catalog, snapshot)


def _step3_ready_session() -> tuple[CompositionState, GuidedSession, Any, MockPayloadStore]:
    state = _empty_state()
    session = GuidedSession.initial(profile=TUTORIAL_PROFILE)
    catalog = _trained_catalog()

    step_1 = handle_step_1_source(
        state=state,
        session=session,
        catalog=catalog,
        plugin_snapshot=catalog.snapshot,
        resolved=SourceResolved(
            plugin="csv",
            options={"path": "x.csv", "schema": {"mode": "observed", "guaranteed_fields": ["price"]}},
            observed_columns=("price",),
            sample_rows=({"price": "1.99"},),
        ),
    )
    step_2 = handle_step_2_sink(
        state=step_1.state,
        session=step_1.session,
        catalog=catalog,
        plugin_snapshot=catalog.snapshot,
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
            _trained_catalog(),
            MockPayloadStore(),
        )

    state, step3_session, catalog, payload_store = _step3_ready_session()
    result = handle_step_3_chain_accept(
        state=state,
        session=step3_session,
        proposal=step3_session.step_3_proposal,
        catalog=catalog,
        plugin_snapshot=catalog.snapshot,
    )
    assert result.tool_result.success is True
    assert result.session.step is GuidedStep.STEP_4_WIRE
    assert result.session.terminal is None
    return result.state, result.session, catalog, payload_store


def _profiled_wire_ready_session() -> tuple[CompositionState, GuidedSession, PolicyCatalogView, Any, MockPayloadStore]:
    """Build the live wire-stage shape: public alias persisted, binding private."""
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
    from elspeth.web.config import WebSettings
    from elspeth.web.plugin_policy.availability import build_plugin_snapshot
    from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
    from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, RuntimeWebPluginConfig

    settings = WebSettings(
        composer_max_composition_turns=4,
        composer_max_discovery_turns=4,
        composer_timeout_seconds=60,
        composer_rate_limit_per_minute=20,
        shareable_link_signing_key=b"0123456789abcdef0123456789abcdef",
        llm_profiles={
            "tutorial": {
                "provider": "bedrock",
                "model": "bedrock/zai.glm-5",
                "region_name": "ap-northeast-1",
            }
        },
        tutorial_llm_profile="tutorial",
    )
    runtime = RuntimeWebPluginConfig.from_settings(settings)
    policy = compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)
    profiles = OperatorProfileRegistry(policy=policy, settings=runtime)
    full_catalog = create_catalog_service()

    class _NoSecrets:
        def has_server_ref(self, name: str) -> bool:
            return False

        def has_user_ref(self, principal: str, name: str) -> bool:
            return False

    snapshot = build_plugin_snapshot(
        policy=policy,
        catalog=full_catalog,
        profiles=profiles,
        principal_scope="local:tutorial-user",
        secret_inventory=_NoSecrets(),
        generation_key=b"wire-profile-lowering-test-key",
    )
    catalog = PolicyCatalogView(full_catalog, snapshot, profiles)
    session = GuidedSession.initial(profile=TUTORIAL_PROFILE)
    step_1 = handle_step_1_source(
        state=_empty_state(),
        session=session,
        catalog=catalog,
        plugin_snapshot=snapshot,
        resolved=SourceResolved(
            plugin="csv",
            options={"path": "x.csv", "schema": {"mode": "observed", "guaranteed_fields": ["text"]}},
            observed_columns=("text",),
            sample_rows=({"text": "hello"},),
        ),
    )
    step_2 = handle_step_2_sink(
        state=step_1.state,
        session=step_1.session,
        catalog=catalog,
        plugin_snapshot=snapshot,
        resolved=SinkResolved(
            outputs=(
                SinkOutputResolved(
                    plugin="json",
                    options={"path": "out.jsonl", "schema": {"mode": "observed"}},
                    required_fields=("summary",),
                    schema_mode="observed",
                ),
            ),
        ),
    )
    proposal = ChainProposal(
        steps=(
            {
                "plugin": "llm",
                "options": {
                    "prompt_template": "Summarise {{ row.text }}",
                    "response_field": "summary",
                    "required_input_fields": ["text"],
                    "schema": {
                        "mode": "observed",
                        "required_fields": ["text"],
                        "guaranteed_fields": ["text", "summary"],
                    },
                },
                "rationale": "summarise each row",
            },
        ),
        why="summarise the tutorial input",
    )
    result = handle_step_3_chain_accept(
        state=step_2.state,
        session=step_2.session,
        proposal=proposal,
        catalog=catalog,
        plugin_snapshot=snapshot,
    )
    assert result.tool_result.success is True, result.tool_result.validation.errors
    assert result.state.nodes[0].options["profile"] == "tutorial"
    assert "provider" not in result.state.nodes[0].options
    assert "model" not in result.state.nodes[0].options
    return result.state, result.session, catalog, profiles, MockPayloadStore()


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
    profile_registry: Any | None = None,
) -> tuple[CompositionState, GuidedSession, Any | None, BufferingRecorder]:
    recorder = recorder or BufferingRecorder()
    state2, guided2, next_turn = await _dispatch_guided_respond(
        state=state,
        guided=guided,
        current_step=current_step,
        current_turn_type=current_turn_type,
        turn_response=turn_response,
        catalog=catalog,
        plugin_snapshot=catalog.snapshot,
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
        profile_registry=profile_registry,
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


@pytest.mark.asyncio
async def test_confirm_wiring_lowers_operator_profile_only_for_validation() -> None:
    """Wire confirmation validates the executable binding, not its public alias."""
    state, wire_session, catalog, profiles, payload_store = _profiled_wire_ready_session()
    authored_validation = state.validate()
    assert authored_validation.is_valid is False
    assert any("Missing fields: [summary]" in error.message for error in authored_validation.errors)
    persisted_is_valid, persisted_errors = _guided_persisted_validity(state, catalog=catalog)
    assert persisted_is_valid is True
    assert persisted_errors is None
    wire_turn = _build_get_guided_turn(state, wire_session, catalog=catalog)
    assert wire_turn is not None
    assert wire_turn["payload"]["edge_contracts"]
    assert all(contract["satisfied"] for contract in wire_turn["payload"]["edge_contracts"])

    _state2, guided2, next_turn, _recorder = await _dispatch(
        state,
        wire_session,
        catalog,
        payload_store=payload_store,
        current_step=GuidedStep.STEP_4_WIRE,
        current_turn_type=TurnType.CONFIRM_WIRING,
        turn_response=_valid_confirm_body(),
        profile_registry=profiles,
    )

    assert next_turn is None
    assert guided2.terminal is not None
    assert guided2.terminal.kind is TerminalKind.COMPLETED
    assert state.nodes[0].options["profile"] == "tutorial"
    assert "provider" not in state.nodes[0].options
    assert "model" not in state.nodes[0].options
    assert "profile: tutorial" in guided2.terminal.pipeline_yaml
    assert "zai.glm-5" not in guided2.terminal.pipeline_yaml


def test_confirm_wiring_invalid_pipeline_returns_failed_tool_result_without_terminal() -> None:
    state, wire_session, _catalog, _payload_store = _wire_ready_session(valid=False)

    result = handle_step_4_wire_confirm(state=state, session=wire_session)

    assert result.tool_result.success is False
    assert result.session.terminal is None
    assert result.session.step is GuidedStep.STEP_4_WIRE


@pytest.mark.asyncio
async def test_confirm_wiring_invalid_pipeline_raises_structured_rejection() -> None:
    """A confirm against an invalid pipeline is an ERROR, not a silent success.

    Pre-fix, this path 200'd, re-emitted the wire turn, and minted a new
    composition-state version per click (elspeth-3b35abf148 variant 3). The
    dispatch now raises the structured rejection; the route maps it to HTTP 409
    without persisting a version.
    """
    from elspeth.web.composer.guided.errors import WireConfirmRejectedError

    state, wire_session, catalog, payload_store = _wire_ready_session(valid=False)
    recorder = BufferingRecorder()

    with pytest.raises(WireConfirmRejectedError) as excinfo:
        await _dispatch(
            state,
            wire_session,
            catalog,
            payload_store=payload_store,
            current_step=GuidedStep.STEP_4_WIRE,
            current_turn_type=TurnType.CONFIRM_WIRING,
            turn_response=_valid_confirm_body(),
            recorder=recorder,
        )

    # Structured rejection names the step and carries the validation issues.
    assert excinfo.value.step == GuidedStep.STEP_4_WIRE.value
    assert len(excinfo.value.issues) > 0
    for issue in excinfo.value.issues:
        assert set(issue) == {"component", "message", "severity"}
    # The rejected confirm never advances the wizard and never re-emits a turn.
    assert _audit_args(recorder, "guided_step_advanced") == []
    assert _audit_args(recorder, "guided_turn_emitted") == []


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


# ---------------------------------------------------------------------------
# Route-level regression: a failed confirm mints NO new composition version and
# returns a structured 409 (elspeth-3b35abf148 variant 3 — the silent no-op
# confirm that minted v11→v19 across 15 clicks).
# ---------------------------------------------------------------------------


def _seed_invalid_wire_session(client: Any, session_id: str) -> int:
    """Persist an INVALID composition parked at STEP_4_WIRE; return its version."""
    import asyncio as _asyncio

    from elspeth.web.composer.guided.state_machine import TurnRecord
    from elspeth.web.sessions.protocol import CompositionStateData

    service = client.app.state.session_service

    state = _empty_state()
    assert state.validate().is_valid is False, "fixture must be invalid to exercise the gate"

    wire_record = TurnRecord(
        step=GuidedStep.STEP_4_WIRE,
        turn_type=TurnType.CONFIRM_WIRING,
        payload_hash="wire-payload-hash",
        response_hash=None,
        emitter="server",
    )
    guided = replace(
        GuidedSession.initial(profile=TUTORIAL_PROFILE),
        step=GuidedStep.STEP_4_WIRE,
        history=(wire_record,),
    )
    state_d = state.to_dict()
    state_data = CompositionStateData(
        sources=state_d["sources"],
        nodes=state_d["nodes"],
        edges=state_d["edges"],
        outputs=state_d["outputs"],
        metadata_=state_d["metadata"],
        is_valid=False,
        validation_errors=None,
        composer_meta={"guided_session": guided.to_dict()},
    )
    record = _asyncio.run(service.save_composition_state(UUID(session_id), state_data, provenance="session_seed"))
    return record.version


def _current_version(client: Any, session_id: str) -> int:
    import asyncio as _asyncio

    record = _asyncio.run(client.app.state.session_service.get_current_state(UUID(session_id)))
    assert record is not None
    return record.version


def test_route_failed_confirm_returns_409_and_mints_no_version(composer_test_client: Any) -> None:
    client = composer_test_client
    resp = client.post("/api/sessions", json={"title": "wire-reject"})
    assert resp.status_code == 201, resp.json()
    session_id = resp.json()["id"]

    seeded_version = _seed_invalid_wire_session(client, session_id)

    confirm_body = {
        "chosen": ["confirm"],
        "edited_values": None,
        "custom_inputs": None,
        "accepted_step_index": None,
        "edit_step_index": None,
        "control_signal": None,
    }

    # Click confirm three times — every attempt is a structured 409, and the
    # composition-version count never moves (pre-fix: +1 per click).
    for _click in range(3):
        resp = client.post(f"/api/sessions/{session_id}/guided/respond", json=confirm_body)
        assert resp.status_code == 409, resp.json()
        detail = resp.json()["detail"]
        assert detail["code"] == "wire_confirm_rejected"
        assert detail["step"] == GuidedStep.STEP_4_WIRE.value
        assert "confirmed yet" in detail["detail"]
        assert isinstance(detail["validation_errors"], list)
        assert len(detail["validation_errors"]) > 0
        for issue in detail["validation_errors"]:
            assert set(issue) == {"component", "message", "severity"}

    assert _current_version(client, session_id) == seeded_version
