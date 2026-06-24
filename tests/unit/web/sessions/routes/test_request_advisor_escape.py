"""Phase P5.8 — REQUEST_ADVISOR whole-pipeline escape at the wire stage;
the existing step-3 chain re-solve is preserved."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.profile import TUTORIAL_PROFILE
from elspeth.web.composer.guided.protocol import ControlSignal, GuidedStep, TurnType
from elspeth.web.composer.service import AdvisorCheckpointVerdict
from elspeth.web.sessions.routes._helpers import _dispatch_guided_respond
from tests.unit.web.sessions.routes._wire_fixtures import make_wire_ready_session_and_state


@pytest.mark.asyncio
async def test_request_advisor_at_wire_runs_whole_pipeline_signoff() -> None:
    session, state = make_wire_ready_session_and_state(profile=TUTORIAL_PROFILE)
    svc = MagicMock()
    svc.run_signoff_checkpoint = AsyncMock(
        return_value=AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: review this")
    )
    svc._validate_advisor_arguments = MagicMock(side_effect=AssertionError("wire-stage REQUEST_ADVISOR must use run_signoff_checkpoint"))
    payload_store = MagicMock()
    payload_store.store.return_value = "payload-id"
    turn_response = {
        "chosen": None,
        "edited_values": None,
        "custom_inputs": None,
        "accepted_step_index": None,
        "edit_step_index": None,
        "control_signal": ControlSignal.REQUEST_ADVISOR,
    }
    recorder = BufferingRecorder()
    _s, guided, next_turn = await _dispatch_guided_respond(
        state=state,
        guided=session,
        current_step=GuidedStep.STEP_4_WIRE,
        current_turn_type=TurnType.CONFIRM_WIRING,
        turn_response=turn_response,
        catalog=MagicMock(),
        recorder=recorder,
        user_id="u1",
        data_dir=None,
        session_engine=None,
        session_id="s1",
        blob_service=MagicMock(),
        payload_store=payload_store,
        model="m",
        temperature=None,
        seed=None,
        composer_service=svc,
        advisor_checkpoint_max_passes=3,
    )
    svc.run_signoff_checkpoint.assert_awaited_once()
    svc._validate_advisor_arguments.assert_not_called()
    assert guided.terminal is None  # on-demand review never auto-completes on a FLAG
    assert next_turn is not None
    assert "review this" in next_turn["payload"]["advisor_findings"]
    assert "composer.signoff.revise" in [inv.tool_name for inv in recorder.invocations]


@pytest.mark.asyncio
async def test_request_advisor_at_wire_clean_re_emits_never_completes() -> None:
    # REQUEST_ADVISOR is advisory, NOT the completion gesture: even a CLEAN
    # verdict RE-EMITS the wire turn (terminal stays None). Only the
    # CONFIRM_WIRING confirm path (P5.6) stamps COMPLETED.
    session, state = make_wire_ready_session_and_state(profile=TUTORIAL_PROFILE)
    svc = MagicMock()
    svc.run_signoff_checkpoint = AsyncMock(return_value=AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="clean"))
    svc._validate_advisor_arguments = MagicMock(side_effect=AssertionError("wire-stage REQUEST_ADVISOR must use run_signoff_checkpoint"))
    payload_store = MagicMock()
    payload_store.store.return_value = "payload-id"
    turn_response = {
        "chosen": None,
        "edited_values": None,
        "custom_inputs": None,
        "accepted_step_index": None,
        "edit_step_index": None,
        "control_signal": ControlSignal.REQUEST_ADVISOR,
    }
    recorder = BufferingRecorder()
    _s, guided, next_turn = await _dispatch_guided_respond(
        state=state,
        guided=session,
        current_step=GuidedStep.STEP_4_WIRE,
        current_turn_type=TurnType.CONFIRM_WIRING,
        turn_response=turn_response,
        catalog=MagicMock(),
        recorder=recorder,
        user_id="u1",
        data_dir=None,
        session_engine=None,
        session_id="s1",
        blob_service=MagicMock(),
        payload_store=payload_store,
        model="m",
        temperature=None,
        seed=None,
        composer_service=svc,
        advisor_checkpoint_max_passes=3,
    )
    svc.run_signoff_checkpoint.assert_awaited_once()
    svc._validate_advisor_arguments.assert_not_called()
    assert guided.terminal is None  # advisory gesture never auto-completes
    assert next_turn is not None
    # The COMPLETE outcome value is carried, but the turn is re-emitted (not terminal).
    assert next_turn["payload"]["signoff_outcome"] == "complete"
    assert "composer.signoff.clean" in [inv.tool_name for inv in recorder.invocations]


@pytest.mark.asyncio
async def test_request_advisor_at_wire_missing_service_or_budget_fails_closed() -> None:
    session, state = make_wire_ready_session_and_state(profile=TUTORIAL_PROFILE)
    payload_store = MagicMock()
    payload_store.store.return_value = "payload-id"
    recorder = BufferingRecorder()
    turn_response = {
        "chosen": None,
        "edited_values": None,
        "custom_inputs": None,
        "accepted_step_index": None,
        "edit_step_index": None,
        "control_signal": ControlSignal.REQUEST_ADVISOR,
    }
    _s, guided, next_turn = await _dispatch_guided_respond(
        state=state,
        guided=session,
        current_step=GuidedStep.STEP_4_WIRE,
        current_turn_type=TurnType.CONFIRM_WIRING,
        turn_response=turn_response,
        catalog=MagicMock(),
        recorder=recorder,
        user_id="u1",
        data_dir=None,
        session_engine=None,
        session_id="s1",
        blob_service=MagicMock(),
        payload_store=payload_store,
        model="m",
        temperature=None,
        seed=None,
        composer_service=None,
        advisor_checkpoint_max_passes=None,
    )
    assert guided.terminal is None
    assert next_turn is not None
    assert "Advisor sign-off service or pass budget is not configured" in str(next_turn["payload"])
    assert next_turn["payload"]["signoff_outcome"] == "blocked_unavailable"
    payload_store.store.assert_called()


@pytest.mark.asyncio
async def test_request_advisor_at_step3_still_resolves_chain(monkeypatch) -> None:
    # Regression guard: the existing STEP_3 chain re-solve path must remain.
    import elspeth.web.sessions.routes._helpers as helpers

    called = {}

    async def fake_solve(**kwargs):
        called["site"] = kwargs.get("site")
        return None, kwargs["session"]

    monkeypatch.setattr(helpers, "solve_chain_with_auto_drop", fake_solve)
    session, state = make_wire_ready_session_and_state(profile=TUTORIAL_PROFILE, at_step3=True)
    svc = MagicMock()
    svc.run_signoff_checkpoint = AsyncMock(side_effect=AssertionError("wire signoff must not run at step3"))
    payload_store = MagicMock()
    payload_store.store.return_value = "payload-id"
    turn_response = {
        "chosen": None,
        "edited_values": None,
        "custom_inputs": None,
        "accepted_step_index": None,
        "edit_step_index": None,
        "control_signal": ControlSignal.REQUEST_ADVISOR,
    }
    await _dispatch_guided_respond(
        state=state,
        guided=session,
        current_step=GuidedStep.STEP_3_TRANSFORMS,
        current_turn_type=TurnType.PROPOSE_CHAIN,
        turn_response=turn_response,
        catalog=MagicMock(),
        recorder=BufferingRecorder(),
        user_id="u1",
        data_dir=None,
        session_engine=None,
        session_id="s1",
        blob_service=MagicMock(),
        payload_store=payload_store,
        model="m",
        temperature=None,
        seed=None,
        composer_service=svc,
        advisor_checkpoint_max_passes=3,
    )
    assert "step_3_request_advisor_solve" in (called.get("site") or "")
    svc.run_signoff_checkpoint.assert_not_awaited()
