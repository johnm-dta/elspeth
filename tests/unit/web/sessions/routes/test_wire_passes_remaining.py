"""Slice B0 — ``passes_remaining`` rides through on RE-EMITTED wire turns only.

The wire-stage advisor sign-off discloses the remaining pass budget on the
explicit re-emit (so the frontend can render "spends 1 of N"). It is folded into
the payload ONLY at the two re-emit sites where ``max_passes`` is in scope; the
INITIAL wire turn carries no cost copy (and so the advisor-off tutorial, which
never re-emits a budgeted pass, shows none). These tests assert delivery through
``_dispatch_guided_respond`` AND the route serializer ``_turn_payload_response``
(an emitter-only unit test cannot prove end-to-end pass-through).
"""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.emitters import build_step_4_wire_turn
from elspeth.web.composer.guided.profile import TUTORIAL_PROFILE
from elspeth.web.composer.guided.protocol import ControlSignal, GuidedStep, TurnType
from elspeth.web.composer.service import AdvisorCheckpointVerdict
from elspeth.web.sessions.routes._helpers import _dispatch_guided_respond
from elspeth.web.sessions.routes.composer.guided import _turn_payload_response
from tests.unit.web.sessions.routes._wire_fixtures import (
    make_trained_plugin_policy_context,
    make_wire_ready_session_and_state,
)

_POLICY_CATALOG, _PLUGIN_SNAPSHOT = make_trained_plugin_policy_context()

# TUTORIAL_PROFILE carries advisor_checkpoints=False (the explicit demo bypass);
# clone it advisor-ON so the END sign-off runs and re-emits a budgeted REVISE.
ADVISOR_ON_PROFILE = dataclasses.replace(TUTORIAL_PROFILE, advisor_checkpoints=True)

_CONFIRM_BODY: dict[str, Any] = {
    "chosen": ["confirm"],
    "edited_values": None,
    "custom_inputs": None,
    "accepted_step_index": None,
    "edit_step_index": None,
    "control_signal": None,
}
_ASK_ADVISOR_BODY: dict[str, Any] = {
    "chosen": None,
    "edited_values": None,
    "custom_inputs": None,
    "accepted_step_index": None,
    "edit_step_index": None,
    "control_signal": ControlSignal.REQUEST_ADVISOR,
}


class _FlaggingServiceFake:
    async def run_signoff_checkpoint(self, *_args: object, **_kwargs: object) -> AdvisorCheckpointVerdict:
        return AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: review")


def _flagging_service() -> _FlaggingServiceFake:
    return _FlaggingServiceFake()


async def _dispatch(session, state, svc, *, turn_response, max_passes=3):
    payload_store = MagicMock(spec_set=["store"])
    payload_store.store.return_value = "payload-id"
    return await _dispatch_guided_respond(
        state=state,
        guided=session,
        current_step=GuidedStep.STEP_4_WIRE,
        current_turn_type=TurnType.CONFIRM_WIRING,
        turn_response=turn_response,
        catalog=_POLICY_CATALOG,
        plugin_snapshot=_PLUGIN_SNAPSHOT,
        recorder=BufferingRecorder(),
        user_id="u1",
        data_dir=None,
        session_engine=None,
        session_id="s1",
        blob_service=MagicMock(spec_set=[]),
        payload_store=payload_store,
        model="m",
        temperature=None,
        seed=None,
        composer_service=svc,
        advisor_checkpoint_max_passes=max_passes,
    )


@pytest.mark.asyncio
async def test_auto_path_flagged_reemit_response_carries_passes_remaining() -> None:
    # A plain confirm on an advisor-ON profile runs the END sign-off; a FLAGGED
    # (blocking) verdict with budget remaining re-emits a REVISE turn. With
    # max_passes=3 and a fresh session (passes_used 0 -> 1), passes_remaining == 2.
    session, state = make_wire_ready_session_and_state(profile=ADVISOR_ON_PROFILE)
    _state, guided, next_turn = await _dispatch(session, state, _flagging_service(), turn_response=_CONFIRM_BODY)

    assert guided.terminal is None
    assert next_turn is not None
    assert next_turn["payload"]["signoff_outcome"] == "revise"
    # Pin the integer (NOT the formula): catches a pre-/post-increment off-by-one.
    assert next_turn["payload"]["passes_remaining"] == 2

    # End-to-end: the route serializer copies every payload key, so the field
    # rides through post_guided_respond's _turn_payload_response unchanged.
    response = _turn_payload_response(next_turn, shield_available=True)
    assert response is not None
    assert response.payload["passes_remaining"] == 2


@pytest.mark.asyncio
async def test_request_advisor_reemit_response_carries_passes_remaining() -> None:
    # The explicit "Ask advisor" re-emit (control_signal=request_advisor) is the
    # other site where max_passes is in scope; it too discloses the budget.
    session, state = make_wire_ready_session_and_state(profile=ADVISOR_ON_PROFILE)
    _state, _guided, next_turn = await _dispatch(session, state, _flagging_service(), turn_response=_ASK_ADVISOR_BODY)

    assert next_turn is not None
    assert next_turn["payload"]["signoff_outcome"] == "revise"
    assert next_turn["payload"]["passes_remaining"] == 2
    response = _turn_payload_response(next_turn, shield_available=True)
    assert response is not None
    assert response.payload["passes_remaining"] == 2


@pytest.mark.asyncio
async def test_initial_wire_response_omits_passes_remaining() -> None:
    # The INITIAL wire turn (built without a sign-off pass) carries no cost copy;
    # the field must be ABSENT so the advisor-off tutorial shows no "uses 1 of N".
    _session, state = make_wire_ready_session_and_state(profile=ADVISOR_ON_PROFILE)
    initial_turn = build_step_4_wire_turn(state)
    assert "passes_remaining" not in initial_turn["payload"]

    response = _turn_payload_response(initial_turn, shield_available=True)
    assert response is not None
    assert "passes_remaining" not in response.payload
