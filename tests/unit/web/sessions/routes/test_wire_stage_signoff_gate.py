"""Phase P5.6 — STEP_4_WIRE terminal is gated on profile.advisor_checkpoints + verdict."""

from __future__ import annotations

import dataclasses

import pytest

from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE
from elspeth.web.composer.guided.protocol import ControlSignal, GuidedStep, TurnType
from elspeth.web.composer.guided.state_machine import GuidedSession, TerminalKind
from elspeth.web.composer.service import AdvisorCheckpointVerdict
from elspeth.web.sessions.routes._helpers import _dispatch_guided_respond
from tests.unit.web.sessions.routes._wire_fixtures import (  # P3 helper; see note
    make_wire_ready_session_and_state,
)

# TUTORIAL_PROFILE carries advisor_checkpoints=False as the explicit demo
# bypass. The gate-behavior cases below use a synthetic clone so they remain
# independent of the live profile constant.
ADVISOR_ON_PROFILE = dataclasses.replace(TUTORIAL_PROFILE, advisor_checkpoints=True)


class _AdvisorCheckpointFake:
    def __init__(self, verdict: AdvisorCheckpointVerdict | None) -> None:
        self._verdict = verdict
        self.await_count = 0

    async def __call__(self, *args, **kwargs) -> AdvisorCheckpointVerdict:
        self.await_count += 1
        if self._verdict is None:
            raise AssertionError("advisor must NOT be called")
        return self._verdict

    def assert_awaited_once(self) -> None:
        assert self.await_count == 1

    def assert_not_awaited(self) -> None:
        assert self.await_count == 0


class _AdvisorServiceFake:
    def __init__(self, verdict: AdvisorCheckpointVerdict | None) -> None:
        self.run_signoff_checkpoint = _AdvisorCheckpointFake(verdict)


class _CatalogPlaceholder:
    pass


class _BlobServicePlaceholder:
    pass


class _PayloadStoreFake:
    def store(self, _payload: bytes) -> str:
        return "payload-id"


def _service(verdict: AdvisorCheckpointVerdict | None) -> _AdvisorServiceFake:
    return _AdvisorServiceFake(verdict)


async def _dispatch(
    session: GuidedSession,
    state,
    svc,
    *,
    max_passes: int | None = 3,
    control=ControlSignal.EXIT_TO_FREEFORM,
    turn_response_override=None,
):
    # CONFIRM_WIRING confirm response: no control signal (a plain confirm).
    turn_response = turn_response_override or {
        "chosen": ["confirm"],
        "edited_values": None,
        "custom_inputs": None,
        "accepted_step_index": None,
        "edit_step_index": None,
        "control_signal": None,
    }
    return await _dispatch_guided_respond(
        state=state,
        guided=session,
        current_step=GuidedStep.STEP_4_WIRE,
        current_turn_type=TurnType.CONFIRM_WIRING,
        turn_response=turn_response,
        catalog=_CatalogPlaceholder(),
        recorder=BufferingRecorder(),
        user_id="u1",
        data_dir=None,
        session_engine=None,
        session_id="s1",
        blob_service=_BlobServicePlaceholder(),
        payload_store=_PayloadStoreFake(),
        model="m",
        temperature=None,
        seed=None,
        composer_service=svc,
        advisor_checkpoint_max_passes=max_passes,
    )


@pytest.mark.asyncio
async def test_live_profile_clean_verdict_calls_advisor_and_completes() -> None:
    session, state = make_wire_ready_session_and_state(profile=EMPTY_PROFILE)
    svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
    _state, guided, _turn = await _dispatch(session, state, svc)
    assert guided.terminal is not None
    assert guided.terminal.kind is TerminalKind.COMPLETED
    assert guided.advisor_checkpoint_passes_used == 1
    svc.run_signoff_checkpoint.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_inputs_never_acknowledge_unavailable_escape() -> None:
    session, state = make_wire_ready_session_and_state(profile=ADVISOR_ON_PROFILE)
    session = dataclasses.replace(session, advisor_checkpoint_passes_used=2)
    svc = _service(
        AdvisorCheckpointVerdict(
            ok=False,
            blocking=False,
            failure_class="unavailable",
            findings_text="advisor unavailable",
        )
    )
    raw_custom_ack = {
        "chosen": ["confirm"],
        "edited_values": None,
        "custom_inputs": ["complete_without_signoff"],
        "accepted_step_index": None,
        "edit_step_index": None,
        "control_signal": None,
    }
    _state, guided, next_turn = await _dispatch(
        session,
        state,
        svc,
        turn_response_override=raw_custom_ack,
    )
    assert guided.terminal is None
    assert next_turn is not None
    assert next_turn["payload"]["signoff_outcome"] == "escape_unavailable"


@pytest.mark.asyncio
async def test_tutorial_profile_clean_completes() -> None:
    # TUTORIAL_PROFILE is the explicit demo bypass, so the tutorial wire-confirm
    # completes clean on a valid pipeline WITHOUT the terminal advisor sign-off.
    session, state = make_wire_ready_session_and_state(profile=TUTORIAL_PROFILE)
    svc = _service(None)  # advisor must NOT be called for the advisor-off profile
    _state, guided, _turn = await _dispatch(session, state, svc)
    assert guided.terminal is not None
    assert guided.terminal.kind is TerminalKind.COMPLETED
    assert guided.advisor_checkpoint_passes_used == 0
    svc.run_signoff_checkpoint.assert_not_awaited()


@pytest.mark.asyncio
async def test_advisor_on_clean_verdict_completes() -> None:
    # Positive gate path: an advisor-ON profile + a clean (non-blocking) verdict
    # COMPLETES, having awaited the provider exactly once and incremented the
    # pass counter. Distinct from the escape-completion path, which completes
    # WITHOUT calling the provider.
    session, state = make_wire_ready_session_and_state(profile=ADVISOR_ON_PROFILE)
    svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
    _state, guided, _turn = await _dispatch(session, state, svc)
    assert guided.terminal is not None
    assert guided.terminal.kind is TerminalKind.COMPLETED
    assert guided.advisor_checkpoint_passes_used == 1
    svc.run_signoff_checkpoint.assert_awaited_once()


@pytest.mark.asyncio
async def test_tutorial_profile_flagged_does_not_complete() -> None:
    session, state = make_wire_ready_session_and_state(profile=ADVISOR_ON_PROFILE)
    svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: x"))
    _state, guided, next_turn = await _dispatch(session, state, svc)
    assert guided.terminal is None  # re-emit a revise turn, never COMPLETED
    assert next_turn is not None
    assert next_turn["type"] == TurnType.CONFIRM_WIRING.value


@pytest.mark.asyncio
async def test_tutorial_profile_missing_service_fails_closed_invariant() -> None:
    session, state = make_wire_ready_session_and_state(profile=ADVISOR_ON_PROFILE)
    _state, guided, next_turn = await _dispatch(session, state, None)
    assert guided.terminal is None
    assert next_turn is not None
    assert "Advisor sign-off service or pass budget is not configured" in str(next_turn["payload"])


@pytest.mark.asyncio
async def test_tutorial_profile_missing_budget_fails_closed_invariant() -> None:
    session, state = make_wire_ready_session_and_state(profile=ADVISOR_ON_PROFILE)
    svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
    _state, guided, next_turn = await _dispatch(session, state, svc, max_passes=None)
    assert guided.terminal is None
    assert next_turn is not None
    assert "Advisor sign-off service or pass budget is not configured" in str(next_turn["payload"])
    svc.run_signoff_checkpoint.assert_not_awaited()


@pytest.mark.asyncio
async def test_acknowledged_unavailable_escape_completes_cross_request() -> None:
    # Positive cross-request escape path THROUGH the dispatcher: the narrowed
    # guard must admit chosen=["complete_without_signoff"], acknowledged_unavailable
    # must compute True (escape_offered + exhausted counter + closed choice +
    # custom_inputs is None), and run_wire_signoff must COMPLETE without re-calling
    # the provider. Guards against a future guard-tightening silently breaking the
    # escape path with no failing test.
    session, state = make_wire_ready_session_and_state(profile=ADVISOR_ON_PROFILE)
    session = dataclasses.replace(
        session,
        advisor_checkpoint_passes_used=3,
        advisor_signoff_escape_offered=True,
    )
    svc = _service(None)  # exhausted + acknowledged => provider must NOT be called
    ack_body = {
        "chosen": ["complete_without_signoff"],
        "edited_values": None,
        "custom_inputs": None,
        "accepted_step_index": None,
        "edit_step_index": None,
        "control_signal": None,
    }
    _state, guided, next_turn = await _dispatch(
        session,
        state,
        svc,
        turn_response_override=ack_body,
    )
    assert guided.terminal is not None
    assert guided.terminal.kind is TerminalKind.COMPLETED
    assert next_turn is None
    svc.run_signoff_checkpoint.assert_not_awaited()
