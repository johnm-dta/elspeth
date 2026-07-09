"""Phase P5.7 — differentiated sign-off audit names + fail-closed findings."""

from __future__ import annotations

import dataclasses

import pytest

from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.profile import TUTORIAL_PROFILE
from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
from elspeth.web.composer.guided.signoff import (
    SignoffDecision,
    SignoffOutcome,
    signoff_audit_event_name,
)
from elspeth.web.composer.guided.state_machine import GuidedSession, TerminalKind
from elspeth.web.composer.service import AdvisorCheckpointVerdict, _advisor_signoff_blocked_validation
from elspeth.web.sessions.routes._helpers import _dispatch_guided_respond
from tests.unit.web.sessions.routes._wire_fixtures import make_wire_ready_session_and_state

# The live profile now uses terminal advisor sign-off; tutorial remains the
# explicit demo bypass. These sign-off-audit tests use a synthetic advisor-ON
# profile so coverage stays independent of either shipped profile constant.
ADVISOR_ON_PROFILE = dataclasses.replace(TUTORIAL_PROFILE, advisor_checkpoints=True)


def _d(outcome: SignoffOutcome, reason: str | None) -> SignoffDecision:
    return SignoffDecision(outcome=outcome, reason=reason, findings_text="f", passes_delta=1)


# --- Audit-name resolver: all six outcomes map to DISTINCT names (D13) --------


def test_clean_audit_name() -> None:
    assert signoff_audit_event_name(_d(SignoffOutcome.COMPLETE, None)) == "composer.signoff.clean"


def test_completed_without_signoff_has_distinct_audit_name() -> None:
    # The audited escape must be DISTINGUISHABLE from a CLEAN sign-off.
    name = signoff_audit_event_name(_d(SignoffOutcome.COMPLETE, "unavailable"))
    assert name == "composer.signoff.completed_without_signoff_advisor_unreachable"
    assert name != "composer.signoff.clean"


def test_revise_audit_name() -> None:
    assert signoff_audit_event_name(_d(SignoffOutcome.REVISE, None)) == "composer.signoff.revise"


def test_blocked_flagged_audit_name() -> None:
    assert signoff_audit_event_name(_d(SignoffOutcome.BLOCKED_FLAGGED, "exhausted")) == "composer.signoff.blocked_flagged"


def test_blocked_unavailable_audit_name() -> None:
    assert signoff_audit_event_name(_d(SignoffOutcome.BLOCKED_UNAVAILABLE, "unavailable")) == "composer.signoff.blocked_unavailable"


def test_escape_offered_audit_name() -> None:
    assert signoff_audit_event_name(_d(SignoffOutcome.ESCAPE_UNAVAILABLE, "unavailable")) == "composer.signoff.escape_offered"


def test_all_six_names_are_distinct() -> None:
    names = {
        signoff_audit_event_name(_d(SignoffOutcome.COMPLETE, None)),
        signoff_audit_event_name(_d(SignoffOutcome.COMPLETE, "unavailable")),
        signoff_audit_event_name(_d(SignoffOutcome.REVISE, None)),
        signoff_audit_event_name(_d(SignoffOutcome.BLOCKED_FLAGGED, "exhausted")),
        signoff_audit_event_name(_d(SignoffOutcome.BLOCKED_UNAVAILABLE, "unavailable")),
        signoff_audit_event_name(_d(SignoffOutcome.ESCAPE_UNAVAILABLE, "unavailable")),
    }
    assert len(names) == 6


# --- The blocked validation the BLOCKED_* revise turn carries is non-runnable -


def test_blocked_validation_is_non_runnable() -> None:
    result = _advisor_signoff_blocked_validation(reason="exhausted", findings="prompt sees no row field")
    assert result.is_valid is False
    assert result.readiness.authoring_valid is False
    assert result.readiness.execution_ready is False
    assert result.readiness.completion_ready is False


# --- End-to-end: the recorded invocation's tool_name equals the distinct name -


class _AdvisorServiceFake:
    def __init__(self, verdict: AdvisorCheckpointVerdict | None) -> None:
        self._verdict = verdict

    async def run_signoff_checkpoint(self, *args, **kwargs) -> AdvisorCheckpointVerdict:
        if self._verdict is None:
            raise AssertionError("advisor must NOT be called")
        return self._verdict


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
    recorder: BufferingRecorder,
    max_passes: int | None = 3,
    turn_response_override=None,
):
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
        recorder=recorder,
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
async def test_dispatch_records_distinct_completed_without_signoff_name() -> None:
    # Differentiated COMPLETE case: an acknowledged advisor-unreachable escape
    # COMPLETES (returns before any guided_turn_emitted), so the LAST recorded
    # invocation is the sign-off audit event — and it carries the DISTINCT
    # advisor-unreachable name, never composer.signoff.clean (the D13 assertion,
    # proven through the real dispatcher).
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
    recorder = BufferingRecorder()
    _state, guided, next_turn = await _dispatch(
        session,
        state,
        svc,
        recorder=recorder,
        turn_response_override=ack_body,
    )
    assert guided.terminal is not None
    assert guided.terminal.kind is TerminalKind.COMPLETED
    assert next_turn is None
    decision = SignoffDecision(outcome=SignoffOutcome.COMPLETE, reason="unavailable", findings_text="", passes_delta=0)
    assert recorder.invocations[-1].tool_name == signoff_audit_event_name(decision)
    assert recorder.invocations[-1].tool_name == "composer.signoff.completed_without_signoff_advisor_unreachable"


@pytest.mark.asyncio
async def test_dispatch_records_clean_name_on_clean_signoff() -> None:
    # CLEAN sign-off completes; the last recorded invocation is the sign-off
    # audit event and it is composer.signoff.clean (NOT the unreachable name).
    session, state = make_wire_ready_session_and_state(profile=ADVISOR_ON_PROFILE)
    svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
    recorder = BufferingRecorder()
    _state, guided, next_turn = await _dispatch(session, state, svc, recorder=recorder)
    assert guided.terminal is not None
    assert guided.terminal.kind is TerminalKind.COMPLETED
    assert next_turn is None
    assert recorder.invocations[-1].tool_name == "composer.signoff.clean"


@pytest.mark.asyncio
async def test_dispatch_blocked_flagged_records_event_and_carries_fail_closed_findings() -> None:
    # FLAGGED on the last budgeted pass => BLOCKED_FLAGGED. The sign-off audit
    # event is recorded (NOT last — guided_turn_emitted follows the re-emit) and
    # the re-emitted turn carries the NON-RUNNABLE blocked-validation findings.
    session, state = make_wire_ready_session_and_state(profile=ADVISOR_ON_PROFILE)
    session = dataclasses.replace(session, advisor_checkpoint_passes_used=2)
    svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: prompt sees no row field"))
    recorder = BufferingRecorder()
    _state, guided, next_turn = await _dispatch(session, state, svc, recorder=recorder)
    assert guided.terminal is None
    assert next_turn is not None
    assert next_turn["payload"]["signoff_outcome"] == SignoffOutcome.BLOCKED_FLAGGED.value
    # The audit event is present (not necessarily last — the turn emit follows).
    names = [inv.tool_name for inv in recorder.invocations]
    assert "composer.signoff.blocked_flagged" in names
    # The carried findings are the fail-closed blocked-validation message, not a
    # plain echo of the advisor text.
    blocked = _advisor_signoff_blocked_validation(reason="exhausted", findings="FLAGGED: prompt sees no row field")
    assert blocked.errors[0].message in str(next_turn["payload"]["advisor_findings"])
