"""Phase P5.5 — persisted-counter-bound wire-stage sign-off runner."""

from __future__ import annotations

import dataclasses

import pytest

from elspeth.web.composer.guided.signoff import (
    SignoffOutcome,
    run_wire_signoff,
)
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.composer.service import _ADVISOR_UNAVAILABLE_USER_DETAIL, AdvisorCheckpointVerdict
from elspeth.web.composer.state import (
    CompositionState,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)


def _state() -> CompositionState:
    return CompositionState(
        source=SourceSpec(plugin="csv", on_success="main", options={"path": "in.csv"}, on_validation_failure="discard"),
        nodes=(),
        edges=(),
        outputs=(OutputSpec(name="out", plugin="csv", options={"path": "out.csv"}, on_write_failure="discard"),),
        metadata=PipelineMetadata(),
        version=2,
    )


class _AdvisorServiceFake:
    def __init__(self, verdict: AdvisorCheckpointVerdict | None) -> None:
        self._verdict = verdict
        self.run_signoff_checkpoint_calls = 0

    async def run_signoff_checkpoint(
        self,
        *,
        state: CompositionState,
        session_id: str | None,
        recorder: object | None,
        progress: object | None = None,
    ) -> AdvisorCheckpointVerdict:
        del state, session_id, recorder, progress
        self.run_signoff_checkpoint_calls += 1
        if self._verdict is None:
            raise AssertionError("advisor sign-off checkpoint must not be called")
        return self._verdict

    def assert_signoff_checkpoint_awaited_once(self) -> None:
        assert self.run_signoff_checkpoint_calls == 1

    def assert_signoff_checkpoint_not_awaited(self) -> None:
        assert self.run_signoff_checkpoint_calls == 0


def _service(verdict: AdvisorCheckpointVerdict | None) -> _AdvisorServiceFake:
    return _AdvisorServiceFake(verdict)


@pytest.mark.asyncio
async def test_clean_completes_and_increments_counter() -> None:
    session = GuidedSession.initial()
    svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
    new_session, decision = await run_wire_signoff(
        session=session,
        state=_state(),
        session_id="s1",
        recorder=None,
        composer_service=svc,
        max_passes=3,
        acknowledged_unavailable=False,
    )
    assert decision.outcome is SignoffOutcome.COMPLETE
    assert decision.reason is None
    assert new_session.advisor_checkpoint_passes_used == 1
    svc.assert_signoff_checkpoint_awaited_once()


@pytest.mark.asyncio
async def test_flagged_last_pass_blocks_no_bypass() -> None:
    session = GuidedSession.initial()
    session = dataclasses.replace(session, advisor_checkpoint_passes_used=2)
    svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: bad"))
    new_session, decision = await run_wire_signoff(
        session=session,
        state=_state(),
        session_id="s1",
        recorder=None,
        composer_service=svc,
        max_passes=3,
        acknowledged_unavailable=False,
    )
    assert decision.outcome is SignoffOutcome.BLOCKED_FLAGGED
    assert decision.reason == "exhausted"
    assert new_session.advisor_checkpoint_passes_used == 3


@pytest.mark.asyncio
async def test_budget_already_spent_does_not_recall_provider() -> None:
    session = GuidedSession.initial()
    session = dataclasses.replace(session, advisor_checkpoint_passes_used=3)
    svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
    new_session, decision = await run_wire_signoff(
        session=session,
        state=_state(),
        session_id="s1",
        recorder=None,
        composer_service=svc,
        max_passes=3,
        acknowledged_unavailable=False,
    )
    assert decision.outcome is SignoffOutcome.BLOCKED_FLAGGED
    assert new_session.advisor_checkpoint_passes_used == 3
    svc.assert_signoff_checkpoint_not_awaited()


@pytest.mark.asyncio
async def test_unavailable_last_pass_offers_escape_when_unacknowledged() -> None:
    session = GuidedSession.initial()
    session = dataclasses.replace(session, advisor_checkpoint_passes_used=2)
    svc = _service(
        AdvisorCheckpointVerdict(
            ok=False,
            blocking=False,
            failure_class="unavailable",
            findings_text=_ADVISOR_UNAVAILABLE_USER_DETAIL,
        )
    )
    _session, decision = await run_wire_signoff(
        session=session,
        state=_state(),
        session_id="s1",
        recorder=None,
        composer_service=svc,
        max_passes=3,
        acknowledged_unavailable=False,
    )
    assert decision.outcome is SignoffOutcome.ESCAPE_UNAVAILABLE
    assert decision.reason == "unavailable"


@pytest.mark.asyncio
async def test_same_request_unavailable_acknowledgement_is_not_honored() -> None:
    session = GuidedSession.initial()
    session = dataclasses.replace(session, advisor_checkpoint_passes_used=2)
    svc = _service(
        AdvisorCheckpointVerdict(
            ok=False,
            blocking=False,
            failure_class="unavailable",
            findings_text=_ADVISOR_UNAVAILABLE_USER_DETAIL,
        )
    )
    new_session, decision = await run_wire_signoff(
        session=session,
        state=_state(),
        session_id="s1",
        recorder=None,
        composer_service=svc,
        max_passes=3,
        acknowledged_unavailable=True,
    )
    # A client must not pre-send the acknowledgement and complete in the same
    # request that first discovers the advisor outage. The server first emits a
    # persisted escape offer; only a later request may acknowledge it.
    assert decision.outcome is SignoffOutcome.ESCAPE_UNAVAILABLE
    assert decision.reason == "unavailable"
    assert new_session.advisor_signoff_escape_offered is True


@pytest.mark.asyncio
async def test_acknowledged_unavailable_never_bypasses_a_flag() -> None:
    # A FLAG on the last pass with acknowledged_unavailable=True must still BLOCK.
    session = GuidedSession.initial()
    session = dataclasses.replace(session, advisor_checkpoint_passes_used=2)
    svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: bad"))
    _session, decision = await run_wire_signoff(
        session=session,
        state=_state(),
        session_id="s1",
        recorder=None,
        composer_service=svc,
        max_passes=3,
        acknowledged_unavailable=True,
    )
    assert decision.outcome is SignoffOutcome.BLOCKED_FLAGGED


@pytest.mark.asyncio
async def test_exhausted_with_acknowledged_outage_completes_cross_request() -> None:
    # D5/B2 regression: the escape is OFFERED on the final pass (one request) and
    # ACKNOWLEDGED on a LATER request, by which time passes_used == max_passes. The
    # persisted escape_offered marker lets the acknowledgement COMPLETE rather than
    # dead-end to BLOCKED_FLAGGED — and the provider is NOT re-called at exhaustion.
    session = dataclasses.replace(
        GuidedSession.initial(),
        advisor_checkpoint_passes_used=3,
        advisor_signoff_escape_offered=True,
    )
    svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
    _session, decision = await run_wire_signoff(
        session=session,
        state=_state(),
        session_id="s1",
        recorder=None,
        composer_service=svc,
        max_passes=3,
        acknowledged_unavailable=True,
    )
    assert decision.outcome is SignoffOutcome.COMPLETE
    assert decision.reason == "unavailable"
    svc.assert_signoff_checkpoint_not_awaited()


@pytest.mark.asyncio
async def test_exhausted_acknowledged_but_prior_was_flag_stays_blocked() -> None:
    # The acknowledgement must NEVER bypass a FLAG: a FLAGGED/MALFORMED-exhausted
    # terminal leaves escape_offered=False, so acknowledging it stays BLOCKED.
    session = dataclasses.replace(
        GuidedSession.initial(),
        advisor_checkpoint_passes_used=3,
        advisor_signoff_escape_offered=False,
    )
    svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
    _session, decision = await run_wire_signoff(
        session=session,
        state=_state(),
        session_id="s1",
        recorder=None,
        composer_service=svc,
        max_passes=3,
        acknowledged_unavailable=True,
    )
    assert decision.outcome is SignoffOutcome.BLOCKED_FLAGGED
    svc.assert_signoff_checkpoint_not_awaited()
