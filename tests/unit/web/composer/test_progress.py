"""Tests for session-scoped composer progress snapshots."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest
from pydantic import ValidationError

from elspeth.web.composer.progress import (
    COMPOSER_PROGRESS_MAX_EVIDENCE,
    NON_TERMINAL_PROGRESS_PHASES,
    ComposerProgressEvent,
    ComposerProgressRegistry,
    client_cancelled_progress_event,
    convergence_progress_event,
)


class TestComposerProgressEvent:
    def test_rejects_unknown_phase(self) -> None:
        """Progress phases are a closed, typed surface for frontend handling."""
        with pytest.raises(ValidationError):
            ComposerProgressEvent(
                phase=cast(Any, "thinking"),
                headline="Thinking about hidden details",
                evidence=(),
            )

    def test_evidence_is_bounded(self) -> None:
        event = ComposerProgressEvent(
            phase="using_tools",
            headline="Checking available tools",
            evidence=tuple(f"visible boundary {index}" for index in range(COMPOSER_PROGRESS_MAX_EVIDENCE + 3)),
        )

        assert len(event.evidence) == COMPOSER_PROGRESS_MAX_EVIDENCE
        assert event.evidence[-1] == f"visible boundary {COMPOSER_PROGRESS_MAX_EVIDENCE - 1}"

    def test_rejects_unknown_reason(self) -> None:
        """Reason codes are a closed Literal so the frontend can branch safely."""
        with pytest.raises(ValidationError):
            ComposerProgressEvent(
                phase="failed",
                headline="The composer could not finish this request.",
                evidence=("Some boundary.",),
                reason=cast(Any, "totally_made_up_reason"),
            )

    def test_failed_phase_requires_reason(self) -> None:
        """phase='failed' MUST carry a stable reason code so UX text drift is impossible."""
        with pytest.raises(ValidationError) as exc_info:
            ComposerProgressEvent(
                phase="failed",
                headline="The composer could not finish this request.",
                evidence=("The bounded composer loop stopped before a final answer.",),
            )
        assert "reason" in str(exc_info.value).lower()

    def test_non_failed_phases_allow_omitted_reason(self) -> None:
        """Non-failed events may omit the reason — they default to None."""
        event = ComposerProgressEvent(
            phase="using_tools",
            headline="The model is updating the pipeline graph.",
            evidence=("A pipeline-editing tool was requested.",),
        )
        assert event.reason is None

    def test_accepts_each_documented_reason_code(self) -> None:
        """Every documented reason value must round-trip through the model."""
        valid_reasons = (
            "convergence_composition_budget",
            "convergence_discovery_budget",
            "convergence_wall_clock_timeout",
            "provider_auth_failed",
            "provider_unavailable",
            "plugin_crash",
            "runtime_preflight_failed",
            "service_setup_failed",
            "client_cancelled",
            "composer_idle",
            "composer_complete",
        )
        for reason in valid_reasons:
            # client_cancelled is paired with phase="cancelled"; everything else
            # currently pairs with phase="failed". The validator only requires
            # *some* reason on terminal-non-success phases, so this is a
            # round-trip check, not a phase/reason coherence check.
            phase: str = "cancelled" if reason == "client_cancelled" else "failed"
            event = ComposerProgressEvent(
                phase=cast(Any, phase),
                headline="A safely bounded terminal event.",
                evidence=("Boundary text.",),
                reason=cast(Any, reason),
            )
            assert event.reason == reason

    def test_cancelled_phase_requires_reason(self) -> None:
        """phase='cancelled' MUST carry a stable reason — same rule as 'failed'.

        Without this, a future operator-initiated cancel reason could be
        added that collapses with client disconnect into one generic
        ``cancelled`` event — exactly the elspeth-5030f7373d failure mode
        but on the cancellation axis.
        """
        with pytest.raises(ValidationError) as exc_info:
            ComposerProgressEvent(
                phase="cancelled",
                headline="The request was cancelled.",
                evidence=("The connection closed.",),
            )
        assert "reason" in str(exc_info.value).lower()

    def test_client_cancelled_event_pairs_with_cancelled_phase(self) -> None:
        """The helper must emit the discriminated cancellation event."""
        event = client_cancelled_progress_event()
        assert event.phase == "cancelled"
        assert event.reason == "client_cancelled"
        assert event.likely_next is not None
        # Recovery copy is for the user, not the operator.
        assert "resubmit" in event.likely_next.lower()


class TestConvergenceProgressEvent:
    """The discriminator that fixes elspeth-5030f7373d.

    The original symptom was that wall-clock timeout, mutation-turn budget,
    and discovery-turn budget all collapsed into one generic ``phase: failed``
    event. These tests pin the three distinct events the helper must emit,
    so any future refactor that re-collapses them fails immediately.
    """

    def test_wall_clock_timeout_emits_distinct_event(self) -> None:
        event = convergence_progress_event(budget_exhausted="timeout")
        assert event.phase == "failed"
        assert event.reason == "convergence_wall_clock_timeout"
        assert "timed out" in event.headline.lower()
        assert event.likely_next is not None
        assert "wall-clock" in event.likely_next.lower()

    def test_composition_budget_emits_distinct_event(self) -> None:
        event = convergence_progress_event(budget_exhausted="composition")
        assert event.phase == "failed"
        assert event.reason == "convergence_composition_budget"
        assert "mutation turn budget" in event.headline.lower()
        assert event.likely_next is not None
        assert "smaller turns" in event.likely_next.lower()

    def test_discovery_budget_emits_distinct_event(self) -> None:
        event = convergence_progress_event(budget_exhausted="discovery")
        assert event.phase == "failed"
        assert event.reason == "convergence_discovery_budget"
        assert "discovery turn budget" in event.headline.lower()
        assert event.likely_next is not None
        assert "narrow" in event.likely_next.lower()

    def test_three_sub_causes_produce_three_distinct_reason_codes(self) -> None:
        """Regression guard: the three causes must NOT collapse to one code."""
        codes = {
            convergence_progress_event(budget_exhausted="timeout").reason,
            convergence_progress_event(budget_exhausted="composition").reason,
            convergence_progress_event(budget_exhausted="discovery").reason,
        }
        assert len(codes) == 3, (
            "Three convergence sub-causes collapsed back into fewer reason codes — "
            "elspeth-5030f7373d regression. Codes seen: " + repr(codes)
        )


class TestComposerProgressRegistry:
    @pytest.mark.asyncio
    async def test_returns_idle_snapshot_when_session_has_no_progress(self) -> None:
        registry = ComposerProgressRegistry()

        snapshot = await registry.get_latest("session-1")

        assert snapshot.session_id == "session-1"
        assert snapshot.request_id is None
        assert snapshot.phase == "idle"
        assert snapshot.headline == "No active composer work."
        assert snapshot.evidence == ()
        assert snapshot.reason == "composer_idle"

    @pytest.mark.asyncio
    async def test_keeps_only_latest_snapshot_per_session(self) -> None:
        registry = ComposerProgressRegistry()

        first = await registry.publish(
            session_id="session-1",
            request_id="message-1",
            user_id="user-1",
            event=ComposerProgressEvent(
                phase="starting",
                headline="I'm reading your request and current pipeline.",
                evidence=("The request was accepted.",),
            ),
        )
        second = await registry.publish(
            session_id="session-1",
            request_id="message-1",
            user_id="user-1",
            event=ComposerProgressEvent(
                phase="calling_model",
                headline="I'm asking the model to choose the next pipeline change.",
                evidence=("The composer prompt was built.",),
            ),
        )

        latest = await registry.get_latest("session-1")

        assert first.updated_at < second.updated_at
        assert latest == second
        assert latest.phase == "calling_model"

    @pytest.mark.asyncio
    async def test_clear_removes_session_snapshot(self) -> None:
        registry = ComposerProgressRegistry()
        await registry.publish(
            session_id="session-1",
            request_id="message-1",
            user_id="user-1",
            event=ComposerProgressEvent(
                phase="failed",
                headline="The composer could not finish this request.",
                evidence=("The safe failure path was reached.",),
                reason="service_setup_failed",
            ),
        )

        await registry.clear("session-1")
        snapshot = await registry.get_latest("session-1")

        assert snapshot.phase == "idle"
        assert snapshot.updated_at <= datetime.now(UTC)

    @pytest.mark.asyncio
    async def test_list_active_returns_only_non_terminal_phases(self) -> None:
        """list_active is the cross-session enumeration primitive used by /_active."""
        registry = ComposerProgressRegistry()
        # Two non-terminal sessions for user-1 and one terminated session.
        await registry.publish(
            session_id="session-running-1",
            request_id="msg-1",
            user_id="user-1",
            event=ComposerProgressEvent(
                phase="calling_model",
                headline="The model is composing.",
                evidence=("Prompt was built.",),
            ),
        )
        await registry.publish(
            session_id="session-running-2",
            request_id="msg-2",
            user_id="user-1",
            event=ComposerProgressEvent(
                phase="using_tools",
                headline="The model is using tools.",
                evidence=("A tool call started.",),
            ),
        )
        await registry.publish(
            session_id="session-done",
            request_id="msg-3",
            user_id="user-1",
            event=ComposerProgressEvent(
                phase="complete",
                headline="The composer response is ready.",
                evidence=("The assistant response was saved.",),
                reason="composer_complete",
            ),
        )

        active = await registry.list_active(user_id="user-1")

        assert {snap.session_id for snap in active} == {"session-running-1", "session-running-2"}
        assert all(snap.phase in NON_TERMINAL_PROGRESS_PHASES for snap in active)

    @pytest.mark.asyncio
    async def test_list_active_scopes_to_user_id(self) -> None:
        """A caller cannot enumerate other users' in-flight sessions.

        The internal user index is the only mechanism enforcing this — there
        is no DB lookup at the endpoint, so this scoping must be airtight.
        """
        registry = ComposerProgressRegistry()
        await registry.publish(
            session_id="session-mine",
            request_id="msg-1",
            user_id="user-alice",
            event=ComposerProgressEvent(
                phase="calling_model",
                headline="The model is composing.",
                evidence=("Prompt built.",),
            ),
        )
        await registry.publish(
            session_id="session-yours",
            request_id="msg-2",
            user_id="user-bob",
            event=ComposerProgressEvent(
                phase="calling_model",
                headline="Different user's request.",
                evidence=("Prompt built.",),
            ),
        )

        alice_active = await registry.list_active(user_id="user-alice")
        bob_active = await registry.list_active(user_id="user-bob")

        assert {snap.session_id for snap in alice_active} == {"session-mine"}
        assert {snap.session_id for snap in bob_active} == {"session-yours"}

    @pytest.mark.asyncio
    async def test_list_active_orders_oldest_first(self) -> None:
        """Triage order: longest-running request at the top, like a DB lock list."""
        registry = ComposerProgressRegistry()
        await registry.publish(
            session_id="session-old",
            request_id="msg-old",
            user_id="user-1",
            event=ComposerProgressEvent(
                phase="calling_model",
                headline="Older request.",
                evidence=("Already in flight.",),
            ),
        )
        await registry.publish(
            session_id="session-new",
            request_id="msg-new",
            user_id="user-1",
            event=ComposerProgressEvent(
                phase="calling_model",
                headline="Newer request.",
                evidence=("Just started.",),
            ),
        )

        active = await registry.list_active(user_id="user-1")

        assert [snap.session_id for snap in active] == ["session-old", "session-new"]

    @pytest.mark.asyncio
    async def test_list_active_excludes_cancelled_phase(self) -> None:
        """A cancelled session is no longer in flight — pin this regression guard."""
        registry = ComposerProgressRegistry()
        await registry.publish(
            session_id="session-cancelled",
            request_id="msg-1",
            user_id="user-1",
            event=client_cancelled_progress_event(),
        )

        active = await registry.list_active(user_id="user-1")

        assert active == ()

    @pytest.mark.asyncio
    async def test_clear_purges_user_index(self) -> None:
        """Clearing a session must drop its user-index entry too.

        Otherwise a re-published snapshot under the same session_id but a
        different user_id (e.g., session ownership transfer in a future
        feature) would still surface to the original user via list_active.
        """
        registry = ComposerProgressRegistry()
        await registry.publish(
            session_id="session-shared-id",
            request_id="msg-1",
            user_id="user-alice",
            event=ComposerProgressEvent(
                phase="calling_model",
                headline="Alice's request.",
                evidence=("Prompt built.",),
            ),
        )
        await registry.clear("session-shared-id")
        await registry.publish(
            session_id="session-shared-id",
            request_id="msg-2",
            user_id="user-bob",
            event=ComposerProgressEvent(
                phase="calling_model",
                headline="Bob's request after Alice's was cleared.",
                evidence=("Prompt built.",),
            ),
        )

        alice_active = await registry.list_active(user_id="user-alice")
        bob_active = await registry.list_active(user_id="user-bob")

        assert alice_active == ()
        assert {snap.session_id for snap in bob_active} == {"session-shared-id"}
