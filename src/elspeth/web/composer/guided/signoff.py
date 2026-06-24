"""Pure D13 verdict-class classifier for the STEP_4_WIRE advisor sign-off.

No ``self`` dependency: the wire-stage dispatcher (``_dispatch_guided_respond``)
and the unit tests both consume :func:`classify_signoff_verdict`. It maps an
:class:`AdvisorCheckpointVerdict` (the non-raising verdict produced by
``ComposerService.run_signoff_checkpoint``) to a terminal/redirect decision,
splitting the two non-CLEAN failure CLASSES per D13:

  * a *quality* FLAG (the advisor judged the pipeline unsafe) stays fully
    fail-closed — re-emit while passes remain, then BLOCKED with no bypass;
  * a *sustained infra* UNAVAILABLE (the advisor never rendered a judgement)
    gets a differentiated audited escape on budget exhaustion, ONLY for
    ``reason="unavailable"`` and NEVER reachable from a FLAG.

The classifier never touches the provider, never raises, and consumes no user
text — it is a pure function of the verdict + the persisted pass budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.contracts.composer_progress import ComposerProgressSink
    from elspeth.web.composer.audit import BufferingRecorder
    from elspeth.web.composer.guided.state_machine import GuidedSession
    from elspeth.web.composer.protocol import ComposerService
    from elspeth.web.composer.service import AdvisorCheckpointVerdict
    from elspeth.web.composer.state import CompositionState


class SignoffOutcome(StrEnum):
    """The terminal/redirect class for a wire-stage sign-off pass."""

    COMPLETE = "complete"  # CLEAN -> stamp COMPLETED
    REVISE = "revise"  # re-emit the wire turn (budget remains)
    BLOCKED_FLAGGED = "blocked_flagged"  # FLAGGED, budget exhausted -> fail-closed, no bypass
    BLOCKED_UNAVAILABLE = "blocked_unavailable"  # UNAVAILABLE escape declined -> fail-closed
    ESCAPE_UNAVAILABLE = "escape_unavailable"  # UNAVAILABLE, budget exhausted -> offer audited escape


@dataclass(frozen=True, slots=True)
class SignoffDecision:
    """Outcome of classifying one wire-stage advisor sign-off pass.

    ``reason`` is ``"exhausted"`` (FLAGGED, no repair left), ``"unavailable"``
    (advisor unreachable), or ``None`` (CLEAN / mid-budget REVISE). It feeds
    the blocked-result reason and the differentiated audit event. ``passes_delta``
    is always 1 — every classified pass consumed one budgeted advisor call.
    """

    outcome: SignoffOutcome
    reason: str | None
    findings_text: str
    passes_delta: int


def classify_signoff_verdict(
    verdict: AdvisorCheckpointVerdict,
    *,
    passes_used: int,
    max_passes: int,
) -> SignoffDecision:
    """Map an END sign-off verdict to a D13 terminal/redirect decision.

    ``passes_used`` is the PERSISTED ``GuidedSession.advisor_checkpoint_passes_used``
    BEFORE this pass; the function computes whether this is the last budgeted pass.
    """
    is_last_pass = (passes_used + 1) >= max_passes
    findings = verdict.findings_text

    if verdict.ok and not verdict.blocking:
        # CLEAN.
        return SignoffDecision(outcome=SignoffOutcome.COMPLETE, reason=None, findings_text=findings, passes_delta=1)

    if verdict.ok and verdict.blocking:
        # FLAGGED — a quality verdict. Fail-closed, no bypass.
        if is_last_pass:
            return SignoffDecision(outcome=SignoffOutcome.BLOCKED_FLAGGED, reason="exhausted", findings_text=findings, passes_delta=1)
        return SignoffDecision(outcome=SignoffOutcome.REVISE, reason=None, findings_text=findings, passes_delta=1)

    # not verdict.ok: the advisor call did not return a usable verdict. NOTE:
    # _run_advisor_checkpoint collapses exceptions to ok=False. So (ok, blocking)
    # ALONE cannot tell malformed/default/unknown failures from a genuine outage;
    # only the exact failure_class "unavailable" is allowed to reach the
    # budget-exhausted escape.
    if verdict.failure_class == "unavailable":
        if is_last_pass:
            return SignoffDecision(outcome=SignoffOutcome.ESCAPE_UNAVAILABLE, reason="unavailable", findings_text=findings, passes_delta=1)
        return SignoffDecision(outcome=SignoffOutcome.REVISE, reason=None, findings_text=findings, passes_delta=1)

    # MALFORMED, NONE/default, or any unknown future class: fail closed exactly
    # like FLAGGED. This prevents a new/omitted failure_class from silently
    # becoming an audited escape.
    if is_last_pass:
        return SignoffDecision(outcome=SignoffOutcome.BLOCKED_FLAGGED, reason="exhausted", findings_text=findings, passes_delta=1)
    return SignoffDecision(outcome=SignoffOutcome.REVISE, reason=None, findings_text=findings, passes_delta=1)


async def run_wire_signoff(
    *,
    session: GuidedSession,
    state: CompositionState,
    session_id: str | None,
    recorder: BufferingRecorder | None,
    composer_service: ComposerService,
    max_passes: int,
    acknowledged_unavailable: bool,
    progress: ComposerProgressSink | None = None,
) -> tuple[GuidedSession, SignoffDecision]:
    """Run one wire-stage END sign-off pass, bounded by the PERSISTED counter.

    Returns the (possibly counter-bumped) session and the D13 decision. The
    persisted ``GuidedSession.advisor_checkpoint_passes_used`` is the re-entry
    bound (D16): guided re-entry crosses separate
    ``POST /api/sessions/{session_id}/guided/respond`` HTTP
    requests, so an unpersisted per-compose local would reset to 0 each request
    and never bound the loop. When the budget is already spent on a prior
    request the provider is NOT re-called.

    ``acknowledged_unavailable`` is the route-validated "complete without
    sign-off (advisor unreachable)" acknowledgement. It is honored only when
    this frozen session already carries ``advisor_signoff_escape_offered=True``
    from a prior server-emitted ESCAPE_UNAVAILABLE turn and the persisted
    counter is exhausted. It can NEVER bypass a FLAG (a FLAG never sets the
    escape-offered marker).
    """
    import dataclasses

    passes_used = session.advisor_checkpoint_passes_used
    if passes_used >= max_passes:
        # Budget spent on a prior request: do not re-call the provider.
        if acknowledged_unavailable and session.advisor_signoff_escape_offered:
            # The prior budget-exhausting terminal was a genuine UNAVAILABLE
            # escape OFFER (persisted marker) and the user has now acknowledged
            # "complete without sign-off (advisor unreachable)". Honour it as the
            # audited COMPLETE-with-reason="unavailable". This can NEVER bypass a
            # FLAG: a FLAGGED-exhausted (or MALFORMED-exhausted) terminal leaves
            # escape_offered=False, so an acknowledgement there falls through to
            # BLOCKED below. The acknowledgement arrives on a SEPARATE
            # POST /api/sessions/{session_id}/guided/respond request than the
            # one that emitted the offer — which
            # is exactly why this cross-request marker is required (D5/B2).
            return session, SignoffDecision(
                outcome=SignoffOutcome.COMPLETE,
                reason="unavailable",
                findings_text="Advisor unreachable; completed without sign-off (acknowledged).",
                passes_delta=0,
            )
        # Otherwise fail closed (no bypass). FLAGGED-exhausted is the safe terminal.
        return session, SignoffDecision(
            outcome=SignoffOutcome.BLOCKED_FLAGGED,
            reason="exhausted",
            findings_text="Advisor sign-off budget exhausted.",
            passes_delta=0,
        )

    verdict = await composer_service.run_signoff_checkpoint(
        state=state,
        session_id=session_id,
        recorder=recorder,
        progress=progress,
    )
    decision = classify_signoff_verdict(verdict, passes_used=passes_used, max_passes=max_passes)
    new_session = dataclasses.replace(
        session,
        advisor_checkpoint_passes_used=passes_used + decision.passes_delta,
        # Persist whether THIS terminal was a genuine-outage escape OFFER, so a
        # later request carrying the user's acknowledgement (handled above) can
        # honour it without re-calling the provider — and so a FLAGGED-exhausted
        # terminal (escape_offered=False) can never be acknowledged into a bypass.
        advisor_signoff_escape_offered=(decision.outcome is SignoffOutcome.ESCAPE_UNAVAILABLE),
    )

    return new_session, decision
