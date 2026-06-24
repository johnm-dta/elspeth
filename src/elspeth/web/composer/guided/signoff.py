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
    from elspeth.web.composer.service import AdvisorCheckpointVerdict


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
