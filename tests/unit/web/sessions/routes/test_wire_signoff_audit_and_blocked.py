"""Phase P5.7 — differentiated sign-off audit names + fail-closed findings."""

from __future__ import annotations

from elspeth.web.composer.guided.signoff import (
    SignoffDecision,
    SignoffOutcome,
    signoff_audit_event_name,
)
from elspeth.web.composer.service import _advisor_signoff_blocked_validation


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
