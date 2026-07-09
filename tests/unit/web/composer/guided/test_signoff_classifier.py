"""Phase P5.3 — pure D13 verdict-class classifier for the wire-stage sign-off."""

from __future__ import annotations

from typing import Any, cast

from elspeth.web.composer.guided.signoff import (
    SignoffOutcome,
    classify_signoff_verdict,
)
from elspeth.web.composer.service import (
    _ADVISOR_UNAVAILABLE_USER_DETAIL,
    AdvisorCheckpointVerdict,
)


def _clean() -> AdvisorCheckpointVerdict:
    return AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN: good")


def _flagged() -> AdvisorCheckpointVerdict:
    return AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: prompt sees no row field")


def _unavailable() -> AdvisorCheckpointVerdict:
    return AdvisorCheckpointVerdict(
        ok=False,
        blocking=False,
        failure_class="unavailable",
        findings_text=_ADVISOR_UNAVAILABLE_USER_DETAIL,
    )


def _malformed() -> AdvisorCheckpointVerdict:
    # The advisor returned output the call core could not parse -> re-raised ->
    # caught -> ok=False with failure_class="malformed". Must FAIL CLOSED (D4/B2).
    return AdvisorCheckpointVerdict(
        ok=False,
        blocking=False,
        failure_class="malformed",
        findings_text="advisor response was malformed",
    )


def _default_none_failure() -> AdvisorCheckpointVerdict:
    # ok=False with the default failure_class="none" is malformed/blocked by policy.
    return AdvisorCheckpointVerdict(
        ok=False,
        blocking=False,
        findings_text="advisor response was malformed",
    )


def _unknown_failure_class() -> AdvisorCheckpointVerdict:
    # Forward-compat guard: only the exact value "unavailable" may escape.
    return AdvisorCheckpointVerdict(
        ok=False,
        blocking=False,
        failure_class=cast(Any, "unknown"),
        findings_text="advisor response was malformed",
    )


def test_clean_completes() -> None:
    d = classify_signoff_verdict(_clean(), passes_used=0, max_passes=3)
    assert d.outcome is SignoffOutcome.COMPLETE
    assert d.reason is None
    assert d.passes_delta == 1


def test_flagged_revises_while_budget_remains() -> None:
    d = classify_signoff_verdict(_flagged(), passes_used=0, max_passes=3)
    assert d.outcome is SignoffOutcome.REVISE
    assert "prompt sees no row field" in d.findings_text
    assert d.passes_delta == 1


def test_flagged_blocks_on_last_pass_no_bypass() -> None:
    d = classify_signoff_verdict(_flagged(), passes_used=2, max_passes=3)
    assert d.outcome is SignoffOutcome.BLOCKED_FLAGGED
    assert d.reason == "exhausted"


def test_unavailable_revises_while_budget_remains() -> None:
    d = classify_signoff_verdict(_unavailable(), passes_used=0, max_passes=3)
    assert d.outcome is SignoffOutcome.REVISE
    assert d.passes_delta == 1


def test_unavailable_offers_escape_on_last_pass() -> None:
    d = classify_signoff_verdict(_unavailable(), passes_used=2, max_passes=3)
    assert d.outcome is SignoffOutcome.ESCAPE_UNAVAILABLE
    assert d.reason == "unavailable"


def test_malformed_revises_while_budget_remains() -> None:
    d = classify_signoff_verdict(_malformed(), passes_used=0, max_passes=3)
    assert d.outcome is SignoffOutcome.REVISE
    assert d.passes_delta == 1


def test_malformed_fails_closed_on_last_pass_never_escapes() -> None:
    # D4/B2 regression: a MALFORMED verdict (ok=False) must NOT take the
    # UNAVAILABLE escape — it fails closed exactly like a FLAG.
    d = classify_signoff_verdict(_malformed(), passes_used=2, max_passes=3)
    assert d.outcome is SignoffOutcome.BLOCKED_FLAGGED
    assert d.outcome is not SignoffOutcome.ESCAPE_UNAVAILABLE
    assert d.reason == "exhausted"


def test_default_none_failure_class_fails_closed_on_last_pass() -> None:
    # ok=False + default/none is not a genuine outage; it fails closed.
    d = classify_signoff_verdict(_default_none_failure(), passes_used=2, max_passes=3)
    assert d.outcome is SignoffOutcome.BLOCKED_FLAGGED
    assert d.outcome is not SignoffOutcome.ESCAPE_UNAVAILABLE
    assert d.reason == "exhausted"


def test_unknown_failure_class_fails_closed_on_last_pass() -> None:
    # Future/unknown classes must not accidentally become escapable.
    d = classify_signoff_verdict(_unknown_failure_class(), passes_used=2, max_passes=3)
    assert d.outcome is SignoffOutcome.BLOCKED_FLAGGED
    assert d.outcome is not SignoffOutcome.ESCAPE_UNAVAILABLE
    assert d.reason == "exhausted"


def test_flagged_never_yields_an_escape() -> None:
    # A FLAG can never reach the unavailable escape — only BLOCKED_FLAGGED.
    d = classify_signoff_verdict(_flagged(), passes_used=2, max_passes=3)
    assert d.outcome is not SignoffOutcome.ESCAPE_UNAVAILABLE
