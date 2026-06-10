"""Release image publication required-check verifier tests."""

from __future__ import annotations

from scripts.cicd.check_release_required_checks import (
    CheckRun,
    RequiredCheckResult,
    evaluate_required_checks,
    extract_required_contexts,
)


def test_extract_required_contexts_from_ruleset() -> None:
    """The verifier must use the live ruleset's required-status context list."""
    ruleset = {
        "rules": [
            {"type": "deletion", "parameters": None},
            {
                "type": "required_status_checks",
                "parameters": {
                    "required_status_checks": [
                        {"context": "CI Success"},
                        {"context": "CodeQL"},
                    ],
                },
            },
        ],
    }

    assert extract_required_contexts(ruleset) == ("CI Success", "CodeQL")


def test_evaluate_required_checks_fails_when_required_context_is_missing() -> None:
    """CI Success alone must not authorize image publication."""
    results = evaluate_required_checks(
        required_contexts=(
            "CI Success",
            "CodeQL",
            "Check cohort-attribution trailers on PR commits",
            "redaction-gate",
        ),
        check_runs=[
            CheckRun(name="CI Success", status="completed", conclusion="success", html_url="https://example.invalid/ci"),
            CheckRun(name="Analyze Python", status="completed", conclusion="success", html_url="https://example.invalid/codeql"),
        ],
        statuses=[],
    )

    assert results == (
        RequiredCheckResult(
            context="CI Success",
            matched_name="CI Success",
            state="success",
            url="https://example.invalid/ci",
        ),
        RequiredCheckResult(
            context="CodeQL",
            matched_name="Analyze Python",
            state="success",
            url="https://example.invalid/codeql",
        ),
        RequiredCheckResult(
            context="Check cohort-attribution trailers on PR commits",
            matched_name=None,
            state="missing",
            url=None,
        ),
        RequiredCheckResult(
            context="redaction-gate",
            matched_name=None,
            state="missing",
            url=None,
        ),
    )


def test_evaluate_required_checks_accepts_all_successful_required_contexts() -> None:
    """Publication can proceed only when every required context has succeeded."""
    results = evaluate_required_checks(
        required_contexts=(
            "CI Success",
            "CodeQL",
            "Check cohort-attribution trailers on PR commits",
            "redaction-gate",
        ),
        check_runs=[
            CheckRun(name="CI Success", status="completed", conclusion="success", html_url=None),
            CheckRun(name="Analyze Python", status="completed", conclusion="success", html_url=None),
            CheckRun(name="Check cohort-attribution trailers on PR commits", status="completed", conclusion="success", html_url=None),
            CheckRun(name="redaction-gate", status="completed", conclusion="success", html_url=None),
        ],
        statuses=[],
    )

    assert all(result.state == "success" for result in results)


def test_evaluate_required_checks_uses_latest_matching_check_run() -> None:
    """An older success must not mask a newer failed run for the same context."""
    results = evaluate_required_checks(
        required_contexts=("CI Success",),
        check_runs=[
            CheckRun(
                name="CI Success",
                status="completed",
                conclusion="success",
                html_url="https://example.invalid/old",
                completed_at="2026-06-01T00:00:00Z",
            ),
            CheckRun(
                name="CI Success",
                status="completed",
                conclusion="failure",
                html_url="https://example.invalid/new",
                completed_at="2026-06-02T00:00:00Z",
            ),
        ],
        statuses=[],
    )

    assert results == (
        RequiredCheckResult(
            context="CI Success",
            matched_name="CI Success",
            state="failure",
            url="https://example.invalid/new",
        ),
    )
