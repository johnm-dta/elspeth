"""Unit coverage for the labelled cicd-judge quality corpus."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from elspeth_lints.core.allowlist import JudgeVerdict
from elspeth_lints.core.cli import main
from elspeth_lints.core.judge import JudgeRequest, JudgeResponse
from elspeth_lints.core.judge_quality import (
    JUDGE_QUALITY_MAX_CASES,
    JUDGE_QUALITY_MIN_CASES,
    JudgeQualityCase,
    JudgeQualityError,
    evaluate_judge_quality_corpus,
    load_judge_quality_corpus,
    render_judge_quality_report_text,
)

SHIPPED_CORPUS = Path("config/cicd/judge-quality-corpus/v1.jsonl")


def test_shipped_judge_quality_corpus_has_labelled_discrimination_cases() -> None:
    cases = load_judge_quality_corpus(SHIPPED_CORPUS)

    assert JUDGE_QUALITY_MIN_CASES <= len(cases) <= JUDGE_QUALITY_MAX_CASES
    assert any(case.expected_verdict is JudgeVerdict.ACCEPTED for case in cases)
    assert any(case.expected_verdict is JudgeVerdict.BLOCKED for case in cases)
    assert any(case.expected_should_use_decorator is not None for case in cases)


def test_load_judge_quality_corpus_rejects_extra_fields(tmp_path: Path) -> None:
    corpus = tmp_path / "bad.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "id": "bad_extra",
                "file_path": "src/elspeth/example.py",
                "rule_id": "R1",
                "symbol": "example",
                "fingerprint": "fp",
                "rationale": "rationale",
                "surrounding_code": "def example(): pass",
                "expected_verdict": "ACCEPTED",
                "expected_should_use_decorator": None,
                "unexpected": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JudgeQualityError, match="extra"):
        load_judge_quality_corpus(corpus)


def test_evaluate_judge_quality_corpus_scores_verdict_and_decorator_mismatches(tmp_path: Path) -> None:
    cases = (
        _case("accept_ok", expected_verdict=JudgeVerdict.ACCEPTED),
        _case(
            "blocked_bad_decorator",
            expected_verdict=JudgeVerdict.BLOCKED,
            expected_should_use_decorator="arguments",
        ),
    )

    def fake_judge(request: JudgeRequest) -> JudgeResponse:
        if request.fingerprint == "accept_ok":
            return _response(JudgeVerdict.ACCEPTED)
        return _response(JudgeVerdict.BLOCKED, should_use_decorator=None)

    report = evaluate_judge_quality_corpus(
        cases=cases,
        corpus_path=tmp_path / "corpus.jsonl",
        min_accuracy=1.0,
        min_cases=2,
        max_cases=2,
        judge_caller=fake_judge,
    )

    assert report.passed_count == 1
    assert report.failed_count == 1
    assert report.accuracy == pytest.approx(0.5)
    assert not report.passes
    rendered = render_judge_quality_report_text(report)
    assert "blocked_bad_decorator" in rendered
    assert "expected=BLOCKED/arguments" in rendered
    assert "actual=BLOCKED/null" in rendered


def test_check_judge_quality_cli_uses_live_judge_boundary_with_mocked_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "quality.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "id": "accept_one",
                "file_path": "src/elspeth/example.py",
                "rule_id": "R1",
                "symbol": "example",
                "fingerprint": "accept_one",
                "rationale": "already decorated Tier-3 boundary",
                "surrounding_code": "def example(arguments):\n    token = 'OPENAI_KEY_REDACTED_PLACEHOLDER'\n    return arguments.get('path')",
                "expected_verdict": "ACCEPTED",
                "expected_should_use_decorator": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_call_judge(request: JudgeRequest, *, model_id: str, max_tokens: int) -> JudgeResponse:
        assert request.fingerprint == "accept_one"
        assert "sk-AAAAAAAA" not in request.surrounding_code
        assert "[REDACTED-SECRET-" in request.surrounding_code
        assert model_id
        assert max_tokens > 0
        return _response(JudgeVerdict.ACCEPTED)

    monkeypatch.setattr("elspeth_lints.core.judge_quality.call_judge", fake_call_judge)

    exit_code = main(
        [
            "check-judge-quality",
            "--corpus",
            str(corpus),
            "--min-accuracy",
            "1.0",
            "--min-cases",
            "1",
            "--max-cases",
            "1",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["passed_count"] == 1
    assert payload["cases"][0]["case_id"] == "accept_one"


def _case(
    case_id: str,
    *,
    expected_verdict: JudgeVerdict,
    expected_should_use_decorator: str | None = None,
) -> JudgeQualityCase:
    return JudgeQualityCase(
        case_id=case_id,
        file_path="src/elspeth/example.py",
        rule_id="R1",
        symbol="example",
        fingerprint=case_id,
        rationale="test rationale",
        surrounding_code="def example(): pass",
        expected_verdict=expected_verdict,
        expected_should_use_decorator=expected_should_use_decorator,
    )


def _response(verdict: JudgeVerdict, *, should_use_decorator: str | None = None) -> JudgeResponse:
    return JudgeResponse(
        verdict=verdict,
        model_id="test-model",
        judge_rationale="test rationale from judge",
        recorded_at=datetime.now(UTC),
        should_use_decorator=should_use_decorator,
        confidence=0.9,
        prompt_tokens_total=100,
        prompt_tokens_cached=50,
        policy_hash="sha256:test",
    )
