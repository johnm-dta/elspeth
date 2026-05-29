"""Labelled quality corpus for the cicd-judge.

The judge gate has two distinct contracts:

* transport / schema integrity: the model returns the exact JSON shape
  required by :mod:`elspeth_lints.core.judge`;
* discrimination quality: the model distinguishes sound allowlist
  rationales from shallow or policy-violating ones.

Parser-only tests prove the first contract. This module is the second
contract: load a small labelled JSONL corpus, call the real judge
boundary for each case, and score exact matches on the expected verdict
and the structured ``should_use_decorator`` signal.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elspeth_lints.core.allowlist import JudgeVerdict
from elspeth_lints.core.judge import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    JudgeRequest,
    JudgeResponse,
    call_judge,
)
from elspeth_lints.core.source_excerpt import scrub_secrets

DEFAULT_JUDGE_QUALITY_MIN_ACCURACY: float = 0.90
JUDGE_QUALITY_MIN_CASES: int = 10
JUDGE_QUALITY_MAX_CASES: int = 30

_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_CASE_FIELDS = frozenset(
    {
        "id",
        "file_path",
        "rule_id",
        "symbol",
        "fingerprint",
        "rationale",
        "surrounding_code",
        "expected_verdict",
        "expected_should_use_decorator",
    }
)

JudgeCaller = Callable[[JudgeRequest], JudgeResponse]


class JudgeQualityError(RuntimeError):
    """The judge-quality corpus or evaluator configuration is invalid."""


@dataclass(frozen=True, slots=True)
class JudgeQualityCase:
    """One labelled judge-quality example."""

    case_id: str
    file_path: str
    rule_id: str
    symbol: str
    fingerprint: str
    rationale: str
    surrounding_code: str
    expected_verdict: JudgeVerdict
    expected_should_use_decorator: str | None

    def to_request(self) -> JudgeRequest:
        scrubbed = scrub_secrets(
            self.surrounding_code,
            salt=f"judge-quality:{self.case_id}:{self.fingerprint}",
            path_hint=self.file_path,
        )
        return JudgeRequest(
            file_path=self.file_path,
            rule_id=self.rule_id,
            symbol=self.symbol,
            fingerprint=self.fingerprint,
            rationale=self.rationale,
            surrounding_code=scrubbed.text,
        )


@dataclass(frozen=True, slots=True)
class JudgeQualityCaseResult:
    """The scored result for one corpus case."""

    case: JudgeQualityCase
    response: JudgeResponse

    @property
    def verdict_matches(self) -> bool:
        return self.response.verdict is self.case.expected_verdict

    @property
    def decorator_matches(self) -> bool:
        return self.response.should_use_decorator == self.case.expected_should_use_decorator

    @property
    def passed(self) -> bool:
        return self.verdict_matches and self.decorator_matches


@dataclass(frozen=True, slots=True)
class JudgeQualityReport:
    """Aggregate score for a corpus evaluation run."""

    corpus_path: Path
    requested_model_id: str
    min_accuracy: float
    results: tuple[JudgeQualityCaseResult, ...]

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed_count(self) -> int:
        return self.total_count - self.passed_count

    @property
    def accuracy(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.passed_count / self.total_count

    @property
    def passes(self) -> bool:
        return self.total_count > 0 and self.accuracy >= self.min_accuracy

    @property
    def failures(self) -> tuple[JudgeQualityCaseResult, ...]:
        return tuple(result for result in self.results if not result.passed)


def load_judge_quality_corpus(path: Path) -> tuple[JudgeQualityCase, ...]:
    """Load a strict JSONL judge-quality corpus."""
    if not path.is_file():
        raise JudgeQualityError(f"corpus {path} is not a file")

    cases: list[JudgeQualityCase] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise JudgeQualityError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise JudgeQualityError(f"{path}:{line_number}: case must be a JSON object")

        missing = _CASE_FIELDS - set(payload)
        extra = set(payload) - _CASE_FIELDS
        if missing or extra:
            parts: list[str] = []
            if missing:
                parts.append(f"missing={sorted(missing)}")
            if extra:
                parts.append(f"extra={sorted(extra)}")
            raise JudgeQualityError(f"{path}:{line_number}: invalid case fields ({'; '.join(parts)})")

        case_id = _required_non_empty_str(payload, "id", path=path, line_number=line_number)
        if _CASE_ID_RE.fullmatch(case_id) is None:
            raise JudgeQualityError(f"{path}:{line_number}: id must match {_CASE_ID_RE.pattern!r}; got {case_id!r}")
        if case_id in seen_ids:
            raise JudgeQualityError(f"{path}:{line_number}: duplicate id {case_id!r}")
        seen_ids.add(case_id)

        expected_verdict = _expected_verdict(payload, path=path, line_number=line_number)
        expected_should_use_decorator = _expected_decorator(payload, path=path, line_number=line_number)
        if expected_verdict is JudgeVerdict.ACCEPTED and expected_should_use_decorator is not None:
            raise JudgeQualityError(f"{path}:{line_number}: expected_should_use_decorator must be null when expected_verdict is ACCEPTED")

        cases.append(
            JudgeQualityCase(
                case_id=case_id,
                file_path=_required_non_empty_str(payload, "file_path", path=path, line_number=line_number),
                rule_id=_required_non_empty_str(payload, "rule_id", path=path, line_number=line_number),
                symbol=_required_non_empty_str(payload, "symbol", path=path, line_number=line_number),
                fingerprint=_required_non_empty_str(payload, "fingerprint", path=path, line_number=line_number),
                rationale=_required_non_empty_str(payload, "rationale", path=path, line_number=line_number),
                surrounding_code=_required_non_empty_str(payload, "surrounding_code", path=path, line_number=line_number),
                expected_verdict=expected_verdict,
                expected_should_use_decorator=expected_should_use_decorator,
            )
        )
    return tuple(cases)


def evaluate_judge_quality_corpus(
    *,
    cases: Sequence[JudgeQualityCase],
    corpus_path: Path,
    min_accuracy: float = DEFAULT_JUDGE_QUALITY_MIN_ACCURACY,
    min_cases: int = JUDGE_QUALITY_MIN_CASES,
    max_cases: int = JUDGE_QUALITY_MAX_CASES,
    model_id: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    judge_caller: JudgeCaller | None = None,
) -> JudgeQualityReport:
    """Call the judge for every case and return the scored report."""
    _validate_evaluation_config(
        case_count=len(cases),
        min_accuracy=min_accuracy,
        min_cases=min_cases,
        max_cases=max_cases,
        max_tokens=max_tokens,
    )

    def default_judge_caller(request: JudgeRequest) -> JudgeResponse:
        return call_judge(request, model_id=model_id, max_tokens=max_tokens)

    caller = judge_caller or default_judge_caller
    results = tuple(JudgeQualityCaseResult(case=case, response=caller(case.to_request())) for case in cases)
    return JudgeQualityReport(
        corpus_path=corpus_path,
        requested_model_id=model_id,
        min_accuracy=min_accuracy,
        results=results,
    )


def render_judge_quality_report_text(report: JudgeQualityReport) -> str:
    """Render an operator-readable quality report."""
    lines = [
        (
            "check-judge-quality: "
            f"corpus={report.corpus_path}, "
            f"requested_model={report.requested_model_id}, "
            f"total={report.total_count}, "
            f"passed={report.passed_count}, "
            f"failed={report.failed_count}, "
            f"accuracy={report.accuracy * 100.0:.2f}% "
            f"(min {report.min_accuracy * 100.0:.2f}%)"
        )
    ]
    if report.passes:
        lines.append("PASS: judge quality corpus meets threshold.")
        return "\n".join(lines) + "\n"

    lines.append("FAIL: judge quality corpus is below threshold.")
    if report.failures:
        lines.append("")
        lines.append("Mismatched cases:")
        for result in report.failures:
            lines.append(
                "  "
                f"{result.case.case_id}: "
                f"expected={result.case.expected_verdict.value}/"
                f"{_decorator_label(result.case.expected_should_use_decorator)}, "
                f"actual={result.response.verdict.value}/"
                f"{_decorator_label(result.response.should_use_decorator)}, "
                f"served_model={result.response.model_id}, "
                f"confidence={result.response.confidence:.2f}"
            )
            lines.append(f"    judge_rationale={_shorten(result.response.judge_rationale, limit=220)}")
    return "\n".join(lines) + "\n"


def render_judge_quality_report_json(report: JudgeQualityReport) -> str:
    """Render a machine-readable quality report."""
    payload = {
        "accuracy": report.accuracy,
        "corpus_path": report.corpus_path.as_posix(),
        "failed": report.failed_count,
        "min_accuracy": report.min_accuracy,
        "passed": report.passes,
        "passed_count": report.passed_count,
        "requested_model": report.requested_model_id,
        "total": report.total_count,
        "cases": [
            {
                "actual_should_use_decorator": result.response.should_use_decorator,
                "actual_verdict": result.response.verdict.value,
                "case_id": result.case.case_id,
                "confidence": result.response.confidence,
                "expected_should_use_decorator": result.case.expected_should_use_decorator,
                "expected_verdict": result.case.expected_verdict.value,
                "judge_rationale": result.response.judge_rationale,
                "passed": result.passed,
                "policy_hash": result.response.policy_hash,
                "prompt_tokens_cached": result.response.prompt_tokens_cached,
                "prompt_tokens_total": result.response.prompt_tokens_total,
                "recorded_at": result.response.recorded_at.isoformat(),
                "served_model": result.response.model_id,
            }
            for result in report.results
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _validate_evaluation_config(
    *,
    case_count: int,
    min_accuracy: float,
    min_cases: int,
    max_cases: int,
    max_tokens: int,
) -> None:
    if min_cases <= 0:
        raise JudgeQualityError(f"min_cases must be positive; got {min_cases}")
    if max_cases < min_cases:
        raise JudgeQualityError(f"max_cases must be >= min_cases; got min={min_cases}, max={max_cases}")
    if not 0.0 <= min_accuracy <= 1.0:
        raise JudgeQualityError(f"min_accuracy must be in [0.0, 1.0]; got {min_accuracy}")
    if max_tokens <= 0:
        raise JudgeQualityError(f"max_tokens must be positive; got {max_tokens}")
    if case_count < min_cases or case_count > max_cases:
        raise JudgeQualityError(f"corpus must contain between {min_cases} and {max_cases} cases; got {case_count}")


def _required_non_empty_str(payload: dict[str, Any], key: str, *, path: Path, line_number: int) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise JudgeQualityError(f"{path}:{line_number}: {key} must be a non-empty string")
    return value


def _expected_verdict(payload: dict[str, Any], *, path: Path, line_number: int) -> JudgeVerdict:
    value = payload["expected_verdict"]
    if not isinstance(value, str):
        raise JudgeQualityError(f"{path}:{line_number}: expected_verdict must be a string")
    try:
        verdict = JudgeVerdict(value)
    except ValueError as exc:
        raise JudgeQualityError(f"{path}:{line_number}: unknown expected_verdict {value!r}") from exc
    if verdict is JudgeVerdict.OVERRIDDEN_BY_OPERATOR:
        raise JudgeQualityError(f"{path}:{line_number}: expected_verdict cannot be OVERRIDDEN_BY_OPERATOR")
    return verdict


def _expected_decorator(payload: dict[str, Any], *, path: Path, line_number: int) -> str | None:
    value = payload["expected_should_use_decorator"]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise JudgeQualityError(f"{path}:{line_number}: expected_should_use_decorator must be a non-empty string or null")
    return value


def _decorator_label(value: str | None) -> str:
    return "null" if value is None else value


def _shorten(value: str, *, limit: int) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."
