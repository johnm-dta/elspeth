"""Integration tests for the judge ↔ ``@trust_boundary`` decorator nudge.

These exercise the structured ``should_use_decorator`` signal that the
judge emits when the proposed allowlist entry is a textbook Tier-3
boundary case that ``@trust_boundary`` covers structurally. The signal
is surfaced two ways:

* the JSON output exposes it as a top-level ``should_use_decorator``
  field so tooling can branch on it;
* the text output extends the BLOCKED message with a human-readable
  recommendation that names the decorator and the ``source_param``.

The Anthropic SDK is mocked at the ``anthropic.Anthropic`` client level
so the tests run offline; the model-response contract is exercised
end-to-end through ``call_judge`` and the CLI's ``_run_justify`` /
``_emit_justify_output`` paths.

See ``notes/cicd-judge-cli-prototype-plan.md`` ("How the two pillars
interact") for the design rationale: the judge's structured nudge is
what enforces decorator adoption — without it, the decorator's reach
stays voluntary and lags.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth_lints.core.allowlist import JudgeVerdict
from elspeth_lints.core.cli import main
from elspeth_lints.core.judge import JudgeRequest, call_judge

# Small synthetic source: an R1 finding inside a function whose external-data
# parameter is named ``arguments`` — the canonical decorator-suggestion case
# from the prototype plan (Pillar B sketch). The model's
# ``should_use_decorator: "arguments"`` response should fire here.
_SYNTHETIC_BOUNDARY_SOURCE = '''\
"""Synthetic boundary module used in judge-decorator integration tests."""


class ToolExecutor:
    def _execute_set_pipeline(self, arguments: dict) -> str:
        # R1: arguments comes from an LLM tool call (Tier-3 external).
        nodes = arguments.get("nodes", [])
        return str(nodes)
'''


# ---------- helpers ----------


def _build_source_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Lay out a minimal source root with one boundary-shaped finding."""
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    target = root / "plugins" / "tool_executor.py"
    target.write_text(_SYNTHETIC_BOUNDARY_SOURCE, encoding="utf-8")
    return root, target


def _build_allowlist_dir(tmp_path: Path) -> Path:
    """Lay out an empty per-module allowlist directory."""
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "_defaults.yaml").write_text(
        "version: 1\ndefaults:\n  fail_on_stale: false\n  fail_on_expired: false\n",
        encoding="utf-8",
    )
    return allowlist_dir


def _mock_openrouter_completion(
    *,
    verdict: str,
    rationale: str,
    should_use_decorator: Any = None,
) -> MagicMock:
    """Build a mock OpenAI-SDK chat-completion result.

    Mirrors ``test_justify._mock_openrouter_completion``; duplicated
    here so this test file is self-contained and can drive
    ``should_use_decorator`` independently. The judge routes through
    OpenRouter via the OpenAI-compatible SDK, so the mock matches the
    chat-completions shape (``.choices[0].message.content`` is a JSON
    string; ``.usage.prompt_tokens`` + optional
    ``.prompt_tokens_details.cached_tokens`` carry token accounting).
    """
    message = MagicMock()
    message.content = json.dumps(
        {
            "verdict": verdict,
            "rationale": rationale,
            "should_use_decorator": should_use_decorator,
        }
    )
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    # Explicit ``completion.model`` is required now that ``call_judge``
    # records the served model id (per C1-1, elspeth-0e1d0978fa). Without
    # this, MagicMock auto-attributes a non-JSON-serialisable Mock object
    # to ``completion.model``, which then flows into JudgeResponse.model_id
    # and breaks JSON serialisation in --format json tests.
    completion.model = "anthropic/claude-opus-4"
    completion.usage = MagicMock(
        prompt_tokens=4000,
        prompt_tokens_details=MagicMock(cached_tokens=0),
    )
    return completion


@contextmanager
def _mock_judge_call(
    *,
    verdict: str,
    rationale: str,
    should_use_decorator: Any = None,
) -> Iterator[MagicMock]:
    """Patch ``openai.OpenAI`` to a fake client emitting one completion."""
    fake_completion = _mock_openrouter_completion(
        verdict=verdict,
        rationale=rationale,
        should_use_decorator=should_use_decorator,
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_completion
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=fake_client) as client_class,
    ):
        yield client_class


# ---------- call_judge: parsing ----------


def test_call_judge_parses_should_use_decorator_with_blocked_verdict() -> None:
    """The model can populate ``should_use_decorator`` alongside BLOCKED."""
    request = JudgeRequest(
        file_path="plugins/tool_executor.py",
        rule_id="R1",
        symbol="ToolExecutor._execute_set_pipeline",
        fingerprint="abc",
        rationale="external tool-call arguments",
        surrounding_code="    nodes = arguments.get('nodes', [])",
    )
    with _mock_judge_call(
        verdict="BLOCKED",
        rationale="this is a Tier-3 boundary; use @trust_boundary on the function",
        should_use_decorator="arguments",
    ):
        response = call_judge(request)
    assert response.verdict is JudgeVerdict.BLOCKED
    assert response.should_use_decorator == "arguments"


def test_call_judge_accepts_null_should_use_decorator_with_blocked_verdict() -> None:
    """Plain BLOCKED without a decorator nudge is still valid."""
    request = JudgeRequest(
        file_path="plugins/tool_executor.py",
        rule_id="R1",
        symbol="ToolExecutor._execute_set_pipeline",
        fingerprint="abc",
        rationale="shallow rationale",
        surrounding_code="    nodes = arguments.get('nodes', [])",
    )
    with _mock_judge_call(
        verdict="BLOCKED",
        rationale="rationale was shallow; fix the code",
        should_use_decorator=None,
    ):
        response = call_judge(request)
    assert response.verdict is JudgeVerdict.BLOCKED
    assert response.should_use_decorator is None


def test_call_judge_crashes_on_accepted_with_decorator_suggestion() -> None:
    """ACCEPTED + non-null should_use_decorator is incoherent — crash.

    The contract: should_use_decorator is the structured alternative to
    landing the proposed allowlist entry. Pairing it with ACCEPTED would
    mean the judge is simultaneously saying "land the entry" and "use the
    decorator instead" — silently picking one would erode the audit
    primitive, so we refuse the response per the project's
    offensive-programming policy.
    """
    request = JudgeRequest(
        file_path="plugins/tool_executor.py",
        rule_id="R1",
        symbol="ToolExecutor._execute_set_pipeline",
        fingerprint="abc",
        rationale="...",
        surrounding_code="...",
    )
    with (
        _mock_judge_call(
            verdict="ACCEPTED",
            rationale="boundary is fine",
            should_use_decorator="arguments",
        ),
        pytest.raises(RuntimeError, match="should_use_decorator"),
    ):
        call_judge(request)


# ---------- CLI: BLOCKED + decorator nudge (text output) ----------


def test_justify_blocked_with_decorator_text_output_names_decorator(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Text-format BLOCKED-with-decorator surfaces source_param and decorator path."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/tool_executor.py",
        "--symbol",
        "ToolExecutor._execute_set_pipeline",
        "--rationale",
        "arguments comes from the composer tool call",
        "--owner",
        "test-agent",
    ]
    with _mock_judge_call(
        verdict="BLOCKED",
        rationale="this is a structural Tier-3 boundary; use @trust_boundary(...)",
        should_use_decorator="arguments",
    ):
        exit_code = main(argv)

    assert exit_code == 1
    target_yaml = allowlist_dir / "plugins.yaml"
    assert not target_yaml.exists()
    out = capsys.readouterr().out
    assert "BLOCKED" in out
    assert "@trust_boundary" in out
    assert "source_param='arguments'" in out
    assert "src/elspeth/contracts/trust_boundary.py" in out


def test_justify_blocked_without_decorator_text_output_lacks_decorator_mention(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Plain BLOCKED keeps the original "agent figures out remediation" copy."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/tool_executor.py",
        "--symbol",
        "ToolExecutor._execute_set_pipeline",
        "--rationale",
        "...",
        "--owner",
        "test-agent",
    ]
    with _mock_judge_call(
        verdict="BLOCKED",
        rationale="rationale was shallow; remediation is the agent's job",
        should_use_decorator=None,
    ):
        exit_code = main(argv)

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "BLOCKED" in out
    assert "@trust_boundary" not in out
    assert "source_param" not in out
    assert "agent's responsibility" in out


def test_justify_accepted_text_output_has_no_decorator_nudge(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ACCEPTED runs never mention the decorator — the entry lands as written."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/tool_executor.py",
        "--symbol",
        "ToolExecutor._execute_set_pipeline",
        "--rationale",
        "Tier-3 boundary, decorator already in place upstream",
        "--owner",
        "test-agent",
    ]
    with _mock_judge_call(
        verdict="ACCEPTED",
        rationale="judge agrees with the rationale",
        should_use_decorator=None,
    ):
        exit_code = main(argv)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "ACCEPTED" in out
    assert "@trust_boundary" not in out


# ---------- CLI: JSON output exposes the structured signal ----------


def test_justify_json_output_includes_should_use_decorator_field(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """JSON output exposes ``should_use_decorator`` as a top-level field."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/tool_executor.py",
        "--symbol",
        "ToolExecutor._execute_set_pipeline",
        "--rationale",
        "arguments is a tool-call payload",
        "--owner",
        "test-agent",
        "--format",
        "json",
    ]
    with _mock_judge_call(
        verdict="BLOCKED",
        rationale="use @trust_boundary instead",
        should_use_decorator="arguments",
    ):
        exit_code = main(argv)

    assert exit_code == 1
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["verdict"] == "BLOCKED"
    assert payload["should_use_decorator"] == "arguments"
    assert payload["blocked"] is True
    assert payload["wrote"] is False


def test_justify_json_output_null_should_use_decorator_when_absent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """JSON output emits ``should_use_decorator: null`` on ordinary verdicts."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/tool_executor.py",
        "--symbol",
        "ToolExecutor._execute_set_pipeline",
        "--rationale",
        "Tier-3 boundary, decorator already present upstream",
        "--owner",
        "test-agent",
        "--format",
        "json",
    ]
    with _mock_judge_call(
        verdict="ACCEPTED",
        rationale="judge agrees",
        should_use_decorator=None,
    ):
        exit_code = main(argv)

    assert exit_code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["verdict"] == "ACCEPTED"
    assert payload["should_use_decorator"] is None


# ---------- CLI: operator override preserves the audit signal ----------


def test_justify_operator_override_past_decorator_suggestion_records_both_signals(
    tmp_path: Path,
) -> None:
    """Override past a should_use_decorator BLOCK preserves the suggestion in rationale.

    The operator override does NOT change the model's verdict or its
    rationale text — both are recorded verbatim. The audit trail therefore
    shows that the operator pushed past a "use the decorator" recommendation
    specifically, not a plain BLOCK. The structured ``should_use_decorator``
    field is not written into the YAML (per the design decision: the audit
    primitive is the rationale text, not a separate field), but the
    rationale text contains the model's "use @trust_boundary..." sentence
    and ``judge_model_verdict=BLOCKED`` records the divergence.
    """
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/tool_executor.py",
        "--symbol",
        "ToolExecutor._execute_set_pipeline",
        "--rationale",
        "shipping under deadline; decorator refactor too risky now",
        "--owner",
        "operator-jdoe",
        "--operator-override",
    ]
    model_rationale = (
        "use @trust_boundary(tier=3, source='LLM tool-call arguments', "
        "source_param='arguments', suppresses=('R1',), invariant='...') "
        "on _execute_set_pipeline and delete this entry."
    )
    with _mock_judge_call(
        verdict="BLOCKED",
        rationale=model_rationale,
        should_use_decorator="arguments",
    ):
        exit_code = main(argv)

    assert exit_code == 0
    target_yaml = allowlist_dir / "plugins.yaml"
    text = target_yaml.read_text(encoding="utf-8")
    assert "judge_verdict: OVERRIDDEN_BY_OPERATOR" in text
    assert "judge_model_verdict: BLOCKED" in text
    # The decorator suggestion sentence is preserved verbatim in
    # judge_rationale — that is the audit signal that the operator pushed
    # past a structured decorator nudge specifically.
    assert "use @trust_boundary" in text
    assert "source_param='arguments'" in text
