"""Unit tests for the ``elspeth-lints justify`` subcommand.

These exercise the judge-gated allowlist-write path. The OpenAI SDK
(pointed at OpenRouter) is mocked at the ``openai.OpenAI`` client level
so the tests run offline; the model-response contract is exercised
end-to-end (JSON shape, verdict parsing, allowlist round-trip) without
making a network call.

The tests deliberately avoid round-tripping through ``yaml.safe_load``
on the written entry because the production write path is text-level;
they instead assert against the rendered YAML text and (separately)
re-read the file via the production loader to confirm parser-side
round-trip parity.
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

from elspeth_lints.core.allowlist import JudgeVerdict, load_allowlist
from elspeth_lints.core.cli import main
from elspeth_lints.core.judge import (
    JudgeConfigurationError,
    JudgeRequest,
    call_judge,
)

# A small synthetic source file that produces exactly one R1 finding
# (`dict.get` on data that is not at a Tier-3 boundary). The exact text
# matters less than the fact that the tier_model rule reports a finding
# at a stable symbol_context.
_SYNTHETIC_SOURCE = '''\
"""Synthetic module used in justify tests."""


class Widget:
    def lookup(self, payload: dict) -> str:
        # R1: dict.get on Tier-2 data — the kind of finding an agent
        # might want to suppress with judge approval.
        return payload.get("name", "anonymous")
'''


# ---------- helpers ----------


def _build_source_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Lay out a minimal source root with one finding-producing file.

    Returns (root_dir, target_file). The root mimics the production
    ``src/elspeth`` layout so the tier_model scanner classifies the file
    as part of L3 (plugins-equivalent).
    """
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    target = root / "plugins" / "widget.py"
    target.write_text(_SYNTHETIC_SOURCE, encoding="utf-8")
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
    prompt_tokens: int = 4000,
    cached_tokens: int | None = 0,
    served_model: str | None = "anthropic/claude-opus-4",
) -> MagicMock:
    """Build a mock OpenAI-SDK ``chat.completions.create`` return value.

    The judge routes through OpenRouter via the OpenAI SDK, so the mock
    mirrors the chat-completions shape: ``.choices[0].message.content``
    is a JSON string the judge will parse, and ``.usage`` carries the
    prompt-token totals plus the (OpenAI-shaped, OpenRouter-forwarded)
    ``prompt_tokens_details.cached_tokens`` field.

    ``cached_tokens=None`` simulates a provider that didn't report the
    cached count (caching off, or transport omitted it). ``0`` simulates
    caching-on with no hit; a positive int simulates a cache hit.
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
    # Explicitly set ``completion.model`` so existing happy-path tests
    # (which assert ``judge_model: anthropic/claude-opus-4`` survives
    # the YAML round-trip) keep passing now that ``call_judge`` records
    # the SERVED model id (not the requested one). Tests that need to
    # exercise routing-divergence or absent-served-id paths pass
    # ``served_model=`` explicitly. See the C1-1 (elspeth-0e1d0978fa)
    # tests below.
    completion.model = served_model
    # Usage shape: total + optional details. cached_tokens=None means
    # the field on details is absent (we model this by setting details
    # to None directly, which judge._extract_cache_accounting treats as
    # "no count reported").
    if cached_tokens is None:
        completion.usage = MagicMock(
            prompt_tokens=prompt_tokens,
            prompt_tokens_details=None,
        )
    else:
        details = MagicMock(cached_tokens=cached_tokens)
        completion.usage = MagicMock(
            prompt_tokens=prompt_tokens,
            prompt_tokens_details=details,
        )
    return completion


@contextmanager
def _mock_judge_call(
    *,
    verdict: str,
    rationale: str,
    prompt_tokens: int = 4000,
    cached_tokens: int | None = 0,
    served_model: str | None = "anthropic/claude-opus-4",
) -> Iterator[MagicMock]:
    """Patch ``openai.OpenAI`` so tests run offline.

    Yields the patched client class so callers can introspect how it was
    invoked (e.g. assert on the prompt the judge would have received and
    on the cache_control marker on the system block).
    """
    fake_completion = _mock_openrouter_completion(
        verdict=verdict,
        rationale=rationale,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        served_model=served_model,
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_completion
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=fake_client) as client_class,
    ):
        yield client_class


# ---------- call_judge contract ----------


def test_call_judge_returns_accepted_for_well_formed_response() -> None:
    request = JudgeRequest(
        file_path="plugins/widget.py",
        rule_id="R1",
        symbol="Widget.lookup",
        fingerprint="abc",
        rationale="dict carries Tier-3 external payload",
        surrounding_code="    return payload.get('name', 'anonymous')",
    )
    with _mock_judge_call(verdict="ACCEPTED", rationale="boundary is genuine"):
        response = call_judge(request)
    assert response.verdict is JudgeVerdict.ACCEPTED
    assert response.judge_rationale == "boundary is genuine"
    assert response.should_use_decorator is None
    assert response.recorded_at.tzinfo is not None


def test_call_judge_raises_configuration_error_when_api_key_absent() -> None:
    request = JudgeRequest(
        file_path="plugins/widget.py",
        rule_id="R1",
        symbol="Widget.lookup",
        fingerprint="abc",
        rationale="...",
        surrounding_code="...",
    )
    # Strip the key out of the environment for this call. The judge
    # routes through OpenRouter, so the gate is OPENROUTER_API_KEY.
    env_without_key = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True), pytest.raises(JudgeConfigurationError, match="OPENROUTER_API_KEY"):
        call_judge(request)


def test_call_judge_crashes_on_malformed_json() -> None:
    """Per the offensive-programming policy, a malformed judge response is fatal."""
    request = JudgeRequest(
        file_path="plugins/widget.py",
        rule_id="R1",
        symbol="Widget.lookup",
        fingerprint="abc",
        rationale="...",
        surrounding_code="...",
    )
    # OpenAI-shape mock with non-JSON content.
    bad_message = MagicMock()
    bad_message.content = "not json at all { ::: }"
    bad_choice = MagicMock()
    bad_choice.message = bad_message
    bad_completion = MagicMock()
    bad_completion.choices = [bad_choice]
    bad_completion.usage = MagicMock(prompt_tokens=100, prompt_tokens_details=None)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = bad_completion
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=fake_client),
        pytest.raises(RuntimeError, match="non-JSON response"),
    ):
        call_judge(request)


def test_call_judge_crashes_on_unknown_verdict_string() -> None:
    """The model is not allowed to emit OVERRIDDEN_BY_OPERATOR — only the CLI does."""
    request = JudgeRequest(
        file_path="plugins/widget.py",
        rule_id="R1",
        symbol="Widget.lookup",
        fingerprint="abc",
        rationale="...",
        surrounding_code="...",
    )
    with _mock_judge_call(verdict="MAYBE", rationale="hedging"), pytest.raises(RuntimeError, match="ACCEPTED or BLOCKED"):
        call_judge(request)


# ---------- CLI: accepted path ----------


def test_justify_accepted_writes_entry_with_judge_metadata(tmp_path: Path) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "payload is Tier-3 external data from upstream tool-call",
        "--owner",
        "test-agent-accepted",
        "--format",
        "json",
    ]
    with _mock_judge_call(verdict="ACCEPTED", rationale="genuine Tier-3 boundary"):
        exit_code = main(argv)

    assert exit_code == 0
    target_yaml = allowlist_dir / "plugins.yaml"
    assert target_yaml.exists()
    text = target_yaml.read_text(encoding="utf-8")
    assert "judge_verdict: ACCEPTED" in text
    assert "judge_model: anthropic/claude-opus-4" in text
    assert "genuine Tier-3 boundary" in text
    assert "plugins/widget.py:R1:Widget:lookup:fp=" in text
    # B3: owner is recorded verbatim from --owner, not fabricated from $USER.
    assert "owner: test-agent-accepted" in text

    # Round-trip parity: the production loader can read the entry back
    # and the judge metadata survives the YAML parse.
    allowlist = load_allowlist(target_yaml, valid_rule_ids={"R1"})
    assert len(allowlist.entries) == 1
    entry = allowlist.entries[0]
    assert entry.judge_verdict is JudgeVerdict.ACCEPTED
    assert entry.judge_model == "anthropic/claude-opus-4"
    assert entry.judge_rationale == "genuine Tier-3 boundary"
    assert entry.judge_recorded_at is not None
    assert entry.judge_recorded_at.tzinfo is not None


# ---------- CLI: blocked path ----------


def test_justify_blocked_does_not_write_and_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "I just don't want to fix this",
        "--owner",
        "lazy-agent",
    ]
    with _mock_judge_call(verdict="BLOCKED", rationale="rationale is shallow; fix the code"):
        exit_code = main(argv)

    assert exit_code == 1
    target_yaml = allowlist_dir / "plugins.yaml"
    assert not target_yaml.exists()
    captured = capsys.readouterr()
    assert "BLOCKED" in captured.out
    assert "rationale is shallow" in captured.out


# ---------- CLI: operator override ----------


def test_justify_operator_override_records_override_with_model_rationale(tmp_path: Path) -> None:
    """Override sets the *entry*'s verdict but preserves the model's actual rationale and verdict.

    The schema captures: judge_verdict (now OVERRIDDEN_BY_OPERATOR),
    judge_model_verdict (the model's actual verdict — typically
    BLOCKED), and judge_rationale (the model's verbatim text). The
    triple asymmetry is the audit signal: a row with
    judge_verdict=OVERRIDDEN_BY_OPERATOR + judge_model_verdict=BLOCKED
    + judge_rationale containing "fix the code" makes plain that the
    operator pushed past a BLOCK. Without judge_model_verdict, a
    downstream aggregator would have to parse rationale text to tell
    overrides-of-BLOCKED apart from overrides-of-ACCEPTED.
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
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "shipping under deadline",
        "--owner",
        "operator-jdoe",
        "--operator-override",
    ]
    with _mock_judge_call(verdict="BLOCKED", rationale="model says: this should be fixed in code"):
        exit_code = main(argv)

    assert exit_code == 0
    target_yaml = allowlist_dir / "plugins.yaml"
    text = target_yaml.read_text(encoding="utf-8")
    assert "judge_verdict: OVERRIDDEN_BY_OPERATOR" in text
    assert "judge_model_verdict: BLOCKED" in text  # model's verdict preserved alongside override
    assert "model says: this should be fixed in code" in text  # model's rationale preserved

    # Verify round-trip: the loaded entry exposes both verdicts as enum members.
    entries = load_allowlist(allowlist_dir, valid_rule_ids={"trust_tier.tier_model"})
    overridden = [e for e in entries.entries if e.judge_verdict == JudgeVerdict.OVERRIDDEN_BY_OPERATOR]
    assert len(overridden) == 1
    assert overridden[0].judge_model_verdict == JudgeVerdict.BLOCKED


def test_justify_non_override_records_judge_model_verdict_as_none(tmp_path: Path) -> None:
    """Non-override ACCEPTED entries don't write judge_model_verdict.

    Per the fabrication-decision test in CLAUDE.md: when the model's
    verdict and the entry's verdict agree, duplicating the model's
    verdict into a separate field would synthesise a divergence signal
    that doesn't exist. None / field-absent is the honest representation.
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
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "genuine Tier-3 boundary",
        "--owner",
        "test-agent",
    ]
    with _mock_judge_call(verdict="ACCEPTED", rationale="judge agrees"):
        exit_code = main(argv)

    assert exit_code == 0
    text = (allowlist_dir / "plugins.yaml").read_text(encoding="utf-8")
    assert "judge_verdict: ACCEPTED" in text
    assert "judge_model_verdict:" not in text  # absence is the signal: no divergence

    entries = load_allowlist(allowlist_dir, valid_rule_ids={"trust_tier.tier_model"})
    accepted = [e for e in entries.entries if e.judge_verdict == JudgeVerdict.ACCEPTED]
    assert len(accepted) == 1
    assert accepted[0].judge_model_verdict is None


# ---------- CLI: dry-run ----------


@pytest.mark.parametrize("verdict_str", ["ACCEPTED", "BLOCKED"])
def test_justify_dry_run_never_writes(tmp_path: Path, verdict_str: str) -> None:
    """--dry-run is a hard non-write guarantee, irrespective of verdict.

    For BLOCKED the exit code is still 1 (the gate decision stands); for
    ACCEPTED the exit code is 0. Neither writes to disk.
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
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "...",
        "--owner",
        "test-agent",
        "--dry-run",
    ]
    with _mock_judge_call(verdict=verdict_str, rationale="judge said something"):
        exit_code = main(argv)

    expected_exit = 0 if verdict_str == "ACCEPTED" else 1
    assert exit_code == expected_exit
    target_yaml = allowlist_dir / "plugins.yaml"
    assert not target_yaml.exists()


# ---------- CLI: missing API key ----------


def test_justify_missing_api_key_emits_configuration_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "...",
        "--owner",
        "test-agent",
    ]
    env_without_key = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        exit_code = main(argv)

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "OPENROUTER_API_KEY" in captured.err


# ---------- CLI: ambiguous symbol ----------


def test_justify_ambiguous_symbol_errors_before_calling_judge(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Two findings at the same symbol_context must error out without calling the judge.

    The synthetic source below produces two R1 findings inside the same
    method (two separate ``dict.get`` calls). The judge gate refuses to
    pick one arbitrarily — the operator must narrow the symbol path or
    run ``elspeth-lints rotate`` first if these are stale fingerprints.
    """
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    target = root / "plugins" / "widget.py"
    target.write_text(
        """\
class Widget:
    def lookup(self, a: dict, b: dict) -> tuple[str, str]:
        # Two R1 findings on the same symbol_context — the judge gate
        # needs them disambiguated before it will gate one of them.
        return a.get("x", ""), b.get("y", "")
""",
        encoding="utf-8",
    )
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "Tier-3 boundary",
        "--owner",
        "test-agent",
    ]
    # Set the API key so we definitely don't fall out via the
    # configuration check — the ambiguity error must fire first.
    judge_called = MagicMock()
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=False), patch("openai.OpenAI", judge_called):
        exit_code = main(argv)

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Ambiguous" in captured.err
    judge_called.assert_not_called()


# ---------- CLI: YAML round-trip preserves judge metadata ----------


def test_justify_round_trip_preserves_judge_metadata_across_reads(tmp_path: Path) -> None:
    """After writing, re-loading and re-writing yields the same effective entry.

    We don't byte-compare (the writer is hand-rolled, not yaml.dump) —
    we compare the parsed dataclass to make sure the JudgeVerdict enum,
    the UTC-aware timestamp, the model id, and the rationale all survive
    the round trip via the production loader.
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
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "tier-3 boundary; payload comes from external tool-call",
        "--owner",
        "round-trip-agent",
    ]
    with _mock_judge_call(verdict="ACCEPTED", rationale="judge's verbatim reasoning"):
        exit_code = main(argv)
    assert exit_code == 0

    target_yaml = allowlist_dir / "plugins.yaml"
    first_load = load_allowlist(target_yaml, valid_rule_ids={"R1"})
    assert len(first_load.entries) == 1
    first = first_load.entries[0]

    # Second load (re-parse the same on-disk YAML) should yield an
    # equivalent dataclass — the YAML parser is idempotent on this file.
    second_load = load_allowlist(target_yaml, valid_rule_ids={"R1"})
    second = second_load.entries[0]

    assert first.judge_verdict == second.judge_verdict == JudgeVerdict.ACCEPTED
    assert first.judge_model == second.judge_model == "anthropic/claude-opus-4"
    assert first.judge_rationale == second.judge_rationale == "judge's verbatim reasoning"
    assert first.judge_recorded_at == second.judge_recorded_at
    assert first.judge_recorded_at is not None
    assert first.judge_recorded_at.tzinfo is not None


# =============================================================================
# Regression: B3 — --owner is required, validated, and recorded verbatim
# =============================================================================
#
# Before the fix, ``_run_justify`` derived the entry's ``owner`` field from
# ``os.environ.get("USER", "agent")``. That is fabrication of audit
# attribution: the audit trail recorded whichever shell user happened to
# launch the process, NOT who took responsibility for the suppression.
# If $USER was unset the literal string "agent" was recorded as the
# owner — even more obviously synthetic.
#
# The fix makes ``--owner`` a required CLI argument, rejects empty /
# whitespace-only values via argparse type-callable, and records the
# operator-supplied value verbatim in the YAML entry.
# =============================================================================


def test_justify_requires_owner_argument(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Omitting --owner causes argparse to exit non-zero before the judge runs.

    argparse uses ``SystemExit(2)`` for "required argument missing", and
    the error message names the missing flag. The judge must not be
    called, so we don't need to mock the API key.
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
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "tier-3 boundary",
        # --owner deliberately omitted
    ]
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--owner" in captured.err


@pytest.mark.parametrize("owner_value", ["", "   ", "\t\t", "\n"])
def test_justify_rejects_empty_owner(tmp_path: Path, capsys: pytest.CaptureFixture[str], owner_value: str) -> None:
    """Empty or whitespace-only --owner is rejected by the argparse type callable.

    The audit signal is the named identity that claimed responsibility;
    an empty owner string is no signal at all. argparse raises
    ``ArgumentTypeError`` from the type callable, which it surfaces as
    SystemExit(2) with a descriptive message.
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
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "tier-3 boundary",
        "--owner",
        owner_value,
    ]
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    # argparse's standard frame is "argument --owner: <our message>"
    assert "--owner" in captured.err
    assert "non-empty" in captured.err or "audit identity" in captured.err


# =============================================================================
# Prompt-caching contract: static policy block carries cache_control;
# dynamic per-call material does not.
# =============================================================================
#
# The judge's system prompt is now structured as a cacheable static
# policy block (CLAUDE.md excerpts, the @trust_boundary teaching, the
# output schema, the decision heuristic) plus a per-call user message
# (file path, rationale, surrounding code). The static block is wrapped
# in ``cache_control: {"type": "ephemeral"}`` so the OpenRouter ->
# Anthropic transport will cache it for the 5-minute TTL window. These
# tests pin the structural contract: the first system block must carry
# the cache marker; the user block must NOT carry it; and the static
# block must contain the load-bearing policy phrases the judge needs
# (so future edits don't accidentally drop a section without breaking a
# test).
# =============================================================================


def test_call_judge_system_block_is_cached_and_user_block_is_not(tmp_path: Path) -> None:
    request = JudgeRequest(
        file_path="plugins/widget.py",
        rule_id="R1",
        symbol="Widget.lookup",
        fingerprint="fp-cache-test",
        rationale="payload is Tier-3 external data",
        surrounding_code="return payload.get('name', 'anonymous')",
    )
    with _mock_judge_call(verdict="ACCEPTED", rationale="boundary is genuine") as client_class:
        call_judge(request)

    # Reach into the underlying mock client instance to see the call
    # kwargs. patch("openai.OpenAI", return_value=fake_client) means
    # client_class.return_value is the fake_client.
    fake_client = client_class.return_value
    create_call = fake_client.chat.completions.create.call_args
    messages = create_call.kwargs["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 2

    # System message: list-of-blocks shape, single text block, with
    # cache_control: ephemeral.
    system_msg = messages[0]
    assert system_msg["role"] == "system"
    system_blocks = system_msg["content"]
    assert isinstance(system_blocks, list)
    assert len(system_blocks) == 1
    sys_block = system_blocks[0]
    assert sys_block["type"] == "text"
    assert sys_block["cache_control"] == {"type": "ephemeral"}

    # User message: also list-of-blocks shape, but no cache_control.
    user_msg = messages[1]
    assert user_msg["role"] == "user"
    user_blocks = user_msg["content"]
    assert isinstance(user_blocks, list)
    assert len(user_blocks) == 1
    user_block = user_blocks[0]
    assert user_block["type"] == "text"
    assert "cache_control" not in user_block


def test_call_judge_static_policy_contains_loadbearing_phrases(tmp_path: Path) -> None:
    """The cached block must contain the policy vocabulary the judge needs.

    These phrases are excerpted verbatim from CLAUDE.md and bind back to
    the Decision Heuristic at the end of the prompt. If a refactor
    accidentally drops one of these sections, the heuristic loses its
    referent and the verdict quality degrades.
    """
    request = JudgeRequest(
        file_path="plugins/widget.py",
        rule_id="R1",
        symbol="Widget.lookup",
        fingerprint="fp-phrases",
        rationale="...",
        surrounding_code="...",
    )
    with _mock_judge_call(verdict="ACCEPTED", rationale="ok") as client_class:
        call_judge(request)

    fake_client = client_class.return_value
    create_call = fake_client.chat.completions.create.call_args
    sys_text = create_call.kwargs["messages"][0]["content"][0]["text"]

    # Tier-model vocabulary
    assert "Tier 1: Our Data" in sys_text
    assert "Tier 2: Pipeline Data" in sys_text
    assert "Tier 3: External Data" in sys_text
    assert "FULL TRUST" in sys_text
    assert "ZERO TRUST" in sys_text

    # Fabrication-decision test (load-bearing for §6 of the heuristic)
    assert "fabrication-decision test" in sys_text

    # Defensive vs offensive (the heading is the canonical phrase)
    assert "Defensive Programming: Forbidden" in sys_text
    assert "Offensive Programming: Encouraged" in sys_text

    # No legacy policy and layer rules — both are heuristic referents
    assert "No Legacy Code" in sys_text
    assert "Layer Dependency Rules" in sys_text

    # The decorator-teaching section is preserved (load-bearing for
    # should_use_decorator output contract)
    assert "@trust_boundary" in sys_text
    assert "should_use_decorator" in sys_text

    # Output schema and decision heuristic
    assert "Output schema" in sys_text
    assert "Decision Heuristic" in sys_text


def test_call_judge_user_block_contains_per_call_material(tmp_path: Path) -> None:
    """Dynamic material (file path, rationale, code) must be in the user message.

    The per-call material must NOT be in the system block (that would
    bust the cache key and defeat the optimisation). The user message
    must carry the substituted template.
    """
    request = JudgeRequest(
        file_path="plugins/my_specific_file.py",
        rule_id="R1",
        symbol="MyClass.my_method",
        fingerprint="fp-dynamic-12345",
        rationale="this is the rationale text the agent supplied",
        surrounding_code="    return external_payload.get('field', 'fallback')",
    )
    with _mock_judge_call(verdict="ACCEPTED", rationale="ok") as client_class:
        call_judge(request)

    fake_client = client_class.return_value
    create_call = fake_client.chat.completions.create.call_args
    sys_text = create_call.kwargs["messages"][0]["content"][0]["text"]
    user_text = create_call.kwargs["messages"][1]["content"][0]["text"]

    # User message carries the dynamic substitutions.
    assert "plugins/my_specific_file.py" in user_text
    assert "MyClass.my_method" in user_text
    assert "fp-dynamic-12345" in user_text
    assert "this is the rationale text the agent supplied" in user_text
    assert "external_payload.get('field', 'fallback')" in user_text

    # The dynamic material must NOT be in the cached system block.
    assert "plugins/my_specific_file.py" not in sys_text
    assert "fp-dynamic-12345" not in sys_text


# =============================================================================
# Cache-hit accounting: the JudgeResponse exposes prompt-token totals.
# =============================================================================


def test_call_judge_returns_cache_accounting_when_provider_reports_it() -> None:
    request = JudgeRequest(
        file_path="plugins/widget.py",
        rule_id="R1",
        symbol="Widget.lookup",
        fingerprint="fp",
        rationale="...",
        surrounding_code="...",
    )
    with _mock_judge_call(
        verdict="ACCEPTED",
        rationale="ok",
        prompt_tokens=4000,
        cached_tokens=3500,
    ):
        response = call_judge(request)
    assert response.prompt_tokens_total == 4000
    assert response.prompt_tokens_cached == 3500


def test_call_judge_distinguishes_cached_zero_from_cached_none() -> None:
    """Provider reported 0 hits != provider didn't report cached count at all.

    Per the fabrication-decision test in CLAUDE.md: absence and zero are
    different facts. ``cached=0`` means caching was on but produced no
    hit (e.g. first call within a TTL window). ``cached=None`` means
    the provider didn't surface the field at all (older transport,
    caching off). The audit trail loses information if we coerce one to
    the other.
    """
    request = JudgeRequest(
        file_path="plugins/widget.py",
        rule_id="R1",
        symbol="Widget.lookup",
        fingerprint="fp",
        rationale="...",
        surrounding_code="...",
    )
    with _mock_judge_call(
        verdict="ACCEPTED",
        rationale="ok",
        prompt_tokens=4000,
        cached_tokens=0,
    ):
        response_zero = call_judge(request)
    with _mock_judge_call(
        verdict="ACCEPTED",
        rationale="ok",
        prompt_tokens=4000,
        cached_tokens=None,
    ):
        response_none = call_judge(request)
    assert response_zero.prompt_tokens_cached == 0
    assert response_none.prompt_tokens_cached is None


def test_justify_text_output_includes_cache_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "Tier-3 boundary",
        "--owner",
        "test-agent",
    ]
    with _mock_judge_call(
        verdict="ACCEPTED",
        rationale="ok",
        prompt_tokens=4000,
        cached_tokens=3200,
    ):
        exit_code = main(argv)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Cache:" in out
    assert "prompt_tokens=4000" in out
    assert "cached=3200" in out
    # 3200 / 4000 = 80%
    assert "80% hit" in out


def test_justify_text_output_renders_cached_none_as_na(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Provider that didn't surface cached count renders as ``n/a``, not 0."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "Tier-3 boundary",
        "--owner",
        "test-agent",
    ]
    with _mock_judge_call(
        verdict="ACCEPTED",
        rationale="ok",
        prompt_tokens=4000,
        cached_tokens=None,
    ):
        exit_code = main(argv)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "cached=n/a" in out
    # No hit ratio if cached is None.
    assert "hit)" not in out


def test_justify_json_output_includes_cache_fields(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "Tier-3 boundary",
        "--owner",
        "test-agent",
        "--format",
        "json",
    ]
    with _mock_judge_call(
        verdict="ACCEPTED",
        rationale="ok",
        prompt_tokens=4000,
        cached_tokens=3500,
    ):
        exit_code = main(argv)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["prompt_tokens_total"] == 4000
    assert payload["prompt_tokens_cached"] == 3500


def test_justify_json_output_cache_fields_when_provider_omits_cached(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """JSON output preserves the absence signal as JSON null, not zero."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "Tier-3 boundary",
        "--owner",
        "test-agent",
        "--format",
        "json",
    ]
    with _mock_judge_call(
        verdict="ACCEPTED",
        rationale="ok",
        prompt_tokens=4000,
        cached_tokens=None,
    ):
        exit_code = main(argv)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["prompt_tokens_total"] == 4000
    assert payload["prompt_tokens_cached"] is None


def test_justify_records_owner_verbatim(tmp_path: Path) -> None:
    """The operator-supplied --owner string is written into the YAML verbatim.

    No coercion, no stripping (we keep the value the operator typed; the
    type callable only rejects all-whitespace inputs, it doesn't
    normalise). The owner field is the audit identity — silently
    transforming it would defeat its purpose.
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
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "tier-3 boundary",
        "--owner",
        "my-test-agent",
    ]
    with _mock_judge_call(verdict="ACCEPTED", rationale="judge agrees"):
        exit_code = main(argv)
    assert exit_code == 0
    text = (allowlist_dir / "plugins.yaml").read_text(encoding="utf-8")
    assert "owner: my-test-agent" in text

    # Round-trip: the loader exposes the same value on the dataclass.
    al = load_allowlist(allowlist_dir / "plugins.yaml", valid_rule_ids={"R1"})
    assert al.entries[0].owner == "my-test-agent"


# =============================================================================
# C1-1 (elspeth-0e1d0978fa): JudgeResponse.model_id records the SERVED model,
# not the requested one. OpenRouter may transparently re-route to a fallback
# (capacity, regional policy); the audit primitive must capture what actually
# answered the prompt, not the requested route — otherwise a subsequent
# reaudit "against the same model" silently runs against a different one.
# =============================================================================


def test_call_judge_records_served_model_when_transport_routes_to_fallback() -> None:
    """When OpenRouter routes to a fallback model, the served id is recorded."""
    request = JudgeRequest(
        file_path="plugins/widget.py",
        rule_id="R1",
        symbol="Widget.lookup",
        fingerprint="fp",
        rationale="...",
        surrounding_code="...",
    )
    fallback_id = "anthropic/claude-opus-4-served-by-fallback"
    with _mock_judge_call(
        verdict="ACCEPTED",
        rationale="ok",
        served_model=fallback_id,
    ):
        response = call_judge(request)
    # The judge was *requested* as anthropic/claude-opus-4 (the default
    # passed via call_judge's keyword), but the transport returned a
    # different served-model id. The JudgeResponse must surface the
    # served id — that's the audit primitive.
    assert response.model_id == fallback_id


def test_call_judge_falls_back_to_requested_model_when_transport_omits_served_id() -> None:
    """Transports that omit completion.model fall back to the requested id.

    Per the Tier-3 record-what-we-got contract: we don't fabricate a
    served id, but we also don't drop the audit primitive when the
    transport omits it. The fallback is the requested model id —
    documented as ``or model_id`` in ``call_judge``.
    """
    request = JudgeRequest(
        file_path="plugins/widget.py",
        rule_id="R1",
        symbol="Widget.lookup",
        fingerprint="fp",
        rationale="...",
        surrounding_code="...",
    )
    # Falsy served_model: simulates a transport that didn't surface
    # the field. None and "" both trigger the fallback branch.
    with _mock_judge_call(verdict="ACCEPTED", rationale="ok", served_model=None):
        response = call_judge(request)
    assert response.model_id == "anthropic/claude-opus-4"  # the requested default


def test_justify_yaml_records_served_model_id(tmp_path: Path) -> None:
    """The on-disk YAML's judge_model field carries the served (not requested) id."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fallback_id = "anthropic/claude-opus-4-served-by-fallback"
    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "tier-3 boundary",
        "--owner",
        "test-agent",
    ]
    with _mock_judge_call(
        verdict="ACCEPTED",
        rationale="judge agrees",
        served_model=fallback_id,
    ):
        exit_code = main(argv)
    assert exit_code == 0
    text = (allowlist_dir / "plugins.yaml").read_text(encoding="utf-8")
    assert f"judge_model: {fallback_id}" in text


# =============================================================================
# C2-4 (elspeth-0c5db2604c): temperature=0 is pinned on the chat-completions
# call so the verdict is reproducible across reaudit runs. Without this,
# OpenRouter's default sampling temperature (~1.0) produces phantom
# WAS_ACCEPTED_NOW_BLOCKED divergences on identical prompts.
# =============================================================================


def test_call_judge_pins_temperature_zero_for_verdict_reproducibility() -> None:
    request = JudgeRequest(
        file_path="plugins/widget.py",
        rule_id="R1",
        symbol="Widget.lookup",
        fingerprint="fp",
        rationale="...",
        surrounding_code="...",
    )
    with _mock_judge_call(verdict="ACCEPTED", rationale="ok") as client_class:
        call_judge(request)
    fake_client = client_class.return_value
    create_call = fake_client.chat.completions.create.call_args
    # The kwarg must be present and exactly 0 (not 0.0-via-coercion,
    # not absent-and-relying-on-SDK-default). The audit primitive is
    # "we asked for greedy decoding on this verdict" — anything else
    # (including omission, which inherits OpenRouter's ~1.0 default)
    # breaks reaudit reproducibility.
    assert create_call.kwargs["temperature"] == 0


# =============================================================================
# C2-3 (elspeth-98c06d159f): --rule must cross-check against
# finding.rule_id. The default ``trust_tier.tier_model`` is the package
# selector and is a no-op (preserves the existing always-package-id call
# sites in tests + CI). A specific sub-rule id (e.g. R5) that mismatches
# the scanner's actual finding must crash before the judge is called.
# =============================================================================


def test_justify_rule_mismatch_crashes_before_calling_judge(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--rule R5 against an R1 finding refuses to write and names the divergence.

    The synthetic file produces R1 (``dict.get`` on Tier-2 data). The
    operator asserts ``--rule R5`` (loop-iteration rule). Refusing
    prevents an audit-attribution lie: the entry would otherwise be
    written claiming the R5 rule was suppressed when in fact R1 was.
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
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "tier-3 boundary",
        "--owner",
        "test-agent",
        "--rule",
        "R5",
    ]
    # Provide the API key + a no-op client so the only failure mode
    # available is the --rule mismatch (not a config error or a call
    # going through).
    judge_called = MagicMock()
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=False), patch("openai.OpenAI", judge_called):
        exit_code = main(argv)
    assert exit_code == 2
    captured = capsys.readouterr()
    # The error must name both ids (operator-asserted + scanner-reported)
    # so the operator can tell which side to correct.
    assert "R5" in captured.err
    assert "R1" in captured.err
    # The judge must NOT have been called — the mismatch is local and
    # the API call would have cost tokens for an entry we can't write.
    judge_called.assert_not_called()
    # And nothing was written.
    assert not (allowlist_dir / "plugins.yaml").exists()


def test_justify_rule_matching_subrule_id_passes(tmp_path: Path) -> None:
    """--rule R1 against an R1 finding completes the write end-to-end.

    Mirror of the ``test_justify_accepted_writes_entry_with_judge_metadata``
    happy path, but with the operator explicitly naming the sub-rule
    id instead of relying on the package-id default. Asserts the
    cross-check accepts the matching case.
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
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "tier-3 boundary",
        "--owner",
        "test-agent",
        "--rule",
        "R1",  # operator names the sub-rule id explicitly
    ]
    with _mock_judge_call(verdict="ACCEPTED", rationale="judge agrees"):
        exit_code = main(argv)
    assert exit_code == 0
    text = (allowlist_dir / "plugins.yaml").read_text(encoding="utf-8")
    # Entry was written and carries the R1 rule id.
    assert "plugins/widget.py:R1:Widget:lookup:fp=" in text


def test_justify_rule_default_package_id_remains_no_op(tmp_path: Path) -> None:
    """The default --rule (package selector) does NOT trigger the cross-check.

    This pins the no-op contract: existing call sites and tests that
    use the default must keep working unchanged. The cross-check is
    only armed when the operator passes a non-default --rule.
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
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "tier-3 boundary",
        "--owner",
        "test-agent",
        # --rule omitted: argparse fills in "trust_tier.tier_model"
    ]
    with _mock_judge_call(verdict="ACCEPTED", rationale="judge agrees"):
        exit_code = main(argv)
    assert exit_code == 0


# =============================================================================
# C8-3: writer emits binding fields, round-trip through loader stays bound
# =============================================================================
#
# The justify writer is the only production producer of judge-gated allowlist
# entries. To close the C8-3 quartet-transplant attack we need the writer to
# emit both binding fields (file_fingerprint + ast_path) so the loader can
# verify the binding still holds at every subsequent load. These tests pin
# the writer-side half of that contract; the loader-side tests live in
# test_allowlist_judge_metadata_integrity.py under the "C8-3" header.
# =============================================================================


def test_justify_writes_file_fingerprint_and_ast_path(tmp_path: Path) -> None:
    """An ACCEPTED entry written by justify carries both C8-3 binding fields.

    Asserts: the emitted YAML contains a ``file_fingerprint:`` whose
    value is the SHA-256 hex digest of plugins/widget.py's bytes, and
    an ``ast_path:`` whose value matches what the tier_model rule
    emits for the same Widget.lookup R1 finding.
    """
    import hashlib

    from elspeth_lints.rules.trust_tier.tier_model.rule import scan_file

    root, target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    expected_file_fp = hashlib.sha256(target.read_bytes()).hexdigest()
    # Pull the live ast_path from the same scanner the writer uses.
    findings = [f for f in scan_file(target, root) if f.rule_id == "R1"]
    assert len(findings) == 1
    expected_ast_path = findings[0].ast_path

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "tier-3 boundary",
        "--owner",
        "binding-test-agent",
    ]
    with _mock_judge_call(verdict="ACCEPTED", rationale="judge agrees"):
        assert main(argv) == 0

    target_yaml = allowlist_dir / "plugins.yaml"
    text = target_yaml.read_text(encoding="utf-8")
    assert f"file_fingerprint: {expected_file_fp}" in text
    # ast_path contains '[' / ']' so the writer single-quotes it
    # (see _yaml_inline_scalar's conservative-quoting rule); accept
    # either quoted or bare form for forward compatibility.
    assert f"ast_path: '{expected_ast_path}'" in text or f"ast_path: {expected_ast_path}" in text

    # The loader (with source_root) verifies the binding lives — proves
    # the writer-side fingerprint actually matches the live source bytes.
    loaded = load_allowlist(target_yaml, valid_rule_ids={"R1"}, source_root=root)
    assert len(loaded.entries) == 1
    entry = loaded.entries[0]
    assert entry.file_fingerprint == expected_file_fp
    assert entry.ast_path == expected_ast_path


def test_justify_override_also_writes_binding_fields(tmp_path: Path) -> None:
    """The operator-override path emits binding fields too.

    Override entries are the most security-sensitive subset of judge-
    gated entries: the operator bypassed the model's verdict. The
    binding fields must travel with the entry whether or not the
    operator overrode — otherwise the override path becomes the
    transplant vector.
    """
    import hashlib

    root, target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    expected_file_fp = hashlib.sha256(target.read_bytes()).hexdigest()

    argv = [
        "justify",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--file-path",
        "plugins/widget.py",
        "--symbol",
        "Widget.lookup",
        "--rationale",
        "shipping under deadline",
        "--owner",
        "operator",
        "--operator-override",
    ]
    with _mock_judge_call(verdict="BLOCKED", rationale="rationale is shallow; fix the code"):
        assert main(argv) == 0

    target_yaml = allowlist_dir / "plugins.yaml"
    text = target_yaml.read_text(encoding="utf-8")
    assert "judge_verdict: OVERRIDDEN_BY_OPERATOR" in text
    assert "judge_model_verdict: BLOCKED" in text
    assert f"file_fingerprint: {expected_file_fp}" in text
    assert "ast_path:" in text
