"""Unit tests for the ``elspeth-lints justify`` subcommand.

These exercise the judge-gated allowlist-write path. The Anthropic SDK
is mocked at the ``anthropic.Anthropic`` client level so the tests run
offline; the model-response contract is exercised end-to-end (JSON
shape, verdict parsing, allowlist round-trip) without making a network
call.

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


def _mock_anthropic_message(*, verdict: str, rationale: str, should_use_decorator: Any = None) -> MagicMock:
    """Build a mock Anthropic ``messages.create`` return value.

    Mirrors the SDK shape: ``.content`` is a list of blocks, each with
    ``.type`` and ``.text``. The judge expects exactly one ``text``
    block whose payload is a JSON object.
    """
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(
        {
            "verdict": verdict,
            "rationale": rationale,
            "should_use_decorator": should_use_decorator,
        }
    )
    message = MagicMock()
    message.content = [block]
    return message


@contextmanager
def _mock_judge_call(*, verdict: str, rationale: str) -> Iterator[MagicMock]:
    """Patch ``anthropic.Anthropic`` so tests run offline.

    Yields the patched client class so callers can introspect how it was
    invoked (e.g. assert on the prompt the judge would have received).
    """
    fake_message = _mock_anthropic_message(verdict=verdict, rationale=rationale)
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_message
    with (
        patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-key"}, clear=False),
        patch("anthropic.Anthropic", return_value=fake_client) as client_class,
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
    # Strip the key out of the environment for this call.
    env_without_key = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True), pytest.raises(JudgeConfigurationError, match="ANTHROPIC_API_KEY"):
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
    bad_block = MagicMock()
    bad_block.type = "text"
    bad_block.text = "not json at all { ::: }"
    bad_message = MagicMock()
    bad_message.content = [bad_block]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = bad_message
    with (
        patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-key"}, clear=False),
        patch("anthropic.Anthropic", return_value=fake_client),
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
        "--root", str(root),
        "--allowlist-dir", str(allowlist_dir),
        "--file-path", "plugins/widget.py",
        "--symbol", "Widget.lookup",
        "--rationale", "payload is Tier-3 external data from upstream tool-call",
        "--owner", "test-agent-accepted",
        "--format", "json",
    ]
    with _mock_judge_call(verdict="ACCEPTED", rationale="genuine Tier-3 boundary"):
        exit_code = main(argv)

    assert exit_code == 0
    target_yaml = allowlist_dir / "plugins.yaml"
    assert target_yaml.exists()
    text = target_yaml.read_text(encoding="utf-8")
    assert "judge_verdict: ACCEPTED" in text
    assert "judge_model: claude-opus-4-7" in text
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
    assert entry.judge_model == "claude-opus-4-7"
    assert entry.judge_rationale == "genuine Tier-3 boundary"
    assert entry.judge_recorded_at is not None
    assert entry.judge_recorded_at.tzinfo is not None


# ---------- CLI: blocked path ----------


def test_justify_blocked_does_not_write_and_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root", str(root),
        "--allowlist-dir", str(allowlist_dir),
        "--file-path", "plugins/widget.py",
        "--symbol", "Widget.lookup",
        "--rationale", "I just don't want to fix this",
        "--owner", "lazy-agent",
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
        "--root", str(root),
        "--allowlist-dir", str(allowlist_dir),
        "--file-path", "plugins/widget.py",
        "--symbol", "Widget.lookup",
        "--rationale", "shipping under deadline",
        "--owner", "operator-jdoe",
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
        "--root", str(root),
        "--allowlist-dir", str(allowlist_dir),
        "--file-path", "plugins/widget.py",
        "--symbol", "Widget.lookup",
        "--rationale", "genuine Tier-3 boundary",
        "--owner", "test-agent",
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
        "--root", str(root),
        "--allowlist-dir", str(allowlist_dir),
        "--file-path", "plugins/widget.py",
        "--symbol", "Widget.lookup",
        "--rationale", "...",
        "--owner", "test-agent",
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
        "--root", str(root),
        "--allowlist-dir", str(allowlist_dir),
        "--file-path", "plugins/widget.py",
        "--symbol", "Widget.lookup",
        "--rationale", "...",
        "--owner", "test-agent",
    ]
    env_without_key = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        exit_code = main(argv)

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "ANTHROPIC_API_KEY" in captured.err


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
        '''\
class Widget:
    def lookup(self, a: dict, b: dict) -> tuple[str, str]:
        # Two R1 findings on the same symbol_context — the judge gate
        # needs them disambiguated before it will gate one of them.
        return a.get("x", ""), b.get("y", "")
''',
        encoding="utf-8",
    )
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "justify",
        "--root", str(root),
        "--allowlist-dir", str(allowlist_dir),
        "--file-path", "plugins/widget.py",
        "--symbol", "Widget.lookup",
        "--rationale", "Tier-3 boundary",
        "--owner", "test-agent",
    ]
    # Set the API key so we definitely don't fall out via the
    # configuration check — the ambiguity error must fire first.
    judge_called = MagicMock()
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False), patch("anthropic.Anthropic", judge_called):
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
        "--root", str(root),
        "--allowlist-dir", str(allowlist_dir),
        "--file-path", "plugins/widget.py",
        "--symbol", "Widget.lookup",
        "--rationale", "tier-3 boundary; payload comes from external tool-call",
        "--owner", "round-trip-agent",
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
    assert first.judge_model == second.judge_model == "claude-opus-4-7"
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
        "--root", str(root),
        "--allowlist-dir", str(allowlist_dir),
        "--file-path", "plugins/widget.py",
        "--symbol", "Widget.lookup",
        "--rationale", "tier-3 boundary",
        # --owner deliberately omitted
    ]
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--owner" in captured.err


@pytest.mark.parametrize("owner_value", ["", "   ", "\t\t", "\n"])
def test_justify_rejects_empty_owner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], owner_value: str
) -> None:
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
        "--root", str(root),
        "--allowlist-dir", str(allowlist_dir),
        "--file-path", "plugins/widget.py",
        "--symbol", "Widget.lookup",
        "--rationale", "tier-3 boundary",
        "--owner", owner_value,
    ]
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    # argparse's standard frame is "argument --owner: <our message>"
    assert "--owner" in captured.err
    assert "non-empty" in captured.err or "audit identity" in captured.err


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
        "--root", str(root),
        "--allowlist-dir", str(allowlist_dir),
        "--file-path", "plugins/widget.py",
        "--symbol", "Widget.lookup",
        "--rationale", "tier-3 boundary",
        "--owner", "my-test-agent",
    ]
    with _mock_judge_call(verdict="ACCEPTED", rationale="judge agrees"):
        exit_code = main(argv)
    assert exit_code == 0
    text = (allowlist_dir / "plugins.yaml").read_text(encoding="utf-8")
    assert "owner: my-test-agent" in text

    # Round-trip: the loader exposes the same value on the dataclass.
    al = load_allowlist(allowlist_dir / "plugins.yaml", valid_rule_ids={"R1"})
    assert al.entries[0].owner == "my-test-agent"
