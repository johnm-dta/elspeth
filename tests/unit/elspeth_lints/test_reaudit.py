"""Unit tests for the ``elspeth-lints reaudit`` subcommand (Slice 3).

Mirrors ``test_justify.py``'s mocking pattern: the Anthropic SDK is
patched at ``anthropic.Anthropic`` so tests run offline; reaudit's
read-only-on-allowlist invariant is asserted by re-loading the YAML
after the run and checking it byte-identical to the pre-run state.

These tests pin the divergence-classification matrix exhaustively:
one test per ``ReauditDivergence`` value, plus filter behaviour
(``--since``, ``--limit``, ``--include-pre-judge``) and report-format
rendering.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth_lints.core.allowlist import AllowlistEntry, JudgeVerdict, compute_judge_metadata_signature
from elspeth_lints.core.cli import main
from elspeth_lints.core.judge import DEFAULT_JUDGE_MODEL, JUDGE_POLICY_HASH
from elspeth_lints.core.reaudit import (
    ReauditCause,
    ReauditDivergence,
    ReauditError,
    ReauditOutcome,
    ReauditReport,
    _apply_filters,
    _parse_entry_key,
    reaudit_entries,
    render_report_json,
    render_report_markdown,
    render_report_text,
)

# Synthetic source: one R1 finding inside Widget.lookup.
_SYNTHETIC_SOURCE = '''\
"""Synthetic module used in reaudit tests."""


class Widget:
    def lookup(self, payload: dict) -> str:
        return payload.get("name", "anonymous")
'''

_TEST_JUDGE_METADATA_HMAC_KEY = "test-judge-metadata-hmac-key-2026-05-24"


@pytest.fixture(autouse=True)
def _judge_metadata_hmac_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep source-root allowlist loads able to verify signed fixture rows."""
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)


# =====================================================================
# Helpers
# =====================================================================


def _build_source_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Lay out a minimal source root with one finding-producing file."""
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    target = root / "plugins" / "widget.py"
    target.write_text(_SYNTHETIC_SOURCE, encoding="utf-8")
    return root, target


def _build_allowlist_dir(tmp_path: Path) -> Path:
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
    served_model: str | None = DEFAULT_JUDGE_MODEL,
    prompt_tokens: int = 4000,
    cached_tokens: int | None = 0,
) -> MagicMock:
    """OpenAI-shape chat-completion mock for OpenRouter routing.

    The judge calls the OpenAI SDK pointed at OpenRouter; this mock
    matches ``client.chat.completions.create(...)``'s return shape —
    ``.choices[0].message.content`` is a JSON string the judge will
    parse, and ``.usage`` carries prompt-token accounting (required;
    the judge reads it offensively for cache-hit telemetry).
    """
    message = MagicMock()
    message.content = json.dumps({"verdict": verdict, "rationale": rationale, "confidence": 0.91, "should_use_decorator": None})
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    completion.model = served_model
    completion.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        prompt_tokens_details=MagicMock(cached_tokens=cached_tokens),
    )
    return completion


@contextmanager
def _mock_judge_call(
    *,
    verdict: str,
    rationale: str,
    served_model: str | None = DEFAULT_JUDGE_MODEL,
    prompt_tokens: int = 4000,
    cached_tokens: int | None = 0,
) -> Iterator[MagicMock]:
    """Patch ``openai.OpenAI`` so reaudit's judge call runs offline."""
    fake_completion = _mock_openrouter_completion(
        verdict=verdict,
        rationale=rationale,
        served_model=served_model,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_completion
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=fake_client) as client_class,
    ):
        yield client_class


def _fixture_judge_metadata_signature(
    *,
    key: str,
    file_fingerprint: str,
    ast_path: str,
    judge_verdict: str = "ACCEPTED",
    judge_model_verdict: str | None = None,
    judge_recorded_at: str = "2024-01-01T00:00:00+00:00",
    judge_model: str = "claude-opus-4-7",
    judge_policy_hash: str = JUDGE_POLICY_HASH,
    judge_rationale: str = "original judge said the boundary was genuine",
) -> str:
    return compute_judge_metadata_signature(
        key=key,
        file_fingerprint=file_fingerprint,
        ast_path=ast_path,
        judge_verdict=JudgeVerdict(judge_verdict),
        judge_model_verdict=JudgeVerdict(judge_model_verdict) if judge_model_verdict is not None else None,
        judge_recorded_at=datetime.fromisoformat(judge_recorded_at),
        judge_model=judge_model,
        judge_policy_hash=judge_policy_hash,
        judge_rationale=judge_rationale,
        hmac_key=_TEST_JUDGE_METADATA_HMAC_KEY.encode("utf-8"),
    )


def _write_widget_lookup_entry(
    allowlist_dir: Path,
    *,
    fingerprint: str,
    judge_verdict: str | None,
    judge_model_verdict: str | None = None,
    judge_recorded_at: str | None = None,
    source_root: Path | None = None,
) -> Path:
    """Write a single allowlist entry whose key targets Widget.lookup's R1.

    The fingerprint is supplied by the caller; tests obtain the real
    fingerprint from a separate scan helper so the entry actually
    matches a live finding. When ``judge_verdict`` is set, the C8-3
    binding fields (file_fingerprint + ast_path) are also written:
    if ``source_root`` is supplied, both are computed against the live
    source tree so reaudit's load gate passes; otherwise synthetic
    placeholders are emitted (only acceptable for tests that load via
    paths passing ``source_root=None`` to ``load_allowlist``).
    """
    key = f"plugins/widget.py:R1:Widget:lookup:fp={fingerprint}"
    lines: list[str] = []
    lines.append("allow_hits:")
    lines.append(f"- key: {key}")
    lines.append("  owner: test-owner")
    lines.append("  reason: |-")
    lines.append("    payload is Tier-3 external data from upstream tool-call")
    lines.append("  safety: |-")
    lines.append("    Suppression gated by cicd-judge; see judge_rationale below.")
    lines.append("  expires: '2030-01-01'")
    if judge_verdict is not None:
        lines.append(f"  judge_verdict: {judge_verdict}")
    if judge_model_verdict is not None:
        lines.append(f"  judge_model_verdict: {judge_model_verdict}")
    if judge_recorded_at is not None:
        lines.append(f"  judge_recorded_at: '{judge_recorded_at}'")
        lines.append("  judge_model: claude-opus-4-7")
        lines.append(f"  judge_policy_hash: '{JUDGE_POLICY_HASH}'")
        lines.append("  judge_rationale: |-")
        lines.append("    original judge said the boundary was genuine")
    # Binding fields (C8-3): required co-presence companions of
    # judge_verdict per invariant 8.
    if judge_verdict is not None:
        if source_root is not None:
            import hashlib as _hashlib  # local import to keep top-of-file footprint minimal

            target_source = source_root / "plugins/widget.py"
            file_fp = _hashlib.sha256(target_source.read_bytes()).hexdigest()
            live_ast_path = _live_widget_finding(source_root).ast_path
        else:
            file_fp = "0" * 64
            live_ast_path = "body[0]/body[0]/body[1]/value"
        lines.append(f"  file_fingerprint: '{file_fp}'")
        lines.append(f"  ast_path: '{live_ast_path}'")
        lines.append(
            "  judge_metadata_signature: '"
            + _fixture_judge_metadata_signature(
                key=key,
                file_fingerprint=file_fp,
                ast_path=live_ast_path,
                judge_verdict=judge_verdict,
                judge_model_verdict=judge_model_verdict,
                judge_recorded_at=judge_recorded_at or "2024-01-01T00:00:00+00:00",
            )
            + "'"
        )
    target = allowlist_dir / "plugins.yaml"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _write_duplicate_widget_lookup_entries(
    allowlist_dir: Path,
    *,
    source_root: Path,
    fingerprint: str,
    count: int,
) -> None:
    """Write ``count`` live allowlist entries for Widget.lookup.

    Duplicate canonical keys are unusual but valid YAML rows and let
    budget tests exercise the orchestration without needing several
    synthetic source files. Each row has distinct owner/rationale
    metadata so the loader still constructs separate entries.
    """
    import hashlib as _hashlib

    finding = _live_widget_finding(source_root)
    live_file_fp = _hashlib.sha256((source_root / "plugins/widget.py").read_bytes()).hexdigest()
    live_ast_path = finding.ast_path
    entry_key = f"plugins/widget.py:R1:Widget:lookup:fp={fingerprint}"
    lines: list[str] = ["allow_hits:"]
    for i in range(count):
        judge_rationale = f"rationale entry {i}"
        lines.append(f"- key: {entry_key}")
        lines.append(f"  owner: agent-{i}")
        lines.append("  reason: |-")
        lines.append("    payload is Tier-3 external data")
        lines.append("  safety: |-")
        lines.append("    bounded coercion")
        lines.append("  expires: '2030-01-01'")
        lines.append("  judge_verdict: ACCEPTED")
        lines.append("  judge_recorded_at: '2024-01-01T00:00:00+00:00'")
        lines.append("  judge_model: claude-opus-4-7")
        lines.append(f"  judge_policy_hash: '{JUDGE_POLICY_HASH}'")
        lines.append("  judge_rationale: |-")
        lines.append(f"    {judge_rationale}")
        lines.append(f"  file_fingerprint: '{live_file_fp}'")
        lines.append(f"  ast_path: '{live_ast_path}'")
        lines.append(
            "  judge_metadata_signature: '"
            + _fixture_judge_metadata_signature(
                key=entry_key,
                file_fingerprint=live_file_fp,
                ast_path=live_ast_path,
                judge_rationale=judge_rationale,
            )
            + "'"
        )
    (allowlist_dir / "plugins.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _live_fingerprint_for_widget(root: Path) -> str:
    """Run the scanner once to grab the fingerprint of the R1 finding."""
    return _live_widget_finding(root).fingerprint


def _live_widget_finding(root: Path) -> Any:
    """Return the R1 ``Finding`` instance for widget.lookup at ``root``.

    Tests use this to grab BOTH the live fingerprint (for the canonical
    key) and the live ast_path (for the binding field) from a single
    scan.
    """
    from elspeth_lints.rules.trust_tier.tier_model.rule import scan_file

    target_file = (root / "plugins/widget.py").resolve()
    findings = list(scan_file(target_file, root))
    r1_findings = [f for f in findings if f.rule_id == "R1"]
    if len(r1_findings) != 1:
        raise AssertionError(f"expected exactly one R1 finding, got {len(r1_findings)}: {r1_findings}")
    return r1_findings[0]


# =====================================================================
# _parse_entry_key
# =====================================================================


def test_parse_entry_key_with_qualified_symbol() -> None:
    key = "plugins/widget.py:R1:Widget:lookup:fp=abc123"
    parsed = _parse_entry_key(key)
    assert parsed == ("plugins/widget.py", "R1", ("Widget", "lookup"), "abc123")


def test_parse_entry_key_with_module_sentinel() -> None:
    key = "plugins/widget.py:TC:_module_:fp=deadbeef"
    parsed = _parse_entry_key(key)
    assert parsed == ("plugins/widget.py", "TC", (), "deadbeef")


def test_parse_entry_key_malformed_returns_none() -> None:
    assert _parse_entry_key("nonsense") is None
    assert _parse_entry_key("a:b:c") is None  # no :fp= suffix
    assert _parse_entry_key("a:b:fp=") is None  # empty fingerprint


# =====================================================================
# Divergence classification
# =====================================================================


def test_still_agrees_when_accepted_stays_accepted(tmp_path: Path) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )
    yaml_text_before = (allowlist_dir / "plugins.yaml").read_text(encoding="utf-8")

    with _mock_judge_call(verdict="ACCEPTED", rationale="boundary is still genuine"):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )

    assert len(report.outcomes) == 1
    outcome = report.outcomes[0]
    assert outcome.divergence is ReauditDivergence.STILL_AGREES
    assert outcome.fresh_verdict is JudgeVerdict.ACCEPTED
    assert outcome.fresh_rationale == "boundary is still genuine"

    # Reaudit is read-only on YAML.
    yaml_text_after = (allowlist_dir / "plugins.yaml").read_text(encoding="utf-8")
    assert yaml_text_before == yaml_text_after


def test_was_accepted_now_blocked(tmp_path: Path) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )

    with _mock_judge_call(verdict="BLOCKED", rationale="code has drifted; boundary no longer applies"):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )

    assert report.outcomes[0].divergence is ReauditDivergence.WAS_ACCEPTED_NOW_BLOCKED
    assert report.outcomes[0].cause is ReauditCause.MODEL_NOISE_OR_POLICY_DRIFT


def test_cli_reaudit_exits_1_on_verdict_divergence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A verdict-change divergence must be CI-gatable by exit status."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )

    argv = [
        "reaudit",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--format",
        "text",
    ]
    with _mock_judge_call(verdict="BLOCKED", rationale="code has drifted; boundary no longer applies"):
        exit_code = main(argv)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "WAS_ACCEPTED_NOW_BLOCKED" in captured.out


def test_override_no_longer_needed(tmp_path: Path) -> None:
    """OVERRIDDEN+model=BLOCKED entry where fresh model now ACCEPTs.

    Operator overrode the original BLOCK; the model has since changed
    its mind. The override is no longer load-bearing.
    """
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="OVERRIDDEN_BY_OPERATOR",
        judge_model_verdict="BLOCKED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )

    with _mock_judge_call(verdict="ACCEPTED", rationale="now agrees the boundary is genuine"):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )

    assert report.outcomes[0].divergence is ReauditDivergence.OVERRIDE_NO_LONGER_NEEDED


def test_override_still_needed(tmp_path: Path) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="OVERRIDDEN_BY_OPERATOR",
        judge_model_verdict="BLOCKED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )

    with _mock_judge_call(verdict="BLOCKED", rationale="still says fix the code"):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )

    assert report.outcomes[0].divergence is ReauditDivergence.OVERRIDE_STILL_NEEDED


def test_pre_judge_fresh_block(tmp_path: Path) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict=None,  # pre-judge entry
    )

    with _mock_judge_call(verdict="BLOCKED", rationale="this should not be allowlisted"):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=True,
        )

    assert report.outcomes[0].divergence is ReauditDivergence.PRE_JUDGE_FRESH_BLOCK


def test_pre_judge_fresh_accept(tmp_path: Path) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict=None,
    )

    with _mock_judge_call(verdict="ACCEPTED", rationale="legitimate Tier-3 boundary"):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=True,
        )

    assert report.outcomes[0].divergence is ReauditDivergence.PRE_JUDGE_FRESH_ACCEPT


def test_entry_obsolete_when_no_current_finding(tmp_path: Path) -> None:
    """Entry references a finding the source no longer produces.

    No judge call is made — ENTRY_OBSOLETE is determined before
    invoking the model, which makes this case testable without
    mocking the Anthropic client at all.
    """
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    # Stale fingerprint that does NOT match any live finding.
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint="stale_fingerprint_that_does_not_match",
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )
    # Don't mock anthropic — if the judge gets called this test would
    # fail with ImportError or ConfigurationError, which is what we want.
    report = reaudit_entries(
        root=root.resolve(),
        allowlist_dir=allowlist_dir,
        rule_filter="trust_tier.tier_model",
        since=None,
        limit=None,
        include_pre_judge=False,
    )

    assert report.outcomes[0].divergence is ReauditDivergence.ENTRY_OBSOLETE
    assert report.outcomes[0].fresh_verdict is None


def test_entry_obsolete_when_source_file_deleted(tmp_path: Path) -> None:
    """Source-file deletion under a judge-gated entry now fails at load (C8-3).

    Before C8-3 binding enforcement, reaudit would classify this as
    ENTRY_OBSOLETE (no live finding because the file is gone). With
    binding enforcement, the load-time gate fires first: a judge-gated
    entry bound to a missing file is corruption (the dependent entry
    should have been removed when the source was deleted), and the
    correct response is to refuse to load — surfacing the dangling
    audit record rather than silently degrading it to "obsolete." The
    operator removes the entry; reaudit is not the cleanup tool for
    this class of drift.
    """
    root, target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )
    target.unlink()

    with pytest.raises(ValueError, match=r"judge-gated entry binds to .* which does not exist"):
        reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )


# =====================================================================
# Filters
# =====================================================================


def test_since_filter_skips_fresh_entries(tmp_path: Path) -> None:
    """Entries with judge_recorded_at >= --since are skipped."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    # Entry recorded yesterday — newer than the --since cutoff below.
    yesterday = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at=yesterday,
    )
    cutoff = datetime.now(UTC) - timedelta(days=2)

    # No mock needed — the entry should be filtered out before any
    # judge call.
    report = reaudit_entries(
        root=root.resolve(),
        allowlist_dir=allowlist_dir,
        rule_filter="trust_tier.tier_model",
        since=cutoff,
        limit=None,
        include_pre_judge=False,
    )
    assert report.outcomes == ()


def test_since_filter_passes_old_entries(tmp_path: Path) -> None:
    """Entries with judge_recorded_at < --since survive the filter."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2020-01-01T00:00:00+00:00",
    )
    cutoff = datetime(2024, 1, 1, tzinfo=UTC)

    with _mock_judge_call(verdict="ACCEPTED", rationale="still fine"):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=cutoff,
            limit=None,
            include_pre_judge=False,
        )
    assert len(report.outcomes) == 1


def test_since_filter_surfaces_future_dated_entries(tmp_path: Path) -> None:
    """Future-dated judge timestamps must not evade a --since sweep."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    reference_time = datetime(2024, 1, 1, tzinfo=UTC)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-02T00:00:00+00:00",
    )

    with _mock_judge_call(verdict="ACCEPTED", rationale="should not be called") as client_class:
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=reference_time - timedelta(days=30),
            limit=None,
            include_pre_judge=False,
            reference_time=reference_time,
        )

    assert len(report.outcomes) == 1
    outcome = report.outcomes[0]
    assert outcome.divergence is ReauditDivergence.FUTURE_DATED_ENTRY
    assert outcome.fresh_verdict is None
    assert outcome.fresh_rationale is not None
    assert "after reaudit reference_time" in outcome.fresh_rationale
    assert client_class.call_count == 0


def test_apply_filters_orders_since_candidates_oldest_first_before_limit() -> None:
    """Limit applies after deterministic recorded-at ordering, not YAML order."""
    newer = AllowlistEntry(
        key="plugins/widget.py:R1:Widget:lookup:fp=newer",
        owner="test-owner",
        reason="newer",
        safety="test safety",
        expires=None,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime(2023, 1, 1, tzinfo=UTC),
    )
    older = AllowlistEntry(
        key="plugins/widget.py:R1:Widget:lookup:fp=older",
        owner="test-owner",
        reason="older",
        safety="test safety",
        expires=None,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime(2020, 1, 1, tzinfo=UTC),
    )

    filtered = _apply_filters(
        entries=[newer, older],
        valid_rule_ids=frozenset({"R1"}),
        include_pre_judge=False,
        since=datetime(2024, 1, 1, tzinfo=UTC),
        limit=1,
    )

    assert [entry.key for entry in filtered] == [older.key]


def test_limit_caps_processed_entries(tmp_path: Path) -> None:
    """``--limit N`` processes only the first N entries surviving filters."""
    import hashlib as _hashlib

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    finding = _live_widget_finding(root)
    fp = finding.fingerprint
    # Live binding values so each entry passes the C8-3 load-time gate.
    live_file_fp = _hashlib.sha256((root / "plugins/widget.py").read_bytes()).hexdigest()
    live_ast_path = finding.ast_path
    # Three entries — duplicate keys are unusual but the loader accepts
    # them and our orchestrator iterates them in order. We test the
    # mechanism by writing three.
    lines: list[str] = ["allow_hits:"]
    for i in range(3):
        entry_key = f"plugins/widget.py:R1:Widget:lookup:fp={fp}"
        lines.append(f"- key: {entry_key}")
        lines.append(f"  owner: agent-{i}")
        lines.append("  reason: |-")
        lines.append("    payload is Tier-3 external data")
        lines.append("  safety: |-")
        lines.append("    bounded coercion")
        lines.append("  expires: '2030-01-01'")
        lines.append("  judge_verdict: ACCEPTED")
        lines.append("  judge_recorded_at: '2024-01-01T00:00:00+00:00'")
        lines.append("  judge_model: claude-opus-4-7")
        lines.append(f"  judge_policy_hash: '{JUDGE_POLICY_HASH}'")
        lines.append("  judge_rationale: |-")
        lines.append(f"    rationale entry {i}")
        lines.append(f"  file_fingerprint: '{live_file_fp}'")
        lines.append(f"  ast_path: '{live_ast_path}'")
        lines.append(
            "  judge_metadata_signature: '"
            + _fixture_judge_metadata_signature(
                key=entry_key,
                file_fingerprint=live_file_fp,
                ast_path=live_ast_path,
                judge_rationale=f"rationale entry {i}",
            )
            + "'"
        )
    (allowlist_dir / "plugins.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with _mock_judge_call(verdict="ACCEPTED", rationale="ok"):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=2,
            include_pre_judge=False,
        )
    assert len(report.outcomes) == 2


def test_max_calls_caps_actual_judge_calls_and_marks_sweep_incomplete(tmp_path: Path) -> None:
    """``--max-calls`` is a judge-call budget, distinct from entry filtering."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_duplicate_widget_lookup_entries(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        count=3,
    )

    with _mock_judge_call(verdict="ACCEPTED", rationale="ok") as client_class:
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            max_calls=1,
            include_pre_judge=False,
        )

    client = client_class.return_value
    assert client.chat.completions.create.call_count == 1
    assert len(report.outcomes) == 1
    assert report.entries_dispatched == 1
    assert report.total_entries == 3
    assert report.max_judge_calls == 1
    assert report.judge_calls_attempted == 1
    assert "INCOMPLETE SWEEP" in render_report_text(report)


def test_reaudit_records_structured_cost_telemetry_in_report_and_json(tmp_path: Path) -> None:
    """Fresh judge token accounting is durable report data, not only CLI chatter."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )

    with _mock_judge_call(verdict="ACCEPTED", rationale="ok", prompt_tokens=5100, cached_tokens=1200):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            max_calls=None,
            include_pre_judge=False,
        )

    assert report.judge_calls_attempted == 1
    assert report.prompt_tokens_total == 5100
    assert report.prompt_tokens_cached == 1200
    assert report.prompt_tokens_uncached == 3900
    outcome = report.outcomes[0]
    assert outcome.judge_call_attempted is True
    assert outcome.fresh_prompt_tokens_total == 5100
    assert outcome.fresh_prompt_tokens_cached == 1200

    payload = json.loads(render_report_json(report))
    assert payload["cost_telemetry"] == {
        "judge_calls_attempted": 1,
        "max_judge_calls": None,
        "prompt_tokens_total": 5100,
        "prompt_tokens_cached": 1200,
        "prompt_tokens_uncached": 3900,
    }
    assert payload["outcomes"][0]["judge_call_attempted"] is True
    assert payload["outcomes"][0]["fresh_prompt_tokens_total"] == 5100
    assert payload["outcomes"][0]["fresh_prompt_tokens_cached"] == 1200


def test_include_pre_judge_off_skips_pre_judge_entries(tmp_path: Path) -> None:
    """Default ``include_pre_judge=False`` excludes None-verdict entries."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict=None,
    )

    # No mock — the entry must be filtered out before any judge call.
    report = reaudit_entries(
        root=root.resolve(),
        allowlist_dir=allowlist_dir,
        rule_filter="trust_tier.tier_model",
        since=None,
        limit=None,
        include_pre_judge=False,
    )
    assert report.outcomes == ()


# =====================================================================
# Validation errors
# =====================================================================


def test_unsupported_rule_filter_raises(tmp_path: Path) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    with pytest.raises(ReauditError, match="not supported"):
        reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="some.other.rule",
            since=None,
            limit=None,
            include_pre_judge=False,
        )


def test_reaudit_rule_filter_skips_entries_from_other_rule_package(tmp_path: Path) -> None:
    """Entries outside the selected rule package are skipped before scanning."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    (allowlist_dir / "plugins.yaml").write_text(
        "\n".join(
            [
                "allow_hits:",
                f"- key: plugins/widget.py:TBE1:Widget:lookup:fp={fp}",
                "  owner: test-owner",
                "  reason: |-",
                "    trust-boundary tests entry in a mixed allowlist file",
                "  safety: |-",
                "    Suppression gated by cicd-judge; see judge_rationale below.",
                "  expires: '2030-01-01'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with _mock_judge_call(verdict="ACCEPTED", rationale="should not be called") as client_class:
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=True,
        )

    assert report.outcomes == ()
    assert report.total_entries == 0
    assert report.entries_skipped_by_rule_filter == 1
    assert client_class.call_count == 0
    assert "entries skipped by rule_filter" in render_report_text(report)
    assert "| _entries skipped by rule_filter_ | 1 |" in render_report_markdown(report)
    assert json.loads(render_report_json(report))["entries_skipped_by_rule_filter"] == 1


def test_missing_allowlist_dir_raises(tmp_path: Path) -> None:
    root, _target = _build_source_tree(tmp_path)
    with pytest.raises(ReauditError, match="not a directory"):
        reaudit_entries(
            root=root.resolve(),
            allowlist_dir=tmp_path / "nonexistent",
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )


# =====================================================================
# Report rendering
# =====================================================================


def _fixed_outcome(
    *,
    key: str,
    divergence: ReauditDivergence,
    original_verdict: JudgeVerdict | None,
    original_model_verdict: JudgeVerdict | None = None,
    fresh_verdict: JudgeVerdict | None,
    fresh_rationale: str | None,
    cause: ReauditCause | None = None,
) -> ReauditOutcome:
    """Construct a ReauditOutcome with a deterministic timestamp.

    Used for snapshot tests where ``call_judge``'s real
    ``datetime.now(UTC)`` would yield a moving target.
    """
    fixed_dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    return ReauditOutcome(
        entry=AllowlistEntry(
            key=key,
            owner="test-owner",
            reason="test reason",
            safety="test safety",
            expires=None,
            judge_verdict=original_verdict,
            judge_model_verdict=original_model_verdict,
            judge_recorded_at=datetime(2024, 1, 1, tzinfo=UTC) if original_verdict is not None else None,
            judge_model="claude-opus-4-7" if original_verdict is not None else None,
            judge_policy_hash=JUDGE_POLICY_HASH if original_verdict is not None else None,
            judge_rationale="original rationale" if original_verdict is not None else None,
        ),
        original_verdict=original_verdict,
        original_model_verdict=original_model_verdict,
        fresh_verdict=fresh_verdict,
        fresh_rationale=fresh_rationale,
        fresh_recorded_at=fixed_dt if fresh_verdict is not None else None,
        fresh_model_id=DEFAULT_JUDGE_MODEL if fresh_verdict is not None else None,
        divergence=divergence,
        cause=cause or ReauditCause.for_divergence(divergence),
        code_snapshot="  >> 10  example code line",
    )


def test_render_text_report_lists_outcomes_and_summary() -> None:
    outcomes = [
        _fixed_outcome(
            key="plugins/a.py:R1:Foo:bar:fp=aaa",
            divergence=ReauditDivergence.STILL_AGREES,
            original_verdict=JudgeVerdict.ACCEPTED,
            fresh_verdict=JudgeVerdict.ACCEPTED,
            fresh_rationale="still good",
        ),
        _fixed_outcome(
            key="plugins/b.py:R2:_module_:fp=bbb",
            divergence=ReauditDivergence.WAS_ACCEPTED_NOW_BLOCKED,
            original_verdict=JudgeVerdict.ACCEPTED,
            fresh_verdict=JudgeVerdict.BLOCKED,
            fresh_rationale="code drifted",
        ),
    ]
    report = ReauditReport.from_outcomes(outcomes)
    text = render_report_text(report)
    assert "plugins/a.py:R1:Foo:bar:fp=aaa  STILL_AGREES  fresh=ACCEPTED" in text
    assert "plugins/b.py:R2:_module_:fp=bbb  WAS_ACCEPTED_NOW_BLOCKED  fresh=BLOCKED" in text
    assert "STILL_AGREES" in text
    assert "Summary:" in text


def test_divergence_order_is_exhaustive() -> None:
    """Every divergence enum member must have an explicit report ordering."""
    from elspeth_lints.core.reaudit import _DIVERGENCE_ORDER

    assert set(_DIVERGENCE_ORDER) == set(ReauditDivergence)


def test_render_json_report_is_valid_json_with_enum_strings() -> None:
    outcomes = [
        _fixed_outcome(
            key="plugins/a.py:R1:Foo:bar:fp=aaa",
            divergence=ReauditDivergence.OVERRIDE_NO_LONGER_NEEDED,
            original_verdict=JudgeVerdict.OVERRIDDEN_BY_OPERATOR,
            original_model_verdict=JudgeVerdict.BLOCKED,
            fresh_verdict=JudgeVerdict.ACCEPTED,
            fresh_rationale="model now agrees",
        ),
    ]
    report = ReauditReport.from_outcomes(outcomes)
    rendered = render_report_json(report)
    payload = json.loads(rendered)
    assert payload["outcomes"][0]["divergence"] == "OVERRIDE_NO_LONGER_NEEDED"
    assert payload["outcomes"][0]["cause"] == "OPERATOR_OVERRIDE_RECHECK"
    assert payload["outcomes"][0]["original_verdict"] == "OVERRIDDEN_BY_OPERATOR"
    assert payload["outcomes"][0]["original_model_verdict"] == "BLOCKED"
    assert payload["outcomes"][0]["fresh_verdict"] == "ACCEPTED"
    assert payload["outcomes"][0]["fresh_model_id"] == DEFAULT_JUDGE_MODEL
    entry_payload = payload["outcomes"][0]["entry"]
    assert "matched" not in entry_payload
    assert set(entry_payload) == {
        "key",
        "owner",
        "reason",
        "safety",
        "expires",
        "file_fingerprint",
        "ast_path",
        "scope_fingerprint",
        "judge_signature_version",
        "pattern",
        "source_file",
        "judge_verdict",
        "judge_recorded_at",
        "judge_model",
        "judge_rationale",
        "judge_confidence",
        "judge_model_verdict",
        "judge_policy_hash",
        "judge_metadata_signature",
        "judge_excerpt_redactions",
        "audit_review",
    }
    # Summary is a list of {divergence, count} dicts, ordered by severity.
    # SOURCE_EXCERPT_REJECTED outranks JUDGE_CALL_FAILED because the
    # former is a security signal (forged path / exfiltration attempt)
    # while future-dated entries are timestamp tampering / clock-skew
    # signals that can evade --since. JUDGE_CALL_FAILED is a transient
    # transport signal. Classification failure is next because the
    # judge returned but the stored/fresh verdict tuple could not be
    # mapped. Ambiguous current finding matches rank before
    # verdict-change divergences because the sweep did not obtain a
    # trustworthy current finding to judge.
    summary_names = [s["divergence"] for s in payload["summary"]]
    assert summary_names[0] == "SOURCE_EXCERPT_REJECTED"
    assert summary_names[1] == "FUTURE_DATED_ENTRY"
    assert summary_names[2] == "JUDGE_CALL_FAILED"
    assert summary_names[3] == "JUDGE_CLASSIFICATION_FAILED"
    assert summary_names[4] == "AMBIGUOUS_FINDING_MATCH"
    assert summary_names[5] == "WAS_ACCEPTED_NOW_BLOCKED"


def test_cause_axis_renders_in_text_markdown_and_json() -> None:
    outcome = _fixed_outcome(
        key="plugins/drift.py:R1:Foo:bar:fp=aaa",
        divergence=ReauditDivergence.WAS_ACCEPTED_NOW_BLOCKED,
        cause=ReauditCause.MODEL_NOISE_OR_POLICY_DRIFT,
        original_verdict=JudgeVerdict.ACCEPTED,
        fresh_verdict=JudgeVerdict.BLOCKED,
        fresh_rationale="fresh model blocked it",
    )
    report = ReauditReport.from_outcomes([outcome])

    text = render_report_text(report)
    assert "cause=MODEL_NOISE_OR_POLICY_DRIFT" in text

    markdown = render_report_markdown(report)
    assert "| Entry | Original Verdict | Fresh Verdict | Fresh Model | Cause | Notes |" in markdown
    assert "| MODEL_NOISE_OR_POLICY_DRIFT |" in markdown

    payload = json.loads(render_report_json(report))
    assert payload["outcomes"][0]["cause"] == "MODEL_NOISE_OR_POLICY_DRIFT"


def test_reaudit_report_records_served_model_id(tmp_path: Path) -> None:
    """Bulk reaudit reports carry the served model id for each fresh verdict."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )

    served_model = f"{DEFAULT_JUDGE_MODEL}-served-by-fallback"
    with _mock_judge_call(verdict="ACCEPTED", rationale="still genuine", served_model=served_model):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )

    assert len(report.outcomes) == 1
    assert report.outcomes[0].fresh_model_id == served_model
    assert f"model={served_model}" in render_report_text(report)
    assert f"| {served_model} |" in render_report_markdown(report)
    assert json.loads(render_report_json(report))["outcomes"][0]["fresh_model_id"] == served_model


def test_render_markdown_report_snapshot() -> None:
    """3-entry report: one of each major divergence kind.

    The snapshot pins the operator-facing markdown shape: table
    columns, severity-ordered sections, override formatting.
    """
    outcomes = [
        _fixed_outcome(
            key="plugins/drift.py:R1:Foo:bar:fp=aaa",
            divergence=ReauditDivergence.WAS_ACCEPTED_NOW_BLOCKED,
            original_verdict=JudgeVerdict.ACCEPTED,
            fresh_verdict=JudgeVerdict.BLOCKED,
            fresh_rationale="code drifted; boundary no longer applies",
        ),
        _fixed_outcome(
            key="plugins/override.py:R2:Baz:method:fp=bbb",
            divergence=ReauditDivergence.OVERRIDE_NO_LONGER_NEEDED,
            original_verdict=JudgeVerdict.OVERRIDDEN_BY_OPERATOR,
            original_model_verdict=JudgeVerdict.BLOCKED,
            fresh_verdict=JudgeVerdict.ACCEPTED,
            fresh_rationale="model now agrees with the boundary",
        ),
        _fixed_outcome(
            key="plugins/stable.py:R1:Widget:lookup:fp=ccc",
            divergence=ReauditDivergence.STILL_AGREES,
            original_verdict=JudgeVerdict.ACCEPTED,
            fresh_verdict=JudgeVerdict.ACCEPTED,
            fresh_rationale="still a genuine Tier-3 boundary",
        ),
    ]
    report = ReauditReport.from_outcomes(outcomes)
    rendered = render_report_markdown(report)

    # Header + summary table.
    assert rendered.startswith("# Reaudit report\n")
    assert "| Divergence | Count |" in rendered
    assert "| WAS_ACCEPTED_NOW_BLOCKED | 1 |" in rendered
    assert "| OVERRIDE_NO_LONGER_NEEDED | 1 |" in rendered
    assert "| STILL_AGREES | 1 |" in rendered

    # Per-divergence sections in severity order.
    pos_was = rendered.index("## WAS_ACCEPTED_NOW_BLOCKED")
    pos_override = rendered.index("## OVERRIDE_NO_LONGER_NEEDED")
    pos_still = rendered.index("## STILL_AGREES")
    assert pos_was < pos_override < pos_still

    # Override row shows underlying model verdict.
    assert "OVERRIDDEN_BY_OPERATOR (model: BLOCKED)" in rendered

    # Entry keys are inline-coded.
    assert "`plugins/drift.py:R1:Foo:bar:fp=aaa`" in rendered


def test_render_markdown_escapes_model_supplied_notes() -> None:
    """Tier-3 model rationale text must not render as active markdown."""
    report = ReauditReport.from_outcomes(
        [
            _fixed_outcome(
                key="plugins/injected.py:R1:Widget:lookup:fp=aaa",
                divergence=ReauditDivergence.WAS_ACCEPTED_NOW_BLOCKED,
                original_verdict=JudgeVerdict.ACCEPTED,
                fresh_verdict=JudgeVerdict.BLOCKED,
                fresh_rationale="`code` <script>alert(1)</script> [link](https://evil.example)\x1b[31m | pipe",
            )
        ]
    )
    rendered = render_report_markdown(report)

    assert "\\`code\\`" in rendered
    assert "&lt;script&gt;alert\\(1\\)&lt;/script&gt;" in rendered
    assert "\\[link\\]\\(https://evil.example\\)" in rendered
    assert "\\| pipe" in rendered
    assert "\x1b" not in rendered
    assert "<script>" not in rendered
    assert "[link](https://evil.example)" not in rendered


def test_markdown_report_includes_obsolete_section() -> None:
    outcomes = [
        _fixed_outcome(
            key="plugins/gone.py:R1:Foo:bar:fp=dead",
            divergence=ReauditDivergence.ENTRY_OBSOLETE,
            original_verdict=JudgeVerdict.ACCEPTED,
            fresh_verdict=None,
            fresh_rationale=None,
        ),
    ]
    report = ReauditReport.from_outcomes(outcomes)
    rendered = render_report_markdown(report)
    assert "## ENTRY_OBSOLETE (1)" in rendered
    assert "<no judge call>" in rendered


# =====================================================================
# CLI integration
# =====================================================================


def test_cli_reaudit_text_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )

    argv = [
        "reaudit",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--format",
        "text",
    ]
    with _mock_judge_call(verdict="ACCEPTED", rationale="still good"):
        exit_code = main(argv)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "STILL_AGREES" in captured.out
    assert "Summary:" in captured.out


def test_cli_reaudit_emits_running_cost_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )

    argv = [
        "reaudit",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--max-calls",
        "1",
        "--format",
        "text",
    ]
    with _mock_judge_call(verdict="ACCEPTED", rationale="still good", prompt_tokens=5100, cached_tokens=1200):
        exit_code = main(argv)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "reaudit progress:" in captured.err
    assert "judge_calls=1/1" in captured.err
    assert "prompt_tokens_total=5100" in captured.err
    assert "prompt_tokens_cached=1200" in captured.err
    assert "prompt_tokens_uncached=3900" in captured.err


def test_cli_reaudit_writes_output_file(tmp_path: Path) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )

    out_path = tmp_path / "report.md"
    argv = [
        "reaudit",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--format",
        "markdown",
        "--output",
        str(out_path),
    ]
    with _mock_judge_call(verdict="ACCEPTED", rationale="still good"):
        exit_code = main(argv)
    assert exit_code == 0
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "# Reaudit report" in content


def test_cli_reaudit_missing_api_key_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Without OPENROUTER_API_KEY the run dies cleanly at the first judge call."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )

    argv = [
        "reaudit",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
    ]
    env_without_key = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        exit_code = main(argv)

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "OPENROUTER_API_KEY" in captured.err


def test_cli_reaudit_invalid_since_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "reaudit",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--since",
        "not-a-date",
    ]
    exit_code = main(argv)
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "ISO-8601" in captured.err


def test_cli_reaudit_unsupported_rule_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    argv = [
        "reaudit",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--rule",
        "some.other.rule",
    ]
    exit_code = main(argv)
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "not supported" in captured.err


def test_cli_reaudit_malformed_allowlist_exits_2_without_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Allowlist loader failures are configuration diagnostics, not tracebacks."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    (allowlist_dir / "plugins.yaml").write_text("allow_hits: definitely-not-a-list\n", encoding="utf-8")

    exit_code = main(
        [
            "reaudit",
            "--root",
            str(root),
            "--allowlist-dir",
            str(allowlist_dir),
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "reaudit error" in captured.err
    assert "allow_hits must be a list" in captured.err
    assert "Traceback" not in captured.err


# =====================================================================
# C3-2 / C3-3: per-entry failure isolation + sweep-completeness banner
# (closes elspeth-9a4e54cc01)
# =====================================================================


@contextmanager
def _mock_judge_call_raising(exc: BaseException) -> Iterator[MagicMock]:
    """Patch ``openai.OpenAI`` so the judge call raises ``exc``.

    Used to simulate transport failures inside reaudit's per-entry
    boundary. The exception is raised when the test code invokes
    ``client.chat.completions.create(...)`` — i.e. exactly where a real
    OpenAI SDK call would surface a network / timeout / rate-limit /
    5xx error.
    """
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = exc
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=fake_client) as client_class,
    ):
        yield client_class


def test_c3_2_transport_failure_classifies_judge_call_failed(tmp_path: Path) -> None:
    """A transient OpenAI APIError inside the sweep records JUDGE_CALL_FAILED.

    The exception classname + message survive on the outcome's
    ``fresh_rationale`` for audit, and the entry's
    ``fresh_verdict`` / ``fresh_recorded_at`` remain ``None`` (no
    fresh verdict landed). The surrounding-code snapshot is still
    populated because reaudit extracts it before the failing call.
    """
    from openai import APIConnectionError

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )

    # APIConnectionError ctor requires a ``request`` kwarg — use a real
    # SDK error so the catch in reaudit exercises the actual SDK class
    # hierarchy (APIError is the umbrella; APIConnectionError is one
    # leaf the operator will see in production).
    transient = APIConnectionError(request=MagicMock())
    with _mock_judge_call_raising(transient):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )

    assert len(report.outcomes) == 1
    outcome = report.outcomes[0]
    assert outcome.divergence is ReauditDivergence.JUDGE_CALL_FAILED
    assert outcome.fresh_verdict is None
    assert outcome.fresh_recorded_at is None
    assert outcome.fresh_rationale is not None
    assert "APIConnectionError" in outcome.fresh_rationale
    # The surrounding code was extracted before the failing call.
    assert outcome.code_snapshot != ""
    assert "Widget" in outcome.code_snapshot
    # Sweep itself completed — all dispatched entries produced an
    # outcome (even if that outcome is JUDGE_CALL_FAILED).
    assert report.entries_dispatched == 1
    assert report.total_entries == 1


def test_c3_2_sweep_continues_after_transport_failure(tmp_path: Path) -> None:
    """One failing entry mid-sweep does not abort the surviving entries.

    Three entries; entry 2 raises a transport error; entries 1 and 3
    produce normal STILL_AGREES outcomes. The report carries all three
    outcomes in order, the failing one classified JUDGE_CALL_FAILED.
    """
    import hashlib as _hashlib

    from openai import APITimeoutError

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    finding = _live_widget_finding(root)
    fp = finding.fingerprint
    live_file_fp = _hashlib.sha256((root / "plugins/widget.py").read_bytes()).hexdigest()
    live_ast_path = finding.ast_path
    # Three duplicate entries — same key, different owners. The loader
    # accepts them and reaudit iterates them in YAML order.
    lines: list[str] = ["allow_hits:"]
    for i in range(3):
        entry_key = f"plugins/widget.py:R1:Widget:lookup:fp={fp}"
        lines.append(f"- key: {entry_key}")
        lines.append(f"  owner: agent-{i}")
        lines.append("  reason: |-")
        lines.append("    payload is Tier-3 external data")
        lines.append("  safety: |-")
        lines.append("    bounded coercion")
        lines.append("  expires: '2030-01-01'")
        lines.append("  judge_verdict: ACCEPTED")
        lines.append("  judge_recorded_at: '2024-01-01T00:00:00+00:00'")
        lines.append("  judge_model: claude-opus-4-7")
        lines.append(f"  judge_policy_hash: '{JUDGE_POLICY_HASH}'")
        lines.append("  judge_rationale: |-")
        lines.append(f"    rationale entry {i}")
        lines.append(f"  file_fingerprint: '{live_file_fp}'")
        lines.append(f"  ast_path: '{live_ast_path}'")
        lines.append(
            "  judge_metadata_signature: '"
            + _fixture_judge_metadata_signature(
                key=entry_key,
                file_fingerprint=live_file_fp,
                ast_path=live_ast_path,
                judge_rationale=f"rationale entry {i}",
            )
            + "'"
        )
    (allowlist_dir / "plugins.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Per-call side_effect: entry 1 success, entry 2 raises, entry 3
    # success. Validates that the failure is isolated to the failing
    # entry and the loop continues.
    ok_completion = _mock_openrouter_completion(verdict="ACCEPTED", rationale="ok")
    transient = APITimeoutError(request=MagicMock())
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [ok_completion, transient, ok_completion]

    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=fake_client),
    ):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )

    assert len(report.outcomes) == 3
    assert report.outcomes[0].divergence is ReauditDivergence.STILL_AGREES
    assert report.outcomes[1].divergence is ReauditDivergence.JUDGE_CALL_FAILED
    assert "APITimeoutError" in (report.outcomes[1].fresh_rationale or "")
    assert report.outcomes[2].divergence is ReauditDivergence.STILL_AGREES
    # The full SDK call count was 3 — the loop did not short-circuit
    # after the middle failure.
    assert fake_client.chat.completions.create.call_count == 3
    # Sweep completed in full — all dispatched, no missing.
    assert report.entries_dispatched == 3
    assert report.total_entries == 3


def test_c3_2_judge_configuration_error_still_aborts_sweep(tmp_path: Path) -> None:
    """JudgeConfigurationError (missing API key / SDK) is sweep-fatal.

    Contrast with APIError (caught per-entry). Configuration errors are
    not transient — there is no value in continuing the sweep when the
    API key is absent — so they propagate out of reaudit_entries to be
    surfaced as a CLI exit-2.
    """
    from elspeth_lints.core.judge import JudgeConfigurationError

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )

    # Remove the API key from the environment for the duration of this
    # call so JudgeConfigurationError fires inside call_judge.
    env_without_key = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    with (
        patch.dict(os.environ, env_without_key, clear=True),
        pytest.raises(JudgeConfigurationError),
    ):
        reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )


def test_classification_error_records_outcome_and_continues(tmp_path: Path) -> None:
    """A per-entry classification exception must not abort the whole sweep."""
    import elspeth_lints.core.reaudit as reaudit_module

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=2)

    with (
        _mock_judge_call(verdict="ACCEPTED", rationale="still good"),
        patch.object(
            reaudit_module,
            "_classify_divergence",
            side_effect=[ReauditError("unexpected stored/fresh verdict tuple"), ReauditDivergence.STILL_AGREES],
        ),
    ):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )

    assert len(report.outcomes) == 2
    assert report.outcomes[0].divergence is ReauditDivergence.JUDGE_CLASSIFICATION_FAILED
    assert "unexpected stored/fresh verdict tuple" in (report.outcomes[0].fresh_rationale or "")
    assert report.outcomes[1].divergence is ReauditDivergence.STILL_AGREES
    assert report.entries_dispatched == report.total_entries == 2


def test_duplicate_matching_findings_record_ambiguous_outcome(tmp_path: Path) -> None:
    """Duplicate canonical_key matches are reported, not silently first-picked."""
    import elspeth_lints.core.reaudit as reaudit_module

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    key = f"plugins/widget.py:R1:Widget:lookup:fp={fp}"
    (allowlist_dir / "plugins.yaml").write_text(
        "\n".join(
            [
                "allow_hits:",
                f"- key: {key}",
                "  owner: test-owner",
                "  reason: |-",
                "    duplicate finding ambiguity",
                "  safety: |-",
                "    Suppression gated by cicd-judge; see judge_rationale below.",
                "  expires: '2030-01-01'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    duplicate_a = MagicMock(canonical_key=key, line=5)
    duplicate_b = MagicMock(canonical_key=key, line=6)

    with (
        patch.object(reaudit_module, "_scan_findings_for_file", return_value=[duplicate_a, duplicate_b]),
        _mock_judge_call(verdict="ACCEPTED", rationale="should not be called") as client_class,
    ):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=True,
        )

    assert len(report.outcomes) == 1
    assert report.outcomes[0].divergence is ReauditDivergence.AMBIGUOUS_FINDING_MATCH
    assert "2 finding(s)" in (report.outcomes[0].fresh_rationale or "")
    assert client_class.call_count == 0


def test_c3_3_entries_dispatched_equals_count_on_complete_sweep(tmp_path: Path) -> None:
    """A sweep that processes N entries reports entries_dispatched == N.

    Confirms the dispatch counter increments on the per-entry boundary
    (not just on successful outcomes) and matches total_entries when
    nothing aborts the loop.
    """
    import hashlib as _hashlib

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    finding = _live_widget_finding(root)
    fp = finding.fingerprint
    live_file_fp = _hashlib.sha256((root / "plugins/widget.py").read_bytes()).hexdigest()
    live_ast_path = finding.ast_path
    lines: list[str] = ["allow_hits:"]
    for i in range(5):
        entry_key = f"plugins/widget.py:R1:Widget:lookup:fp={fp}"
        lines.append(f"- key: {entry_key}")
        lines.append(f"  owner: agent-{i}")
        lines.append("  reason: |-")
        lines.append("    payload is Tier-3 external data")
        lines.append("  safety: |-")
        lines.append("    bounded coercion")
        lines.append("  expires: '2030-01-01'")
        lines.append("  judge_verdict: ACCEPTED")
        lines.append("  judge_recorded_at: '2024-01-01T00:00:00+00:00'")
        lines.append("  judge_model: claude-opus-4-7")
        lines.append(f"  judge_policy_hash: '{JUDGE_POLICY_HASH}'")
        lines.append("  judge_rationale: |-")
        lines.append(f"    rationale entry {i}")
        lines.append(f"  file_fingerprint: '{live_file_fp}'")
        lines.append(f"  ast_path: '{live_ast_path}'")
        lines.append(
            "  judge_metadata_signature: '"
            + _fixture_judge_metadata_signature(
                key=entry_key,
                file_fingerprint=live_file_fp,
                ast_path=live_ast_path,
                judge_rationale=f"rationale entry {i}",
            )
            + "'"
        )
    (allowlist_dir / "plugins.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with _mock_judge_call(verdict="ACCEPTED", rationale="ok"):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )

    assert report.entries_dispatched == 5
    assert report.total_entries == 5
    assert _incomplete_sweep_banner_is_absent(report)


def _incomplete_sweep_banner_is_absent(report: ReauditReport) -> bool:
    """Helper: assert the renderers do not emit the banner on a complete sweep."""
    text = render_report_text(report)
    md = render_report_markdown(report)
    js = render_report_json(report)
    return ("INCOMPLETE SWEEP" not in text) and ("INCOMPLETE SWEEP" not in md) and ("incomplete_sweep" not in json.loads(js))


def test_c3_3_incomplete_sweep_banner_renders_on_partial_report() -> None:
    """A ReauditReport with entries_dispatched < total_entries renders banners.

    Simulates the "sweep was killed mid-loop after dispatching K of N"
    case by constructing the report directly: the orchestrator does not
    yet persist partial state (out of scope per the C3-3 plan), so the
    test validates the *renderer* contract that any caller who DOES
    surface a partial-state report will get the operator-visible
    banner. The text, markdown, and JSON outputs all carry the signal.
    """
    outcomes = [
        _fixed_outcome(
            key=f"plugins/a.py:R1:Foo:bar:fp=aaa{i:03d}",
            divergence=ReauditDivergence.STILL_AGREES,
            original_verdict=JudgeVerdict.ACCEPTED,
            fresh_verdict=JudgeVerdict.ACCEPTED,
            fresh_rationale="still good",
        )
        for i in range(3)
    ]
    # Three outcomes recorded, but the sweep planned to dispatch ten —
    # seven entries were never reached.
    report = ReauditReport.from_outcomes(outcomes, entries_dispatched=3, total_entries=10)

    text = render_report_text(report)
    assert "INCOMPLETE SWEEP" in text
    assert "7" in text  # 10 - 3 missing
    assert "10" in text

    md = render_report_markdown(report)
    assert "INCOMPLETE SWEEP" in md
    # The dispatched-vs-planned row is in the summary table.
    assert "3 / 10" in md

    payload = json.loads(render_report_json(report))
    assert payload["entries_dispatched"] == 3
    assert payload["total_entries"] == 10
    assert payload["incomplete_sweep"]["missing"] == 7
    assert "INCOMPLETE SWEEP" in payload["incomplete_sweep"]["message"]


def test_c3_3_judge_call_failed_does_not_trigger_incomplete_sweep(tmp_path: Path) -> None:
    """A sweep that completed but had JUDGE_CALL_FAILED outcomes is NOT incomplete.

    The two failure modes are operator-actionable for different
    reasons: JUDGE_CALL_FAILED says "I tried this entry and the
    transport rejected the call", INCOMPLETE_SWEEP says "I never tried
    these entries at all". The banner must not fire for the former
    (the renderer's banner is reserved for entries the sweep never
    reached).
    """
    from openai import APIConnectionError

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )

    transient = APIConnectionError(request=MagicMock())
    with _mock_judge_call_raising(transient):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )

    assert report.entries_dispatched == report.total_entries == 1
    text = render_report_text(report)
    assert "INCOMPLETE SWEEP" not in text
    # But the JUDGE_CALL_FAILED divergence does surface — it is the
    # operator-actionable signal for this sweep.
    assert "JUDGE_CALL_FAILED" in text


def test_c3_2_cli_exits_1_on_judge_call_failed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI exits 1 when any entry hits JUDGE_CALL_FAILED.

    Distinct from exit 2 (configuration errors) and exit 0 (clean
    sweep, only verdict-change divergences). Operators running reaudit
    in CI need a non-zero exit when data-collection gaps appear, so a
    failed transport doesn't quietly produce a green build.
    """
    from openai import APIConnectionError

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    fp = _live_fingerprint_for_widget(root)
    _write_widget_lookup_entry(
        allowlist_dir,
        source_root=root,
        fingerprint=fp,
        judge_verdict="ACCEPTED",
        judge_recorded_at="2024-01-01T00:00:00+00:00",
    )

    argv = [
        "reaudit",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--format",
        "text",
    ]
    transient = APIConnectionError(request=MagicMock())
    with _mock_judge_call_raising(transient):
        exit_code = main(argv)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "JUDGE_CALL_FAILED" in captured.out


# =====================================================================
# T6b: sidecar JSONL crash-recovery (extends T6 commit 6b33ee5b3,
# closes the remaining failure mode of P0 elspeth-9a4e54cc01)
# =====================================================================


def _write_n_duplicate_entries(allowlist_dir: Path, root: Path, count: int) -> Path:
    """Write ``count`` allowlist entries all targeting Widget.lookup.

    Each entry has a distinct ``owner`` so the loader's de-dup never
    fires; the YAML carries ``count`` distinct rows that all dispatch
    through the judge. Returns the YAML path.
    """
    import hashlib as _hashlib

    finding = _live_widget_finding(root)
    fp = finding.fingerprint
    live_file_fp = _hashlib.sha256((root / "plugins/widget.py").read_bytes()).hexdigest()
    live_ast_path = finding.ast_path
    lines: list[str] = ["allow_hits:"]
    for i in range(count):
        entry_key = f"plugins/widget.py:R1:Widget:lookup:fp={fp}"
        lines.append(f"- key: {entry_key}")
        lines.append(f"  owner: agent-{i}")
        lines.append("  reason: |-")
        lines.append("    payload is Tier-3 external data")
        lines.append("  safety: |-")
        lines.append("    bounded coercion")
        lines.append("  expires: '2030-01-01'")
        lines.append("  judge_verdict: ACCEPTED")
        lines.append("  judge_recorded_at: '2024-01-01T00:00:00+00:00'")
        lines.append("  judge_model: claude-opus-4-7")
        lines.append(f"  judge_policy_hash: '{JUDGE_POLICY_HASH}'")
        lines.append("  judge_rationale: |-")
        lines.append(f"    rationale entry {i}")
        lines.append(f"  file_fingerprint: '{live_file_fp}'")
        lines.append(f"  ast_path: '{live_ast_path}'")
        lines.append(
            "  judge_metadata_signature: '"
            + _fixture_judge_metadata_signature(
                key=entry_key,
                file_fingerprint=live_file_fp,
                ast_path=live_ast_path,
                judge_rationale=f"rationale entry {i}",
            )
            + "'"
        )
    target = allowlist_dir / "plugins.yaml"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _run_cli_reaudit(*, root: Path, allowlist_dir: Path, extra_args: list[str] | None = None) -> int:
    argv = [
        "reaudit",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--format",
        "text",
    ]
    if extra_args:
        argv.extend(extra_args)
    return main(argv)


def _read_sidecar_lines(allowlist_dir: Path, run_id: str) -> list[dict[str, Any]]:
    from elspeth_lints.core.reaudit_sidecar import sidecar_path_for

    sidecar = sidecar_path_for(allowlist_dir, run_id)
    return [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]


def _extract_run_id_from_stderr(stderr: str) -> str:
    import re

    match = re.search(r"run_id=([0-9a-f]{32})", stderr)
    if match is None:
        raise AssertionError(f"could not find run_id in stderr: {stderr!r}")
    return match.group(1)


def test_t6b_sidecar_happy_path_writes_header_outcomes_trailer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A normal sweep writes header + N outcome lines + trailer.

    Verifies the full JSONL structure end-to-end on the CLI surface
    and confirms the in-memory report's dispatch counts match.
    """
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=3)

    with _mock_judge_call(verdict="ACCEPTED", rationale="boundary still genuine"):
        exit_code = _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir)

    assert exit_code == 0
    captured = capsys.readouterr()
    run_id = _extract_run_id_from_stderr(captured.err)
    lines = _read_sidecar_lines(allowlist_dir, run_id)
    assert lines[0]["type"] == "header"
    assert lines[0]["run_id"] == run_id
    assert lines[0]["total_entries"] == 3
    assert lines[0]["schema_version"] == 5
    assert lines[0]["rule_filter"] == "trust_tier.tier_model"
    outcome_lines = [line for line in lines if line["type"] == "outcome"]
    assert len(outcome_lines) == 3
    for outcome_line in outcome_lines:
        assert outcome_line["divergence"] == "STILL_AGREES"
        assert outcome_line["judge_call_attempted"] is True
        assert outcome_line["fresh_prompt_tokens_total"] == 4000
        assert outcome_line["fresh_prompt_tokens_cached"] == 0
        # The entry payload round-tripped with the canonical key
        # produced by the live scanner.
        assert outcome_line["entry"]["key"].startswith("plugins/widget.py:R1:Widget:lookup:fp=")
        assert outcome_line["entry"]["judge_verdict"] == "ACCEPTED"
        assert outcome_line["entry"]["owner"].startswith("agent-")
    assert lines[-1]["type"] == "trailer"
    assert lines[-1]["outcomes_written"] == 3
    assert lines[-1]["run_id"] == run_id


def test_cli_reaudit_max_calls_leaves_resumable_sidecar_without_trailer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A spent call budget stops cleanly but keeps the sweep resumable."""
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=3)

    with _mock_judge_call(verdict="ACCEPTED", rationale="boundary still genuine"):
        exit_code = _run_cli_reaudit(
            root=root,
            allowlist_dir=allowlist_dir,
            extra_args=["--max-calls", "1"],
        )

    assert exit_code == 1
    captured = capsys.readouterr()
    run_id = _extract_run_id_from_stderr(captured.err)
    lines = _read_sidecar_lines(allowlist_dir, run_id)
    assert sum(1 for line in lines if line["type"] == "outcome") == 1
    assert not any(line["type"] == "trailer" for line in lines)
    assert "INCOMPLETE SWEEP" in captured.out

    with _mock_judge_call(verdict="ACCEPTED", rationale="boundary still genuine") as client_class:
        resume_exit = _run_cli_reaudit(
            root=root,
            allowlist_dir=allowlist_dir,
            extra_args=["--resume", run_id],
        )

    assert resume_exit == 0
    assert client_class.return_value.chat.completions.create.call_count == 2
    lines_after_resume = _read_sidecar_lines(allowlist_dir, run_id)
    assert sum(1 for line in lines_after_resume if line["type"] == "outcome") == 3
    assert any(line["type"] == "trailer" for line in lines_after_resume)


def test_t6b_mid_sweep_keyboard_interrupt_leaves_no_trailer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """KeyboardInterrupt during sweep leaves partial sidecar without trailer.

    Simulates SIGINT after entry 3 of 5: the sidecar has header + 3
    outcome lines + NO trailer. --render-incomplete then surfaces the
    partial state via the renderer's INCOMPLETE SWEEP banner.
    """
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=5)

    # Mock the judge so the 4th call raises KeyboardInterrupt — the
    # first 3 entries classify normally, the 4th aborts the loop.
    ok_completion = _mock_openrouter_completion(verdict="ACCEPTED", rationale="ok")
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        ok_completion,
        ok_completion,
        ok_completion,
        KeyboardInterrupt(),
    ]

    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=fake_client),
        pytest.raises(KeyboardInterrupt),
    ):
        _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir)

    captured = capsys.readouterr()
    run_id = _extract_run_id_from_stderr(captured.err)
    lines = _read_sidecar_lines(allowlist_dir, run_id)
    assert lines[0]["type"] == "header"
    outcome_lines = [line for line in lines if line["type"] == "outcome"]
    assert len(outcome_lines) == 3
    trailer_lines = [line for line in lines if line["type"] == "trailer"]
    assert trailer_lines == []

    # --render-incomplete surfaces the partial state.
    render_exit = _run_cli_reaudit(
        root=root,
        allowlist_dir=allowlist_dir,
        extra_args=["--render-incomplete", run_id],
    )
    assert render_exit == 1
    rendered = capsys.readouterr()
    assert "INCOMPLETE SWEEP" in rendered.out
    assert "3" in rendered.out and "5" in rendered.out


def test_t6b_resume_completes_killed_sweep(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--resume after KeyboardInterrupt continues from un-classified entries.

    Three entries; first sweep classifies 1 then aborts; --resume
    classifies the remaining 2 and writes the trailer. The final
    sidecar has 3 outcome lines + trailer.
    """
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=3)

    ok_completion = _mock_openrouter_completion(verdict="ACCEPTED", rationale="ok")
    aborting_client = MagicMock()
    aborting_client.chat.completions.create.side_effect = [ok_completion, KeyboardInterrupt()]
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=aborting_client),
        pytest.raises(KeyboardInterrupt),
    ):
        _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir)

    captured = capsys.readouterr()
    run_id = _extract_run_id_from_stderr(captured.err)
    lines_after_abort = _read_sidecar_lines(allowlist_dir, run_id)
    outcome_count_after_abort = sum(1 for line in lines_after_abort if line["type"] == "outcome")
    assert outcome_count_after_abort == 1
    assert not any(line["type"] == "trailer" for line in lines_after_abort)

    # Resume — provide enough successful responses for the remaining 2.
    resume_client = MagicMock()
    resume_client.chat.completions.create.side_effect = [ok_completion, ok_completion]
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=resume_client),
    ):
        resume_exit = _run_cli_reaudit(
            root=root,
            allowlist_dir=allowlist_dir,
            extra_args=["--resume", run_id],
        )

    assert resume_exit == 0
    # Resume should have called the judge exactly twice (the remaining
    # entries), confirming it did not re-classify the already-done ones.
    assert resume_client.chat.completions.create.call_count == 2

    lines_after_resume = _read_sidecar_lines(allowlist_dir, run_id)
    outcome_count_after_resume = sum(1 for line in lines_after_resume if line["type"] == "outcome")
    assert outcome_count_after_resume == 3
    trailer_lines = [line for line in lines_after_resume if line["type"] == "trailer"]
    assert len(trailer_lines) == 1
    assert trailer_lines[0]["outcomes_written"] == 2  # this resume run wrote 2 (not the unified 3)


def test_t6b_resume_rejects_allowlist_edit_via_header_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Editing the allowlist YAML between sweep and resume crashes the resume.

    Tier-1 honesty: the resumed sweep would re-derive a different
    filtered list, breaking the "skip already-classified" guarantee.
    The hash mismatch fires at validation time.
    """
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=3)

    ok_completion = _mock_openrouter_completion(verdict="ACCEPTED", rationale="ok")
    aborting_client = MagicMock()
    aborting_client.chat.completions.create.side_effect = [ok_completion, KeyboardInterrupt()]
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=aborting_client),
        pytest.raises(KeyboardInterrupt),
    ):
        _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir)
    captured = capsys.readouterr()
    run_id = _extract_run_id_from_stderr(captured.err)

    # Mutate the allowlist YAML — change one owner string. The C8-3
    # binding fields stay valid, so the load itself still succeeds,
    # but the hash drifts.
    yaml_path = allowlist_dir / "plugins.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    yaml_path.write_text(text.replace("agent-1", "agent-1-edited"), encoding="utf-8")

    resume_exit = _run_cli_reaudit(
        root=root,
        allowlist_dir=allowlist_dir,
        extra_args=["--resume", run_id],
    )
    assert resume_exit == 2
    err = capsys.readouterr().err
    assert "hash drift" in err


def test_t6b_fsync_called_per_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each outcome write triggers a flush + os.fsync for durability.

    Without fsync, a SIGKILL between write() and the kernel's writeback
    could swallow recently-appended lines. We spy on os.fsync at the
    sidecar module's namespace and assert it fires at header, every
    outcome, and trailer.
    """
    import os as _os

    from elspeth_lints.core import reaudit_sidecar

    fsync_calls: list[int] = []
    real_fsync = _os.fsync

    def spy_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(reaudit_sidecar.os, "fsync", spy_fsync)

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=2)

    with _mock_judge_call(verdict="ACCEPTED", rationale="ok"):
        exit_code = _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir)
    assert exit_code == 0
    # At least: 1 header + 2 outcomes + 1 trailer = 4 fsync calls.
    assert len(fsync_calls) >= 4


def test_t6b_malformed_judge_json_classifies_judge_call_failed(tmp_path: Path) -> None:
    """A JudgeContractError from _parse_judge_payload is now classified.

    Pre-T6b this would abort the sweep (contract error propagated). The
    T6b widening makes malformed-model-response a per-entry isolation
    point: the entry records JUDGE_CALL_FAILED and the sweep continues.
    """
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=2)

    # First response is malformed (not JSON); second is normal. The
    # loop should record JUDGE_CALL_FAILED for #1 and STILL_AGREES
    # for #2.
    malformed = MagicMock()
    malformed_choice = MagicMock()
    malformed_choice.message.content = "this is not JSON at all"
    malformed.choices = [malformed_choice]
    malformed.usage = MagicMock(
        prompt_tokens=4000,
        prompt_tokens_details=MagicMock(cached_tokens=0),
    )
    ok_completion = _mock_openrouter_completion(verdict="ACCEPTED", rationale="ok")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [malformed, ok_completion]
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=fake_client),
    ):
        report = reaudit_entries(
            root=root.resolve(),
            allowlist_dir=allowlist_dir,
            rule_filter="trust_tier.tier_model",
            since=None,
            limit=None,
            include_pre_judge=False,
        )

    assert len(report.outcomes) == 2
    assert report.outcomes[0].divergence is ReauditDivergence.JUDGE_CALL_FAILED
    rationale_0 = report.outcomes[0].fresh_rationale or ""
    assert "JudgeContractError" in rationale_0
    assert report.outcomes[1].divergence is ReauditDivergence.STILL_AGREES


def test_t6b_runtime_error_outside_judge_call_still_propagates(tmp_path: Path) -> None:
    """A RuntimeError raised by _classify_divergence (registry corruption) propagates.

    The T6b widening of the catch is scoped to ``call_judge()`` only.
    RuntimeErrors raised AFTER the call (during divergence
    classification, scanner dispatch, etc.) must still abort the sweep
    — they signal bugs in our code, not transport failures.

    We simulate this by having call_judge return a verdict triple that
    can't be classified: a BLOCKED prior verdict, which the loader is
    supposed to reject — but if we synthesize it directly on the entry
    via reaudit_entries' filter loop, _classify_divergence will crash
    with ReauditError (a RuntimeError subclass). That crash must
    propagate.
    """
    from elspeth_lints.core.reaudit import _classify_divergence

    # Direct unit-test of the classifier-level invariant: BLOCKED on a
    # persisted entry crashes with ReauditError. This is the runtime
    # error path that must NOT be wrapped into JUDGE_CALL_FAILED.
    with pytest.raises(ReauditError, match="BLOCKED"):
        _classify_divergence(
            entry_verdict=JudgeVerdict.BLOCKED,
            entry_model_verdict=None,
            fresh_verdict=JudgeVerdict.ACCEPTED,
        )


def test_t6c_concurrent_resume_rejected_across_processes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two separate processes attempting --resume on the same run_id: only one wins.

    This is the cross-process counterpart to T6b's flock test (which
    was single-process and only exercised per-OFD flock semantics
    indirectly). The TOCTOU concern T6c closes is: two ``--resume``
    invocations both pass validation (no flock yet), then serialise on
    flock acquisition, then both append outcomes past each other's
    trailers. With validation pulled inside the flock window, only one
    --resume can acquire the lock; the other fails fast.

    Mechanism: subprocess.Popen (NOT multiprocessing) — pytest-xdist
    workers and spawn-context multiprocessing have well-known pickling
    pain around mocks and module-level state. A tiny inline Python
    script that opens the sidecar + flocks it + waits on a sentinel
    file is deterministic and free of those concerns.
    """
    import subprocess
    import sys as _sys
    import textwrap

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=3)

    # Pre-create an incomplete sidecar by running a sweep that aborts
    # mid-loop. This gives us a real run_id for --resume to target.
    ok_completion = _mock_openrouter_completion(verdict="ACCEPTED", rationale="ok")
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [ok_completion, KeyboardInterrupt()]
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=fake_client),
        pytest.raises(KeyboardInterrupt),
    ):
        _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir)
    captured = capsys.readouterr()
    run_id = _extract_run_id_from_stderr(captured.err)

    from elspeth_lints.core.reaudit_sidecar import sidecar_path_for

    sidecar_path = sidecar_path_for(allowlist_dir, run_id)
    sentinel_ready = tmp_path / "child_holding_lock.sentinel"
    sentinel_release = tmp_path / "child_may_release.sentinel"

    # Process A: open sidecar, flock LOCK_EX | LOCK_NB, touch ready
    # sentinel, then block on release sentinel. This faithfully
    # represents the in-lock window the real SidecarWriter holds.
    holder_script = textwrap.dedent(
        f"""\
        import fcntl
        import pathlib
        import time
        path = pathlib.Path({str(sidecar_path)!r})
        ready = pathlib.Path({str(sentinel_ready)!r})
        release = pathlib.Path({str(sentinel_release)!r})
        with path.open("a", encoding="utf-8") as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            ready.touch()
            deadline = time.time() + 30
            while not release.exists() and time.time() < deadline:
                time.sleep(0.05)
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        """
    )
    holder = subprocess.Popen([_sys.executable, "-c", holder_script])
    try:
        # Barrier: wait for child to confirm it holds the lock.
        deadline = time.time() + 10.0
        while not sentinel_ready.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert sentinel_ready.exists(), "subprocess A failed to acquire the sidecar lock"

        # Process B (this process): attempt --resume. The TOCTOU fix
        # means SidecarWriter.__enter__'s flock acquisition is the
        # rejection point — IMMEDIATE, not delayed. The CLI surfaces
        # SidecarConflictError (subclass of ReauditError) via exit 2 +
        # the "locked by another process" stderr message.
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False):
            rejection_start = time.time()
            resume_exit = _run_cli_reaudit(
                root=root,
                allowlist_dir=allowlist_dir,
                extra_args=["--resume", run_id],
            )
            rejection_elapsed = time.time() - rejection_start
        rejection_err = capsys.readouterr().err
        assert resume_exit == 2, f"expected exit 2 (ReauditError), got {resume_exit}"
        # The CLI surfaces the conflict as a ReauditError ("reaudit
        # error: ..."); the underlying class is SidecarConflictError.
        assert "locked by another process" in rejection_err, f"expected lock-conflict message, got: {rejection_err!r}"
        # Lock rejection is fail-fast — LOCK_NB returns EWOULDBLOCK in
        # microseconds. _run_cli_reaudit drives the CLI in-process
        # (no subprocess interpreter cold start), so the upper bound
        # is dominated by argparse + module imports + the actual
        # flock syscall. 0.5s gives ~3 orders of magnitude headroom
        # over the real per-flock cost while still proving "not
        # waiting for the held-lock 30s sleep to release" — the
        # property that fails if LOCK_NB regresses to LOCK_EX.
        assert rejection_elapsed < 0.5, f"rejection took {rejection_elapsed:.2f}s; LOCK_NB should reject immediately, not block"
    finally:
        sentinel_release.touch()
        holder.wait(timeout=10)
    assert holder.returncode == 0, f"holder process failed: returncode={holder.returncode}"

    # Process C: holder is gone; --resume now succeeds (the
    # sidecar is unlocked and validation passes).
    ok_completion_c = _mock_openrouter_completion(verdict="ACCEPTED", rationale="ok")
    fake_client_c = MagicMock()
    fake_client_c.chat.completions.create.return_value = ok_completion_c
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=fake_client_c),
    ):
        success_exit = _run_cli_reaudit(
            root=root,
            allowlist_dir=allowlist_dir,
            extra_args=["--resume", run_id],
        )
    assert success_exit == 0
    final_lines = _read_sidecar_lines(allowlist_dir, run_id)
    assert any(line["type"] == "trailer" for line in final_lines)


def test_t6c_load_sidecar_recovers_truncated_final_outcome(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """SIGKILL between write() and flush() leaves a partial last line — recover prior outcomes.

    Hand-constructs the exact byte-pattern that a SIGKILL between the
    writer's ``write()`` and the next ``flush()+fsync()`` would leave:
    header line + 3 complete outcome lines (each terminated with ``\\n``)
    + a partial 4th outcome (first 30 bytes of the JSON payload, NO
    trailing newline).

    The loader must:
    * NOT raise (this is the recovery surface's whole purpose)
    * Return the 3 prior complete outcomes
    * Write a structured warning to stderr naming the byte offset of
      the partial line and the recovered-outcome count

    --render-incomplete on the recovered sidecar must then surface the
    INCOMPLETE SWEEP banner naming "1 entry in-flight when sweep ended"
    (entries_dispatched=3, total_entries=4).
    """
    from elspeth_lints.core.reaudit_sidecar import load_sidecar, sidecar_path_for

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=4)

    # First, run a normal sweep of 3 entries' worth (using --limit) so
    # we have a real well-formed sidecar (header + 3 outcomes + trailer)
    # built by production code. Then surgically replace the trailer with
    # a partial 4th outcome and update the header's total_entries to 4.
    with _mock_judge_call(verdict="ACCEPTED", rationale="ok"):
        exit_code = _run_cli_reaudit(
            root=root,
            allowlist_dir=allowlist_dir,
            extra_args=["--limit", "3"],
        )
    assert exit_code == 0
    captured = capsys.readouterr()
    run_id = _extract_run_id_from_stderr(captured.err)
    sidecar = sidecar_path_for(allowlist_dir, run_id)

    raw = sidecar.read_text(encoding="utf-8")
    lines_in_order = raw.splitlines(keepends=True)
    # Surgical edit: rewrite the header to advertise 4 total entries
    # (so --render-incomplete will report 3-of-4 dispatched), drop the
    # trailer, and append a partial 4th outcome with NO trailing
    # newline.
    header_payload = json.loads(lines_in_order[0])
    header_payload["total_entries"] = 4
    new_header = json.dumps(header_payload, sort_keys=True, ensure_ascii=False) + "\n"
    outcome_lines = [line for line in lines_in_order if json.loads(line).get("type") == "outcome"]
    assert len(outcome_lines) == 3
    partial_outcome = '{"type": "outcome", "appended_at": "2026-05-24'  # 47 bytes, no newline, deliberately truncated
    new_content = new_header + "".join(outcome_lines) + partial_outcome
    sidecar.write_bytes(new_content.encode("utf-8"))
    assert not new_content.endswith("\n")

    # Recovery: load must return 3 outcomes, no trailer, warning on
    # stderr with byte offset.
    loaded = load_sidecar(sidecar)
    err_capture = capsys.readouterr().err
    assert len(loaded.outcomes) == 3
    assert loaded.trailer is None
    assert "partial final line" in err_capture
    expected_offset = len(new_header.encode("utf-8")) + len("".join(outcome_lines).encode("utf-8"))
    assert f"byte offset {expected_offset}" in err_capture
    assert "3 prior complete outcome(s) recovered" in err_capture

    # --render-incomplete surfaces the partial state — banner fires
    # because entries_dispatched=3 < total_entries=4.
    render_exit = _run_cli_reaudit(
        root=root,
        allowlist_dir=allowlist_dir,
        extra_args=["--render-incomplete", run_id],
    )
    assert render_exit == 1
    rendered = capsys.readouterr().out
    assert "INCOMPLETE SWEEP" in rendered
    assert "3" in rendered and "4" in rendered


def test_t6d_resume_after_sigkill_truncates_partial_line_and_appends_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CRITICAL — operator's natural recovery action preserves prior outcomes.

    Reproduces the exact write-side failure the T6c reviewer identified:
    SIGKILL leaves N complete outcomes + 1 partial line (no trailing
    newline). The T6c read-side correctly recovers the prior N on
    --render-incomplete. But --resume opens mode="a" which positions
    POSIX writes at EOF — mid-partial-line. Without T6d, the first
    appended outcome glues onto the truncated tail, the file ends in
    \\n again, and the next load_sidecar crashes with
    SidecarCorruptError because the glued line is non-final malformed
    JSON — destroying everything the read-side fix preserved.

    With T6d, SidecarWriter.__enter__ truncates the partial last line
    inside the flock window (after on_resume_locked validates the
    header). The resume then appends cleanly past the last newline
    boundary, the next load succeeds, and the final sidecar contains
    the prior complete outcomes plus the resumed ones plus a clean
    trailer.

    Setup:
    * 3 allowlist entries
    * First sweep: classify 1, KeyboardInterrupt aborts (real
      abort, real sidecar state, real production code path)
    * Simulate SIGKILL: open the (already-not-trailered) sidecar and
      append a partial 4th-outcome-prefix WITHOUT a trailing newline.
      This is byte-for-byte what a real SIGKILL between write() and
      flush()+fsync() would leave.
    * Sidecar now: header + 1 complete outcome + partial line, NO
      trailer, NOT ending in \\n.

    Assertions:
    * --resume exits 0 (no SidecarCorruptError)
    * Final sidecar reloads cleanly via load_sidecar (no crash)
    * Outcomes: 1 pre-kill + 2 resumed = 3 (the partial in-flight is
      dropped; resume re-classifies it from scratch)
    * Trailer present
    * NO glued line: the raw bytes contain no occurrence of the
      truncated prefix immediately followed by a fresh outcome JSON
    """
    from elspeth_lints.core.reaudit_sidecar import load_sidecar, sidecar_path_for

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=3)

    # Phase 1: real sweep, classify 1, abort on 2nd judge call.
    ok_completion = _mock_openrouter_completion(verdict="ACCEPTED", rationale="ok")
    aborting_client = MagicMock()
    aborting_client.chat.completions.create.side_effect = [ok_completion, KeyboardInterrupt()]
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=aborting_client),
        pytest.raises(KeyboardInterrupt),
    ):
        _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir)
    captured = capsys.readouterr()
    run_id = _extract_run_id_from_stderr(captured.err)
    sidecar = sidecar_path_for(allowlist_dir, run_id)

    # Confirm the post-abort state: header + 1 outcome, ends in \n,
    # no trailer.
    after_abort = sidecar.read_bytes()
    assert after_abort.endswith(b"\n")
    after_abort_lines = sidecar.read_text(encoding="utf-8").splitlines()
    assert len(after_abort_lines) == 2  # header + 1 outcome
    assert json.loads(after_abort_lines[0])["type"] == "header"
    assert json.loads(after_abort_lines[1])["type"] == "outcome"

    # Phase 2: simulate the SIGKILL — append a partial 4th-outcome-
    # prefix to the sidecar with NO trailing newline. This is
    # byte-for-byte what a write() of a JSON line that died before
    # flush()+fsync() would leave on disk.
    partial_prefix = b'{"type": "outcome", "appended_at": "2026-05-24'  # 47 bytes, no \n
    with sidecar.open("ab") as fp:
        fp.write(partial_prefix)
    after_kill = sidecar.read_bytes()
    assert not after_kill.endswith(b"\n"), "setup precondition: file must end mid-partial-line"
    assert partial_prefix in after_kill

    # Phase 3: --resume. With T6d, this truncates the partial line and
    # appends cleanly. Without T6d, the first resumed write glues
    # onto the partial prefix and the next reload crashes.
    resume_client = MagicMock()
    resume_client.chat.completions.create.side_effect = [ok_completion, ok_completion]
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=resume_client),
    ):
        resume_exit = _run_cli_reaudit(
            root=root,
            allowlist_dir=allowlist_dir,
            extra_args=["--resume", run_id],
        )
    assert resume_exit == 0, f"--resume must succeed after partial-line truncation (got exit {resume_exit})"
    assert resume_client.chat.completions.create.call_count == 2, (
        "resume must re-classify the 2 unclassified entries (the dropped partial is one of them)"
    )

    # Phase 4: the final sidecar reloads cleanly — this is the
    # property that fails without T6d.
    loaded = load_sidecar(sidecar)
    assert len(loaded.outcomes) == 3, f"expected 3 outcomes (1 pre-kill + 2 resumed); got {len(loaded.outcomes)}"
    assert loaded.trailer is not None, "trailer must be written after successful resume"

    # Phase 5: assert NO glued line in the raw bytes — the partial
    # prefix must NOT appear immediately followed by a fresh JSON open-
    # brace (which would be the diagnostic for the bug). Specifically,
    # the bytes "appended_at\": \"2026-05-24{" would only exist if the
    # truncation didn't happen.
    final_bytes = sidecar.read_bytes()
    glued_signature = partial_prefix + b"{"
    assert glued_signature not in final_bytes, "partial line was glued onto a fresh outcome — T6d truncation did not fire"
    # Also: the partial prefix should NOT appear anywhere in the
    # truncated file (it was dropped wholesale).
    assert partial_prefix not in final_bytes, "partial prefix bytes survived truncation — T6d truncation point was wrong"


def test_t6d_truncate_is_idempotent_on_clean_sidecar(tmp_path: Path) -> None:
    """A newline-terminated sidecar is byte-identical before/after writer entry.

    The truncation logic is gated on "final byte is not \\n". A clean
    sidecar (every write made it to disk before the next read) must
    skip the truncation entirely — no ftruncate, no fsync, no
    observable side effect on the bytes.
    """
    from elspeth_lints.core.reaudit_sidecar import (
        SidecarHeader,
        SidecarWriter,
        compute_allowlist_hash,
    )

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=3)

    # Build a real clean sidecar via a normal completed sweep.
    with _mock_judge_call(verdict="ACCEPTED", rationale="ok"):
        exit_code = _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir)
    assert exit_code == 0
    sidecar_dir = allowlist_dir / ".reaudit-state"
    [sidecar] = list(sidecar_dir.glob("*.jsonl"))
    run_id = sidecar.stem

    before_bytes = sidecar.read_bytes()
    assert before_bytes.endswith(b"\n"), "clean sidecar precondition: ends in \\n"

    # Drop the trailer so we can enter append-mode (a completed sweep
    # with a trailer cannot be resumed). Re-strip after rebuild so the
    # file still ends in \n.
    text = sidecar.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    non_trailer = [line for line in lines if json.loads(line).get("type") != "trailer"]
    sidecar.write_bytes("".join(non_trailer).encode("utf-8"))
    clean_pre_enter = sidecar.read_bytes()
    assert clean_pre_enter.endswith(b"\n")

    # Enter as a writer in append mode WITHOUT a resume callback (so
    # validation is skipped and we exercise only the truncation
    # gating). The header argument is required by the constructor but
    # is not re-written in append mode.
    header = SidecarHeader(
        run_id=run_id,
        started_at=datetime.now(UTC),
        total_entries=3,
        allowlist_path=str(allowlist_dir),
        allowlist_hash=compute_allowlist_hash(allowlist_dir),
        rule_filter="trust_tier.tier_model",
        since_iso=None,
        limit=None,
        include_pre_judge=False,
    )
    with SidecarWriter(sidecar, header, append=True):
        pass  # enter + exit; truncation gating is the test surface

    after_enter = sidecar.read_bytes()
    assert after_enter == clean_pre_enter, "clean sidecar must be byte-identical after writer entry — truncation must have been a no-op"


def test_t6d_truncation_shrinks_file_to_last_newline_offset(tmp_path: Path) -> None:
    """After truncation, os.path.getsize equals last_newline_offset + 1.

    Verifies the durable size change: the file's inode shrinks to drop
    the partial bytes after the last newline, not merely the
    in-process buffer.
    """
    from elspeth_lints.core.reaudit_sidecar import (
        SidecarHeader,
        SidecarWriter,
        compute_allowlist_hash,
    )

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=3)

    # Real sweep → real sidecar → drop trailer → append partial line.
    with _mock_judge_call(verdict="ACCEPTED", rationale="ok"):
        exit_code = _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir)
    assert exit_code == 0
    sidecar_dir = allowlist_dir / ".reaudit-state"
    [sidecar] = list(sidecar_dir.glob("*.jsonl"))
    run_id = sidecar.stem

    text = sidecar.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    non_trailer = [line for line in lines if json.loads(line).get("type") != "trailer"]
    sidecar.write_bytes("".join(non_trailer).encode("utf-8"))
    expected_size_after_truncate = sidecar.stat().st_size
    # Append partial line.
    partial_prefix = b'{"partial line with no newline'
    with sidecar.open("ab") as fp:
        fp.write(partial_prefix)
    pre_enter_size = sidecar.stat().st_size
    assert pre_enter_size == expected_size_after_truncate + len(partial_prefix)
    # Last newline is at index expected_size_after_truncate - 1; the
    # truncation point is expected_size_after_truncate.
    raw_before = sidecar.read_bytes()
    last_nl = raw_before.rfind(b"\n")
    assert last_nl + 1 == expected_size_after_truncate

    header = SidecarHeader(
        run_id=run_id,
        started_at=datetime.now(UTC),
        total_entries=3,
        allowlist_path=str(allowlist_dir),
        allowlist_hash=compute_allowlist_hash(allowlist_dir),
        rule_filter="trust_tier.tier_model",
        since_iso=None,
        limit=None,
        include_pre_judge=False,
    )
    with SidecarWriter(sidecar, header, append=True):
        # Inside the writer context: the truncation has fired. Don't
        # write anything; we want to observe the post-truncate size
        # before any new append changes it.
        post_truncate_size = sidecar.stat().st_size
        assert post_truncate_size == expected_size_after_truncate, (
            f"post-truncate size {post_truncate_size} != expected last_newline_offset+1 {expected_size_after_truncate}"
        )


def test_t6d_truncation_runs_inside_flock_window(tmp_path: Path) -> None:
    """A second writer attempting to enter while a holder has the lock fails fast — does NOT truncate.

    Constructs a holder process that opens the sidecar and holds
    LOCK_EX | LOCK_NB, then attempts to enter a second SidecarWriter
    in this process. The second writer must hit BlockingIOError on
    its own flock attempt (raising SidecarConflictError) BEFORE
    reaching the truncation logic. Verified by asserting the sidecar
    bytes are byte-identical after the failed attempt.

    The simpler test (this one is in-process to avoid subprocess
    machinery): pre-acquire the lock via a separate fd, then attempt
    SidecarWriter entry. The fcntl semantics treat the two fds as
    independent for LOCK_EX testing — and the operator-visible bug
    we're guarding against is "another reaudit invocation grabbed
    the lock, ours fails to acquire, the partial-line state on disk
    is untouched until our retry."
    """
    from elspeth_lints.core.reaudit_sidecar import (
        SidecarConflictError,
        SidecarHeader,
        SidecarWriter,
        compute_allowlist_hash,
    )

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=3)
    with _mock_judge_call(verdict="ACCEPTED", rationale="ok"):
        exit_code = _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir)
    assert exit_code == 0
    sidecar_dir = allowlist_dir / ".reaudit-state"
    [sidecar] = list(sidecar_dir.glob("*.jsonl"))
    run_id = sidecar.stem

    # Drop trailer + append partial line (mimics post-SIGKILL state).
    text = sidecar.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    non_trailer = [line for line in lines if json.loads(line).get("type") != "trailer"]
    sidecar.write_bytes("".join(non_trailer).encode("utf-8"))
    partial_prefix = b'{"partial line with no newline'
    with sidecar.open("ab") as fp:
        fp.write(partial_prefix)
    pre_attempt_bytes = sidecar.read_bytes()

    # Holder: acquire exclusive flock via a separate fd. Use a
    # separate Python file object (not the SidecarWriter context) so
    # the lock survives across our attempt. fcntl flocks are per-fd
    # in the Linux fcntl-emulating-flock semantics used here; a
    # second open+flock attempt from the same process WILL block.
    # subprocess-based test for the cross-process case lives at
    # test_t6c_concurrent_resume_rejected_across_processes; this
    # test's narrow purpose is "truncation does not happen on the
    # rejection path."
    import subprocess
    import sys as _sys
    import textwrap

    sentinel_ready = tmp_path / "t6d_holder_ready.sentinel"
    sentinel_release = tmp_path / "t6d_holder_release.sentinel"
    holder_script = textwrap.dedent(
        f"""\
        import fcntl
        import pathlib
        import time
        path = pathlib.Path({str(sidecar)!r})
        ready = pathlib.Path({str(sentinel_ready)!r})
        release = pathlib.Path({str(sentinel_release)!r})
        with path.open("a", encoding="utf-8") as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            ready.touch()
            deadline = time.time() + 30
            while not release.exists() and time.time() < deadline:
                time.sleep(0.05)
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        """
    )
    holder = subprocess.Popen([_sys.executable, "-c", holder_script])
    try:
        deadline = time.time() + 10.0
        while not sentinel_ready.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert sentinel_ready.exists(), "holder process failed to acquire the sidecar lock"

        header = SidecarHeader(
            run_id=run_id,
            started_at=datetime.now(UTC),
            total_entries=3,
            allowlist_path=str(allowlist_dir),
            allowlist_hash=compute_allowlist_hash(allowlist_dir),
            rule_filter="trust_tier.tier_model",
            since_iso=None,
            limit=None,
            include_pre_judge=False,
        )
        with pytest.raises(SidecarConflictError), SidecarWriter(sidecar, header, append=True):
            raise AssertionError("writer entry must have raised before reaching the body")

        # The lock-rejected attempt MUST NOT have truncated the file.
        post_attempt_bytes = sidecar.read_bytes()
        assert post_attempt_bytes == pre_attempt_bytes, (
            "lock-rejected writer entry mutated the sidecar — truncation ran outside the flock window"
        )
    finally:
        sentinel_release.touch()
        holder.wait(timeout=10)
    assert holder.returncode == 0


def test_t6d_render_incomplete_does_not_truncate_partial_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--render-incomplete is read-only — the sidecar bytes must be byte-identical after.

    Only the writer truncates; the renderer must surface the partial
    state without mutating disk. This is operator-actionable: the
    operator inspects the killed sweep via --render-incomplete BEFORE
    deciding whether to --resume, and the inspection must not change
    the file under them.
    """
    from elspeth_lints.core.reaudit_sidecar import sidecar_path_for

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=4)

    # Reuse the synthetic-partial-line pattern from the T6c truncation
    # recovery test: real --limit 3 sweep, then surgically modify
    # header total_entries=4, drop trailer, append partial line.
    with _mock_judge_call(verdict="ACCEPTED", rationale="ok"):
        exit_code = _run_cli_reaudit(
            root=root,
            allowlist_dir=allowlist_dir,
            extra_args=["--limit", "3"],
        )
    assert exit_code == 0
    captured = capsys.readouterr()
    run_id = _extract_run_id_from_stderr(captured.err)
    sidecar = sidecar_path_for(allowlist_dir, run_id)

    raw = sidecar.read_text(encoding="utf-8")
    lines_in_order = raw.splitlines(keepends=True)
    header_payload = json.loads(lines_in_order[0])
    header_payload["total_entries"] = 4
    new_header = json.dumps(header_payload, sort_keys=True, ensure_ascii=False) + "\n"
    outcome_lines = [line for line in lines_in_order if json.loads(line).get("type") == "outcome"]
    partial_outcome = '{"type": "outcome", "appended_at": "2026-05-24'
    new_content = (new_header + "".join(outcome_lines) + partial_outcome).encode("utf-8")
    sidecar.write_bytes(new_content)
    pre_render_bytes = sidecar.read_bytes()
    assert not pre_render_bytes.endswith(b"\n"), "setup precondition"

    # --render-incomplete must not mutate the file. (It will fire the
    # T6c partial-line warning during load_sidecar, but that's
    # stderr, not disk state.)
    render_exit = _run_cli_reaudit(
        root=root,
        allowlist_dir=allowlist_dir,
        extra_args=["--render-incomplete", run_id],
    )
    # render-incomplete exits 1 because the rendered sweep is
    # incomplete by definition; that is the existing behaviour.
    assert render_exit == 1
    capsys.readouterr()  # drain stderr (partial-line warning + report)

    post_render_bytes = sidecar.read_bytes()
    assert post_render_bytes == pre_render_bytes, "render-incomplete mutated the sidecar — only writers may truncate"


def test_t6c_load_sidecar_crashes_on_mid_file_corruption(tmp_path: Path) -> None:
    """A non-final corrupt line is unrecoverable — Tier-1 crash.

    The truncation recovery is bounded to the LAST line of a no-trailing-
    newline file. A corrupt line in the middle (well-formed-looking
    surroundings, broken JSON in the middle) means the file was edited
    by hand or bytes were dropped — there is no recovery path; the
    sidecar must crash with ``SidecarCorruptError`` naming the offending
    line and byte offset.
    """
    from elspeth_lints.core.reaudit_sidecar import SidecarCorruptError, load_sidecar, sidecar_path_for

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=3)

    with _mock_judge_call(verdict="ACCEPTED", rationale="ok"):
        exit_code = _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir)
    assert exit_code == 0
    import re

    err_text = ""
    # capsys not requested; pull from sidecar directly.
    # Recover run_id by listing the sidecar dir.
    sidecar_dir = allowlist_dir / ".reaudit-state"
    sidecars = list(sidecar_dir.glob("*.jsonl"))
    assert len(sidecars) == 1
    run_id = sidecars[0].stem
    assert re.match(r"^[0-9a-f]{32}$", run_id), err_text
    sidecar = sidecar_path_for(allowlist_dir, run_id)

    raw = sidecar.read_text(encoding="utf-8")
    lines_in_order = raw.splitlines(keepends=True)
    # Replace the MIDDLE outcome line with broken JSON. File still ends
    # in newline (no truncation signal), so the loader must crash.
    assert len(lines_in_order) >= 4  # header + 3 outcomes + trailer
    lines_in_order[2] = "{this is not valid JSON\n"
    sidecar.write_text("".join(lines_in_order), encoding="utf-8")
    assert raw.endswith("\n") and "".join(lines_in_order).endswith("\n")

    with pytest.raises(SidecarCorruptError) as exc_info:
        load_sidecar(sidecar)
    assert "line 3" in str(exc_info.value)
    assert "byte offset" in str(exc_info.value)
    assert "malformed JSON" in str(exc_info.value)


def test_t6c_compute_allowlist_hash_includes_yml_extension(tmp_path: Path) -> None:
    """A .yml allowlist file is hashed (not just .yaml).

    Operators in the wild author allowlist files under both .yaml and
    .yml suffixes (both are YAML 1.2-permitted). The original glob
    silently excluded .yml, so an operator editing a .yml file would
    have their --resume succeed against a header hash that didn't
    cover the edit — silent corruption of the resumed sweep.
    """
    from elspeth_lints.core.reaudit_sidecar import compute_allowlist_hash

    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    yaml_file = allowlist_dir / "a.yaml"
    yaml_file.write_text("version: 1\n", encoding="utf-8")
    yml_file = allowlist_dir / "b.yml"
    yml_file.write_text("entries: []\n", encoding="utf-8")

    hash_before = compute_allowlist_hash(allowlist_dir)
    yml_file.write_text("entries: [{key: foo}]\n", encoding="utf-8")
    hash_after_yml_edit = compute_allowlist_hash(allowlist_dir)
    assert hash_before != hash_after_yml_edit, (
        "compute_allowlist_hash must hash .yml files — editing a .yml allowlist "
        "file must change the hash, otherwise --resume drift detection silently misses .yml edits"
    )

    yml_file.write_text("entries: []\n", encoding="utf-8")
    yaml_file.write_text("version: 2\n", encoding="utf-8")
    hash_after_yaml_edit = compute_allowlist_hash(allowlist_dir)
    assert hash_before != hash_after_yaml_edit


def test_t6c_completed_sidecar_pruned_after_retention_horizon(tmp_path: Path) -> None:
    """Lazy cleanup: completed sweeps older than retention are deleted at next writer open.

    Incomplete sweeps (no trailer) are recoverable Tier-1 data and MUST
    NOT be deleted regardless of age — only the operator can retire
    them via --resume or by removing the run_id manually.
    """
    from datetime import UTC as _UTC
    from datetime import datetime as _datetime

    from elspeth_lints.core.reaudit_sidecar import (
        COMPLETED_SIDECAR_RETENTION_DAYS,
        SidecarHeader,
        SidecarWriter,
        sidecar_path_for,
    )

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=1)

    # Build one completed sidecar (with trailer) and one incomplete
    # sidecar (no trailer), both via real production code paths.
    with _mock_judge_call(verdict="ACCEPTED", rationale="ok"):
        assert _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir) == 0
    completed_run_id = next(p.stem for p in (allowlist_dir / ".reaudit-state").glob("*.jsonl"))
    completed_sidecar = sidecar_path_for(allowlist_dir, completed_run_id)

    # Force an incomplete sweep alongside.
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = KeyboardInterrupt()
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=fake_client),
        pytest.raises(KeyboardInterrupt),
    ):
        _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir)
    incomplete_run_id = next(p.stem for p in (allowlist_dir / ".reaudit-state").glob("*.jsonl") if p.stem != completed_run_id)
    incomplete_sidecar = sidecar_path_for(allowlist_dir, incomplete_run_id)

    # Age both sidecars past the retention horizon.
    ancient = time.time() - (COMPLETED_SIDECAR_RETENTION_DAYS + 5) * 86400
    os.utime(completed_sidecar, (ancient, ancient))
    os.utime(incomplete_sidecar, (ancient, ancient))

    # Trigger lazy cleanup by entering a fresh SidecarWriter (a new
    # run, distinct run_id). The completed-ancient sidecar must be
    # deleted; the incomplete-ancient one must persist.
    trigger_path = allowlist_dir / ".reaudit-state" / "trigger.jsonl"
    trigger_header = SidecarHeader(
        run_id="trigger",
        started_at=_datetime.now(_UTC),
        total_entries=0,
        allowlist_path=str(allowlist_dir.resolve()),
        allowlist_hash="0" * 64,
        rule_filter="trust_tier.tier_model",
        since_iso=None,
        limit=None,
        include_pre_judge=False,
    )
    with SidecarWriter(trigger_path, trigger_header):
        pass
    trigger_path.unlink()

    assert not completed_sidecar.exists(), "completed sidecar older than COMPLETED_SIDECAR_RETENTION_DAYS must be deleted"
    assert incomplete_sidecar.exists(), "incomplete sidecar must NOT be deleted regardless of age (recoverable Tier-1 data)"


def test_t6c_gitignore_excludes_reaudit_state_directories() -> None:
    """The project .gitignore catches sidecars in tracked config/cicd/ dirs.

    Without a gitignore rule, an operator running `git status` after a
    reaudit sweep would see the sidecar as a new untracked file, and a
    careless `git add -A` would commit operator-local run state into
    the audit trail's git history. Verify the rule actually fires for
    the exact path shape the sidecar uses.
    """
    import subprocess

    repo_root = Path(__file__).resolve().parents[3]
    assert (repo_root / ".gitignore").exists()
    # Use a path inside a tracked config/cicd/ directory so the test
    # exercises the real risk surface.
    candidate = repo_root / "config" / "cicd" / "enforce_tier_model" / ".reaudit-state" / "abc.jsonl"
    result = subprocess.run(
        ["git", "check-ignore", "--verbose", str(candidate.relative_to(repo_root))],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    # git check-ignore exits 0 if the path IS ignored, 1 if not.
    assert result.returncode == 0, (
        f"sidecar path {candidate.relative_to(repo_root)} is NOT covered by .gitignore. stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert ".reaudit-state" in result.stdout


def test_t6b_render_incomplete_on_completed_sweep_still_renders(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--render-incomplete on a cleanly-finished sweep renders the full report.

    Edge case: operator passes --render-incomplete on a run_id that
    actually completed (trailer present). The renderer should still
    work — entries_dispatched == total_entries means no banner, but
    the outcomes still render. Exit is non-zero per the
    "incomplete-by-intent" exit-code policy.
    """
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=2)

    with _mock_judge_call(verdict="ACCEPTED", rationale="ok"):
        assert _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir) == 0
    captured = capsys.readouterr()
    run_id = _extract_run_id_from_stderr(captured.err)

    render_exit = _run_cli_reaudit(
        root=root,
        allowlist_dir=allowlist_dir,
        extra_args=["--render-incomplete", run_id],
    )
    assert render_exit == 1
    rendered = capsys.readouterr().out
    assert "INCOMPLETE SWEEP" not in rendered  # complete sweep → no banner
    assert "STILL_AGREES" in rendered  # but outcomes still render


def test_t6b_resume_on_completed_sweep_refuses(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--resume on a run that already finished crashes with a clear error.

    Without this guard, --resume would re-open the sidecar in append
    mode and write a second (or third) trailer on top of an existing
    one, corrupting the audit record. Refuse instead.
    """
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_n_duplicate_entries(allowlist_dir, root, count=2)

    with _mock_judge_call(verdict="ACCEPTED", rationale="ok"):
        assert _run_cli_reaudit(root=root, allowlist_dir=allowlist_dir) == 0
    captured = capsys.readouterr()
    run_id = _extract_run_id_from_stderr(captured.err)

    resume_exit = _run_cli_reaudit(
        root=root,
        allowlist_dir=allowlist_dir,
        extra_args=["--resume", run_id],
    )
    assert resume_exit == 2
    err = capsys.readouterr().err
    assert "already completed" in err


def test_t6b_sidecar_load_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """A sidecar from a future schema version crashes at load.

    Tier-1: the on-disk format is bound to the running build. A
    schema_version bump must not silently re-interpret old lines or
    forward-compatibly read newer ones — the resume semantics are
    schema-coupled.
    """
    from elspeth_lints.core.reaudit_sidecar import SidecarCorruptError, load_sidecar

    sidecar = tmp_path / "fake.jsonl"
    sidecar.write_text(
        json.dumps(
            {
                "type": "header",
                "schema_version": 999,
                "run_id": "abc",
                "started_at": "2026-05-24T00:00:00+00:00",
                "total_entries": 0,
                "allowlist_path": str(tmp_path),
                "allowlist_hash": "0" * 64,
                "rule_filter": "trust_tier.tier_model",
                "since_iso": None,
                "limit": None,
                "include_pre_judge": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SidecarCorruptError, match="schema_version"):
        load_sidecar(sidecar)


def test_t6b_sidecar_load_rejects_malformed_jsonl(tmp_path: Path) -> None:
    """Bad JSON in any sidecar line crashes the load."""
    from elspeth_lints.core.reaudit_sidecar import SidecarCorruptError, load_sidecar

    sidecar = tmp_path / "bad.jsonl"
    sidecar.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(SidecarCorruptError, match="malformed JSON"):
        load_sidecar(sidecar)
