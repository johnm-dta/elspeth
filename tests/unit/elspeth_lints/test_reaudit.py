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
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth_lints.core.allowlist import AllowlistEntry, JudgeVerdict
from elspeth_lints.core.cli import main
from elspeth_lints.core.reaudit import (
    ReauditDivergence,
    ReauditError,
    ReauditOutcome,
    ReauditReport,
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


def _mock_openrouter_completion(*, verdict: str, rationale: str) -> MagicMock:
    """OpenAI-shape chat-completion mock for OpenRouter routing.

    The judge calls the OpenAI SDK pointed at OpenRouter; this mock
    matches ``client.chat.completions.create(...)``'s return shape —
    ``.choices[0].message.content`` is a JSON string the judge will
    parse, and ``.usage`` carries prompt-token accounting (required;
    the judge reads it offensively for cache-hit telemetry).
    """
    message = MagicMock()
    message.content = json.dumps({"verdict": verdict, "rationale": rationale, "should_use_decorator": None})
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock(
        prompt_tokens=4000,
        prompt_tokens_details=MagicMock(cached_tokens=0),
    )
    return completion


@contextmanager
def _mock_judge_call(*, verdict: str, rationale: str) -> Iterator[MagicMock]:
    """Patch ``openai.OpenAI`` so reaudit's judge call runs offline."""
    fake_completion = _mock_openrouter_completion(verdict=verdict, rationale=rationale)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_completion
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=False),
        patch("openai.OpenAI", return_value=fake_client) as client_class,
    ):
        yield client_class


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
    target = allowlist_dir / "plugins.yaml"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


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
        lines.append(f"- key: plugins/widget.py:R1:Widget:lookup:fp={fp}")
        lines.append(f"  owner: agent-{i}")
        lines.append("  reason: |-")
        lines.append("    payload is Tier-3 external data")
        lines.append("  safety: |-")
        lines.append("    bounded coercion")
        lines.append("  expires: '2030-01-01'")
        lines.append("  judge_verdict: ACCEPTED")
        lines.append("  judge_recorded_at: '2024-01-01T00:00:00+00:00'")
        lines.append("  judge_model: claude-opus-4-7")
        lines.append("  judge_rationale: |-")
        lines.append(f"    rationale entry {i}")
        lines.append(f"  file_fingerprint: '{live_file_fp}'")
        lines.append(f"  ast_path: '{live_ast_path}'")
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
            judge_rationale="original rationale" if original_verdict is not None else None,
        ),
        original_verdict=original_verdict,
        original_model_verdict=original_model_verdict,
        fresh_verdict=fresh_verdict,
        fresh_rationale=fresh_rationale,
        fresh_recorded_at=fixed_dt if fresh_verdict is not None else None,
        divergence=divergence,
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
    assert payload["outcomes"][0]["original_verdict"] == "OVERRIDDEN_BY_OPERATOR"
    assert payload["outcomes"][0]["original_model_verdict"] == "BLOCKED"
    assert payload["outcomes"][0]["fresh_verdict"] == "ACCEPTED"
    # Summary is a list of {divergence, count} dicts, ordered by severity.
    # JUDGE_CALL_FAILED outranks WAS_ACCEPTED_NOW_BLOCKED because a
    # data-collection failure (judge transport rejected the call) is
    # more urgent than any verdict-change signal — the entry simply
    # wasn't re-judged. The first verdict-change divergence (the most
    # urgent operator-actionable verdict signal) is the next slot.
    summary_names = [s["divergence"] for s in payload["summary"]]
    assert summary_names[0] == "JUDGE_CALL_FAILED"
    assert summary_names[1] == "WAS_ACCEPTED_NOW_BLOCKED"


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
        lines.append(f"- key: plugins/widget.py:R1:Widget:lookup:fp={fp}")
        lines.append(f"  owner: agent-{i}")
        lines.append("  reason: |-")
        lines.append("    payload is Tier-3 external data")
        lines.append("  safety: |-")
        lines.append("    bounded coercion")
        lines.append("  expires: '2030-01-01'")
        lines.append("  judge_verdict: ACCEPTED")
        lines.append("  judge_recorded_at: '2024-01-01T00:00:00+00:00'")
        lines.append("  judge_model: claude-opus-4-7")
        lines.append("  judge_rationale: |-")
        lines.append(f"    rationale entry {i}")
        lines.append(f"  file_fingerprint: '{live_file_fp}'")
        lines.append(f"  ast_path: '{live_ast_path}'")
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
        lines.append(f"- key: plugins/widget.py:R1:Widget:lookup:fp={fp}")
        lines.append(f"  owner: agent-{i}")
        lines.append("  reason: |-")
        lines.append("    payload is Tier-3 external data")
        lines.append("  safety: |-")
        lines.append("    bounded coercion")
        lines.append("  expires: '2030-01-01'")
        lines.append("  judge_verdict: ACCEPTED")
        lines.append("  judge_recorded_at: '2024-01-01T00:00:00+00:00'")
        lines.append("  judge_model: claude-opus-4-7")
        lines.append("  judge_rationale: |-")
        lines.append(f"    rationale entry {i}")
        lines.append(f"  file_fingerprint: '{live_file_fp}'")
        lines.append(f"  ast_path: '{live_ast_path}'")
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
