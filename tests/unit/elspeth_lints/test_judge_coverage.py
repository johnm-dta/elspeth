"""Adversarial-boundary tests for the new-entry judge-coverage gate (C1).

Convergent panel finding C1: judge enforcement is voluntary in the
prototype because the loader treats all-``None`` ``judge_*`` fields
as honest pre-judge representation. An agent who hand-edits a YAML
entry without judge metadata produces output observationally
identical to a grandfathered pre-judge entry — the audit trail is
silently eroded.

The ``check_judge_coverage`` function closes this loop on a PR diff
against a baseline ref. These tests probe the boundaries of:

1. Rotation grandfathering: an entry whose fingerprint shifted in
   HEAD still matches its baseline counterpart by the
   ``(file, rule, symbol, owner, reason)`` discriminator. The
   policy is operator-confirmed (2026-05-23): rotation = mechanical
   rename, not a re-justification trigger.
2. Genuine new entries lacking judge fields surface as violations.
3. Genuine new entries WITH the atomic quartet are accepted.
4. Pre-judge entries that existed in baseline stay grandfathered
   (not flagged), but the SAME pre-judge entry shape added new in
   this PR is a violation (the gate's whole point).
5. Baseline-file-absent / directory-absent / git-rev-bad failure
   modes produce actionable diagnostics, not silent passes.
6. The ``allow_classes:`` and private-``entries:`` legacy shapes
   are silently skipped (scope discipline).

Test discipline per M5: parameterise invariant violations first.
Each "happy path" assertion is paired with a "what breaks the
contract" probe (rotation policy is the canonical example —
verifying grandfathering is half the story; verifying that a
genuinely-different-owner entry is NOT grandfathered is the other
half).
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from elspeth_lints.core.judge_coverage import (
    JudgeCoverageError,
    JudgeCoverageReport,
    _discriminator,
    _missing_judge_fields,
    check_judge_coverage,
    check_one_directory,
)

# =========================================================================
# Discriminator: rotation policy unit tests (no git, no filesystem)
# =========================================================================


def _make_entry(
    *,
    key: str,
    owner: str = "alice",
    reason: str = "permitted boundary",
    judge_verdict=None,
    judge_recorded_at=None,
    judge_model=None,
    judge_rationale=None,
    judge_model_verdict=None,
):
    """Construct an AllowlistEntry for discriminator/judge-field tests."""
    from elspeth_lints.core.allowlist import AllowlistEntry

    return AllowlistEntry(
        key=key,
        owner=owner,
        reason=reason,
        safety="contained",
        expires=None,
        file_fingerprint=None,
        ast_path=None,
        pattern=None,
        source_file="test.yaml",
        judge_verdict=judge_verdict,
        judge_recorded_at=judge_recorded_at,
        judge_model=judge_model,
        judge_rationale=judge_rationale,
        judge_model_verdict=judge_model_verdict,
    )


def test_discriminator_strips_fingerprint_suffix() -> None:
    """The fp=<hex> segment must not be part of the identity tuple."""
    rotated = _make_entry(key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa")
    same_after_rotate = _make_entry(key="web/x.py:R1:fn:fp=bbbbbbbbbbbbbbbb")
    assert _discriminator(rotated) == _discriminator(same_after_rotate)


def test_discriminator_distinguishes_different_symbols() -> None:
    """Same file + rule but different symbol is a different entry."""
    a = _make_entry(key="web/x.py:R1:fn_a:fp=aaaaaaaaaaaaaaaa")
    b = _make_entry(key="web/x.py:R1:fn_b:fp=aaaaaaaaaaaaaaaa")
    assert _discriminator(a) != _discriminator(b)


def test_discriminator_distinguishes_different_reasons() -> None:
    """Same key but different reason is a different audit assertion.

    Without this, two distinct overrides justifying different aspects
    of the same suppression would collapse — destroying the audit
    signal that the operator wrote them deliberately.
    """
    a = _make_entry(key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa", reason="reason A")
    b = _make_entry(key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa", reason="reason B")
    assert _discriminator(a) != _discriminator(b)


def test_discriminator_distinguishes_different_owners() -> None:
    """Same key but different owner is a different audit assertion."""
    a = _make_entry(key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa", owner="alice")
    b = _make_entry(key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa", owner="bob")
    assert _discriminator(a) != _discriminator(b)


def test_discriminator_normalises_whitespace_in_reason() -> None:
    """Line-wrap reformatting of reason must not look like a new entry.

    YAML's folded-scalar quirks (``>``, ``>-``) can rewrap text on
    save; the discriminator collapses internal whitespace so a
    reformat-only diff is grandfathered.
    """
    a = _make_entry(
        key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa",
        reason="multi\nline   reason",
    )
    b = _make_entry(
        key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa",
        reason="multi line reason",
    )
    assert _discriminator(a) == _discriminator(b)


# =========================================================================
# Missing-fields validation: atomic-quartet invariant probes
# =========================================================================


def test_missing_judge_fields_reports_all_absent_for_pre_judge() -> None:
    """A bare entry with no judge_* fields fails on the full quartet."""
    entry = _make_entry(key="web/x.py:R1:fn:fp=aa")
    assert _missing_judge_fields(entry) == (
        "judge_verdict",
        "judge_recorded_at",
        "judge_model",
        "judge_rationale",
    )


def test_missing_judge_fields_empty_for_complete_entry() -> None:
    """An entry with all atomic-quartet fields passes."""
    from datetime import UTC, datetime

    from elspeth_lints.core.allowlist import JudgeVerdict

    entry = _make_entry(
        key="web/x.py:R1:fn:fp=aa",
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime(2026, 5, 23, tzinfo=UTC),
        judge_model="anthropic/claude-opus-4",
        judge_rationale="rationale",
    )
    assert _missing_judge_fields(entry) == ()


# =========================================================================
# End-to-end: git diff over a real fixture repo
# =========================================================================


def _init_git_fixture(tmp_path: Path) -> Path:
    """Create a minimal repo with config/cicd/enforce_tier_model/."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"],
        check=True,
    )
    enforce_dir = tmp_path / "config" / "cicd" / "enforce_tier_model"
    enforce_dir.mkdir(parents=True)
    return enforce_dir


def _commit(tmp_path: Path, msg: str) -> str:
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", msg],
        check=True,
    )
    result = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_e2e_pre_judge_baseline_with_rotation_is_grandfathered(tmp_path: Path) -> None:
    """An entry whose fingerprint rotated between baseline and HEAD is grandfathered.

    This is the load-bearing rotation-policy assertion. The same
    (file, rule, symbol, owner, reason) entry with a different fp=
    must NOT be flagged as new, because the operator confirmed
    rotation = mechanical rename.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: legitimate boundary
          safety: contained
    """)
    )
    baseline = _commit(tmp_path, "initial: pre-judge entry")

    # Simulate a rotation: same logical entry, fresh fingerprint.
    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:fn:fp=bbbbbbbbbbbbbbbb
          owner: alice
          reason: legitimate boundary
          safety: contained
    """)
    )
    _commit(tmp_path, "rotation: fingerprint shifted")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert report.head_entry_count == 1
    assert report.grandfathered_count == 1
    assert report.new_entry_count == 0
    assert report.violations == ()
    assert report.passes


def test_e2e_new_pre_judge_entry_added_in_pr_is_flagged(tmp_path: Path) -> None:
    """A genuinely-new entry without judge metadata is the C1 violation."""
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:existing:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: existing boundary
          safety: contained
    """)
    )
    baseline = _commit(tmp_path, "initial: one existing entry")

    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:existing:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: existing boundary
          safety: contained
        - key: web/x.py:R5:freshly_added:fp=cccccccccccccccc
          owner: alice
          reason: hand-added without judge run
          safety: contained
    """)
    )
    _commit(tmp_path, "PR: add unjudged entry")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert report.head_entry_count == 2
    assert report.grandfathered_count == 1
    assert report.new_entry_count == 1
    assert len(report.violations) == 1
    assert report.violations[0].entry_key.endswith("freshly_added:fp=cccccccccccccccc")
    assert "judge_verdict" in report.violations[0].missing_fields
    assert not report.passes


def test_e2e_new_entry_with_full_judge_quartet_passes(tmp_path: Path) -> None:
    """A new entry that records the atomic quartet satisfies the gate."""
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")
    baseline = _commit(tmp_path, "initial: empty allowlist")

    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R5:judged:fp=dddddddddddddddd
          owner: alice
          reason: new judged entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-23T12:00:00+00:00'
          judge_model: anthropic/claude-opus-4
          judge_rationale: model reasoned that this boundary is legitimate.
          file_fingerprint: '0000000000000000000000000000000000000000000000000000000000000000'
          ast_path: body[0]
    """)
    )
    _commit(tmp_path, "PR: judged new entry")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert report.head_entry_count == 1
    assert report.grandfathered_count == 0
    assert report.new_entry_count == 1
    assert report.violations == ()
    assert report.passes


def test_e2e_baseline_absent_directory_treats_all_entries_as_new(tmp_path: Path) -> None:
    """A directory added in this PR has no baseline — every entry must be judged."""
    enforce_dir = _init_git_fixture(tmp_path)
    # Initial commit with empty repo state for the enforce_tier_model dir.
    (tmp_path / "README.md").write_text("seed\n")
    baseline = _commit(tmp_path, "initial: just a readme, no enforce dir")

    # Now the directory + entries appear in this PR.
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:fresh:fp=ee
          owner: alice
          reason: new
          safety: contained
    """)
    )
    _commit(tmp_path, "PR: add enforce dir")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert report.grandfathered_count == 0
    assert report.new_entry_count == 1
    assert len(report.violations) == 1
    assert not report.passes


def test_check_judge_coverage_rejects_bad_baseline_ref(tmp_path: Path) -> None:
    """An unknown baseline ref must error, not silently pass.

    The CI agent must never get a green gate because the workflow
    passed a typo for the baseline ref.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")
    _commit(tmp_path, "initial")

    with pytest.raises(JudgeCoverageError) as exc_info:
        check_one_directory(
            allowlist_dir=enforce_dir,
            baseline_ref="nonexistent-ref-xyz",
            repo_root=tmp_path,
        )
    assert "baseline-ref" in str(exc_info.value)


def test_check_judge_coverage_skips_legacy_allow_classes_shape(tmp_path: Path) -> None:
    """Directories using ``allow_classes:`` (private legacy shape) are skipped."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "t@e.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "T"],
        check=True,
    )
    enforce_root = tmp_path / "config" / "cicd"
    legacy = enforce_root / "enforce_audit_evidence_nominal"
    legacy.mkdir(parents=True)
    (legacy / "errors.yaml").write_text(
        textwrap.dedent("""\
        allow_classes:
        - key: x.py:AEN1:Class
          owner: alice
          reason: legacy
          safety: contained
    """)
    )
    baseline = _commit(tmp_path, "initial")

    reports = check_judge_coverage(
        allowlist_root=enforce_root,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    # Legacy-format directory should not appear in the report keys.
    assert "enforce_audit_evidence_nominal" not in reports


def test_check_judge_coverage_root_not_a_dir_errors(tmp_path: Path) -> None:
    """A missing allowlist root produces a clear error, not silent zero-report."""
    with pytest.raises(JudgeCoverageError) as exc_info:
        check_judge_coverage(
            allowlist_root=tmp_path / "no_such_dir",
            baseline_ref="HEAD",
            repo_root=tmp_path,
        )
    assert "not a directory" in str(exc_info.value)


def test_report_passes_returns_true_when_no_violations() -> None:
    """JudgeCoverageReport.passes is the inverse of violations being non-empty."""
    empty = JudgeCoverageReport(
        head_entry_count=0,
        grandfathered_count=0,
        new_entry_count=0,
        violations=(),
    )
    assert empty.passes


# =========================================================================
# C7-4a: OSError during routing must fail closed, not silent-skip
# =========================================================================


def test_directory_routing_oserror_raises_judge_coverage_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreadable YAML file during the allow_hits routing check fails the gate (C7-4a).

    The prior behaviour caught ``OSError`` and silently continued —
    indistinguishable from "this file has no allow_hits". A silent
    skip on read failure means the gate would route the *whole
    directory* out of inspection when only one file was unreadable.
    We surface as ``JudgeCoverageError`` (CLI exit 2 — gate broken).
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")

    from typing import Any

    real_read_text = Path.read_text

    def fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == yaml_path:
            raise OSError("permission denied (simulated)")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    with pytest.raises(JudgeCoverageError) as exc_info:
        check_judge_coverage(
            allowlist_root=tmp_path / "config" / "cicd",
            baseline_ref="HEAD",
            repo_root=tmp_path,
        )
    assert "web.yaml" in str(exc_info.value)
    assert "routing for allow_hits" in str(exc_info.value)


def test_directory_routing_oserror_cli_exit_code_is_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI surfaces the OSError-routing failure as exit 2 (C7-4a)."""
    from elspeth_lints.core.cli import main

    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")
    _commit(tmp_path, "initial")

    from typing import Any

    real_read_text = Path.read_text

    def fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == yaml_path:
            raise OSError("permission denied (simulated)")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    exit_code = main(
        [
            "check-judge-coverage",
            "--allowlist-root",
            str(tmp_path / "config" / "cicd"),
            "--baseline-ref",
            "HEAD",
            "--repo-root",
            str(tmp_path),
        ]
    )
    assert exit_code == 2


# =========================================================================
# C7-4b: CRLF + start-of-file allow_hits detection
# =========================================================================


def test_directory_routing_detects_crlf_line_endings(tmp_path: Path) -> None:
    """A YAML file with CRLF line endings and ``allow_hits:`` mid-file is routed in (C7-4b).

    The prior substring-match for ``"\\nallow_hits:"`` missed CRLF
    (``"\\r\\nallow_hits:"``). A missed routing decision silently
    excludes the whole directory from the gate's purview — the
    operator sees the gate pass on a corpus the gate never actually
    inspected.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    # CRLF line endings with allow_hits NOT at start-of-file.
    yaml_content = "# leading comment\r\nallow_hits:\r\n- key: web/x.py:R1:fn:fp=aa\r\n  owner: alice\r\n  reason: r\r\n  safety: contained\r\n  judge_verdict: ACCEPTED\r\n  judge_recorded_at: '2026-05-23T00:00:00+00:00'\r\n  judge_model: m\r\n  judge_rationale: r\r\n  file_fingerprint: '0000000000000000000000000000000000000000000000000000000000000000'\r\n  ast_path: body[0]\r\n"
    yaml_path.write_bytes(yaml_content.encode("utf-8"))
    baseline = _commit(tmp_path, "initial: CRLF allow_hits")

    reports = check_judge_coverage(
        allowlist_root=tmp_path / "config" / "cicd",
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    # The enforce_tier_model directory MUST have been routed in.
    assert "enforce_tier_model" in reports
    assert reports["enforce_tier_model"].head_entry_count == 1


def test_directory_routing_detects_allow_hits_at_start_of_file(tmp_path: Path) -> None:
    """A YAML file beginning with ``allow_hits:`` at byte 0 is routed in (C7-4b).

    The prior substring-match required a preceding newline. A
    YAML file authored with ``allow_hits:`` as the very first
    token (no header comment, no blank line) would be silently
    excluded from inspection.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    # allow_hits: as the very first bytes of the file.
    yaml_path.write_text(
        "allow_hits:\n"
        "- key: web/x.py:R1:fn:fp=aa\n"
        "  owner: alice\n"
        "  reason: r\n"
        "  safety: contained\n"
        "  judge_verdict: ACCEPTED\n"
        "  judge_recorded_at: '2026-05-23T00:00:00+00:00'\n"
        "  judge_model: m\n"
        "  judge_rationale: r\n"
        "  file_fingerprint: '0000000000000000000000000000000000000000000000000000000000000000'\n"
        "  ast_path: body[0]\n"
    )
    baseline = _commit(tmp_path, "initial: start-of-file allow_hits")

    reports = check_judge_coverage(
        allowlist_root=tmp_path / "config" / "cicd",
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert "enforce_tier_model" in reports
    assert reports["enforce_tier_model"].head_entry_count == 1


# =========================================================================
# C7-4c: baseline parse failure must fail the gate (not silently empty)
# =========================================================================


def test_baseline_parse_failure_raises_judge_coverage_error(tmp_path: Path) -> None:
    """A baseline YAML that doesn't parse fails the gate; it does NOT degrade to empty-baseline (C7-4c).

    The prior behaviour silently caught ``ValueError`` and treated
    the unparseable baseline file as contributing zero entries.
    That converted every HEAD entry to "new", which floods the
    operator with bogus "missing judge metadata" violations and
    hides the real problem (baseline corruption / schema drift).

    Fail-closed: raise ``JudgeCoverageError`` with the offending
    baseline path. The operator fixes the baseline (or picks a
    different ref), not the cascading symptoms.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    # Baseline content is malformed (YAML mapping must be a dict;
    # a scalar string at root is rejected by _load_yaml_strict).
    yaml_path.write_text("just a scalar string at root\n")
    baseline = _commit(tmp_path, "initial: malformed-mapping baseline")

    # HEAD content is a well-formed allow_hits with one entry.
    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:fn:fp=aa
          owner: alice
          reason: legitimate boundary
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-23T00:00:00+00:00'
          judge_model: anthropic/claude-opus-4
          judge_rationale: rationale
          file_fingerprint: '0000000000000000000000000000000000000000000000000000000000000000'
          ast_path: body[0]
    """)
    )
    _commit(tmp_path, "PR: fix baseline shape")

    with pytest.raises(JudgeCoverageError) as exc_info:
        check_one_directory(
            allowlist_dir=enforce_dir,
            baseline_ref=baseline,
            repo_root=tmp_path,
        )
    assert "baseline" in str(exc_info.value)
    assert "web.yaml" in str(exc_info.value)


def test_baseline_yamlerror_raises_judge_coverage_error(tmp_path: Path) -> None:
    """A baseline file with un-tokenisable YAML (yaml.YAMLError) fails the gate (C7-4c + C7-5).

    Combines the baseline-parse-must-fail-closed contract (C7-4c)
    with the YAMLError-not-subclass-of-ValueError fix (C7-5): a
    baseline whose YAML cannot even be tokenised must still produce
    a ``JudgeCoverageError``, not an uncaught traceback.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    # ``:- invalid yaml -:`` is a tokeniser-level error (yaml.YAMLError).
    yaml_path.write_text("key: [unclosed\n")
    baseline = _commit(tmp_path, "initial: tokenizer-error baseline")

    # Replace HEAD with valid content.
    yaml_path.write_text("allow_hits: []\n")
    _commit(tmp_path, "PR: clean head")

    with pytest.raises(JudgeCoverageError) as exc_info:
        check_one_directory(
            allowlist_dir=enforce_dir,
            baseline_ref=baseline,
            repo_root=tmp_path,
        )
    assert "baseline" in str(exc_info.value)
    assert "failed to parse" in str(exc_info.value)


# =========================================================================
# C7-5: yaml.YAMLError catch at HEAD parse path
# =========================================================================


def test_head_yamlerror_raises_judge_coverage_error(tmp_path: Path) -> None:
    """A HEAD file with un-tokenisable YAML raises JudgeCoverageError (C7-5).

    ``yaml.safe_load`` raises ``yaml.YAMLError`` for malformed
    input — a sibling of Exception, NOT a ValueError subclass. The
    prior handler caught only ``ValueError`` and let YAMLError
    escape as a raw traceback (exit-1 "gate broken" via wrong
    channel) rather than the documented exit-2 with a structured
    JudgeCoverageError message.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")
    baseline = _commit(tmp_path, "initial: clean baseline")

    # Replace HEAD with a tokeniser-error file.
    yaml_path.write_text("key: [unclosed\n")
    _commit(tmp_path, "PR: malformed HEAD")

    with pytest.raises(JudgeCoverageError) as exc_info:
        check_one_directory(
            allowlist_dir=enforce_dir,
            baseline_ref=baseline,
            repo_root=tmp_path,
        )
    assert "web.yaml" in str(exc_info.value)
    assert "failed to parse" in str(exc_info.value)


def test_head_yamlerror_cli_exit_code_is_two(tmp_path: Path) -> None:
    """CLI returns exit 2 for HEAD YAMLError (C7-5 surface)."""
    from elspeth_lints.core.cli import main

    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")
    _commit(tmp_path, "initial")
    yaml_path.write_text("key: [unclosed\n")
    _commit(tmp_path, "PR: malformed HEAD")

    exit_code = main(
        [
            "check-judge-coverage",
            "--allowlist-root",
            str(tmp_path / "config" / "cicd"),
            "--baseline-ref",
            "HEAD~1",
            "--repo-root",
            str(tmp_path),
        ]
    )
    assert exit_code == 2
