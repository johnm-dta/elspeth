"""Unit tests for the ``elspeth-lints migrate-judge-scope`` subcommand (SF10).

This command mechanically re-signs currently-VALID v1 (whole-file
``file_fingerprint``) judge-gated allowlist entries as v2 (enclosing-scope
``scope_fingerprint``) WITHOUT re-running the LLM judge. It applies two
independent gates per v1 entry:

* an **integrity gate** that verifies the entry's EXISTING v1
  ``judge_metadata_signature`` with the operator HMAC key (a mismatch is
  tampering and STOPS the whole run), and
* a **relevance gate** that re-locates the suppressed finding by canonical
  key in a fresh scan of the source tree (no live finding ⇒ already stale
  ⇒ refuse-and-continue, "re-justify required").

A passing v1 entry is rewritten in place: the ``file_fingerprint`` line is
replaced with ``judge_signature_version: 2`` + ``scope_fingerprint`` (from
the LIVE finding) and the ``judge_metadata_signature`` is recomputed as a
``hmac-sha256:v2:`` signature. All non-binding audit fields
(``judge_verdict`` / ``judge_rationale`` / ``judge_recorded_at`` / owner /
reason / ...) carry forward BYTE-IDENTICAL.

The HMAC key is a fixture (``"x" * 32``) set via ``monkeypatch.setenv`` —
never the real operator key.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from elspeth_lints.core.allowlist import (
    JudgeVerdict,
    compute_judge_metadata_signature,
    load_allowlist,
    verify_entry_binding_against_finding,
)
from elspeth_lints.core.cli import main
from elspeth_lints.core.judge import JUDGE_POLICY_HASH

# Synthetic source: one R1 finding (dict.get on Tier-2 data) inside
# Widget.lookup. Mirrors the justify/reaudit fixtures.
_SYNTHETIC_SOURCE = '''\
"""Synthetic module used in migrate-judge-scope tests."""


class Widget:
    def lookup(self, payload: dict) -> str:
        return payload.get("name", "anonymous")
'''

# A drifted variant: the suppressed node (Widget.lookup) is gone, so no
# live finding matches the v1 entry's canonical key. Used for the
# already-stale relevance-gate test.
_DRIFTED_SOURCE = '''\
"""Synthetic module — Widget.lookup removed, so the R1 finding is gone."""


class Widget:
    def describe(self) -> str:
        return "no dict.get here"
'''

# A second finding-producing file in the same module dir, so a single
# per-module YAML (plugins.yaml) can carry two entries with DISTINCT
# canonical keys — used to prove tamper-detection is atomic across entries.
_GADGET_SOURCE = '''\
"""Second synthetic module: one R1 finding inside Gadget.fetch."""


class Gadget:
    def fetch(self, payload: dict) -> str:
        return payload.get("id", "unknown")
'''

_FIXTURE_HMAC_KEY = "x" * 32
_JUDGE_RATIONALE = "original judge said the boundary was genuine"
_JUDGE_RECORDED_AT = "2024-01-01T00:00:00+00:00"
_JUDGE_MODEL = "claude-opus-4-7"


@pytest.fixture(autouse=True)
def _set_fixture_hmac_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the operator-side HMAC key as a FIXTURE (never the real key).

    Source-root allowlist loads and the migrate command's integrity gate
    both verify/sign with this key; the real
    ``ELSPETH_JUDGE_METADATA_HMAC_KEY`` is never read.
    """
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _FIXTURE_HMAC_KEY)


# =====================================================================
# Helpers
# =====================================================================


def _build_source_tree(tmp_path: Path, *, source: str = _SYNTHETIC_SOURCE) -> tuple[Path, Path]:
    """Lay out a minimal L3 source root with one finding-producing file."""
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    target = root / "plugins" / "widget.py"
    target.write_text(source, encoding="utf-8")
    return root, target


def _build_allowlist_dir(tmp_path: Path) -> Path:
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "_defaults.yaml").write_text(
        "version: 1\ndefaults:\n  fail_on_stale: false\n  fail_on_expired: false\n",
        encoding="utf-8",
    )
    return allowlist_dir


def _live_widget_finding(root: Path) -> Any:
    """Return the single live R1 ``Finding`` for Widget.lookup at ``root``."""
    from elspeth_lints.rules.trust_tier.tier_model.rule import scan_file

    target_file = (root / "plugins/widget.py").resolve()
    findings = list(scan_file(target_file, root))
    r1_findings = [f for f in findings if f.rule_id == "R1"]
    if len(r1_findings) != 1:
        raise AssertionError(f"expected exactly one R1 finding, got {len(r1_findings)}: {r1_findings}")
    return r1_findings[0]


def _write_valid_v1_entry(
    allowlist_dir: Path,
    *,
    source_root: Path,
    rationale: str = _JUDGE_RATIONALE,
) -> tuple[Path, str]:
    """Write a single VALID v1 (file_fingerprint-bound) Widget.lookup entry.

    "Valid" means: the canonical key matches the live R1 finding, the
    ``file_fingerprint`` matches the live source bytes, and the
    ``judge_metadata_signature`` is a correct v1 HMAC over the metadata.
    Returns (yaml_path, canonical_key).
    """
    entry_lines, key = _render_v1_entry_lines(source_root, "plugins/widget.py", rationale=rationale)
    lines = ["allow_hits:", *entry_lines]
    target = allowlist_dir / "plugins.yaml"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target, key


def _live_finding_for_file(root: Path, rel_path: str) -> Any:
    """Return the single live R1 ``Finding`` for ``rel_path`` under ``root``."""
    from elspeth_lints.rules.trust_tier.tier_model.rule import scan_file

    target_file = (root / rel_path).resolve()
    findings = list(scan_file(target_file, root))
    r1_findings = [f for f in findings if f.rule_id == "R1"]
    if len(r1_findings) != 1:
        raise AssertionError(f"expected exactly one R1 finding in {rel_path}, got {len(r1_findings)}: {r1_findings}")
    return r1_findings[0]


def _render_v1_entry_lines(root: Path, rel_path: str, *, rationale: str = _JUDGE_RATIONALE) -> tuple[list[str], str]:
    """Render the YAML lines for one VALID v1 entry targeting ``rel_path``.

    Returns (lines_without_header, canonical_key). The lines start at the
    ``- key:`` line so callers can concatenate several under one
    ``allow_hits:`` header.
    """
    finding = _live_finding_for_file(root, rel_path)
    key = finding.canonical_key
    if callable(key):
        key = key()
    file_fp = hashlib.sha256((root / rel_path).read_bytes()).hexdigest()
    ast_path = finding.ast_path
    signature = compute_judge_metadata_signature(
        key=key,
        signature_version=1,
        file_fingerprint=file_fp,
        ast_path=ast_path,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime.fromisoformat(_JUDGE_RECORDED_AT),
        judge_model=_JUDGE_MODEL,
        judge_policy_hash=JUDGE_POLICY_HASH,
        judge_rationale=rationale,
        hmac_key=_FIXTURE_HMAC_KEY.encode("utf-8"),
    )
    lines = [
        f"- key: {key}",
        "  owner: test-owner",
        "  reason: |-",
        "    payload is Tier-3 external data from upstream tool-call",
        "  safety: |-",
        "    Suppression gated by cicd-judge; see judge_rationale below.",
        "  expires: '2030-01-01'",
        "  judge_verdict: ACCEPTED",
        f"  judge_recorded_at: '{_JUDGE_RECORDED_AT}'",
        f"  judge_model: {_JUDGE_MODEL}",
        f"  judge_policy_hash: '{JUDGE_POLICY_HASH}'",
        "  judge_rationale: |-",
        f"    {rationale}",
        f"  file_fingerprint: '{file_fp}'",
        f"  ast_path: '{ast_path}'",
        f"  judge_metadata_signature: '{signature}'",
    ]
    return lines, key


def _render_v1_entry_lines_with_bodies(
    root: Path,
    rel_path: str,
    *,
    reason_body: list[str],
    rationale_body: list[str],
) -> tuple[list[str], str]:
    """Render a VALID v1 entry with caller-supplied multi-line block scalars.

    ``reason_body`` (an UNSIGNED field) and ``rationale_body`` (a SIGNED
    field) are emitted as 4-space-indented block-scalar (``|-``) body lines.
    The signature is computed over the rationale joined by newlines, exactly
    as the production loader reconstructs it, so the entry is genuinely valid.
    Used to plant body lines that strip to binding-field prefixes (the C1
    corruption hazard).
    """
    finding = _live_finding_for_file(root, rel_path)
    key = finding.canonical_key
    if callable(key):
        key = key()
    file_fp = hashlib.sha256((root / rel_path).read_bytes()).hexdigest()
    ast_path = finding.ast_path
    rationale = "\n".join(rationale_body)
    signature = compute_judge_metadata_signature(
        key=key,
        signature_version=1,
        file_fingerprint=file_fp,
        ast_path=ast_path,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime.fromisoformat(_JUDGE_RECORDED_AT),
        judge_model=_JUDGE_MODEL,
        judge_policy_hash=JUDGE_POLICY_HASH,
        judge_rationale=rationale,
        hmac_key=_FIXTURE_HMAC_KEY.encode("utf-8"),
    )
    lines = [f"- key: {key}", "  owner: test-owner", "  reason: |-"]
    lines.extend(f"    {body_line}" for body_line in reason_body)
    lines.extend(
        [
            "  safety: |-",
            "    Suppression gated by cicd-judge; see judge_rationale below.",
            "  expires: '2030-01-01'",
            "  judge_verdict: ACCEPTED",
            f"  judge_recorded_at: '{_JUDGE_RECORDED_AT}'",
            f"  judge_model: {_JUDGE_MODEL}",
            f"  judge_policy_hash: '{JUDGE_POLICY_HASH}'",
            "  judge_rationale: |-",
        ]
    )
    lines.extend(f"    {body_line}" for body_line in rationale_body)
    lines.extend(
        [
            f"  file_fingerprint: '{file_fp}'",
            f"  ast_path: '{ast_path}'",
            f"  judge_metadata_signature: '{signature}'",
        ]
    )
    return lines, key


def _entry_binding_key_counts(yaml_text: str, *, entry_key: str) -> dict[str, int]:
    """Count occurrences of each binding key WITHIN one entry's line range.

    Locates the entry by its ``- key:`` line and counts indent-exact
    (2-space) binding-field lines until the next ``- `` entry boundary or a
    new top-level key. Used to assert the surgery never injects a duplicate
    binding key into the entry.
    """
    lines = yaml_text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.rstrip("\r") == f"- key: {entry_key}":
            start = idx
            break
    if start is None:
        raise AssertionError(f"entry {entry_key!r} not found in YAML")
    counts = {"file_fingerprint": 0, "scope_fingerprint": 0, "judge_signature_version": 0, "ast_path": 0, "judge_metadata_signature": 0}
    for line in lines[start + 1 :]:
        if line.startswith("- "):
            break
        if line and not line.startswith(" "):
            break  # new top-level key
        for field in counts:
            if line.startswith(f"  {field}:"):
                counts[field] += 1
    return counts


def _run_migrate(root: Path, allowlist_dir: Path, *, dry_run: bool = False) -> int:
    argv = [
        "migrate-judge-scope",
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--owner",
        "test-operator",
    ]
    if dry_run:
        argv.append("--dry-run")
    return main(argv)


def _valid_rule_ids() -> frozenset[str]:
    from elspeth_lints.rules.trust_tier.tier_model.rule import RULES

    return frozenset(RULES.keys())


# =====================================================================
# Test 1 — happy path: a valid v1 entry is rewritten as v2
# =====================================================================


def test_migrate_rewrites_valid_v1_entry_as_v2(tmp_path: Path) -> None:
    root, _ = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    yaml_path, key = _write_valid_v1_entry(allowlist_dir, source_root=root)

    exit_code = _run_migrate(root, allowlist_dir)
    assert exit_code == 0

    text = yaml_path.read_text(encoding="utf-8")
    assert "judge_signature_version: 2" in text
    assert "scope_fingerprint:" in text
    assert "file_fingerprint:" not in text
    assert "hmac-sha256:v2:" in text
    # Non-binding audit fields carried forward unchanged.
    assert "judge_verdict: ACCEPTED" in text
    assert _JUDGE_RATIONALE in text
    assert f"judge_recorded_at: '{_JUDGE_RECORDED_AT}'" in text

    # The migrated allowlist reloads CLEAN with source_root set — this
    # exercises BOTH the HMAC signature gate and the v1 file-binding gate
    # (the latter is now skipped for the v2 entry; v2's binding is checked
    # at match time below).
    allowlist = load_allowlist(allowlist_dir, valid_rule_ids=_valid_rule_ids(), source_root=root)
    entries = [e for e in allowlist.entries if e.key == key]
    assert len(entries) == 1
    migrated = entries[0]
    assert migrated.judge_signature_version == 2
    assert migrated.scope_fingerprint is not None
    assert migrated.file_fingerprint is None
    assert migrated.judge_verdict is JudgeVerdict.ACCEPTED
    assert migrated.judge_rationale == _JUDGE_RATIONALE

    # The live finding still matches the migrated v2 binding (the gate the
    # next CI run will apply).
    finding = _live_widget_finding(root)
    live_scope_fp = finding.scope_fingerprint
    verify_entry_binding_against_finding(
        migrated,
        file_path="plugins/widget.py",
        ast_path=finding.ast_path,
        scope_fingerprint=live_scope_fp,
    )


# =====================================================================
# Test 2 — already-stale v1 entry is refused (relevance gate)
# =====================================================================


def test_migrate_refuses_and_reports_already_stale_v1_entry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Build the entry against the ORIGINAL source so it is a valid v1 entry,
    # then drift the source so no live finding matches its canonical key.
    root, target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    yaml_path, _ = _write_valid_v1_entry(allowlist_dir, source_root=root)
    before = yaml_path.read_text(encoding="utf-8")

    # Replace the suppressed node — the R1 finding disappears, so the
    # entry's canonical key no longer locates a live finding.
    target.write_text(_DRIFTED_SOURCE, encoding="utf-8")

    exit_code = _run_migrate(root, allowlist_dir)
    assert exit_code != 0

    # Untouched: still v1, still file_fingerprint, byte-identical on disk.
    after = yaml_path.read_text(encoding="utf-8")
    assert after == before
    assert "file_fingerprint:" in after
    assert "judge_signature_version: 2" not in after

    captured = capsys.readouterr()
    report = captured.out + captured.err
    assert "re-justify" in report.lower()


# =====================================================================
# Test 3 — tampered v1 entry is refused as TAMPERING (integrity gate)
# =====================================================================


def test_migrate_refuses_tampered_v1_entry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    yaml_path, _ = _write_valid_v1_entry(allowlist_dir, source_root=root)

    # Tamper a SIGNED field's on-disk text WITHOUT re-signing: rewrite the
    # judge_rationale block scalar. The signature still binds the original
    # rationale, so it no longer verifies. (We do NOT touch file_fingerprint
    # — that is the source-file hash, not a YAML-derived value; editing it
    # would not help an attacker and would muddy the construction.)
    original = yaml_path.read_text(encoding="utf-8")
    tampered = original.replace(
        f"    {_JUDGE_RATIONALE}",
        "    forged: the boundary is fake and this rationale was hand-edited",
    )
    assert tampered != original
    yaml_path.write_text(tampered, encoding="utf-8")

    exit_code = _run_migrate(root, allowlist_dir)
    assert exit_code != 0

    # Untouched: NOT migrated, left byte-identical to the tampered-but-present
    # state (the command must not "fix" tampering by re-signing).
    after = yaml_path.read_text(encoding="utf-8")
    assert after == tampered
    assert "judge_signature_version: 2" not in after
    assert "hmac-sha256:v2:" not in after

    captured = capsys.readouterr()
    report = (captured.out + captured.err).lower()
    assert "tamper" in report


# =====================================================================
# Test 4 — non-binding fields are byte-identical after migration
# =====================================================================


def test_migrate_preserves_non_binding_fields_byte_identical(tmp_path: Path) -> None:
    root, _ = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    yaml_path, _ = _write_valid_v1_entry(allowlist_dir, source_root=root)
    before = yaml_path.read_text(encoding="utf-8")

    exit_code = _run_migrate(root, allowlist_dir)
    assert exit_code == 0
    after = yaml_path.read_text(encoding="utf-8")

    before_lines = before.splitlines()
    after_lines = after.splitlines()

    # Every non-binding line is byte-identical and in the same relative
    # order. The ONLY differences are the three binding lines:
    #   removed: file_fingerprint
    #   added:   judge_signature_version, scope_fingerprint
    #   changed: judge_metadata_signature (v1 -> v2 value)
    def non_binding(lines: list[str]) -> list[str]:
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("file_fingerprint:"):
                continue
            if stripped.startswith("scope_fingerprint:"):
                continue
            if stripped.startswith("judge_signature_version:"):
                continue
            if stripped.startswith("judge_metadata_signature:"):
                continue
            kept.append(line)
        return kept

    assert non_binding(before_lines) == non_binding(after_lines)

    # And the binding surface changed exactly as specified.
    assert "  file_fingerprint:" in before
    assert "  file_fingerprint:" not in after
    assert "  scope_fingerprint:" in after
    assert "  judge_signature_version: 2" in after
    assert "hmac-sha256:v1:" in before
    assert "hmac-sha256:v2:" in after


# =====================================================================
# Test 5 — dry-run reports without writing
# =====================================================================


def test_migrate_dry_run_reports_without_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    yaml_path, key = _write_valid_v1_entry(allowlist_dir, source_root=root)
    before = yaml_path.read_text(encoding="utf-8")

    exit_code = _run_migrate(root, allowlist_dir, dry_run=True)
    # Dry-run found a migratable entry and changed nothing; exit 0 (nothing
    # refused, nothing tampered).
    assert exit_code == 0

    after = yaml_path.read_text(encoding="utf-8")
    assert after == before  # byte-identical: nothing written

    captured = capsys.readouterr()
    report = captured.out + captured.err
    # The report still lists what WOULD migrate.
    assert key in report
    assert "migrat" in report.lower()


# =====================================================================
# Test 6 — tamper-detection is ATOMIC across entries (no partial write)
# =====================================================================


def test_migrate_tampered_entry_leaves_earlier_valid_entry_unwritten(tmp_path: Path) -> None:
    """A tampered entry must abort the run BEFORE any valid entry is written.

    The integrity gate runs as a full first pass over every selected v1
    entry before any re-sign/write. So when entry B (gadget, listed second)
    is tampered, entry A (widget, listed first and perfectly valid) must NOT
    be migrated — the YAML stays byte-identical and the command's "NO entries
    were written" claim is true. Without the two-pass ordering a one-pass
    loop would have written A before discovering B's tampering, leaving the
    allowlist half-migrated and the report lying.
    """
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    (root / "plugins" / "widget.py").write_text(_SYNTHETIC_SOURCE, encoding="utf-8")
    (root / "plugins" / "gadget.py").write_text(_GADGET_SOURCE, encoding="utf-8")
    allowlist_dir = _build_allowlist_dir(tmp_path)

    widget_lines, _ = _render_v1_entry_lines(root, "plugins/widget.py")
    gadget_lines, _ = _render_v1_entry_lines(root, "plugins/gadget.py")
    # Widget entry first (valid), gadget entry second.
    yaml_path = allowlist_dir / "plugins.yaml"
    yaml_path.write_text("\n".join(["allow_hits:", *widget_lines, *gadget_lines]) + "\n", encoding="utf-8")

    # Tamper the SECOND entry's signed rationale text without re-signing.
    original = yaml_path.read_text(encoding="utf-8")
    # Both entries share the same rationale text; tamper only the gadget
    # occurrence (the last one in file order) so the widget entry stays valid.
    idx = original.rindex(f"    {_JUDGE_RATIONALE}")
    tampered = original[:idx] + "    forged: gadget rationale was hand-edited" + original[idx + len(f"    {_JUDGE_RATIONALE}") :]
    assert tampered != original
    yaml_path.write_text(tampered, encoding="utf-8")

    exit_code = _run_migrate(root, allowlist_dir)
    assert exit_code != 0

    # Atomicity: NOTHING was written. The valid widget entry is still v1, and
    # the whole file is byte-identical to the tampered-but-present state.
    after = yaml_path.read_text(encoding="utf-8")
    assert after == tampered
    assert "judge_signature_version: 2" not in after
    assert "hmac-sha256:v2:" not in after
    # The widget entry specifically is untouched (still v1-bound).
    assert "file_fingerprint:" in after


# =====================================================================
# Test 7 — block-scalar body lines that strip to binding prefixes (C1)
# =====================================================================


def test_migrate_does_not_corrupt_block_scalar_body_lines(tmp_path: Path) -> None:
    """A body line that strips to a binding prefix must NOT be rewritten.

    This is the C1 regression: ``str.strip()``-based matching is
    indentation-blind, so a 4-space-indented block-scalar BODY line inside
    ``reason`` (unsigned) or ``judge_rationale`` (signed) that strips to
    ``ast_path:`` / ``file_fingerprint:`` would be misidentified as a binding
    line and rewritten — destroying audit text and injecting a duplicate YAML
    key (which PyYAML accepts last-wins, so it reloads silently when the
    collision lands in an unsigned field). Indent-exact matching on the
    writer's 2-space prefix fixes it.
    """
    root, _ = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    reason_body = [
        "payload is Tier-3 external data from upstream tool-call",
        "ast_path: discussed in PR #42 (this is prose, not a binding field)",
    ]
    rationale_body = [
        "original judge said the boundary was genuine",
        "file_fingerprint: was the wrong word to use here, but it is prose",
    ]
    entry_lines, key = _render_v1_entry_lines_with_bodies(
        root,
        "plugins/widget.py",
        reason_body=reason_body,
        rationale_body=rationale_body,
    )
    yaml_path = allowlist_dir / "plugins.yaml"
    yaml_path.write_text("\n".join(["allow_hits:", *entry_lines]) + "\n", encoding="utf-8")
    before = yaml_path.read_text(encoding="utf-8")

    exit_code = _run_migrate(root, allowlist_dir)
    assert exit_code == 0
    after = yaml_path.read_text(encoding="utf-8")

    # The block-scalar body lines are byte-identical (prose untouched).
    assert "    ast_path: discussed in PR #42 (this is prose, not a binding field)" in after
    assert "    file_fingerprint: was the wrong word to use here, but it is prose" in after

    # The reason/safety/rationale bodies are byte-identical to before (the
    # only changed lines are the three real binding lines at 2-space indent).
    def non_binding(text: str) -> list[str]:
        kept: list[str] = []
        for line in text.splitlines():
            if line.startswith("  file_fingerprint:"):
                continue
            if line.startswith("  scope_fingerprint:"):
                continue
            if line.startswith("  judge_signature_version:"):
                continue
            if line.startswith("  judge_metadata_signature:"):
                continue
            kept.append(line)
        return kept

    assert non_binding(before) == non_binding(after)

    # NO duplicate binding keys were injected into the entry. The real
    # binding fields each appear exactly once; the prose collisions did not
    # mint extra keys.
    counts = _entry_binding_key_counts(after, entry_key=key)
    assert counts["scope_fingerprint"] == 1
    assert counts["judge_signature_version"] == 1
    assert counts["ast_path"] == 1
    assert counts["judge_metadata_signature"] == 1
    assert counts["file_fingerprint"] == 0

    # And it reloads CLEAN with source_root (HMAC + binding gates pass): the
    # signed rationale was preserved byte-identical, so the v2 signature the
    # migration computed over the loaded fields re-verifies.
    allowlist = load_allowlist(allowlist_dir, valid_rule_ids=_valid_rule_ids(), source_root=root)
    migrated = [e for e in allowlist.entries if e.key == key]
    assert len(migrated) == 1
    assert migrated[0].judge_signature_version == 2
    # The reason body (including the colliding prose line) survived intact.
    assert "ast_path: discussed in PR #42" in migrated[0].reason


# =====================================================================
# Test 8 — key-absent fails closed (exit 2, no write)
# =====================================================================


def test_migrate_fails_closed_when_hmac_key_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the operator HMAC key the command must refuse (exit 2, no write).

    Fail-closed is load-bearing: the integrity gate's HMAC recompute and the
    re-sign both require the key. A keyless run must not silently no-op the
    integrity check or write anything.
    """
    root, _ = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    yaml_path, _ = _write_valid_v1_entry(allowlist_dir, source_root=root)
    before = yaml_path.read_text(encoding="utf-8")

    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)

    exit_code = _run_migrate(root, allowlist_dir)
    assert exit_code == 2

    after = yaml_path.read_text(encoding="utf-8")
    assert after == before  # nothing written
    assert "judge_signature_version: 2" not in after


# =====================================================================
# Test 9 — CRLF input is normalized to LF (the toolchain is LF-only)
# =====================================================================


def test_migrate_normalizes_crlf_input_to_lf(tmp_path: Path) -> None:
    """A CRLF-terminated YAML migrates successfully and lands as LF on disk.

    The allowlist toolchain is LF-only by construction: ``atomic_update_text``
    reads via ``Path.read_text`` (universal newlines), so a CRLF input file is
    normalized to LF on read before the surgery runs, and the canonical writer
    emits LF unconditionally. There is no CRLF "preservation" to test — the
    honest contract is CRLF→LF normalization. This pins that a CRLF input is
    exercised end to end (migrates, reloads clean) and that the result is LF.
    """
    root, _ = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    entry_lines, key = _render_v1_entry_lines(root, "plugins/widget.py")
    yaml_path = allowlist_dir / "plugins.yaml"
    # Write the input with CRLF endings.
    crlf_text = "\r\n".join(["allow_hits:", *entry_lines]) + "\r\n"
    yaml_path.write_bytes(crlf_text.encode("utf-8"))

    exit_code = _run_migrate(root, allowlist_dir)
    assert exit_code == 0

    raw = yaml_path.read_bytes()
    # Migration happened.
    assert b"judge_signature_version: 2" in raw
    assert b"hmac-sha256:v2:" in raw
    # Result is LF on disk — no CRLF survives (the toolchain is LF-only).
    assert b"\r\n" not in raw
    assert b"\r" not in raw

    # Reloads clean with source_root (HMAC + binding gates pass).
    allowlist = load_allowlist(allowlist_dir, valid_rule_ids=_valid_rule_ids(), source_root=root)
    migrated = [e for e in allowlist.entries if e.key == key]
    assert len(migrated) == 1
    assert migrated[0].judge_signature_version == 2


# =====================================================================
# Test 10 — per-file mix: one migrates, one stale-refused (I1)
# =====================================================================


def test_migrate_mixed_file_migrates_valid_and_refuses_stale(tmp_path: Path) -> None:
    """In one YAML file: a valid entry migrates, a stale entry is refused.

    Exercises the per-file batch write path with a heterogeneous file. The
    valid widget entry migrates to v2; the gadget entry is stale (its source
    node was removed so no live finding matches) and is refused / left v1.
    Exit is non-zero (a refusal occurred) but the valid entry IS migrated —
    the per-file atomic write applies only the resolved specs.
    """
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    (root / "plugins" / "widget.py").write_text(_SYNTHETIC_SOURCE, encoding="utf-8")
    (root / "plugins" / "gadget.py").write_text(_GADGET_SOURCE, encoding="utf-8")
    allowlist_dir = _build_allowlist_dir(tmp_path)

    widget_lines, widget_key = _render_v1_entry_lines(root, "plugins/widget.py")
    gadget_lines, gadget_key = _render_v1_entry_lines(root, "plugins/gadget.py")
    yaml_path = allowlist_dir / "plugins.yaml"
    yaml_path.write_text("\n".join(["allow_hits:", *widget_lines, *gadget_lines]) + "\n", encoding="utf-8")

    # Drift gadget.py so its entry goes stale (R1 finding disappears), while
    # widget.py stays valid.
    (root / "plugins" / "gadget.py").write_text(
        '"""Gadget.fetch removed — its R1 finding is gone."""\n\n\nclass Gadget:\n    def describe(self) -> str:\n        return "no dict.get"\n',
        encoding="utf-8",
    )

    exit_code = _run_migrate(root, allowlist_dir)
    assert exit_code != 0  # a refusal occurred

    # Load WITHOUT source_root: the stale gadget entry is still v1-bound to the
    # ORIGINAL gadget.py bytes, which no longer match the drifted source, so a
    # source_root load would (correctly) crash on its file_fingerprint gate.
    # We only need to inspect the persisted fields here.
    allowlist = load_allowlist(allowlist_dir, valid_rule_ids=_valid_rule_ids(), source_root=None)
    by_key = {e.key: e for e in allowlist.entries}
    # Widget migrated to v2.
    assert by_key[widget_key].judge_signature_version == 2
    assert by_key[widget_key].scope_fingerprint is not None
    assert by_key[widget_key].file_fingerprint is None
    # Gadget left untouched as v1 (stale → re-justify, not migrated).
    assert by_key[gadget_key].judge_signature_version in (None, 1)
    assert by_key[gadget_key].file_fingerprint is not None
    assert by_key[gadget_key].scope_fingerprint is None
