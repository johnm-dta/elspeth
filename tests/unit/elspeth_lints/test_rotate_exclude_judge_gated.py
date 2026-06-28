"""Primitive-level pin for ``scan_for_rotations(exclude_judge_gated=...)``.

The standalone ``rotate`` CLI must keep its raise-by-design when a judge-gated
entry has positional (fp) drift (``exclude_judge_gated=False``, the default).
The survey/verify paths that feed ``sign-bundle`` instead pass
``exclude_judge_gated=True`` so the read-only scan over the canonical
(mostly judge-gated) corpus never hits the scan-time
``_refuse_rotation_of_judge_gated_entry`` raise. This is a NEW file rather than
an append to the byte-frozen ``test_rotate_tier_model.py``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from elspeth_lints.core.allowlist import JudgeVerdict, compute_judge_metadata_signature
from elspeth_lints.core.judge import JUDGE_POLICY_HASH
from elspeth_lints.rules.trust_tier.tier_model.rotate import scan_for_rotations

_HMAC_KEY = "x" * 32
_RECORDED_AT = "2024-01-01T00:00:00+00:00"
_MODEL = "claude-opus-4-7"
_RATIONALE = "original judge said the boundary was genuine"

# An R1-bearing module. A real module-level statement prepended to it shifts
# the AST body index (the leading ``ast_path`` segment) and therefore the
# ``:fp=`` suffix, while the enclosing scope content stays byte-identical -- the
# canonical AST-position-cascade drift that produces a judge-gated fp shift.
_SOURCE = '''\
"""Synthetic module used in rotation-filter tests."""


class Widget:
    def lookup(self, payload: dict) -> str:
        return payload.get("name", "anonymous")
'''


def _build_source_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    target = root / "plugins" / "widget.py"
    target.write_text(_SOURCE, encoding="utf-8")
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
    from elspeth_lints.rules.trust_tier.tier_model.rule import scan_file

    findings = [f for f in scan_file((root / "plugins/widget.py").resolve(), root) if f.rule_id == "R1"]
    if len(findings) != 1:
        raise AssertionError(f"expected one R1 finding, got {findings!r}")
    return findings[0]


def _canonical_key(finding: Any) -> str:
    key = finding.canonical_key
    if callable(key):
        key = key()
    if not isinstance(key, str):
        raise AssertionError(f"canonical_key must be str, got {type(key).__name__}")
    return key


def _write_judge_gated_v2_entry(allowlist_dir: Path, *, source_root: Path) -> str:
    finding = _live_widget_finding(source_root)
    key = _canonical_key(finding)
    signature = compute_judge_metadata_signature(
        key=key,
        ast_path=finding.ast_path,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime.fromisoformat(_RECORDED_AT),
        judge_model=_MODEL,
        judge_rationale=_RATIONALE,
        judge_policy_hash=JUDGE_POLICY_HASH,
        signature_version=2,
        scope_fingerprint=finding.scope_fingerprint,
        judge_transport="openrouter",
        hmac_key=_HMAC_KEY.encode("utf-8"),
    )
    lines = [
        "allow_hits:",
        f"- key: {key}",
        "  owner: test-owner",
        "  reason: |-",
        "    payload is Tier-3 external data from upstream tool-call",
        "  safety: |-",
        "    Suppression gated by cicd-judge; see judge_rationale below.",
        "  expires: '2030-01-01'",
        "  judge_verdict: ACCEPTED",
        f"  judge_recorded_at: '{_RECORDED_AT}'",
        f"  judge_model: {_MODEL}",
        f"  judge_policy_hash: '{JUDGE_POLICY_HASH}'",
        "  judge_rationale: |-",
        f"    {_RATIONALE}",
        "  judge_signature_version: 2",
        f"  scope_fingerprint: '{finding.scope_fingerprint}'",
        "  judge_transport: openrouter",
        f"  ast_path: '{finding.ast_path}'",
        f"  judge_metadata_signature: '{signature}'",
    ]
    (allowlist_dir / "plugins.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return key


def test_scan_for_rotations_exclude_judge_gated_returns_clean_plan(tmp_path: Path) -> None:
    root, target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    # Bind the judge-gated entry to the original finding, THEN shift the module
    # body so the live finding's fp moves while the enclosing scope is unchanged.
    stale_key = _write_judge_gated_v2_entry(allowlist_dir, source_root=root)
    target.write_text("_SHIM = 1\n\n\n" + _SOURCE, encoding="utf-8")

    # Default (rotate CLI behaviour): the judge-gated fp shift is a rotation
    # candidate, so plan_rotations refuses it at scan time.
    with pytest.raises(RuntimeError, match="refusing to rotate judge-gated"):
        scan_for_rotations(source_root=root, allowlist_path=allowlist_dir, exclude_judge_gated=False)

    # Survey/verify path: the judge-gated entry is filtered out before
    # plan_rotations, so no raise and no rotation for that key.
    plan = scan_for_rotations(source_root=root, allowlist_path=allowlist_dir, exclude_judge_gated=True)
    assert all(rotation.old_key != stale_key for rotation in plan.rotations)
    assert plan.rotations == ()
