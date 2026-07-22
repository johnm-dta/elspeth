"""From-tree re-verify -- the ``sign-bundle`` linchpin.

``verify_bundle_against_tree`` is the all-or-nothing gate: it re-derives every
binding from the *current* source and refuses (``ok is False``) on any staleness
mismatch, before a single write. Every action ``kind`` in the vocabulary has its
own from-tree verify rule (``new_judgment`` / ``drift_repair`` / ``rotation`` /
``stale_delete``), so a stale action of *any* kind makes the whole bundle fail.

Fixtures are replicated locally (rather than imported from
``test_judge_signature_diagnosis``) because there is no
``tests/unit/elspeth_lints`` package and no precedent for cross-test imports.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from elspeth_lints.core.allowlist import JudgeVerdict, compute_judge_metadata_signature
from elspeth_lints.core.bundle_verify import verify_bundle_against_tree
from elspeth_lints.core.judge import JUDGE_POLICY_HASH
from elspeth_lints.core.review_bundle import BundleAction, ReviewBundle
from elspeth_lints.rules.trust_tier.tier_model.rotate import identity_prefix

_HMAC_KEY = "x" * 32
_RECORDED_AT = "2024-01-01T00:00:00+00:00"
_MODEL = "claude-opus-4-7"
_RATIONALE = "original judge said the boundary was genuine"


@pytest.fixture(autouse=True)
def _keyless(monkeypatch: pytest.MonkeyPatch) -> None:
    # Verify is keyless: diagnosis runs shape-only (binding checks still fire).
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)


def _src(doc: str, *, active: bool = True) -> str:
    body = '        return payload.get("name", "anonymous")' if active else '        return "anonymous"'
    return f'"""{doc}"""\n\n\nclass Widget:\n    def lookup(self, payload: dict) -> str:\n{body}\n'


def _build_root(tmp_path: Path) -> Path:
    root = tmp_path / "src_root"
    (root / "plugins").mkdir(parents=True)
    return root


def _write_source(root: Path, rel: str, doc: str, *, active: bool = True) -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_src(doc, active=active), encoding="utf-8")
    return target


def _build_allowlist_dir(tmp_path: Path) -> Path:
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "_defaults.yaml").write_text(
        "version: 1\ndefaults:\n  fail_on_stale: false\n  fail_on_expired: false\n",
        encoding="utf-8",
    )
    return allowlist_dir


def _live_finding(root: Path, rel: str) -> Any:
    from elspeth_lints.rules.trust_tier.tier_model.rule import scan_file

    findings = [f for f in scan_file((root / rel).resolve(), root) if f.rule_id == "R1"]
    if len(findings) != 1:
        raise AssertionError(f"expected one R1 finding in {rel}, got {findings!r}")
    return findings[0]


def _canonical_key(finding: Any) -> str:
    key = finding.canonical_key
    if callable(key):
        key = key()
    if not isinstance(key, str):
        raise AssertionError(f"canonical_key must be str, got {type(key).__name__}")
    return key


def _write_signed_v2_entry(
    allowlist_dir: Path,
    yaml_name: str,
    *,
    finding: Any,
    scope_fingerprint: str | None = None,
) -> str:
    key = _canonical_key(finding)
    stored_scope = finding.scope_fingerprint if scope_fingerprint is None else scope_fingerprint
    signature = compute_judge_metadata_signature(
        key=key,
        ast_path=finding.ast_path,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime.fromisoformat(_RECORDED_AT),
        judge_model=_MODEL,
        judge_rationale=_RATIONALE,
        judge_policy_hash=JUDGE_POLICY_HASH,
        signature_version=2,
        scope_fingerprint=stored_scope,
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
        f"  scope_fingerprint: '{stored_scope}'",
        "  judge_transport: openrouter",
        f"  ast_path: '{finding.ast_path}'",
        f"  judge_metadata_signature: '{signature}'",
    ]
    (allowlist_dir / yaml_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return key


def _write_pre_judge_entry(allowlist_dir: Path, yaml_name: str, *, key: str) -> str:
    lines = [
        "allow_hits:",
        f"- key: {key}",
        "  owner: test-owner",
        "  reason: |-",
        "    payload is Tier-3 external data from upstream tool-call",
        "  safety: |-",
        "    suppression",
        "  expires: '2030-01-01'",
    ]
    (allowlist_dir / yaml_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return key


def _bundle(root: Path, allowlist_dir: Path, actions: tuple[BundleAction, ...]) -> ReviewBundle:
    return ReviewBundle(
        bundle_id="verify-bundle",
        schema_version=1,
        created_at="2026-06-28T00:00:00+00:00",
        staged_by="agent-x",
        root=str(root),
        allowlist_dir=str(allowlist_dir),
        source_rev=None,
        source_dirty=False,
        actions=actions,
    )


def _new_judgment_action(finding: Any, rel: str) -> BundleAction:
    key = _canonical_key(finding)
    return BundleAction(
        lane="new_judgment",
        kind="justify",
        key=key,
        file_path=rel,
        symbol="Widget.lookup",
        rule="R1",
        fingerprint=key.rsplit(":fp=", 1)[1],
        draft_rationale="payload is Tier-3 external data",
    )


def test_verify_passes_when_claims_match_tree(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    # drift_repair: a signed entry whose stored scope no longer matches the tree.
    _write_source(root, "plugins/widget.py", "widget")
    drift_finding = _live_finding(root, "plugins/widget.py")
    drift_key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=drift_finding, scope_fingerprint="b" * 64)
    # new_judgment: a live, uncovered finding (no allowlist entry).
    _write_source(root, "plugins/gadget.py", "gadget")
    gadget_finding = _live_finding(root, "plugins/gadget.py")

    bundle = _bundle(
        root,
        allowlist_dir,
        (
            BundleAction(lane="resign", kind="drift_repair", key=drift_key, diagnosis_status="SCOPE_BINDING_DRIFT"),
            _new_judgment_action(gadget_finding, "plugins/gadget.py"),
        ),
    )

    report = verify_bundle_against_tree(bundle, root=root, allowlist_dir=allowlist_dir)
    assert report.ok is True
    assert report.mismatches == ()
    assert report.rotation_plan is None  # no rotation action


def test_verify_passes_with_cli_style_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented sign-bundle defaults are relative to the repository root."""
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/gadget.py", "gadget")
    finding = _live_finding(root, "plugins/gadget.py")
    bundle = _bundle(root, allowlist_dir, (_new_judgment_action(finding, "plugins/gadget.py"),))

    monkeypatch.chdir(tmp_path)
    report = verify_bundle_against_tree(
        bundle,
        root=root.relative_to(tmp_path),
        allowlist_dir=allowlist_dir.relative_to(tmp_path),
    )

    assert report.ok is True
    assert report.mismatches == ()


def test_verify_scans_each_new_judgment_file_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Large bundles commonly carry hundreds of findings from one source file."""
    from elspeth_lints.core import bundle_verify
    from elspeth_lints.rules.trust_tier.tier_model.rule import scan_file

    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    target = root / "plugins/gadget.py"
    target.write_text(
        "class Widget:\n"
        "    def lookup(self, payload: dict) -> str:\n"
        "        return payload.get('name', 'anonymous')\n\n"
        "    def lookup_other(self, payload: dict) -> str:\n"
        "        return payload.get('other', 'anonymous')\n",
        encoding="utf-8",
    )
    findings = [finding for finding in scan_file(target.resolve(), root) if finding.rule_id == "R1"]
    assert len(findings) == 2
    bundle = _bundle(
        root,
        allowlist_dir,
        tuple(_new_judgment_action(finding, "plugins/gadget.py") for finding in findings),
    )

    real_scan = bundle_verify.scan_single_file_findings
    scan_calls = 0

    def counted_scan(*, target_file: Path, root: Path) -> list[Any]:
        nonlocal scan_calls
        scan_calls += 1
        return real_scan(target_file=target_file, root=root)

    monkeypatch.setattr(bundle_verify, "scan_single_file_findings", counted_scan)
    report = verify_bundle_against_tree(bundle, root=root, allowlist_dir=allowlist_dir)

    assert report.ok is True
    assert scan_calls == 1


def test_verify_aborts_on_drift_mismatch(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/widget.py", "widget")
    finding = _live_finding(root, "plugins/widget.py")
    # Tree reports SCOPE_BINDING_DRIFT (scope content changed)...
    key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=finding, scope_fingerprint="b" * 64)
    # ...but the bundle claims it is only positional AST drift.
    bundle = _bundle(
        root,
        allowlist_dir,
        (BundleAction(lane="resign", kind="drift_repair", key=key, diagnosis_status="AST_PATH_BINDING_DRIFT"),),
    )

    report = verify_bundle_against_tree(bundle, root=root, allowlist_dir=allowlist_dir)
    assert report.ok is False
    joined = " ".join(report.mismatches)
    assert "AST_PATH_BINDING_DRIFT" in joined
    assert "SCOPE_BINDING_DRIFT" in joined


def test_verify_aborts_on_vanished_new_judgment_finding(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/gadget.py", "gadget")
    finding = _live_finding(root, "plugins/gadget.py")
    action = _new_judgment_action(finding, "plugins/gadget.py")
    # Edit the source so the staged finding no longer exists.
    _write_source(root, "plugins/gadget.py", "gadget", active=False)

    bundle = _bundle(root, allowlist_dir, (action,))
    report = verify_bundle_against_tree(bundle, root=root, allowlist_dir=allowlist_dir)
    assert report.ok is False
    assert any(action.key in m for m in report.mismatches)


def test_verify_passes_when_new_judgment_finding_became_covered(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/gadget.py", "gadget")
    finding = _live_finding(root, "plugins/gadget.py")
    action = _new_judgment_action(finding, "plugins/gadget.py")
    # An allowlist entry has since come to cover the same key (covered between
    # stage and fire). Verify keys off finding existence, not the coverage delta.
    _write_signed_v2_entry(allowlist_dir, "gadget.yaml", finding=finding)

    bundle = _bundle(root, allowlist_dir, (action,))
    report = verify_bundle_against_tree(bundle, root=root, allowlist_dir=allowlist_dir)
    assert report.ok is True


def test_verify_aborts_on_reappeared_stale_delete_finding(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/widget.py", "widget")
    finding = _live_finding(root, "plugins/widget.py")
    # A live finding still covers this key (diagnose reports it OK), so deleting
    # it would drop a live covered entry.
    key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=finding)

    bundle = _bundle(
        root,
        allowlist_dir,
        (BundleAction(lane="resign", kind="stale_delete", key=key, source_file="widget.yaml"),),
    )
    report = verify_bundle_against_tree(bundle, root=root, allowlist_dir=allowlist_dir)
    assert report.ok is False
    assert any(key in m for m in report.mismatches)


def test_verify_aborts_on_rotation_no_longer_applicable(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/gadget.py", "gadget")
    finding = _live_finding(root, "plugins/gadget.py")
    # A non-judge-gated entry that MATCHES the live finding (unchanged -> not a
    # rotation candidate). Staging a rotation for it is no longer applicable.
    key = _write_pre_judge_entry(allowlist_dir, "gadget.yaml", key=_canonical_key(finding))

    bundle = _bundle(
        root,
        allowlist_dir,
        (BundleAction(lane="resign", kind="rotation", key=key, source_file="gadget.yaml"),),
    )
    report = verify_bundle_against_tree(bundle, root=root, allowlist_dir=allowlist_dir)
    assert report.ok is False
    assert report.rotation_plan is not None
    assert any(key in m for m in report.mismatches)


def test_verify_does_not_crash_on_judge_gated_fp_shift_in_scanned_dir(tmp_path: Path) -> None:
    """Verify-site half of the rotation-crash fix (mirrors stage_scan onto the
    SECOND ``scan_for_rotations`` call site).

    Places a judge-gated, fp-SHIFTED entry (the AST-position cascade: a real
    statement prepended so the leading ``ast_path`` index and ``:fp=`` shift
    while the enclosing scope stays byte-identical) as a NON-action in the
    scanned dir, coexisting with a normal non-judge-gated ``rotation`` action.
    Against the pre-fix unfiltered whole-dir scan this would ``RuntimeError`` at
    ``plan_rotations`` regardless of bundle membership; the fixed
    ``exclude_judge_gated=True`` scan filters it out first, so verify must NOT
    raise. The non-action placement keeps the rotation assertions intact.
    """
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    # (1) Judge-gated fp-shifted NON-action entry.
    widget = _write_source(root, "plugins/widget.py", "widget")
    widget_finding = _live_finding(root, "plugins/widget.py")
    _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=widget_finding)
    widget.write_text("_SHIM = 1\n\n\n" + _src("widget"), encoding="utf-8")  # shift -> fp drift

    # (2) Non-judge-gated rotation ACTION (positional drift via stale fp).
    _write_source(root, "plugins/gadget.py", "gadget")
    gadget_finding = _live_finding(root, "plugins/gadget.py")
    stale_key = identity_prefix(_canonical_key(gadget_finding)) + ":fp=deadbeefdeadbeef"
    _write_pre_judge_entry(allowlist_dir, "gadget.yaml", key=stale_key)

    bundle = _bundle(
        root,
        allowlist_dir,
        (BundleAction(lane="resign", kind="rotation", key=stale_key, source_file="gadget.yaml"),),
    )

    # The unfiltered whole-dir scan would raise here; the filtered scan does not.
    report = verify_bundle_against_tree(bundle, root=root, allowlist_dir=allowlist_dir)
    assert report.ok is True
    assert report.rotation_plan is not None


def test_verify_aborts_on_mixed_bundle_with_one_stale_action(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    # One valid drift_repair (matches the tree).
    _write_source(root, "plugins/widget.py", "widget")
    widget_finding = _live_finding(root, "plugins/widget.py")
    drift_key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=widget_finding, scope_fingerprint="b" * 64)
    # One stale action of a DIFFERENT kind (new_judgment whose finding vanished).
    _write_source(root, "plugins/gadget.py", "gadget")
    gadget_finding = _live_finding(root, "plugins/gadget.py")
    vanished_action = _new_judgment_action(gadget_finding, "plugins/gadget.py")
    _write_source(root, "plugins/gadget.py", "gadget", active=False)

    bundle = _bundle(
        root,
        allowlist_dir,
        (
            BundleAction(lane="resign", kind="drift_repair", key=drift_key, diagnosis_status="SCOPE_BINDING_DRIFT"),
            vanished_action,
        ),
    )
    report = verify_bundle_against_tree(bundle, root=root, allowlist_dir=allowlist_dir)
    assert report.ok is False
    assert any(vanished_action.key in m for m in report.mismatches)
