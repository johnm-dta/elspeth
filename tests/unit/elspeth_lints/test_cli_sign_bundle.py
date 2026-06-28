"""``sign-bundle`` -- the operator (key-bearing) firing command.

``sign-bundle`` is the *only* place a judge signature is minted from a staged
review bundle. It re-verifies every staged claim against the live tree (the
atomicity gate -- abort before any write), then fires per-action:

* ``drift_repair`` re-runs the real judge through the ``sign-judge-signatures``
  ceremony (re-judging prevents laundering a stale verdict over drifted content);
* ``new_judgment`` runs the real judge inside the keyed step;
* ``rotation`` mechanically re-binds a *non-judge-gated* key (no judge);
* ``stale_delete`` removes an orphaned entry (no judge).

These tests run with the operator HMAC key PRESENT (so diagnose is authoritative,
unlike the keyless ``test_bundle_verify`` suite); the signing key the fixtures
sign with and the env key are the one shared ``_HMAC_KEY`` constant. The real
judge is patched at the lazy-import seam ``elspeth_lints.core.judge.call_judge``
(patching ``core.cli`` is a no-op -- see ``test_justify.py``).

Fixtures are replicated locally (rather than imported from
``test_judge_signature_diagnosis`` / ``test_bundle_verify``) because there is no
``tests/unit/elspeth_lints`` package and no precedent for cross-test imports.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from elspeth_lints.core.allowlist import JudgeVerdict, compute_judge_metadata_signature
from elspeth_lints.core.cli import main
from elspeth_lints.core.judge import JUDGE_POLICY_HASH, JudgeResponse
from elspeth_lints.core.judge_signature_diagnosis import diagnose_judge_signatures
from elspeth_lints.core.review_bundle import (
    ActionPreview,
    BundleAction,
    ReviewBundle,
    write_bundle,
)
from elspeth_lints.rules.trust_tier.tier_model.rotate import identity_prefix

_HMAC_KEY = "x" * 32
_RECORDED_AT = "2024-01-01T00:00:00+00:00"
_MODEL = "claude-opus-4-7"
_RATIONALE = "original judge said the boundary was genuine"


@pytest.fixture(autouse=True)
def _signing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # sign-bundle is key-bearing: the operator HMAC key is present so diagnose
    # runs authoritative (recomputes signatures). Override tokens are cleared so
    # the override-required test isolates that one cause.
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _HMAC_KEY)
    monkeypatch.delenv("ELSPETH_JUDGE_OVERRIDE_TOKEN", raising=False)
    monkeypatch.delenv("ELSPETH_JUDGE_OVERRIDE_TOKEN_SHA256", raising=False)


# --------------------------------------------------------------------------- #
# source-tree + allowlist fixtures
# --------------------------------------------------------------------------- #


def _src(doc: str, *, active: bool = True) -> str:
    body = '        return payload.get("name", "anonymous")' if active else '        return "anonymous"'
    return f'"""{doc}"""\n\n\nclass Widget:\n    def lookup(self, payload: dict) -> str:\n{body}\n'


def _build_root(tmp_path: Path) -> Path:
    root = tmp_path / "src_root"
    root.mkdir(parents=True)
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


def _signed_entry_lines(key: str, *, ast_path: str, scope_fingerprint: str) -> list[str]:
    signature = compute_judge_metadata_signature(
        key=key,
        ast_path=ast_path,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime.fromisoformat(_RECORDED_AT),
        judge_model=_MODEL,
        judge_rationale=_RATIONALE,
        judge_policy_hash=JUDGE_POLICY_HASH,
        signature_version=2,
        scope_fingerprint=scope_fingerprint,
        judge_transport="openrouter",
        hmac_key=_HMAC_KEY.encode("utf-8"),
    )
    return [
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
        f"  scope_fingerprint: '{scope_fingerprint}'",
        "  judge_transport: openrouter",
        f"  ast_path: '{ast_path}'",
        f"  judge_metadata_signature: '{signature}'",
    ]


def _write_signed_v2_entry(
    allowlist_dir: Path,
    yaml_name: str,
    *,
    finding: Any,
    scope_fingerprint: str | None = None,
) -> str:
    key = _canonical_key(finding)
    stored_scope = finding.scope_fingerprint if scope_fingerprint is None else scope_fingerprint
    lines = ["allow_hits:", *_signed_entry_lines(key, ast_path=finding.ast_path, scope_fingerprint=stored_scope)]
    (allowlist_dir / yaml_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return key


def _pre_judge_entry_lines(key: str) -> list[str]:
    return [
        f"- key: {key}",
        "  owner: test-owner",
        "  reason: |-",
        "    payload is Tier-3 external data from upstream tool-call",
        "  safety: |-",
        "    suppression",
        "  expires: '2030-01-01'",
    ]


def _write_pre_judge_entry(allowlist_dir: Path, yaml_name: str, *, key: str) -> str:
    lines = ["allow_hits:", *_pre_judge_entry_lines(key)]
    (allowlist_dir / yaml_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return key


def _stale_rotation_key(finding: Any, *, fp: str = "deadbeefdeadbeef") -> str:
    return identity_prefix(_canonical_key(finding)) + f":fp={fp}"


# A well-formed canonical key for a SPARE pre-judge entry that coexists with the
# drifted signed entry in one YAML. It keeps the file's ``allow_hits`` a non-empty
# list after the drift_repair lane pops the drifted row -- mirroring the canonical
# multi-entry allowlist files, so ``_run_justify``'s similarity scan loads cleanly
# (a bare ``allow_hits:`` with no items is rejected by the loader). It is never a
# bundle action key, so verify ignores it.
_SPARE_PRE_JUDGE_KEY = "plugins/spare.py:R1:Widget:lookup:fp=feedface00000000"


def _write_signed_entry_with_spare(
    allowlist_dir: Path,
    yaml_name: str,
    *,
    finding: Any,
    scope_fingerprint: str | None = None,
) -> str:
    """Write a YAML with a leading spare pre-judge entry then the (drifted) signed entry.

    The signed entry is LAST, so a pop->restore round-trip re-appends it to the
    same position and the file stays byte-identical (the block_not_laundered pin).
    """
    key = _canonical_key(finding)
    stored_scope = finding.scope_fingerprint if scope_fingerprint is None else scope_fingerprint
    lines = [
        "allow_hits:",
        *_pre_judge_entry_lines(_SPARE_PRE_JUDGE_KEY),
        *_signed_entry_lines(key, ast_path=finding.ast_path, scope_fingerprint=stored_scope),
    ]
    (allowlist_dir / yaml_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return key


# --------------------------------------------------------------------------- #
# bundle + argv helpers
# --------------------------------------------------------------------------- #


def _bundle(
    root: Path,
    allowlist_dir: Path,
    actions: tuple[BundleAction, ...],
    *,
    bundle_id: str = "sign-bundle-under-test",
) -> ReviewBundle:
    return ReviewBundle(
        bundle_id=bundle_id,
        schema_version=1,
        created_at="2026-06-28T00:00:00+00:00",
        staged_by="agent-x",
        root=str(root),
        allowlist_dir=str(allowlist_dir),
        source_rev=None,
        source_dirty=False,
        actions=actions,
    )


def _write_bundle_file(tmp_path: Path, bundle: ReviewBundle) -> Path:
    return write_bundle(bundle, staged_dir=tmp_path / "staged")


def _new_judgment_action(finding: Any, rel: str, *, preview: ActionPreview | None = None) -> BundleAction:
    key = _canonical_key(finding)
    return BundleAction(
        lane="new_judgment",
        kind="justify",
        key=key,
        file_path=rel,
        symbol="Widget.lookup",
        rule="R1",
        fingerprint=key.rsplit(":fp=", 1)[1],
        draft_rationale="payload is Tier-3 external data from upstream tool-call",
        preview=preview,
    )


def _argv(
    bundle_path: Path,
    root: Path,
    allowlist_dir: Path,
    *,
    owner: str = "test-operator",
    extra: tuple[str, ...] = (),
) -> list[str]:
    return [
        "sign-bundle",
        str(bundle_path),
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        "--owner",
        owner,
        *extra,
    ]


@contextmanager
def _patch_judge(verdict_for: Callable[[str], JudgeVerdict]) -> Iterator[list[str]]:
    """Patch the real judge at the lazy-import seam; dispatch verdict by file_path."""
    calls: list[str] = []

    def _fake(request: Any, **kwargs: Any) -> JudgeResponse:
        calls.append(request.file_path)
        verdict = verdict_for(request.file_path)
        return JudgeResponse(
            verdict=verdict,
            model_id=_MODEL,
            judge_rationale=(
                "re-judged: genuine Tier-3 boundary" if verdict is JudgeVerdict.ACCEPTED else "blocked: not a genuine boundary"
            ),
            recorded_at=datetime.now(UTC),
            should_use_decorator=None,
            confidence=0.91,
            prompt_tokens_total=4000,
            prompt_tokens_cached=0,
            policy_hash=JUDGE_POLICY_HASH,
            judge_transport="openrouter",
        )

    with patch("elspeth_lints.core.judge.call_judge", side_effect=_fake):
        yield calls


def _accept_all(_file_path: str) -> JudgeVerdict:
    return JudgeVerdict.ACCEPTED


def _block_all(_file_path: str) -> JudgeVerdict:
    return JudgeVerdict.BLOCKED


def _diagnose(root: Path, allowlist_dir: Path) -> Any:
    return diagnose_judge_signatures(root=root, allowlist_dir=allowlist_dir)


# =========================================================================== #
# Task 2.1 -- subparser, dispatch, fail-closed key hoist, load + integrity
# =========================================================================== #


def test_sign_bundle_fails_closed_without_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """[O1] §5.4: a keyless run aborts before any tree read, even stale_delete-only."""
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/widget.py", "widget")
    finding = _live_finding(root, "plugins/widget.py")
    key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=finding)
    bundle = _bundle(root, allowlist_dir, (BundleAction(lane="resign", kind="stale_delete", key=key, source_file="widget.yaml"),))
    bundle_path = _write_bundle_file(tmp_path, bundle)

    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)
    assert main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",))) == 2
    assert "ELSPETH_JUDGE_METADATA_HMAC_KEY" in capsys.readouterr().err


def test_sign_bundle_rejects_malformed_bundle(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    bundle_path = tmp_path / "bad.json"
    bundle_path.write_text(json.dumps({"actions": []}), encoding="utf-8")  # missing schema_version
    assert main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",))) == 2


def test_sign_bundle_rejects_readonly_judge_tools(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/widget.py", "widget")
    finding = _live_finding(root, "plugins/widget.py")
    key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=finding)
    bundle = _bundle(root, allowlist_dir, (BundleAction(lane="resign", kind="stale_delete", key=key, source_file="widget.yaml"),))
    bundle_path = _write_bundle_file(tmp_path, bundle)
    assert main(_argv(bundle_path, root, allowlist_dir, extra=("--yes", "--judge-tools", "readonly"))) == 2
    assert "readonly" in capsys.readouterr().err


# =========================================================================== #
# Task 2.2 -- re-verify gate (abort before any write)
# =========================================================================== #


def test_sign_bundle_aborts_on_tree_drift_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/widget.py", "widget")
    finding = _live_finding(root, "plugins/widget.py")
    # Tree reports SCOPE_BINDING_DRIFT (synthetic scope drift)...
    key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=finding, scope_fingerprint="b" * 64)
    yaml_path = allowlist_dir / "widget.yaml"
    before = yaml_path.read_text(encoding="utf-8")
    # ...but the bundle claims it is only positional AST drift.
    bundle = _bundle(
        root, allowlist_dir, (BundleAction(lane="resign", kind="drift_repair", key=key, diagnosis_status="AST_PATH_BINDING_DRIFT"),)
    )
    bundle_path = _write_bundle_file(tmp_path, bundle)

    assert main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",))) == 2
    err = capsys.readouterr().err
    assert "SCOPE_BINDING_DRIFT" in err
    assert yaml_path.read_text(encoding="utf-8") == before  # no write


# =========================================================================== #
# Task 2.3 -- resign lane (drift_repair re-judges; rotation/stale_delete no judge)
# =========================================================================== #


def _drift_repair_ast_path_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    """A signed entry whose live AST position shifted -> AST_PATH_BINDING_DRIFT.

    The signed v2 entry binds the pre-shim ast_path; prepending a real statement
    shifts ``Module.body`` so ast_path drifts while the enclosing scope content
    stays byte-identical (scope_fingerprint stable).
    """
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    src = _write_source(root, "plugins/widget.py", "widget")
    finding = _live_finding(root, "plugins/widget.py")
    key = _write_signed_entry_with_spare(allowlist_dir, "plugins.yaml", finding=finding)
    src.write_text("_SHIM = 1\n\n\n" + _src("widget"), encoding="utf-8")
    return root, allowlist_dir, key


def test_sign_bundle_drift_repair_rejudges(tmp_path: Path) -> None:
    root, allowlist_dir, key = _drift_repair_ast_path_fixture(tmp_path)
    # Sanity: the tree genuinely reports the claimed status.
    assert any(i.status == "AST_PATH_BINDING_DRIFT" for i in _diagnose(root, allowlist_dir).items)
    bundle = _bundle(
        root, allowlist_dir, (BundleAction(lane="resign", kind="drift_repair", key=key, diagnosis_status="AST_PATH_BINDING_DRIFT"),)
    )
    bundle_path = _write_bundle_file(tmp_path, bundle)

    with _patch_judge(_accept_all) as calls:
        rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))

    assert rc == 0
    assert calls == ["plugins/widget.py"]  # the real judge WAS re-run
    post = _diagnose(root, allowlist_dir)
    assert not any(i.status == "AST_PATH_BINDING_DRIFT" for i in post.items)
    assert any(i.status == "OK_AUTHORITATIVE" for i in post.items)


def test_sign_bundle_drift_repair_block_not_laundered(tmp_path: Path) -> None:
    """§5.5/§7: an honest SCOPE drift that the judge BLOCKs is not laundered.

    The reused ceremony pops the stale row before judging and re-appends it on
    judge failure -- a pop-WITHOUT-restore would silently delete a signed entry.
    Pin the pop->block->restore contract by byte-comparing the YAML survives.
    """
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/widget.py", "widget")
    finding = _live_finding(root, "plugins/widget.py")
    key = _write_signed_entry_with_spare(allowlist_dir, "plugins.yaml", finding=finding, scope_fingerprint="b" * 64)
    yaml_path = allowlist_dir / "plugins.yaml"
    before = yaml_path.read_text(encoding="utf-8")
    assert any(i.status == "SCOPE_BINDING_DRIFT" for i in _diagnose(root, allowlist_dir).items)
    bundle = _bundle(
        root, allowlist_dir, (BundleAction(lane="resign", kind="drift_repair", key=key, diagnosis_status="SCOPE_BINDING_DRIFT"),)
    )
    bundle_path = _write_bundle_file(tmp_path, bundle)

    with _patch_judge(_block_all) as calls:
        rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))

    assert rc != 0
    assert calls == ["plugins/widget.py"]  # judge ran and BLOCKed
    assert yaml_path.read_text(encoding="utf-8") == before  # restored intact -- NOT deleted, NOT re-signed
    assert "b" * 64 in before  # the original drifted scope binding is still on disk


def test_sign_bundle_rotation_no_judge(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/gadget.py", "gadget")
    finding = _live_finding(root, "plugins/gadget.py")
    live_key = _canonical_key(finding)
    stale_key = _stale_rotation_key(finding)
    _write_pre_judge_entry(allowlist_dir, "gadget.yaml", key=stale_key)
    bundle = _bundle(root, allowlist_dir, (BundleAction(lane="resign", kind="rotation", key=stale_key, source_file="gadget.yaml"),))
    bundle_path = _write_bundle_file(tmp_path, bundle)

    def _raise(_file_path: str) -> JudgeVerdict:
        raise AssertionError("the judge must not run for a rotation action")

    with _patch_judge(_raise):
        rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes", "--rotation-log", str(tmp_path / "rotations.log"))))

    assert rc == 0
    text = (allowlist_dir / "gadget.yaml").read_text(encoding="utf-8")
    assert f"- key: {live_key}" in text
    assert stale_key not in text


def test_sign_bundle_rotation_records_rotation_manifest(tmp_path: Path) -> None:
    """A rotation action MUST append the .elspeth/rotations.log manifest record the
    governance gate (check-rotation-audit) consumes -- the gate derives expected
    rotations from the git diff of the allowlist and fails any old->new key change
    with no covering manifest record. Regression for the hardcoded
    ``rotation_log_path=None`` that rewrote the key but suppressed the manifest.
    """
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/gadget.py", "gadget")
    finding = _live_finding(root, "plugins/gadget.py")
    live_key = _canonical_key(finding)
    stale_key = _stale_rotation_key(finding)
    _write_pre_judge_entry(allowlist_dir, "gadget.yaml", key=stale_key)
    rot_log = tmp_path / "rotations.log"
    bundle = _bundle(root, allowlist_dir, (BundleAction(lane="resign", kind="rotation", key=stale_key, source_file="gadget.yaml"),))
    bundle_path = _write_bundle_file(tmp_path, bundle)

    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes", "--rotation-log", str(rot_log))))

    assert rc == 0
    assert f"- key: {live_key}" in (allowlist_dir / "gadget.yaml").read_text(encoding="utf-8")
    assert rot_log.exists(), "rotation applied but no .elspeth/rotations.log manifest record was written -- check-rotation-audit will fail"
    records = [json.loads(line) for line in rot_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    rotations = [item for rec in records for item in rec.get("rotations", [])]
    assert {"source_file": "gadget.yaml", "old_key": stale_key, "new_key": live_key} in rotations


def test_sign_bundle_rotation_execute_minimal_plan_no_unfiltered_rescan(tmp_path: Path) -> None:
    """Third-consumer regression: no unfiltered re-scan at execute + no over-application.

    Populations: (1) a judge-gated, fp-SHIFTED NON-action entry in the scanned
    dir (would crash an unfiltered whole-dir scan at ``plan_rotations``);
    (2) ONE non-judge-gated rotation that IS the staged action; (3) a SECOND,
    surveyed-but-UNSTAGED non-judge-gated rotation. Against the pre-fix lane that
    copies ``_run_rotate``'s default unfiltered ``scan_for_rotations`` this would
    ``RuntimeError`` at execute (population 1) and/or over-apply population (3);
    the fixed lane reuses the carried filtered plan and applies a one-``Rotation``
    minimal plan, so it neither raises nor touches population (3).
    """
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)

    # (1) judge-gated, fp-shifted NON-action.
    widget = _write_source(root, "plugins/widget.py", "widget")
    widget_finding = _live_finding(root, "plugins/widget.py")
    _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=widget_finding)
    widget.write_text("_SHIM = 1\n\n\n" + _src("widget"), encoding="utf-8")  # fp drift -> would crash unfiltered scan

    # (2) staged non-judge-gated rotation.
    _write_source(root, "plugins/gadget.py", "gadget")
    gadget_finding = _live_finding(root, "plugins/gadget.py")
    gadget_live = _canonical_key(gadget_finding)
    gadget_stale = _stale_rotation_key(gadget_finding, fp="deadbeefdeadbeef")
    _write_pre_judge_entry(allowlist_dir, "gadget.yaml", key=gadget_stale)

    # (3) surveyed-but-unstaged non-judge-gated rotation.
    _write_source(root, "plugins/sprocket.py", "sprocket")
    sprocket_finding = _live_finding(root, "plugins/sprocket.py")
    sprocket_stale = _stale_rotation_key(sprocket_finding, fp="cafebabecafebabe")
    _write_pre_judge_entry(allowlist_dir, "sprocket.yaml", key=sprocket_stale)
    sprocket_before = (allowlist_dir / "sprocket.yaml").read_text(encoding="utf-8")

    bundle = _bundle(root, allowlist_dir, (BundleAction(lane="resign", kind="rotation", key=gadget_stale, source_file="gadget.yaml"),))
    bundle_path = _write_bundle_file(tmp_path, bundle)

    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes", "--rotation-log", str(tmp_path / "rotations.log"))))

    assert rc == 0  # (a) did NOT raise on the judge-gated fp-shifted non-action
    gadget_text = (allowlist_dir / "gadget.yaml").read_text(encoding="utf-8")
    assert f"- key: {gadget_live}" in gadget_text  # staged key rotated
    assert gadget_stale not in gadget_text
    # (b) the unstaged surveyed rotation is byte-untouched -> a minimal one-Rotation plan was built.
    assert (allowlist_dir / "sprocket.yaml").read_text(encoding="utf-8") == sprocket_before


def test_sign_bundle_stale_delete_removes_entry(tmp_path: Path) -> None:
    """Multi-file allowlist: the action routes to its OWNING source_file."""
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    # Orphan to delete: a signed entry whose finding has since vanished.
    _write_source(root, "plugins/widget.py", "widget")
    widget_finding = _live_finding(root, "plugins/widget.py")
    orphan_key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=widget_finding)
    _write_source(root, "plugins/widget.py", "widget", active=False)  # finding gone -> NO_MATCHING_FINDING
    # Sibling in a DIFFERENT file that must remain untouched.
    sibling_before = _write_pre_judge_entry(allowlist_dir, "gadget.yaml", key="plugins/gadget.py:R1:Widget:lookup:fp=feedface00000000")
    gadget_before = (allowlist_dir / "gadget.yaml").read_text(encoding="utf-8")

    bundle = _bundle(root, allowlist_dir, (BundleAction(lane="resign", kind="stale_delete", key=orphan_key, source_file="widget.yaml"),))
    bundle_path = _write_bundle_file(tmp_path, bundle)

    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))

    assert rc == 0
    assert orphan_key not in (allowlist_dir / "widget.yaml").read_text(encoding="utf-8")
    assert (allowlist_dir / "gadget.yaml").read_text(encoding="utf-8") == gadget_before  # sibling intact
    assert sibling_before in gadget_before


# =========================================================================== #
# Task 2.4 -- new-judgment lane (real judge + sign) + BLOCK + override
# =========================================================================== #


def test_sign_bundle_new_judgment_runs_real_judge(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/gadget.py", "gadget")
    finding = _live_finding(root, "plugins/gadget.py")
    bundle = _bundle(root, allowlist_dir, (_new_judgment_action(finding, "plugins/gadget.py"),))
    bundle_path = _write_bundle_file(tmp_path, bundle)

    with _patch_judge(_accept_all) as calls:
        rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))

    assert rc == 0
    assert calls == ["plugins/gadget.py"]
    post = _diagnose(root, allowlist_dir)
    assert any(i.status == "OK_AUTHORITATIVE" and i.key == _canonical_key(finding) for i in post.items)


def test_sign_bundle_block_contradicting_preview_not_signed(tmp_path: Path) -> None:
    """§7: an ACCEPTED preview does not survive a BLOCK from the authoritative judge."""
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/gadget.py", "gadget")
    finding = _live_finding(root, "plugins/gadget.py")
    preview = ActionPreview(verdict="ACCEPTED", rationale="agent preview said genuine", model="preview-model", transport="claude_agent_sdk")
    bundle = _bundle(root, allowlist_dir, (_new_judgment_action(finding, "plugins/gadget.py", preview=preview),))
    bundle_path = _write_bundle_file(tmp_path, bundle)

    with _patch_judge(_block_all) as calls:
        rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))

    assert rc != 0
    assert calls == ["plugins/gadget.py"]
    assert not (allowlist_dir / "plugins.yaml").exists()  # nothing signed
    post = _diagnose(root, allowlist_dir)
    assert all(i.key != _canonical_key(finding) for i in post.items)


def test_sign_bundle_override_token_required(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/gadget.py", "gadget")
    finding = _live_finding(root, "plugins/gadget.py")
    bundle = _bundle(root, allowlist_dir, (_new_judgment_action(finding, "plugins/gadget.py"),))
    bundle_path = _write_bundle_file(tmp_path, bundle)

    # --operator-override but no override token in the env (cleared by the fixture).
    with _patch_judge(_accept_all):
        rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes", "--operator-override")))

    assert rc == 2
    assert "ELSPETH_JUDGE_OVERRIDE_TOKEN" in capsys.readouterr().err
    assert not (allowlist_dir / "plugins.yaml").exists()


def test_sign_bundle_partial_block_writes_accepted_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Per-action non-transactional contract: A is written, B blocked, M/K reported."""
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "alpha/mod.py", "alpha")
    _write_source(root, "beta/mod.py", "beta")
    alpha_finding = _live_finding(root, "alpha/mod.py")
    beta_finding = _live_finding(root, "beta/mod.py")
    bundle = _bundle(
        root,
        allowlist_dir,
        (
            _new_judgment_action(alpha_finding, "alpha/mod.py"),
            _new_judgment_action(beta_finding, "beta/mod.py"),
        ),
    )
    bundle_path = _write_bundle_file(tmp_path, bundle)

    def _verdict(file_path: str) -> JudgeVerdict:
        return JudgeVerdict.ACCEPTED if file_path.startswith("alpha/") else JudgeVerdict.BLOCKED

    with _patch_judge(_verdict):
        rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))

    assert rc != 0
    # A written, B not.
    assert (allowlist_dir / "alpha.yaml").exists()
    assert not (allowlist_dir / "beta.yaml").exists()
    post = _diagnose(root, allowlist_dir)
    assert any(i.key == _canonical_key(alpha_finding) and i.status == "OK_AUTHORITATIVE" for i in post.items)
    assert all(i.key != _canonical_key(beta_finding) for i in post.items)
    err = capsys.readouterr().err
    assert "succeeded" in err and "failed" in err


# =========================================================================== #
# Task 2.5 -- summary / confirm / dry-run / dup-key / baseline-regen
# =========================================================================== #


def test_sign_bundle_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/gadget.py", "gadget")
    finding = _live_finding(root, "plugins/gadget.py")
    bundle = _bundle(root, allowlist_dir, (_new_judgment_action(finding, "plugins/gadget.py"),))
    bundle_path = _write_bundle_file(tmp_path, bundle)

    def _raise(_file_path: str) -> JudgeVerdict:
        raise AssertionError("dry-run must not call the judge")

    with _patch_judge(_raise):
        rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--dry-run",)))

    assert rc == 0
    assert not (allowlist_dir / "plugins.yaml").exists()
    out = capsys.readouterr().out
    assert "new_judgment" in out


def test_sign_bundle_dry_run_reports_planned_override_count(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "alpha/mod.py", "alpha")
    _write_source(root, "beta/mod.py", "beta")
    alpha_finding = _live_finding(root, "alpha/mod.py")
    beta_finding = _live_finding(root, "beta/mod.py")
    bundle = _bundle(
        root,
        allowlist_dir,
        (
            _new_judgment_action(alpha_finding, "alpha/mod.py"),
            _new_judgment_action(beta_finding, "beta/mod.py"),
        ),
    )
    bundle_path = _write_bundle_file(tmp_path, bundle)

    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--dry-run", "--operator-override")))

    assert rc == 0
    out = capsys.readouterr().out
    # K = 2 planned override actions, surfaced as the load-bearing integer.
    assert "planned operator-override actions: 2" in out
    assert "approx" in out.lower()


def test_sign_bundle_requires_confirmation_without_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/gadget.py", "gadget")
    finding = _live_finding(root, "plugins/gadget.py")
    bundle = _bundle(root, allowlist_dir, (_new_judgment_action(finding, "plugins/gadget.py"),))
    bundle_path = _write_bundle_file(tmp_path, bundle)

    monkeypatch.setattr("sys.stdin", io.StringIO("no\n"))

    def _raise(_file_path: str) -> JudgeVerdict:
        raise AssertionError("a declined confirmation must not call the judge")

    with _patch_judge(_raise):
        rc = main(_argv(bundle_path, root, allowlist_dir))  # no --yes

    assert rc == 0
    assert not (allowlist_dir / "plugins.yaml").exists()  # nothing written


def _dup_key_signed_block(key: str) -> list[str]:
    return _signed_entry_lines(key, ast_path="body[1]/body[0]/body[0]/value", scope_fingerprint="a" * 64)


def test_sign_bundle_dup_key_bundle_aborts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Dup-key dataloss trap: apply_plan refuses span!=1 -> caught -> return 2, both copies intact.

    The same key K appears twice -- once judge-gated (filtered out of the
    non-judge-gated rotation survey, so verify still sees ONE clean rotation) and
    once non-judge-gated (the staged rotation). At write time ``apply_plan`` finds
    K twice in the text and raises 'occurs 2x'; the narrow catch converts it to a
    clean return 2 rather than deleting both copies.
    """
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/gadget.py", "gadget")
    finding = _live_finding(root, "plugins/gadget.py")
    stale_key = _stale_rotation_key(finding)
    text = "\n".join(["allow_hits:", *_dup_key_signed_block(stale_key), *_pre_judge_entry_lines(stale_key)]) + "\n"
    yaml_path = allowlist_dir / "gadget.yaml"
    yaml_path.write_text(text, encoding="utf-8")

    bundle = _bundle(root, allowlist_dir, (BundleAction(lane="resign", kind="rotation", key=stale_key, source_file="gadget.yaml"),))
    bundle_path = _write_bundle_file(tmp_path, bundle)

    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))

    assert rc == 2
    assert yaml_path.read_text(encoding="utf-8").count(f"- key: {stale_key}") == 2  # both copies preserved
    assert "occurs" in capsys.readouterr().err.lower()


def test_sign_bundle_noncanonical_allowlist_skips_baseline_regen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hermeticity: a tmp_path run never shells regen_fingerprint_baseline.py."""
    import subprocess

    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/widget.py", "widget")
    widget_finding = _live_finding(root, "plugins/widget.py")
    orphan_key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=widget_finding)
    _write_source(root, "plugins/widget.py", "widget", active=False)  # orphan
    bundle = _bundle(root, allowlist_dir, (BundleAction(lane="resign", kind="stale_delete", key=orphan_key, source_file="widget.yaml"),))
    bundle_path = _write_bundle_file(tmp_path, bundle)

    calls: list[Any] = []
    real_run = subprocess.run

    def _spy_run(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return real_run(*args, **kwargs)

    monkeypatch.setattr("subprocess.run", _spy_run)

    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))

    assert rc == 0
    assert calls == []  # never shelled the regen script
    assert "canonical-allowlist-only" in capsys.readouterr().out
