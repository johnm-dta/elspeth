"""``rekey`` -- the operator dual-key custody window.

``rekey`` rotates the HMAC key that binds judge metadata: it verifies every
judge-gated entry under the OLD key, recomputes its signature under the NEW key,
and atomically rewrites only the ``judge_metadata_signature`` line (binding +
audit lines stay byte-identical -- a *scheme-preserving signature-only swap*).

Two security properties are first-class:

* **No laundering** -- an entry that verifies under *neither* the old nor the new
  key aborts the whole run with no write (a broken entry is never re-keyed clean).
* **Re-runnable** -- Pass-1 accepts an entry verifying under OLD *or* NEW and
  Pass-2 skips already-NEW entries, so a partial/interrupted run self-heals on a
  second invocation rather than self-bricking.

The full judge-gated set is re-derived from the live tree at fire time; the
staged ``RekeyPlan`` is advisory provenance (a tree entry absent from its ``keys``
is still re-keyed). The bundle's ``old_key_env``/``new_key_env`` cross-check the
CLI flags and abort a stale/transposed flag set before Pass-1.

Fixtures are replicated locally (mirroring ``test_cli_sign_bundle``) because
there is no ``tests/unit/elspeth_lints`` package and no cross-test-import
precedent.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from elspeth_lints.core.allowlist import (
    JudgeVerdict,
    compute_judge_metadata_signature,
    load_allowlist,
)
from elspeth_lints.core.cli import main
from elspeth_lints.core.judge import JUDGE_POLICY_HASH
from elspeth_lints.core.review_bundle import RekeyPlan, ReviewBundle, write_bundle
from elspeth_lints.rules.trust_tier.tier_model.rule import RULES

_OLD_KEY = "o" * 32
_NEW_KEY = "n" * 32
_THIRD_KEY = "z" * 32
_RECORDED_AT = "2024-01-01T00:00:00+00:00"
_MODEL = "claude-opus-4-7"
_RATIONALE = "original judge said the boundary was genuine"

_OLD_ENV = "ELSPETH_REKEY_OLD_KEY"
_NEW_ENV = "ELSPETH_REKEY_NEW_KEY"


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


def _signed_entry_lines(key: str, *, ast_path: str, scope_fingerprint: str, hmac_key: str) -> list[str]:
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
        hmac_key=hmac_key.encode("utf-8"),
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
    hmac_key: str,
    scope_fingerprint: str | None = None,
) -> str:
    key = _canonical_key(finding)
    stored_scope = finding.scope_fingerprint if scope_fingerprint is None else scope_fingerprint
    lines = ["allow_hits:", *_signed_entry_lines(key, ast_path=finding.ast_path, scope_fingerprint=stored_scope, hmac_key=hmac_key)]
    (allowlist_dir / yaml_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return key


def _pre_judge_entry_lines(key: str) -> list[str]:
    """A non-judge-gated (pre-judge) entry: no signature line, ignored by rekey."""
    return [
        f"- key: {key}",
        "  owner: test-owner",
        "  reason: |-",
        "    payload is Tier-3 external data from upstream tool-call",
        "  safety: |-",
        "    suppression",
        "  expires: '2030-01-01'",
    ]


def _entry_signature(finding: Any, *, hmac_key: str) -> str:
    """The exact judge_metadata_signature value the fixtures sign with, under ``hmac_key``."""
    return compute_judge_metadata_signature(
        key=_canonical_key(finding),
        ast_path=finding.ast_path,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime.fromisoformat(_RECORDED_AT),
        judge_model=_MODEL,
        judge_rationale=_RATIONALE,
        judge_policy_hash=JUDGE_POLICY_HASH,
        signature_version=2,
        scope_fingerprint=finding.scope_fingerprint,
        judge_transport="openrouter",
        hmac_key=hmac_key.encode("utf-8"),
    )


def _load_entry(allowlist_dir: Path, key: str) -> Any:
    """Load source-less (no production HMAC/file-fingerprint gate) and return the entry."""
    allowlist = load_allowlist(allowlist_dir, valid_rule_ids=frozenset(RULES.keys()), source_root=None)
    for entry in allowlist.entries:
        if entry.key == key:
            return entry
    raise AssertionError(f"entry {key!r} not found in {allowlist_dir}")


def _rekey_bundle(
    tmp_path: Path,
    root: Path,
    allowlist_dir: Path,
    *,
    keys: tuple[str, ...] = (),
    broken_keys: tuple[str, ...] = (),
    old_key_env: str = _OLD_ENV,
    new_key_env: str = _NEW_ENV,
    bundle_id: str = "rekey-under-test",
) -> Path:
    bundle = ReviewBundle(
        bundle_id=bundle_id,
        schema_version=1,
        created_at="2026-06-28T00:00:00+00:00",
        staged_by="agent-x",
        root=str(root),
        allowlist_dir=str(allowlist_dir),
        source_rev=None,
        source_dirty=False,
        actions=(),
        rekey=RekeyPlan(old_key_env=old_key_env, new_key_env=new_key_env, keys=keys, broken_keys=broken_keys),
    )
    return write_bundle(bundle, staged_dir=tmp_path / "staged")


def _argv(
    bundle_path: Path,
    root: Path,
    allowlist_dir: Path,
    *,
    old_key_env: str = _OLD_ENV,
    new_key_env: str = _NEW_ENV,
    extra: tuple[str, ...] = (),
) -> list[str]:
    return [
        "rekey",
        "--in",
        str(bundle_path),
        "--old-key-env",
        old_key_env,
        "--new-key-env",
        new_key_env,
        "--root",
        str(root),
        "--allowlist-dir",
        str(allowlist_dir),
        *extra,
    ]


def _set_keys(monkeypatch: pytest.MonkeyPatch, *, old: str | None = _OLD_KEY, new: str | None = _NEW_KEY) -> None:
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)
    if old is None:
        monkeypatch.delenv(_OLD_ENV, raising=False)
    else:
        monkeypatch.setenv(_OLD_ENV, old)
    if new is None:
        monkeypatch.delenv(_NEW_ENV, raising=False)
    else:
        monkeypatch.setenv(_NEW_ENV, new)


def _verifies_under(entry: Any, hmac_key: str) -> bool:
    from elspeth_lints.core.allowlist import verify_entry_signature_with_key

    try:
        verify_entry_signature_with_key(entry, hmac_key=hmac_key.encode("utf-8"))
    except ValueError:
        return False
    return True


def _production_verify(allowlist_dir: Path, key: str, *, hmac_key: str, monkeypatch: pytest.MonkeyPatch) -> bool:
    """Re-verify an entry through the PRODUCTION load-time verifier under ``hmac_key``."""
    from elspeth_lints.core.allowlist import _verify_judge_metadata_signature_at_load

    entry = _load_entry(allowlist_dir, key)
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", hmac_key)
    try:
        _verify_judge_metadata_signature_at_load(entry, context=f"rekey-test {key!r}", allow_shape_only=False)
    except ValueError:
        return False
    finally:
        monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)
    return True


# =========================================================================== #
# Task 4.1 -- keyed-verify sibling in allowlist.py
# =========================================================================== #


def test_verify_entry_signature_with_key(tmp_path: Path) -> None:
    """A v2 entry signed under key A verifies under A and raises under key B."""
    from elspeth_lints.core.allowlist import verify_entry_signature_with_key

    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/widget.py", "widget")
    finding = _live_finding(root, "plugins/widget.py")
    key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=finding, hmac_key=_OLD_KEY)
    entry = _load_entry(allowlist_dir, key)

    # Verifies under the signing key (no raise) ...
    verify_entry_signature_with_key(entry, hmac_key=_OLD_KEY.encode("utf-8"))
    # ... and raises under a different key.
    with pytest.raises(ValueError):
        verify_entry_signature_with_key(entry, hmac_key=_NEW_KEY.encode("utf-8"))


# =========================================================================== #
# Task 4.3 -- rekey CLI (dual-key verify -> recompute -> atomic write)
# =========================================================================== #


def test_rekey_reseals_under_new_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dual-key window: verify-under-OLD -> sign-under-NEW, signature-only swap.

    Proven through the PRODUCTION loader (a shared-marshalling bug would agree with
    ``verify_entry_signature_with_key`` but brick real loads), and pinned as a
    *signature-only* swap by byte-comparing every non-signature line.
    """
    _set_keys(monkeypatch)
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/widget.py", "widget")
    finding = _live_finding(root, "plugins/widget.py")
    key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=finding, hmac_key=_OLD_KEY)
    yaml_path = allowlist_dir / "widget.yaml"
    before = yaml_path.read_text(encoding="utf-8")
    bundle_path = _rekey_bundle(tmp_path, root, allowlist_dir, keys=(key,))

    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))
    assert rc == 0

    # Signature-only swap: exactly the judge_metadata_signature line changed.
    before_lines = before.splitlines()
    after_lines = yaml_path.read_text(encoding="utf-8").splitlines()
    assert len(before_lines) == len(after_lines)
    changed = [(b, a) for b, a in zip(before_lines, after_lines, strict=True) if b != a]
    assert len(changed) == 1
    assert changed[0][0].lstrip().startswith("judge_metadata_signature:")

    # Production loader: verifies under NEW, fails under OLD.
    assert _production_verify(allowlist_dir, key, hmac_key=_NEW_KEY, monkeypatch=monkeypatch)
    assert not _production_verify(allowlist_dir, key, hmac_key=_OLD_KEY, monkeypatch=monkeypatch)


def test_rekey_refuses_nonverifying_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """§5.5 no-laundering: an entry verifying under NEITHER key aborts with no write."""
    _set_keys(monkeypatch)
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/widget.py", "widget")
    finding = _live_finding(root, "plugins/widget.py")
    # Signed under a THIRD key -> verifies under neither OLD nor NEW.
    key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=finding, hmac_key=_THIRD_KEY)
    yaml_path = allowlist_dir / "widget.yaml"
    before = yaml_path.read_text(encoding="utf-8")
    bundle_path = _rekey_bundle(tmp_path, root, allowlist_dir, keys=(key,))

    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))

    assert rc == 2
    # The abort is specifically the no-laundering Pass-1 path, not an incidental error.
    err = capsys.readouterr().err
    assert "NEITHER" in err and key in err
    assert yaml_path.read_text(encoding="utf-8") == before  # not laundered into NEW; no write
    # The broken entry was NOT re-keyed (it does not verify under NEW).
    assert not _production_verify(allowlist_dir, key, hmac_key=_NEW_KEY, monkeypatch=monkeypatch)


def test_rekey_partial_pass2_is_re_runnable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recovery: a mid-Pass-2 state (some files NEW, some OLD) self-heals on re-run.

    Pass-1 accepts an entry verifying under OLD *or* NEW; Pass-2 skips already-NEW.
    """
    _set_keys(monkeypatch)
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    # File A: a COMPLETED prior-run file (already under NEW).
    _write_source(root, "alpha/mod.py", "alpha")
    alpha_finding = _live_finding(root, "alpha/mod.py")
    alpha_key = _write_signed_v2_entry(allowlist_dir, "alpha.yaml", finding=alpha_finding, hmac_key=_NEW_KEY)
    # File B: the remaining work (still under OLD).
    _write_source(root, "beta/mod.py", "beta")
    beta_finding = _live_finding(root, "beta/mod.py")
    beta_key = _write_signed_v2_entry(allowlist_dir, "beta.yaml", finding=beta_finding, hmac_key=_OLD_KEY)
    alpha_before = (allowlist_dir / "alpha.yaml").read_text(encoding="utf-8")

    # Pre-assert the partial state genuinely exercises OLD-or-NEW Pass-1 acceptance.
    alpha_entry = _load_entry(allowlist_dir, alpha_key)
    assert _verifies_under(alpha_entry, _NEW_KEY)
    assert not _verifies_under(alpha_entry, _OLD_KEY)

    bundle_path = _rekey_bundle(tmp_path, root, allowlist_dir, keys=(beta_key,))
    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))

    assert rc == 0
    # Already-NEW file skipped (byte-identical -> the Pass-2 skip is real).
    assert (allowlist_dir / "alpha.yaml").read_text(encoding="utf-8") == alpha_before
    # All entries end under NEW; the OLD file was re-keyed.
    assert _production_verify(allowlist_dir, alpha_key, hmac_key=_NEW_KEY, monkeypatch=monkeypatch)
    assert _production_verify(allowlist_dir, beta_key, hmac_key=_NEW_KEY, monkeypatch=monkeypatch)
    assert not _production_verify(allowlist_dir, beta_key, hmac_key=_OLD_KEY, monkeypatch=monkeypatch)


def test_rekey_rekeys_tree_entry_absent_from_staged_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """M1 completeness: a judge-gated tree entry absent from RekeyPlan.keys is STILL re-keyed."""
    _set_keys(monkeypatch)
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "alpha/mod.py", "alpha")
    alpha_finding = _live_finding(root, "alpha/mod.py")
    alpha_key = _write_signed_v2_entry(allowlist_dir, "alpha.yaml", finding=alpha_finding, hmac_key=_OLD_KEY)
    # A SECOND judge-gated entry, added to the tree after stage_rekey -> absent from the plan.
    _write_source(root, "beta/mod.py", "beta")
    beta_finding = _live_finding(root, "beta/mod.py")
    beta_key = _write_signed_v2_entry(allowlist_dir, "beta.yaml", finding=beta_finding, hmac_key=_OLD_KEY)
    # The staged plan lists ONLY alpha.
    bundle_path = _rekey_bundle(tmp_path, root, allowlist_dir, keys=(alpha_key,))

    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))

    assert rc == 0
    # The plan-absent tree entry was re-keyed too (not left to rot under the retired OLD key).
    assert _production_verify(allowlist_dir, beta_key, hmac_key=_NEW_KEY, monkeypatch=monkeypatch)
    assert _production_verify(allowlist_dir, alpha_key, hmac_key=_NEW_KEY, monkeypatch=monkeypatch)


def test_rekey_aborts_on_env_name_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Stale/transposed bundle env-names must not steer key selection: abort before Pass-1, no write."""
    _set_keys(monkeypatch)
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/widget.py", "widget")
    finding = _live_finding(root, "plugins/widget.py")
    key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=finding, hmac_key=_OLD_KEY)
    yaml_path = allowlist_dir / "widget.yaml"
    before = yaml_path.read_text(encoding="utf-8")
    # Bundle records a DIFFERENT old_key_env than the --old-key-env flag passes.
    bundle_path = _rekey_bundle(tmp_path, root, allowlist_dir, keys=(key,), old_key_env="SOME_OTHER_OLD_ENV")

    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))

    assert rc == 2
    assert yaml_path.read_text(encoding="utf-8") == before  # aborted before Pass-1; no write
    assert "mismatch" in capsys.readouterr().err.lower()


def test_rekey_dry_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """--dry-run: zero writes, returns 0, prints the planned re-key (still-OLD) count."""
    _set_keys(monkeypatch)
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "alpha/mod.py", "alpha")
    alpha_finding = _live_finding(root, "alpha/mod.py")
    alpha_key = _write_signed_v2_entry(allowlist_dir, "alpha.yaml", finding=alpha_finding, hmac_key=_OLD_KEY)
    _write_source(root, "beta/mod.py", "beta")
    beta_finding = _live_finding(root, "beta/mod.py")
    beta_key = _write_signed_v2_entry(allowlist_dir, "beta.yaml", finding=beta_finding, hmac_key=_OLD_KEY)
    alpha_before = (allowlist_dir / "alpha.yaml").read_text(encoding="utf-8")
    beta_before = (allowlist_dir / "beta.yaml").read_text(encoding="utf-8")
    bundle_path = _rekey_bundle(tmp_path, root, allowlist_dir, keys=(alpha_key, beta_key))

    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--dry-run",)))

    assert rc == 0
    assert (allowlist_dir / "alpha.yaml").read_text(encoding="utf-8") == alpha_before
    assert (allowlist_dir / "beta.yaml").read_text(encoding="utf-8") == beta_before
    out = capsys.readouterr().out
    assert "planned re-key actions: 2" in out  # both still-OLD entries Pass-2 WOULD write


def test_rekey_dry_run_planned_count_excludes_already_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Planned count = the REMAINING work (still-OLD), not the full judge-gated count."""
    _set_keys(monkeypatch)
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "alpha/mod.py", "alpha")
    alpha_finding = _live_finding(root, "alpha/mod.py")
    _write_signed_v2_entry(allowlist_dir, "alpha.yaml", finding=alpha_finding, hmac_key=_NEW_KEY)  # already NEW
    _write_source(root, "beta/mod.py", "beta")
    beta_finding = _live_finding(root, "beta/mod.py")
    beta_key = _write_signed_v2_entry(allowlist_dir, "beta.yaml", finding=beta_finding, hmac_key=_OLD_KEY)  # still OLD
    bundle_path = _rekey_bundle(tmp_path, root, allowlist_dir, keys=(beta_key,))

    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--dry-run",)))

    assert rc == 0
    out = capsys.readouterr().out
    assert "planned re-key actions: 1" in out  # only the still-OLD entry, not the already-NEW one


def test_rekey_requires_confirmation_without_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A declined confirmation (stdin 'no') aborts with no write."""
    _set_keys(monkeypatch)
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/widget.py", "widget")
    finding = _live_finding(root, "plugins/widget.py")
    key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=finding, hmac_key=_OLD_KEY)
    yaml_path = allowlist_dir / "widget.yaml"
    before = yaml_path.read_text(encoding="utf-8")
    bundle_path = _rekey_bundle(tmp_path, root, allowlist_dir, keys=(key,))

    monkeypatch.setattr("sys.stdin", io.StringIO("no\n"))
    rc = main(_argv(bundle_path, root, allowlist_dir))  # no --yes

    assert rc == 0
    assert yaml_path.read_text(encoding="utf-8") == before  # nothing written


@pytest.mark.parametrize("missing", ["old", "new"])
def test_rekey_fails_closed_without_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """Missing OLD or NEW key env -> return 2, no write (fails closed without both keys)."""
    if missing == "old":
        _set_keys(monkeypatch, old=None, new=_NEW_KEY)
    else:
        _set_keys(monkeypatch, old=_OLD_KEY, new=None)
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/widget.py", "widget")
    finding = _live_finding(root, "plugins/widget.py")
    key = _write_signed_v2_entry(allowlist_dir, "widget.yaml", finding=finding, hmac_key=_OLD_KEY)
    yaml_path = allowlist_dir / "widget.yaml"
    before = yaml_path.read_text(encoding="utf-8")
    bundle_path = _rekey_bundle(tmp_path, root, allowlist_dir, keys=(key,))

    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))

    assert rc == 2
    assert yaml_path.read_text(encoding="utf-8") == before  # fail-closed before any write


def test_rekey_multi_entry_file_splices_only_still_old(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ONE YAML, THREE entries -> exercise the multi-entry splice machinery.

    The canonical corpus is multi-entry files; the single-entry tests above would
    pass against a naive single-line replacement, leaving ``_rekey_entries_in_yaml``'s
    range-subset selection + bottom-up splice + in-file Pass-2 skip + non-judge-gated
    passthrough untested. This pins all four in one file:

    * (a) a judge-gated entry still under OLD -> re-keyed;
    * (b) a judge-gated entry already under NEW -> byte-identical (in-file skip);
    * (c) a pre-judge (non-judge-gated) entry -> byte-identical (subset ignores it).
    """
    _set_keys(monkeypatch)
    root = _build_root(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_source(root, "plugins/widget.py", "widget")
    _write_source(root, "plugins/gadget.py", "gadget")
    widget_finding = _live_finding(root, "plugins/widget.py")
    gadget_finding = _live_finding(root, "plugins/gadget.py")
    old_entry_key = _canonical_key(widget_finding)  # signed under OLD -> re-keyed
    new_entry_key = _canonical_key(gadget_finding)  # signed under NEW -> skipped
    pre_judge_key = "plugins/spare.py:R1:Widget:lookup:fp=feedface00000000"

    # All three entries live in ONE plugins.yaml (their source paths differ, but the
    # YAML file -- hence each entry.source_file -- is the same; Pass-2 groups them).
    lines = [
        "allow_hits:",
        *_signed_entry_lines(
            old_entry_key, ast_path=widget_finding.ast_path, scope_fingerprint=widget_finding.scope_fingerprint, hmac_key=_OLD_KEY
        ),
        *_signed_entry_lines(
            new_entry_key, ast_path=gadget_finding.ast_path, scope_fingerprint=gadget_finding.scope_fingerprint, hmac_key=_NEW_KEY
        ),
        *_pre_judge_entry_lines(pre_judge_key),
    ]
    yaml_path = allowlist_dir / "plugins.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    before = yaml_path.read_text(encoding="utf-8")

    bundle_path = _rekey_bundle(tmp_path, root, allowlist_dir, keys=(old_entry_key,))
    rc = main(_argv(bundle_path, root, allowlist_dir, extra=("--yes",)))
    assert rc == 0

    # Production loader: the still-OLD entry is re-keyed; the already-NEW entry stays valid.
    assert _production_verify(allowlist_dir, old_entry_key, hmac_key=_NEW_KEY, monkeypatch=monkeypatch)
    assert _production_verify(allowlist_dir, new_entry_key, hmac_key=_NEW_KEY, monkeypatch=monkeypatch)

    after = yaml_path.read_text(encoding="utf-8")
    # Exactly one line changed, and it is a judge_metadata_signature line (the splice
    # touched only the OLD entry's signature; the pre-judge entry has no signature line).
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    assert len(before_lines) == len(after_lines)
    changed = [(b, a) for b, a in zip(before_lines, after_lines, strict=True) if b != a]
    assert len(changed) == 1
    assert changed[0][0].lstrip().startswith("judge_metadata_signature:")

    # Byte-level proof of WHICH entry changed: the OLD entry's original signature is
    # gone (re-keyed); the already-NEW entry's signature survives verbatim (skipped).
    assert _entry_signature(widget_finding, hmac_key=_OLD_KEY) not in after
    assert _entry_signature(gadget_finding, hmac_key=_NEW_KEY) in after
    # The non-judge-gated entry is wholly untouched.
    assert f"- key: {pre_judge_key}" in after
