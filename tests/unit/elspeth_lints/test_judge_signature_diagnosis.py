"""Tests for read-only signed judge metadata diagnosis.

The diagnosis command is the keyless agent/operator handoff surface: it must
inspect signed allowlist state without requiring the operator HMAC key, classify
what kind of repair is needed, and print exact commands that only an
operator-held environment may run.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from elspeth_lints.core.allowlist import JudgeVerdict, compute_judge_metadata_signature
from elspeth_lints.core.cli import main
from elspeth_lints.core.judge import JUDGE_POLICY_HASH

_HMAC_KEY = "x" * 32
_RECORDED_AT = "2024-01-01T00:00:00+00:00"
_MODEL = "claude-opus-4-7"
_RATIONALE = "original judge said the boundary was genuine"

_SOURCE = '''\
"""Synthetic module used in signature-diagnosis tests."""


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

    findings = list(scan_file((root / "plugins/widget.py").resolve(), root))
    r1_findings = [finding for finding in findings if finding.rule_id == "R1"]
    if len(r1_findings) != 1:
        raise AssertionError(f"expected one R1 finding, got {r1_findings!r}")
    return r1_findings[0]


def _canonical_key(finding: Any) -> str:
    key = finding.canonical_key
    if callable(key):
        key = key()
    if not isinstance(key, str):
        raise AssertionError(f"canonical_key must be str, got {type(key).__name__}")
    return key


def _write_v2_entry(
    allowlist_dir: Path,
    *,
    source_root: Path,
    scope_fingerprint: str | None = None,
    include_signature: bool = True,
    rationale: str = _RATIONALE,
) -> str:
    finding = _live_widget_finding(source_root)
    key = _canonical_key(finding)
    stored_scope = finding.scope_fingerprint if scope_fingerprint is None else scope_fingerprint
    signature = compute_judge_metadata_signature(
        key=key,
        ast_path=finding.ast_path,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime.fromisoformat(_RECORDED_AT),
        judge_model=_MODEL,
        judge_rationale=rationale,
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
        f"    {rationale}",
        "  judge_signature_version: 2",
        f"  scope_fingerprint: '{stored_scope}'",
        "  judge_transport: openrouter",
        f"  ast_path: '{finding.ast_path}'",
    ]
    if include_signature:
        lines.append(f"  judge_metadata_signature: '{signature}'")
    (allowlist_dir / "plugins.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return key


def _write_v1_entry(allowlist_dir: Path, *, source_root: Path) -> str:
    finding = _live_widget_finding(source_root)
    key = _canonical_key(finding)
    file_fingerprint = hashlib.sha256((source_root / "plugins/widget.py").read_bytes()).hexdigest()
    signature = compute_judge_metadata_signature(
        key=key,
        ast_path=finding.ast_path,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime.fromisoformat(_RECORDED_AT),
        judge_model=_MODEL,
        judge_rationale=_RATIONALE,
        judge_policy_hash=JUDGE_POLICY_HASH,
        signature_version=1,
        file_fingerprint=file_fingerprint,
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
        f"  file_fingerprint: '{file_fingerprint}'",
        f"  ast_path: '{finding.ast_path}'",
        f"  judge_metadata_signature: '{signature}'",
    ]
    (allowlist_dir / "plugins.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return key


def _run_diagnose(root: Path, allowlist_dir: Path, *extra_args: str) -> int:
    return main(
        [
            "diagnose-judge-signatures",
            "--root",
            str(root),
            "--allowlist-dir",
            str(allowlist_dir),
            *extra_args,
        ]
    )


def _run_sign(root: Path, allowlist_dir: Path, *extra_args: str) -> int:
    return main(
        [
            "sign-judge-signatures",
            "--root",
            str(root),
            "--allowlist-dir",
            str(allowlist_dir),
            *extra_args,
        ]
    )


def test_cli_diagnose_reports_shape_only_without_hmac_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    key = _write_v2_entry(allowlist_dir, source_root=root)
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)

    assert _run_diagnose(root, allowlist_dir) == 0

    captured = capsys.readouterr()
    assert "verification_mode: shape-only" in captured.out
    assert "OK_SHAPE_ONLY" in captured.out
    assert key in captured.out
    assert _HMAC_KEY not in captured.out


def test_standalone_diagnose_command_delegates_to_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from elspeth_lints.core.judge_signature_diagnosis import main as diagnose_main

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    key = _write_v2_entry(allowlist_dir, source_root=root)
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)

    assert diagnose_main(["--root", str(root), "--allowlist-dir", str(allowlist_dir)]) == 0

    captured = capsys.readouterr()
    assert "verification_mode: shape-only" in captured.out
    assert "OK_SHAPE_ONLY" in captured.out
    assert key in captured.out


def test_cli_diagnose_loads_authoritative_hmac_from_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    key = _write_v2_entry(allowlist_dir, source_root=root)
    env_file = tmp_path / "operator.env"
    env_file.write_text(
        f"ELSPETH_JUDGE_METADATA_HMAC_KEY={_HMAC_KEY}\nUNRELATED_SECRET=this-must-not-be-imported\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)
    monkeypatch.delenv("UNRELATED_SECRET", raising=False)

    try:
        assert _run_diagnose(root, allowlist_dir, "--env-file", str(env_file)) == 0

        captured = capsys.readouterr()
        assert "verification_mode: authoritative" in captured.out
        assert "OK_AUTHORITATIVE" in captured.out
        assert key in captured.out
        assert _HMAC_KEY not in captured.out
        assert "UNRELATED_SECRET" not in os.environ
    finally:
        monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)


def test_standalone_diagnose_loads_authoritative_hmac_from_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from elspeth_lints.core.judge_signature_diagnosis import main as diagnose_main

    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    key = _write_v2_entry(allowlist_dir, source_root=root)
    env_file = tmp_path / "operator.env"
    env_file.write_text(f"ELSPETH_JUDGE_METADATA_HMAC_KEY={_HMAC_KEY}\n", encoding="utf-8")
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)

    try:
        assert diagnose_main(["--root", str(root), "--allowlist-dir", str(allowlist_dir), "--env-file", str(env_file)]) == 0

        captured = capsys.readouterr()
        assert "verification_mode: authoritative" in captured.out
        assert "OK_AUTHORITATIVE" in captured.out
        assert key in captured.out
        assert _HMAC_KEY not in captured.out
    finally:
        monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)


def test_cli_diagnose_rejects_missing_env_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    _write_v2_entry(allowlist_dir, source_root=root)

    assert _run_diagnose(root, allowlist_dir, "--env-file", str(tmp_path / "missing.env")) == 2

    captured = capsys.readouterr()
    assert "diagnose-judge-signatures error: --env-file:" in captured.err
    assert "missing.env" in captured.err


def test_cli_diagnose_reports_scope_drift_with_rejustify_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    key = _write_v2_entry(allowlist_dir, source_root=root, scope_fingerprint="b" * 64)
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)

    assert _run_diagnose(root, allowlist_dir) == 1

    captured = capsys.readouterr()
    assert "SCOPE_BINDING_DRIFT" in captured.out
    assert key in captured.out
    assert "re-justify required" in captured.out
    assert "ELSPETH_JUDGE_METADATA_HMAC_KEY=<operator-held-key>" in captured.out
    assert "elspeth_lints.core.cli justify" in captured.out
    assert "--file-path plugins/widget.py" in captured.out
    assert "--rule R1" in captured.out
    assert "--symbol Widget.lookup" in captured.out


def test_cli_diagnose_reports_missing_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    key = _write_v2_entry(allowlist_dir, source_root=root, include_signature=False)
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)

    assert _run_diagnose(root, allowlist_dir) == 1

    captured = capsys.readouterr()
    assert "MISSING_SIGNATURE" in captured.out
    assert key in captured.out
    assert "re-justify required" in captured.out


def test_cli_diagnose_reports_v1_file_drift_with_migrate_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    key = _write_v1_entry(allowlist_dir, source_root=root)
    target.write_text(_SOURCE + "\n# harmless comment that changes only file bytes\n", encoding="utf-8")
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)

    assert _run_diagnose(root, allowlist_dir) == 1

    captured = capsys.readouterr()
    assert "V1_FILE_FINGERPRINT_DRIFT" in captured.out
    assert key in captured.out
    assert "migrate-judge-scope" in captured.out
    assert '--owner "$USER"' in captured.out


def test_cli_diagnose_reports_authoritative_signature_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    key = _write_v2_entry(allowlist_dir, source_root=root)
    yaml_path = allowlist_dir / "plugins.yaml"
    yaml_path.write_text(
        yaml_path.read_text(encoding="utf-8").replace(_RATIONALE, "tampered signed rationale"),
        encoding="utf-8",
    )
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _HMAC_KEY)

    assert _run_diagnose(root, allowlist_dir) == 1

    captured = capsys.readouterr()
    assert "verification_mode: authoritative" in captured.out
    assert "INVALID_SIGNATURE" in captured.out
    assert key in captured.out
    assert "re-justify required" in captured.out


def test_cli_sign_loads_env_removes_stale_entry_and_invokes_justify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    key = _write_v2_entry(allowlist_dir, source_root=root, scope_fingerprint="b" * 64)
    env_file = tmp_path / "operator.env"
    env_file.write_text(
        "\n".join(
            [
                f"ELSPETH_JUDGE_METADATA_HMAC_KEY={_HMAC_KEY}",
                "OPENROUTER_API_KEY=test-openrouter-key",
                "UNRELATED_SECRET=must-not-load",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for env_key in ("ELSPETH_JUDGE_METADATA_HMAC_KEY", "OPENROUTER_API_KEY", "UNRELATED_SECRET"):
        monkeypatch.delenv(env_key, raising=False)

    calls: list[Any] = []

    def fake_justify(args: Any) -> int:
        calls.append(args)
        return 0

    try:
        with patch("elspeth_lints.core.cli._run_justify", side_effect=fake_justify):
            assert _run_sign(root, allowlist_dir, "--env-file", str(env_file), "--owner", "test-operator") == 0

        captured = capsys.readouterr()
        assert "1 diagnosed repair(s)" in captured.out
        assert "SCOPE_BINDING_DRIFT" in captured.out
        assert len(calls) == 1
        assert calls[0].file_path == "plugins/widget.py"
        assert calls[0].rule == "R1"
        assert calls[0].symbol == "Widget.lookup"
        assert calls[0].owner == "test-operator"
        assert calls[0].judge_transport == "openrouter"
        assert calls[0].fingerprint == key.rsplit(":fp=", 1)[1]
        assert "payload is Tier-3 external data" in calls[0].rationale
        assert key not in (allowlist_dir / "plugins.yaml").read_text(encoding="utf-8")
        assert os.environ["OPENROUTER_API_KEY"] == "test-openrouter-key"
        assert "UNRELATED_SECRET" not in os.environ
    finally:
        for env_key in ("ELSPETH_JUDGE_METADATA_HMAC_KEY", "OPENROUTER_API_KEY", "UNRELATED_SECRET"):
            os.environ.pop(env_key, None)


def test_cli_sign_restores_stale_entry_when_justify_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    key = _write_v2_entry(allowlist_dir, source_root=root, scope_fingerprint="b" * 64)
    env_file = tmp_path / "operator.env"
    env_file.write_text(f"ELSPETH_JUDGE_METADATA_HMAC_KEY={_HMAC_KEY}\n", encoding="utf-8")
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)

    try:
        with patch("elspeth_lints.core.cli._run_justify", return_value=1):
            assert _run_sign(root, allowlist_dir, "--env-file", str(env_file), "--owner", "test-operator") == 1

        captured = capsys.readouterr()
        assert "restored the stale row" in captured.err
        assert key in (allowlist_dir / "plugins.yaml").read_text(encoding="utf-8")
    finally:
        os.environ.pop("ELSPETH_JUDGE_METADATA_HMAC_KEY", None)


def test_cli_sign_manifest_invokes_justify_for_new_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    env_file = tmp_path / "operator.env"
    env_file.write_text(f"ELSPETH_JUDGE_METADATA_HMAC_KEY={_HMAC_KEY}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """\
entries:
  - file_path: plugins/widget.py
    rule: R1
    symbol: Widget.lookup
    fingerprint: abc123
    rationale: payload is intentionally judged for this manifest repair
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)
    calls: list[Any] = []

    try:
        with patch("elspeth_lints.core.cli._run_justify", side_effect=lambda args: calls.append(args) or 0):
            assert (
                _run_sign(
                    root,
                    allowlist_dir,
                    "--env-file",
                    str(env_file),
                    "--owner",
                    "manifest-operator",
                    "--manifest",
                    str(manifest),
                )
                == 0
            )

        assert len(calls) == 1
        assert calls[0].file_path == "plugins/widget.py"
        assert calls[0].rule == "R1"
        assert calls[0].symbol == "Widget.lookup"
        assert calls[0].fingerprint == "abc123"
        assert calls[0].owner == "manifest-operator"
        assert calls[0].rationale == "payload is intentionally judged for this manifest repair"
    finally:
        os.environ.pop("ELSPETH_JUDGE_METADATA_HMAC_KEY", None)


def test_cli_sign_manifest_skips_already_signed_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    key = _write_v2_entry(allowlist_dir, source_root=root)
    fingerprint = key.rsplit(":fp=", 1)[1]
    env_file = tmp_path / "operator.env"
    env_file.write_text(f"ELSPETH_JUDGE_METADATA_HMAC_KEY={_HMAC_KEY}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""\
entries:
  - file_path: plugins/widget.py
    rule: R1
    symbol: Widget.lookup
    fingerprint: {fingerprint}
    rationale: payload already has healthy signed judge metadata
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)

    try:
        with patch("elspeth_lints.core.cli._run_justify") as fake_justify:
            assert (
                _run_sign(
                    root,
                    allowlist_dir,
                    "--env-file",
                    str(env_file),
                    "--owner",
                    "manifest-operator",
                    "--manifest",
                    str(manifest),
                )
                == 0
            )

        captured = capsys.readouterr()
        assert "nothing to sign" in captured.out
        assert fake_justify.call_count == 0
    finally:
        os.environ.pop("ELSPETH_JUDGE_METADATA_HMAC_KEY", None)


def test_cli_sign_manifest_failure_reports_no_stale_row_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    env_file = tmp_path / "operator.env"
    env_file.write_text(f"ELSPETH_JUDGE_METADATA_HMAC_KEY={_HMAC_KEY}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """\
entries:
  - file_path: plugins/widget.py
    rule: R1
    symbol: Widget.lookup
    fingerprint: abc123
    rationale: payload is intentionally judged for this manifest repair
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)

    try:
        with patch("elspeth_lints.core.cli._run_justify", return_value=1):
            assert (
                _run_sign(
                    root,
                    allowlist_dir,
                    "--env-file",
                    str(env_file),
                    "--owner",
                    "manifest-operator",
                    "--manifest",
                    str(manifest),
                )
                == 1
            )

        captured = capsys.readouterr()
        assert "manifest entry" in captured.err
        assert "no diagnosed stale row was removed" in captured.err
    finally:
        os.environ.pop("ELSPETH_JUDGE_METADATA_HMAC_KEY", None)


def test_cli_sign_manifest_failure_continues_to_later_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    env_file = tmp_path / "operator.env"
    env_file.write_text(f"ELSPETH_JUDGE_METADATA_HMAC_KEY={_HMAC_KEY}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """\
entries:
  - file_path: plugins/widget.py
    rule: R1
    symbol: Widget.lookup
    fingerprint: blocked123
    rationale: first suppression is rejected by judge
  - file_path: plugins/widget.py
    rule: R1
    symbol: Widget.lookup
    fingerprint: accepted456
    rationale: second suppression can still be signed
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)
    calls: list[Any] = []

    def fake_justify(args: Any) -> int:
        calls.append(args)
        return 1 if args.fingerprint == "blocked123" else 0

    try:
        with patch("elspeth_lints.core.cli._run_justify", side_effect=fake_justify):
            assert (
                _run_sign(
                    root,
                    allowlist_dir,
                    "--env-file",
                    str(env_file),
                    "--owner",
                    "manifest-operator",
                    "--manifest",
                    str(manifest),
                )
                == 1
            )

        captured = capsys.readouterr()
        assert [call.fingerprint for call in calls] == ["blocked123", "accepted456"]
        assert "continuing with remaining entries" in captured.err
        assert "completed with 1 failed justify call(s)" in captured.err
        assert "blocked123" in captured.err
    finally:
        os.environ.pop("ELSPETH_JUDGE_METADATA_HMAC_KEY", None)


def test_cli_sign_dry_run_prints_plan_without_hmac_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _target = _build_source_tree(tmp_path)
    allowlist_dir = _build_allowlist_dir(tmp_path)
    key = _write_v2_entry(allowlist_dir, source_root=root, scope_fingerprint="b" * 64)
    before = (allowlist_dir / "plugins.yaml").read_text(encoding="utf-8")
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)

    with patch("elspeth_lints.core.cli._run_justify") as fake_justify:
        assert _run_sign(root, allowlist_dir, "--owner", "dry-run-operator", "--dry-run") == 0

    captured = capsys.readouterr()
    assert "stale rows that would be removed" in captured.out
    assert "justify calls that would run" in captured.out
    assert key in captured.out
    assert fake_justify.call_count == 0
    assert (allowlist_dir / "plugins.yaml").read_text(encoding="utf-8") == before
