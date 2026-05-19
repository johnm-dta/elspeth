"""Tests for the elspeth-lints parity harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.cicd.parity_harness import load_manifest, run_parity


def test_empty_manifest_has_no_comparisons(tmp_path: Path) -> None:
    """A manifest with no shadow entries is a successful no-op."""
    manifest = tmp_path / "lint_migration_status.yaml"
    manifest.write_text("version: 1\nrules: []\n", encoding="utf-8")

    config = load_manifest(manifest)
    result = run_parity(config, root=tmp_path)

    assert result.ok
    assert result.comparisons == []


def test_matching_old_and_new_findings_pass(tmp_path: Path) -> None:
    """Old and new commands with identical normalized findings pass."""
    old_script = _write_emitter(tmp_path / "old.py", line=4)
    new_script = _write_emitter(tmp_path / "new.py", line=4)
    manifest = _write_manifest(tmp_path, old_script=old_script, new_script=new_script)

    result = run_parity(load_manifest(manifest), root=tmp_path)

    assert result.ok
    assert len(result.comparisons) == 1
    assert result.comparisons[0].missing_from_new == []
    assert result.comparisons[0].unexpected_in_new == []


def test_one_line_drift_fails(tmp_path: Path) -> None:
    """The harness catches a one-line drift between old and new outputs."""
    old_script = _write_emitter(tmp_path / "old.py", line=4)
    new_script = _write_emitter(tmp_path / "new.py", line=5)
    manifest = _write_manifest(tmp_path, old_script=old_script, new_script=new_script)

    result = run_parity(load_manifest(manifest), root=tmp_path)

    assert not result.ok
    assert len(result.comparisons) == 1
    comparison = result.comparisons[0]
    assert [finding.line for finding in comparison.missing_from_new] == [4]
    assert [finding.line for finding in comparison.unexpected_in_new] == [5]


def test_legacy_violations_mapping_matches_new_list(tmp_path: Path) -> None:
    """Legacy report_json-style payloads compare against elspeth-lints finding lists."""
    old_script = _write_emitter(tmp_path / "old.py", line=4, envelope="violations")
    new_script = _write_emitter(tmp_path / "new.py", line=4)
    manifest = _write_manifest(tmp_path, old_script=old_script, new_script=new_script)

    result = run_parity(load_manifest(manifest), root=tmp_path)

    assert result.ok


def _write_emitter(path: Path, *, line: int, envelope: str | None = None) -> Path:
    payload = [
        {
            "rule_id": "demo.rule",
            "file_path": "src/example.py",
            "line": line,
            "column": 2,
            "message": "demo finding",
            "fingerprint": f"demo:{line}:2",
        }
    ]
    emitted: object = {envelope: payload} if envelope is not None else payload
    path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import json",
                f"print(json.dumps({json.dumps(emitted)}))",
                "raise SystemExit(1)",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_manifest(tmp_path: Path, *, old_script: Path, new_script: Path) -> Path:
    manifest = tmp_path / "lint_migration_status.yaml"
    manifest.write_text(
        f"""
version: 1
rules:
  - old_script: scripts/cicd/enforce_demo.py
    new_rule: demo.rule
    status: shadow
    old_command: ["{sys.executable}", "{old_script}"]
    new_command: ["{sys.executable}", "{new_script}"]
""",
        encoding="utf-8",
    )
    return manifest
