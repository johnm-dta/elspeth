"""Tests for the --allowlist-dir CLI override (Plan A Task 8).

The --allowlist-dir flag at core/cli.py was historically parsed but never
read — _run_check built RuleContext with no override. This module covers the
wired-through behaviour:

1. unset → each rule resolves its own per-rule default directory (no change).
2. set to an empty directory → no entries match → previously-suppressed
   findings emerge (gate still works; allowlist is just empty).
3. set to a non-existent path → exit 2 with a clear diagnostic.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ELSPETH_LINTS_SRC = REPO_ROOT / "elspeth-lints" / "src"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def _write_tier_model_fixture(root: Path) -> Path:
    """Create a tiny repo root whose default tier-model allowlist suppresses one finding."""
    fixture_root = root / "fixture"
    fixture_root.mkdir()
    (fixture_root / "demo.py").write_text(
        'def read_value(options):\n    return options.get("value")\n',
        encoding="utf-8",
    )
    allowlist_dir = fixture_root / "config" / "cicd" / "enforce_tier_model"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "web.yaml").write_text(
        """per_file_rules:
- pattern: demo.py
  rules:
  - R1
  reason: Isolated CLI fixture proves default per-rule allowlist discovery.
  expires: null
  max_hits: 1
""",
        encoding="utf-8",
    )
    return fixture_root


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(ELSPETH_LINTS_SRC)}
    return subprocess.run(
        [str(PYTHON), "-m", "elspeth_lints.core.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(REPO_ROOT),
    )


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 13),
    reason="Python 3.13 is the canonical tier-model lint runtime",
)
def test_allowlist_dir_unset_uses_per_rule_defaults(tmp_path: Path) -> None:
    """When --allowlist-dir is unset, rules resolve their own default directories."""
    fixture_root = _write_tier_model_fixture(tmp_path)
    result = _run_cli(["check", "--rules", "trust_tier.tier_model", "--root", str(fixture_root)])
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"


def test_allowlist_dir_overrides_per_rule_default(tmp_path: Path) -> None:
    """A shadow (empty) allowlist directory removes all suppressions."""
    fixture_root = _write_tier_model_fixture(tmp_path)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    # Empty directory → no allowlist entries → every previously-suppressed
    # finding emerges.
    result = _run_cli(
        [
            "check",
            "--rules",
            "trust_tier.tier_model",
            "--allowlist-dir",
            str(shadow),
            "--root",
            str(fixture_root),
            "--format",
            "json",
        ]
    )
    # Exit 1 = findings (gate worked); the shadow allowlist suppresses nothing.
    assert result.returncode == 1, f"expected findings, got rc={result.returncode}; stderr: {result.stderr}"
    findings = json.loads(result.stdout)
    assert [finding["file_path"] for finding in findings] == ["demo.py"]


def test_allowlist_dir_nonexistent_exits_2(tmp_path: Path) -> None:
    """A non-existent --allowlist-dir is a configuration error → exit 2."""
    missing = tmp_path / "does-not-exist"
    result = _run_cli(
        [
            "check",
            "--rules",
            "trust_tier.tier_model",
            "--allowlist-dir",
            str(missing),
            "--root",
            "src/elspeth",
        ]
    )
    assert result.returncode == 2
    assert "is not a directory" in result.stderr
