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
    reason="tier-model allowlist fingerprints are version-specific; Python 3.13 is the canonical lint runtime",
)
def test_allowlist_dir_unset_uses_per_rule_defaults() -> None:
    """When --allowlist-dir is unset, rules resolve their own default directories."""
    # trust_tier.tier_model with its real allowlist → 0 findings (gate is green)
    result = _run_cli(["check", "--rules", "trust_tier.tier_model", "--root", "src/elspeth"])
    assert result.returncode == 0, f"stderr: {result.stderr}"


def test_allowlist_dir_overrides_per_rule_default(tmp_path: Path) -> None:
    """A shadow (empty) allowlist directory removes all suppressions."""
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
            "src/elspeth",
            "--format",
            "json",
        ]
    )
    # Exit 1 = findings (gate worked); the shadow allowlist suppresses nothing.
    assert result.returncode == 1, f"expected findings, got rc={result.returncode}; stderr: {result.stderr}"
    findings = json.loads(result.stdout)
    assert len(findings) > 0, "shadow allowlist should let real findings through"


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
