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


def _write_freeze_guard_governance_fixture(root: Path) -> Path:
    """Create a fixture whose code findings are suppressed but allowlist governance is bad."""
    fixture_root = root / "freeze-governance"
    fixture_root.mkdir()
    (fixture_root / "example.py").write_text(
        """from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Example:
    data: dict[str, str]

    def __post_init__(self):
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))
        object.__setattr__(self, "data2", MappingProxyType(dict(self.data)))
""",
        encoding="utf-8",
    )
    allowlist_dir = fixture_root / "config" / "cicd" / "enforce_freeze_guards"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "_defaults.yaml").write_text(
        """version: 1
defaults:
  fail_on_stale: true
  fail_on_expired: true
  allowlist_budget:
    max_total_entries: 0
""",
        encoding="utf-8",
    )
    (allowlist_dir / "fixture.yaml").write_text(
        """allow_hits:
- key: stale.py:FG1:_module_:fp=deadbeef
  owner: test-owner
  reason: stale exact entry fixture
  safety: test
  expires: null
- key: expired.py:FG2:_module_:fp=cafebabe
  owner: test-owner
  reason: expired exact entry fixture
  safety: test
  expires: 2000-01-01
per_file_rules:
- pattern: example.py
  rules:
  - FG1
  reason: suppress fixture FG1 findings but exceed max_hits
  expires: null
  max_hits: 1
- pattern: expired.py
  rules:
  - FG2
  reason: expired per-file rule fixture
  expires: 2000-01-01
- pattern: unused.py
  rules:
  - FG3
  reason: unused per-file rule fixture
  expires: null
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


def test_non_tier_allowlist_governance_failures_reach_json_output(tmp_path: Path) -> None:
    """Non-tier shared allowlists must fail full checks on governance debt."""
    fixture_root = _write_freeze_guard_governance_fixture(tmp_path)
    result = _run_cli(
        [
            "check",
            "--rules",
            "immutability.freeze_guards",
            "--root",
            str(fixture_root),
            "--format",
            "json",
        ]
    )

    assert result.returncode == 1, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    findings = json.loads(result.stdout)
    rule_ids = {finding["rule_id"] for finding in findings}
    assert {
        "allowlist.stale_entry",
        "allowlist.expired_entry",
        "allowlist.expired_rule",
        "allowlist.unused_rule",
        "allowlist.max_hits_exceeded",
        "allowlist.budget_exceeded",
    }.issubset(rule_ids)
    assert "FG1" not in rule_ids


def test_non_tier_allowlist_governance_is_distinct_in_text_and_sarif(tmp_path: Path) -> None:
    """Text and SARIF emit governance findings as distinct allowlist rule ids."""
    fixture_root = _write_freeze_guard_governance_fixture(tmp_path)
    text_result = _run_cli(
        [
            "check",
            "--rules",
            "immutability.freeze_guards",
            "--root",
            str(fixture_root),
            "--format",
            "text",
        ]
    )
    assert text_result.returncode == 1
    assert "allowlist.stale_entry" in text_result.stdout
    assert "allowlist.max_hits_exceeded" in text_result.stdout
    assert ": FG1:" not in text_result.stdout

    sarif_result = _run_cli(
        [
            "check",
            "--rules",
            "immutability.freeze_guards",
            "--root",
            str(fixture_root),
            "--format",
            "sarif",
        ]
    )
    assert sarif_result.returncode == 1
    sarif_payload = json.loads(sarif_result.stdout)
    sarif_rule_ids = {result["ruleId"] for run in sarif_payload["runs"] for result in run["results"]}
    assert "allowlist.expired_rule" in sarif_rule_ids
    assert "allowlist.budget_exceeded" in sarif_rule_ids
    assert "FG1" not in sarif_rule_ids
