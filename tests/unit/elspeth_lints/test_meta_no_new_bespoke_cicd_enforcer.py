"""Tests for the meta gate that blocks new bespoke CI enforcer scripts."""

from __future__ import annotations

import json
from pathlib import Path

from pytest import CaptureFixture

from elspeth_lints.core.protocols import RuleContext
from elspeth_lints.rules.meta_no_new_bespoke_cicd_enforcer import RULE


def test_meta_gate_allows_manifested_legacy_enforcers(tmp_path: Path) -> None:
    """Existing enforce_*.py scripts are allowed only when tracked in the manifest."""
    _write_file(tmp_path / "scripts/cicd/enforce_existing.py", "print('legacy')\n")
    _write_file(
        tmp_path / ".github/workflows/ci.yaml",
        """
jobs:
  lint:
    steps:
      - run: python scripts/cicd/enforce_existing.py
""",
    )
    _write_file(
        tmp_path / "config/cicd/lint_migration_status.yaml",
        """
version: 1
rules:
  - old_script: scripts/cicd/enforce_existing.py
    new_rule: null
    status: pending
    migration_issue: elspeth-test
""",
    )

    findings = list(RULE.analyze_repository(tmp_path, RuleContext(root=tmp_path)))

    assert findings == []


def test_meta_gate_blocks_unexercised_pending_enforcers(tmp_path: Path) -> None:
    """Pending legacy scripts must be visibly exercised by CI while awaiting migration."""
    _write_file(tmp_path / "scripts/cicd/enforce_existing.py", "print('legacy')\n")
    _write_file(
        tmp_path / ".github/workflows/ci.yaml",
        """
jobs:
  lint:
    steps:
      - run: echo no legacy scripts here
""",
    )
    _write_file(
        tmp_path / "config/cicd/lint_migration_status.yaml",
        """
version: 1
rules:
  - old_script: scripts/cicd/enforce_existing.py
    new_rule: null
    status: pending
    migration_issue: elspeth-test
""",
    )

    findings = list(RULE.analyze_repository(tmp_path, RuleContext(root=tmp_path)))

    assert len(findings) == 1
    assert findings[0].rule_id == "meta.no-new-bespoke-cicd-enforcer"
    assert "pending" in findings[0].message
    assert "not exercised by CI" in findings[0].message
    assert "scripts/cicd/enforce_existing.py" in findings[0].message


def test_meta_gate_allows_manifested_adr019_inventory_scripts(tmp_path: Path) -> None:
    """The ADR-019 inventory scripts are tracked legacy scripts during migration."""
    _write_file(tmp_path / "scripts/cicd/adr019_symbol_inventory.py", "print('legacy')\n")
    _write_file(tmp_path / "scripts/cicd/adr019_test_inventory.py", "print('legacy')\n")
    _write_file(
        tmp_path / "config/cicd/lint_migration_status.yaml",
        """
version: 1
rules:
  - old_script: scripts/cicd/adr019_symbol_inventory.py
    new_rule: manifest.symbol_inventory
    status: shadow
    migration_issue: elspeth-test
  - old_script: scripts/cicd/adr019_test_inventory.py
    new_rule: manifest.test_to_source_mapping
    status: shadow
    migration_issue: elspeth-test
""",
    )

    findings = list(RULE.analyze_repository(tmp_path, RuleContext(root=tmp_path)))

    assert findings == []


def test_meta_gate_blocks_unmanifested_new_enforcers(tmp_path: Path) -> None:
    """A new bespoke enforce_*.py file fails until it is represented in the manifest."""
    _write_file(tmp_path / "scripts/cicd/enforce_new_thing.py", "print('new')\n")
    _write_file(
        tmp_path / "config/cicd/lint_migration_status.yaml",
        """
version: 1
rules: []
""",
    )

    findings = list(RULE.analyze_repository(tmp_path, RuleContext(root=tmp_path)))

    assert len(findings) == 1
    assert findings[0].rule_id == "meta.no-new-bespoke-cicd-enforcer"
    assert "scripts/cicd/enforce_new_thing.py" in findings[0].message


def test_meta_gate_blocks_unmanifested_workflow_invoked_policy_scripts(tmp_path: Path) -> None:
    """Workflow-invoked policy scripts are tracked even without the enforce_ prefix."""
    _write_file(tmp_path / "scripts/cicd/check_new_policy.py", "print('new')\n")
    _write_file(
        tmp_path / ".github/workflows/ci.yaml",
        """
jobs:
  policy:
    steps:
      - run: uv run python scripts/cicd/check_new_policy.py
""",
    )
    _write_file(
        tmp_path / "config/cicd/lint_migration_status.yaml",
        """
version: 1
rules: []
""",
    )

    findings = list(RULE.analyze_repository(tmp_path, RuleContext(root=tmp_path)))

    assert len(findings) == 1
    assert findings[0].rule_id == "meta.no-new-bespoke-cicd-enforcer"
    assert "scripts/cicd/check_new_policy.py" in findings[0].message


def test_meta_gate_allows_manifested_workflow_invoked_policy_scripts(tmp_path: Path) -> None:
    """Non-enforce policy scripts can be intentionally tracked in the migration manifest."""
    _write_file(tmp_path / "scripts/cicd/check_new_policy.py", "print('new')\n")
    _write_file(
        tmp_path / ".github/workflows/ci.yaml",
        """
jobs:
  policy:
    steps:
      - run: python3 scripts/cicd/check_new_policy.py
""",
    )
    _write_file(
        tmp_path / "config/cicd/lint_migration_status.yaml",
        """
version: 1
rules:
  - old_script: scripts/cicd/check_new_policy.py
    new_rule: null
    status: pending
    migration_issue: elspeth-test
""",
    )

    findings = list(RULE.analyze_repository(tmp_path, RuleContext(root=tmp_path)))

    assert findings == []


def test_cli_renders_meta_gate_sarif(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """The check CLI can render rule findings as SARIF."""
    from elspeth_lints.core.cli import main

    _write_file(tmp_path / "scripts/cicd/enforce_new_thing.py", "print('new')\n")
    _write_file(tmp_path / "config/cicd/lint_migration_status.yaml", "version: 1\nrules: []\n")

    exit_code = main(
        [
            "check",
            "--rules",
            "meta.no-new-bespoke-cicd-enforcer",
            "--root",
            str(tmp_path),
            "--format",
            "sarif",
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["version"] == "2.1.0"
    assert payload["runs"][0]["results"][0]["ruleId"] == "meta.no-new-bespoke-cicd-enforcer"


def test_cli_renders_meta_gate_github_annotation(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """The check CLI can render rule findings as GitHub annotations."""
    from elspeth_lints.core.cli import main

    _write_file(tmp_path / "scripts/cicd/enforce_new_thing.py", "print('new')\n")
    _write_file(tmp_path / "config/cicd/lint_migration_status.yaml", "version: 1\nrules: []\n")

    exit_code = main(
        [
            "check",
            "--rules",
            "meta.no-new-bespoke-cicd-enforcer",
            "--root",
            str(tmp_path),
            "--format",
            "github",
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("::error file=scripts/cicd/enforce_new_thing.py,line=1,col=1,title=meta.no-new-bespoke-cicd-enforcer::")


def _write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
