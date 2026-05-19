"""Tests for the ADR-019 tests-tree migration inventory."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from scripts.cicd.adr019_test_inventory import FindingKind, main, scan_file


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def test_positive_fixture_reports_required_finding_kinds(tmp_path: Path) -> None:
    legacy_name = "Row" + "Outcome"
    source = _write(
        tmp_path / "tests/unit/test_old_expectations.py",
        """
        from elspeth.contracts.enums import LEGACY, TerminalOutcome, TerminalPath
        from elspeth.core.landscape.schema import token_outcomes_table
        from sqlalchemy import select, text


        def test_old_expectations(result, row, outcome_values, actual, outcomes):
            assert result[0] == LEGACY.FORKED
            actual = LEGACY(row.outcome)
            assert actual == LEGACY.CONSUMED_IN_BATCH
            assert outcomes == [LEGACY.BUFFERED, LEGACY.FAILED]
            assert LEGACY.COMPLETED in outcome_values
            assert actual in {LEGACY.COMPLETED, LEGACY.FAILED}
            assert row.outcome == "routed"
            assert row.outcome in {"completed", "failed"}
            text("SELECT outcome FROM token_outcomes WHERE token_id = :token_id")
            select(token_outcomes_table.c.outcome).where(token_outcomes_table.c.is_terminal == 1)

            assert result.outcome == TerminalOutcome.SUCCESS
            assert result.path == TerminalPath.DEFAULT_FLOW
        """.replace("LEGACY", legacy_name),
    )

    findings = scan_file(source, tmp_path)
    kinds = {finding.kind for finding in findings}

    assert kinds == {
        FindingKind.ROW_OUTCOME_ATTRIBUTE,
        FindingKind.ROW_OUTCOME_COMPARE,
        FindingKind.ROW_OUTCOME_COLLECTION,
        FindingKind.ROW_OUTCOME_MEMBERSHIP,
        FindingKind.OLD_OUTCOME_STRING_COMPARE,
        FindingKind.OLD_OUTCOME_STRING_MEMBERSHIP,
        FindingKind.RAW_TOKEN_OUTCOMES_SQL,
        FindingKind.TOKEN_OUTCOMES_SCHEMA_READ,
    }


def test_negative_fixture_ignores_migrated_assertions_and_unrelated_strings(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "tests/unit/test_new_expectations.py",
        """
        from elspeth.contracts.enums import TerminalOutcome, TerminalPath

        '''Commentary may mention token_outcomes.outcome during migration.'''


        def test_new_expectations(result, status):
            assert result.outcome == TerminalOutcome.SUCCESS
            assert result.path == TerminalPath.DEFAULT_FLOW
            assert status in {"completed", "failed"}
            payload = {"completed": True}
            note = "Early-exit path reports truthful counts from token_outcomes"
            migrated_sql = "SELECT outcome, path FROM token_outcomes WHERE token_id = :token_id"
            assert note
            return payload, migrated_sql
        """,
    )

    assert scan_file(source, tmp_path) == []


def test_cli_uses_directory_allowlist_and_emits_json_lines(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    legacy_name = "Row" + "Outcome"
    _write(
        tmp_path / "tests/unit/contracts/test_enums.py",
        """
        from elspeth.contracts.enums import LEGACY

        EXPECTED = [LEGACY.COMPLETED]
        """.replace("LEGACY", legacy_name),
    )
    _write(
        tmp_path / "tests/integration/test_real_output.py",
        """
        from elspeth.contracts.enums import LEGACY

        def test_real_output(result):
            assert result.outcome == LEGACY.COMPLETED
        """.replace("LEGACY", legacy_name),
    )
    allowlist = tmp_path / "config/cicd/adr019_test_inventory"
    _write(
        allowlist / "migration_files.yaml",
        """
        allowed:
          - file: tests/unit/contracts/test_enums.py
            justification: compatibility mapping fixture
        """,
    )

    exit_code = main(
        [
            "check",
            "--root",
            str(tmp_path / "tests"),
            "--allowlist",
            str(allowlist),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert '"kind": "row_outcome_compare"' in captured.out
    assert '"path": "tests/integration/test_real_output.py"' in captured.out
    assert "tests/unit/contracts/test_enums.py" not in captured.out


def test_cli_json_mode_emits_parity_findings(tmp_path: Path) -> None:
    legacy_name = "Row" + "Outcome"
    _write(
        tmp_path / "tests/unit/test_old_expectations.py",
        """
        from elspeth.contracts.enums import LEGACY

        def test_real_output(result):
            assert result.outcome == LEGACY.COMPLETED
        """.replace("LEGACY", legacy_name),
    )
    project_root = Path(__file__).resolve().parents[4]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/cicd/adr019_test_inventory.py",
            "check",
            "--root",
            str(tmp_path / "tests"),
            "--format",
            "json",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert [finding["rule_id"] for finding in payload] == [
        FindingKind.ROW_OUTCOME_COMPARE.value,
        FindingKind.ROW_OUTCOME_ATTRIBUTE.value,
    ]
    assert payload[0]["file_path"] == "tests/unit/test_old_expectations.py"
    assert "fingerprint" in payload[0]
    assert "message" in payload[0]
