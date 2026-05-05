"""Tests for the ADR-019 tests-tree migration inventory."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from scripts.cicd.adr019_test_inventory import FindingKind, main, scan_file


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def test_positive_fixture_reports_required_finding_kinds(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "tests/unit/test_old_expectations.py",
        """
        from elspeth.contracts.enums import RowOutcome, TerminalOutcome, TerminalPath
        from elspeth.core.landscape.schema import token_outcomes_table
        from sqlalchemy import select, text


        def test_old_expectations(result, row, outcome_values, actual, outcomes):
            assert result[0] == RowOutcome.FORKED
            actual = RowOutcome(row.outcome)
            assert actual == RowOutcome.CONSUMED_IN_BATCH
            assert outcomes == [RowOutcome.BUFFERED, RowOutcome.FAILED]
            assert RowOutcome.COMPLETED in outcome_values
            assert actual in {RowOutcome.COMPLETED, RowOutcome.FAILED}
            assert row.outcome == "routed"
            assert row.outcome in {"completed", "failed"}
            text("SELECT outcome FROM token_outcomes WHERE token_id = :token_id")
            select(token_outcomes_table.c.outcome).where(token_outcomes_table.c.is_terminal == 1)

            assert result.outcome == TerminalOutcome.SUCCESS
            assert result.path == TerminalPath.DEFAULT_FLOW
        """,
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
    _write(
        tmp_path / "tests/unit/contracts/test_enums.py",
        """
        from elspeth.contracts.enums import RowOutcome

        EXPECTED = [RowOutcome.COMPLETED]
        """,
    )
    _write(
        tmp_path / "tests/integration/test_real_output.py",
        """
        from elspeth.contracts.enums import RowOutcome

        def test_real_output(result):
            assert result.outcome == RowOutcome.COMPLETED
        """,
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
