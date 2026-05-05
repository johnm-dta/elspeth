"""Tests for the ADR-019 RowOutcome migration guard."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from scripts.cicd.forbid_new_row_outcome import RULE_ID, main


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def test_detects_fnr1_and_hardcoded_row_outcome_value_strings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(
        tmp_path / "src/elspeth/core/example.py",
        """
        from elspeth.contracts.enums import RowOutcome


        def check(row, outcome, status):
            if RowOutcome.COMPLETED:
                pass
            if outcome == "quarantined":
                pass
            if "completed" == row.outcome:
                pass
            if outcome in {"completed", "failed"}:
                pass
            if row.outcome not in {"diverted", "buffered"}:
                pass
            if outcome == "active":
                pass
            if status in {"completed", "failed"}:
                pass
        """,
    )

    exit_code = main(["check", "--root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert f"[{RULE_ID}]" in captured.out
    assert "[FNR2]" in captured.out
    assert "no-hardcoded-row-outcome-value-string" in captured.out
    assert "quarantined" in captured.out
    assert "completed,failed" in captured.out
    assert "active" not in captured.out


def test_fnr2_is_scoped_to_src_elspeth(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(
        tmp_path / "tests/unit/test_fixture.py",
        """
        def test_fixture(row, outcome):
            assert outcome == "completed"
            assert row.outcome in {"failed"}
        """,
    )

    exit_code = main(["check", "--root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""


def test_allowlist_suppresses_fnr1_and_fnr2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(
        tmp_path / "src/elspeth/contracts/enums.py",
        """
        from elspeth.contracts.enums import RowOutcome


        def compatibility(outcome):
            assert RowOutcome.COMPLETED
            assert outcome == "completed"
        """,
    )
    allowlist = tmp_path / "config/cicd/forbid_new_row_outcome"
    _write(
        allowlist / "migration_files.yaml",
        """
        allowed:
          - file: src/elspeth/contracts/enums.py
            justification: compatibility migration file
        """,
    )

    exit_code = main(
        [
            "check",
            "--root",
            str(tmp_path),
            "--allowlist",
            str(allowlist),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""
