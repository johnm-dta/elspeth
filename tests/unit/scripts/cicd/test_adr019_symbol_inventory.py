"""Tests for the ADR-019 migration symbol inventory."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.cicd.adr019_symbol_inventory import FindingKind, main, scan_file

FIXTURE_DIR = Path("tests/fixtures/cicd/adr019_symbol_inventory")


def test_positive_fixture_reports_every_finding_kind() -> None:
    findings = scan_file(FIXTURE_DIR / "positive.py", Path.cwd())
    kinds = {finding.kind for finding in findings}

    assert kinds == {
        FindingKind.IS_TERMINAL_ANNOTATION,
        FindingKind.IS_TERMINAL_ATTRIBUTE,
        FindingKind.IS_TERMINAL_KEYWORD,
        FindingKind.IS_TERMINAL_DICT_KEY,
        FindingKind.ROW_OUTCOME_STRING_COMPARE,
        FindingKind.TERMINAL_OUTCOME_STRING_COMPARE,
        FindingKind.TERMINAL_PATH_STRING_COMPARE,
        FindingKind.ROW_OUTCOME_STRING_MEMBERSHIP,
        FindingKind.TERMINAL_OUTCOME_STRING_MEMBERSHIP,
        FindingKind.TERMINAL_PATH_STRING_MEMBERSHIP,
    }


def test_negative_fixture_avoids_false_positives() -> None:
    findings = scan_file(FIXTURE_DIR / "negative.py", Path.cwd())

    assert findings == []


def test_cli_uses_allowlist_directory_and_emits_json_lines(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    allowed = tmp_path / "src/elspeth/contracts"
    denied = tmp_path / "src/elspeth/mcp"
    allowed.mkdir(parents=True)
    denied.mkdir(parents=True)
    source = "def f(record):\n    return record.is_terminal\n"
    (allowed / "enums.py").write_text(source, encoding="utf-8")
    (denied / "types.py").write_text(source, encoding="utf-8")

    exit_code = main(
        [
            "check",
            "--root",
            str(tmp_path),
            "--allowlist",
            "config/cicd/adr019_symbol_inventory",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert '"kind": "is_terminal_attribute"' in captured.out
    assert '"path": "src/elspeth/mcp/types.py"' in captured.out
    assert "src/elspeth/contracts/enums.py" not in captured.out
