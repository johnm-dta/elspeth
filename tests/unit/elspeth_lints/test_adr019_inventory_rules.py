"""Tests for ADR-019 inventory elspeth-lints rules."""

from __future__ import annotations

import ast
from pathlib import Path

from elspeth_lints.core.protocols import RuleContext
from elspeth_lints.rules.manifest.symbol_inventory import RULE as SYMBOL_INVENTORY_RULE
from elspeth_lints.rules.manifest.test_to_source_mapping import RULE as TEST_TO_SOURCE_MAPPING_RULE


def test_symbol_inventory_reports_is_terminal_attribute(tmp_path: Path) -> None:
    source = tmp_path / "src" / "elspeth" / "mcp" / "types.py"
    source.parent.mkdir(parents=True)
    source.write_text("def f(record):\n    return record.is_terminal\n", encoding="utf-8")
    root = tmp_path / "src" / "elspeth"

    findings = list(SYMBOL_INVENTORY_RULE.analyze(ast.Module(body=[], type_ignores=[]), root, RuleContext(root=root)))

    assert [finding.rule_id for finding in findings] == ["is_terminal_attribute"]
    assert findings[0].file_path == "src/elspeth/mcp/types.py"
    assert "is_terminal" in findings[0].message


def test_test_to_source_mapping_reports_row_outcome_compare(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"
    source = tests_root / "unit" / "test_old.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from elspeth.contracts.enums import RowOutcome\n\ndef test_old(result):\n    assert result.outcome == RowOutcome.COMPLETED\n",
        encoding="utf-8",
    )

    findings = list(TEST_TO_SOURCE_MAPPING_RULE.analyze(ast.Module(body=[], type_ignores=[]), tests_root, RuleContext(root=tests_root)))

    assert [finding.rule_id for finding in findings] == ["row_outcome_compare", "row_outcome_attribute"]
    assert findings[0].file_path == "tests/unit/test_old.py"
    assert "RowOutcome" in findings[0].message
