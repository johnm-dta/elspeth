"""Tests for ADR-019 inventory elspeth-lints rules."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from elspeth_lints.core.protocols import RuleContext
from elspeth_lints.rules.manifest.symbol_inventory import RULE as SYMBOL_INVENTORY_RULE
from elspeth_lints.rules.manifest.symbol_inventory.rule import FindingKind as SymbolFindingKind
from elspeth_lints.rules.manifest.symbol_inventory.rule import scan_file as scan_symbol_file
from elspeth_lints.rules.manifest.test_to_source_mapping import RULE as TEST_TO_SOURCE_MAPPING_RULE
from elspeth_lints.rules.manifest.test_to_source_mapping.rule import FindingKind as MappingFindingKind
from elspeth_lints.rules.manifest.test_to_source_mapping.rule import scan_file as scan_test_file

SYMBOL_FIXTURE_DIR = Path("tests/fixtures/cicd/symbol_inventory")


def test_symbol_inventory_reports_every_finding_kind() -> None:
    findings = scan_symbol_file(SYMBOL_FIXTURE_DIR / "positive.py", Path.cwd())
    kinds = {finding.kind for finding in findings}

    assert kinds == {
        SymbolFindingKind.IS_TERMINAL_ANNOTATION,
        SymbolFindingKind.IS_TERMINAL_ATTRIBUTE,
        SymbolFindingKind.IS_TERMINAL_KEYWORD,
        SymbolFindingKind.IS_TERMINAL_DICT_KEY,
        SymbolFindingKind.ROW_OUTCOME_STRING_COMPARE,
        SymbolFindingKind.TERMINAL_OUTCOME_STRING_COMPARE,
        SymbolFindingKind.TERMINAL_PATH_STRING_COMPARE,
        SymbolFindingKind.ROW_OUTCOME_STRING_MEMBERSHIP,
        SymbolFindingKind.TERMINAL_OUTCOME_STRING_MEMBERSHIP,
        SymbolFindingKind.TERMINAL_PATH_STRING_MEMBERSHIP,
    }


def test_symbol_inventory_negative_fixture_avoids_false_positives() -> None:
    findings = scan_symbol_file(SYMBOL_FIXTURE_DIR / "negative.py", Path.cwd())

    assert findings == []


def test_symbol_inventory_reports_is_terminal_attribute(tmp_path: Path) -> None:
    source = tmp_path / "src" / "elspeth" / "mcp" / "types.py"
    source.parent.mkdir(parents=True)
    source.write_text("def f(record):\n    return record.is_terminal\n", encoding="utf-8")
    root = tmp_path / "src" / "elspeth"

    findings = list(SYMBOL_INVENTORY_RULE.analyze(ast.Module(body=[], type_ignores=[]), root, RuleContext(root=root)))

    assert [finding.rule_id for finding in findings] == ["is_terminal_attribute"]
    assert findings[0].file_path == "src/elspeth/mcp/types.py"
    assert "is_terminal" in findings[0].message


def test_symbol_inventory_uses_directory_allowlist(tmp_path: Path) -> None:
    allowed = tmp_path / "src" / "elspeth" / "contracts"
    denied = tmp_path / "src" / "elspeth" / "mcp"
    allowed.mkdir(parents=True)
    denied.mkdir(parents=True)
    source = "def f(record):\n    return record.is_terminal\n"
    (allowed / "enums.py").write_text(source, encoding="utf-8")
    (denied / "types.py").write_text(source, encoding="utf-8")
    allowlist = tmp_path / "config" / "cicd" / "symbol_inventory"
    allowlist.mkdir(parents=True)
    (allowlist / "migration_files.yaml").write_text(
        textwrap.dedent(
            """
            per_file_rules:
              - pattern: "src/elspeth/contracts/enums.py"
                rules: ["manifest.symbol_inventory"]
                reason: "compatibility mapping fixture"
            """
        ).lstrip(),
        encoding="utf-8",
    )

    findings = list(SYMBOL_INVENTORY_RULE.analyze(ast.Module(body=[], type_ignores=[]), tmp_path, RuleContext(root=tmp_path)))

    assert [finding.file_path for finding in findings] == ["src/elspeth/mcp/types.py"]


def test_test_to_source_mapping_reports_required_finding_kinds(tmp_path: Path) -> None:
    legacy_name = "Row" + "Outcome"
    token_table = "token_" + "outcomes"
    token_table_symbol = token_table + "_table"
    raw_sql = f"SELECT outcome FROM {token_table} WHERE token_id = :token_id"
    source = _write(
        tmp_path / "tests/unit/test_old_expectations.py",
        """
        from elspeth.contracts.enums import LEGACY, TerminalOutcome, TerminalPath
        from elspeth.core.landscape.schema import TOKEN_TABLE_SYMBOL
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
            text("RAW_SQL")
            select(TOKEN_TABLE_SYMBOL.c.outcome).where(TOKEN_TABLE_SYMBOL.c.is_terminal == 1)

            assert result.outcome == TerminalOutcome.SUCCESS
            assert result.path == TerminalPath.DEFAULT_FLOW
        """.replace("LEGACY", legacy_name)
        .replace("TOKEN_TABLE_SYMBOL", token_table_symbol)
        .replace("RAW_SQL", raw_sql),
    )

    findings = scan_test_file(source, tmp_path)
    kinds = {finding.kind for finding in findings}

    assert kinds == {
        MappingFindingKind.ROW_OUTCOME_ATTRIBUTE,
        MappingFindingKind.ROW_OUTCOME_COMPARE,
        MappingFindingKind.ROW_OUTCOME_COLLECTION,
        MappingFindingKind.ROW_OUTCOME_MEMBERSHIP,
        MappingFindingKind.OLD_OUTCOME_STRING_COMPARE,
        MappingFindingKind.OLD_OUTCOME_STRING_MEMBERSHIP,
        MappingFindingKind.RAW_TOKEN_OUTCOMES_SQL,
        MappingFindingKind.TOKEN_OUTCOMES_SCHEMA_READ,
    }


def test_test_to_source_mapping_ignores_migrated_assertions_and_unrelated_strings(tmp_path: Path) -> None:
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

    assert scan_test_file(source, tmp_path) == []


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


def test_test_to_source_mapping_uses_directory_allowlist(tmp_path: Path) -> None:
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
    allowlist = tmp_path / "config" / "cicd" / "test_to_source_mapping"
    _write(
        allowlist / "migration_files.yaml",
        """
        per_file_rules:
          - pattern: "tests/unit/contracts/test_enums.py"
            rules: ["manifest.test_to_source_mapping"]
            reason: "compatibility mapping fixture"
        """,
    )

    findings = list(TEST_TO_SOURCE_MAPPING_RULE.analyze(ast.Module(body=[], type_ignores=[]), tmp_path, RuleContext(root=tmp_path)))

    assert [finding.rule_id for finding in findings] == ["row_outcome_compare", "row_outcome_attribute"]
    assert {finding.file_path for finding in findings} == {"tests/integration/test_real_output.py"}


def test_test_to_source_mapping_rule_uses_core_loader() -> None:
    """After Tasks 1+5 the rule must not define its own allowlist loader."""
    from elspeth_lints.rules.manifest.test_to_source_mapping import rule as r

    assert "_load_allowlist" not in vars(r), "degenerate loader must be removed"
    assert "_is_allowed" not in vars(r), "degenerate matcher must be removed"


def test_symbol_inventory_rule_uses_core_loader() -> None:
    """After Tasks 1+5 the rule must not define its own allowlist loader."""
    from elspeth_lints.rules.manifest.symbol_inventory import rule as r

    assert "_load_allowlist" not in vars(r), "degenerate loader must be removed"
    assert "_is_allowed" not in vars(r), "degenerate matcher must be removed"


def test_test_to_source_mapping_yaml_loads_with_core_loader() -> None:
    """The committed YAML must parse under the shared per_file_rules schema."""
    from elspeth_lints.core.allowlist import load_allowlist

    path = Path("config/cicd/test_to_source_mapping/migration_files.yaml")
    result = load_allowlist(path, valid_rule_ids={"manifest.test_to_source_mapping"})
    assert len(result.per_file_rules) == 3


def test_symbol_inventory_yaml_loads_with_core_loader() -> None:
    """The committed YAML must parse under the shared per_file_rules schema."""
    from elspeth_lints.core.allowlist import load_allowlist

    path = Path("config/cicd/symbol_inventory/migration_files.yaml")
    result = load_allowlist(path, valid_rule_ids={"manifest.symbol_inventory"})
    assert len(result.per_file_rules) == 0


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path
