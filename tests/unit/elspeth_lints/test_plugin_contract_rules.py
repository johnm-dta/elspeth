"""Tests for plugin-contract elspeth-lints rules."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from elspeth_lints.core.protocols import RuleContext
from elspeth_lints.rules.plugin_contract.component_type import RULE as COMPONENT_TYPE_RULE
from elspeth_lints.rules.plugin_contract.plugin_hashes import RULE as PLUGIN_HASHES_RULE


def test_component_type_reports_data_config_without_component_type() -> None:
    findings = list(
        COMPONENT_TYPE_RULE.analyze(
            _tree("""
            class BadConfig(DataPluginConfig):
                path: str
            """),
            Path("bad.py"),
            RuleContext(root=Path(".")),
        )
    )

    assert [finding.rule_id for finding in findings] == ["CT1"]
    assert "BadConfig" in findings[0].message


def test_plugin_hashes_reports_missing_source_file_hash(tmp_path: Path) -> None:
    _write(
        tmp_path / "plugins" / "sources" / "missing_hash.py",
        """
        class MissingHashSource:
            name = "missing-hash"
            plugin_version = "1.0.0"
        """,
    )

    findings = list(
        PLUGIN_HASHES_RULE.analyze(
            ast.Module(body=[], type_ignores=[]),
            tmp_path,
            RuleContext(root=tmp_path),
        )
    )

    assert [finding.rule_id for finding in findings] == ["PH2"]
    assert "source_file_hash" in findings[0].message


def _tree(source: str) -> ast.Module:
    return ast.parse(textwrap.dedent(source))


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
