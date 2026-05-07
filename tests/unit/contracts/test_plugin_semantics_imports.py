"""Verify L0 purity of the plugin semantics + assistance contract modules.

These modules sit in src/elspeth/contracts/ which is L0 — they may not
import runtime code from core/ (L1), engine/ (L2), or plugins/web/mcp/tui/
(L3). TYPE_CHECKING imports are annotation-only warnings in the tier model,
not runtime coupling. The CI script enforce_tier_model.py also catches this;
this test gives faster feedback during development.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = _PROJECT_ROOT / "src"

_FORBIDDEN_PREFIXES = (
    "elspeth.core",
    "elspeth.engine",
    "elspeth.plugins",
    "elspeth.web",
    "elspeth.mcp",
    "elspeth.composer_mcp",
    "elspeth.tui",
    "elspeth.cli",
    "elspeth.telemetry",
    "elspeth.testing",
)


def _is_type_checking_guard(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING" and isinstance(test.value, ast.Name) and test.value.id == "typing"
    )


class _RuntimeImportCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: list[str] = []
        self._type_checking_depth = 0

    def visit_Import(self, node: ast.Import) -> None:
        if self._type_checking_depth == 0:
            self.imports.extend(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._type_checking_depth == 0 and node.module is not None:
            self.imports.append(node.module)

    def visit_If(self, node: ast.If) -> None:
        if not _is_type_checking_guard(node.test):
            self.generic_visit(node)
            return

        self._type_checking_depth += 1
        for child in node.body:
            self.visit(child)
        self._type_checking_depth -= 1

        for child in node.orelse:
            self.visit(child)


def _module_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    collector = _RuntimeImportCollector()
    collector.visit(tree)
    return collector.imports


def _forbidden_l0_runtime_imports(imports: list[str]) -> list[str]:
    return [imp for imp in imports if any(imp == prefix or imp.startswith(f"{prefix}.") for prefix in _FORBIDDEN_PREFIXES)]


def test_type_checking_imports_do_not_count_as_runtime_l0_violations(tmp_path: Path) -> None:
    module = tmp_path / "contract_module.py"
    module.write_text(
        "from typing import TYPE_CHECKING\n"
        "from elspeth.contracts import PluginSchema\n"
        "if TYPE_CHECKING:\n"
        "    from elspeth.core.config import Settings\n"
    )

    assert _forbidden_l0_runtime_imports(_module_imports(module)) == []


def test_runtime_imports_inside_regular_branches_still_violate_l0(tmp_path: Path) -> None:
    module = tmp_path / "contract_module.py"
    module.write_text("if True:\n    from elspeth.core.config import Settings\n")

    assert _forbidden_l0_runtime_imports(_module_imports(module)) == ["elspeth.core.config"]


@pytest.mark.parametrize(
    "module_path",
    [
        "src/elspeth/contracts/plugin_semantics.py",
        "src/elspeth/contracts/plugin_assistance.py",
    ],
)
def test_module_does_not_import_above_l0(module_path: str):
    path = _PROJECT_ROOT / module_path
    violations = _forbidden_l0_runtime_imports(_module_imports(path))
    assert not violations, f"{module_path} imports L1+ modules: {violations}. Contracts must remain L0-pure."
