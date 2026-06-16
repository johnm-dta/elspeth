from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]

_OPTIMIZATION_SAFE_INVARIANT_GUARDS: tuple[tuple[str, Path], ...] = (
    ("_handle_upsert_node", Path("src/elspeth/web/composer/tools/transforms.py")),
    ("_handle_patch_node_options", Path("src/elspeth/web/composer/tools/transforms.py")),
    ("_handle_patch_output_options", Path("src/elspeth/web/composer/tools/outputs.py")),
)


def _assert_lines_in_function(module_path: Path, function_name: str) -> list[int]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return [child.lineno for child in ast.walk(node) if isinstance(child, ast.Assert)]
    raise AssertionError(f"{function_name} was not found in {module_path}")


@pytest.mark.parametrize(
    ("function_name", "relative_path"),
    _OPTIMIZATION_SAFE_INVARIANT_GUARDS,
    ids=lambda value: value if isinstance(value, str) else value.name,
)
def test_composer_tool_success_invariants_are_not_bare_asserts(function_name: str, relative_path: Path) -> None:
    """Post-success invariant checks must survive ``python -O``."""
    module_path = _REPO_ROOT / relative_path

    assert _assert_lines_in_function(module_path, function_name) == [], (
        f"{function_name} in {relative_path} uses bare assert for a runtime invariant; "
        "use an explicit raise so the guard survives python -O."
    )
