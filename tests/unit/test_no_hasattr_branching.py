"""Mechanical gate against weak hasattr-based branching in tests."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"


def _is_direct_assert_surface_check(node: ast.Call, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    if isinstance(parent, ast.Assert):
        return parent.test is node
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
        grandparent = parents.get(parent)
        return isinstance(grandparent, ast.Assert) and grandparent.test is parent
    return False


def test_hasattr_in_tests_is_limited_to_direct_surface_assertions() -> None:
    """Allow API presence/absence assertions, but forbid branching on hasattr()."""
    violations: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "hasattr":
                continue
            if _is_direct_assert_surface_check(node, parents):
                continue
            rel_path = path.relative_to(REPO_ROOT)
            violations.append(f"{rel_path}:{node.lineno}:{node.col_offset}")

    assert violations == []
