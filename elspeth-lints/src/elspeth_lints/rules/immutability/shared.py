"""Shared AST helpers for immutability rules."""

from __future__ import annotations

import ast
from pathlib import Path


def is_frozen_dataclass(node: ast.ClassDef) -> bool:
    """Return whether a class has a dataclass(frozen=True) decorator."""
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if not ((isinstance(func, ast.Name) and func.id == "dataclass") or (isinstance(func, ast.Attribute) and func.attr == "dataclass")):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "frozen" and _is_true_literal(keyword.value):
                return True
    return False


def target_name(target: ast.expr) -> str | None:
    """Return the assignment target name for simple name/attribute targets."""
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def source_line(source_lines: list[str], lineno: int) -> str:
    """Return one stripped source line for diagnostics."""
    if 1 <= lineno <= len(source_lines):
        return source_lines[lineno - 1].strip()
    return "<source unavailable>"


def display_path(file_path: Path, root: Path) -> str:
    """Return the path format used by legacy CI scripts for this root."""
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError:
        return file_path.as_posix()


def repo_relative_display_path(file_path: Path, root: Path) -> str:
    """Return the legacy repo-relative path for source roots under src/elspeth."""
    display_root = root.parent.parent if root.name == "elspeth" and root.parent.name == "src" else root
    try:
        return file_path.relative_to(display_root).as_posix()
    except ValueError:
        return file_path.as_posix()


def allowlist_path_for_root(root: Path, directory_name: str) -> Path:
    """Find a CI allowlist directory from a scan root or repository cwd."""
    relative = Path("config") / "cicd" / directory_name
    candidates = [root, Path.cwd(), *root.parents]
    for candidate in candidates:
        path = candidate / relative
        if path.exists():
            return path
    return root / relative


def _is_true_literal(node: ast.expr) -> bool:
    return (isinstance(node, ast.Constant) and node.value is True) or (isinstance(node, ast.Name) and node.id == "True")
