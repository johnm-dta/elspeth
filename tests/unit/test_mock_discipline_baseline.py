"""Zero-tolerance gate for direct unspecced ``unittest.mock`` constructors.

Test doubles should be named fakes, real domain objects, or mocks with a
meaningful ``spec``/``spec_set``/``autospec``/``wraps`` boundary. Direct
unspecced mock constructors hide interface drift and are not allowed.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOT = REPO_ROOT
PRUNE_DIRS = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".uv-cache",
        ".venv",
        ".worktrees",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)
MOCK_NAMES = frozenset(
    {
        "AsyncMock",
        "MagicMock",
        "Mock",
        "NonCallableMagicMock",
        "NonCallableMock",
        "PropertyMock",
    }
)
SPEC_KEYWORDS = frozenset({"autospec", "spec", "spec_set", "wraps"})


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_specced_mock_call(node: ast.Call) -> bool:
    return any(keyword.arg in SPEC_KEYWORDS for keyword in node.keywords)


def _iter_python_files(root: Path) -> list[Path]:
    python_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [dirname for dirname in dirnames if dirname not in PRUNE_DIRS]
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                python_files.append(Path(dirpath) / filename)
    return sorted(python_files)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path.relative_to(SCAN_ROOT))


def _unspecced_mock_calls_by_file() -> dict[str, list[str]]:
    calls_by_file: dict[str, list[str]] = {}
    for path in _iter_python_files(SCAN_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel_path = _display_path(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node) not in MOCK_NAMES:
                continue
            if _is_specced_mock_call(node):
                continue
            calls_by_file.setdefault(rel_path, []).append(f"{rel_path}:{node.lineno}:{node.col_offset}")
    return calls_by_file


def _failure_lines(calls_by_file: dict[str, list[str]]) -> list[str]:
    lines: list[str] = []
    for rel_path, calls in sorted(calls_by_file.items()):
        examples = ", ".join(calls[:5])
        suffix = "" if len(calls) <= 5 else f", ... (+{len(calls) - 5} more)"
        lines.append(f"{rel_path}: {len(calls)} unspecced direct mock call(s); examples: {examples}{suffix}")
    return lines


def test_unspecced_mock_gate_rejects_new_file_regressions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A new unspecced mock must fail the zero-tolerance guard."""
    (tmp_path / "changed.py").write_text(
        "from unittest.mock import AsyncMock, MagicMock\n\nasync def test_changed() -> None:\n    AsyncMock()\n    MagicMock()\n",
        encoding="utf-8",
    )
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "SCAN_ROOT", tmp_path)

    with pytest.raises(AssertionError, match=r"changed\.py"):
        test_no_unspecced_direct_mock_constructors()


def test_no_unspecced_direct_mock_constructors() -> None:
    calls_by_file = _unspecced_mock_calls_by_file()
    assert not calls_by_file, "direct unspecced unittest.mock constructors are forbidden:\n" + "\n".join(_failure_lines(calls_by_file))
