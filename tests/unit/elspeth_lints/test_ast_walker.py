"""Tests for shared AST walking and parse-error helpers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from elspeth_lints.core.ast_walker import (
    PythonFileReadError,
    parse_python_file,
    walk_function_own_scope,
)


def test_parse_python_file_returns_read_error_for_unicode_decode_error(tmp_path: Path) -> None:
    source_path = tmp_path / "invalid_utf8.py"
    source_path.write_bytes(b"\xff")

    result = parse_python_file(source_path)

    assert isinstance(result, PythonFileReadError)
    assert result.path == source_path
    assert result.error_type == "UnicodeDecodeError"
    assert "could not decode as UTF-8" in result.message
    assert "byte 0" in result.message


def test_parse_python_file_returns_read_error_for_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "locked.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")

    def raise_permission_error(self: Path, *args: object, **kwargs: object) -> str:
        assert self == source_path
        _ = (args, kwargs)
        raise PermissionError("synthetic permission fault")

    monkeypatch.setattr(Path, "read_text", raise_permission_error)

    result = parse_python_file(source_path)

    assert isinstance(result, PythonFileReadError)
    assert result.path == source_path
    assert result.error_type == "PermissionError"
    assert "permission denied:" in result.message
    assert "synthetic permission fault" in result.message


def test_parse_python_file_returns_read_error_for_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "vanished.py"

    def raise_os_error(self: Path, *args: object, **kwargs: object) -> str:
        assert self == source_path
        _ = (args, kwargs)
        raise OSError("synthetic I/O fault")

    monkeypatch.setattr(Path, "read_text", raise_os_error)

    result = parse_python_file(source_path)

    assert isinstance(result, PythonFileReadError)
    assert result.path == source_path
    assert result.error_type == "OSError"
    assert result.message == "OSError: synthetic I/O fault"


def test_walk_function_own_scope_keeps_comprehension_nodes_visible() -> None:
    tree = ast.parse("""
def handler(arguments):
    values = [item.get("k") for item in arguments]
    return values
""")
    func_node = tree.body[0]
    assert isinstance(func_node, ast.FunctionDef)

    nodes = list(walk_function_own_scope(func_node))

    assert any(isinstance(node, ast.ListComp) for node in nodes)
    assert any(isinstance(node, ast.comprehension) for node in nodes)
    assert any(isinstance(node, ast.Name) and node.id == "item" and isinstance(node.ctx, ast.Store) for node in nodes)
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "item"
        for node in nodes
    )
