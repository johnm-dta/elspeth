"""Connection-provider boundary tests for Landscape persistence helpers."""

from __future__ import annotations

import ast
from pathlib import Path


def _protocol_methods(source_path: Path, class_name: str) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
    raise AssertionError(f"{class_name} not found in {source_path}")


def test_broad_landscape_connection_provider_lives_in_ports_module() -> None:
    """The broad DB port belongs to a boundary module, not _database_ops."""
    from elspeth.core.landscape import _database_ops, ports

    assert not hasattr(_database_ops, "LandscapeConnectionProvider")
    assert hasattr(ports, "LandscapeConnectionProvider")

    methods = _protocol_methods(Path(ports.__file__), "LandscapeConnectionProvider")
    assert methods == {"engine", "read_only_connection", "connection", "write_connection"}


def test_database_ops_uses_narrow_connection_provider_protocol() -> None:
    """DatabaseOps should depend only on the methods it calls."""
    from elspeth.core.landscape import _database_ops

    methods = _protocol_methods(Path(_database_ops.__file__), "DatabaseOpsConnectionProvider")
    assert methods == {"read_only_connection", "write_connection"}
