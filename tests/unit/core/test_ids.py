"""Tests for neutral core ID primitives."""

from __future__ import annotations

import ast
import uuid
from pathlib import Path


def test_generate_id_lives_on_neutral_core_surface() -> None:
    """ID generation is a core primitive, not a landscape-private helper."""
    from elspeth.core.ids import generate_id
    from elspeth.core.landscape import _helpers

    generated = generate_id()

    assert len(generated) == 32
    assert uuid.UUID(hex=generated).hex == generated
    assert _helpers.generate_id is generate_id


def test_production_code_does_not_import_generate_id_from_landscape_helpers() -> None:
    """Production modules should depend on the neutral ID surface."""
    root = Path(__file__).parents[3] / "src" / "elspeth"
    offenders: list[tuple[str, int]] = []

    for source_path in root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if any(alias.name == "generate_id" for alias in node.names):
                imports_landscape_helpers = node.module == "elspeth.core.landscape._helpers" or (
                    node.level > 0 and node.module == "_helpers"
                )
                if imports_landscape_helpers:
                    offenders.append((str(source_path.relative_to(root)), node.lineno))

    assert offenders == []
