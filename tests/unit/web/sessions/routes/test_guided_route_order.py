"""Structural contract for grouping the guided mode-transition routes."""

from __future__ import annotations

import ast
from pathlib import Path

GUIDED_ROUTES = Path(__file__).resolve().parents[5] / "src" / "elspeth" / "web" / "sessions" / "routes" / "composer" / "guided.py"


def test_guided_convert_is_grouped_between_start_and_respond() -> None:
    """The explicit conversion route belongs beside the guided start/respond routes."""
    module = ast.parse(GUIDED_ROUTES.read_text(encoding="utf-8"), filename=str(GUIDED_ROUTES))
    function_names = [node.name for node in module.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]

    assert function_names.index("post_guided_start") < function_names.index("post_guided_convert")
    assert function_names.index("post_guided_convert") < function_names.index("post_guided_respond")
