"""Structural contract for grouping the guided mode-transition routes."""

from __future__ import annotations

import ast
from pathlib import Path

GUIDED_ROUTES = Path(__file__).resolve().parents[5] / "src" / "elspeth" / "web" / "sessions" / "routes" / "composer" / "guided.py"


def test_guided_convert_follows_start_and_respond() -> None:
    """post_guided_convert is the last guided mode-transition route.

    The signed AST layout requires post_guided_convert to be the final guided
    handler (enforced by the signed-layout contract in
    tests/unit/web/composer/guided/test_no_chain_authoring_path.py), so it is
    defined after both post_guided_start and post_guided_respond — not grouped
    between them, as an earlier revision assumed.
    """
    module = ast.parse(GUIDED_ROUTES.read_text(encoding="utf-8"), filename=str(GUIDED_ROUTES))
    function_names = [node.name for node in module.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]

    assert function_names.index("post_guided_start") < function_names.index("post_guided_convert")
    assert function_names.index("post_guided_respond") < function_names.index("post_guided_convert")
