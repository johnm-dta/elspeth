"""Composer recipes plane — pipeline-recipe discovery handler.

The recipe-application handler (``_execute_apply_pipeline_recipe``) is
co-located with ``_execute_set_pipeline`` in ``tools/sessions.py`` because
the application path delegates state replacement to the set_pipeline
executor; keeping them in the same plane closes the cross-plane private-
access edge the slice 1-5 refactor introduced.
"""

from __future__ import annotations

from typing import Any

from elspeth.web.composer.recipes import list_recipes
from elspeth.web.composer.state import (
    CompositionState,
)
from elspeth.web.composer.tools._common import (
    ToolContext,
    ToolResult,
    _discovery_result,
)


def _execute_list_recipes(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Return discovery metadata for every registered pipeline recipe."""
    del context  # unused; signature uniformity with the other handlers.
    return _discovery_result(state, {"recipes": list_recipes()})
