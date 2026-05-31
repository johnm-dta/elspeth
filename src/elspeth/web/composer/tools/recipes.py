"""Composer recipes plane — pipeline-recipe discovery handler.

The recipe-application handler (``_execute_apply_pipeline_recipe``) lives
in ``tools/sessions.py`` next to ``_execute_set_pipeline`` — see the
``_execute_apply_pipeline_recipe`` docstring there for the delegation
rationale.
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
from elspeth.web.composer.tools.declarations import (
    ToolDeclaration,
    ToolKind,
)


def _execute_list_recipes(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Return discovery metadata for every registered pipeline recipe."""
    del context  # unused; signature uniformity with the other handlers.
    return _discovery_result(state, {"recipes": list_recipes()})


_LIST_RECIPES_DECLARATION = ToolDeclaration(
    name="list_recipes",
    handler=_execute_list_recipes,
    kind=ToolKind.DISCOVERY,
    description=(
        "List the registered pipeline recipes — deterministic scaffolds for common simple "
        "intents. Each recipe declares its required slots; apply_pipeline_recipe then "
        "instantiates the recipe with operator-supplied slot values. Recipes accelerate "
        "the highest-frequency 'classify CSV with LLM' and 'split rows by threshold' "
        "patterns; for shapes outside the recipe set, hand-author with set_pipeline."
    ),
    json_schema={"type": "object", "properties": {}, "required": []},
    cacheable=True,
)


TOOLS_IN_MODULE: tuple[ToolDeclaration, ...] = (_LIST_RECIPES_DECLARATION,)
"""Every tool declared in this module, in stable order.

``_dispatch.py`` aggregates this tuple alongside every other plane's
TOOLS_IN_MODULE to build the registered-tool universe."""
