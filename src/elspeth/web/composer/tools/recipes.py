"""Composer recipes plane — pipeline-recipe discovery and application."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.recipes import (
    RecipeValidationError,
    apply_recipe,
    list_recipes,
)
from elspeth.web.composer.redaction import (
    ApplyPipelineRecipeArgumentsModel,
)
from elspeth.web.composer.state import (
    CompositionState,
)
from elspeth.web.composer.tools._common import (
    ToolContext,
    ToolResult,
    _discovery_result,
    _failure_result,
)
from elspeth.web.composer.tools.sessions import (
    _execute_set_pipeline,
)


def _execute_list_recipes(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Return discovery metadata for every registered pipeline recipe."""
    del context  # unused; signature uniformity with the other handlers.
    return _discovery_result(state, {"recipes": list_recipes()})


def _execute_apply_pipeline_recipe(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Validate a recipe's slots, build set_pipeline args, and dispatch to set_pipeline.

    Tier-3 boundary: ``arguments`` is an LLM-supplied dict.  Validated
    via :class:`ApplyPipelineRecipeArgumentsModel` (the single source of
    truth for the argument schema — supersedes the deleted
    ``_TOOL_REQUIRED_PATHS["apply_pipeline_recipe"]`` entry in
    ``service.py``, rev-3 N7 / rev-4 M1).  On
    :class:`pydantic.ValidationError` the handler re-raises as
    :class:`ToolArgumentError` so the compose loop's ARG_ERROR routing
    at ``service.py:2480`` receives the right exception class.

    Semantic vs argument-shape failures
    ------------------------------------
    Pydantic enforces argument shape (type, required-fields, extra=forbid).
    The empty-``recipe_name`` semantic check and the
    :class:`RecipeValidationError` slot-shape check remain in this handler
    and produce recoverable ``_failure_result`` responses with repair
    hints (``Call list_recipes to discover available recipes``).  Two
    channels for two failure shapes (type vs semantic) — same pattern as
    :class:`SetSourceArgumentsModel` plugin-not-in-catalog handling.

    ``set_pipeline`` is full state replacement, so a ``replaced_pipeline_note`` is
    emitted to make the destructive replacement visible to the LLM/operator. The
    note is suppressed when the prior pipeline is empty — a no-op replacement
    needs no flag, and emitting one would be noise on a fresh-session apply.
    """
    try:
        validated = ApplyPipelineRecipeArgumentsModel.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="apply_pipeline_recipe arguments",
            expected="object conforming to ApplyPipelineRecipeArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc

    recipe_name = validated.recipe_name
    raw_slots = validated.slots
    if not recipe_name:
        # Empty-string recipe_name passes Pydantic's ``str`` validation
        # but the handler treats it as a recoverable semantic failure
        # with a repair-hint pointing the LLM at list_recipes (rather
        # than the generic ARG_ERROR envelope a Pydantic min_length=1
        # would produce).
        return _failure_result(
            state,
            "apply_pipeline_recipe requires a non-empty 'recipe_name' string. Call list_recipes to discover available recipes.",
        )

    try:
        pipeline_args = apply_recipe(recipe_name, dict(raw_slots))
    except RecipeValidationError as exc:
        return _failure_result(state, str(exc))

    # Capture pre-replacement counts BEFORE delegating to the destructive
    # set_pipeline path. Frozen-dataclass fields, so capturing the integers
    # now is sufficient — the post-call result.updated_state is a fresh
    # CompositionState produced by set_pipeline.
    pre_source_present = state.source is not None
    pre_node_count = len(state.nodes)
    pre_output_count = len(state.outputs)

    # Delegate to the existing set_pipeline executor — recipes produce the
    # exact arguments shape set_pipeline accepts, so validation and state
    # mutation flow through the canonical mutation path.
    result = _execute_set_pipeline(pipeline_args, state, context)

    # Only annotate successful replacements over a non-empty prior state.
    # On failure, set_pipeline returned ``state`` unchanged and the note
    # would be misleading. On a fresh-session apply, there is nothing to
    # call out and a note would be noise.
    if not result.success:
        return result
    if not (pre_source_present or pre_node_count or pre_output_count):
        return result

    note = (
        f"apply_pipeline_recipe replaced the existing pipeline "
        f"(prior state had source={'set' if pre_source_present else 'unset'}, "
        f"{pre_node_count} node(s), {pre_output_count} output(s)). "
        "Recipes are full-state scaffolds; the prior composition was discarded."
    )

    # Preserve any existing data payload from set_pipeline (e.g. inline-blob
    # creation summary) by merging into a single dict. set_pipeline's
    # ``data`` is currently None on the recipe path because recipes don't
    # use inline_blob, but merging is forward-compatible.
    existing_data = result.data
    if existing_data is None:
        merged_data: Any = {"replaced_pipeline_note": note}
    elif isinstance(existing_data, Mapping):
        merged_data = {**dict(existing_data), "replaced_pipeline_note": note}
    else:
        # set_pipeline contract: ``data`` is None or a Mapping. Anything
        # else is a contract drift bug — surface the note alongside in a
        # wrapper rather than silently dropping either.
        merged_data = {"replaced_pipeline_note": note, "set_pipeline_data": existing_data}

    return replace(result, data=merged_data)
