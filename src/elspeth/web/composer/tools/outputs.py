"""Composer outputs plane — output (sink-instance) mutation handlers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.redaction import (
    PatchOutputOptionsArgumentsModel,
)
from elspeth.web.composer.state import (
    CompositionState,
    OutputSpec,
)
from elspeth.web.composer.tools._common import (
    ToolContext,
    ToolResult,
    _apply_merge_patch,
    _attach_post_call_hints,
    _credential_wiring_contract_failure,
    _failure_result,
    _mutation_result,
    _prevalidate_sink,
    _validate_mutation_arguments,
    _validate_plugin_name,
    _validate_sink_path,
    validate_composer_file_sink_collision_policy,
)
from elspeth.web.composer.tools.declarations import (
    ToolDeclaration,
    ToolKind,
)


class _SetOutputArgumentsModel(BaseModel):
    sink_name: str
    plugin: str
    options: dict[str, Any]
    on_write_failure: str = "discard"

    model_config = ConfigDict(extra="forbid")


class _RemoveOutputArgumentsModel(BaseModel):
    sink_name: str

    model_config = ConfigDict(extra="forbid")


def _handle_set_output(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    return _execute_set_output(arguments, state, context)


_SET_OUTPUT_DECLARATION = ToolDeclaration(
    name="set_output",
    handler=_handle_set_output,
    kind=ToolKind.MUTATION,
    description="Add or replace a pipeline output (sink).",
    json_schema={
        "type": "object",
        "properties": {
            "sink_name": {
                "type": "string",
                "description": (
                    "Sink name. This string is BOTH the sink's identifier (used by "
                    "patch_output_options/remove_output) AND the connection-name the sink "
                    "consumes — it MUST equal some upstream's on_success value. Pick a name "
                    "describing the data being written; it does not need to match an upstream "
                    "node's id."
                ),
                "examples": ["lines_out", "scored_results", "errors_quarantine"],
            },
            "plugin": {"type": "string", "description": "Sink plugin name (e.g. 'csv', 'json')."},
            "options": {
                "type": "object",
                "description": (
                    "Plugin-specific config. For csv/json file sinks in runnable web pipelines, "
                    "include path, schema, and explicit collision_policy."
                ),
            },
            "on_write_failure": {
                "type": "string",
                "description": "How to handle per-row write failures. Use 'discard' to drop with audit record, or a sink name (e.g. 'results_failures') to divert failed rows to that failsink.",
                "default": "discard",
            },
        },
        "required": ["sink_name", "plugin", "options"],
    },
    augments_on_failure=True,
)


def _handle_remove_output(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    return _execute_remove_output(arguments, state, context)


_REMOVE_OUTPUT_DECLARATION = ToolDeclaration(
    name="remove_output",
    handler=_handle_remove_output,
    kind=ToolKind.MUTATION,
    description="Remove a pipeline output (sink) by name.",
    json_schema={
        "type": "object",
        "properties": {
            "sink_name": {"type": "string", "description": "Sink name to remove."},
        },
        "required": ["sink_name"],
    },
)


def _execute_set_output(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Add or replace a pipeline output (sink)."""
    validated = cast(_SetOutputArgumentsModel, _validate_mutation_arguments(_SetOutputArgumentsModel, args, "set_output arguments"))
    plugin = validated.plugin
    # Validate plugin exists in catalog
    plugin_error = _validate_plugin_name(context.catalog, "sink", plugin)
    if plugin_error is not None:
        return _failure_result(state, plugin_error)

    # S2: Validate sink path allowlist (mirrors source path check)
    sink_options = validated.options
    credential_error = _credential_wiring_contract_failure(
        state,
        component_id=validated.sink_name,
        component_type="output",
        options=sink_options,
    )
    if credential_error is not None:
        return credential_error
    path_error = _validate_sink_path(sink_options, context.data_dir)
    if path_error is not None:
        return _failure_result(state, path_error)

    prevalidation_error = _prevalidate_sink(plugin, sink_options)
    if prevalidation_error is not None:
        return _failure_result(state, prevalidation_error)
    collision_error = validate_composer_file_sink_collision_policy(
        plugin,
        sink_options,
        require_explicit=context.data_dir is not None,
    )
    if collision_error is not None:
        return _failure_result(state, collision_error)

    output = OutputSpec(
        name=validated.sink_name,
        plugin=plugin,
        options=sink_options,
        on_write_failure=validated.on_write_failure,
    )
    new_state = state.with_output(output)
    return _mutation_result(new_state, (validated.sink_name,))


def _execute_remove_output(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Remove a pipeline output (sink) by name."""
    del context  # unused; signature uniformity with the other handlers.
    validated = cast(
        _RemoveOutputArgumentsModel, _validate_mutation_arguments(_RemoveOutputArgumentsModel, args, "remove_output arguments")
    )
    sink_name = validated.sink_name
    new_state = state.without_output(sink_name)
    if new_state is None:
        return _failure_result(state, f"Output '{sink_name}' not found.")
    return _mutation_result(new_state, (sink_name,))


def _execute_patch_output_options(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Apply a merge-patch to an output's plugin options.

    Tier-3 boundary: ``args`` is an LLM-supplied dict.  Validated via the
    Pydantic redaction-bearing model :class:`PatchOutputOptionsArgumentsModel`
    (the single source of truth for the argument schema — supersedes the
    deleted ``_TOOL_REQUIRED_PATHS["patch_output_options"]`` entry in
    ``service.py``, rev-3 N7 / rev-4 M1).

    On :class:`pydantic.ValidationError` the handler re-raises as
    :class:`ToolArgumentError` so the compose loop's ARG_ERROR routing at
    ``service.py:2480`` receives the right exception class.
    """
    try:
        validated = PatchOutputOptionsArgumentsModel.model_validate(args)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="patch_output_options arguments",
            expected="object conforming to PatchOutputOptionsArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc
    sink_name = validated.sink_name
    patch = validated.patch
    current = next((o for o in state.outputs if o.name == sink_name), None)
    if current is None:
        return _failure_result(state, f"Output '{sink_name}' not found.")
    new_options = _apply_merge_patch(current.options, patch)
    credential_error = _credential_wiring_contract_failure(
        state,
        component_id=sink_name,
        component_type="output",
        options=new_options,
    )
    if credential_error is not None:
        return credential_error

    # S2: Validate patched sink paths against allowlist
    path_error = _validate_sink_path(new_options, context.data_dir)
    if path_error is not None:
        return _failure_result(state, path_error)

    prevalidation_error = _prevalidate_sink(current.plugin, new_options)
    if prevalidation_error is not None:
        return _failure_result(state, prevalidation_error)
    collision_error = validate_composer_file_sink_collision_policy(
        current.plugin,
        new_options,
        require_explicit=context.data_dir is not None,
    )
    if collision_error is not None:
        return _failure_result(state, collision_error)

    new_output = replace(current, options=new_options)
    new_state = state.with_output(new_output)
    return _mutation_result(new_state, (sink_name,))


def _handle_patch_output_options(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    # _execute_patch_output_options validates arguments via the Pydantic model
    # and re-raises as ToolArgumentError; PydanticValidationError cannot
    # escape into this caller. Re-validation on the success branch is
    # deterministic by the same model.
    result = _execute_patch_output_options(arguments, state, context)
    if not result.success:
        return result
    validated = PatchOutputOptionsArgumentsModel.model_validate(arguments)
    sink_name = validated.sink_name
    output = next((o for o in result.updated_state.outputs if o.name == sink_name), None)
    # Offensive programming: _execute_patch_output_options succeeded above, so
    # the output it just upserted MUST be on the post-mutation state. Absence
    # here would be a bug in _execute_patch_output_options' state-update path
    # (or in CompositionState.with_output), not a recoverable runtime branch.
    assert output is not None, (
        f"_execute_patch_output_options succeeded for output '{sink_name}' but "
        "the post-mutation state does not contain it — invariant violation."
    )
    return _attach_post_call_hints(
        result,
        context.catalog,
        plugin_type="sink",
        tool_name="patch_output_options",
        plugin_name=output.plugin,
        config_snapshot=output.options,
    )


_PATCH_OUTPUT_OPTIONS_DECLARATION = ToolDeclaration(
    name="patch_output_options",
    handler=_handle_patch_output_options,
    kind=ToolKind.MUTATION,
    description="Apply a shallow merge-patch to an output's options. "
    "Keys in the patch overwrite existing keys. "
    "Keys set to null are deleted. Missing keys are unchanged.",
    json_schema={
        "type": "object",
        "properties": {
            "sink_name": {
                "type": "string",
                "description": "Name of the output (sink) to patch.",
            },
            "patch": {
                "type": "object",
                "description": "Merge-patch to apply to output options.",
            },
        },
        "required": ["sink_name", "patch"],
    },
    augments_on_failure=True,
)


TOOLS_IN_MODULE: tuple[ToolDeclaration, ...] = (
    _SET_OUTPUT_DECLARATION,
    _REMOVE_OUTPUT_DECLARATION,
    _PATCH_OUTPUT_OPTIONS_DECLARATION,
)
"""Every tool declared in this module, in stable order.

``_dispatch.py`` aggregates this tuple alongside every other plane's
TOOLS_IN_MODULE to build the registered-tool universe."""
