"""Composer secrets plane — secret-reference discovery, validation, and wiring."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from elspeth.contracts.freeze import deep_thaw
from elspeth.web.composer.state import (
    CompositionState,
)
from elspeth.web.composer.tools._common import (
    ToolContext,
    ToolResult,
    _discovery_result,
    _failure_result,
    _mutation_result,
    _secret_ref_placement_error,
)


def _handle_list_secret_refs(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    if context.secret_service is None or context.user_id is None:
        return _failure_result(state, "Secret tools require secret service context.")
    items = context.secret_service.list_refs(context.user_id)
    # Return inventory dicts — NEVER include values
    data = [{"name": item.name, "scope": item.scope, "available": item.available} for item in items]
    return _discovery_result(state, data)


def _handle_validate_secret_ref(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    if context.secret_service is None or context.user_id is None:
        return _failure_result(state, "Secret tools require secret service context.")
    name = arguments["name"]
    available = context.secret_service.has_ref(context.user_id, name)
    return _discovery_result(state, {"name": name, "available": available})


def _execute_wire_secret_ref(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    if context.secret_service is None or context.user_id is None:
        return _failure_result(state, "Secret tools require secret service context.")

    name = arguments["name"]
    target = arguments["target"]
    option_key = arguments["option_key"]
    target_id = arguments.get("target_id")

    # Validate the secret ref exists
    if not context.secret_service.has_ref(context.user_id, name):
        return _failure_result(state, f"Secret reference '{name}' not found or not accessible.")

    marker = {"secret_ref": name}

    if target == "source":
        if state.source is None:
            return _failure_result(state, "No source configured — set a source first.")
        patched_options = dict(deep_thaw(state.source.options))
        patched_options[option_key] = marker
        placement_error = _secret_ref_placement_error("source", state.source.plugin, patched_options)
        if placement_error is not None:
            return _failure_result(state, placement_error)
        new_source = replace(state.source, options=patched_options)
        new_state = state.with_source(new_source)
        return _mutation_result(new_state, ("source",))

    elif target == "node":
        if target_id is None:
            return _failure_result(state, "target_id is required for node targets.")
        node = next((n for n in state.nodes if n.id == target_id), None)
        if node is None:
            return _failure_result(state, f"Node '{target_id}' not found.")
        if node.node_type not in ("transform", "aggregation") or node.plugin is None:
            return _failure_result(
                state,
                "Secret references can only be wired into source, transform, aggregation, or output plugin options.",
            )
        patched_options = dict(deep_thaw(node.options))
        patched_options[option_key] = marker
        placement_error = _secret_ref_placement_error("transform", node.plugin, patched_options)
        if placement_error is not None:
            return _failure_result(state, placement_error)
        new_node = replace(node, options=patched_options)
        new_state = state.with_node(new_node)
        return _mutation_result(new_state, (target_id,))

    elif target == "output":
        if target_id is None:
            return _failure_result(state, "target_id is required for output targets.")
        output = next((o for o in state.outputs if o.name == target_id), None)
        if output is None:
            return _failure_result(state, f"Output '{target_id}' not found.")
        patched_options = dict(deep_thaw(output.options))
        patched_options[option_key] = marker
        placement_error = _secret_ref_placement_error("sink", output.plugin, patched_options)
        if placement_error is not None:
            return _failure_result(state, placement_error)
        new_output = replace(output, options=patched_options)
        new_state = state.with_output(new_output)
        return _mutation_result(new_state, (target_id,))

    else:
        return _failure_result(state, f"Unknown target type: '{target}'.")
