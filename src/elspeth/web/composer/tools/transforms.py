"""Composer transforms plane — node, edge, and metadata graph-mutation handlers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Final, cast

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.redaction import (
    PatchNodeOptionsArgumentsModel,
)
from elspeth.web.composer.state import (
    CoalesceBranches,
    CompositionState,
    EdgeSpec,
    EdgeType,
    NodeSpec,
    NodeType,
    _batch_aware_placement_error,
    _batch_aware_required_input_fields_error,
    _validate_gate_expression,
)
from elspeth.web.composer.tools._common import (
    ToolResult,
    _apply_merge_patch,
    _attach_post_call_hints,
    _credential_wiring_contract_failure,
    _discovery_result,
    _failure_result,
    _mutation_result,
    _prevalidate_transform,
    _validate_aggregation_trigger,
    _validate_mutation_arguments,
    _validate_plugin_name,
)

_NODE_ROUTING_OPTION_PATCH_KEYS: Final[frozenset[str]] = frozenset({"input", "on_success", "on_error", "routes", "fork_to"})


class _UpsertNodeArgumentsModel(BaseModel):
    id: str
    node_type: NodeType
    input: str
    plugin: str | None = None
    on_success: str | None = None
    on_error: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    condition: str | None = None
    routes: dict[str, str] | None = None
    fork_to: list[str] | None = None
    branches: list[str] | dict[str, str] | None = None
    policy: str | None = None
    merge: str | None = None
    trigger: dict[str, Any] | None = None
    output_mode: str | None = None
    expected_output_count: int | None = None

    model_config = ConfigDict(extra="forbid")


class _UpsertEdgeArgumentsModel(BaseModel):
    id: str
    from_node: str
    to_node: str
    edge_type: EdgeType
    label: str | None = None

    model_config = ConfigDict(extra="forbid")


class _RemoveByIdArgumentsModel(BaseModel):
    id: str

    model_config = ConfigDict(extra="forbid")


class _SetMetadataPatchModel(BaseModel):
    name: str | None = None
    description: str | None = None

    model_config = ConfigDict(extra="forbid")


class _SetMetadataArgumentsModel(BaseModel):
    patch: _SetMetadataPatchModel

    model_config = ConfigDict(extra="forbid")


def _handle_list_transforms(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _discovery_result(state, catalog.list_transforms())


def _handle_upsert_node(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    result = _execute_upsert_node(arguments, state, catalog)
    # The node may be a gate or coalesce (plugin=None) — _attach handles
    # that case. Extract the node identity from validated args so we
    # look up the right entry on the post-mutation state.
    try:
        validated = _UpsertNodeArgumentsModel.model_validate(arguments)
    except PydanticValidationError:
        # Validation failed inside _execute_upsert_node; the result
        # carries the failure. Skip hint resolution.
        return result
    node_id = validated.id
    node = next((n for n in result.updated_state.nodes if n.id == node_id), None)
    if node is None:
        return result
    return _attach_post_call_hints(
        result,
        catalog,
        plugin_type="transform",
        tool_name="upsert_node",
        plugin_name=node.plugin,
        config_snapshot=node.options,
    )


def _handle_upsert_edge(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_upsert_edge(arguments, state)


def _handle_remove_node(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_remove_node(arguments, state)


def _handle_remove_edge(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_remove_edge(arguments, state)


def _handle_set_metadata(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_set_metadata(arguments, state)


def _execute_upsert_node(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
) -> ToolResult:
    """Add or update a pipeline node."""
    validated = cast(_UpsertNodeArgumentsModel, _validate_mutation_arguments(_UpsertNodeArgumentsModel, args, "upsert_node arguments"))
    node_id = validated.id
    node_type = validated.node_type
    plugin = validated.plugin
    node_options = validated.options
    credential_error = _credential_wiring_contract_failure(
        state,
        component_id=node_id,
        component_type="node",
        options=node_options,
    )
    if credential_error is not None:
        return credential_error

    # Validate plugin for types that require one.
    # Gates and coalesces intentionally have plugin=None (they're expression-based or
    # structural, not plugin-driven), so the "and plugin is not None" guard covers them.
    # NodeSpec documents this: "plugin: Plugin name. None for gates and coalesces."
    if node_type in ("transform", "aggregation") and plugin is not None:
        plugin_error = _validate_plugin_name(catalog, "transform", plugin)
        if plugin_error is not None:
            return _failure_result(state, plugin_error)

        batch_placement_error = _batch_aware_placement_error(node_id, node_type, plugin, validated.output_mode)
        if batch_placement_error is not None:
            return _failure_result(state, batch_placement_error)

        batch_required_error = _batch_aware_required_input_fields_error(node_id, plugin, node_options)
        if batch_required_error is not None:
            return _failure_result(state, batch_required_error)

        prevalidation_error = _prevalidate_transform(plugin, node_options)
        if prevalidation_error is not None:
            return _failure_result(state, prevalidation_error)

    # Validate gate condition expression at composition time.
    # Gives the LLM immediate feedback on syntax/security errors.
    condition = validated.condition
    if node_type == "gate" and condition is not None:
        expr_error = _validate_gate_expression(condition)
        if expr_error is not None:
            return _failure_result(state, f"Node '{node_id}': {expr_error}")
    if node_type == "aggregation":
        trigger_error = _validate_aggregation_trigger(validated.trigger)
        if trigger_error is not None:
            return _failure_result(state, f"Node '{node_id}': {trigger_error}")

    fork_to: tuple[str, ...] | None = tuple(validated.fork_to) if validated.fork_to is not None else None

    branches: CoalesceBranches | None = None
    if validated.branches is not None:
        branches = dict(validated.branches) if isinstance(validated.branches, Mapping) else tuple(validated.branches)

    node = NodeSpec(
        id=node_id,
        node_type=node_type,
        plugin=plugin,
        input=validated.input,
        on_success=validated.on_success,
        on_error=validated.on_error or ("discard" if node_type in ("transform", "aggregation") else None),
        options=node_options,
        condition=validated.condition,
        routes=validated.routes,
        fork_to=fork_to,
        branches=branches,
        policy=validated.policy,
        merge=validated.merge,
        trigger=validated.trigger,
        output_mode=validated.output_mode,
        expected_output_count=validated.expected_output_count,
    )

    new_state = state.with_node(node)

    # Affected: the node itself plus nodes with edges referencing it
    affected = {node_id}
    for edge in new_state.edges:
        if edge.from_node == node_id or edge.to_node == node_id:
            affected.add(edge.from_node)
            affected.add(edge.to_node)

    return _mutation_result(new_state, tuple(sorted(affected)))


def _execute_upsert_edge(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Add or update an edge.

    When the edge targets an output (sink), synchronises the source
    node's connection field so that generate_yaml() produces a
    working pipeline.  Edges to non-output nodes are visual only.
    """
    validated = cast(_UpsertEdgeArgumentsModel, _validate_mutation_arguments(_UpsertEdgeArgumentsModel, args, "upsert_edge arguments"))
    from_node = validated.from_node
    to_node = validated.to_node
    edge_type = validated.edge_type

    edge = EdgeSpec(
        id=validated.id,
        from_node=from_node,
        to_node=to_node,
        edge_type=edge_type,
        label=validated.label,
    )
    new_state = state.with_edge(edge)

    # Synchronise connection field when the edge targets an output.
    # generate_yaml() and the engine use on_success/on_error values
    # (not edges) to route data to sinks, so the connection field
    # must match the output name for the pipeline to work at runtime.
    output_names = {o.name for o in new_state.outputs}
    if to_node in output_names:
        if from_node == "source":
            if edge_type != "on_success":
                return _failure_result(state, "Source sink edges must use 'on_success'.")
            if new_state.source is not None and new_state.source.on_success != to_node:
                new_source = replace(new_state.source, on_success=to_node)
                new_state = new_state.with_source(new_source)
        else:
            node = next((n for n in new_state.nodes if n.id == from_node), None)
            if node is not None:
                if edge_type == "on_success":
                    if node.node_type == "gate":
                        return _failure_result(state, f"Gate '{from_node}' sink edges must use route_true, route_false, or fork.")
                    if node.on_success != to_node:
                        new_state = new_state.with_node(replace(node, on_success=to_node))
                elif edge_type == "on_error":
                    if node.node_type == "gate":
                        return _failure_result(state, f"Gate '{from_node}' sink edges must use route_true, route_false, or fork.")
                    if node.on_error != to_node:
                        new_state = new_state.with_node(replace(node, on_error=to_node))
                elif edge_type in ("route_true", "route_false"):
                    if node.node_type != "gate":
                        return _failure_result(state, f"Only gates can use '{edge_type}' edges to sinks.")
                    route_key = "true" if edge_type == "route_true" else "false"
                    routes = dict(node.routes or {})
                    if routes.get(route_key) != to_node:
                        routes[route_key] = to_node
                        new_state = new_state.with_node(replace(node, routes=routes))
                elif edge_type == "fork":
                    if node.node_type != "gate":
                        return _failure_result(state, "Only gates can use 'fork' edges to sinks.")
                    fork_targets = tuple(dict.fromkeys((*(node.fork_to or ()), to_node)))
                    if node.fork_to != fork_targets:
                        new_state = new_state.with_node(replace(node, fork_to=fork_targets))

    return _mutation_result(new_state, (from_node, to_node))


def _execute_remove_node(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Remove a node and its edges."""
    validated = cast(_RemoveByIdArgumentsModel, _validate_mutation_arguments(_RemoveByIdArgumentsModel, args, "remove_node arguments"))
    node_id = validated.id

    # Collect affected nodes before removal (edges that reference this node)
    affected = {node_id}
    for edge in state.edges:
        if edge.from_node == node_id or edge.to_node == node_id:
            affected.add(edge.from_node)
            affected.add(edge.to_node)

    new_state = state.without_node(node_id)
    if new_state is None:
        return _failure_result(state, f"Node '{node_id}' not found.")

    return _mutation_result(new_state, tuple(sorted(affected)))


def _execute_remove_edge(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Remove an edge."""
    validated = cast(_RemoveByIdArgumentsModel, _validate_mutation_arguments(_RemoveByIdArgumentsModel, args, "remove_edge arguments"))
    edge_id = validated.id

    # Find the edge to get affected nodes
    edge = next((e for e in state.edges if e.id == edge_id), None)
    if edge is None:
        return _failure_result(state, f"Edge '{edge_id}' not found.")

    affected = (edge.from_node, edge.to_node)
    new_state = state.without_edge(edge_id)
    if new_state is None:
        return _failure_result(state, f"Edge '{edge_id}' not found.")

    return _mutation_result(new_state, affected)


def _execute_set_metadata(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Update pipeline metadata."""
    validated = cast(_SetMetadataArgumentsModel, _validate_mutation_arguments(_SetMetadataArgumentsModel, args, "set_metadata arguments"))
    patch = validated.patch.model_dump(exclude_none=True)

    new_state = state.with_metadata(patch)
    return _mutation_result(new_state, ())


def _node_routing_option_patch_error(patch: Mapping[str, Any]) -> str | None:
    """Return guidance when plugin-option patches contain node routing fields."""
    if not (_NODE_ROUTING_OPTION_PATCH_KEYS & patch.keys()):
        return None
    for key in ("on_error", "on_success", "input", "routes", "fork_to"):
        if key not in patch:
            continue
        if key == "on_error":
            return (
                "on_error is a node-level routing field, not a plugin option. "
                "Use upsert_edge with edge_type='on_error' when routing failures to an existing sink, "
                "or use upsert_node with on_error as a sibling of options for other routing edits."
            )
        if key == "on_success":
            return (
                "on_success is a node-level routing field, not a plugin option. "
                "Use upsert_edge with edge_type='on_success' when routing success rows to an existing sink, "
                "or use upsert_node with on_success as a sibling of options for other routing edits."
            )
        if key == "input":
            return (
                "input is a node-level connection field, not a plugin option. "
                "Use upsert_node with input as a sibling of options to change the connection this node consumes."
            )
        if key in {"routes", "fork_to"}:
            return (
                f"{key} is a gate-level routing field, not a plugin option. "
                "Use upsert_edge with edge_type='route_true', edge_type='route_false', or edge_type='fork' "
                f"for sink routing, or use upsert_node with {key} as a sibling of options."
            )
    return None


def _execute_patch_node_options(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Apply a merge-patch to a node's plugin options.

    Tier-3 boundary: ``args`` is an LLM-supplied dict.  Validated via the
    Pydantic redaction-bearing model :class:`PatchNodeOptionsArgumentsModel`
    (the single source of truth for the argument schema — supersedes the
    deleted ``_TOOL_REQUIRED_PATHS["patch_node_options"]`` entry in
    ``service.py``, rev-3 N7 / rev-4 M1).

    On :class:`pydantic.ValidationError` the handler re-raises as
    :class:`ToolArgumentError` so the compose loop's ARG_ERROR routing at
    ``service.py:2480`` receives the right exception class.

    Routing-key guard: :func:`_node_routing_option_patch_error` rejects
    routing-field keys in ``patch`` (on_error, on_success, input, routes,
    fork_to).  This is a value-domain check that Pydantic cannot express;
    it runs AFTER Pydantic validation — same discipline as
    ``set_pipeline``'s blob_id/inline_blob mutual-exclusion check.
    """
    try:
        validated = PatchNodeOptionsArgumentsModel.model_validate(args)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="patch_node_options arguments",
            expected="object conforming to PatchNodeOptionsArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc
    node_id = validated.node_id
    patch = validated.patch
    current = next((n for n in state.nodes if n.id == node_id), None)
    if current is None:
        return _failure_result(state, f"Node '{node_id}' not found.")
    routing_patch_error = _node_routing_option_patch_error(patch)
    if routing_patch_error is not None:
        return _failure_result(state, routing_patch_error)
    new_options = _apply_merge_patch(current.options, patch)
    credential_error = _credential_wiring_contract_failure(
        state,
        component_id=node_id,
        component_type="node",
        options=new_options,
    )
    if credential_error is not None:
        return credential_error

    if current.node_type in ("transform", "aggregation") and current.plugin is not None:
        prevalidation_error = _prevalidate_transform(current.plugin, new_options)
        if prevalidation_error is not None:
            return _failure_result(state, prevalidation_error)

    new_node = NodeSpec(
        id=current.id,
        node_type=current.node_type,
        plugin=current.plugin,
        input=current.input,
        on_success=current.on_success,
        on_error=current.on_error,
        options=new_options,
        condition=current.condition,
        routes=current.routes,
        fork_to=current.fork_to,
        branches=current.branches,
        policy=current.policy,
        merge=current.merge,
        trigger=current.trigger,
        output_mode=current.output_mode,
        expected_output_count=current.expected_output_count,
    )
    new_state = state.with_node(new_node)
    return _mutation_result(new_state, (node_id,))


def _handle_patch_node_options(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    result = _execute_patch_node_options(arguments, state)
    if not result.success:
        return result
    try:
        validated = PatchNodeOptionsArgumentsModel.model_validate(arguments)
    except PydanticValidationError:
        return result
    node_id = validated.node_id
    node = next((n for n in result.updated_state.nodes if n.id == node_id), None)
    if node is None:
        return result
    return _attach_post_call_hints(
        result,
        catalog,
        plugin_type="transform",
        tool_name="patch_node_options",
        plugin_name=node.plugin,
        config_snapshot=node.options,
    )
