"""Composer transforms plane — node, edge, and metadata graph-mutation handlers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Final, cast

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

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
    _validate_gate_route_parity,
)
from elspeth.web.composer.tools._common import (
    ToolContext,
    ToolResult,
    _apply_merge_patch,
    _attach_post_call_hints,
    _credential_wiring_contract_failure,
    _discovery_result,
    _failure_result,
    _mutation_result,
    _options_with_default_llm_reviews,
    _prevalidate_transform,
    _validate_aggregation_trigger,
    _validate_mutation_arguments,
    _validate_plugin_name,
    _validate_transform_provider_config_path,
)
from elspeth.web.composer.tools.declarations import (
    ToolDeclaration,
    ToolKind,
)
from elspeth.web.interpretation_state import composition_review_contract_error

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
    context: ToolContext,
) -> ToolResult:
    return _discovery_result(state, context.catalog.list_transforms())


_LIST_TRANSFORMS_DECLARATION = ToolDeclaration(
    name="list_transforms",
    handler=_handle_list_transforms,
    kind=ToolKind.DISCOVERY,
    description="List available transform plugins with name and summary.",
    json_schema={"type": "object", "properties": {}, "required": []},
    cacheable=True,
)


def _handle_list_sinks(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    return _discovery_result(state, context.catalog.list_sinks())


_LIST_SINKS_DECLARATION = ToolDeclaration(
    name="list_sinks",
    handler=_handle_list_sinks,
    kind=ToolKind.DISCOVERY,
    description="List available sink plugins with name and summary.",
    json_schema={"type": "object", "properties": {}, "required": []},
    cacheable=True,
)


_UPSERT_NODE_DECLARATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Unique node identifier."},
        "node_type": {
            "type": "string",
            "enum": ["transform", "gate", "aggregation", "coalesce"],
        },
        "plugin": {
            "type": ["string", "null"],
            "description": "Plugin name. Required for transform/aggregation. Null for gate/coalesce.",
        },
        "input": {
            "type": "string",
            "description": (
                "Connection-name string this node CONSUMES. MUST equal the value of some "
                "upstream's on_success (or routes value, or on_error) field. NOT the upstream "
                "node's id — connections are matched by string, not by graph topology. "
                "Example: if source.on_success='raw_url_rows', this node sets input='raw_url_rows'."
            ),
            "examples": ["raw_url_rows", "fetched_text", "scored_rows"],
        },
        "on_success": {
            "type": ["string", "null"],
            "description": (
                "Output connection. Required for transform/aggregation/coalesce. Null for "
                "gates (routing is via condition/routes). When set, this is the connection-name "
                "string the node PUBLISHES — some downstream input/sink_name MUST equal this "
                "value. The runtime matches strings, not topology."
            ),
            "examples": ["fetched_text", "scored_rows", "lines_out"],
        },
        "on_error": {"type": ["string", "null"], "description": "Error output connection (transform/aggregation only)."},
        "options": {"type": "object", "description": "Plugin-specific config (transform/aggregation only)."},
        "condition": {"type": ["string", "null"], "description": "Boolean expression (gate only). Evaluated per row."},
        "routes": {
            "type": ["object", "null"],
            "description": (
                "Route mapping {true: sink_or_connection_or_discard, false: sink_or_connection_or_discard} "
                "(gate only, mutually exclusive with fork_to). Use 'discard' to drop that route with "
                "an audited gate_discarded terminal outcome."
            ),
        },
        "fork_to": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Fork destinations — row is copied to all listed paths (gate only, mutually exclusive with routes).",
        },
        "branches": {
            "type": ["array", "object", "null"],
            "items": {"type": "string"},
            "additionalProperties": {"type": "string"},
            "description": (
                "Branches to merge (coalesce only). Use list form when branch identity and input "
                "connection are the same, or object form {branch_name: input_connection} when a "
                "branch flows through transforms before coalescing."
            ),
        },
        "policy": {"type": ["string", "null"], "description": "Merge trigger policy (coalesce only)."},
        "merge": {"type": ["string", "null"], "description": "Field merge strategy (coalesce only)."},
        "trigger": {
            "type": ["object", "null"],
            "description": "Optional early batch trigger config (aggregation only). Omit, null, or {} for end-of-source-only aggregation.",
            "additionalProperties": False,
            "properties": {
                "count": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "description": "Flush after this many accepted rows.",
                },
                "timeout_seconds": {
                    "type": ["number", "null"],
                    "exclusiveMinimum": 0,
                    "description": "Flush after this many seconds since the first accepted row.",
                },
                "condition": {
                    "type": ["string", "null"],
                    "description": "Boolean expression over row['batch_count'] and row['batch_age_seconds']; do not use end_of_source here.",
                },
            },
        },
        "output_mode": {
            "type": ["string", "null"],
            "enum": ["passthrough", "transform", None],
            "description": "Aggregation output mode (aggregation only). Defaults to 'transform' if omitted.",
        },
        "expected_output_count": {
            "type": ["integer", "null"],
            "description": "Expected number of output rows from aggregation (aggregation only). Optional; omit when output count depends on group_by distinct values.",
        },
    },
    "required": ["id", "node_type", "input"],
}


def _handle_upsert_node(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    # _execute_upsert_node validates arguments via _validate_mutation_arguments,
    # which raises ToolArgumentError on Pydantic failure — a
    # PydanticValidationError can never escape into this caller. Re-validation
    # on the success branch is deterministic by the same model; we only need
    # the validated.id to look up the post-mutation node for hint resolution.
    result = _execute_upsert_node(arguments, state, context)
    if not result.success:
        return result
    validated = _UpsertNodeArgumentsModel.model_validate(arguments)
    node_id = validated.id
    node = next((n for n in result.updated_state.nodes if n.id == node_id), None)
    # Offensive programming: _execute_upsert_node succeeded above, so the
    # node it just upserted MUST be on the post-mutation state. Absence
    # here would be a bug in state.with_node, not a runtime condition.
    assert node is not None, (
        f"_execute_upsert_node succeeded for node '{node_id}' but the post-mutation state does not contain it — invariant violation."
    )
    return _attach_post_call_hints(
        result,
        context.catalog,
        plugin_type="transform",
        tool_name="upsert_node",
        plugin_name=node.plugin,
        config_snapshot=node.options,
    )


_UPSERT_NODE_DECLARATION = ToolDeclaration(
    name="upsert_node",
    handler=_handle_upsert_node,
    kind=ToolKind.MUTATION,
    description=(
        "Add or update a pipeline node. "
        "Fields are node_type-dependent: "
        "transform/aggregation use plugin+options; "
        "gate uses condition+routes (or fork_to); "
        "coalesce uses branches+policy+merge. "
        "Omit fields that don't apply to your node_type."
    ),
    json_schema=_UPSERT_NODE_DECLARATION_JSON_SCHEMA,
    augments_on_failure=True,
)


def _handle_upsert_edge(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    return _execute_upsert_edge(arguments, state, context)


_UPSERT_EDGE_DECLARATION = ToolDeclaration(
    name="upsert_edge",
    handler=_handle_upsert_edge,
    kind=ToolKind.MUTATION,
    description=(
        "Add or update a connection between nodes. When the edge targets a sink, "
        "this also updates the source/node routing field used by runtime "
        "(on_success, on_error, gate routes, or fork destinations)."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Unique edge identifier."},
            "from_node": {"type": "string", "description": "Source node ID or 'source'."},
            "to_node": {"type": "string", "description": "Destination node ID or sink name."},
            "edge_type": {
                "type": "string",
                "enum": ["on_success", "on_error", "route_true", "route_false", "fork"],
            },
            "label": {"type": ["string", "null"], "description": "Display label."},
        },
        "required": ["id", "from_node", "to_node", "edge_type"],
        "examples": [
            {
                "id": "e_judge_layers_error",
                "from_node": "judge_layers",
                "to_node": "llm_failures",
                "edge_type": "on_error",
                "label": "LLM failures",
            }
        ],
    },
)


def _handle_remove_node(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    return _execute_remove_node(arguments, state, context)


_REMOVE_NODE_DECLARATION = ToolDeclaration(
    name="remove_node",
    handler=_handle_remove_node,
    kind=ToolKind.MUTATION,
    description="Remove a node and all its edges.",
    json_schema={
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Node ID to remove."},
        },
        "required": ["id"],
    },
)


def _handle_remove_edge(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    return _execute_remove_edge(arguments, state, context)


_REMOVE_EDGE_DECLARATION = ToolDeclaration(
    name="remove_edge",
    handler=_handle_remove_edge,
    kind=ToolKind.MUTATION,
    description="Remove an edge by ID.",
    json_schema={
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Edge ID to remove."},
        },
        "required": ["id"],
    },
)


def _handle_set_metadata(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    return _execute_set_metadata(arguments, state, context)


_SET_METADATA_DECLARATION = ToolDeclaration(
    name="set_metadata",
    handler=_handle_set_metadata,
    kind=ToolKind.MUTATION,
    description="Update pipeline metadata (name and description only).",
    json_schema={
        "type": "object",
        "properties": {
            "patch": {
                "type": "object",
                "description": "Partial metadata update. Only included fields are changed.",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
        "required": ["patch"],
    },
)


def _execute_upsert_node(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
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
        plugin_error = _validate_plugin_name(context.catalog, "transform", plugin)
        if plugin_error is not None:
            return _failure_result(state, plugin_error)

        batch_placement_error = _batch_aware_placement_error(node_id, node_type, plugin, validated.output_mode)
        if batch_placement_error is not None:
            return _failure_result(state, batch_placement_error)

        batch_required_error = _batch_aware_required_input_fields_error(node_id, plugin, node_options)
        if batch_required_error is not None:
            return _failure_result(state, batch_required_error)

        review_options = _options_with_default_llm_reviews(
            node_id=node_id,
            plugin=plugin,
            options=node_options,
        )
        prevalidation_error = _prevalidate_transform(plugin, review_options)
        if prevalidation_error is not None:
            return _failure_result(state, prevalidation_error)

        # S2: confine nested provider_config persist_directory (RAG retrieval).
        provider_path_error = _validate_transform_provider_config_path(node_options, context.data_dir)
        if provider_path_error is not None:
            return _failure_result(state, f"Node '{node_id}': {provider_path_error}")

    # Validate gate condition expression at composition time.
    # Gives the LLM immediate feedback on syntax/security errors.
    condition = validated.condition
    if node_type == "gate" and condition is not None:
        expr_error = _validate_gate_expression(condition)
        if expr_error is not None:
            return _failure_result(state, f"Node '{node_id}': {expr_error}")
        parity_error = _validate_gate_route_parity(condition, validated.routes)
        if parity_error is not None:
            return _failure_result(state, f"Node '{node_id}': {parity_error}")
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
        options=_options_with_default_llm_reviews(
            node_id=node_id,
            plugin=plugin,
            options=node_options,
        ),
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
    review_contract_error = composition_review_contract_error(new_state)
    if review_contract_error is not None:
        return _failure_result(state, review_contract_error)

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
    context: ToolContext,
) -> ToolResult:
    """Add or update an edge.

    When the edge targets an output (sink), synchronises the source
    node's connection field so that generate_yaml() produces a
    working pipeline.  Edges to non-output nodes are visual only.
    """
    del context  # unused; signature uniformity with the other handlers.
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
    context: ToolContext,
) -> ToolResult:
    """Remove a node and its edges."""
    del context  # unused; signature uniformity with the other handlers.
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
    context: ToolContext,
) -> ToolResult:
    """Remove an edge."""
    del context  # unused; signature uniformity with the other handlers.
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
    context: ToolContext,
) -> ToolResult:
    """Update pipeline metadata."""
    del context  # unused; signature uniformity with the other handlers.
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
    context: ToolContext,
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
    new_options: Mapping[str, Any] = _apply_merge_patch(current.options, patch)
    new_options = _options_with_default_llm_reviews(
        node_id=node_id,
        plugin=current.plugin,
        options=new_options,
    )
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

        # S2: confine nested provider_config persist_directory (RAG retrieval).
        # A merge-patch can introduce an escaping path just as upsert_node can.
        provider_path_error = _validate_transform_provider_config_path(new_options, context.data_dir)
        if provider_path_error is not None:
            return _failure_result(state, f"Node '{node_id}': {provider_path_error}")

    new_node = replace(current, options=new_options)
    new_state = state.with_node(new_node)
    review_contract_error = composition_review_contract_error(new_state)
    if review_contract_error is not None:
        return _failure_result(state, review_contract_error)
    return _mutation_result(new_state, (node_id,))


def _handle_patch_node_options(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    # _execute_patch_node_options validates arguments via the Pydantic model
    # and re-raises as ToolArgumentError; PydanticValidationError cannot
    # escape into this caller. Re-validation on the success branch is
    # deterministic by the same model.
    result = _execute_patch_node_options(arguments, state, context)
    if not result.success:
        return result
    validated = PatchNodeOptionsArgumentsModel.model_validate(arguments)
    node_id = validated.node_id
    node = next((n for n in result.updated_state.nodes if n.id == node_id), None)
    # Offensive programming: _execute_patch_node_options succeeded above, so
    # the node it just upserted MUST be on the post-mutation state. Absence
    # here would be a bug in state.with_node, not a runtime condition.
    assert node is not None, (
        f"_execute_patch_node_options succeeded for node '{node_id}' but the post-mutation state does not contain it — invariant violation."
    )
    return _attach_post_call_hints(
        result,
        context.catalog,
        plugin_type="transform",
        tool_name="patch_node_options",
        plugin_name=node.plugin,
        config_snapshot=node.options,
    )


_PATCH_NODE_OPTIONS_DECLARATION = ToolDeclaration(
    name="patch_node_options",
    handler=_handle_patch_node_options,
    kind=ToolKind.MUTATION,
    description="Apply a shallow merge-patch to a node's options. "
    "Keys in the patch overwrite existing keys. "
    "Keys set to null are deleted. Missing keys are unchanged. "
    "Do not use this for node routing fields such as on_success/on_error/input/routes; "
    "use upsert_edge or upsert_node for routing edits.",
    json_schema={
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": "ID of the node to patch.",
            },
            "patch": {
                "type": "object",
                "description": (
                    "Merge-patch to apply to plugin options only. "
                    "Node-level routing fields such as on_success, on_error, input, routes, "
                    "and fork_to are siblings of options; edit them with upsert_edge or upsert_node."
                ),
            },
        },
        "required": ["node_id", "patch"],
    },
    augments_on_failure=True,
)


TOOLS_IN_MODULE: tuple[ToolDeclaration, ...] = (
    _LIST_TRANSFORMS_DECLARATION,
    _LIST_SINKS_DECLARATION,
    _UPSERT_NODE_DECLARATION,
    _UPSERT_EDGE_DECLARATION,
    _REMOVE_NODE_DECLARATION,
    _REMOVE_EDGE_DECLARATION,
    _SET_METADATA_DECLARATION,
    _PATCH_NODE_OPTIONS_DECLARATION,
)
"""Every tool declared in this module, in stable order.

``_dispatch.py`` aggregates this tuple alongside every other plane's
TOOLS_IN_MODULE to build the registered-tool universe."""
