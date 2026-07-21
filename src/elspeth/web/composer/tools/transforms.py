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
    SpliceTransformArgumentsModel,
)
from elspeth.web.composer.state import (
    CoalesceBranches,
    CompositionState,
    EdgeSpec,
    EdgeType,
    NodeSpec,
    NodeType,
    SourceSpec,
    _batch_aware_placement_error,
    _batch_aware_required_input_fields_error,
    _validate_gate_expression,
    _validate_gate_route_parity,
    queue_node_contract_error,
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
    _plugin_policy_failure,
    _prevalidate_transform_for_context,
    _reserved_connection_names,
    _runtime_owned_llm_option_error,
    _validate_aggregation_trigger,
    _validate_mutation_arguments,
    _validate_plugin_name,
    _validate_transform_provider_config_path,
    _validate_transform_provider_config_policy,
)
from elspeth.web.composer.tools.declarations import (
    ToolDeclaration,
    ToolKind,
)
from elspeth.web.interpretation_state import (
    composition_review_contract_error,
    reconcile_authoritative_reviews,
    serialize_authoring_review_options,
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
    context: ToolContext,
) -> ToolResult:
    return _discovery_result(state, context.catalog.list_transforms())


_LIST_TRANSFORMS_DECLARATION = ToolDeclaration(
    name="list_transforms",
    handler=_handle_list_transforms,
    kind=ToolKind.DISCOVERY,
    description="List available transform plugins with name and summary.",
    json_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
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
    json_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    cacheable=True,
)


_UPSERT_NODE_DECLARATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Unique node identifier."},
        "node_type": {
            "type": "string",
            "enum": ["transform", "gate", "aggregation", "coalesce", "queue"],
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
    "additionalProperties": False,
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
    if node is None:
        raise AssertionError(
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
        "coalesce uses branches+policy+merge; "
        "queue is a structural fan-in point — set id == input to the shared "
        "connection name, omit plugin and every routing field (on_success/"
        "on_error/routes/fork_to), and options accepts only an optional "
        "description. Multiple producers may publish that name precisely "
        "because the queue is declared. "
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
        "additionalProperties": False,
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
        "additionalProperties": False,
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
        "additionalProperties": False,
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
        "additionalProperties": False,
    },
)


def _execute_upsert_queue_node(
    validated: _UpsertNodeArgumentsModel,
    state: CompositionState,
) -> ToolResult:
    """Insert/update a canonical structural queue node.

    Construct the NodeSpec from the validated arguments verbatim, run ONLY the
    intrinsic ``queue_node_contract_error`` guard, and mutate (``with_node``)
    only after that check passes. A malformed queue (id != input, any
    forbidden field, unknown/typed option) returns an ordinary
    ``_failure_result`` and leaves the exact prior state/version unchanged —
    ``with_node`` is never reached, so the mutation is atomic. A canonical
    queue succeeds and persists even when the resulting pipeline is
    incomplete (missing producers/downstream); completeness is validation
    telemetry surfaced on the returned ``ToolResult.validation``, not a
    mutation rejection.
    """
    fork_to: tuple[str, ...] | None = tuple(validated.fork_to) if validated.fork_to is not None else None
    branches: CoalesceBranches | None = None
    if validated.branches is not None:
        branches = dict(validated.branches) if isinstance(validated.branches, Mapping) else tuple(validated.branches)
    node = NodeSpec(
        id=validated.id,
        node_type="queue",
        plugin=validated.plugin,
        input=validated.input,
        on_success=validated.on_success,
        on_error=validated.on_error,
        options=dict(validated.options),
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
    contract_error = queue_node_contract_error(node)
    if contract_error is not None:
        return _failure_result(state, contract_error)

    new_state = state.with_node(node)
    affected = {validated.id}
    for edge in new_state.edges:
        if edge.from_node == validated.id or edge.to_node == validated.id:
            affected.add(edge.from_node)
            affected.add(edge.to_node)
    return _mutation_result(new_state, tuple(sorted(affected)))


def _execute_upsert_node(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Add or update a pipeline node."""
    validated = cast(_UpsertNodeArgumentsModel, _validate_mutation_arguments(_UpsertNodeArgumentsModel, args, "upsert_node arguments"))
    node_id = validated.id
    node_type = validated.node_type
    if node_type == "queue":
        # A queue is a structural pass-through fan-in point with no plugin,
        # no routing, and no plugin options — so the plugin/credential/review
        # gates below do not apply. The ONLY intrinsic constraint is
        # queue_node_contract_error (state.py), the single source of truth
        # shared with state validation and YAML generation. Pipeline
        # completeness (producers/downstream) stays validation telemetry, so an
        # orphan queue inserted during incremental authoring still persists.
        return _execute_upsert_queue_node(validated, state)
    plugin = validated.plugin
    node_options = validated.options
    runtime_owned_error = _runtime_owned_llm_option_error(
        plugin,
        node_options,
        tool_name="upsert_node",
    )
    if runtime_owned_error is not None:
        return _failure_result(state, f"Node '{node_id}': {runtime_owned_error}")
    credential_error = _credential_wiring_contract_failure(
        state,
        component_id=node_id,
        component_type="node",
        plugin_type="transform" if plugin is not None else None,
        plugin_name=plugin,
        options=node_options,
    )
    if credential_error is not None:
        return credential_error

    # Validate plugin for types that require one.
    # Gates and coalesces intentionally have plugin=None (they're expression-based or
    # structural, not plugin-driven), so the "and plugin is not None" guard covers them.
    # NodeSpec documents this: "plugin: Plugin name. None for gates and coalesces."
    if node_type in ("transform", "aggregation") and plugin is not None:
        plugin_error = _validate_plugin_name(context, "transform", plugin)
        if plugin_error is not None:
            return _plugin_policy_failure(state, plugin_error)

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
        prevalidation_error = _prevalidate_transform_for_context(context, plugin, review_options)
        if prevalidation_error is not None:
            return _failure_result(state, prevalidation_error)

        # Operator-profiled nodes carry their private provider config (retry
        # budget / provider binding) in the profile, injected only at lowering;
        # ``_prevalidate_transform_for_context`` above already validated the
        # LOWERED executable. Running the raw provider-config policy on the
        # authored options would false-positive on the absent private retry
        # budget (see the fuller rationale at set_pipeline in sessions.py).
        if "profile" not in node_options:
            provider_policy_error = _validate_transform_provider_config_policy(node_options, plugin=plugin)
            if provider_policy_error is not None:
                return _failure_result(state, f"Node '{node_id}': {provider_policy_error}")

        provider_path_error = _validate_transform_provider_config_path(node_options, context.data_dir, session_id=context.session_id)
        if provider_path_error is not None:
            return _failure_result(state, f"Node '{node_id}': {provider_path_error}")

    condition = validated.condition
    if node_type == "gate" and condition is not None:
        expr_error = _validate_gate_expression(condition)
        if expr_error is not None:
            return _failure_result(state, f"Node '{node_id}': {expr_error}")
        parity_error = _validate_gate_route_parity(condition, validated.routes)
        if parity_error is not None:
            return _failure_result(state, f"Node '{node_id}': {parity_error}", error_code="gate_route_labels_mismatch")
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


def _splice_connection_name(node_id: str, state: CompositionState) -> str | None:
    fragment = "".join(character if character.isalnum() or character in "_-" else "_" for character in node_id).strip("_-")
    if not fragment:
        fragment = "transform"
    reserved = (
        _reserved_connection_names(state)
        | set(state.sources)
        | {node.id for node in state.nodes}
        | {output.name for output in state.outputs}
    )
    for attempt in range(1, _SPLICE_CONNECTION_ATTEMPTS + 1):
        suffix = "" if attempt == 1 else f"_{attempt}"
        stem_length = _SPLICE_CONNECTION_MAX_LENGTH - len("_out") - len(suffix)
        candidate = f"{fragment[:stem_length]}_out{suffix}"
        if candidate not in reserved:
            return candidate
    return None


def _splice_edge_id(direct_edge_id: str, node_id: str) -> str:
    marker = "__splice__"
    node_fragment_length = max(1, _SPLICE_EDGE_ID_MAX_LENGTH - len(marker) - 1)
    suffix = f"{marker}{node_id[:node_fragment_length]}"
    stem_length = max(1, _SPLICE_EDGE_ID_MAX_LENGTH - len(suffix))
    return f"{direct_edge_id[:stem_length]}{suffix}"


def _normalized_splice_node_projection(node: NodeSpec) -> dict[str, Any]:
    return {
        "id": node.id,
        "plugin": node.plugin,
        "on_error": node.on_error or "discard",
        "options": serialize_authoring_review_options(node.options),
    }


def _clear_removed_sink_edge_route(state: CompositionState, edge: EdgeSpec) -> CompositionState:
    """Clear runtime routing that was written for a removed sink edge."""
    output_names = {output.name for output in state.outputs}
    if edge.to_node not in output_names:
        return state

    if edge.from_node in state.sources:
        if edge.edge_type != "on_success":
            return state
        source = state.sources[edge.from_node]
        if source.on_success == edge.to_node:
            return state.with_named_source(edge.from_node, replace(source, on_success="discard"))
        return state

    node = next((candidate for candidate in state.nodes if candidate.id == edge.from_node), None)
    if node is None:
        return state
    if edge.edge_type == "on_success":
        if node.on_success == edge.to_node:
            return state.with_node(replace(node, on_success=None))
        return state
    if edge.edge_type == "on_error":
        if node.on_error == edge.to_node:
            return state.with_node(replace(node, on_error=None))
        return state
    if edge.edge_type in ("route_true", "route_false"):
        route_key = "true" if edge.edge_type == "route_true" else "false"
        routes = dict(node.routes or {})
        if routes.get(route_key) != edge.to_node:
            return state
        del routes[route_key]
        return state.with_node(replace(node, routes=routes or None))
    if edge.edge_type == "fork":
        fork_to = tuple(target for target in (node.fork_to or ()) if target != edge.to_node)
        if fork_to == (node.fork_to or ()):
            return state
        return state.with_node(replace(node, fork_to=fork_to or None))
    return state


def _splice_predecessor(
    state: CompositionState,
    predecessor_id: str,
) -> SourceSpec | NodeSpec | None:
    if predecessor_id in state.sources:
        return state.sources[predecessor_id]
    return next((node for node in state.nodes if node.id == predecessor_id), None)


def _splice_topology_error(
    state: CompositionState,
    *,
    predecessor_id: str,
    successor: NodeSpec,
    inserted: NodeSpec | None = None,
) -> str | None:
    predecessor = _splice_predecessor(state, predecessor_id)
    if predecessor is None:
        return f"Splice predecessor '{predecessor_id}' not found."
    if type(predecessor) is NodeSpec and predecessor.node_type != "transform":
        return "splice_transform supports only source or transform predecessors on a direct linear path."
    if successor.node_type != "transform":
        return "splice_transform requires a transform successor on a direct linear path."
    for node in (predecessor, successor):
        if type(node) is NodeSpec and (
            node.condition is not None
            or node.routes is not None
            or node.fork_to is not None
            or node.branches is not None
            or node.node_type in {"gate", "coalesce", "queue"}
        ):
            return "splice_transform does not support gates, forks, queues, coalesces, or branched paths."
        if type(node) is NodeSpec and node.on_error not in (None, "discard"):
            return "splice_transform does not support predecessors or successors with routed error branches."

    if inserted is None:
        predecessor_output = predecessor.on_success
        if type(predecessor_output) is not str or not predecessor_output or predecessor_output == "discard":
            return "Splice predecessor has no direct on_success connection."
        if predecessor_output != successor.input:
            return "Splice predecessor and successor do not share one direct on_success connection."
        matching = [
            edge
            for edge in state.edges
            if edge.from_node == predecessor_id and edge.to_node == successor.id and edge.edge_type == "on_success"
        ]
        if len(matching) != 1:
            return "Splice path must have exactly one direct visual on_success edge."
        if any(edge.from_node == predecessor_id and edge is not matching[0] for edge in state.edges):
            return "Splice predecessor has an ambiguous or branched visual path."
        if any(edge.to_node == successor.id and edge is not matching[0] for edge in state.edges):
            return "Splice successor has an ambiguous or branched visual path."
        other_consumers = [node.id for node in state.nodes if node.id != successor.id and node.input == predecessor_output]
        if other_consumers or predecessor_output in {output.name for output in state.outputs}:
            return "Splice connection has multiple consumers or terminates at a sink."
        return None

    if (
        inserted.node_type != "transform"
        or inserted.condition is not None
        or inserted.routes is not None
        or inserted.fork_to is not None
        or inserted.branches is not None
        or inserted.policy is not None
        or inserted.merge is not None
        or inserted.trigger is not None
        or inserted.output_mode is not None
        or inserted.expected_output_count is not None
    ):
        return "Existing splice node is not a canonical transform."
    if predecessor.on_success != inserted.input or inserted.on_success != successor.input:
        return "Existing splice topology does not match the server-derived direct path."
    predecessor_consumers = [node.id for node in state.nodes if node.id != inserted.id and node.input == predecessor.on_success]
    inserted_consumers = [node.id for node in state.nodes if node.id != successor.id and node.input == inserted.on_success]
    output_names = {output.name for output in state.outputs}
    if predecessor_consumers or inserted_consumers or predecessor.on_success in output_names or inserted.on_success in output_names:
        return "Existing splice topology contains multiple consumers or terminates at a sink."
    try:
        inserted_index = next(index for index, node in enumerate(state.nodes) if node is inserted)
        successor_index = next(index for index, node in enumerate(state.nodes) if node is successor)
    except StopIteration:
        return "Existing splice topology is incomplete."
    if inserted_index + 1 != successor_index:
        return "Existing splice node is not immediately before its successor."
    predecessor_edges = [
        (index, edge)
        for index, edge in enumerate(state.edges)
        if edge.from_node == predecessor_id and edge.to_node == inserted.id and edge.edge_type == "on_success"
    ]
    successor_edges = [
        (index, edge)
        for index, edge in enumerate(state.edges)
        if edge.from_node == inserted.id and edge.to_node == successor.id and edge.edge_type == "on_success"
    ]
    if len(predecessor_edges) != 1 or len(successor_edges) != 1:
        return "Existing splice topology does not have exactly two direct visual edges."
    predecessor_edge_index, predecessor_edge = predecessor_edges[0]
    successor_edge_index, successor_edge = successor_edges[0]
    if successor_edge_index != predecessor_edge_index + 1:
        return "Existing splice edges are not in canonical adjacent order."
    if successor_edge.id != _splice_edge_id(predecessor_edge.id, inserted.id):
        return "Existing splice edge identity differs from the server-derived identity."
    allowed_edge_ids = {predecessor_edge.id, successor_edge.id}
    if any(
        edge.id not in allowed_edge_ids and (edge.from_node in {predecessor_id, inserted.id} or edge.to_node in {inserted.id, successor.id})
        for edge in state.edges
    ):
        return "Existing splice topology contains a conflicting visual path."
    return None


def _execute_splice_transform(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Atomically insert one transform on an existing direct linear path."""
    validated = cast(
        SpliceTransformArgumentsModel,
        _validate_mutation_arguments(SpliceTransformArgumentsModel, args, "splice_transform arguments"),
    )
    predecessor_id = validated.predecessor_id
    successor_id = validated.successor_id
    node_args = validated.node
    if predecessor_id == successor_id or node_args.id in {predecessor_id, successor_id}:
        return _failure_result(state, "Splice predecessor, successor, and inserted node IDs must be distinct.")
    if len({node.id for node in state.nodes}) != len(state.nodes):
        return _failure_result(state, "splice_transform refuses a state with duplicate node IDs.")
    if len({edge.id for edge in state.edges}) != len(state.edges):
        return _failure_result(state, "splice_transform refuses a state with duplicate edge IDs.")
    successor = next((node for node in state.nodes if node.id == successor_id), None)
    if successor is None:
        return _failure_result(state, f"Splice successor '{successor_id}' not found or is not a transform.")

    existing = next((node for node in state.nodes if node.id == node_args.id), None)
    if existing is not None:
        topology_error = _splice_topology_error(
            state,
            predecessor_id=predecessor_id,
            successor=successor,
            inserted=existing,
        )
        if topology_error is not None:
            return _failure_result(state, topology_error)
        prepared_replay = _prepare_transform_candidate(
            state=state,
            context=context,
            tool_name="splice_transform",
            node_id=node_args.id,
            node_type="transform",
            plugin=node_args.plugin,
            input_name=existing.input,
            on_success=existing.on_success,
            on_error=node_args.on_error,
            options=node_args.options,
        )
        if type(prepared_replay) is ToolResult:
            return prepared_replay
        replay_node = cast(NodeSpec, prepared_replay)
        try:
            identical = _normalized_splice_node_projection(replay_node) == _normalized_splice_node_projection(existing)
        except (KeyError, TypeError, ValueError):
            return _failure_result(state, f"Node '{node_args.id}' already exists with a divergent splice definition.")
        if not identical:
            return _failure_result(state, f"Node '{node_args.id}' already exists with a divergent splice definition.")
        return _mutation_result(
            state,
            (predecessor_id, node_args.id, successor_id),
            data={
                "already_applied": True,
                "predecessor_id": predecessor_id,
                "successor_id": successor_id,
                "inserted_node_id": node_args.id,
                "derived_connection": existing.on_success,
            },
        )

    if node_args.id in state.sources or node_args.id in {output.name for output in state.outputs}:
        return _failure_result(state, f"Inserted node ID '{node_args.id}' collides with an existing source or sink.")
    topology_error = _splice_topology_error(state, predecessor_id=predecessor_id, successor=successor)
    if topology_error is not None:
        return _failure_result(state, topology_error)
    predecessor = _splice_predecessor(state, predecessor_id)
    assert predecessor is not None
    predecessor_output = predecessor.on_success
    assert type(predecessor_output) is str
    direct_edge_index, direct_edge = next(
        (index, edge)
        for index, edge in enumerate(state.edges)
        if edge.from_node == predecessor_id and edge.to_node == successor_id and edge.edge_type == "on_success"
    )
    connection_name = _splice_connection_name(node_args.id, state)
    if connection_name is None:
        return _failure_result(state, "No bounded collision-free splice connection name is available.")
    new_edge_id = _splice_edge_id(direct_edge.id, node_args.id)
    if new_edge_id in {edge.id for edge in state.edges}:
        return _failure_result(state, f"Derived splice edge ID '{new_edge_id}' collides with an existing edge.")
    prepared = _prepare_transform_candidate(
        state=state,
        context=context,
        tool_name="splice_transform",
        node_id=node_args.id,
        node_type="transform",
        plugin=node_args.plugin,
        input_name=predecessor_output,
        on_success=connection_name,
        on_error=node_args.on_error,
        options=node_args.options,
    )
    if type(prepared) is ToolResult:
        return prepared
    prepared_node = cast(NodeSpec, prepared)

    successor_rewired = replace(successor, input=connection_name)
    successor_index = next(index for index, node in enumerate(state.nodes) if node is successor)
    nodes = (*state.nodes[:successor_index], prepared_node, successor_rewired, *state.nodes[successor_index + 1 :])
    predecessor_edge = replace(direct_edge, to_node=prepared_node.id)
    successor_edge = EdgeSpec(
        id=new_edge_id,
        from_node=prepared_node.id,
        to_node=successor.id,
        edge_type="on_success",
        label=None,
    )
    edges = (*state.edges[:direct_edge_index], predecessor_edge, successor_edge, *state.edges[direct_edge_index + 1 :])
    proposed = CompositionState(
        sources=state.sources,
        nodes=nodes,
        edges=edges,
        outputs=state.outputs,
        metadata=state.metadata,
        version=state.version,
        guided_session=state.guided_session,
    )
    try:
        reconciled = reconcile_authoritative_reviews(state, proposed)
    except (KeyError, TypeError, ValueError):
        return _failure_result(
            state,
            "Authoritative interpretation-review reconciliation failed. Re-inspect the pipeline and retry.",
            error_code="review_reconciliation_failed",
        )
    review_contract_error = composition_review_contract_error(reconciled)
    if review_contract_error is not None:
        return _failure_result(state, review_contract_error)
    profile_validation = context.catalog.validate_composition_state(reconciled)
    if not profile_validation.validation.is_valid:
        return _failure_result(
            state,
            "Spliced pipeline failed context-aware validation.",
            error_code="splice_validation_failed",
        )
    new_state = replace(reconciled, version=state.version + 1)
    return _mutation_result(
        new_state,
        (predecessor_id, prepared_node.id, successor.id),
        data={
            "already_applied": False,
            "predecessor_id": predecessor_id,
            "successor_id": successor.id,
            "inserted_node_id": prepared_node.id,
            "derived_connection": connection_name,
            "replaced_edge_id": direct_edge.id,
            "new_edge_id": new_edge_id,
        },
    )


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
        if from_node in new_state.sources:
            if edge_type != "on_success":
                return _failure_result(state, "Source sink edges must use 'on_success'.")
            source = new_state.sources[from_node]
            if source.on_success != to_node:
                new_state = new_state.with_named_source(from_node, replace(source, on_success=to_node))
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
                    if route_key not in routes or routes[route_key] != to_node:
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
    new_state = _clear_removed_sink_edge_route(new_state, edge)

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
    runtime_owned_error = _runtime_owned_llm_option_error(
        current.plugin,
        patch,
        tool_name="patch_node_options",
    )
    if runtime_owned_error is not None:
        return _failure_result(state, f"Node '{node_id}': {runtime_owned_error}")
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
        plugin_type="transform" if current.plugin is not None else None,
        plugin_name=current.plugin,
        options=new_options,
    )
    if credential_error is not None:
        return credential_error

    if current.node_type in ("transform", "aggregation") and current.plugin is not None:
        prevalidation_error = _prevalidate_transform_for_context(context, current.plugin, new_options)
        if prevalidation_error is not None:
            return _failure_result(state, prevalidation_error)

        # Operator-profiled nodes carry their private provider config (retry
        # budget / provider binding) in the profile, injected only at lowering;
        # the prevalidation above already validated the LOWERED executable. The
        # raw provider-config policy would false-positive on the absent private
        # retry budget (see set_pipeline in sessions.py for the full rationale).
        if "profile" not in new_options:
            provider_policy_error = _validate_transform_provider_config_policy(new_options, plugin=current.plugin)
            if provider_policy_error is not None:
                return _failure_result(state, f"Node '{node_id}': {provider_policy_error}")

        # S2: confine nested provider_config persist_directory (RAG retrieval).
        # A merge-patch can introduce an escaping path just as upsert_node can.
        provider_path_error = _validate_transform_provider_config_path(new_options, context.data_dir, session_id=context.session_id)
        if provider_path_error is not None:
            return _failure_result(state, f"Node '{node_id}': {provider_path_error}")

    new_node = replace(current, options=new_options)
    # Third canonical mutation boundary: a patch that would break a queue's
    # intrinsic contract (unknown option, non-string description) is rejected
    # by the single shared guard before with_node, leaving state atomically
    # unchanged. Returns None for every non-queue node, so this is a no-op for
    # transform/gate/aggregation/coalesce patches.
    queue_contract_error = queue_node_contract_error(new_node)
    if queue_contract_error is not None:
        return _failure_result(state, queue_contract_error)
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
    if node is None:
        raise AssertionError(
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
    description="Apply a shallow merge-patch to a node's options. Use this for option-only edits. "
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
        "additionalProperties": False,
    },
    augments_on_failure=True,
)


_SPLICE_CONNECTION_ATTEMPTS: Final[int] = 32
_SPLICE_CONNECTION_MAX_LENGTH: Final[int] = 64
_SPLICE_EDGE_ID_MAX_LENGTH: Final[int] = 160


def _prepare_transform_candidate(
    *,
    state: CompositionState,
    context: ToolContext,
    tool_name: str,
    node_id: str,
    node_type: NodeType,
    plugin: str | None,
    input_name: str,
    on_success: str | None,
    on_error: str | None,
    options: Mapping[str, Any],
    trigger: Mapping[str, Any] | None = None,
    output_mode: str | None = None,
    expected_output_count: int | None = None,
) -> NodeSpec | ToolResult:
    """Validate and prepare one transform candidate without mutating state."""
    if plugin is None:
        return _failure_result(state, f"Node '{node_id}': transform plugin is required.")
    runtime_owned_error = _runtime_owned_llm_option_error(plugin, options, tool_name=tool_name)
    if runtime_owned_error is not None:
        return _failure_result(state, f"Node '{node_id}': {runtime_owned_error}")
    credential_error = _credential_wiring_contract_failure(
        state,
        component_id=node_id,
        component_type="node",
        plugin_type="transform",
        plugin_name=plugin,
        options=options,
    )
    if credential_error is not None:
        return credential_error
    plugin_error = _validate_plugin_name(context, "transform", plugin)
    if plugin_error is not None:
        return _plugin_policy_failure(state, plugin_error)
    batch_placement_error = _batch_aware_placement_error(node_id, node_type, plugin, output_mode)
    if batch_placement_error is not None:
        return _failure_result(state, batch_placement_error)
    batch_required_error = _batch_aware_required_input_fields_error(node_id, plugin, options)
    if batch_required_error is not None:
        return _failure_result(state, batch_required_error)

    review_options = _options_with_default_llm_reviews(node_id=node_id, plugin=plugin, options=options)
    prevalidation_error = _prevalidate_transform_for_context(context, plugin, review_options)
    if prevalidation_error is not None:
        return _failure_result(state, prevalidation_error)
    # Operator-profiled nodes carry their private provider config (retry budget /
    # provider binding) in the profile, injected only at lowering; the
    # prevalidation above already validated the LOWERED executable. The raw
    # provider-config policy would false-positive on the absent private retry
    # budget (see set_pipeline in sessions.py for the full rationale).
    if "profile" not in options:
        provider_policy_error = _validate_transform_provider_config_policy(options, plugin=plugin)
        if provider_policy_error is not None:
            return _failure_result(state, f"Node '{node_id}': {provider_policy_error}")
    provider_path_error = _validate_transform_provider_config_path(options, context.data_dir, session_id=context.session_id)
    if provider_path_error is not None:
        return _failure_result(state, f"Node '{node_id}': {provider_path_error}")
    if node_type == "aggregation":
        trigger_error = _validate_aggregation_trigger(dict(trigger) if trigger is not None else None)
        if trigger_error is not None:
            return _failure_result(state, f"Node '{node_id}': {trigger_error}")

    return NodeSpec(
        id=node_id,
        node_type=node_type,
        plugin=plugin,
        input=input_name,
        on_success=on_success,
        on_error=on_error or "discard",
        options=review_options,
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
        trigger=trigger,
        output_mode=output_mode,
        expected_output_count=expected_output_count,
    )


def _handle_splice_transform(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    result = _execute_splice_transform(arguments, state, context)
    if not result.success:
        return result
    validated = SpliceTransformArgumentsModel.model_validate(arguments)
    return _attach_post_call_hints(
        result,
        context.catalog,
        plugin_type="transform",
        tool_name="splice_transform",
        plugin_name=validated.node.plugin,
        config_snapshot=validated.node.options,
    )


_SPLICE_TRANSFORM_DECLARATION = ToolDeclaration(
    name="splice_transform",
    handler=_handle_splice_transform,
    kind=ToolKind.MUTATION,
    description=(
        "Insert one transform between a predecessor and successor on an existing direct linear on_success path. "
        "Use this for insert/between/before/after edits; the server derives input, on_success, connection, and edge IDs."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "predecessor_id": {
                "type": "string",
                "description": "Existing source or transform immediately before the insertion point.",
            },
            "successor_id": {
                "type": "string",
                "description": "Existing transform immediately after the insertion point.",
            },
            "node": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique ID for the inserted transform."},
                    "plugin": {"type": "string", "description": "Transform plugin name."},
                    "options": {"type": "object", "description": "Plugin-specific authored options."},
                    "on_error": {
                        "type": ["string", "null"],
                        "description": "Optional error route; defaults to discard.",
                    },
                },
                "required": ["id", "plugin", "options"],
                "additionalProperties": False,
            },
        },
        "required": ["predecessor_id", "successor_id", "node"],
        "additionalProperties": False,
    },
    augments_on_failure=True,
)


TOOLS_IN_MODULE: tuple[ToolDeclaration, ...] = (
    _LIST_TRANSFORMS_DECLARATION,
    _LIST_SINKS_DECLARATION,
    _UPSERT_NODE_DECLARATION,
    _SPLICE_TRANSFORM_DECLARATION,
    _UPSERT_EDGE_DECLARATION,
    _REMOVE_NODE_DECLARATION,
    _REMOVE_EDGE_DECLARATION,
    _SET_METADATA_DECLARATION,
    _PATCH_NODE_OPTIONS_DECLARATION,
)
"""Every tool declared in this module, in stable order.

``_dispatch.py`` aggregates this tuple alongside every other plane's
TOOLS_IN_MODULE to build the registered-tool universe."""
