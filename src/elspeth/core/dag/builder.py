# src/elspeth/core/dag/builder.py
"""DAG construction from plugin instances.

Extracts the graph-building logic from ExecutionGraph.from_plugin_instances()
into a module-level function. The classmethod facade on ExecutionGraph delegates
here via lazy import to avoid circular dependencies.

Dependency: models.py (leaf) — no import of graph.py at module level.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import networkx as nx

from elspeth.contracts import RouteDestination, RoutingMode, error_edge_label
from elspeth.contracts.enums import NodeType
from elspeth.contracts.types import (
    AggregationName,
    BranchName,
    CoalesceName,
    GateName,
    NodeID,
    SinkName,
)
from elspeth.core.canonical import canonical_json
from elspeth.core.dag.models import (
    _NODE_ID_MAX_LENGTH,
    GraphValidationError,
    _GateEntry,
    _suggest_similar,
)

if TYPE_CHECKING:
    from elspeth.core.config import (
        AggregationSettings,
        CoalesceSettings,
        GateSettings,
        SourceSettings,
    )
    from elspeth.core.dag.graph import ExecutionGraph
    from elspeth.core.dag.models import NodeConfig, WiredTransform
    from elspeth.plugins.protocols import SinkProtocol, SourceProtocol, TransformProtocol


def _field_name_type(field_spec: Any) -> tuple[str, str]:
    """Extract (field_name, field_type) from a field spec in any format.

    Handles:
    - String: ``"name: str"`` or ``"name: str?"``
    - to_dict() dict: ``{"name": "x", "type": "str", "required": true}``
    - YAML dict: ``{"id": "int"}``
    """
    if isinstance(field_spec, str):
        name, _, type_part = field_spec.partition(":")
        return name.strip(), type_part.strip().rstrip("?")
    if isinstance(field_spec, dict):
        if "name" in field_spec and "type" in field_spec:
            return field_spec["name"], field_spec["type"]
        if len(field_spec) == 1:
            name, ftype = next(iter(field_spec.items()))
            return str(name), str(ftype).rstrip("?")
    msg = f"Cannot parse field spec: {field_spec!r}"
    raise ValueError(msg)


def _field_required(field_spec: Any) -> bool:
    """Extract required status from a field spec.

    - String ending with ``?``: optional (``False``)
    - Dict with ``"required"`` key: use that value
    - YAML dict ``{"id": "int?"}`` ending with ``?``: optional
    - Otherwise: required (``True``)
    """
    if isinstance(field_spec, str):
        return not field_spec.strip().endswith("?")
    if isinstance(field_spec, dict):
        if "required" in field_spec:
            return bool(field_spec["required"])
        if len(field_spec) == 1:
            ftype = str(next(iter(field_spec.values())))
            return not ftype.strip().endswith("?")
    return True


def build_execution_graph(
    cls: type[ExecutionGraph],
    source: SourceProtocol,
    source_settings: SourceSettings,
    transforms: list[WiredTransform],
    sinks: dict[str, SinkProtocol],
    aggregations: dict[str, tuple[TransformProtocol, AggregationSettings]],
    gates: list[GateSettings],
    coalesce_settings: list[CoalesceSettings] | None = None,
) -> ExecutionGraph:
    """Build an ExecutionGraph from plugin instances.

    Called by ExecutionGraph.from_plugin_instances() facade. See that method
    for full documentation of parameters and semantics.
    """
    graph = cls()

    def node_id(prefix: str, name: str, config: NodeConfig, sequence: int | None = None) -> NodeID:
        """Generate deterministic node ID based on plugin type and config.

        Node IDs must be deterministic for checkpoint/resume compatibility.
        If a pipeline is checkpointed and later resumed, the node IDs must
        be identical so checkpoint state can be restored correctly.

        For nodes that can appear multiple times with identical configs
        (transforms, aggregations), include sequence number to ensure uniqueness.

        Args:
            prefix: Node type prefix (source_, transform_, sink_, etc.)
            name: Plugin name
            config: Plugin configuration dict
            sequence: Optional sequence number for duplicate configs (transforms, aggregations)

        Returns:
            Deterministic node ID
        """
        # Create stable hash of config using RFC 8785 canonical JSON
        # CRITICAL: Must use canonical_json() not json.dumps() for true determinism
        # (floats, nested dicts, datetime serialization must be consistent)
        config_str = canonical_json(config)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]  # 48 bits

        # Include sequence number for nodes that can have duplicates
        if sequence is not None:
            generated = f"{prefix}_{name}_{config_hash}_{sequence}"
        else:
            generated = f"{prefix}_{name}_{config_hash}"

        if len(generated) > _NODE_ID_MAX_LENGTH:
            raise GraphValidationError(
                f"Generated node_id exceeds {_NODE_ID_MAX_LENGTH} characters: "
                f"'{generated}' (length={len(generated)}). "
                "Use shorter transform/gate/aggregation/source/sink names."
            )

        return NodeID(generated)

    def _best_schema_dict(nid: NodeID) -> dict[str, Any]:
        """Get best available schema dict from a node.

        Prefers computed output_schema_config (includes guaranteed_fields,
        audit_fields from e.g., LLM transforms) over raw config["schema"].
        Pass-through nodes (gates, coalesce) should inherit the computed
        schema so audit records reflect actual data contracts.
        """
        info = graph.get_node_info(nid)
        if info.output_schema_config is not None:
            return info.output_schema_config.to_dict()
        # config["schema"] is Any from NodeConfig (dict[str, Any] value access).
        # It's always a dict at runtime — ensured by DataPluginConfig validation.
        schema: dict[str, Any] = info.config["schema"]
        return schema

    def _sink_name_set() -> set[str]:
        return {str(name) for name in sink_ids}

    # Add source
    source_config = source.config
    source_id = node_id("source", source.name, source_config)
    graph.add_node(
        source_id,
        node_type=NodeType.SOURCE,
        plugin_name=source.name,
        config=source_config,
        output_schema=source.output_schema,  # SourceProtocol requires this
    )

    # Add sinks
    sink_ids: dict[SinkName, NodeID] = {}
    for sink_name, sink in sinks.items():
        sink_config = sink.config
        sid = node_id("sink", sink_name, sink_config)
        sink_ids[SinkName(sink_name)] = sid
        graph.add_node(
            sid,
            node_type=NodeType.SINK,
            plugin_name=sink.name,
            config=sink_config,
            input_schema=sink.input_schema,  # SinkProtocol requires this
        )

    graph.set_sink_id_map(sink_ids)

    # Build transforms
    transform_ids_by_name: dict[str, NodeID] = {}
    transform_ids_by_seq: dict[int, NodeID] = {}
    gate_entries: list[_GateEntry] = []
    gate_route_connections: list[tuple[NodeID, str, str]] = []

    for seq, wired in enumerate(transforms):
        transform = wired.plugin
        transform_config = transform.config
        tid = node_id("transform", wired.settings.name, transform_config)
        transform_ids_by_name[wired.settings.name] = tid
        transform_ids_by_seq[seq] = tid

        node_config = dict(transform_config)
        node_type = NodeType.TRANSFORM

        # Extract computed output schema config if available (e.g., LLM transforms
        # compute guaranteed_fields and audit_fields from their configuration).
        # getattr is appropriate here: this is a framework boundary where the DAG
        # builder queries an optional plugin capability (not all transforms compute
        # output schemas). See CLAUDE.md "Legitimate Uses: Framework boundaries."
        output_schema_config = getattr(transform, "_output_schema_config", None)

        graph.add_node(
            tid,
            node_type=node_type,
            plugin_name=transform.name,
            config=node_config,
            input_schema=transform.input_schema,  # TransformProtocol requires this
            output_schema=transform.output_schema,  # TransformProtocol requires this
            output_schema_config=output_schema_config,
        )

    graph.set_transform_id_map(transform_ids_by_seq)

    # Build aggregations
    aggregation_ids: dict[AggregationName, NodeID] = {}
    for agg_name, (transform, agg_config) in aggregations.items():
        transform_config = transform.config
        agg_node_config = {
            "trigger": agg_config.trigger.model_dump(),
            "output_mode": agg_config.output_mode,
            "options": dict(agg_config.options),
            "schema": transform_config["schema"],
        }
        aid = node_id("aggregation", agg_name, agg_node_config)
        aggregation_ids[AggregationName(agg_name)] = aid

        # Same framework-boundary getattr as transform case above.
        agg_output_schema_config = getattr(transform, "_output_schema_config", None)

        graph.add_node(
            aid,
            node_type=NodeType.AGGREGATION,
            plugin_name=agg_config.plugin,
            config=agg_node_config,
            input_schema=transform.input_schema,  # TransformProtocol requires this (aggregations use transforms)
            output_schema=transform.output_schema,  # TransformProtocol requires this (aggregations use transforms)
            output_schema_config=agg_output_schema_config,
        )

    graph.set_aggregation_id_map(aggregation_ids)

    # Build config gates (no plugin instances)
    config_gate_ids: dict[GateName, NodeID] = {}
    config_gate_schema_inputs: list[tuple[NodeID, str, str]] = []

    for gate_config in gates:
        gate_node_config = {
            "condition": gate_config.condition,
            "routes": dict(gate_config.routes),
        }
        if gate_config.fork_to:
            gate_node_config["fork_to"] = list(gate_config.fork_to)

        gid = node_id("config_gate", gate_config.name, gate_node_config)
        config_gate_ids[GateName(gate_config.name)] = gid

        graph.add_node(
            gid,
            node_type=NodeType.GATE,
            plugin_name=f"config_gate:{gate_config.name}",
            config=gate_node_config,
        )

        config_gate_schema_inputs.append((gid, gate_config.name, gate_config.input))

        # Gate routes to sinks; connection-name routes are deferred.
        for route_label, target in gate_config.routes.items():
            if target == "fork":
                # Fork is a special routing mode - handled by fork_to branches
                graph.add_route_resolution_entry(gid, route_label, RouteDestination.fork())
            elif SinkName(target) in sink_ids:
                target_sink_id = sink_ids[SinkName(target)]
                graph.add_edge(gid, target_sink_id, label=route_label, mode=RoutingMode.MOVE)
                graph.add_route_label_entry(gid, target, route_label)
                graph.add_route_resolution_entry(gid, route_label, RouteDestination.sink(SinkName(target)))
            else:
                gate_route_connections.append((gid, route_label, target))

        gate_entries.append(
            _GateEntry(
                node_id=gid,
                name=gate_config.name,
                fork_to=tuple(gate_config.fork_to) if gate_config.fork_to is not None else None,
                routes=MappingProxyType(dict(gate_config.routes)),
            )
        )

    graph.set_config_gate_id_map(config_gate_ids)

    # ===== COALESCE IMPLEMENTATION (BUILD NODES AND MAPPINGS FIRST) =====
    # Build coalesce nodes BEFORE connecting gates (needed for branch routing)
    coalesce_ids: dict[CoalesceName, NodeID] = {}
    if coalesce_settings:
        branch_to_coalesce: dict[BranchName, CoalesceName] = {}

        for coalesce_config in coalesce_settings:
            # Coalesce merges - no schema transformation
            # Note: Pydantic validates min_length=2 for branches field
            config_dict: NodeConfig = {
                "branches": dict(coalesce_config.branches),
                "policy": coalesce_config.policy,
                "merge": coalesce_config.merge,
            }
            if coalesce_config.timeout_seconds is not None:
                config_dict["timeout_seconds"] = coalesce_config.timeout_seconds
            if coalesce_config.quorum_count is not None:
                config_dict["quorum_count"] = coalesce_config.quorum_count
            if coalesce_config.select_branch is not None:
                config_dict["select_branch"] = coalesce_config.select_branch

            cid = node_id("coalesce", coalesce_config.name, config_dict)
            coalesce_ids[CoalesceName(coalesce_config.name)] = cid

            # Map branches to this coalesce - check for duplicates
            for branch_name in coalesce_config.branches:
                if BranchName(branch_name) in branch_to_coalesce:
                    # Branch already mapped to another coalesce
                    existing_coalesce = branch_to_coalesce[BranchName(branch_name)]
                    raise GraphValidationError(
                        f"Duplicate branch name '{branch_name}' found in coalesce settings.\n"
                        f"Branch '{branch_name}' is already mapped to coalesce '{existing_coalesce}', "
                        f"but coalesce '{coalesce_config.name}' also declares it.\n"
                        f"Each fork branch can only merge at one coalesce point."
                    )
                branch_to_coalesce[BranchName(branch_name)] = CoalesceName(coalesce_config.name)

            graph.add_node(
                cid,
                node_type=NodeType.COALESCE,
                plugin_name=f"coalesce:{coalesce_config.name}",
                config=config_dict,
            )

        graph.set_coalesce_id_map(coalesce_ids)
        graph.set_branch_to_coalesce(branch_to_coalesce)
    else:
        branch_to_coalesce = {}

    # ===== DETERMINE WHICH BRANCHES HAVE TRANSFORMS =====
    # A branch has transforms when its coalesce input_connection differs from
    # the branch name (identity mapping = no transforms).
    transformed_branches: set[str] = set()
    branch_input_connections: dict[str, str] = {}  # branch_name → input_connection
    if coalesce_settings:
        for coalesce_config in coalesce_settings:
            for branch_name, input_connection in coalesce_config.branches.items():
                branch_input_connections[branch_name] = input_connection
                if input_connection != branch_name:
                    transformed_branches.add(branch_name)

    # ===== CONNECT FORK GATES - EXPLICIT DESTINATIONS ONLY =====
    # CRITICAL: No fallback behavior. All fork branches must have explicit destinations.
    # This prevents silent configuration bugs (typos, missing destinations).
    fork_branch_owner: dict[str, str] = {}
    # Track which coalesce nodes need consumer registration for transform branches
    coalesce_transform_consumers: list[tuple[str, str, CoalesceName]] = []  # (branch, input_conn, coalesce)
    for gate_entry in gate_entries:
        if gate_entry.fork_to:
            branch_counts = Counter(gate_entry.fork_to)
            duplicates = sorted([branch for branch, count in branch_counts.items() if count > 1])
            if duplicates:
                raise GraphValidationError(
                    f"Gate '{gate_entry.name}' has duplicate fork branches: {duplicates}. Each fork branch name must be unique."
                )
            for branch_name in gate_entry.fork_to:
                if branch_name in fork_branch_owner:
                    raise GraphValidationError(
                        f"Fork branch '{branch_name}' is declared by multiple gates: "
                        f"'{fork_branch_owner[branch_name]}' and '{gate_entry.name}'. "
                        "Fork branch names must be globally unique across all gates."
                    )
                fork_branch_owner[branch_name] = gate_entry.name
                if BranchName(branch_name) in branch_to_coalesce:
                    coalesce_name = branch_to_coalesce[BranchName(branch_name)]
                    coalesce_nid = coalesce_ids[coalesce_name]
                    if branch_name in transformed_branches:
                        # Transform branch: branch name becomes a produced connection
                        # from the gate. The coalesce consumes the final transform's
                        # output (input_connection). Connection resolution wires the chain.
                        # Direct COPY edge NOT created — routing goes through transforms.
                        input_conn = branch_input_connections[branch_name]
                        coalesce_transform_consumers.append((branch_name, input_conn, coalesce_name))
                    else:
                        # Identity branch: direct COPY edge (current behavior)
                        graph.add_edge(gate_entry.node_id, coalesce_nid, label=branch_name, mode=RoutingMode.COPY)
                elif SinkName(branch_name) in sink_ids:
                    # Explicit sink destination (branch name matches sink name)
                    graph.add_edge(
                        gate_entry.node_id,
                        sink_ids[SinkName(branch_name)],
                        label=branch_name,
                        mode=RoutingMode.COPY,
                    )
                else:
                    # NO FALLBACK - this is a configuration error
                    raise GraphValidationError(
                        f"Gate '{gate_entry.name}' has fork branch '{branch_name}' with no destination.\n"
                        f"Fork branches must either:\n"
                        f"  1. Be listed in a coalesce 'branches' dict/list, or\n"
                        f"  2. Match a sink name exactly\n"
                        f"\n"
                        f"Available coalesce branches: {sorted(branch_to_coalesce.keys())}\n"
                        f"Available sinks: {sorted(sink_ids.keys())}"
                    )

    # ===== VALIDATE COALESCE BRANCHES ARE PRODUCED BY GATES =====
    # All branches declared in coalesce settings must be produced by some fork gate
    if coalesce_settings and branch_to_coalesce:
        # Collect all branches produced by gates
        produced_branches: set[str] = set()
        for gate_entry in gate_entries:
            if gate_entry.fork_to:
                produced_branches.update(gate_entry.fork_to)

        # Check that all coalesce branches are produced
        for branch_name, coalesce_name in branch_to_coalesce.items():
            if branch_name not in produced_branches:
                raise GraphValidationError(
                    f"Coalesce '{coalesce_name}' declares branch '{branch_name}', "
                    f"but no gate produces this branch.\n"
                    f"Branches must be listed in a gate's fork_to list to be valid.\n"
                    f"\n"
                    f"Branches produced by gates: {sorted(produced_branches) if produced_branches else '(none)'}\n"
                    f"Coalesce '{coalesce_name}' expects branches: "
                    f"{sorted([b for b, c in branch_to_coalesce.items() if c == coalesce_name])}"
                )

    # ===== BUILD PRODUCER REGISTRY =====
    producers: dict[str, tuple[NodeID, str]] = {}
    producer_desc: dict[str, str] = {}
    gate_connection_route_labels: dict[tuple[NodeID, str], list[str]] = {}

    def register_producer(connection_name: str, node_id: NodeID, label: str, description: str) -> None:
        if connection_name in producers:
            existing_node, _existing_label = producers[connection_name]
            raise GraphValidationError(
                f"Duplicate producer for connection '{connection_name}': "
                f"{producer_desc[connection_name]} ({existing_node}) and {description} ({node_id})."
            )
        producers[connection_name] = (node_id, label)
        producer_desc[connection_name] = description

    source_on_success = source_settings.on_success
    if SinkName(source_on_success) not in sink_ids:
        register_producer(
            source_on_success,
            source_id,
            "continue",
            f"source '{source.name}'",
        )

    for wired in transforms:
        tid = transform_ids_by_name[wired.settings.name]
        on_success = wired.settings.on_success
        if SinkName(on_success) not in sink_ids:
            register_producer(on_success, tid, "continue", f"transform '{wired.settings.name}'")

    for agg_name, (_transform, agg_settings) in aggregations.items():
        aid = aggregation_ids[AggregationName(agg_name)]
        if agg_settings.on_success is None:
            register_producer(agg_settings.name, aid, "continue", f"aggregation '{agg_settings.name}'")
        elif SinkName(agg_settings.on_success) not in sink_ids:
            register_producer(agg_settings.on_success, aid, "continue", f"aggregation '{agg_settings.name}'")

    if coalesce_settings:
        for coalesce_config in coalesce_settings:
            if coalesce_config.on_success is None:
                coalesce_id = coalesce_ids[CoalesceName(coalesce_config.name)]
                register_producer(
                    coalesce_config.name,
                    coalesce_id,
                    "continue",
                    f"coalesce '{coalesce_config.name}'",
                )

    for gate_id, route_label, target in gate_route_connections:
        gate_connection_key = (gate_id, target)
        gate_connection_route_labels.setdefault(gate_connection_key, []).append(route_label)

        # Multiple routes from the same gate may converge to the same target
        # (e.g., {"true": "next_gate", "false": "next_gate"}). Only register
        # the producer once — the connection is the same regardless of which
        # route label was taken.
        if target in producers and producers[target][0] == gate_id:
            continue
        register_producer(target, gate_id, route_label, f"gate route '{route_label}' from '{gate_id}'")

    # Register fork branches as produced connections (only for branches with transforms).
    # Identity branches use direct COPY edges and don't need connection registration.
    for branch_name in transformed_branches:
        gate_name = fork_branch_owner[branch_name]
        gate_nid = config_gate_ids[GateName(gate_name)]
        register_producer(
            branch_name,
            gate_nid,
            branch_name,
            f"fork branch '{branch_name}' from gate '{gate_name}'",
        )

    # ===== BUILD CONSUMER REGISTRY =====
    consumers: dict[str, NodeID] = {}
    consumer_claims: list[tuple[str, NodeID, str]] = []

    def register_consumer(connection_name: str, node_id: NodeID, description: str) -> None:
        consumer_claims.append((connection_name, node_id, description))
        if connection_name not in consumers:
            consumers[connection_name] = node_id

    for wired in transforms:
        register_consumer(
            wired.settings.input,
            transform_ids_by_name[wired.settings.name],
            f"transform '{wired.settings.name}'",
        )

    for agg_name, (_transform, agg_settings) in aggregations.items():
        register_consumer(
            agg_settings.input,
            aggregation_ids[AggregationName(agg_name)],
            f"aggregation '{agg_settings.name}'",
        )

    for gate_settings in gates:
        register_consumer(
            gate_settings.input,
            config_gate_ids[GateName(gate_settings.name)],
            f"gate '{gate_settings.name}'",
        )

    # Register coalesce nodes as consumers of transform branch input connections.
    # For transform branches, the coalesce consumes from the final transform's
    # output connection (not the branch name). The connection resolution system
    # will create MOVE edges through the transform chain automatically.
    for branch_name, input_conn, coal_name in coalesce_transform_consumers:
        register_consumer(
            input_conn,
            coalesce_ids[coal_name],
            f"coalesce '{coal_name}' branch '{branch_name}'",
        )

    # ===== VALIDATE CONNECTION NAMESPACES =====
    cls._validate_connection_namespaces(
        producers=producers,
        consumers=consumers,
        consumer_claims=consumer_claims,
        sink_names=_sink_name_set(),
        check_dangling=False,
    )

    # Config gate schema resolution (pass 1): resolve gates whose upstream
    # producer already has a schema. Gates downstream of coalesce nodes are
    # deferred to pass 2 (after coalesce schema population).
    deferred_config_gate_schemas: list[tuple[NodeID, str, str]] = []
    for gate_id, gate_name, input_connection in config_gate_schema_inputs:
        if input_connection not in producers:
            suggestions = _suggest_similar(input_connection, sorted(producers.keys()))
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise GraphValidationError(
                f"Gate '{gate_name}' input '{input_connection}' has no producer.{hint}\nAvailable connections: {', '.join(sorted(producers.keys()))}"
            )
        producer_id, _producer_label = producers[input_connection]
        upstream_info = graph.get_node_info(producer_id)
        if upstream_info.output_schema_config is not None or "schema" in upstream_info.config:
            graph.get_node_info(gate_id).config["schema"] = _best_schema_dict(producer_id)
        else:
            deferred_config_gate_schemas.append((gate_id, gate_name, input_connection))

    # ===== MATCH PRODUCERS TO CONSUMERS =====
    gate_node_ids = {entry.node_id for entry in gate_entries}

    gate_default_continue_targets: dict[NodeID, NodeID] = {}
    ambiguous_continue_gates: set[NodeID] = set()

    for connection_name, consumer_id in consumers.items():
        producer_id, producer_label = producers[connection_name]
        if producer_id in gate_node_ids and producer_label != "continue":
            route_labels = gate_connection_route_labels.get((producer_id, connection_name))
            if route_labels:
                for route_label in route_labels:
                    graph.add_edge(producer_id, consumer_id, label=route_label, mode=RoutingMode.MOVE)
            else:
                graph.add_edge(producer_id, consumer_id, label=producer_label, mode=RoutingMode.MOVE)
            # Preserve gate fallthrough semantics for RoutingAction.continue_():
            # when a gate has a single downstream processing target, continue
            # should route there even if explicit route labels are present.
            existing_target = gate_default_continue_targets.get(producer_id)
            if existing_target is None:
                gate_default_continue_targets[producer_id] = consumer_id
            elif existing_target != consumer_id:
                # Ambiguous continue fallthrough (multiple processing targets).
                # Leave unresolved; GateExecutor will fail closed if a gate
                # emits continue_() without a unique continuation edge.
                ambiguous_continue_gates.add(producer_id)
        else:
            graph.add_edge(producer_id, consumer_id, label="continue", mode=RoutingMode.MOVE)

    for gate_id, continue_target in gate_default_continue_targets.items():
        if gate_id in ambiguous_continue_gates:
            continue
        graph.add_edge(gate_id, continue_target, label="continue", mode=RoutingMode.MOVE)

    # ===== RESOLVE DEFERRED GATE ROUTES =====
    for gate_id, route_label, target in gate_route_connections:
        if target not in consumers:
            suggestions = _suggest_similar(target, sorted(consumers.keys()))
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise GraphValidationError(f"Gate route target '{target}' is neither a sink nor a known connection name.{hint}")
        graph.add_route_resolution_entry(gate_id, route_label, RouteDestination.processing_node(consumers[target]))

    # Ensure all declared gate route labels are resolvable before runtime.
    graph._validate_route_resolution_map_complete()

    # ===== TERMINAL ROUTING (on_success -> sinks) =====
    for wired in transforms:
        on_success = wired.settings.on_success
        tid = transform_ids_by_name[wired.settings.name]
        if SinkName(on_success) in sink_ids:
            graph.add_edge(tid, sink_ids[SinkName(on_success)], label="on_success", mode=RoutingMode.MOVE)
        elif on_success not in consumers:
            suggestions = _suggest_similar(on_success, sorted(consumers.keys()))
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise GraphValidationError(
                f"Transform '{wired.settings.name}' on_success '{on_success}' is neither a sink nor a known connection.{hint}"
            )

    for agg_name, (_transform, agg_settings) in aggregations.items():
        agg_on_success = agg_settings.on_success
        if agg_on_success is None:
            continue
        aid = aggregation_ids[AggregationName(agg_name)]
        if SinkName(agg_on_success) in sink_ids:
            graph.add_edge(aid, sink_ids[SinkName(agg_on_success)], label="on_success", mode=RoutingMode.MOVE)
        elif agg_on_success not in consumers:
            suggestions = _suggest_similar(agg_on_success, sorted(consumers.keys()))
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise GraphValidationError(
                f"Aggregation '{agg_settings.name}' on_success '{agg_on_success}' is neither a sink nor a known connection.{hint}"
            )

    if coalesce_settings:
        for coalesce_config in coalesce_settings:
            if coalesce_config.on_success is None:
                continue
            if coalesce_config.on_success in consumers:
                raise GraphValidationError(
                    f"Coalesce '{coalesce_config.name}' has on_success='{coalesce_config.on_success}'. "
                    "Coalesce on_success must point to a sink when configured."
                )
            on_success_sink = SinkName(coalesce_config.on_success)
            if on_success_sink not in sink_ids:
                raise GraphValidationError(
                    f"Coalesce '{coalesce_config.name}' on_success references unknown sink "
                    f"'{coalesce_config.on_success}'. Available sinks: {sorted(sink_ids.keys())}"
                )
            graph.add_edge(
                coalesce_ids[CoalesceName(coalesce_config.name)],
                sink_ids[on_success_sink],
                label="on_success",
                mode=RoutingMode.MOVE,
            )

    if SinkName(source_on_success) in sink_ids:
        # For source-only pipelines, create direct source -> sink edge.
        if not transforms and not gates and not aggregations:
            graph.add_edge(
                source_id,
                sink_ids[SinkName(source_on_success)],
                label="on_success",
                mode=RoutingMode.MOVE,
            )
    elif source_on_success not in consumers:
        suggestions = _suggest_similar(source_on_success, sorted(str(s) for s in sink_ids))
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise GraphValidationError(
            f"Source '{source.name}' on_success '{source_on_success}' is neither a sink nor a known connection.{hint}"
        )

    # Re-run namespace validation with dangling-output checks enabled now
    # that terminal on_success sink/connection validation has completed.
    cls._validate_connection_namespaces(
        producers=producers,
        consumers=consumers,
        consumer_claims=consumer_claims,
        sink_names=_sink_name_set(),
        check_dangling=True,
    )

    # ===== ADD DIVERT EDGES (quarantine/error sinks) =====
    # Divert edges represent error/quarantine data flows that bypass the
    # normal DAG execution path. They make quarantine/error sinks reachable
    # in the graph (required for node_ids and audit trail).
    #
    # These are STRUCTURAL markers, not execution paths. Rows reach these
    # sinks via exception handling (processor.py) or source validation
    # failures (orchestrator.py), not by traversing the edge during
    # normal processing.

    # Source quarantine edge
    # _on_validation_failure is defined on SourceProtocol (protocols.py:78)
    quarantine_dest = source._on_validation_failure
    if quarantine_dest != "discard" and SinkName(quarantine_dest) in sink_ids:
        graph.add_edge(
            source_id,
            sink_ids[SinkName(quarantine_dest)],
            label="__quarantine__",
            mode=RoutingMode.DIVERT,
        )

    # Transform error edges
    for wired in transforms:
        on_error = wired.settings.on_error
        if on_error != "discard":
            if SinkName(on_error) not in sink_ids:
                suggestions = _suggest_similar(on_error, sorted(str(s) for s in sink_ids))
                hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise GraphValidationError(
                    f"Transform '{wired.settings.name}' on_error '{on_error}' references unknown sink.{hint} "
                    f"Available sinks: {', '.join(sorted(str(s) for s in sink_ids))}"
                )
            graph.add_edge(
                transform_ids_by_name[wired.settings.name],
                sink_ids[SinkName(on_error)],
                label=error_edge_label(wired.settings.name),
                mode=RoutingMode.DIVERT,
            )

    # ===== PIPELINE ORDERING (TOPOLOGICAL) =====
    processing_node_ids: set[NodeID] = set()
    processing_node_ids.update(transform_ids_by_name.values())
    processing_node_ids.update(aggregation_ids.values())
    processing_node_ids.update(config_gate_ids.values())
    processing_node_ids.update(coalesce_ids.values())

    try:
        topo_order = [NodeID(raw_id) for raw_id in nx.topological_sort(graph._graph)]
    except nx.NetworkXUnfeasible:
        try:
            cycle = nx.find_cycle(graph._graph)
            cycle_str = " -> ".join(f"{edge[0]}" for edge in cycle)
            raise GraphValidationError(f"Pipeline contains a cycle: {cycle_str}")
        except nx.NetworkXNoCycle:
            raise GraphValidationError("Pipeline contains a cycle") from None
    pipeline_nodes = [node_id for node_id in topo_order if node_id in processing_node_ids]

    branch_gate_map: dict[BranchName, NodeID] = {}
    if coalesce_settings:
        for gate_entry in gate_entries:
            if gate_entry.fork_to is None:
                continue
            for branch_name in gate_entry.fork_to:
                branch_key = BranchName(branch_name)
                if branch_key in branch_to_coalesce:
                    branch_gate_map[branch_key] = gate_entry.node_id
    graph.set_branch_gate_map(branch_gate_map)

    # ===== POPULATE COALESCE SCHEMA CONFIG =====
    # Coalesce nodes are structural pass-throughs; record the upstream schema
    # so audit logs reflect the actual data contract at the merge point.
    # Schema validation is strategy-aware:
    #   union:  require compatible types on overlapping fields
    #   nested: no cross-branch constraint (each branch keyed separately)
    #   select: no cross-branch constraint (only selected branch matters)
    coalesce_id_to_config: dict[NodeID, CoalesceSettings] = {}
    if coalesce_settings:
        for coalesce_config in coalesce_settings:
            cid = coalesce_ids[CoalesceName(coalesce_config.name)]
            coalesce_id_to_config[cid] = coalesce_config

    for coalesce_id in coalesce_ids.values():
        incoming_edges_with_data = list(graph._graph.in_edges(coalesce_id, data=True, keys=True))
        if not incoming_edges_with_data:
            raise GraphValidationError(f"Coalesce node '{coalesce_id}' has no incoming branches; cannot determine schema for audit.")

        coal_config = coalesce_id_to_config[coalesce_id]

        # Build a branch_name → schema mapping using edge labels.
        # Identity branches have COPY edges labelled with branch_name.
        # Transform branches have MOVE edges from the last transform — we
        # correlate via the coalesce config's branch_input → branch_name mapping.
        branch_to_schema: dict[str, dict[str, Any]] = {}

        for from_id, _to_id, _key, data in incoming_edges_with_data:
            edge_label = data["label"]
            edge_mode = data["mode"]
            schema = _best_schema_dict(NodeID(from_id))

            if edge_mode == RoutingMode.COPY and edge_label in coal_config.branches:
                # Identity branch: COPY edge labelled with branch name
                branch_to_schema[edge_label] = schema
            elif edge_mode == RoutingMode.MOVE:
                # Transform branch: MOVE edge from last transform in chain.
                # The producer connection name was registered as the branch's
                # input_connection — look up the corresponding branch name.
                # For "continue" edges from connection resolution, we need to
                # match via the source node.  Check each branch's input
                # connection to find which branch this edge serves.
                for branch_name, input_conn in coal_config.branches.items():
                    if input_conn != branch_name and input_conn in producers and producers[input_conn][0] == NodeID(from_id):
                        branch_to_schema[branch_name] = schema
                        break

        if coal_config.merge == "union":
            # Union merge: require compatible types on ALL pairwise overlapping fields.
            # Parse each branch's SchemaConfig dict to extract field definitions.
            # Tracks (type, required, first_branch) to preserve optionality markers.
            seen_types: dict[str, tuple[str, bool, str]] = {}  # field → (type, required, first_branch)
            all_observed = False
            for branch_name, schema_dict in branch_to_schema.items():
                if schema_dict.get("mode") == "observed":
                    all_observed = True
                    break
                fields_list = schema_dict.get("fields")
                if not fields_list:
                    continue
                for field_spec in fields_list:
                    fname, ftype = _field_name_type(field_spec)
                    freq = _field_required(field_spec)
                    if fname in seen_types:
                        prior_type, _prior_req, prior_branch = seen_types[fname]
                        if prior_type != ftype:
                            raise GraphValidationError(
                                f"Coalesce node '{coalesce_id}' receives incompatible "
                                f"types for field '{fname}' in union merge: "
                                f"branch '{prior_branch}' has {prior_type!r}, "
                                f"branch '{branch_name}' has {ftype!r}. "
                                "Union merge requires compatible types on shared fields."
                            )
                        # If optional in ANY branch, optional in the merged output.
                        if not freq:
                            seen_types[fname] = (prior_type, False, prior_branch)
                    else:
                        seen_types[fname] = (ftype, freq, branch_name)
            # Build merged schema preserving contract fields.
            if all_observed or not seen_types:
                merged: dict[str, Any] = {"mode": "observed"}
            else:
                merged = {
                    "mode": "flexible",
                    "fields": [f"{name}: {ftype}{'?' if not req else ''}" for name, (ftype, req, _) in seen_types.items()],
                }
            # Propagate contract fields from branches:
            #   guaranteed_fields = intersection (guaranteed by ALL branches)
            #   audit_fields = union (any audit field from any branch)
            guaranteed_sets: list[set[str]] = []
            audit_sets: list[set[str]] = []
            for schema_dict in branch_to_schema.values():
                gf = schema_dict.get("guaranteed_fields")
                if gf is not None:
                    guaranteed_sets.append(set(gf))
                af = schema_dict.get("audit_fields")
                if af is not None:
                    audit_sets.append(set(af))
            if guaranteed_sets:
                merged["guaranteed_fields"] = sorted(set.intersection(*guaranteed_sets))
            if audit_sets:
                merged["audit_fields"] = sorted(set.union(*audit_sets))
            graph.get_node_info(coalesce_id).config["schema"] = merged
        elif coal_config.merge == "select":
            # Select merge: use selected branch's schema directly.
            # _best_schema_dict() returns a SchemaConfig-compatible dict.
            select_branch = coal_config.select_branch
            assert select_branch is not None  # Guaranteed by validate_merge_requirements
            if select_branch not in branch_to_schema:
                raise GraphValidationError(
                    f"Coalesce node '{coalesce_id}' select_branch '{select_branch}' "
                    f"has no schema mapping. Available branches: "
                    f"{sorted(branch_to_schema.keys())}. "
                    "This indicates a graph construction bug."
                )
            graph.get_node_info(coalesce_id).config["schema"] = branch_to_schema[select_branch]
        else:
            # Nested merge: output has branch names as top-level fields, each
            # containing the branch's row data as a nested dict.  Since the type
            # system only supports flat types, declare branch fields as "any".
            graph.get_node_info(coalesce_id).config["schema"] = {
                "mode": "flexible",
                "fields": [f"{branch}: any" for branch in branch_to_schema],
            }

    # Config gate schema resolution (pass 2): resolve gates that were deferred
    # because their upstream producer (e.g., coalesce) didn't have schema yet.
    for gate_id, _gate_name, input_connection in deferred_config_gate_schemas:
        producer_id, _producer_label = producers[input_connection]
        graph.get_node_info(gate_id).config["schema"] = _best_schema_dict(producer_id)

    # PHASE 2 VALIDATION: Validate schema compatibility AFTER graph is built
    graph.validate_edge_compatibility()

    # Warn about DIVERT edges feeding require_all coalesces (non-fatal).
    if coalesce_id_to_config:
        graph.warn_divert_coalesce_interactions(coalesce_id_to_config)

    # Freeze all NodeInfo configs now that schema resolution is complete.
    # NodeInfo is frozen=True so we use object.__setattr__ to replace the
    # mutable dict with an immutable MappingProxyType.  This prevents
    # accidental mutation of node configs after graph construction.
    for _, attrs in graph._graph.nodes(data=True):
        info = attrs["info"]
        if isinstance(info.config, dict):
            object.__setattr__(info, "config", MappingProxyType(info.config))

    # Step maps and node sequence support node_id-based processor traversal.
    graph.set_pipeline_nodes(pipeline_nodes)
    graph.set_node_step_map(graph.build_step_map())

    return graph
