# src/elspeth/core/dag.py
"""DAG (Directed Acyclic Graph) operations for execution planning.

Uses NetworkX for graph operations including:
- Acyclicity validation
- Topological sorting
- Path finding for lineage queries
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
from networkx import MultiDiGraph

from elspeth.contracts import EdgeInfo, RoutingMode, check_compatibility
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.types import (
    AggregationName,
    BranchName,
    CoalesceName,
    GateName,
    NodeID,
    SinkName,
)

if TYPE_CHECKING:
    from elspeth.contracts import PluginSchema
    from elspeth.core.config import AggregationSettings, CoalesceSettings, GateSettings
    from elspeth.plugins.protocols import SinkProtocol, SourceProtocol, TransformProtocol


class GraphValidationError(Exception):
    """Raised when graph validation fails."""

    pass


@dataclass
class NodeInfo:
    """Information about a node in the execution graph.

    Schemas are immutable after graph construction. Even dynamic schemas
    (determined by data inspection) are locked at launch and never change
    during the run. This guarantees audit trail consistency.

    Schema Contracts:
        input_schema_config and output_schema_config store the original
        SchemaConfig from plugin configuration. These contain contract
        declarations (guaranteed_fields, required_fields) used for DAG
        validation. The input_schema and output_schema are the generated
        Pydantic model types used for runtime validation.
    """

    node_id: NodeID
    node_type: str  # source, transform, gate, aggregation, coalesce, sink
    plugin_name: str
    config: dict[str, Any] = field(default_factory=dict)
    input_schema: type[PluginSchema] | None = None  # Immutable after graph construction
    output_schema: type[PluginSchema] | None = None  # Immutable after graph construction
    # Schema configs for contract validation (guaranteed/required fields)
    input_schema_config: SchemaConfig | None = None
    output_schema_config: SchemaConfig | None = None


@dataclass(frozen=True)
class _GateEntry:
    """Internal gate metadata for coalesce and routing wiring."""

    node_id: NodeID
    name: str
    fork_to: list[str] | None
    routes: dict[str, str]


class ExecutionGraph:
    """Execution graph for pipeline configuration.

    Wraps NetworkX MultiDiGraph with domain-specific operations.
    Uses MultiDiGraph to support multiple edges between the same node pair
    (e.g., fork gates routing multiple labels to the same sink).
    """

    def __init__(self) -> None:
        self._graph: MultiDiGraph[str] = nx.MultiDiGraph()
        self._sink_id_map: dict[SinkName, NodeID] = {}
        self._transform_id_map: dict[int, NodeID] = {}
        self._config_gate_id_map: dict[GateName, NodeID] = {}  # gate_name -> node_id
        self._aggregation_id_map: dict[AggregationName, NodeID] = {}  # agg_name -> node_id
        self._coalesce_id_map: dict[CoalesceName, NodeID] = {}  # coalesce_name -> node_id
        self._branch_to_coalesce: dict[BranchName, CoalesceName] = {}  # branch_name -> coalesce_name
        self._default_sink: str = ""
        self._route_label_map: dict[tuple[NodeID, str], str] = {}  # (gate_node, sink_name) -> route_label
        self._route_resolution_map: dict[tuple[NodeID, str], str] = {}  # (gate_node, label) -> sink_name | "continue"
        self._coalesce_gate_index: dict[CoalesceName, int] = {}  # coalesce_name -> gate pipeline index

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return self._graph.number_of_edges()

    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return self._graph.has_node(node_id)

    def get_nx_graph(self) -> MultiDiGraph[str]:
        """Return the underlying NetworkX graph for advanced operations.

        Use this for topology analysis, subgraph operations, and other
        NetworkX algorithms that require direct graph access.

        Returns:
            The underlying NetworkX MultiDiGraph.

        Warning:
            Direct graph manipulation should be avoided. Use ExecutionGraph
            methods (add_node, add_edge) to ensure integrity constraints.
        """
        return self._graph

    def add_node(
        self,
        node_id: str,
        *,
        node_type: str,
        plugin_name: str,
        config: dict[str, Any] | None = None,
        input_schema: type[PluginSchema] | None = None,
        output_schema: type[PluginSchema] | None = None,
        input_schema_config: SchemaConfig | None = None,
        output_schema_config: SchemaConfig | None = None,
    ) -> None:
        """Add a node to the execution graph.

        Args:
            node_id: Unique node identifier
            node_type: One of: source, transform, gate, aggregation, coalesce, sink
            plugin_name: Plugin identifier
            config: Node configuration
            input_schema: Input schema Pydantic type (None for dynamic or N/A like sources)
            output_schema: Output schema Pydantic type (None for dynamic or N/A like sinks)
            input_schema_config: Input schema config for contract validation
            output_schema_config: Output schema config for contract validation
        """
        info = NodeInfo(
            node_id=NodeID(node_id),
            node_type=node_type,
            plugin_name=plugin_name,
            config=config or {},
            input_schema=input_schema,
            output_schema=output_schema,
            input_schema_config=input_schema_config,
            output_schema_config=output_schema_config,
        )
        self._graph.add_node(node_id, info=info)

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        *,
        label: str,
        mode: RoutingMode = RoutingMode.MOVE,
    ) -> None:
        """Add an edge between nodes.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            label: Edge label (e.g., "continue", "suspicious") - also used as edge key
            mode: Routing mode (MOVE or COPY)
        """
        # Use label as key to allow multiple edges between same nodes
        self._graph.add_edge(from_node, to_node, key=label, label=label, mode=mode)

    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic (a valid DAG)."""
        return nx.is_directed_acyclic_graph(self._graph)

    def validate(self) -> None:
        """Validate the execution graph structure.

        Validates:
        1. Graph is acyclic (no cycles)
        2. Exactly one source node exists
        3. At least one sink node exists
        4. Edge labels are unique per source node

        Does NOT check schema compatibility - plugins validate their own
        schemas during construction.

        Raises:
            GraphValidationError: If validation fails
        """
        # Check for cycles
        if not self.is_acyclic():
            try:
                cycle = nx.find_cycle(self._graph)
                # MultiDiGraph returns (u, v, key) tuples; extract just u for display
                cycle_str = " -> ".join(f"{edge[0]}" for edge in cycle)
                raise GraphValidationError(f"Graph contains a cycle: {cycle_str}")
            except nx.NetworkXNoCycle:
                raise GraphValidationError("Graph contains a cycle") from None

        # Check for exactly one source
        # All nodes have "info" - added via add_node(), direct access is safe
        sources = [node_id for node_id, data in self._graph.nodes(data=True) if data["info"].node_type == "source"]
        if len(sources) != 1:
            raise GraphValidationError(f"Graph must have exactly one source, found {len(sources)}")

        # Check for at least one sink
        sinks = self.get_sinks()
        if len(sinks) < 1:
            raise GraphValidationError("Graph must have at least one sink")

        # Check outgoing edge labels are unique per node.
        # The orchestrator's edge_map keys by (from_node, label), so duplicate
        # labels from the same node would cause silent overwrites, leading to
        # routing events recorded against the wrong edge (audit corruption).
        for node_id in self._graph.nodes():
            labels_seen: set[str] = set()
            # out_edges returns (from, to, key) for MultiDiGraph
            for _, _, edge_key in self._graph.out_edges(node_id, keys=True):
                if edge_key in labels_seen:
                    raise GraphValidationError(
                        f"Node '{node_id}' has duplicate outgoing edge label '{edge_key}'. "
                        "Edge labels must be unique per source node to ensure correct "
                        "routing event recording."
                    )
                labels_seen.add(edge_key)

    def topological_order(self) -> list[str]:
        """Return nodes in topological order.

        Returns:
            List of node IDs in execution order

        Raises:
            GraphValidationError: If graph has cycles
        """
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible as e:
            raise GraphValidationError(f"Cannot sort graph: {e}") from e

    def get_source(self) -> NodeID | None:
        """Get the source node ID.

        Returns:
            The source node ID, or None if not exactly one source exists.
        """
        # All nodes have "info" - added via add_node(), direct access is safe
        sources = [NodeID(node_id) for node_id, data in self._graph.nodes(data=True) if data["info"].node_type == "source"]
        return sources[0] if len(sources) == 1 else None

    def get_sinks(self) -> list[NodeID]:
        """Get all sink node IDs.

        Returns:
            List of sink node IDs.
        """
        # All nodes have "info" - added via add_node(), direct access is safe
        return [NodeID(node_id) for node_id, data in self._graph.nodes(data=True) if data["info"].node_type == "sink"]

    def get_node_info(self, node_id: str) -> NodeInfo:
        """Get NodeInfo for a node.

        Args:
            node_id: The node ID

        Returns:
            NodeInfo for the node

        Raises:
            KeyError: If node doesn't exist
        """
        if not self._graph.has_node(node_id):
            raise KeyError(f"Node not found: {node_id}")
        return cast(NodeInfo, self._graph.nodes[node_id]["info"])

    def get_nodes(self) -> list[NodeInfo]:
        """Get all nodes as NodeInfo objects.

        Returns:
            List of NodeInfo objects for all nodes in the graph.
        """
        return [cast(NodeInfo, attrs["info"]) for _node_id, attrs in self._graph.nodes(data=True)]

    def get_edges(self) -> list[EdgeInfo]:
        """Get all edges with their data as typed EdgeInfo.

        Returns:
            List of EdgeInfo contracts (not tuples)
        """
        # Note: _key is unused but required for MultiDiGraph iteration signature
        return [
            EdgeInfo(
                from_node=u,
                to_node=v,
                label=data["label"],
                mode=data["mode"],  # Already RoutingMode after add_edge change
            )
            for u, v, _key, data in self._graph.edges(data=True, keys=True)
        ]

    def get_incoming_edges(self, node_id: str) -> list[EdgeInfo]:
        """Get all edges pointing TO this node.

        Args:
            node_id: The target node ID

        Returns:
            List of EdgeInfo for edges where to_node == node_id
        """
        # NetworkX in_edges returns (from, to, key) tuples for MultiDiGraph
        return [
            EdgeInfo(
                from_node=u,
                to_node=v,
                label=data["label"],
                mode=data["mode"],
            )
            for u, v, _key, data in self._graph.in_edges(node_id, data=True, keys=True)
        ]

    @classmethod
    def from_plugin_instances(
        cls,
        source: SourceProtocol,
        transforms: list[TransformProtocol],
        sinks: dict[str, SinkProtocol],
        aggregations: dict[str, tuple[TransformProtocol, AggregationSettings]],
        gates: list[GateSettings],
        default_sink: str,
        coalesce_settings: list[CoalesceSettings] | None = None,
    ) -> ExecutionGraph:
        """Build ExecutionGraph from plugin instances.

        CORRECT method for graph construction - enables schema validation.
        Schemas extracted directly from instance attributes.

        Args:
            source: Instantiated source plugin
            transforms: Instantiated transforms (row_plugins only, NOT aggregations)
            sinks: Dict of sink_name -> instantiated sink
            aggregations: Dict of agg_name -> (transform_instance, AggregationSettings)
            gates: Config-driven gate settings
            default_sink: Default output sink name
            coalesce_settings: Coalesce configs for fork/join patterns

        Returns:
            ExecutionGraph with schemas populated

        Raises:
            GraphValidationError: If gate routes reference unknown sinks
        """
        import hashlib

        from elspeth.core.canonical import canonical_json
        from elspeth.plugins.protocols import GateProtocol

        graph = cls()

        def node_id(prefix: str, name: str, config: dict[str, Any], sequence: int | None = None) -> NodeID:
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
                return NodeID(f"{prefix}_{name}_{config_hash}_{sequence}")
            return NodeID(f"{prefix}_{name}_{config_hash}")

        # Add source - extract schema from instance
        source_config = source.config  # type: ignore[attr-defined]
        source_id = node_id("source", source.name, source_config)
        graph.add_node(
            source_id,
            node_type="source",
            plugin_name=source.name,
            config=source_config,
            output_schema=source.output_schema,  # SourceProtocol requires this
        )

        # Add sinks
        sink_ids: dict[SinkName, NodeID] = {}
        for sink_name, sink in sinks.items():
            sink_config = sink.config  # type: ignore[attr-defined]
            sid = node_id("sink", sink_name, sink_config)
            sink_ids[SinkName(sink_name)] = sid
            graph.add_node(
                sid,
                node_type="sink",
                plugin_name=sink.name,
                config=sink_config,
                input_schema=sink.input_schema,  # SinkProtocol requires this
            )

        graph._sink_id_map = dict(sink_ids)
        graph._default_sink = default_sink

        # Build transform chain (includes plugin gates)
        transform_ids: dict[int, NodeID] = {}
        gate_entries: list[_GateEntry] = []
        pipeline_nodes: list[NodeID] = []
        prev_node_id = source_id

        for i, transform in enumerate(transforms):
            transform_config = transform.config  # type: ignore[attr-defined]
            is_gate = isinstance(transform, GateProtocol)
            # Include sequence to prevent ID collisions when configs are identical
            tid = node_id("transform", transform.name, transform_config, sequence=i)
            transform_ids[i] = tid

            node_config = dict(transform_config)
            node_type = "gate" if is_gate else "transform"

            if is_gate:
                # Type narrowing: we know it's a GateProtocol from isinstance check
                gate = cast(GateProtocol, transform)
                upstream_schema = graph.get_node_info(prev_node_id).config["schema"]
                if "schema" in node_config and node_config["schema"] != upstream_schema:
                    raise GraphValidationError(
                        f"Gate '{gate.name}' declares schema config that differs from upstream. "
                        f"Upstream schema config: {upstream_schema}, gate schema config: {node_config['schema']}"
                    )
                node_config["schema"] = upstream_schema
                node_config["routes"] = dict(gate.routes)
                if gate.fork_to is not None:
                    node_config["fork_to"] = list(gate.fork_to)

            # Extract computed output schema config if available (e.g., LLM transforms
            # compute guaranteed_fields and audit_fields from their configuration)
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

            graph.add_edge(prev_node_id, tid, label="continue", mode=RoutingMode.MOVE)
            prev_node_id = tid
            pipeline_nodes.append(tid)

            if is_gate:
                # Type narrowing: we know it's a GateProtocol from isinstance check
                gate = cast(GateProtocol, transform)
                gate_entries.append(
                    _GateEntry(
                        node_id=tid,
                        name=gate.name,
                        fork_to=list(gate.fork_to) if gate.fork_to is not None else None,
                        routes=dict(gate.routes),
                    )
                )

                # Gate routes to sinks via route labels
                for route_label, target in gate.routes.items():
                    if target == "continue":
                        graph._route_resolution_map[(tid, route_label)] = "continue"
                    elif target == "fork":
                        raise GraphValidationError(
                            f"Gate '{transform.name}' route '{route_label}' resolves to 'fork'. "
                            "Plugin gates must use RoutingAction.fork_to_paths() for forks."
                        )
                    else:
                        if SinkName(target) not in sink_ids:
                            raise GraphValidationError(f"Gate '{transform.name}' route '{route_label}' references unknown sink '{target}'")
                        target_sink_id = sink_ids[SinkName(target)]
                        graph.add_edge(tid, target_sink_id, label=route_label, mode=RoutingMode.MOVE)
                        graph._route_label_map[(tid, target)] = route_label
                        graph._route_resolution_map[(tid, route_label)] = target

        graph._transform_id_map = transform_ids

        # Build aggregations - dual schemas
        aggregation_ids: dict[AggregationName, NodeID] = {}
        for agg_name, (transform, agg_config) in aggregations.items():
            transform_config = transform.config  # type: ignore[attr-defined]
            agg_node_config = {
                "trigger": agg_config.trigger.model_dump(),
                "output_mode": agg_config.output_mode,
                "options": dict(agg_config.options),
                "schema": transform_config["schema"],
            }
            aid = node_id("aggregation", agg_name, agg_node_config)
            aggregation_ids[AggregationName(agg_name)] = aid

            # Extract computed output schema config if available (e.g., LLM aggregations
            # compute guaranteed_fields and audit_fields from their configuration)
            agg_output_schema_config = getattr(transform, "_output_schema_config", None)

            graph.add_node(
                aid,
                node_type="aggregation",
                plugin_name=agg_config.plugin,
                config=agg_node_config,
                input_schema=transform.input_schema,  # TransformProtocol requires this (aggregations use transforms)
                output_schema=transform.output_schema,  # TransformProtocol requires this (aggregations use transforms)
                output_schema_config=agg_output_schema_config,
            )

            graph.add_edge(prev_node_id, aid, label="continue", mode=RoutingMode.MOVE)
            prev_node_id = aid
            pipeline_nodes.append(aid)

        graph._aggregation_id_map = aggregation_ids

        # Build gates (config-driven, no instances)
        config_gate_ids: dict[GateName, NodeID] = {}

        for gate_config in gates:
            gate_node_config = {
                "condition": gate_config.condition,
                "routes": dict(gate_config.routes),
                "schema": graph.get_node_info(prev_node_id).config["schema"],
            }
            if gate_config.fork_to:
                gate_node_config["fork_to"] = list(gate_config.fork_to)

            gid = node_id("config_gate", gate_config.name, gate_node_config)
            config_gate_ids[GateName(gate_config.name)] = gid

            graph.add_node(
                gid,
                node_type="gate",
                plugin_name=f"config_gate:{gate_config.name}",
                config=gate_node_config,
            )

            graph.add_edge(prev_node_id, gid, label="continue", mode=RoutingMode.MOVE)
            prev_node_id = gid  # Advance chain to this gate
            pipeline_nodes.append(gid)

            # Gate routes to sinks
            for route_label, target in gate_config.routes.items():
                if target == "continue":
                    graph._route_resolution_map[(gid, route_label)] = "continue"
                elif target == "fork":
                    # Fork is a special routing mode - handled by fork_to branches
                    graph._route_resolution_map[(gid, route_label)] = "fork"
                else:
                    if SinkName(target) not in sink_ids:
                        raise GraphValidationError(f"Gate '{gate_config.name}' route '{route_label}' references unknown sink '{target}'")
                    target_sink_id = sink_ids[SinkName(target)]
                    graph.add_edge(gid, target_sink_id, label=route_label, mode=RoutingMode.MOVE)
                    graph._route_label_map[(gid, target)] = route_label
                    graph._route_resolution_map[(gid, route_label)] = target

            gate_entries.append(
                _GateEntry(
                    node_id=gid,
                    name=gate_config.name,
                    fork_to=list(gate_config.fork_to) if gate_config.fork_to is not None else None,
                    routes=dict(gate_config.routes),
                )
            )

        graph._config_gate_id_map = config_gate_ids

        # ===== COALESCE IMPLEMENTATION (BUILD NODES AND MAPPINGS FIRST) =====
        # Build coalesce nodes BEFORE connecting gates (needed for branch routing)
        coalesce_ids: dict[CoalesceName, NodeID] = {}
        if coalesce_settings:
            branch_to_coalesce: dict[BranchName, CoalesceName] = {}

            for coalesce_config in coalesce_settings:
                # Coalesce merges - no schema transformation
                # Note: Pydantic validates min_length=2 for branches field
                config_dict: dict[str, Any] = {
                    "branches": list(coalesce_config.branches),
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
                    node_type="coalesce",
                    plugin_name=f"coalesce:{coalesce_config.name}",
                    config=config_dict,
                )

            graph._coalesce_id_map = coalesce_ids
            graph._branch_to_coalesce = branch_to_coalesce
        else:
            branch_to_coalesce = {}

        # ===== CONNECT FORK GATES - EXPLICIT DESTINATIONS ONLY =====
        # CRITICAL: No fallback behavior. All fork branches must have explicit destinations.
        # This prevents silent configuration bugs (typos, missing destinations).
        for gate_entry in gate_entries:
            if gate_entry.fork_to:
                branch_counts = Counter(gate_entry.fork_to)
                duplicates = sorted([branch for branch, count in branch_counts.items() if count > 1])
                if duplicates:
                    raise GraphValidationError(
                        f"Gate '{gate_entry.name}' has duplicate fork branches: {duplicates}. Each fork branch name must be unique."
                    )
                for branch_name in gate_entry.fork_to:
                    if BranchName(branch_name) in branch_to_coalesce:
                        # Explicit coalesce destination
                        coalesce_name = branch_to_coalesce[BranchName(branch_name)]
                        coalesce_id = coalesce_ids[coalesce_name]
                        graph.add_edge(gate_entry.node_id, coalesce_id, label=branch_name, mode=RoutingMode.COPY)
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
                            f"  1. Be listed in a coalesce 'branches' list, or\n"
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

        # ===== COMPUTE COALESCE INSERTION POINTS =====
        # Coalesce continues after the latest gate that produces any of its branches.
        pipeline_index: dict[NodeID, int] = {node_id: idx for idx, node_id in enumerate(pipeline_nodes)}
        coalesce_gate_index: dict[CoalesceName, int] = {}
        if coalesce_settings:
            for gate_entry in gate_entries:
                if gate_entry.fork_to:
                    gate_idx = pipeline_index[gate_entry.node_id]
                    for branch_name in gate_entry.fork_to:
                        branch_key = BranchName(branch_name)
                        if branch_key in branch_to_coalesce:
                            coalesce_name = branch_to_coalesce[branch_key]
                            if coalesce_name in coalesce_gate_index:
                                existing = coalesce_gate_index[coalesce_name]
                                if gate_idx > existing:
                                    coalesce_gate_index[coalesce_name] = gate_idx
                            else:
                                coalesce_gate_index[coalesce_name] = gate_idx

            for coalesce_name in coalesce_ids:
                if coalesce_name not in coalesce_gate_index:
                    raise GraphValidationError(
                        f"Coalesce '{coalesce_name}' has no producing gate. This should have been caught by branch validation."
                    )

        # Store for external access
        graph._coalesce_gate_index = coalesce_gate_index

        # ===== CONNECT GATE CONTINUE ROUTES =====
        # CRITICAL FIX: Handle ALL continue routes, not just "true"
        for gate_config in gates:
            gid = config_gate_ids[GateName(gate_config.name)]
            # Check if ANY route resolves to "continue"
            has_continue_route = any(target == "continue" for target in gate_config.routes.values())

            if has_continue_route:
                # Determine next node in chain
                gate_idx = pipeline_index[gid]
                if gate_idx + 1 < len(pipeline_nodes):
                    next_node_id = pipeline_nodes[gate_idx + 1]
                else:
                    if SinkName(default_sink) not in sink_ids:
                        raise GraphValidationError(
                            f"Gate '{gate_config.name}' has 'continue' route but is the last gate "
                            f"and default_sink '{default_sink}' is not in configured sinks. "
                            f"Available sinks: {sorted(sink_ids.keys())}"
                        )
                    next_node_id = sink_ids[SinkName(default_sink)]

                if not graph._graph.has_edge(gid, next_node_id, key="continue"):
                    graph.add_edge(gid, next_node_id, label="continue", mode=RoutingMode.MOVE)

        # ===== CONNECT FINAL NODE TO OUTPUT (NO GATES CASE) =====
        if not gates and SinkName(default_sink) in sink_ids:
            graph.add_edge(prev_node_id, sink_ids[SinkName(default_sink)], label="continue", mode=RoutingMode.MOVE)

        # ===== CONNECT COALESCE TO NEXT NODE =====
        if coalesce_settings:
            for coalesce_name, coalesce_id in coalesce_ids.items():
                if SinkName(default_sink) not in sink_ids:
                    raise GraphValidationError(f"Coalesce '{coalesce_name}' has no default sink '{default_sink}' configured.")
                gate_idx = coalesce_gate_index[coalesce_name]
                if gate_idx + 1 < len(pipeline_nodes):
                    next_node_id = pipeline_nodes[gate_idx + 1]
                else:
                    next_node_id = sink_ids[SinkName(default_sink)]
                graph.add_edge(coalesce_id, next_node_id, label="continue", mode=RoutingMode.MOVE)

        # ===== POPULATE COALESCE SCHEMA CONFIG =====
        # Coalesce nodes are structural pass-throughs; record the upstream schema
        # so audit logs reflect the actual data contract at the merge point.
        for coalesce_id in coalesce_ids.values():
            incoming_edges = list(graph._graph.in_edges(coalesce_id))
            if not incoming_edges:
                raise GraphValidationError(f"Coalesce node '{coalesce_id}' has no incoming branches; cannot determine schema for audit.")

            first_from_node = incoming_edges[0][0]
            first_schema = graph.get_node_info(first_from_node).config["schema"]

            for from_node, _to_node in incoming_edges[1:]:
                other_schema = graph.get_node_info(from_node).config["schema"]
                if other_schema != first_schema:
                    raise GraphValidationError(
                        f"Coalesce node '{coalesce_id}' receives mismatched schema configs from branches. "
                        "Schemas must be identical at merge points."
                    )

            graph.get_node_info(coalesce_id).config["schema"] = first_schema

        # PHASE 2 VALIDATION: Validate schema compatibility AFTER graph is built
        graph.validate_edge_compatibility()

        return graph

    def get_sink_id_map(self) -> dict[SinkName, NodeID]:
        """Get explicit sink_name -> node_id mapping.

        Returns:
            Dict mapping each sink's logical name to its graph node ID.
            No substring matching required - use this for direct lookup.
        """
        return dict(self._sink_id_map)

    def get_transform_id_map(self) -> dict[int, NodeID]:
        """Get explicit sequence -> node_id mapping for transforms.

        Returns:
            Dict mapping transform sequence position (0-indexed) to node ID.
        """
        return dict(self._transform_id_map)

    def get_config_gate_id_map(self) -> dict[GateName, NodeID]:
        """Get explicit gate_name -> node_id mapping for config-driven gates.

        Returns:
            Dict mapping gate name to its graph node ID.
        """
        return dict(self._config_gate_id_map)

    def get_aggregation_id_map(self) -> dict[AggregationName, NodeID]:
        """Get explicit agg_name -> node_id mapping for aggregations.

        Returns:
            Dict mapping aggregation name to its graph node ID.
        """
        return dict(self._aggregation_id_map)

    def get_coalesce_id_map(self) -> dict[CoalesceName, NodeID]:
        """Get explicit coalesce_name -> node_id mapping.

        Returns:
            Dict mapping coalesce name to its graph node ID.
        """
        return dict(self._coalesce_id_map)

    def get_branch_to_coalesce_map(self) -> dict[BranchName, CoalesceName]:
        """Get branch_name -> coalesce_name mapping.

        Returns:
            Dict mapping fork branch names to their coalesce destination.
            Branches not in this map route to the output sink.
        """
        return dict(self._branch_to_coalesce)

    def get_coalesce_gate_index(self) -> dict[CoalesceName, int]:
        """Get coalesce_name -> producing gate pipeline index mapping.

        Returns the pipeline index of the gate that produces each coalesce's
        branches. Used by orchestrator to compute coalesce_step_map aligned
        with graph topology.

        Returns:
            Dict mapping coalesce name to the pipeline index of its producing
            fork gate. Empty dict if no coalesce configured.
        """
        return dict(self._coalesce_gate_index)  # Return copy to prevent mutation

    def get_default_sink(self) -> str:
        """Get the default sink name."""
        return self._default_sink

    def get_route_label(self, from_node_id: str, sink_name: str) -> str:
        """Get the route label for an edge from a gate to a sink.

        Args:
            from_node_id: The gate node ID
            sink_name: The sink name (not node ID)

        Returns:
            The route label (e.g., "suspicious") or "continue" for default path
        """
        # Check explicit route mapping first
        if (NodeID(from_node_id), sink_name) in self._route_label_map:
            return self._route_label_map[(NodeID(from_node_id), sink_name)]

        # Default path uses "continue" label
        return "continue"

    def get_route_resolution_map(self) -> dict[tuple[NodeID, str], str]:
        """Get the route resolution map for all gates.

        Returns:
            Dict mapping (gate_node_id, route_label) -> destination.
            Destination is either "continue" or a sink name.
            Used by the executor to resolve route labels from gates.
        """
        return dict(self._route_resolution_map)

    def validate_edge_compatibility(self) -> None:
        """Validate schema compatibility for all edges in the graph.

        Called AFTER graph construction is complete. Validates that each edge
        connects compatible schemas.

        Raises:
            ValueError: If any edge has incompatible schemas

        Note:
            This is PHASE 2 validation (cross-plugin compatibility). Plugin
            SELF-validation happens in PHASE 1 during plugin construction.
        """
        # Validate each edge
        for from_id, to_id, _edge_data in self._graph.edges(data=True):
            self._validate_single_edge(from_id, to_id)

        # Validate all coalesce nodes (must have compatible schemas from all branches)
        coalesce_nodes = [node_id for node_id, data in self._graph.nodes(data=True) if data["info"].node_type == "coalesce"]
        for coalesce_id in coalesce_nodes:
            self._validate_coalesce_compatibility(coalesce_id)

    def _validate_single_edge(self, from_node_id: str, to_node_id: str) -> None:
        """Validate schema compatibility for a single edge.

        Validation is performed in two phases:
        1. CONTRACT VALIDATION: Check required/guaranteed field names
        2. TYPE VALIDATION: Check field type compatibility

        Contract validation catches missing fields even for dynamic schemas,
        which is critical for template-based transforms (e.g., LLM) that
        reference specific row fields.

        Args:
            from_node_id: Source node ID
            to_node_id: Destination node ID

        Raises:
            ValueError: If schemas are incompatible or contracts violated
        """
        to_info = self.get_node_info(to_node_id)

        # Skip edge validation for coalesce nodes - they have special validation
        # that checks all incoming branches together
        if to_info.node_type == "coalesce":
            return

        # Rule 0: Gates must preserve schema (input == output)
        if (
            to_info.node_type == "gate"
            and to_info.input_schema is not None
            and to_info.output_schema is not None
            and to_info.input_schema != to_info.output_schema
        ):
            raise ValueError(
                f"Gate '{to_node_id}' must preserve schema: "
                f"input_schema={to_info.input_schema.__name__}, "
                f"output_schema={to_info.output_schema.__name__}"
            )

        # ===== PHASE 1: CONTRACT VALIDATION (field name requirements) =====
        # This catches missing fields even for dynamic schemas
        consumer_required = self._get_required_fields(to_node_id)

        if consumer_required:
            # Get effective guaranteed fields (walks through pass-through nodes)
            producer_guaranteed = self._get_effective_guaranteed_fields(from_node_id)

            missing = consumer_required - producer_guaranteed
            if missing:
                # Build actionable error message
                from_info = self.get_node_info(from_node_id)
                raise ValueError(
                    f"Schema contract violation: edge '{from_node_id}' → '{to_node_id}'\n"
                    f"  Consumer ({to_info.plugin_name}) requires fields: {sorted(consumer_required)}\n"
                    f"  Producer ({from_info.plugin_name}) guarantees: "
                    f"{sorted(producer_guaranteed) if producer_guaranteed else '(none - dynamic schema)'}\n"
                    f"  Missing fields: {sorted(missing)}\n"
                    f"\n"
                    f"Fix: Either:\n"
                    f"  1. Add missing fields to producer's schema or guaranteed_fields, or\n"
                    f"  2. Remove from consumer's required_input_fields if truly optional"
                )

        # ===== PHASE 2: TYPE VALIDATION (schema compatibility) =====
        # Get EFFECTIVE producer schema (walks through gates if needed)
        producer_schema = self._get_effective_producer_schema(from_node_id)
        consumer_schema = to_info.input_schema

        # Rule 1: Dynamic schemas (None) bypass type validation
        if producer_schema is None or consumer_schema is None:
            return  # Dynamic schema - compatible with anything

        # Handle dynamic schemas (no explicit fields + extra='allow')
        # These are created by _create_dynamic_schema and accept anything
        # NOTE: We control all schemas via PluginSchema base class which sets model_config["extra"].
        # Direct access is correct per Tier 1 trust model - missing key would be our bug.
        producer_is_dynamic = len(producer_schema.model_fields) == 0 and producer_schema.model_config["extra"] == "allow"
        consumer_is_dynamic = len(consumer_schema.model_fields) == 0 and consumer_schema.model_config["extra"] == "allow"
        if producer_is_dynamic or consumer_is_dynamic:
            return  # Dynamic schemas bypass static type validation

        # Rule 2: Full compatibility check (missing fields, type mismatches, extra fields)
        result = check_compatibility(producer_schema, consumer_schema)
        if not result.compatible:
            raise ValueError(
                f"Edge from '{from_node_id}' to '{to_node_id}' invalid: "
                f"producer schema '{producer_schema.__name__}' incompatible with "
                f"consumer schema '{consumer_schema.__name__}': {result.error_message}"
            )

    def _get_effective_producer_schema(self, node_id: str) -> type[PluginSchema] | None:
        """Get effective output schema, walking through pass-through nodes (gates, coalesce).

        Gates and coalesce nodes don't transform data - they inherit schema from their
        upstream producers. This method walks backwards through the graph to find the
        nearest schema-carrying producer.

        Args:
            node_id: Node to get effective schema for

        Returns:
            Output schema type, or None if dynamic

        Raises:
            ValueError: If pass-through node has no incoming edges (graph construction bug)
        """
        node_info = self.get_node_info(node_id)

        # If node has output_schema, return it directly
        if node_info.output_schema is not None:
            return node_info.output_schema

        # Node has no schema - check if it's a pass-through type (gate or coalesce)
        if node_info.node_type in ("gate", "coalesce"):
            # Pass-through nodes inherit schema from upstream producers
            incoming = list(self._graph.in_edges(node_id, data=True))

            if not incoming:
                # Pass-through node with no inputs is a graph construction bug - CRASH
                raise ValueError(
                    f"{node_info.node_type.capitalize()} node '{node_id}' has no incoming edges - "
                    f"this indicates a bug in graph construction"
                )

            # Gather all input schemas for validation
            all_schemas: list[tuple[str, type[PluginSchema] | None]] = []
            for from_id, _, _ in incoming:
                schema = self._get_effective_producer_schema(from_id)
                all_schemas.append((from_id, schema))

            # For multi-input nodes, check for mixed dynamic/explicit schemas first
            # BUG FIX: P2-2026-02-01-dynamic-branch-schema-mismatch-not-detected
            # Mixed dynamic/explicit branches create semantic mismatches that cause runtime failures
            if len(all_schemas) > 1:
                dynamic_branches = [(nid, s) for nid, s in all_schemas if self._is_dynamic_schema(s)]
                explicit_branches = [(nid, s) for nid, s in all_schemas if not self._is_dynamic_schema(s)]

                if dynamic_branches and explicit_branches:
                    # Mixed dynamic/explicit - reject with clear error
                    dynamic_names = [nid for nid, _ in dynamic_branches]
                    # Schema is guaranteed non-None here (explicit_branches filtered out dynamic/None)
                    explicit_names = [f"{nid} ({s.__name__})" for nid, s in explicit_branches if s is not None]
                    raise ValueError(
                        f"{node_info.node_type.capitalize()} '{node_id}' has mixed dynamic/explicit schemas - "
                        f"this is not allowed because dynamic branches may produce rows missing fields "
                        f"expected by downstream consumers. "
                        f"Dynamic branches: {dynamic_names}, explicit branches: {explicit_names}. "
                        f"Fix: ensure all branches produce explicit schemas with compatible fields, "
                        f"or all branches produce dynamic schemas."
                    )

                # All explicit - verify structural compatibility
                if len(explicit_branches) > 1:
                    _first_id, first_schema = explicit_branches[0]
                    for _other_id, other_schema in explicit_branches[1:]:
                        compatible, error_msg = self._schemas_structurally_compatible(first_schema, other_schema)
                        if not compatible:
                            # Schemas are guaranteed non-None here (explicit_branches filtered out dynamic/None)
                            first_name = first_schema.__name__ if first_schema is not None else "dynamic"
                            other_name = other_schema.__name__ if other_schema is not None else "dynamic"
                            raise ValueError(
                                f"{node_info.node_type.capitalize()} '{node_id}' receives incompatible schemas from "
                                f"multiple inputs - this is a graph construction bug. "
                                f"First input: {first_name}, other input: {other_name}. {error_msg}"
                            )

            # Return first schema (all are now either all-dynamic or all-explicit-compatible)
            return all_schemas[0][1]

        # Not a pass-through node and no schema - return None (dynamic)
        return None

    def _is_dynamic_schema(self, schema: type[PluginSchema] | None) -> bool:
        """Check if a schema is dynamic (accepts any fields).

        A schema is dynamic if:
        - It is None (unspecified output_schema)
        - It has no fields and allows extra fields (structural dynamic)

        Args:
            schema: Schema class or None

        Returns:
            True if schema is dynamic, False if explicit
        """
        if schema is None:
            return True

        # Structural dynamic: no fields + extra="allow"
        # NOTE: We control all schemas via PluginSchema base class which sets model_config["extra"].
        # Direct access is correct per Tier 1 trust model - missing key would be our bug.
        return len(schema.model_fields) == 0 and schema.model_config["extra"] == "allow"

    def _schemas_structurally_compatible(
        self, schema_a: type[PluginSchema] | None, schema_b: type[PluginSchema] | None
    ) -> tuple[bool, str]:
        """Check if two schemas are structurally compatible (not by class identity).

        Uses check_compatibility() for structural comparison. Handles dynamic schemas
        which are compatible with anything.

        Args:
            schema_a: First schema (or None for dynamic)
            schema_b: Second schema (or None for dynamic)

        Returns:
            Tuple of (is_compatible, error_message). If compatible, error_message is empty.
        """
        # Both dynamic - compatible
        if self._is_dynamic_schema(schema_a) and self._is_dynamic_schema(schema_b):
            return True, ""

        # One dynamic, one explicit - for general compatibility, allow this
        # (Pass-through nodes use stricter checking via _check_passthrough_schema_homogeneity)
        if self._is_dynamic_schema(schema_a) or self._is_dynamic_schema(schema_b):
            return True, ""

        # Both explicit schemas - same class is trivially compatible
        if schema_a is schema_b:
            return True, ""

        # At this point both schemas are explicit (not None, not dynamic)
        # Type narrowing for mypy: we've already returned if either is dynamic/None
        assert schema_a is not None and schema_b is not None

        # Both explicit schemas - use bidirectional structural comparison
        # For coalesce/pass-through nodes, schemas must be mutually compatible
        result_ab = check_compatibility(schema_a, schema_b)
        result_ba = check_compatibility(schema_b, schema_a)

        if result_ab.compatible and result_ba.compatible:
            return True, ""

        # Build error message showing what's incompatible
        errors = []
        if not result_ab.compatible:
            errors.append(f"{schema_a.__name__} -> {schema_b.__name__}: {result_ab.error_message}")
        if not result_ba.compatible:
            errors.append(f"{schema_b.__name__} -> {schema_a.__name__}: {result_ba.error_message}")
        return False, "; ".join(errors)

    def _validate_coalesce_compatibility(self, coalesce_id: str) -> None:
        """Validate all inputs to coalesce node have compatible schemas.

        Args:
            coalesce_id: Coalesce node ID

        Raises:
            ValueError: If branches have incompatible schemas
        """
        incoming = list(self._graph.in_edges(coalesce_id, data=True))

        if len(incoming) < 2:
            return  # Degenerate case (1 branch) - always compatible

        # Get effective schema from first branch
        first_edge_source = incoming[0][0]
        first_schema = self._get_effective_producer_schema(first_edge_source)

        # Verify all other branches have structurally compatible schemas
        # Note: Uses structural comparison, not class identity (P2-2026-01-30 fix)
        for from_id, _, _ in incoming[1:]:
            other_schema = self._get_effective_producer_schema(from_id)
            compatible, error_msg = self._schemas_structurally_compatible(first_schema, other_schema)
            if not compatible:
                first_name = first_schema.__name__ if first_schema else "dynamic"
                other_name = other_schema.__name__ if other_schema else "dynamic"
                raise ValueError(
                    f"Coalesce '{coalesce_id}' receives incompatible schemas from "
                    f"multiple branches: first branch has {first_name}, "
                    f"branch from '{from_id}' has {other_name}. {error_msg}"
                )

    # ===== CONTRACT VALIDATION HELPERS =====

    def _get_schema_config_from_node(self, node_id: str) -> SchemaConfig | None:
        """Extract SchemaConfig from node.

        Priority:
        1. output_schema_config from NodeInfo (computed by transform)
        2. schema from config dict (raw config)

        Transforms may compute their schema config dynamically (e.g., LLM transforms
        determine guaranteed_fields and audit_fields from their configuration). When
        this computed schema config is available in NodeInfo, it takes precedence
        over the raw config dict.

        Args:
            node_id: Node ID to get schema config from

        Returns:
            SchemaConfig if available, None if schema not in config
        """
        node_info = self.get_node_info(node_id)

        # First check if we have computed schema config in NodeInfo
        # (populated by from_plugin_instances when transform has _output_schema_config)
        if node_info.output_schema_config is not None:
            return node_info.output_schema_config

        # Fall back to parsing from raw config dict
        schema_dict = node_info.config.get("schema")
        if schema_dict is None:
            return None

        # Parse the schema dict into SchemaConfig
        # Handle both raw dict form and serialized form
        if isinstance(schema_dict, dict):
            return SchemaConfig.from_dict(schema_dict)

        # Dynamic schema stored as string
        if schema_dict == "dynamic":
            return SchemaConfig.from_dict({"fields": "dynamic"})

        return None

    def _get_guaranteed_fields(self, node_id: str) -> frozenset[str]:
        """Get fields that a node guarantees in its output.

        Priority:
        1. Explicit guaranteed_fields in schema config
        2. Declared fields in free/strict mode schemas
        3. Empty set for pure dynamic schemas

        Args:
            node_id: Node ID to get guarantees from

        Returns:
            Frozenset of field names the node guarantees to output
        """
        schema_config = self._get_schema_config_from_node(node_id)

        if schema_config is None:
            return frozenset()

        return schema_config.get_effective_guaranteed_fields()

    def _get_required_fields(self, node_id: str) -> frozenset[str]:
        """Get fields that a node EXPLICITLY requires in its input.

        This returns only explicit contract declarations, not implicit
        requirements from typed schemas. The existing type validation
        handles typed schema compatibility separately.

        Priority:
        1. Explicit required_input_fields from plugin config (TransformDataConfig)
        2. Explicit required_fields in schema config

        Note: This deliberately does NOT include implicit requirements from
        strict/free mode schemas. Those are handled by type validation, which
        correctly skips validation when either side is dynamic.

        Args:
            node_id: Node ID to get requirements from

        Returns:
            Frozenset of field names explicitly required
        """
        node_info = self.get_node_info(node_id)

        # Check plugin config for required_input_fields (highest priority)
        # This is the explicit declaration from TransformDataConfig
        required_input = node_info.config.get("required_input_fields")
        if required_input is not None and len(required_input) > 0:
            return frozenset(required_input)

        # For aggregation nodes, also check inside "options" where transform config is nested
        if node_info.node_type == "aggregation":
            options = node_info.config["options"]
            if not isinstance(options, dict):
                raise TypeError(f"Aggregation node config 'options' must be dict, got {type(options).__name__}")
            if "required_input_fields" in options:
                required_input = options["required_input_fields"]
                if required_input is not None and len(required_input) > 0:
                    return frozenset(required_input)

        # Check for explicit required_fields in schema config
        schema_config = self._get_schema_config_from_node(node_id)

        if schema_config is None:
            return frozenset()

        # Only return explicit required_fields declaration, NOT implicit from typed schemas
        if schema_config.required_fields is not None:
            return frozenset(schema_config.required_fields)

        return frozenset()

    def _get_effective_guaranteed_fields(self, node_id: str) -> frozenset[str]:
        """Get effective output guarantees, walking through pass-through nodes.

        Gates and coalesce nodes don't transform data - they inherit guarantees
        from their upstream producers. This method walks backwards through the
        graph to find actual guarantees.

        For coalesce nodes, returns the intersection of all branch guarantees
        (only fields guaranteed by ALL branches are guaranteed after merge).

        IMPORTANT: Gates ALWAYS inherit from upstream, even if they have raw schema
        guarantees. This is because gates copy raw config["schema"] from upstream,
        which may not include computed guarantees from output_schema_config
        (e.g., LLM transforms compute additional guaranteed_fields like *_usage).
        See P1-2026-01-31-gate-drops-computed-schema-guarantees for details.

        Args:
            node_id: Node to get effective guarantees for

        Returns:
            Frozenset of field names effectively guaranteed at this point
        """
        node_info = self.get_node_info(node_id)

        # Gates ALWAYS inherit from upstream - they don't compute schemas.
        # Their raw config["schema"] may miss computed guarantees from upstream's
        # output_schema_config (e.g., LLM *_usage fields).
        if node_info.node_type == "gate":
            incoming = list(self._graph.in_edges(node_id, data=True))
            if not incoming:
                return frozenset()
            # Gates pass through - inherit from single upstream
            return self._get_effective_guaranteed_fields(incoming[0][0])

        # Coalesce nodes return intersection of branch guarantees
        if node_info.node_type == "coalesce":
            incoming = list(self._graph.in_edges(node_id, data=True))
            if not incoming:
                return frozenset()
            # Coalesce guarantees the INTERSECTION of branch guarantees
            branch_guarantees = [self._get_effective_guaranteed_fields(from_id) for from_id, _, _ in incoming]
            if not branch_guarantees:
                return frozenset()
            # Start with first, intersect with rest
            result = branch_guarantees[0]
            for guarantees in branch_guarantees[1:]:
                result = result & guarantees
            return result

        # Non-pass-through nodes return their own guarantees
        return self._get_guaranteed_fields(node_id)
