# src/elspeth/core/dag/graph.py
"""ExecutionGraph class — query, validation, and traversal operations.

Construction logic lives in builder.py; this module contains the graph
class with all runtime methods. The from_plugin_instances() classmethod
is a thin facade that delegates to builder.build_execution_graph().
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, cast

import networkx as nx
from networkx import MultiDiGraph

from elspeth.contracts import (
    EdgeInfo,
    RouteDestination,
    RoutingMode,
    check_compatibility,
)
from elspeth.contracts.enums import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.types import (
    AggregationName,
    BranchName,
    CoalesceName,
    GateName,
    NodeID,
    SinkName,
)
from elspeth.core.dag.models import (
    GraphValidationError,
    GraphValidationWarning,
    NodeConfig,
    NodeInfo,
    _suggest_similar,
)

if TYPE_CHECKING:
    from elspeth.contracts import PluginSchema
    from elspeth.core.config import (
        AggregationSettings,
        CoalesceSettings,
        GateSettings,
        SourceSettings,
    )
    from elspeth.core.dag.models import WiredTransform
    from elspeth.plugins.protocols import SinkProtocol, SourceProtocol, TransformProtocol


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
        self._route_label_map: dict[tuple[NodeID, str], str] = {}  # (gate_node, sink_name) -> route_label
        self._route_resolution_map: dict[tuple[NodeID, str], RouteDestination] = {}
        self._branch_gate_map: dict[BranchName, NodeID] = {}  # branch_name -> producing gate node ID
        self._pipeline_nodes: list[NodeID] | None = None  # Ordered processing nodes (no source/sinks); None = not yet populated
        self._node_step_map: dict[NodeID, int] = {}  # node_id -> audit step (source=0)

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
        """Return a frozen copy of the underlying NetworkX graph.

        Use this for topology analysis, subgraph operations, and other
        NetworkX algorithms that require direct graph access.

        Returns:
            A frozen (immutable) copy of the internal MultiDiGraph.
            Mutation attempts raise nx.NetworkXError.
        """
        return nx.freeze(self._graph.copy())  # type: ignore[no-any-return]

    def add_node(
        self,
        node_id: str,
        *,
        node_type: NodeType,
        plugin_name: str,
        config: NodeConfig | None = None,
        input_schema: type[PluginSchema] | None = None,
        output_schema: type[PluginSchema] | None = None,
        input_schema_config: SchemaConfig | None = None,
        output_schema_config: SchemaConfig | None = None,
    ) -> None:
        """Add a node to the execution graph.

        Args:
            node_id: Unique node identifier
            node_type: NodeType enum value
            plugin_name: Plugin identifier
            config: Node configuration (see NodeConfig type alias for per-node-type structure)
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
            mode: Routing mode (MOVE, COPY, or DIVERT)
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
        4. All nodes are reachable from source (no disconnected/orphaned nodes)
        5. Edge labels are unique per source node

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
        sources = [node_id for node_id, data in self._graph.nodes(data=True) if data["info"].node_type == NodeType.SOURCE]
        if len(sources) != 1:
            raise GraphValidationError(f"Graph must have exactly one source, found {len(sources)}")

        # Check for at least one sink
        sinks = self.get_sinks()
        if len(sinks) < 1:
            raise GraphValidationError("Graph must have at least one sink")

        # Check for unreachable nodes (nodes not reachable from source)
        source_id = sources[0]  # We already validated exactly one source exists
        reachable = nx.descendants(self._graph, source_id)
        reachable.add(source_id)  # Include source itself in reachable set

        all_nodes = set(self._graph.nodes())
        unreachable = all_nodes - reachable

        if unreachable:
            # Build detailed error message with node types
            unreachable_details = [f"{node_id} ({self._graph.nodes[node_id]['info'].node_type})" for node_id in sorted(unreachable)]
            raise GraphValidationError(
                f"Graph validation failed: {len(unreachable)} unreachable node(s) detected:\n"
                f"  {', '.join(unreachable_details)}\n"
                f"All nodes must be reachable from the source node '{source_id}'."
            )

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
        sources = [NodeID(node_id) for node_id, data in self._graph.nodes(data=True) if data["info"].node_type == NodeType.SOURCE]
        return sources[0] if len(sources) == 1 else None

    def get_sinks(self) -> list[NodeID]:
        """Get all sink node IDs.

        Returns:
            List of sink node IDs.
        """
        # All nodes have "info" - added via add_node(), direct access is safe
        return [NodeID(node_id) for node_id, data in self._graph.nodes(data=True) if data["info"].node_type == NodeType.SINK]

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

    def _validate_route_resolution_map_complete(self) -> None:
        """Ensure every declared gate route label resolves to a destination.

        This is a construction-time safety check to prevent runtime failures
        when a gate emits a declared label that is missing from
        _route_resolution_map.
        """
        for node_id, attrs in self._graph.nodes(data=True):
            info = cast(NodeInfo, attrs["info"])
            if info.node_type != NodeType.GATE:
                continue

            routes = cast(dict[str, str], info.config["routes"])
            for route_label in routes:
                key = (NodeID(node_id), route_label)
                if key not in self._route_resolution_map:
                    raise GraphValidationError(
                        f"Gate '{info.plugin_name}' route label '{route_label}' has no destination in route resolution map. "
                        "All declared route labels must resolve during graph construction."
                    )

    @staticmethod
    def _validate_connection_namespaces(
        *,
        producers: dict[str, tuple[NodeID, str]],
        consumers: dict[str, NodeID],
        consumer_claims: list[tuple[str, NodeID, str]],
        sink_names: set[str],
        check_dangling: bool = True,
    ) -> None:
        """Validate declarative connection namespace integrity.

        Enforces:
        - Duplicate consumers are forbidden (fan-out requires explicit gate)
        - Every consumed connection has a producer
        - Connection and sink namespaces are disjoint
        - Every produced connection is consumed (or emitted to sink directly)
        """
        consumer_counts = Counter(name for name, _node_id, _desc in consumer_claims)
        duplicate_consumers = sorted(name for name, count in consumer_counts.items() if count > 1)
        if duplicate_consumers:
            error_parts: list[str] = []
            for dup_name in duplicate_consumers:
                dup_entries = [(node_id, desc) for name, node_id, desc in consumer_claims if name == dup_name]
                first_node, first_desc = dup_entries[0]
                second_node, second_desc = dup_entries[1]
                error_parts.append(f"'{dup_name}': {first_desc} ({first_node}) and {second_desc} ({second_node})")
            raise GraphValidationError(
                f"Duplicate consumers for {len(duplicate_consumers)} connection(s): " + "; ".join(error_parts) + ". Use a gate for fan-out."
            )

        for connection_name in consumers:
            if connection_name not in producers:
                suggestions = _suggest_similar(connection_name, sorted(producers.keys()))
                hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise GraphValidationError(
                    f"No producer for connection '{connection_name}'.{hint}\nAvailable connections: {', '.join(sorted(producers.keys()))}"
                )

        connection_names = set(producers.keys()) | set(consumers.keys())
        overlap = connection_names & sink_names
        if overlap:
            raise GraphValidationError(
                f"Connection names overlap with sink names: {sorted(overlap)}. Connection names and sink names must be disjoint."
            )

        if check_dangling:
            dangling_connections = sorted(set(producers.keys()) - set(consumers.keys()))
            if dangling_connections:
                raise GraphValidationError(
                    f"Dangling output connections with no consumer: {dangling_connections}. "
                    "Every produced connection must be consumed or routed to a sink."
                )

    def is_sink_node(self, node_id: NodeID) -> bool:
        """Check if a node is a sink node."""
        return self.get_node_info(node_id).node_type == NodeType.SINK

    def get_first_transform_node(self) -> NodeID | None:
        """Get the first processing node after source via continue edge.

        Returns:
            Node ID of the first processing node (transform/gate/aggregation),
            or None for source-only pipelines.
        """
        source_id = self.get_source()
        if source_id is None:
            return None
        return self.get_next_node(source_id)

    def get_next_node(self, node_id: NodeID) -> NodeID | None:
        """Follow the continue MOVE edge to the next processing node.

        Returns:
            Next processing node ID, or None if node is terminal.
        """
        next_nodes: list[NodeID] = []
        for _from_id, to_id, edge_key, edge_data in self._graph.out_edges(node_id, keys=True, data=True):
            if edge_key != "continue":
                continue
            if edge_data["mode"] != RoutingMode.MOVE:
                continue
            next_node_id = NodeID(to_id)
            if self.is_sink_node(next_node_id):
                continue
            next_nodes.append(next_node_id)

        if len(next_nodes) > 1:
            raise GraphValidationError(f"Node '{node_id}' has multiple continue MOVE edges to processing nodes: {sorted(next_nodes)}")
        if len(next_nodes) == 1:
            return next_nodes[0]
        return None

    def get_pipeline_node_sequence(self) -> list[NodeID]:
        """Get ordered processing nodes in pipeline traversal order."""
        if self._pipeline_nodes is not None:
            return list(self._pipeline_nodes)

        first_node = self.get_first_transform_node()
        if first_node is None:
            return []

        reachable: set[NodeID] = set()
        pending: list[NodeID] = [first_node]
        while pending:
            current = pending.pop()
            if current in reachable:
                continue
            reachable.add(current)

            for _from_id, to_id, _edge_key, edge_data in self._graph.out_edges(current, keys=True, data=True):
                if edge_data["mode"] != RoutingMode.MOVE:
                    continue
                target = NodeID(to_id)
                if self.is_sink_node(target):
                    continue
                pending.append(target)

        return [node_id for node_id in (NodeID(node) for node in self.topological_order()) if node_id in reachable]

    def build_step_map(self) -> dict[NodeID, int]:
        """Build node -> audit step map (source=0, processing nodes start at 1)."""
        source_id = self.get_source()
        if source_id is None:
            return {}

        step_map: dict[NodeID, int] = {source_id: 0}
        for idx, node_id in enumerate(self.get_pipeline_node_sequence(), start=1):
            step_map[node_id] = idx

        self._node_step_map = dict(step_map)
        return dict(step_map)

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
                from_node=NodeID(u),
                to_node=NodeID(v),
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
                from_node=NodeID(u),
                to_node=NodeID(v),
                label=data["label"],
                mode=data["mode"],
            )
            for u, v, _key, data in self._graph.in_edges(node_id, data=True, keys=True)
        ]

    @classmethod
    def from_plugin_instances(
        cls,
        source: SourceProtocol,
        source_settings: SourceSettings,
        transforms: list[WiredTransform],
        sinks: dict[str, SinkProtocol],
        aggregations: dict[str, tuple[TransformProtocol, AggregationSettings]],
        gates: list[GateSettings],
        coalesce_settings: list[CoalesceSettings] | None = None,
    ) -> ExecutionGraph:
        """Build ExecutionGraph from plugin instances.

        CORRECT method for graph construction - enables schema validation.
        Schemas extracted directly from instance attributes.

        Routing is explicit: terminal transforms and sources declare their
        output sink via on_success. There is no default_sink fallback.

        Args:
            source: Instantiated source plugin
            source_settings: Source settings (wiring metadata)
            transforms: Wired transforms (plugin instance + settings metadata)
            sinks: Dict of sink_name -> instantiated sink
            aggregations: Dict of agg_name -> (transform_instance, AggregationSettings)
            gates: Config-driven gate settings
            coalesce_settings: Coalesce configs for fork/join patterns

        Returns:
            ExecutionGraph with schemas populated

        Raises:
            GraphValidationError: If gate routes reference unknown sinks,
                terminal nodes lack on_success, or on_success references
                unknown sinks.
        """
        from elspeth.core.dag.builder import build_execution_graph

        return build_execution_graph(
            cls=cls,
            source=source,
            source_settings=source_settings,
            transforms=transforms,
            sinks=sinks,
            aggregations=aggregations,
            gates=gates,
            coalesce_settings=coalesce_settings,
        )

    # ===== PUBLIC SETTERS (construction-time) =====

    def set_sink_id_map(self, mapping: dict[SinkName, NodeID]) -> None:
        """Set the sink_name -> node_id mapping."""
        self._sink_id_map = dict(mapping)

    def set_transform_id_map(self, mapping: dict[int, NodeID]) -> None:
        """Set the transform sequence -> node_id mapping."""
        self._transform_id_map = dict(mapping)

    def set_config_gate_id_map(self, mapping: dict[GateName, NodeID]) -> None:
        """Set the gate_name -> node_id mapping."""
        self._config_gate_id_map = dict(mapping)

    def set_route_resolution_map(self, mapping: dict[tuple[NodeID, str], RouteDestination]) -> None:
        """Set the (gate_node_id, route_label) -> destination mapping."""
        self._route_resolution_map = dict(mapping)

    def set_aggregation_id_map(self, mapping: dict[AggregationName, NodeID]) -> None:
        """Set the agg_name -> node_id mapping."""
        self._aggregation_id_map = dict(mapping)

    def set_coalesce_id_map(self, mapping: dict[CoalesceName, NodeID]) -> None:
        """Set the coalesce_name -> node_id mapping."""
        self._coalesce_id_map = dict(mapping)

    def set_branch_to_coalesce(self, mapping: dict[BranchName, CoalesceName]) -> None:
        """Set the branch_name -> coalesce_name mapping."""
        self._branch_to_coalesce = dict(mapping)

    def set_route_label_map(self, mapping: dict[tuple[NodeID, str], str]) -> None:
        """Set the (gate_node, sink_name) -> route_label mapping."""
        self._route_label_map = dict(mapping)

    def set_branch_gate_map(self, mapping: dict[BranchName, NodeID]) -> None:
        """Set the branch_name -> producing gate node ID mapping."""
        self._branch_gate_map = dict(mapping)

    def set_pipeline_nodes(self, nodes: list[NodeID]) -> None:
        """Set the ordered processing node sequence."""
        self._pipeline_nodes = list(nodes)

    def set_node_step_map(self, mapping: dict[NodeID, int]) -> None:
        """Set the node_id -> audit step mapping."""
        self._node_step_map = dict(mapping)

    def add_route_resolution_entry(self, gate_id: NodeID, label: str, dest: RouteDestination) -> None:
        """Add a single entry to the route resolution map."""
        self._route_resolution_map[(gate_id, label)] = dest

    def add_route_label_entry(self, gate_id: NodeID, sink_name: str, label: str) -> None:
        """Add a single entry to the route label map."""
        self._route_label_map[(gate_id, sink_name)] = label

    # ===== PUBLIC GETTERS =====

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

    def get_branch_first_nodes(self) -> dict[str, NodeID]:
        """Get mapping of branch names to their first processing node.

        For every branch that routes to a coalesce node, returns the first
        node the token should visit:
        - Identity branches (COPY edge gate→coalesce): maps to coalesce node ID
        - Transform branches (MOVE edge chain→coalesce): maps to the first
          transform's node ID in the branch chain

        The mapping covers ALL coalesce branches, eliminating the need for
        defensive .get() at runtime.

        Returns:
            Dict mapping branch name (str) to the first processing NodeID.
            Empty dict if no coalesce branches exist.
        """
        result: dict[str, NodeID] = {}

        for branch_name, coalesce_name in self._branch_to_coalesce.items():
            coalesce_nid = self._coalesce_id_map[coalesce_name]

            # Check if this is an identity branch (direct COPY edge from gate to coalesce).
            # Identity branches have a COPY edge labelled with the branch name pointing
            # at the coalesce node — the token goes straight to coalesce.
            is_identity = False
            for _from_id, _to_id, _key, data in self._graph.in_edges(coalesce_nid, keys=True, data=True):
                if data["mode"] == RoutingMode.COPY and data["label"] == branch_name:
                    is_identity = True
                    break

            if is_identity:
                result[branch_name] = coalesce_nid
            else:
                # Transform branch: trace backwards from coalesce through MOVE edges
                # to find the first node in this branch's transform chain.
                # The chain is: gate -[branch_name MOVE]-> T1 -[... MOVE]-> Tn -[... MOVE]-> coalesce
                # We need T1 (the first transform after the gate).
                first_node, _last_node = self._trace_branch_endpoints(coalesce_nid, branch_name)
                result[branch_name] = first_node

        return result

    def _trace_branch_endpoints(self, coalesce_nid: NodeID, branch_name: str) -> tuple[NodeID, NodeID]:
        """Trace backwards from coalesce to find the first AND last transforms in a branch chain.

        Walks backwards through MOVE edges from the coalesce node to find both
        endpoints of the transform chain for a given branch. The chain terminates
        at the fork gate node (which produces the branch via a MOVE edge labelled
        with the branch name).

        The backward walk follows ANY MOVE edge, not just ``"continue"`` edges,
        because branch chains may include intermediate routing gates whose
        outgoing edges carry route-specific labels (e.g., ``"approved"``).

        Branch entry identification requires matching BOTH the edge label AND the
        edge origin (the fork gate), because intermediate gates within the branch
        may produce MOVE edges whose labels collide with the branch name.

        Args:
            coalesce_nid: The coalesce node to trace back from
            branch_name: The branch name to trace

        Returns:
            ``(first_node, last_node)`` — first_node is the first transform
            after the gate (receives the branch_name MOVE edge); last_node is
            the immediate MOVE predecessor of the coalesce.

        Raises:
            GraphValidationError: If the branch chain cannot be traced
        """
        # Resolve the fork gate that originates this specific branch.
        fork_gate_nid = self._branch_gate_map[BranchName(branch_name)]

        visited: set[NodeID] = set()
        candidates: list[NodeID] = []

        # Collect MOVE-edge predecessors of coalesce (these are the last transforms in branch chains)
        for from_id, _to_id, _key, data in self._graph.in_edges(coalesce_nid, keys=True, data=True):
            if data["mode"] == RoutingMode.MOVE:
                candidates.append(NodeID(from_id))

        # For each candidate, walk backwards through MOVE edges
        # until we find the node whose incoming edge has label == branch_name
        # AND originates from the fork gate (not an intermediate gate).
        for candidate in candidates:
            current = candidate
            visited.clear()

            while current not in visited:
                visited.add(current)
                # Look for incoming MOVE edge with label == branch_name FROM the fork gate
                found_branch_entry = False
                for from_id, _to_id, _key, data in self._graph.in_edges(current, keys=True, data=True):
                    if data["mode"] == RoutingMode.MOVE and data["label"] == branch_name and NodeID(from_id) == fork_gate_nid:
                        found_branch_entry = True
                        break

                if found_branch_entry:
                    # current = first node; candidate = last node before coalesce
                    return current, candidate

                # Walk backwards through any MOVE edge to find predecessor.
                # Not restricted to "continue" because intermediate routing gates
                # produce edges with route-specific labels (e.g., "approved").
                predecessor: NodeID | None = None
                for from_id, _to_id, _key, data in self._graph.in_edges(current, keys=True, data=True):
                    if data["mode"] == RoutingMode.MOVE:
                        predecessor = NodeID(from_id)
                        break

                if predecessor is None:
                    break  # No MOVE predecessor — try next candidate
                current = predecessor

        raise GraphValidationError(
            f"Cannot trace first transform for branch '{branch_name}' leading to "
            f"coalesce node '{coalesce_nid}'. This indicates a graph construction bug — "
            f"transform branches must have MOVE edge chains from gate to coalesce."
        )

    def get_branch_to_sink_map(self) -> dict[BranchName, SinkName]:
        """Get fork branches that route directly to sinks (not to coalesce).

        Scans COPY-mode edges from gate nodes to sink nodes to build the
        mapping. Branches that route to coalesce nodes are excluded — they
        are handled by the coalesce executor, not terminal sink routing.

        Returns:
            Dict mapping branch names to their target sink names.
            Empty dict if no fork-to-sink branches exist.
        """
        result: dict[BranchName, SinkName] = {}
        sink_node_to_name: dict[NodeID, SinkName] = {nid: name for name, nid in self._sink_id_map.items()}
        for _from_id, to_id, _key, data in self._graph.edges(data=True, keys=True):
            if data["mode"] == RoutingMode.COPY and NodeID(to_id) in sink_node_to_name:
                result[BranchName(data["label"])] = sink_node_to_name[NodeID(to_id)]
        return result

    def get_branch_gate_map(self) -> dict[BranchName, NodeID]:
        """Get branch_name -> producing gate node ID mapping.

        Returns the node ID of the gate that produces each coalesce branch.
        Each branch has exactly one producing gate (validated at build time).

        Returns:
            Dict mapping branch name to the node ID of its producing fork gate.
            Empty dict if no coalesce configured.
        """
        return dict(self._branch_gate_map)  # Return copy to prevent mutation

    def get_terminal_sink_map(self) -> dict[NodeID, SinkName]:
        """Get mapping of terminal node IDs to their on_success sink names.

        Scans outgoing edges labelled "on_success" with MOVE mode to build
        the mapping. Terminal nodes are transforms or coalesce nodes that
        route completed rows directly to a sink.

        Returns:
            Dict mapping terminal node IDs to their declared sink names.
            Empty dict if no terminal nodes (e.g., all paths end at gates).
        """
        result: dict[NodeID, SinkName] = {}
        # Invert the sink_id_map for reverse lookup
        sink_node_to_name: dict[NodeID, SinkName] = {node_id: sink_name for sink_name, node_id in self._sink_id_map.items()}
        for from_id, to_id, _key, data in self._graph.edges(data=True, keys=True):
            if data["label"] == "on_success" and data["mode"] == RoutingMode.MOVE and NodeID(to_id) in sink_node_to_name:
                result[NodeID(from_id)] = sink_node_to_name[NodeID(to_id)]
        return result

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

    def get_route_resolution_map(self) -> dict[tuple[NodeID, str], RouteDestination]:
        """Get the route resolution map for all gates.

        Returns:
            Dict mapping (gate_node_id, route_label) -> destination.
            Destination can be continue, fork, sink, or a processing node.
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
        # Validate each edge (skip divert edges — quarantine/error data doesn't
        # conform to producer schemas because it failed validation or errored)
        for from_id, to_id, edge_data in self._graph.edges(data=True):
            if edge_data["mode"] == RoutingMode.DIVERT:
                continue
            self._validate_single_edge(from_id, to_id)

        # Validate all coalesce nodes (must have compatible schemas from all branches)
        coalesce_nodes = [node_id for node_id, data in self._graph.nodes(data=True) if data["info"].node_type == NodeType.COALESCE]
        for coalesce_id in coalesce_nodes:
            self._validate_coalesce_compatibility(coalesce_id)

    def warn_divert_coalesce_interactions(
        self,
        coalesce_configs: dict[NodeID, CoalesceSettings],
    ) -> list[GraphValidationWarning]:
        """Detect transforms with DIVERT edges that feed require_all coalesces.

        When a branch transform has ``on_error`` routing (DIVERT edge), rows that
        hit the error path are diverted to an error sink and never reach the
        coalesce. If the coalesce uses ``require_all`` policy, it will wait
        indefinitely for the missing branch, holding the other branches' tokens
        in memory until end-of-source flush.

        This is a build-time warning, not an error — the configuration is valid
        but likely to cause operational surprises.

        Algorithm:
          1. Pre-compute set of transform node IDs that have outgoing DIVERT edges.
          2. For each ``require_all`` coalesce, walk backwards from each incoming
             MOVE edge (transform branch) to check if any transform in the chain
             has a DIVERT edge.

        Args:
            coalesce_configs: Mapping of coalesce node IDs to their settings.

        Returns:
            List of warnings (also logged via structlog).
        """
        import structlog

        log = structlog.get_logger()

        # Step 1: pre-compute transforms with DIVERT edges (exit early if none)
        divert_transforms: set[NodeID] = set()
        for edge in self.get_edges():
            if edge.mode == RoutingMode.DIVERT:
                from_info = self.get_node_info(edge.from_node)
                if from_info.node_type == NodeType.TRANSFORM:
                    divert_transforms.add(edge.from_node)

        if not divert_transforms:
            return []

        warnings: list[GraphValidationWarning] = []

        # Step 2: check each require_all coalesce
        for coalesce_nid, coal_config in coalesce_configs.items():
            if coal_config.policy != "require_all":
                continue

            for from_id, _to_id, _key, data in self._graph.in_edges(coalesce_nid, keys=True, data=True):
                edge_mode = data["mode"]

                # Identity branches (COPY from gate) have no transforms — skip
                if edge_mode == RoutingMode.COPY:
                    continue

                # Transform branch (MOVE edge from last transform in chain).
                # Walk backwards through predecessor transforms.
                if edge_mode != RoutingMode.MOVE:
                    continue

                current = NodeID(from_id)
                visited: set[NodeID] = set()

                while current not in visited:
                    visited.add(current)
                    current_info = self.get_node_info(current)
                    if current_info.node_type != NodeType.TRANSFORM:
                        break  # Hit gate/source — stop walking

                    if current in divert_transforms:
                        warning = GraphValidationWarning(
                            code="DIVERT_COALESCE_REQUIRE_ALL",
                            message=(
                                f"Transform '{current}' has on_error routing (DIVERT edge) "
                                f"and feeds require_all coalesce '{coalesce_nid}'. "
                                f"Rows diverted on error will never reach the coalesce, "
                                f"causing other branches to wait until end-of-source flush."
                            ),
                            node_ids=(str(current), str(coalesce_nid)),
                        )
                        warnings.append(warning)
                        log.warning(
                            "divert_coalesce_interaction",
                            code=warning.code,
                            transform=str(current),
                            coalesce=str(coalesce_nid),
                            message=warning.message,
                        )
                        break  # One warning per branch is enough

                    # Walk backwards: find incoming MOVE edge (stay on main chain)
                    predecessor: NodeID | None = None
                    for pred_from, _pred_to, _pred_key, pred_data in self._graph.in_edges(current, keys=True, data=True):
                        if pred_data["mode"] == RoutingMode.DIVERT:
                            continue  # Skip DIVERT edges — stay on main chain
                        if pred_data["mode"] == RoutingMode.MOVE:
                            predecessor = NodeID(pred_from)
                            break

                    if predecessor is None:
                        break
                    current = predecessor

        return warnings

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
            GraphValidationError: If schemas are incompatible or contracts violated
        """
        to_info = self.get_node_info(to_node_id)

        # Skip edge validation for coalesce nodes - they have special validation
        # that checks all incoming branches together
        if to_info.node_type == NodeType.COALESCE:
            return

        # Rule 0: Gates must preserve schema (input == output)
        if (
            to_info.node_type == NodeType.GATE
            and to_info.input_schema is not None
            and to_info.output_schema is not None
            and to_info.input_schema != to_info.output_schema
        ):
            raise GraphValidationError(
                f"Gate '{to_node_id}' must preserve schema: "
                f"input_schema={to_info.input_schema.__name__}, "
                f"output_schema={to_info.output_schema.__name__}"
            )

        # ===== PHASE 1: CONTRACT VALIDATION (field name requirements) =====
        # This catches missing fields even for dynamic schemas
        consumer_required = self.get_required_fields(to_node_id)

        if consumer_required:
            # Get effective guaranteed fields (walks through pass-through nodes)
            producer_guaranteed = self.get_effective_guaranteed_fields(from_node_id)

            missing = consumer_required - producer_guaranteed
            if missing:
                # Build actionable error message
                from_info = self.get_node_info(from_node_id)
                raise GraphValidationError(
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
        producer_schema = self.get_effective_producer_schema(from_node_id)
        consumer_schema = to_info.input_schema

        # Rule 1: Dynamic schemas (None) bypass type validation
        if producer_schema is None or consumer_schema is None:
            return  # Observed schema - compatible with anything

        # Handle observed schemas (no explicit fields + extra='allow')
        # These are created by _create_dynamic_schema and accept anything
        # NOTE: We control all schemas via PluginSchema base class which sets model_config["extra"].
        # Direct access is correct per Tier 1 trust model - missing key would be our bug.
        producer_is_observed = len(producer_schema.model_fields) == 0 and producer_schema.model_config["extra"] == "allow"
        consumer_is_observed = len(consumer_schema.model_fields) == 0 and consumer_schema.model_config["extra"] == "allow"
        if producer_is_observed or consumer_is_observed:
            return  # Observed schemas bypass static type validation

        # Rule 2: Full compatibility check (missing fields, type mismatches, extra fields)
        result = check_compatibility(producer_schema, consumer_schema)
        if not result.compatible:
            raise GraphValidationError(
                f"Edge from '{from_node_id}' to '{to_node_id}' invalid: "
                f"producer schema '{producer_schema.__name__}' incompatible with "
                f"consumer schema '{consumer_schema.__name__}': {result.error_message}"
            )

    def get_effective_producer_schema(self, node_id: str) -> type[PluginSchema] | None:
        """Get effective output schema, walking through pass-through nodes (gates, coalesce).

        Gates and coalesce nodes don't transform data - they inherit schema from their
        upstream producers. This method walks backwards through the graph to find the
        nearest schema-carrying producer.

        Args:
            node_id: Node to get effective schema for

        Returns:
            Output schema type, or None if dynamic

        Raises:
            GraphValidationError: If pass-through node has no incoming edges (graph construction bug)
        """
        node_info = self.get_node_info(node_id)

        # If node has output_schema, return it directly
        if node_info.output_schema is not None:
            return node_info.output_schema

        # Coalesce nodes are NOT pass-throughs — they transform data via merge
        # strategy (nested wraps in {branch: data}, union merges fields, select
        # picks a branch).  Strategy-aware handling:
        #   - select: passes through one branch unchanged → trace that branch's schema
        #   - union/nested: output shape differs from any input → return None (dynamic)
        if node_info.node_type == NodeType.COALESCE:
            merge_strategy = node_info.config["merge"]
            if merge_strategy == "select":
                # Select merge passes through the selected branch's data unchanged.
                # Trace back to that branch's producer schema for type validation.
                select_branch = node_info.config.get("select_branch")
                if select_branch is not None:
                    # Identity branch: COPY edge from gate to coalesce with label == select_branch
                    for from_id, _, edge_data in self._graph.in_edges(node_id, data=True):
                        if edge_data.get("mode") == RoutingMode.COPY and edge_data.get("label") == select_branch:
                            return self.get_effective_producer_schema(from_id)
                    # Transform branch: last transform's edge has label "continue", not
                    # the branch name. Trace backward to find the last transform node.
                    try:
                        _first, last = self._trace_branch_endpoints(NodeID(node_id), select_branch)
                        return self.get_effective_producer_schema(last)
                    except GraphValidationError:
                        pass  # Fall through to None if trace fails
            return None

        # Gates are true pass-throughs — inherit schema from upstream producers
        if node_info.node_type == NodeType.GATE:
            incoming = list(self._graph.in_edges(node_id, data=True))

            if not incoming:
                # Pass-through node with no inputs is a graph construction bug - CRASH
                raise GraphValidationError(
                    f"{node_info.node_type.capitalize()} node '{node_id}' has no incoming edges - "
                    f"this indicates a bug in graph construction"
                )

            # Gather all input schemas for validation
            all_schemas: list[tuple[str, type[PluginSchema] | None]] = []
            for from_id, _, _ in incoming:
                schema = self.get_effective_producer_schema(from_id)
                all_schemas.append((from_id, schema))

            # For multi-input nodes, check for mixed observed/explicit schemas first
            # BUG FIX: P2-2026-02-01-dynamic-branch-schema-mismatch-not-detected
            # Mixed observed/explicit branches create semantic mismatches that cause runtime failures
            if len(all_schemas) > 1:
                observed_branches = [(nid, s) for nid, s in all_schemas if self._is_observed_schema(s)]
                explicit_branches = [(nid, s) for nid, s in all_schemas if not self._is_observed_schema(s)]

                if observed_branches and explicit_branches:
                    # Mixed observed/explicit - reject with clear error
                    observed_names = [nid for nid, _ in observed_branches]
                    # Schema is guaranteed non-None here (explicit_branches filtered out observed/None)
                    explicit_names = [f"{nid} ({s.__name__})" for nid, s in explicit_branches if s is not None]
                    raise GraphValidationError(
                        f"{node_info.node_type.capitalize()} '{node_id}' has mixed observed/explicit schemas - "
                        f"this is not allowed because observed branches may produce rows missing fields "
                        f"expected by downstream consumers. "
                        f"Observed branches: {observed_names}, explicit branches: {explicit_names}. "
                        f"Fix: ensure all branches produce explicit schemas with compatible fields, "
                        f"or all branches produce observed schemas."
                    )

                # All explicit - verify structural compatibility
                if len(explicit_branches) > 1:
                    _first_id, first_schema = explicit_branches[0]
                    for _other_id, other_schema in explicit_branches[1:]:
                        compatible, error_msg = self._schemas_structurally_compatible(first_schema, other_schema)
                        if not compatible:
                            # Schemas are guaranteed non-None here (explicit_branches filtered out observed/None)
                            first_name = first_schema.__name__ if first_schema is not None else "observed"
                            other_name = other_schema.__name__ if other_schema is not None else "observed"
                            raise GraphValidationError(
                                f"{node_info.node_type.capitalize()} '{node_id}' receives incompatible schemas from "
                                f"multiple inputs - this is a graph construction bug. "
                                f"First input: {first_name}, other input: {other_name}. {error_msg}"
                            )

            # Return first schema (all are now either all-observed or all-explicit-compatible)
            return all_schemas[0][1]

        # Not a pass-through node and no schema - return None (observed)
        return None

    def _is_observed_schema(self, schema: type[PluginSchema] | None) -> bool:
        """Check if a schema is observed (accepts any fields, types inferred from data).

        A schema is observed if:
        - It is None (unspecified output_schema)
        - It has no fields and allows extra fields (structural observed)

        Args:
            schema: Schema class or None

        Returns:
            True if schema is observed, False if explicit (fixed/flexible)
        """
        if schema is None:
            return True

        # Structural observed: no fields + extra="allow"
        # NOTE: We control all schemas via PluginSchema base class which sets model_config["extra"].
        # Direct access is correct per Tier 1 trust model - missing key would be our bug.
        return len(schema.model_fields) == 0 and schema.model_config["extra"] == "allow"

    def _schemas_structurally_compatible(
        self, schema_a: type[PluginSchema] | None, schema_b: type[PluginSchema] | None
    ) -> tuple[bool, str]:
        """Check if two schemas are structurally compatible (not by class identity).

        Uses check_compatibility() for structural comparison. Handles observed schemas
        which are compatible with anything.

        Args:
            schema_a: First schema (or None for observed)
            schema_b: Second schema (or None for observed)

        Returns:
            Tuple of (is_compatible, error_message). If compatible, error_message is empty.
        """
        # Both observed - compatible
        if self._is_observed_schema(schema_a) and self._is_observed_schema(schema_b):
            return True, ""

        # One observed, one explicit - for general compatibility, allow this
        # (Pass-through nodes use stricter checking via _check_passthrough_schema_homogeneity)
        if self._is_observed_schema(schema_a) or self._is_observed_schema(schema_b):
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

        Strategy-aware: only ``union`` requires cross-branch schema compatibility.
        ``nested`` and ``select`` strategies have no cross-branch constraint because
        branches are keyed separately (nested) or only one branch is used (select).

        Args:
            coalesce_id: Coalesce node ID

        Raises:
            GraphValidationError: If branches have incompatible schemas
        """
        incoming = list(self._graph.in_edges(coalesce_id, data=True))

        if len(incoming) < 2:
            return  # Degenerate case (1 branch) - always compatible

        # Determine merge strategy from node config.
        # Config is populated by the builder — direct access is correct (Tier 1).
        node_info = self.get_node_info(coalesce_id)
        merge_strategy = node_info.config["merge"]

        # nested/select strategies have no cross-branch schema constraint
        if merge_strategy in ("nested", "select"):
            return

        # union strategy: gather all branch schemas and validate
        all_schemas: list[tuple[str, type[PluginSchema] | None]] = []
        for from_id, _, _ in incoming:
            schema = self.get_effective_producer_schema(from_id)
            all_schemas.append((from_id, schema))

        # Reject mixed observed/explicit schemas (P2-2026-02-01 fix)
        observed_branches = [(nid, s) for nid, s in all_schemas if self._is_observed_schema(s)]
        explicit_branches = [(nid, s) for nid, s in all_schemas if not self._is_observed_schema(s)]

        if observed_branches and explicit_branches:
            observed_names = [nid for nid, _ in observed_branches]
            explicit_names = [f"{nid} ({s.__name__})" for nid, s in explicit_branches if s is not None]
            raise GraphValidationError(
                f"Coalesce '{coalesce_id}' has mixed observed/explicit schemas - "
                f"this is not allowed because observed branches may produce rows missing fields "
                f"expected by downstream consumers. "
                f"Observed branches: {observed_names}, explicit branches: {explicit_names}. "
                f"Fix: ensure all branches produce explicit schemas with compatible fields, "
                f"or all branches produce observed schemas."
            )

        # All explicit: verify structural compatibility across branches
        if len(explicit_branches) > 1:
            _first_id, first_schema = explicit_branches[0]
            for other_id, other_schema in explicit_branches[1:]:
                compatible, error_msg = self._schemas_structurally_compatible(first_schema, other_schema)
                if not compatible:
                    first_name = first_schema.__name__ if first_schema else "observed"
                    other_name = other_schema.__name__ if other_schema else "observed"
                    raise GraphValidationError(
                        f"Coalesce '{coalesce_id}' receives incompatible schemas from "
                        f"multiple branches: first branch has {first_name}, "
                        f"branch from '{other_id}' has {other_name}. {error_msg}"
                    )

    # ===== CONTRACT VALIDATION HELPERS =====

    def get_schema_config_from_node(self, node_id: str) -> SchemaConfig | None:
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
        # Handle raw dict form (schema_dict should always be a dict now)
        if isinstance(schema_dict, dict):
            return SchemaConfig.from_dict(schema_dict)

        return None

    def get_guaranteed_fields(self, node_id: str) -> frozenset[str]:
        """Get fields that a node guarantees in its output.

        Priority:
        1. Explicit guaranteed_fields in schema config
        2. Declared fields in flexible/fixed mode schemas
        3. Empty set for observed schemas

        Args:
            node_id: Node ID to get guarantees from

        Returns:
            Frozenset of field names the node guarantees to output
        """
        schema_config = self.get_schema_config_from_node(node_id)

        if schema_config is None:
            return frozenset()

        return schema_config.get_effective_guaranteed_fields()

    def get_required_fields(self, node_id: str) -> frozenset[str]:
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
        if node_info.node_type == NodeType.AGGREGATION:
            options = node_info.config["options"]
            if not isinstance(options, dict):
                raise GraphValidationError(f"Aggregation node config 'options' must be dict, got {type(options).__name__}")
            if "required_input_fields" in options:
                required_input = options["required_input_fields"]
                if required_input is not None and len(required_input) > 0:
                    return frozenset(required_input)

        # Check for explicit required_fields in schema config
        schema_config = self.get_schema_config_from_node(node_id)

        if schema_config is None:
            return frozenset()

        # Only return explicit required_fields declaration, NOT implicit from typed schemas
        if schema_config.required_fields is not None:
            return frozenset(schema_config.required_fields)

        return frozenset()

    def get_effective_guaranteed_fields(self, node_id: str) -> frozenset[str]:
        """Get effective output guarantees, walking through pass-through nodes.

        Gates inherit guarantees from upstream. Coalesce nodes are strategy-aware:
        - **union**: intersection of branch guarantees (only fields in ALL branches)
        - **nested**: the node's own guarantees (branch names, not inner fields)
        - **select**: the node's own guarantees (selected branch's schema)

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
        if node_info.node_type == NodeType.GATE:
            incoming = list(self._graph.in_edges(node_id, data=True))
            if not incoming:
                return frozenset()
            # Gates pass through - inherit from single upstream
            return self.get_effective_guaranteed_fields(incoming[0][0])

        # Coalesce nodes: strategy-aware guaranteed fields
        if node_info.node_type == NodeType.COALESCE:
            merge_strategy = node_info.config["merge"]

            # nested/select: use the node's own config schema (set by builder).
            # - Nested output is {branch_a: data, branch_b: data} — guarantees
            #   are the branch names, NOT the inner field names.
            # - Select output is the selected branch's data — its schema was
            #   copied by the builder into this node's config.
            if merge_strategy in ("nested", "select"):
                return self.get_guaranteed_fields(node_id)

            # union: intersection of branch guarantees. Only fields present in
            # ALL branches are guaranteed in the flat merged output.
            incoming = list(self._graph.in_edges(node_id, data=True))
            if not incoming:
                return frozenset()
            branch_guarantees = [self.get_effective_guaranteed_fields(from_id) for from_id, _, _ in incoming]
            if not branch_guarantees:
                return frozenset()
            result = branch_guarantees[0]
            for guarantees in branch_guarantees[1:]:
                result = result & guarantees
            return result

        # Non-pass-through nodes return their own guarantees
        return self.get_guaranteed_fields(node_id)
