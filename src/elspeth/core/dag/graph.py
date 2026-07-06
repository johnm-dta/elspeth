"""ExecutionGraph class — query, validation, and traversal operations.

Construction logic lives in builder.py; this module contains the graph
class with all runtime methods. The from_plugin_instances() classmethod
is a thin facade that delegates to builder.build_execution_graph().
Schema/contract validation policy lives in schema_validation.py,
coalesce_warnings.py, guarantees.py and schema_factory.py; the
corresponding ExecutionGraph methods are thin delegating facades
(elspeth-b2c6ab6db8).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import networkx as nx
from networkx import MultiDiGraph

from elspeth.contracts import (
    EdgeInfo,
    PluginSchema,
    RouteDestination,
    RoutingMode,
)
from elspeth.contracts.enums import NodeType
from elspeth.contracts.freeze import deep_freeze
from elspeth.contracts.schema import SchemaConfig, get_raw_schema_config
from elspeth.contracts.types import (
    AggregationName,
    BranchName,
    CoalesceName,
    GateName,
    NodeID,
    SinkName,
)
from elspeth.core.dag import coalesce_warnings, guarantees, schema_validation
from elspeth.core.dag.guarantees import EffectiveGuaranteeVote as _EffectiveGuaranteeVote
from elspeth.core.dag.models import (
    BranchInfo,
    GraphValidationError,
    GraphValidationWarning,
    NodeConfig,
    NodeInfo,
    _suggest_similar,
)
from elspeth.core.dag.schema_factory import (
    build_coalesce_schema as _build_coalesce_schema,  # noqa: F401 — compat re-export; tests import _build_coalesce_schema from graph
)

if TYPE_CHECKING:
    from elspeth.contracts import SinkProtocol, SourceProtocol, TransformProtocol
    from elspeth.core.config import (
        AggregationSettings,
        CoalesceSettings,
        GateSettings,
        QueueSettings,
        SourceSettings,
    )
    from elspeth.core.dag.models import WiredTransform


# Sentinel for the empty declared_required_fields default. Sharing a single
# frozenset instance is safe (frozensets are immutable) and avoids the bare
# `frozenset()` literal default-argument anti-pattern, which would be unsafe
# if the type were ever changed to a mutable container.
_EMPTY_DECLARED_REQUIRED_FIELDS: frozenset[str] = frozenset()


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
        self._branch_info: dict[BranchName, BranchInfo] = {}  # branch_name -> coalesce + gate info
        self._route_label_map: dict[tuple[NodeID, SinkName], str] = {}  # (gate_node, sink_name) -> route_label
        self._route_resolution_map: dict[tuple[NodeID, str], RouteDestination] = {}
        self._pipeline_nodes: list[NodeID] | None = None  # Ordered processing nodes (no source/sinks); None = not yet populated
        self._node_step_map: dict[NodeID, int] = {}  # node_id -> audit step (source=0)
        self._validation_warnings: tuple[GraphValidationWarning, ...] = ()
        self._build_metadata_frozen = False

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
        declared_required_fields: frozenset[str] = _EMPTY_DECLARED_REQUIRED_FIELDS,
        passes_through_input: bool = False,
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
            output_schema_config: Output schema config for contract validation.
                Parsed from config["schema"] or config["schema_config"]
                when not provided explicitly.
            declared_required_fields: For SINK nodes only — the set of fields the
                sink requires in its input rows. Populated by the builder from
                SinkProtocol.declared_required_fields. Empty frozenset otherwise.
            passes_through_input: For TRANSFORM nodes only — True iff the transform
                unconditionally emits rows containing every input field
                (ADR-007). Validator walk propagates predecessor guarantees
                through nodes where this is True. Must be False for non-TRANSFORM
                nodes; NodeInfo guards against misuse.
        """
        self._assert_build_metadata_mutable()
        resolved_config = config or {}

        # Populate output_schema_config from the raw config when the caller
        # doesn't provide it explicitly. The builder always passes it; this
        # keeps direct add_node() callers aligned with the same alias rules.
        if output_schema_config is None:
            try:
                output_schema_config = get_raw_schema_config(
                    resolved_config,
                    owner=f"node:{node_id}",
                )
            except ValueError as exc:
                raise GraphValidationError(
                    f"Invalid schema config: {exc}",
                    component_id=node_id,
                    component_type=node_type.value if isinstance(node_type, NodeType) else str(node_type),
                ) from exc

        info = NodeInfo(
            node_id=NodeID(node_id),
            node_type=node_type,
            plugin_name=plugin_name,
            config=resolved_config,
            input_schema=input_schema,
            output_schema=output_schema,
            input_schema_config=input_schema_config,
            output_schema_config=output_schema_config,
            declared_required_fields=declared_required_fields,
            passes_through_input=passes_through_input,
        )
        self._graph.add_node(node_id, info=info)

    def set_node_output_schema(self, node_id: str, schema: SchemaConfig) -> None:
        """Set a node's output schema during graph construction."""
        self._assert_build_metadata_mutable()
        info = self.get_node_info(node_id)
        self._graph.nodes[node_id]["info"] = replace(info, output_schema_config=schema)

    def finalize_node_configs(self) -> None:
        """Deep-freeze mutable node configs after construction is complete."""
        for node_id, attrs in self._graph.nodes(data=True):
            info = cast(NodeInfo, attrs["info"])
            if isinstance(info.config, dict):
                self._graph.nodes[node_id]["info"] = replace(info, config=deep_freeze(info.config))

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
        self._assert_build_metadata_mutable()
        # Use label as key to allow multiple edges between same nodes
        self._graph.add_edge(from_node, to_node, key=label, label=label, mode=mode)

    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic (a valid DAG)."""
        return nx.is_directed_acyclic_graph(self._graph)

    def validate(self) -> None:
        """Validate the execution graph structure.

        Validates:
        1. Graph is acyclic (no cycles)
        2. One or more source nodes exist
        3. At least one sink node exists
        4. All nodes are reachable from at least one source (no disconnected/orphaned nodes)
        5. Edge labels are unique per source node
        6. Every gate→sink MOVE edge has a corresponding route label entry
        7. Fan-in to ordinary executable nodes is explicit through QUEUE nodes

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
            except nx.NetworkXNoCycle as exc:
                raise GraphValidationError("Graph contains a cycle") from exc

        # Check for one or more sources
        # All nodes have "info" - added via add_node(), direct access is safe
        sources = [node_id for node_id, data in self._graph.nodes(data=True) if data["info"].node_type == NodeType.SOURCE]
        if not sources:
            raise GraphValidationError("Graph must have at least one source")

        # Check for at least one sink
        sinks = self.get_sinks()
        if len(sinks) < 1:
            raise GraphValidationError("Graph must have at least one sink")

        # Check for unreachable nodes (nodes not reachable from any source)
        reachable: set[str] = set()
        for source_id in sources:
            reachable.update(nx.descendants(self._graph, source_id))
            reachable.add(source_id)

        all_nodes = set(self._graph.nodes())
        unreachable = all_nodes - reachable

        if unreachable:
            # Build detailed error message with node types
            unreachable_details = [f"{node_id} ({self._graph.nodes[node_id]['info'].node_type})" for node_id in sorted(unreachable)]
            raise GraphValidationError(
                f"Graph validation failed: {len(unreachable)} unreachable node(s) detected:\n"
                f"  {', '.join(unreachable_details)}\n"
                f"All nodes must be reachable from at least one source node."
            )

        for node_id_str, node_attrs in self._graph.nodes(data=True):
            node_info = cast(NodeInfo, node_attrs["info"])
            # QUEUE and COALESCE are structural join primitives. SINK is a
            # terminal write boundary: ADR-025 Decision 9 allows direct
            # multi-source fan-in here, with ingest_sequence as the ordering
            # authority. Ordinary processing nodes must still route through
            # an explicit QUEUE.
            if node_info.node_type in {NodeType.QUEUE, NodeType.SINK, NodeType.COALESCE}:
                continue
            incoming_move_predecessors = {
                from_id
                for from_id, _to_id, _key, edge_data in self._graph.in_edges(node_id_str, keys=True, data=True)
                if edge_data["mode"] == RoutingMode.MOVE
            }
            if len(incoming_move_predecessors) > 1:
                raise GraphValidationError(
                    f"Node '{node_id_str}' has fan-in from multiple producers without a queue. "
                    "Route multiple producers through an explicit queue node before ordinary processing.",
                    component_id=node_id_str,
                    component_type=node_info.node_type.value,
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
                    node_info = cast(NodeInfo, self._graph.nodes[node_id]["info"])
                    raise GraphValidationError(
                        f"Node '{node_id}' has duplicate outgoing edge label '{edge_key}'. "
                        "Edge labels must be unique per source node to ensure correct "
                        "routing event recording.",
                        component_id=node_id,
                        component_type=node_info.node_type.value,
                    )
                labels_seen.add(edge_key)

        # Route label completeness: every gate→sink MOVE edge must have a
        # corresponding _route_label_map entry.  Missing entries mean the builder
        # created the edge but forgot to register the label — a construction bug
        # that would silently misrecord routing decisions in the audit trail.
        #
        # WHY MOVE-only:  Gates produce two kinds of edges to sinks:
        #   - MOVE edges: routing decisions (gate evaluates condition → routes row
        #     to a sink).  These MUST have route labels for audit recording.
        #   - COPY edges: fork paths (gate forks token → copies flow to parallel
        #     destinations).  These are NOT routing decisions — they're structural
        #     fan-out.  No route label needed.
        # The builder creates MOVE edges via routes={} and COPY edges via fork_to=[].
        # See builder.py lines ~293 (MOVE+label) vs ~414 (COPY, no label).
        #
        # WHY skip when _sink_id_map is empty:  Manual unit-test graphs for
        # isolated algorithms (cycle detection, topo sort) don't populate the
        # sink ID map.  This check only applies to builder-constructed graphs.
        if not self._sink_id_map:
            return
        for node_id_str in self._graph.nodes():
            node_info = cast(NodeInfo, self._graph.nodes[node_id_str]["info"])
            if node_info.node_type != NodeType.GATE:
                continue
            for _, to_id, _key, edge_data in self._graph.out_edges(node_id_str, keys=True, data=True):
                to_info = cast(NodeInfo, self._graph.nodes[to_id]["info"])
                if to_info.node_type != NodeType.SINK:
                    continue
                if edge_data["mode"] != RoutingMode.MOVE:
                    continue
                sink_name = next(
                    (name for name, nid in self._sink_id_map.items() if nid == NodeID(to_id)),
                    None,
                )
                if sink_name is None:
                    raise GraphValidationError(
                        f"Sink node '{to_id}' exists in the graph but is not registered "
                        "in the sink ID map. This indicates a graph construction bug.",
                        component_id=str(to_id),
                        component_type="sink",
                    )
                if (NodeID(node_id_str), sink_name) not in self._route_label_map:
                    raise GraphValidationError(
                        f"Gate '{node_id_str}' has a direct edge to sink node '{to_id}' "
                        f"(sink name '{sink_name}') but no registered route label. "
                        "This indicates a graph construction bug — every gate→sink edge "
                        "must have a corresponding route label entry.",
                        component_id=node_id_str,
                        component_type="gate",
                    )

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

    def topological_processing_order(self, processing_node_ids: Iterable[NodeID]) -> list[NodeID]:
        """Return processing nodes filtered from the graph's topological order."""
        processing_nodes = set(processing_node_ids)
        try:
            topo_order = [NodeID(raw_id) for raw_id in nx.topological_sort(self._graph)]
        except nx.NetworkXUnfeasible as unfeasible_exc:
            try:
                cycle = nx.find_cycle(self._graph)
                cycle_str = " -> ".join(f"{edge[0]}" for edge in cycle)
                raise GraphValidationError(f"Pipeline contains a cycle: {cycle_str}") from unfeasible_exc
            except nx.NetworkXNoCycle as exc:
                raise GraphValidationError("Pipeline contains a cycle") from exc
        return [node_id for node_id in topo_order if node_id in processing_nodes]

    def get_sources(self) -> list[NodeID]:
        """Get all source node IDs."""
        return [NodeID(node_id) for node_id, data in self._graph.nodes(data=True) if data["info"].node_type == NodeType.SOURCE]

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
                        "All declared route labels must resolve during graph construction.",
                        component_id=str(node_id),
                        component_type="gate",
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
            first_dup_node, _first_dup_desc = next(
                (node_id, desc) for name, node_id, desc in consumer_claims if name == duplicate_consumers[0]
            )
            raise GraphValidationError(
                f"Duplicate consumers for {len(duplicate_consumers)} connection(s): "
                + "; ".join(error_parts)
                + ". Use a gate for fan-out.",
                component_id=str(first_dup_node),
            )

        for connection_name in consumers:
            if connection_name not in producers:
                suggestions = _suggest_similar(connection_name, sorted(producers.keys()))
                hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                consumer_node = consumers[connection_name]
                raise GraphValidationError(
                    f"No producer for connection '{connection_name}'.{hint}\nAvailable connections: {', '.join(sorted(producers.keys()))}",
                    component_id=str(consumer_node),
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
            node_info = self.get_node_info(node_id)
            raise GraphValidationError(
                f"Node '{node_id}' has multiple continue MOVE edges to processing nodes: {sorted(next_nodes)}",
                component_id=str(node_id),
                component_type=node_info.node_type.value,
            )
        if len(next_nodes) == 1:
            return next_nodes[0]
        return None

    def get_pipeline_node_sequence(self) -> list[NodeID]:
        """Get ordered processing nodes in pipeline traversal order."""
        if self._pipeline_nodes is not None:
            return list(self._pipeline_nodes)

        sources = self.get_sources()
        if not sources:
            return []

        reachable: set[NodeID] = set()
        pending: list[NodeID] = []
        for source_id in sources:
            next_node = self.get_next_node(source_id)
            if next_node is not None:
                pending.append(next_node)
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
        """Build node -> audit step map (sources=0, processing nodes start at 1)."""
        source_ids = self.get_sources()

        step_map: dict[NodeID, int] = dict.fromkeys(source_ids, 0)
        for idx, node_id in enumerate(self.get_pipeline_node_sequence(), start=1):
            step_map[node_id] = idx

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
        *,
        sources: Mapping[str, SourceProtocol],
        source_settings_map: Mapping[str, SourceSettings],
        transforms: Sequence[WiredTransform] = (),
        sinks: Mapping[str, SinkProtocol] | None = None,
        aggregations: Mapping[str, tuple[TransformProtocol, AggregationSettings]] | None = None,
        gates: Sequence[GateSettings] = (),
        coalesce_settings: Sequence[CoalesceSettings] | None = None,
        queues: Mapping[str, QueueSettings] | None = None,
    ) -> ExecutionGraph:
        """Build ExecutionGraph from plugin instances.

        CORRECT method for graph construction - enables schema validation.
        Schemas extracted directly from instance attributes.

        Routing is explicit: terminal transforms and sources declare their
        output sink via on_success. There is no default_sink fallback.

        Per ADR-025 §2 the source surface is plural-only: callers pass
        ``sources`` and ``source_settings_map`` keyed by source name. The
        prior singular ``source=`` / ``source_settings=`` keyword
        arguments and their legacy single-source facade are deleted.

        Args:
            sources: Named source plugin instances (one or more per graph).
            source_settings_map: Source settings keyed by the same source names.
            transforms: Wired transforms (plugin instance + settings metadata)
            sinks: Dict of sink_name -> instantiated sink
            aggregations: Dict of agg_name -> (transform_instance, AggregationSettings)
            gates: Config-driven gate settings
            coalesce_settings: Coalesce configs for fork/join patterns
            queues: Declared pass-through scheduling queues

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
            sources=sources,
            source_settings_map=source_settings_map,
            transforms=transforms,
            sinks=sinks,
            aggregations=aggregations,
            gates=gates,
            coalesce_settings=coalesce_settings,
            queues=queues,
        )

    # ===== PUBLIC SETTERS (construction-time) =====

    def set_sink_id_map(self, mapping: dict[SinkName, NodeID]) -> None:
        """Set the sink_name -> node_id mapping."""
        self._assert_build_metadata_mutable()
        self._sink_id_map = dict(mapping)

    def set_transform_id_map(self, mapping: dict[int, NodeID]) -> None:
        """Set the transform sequence -> node_id mapping."""
        self._assert_build_metadata_mutable()
        self._transform_id_map = dict(mapping)

    def set_config_gate_id_map(self, mapping: dict[GateName, NodeID]) -> None:
        """Set the gate_name -> node_id mapping."""
        self._assert_build_metadata_mutable()
        self._config_gate_id_map = dict(mapping)

    def set_route_resolution_map(self, mapping: dict[tuple[NodeID, str], RouteDestination]) -> None:
        """Set the (gate_node_id, route_label) -> destination mapping."""
        self._assert_build_metadata_mutable()
        self._route_resolution_map = dict(mapping)

    def set_aggregation_id_map(self, mapping: dict[AggregationName, NodeID]) -> None:
        """Set the agg_name -> node_id mapping."""
        self._assert_build_metadata_mutable()
        self._aggregation_id_map = dict(mapping)

    def set_coalesce_id_map(self, mapping: dict[CoalesceName, NodeID]) -> None:
        """Set the coalesce_name -> node_id mapping."""
        self._assert_build_metadata_mutable()
        self._coalesce_id_map = dict(mapping)

    def set_branch_info(self, mapping: dict[BranchName, BranchInfo]) -> None:
        """Set the branch_name -> BranchInfo mapping (coalesce + gate)."""
        self._assert_build_metadata_mutable()
        self._branch_info = dict(mapping)

    def set_route_label_map(self, mapping: dict[tuple[NodeID, SinkName], str]) -> None:
        """Set the (gate_node, sink_name) -> route_label mapping."""
        self._assert_build_metadata_mutable()
        self._route_label_map = dict(mapping)

    def set_pipeline_nodes(self, nodes: list[NodeID]) -> None:
        """Set the ordered processing node sequence."""
        self._assert_build_metadata_mutable()
        self._pipeline_nodes = list(nodes)

    def set_node_step_map(self, mapping: dict[NodeID, int]) -> None:
        """Set the node_id -> audit step mapping."""
        self._assert_build_metadata_mutable()
        self._node_step_map = dict(mapping)

    def set_validation_warnings(self, warnings: Sequence[GraphValidationWarning]) -> None:
        """Set non-fatal graph construction warnings."""
        self._assert_build_metadata_mutable()
        self._validation_warnings = tuple(warnings)

    def add_route_resolution_entry(self, gate_id: NodeID, label: str, dest: RouteDestination) -> None:
        """Add a single entry to the route resolution map."""
        self._assert_build_metadata_mutable()
        self._route_resolution_map[(gate_id, label)] = dest

    def add_route_label_entry(self, gate_id: NodeID, sink_name: SinkName, label: str) -> None:
        """Add a single entry to the route label map."""
        self._assert_build_metadata_mutable()
        self._route_label_map[(gate_id, sink_name)] = label

    def _freeze_build_metadata(self) -> None:
        """Reject topology and metadata mutation after builder finalization."""
        self._build_metadata_frozen = True

    def _assert_build_metadata_mutable(self) -> None:
        if self._build_metadata_frozen:
            raise GraphValidationError(
                "ExecutionGraph build metadata is frozen after construction; rebuild the graph instead of mutating it."
            )

    # ===== PUBLIC GETTERS =====

    def get_sink_id_map(self) -> dict[SinkName, NodeID]:
        """Get explicit sink_name -> node_id mapping.

        Returns:
            Dict mapping each sink's logical name to its graph node ID.
            No substring matching required - use this for direct lookup.
        """
        return dict(self._sink_id_map)

    @property
    def validation_warnings(self) -> tuple[GraphValidationWarning, ...]:
        """Non-fatal graph construction warnings emitted during build."""
        return self._validation_warnings

    def get_transform_id_map(self) -> dict[int, NodeID]:
        """Get explicit sequence -> node_id mapping for transforms.

        Returns:
            Dict mapping transform sequence position (0-indexed) to node ID.
        """
        return dict(self._transform_id_map)

    def get_node_step_map(self) -> dict[NodeID, int]:
        """Get the builder-assigned node_id -> audit step mapping."""
        return dict(self._node_step_map)

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
        return {name: info.coalesce_name for name, info in self._branch_info.items()}

    def get_branch_info_map(self) -> dict[BranchName, BranchInfo]:
        """Get immutable branch routing plans keyed by branch name."""
        return dict(self._branch_info)

    def get_coalesce_branch_schemas(
        self,
        coalesce_name: CoalesceName,
    ) -> dict[str, SchemaConfig]:
        """Get per-branch schemas for a specific coalesce.

        Returns a mapping of branch name to the SchemaConfig that branch
        produces. This enables runtime tracking of which fields would have
        been contributed by a branch that was lost (diverted to error sink).

        Schema source: Branch schemas are computed by ``builder.py`` during
        DAG construction (see ``build_execution_graph`` coalesce loop around
        line 848) and stored in ``BranchInfo.schema``. This method reads from
        that authoritative source — it does not re-derive schemas from nodes.

        Args:
            coalesce_name: The coalesce to get branch schemas for.

        Returns:
            Dict mapping branch name (str) to SchemaConfig.
            Branches without schema info (observed mode or not in _branch_info)
            are omitted.
        """
        result: dict[str, SchemaConfig] = {}
        for branch_name, info in self._branch_info.items():
            if info.coalesce_name == coalesce_name and info.schema is not None:
                result[branch_name] = info.schema
        return result

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

        for branch_name, branch_info in self._branch_info.items():
            coalesce_nid = self._coalesce_id_map[branch_info.coalesce_name]

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
        fork_gate_nid = self._branch_info[BranchName(branch_name)].gate_node_id

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
            f"transform branches must have MOVE edge chains from gate to coalesce.",
            component_id=str(coalesce_nid),
            component_type="coalesce",
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
        return {name: info.gate_node_id for name, info in self._branch_info.items()}

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

    def get_route_label(self, from_node_id: str, sink_name: SinkName) -> str:
        """Get the route label for an edge from a node to a sink.

        Pure map lookup with "continue" fallback for indirect paths.
        Route-label completeness is enforced at construction time by validate();
        a missing entry here means the node reaches the sink via continue edges.

        Args:
            from_node_id: The originating node ID
            sink_name: The sink name

        Returns:
            The route label (e.g., "suspicious") or "continue" for default path.
        """
        key = (NodeID(from_node_id), sink_name)
        if key in self._route_label_map:
            return self._route_label_map[key]
        return "continue"

    def get_route_resolution_map(self) -> dict[tuple[NodeID, str], RouteDestination]:
        """Get the route resolution map for all gates.

        Returns:
            Dict mapping (gate_node_id, route_label) -> destination.
            Destination can be continue, fork, sink, or a processing node.
            Used by the executor to resolve route labels from gates.
        """
        return dict(self._route_resolution_map)

    # ===== SCHEMA/CONTRACT VALIDATION (delegating facade) =====
    # Implementations extracted to dag.schema_validation, dag.coalesce_warnings,
    # dag.guarantees and dag.schema_factory (elspeth-b2c6ab6db8). The delegators
    # below preserve the historical ExecutionGraph surface for callers and tests.

    def validate_edge_compatibility(self) -> None:
        """Validate schema compatibility for all edges in the graph.

        Delegates to ``dag.schema_validation.validate_edge_compatibility``
        (PHASE 2 cross-plugin validation: edges, coalesce branches, sink
        required fields).
        """
        schema_validation.validate_edge_compatibility(self)

    def warn_divert_coalesce_interactions(
        self,
        coalesce_configs: dict[NodeID, CoalesceSettings],
    ) -> list[GraphValidationWarning]:
        """Detect DIVERT edges that feed coalesces with audit/data implications.

        Delegates to ``dag.coalesce_warnings.warn_divert_coalesce_interactions``.
        """
        return coalesce_warnings.warn_divert_coalesce_interactions(self, coalesce_configs)

    def _validate_single_edge(
        self,
        from_node_id: str,
        to_node_id: str,
        *,
        _schema_cache: dict[str, type[PluginSchema] | None] | None = None,
    ) -> None:
        """Validate schema compatibility for a single edge.

        Delegates to ``dag.schema_validation.validate_single_edge``.
        """
        schema_validation.validate_single_edge(self, from_node_id, to_node_id, _schema_cache=_schema_cache)

    def get_effective_producer_schema(
        self,
        node_id: str,
        _cache: dict[str, type[PluginSchema] | None] | None = None,
    ) -> type[PluginSchema] | None:
        """Get effective output schema, walking through pass-through nodes.

        Delegates to ``dag.schema_validation.get_effective_producer_schema``.
        """
        return schema_validation.get_effective_producer_schema(self, node_id, _cache)

    def _validate_coalesce_compatibility(
        self,
        coalesce_id: str,
        *,
        _schema_cache: dict[str, type[PluginSchema] | None] | None = None,
    ) -> None:
        """Validate all inputs to coalesce node have compatible schemas.

        Delegates to ``dag.schema_validation.validate_coalesce_compatibility``.
        """
        schema_validation.validate_coalesce_compatibility(self, coalesce_id, _schema_cache=_schema_cache)

    def _validate_sink_required_fields(self) -> None:
        """Validate each sink's declared_required_fields against upstream guarantees.

        Delegates to ``dag.schema_validation.validate_sink_required_fields``.
        """
        schema_validation.validate_sink_required_fields(self)

    # ===== CONTRACT VALIDATION HELPERS =====

    def get_schema_config_from_node(self, node_id: str) -> SchemaConfig | None:
        """Extract SchemaConfig from node.

        Delegates to ``dag.guarantees.get_schema_config_from_node``.
        """
        return guarantees.get_schema_config_from_node(self, node_id)

    def get_guaranteed_fields(self, node_id: str) -> frozenset[str]:
        """Get fields that a node guarantees in its output.

        Delegates to ``dag.guarantees.get_guaranteed_fields``.
        """
        return guarantees.get_guaranteed_fields(self, node_id)

    def get_required_fields(self, node_id: str) -> frozenset[str]:
        """Get fields that a node EXPLICITLY requires in its input.

        Delegates to ``dag.guarantees.get_required_fields``.
        """
        return guarantees.get_required_fields(self, node_id)

    def get_effective_guaranteed_fields(self, node_id: str) -> frozenset[str]:
        """Get effective output guarantees for a node (propagation-aware, ADR-007).

        Delegates to ``dag.guarantees.get_effective_guaranteed_fields``.
        Callers that want the raw per-node declarations (without propagation)
        must call ``get_guaranteed_fields`` instead.
        """
        return guarantees.get_effective_guaranteed_fields(self, node_id)

    def _walk_effective_guarantee_vote(
        self,
        node_id: str,
        cache: dict[str, _EffectiveGuaranteeVote],
        field_cache: dict[str, frozenset[str]] | None = None,
    ) -> _EffectiveGuaranteeVote:
        """Recursive guarantee walk that preserves participation state.

        Delegates to ``dag.guarantees.walk_effective_guarantee_vote``.
        """
        return guarantees.walk_effective_guarantee_vote(self, node_id, cache, field_cache)

    def _walk_effective_guaranteed_fields(
        self,
        node_id: str,
        cache: dict[str, frozenset[str]],
    ) -> frozenset[str]:
        """Recursive implementation of get_effective_guaranteed_fields.

        Delegates to ``dag.guarantees.walk_effective_guaranteed_fields``.
        """
        return guarantees.walk_effective_guaranteed_fields(self, node_id, cache)
