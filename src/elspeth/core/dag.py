# src/elspeth/core/dag.py
"""DAG (Directed Acyclic Graph) operations for execution planning.

Uses NetworkX for graph operations including:
- Acyclicity validation
- Topological sorting
- Path finding for lineage queries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
from networkx import DiGraph

if TYPE_CHECKING:
    from elspeth.core.config import ElspethSettings


class GraphValidationError(Exception):
    """Raised when graph validation fails."""

    pass


@dataclass
class NodeInfo:
    """Information about a node in the execution graph."""

    node_id: str
    node_type: str  # source, transform, gate, aggregation, coalesce, sink
    plugin_name: str
    config: dict[str, Any] = field(default_factory=dict)


class ExecutionGraph:
    """Execution graph for pipeline configuration.

    Wraps NetworkX DiGraph with domain-specific operations.
    """

    def __init__(self) -> None:
        self._graph: DiGraph[str] = nx.DiGraph()

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

    def add_node(
        self,
        node_id: str,
        *,
        node_type: str,
        plugin_name: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Add a node to the execution graph."""
        info = NodeInfo(
            node_id=node_id,
            node_type=node_type,
            plugin_name=plugin_name,
            config=config or {},
        )
        self._graph.add_node(node_id, info=info)

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        *,
        label: str,
        mode: str = "move",
    ) -> None:
        """Add an edge between nodes.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            label: Edge label (e.g., "continue", "suspicious")
            mode: Routing mode ("move" or "copy")
        """
        self._graph.add_edge(from_node, to_node, label=label, mode=mode)

    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic (a valid DAG)."""
        return nx.is_directed_acyclic_graph(self._graph)

    def validate(self) -> None:
        """Validate the execution graph.

        Validates:
        1. Graph is acyclic (no cycles)
        2. Exactly one source node exists
        3. At least one sink node exists

        Raises:
            GraphValidationError: If validation fails
        """
        # Check for cycles
        if not self.is_acyclic():
            try:
                cycle = nx.find_cycle(self._graph)
                cycle_str = " -> ".join(f"{u}" for u, v in cycle)
                raise GraphValidationError(f"Graph contains a cycle: {cycle_str}")
            except nx.NetworkXNoCycle:
                raise GraphValidationError("Graph contains a cycle") from None

        # Check for exactly one source
        sources = [
            node_id
            for node_id, data in self._graph.nodes(data=True)
            if data.get("info") and data["info"].node_type == "source"
        ]
        if len(sources) != 1:
            raise GraphValidationError(
                f"Graph must have exactly one source, found {len(sources)}"
            )

        # Check for at least one sink
        sinks = self.get_sinks()
        if len(sinks) < 1:
            raise GraphValidationError("Graph must have at least one sink")

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

    def get_source(self) -> str | None:
        """Get the source node ID.

        Returns:
            The source node ID, or None if not exactly one source exists.
        """
        sources = [
            node_id
            for node_id, data in self._graph.nodes(data=True)
            if data.get("info") and data["info"].node_type == "source"
        ]
        return sources[0] if len(sources) == 1 else None

    def get_sinks(self) -> list[str]:
        """Get all sink node IDs.

        Returns:
            List of sink node IDs.
        """
        return [
            node_id
            for node_id, data in self._graph.nodes(data=True)
            if data.get("info") and data["info"].node_type == "sink"
        ]

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

    def get_edges(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Get all edges with their data.

        Returns:
            List of (from_node, to_node, edge_data) tuples
        """
        return [
            (u, v, dict(data))
            for u, v, data in self._graph.edges(data=True)
        ]

    @classmethod
    def from_config(cls, config: ElspethSettings) -> ExecutionGraph:
        """Build an ExecutionGraph from validated settings.

        Creates nodes for:
        - Source (from config.datasource)
        - Transforms (from config.row_plugins, in order)
        - Sinks (from config.sinks)

        Creates edges for:
        - Linear flow: source -> transforms -> output_sink

        Args:
            config: Validated ElspethSettings

        Returns:
            ExecutionGraph ready for validation and execution
        """
        import uuid

        graph = cls()

        def node_id(prefix: str, name: str) -> str:
            return f"{prefix}_{name}_{uuid.uuid4().hex[:8]}"

        # Add source node
        source_id = node_id("source", config.datasource.plugin)
        graph.add_node(
            source_id,
            node_type="source",
            plugin_name=config.datasource.plugin,
            config=config.datasource.options,
        )

        # Add sink nodes
        sink_ids: dict[str, str] = {}
        for sink_name, sink_config in config.sinks.items():
            sid = node_id("sink", sink_name)
            sink_ids[sink_name] = sid
            graph.add_node(
                sid,
                node_type="sink",
                plugin_name=sink_config.plugin,
                config=sink_config.options,
            )

        # Build transform chain
        prev_node_id = source_id
        for plugin_config in config.row_plugins:
            is_gate = plugin_config.type == "gate"
            ntype = "gate" if is_gate else "transform"
            tid = node_id(ntype, plugin_config.plugin)

            graph.add_node(
                tid,
                node_type=ntype,
                plugin_name=plugin_config.plugin,
                config=plugin_config.options,
            )

            # Edge from previous node
            graph.add_edge(prev_node_id, tid, label="continue", mode="move")
            prev_node_id = tid

        # Edge from last transform (or source) to output sink
        graph.add_edge(
            prev_node_id,
            sink_ids[config.output_sink],
            label="continue",
            mode="move",
        )

        return graph
