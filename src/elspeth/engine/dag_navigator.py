"""DAGNavigator: Pure topology queries for DAG traversal.

Extracted from RowProcessor to create a clean service boundary for
DAG navigation concerns. All methods are pure queries on immutable
topology data — no mutable state dependencies.

Used by:
- RowProcessor (node and terminal route resolution)
- Future: aggregation flush helpers (routing without RowProcessor coupling)
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol

from elspeth.contracts import TransformProtocol
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.core.config import GateSettings
from elspeth.engine.orchestrator.plugin_types import RowPlugin


class DAGTraversalSnapshot(Protocol):
    """Traversal fields consumed by DAGNavigator.from_traversal_context()."""

    @property
    def coalesce_node_map(self) -> Mapping[CoalesceName, NodeID]: ...

    @property
    def node_to_plugin(self) -> Mapping[NodeID, RowPlugin | GateSettings]: ...

    @property
    def node_to_next(self) -> Mapping[NodeID, NodeID | None]: ...

    @property
    def branch_first_node(self) -> Mapping[str, NodeID]: ...

    @property
    def structural_node_ids(self) -> frozenset[NodeID]: ...


class DAGNavigator:
    """Pure topology queries for DAG traversal.

    Resolves next-nodes, coalesce identifiers, branch starts, and terminal
    sinks. All methods are pure queries on immutable data — no mutable state
    mutations.

    Constructed from a DAGTraversalContext (built by orchestrator) plus
    supplementary routing data from RowProcessor's constructor params.
    """

    def __init__(
        self,
        *,
        node_to_plugin: Mapping[NodeID, RowPlugin | GateSettings],
        node_to_next: Mapping[NodeID, NodeID | None],
        coalesce_node_ids: Mapping[CoalesceName, NodeID],
        structural_node_ids: frozenset[NodeID],
        coalesce_name_by_node_id: Mapping[NodeID, CoalesceName],
        coalesce_on_success_map: Mapping[CoalesceName, str],
        sink_names: frozenset[str],
        branch_first_node: Mapping[str, NodeID] | None = None,
    ) -> None:
        # Wrap all mappings in MappingProxyType for true immutability
        self._node_to_plugin: Mapping[NodeID, RowPlugin | GateSettings] = MappingProxyType(dict(node_to_plugin))
        self._node_to_next: Mapping[NodeID, NodeID | None] = MappingProxyType(dict(node_to_next))
        self._coalesce_node_ids: Mapping[CoalesceName, NodeID] = MappingProxyType(dict(coalesce_node_ids))
        self._structural_node_ids = structural_node_ids
        self._coalesce_name_by_node_id: Mapping[NodeID, CoalesceName] = MappingProxyType(dict(coalesce_name_by_node_id))
        self._coalesce_on_success_map: Mapping[CoalesceName, str] = MappingProxyType(dict(coalesce_on_success_map))
        self._sink_names = sink_names
        self._branch_first_node: Mapping[str, NodeID] = MappingProxyType(dict(branch_first_node or {}))

    @classmethod
    def from_traversal_context(
        cls,
        traversal: DAGTraversalSnapshot,
        *,
        coalesce_on_success_map: Mapping[CoalesceName, str] | None = None,
        sink_names: frozenset[str] | None = None,
    ) -> DAGNavigator:
        """Create a DAGNavigator from a DAGTraversalContext plus supplementary params.

        Consumes the context's explicit structural_node_ids allowlist —
        never the complement of node_to_plugin, which silently classified
        unmapped plugin nodes as skippable (elspeth-c522931bd1) — and
        derives coalesce_name_by_node_id automatically.
        """
        coalesce_node_ids = dict(traversal.coalesce_node_map)
        node_to_plugin = dict(traversal.node_to_plugin)
        node_to_next = dict(traversal.node_to_next)

        # Coalesce nodes are structural by definition; the union keeps that
        # invariant even for snapshot implementations that omit them.
        structural_node_ids = frozenset(traversal.structural_node_ids) | frozenset(coalesce_node_ids.values())
        coalesce_name_by_node_id = {node_id: coalesce_name for coalesce_name, node_id in coalesce_node_ids.items()}

        return cls(
            node_to_plugin=node_to_plugin,
            node_to_next=node_to_next,
            coalesce_node_ids=coalesce_node_ids,
            structural_node_ids=structural_node_ids,
            coalesce_name_by_node_id=coalesce_name_by_node_id,
            coalesce_on_success_map=coalesce_on_success_map or {},
            sink_names=sink_names or frozenset(),
            branch_first_node=dict(traversal.branch_first_node),
        )

    def resolve_plugin_for_node(self, node_id: NodeID) -> TransformProtocol | GateSettings | None:
        """Resolve the plugin/gate associated with a processing node.

        Returns None for structural nodes (e.g. coalesce points) that exist in
        the DAG traversal but have no plugin to execute. The caller skips these
        nodes and continues to the next processing node.

        Raises OrchestrationInvariantError for unknown nodes that are neither
        plugin-bearing nor structural — this would indicate a graph construction bug.
        """
        if node_id in self._node_to_plugin:
            return self._node_to_plugin[node_id]
        if node_id in self._structural_node_ids:
            return None
        raise OrchestrationInvariantError(
            f"Node ID '{node_id}' is neither a plugin node nor a known structural node (coalesce). "
            f"Plugin nodes: {sorted(self._node_to_plugin.keys())}, "
            f"structural nodes: {sorted(self._structural_node_ids)}"
        )

    def resolve_next_node(self, node_id: NodeID) -> NodeID | None:
        """Resolve the next processing node from traversal metadata."""
        if node_id not in self._node_to_next:
            raise OrchestrationInvariantError(
                f"Node ID '{node_id}' missing from traversal next-node map (terminal nodes must have explicit None entries)"
            )
        return self._node_to_next[node_id]

    def resolve_coalesce_sink(self, coalesce_name: CoalesceName, *, context: str) -> str:
        """Resolve terminal sink for coalesce outcomes with invariant validation."""
        if coalesce_name not in self._coalesce_on_success_map:
            raise OrchestrationInvariantError(
                f"Coalesce '{coalesce_name}' not in on_success map. "
                f"Available: {sorted(self._coalesce_on_success_map.keys())}. "
                f"Context: {context}"
            )
        return self._coalesce_on_success_map[coalesce_name]

    def resolve_coalesce_node(self, coalesce_name: CoalesceName) -> NodeID:
        """Resolve a coalesce node id from its configured coalesce name."""
        try:
            return self._coalesce_node_ids[coalesce_name]
        except KeyError as exc:
            raise OrchestrationInvariantError(
                f"Unknown coalesce name '{coalesce_name}' — "
                f"not in coalesce_node_ids map. "
                f"Known coalesce names: {sorted(self._coalesce_node_ids.keys())}"
            ) from exc

    def resolve_coalesce_name(self, coalesce_node_id: NodeID) -> CoalesceName:
        """Resolve a coalesce name from its structural node id."""
        try:
            return self._coalesce_name_by_node_id[coalesce_node_id]
        except KeyError as exc:
            raise OrchestrationInvariantError(
                f"Unknown coalesce node id '{coalesce_node_id}' — "
                f"not in coalesce_name_by_node_id map. "
                f"Known coalesce nodes: {sorted(self._coalesce_name_by_node_id.keys())}"
            ) from exc

    def resolve_jump_target_sink(self, start_node_id: NodeID) -> str | None:
        """Resolve terminal on_success sink reachable from a route jump target.

        Returns None when the jump target contains a gate that will self-route
        at execution time (gates determine sink destinations dynamically via
        their routes config, so no static on_success resolution is needed).
        """
        node_id: NodeID | None = start_node_id
        resolved_sink: str | None = None
        encountered_gate = False
        iterations = 0
        max_iterations = len(self._node_to_next) + 1

        while node_id is not None:
            iterations += 1
            if iterations > max_iterations:
                raise OrchestrationInvariantError(
                    f"Jump-target sink resolution exceeded {max_iterations} iterations from node '{start_node_id}'. "
                    "Possible cycle in traversal map."
                )

            plugin = self.resolve_plugin_for_node(node_id)
            if isinstance(plugin, GateSettings):
                encountered_gate = True
            elif isinstance(plugin, TransformProtocol) and plugin.on_success is not None:
                candidate_sink = plugin.on_success
                if not self._sink_names or candidate_sink in self._sink_names:
                    resolved_sink = candidate_sink

            next_node_id = self.resolve_next_node(node_id)
            if next_node_id is None and node_id in self._coalesce_name_by_node_id:
                coalesce_name = self._coalesce_name_by_node_id[node_id]
                resolved_sink = self.resolve_coalesce_sink(
                    coalesce_name,
                    context=f"walk started at node '{start_node_id}'",
                )

            node_id = next_node_id

        if resolved_sink is None and not encountered_gate:
            raise OrchestrationInvariantError(
                f"Jump-target sink resolution reached terminal path with no sink from node '{start_node_id}'. "
                "A gate route jump must resolve to a terminal sink to avoid stale routing state."
            )

        if resolved_sink is not None and self._sink_names and resolved_sink not in self._sink_names:
            raise OrchestrationInvariantError(
                f"Jump-target sink resolution returned '{resolved_sink}' which is not a configured sink. "
                f"Available sinks: {sorted(self._sink_names)}. Walk started at node '{start_node_id}'."
            )
        return resolved_sink

    def resolve_branch_first_node(self, branch_name: str) -> NodeID:
        """First processing node for a fork branch routed to a coalesce.

        Exposes the _branch_first_node lookup for fresh fork children and the
        resume path (RowProcessor.resume_incomplete_token).

        _branch_first_node covers all coalesce-bound branches (built by
        ExecutionGraph.get_branch_first_nodes). Callers must only invoke this for
        branches that are in _branch_to_coalesce; calling it for a fork→sink branch
        will raise because those branches are not in the map.

        Raises:
            OrchestrationInvariantError: If branch_name is not in the branch_first_node
                map. This indicates a logic error in the caller: only coalesce-bound
                branches are registered, not fork→sink branches.
        """
        try:
            return self._branch_first_node[branch_name]
        except KeyError as exc:
            raise OrchestrationInvariantError(
                f"Unknown branch name '{branch_name}' — not in branch_first_node map. "
                f"Only coalesce-bound branches are registered here; fork→sink branches "
                f"are not. Known: {sorted(self._branch_first_node.keys())}"
            ) from exc
