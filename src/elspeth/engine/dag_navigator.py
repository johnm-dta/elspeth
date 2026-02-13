# src/elspeth/engine/dag_navigator.py
"""DAGNavigator: Pure topology queries for DAG traversal.

Extracted from RowProcessor to create a clean service boundary for
DAG navigation concerns. All methods are pure queries on immutable
topology data — no mutable state dependencies.

Used by:
- RowProcessor (work item creation, node resolution)
- Future: aggregation flush helpers (routing without RowProcessor coupling)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.core.config import GateSettings
from elspeth.plugins.protocols import TransformProtocol

if TYPE_CHECKING:
    from elspeth.contracts import TokenInfo
    from elspeth.engine.orchestrator.types import RowPlugin
    from elspeth.engine.processor import DAGTraversalContext


@dataclass(frozen=True)
class WorkItem:
    """Item in the work queue for DAG processing.

    Frozen to prevent post-construction mutation. Use DAGNavigator.create_work_item()
    factory method for construction with coalesce node resolution.
    """

    token: TokenInfo
    current_node_id: NodeID | None
    coalesce_node_id: NodeID | None = None
    coalesce_name: CoalesceName | None = None  # Name of the coalesce point (if any)
    on_success_sink: str | None = None  # Inherited sink for terminal children (deagg)

    def __post_init__(self) -> None:
        has_id = self.coalesce_node_id is not None
        has_name = self.coalesce_name is not None
        if has_id != has_name:
            raise OrchestrationInvariantError(
                f"WorkItem coalesce fields must be both set or both None: "
                f"coalesce_node_id={self.coalesce_node_id}, coalesce_name={self.coalesce_name}"
            )


class DAGNavigator:
    """Pure topology queries for DAG traversal.

    Resolves next-nodes, creates work items, and walks the DAG to find
    terminal sinks. All methods are pure queries on immutable data — no
    mutable state mutations.

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
        traversal: DAGTraversalContext,
        *,
        coalesce_on_success_map: Mapping[CoalesceName, str] | None = None,
        sink_names: frozenset[str] | None = None,
    ) -> DAGNavigator:
        """Create a DAGNavigator from a DAGTraversalContext plus supplementary params.

        Derives structural_node_ids and coalesce_name_by_node_id automatically.
        """
        coalesce_node_ids = dict(traversal.coalesce_node_map)
        node_to_plugin = dict(traversal.node_to_plugin)
        node_to_next = dict(traversal.node_to_next)

        structural_node_ids = frozenset(nid for nid in node_to_next if nid not in node_to_plugin)
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

    def create_work_item(
        self,
        *,
        token: TokenInfo,
        current_node_id: NodeID | None,
        coalesce_name: CoalesceName | None = None,
        coalesce_node_id: NodeID | None = None,
        on_success_sink: str | None = None,
    ) -> WorkItem:
        """Create node-id based work item."""
        resolved_coalesce_node_id = coalesce_node_id
        resolved_coalesce_name = coalesce_name

        # Resolve missing node id from name (existing behavior)
        if resolved_coalesce_node_id is None and resolved_coalesce_name is not None:
            resolved_coalesce_node_id = self._coalesce_node_ids[resolved_coalesce_name]
        # Resolve missing name from node id (symmetric resolution)
        elif resolved_coalesce_node_id is not None and resolved_coalesce_name is None:
            try:
                resolved_coalesce_name = self._coalesce_name_by_node_id[resolved_coalesce_node_id]
            except KeyError as exc:
                raise OrchestrationInvariantError(
                    f"Unknown coalesce node id '{resolved_coalesce_node_id}' — "
                    f"not in coalesce_name_by_node_id map. "
                    f"Known coalesce nodes: {sorted(self._coalesce_name_by_node_id.keys())}"
                ) from exc

        return WorkItem(
            token=token,
            current_node_id=current_node_id,
            coalesce_node_id=resolved_coalesce_node_id,
            coalesce_name=resolved_coalesce_name,
            on_success_sink=on_success_sink,
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

    def create_continuation_work_item(
        self,
        *,
        token: TokenInfo,
        current_node_id: NodeID,
        coalesce_name: CoalesceName | None = None,
        on_success_sink: str | None = None,
    ) -> WorkItem:
        """Create child work item that continues after current node or resumes at coalesce.

        For fork children (coalesce_name is set), routes the token to the first
        processing node for its branch:
        - Identity branches: first_node == coalesce_node_id (same as before)
        - Transform branches: first_node is the first transform in the chain
        """
        if coalesce_name is not None:
            coalesce_node_id = self._coalesce_node_ids[coalesce_name]
            # Direct access — all coalesce branches are populated in _branch_first_node.
            # branch_name is always set on fork children; None here is an invariant violation.
            branch_name = token.branch_name
            if branch_name is None:
                raise OrchestrationInvariantError(
                    f"Token '{token.token_id}' has coalesce_name='{coalesce_name}' but branch_name is None. "
                    "Fork children must have branch_name set."
                )
            first_node = self._branch_first_node[branch_name]
            return self.create_work_item(
                token=token,
                current_node_id=first_node,
                coalesce_name=coalesce_name,
                coalesce_node_id=coalesce_node_id,
                on_success_sink=on_success_sink,
            )

        return self.create_work_item(
            token=token,
            current_node_id=self.resolve_next_node(current_node_id),
            on_success_sink=on_success_sink,
        )
