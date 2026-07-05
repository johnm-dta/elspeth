"""Graph/plugin wiring setup functions for the orchestrator.

This module contains pure functions for:
- Assigning ``node_id`` to every plugin instance from the compiled graph's
  ID maps (part of the plugin protocol contract — the orchestrator populates
  ``node_id`` after registering each node)
- Building the :class:`DAGTraversalContext` that the RowProcessor consumes,
  from the execution graph plus the pipeline config

All functions operate on external state passed via parameters - they don't
maintain internal state. This follows the same pattern as aggregation.py and
outcomes.py: pure delegation targets for the Orchestrator.

These functions were extracted from ``Orchestrator`` (where they lived as
``_assign_plugin_node_ids`` and ``_build_dag_traversal_context``) to shrink
``core.py`` and to make the wiring logic independently testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from typing import TYPE_CHECKING

from elspeth.contracts.enums import NodeType
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.types import GateName, NodeID, SinkName
from elspeth.engine.processor import DAGTraversalContext

# Node types that legitimately appear in traversal with no plugin to execute.
# Closed vocabulary — extend deliberately, never derive by complement.
_STRUCTURAL_NODE_TYPES = frozenset({NodeType.SOURCE, NodeType.QUEUE, NodeType.COALESCE})

if TYPE_CHECKING:
    from elspeth.contracts import SinkProtocol, SourceProtocol
    from elspeth.core.config import GateSettings
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.landscape.data_flow_repository import DataFlowRepository
    from elspeth.engine.orchestrator.plugin_types import RowPlugin
    from elspeth.engine.orchestrator.types import PipelineConfig


def assign_plugin_node_ids(
    sources: Mapping[str, SourceProtocol],
    transforms: Sequence[RowPlugin],
    sinks: Mapping[str, SinkProtocol],
    source_id_map: Mapping[str, NodeID],
    transform_id_map: Mapping[int, NodeID],
    sink_id_map: Mapping[SinkName, NodeID],
    aggregation_node_ids: Set[NodeID] = frozenset(),
) -> None:
    """Explicitly assign node_id to all plugins with validation.

    This is part of the plugin protocol contract - all plugins define
    node_id: str | None and the orchestrator populates it.

    Args:
        sources: Source plugin instances keyed by source name
        transforms: List of transform plugins
        sinks: Dict of sink_name -> sink plugin
        source_id_map: Source name -> source node ID
        transform_id_map: Maps transform sequence -> node_id
        sink_id_map: Maps sink_name -> node_id
        aggregation_node_ids: Graph aggregation node IDs whose transforms are
            legitimately pre-populated before assignment

    Raises:
        OrchestrationInvariantError: If transform/sink not in ID map
    """
    # Set node_id on sources
    for source_name, source in sources.items():
        if source_name not in source_id_map:
            raise OrchestrationInvariantError(
                f"Source '{source_name}' not found in graph source map. Graph has mappings for sources: {list(source_id_map.keys())}"
            )
        source.node_id = source_id_map[source_name]

    # Set node_id on transforms.
    # Aggregation transforms already have node_id set by CLI (mapped from the
    # graph's aggregation_id_map). Only those graph-confirmed aggregation IDs
    # may skip sequence-based assignment; stale regular transform IDs fail closed.
    for seq, transform in enumerate(transforms):
        if transform.node_id is not None:
            existing_node_id = NodeID(transform.node_id)
            if existing_node_id in aggregation_node_ids:
                continue
            if seq not in transform_id_map:
                raise OrchestrationInvariantError(
                    f"Transform at sequence {seq} has pre-set node_id {transform.node_id!r}, "
                    "but that sequence is not in the graph transform map and the node_id is not a graph aggregation node. "
                    f"Graph has transform sequences: {list(transform_id_map.keys())}; "
                    f"aggregation node IDs: {sorted(aggregation_node_ids)}."
                )
            expected_node_id = transform_id_map[seq]
            if existing_node_id != expected_node_id:
                raise OrchestrationInvariantError(
                    f"Transform at sequence {seq} has pre-set node_id {transform.node_id!r}, "
                    f"but the graph maps that sequence to {expected_node_id!r}. "
                    "Only graph aggregation transforms may keep pre-set node IDs."
                )
            continue
        if seq not in transform_id_map:
            raise OrchestrationInvariantError(
                f"Transform at sequence {seq} not found in graph. Graph has mappings for sequences: {list(transform_id_map.keys())}"
            )
        transform.node_id = transform_id_map[seq]

    # Set node_id on sinks
    # Note: Sinks not in graph are skipped (e.g., export sinks used post-run)
    for sink_name, sink in sinks.items():
        sink_name_typed = SinkName(sink_name)
        if sink_name_typed not in sink_id_map:
            # Sink not in execution graph - skip silently
            # This happens for post-run sinks (e.g., landscape.export.sink)
            continue
        sink.node_id = sink_id_map[sink_name_typed]


def build_source_id_map(graph: ExecutionGraph) -> dict[str, NodeID]:
    """Source-name -> source-node-id map from the graph's source nodes.

    ONE loader for every processor-assembly seam — leader
    (``Orchestrator._register_graph_nodes_and_edges``), resume
    (``setup_resume_context``), and follower (``build_follower_processor``)
    — which previously each hand-rolled this loop (elspeth-07b2031e41).

    Per ADR-025 §2 the DAG builder unconditionally sets ``source_name`` on
    every source node; a missing key would collide entries across multiple
    sources, silently overwriting earlier ones, so it fails closed.
    """
    source_id_map: dict[str, NodeID] = {}
    for candidate_source_id in graph.get_sources():
        source_info = graph.get_node_info(candidate_source_id)
        if "source_name" not in source_info.config:
            raise OrchestrationInvariantError(
                f"DAG source node {candidate_source_id!r} is missing 'source_name' in its config. "
                f"Per ADR-025 §2 the DAG builder MUST set source_name on every source node. "
                f"This is a graph-construction bug — node config keys: {sorted(source_info.config.keys())}."
            )
        source_id_map[str(source_info.config["source_name"])] = candidate_source_id
    return source_id_map


def load_edge_map(data_flow: DataFlowRepository, run_id: str) -> dict[tuple[NodeID, str], str]:
    """Load the run's registered edge map from Landscape, keyed for RowProcessor.

    The leader registered nodes and edges at run start; resume and follower
    load the REAL edge ids (FK integrity for routing events) and rekey from
    the DB's ``(str, str)`` to RowProcessor's ``(NodeID, label)``.
    """
    raw_edge_map = data_flow.get_edge_map(run_id)
    return {(NodeID(key[0]), key[1]): edge_id for key, edge_id in raw_edge_map.items()}


def build_dag_traversal_context(
    graph: ExecutionGraph,
    config: PipelineConfig,
    config_gate_id_map: dict[GateName, NodeID],
) -> DAGTraversalContext:
    """Build traversal context for RowProcessor from graph + pipeline config."""
    node_step_map = graph.build_step_map()
    node_to_plugin: dict[NodeID, RowPlugin | GateSettings] = {}

    for transform in config.transforms:
        node_id_raw = transform.node_id
        if node_id_raw is None:
            raise OrchestrationInvariantError(f"Transform '{transform.name}' missing node_id for traversal context")
        node_to_plugin[NodeID(node_id_raw)] = transform

    for gate in config.gates:
        gate_node_id = config_gate_id_map[GateName(gate.name)]
        node_to_plugin[gate_node_id] = gate

    node_to_next: dict[NodeID, NodeID | None] = {}
    for source_id in graph.get_sources():
        node_to_next[source_id] = graph.get_next_node(source_id)
    for node_id in graph.get_pipeline_node_sequence():
        node_to_next[node_id] = graph.get_next_node(node_id)
    for coalesce_node_id in graph.get_coalesce_id_map().values():
        node_to_next[coalesce_node_id] = graph.get_next_node(coalesce_node_id)

    # Structural nodes are the explicit closed set of node types that carry no
    # plugin: sources (traversal origins), queues (ADR-025 fan-in primitives),
    # and coalesce points. Anything else in node_to_next must be plugin-bearing
    # — a transform/gate missing from node_to_plugin means the graph and the
    # config have drifted, and skipping it would bypass whatever that node
    # enforced (elspeth-c522931bd1).
    structural_node_ids = frozenset(node_id for node_id in node_to_next if graph.get_node_info(node_id).node_type in _STRUCTURAL_NODE_TYPES)
    unaccounted = sorted(node_id for node_id in node_to_next if node_id not in node_to_plugin and node_id not in structural_node_ids)
    if unaccounted:
        raise OrchestrationInvariantError(
            f"DAG traversal contains node(s) with neither a plugin mapping nor a structural role: {unaccounted}. "
            "Every traversal node must be plugin-bearing (transform/gate) or structural (source/queue/coalesce). "
            "This indicates graph/config construction drift."
        )

    return DAGTraversalContext(
        node_step_map=node_step_map,
        node_to_plugin=node_to_plugin,
        node_to_next=node_to_next,
        coalesce_node_map=graph.get_coalesce_id_map(),
        branch_first_node=graph.get_branch_first_nodes(),
        structural_node_ids=structural_node_ids,
    )
