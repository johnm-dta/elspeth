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

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.types import GateName, NodeID, SinkName
from elspeth.engine.processor import DAGTraversalContext

if TYPE_CHECKING:
    from elspeth.contracts import SinkProtocol, SourceProtocol
    from elspeth.core.config import GateSettings
    from elspeth.core.dag import ExecutionGraph
    from elspeth.engine.orchestrator.types import PipelineConfig, RowPlugin


def assign_plugin_node_ids(
    sources: Mapping[str, SourceProtocol],
    transforms: Sequence[RowPlugin],
    sinks: Mapping[str, SinkProtocol],
    source_id_map: Mapping[str, NodeID],
    transform_id_map: Mapping[int, NodeID],
    sink_id_map: Mapping[SinkName, NodeID],
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

    # Set node_id on transforms
    # Note: Aggregation transforms already have node_id set by CLI (mapped from
    # aggregation_id_map), so only assign for transforms without node_id.
    for seq, transform in enumerate(transforms):
        if transform.node_id is not None:
            # Already has node_id (e.g., aggregation transform) - skip
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

    return DAGTraversalContext(
        node_step_map=node_step_map,
        node_to_plugin=node_to_plugin,
        first_transform_node_id=graph.get_first_transform_node(),
        node_to_next=node_to_next,
        coalesce_node_map=graph.get_coalesce_id_map(),
        branch_first_node=graph.get_branch_first_nodes(),
    )
