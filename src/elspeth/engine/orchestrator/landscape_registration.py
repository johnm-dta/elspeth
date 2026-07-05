"""Landscape run-setup registration/recording functions for the orchestrator.

This module contains pure functions that write run-setup records to the
Landscape audit trail:
- register_nodes with the data-flow repository for every node in the execution
  graph (resolving plugin metadata, schema config, output contract per node)
- record_schema_contract: persist the source schema contract at source-node
  level and expose it to transforms via the plugin context

All functions operate on external state passed via parameters - they don't
maintain internal state. This follows the same pattern as aggregation.py and
outcomes.py: pure delegation targets for the Orchestrator. (Moving them does not
change audit ordering - the bodies are unchanged.)

These functions were extracted from ``Orchestrator`` (where they lived as
``_register_nodes_with_landscape`` and ``_record_schema_contract``) to shrink
``core.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from elspeth import __version__ as ENGINE_VERSION
from elspeth.contracts import Determinism, NodeType
from elspeth.contracts.errors import FrameworkBugError, OrchestrationInvariantError
from elspeth.contracts.types import NodeID, SinkName

if TYPE_CHECKING:
    from elspeth.contracts import SourceProtocol
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.landscape.factory import RecorderFactory
    from elspeth.engine.orchestrator.types import PipelineConfig


@dataclass(frozen=True, slots=True)
class NodeAuditMetadata:
    """Audit metadata Landscape stores for an execution-graph node."""

    plugin_version: str
    determinism: Determinism
    source_file_hash: str | None


class _AuditMetadataPlugin(Protocol):
    """Plugin attributes required for node audit registration."""

    @property
    def plugin_version(self) -> str: ...

    @property
    def determinism(self) -> Determinism: ...

    @property
    def source_file_hash(self) -> str | None: ...


_ENGINE_NODE_AUDIT_METADATA = NodeAuditMetadata(
    plugin_version=f"engine:{ENGINE_VERSION}",
    determinism=Determinism.DETERMINISTIC,
    source_file_hash=None,
)


def resolve_node_audit_metadata(
    config: PipelineConfig,
    graph: ExecutionGraph,
    *,
    source_id_map: Mapping[str, NodeID],
    transform_id_map: Mapping[int, NodeID],
    sink_id_map: Mapping[SinkName, NodeID],
    config_gate_node_ids: set[NodeID],
    aggregation_node_ids: set[NodeID],
    coalesce_node_ids: set[NodeID],
) -> dict[NodeID, NodeAuditMetadata]:
    """Resolve audit metadata for every graph node before Landscape writes."""

    plugin_by_node: dict[NodeID, _AuditMetadataPlugin] = {}
    for source_name, source_node_id in source_id_map.items():
        plugin_by_node[source_node_id] = config.sources[source_name]

    for seq, transform in enumerate(config.transforms):
        if seq in transform_id_map:
            plugin_by_node[transform_id_map[seq]] = transform
        elif transform.node_id is not None and NodeID(transform.node_id) in aggregation_node_ids:
            plugin_by_node[NodeID(transform.node_id)] = transform

    for sink_name, sink in config.sinks.items():
        if SinkName(sink_name) in sink_id_map:
            plugin_by_node[sink_id_map[SinkName(sink_name)]] = sink

    metadata_by_node: dict[NodeID, NodeAuditMetadata] = {}
    for raw_node_id in graph.topological_order():
        node_id = NodeID(raw_node_id)
        node_info = graph.get_node_info(raw_node_id)
        if node_id in config_gate_node_ids or node_id in coalesce_node_ids or node_info.node_type == NodeType.QUEUE:
            metadata_by_node[node_id] = _ENGINE_NODE_AUDIT_METADATA
            continue

        plugin = plugin_by_node.get(node_id)
        if plugin is None:
            raise OrchestrationInvariantError(
                f"Node {raw_node_id!r} ({node_info.node_type.value}/{node_info.plugin_name}) requires plugin-backed audit metadata, "
                "but no plugin instance was resolved from the graph ID maps."
            )
        metadata_by_node[node_id] = NodeAuditMetadata(
            plugin_version=plugin.plugin_version,
            determinism=plugin.determinism,
            source_file_hash=plugin.source_file_hash,
        )

    return metadata_by_node


def register_nodes_with_landscape(
    factory: RecorderFactory,
    run_id: str,
    config: PipelineConfig,
    graph: ExecutionGraph,
    execution_order: list[str],
    audit_metadata_by_node: Mapping[NodeID, NodeAuditMetadata],
) -> None:
    """Register each node in the execution graph with Landscape.

    Iterates the topological execution order, resolves plugin metadata
    (version, determinism), schema config, and output contract for each node,
    then calls factory.data_flow.register_node().

    Args:
        factory: RecorderFactory for audit trail.
        run_id: Run identifier.
        config: Pipeline configuration (for source contract).
        graph: Execution graph (for node info lookup).
        execution_order: Topological ordering of node IDs.
        audit_metadata_by_node: Pre-resolved plugin/engine metadata by node ID.
    """

    for node_id in execution_order:
        node_info = graph.get_node_info(node_id)
        try:
            audit_metadata = audit_metadata_by_node[NodeID(node_id)]
        except KeyError as exc:
            raise FrameworkBugError(f"Node '{node_id}' has no resolved audit metadata.") from exc

        # Schema config is always available via output_schema_config —
        # populated at construction time for all node types.
        schema_config = node_info.output_schema_config
        if schema_config is None:
            raise FrameworkBugError(
                f"Node '{node_id}' has no output_schema_config. "
                "All nodes in execution order must have schema config "
                "populated by the builder."
            )

        # Get output_contract for source nodes
        # Sources have get_schema_contract() method that returns their output contract
        output_contract = None
        if node_info.node_type == NodeType.SOURCE:
            if "source_name" not in node_info.config:
                raise OrchestrationInvariantError(
                    f"DAG source node '{node_info.node_id}' is missing 'source_name' in its config. "
                    f"Per ADR-025 §2 the DAG builder MUST set source_name on every source node. "
                    f"This is a graph-construction bug — node config keys: {sorted(node_info.config.keys())}."
                )
            source_name = str(node_info.config["source_name"])
            output_contract = config.sources[source_name].get_schema_contract()

        factory.data_flow.register_node(
            run_id=run_id,
            node_id=node_id,
            plugin_name=node_info.plugin_name,
            node_type=NodeType(node_info.node_type),  # Already lowercase
            plugin_version=audit_metadata.plugin_version,
            config=node_info.config,
            determinism=audit_metadata.determinism,
            schema_config=schema_config,
            output_contract=output_contract,
            source_file_hash=audit_metadata.source_file_hash,
        )


def record_schema_contract(
    factory: RecorderFactory,
    run_id: str,
    source_id: NodeID,
    ctx: PluginContext,
    *,
    active_source: SourceProtocol,
) -> bool:
    """Record source schema contract if available.

    Called once per run — on the first VALID row (quarantined rows don't
    trigger contract population) or post-loop for runs with no valid rows
    (empty input or all-quarantined).

    Returns:
        True if schema contract was recorded, False otherwise.
    """
    schema_contract = active_source.get_schema_contract()
    if schema_contract is None:
        return False

    # Update source-scoped resume metadata before row processing can fail.
    # Per ADR-025 §3 Decision 5 (G6), ``run_sources.schema_contract_json`` is
    # the single authoritative writer/reader for resume contracts. Do not also
    # write the legacy run-level singleton surface.
    factory.run_lifecycle.update_run_source_contract(
        run_id=run_id,
        source_node_id=source_id,
        schema_contract=schema_contract,
    )
    # Update source node's output_contract (was NULL at registration)
    factory.data_flow.update_node_output_contract(run_id, source_id, schema_contract)
    # Make contract available to transforms via context
    ctx.contract = schema_contract
    return True
