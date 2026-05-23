"""Landscape run-setup registration/recording functions for the orchestrator.

This module contains pure functions that write run-setup records to the
Landscape audit trail:
- register_nodes with the data-flow repository for every node in the execution
  graph (resolving plugin metadata, schema config, output contract per node)
- record_schema_contract: persist the source schema contract at run and
  source-node level and expose it to transforms via the plugin context

All functions operate on external state passed via parameters - they don't
maintain internal state. This follows the same pattern as aggregation.py and
outcomes.py: pure delegation targets for the Orchestrator. (Moving them does not
change audit ordering - the bodies are unchanged.)

These functions were extracted from ``Orchestrator`` (where they lived as
``_register_nodes_with_landscape`` and ``_record_schema_contract``) to shrink
``core.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elspeth import __version__ as ENGINE_VERSION
from elspeth.contracts import Determinism, NodeType
from elspeth.contracts.errors import FrameworkBugError, OrchestrationInvariantError
from elspeth.contracts.types import NodeID

if TYPE_CHECKING:
    from elspeth.contracts import SourceProtocol
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.landscape.factory import RecorderFactory
    from elspeth.engine.orchestrator.types import PipelineConfig


def register_nodes_with_landscape(
    factory: RecorderFactory,
    run_id: str,
    config: PipelineConfig,
    graph: ExecutionGraph,
    execution_order: list[str],
    node_to_plugin: dict[NodeID, Any],
    source_id: NodeID,
    config_gate_node_ids: set[NodeID],
    coalesce_node_ids: set[NodeID],
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
        node_to_plugin: Mapping from node ID to plugin instance.
        source_id: Source node ID (for output contract).
        config_gate_node_ids: Set of config gate node IDs (structural, no plugin).
        coalesce_node_ids: Set of coalesce node IDs (structural, no plugin).
    """

    for node_id in execution_order:
        node_info = graph.get_node_info(node_id)

        # Config gates and coalesce nodes are structural (no plugin instances)
        # Aggregations have plugin instances in node_to_plugin (transforms with metadata)
        if node_id in config_gate_node_ids:
            # Config gates are deterministic (expression evaluation is deterministic)
            # Use engine version to track which version of ExpressionParser was used
            plugin_version = f"engine:{ENGINE_VERSION}"
            determinism = Determinism.DETERMINISTIC
            source_file_hash = None  # Engine-internal nodes have no source file
        elif node_id in coalesce_node_ids or node_info.node_type == NodeType.QUEUE:
            # Coalesce/queue nodes are engine-internal deterministic
            # coordination nodes. Use engine version to track which version of
            # the structural logic was used.
            plugin_version = f"engine:{ENGINE_VERSION}"
            determinism = Determinism.DETERMINISTIC
            source_file_hash = None  # Engine-internal nodes have no source file
        else:
            # Direct access - if node_id is in execution_order (from graph.topological_order()),
            # it MUST be in node_to_plugin (built from the same graph's source, transforms, sinks).
            # A KeyError here indicates a bug in graph construction or node_to_plugin building.
            plugin = node_to_plugin[NodeID(node_id)]

            # Extract plugin metadata - all protocols define these attributes,
            # all base classes provide defaults. Direct access is safe.
            plugin_version = plugin.plugin_version
            determinism = plugin.determinism
            source_file_hash = plugin.source_file_hash

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
            plugin_version=plugin_version,
            config=node_info.config,
            determinism=determinism,
            schema_config=schema_config,
            output_contract=output_contract,
            source_file_hash=source_file_hash,
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
    factory.run_lifecycle.update_run_source_contract(
        run_id=run_id,
        source_node_id=source_id,
        schema_contract=schema_contract,
    )
    # Update source node's output_contract (was NULL at registration)
    factory.data_flow.update_node_output_contract(run_id, source_id, schema_contract)
    # Preserve the legacy run-level singleton for single-source consumers,
    # but never let it block later source-scoped contracts in multi-source
    # runs. Per-source run_sources records are authoritative for resume.
    if factory.run_lifecycle.get_run_contract(run_id) is None:
        factory.run_lifecycle.update_run_contract(run_id, schema_contract)
    # Make contract available to transforms via context
    ctx.contract = schema_contract
    return True
