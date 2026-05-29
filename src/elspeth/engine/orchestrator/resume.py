"""Resume-path execution helpers for the orchestrator.

This module contains pure functions for the resume code path:
- setup_resume_context: rebuild GraphArtifacts from existing Landscape records
  (the resume-path equivalent of graph node/edge registration)
- run_resume_processing_loop: iterate the unprocessed rows of a resumed run,
  transform/flush/accumulate, with end-of-source aggregation + coalesce flushes
  honoured only when the resume source is truly exhausted

All functions operate on external state passed via parameters - they don't
maintain internal state. This follows the same pattern as aggregation.py and
outcomes.py: pure delegation targets for the Orchestrator.

These functions were extracted from ``Orchestrator`` (where they lived as
``_setup_resume_context`` and ``_run_resume_processing_loop``) to shrink
``core.py``; the resume orchestration that wires them together (``resume`` and
``_reconstruct_resume_state``) remains on the Orchestrator.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from elspeth.contracts import PipelineRow
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.types import NodeID
from elspeth.engine.orchestrator.aggregation import (
    check_aggregation_timeouts,
    flush_remaining_aggregation_buffers,
)
from elspeth.engine.orchestrator.outcomes import (
    accumulate_row_outcomes,
    flush_coalesce_pending,
    handle_coalesce_timeouts,
)
from elspeth.engine.orchestrator.types import GraphArtifacts
from elspeth.engine.orchestrator.validation import (
    validate_route_destinations,
    validate_sink_failsink_destinations,
    validate_source_quarantine_destination,
    validate_transform_error_sinks,
)

if TYPE_CHECKING:
    from elspeth.contracts import SchemaContract
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.landscape.factory import RecorderFactory
    from elspeth.engine.orchestrator.types import LoopContext, PipelineConfig


def setup_resume_context(
    factory: RecorderFactory,
    run_id: str,
    config: PipelineConfig,
    graph: ExecutionGraph,
) -> GraphArtifacts:
    """Resume-path equivalent of _register_graph_nodes_and_edges().

    Loads node ID maps and edge_map from database records instead of
    registering new ones. The graph is the same as the original run,
    but nodes/edges already exist in Landscape.

    Returns:
        GraphArtifacts populated from existing Landscape records.
    """
    # Get explicit node ID mappings from graph
    source_id = graph.get_source()
    sink_id_map = graph.get_sink_id_map()
    transform_id_map = graph.get_transform_id_map()
    config_gate_id_map = graph.get_config_gate_id_map()
    coalesce_id_map = graph.get_coalesce_id_map()

    # Build edge_map from database (load real edge IDs registered in original run)
    # CRITICAL: Must use real edge_ids for FK integrity when recording routing events
    # Convert keys from (str, str) to (NodeID, str) to match RowProcessor's type
    raw_edge_map = factory.data_flow.get_edge_map(run_id)
    edge_map: dict[tuple[NodeID, str], str] = {(NodeID(k[0]), k[1]): v for k, v in raw_edge_map.items()}

    # Get route resolution map for validation
    route_resolution_map = graph.get_route_resolution_map()

    # Validate route destinations (config may have changed since original run)
    # This catches config errors early instead of after partial processing
    # Call module function directly (no wrapper method)
    validate_route_destinations(
        route_resolution_map=route_resolution_map,
        available_sinks=set(config.sinks.keys()),
        transform_id_map=transform_id_map,
        transforms=config.transforms,
        config_gate_id_map=config_gate_id_map,
        config_gates=config.gates,
    )

    # Validate transform error sink destinations
    # Call module function directly (no wrapper method)
    validate_transform_error_sinks(
        transforms=config.transforms,
        available_sinks=set(config.sinks.keys()),
    )

    # Validate source quarantine destination
    # Call module function directly (no wrapper method)
    validate_source_quarantine_destination(
        source=config.source,
        available_sinks=set(config.sinks.keys()),
    )

    # Validate sink failsink destinations
    sink_validation_stubs = {name: SimpleNamespace(on_write_failure=sink._on_write_failure) for name, sink in config.sinks.items()}
    sink_plugins = {name: sink.name for name, sink in config.sinks.items()}
    validate_sink_failsink_destinations(
        sink_configs=sink_validation_stubs,
        available_sinks=set(config.sinks.keys()),
        sink_plugins=sink_plugins,
    )

    return GraphArtifacts(
        edge_map=edge_map,
        source_id=source_id,
        sink_id_map=sink_id_map,
        transform_id_map=transform_id_map,
        config_gate_id_map=config_gate_id_map,
        coalesce_id_map=coalesce_id_map,
    )


def run_resume_processing_loop(
    loop_ctx: LoopContext,
    unprocessed_rows: Sequence[tuple[str, int, dict[str, Any]]],
    schema_contract: SchemaContract,
    *,
    shutdown_event: threading.Event | None = None,
) -> bool:
    """Run the resume processing loop: iterate unprocessed rows, transform, flush, accumulate.

    Includes end-of-loop aggregation/coalesce flushes only when the resume
    source is actually exhausted. On graceful shutdown we keep buffered state
    pending rather than forcing end-of-source semantics.

    Simpler than the main loop:
    - No quarantine handling (rows already validated)
    - No field resolution (already recorded in original run)
    - No schema contract recording (passed via parameter)
    - No operation_id lifecycle (no source track_operation)
    - No progress emission (known gap — see design doc)

    Returns:
        True if interrupted by shutdown, False otherwise.
    """
    # Destructure loop_ctx for local access
    config = loop_ctx.config
    ctx = loop_ctx.ctx
    processor = loop_ctx.processor
    counters = loop_ctx.counters
    pending_tokens = loop_ctx.pending_tokens
    coalesce_executor = loop_ctx.coalesce_executor
    coalesce_node_map = dict(loop_ctx.coalesce_node_map)
    agg_transform_lookup = dict(loop_ctx.agg_transform_lookup)

    # A buffered-only resume can have zero unprocessed rows but still carry
    # restored aggregation/coalesce state. If shutdown is already requested,
    # honor it before any end-of-source flush work so buffered state is
    # checkpointed again instead of being flushed to sinks.
    interrupted_by_shutdown = shutdown_event is not None and shutdown_event.is_set()

    # Process each unprocessed row using process_existing_row
    # (rows already exist in DB, only tokens need to be created)
    for row_id, _row_index, row_data in unprocessed_rows:
        if interrupted_by_shutdown:
            break
        counters.rows_processed += 1

        # ─────────────────────────────────────────────────────────────────
        # Check for timed-out aggregations BEFORE processing this row
        # Ensures timeout flushes OLD batch before processing new row
        # ─────────────────────────────────────────────────────────────────
        # Call module function directly (no wrapper method)
        timeout_result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=ctx,
            pending_tokens=pending_tokens,
            agg_transform_lookup=agg_transform_lookup,
        )
        counters.accumulate_flush_result(timeout_result)

        # Wrap row_data in PipelineRow with contract (PIPELINEROW MIGRATION)
        # Row data from resume is a plain dict, but process_existing_row expects PipelineRow
        pipeline_row = PipelineRow(data=row_data, contract=schema_contract)

        results = processor.process_existing_row(
            row_id=row_id,
            row_data=pipeline_row,
            transforms=config.transforms,
            ctx=ctx,
        )
        if results:
            loop_ctx.last_token_id = results[-1].token.token_id

        # Handle all results from this row
        accumulate_row_outcomes(results, counters, pending_tokens)

        # ─────────────────────────────────────────────────────────────────
        # Check for timed-out coalesces after processing each row
        # Must check coalesce timeouts after each row to flush stale barriers
        # ─────────────────────────────────────────────────────────────────
        if coalesce_executor is not None:
            handle_coalesce_timeouts(
                coalesce_executor=coalesce_executor,
                coalesce_node_map=coalesce_node_map,
                processor=processor,
                ctx=ctx,
                counters=counters,
                pending_tokens=pending_tokens,
            )

        # ─────────────────────────────────────────────────────────────
        # GRACEFUL SHUTDOWN CHECK
        # Check between row iterations — current row is fully
        # processed, outcomes recorded, safe to stop here.
        # No quarantine path in resume (rows already validated).
        # ─────────────────────────────────────────────────────────────
        if shutdown_event is not None and shutdown_event.is_set():
            interrupted_by_shutdown = True
            break

    if not interrupted_by_shutdown:
        # CRITICAL: Flush remaining aggregation buffers only at true end-of-source.
        if config.aggregation_settings:
            # Call module function directly (no wrapper method)
            flush_result = flush_remaining_aggregation_buffers(
                config=config,
                processor=processor,
                ctx=ctx,
                pending_tokens=pending_tokens,
            )
            counters.accumulate_flush_result(flush_result)

            # TERMINAL GUARANTEE: same assertion as _post_source_iteration_work.
            for agg_node_id_str in config.aggregation_settings:
                remaining = processor.get_aggregation_buffer_count(NodeID(agg_node_id_str))
                if remaining > 0:
                    raise OrchestrationInvariantError(
                        f"Aggregation buffer for node '{agg_node_id_str}' still has "
                        f"{remaining} tokens after end-of-source flush. "
                        f"These tokens would never reach a terminal state."
                    )

        # Flush pending coalesce operations only when resume processing exhausted all rows.
        if coalesce_executor is not None:
            flush_coalesce_pending(
                coalesce_executor=coalesce_executor,
                coalesce_node_map=coalesce_node_map,
                processor=processor,
                ctx=ctx,
                counters=counters,
                pending_tokens=pending_tokens,
            )

    return interrupted_by_shutdown
