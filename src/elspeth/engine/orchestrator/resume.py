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
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from elspeth.contracts import PipelineRow
from elspeth.contracts.errors import AuditIntegrityError, OrchestrationInvariantError
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
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.checkpoint.recovery import IncompleteTokenSpec, RecoveryManager
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
    source_id_map: dict[str, NodeID] = {}
    for candidate_source_id in graph.get_sources():
        source_info = graph.get_node_info(candidate_source_id)
        if "source_name" not in source_info.config:
            raise OrchestrationInvariantError(
                f"DAG source node '{source_info.node_id}' is missing 'source_name' in its config. "
                f"Per ADR-025 §2 the DAG builder MUST set source_name on every source node. "
                f"This is a graph-construction bug — node config keys: {sorted(source_info.config.keys())}."
            )
        source_id_map[str(source_info.config["source_name"])] = candidate_source_id
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
        source_id_map=source_id_map,
        sink_id_map=sink_id_map,
        transform_id_map=transform_id_map,
        config_gate_id_map=config_gate_id_map,
        coalesce_id_map=coalesce_id_map,
    )


def run_resume_processing_loop(
    loop_ctx: LoopContext,
    unprocessed_rows: Sequence[tuple[str, int, dict[str, Any]] | tuple[str, int, NodeID, dict[str, Any]]],
    schema_contract: SchemaContract,
    *,
    schema_contracts_by_source: Mapping[NodeID, SchemaContract] | None = None,
    source_on_success_by_source: Mapping[NodeID, str] | None = None,
    incomplete_by_row: Mapping[str, Sequence[IncompleteTokenSpec]],
    recovery_manager: RecoveryManager,
    payload_store: PayloadStore,
    run_id: str,
    resume_checkpoint_id: str,
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

    Per-row dispatch (F1 fix):
    - If the row has incomplete child tokens (partial fork/expand/coalesce):
      drive ONLY the incomplete children via resume_incomplete_token.
      Restarting from source (process_existing_row) would re-fork to ALL branches
      and re-emit the completed ones (the F1 double-emission defect).
    - Otherwise (never started, or fully linear): whole-row restart from source
      via process_existing_row is correct.

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

    if not interrupted_by_shutdown and processor.has_scheduled_work():
        recovered_row_ids = frozenset(row[0] for row in unprocessed_rows)
        scheduled_row_ids = processor.active_scheduled_row_ids()
        uncovered_row_ids = recovered_row_ids - scheduled_row_ids
        if uncovered_row_ids:
            formatted_uncovered = ", ".join(sorted(uncovered_row_ids))
            formatted_scheduled = ", ".join(sorted(scheduled_row_ids)) or "<none>"
            raise AuditIntegrityError(
                "Resume scheduler coverage is incomplete: active scheduler work exists, "
                "but recovered rows are not represented by scheduler work items. "
                f"Uncovered row_id(s): {formatted_uncovered}. "
                f"Scheduled row_id(s): {formatted_scheduled}. "
                "Refusing mixed scheduler/source replay to avoid skipped or duplicated rows."
            )
        results = processor.drain_scheduled_work(ctx)
        counters.rows_processed += len({result.token.row_id for result in results})
        accumulate_row_outcomes(results, counters, pending_tokens)
        unprocessed_rows = ()

    # Process each unprocessed row. Rows already exist in DB; only tokens need to
    # be created. Dispatch: partial-fork/expand/coalesce rows use mid-DAG continuation;
    # never-started and fully-linear rows use whole-row restart (process_existing_row).
    for resumed_row in unprocessed_rows:
        if interrupted_by_shutdown:
            break
        if len(resumed_row) == 4:
            row_id, _row_index, source_node_id, row_data = resumed_row
            if schema_contracts_by_source is None or source_node_id not in schema_contracts_by_source:
                raise OrchestrationInvariantError(
                    f"Cannot resume row {row_id!r} from source node {source_node_id!r}: "
                    "source-scoped schema contract is missing from resume state."
                )
            row_contract = schema_contracts_by_source[source_node_id]
            if source_on_success_by_source is None or source_node_id not in source_on_success_by_source:
                raise OrchestrationInvariantError(
                    f"Cannot resume row {row_id!r} from source node {source_node_id!r}: "
                    "source-scoped on_success routing is missing from resume state."
                )
            source_on_success = source_on_success_by_source[source_node_id]
        else:
            row_id, _row_index, row_data = resumed_row
            source_node_id = None
            row_contract = schema_contract
            source_on_success = None
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
        ctx.contract = row_contract
        pipeline_row = PipelineRow(data=row_data, contract=row_contract)

        # F1 fix: dispatch on whether this row has incomplete fork/expand/coalesce child tokens.
        #
        # incomplete_by_row ⊆ unprocessed_rows by construction (both queries share
        # _DELEGATION_PATHS and "incomplete non-delegation token" is Case 2 of
        # get_unprocessed_rows), so every partial-fork/expand/coalesce row IS visited
        # by this loop and its specs are found here.
        #
        # Lineage-field filter: get_incomplete_tokens_by_row returns ALL incomplete
        # non-delegation tokens — including linear-pipeline tokens that were interrupted
        # mid-transform (branch_name=None, fork_group_id=None, expand_group_id=None,
        # join_group_id=None). Those linear tokens are correctly handled by
        # process_existing_row (whole-row restart mints a fresh token); routing them to
        # resume_incomplete_token raises OrchestrationInvariantError (F1 regression).
        # Only dispatch specs that are provably fork/expand/coalesce children (at least
        # one lineage field set).
        # Direct key check (not .get()) — incomplete_by_row is our pre-built index
        # (Tier-1 audit data), not an external boundary. A missing key is the normal
        # "no incomplete children for this row" case.
        fork_expand_coalesce_specs = (
            [
                s
                for s in incomplete_by_row[row_id]
                if s.branch_name is not None or s.fork_group_id is not None or s.expand_group_id is not None or s.join_group_id is not None
            ]
            if row_id in incomplete_by_row
            else []
        )

        if fork_expand_coalesce_specs:
            # Partial fork/expand/coalesce completion: drive ONLY the incomplete
            # children to completion under the original parent. Restarting from
            # source (process_existing_row) would re-fork to ALL branches and
            # re-emit the completed ones (F1 double-emission defect).
            results = []
            for spec in fork_expand_coalesce_specs:
                token_row = recovery_manager.reconstruct_token_row(spec, run_id, source_row=pipeline_row, payload_store=payload_store)
                results.extend(processor.resume_incomplete_token(spec, token_row, ctx, resume_checkpoint_id=resume_checkpoint_id))
        else:
            # No incomplete fork/expand/coalesce tokens for this row (never started,
            # fully linear, or interrupted linear token): whole-row restart from source
            # is correct. process_existing_row mints a fresh token and re-traverses.
            results = processor.process_existing_row(
                row_id=row_id,
                row_data=pipeline_row,
                transforms=config.transforms,
                ctx=ctx,
                source_node_id=source_node_id,
                source_on_success=source_on_success,
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

        if processor.has_scheduled_work():
            active_work = "; ".join(processor.summarize_scheduled_work()) or "<unknown>"
            raise OrchestrationInvariantError(
                f"Resume for run '{processor.run_id}' left non-terminal scheduler work after end-of-source flush. "
                "Blocked or future WAITING scheduler state must be recovered explicitly before run completion. "
                f"Active scheduler work: {active_work}."
            )

    return interrupted_by_shutdown
