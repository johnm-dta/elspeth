"""Aggregation handling functions for the orchestrator.

This module contains functions for:
- Finding aggregation transforms by node ID
- Checking and flushing aggregation timeouts
- Flushing remaining aggregation buffers at end-of-source

All functions operate on external state passed via parameters - they don't
maintain internal state. This enables the Orchestrator to use them as
pure delegation targets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from elspeth.contracts.enums import TriggerType
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.types import NodeID
from elspeth.engine.orchestrator.outcomes import accumulate_row_outcomes
from elspeth.engine.orchestrator.types import (
    AggNodeEntry,
    AggregationFlushResult,
    ExecutionCounters,
    PendingTokenMap,
    PipelineConfig,
    RowProcessorHandle,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from elspeth.contracts import TransformProtocol
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.contracts.results import RowResult
    from elspeth.engine.dag_navigator import WorkItem


def find_aggregation_transform(
    config: PipelineConfig,
    agg_node_id_str: str,
    agg_name: str,
) -> tuple[TransformProtocol, NodeID]:
    """Find the batch-aware transform for an aggregation node.

    Args:
        config: Pipeline configuration with transforms
        agg_node_id_str: The aggregation node ID as string
        agg_name: Human-readable aggregation name (for error messages)

    Returns:
        Tuple of (transform, aggregation_node_id)

    Raises:
        RuntimeError: If no batch-aware transform found for the aggregation
    """
    from elspeth.contracts import TransformProtocol

    agg_transform: TransformProtocol | None = None
    agg_node_id = NodeID(agg_node_id_str)

    for t in config.transforms:
        if isinstance(t, TransformProtocol) and t.node_id == agg_node_id_str and t.is_batch_aware:
            agg_transform = t
            break

    if agg_transform is None:
        raise OrchestrationInvariantError(
            f"No batch-aware transform found for aggregation '{agg_name}' "
            f"(node_id={agg_node_id_str}). This indicates a bug in graph construction "
            f"or pipeline configuration. "
            f"Available transforms: {[t.node_id for t in config.transforms]}"
        )

    return agg_transform, agg_node_id


def _process_flush_results(
    completed_results: Sequence[RowResult],
    work_items: Sequence[WorkItem],
    processor: RowProcessorHandle,
    ctx: PluginContext,
    counters: ExecutionCounters,
    pending_tokens: PendingTokenMap,
) -> None:
    """Accumulate completed results and route work items through remaining transforms.

    Extracted from check_aggregation_timeouts and flush_remaining_aggregation_buffers
    which had identical post-flush continuation loops.
    """
    accumulate_row_outcomes(completed_results, counters, pending_tokens)

    for work_item in work_items:
        if work_item.current_node_id is None:
            raise OrchestrationInvariantError("Aggregation continuation work item missing current_node_id")
        downstream_results = processor.process_token(
            token=work_item.token,
            ctx=ctx,
            current_node_id=work_item.current_node_id,
            coalesce_node_id=work_item.coalesce_node_id,
            coalesce_name=work_item.coalesce_name,
        )
        accumulate_row_outcomes(downstream_results, counters, pending_tokens)


def check_aggregation_timeouts(
    config: PipelineConfig,
    processor: RowProcessorHandle,
    ctx: PluginContext,
    pending_tokens: PendingTokenMap,
    agg_transform_lookup: dict[str, AggNodeEntry] | None = None,
) -> AggregationFlushResult:
    """Check and flush any aggregations whose timeout has expired.

    Called BEFORE processing each row and by the orchestrator's idle source
    polling path. Checking BEFORE buffering ensures timed-out batches don't
    include the newly arriving row; polling while source iteration waits lets
    timeout batches flush even when no additional row arrives.

    Routing uses result.sink_name (set by on_success in the processor) rather
    than a default_sink_name parameter.

    Note: Checkpointing is NOT done here. Tokens routed to pending_tokens are
    only checkpointed after SinkExecutor.write() achieves sink durability,
    via the checkpoint_after_sink callback. Fix: elspeth-rapid-xtmo.

    Args:
        config: Pipeline configuration with aggregation_settings
        processor: RowProcessor-compatible handle with public aggregation timeout API
        ctx: Plugin context for transform execution
        pending_tokens: Dict of sink_name -> tokens to append results to
        agg_transform_lookup: Pre-computed dict mapping node_id_str ->
            AggNodeEntry(transform, node_id).
            If None, lookup is computed on each call (less efficient).

    Returns:
        AggregationFlushResult with counts for succeeded, failed, routed,
        quarantined, coalesced, forked, expanded, buffered rows and routed_destinations
    """
    counters = ExecutionCounters()

    for agg_node_id_str, agg_settings in config.aggregation_settings.items():
        agg_node_id = NodeID(agg_node_id_str)

        # Use public facade method to check timeout (no private member access)
        should_flush, trigger_type = processor.check_aggregation_timeout(agg_node_id)

        if not should_flush:
            continue

        # Skip count triggers — they are handled in buffer_row.
        # Timeout AND condition triggers can be time-based (e.g.,
        # batch_age_seconds >= 5) and must flush before the next row is buffered.
        # Condition triggers can also be time-based and must be checked here.
        if trigger_type not in (TriggerType.TIMEOUT, TriggerType.CONDITION):
            continue

        # Check if there are buffered rows
        buffered_count = processor.get_aggregation_buffer_count(agg_node_id)
        if buffered_count == 0:
            continue

        # Get transform and aggregation node from pre-computed lookup (O(1)) or compute (O(n))
        if agg_transform_lookup and agg_node_id_str in agg_transform_lookup:
            entry = agg_transform_lookup[agg_node_id_str]
            agg_transform = entry.transform
        else:
            # Fallback: use helper method if lookup not provided
            agg_transform, _agg_node_id = find_aggregation_transform(config, agg_node_id_str, agg_settings.name)

        # Use handle_timeout_flush for proper output_mode handling.
        # Continuation is node-based inside the processor.
        # Pass actual trigger_type (TIMEOUT or CONDITION) for correct audit records.
        completed_results, work_items = processor.handle_timeout_flush(
            node_id=agg_node_id,
            transform=agg_transform,
            ctx=ctx,
            trigger_type=trigger_type,
        )

        _process_flush_results(
            completed_results,
            work_items,
            processor,
            ctx,
            counters,
            pending_tokens,
        )

    return counters.to_flush_result()


def flush_remaining_aggregation_buffers(
    config: PipelineConfig,
    processor: RowProcessorHandle,
    ctx: PluginContext,
    pending_tokens: PendingTokenMap,
) -> AggregationFlushResult:
    """Flush remaining aggregation buffers at end-of-source.

    Without this, rows buffered but not yet flushed (e.g., 50 rows
    when trigger is count=100) would be silently lost.

    Uses handle_timeout_flush with END_OF_SOURCE trigger to properly handle
    all output_mode semantics (single, passthrough, transform) and route
    tokens through remaining transforms if any exist after the aggregation.

    Routing uses result.sink_name (set by on_success in the processor) rather
    than a default_sink_name parameter.

    Note: Checkpointing is NOT done here. Tokens routed to pending_tokens are
    only checkpointed after SinkExecutor.write() achieves sink durability,
    via the checkpoint_after_sink callback. Fix: elspeth-rapid-xtmo.

    Args:
        config: Pipeline configuration with aggregation_settings
        processor: RowProcessor-compatible handle with public aggregation facades
        ctx: Plugin context for transform execution
        pending_tokens: Dict of sink_name -> tokens to append results to

    Returns:
        AggregationFlushResult with counts for succeeded, failed, routed,
        quarantined, coalesced, forked, expanded, buffered rows and routed_destinations

    Raises:
        RuntimeError: If no batch-aware transform found for an aggregation
                     (indicates bug in graph construction or pipeline config)
    """
    counters = ExecutionCounters()

    for agg_node_id_str, agg_settings in config.aggregation_settings.items():
        agg_node_id = NodeID(agg_node_id_str)

        # Use public facade (not private member)
        buffered_count = processor.get_aggregation_buffer_count(agg_node_id)
        if buffered_count == 0:
            continue

        # Use helper method for transform lookup
        agg_transform, _agg_node_id = find_aggregation_transform(config, agg_node_id_str, agg_settings.name)

        # Use handle_timeout_flush with END_OF_SOURCE trigger
        # This properly handles output_mode and routes through remaining transforms
        completed_results, work_items = processor.handle_timeout_flush(
            node_id=agg_node_id,
            transform=agg_transform,
            ctx=ctx,
            trigger_type=TriggerType.END_OF_SOURCE,
        )

        _process_flush_results(
            completed_results,
            work_items,
            processor,
            ctx,
            counters,
            pending_tokens,
        )

    return counters.to_flush_result()
