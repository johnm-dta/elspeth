"""SinkFlushCoordinator: pending-token sink writes and post-loop flush bookkeeping.

Split out of the old ``RunExecutionCore`` by lifecycle boundary
(elspeth-b53a093321): this module owns the sink-write cluster — pending-token
consumption, failsink resolution, the composed post-sink checkpoint /
scheduler-terminalization callbacks, diversion-counter reconciliation, and
graceful-shutdown checkpointing. Shared verbatim by the main run path and
the resume path.

Dependencies held by the coordinator:
- ``_span_factory``: SpanFactory for SinkExecutor spans
- ``_checkpoints``: CheckpointCoordinator for interrupted-progress checkpoints
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from elspeth.contracts.errors import (
    GracefulShutdownError,
    OrchestrationInvariantError,
)
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.types import (
    NodeID,
    SinkName,
)
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.executors.sink import DiversionCounts
from elspeth.engine.orchestrator.outcomes import (
    reconcile_sink_write_diversions,
)

if TYPE_CHECKING:
    from elspeth.contracts import (
        PendingOutcome,
        SinkProtocol,
        TokenInfo,
    )
    from elspeth.engine.orchestrator.checkpointing import CheckpointCoordinator
    from elspeth.engine.orchestrator.types import (
        CheckpointAfterSinkCallback,
        ExecutionCounters,
        LoopContext,
        PendingTokenMap,
        PipelineConfig,
        SchedulerTerminalizer,
        _CheckpointFactory,
    )
    from elspeth.engine.spans import SpanFactory


class _SchedulerTerminalizationCallback:
    """Terminalize durable scheduler sink-handoffs in batches after sink writes.

    The scheduler-handoff half of what used to be a single dual-purpose
    post-sink callback (elspeth-107a29d02e): accumulate sink-written token ids
    and mark them terminal in batches of 64, with an explicit end-of-write
    flush. Owned here next to the sink-flush coordinator rather than in
    checkpointing.py, because scheduler terminalization is a run-execution
    concern, not a checkpoint concern; it depends only on the narrow
    :class:`SchedulerTerminalizer` slice of the processor.
    """

    def __init__(self, terminalizer: SchedulerTerminalizer) -> None:
        self._terminalizer = terminalizer
        self._pending_terminal_tokens: list[str] = []

    def __call__(self, token: TokenInfo) -> None:
        self._pending_terminal_tokens.append(token.token_id)
        if len(self._pending_terminal_tokens) >= 64:
            self.flush()

    def flush(self) -> None:
        if not self._pending_terminal_tokens:
            return
        token_ids = tuple(self._pending_terminal_tokens)
        self._pending_terminal_tokens.clear()
        self._terminalizer.mark_sink_bound_scheduler_terminal_many(token_ids)


class _CompositeAfterSinkCallback:
    """Fan a post-sink token event out to several ``CheckpointAfterSinkCallback``s.

    Composes the independent post-sink lifecycles (checkpoint progress +
    scheduler terminalization) at the sink-write call site while keeping each
    behind its own flush boundary (elspeth-107a29d02e). Per-token ``__call__``
    runs the callbacks in list order (checkpoint-progress before
    terminalization, matching the pre-split single callback); ``flush`` runs
    them in the same order after the write completes.
    """

    def __init__(self, callbacks: tuple[CheckpointAfterSinkCallback, ...]) -> None:
        self._callbacks = callbacks

    def __call__(self, token: TokenInfo) -> None:
        for callback in self._callbacks:
            callback(token)

    def flush(self) -> None:
        for callback in self._callbacks:
            callback.flush()


class SinkFlushCoordinator:
    """Owns the sink-write cluster of an Orchestrator run.

    Holds the two methods shared verbatim by the main run path and the resume
    path: the post-loop flush entry point (:meth:`flush_and_write_sinks`) and
    the underlying pending-token sink writer (:meth:`write_pending_to_sinks`).
    """

    def __init__(
        self,
        *,
        span_factory: SpanFactory,
        checkpoints: CheckpointCoordinator,
    ) -> None:
        self._span_factory = span_factory
        self._checkpoints = checkpoints

    def write_pending_to_sinks(
        self,
        factory: RecorderFactory,
        run_id: str,
        config: PipelineConfig,
        ctx: PluginContext,
        counters: ExecutionCounters,
        pending_tokens: PendingTokenMap,
        sink_id_map: dict[SinkName, NodeID],
        edge_map: Mapping[tuple[NodeID, str], str],
        sink_step: int,
        *,
        on_token_written_factory: _CheckpointFactory | None = None,
        scheduler_terminalizer: SchedulerTerminalizer | None = None,
    ) -> DiversionCounts:
        """Write pending tokens to sinks using SinkExecutor.

        Extracted from _execute_run() and _process_resumed_rows() to eliminate
        duplication of the sink write orchestration pattern.

        Args:
            factory: RecorderFactory for audit trail
            run_id: Current run ID
            config: Pipeline configuration
            ctx: Plugin context
            pending_tokens: Dict of sink_name -> list of (token, pending_outcome) pairs
            sink_id_map: Maps SinkName -> NodeID for checkpoint callbacks
            sink_step: Audit step index for sink writes (from processor.resolve_sink_step())
            on_token_written_factory: Optional factory that creates per-sink checkpoint
                PROGRESS callbacks. Takes sink_node_id, returns callback(TokenInfo) -> None.
                When None, no checkpoint-progress callbacks are used.
            scheduler_terminalizer: Optional narrow processor surface used to build a
                batched scheduler-terminalization callback, composed alongside the
                checkpoint-progress callback for grouped batches whose pending outcome
                carries a durable scheduler PENDING_SINK handoff (elspeth-107a29d02e).
                When None, no scheduler terminalization is performed.
        """
        from itertools import groupby

        from elspeth.engine.executors.sink import DiversionCounts, SinkExecutor

        sink_executor = SinkExecutor(factory.execution, factory.data_flow, self._span_factory, run_id)
        step = sink_step
        total_diversions = DiversionCounts()

        def consume_group(
            live_pairs: list[tuple[TokenInfo, PendingOutcome | None]], group_pairs: list[tuple[TokenInfo, PendingOutcome | None]]
        ) -> None:
            consumed_pair_ids = {id(pair) for pair in group_pairs}
            live_pairs[:] = [pair for pair in live_pairs if id(pair) not in consumed_pair_ids]

        for sink_name, token_outcome_pairs in pending_tokens.items():
            if not token_outcome_pairs:
                continue
            if sink_name not in config.sinks:
                raise OrchestrationInvariantError(
                    f"Sink '{sink_name}' in pending_tokens not found in config.sinks. "
                    f"Available: {sorted(config.sinks.keys())}. "
                    f"This indicates a token routing bug."
                )
            sink = config.sinks[sink_name]
            sink_node_id = sink_id_map[SinkName(sink_name)]

            # Resolve failsink reference (if configured and not 'discard')

            failsink: SinkProtocol | None = None
            failsink_config_name: str | None = None
            failsink_edge_id: str | None = None
            on_write_failure = sink._on_write_failure
            if on_write_failure is not None and on_write_failure != "discard":
                if on_write_failure not in config.sinks:
                    raise OrchestrationInvariantError(
                        f"Sink '{sink_name}' on_write_failure references '{on_write_failure}' "
                        f"which passed validation but is not in config.sinks at runtime. "
                        f"Available: {sorted(config.sinks.keys())}."
                    )
                failsink = config.sinks[on_write_failure]
                failsink_config_name = on_write_failure
                failsink_edge_key = (sink_node_id, "__failsink__")
                try:
                    failsink_edge_id = edge_map[failsink_edge_key]
                except KeyError as exc:
                    raise OrchestrationInvariantError(
                        f"Sink '{sink_name}' on_write_failure='{on_write_failure}' "
                        f"but no __failsink__ DIVERT edge exists in DAG for node '{sink_node_id}'. "
                        f"This is a DAG construction bug — on_write_failure should have "
                        f"created a DIVERT edge in from_plugin_instances()."
                    ) from exc

            # Group tokens by pending_outcome for separate write() calls
            # (sink_executor.write() takes a single PendingOutcome for all tokens in a batch)
            # PendingOutcome carries error_hash for QUARANTINED tokens
            def pending_sort_key(pair: tuple[TokenInfo, PendingOutcome | None]) -> tuple[bool, str, str, str, bool]:
                pending = pair[1]
                if pending is None:
                    return (True, "", "", "", False)  # None sorts last
                outcome_value = pending.outcome.value if pending.outcome is not None else ""
                return (False, outcome_value, pending.path.value, pending.error_hash or "", pending.scheduler_pending_sink)

            sorted_pairs = sorted(token_outcome_pairs, key=pending_sort_key)

            for _group_key, group in groupby(sorted_pairs, key=pending_sort_key):
                group_pairs = list(group)
                pending_outcome = group_pairs[0][1]
                group_tokens = [token for token, _pending in group_pairs]
                # Only tokens with a proven durable PENDING_SINK handoff are
                # terminalized after sink durability. Aggregation flush
                # outputs carry that handoff since F1/D6 (the atomic barrier
                # completion inserts their PENDING_SINK rows); terminal
                # coalesce merges and source-quarantine rows still need
                # checkpoints but have no scheduler row to close.
                terminalize_scheduler = bool(pending_outcome is not None and pending_outcome.scheduler_pending_sink)
                # Compose the two independent post-sink lifecycles here
                # (elspeth-107a29d02e): checkpoint progress (from the factory) and,
                # only when this grouped batch carries a durable scheduler
                # PENDING_SINK handoff, batched scheduler terminalization. Each keeps
                # its own flush boundary; the composite preserves the pre-split order
                # (checkpoint-progress before terminalization).
                after_sink_callbacks: list[CheckpointAfterSinkCallback] = []
                if on_token_written_factory is not None:
                    after_sink_callbacks.append(on_token_written_factory(sink_node_id))
                if terminalize_scheduler and scheduler_terminalizer is not None:
                    after_sink_callbacks.append(_SchedulerTerminalizationCallback(scheduler_terminalizer))
                on_token_written: CheckpointAfterSinkCallback | None = (
                    _CompositeAfterSinkCallback(tuple(after_sink_callbacks)) if after_sink_callbacks else None
                )
                _, diversion_counts = sink_executor.write(
                    sink=sink,
                    tokens=group_tokens,
                    ctx=ctx,
                    step_in_pipeline=step,
                    sink_name=sink_name,
                    pending_outcome=pending_outcome,
                    failsink=failsink,
                    failsink_name=failsink_config_name,
                    failsink_edge_id=failsink_edge_id,
                    on_token_written=on_token_written,
                )
                if on_token_written is not None:
                    on_token_written.flush()
                consume_group(token_outcome_pairs, group_pairs)
                reconcile_sink_write_diversions(
                    counters=counters,
                    sink_name=sink_name,
                    pending_outcome=pending_outcome,
                    diversion_count=diversion_counts.total,
                )
                total_diversions = DiversionCounts(
                    failsink_mode=total_diversions.failsink_mode + diversion_counts.failsink_mode,
                    discard_mode=total_diversions.discard_mode + diversion_counts.discard_mode,
                )

        return total_diversions

    def flush_and_write_sinks(
        self,
        factory: RecorderFactory,
        run_id: str,
        loop_ctx: LoopContext,
        sink_id_map: Mapping[SinkName, NodeID],
        edge_map: Mapping[tuple[NodeID, str], str],
        interrupted_by_shutdown: bool,
        *,
        on_token_written_factory: _CheckpointFactory | None = None,
        scheduler_terminalizer: SchedulerTerminalizer | None = None,
    ) -> None:
        """Write all pending tokens to sinks and handle post-loop bookkeeping.

        IMPORTANT: Aggregation flush and coalesce flush are NOT in this method.
        They stay inside the processing loop because they must execute inside
        the track_operation(source_load) context to preserve audit attribution.

        Handles:
        1. Write pending tokens to sinks (each sink has its own track_operation)
        2. Raise GracefulShutdownError if interrupted
        """
        counters = loop_ctx.counters

        diversion_counts = self.write_pending_to_sinks(
            factory=factory,
            run_id=run_id,
            config=loop_ctx.config,
            ctx=loop_ctx.ctx,
            counters=loop_ctx.counters,
            pending_tokens=loop_ctx.pending_tokens,
            sink_id_map=dict(sink_id_map),
            edge_map=edge_map,
            sink_step=loop_ctx.processor.resolve_sink_step(),
            on_token_written_factory=on_token_written_factory,
            scheduler_terminalizer=scheduler_terminalizer,
        )
        # ADR-019: failsink-mode diversions are TRANSIENT structural evidence;
        # discard-mode diversions are FAILURE predicate inputs as well.
        loop_ctx.counters.rows_diverted += diversion_counts.total
        loop_ctx.counters.rows_failed += diversion_counts.discard_mode

        # If shutdown interrupted the loop, raise after all pending work is flushed.
        # At this point: sink writes are done, and any buffered aggregation/coalesce
        # state that we intentionally preserved can be checkpointed for resume.
        if interrupted_by_shutdown:
            self._checkpoints.checkpoint_interrupted_progress(
                run_id=run_id,
                loop_ctx=loop_ctx,
            )
            raise GracefulShutdownError(
                rows_processed=counters.rows_processed,
                run_id=run_id,
                rows_succeeded=counters.rows_succeeded,
                rows_failed=counters.rows_failed,
                rows_quarantined=counters.rows_quarantined,
                rows_routed_success=counters.rows_routed_success,
                rows_routed_failure=counters.rows_routed_failure,
                routed_destinations=dict(counters.routed_destinations),
            )
