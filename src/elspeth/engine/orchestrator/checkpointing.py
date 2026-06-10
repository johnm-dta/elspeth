"""CheckpointCoordinator: checkpoint sequencing and creation helpers.

Extracted from Orchestrator (core.py) — these methods own:
- ``_sequence_number``: monotonic counter for checkpoint ordering
- ``_active_graph``: the current ExecutionGraph (late-bound at fire time)
- ``_checkpoint_manager``: persists checkpoints to the database
- ``_checkpoint_config``: determines whether/how often to checkpoint
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import structlog

from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.types import CoalesceName, NodeID, SinkName

if TYPE_CHECKING:
    from elspeth.contracts.aggregation_checkpoint import AggregationCheckpointState
    from elspeth.contracts.coalesce_checkpoint import CoalesceCheckpointState
    from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
    from elspeth.contracts.identity import TokenInfo
    from elspeth.core.checkpoint import CheckpointManager
    from elspeth.core.dag import ExecutionGraph
    from elspeth.engine.orchestrator.types import CheckpointAfterSinkCallback, LoopContext, RowProcessorHandle, _CheckpointFactory

slog = structlog.get_logger(__name__)


class CheckpointCoordinator:
    def __init__(
        self,
        *,
        checkpoint_manager: CheckpointManager | None,
        checkpoint_config: RuntimeCheckpointConfig | None,
    ) -> None:
        self._checkpoint_manager = checkpoint_manager
        self._checkpoint_config = checkpoint_config
        self._sequence_number = 0
        self._active_graph: ExecutionGraph | None = None  # relocated from Orchestrator._current_graph; late-bound at fire time

    def set_active_graph(self, graph: ExecutionGraph | None) -> None:
        """Set (or clear) the active execution graph for late-bound checkpoint calls."""
        self._active_graph = graph

    def reset_sequence(self) -> None:
        """Reset checkpoint ordering for a fresh run."""
        self._sequence_number = 0

    def rebase_sequence(self, sequence_number: int) -> None:
        """Continue checkpoint ordering from a previously persisted checkpoint."""
        self._sequence_number = sequence_number

    def maybe_checkpoint(
        self,
        run_id: str,
        token_id: str,
        node_id: str,
        aggregation_state: AggregationCheckpointState | None = None,
        coalesce_state: CoalesceCheckpointState | None = None,
    ) -> None:
        """Create checkpoint if configured.

        Called after a token has been durably written to its terminal sink.
        The checkpoint represents a durable progress marker - recovery can
        safely skip any row whose token has a checkpoint with a sink node_id.

        IMPORTANT: Checkpoints are created AFTER sink writes, not during
        the main processing loop. This ensures the checkpoint represents
        actual durable output, not just processing completion.

        Args:
            run_id: Current run ID
            token_id: Token that was just written to sink
            node_id: Sink node that received the token
            aggregation_state: Typed aggregation checkpoint state for crash recovery
            coalesce_state: Typed pending coalesce state for crash recovery
        """
        if not self._checkpoint_config or not self._checkpoint_config.enabled:
            return
        if self._checkpoint_manager is None:
            return
        if self._active_graph is None:
            # Should never happen - graph is set during execution
            raise OrchestrationInvariantError("Cannot create checkpoint: execution graph not available")

        self._sequence_number += 1

        # RuntimeCheckpointConfig.frequency is an int:
        # - 1 = every_row
        # - 0 = aggregation_only
        # - N = every N rows
        frequency = self._checkpoint_config.frequency
        should_checkpoint = False
        if frequency == 0:
            # aggregation_only: checkpoint unconditionally. In the post-sink
            # architecture (elspeth-rapid-xtmo), _maybe_checkpoint is only
            # called from checkpoint_after_sink — i.e., after sink durability.
            # Aggregation already reduces cardinality (many rows → fewer
            # aggregated results), so the I/O reduction is inherent.
            should_checkpoint = True
        elif frequency == 1:
            should_checkpoint = True  # every_row
        elif frequency > 1:
            should_checkpoint = (self._sequence_number % frequency) == 0  # every_n

        if should_checkpoint:
            self._checkpoint_manager.create_checkpoint(
                run_id=run_id,
                token_id=token_id,
                node_id=node_id,
                sequence_number=self._sequence_number,
                graph=self._active_graph,
                aggregation_state=aggregation_state,
                coalesce_state=coalesce_state,
            )

    def make_checkpoint_after_sink_factory(
        self,
        run_id: str,
        processor: RowProcessorHandle,
    ) -> _CheckpointFactory:
        """Create a per-sink checkpoint callback factory.

        Returns a factory that, given a sink_node_id, produces a callback
        invoked after each token is durably written to that sink.  Used by
        both the normal execution path and the resume path.
        """

        coordinator = self

        class BatchCheckpointCallback:
            """Checkpoint tokens immediately and terminalize scheduler work in batches."""

            def __init__(self, sink_node_id: str, *, terminalize_scheduler: bool) -> None:
                self._sink_node_id = sink_node_id
                self._terminalize_scheduler = terminalize_scheduler
                self._pending_terminal_tokens: list[str] = []

            def __call__(self, token: TokenInfo) -> None:
                agg_state = processor.get_aggregation_checkpoint_state()
                coalesce_state = processor.get_coalesce_checkpoint_state()
                coordinator.maybe_checkpoint(
                    run_id=run_id,
                    token_id=token.token_id,
                    node_id=self._sink_node_id,
                    aggregation_state=agg_state,
                    coalesce_state=coalesce_state if coalesce_state is not None and coalesce_state.has_resumable_state else None,
                )
                if self._terminalize_scheduler:
                    self._pending_terminal_tokens.append(token.token_id)
                if len(self._pending_terminal_tokens) >= 64:
                    self.flush()

            def flush(self) -> None:
                if not self._pending_terminal_tokens:
                    return
                token_ids = tuple(self._pending_terminal_tokens)
                self._pending_terminal_tokens.clear()
                processor.mark_sink_bound_scheduler_terminal_many(token_ids)

        def factory(sink_node_id: str, *, terminalize_scheduler: bool = True) -> CheckpointAfterSinkCallback:
            return BatchCheckpointCallback(sink_node_id, terminalize_scheduler=terminalize_scheduler)

        return factory

    def checkpoint_interrupted_progress(
        self,
        run_id: str,
        loop_ctx: LoopContext,
        sink_id_map: Mapping[SinkName, NodeID],
        source_id: NodeID,
    ) -> None:
        """Persist a resumable checkpoint for graceful shutdown.

        Shutdown is an explicit operator action, so it creates a recovery
        checkpoint even if normal checkpoint frequency would skip this row.
        This preserves resumability for runs that stop before any sink-token
        checkpoint has been emitted, especially buffered aggregation/coalesce
        pipelines that intentionally skip end-of-source flushes on shutdown.

        The checkpoint anchor (token_id/node_id) must reference a token that
        actually exists: aggregation nodes whose buffers were flush-emptied
        (counter-only snapshots) cannot anchor the checkpoint, but their
        durable counters are still persisted because resume restores the full
        aggregation state regardless of the anchor.
        """
        if not self._checkpoint_config or not self._checkpoint_config.enabled:
            return
        if self._checkpoint_manager is None:
            return
        if self._active_graph is None:
            raise OrchestrationInvariantError("Cannot create shutdown checkpoint: execution graph not available")

        aggregation_state = loop_ctx.processor.get_aggregation_checkpoint_state()
        raw_coalesce = loop_ctx.processor.get_coalesce_checkpoint_state()
        # Persist coalesce state when it has pending barriers or completed keys
        # needed for late-arrival detection on resume
        coalesce_state = raw_coalesce if raw_coalesce is not None and raw_coalesce.has_resumable_state else None

        token_id: str | None = None
        node_id: str | None = None
        checkpoint_agg_state: AggregationCheckpointState | None = None

        if aggregation_state.nodes:
            # Persist the full aggregation state whenever any node is present.
            # Counter-only nodes (flush-emptied buffers) carry durable counters
            # (accepted_count_total / completed_flush_count) that resume restores
            # wholesale via restore_from_checkpoint, independent of which token
            # anchors the checkpoint row.
            checkpoint_agg_state = aggregation_state
            for agg_node_id, agg_node_state in aggregation_state.nodes.items():
                if agg_node_state.tokens:
                    token_id = agg_node_state.tokens[-1].token_id
                    node_id = agg_node_id
                    break

        if token_id is None and node_id is None:
            if coalesce_state is not None and coalesce_state.pending:
                pending_entry = coalesce_state.pending[-1]
                node_id = str(loop_ctx.coalesce_node_map[CoalesceName(pending_entry.coalesce_name)])
                if pending_entry.branches:
                    last_branch = list(pending_entry.branches.values())[-1]
                    token_id = last_branch.token_id
            else:
                for sink_name, token_outcome_pairs in loop_ctx.pending_tokens.items():
                    if not token_outcome_pairs:
                        continue
                    token_id = token_outcome_pairs[-1][0].token_id
                    node_id = str(sink_id_map[SinkName(sink_name)])
                    break

        if token_id is None and loop_ctx.last_token_id is not None:
            token_id = loop_ctx.last_token_id
            if node_id is None:
                last_token_source_id = loop_ctx.last_token_source_id
                node_id = str(last_token_source_id) if last_token_source_id is not None else str(source_id)

        if token_id is None or node_id is None:
            slog.warning(
                "shutdown_checkpoint_skipped",
                run_id=run_id,
                reason="no_token_or_node_id_available",
                has_aggregation_nodes=bool(aggregation_state.nodes),
                has_coalesce_pending=coalesce_state is not None,
                has_pending_sink_tokens=any(bool(pairs) for pairs in loop_ctx.pending_tokens.values()),
                last_token_id=loop_ctx.last_token_id,
                resolved_token_id=token_id,
                resolved_node_id=node_id,
            )
            return

        self._sequence_number += 1
        self._checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id=token_id,
            node_id=node_id,
            sequence_number=self._sequence_number,
            graph=self._active_graph,
            aggregation_state=checkpoint_agg_state,
            coalesce_state=coalesce_state,
        )

    def delete_checkpoints(self, run_id: str) -> None:
        """Delete all checkpoints for a run after successful completion.

        Args:
            run_id: Run to clean up checkpoints for
        """
        if self._checkpoint_manager is not None:
            self._checkpoint_manager.delete_checkpoints(run_id)
