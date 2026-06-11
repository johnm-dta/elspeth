"""CheckpointCoordinator: checkpoint sequencing and creation helpers.

Extracted from Orchestrator (core.py) — these methods own:
- ``_sequence_number``: monotonic counter for checkpoint ordering
- ``_active_graph``: the current ExecutionGraph (late-bound at fire time)
- ``_checkpoint_manager``: persists checkpoints to the database
- ``_checkpoint_config``: determines whether/how often to checkpoint
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from elspeth.contracts.errors import OrchestrationInvariantError

if TYPE_CHECKING:
    from elspeth.contracts.barrier_scalars import BarrierScalars
    from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
    from elspeth.contracts.identity import TokenInfo
    from elspeth.core.checkpoint import CheckpointManager
    from elspeth.core.dag import ExecutionGraph
    from elspeth.engine.orchestrator.types import CheckpointAfterSinkCallback, LoopContext, RowProcessorHandle, _CheckpointFactory


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
        *,
        barrier_scalars: BarrierScalars | None,
    ) -> None:
        """Create checkpoint if configured.

        Called after a token has been durably written to its terminal sink.
        The checkpoint represents a durable progress marker.

        IMPORTANT: Checkpoints are created AFTER sink writes, not during
        the main processing loop. This ensures the checkpoint represents
        actual durable output, not just processing completion.

        Args:
            run_id: Current run ID
            barrier_scalars: Composed barrier scalars from the live executors
                (``processor.get_barrier_scalars()``, F1 Task 2.4). Passed to
                ``create_checkpoint`` unconditionally — the manager serializes
                NULL when ``has_state`` is False.

        F1: only scalar barrier metadata is persisted; buffered tokens live in
        journal BLOCKED rows.
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
                sequence_number=self._sequence_number,
                barrier_scalars=barrier_scalars,
                graph=self._active_graph,
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

            def __init__(self, *, terminalize_scheduler: bool) -> None:
                self._terminalize_scheduler = terminalize_scheduler
                self._pending_terminal_tokens: list[str] = []

            def __call__(self, token: TokenInfo) -> None:
                coordinator.maybe_checkpoint(
                    run_id=run_id,
                    barrier_scalars=processor.get_barrier_scalars(),
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
            # sink_node_id is the factory's per-sink discriminator; the callback
            # itself no longer persists a per-sink anchor (F2).
            del sink_node_id
            return BatchCheckpointCallback(terminalize_scheduler=terminalize_scheduler)

        return factory

    def checkpoint_interrupted_progress(
        self,
        run_id: str,
        loop_ctx: LoopContext,
    ) -> None:
        """Persist a resumable checkpoint for graceful shutdown.

        Shutdown is an explicit operator action, so it creates a recovery
        checkpoint even if normal checkpoint frequency would skip this row.
        This preserves resumability for runs that stop before any sink-token
        checkpoint has been emitted, especially buffered aggregation/coalesce
        pipelines that intentionally skip end-of-source flushes on shutdown.
        """
        if not self._checkpoint_config or not self._checkpoint_config.enabled:
            return
        if self._checkpoint_manager is None:
            return
        if self._active_graph is None:
            raise OrchestrationInvariantError("Cannot create shutdown checkpoint: execution graph not available")

        self._sequence_number += 1
        self._checkpoint_manager.create_checkpoint(
            run_id=run_id,
            sequence_number=self._sequence_number,
            barrier_scalars=loop_ctx.processor.get_barrier_scalars(),
            graph=self._active_graph,
        )

    def delete_checkpoints(self, run_id: str) -> None:
        """Delete all checkpoints for a run after successful completion.

        Args:
            run_id: Run to clean up checkpoints for
        """
        if self._checkpoint_manager is not None:
            self._checkpoint_manager.delete_checkpoints(run_id)
