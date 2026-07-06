"""CheckpointCoordinator: checkpoint sequencing and creation helpers.

Extracted from Orchestrator (core.py) — these methods own:
- ``_sequence_number``: monotonic counter for checkpoint ordering
- ``_active_graph``: the current ExecutionGraph (late-bound at fire time)
- ``_checkpoint_manager``: persists checkpoints to the database
- ``_checkpoint_config``: determines whether/how often to checkpoint
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from elspeth.contracts import CheckpointDraft
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.core.checkpoint.compatibility import CheckpointCompatibilityValidator

if TYPE_CHECKING:
    from elspeth.contracts.barrier_scalars import BarrierScalars
    from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
    from elspeth.contracts.coordination import CoordinationToken
    from elspeth.contracts.identity import TokenInfo
    from elspeth.core.checkpoint import CheckpointManager
    from elspeth.core.dag import ExecutionGraph
    from elspeth.engine.orchestrator.ports import (
        BarrierScalarsSource,
        CheckpointAfterSinkCallback,
        _CheckpointFactory,
    )
    from elspeth.engine.orchestrator.run_state import (
        LoopContext,
    )


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
        # ADR-030 leader fencing token; bound at run/resume start and
        # threaded into every CheckpointManager write so the checkpoint
        # INSERT/DELETE carries the verify-and-extend epoch fence.
        self._coordination_token: CoordinationToken | None = None

    def set_active_graph(self, graph: ExecutionGraph | None) -> None:
        """Set (or clear) the active execution graph for late-bound checkpoint calls."""
        self._active_graph = graph

    def bind_coordination(self, token: CoordinationToken | None) -> None:
        """Bind (or clear) the leader fencing token for this run's checkpoint writes.

        Called once at run/resume start with the token minted by
        ``begin_run`` / ``acquire_run_leadership``. The token is carried by
        value and never re-read mid-run (ADR-030 §G).
        """
        self._coordination_token = token

    def _checkpoint_gate(self, *, action: str) -> tuple[RuntimeCheckpointConfig, CheckpointManager, ExecutionGraph] | None:
        """Shared enabled/manager/graph precondition gate for checkpoint writes.

        Returns the narrowed (config, manager, graph) triple when checkpointing
        should proceed, or None when checkpointing is disabled/unconfigured. A
        missing graph with checkpointing enabled should never happen — the
        graph is set during execution — so that arm raises instead of silently
        skipping.
        """
        if not self._checkpoint_config or not self._checkpoint_config.enabled:
            return None
        if self._checkpoint_manager is None:
            return None
        if self._active_graph is None:
            raise OrchestrationInvariantError(f"Cannot create {action}: execution graph not available")
        return self._checkpoint_config, self._checkpoint_manager, self._active_graph

    def _require_fence(self, run_id: str) -> CoordinationToken:
        """Fail closed unless a leader token bound to THIS run is held.

        ADR-030 defense-in-depth (elspeth-fab455790d): checkpoint create and
        delete are leader-only writes. A missing token would fall through to
        CheckpointManager's unfenced plain-write arm (a deliberate seam for
        direct repository/test/tooling callers, NOT the coordinator runtime
        path); a token minted for a different run would fence against the
        wrong run's epoch seat. Both are wiring bugs and must crash before
        any manager call. Callers invoke this AFTER their enabled/manager
        gate so disabled-checkpointing runs stay token-free.
        """
        token = self._coordination_token
        if token is None:
            raise OrchestrationInvariantError(
                f"Checkpoint write for run {run_id!r} attempted with no bound leader token; "
                "bind_coordination must run at run/resume start before any checkpoint write (ADR-030)."
            )
        if token.run_id != run_id:
            raise OrchestrationInvariantError(
                f"Checkpoint write for run {run_id!r} attempted under a leader token for run "
                f"{token.run_id!r}; the coordinator's bound token must belong to the run being written (ADR-030)."
            )
        return token

    def _build_checkpoint_draft(
        self,
        *,
        run_id: str,
        sequence_number: int,
        barrier_scalars: BarrierScalars | None,
        graph: ExecutionGraph,
    ) -> CheckpointDraft:
        """Build persistence-ready checkpoint data at the topology boundary."""
        return CheckpointDraft(
            run_id=run_id,
            sequence_number=sequence_number,
            barrier_scalars=barrier_scalars,
            upstream_topology_hash=CheckpointCompatibilityValidator().compute_full_topology_hash(graph),
        )

    def reset_sequence(self) -> None:
        """Reset checkpoint ordering for a fresh run."""
        self._sequence_number = 0

    def rebase_sequence(self, sequence_number: int) -> None:
        """Continue checkpoint ordering from a previously persisted checkpoint."""
        self._sequence_number = sequence_number

    def checkpoint_run_start(self, run_id: str) -> None:
        """Write the sequence-0 run-start checkpoint (F1 design D4).

        Called once per fresh run, before source iteration. Every
        checkpointing-enabled run therefore has a baseline checkpoint row,
        so resume topology validation is unconditional and ``can_resume``'s
        missing-baseline arm is a genuine "run predates run-start
        checkpointing or checkpointing was disabled" refusal.

        Sequencing: the post-sink and shutdown paths PRE-increment
        ``_sequence_number`` (0 -> 1 on first fire), so the baseline's
        sequence 0 never collides; the manager's duplicate-sequence guard
        is the backstop. The resume path must NOT call this — it rebases
        onto the persisted sequence via :meth:`rebase_sequence`.

        Errors propagate deliberately: a run that cannot persist its
        baseline cannot checkpoint at all, so it crashes before any source
        row is processed rather than running un-resumable.
        """
        gate = self._checkpoint_gate(action="run-start checkpoint")
        if gate is None:
            return
        _config, manager, graph = gate
        token = self._require_fence(run_id)

        manager.create_checkpoint(
            draft=self._build_checkpoint_draft(
                run_id=run_id,
                sequence_number=0,
                barrier_scalars=None,
                graph=graph,
            ),
            coordination_token=token,
        )

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
        gate = self._checkpoint_gate(action="checkpoint")
        if gate is None:
            return
        config, manager, graph = gate
        # Before the sequence increment: every-N runs fail closed even on
        # rows the frequency gate would skip.
        token = self._require_fence(run_id)

        self._sequence_number += 1

        # RuntimeCheckpointConfig.frequency is an int:
        # - 1 = every_row
        # - 0 = aggregation_only
        # - N = every N rows
        frequency = config.frequency
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
            manager.create_checkpoint(
                draft=self._build_checkpoint_draft(
                    run_id=run_id,
                    sequence_number=self._sequence_number,
                    barrier_scalars=barrier_scalars,
                    graph=graph,
                ),
                coordination_token=token,
            )

    def make_checkpoint_after_sink_factory(
        self,
        run_id: str,
        barrier_scalars_source: BarrierScalarsSource,
    ) -> _CheckpointFactory:
        """Create a per-sink checkpoint-PROGRESS callback factory.

        Returns a factory that, given a sink_node_id, produces a callback
        invoked after each token is durably written to that sink. The callback
        records checkpoint progress ONLY — scheduler terminalization used to
        share this callback but is now a separate lifecycle composed at the
        sink-write call site (``SinkFlushCoordinator.flush_and_write_sinks``,
        elspeth-107a29d02e). Used by both the normal execution path and resume.

        Depends on the narrow :class:`BarrierScalarsSource` slice of the
        processor rather than the broad ``RowProcessorHandle``.
        """

        coordinator = self

        class CheckpointProgressCallback:
            """Record checkpoint progress after each durable sink write.

            One of the two lifecycles that used to share the post-sink callback
            (elspeth-107a29d02e); the scheduler-terminalization sibling lives in
            sink_flush.py. Progress checkpoints are created eagerly per call, so
            ``flush`` is a no-op — the explicit flush boundary exists only to
            satisfy the :class:`CheckpointAfterSinkCallback` protocol shared with
            the batched terminalization callback.
            """

            def __call__(self, token: TokenInfo) -> None:
                # token identity is not needed for checkpoint progress (it was
                # only consumed by the terminalization lifecycle, now split out).
                del token
                coordinator.maybe_checkpoint(
                    run_id=run_id,
                    barrier_scalars=barrier_scalars_source.get_barrier_scalars(),
                )

            def flush(self) -> None:
                # Progress checkpoints are written eagerly per call; nothing is
                # batched, so there is no deferred work to flush.
                return

        def factory(sink_node_id: str) -> CheckpointAfterSinkCallback:
            # sink_node_id is the factory's per-sink discriminator; the callback
            # itself no longer persists a per-sink anchor (F2).
            del sink_node_id
            return CheckpointProgressCallback()

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
        gate = self._checkpoint_gate(action="shutdown checkpoint")
        if gate is None:
            return
        _config, manager, graph = gate
        token = self._require_fence(run_id)

        self._sequence_number += 1
        manager.create_checkpoint(
            draft=self._build_checkpoint_draft(
                run_id=run_id,
                sequence_number=self._sequence_number,
                barrier_scalars=loop_ctx.processor.get_barrier_scalars(),
                graph=graph,
            ),
            coordination_token=token,
        )

    def delete_checkpoints(self, run_id: str) -> None:
        """Delete all checkpoints for a run after successful completion.

        Args:
            run_id: Run to clean up checkpoints for
        """
        if self._checkpoint_manager is not None:
            token = self._require_fence(run_id)
            self._checkpoint_manager.delete_checkpoints(run_id, coordination_token=token)
