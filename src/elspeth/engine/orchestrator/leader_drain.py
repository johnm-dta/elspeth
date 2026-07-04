"""LeaderDrainCoordinator: fresh-run phase sequencing for the leader worker.

Extracted from ``Orchestrator._execute_run`` (filigree elspeth-9e71ae82a4).
The facade keeps a thin ``_execute_run`` delegator (characterization tests
drive it directly) and injects ``register_graph_nodes_and_edges`` per call as
a bound method, so tests that stub that method on the orchestrator instance
keep intercepting the GRAPH phase.

Owns the run-body ordering: run-start checkpoint -> graph registration ->
context init -> transform runtime preflights -> sequential multi-source
ingest -> bounded peer-lease wait -> unresolved-work invariant -> leader sink
flush -> follower pending-sink drain -> deferred invariant sweep -> final
progress + PROCESS completion. The bounded-poll mechanics themselves live in
:class:`~elspeth.engine.orchestrator.leader_follower_drain.LeaderFollowerDrain`.

Behaviour-preserving: exception translation (``GracefulShutdownError``
re-raised, everything else wrapped in ``_RunFailedWithPartialResultError``
with live partial counters), the ``cleanup_plugins`` finally arm, and the
RUNNING-status return (the run-lifecycle wrapper owns terminal finalize) are
unchanged. Tests that previously patched
``…orchestrator.core.run_transform_runtime_preflights`` or
``…orchestrator.core.time.sleep`` / ``.monotonic`` patch this module instead.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from elspeth.contracts import RunStatus
from elspeth.contracts.cli import ProgressEvent
from elspeth.contracts.config import RuntimeRetryConfig
from elspeth.contracts.errors import (
    GracefulShutdownError,
    OrchestrationInvariantError,
)
from elspeth.contracts.events import PhaseCompleted, PipelinePhase
from elspeth.engine.orchestrator.cleanup import cleanup_plugins
from elspeth.engine.orchestrator.leader_follower_drain import LeaderFollowerDrain
from elspeth.engine.orchestrator.outcomes import accumulate_row_outcomes
from elspeth.engine.orchestrator.runtime_preflight import run_transform_runtime_preflights
from elspeth.engine.orchestrator.types import (
    ExecutionCounters,
    LoopContext,
    LoopResult,
    _RunFailedWithPartialResultError,
)
from elspeth.engine.retry import RetryManager

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable

    from elspeth.contracts.coordination import CoordinationToken
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.config import ElspethSettings
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.events import EventBusProtocol
    from elspeth.core.landscape.factory import RecorderFactory
    from elspeth.engine.orchestrator.checkpointing import CheckpointCoordinator
    from elspeth.engine.orchestrator.run_context_factory import RunContextFactory
    from elspeth.engine.orchestrator.sink_flush import SinkFlushCoordinator
    from elspeth.engine.orchestrator.source_iteration import SourceIterationDriver
    from elspeth.engine.orchestrator.types import (
        GraphArtifacts,
        PipelineConfig,
        RunResult,
    )

    RegisterGraphNodesAndEdges = Callable[
        [RecorderFactory, str, PipelineConfig, ExecutionGraph],
        GraphArtifacts,
    ]


class LeaderDrainCoordinator:
    """Owns the leader's source/process/sink phase sequencing for a fresh run."""

    def __init__(
        self,
        *,
        events: EventBusProtocol,
        checkpoints: CheckpointCoordinator,
        context_factory: RunContextFactory,
        sink_flush: SinkFlushCoordinator,
        source_driver: SourceIterationDriver,
    ) -> None:
        self._events = events
        self._checkpoints = checkpoints
        self._context_factory = context_factory
        self._sink_flush = sink_flush
        self._source_driver = source_driver

    def execute_run(
        self,
        factory: RecorderFactory,
        run_id: str,
        config: PipelineConfig,
        graph: ExecutionGraph,
        settings: ElspethSettings | None = None,
        *,
        payload_store: PayloadStore,
        shutdown_event: threading.Event | None = None,
        coordination_token: CoordinationToken | None = None,
        check_coordination_latch: Callable[[], None] | None = None,
        register_graph_nodes_and_edges: RegisterGraphNodesAndEdges,
    ) -> RunResult:
        """Execute the run using the execution graph.

        Orchestrates the four phases: graph registration, context initialization,
        source+process loop, sink writes. Returns RunStatus.RUNNING — the public
        run() wrapper transitions to COMPLETED after finalize_run().

        Parameters
        ----------
        check_coordination_latch:
            Optional callable forwarded to the per-source processing loop.
            Pass ``RunHeartbeatThread.check_and_raise`` from the enclosing
            ``run()`` method so the drain loop surfaces
            :class:`~elspeth.contracts.errors.RunWorkerEvictedError` at each
            row boundary when the heartbeat thread detects seat deposition.
            ``None`` disables latch polling (non-coordinated runs).
        register_graph_nodes_and_edges:
            The GRAPH-phase entry point. The facade passes its bound
            ``Orchestrator._register_graph_nodes_and_edges`` delegator so
            instance-level stubs on the orchestrator keep intercepting.
        """
        self._checkpoints.set_active_graph(graph)

        # F1 design D4: sequence-0 run-start checkpoint. Written before any
        # source iteration so every checkpointing-enabled run carries a
        # topology baseline; a run with NO checkpoint row then genuinely
        # predates run-start checkpointing or ran with checkpointing
        # disabled (can_resume's missing-baseline refusal, Task 3.2).
        # The resume path does NOT write this — it rebases onto the
        # persisted sequence (ResumeCoordinator.resume -> rebase_sequence).
        # Failures propagate: no baseline means the run cannot checkpoint.
        self._checkpoints.checkpoint_run_start(run_id)

        # 1. Register graph nodes and edges
        artifacts = register_graph_nodes_and_edges(factory, run_id, config, graph)

        # 2. Initialize context + processor
        run_ctx = self._context_factory.initialize_run_context(
            factory,
            run_id,
            config,
            graph,
            settings,
            artifacts,
            payload_store,
            shutdown_event=shutdown_event,
            coordination_token=coordination_token,
        )
        preflight_retry_manager = RetryManager(RuntimeRetryConfig.from_settings(settings.retry)) if settings is not None else None
        try:
            run_transform_runtime_preflights(
                factory,
                run_id,
                config,
                run_ctx.ctx,
                retry_manager=preflight_retry_manager,
                shutdown_event=shutdown_event,
            )
        except BaseException:
            cleanup_plugins(config, run_ctx.ctx, include_source=True)
            raise

        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={name: [] for name in config.sinks},
            processor=run_ctx.processor,
            ctx=run_ctx.ctx,
            config=config,
            agg_transform_lookup=run_ctx.agg_transform_lookup,
            coalesce_executor=run_ctx.coalesce_executor,
            coalesce_node_map=run_ctx.coalesce_node_map,
        )

        try:
            # 3. Source + Process phase. This is sequential multi-source ingest:
            # each declared source is iterated in turn with the active
            # ``SourceProtocol`` passed explicitly into the loop.
            # YAML declaration order is the determinism anchor for cross-source
            # ``ingest_sequence`` assignment. The scheduler's concurrency
            # contract is worker-token concurrency, not concurrent source iteration.
            # Per ADR-025 §1, the prior synthetic
            # ``replace(config, source=..., sources={...})`` per-iteration
            # config-mutation pattern is deleted.
            loop_result: LoopResult | None = None
            source_items = tuple(artifacts.source_id_map.items())
            for source_ordinal, (source_name, source_id) in enumerate(source_items):
                active_source = config.sources[source_name]
                source_loop_ctx = LoopContext(
                    counters=loop_ctx.counters,
                    pending_tokens=loop_ctx.pending_tokens,
                    processor=loop_ctx.processor,
                    ctx=loop_ctx.ctx,
                    config=config,
                    agg_transform_lookup=loop_ctx.agg_transform_lookup,
                    coalesce_executor=loop_ctx.coalesce_executor,
                    coalesce_node_map=loop_ctx.coalesce_node_map,
                )
                loop_result = self._source_driver.run_main_processing_loop(
                    source_loop_ctx,
                    factory,
                    run_id,
                    source_id,
                    artifacts.edge_map,
                    active_source_name=source_name,
                    active_source=active_source,
                    shutdown_event=shutdown_event,
                    flush_end_of_input=source_ordinal == len(source_items) - 1,
                    check_coordination_latch=check_coordination_latch,
                )
                if loop_result.interrupted:
                    break

            if loop_result is None:
                raise OrchestrationInvariantError("Pipeline has no sources to process")

            # 4b-pre. ADR-030 multi-worker: BEFORE checking for unresolved scheduler
            # work, wait for peer followers to finish any in-flight LEASED items.
            # A follower that claimed an item just before the leader's source loop
            # exited will still hold a LEASED row (pending_sink_name IS NULL) that
            # has_unresolved_scheduler_work() counts as unresolved.  Waiting here
            # ensures followers complete their claims (LEASED → PENDING_SINK) before
            # the invariant check fires.  In the single-worker case,
            # has_peer_active_leases() returns False immediately and the loop is
            # skipped.
            #
            # The wait is BOUNDED by the active item lease plus the stall budget:
            # a live peer gets the full lease window to finish legitimate work,
            # while a wedged-but-alive peer that keeps its lease refreshed must
            # not hang a deposed/interrupted leader forever.
            # Each iteration also (a) honours the in-scope shutdown_event (SIGINT)
            # and check_coordination_latch (epoch deposition) so the leader can break
            # out, and (b) drives lease maintenance so a peer that DIED mid-lease is
            # actively reaped to READY within the liveness window rather than waiting
            # out the full item TTL.  On timeout we fall through to the existing
            # has_unresolved_scheduler_work raise, which names the still-leased peers.
            def _shutdown_during_wait() -> GracefulShutdownError:
                """Build the canonical INTERRUPTED signal from the live counters.

                A SIGINT observed while waiting on / draining peer work must surface
                the same resumable GracefulShutdownError the source loop and sink
                flush raise (counter-bearing, run_id-scoped), not a bare message.
                """
                _c = loop_ctx.counters
                return GracefulShutdownError(
                    rows_processed=_c.rows_processed,
                    run_id=run_id,
                    rows_succeeded=_c.rows_succeeded,
                    rows_failed=_c.rows_failed,
                    rows_quarantined=_c.rows_quarantined,
                    rows_routed_success=_c.rows_routed_success,
                    rows_routed_failure=_c.rows_routed_failure,
                    routed_destinations=dict(_c.routed_destinations),
                )

            _leader_follower_drain = LeaderFollowerDrain(
                processor=loop_ctx.processor,
                run_id=run_id,
                shutdown_event=shutdown_event,
                check_coordination_latch=check_coordination_latch,
                make_shutdown_error=_shutdown_during_wait,
            )

            if not loop_result.interrupted:
                _leader_follower_drain.wait_for_peer_leases()

            if not loop_result.interrupted and loop_ctx.processor.has_unresolved_scheduler_work():
                active_work = "; ".join(loop_ctx.processor.summarize_unresolved_scheduler_work()) or "<unknown>"
                raise OrchestrationInvariantError(
                    f"Run '{run_ctx.processor.run_id}' left non-terminal scheduler work after final source flush. "
                    "Blocked or READY scheduler state must be resolved before run completion. "
                    f"Active scheduler work: {active_work}."
                )

            # 4. Sink writes — outside source_load track_operation context.
            # Each sink write has its own track_operation (sink_write) in SinkExecutor.
            self._sink_flush.flush_and_write_sinks(
                factory,
                run_id,
                loop_ctx,
                artifacts.sink_id_map,
                artifacts.edge_map,
                loop_result.interrupted,
                on_token_written_factory=self._checkpoints.make_checkpoint_after_sink_factory(run_id, run_ctx.processor),
                scheduler_terminalizer=run_ctx.processor,
            )

            # 4b. ADR-030 multi-worker: after the leader's own sink writes are done
            # (all leader PENDING_SINK rows are now TERMINAL), drain the PENDING_SINK
            # rows produced by follower workers and write those to sinks.  In the
            # single-worker case, has_scheduled_work() returns False immediately
            # (no follower PENDING_SINK rows exist) and the loop body never runs.
            #
            # LOOP (not single-pass): a follower that transitions LEASED→PENDING_SINK
            # AFTER drain_scheduled_work's claim loop returns would leave a late
            # PENDING_SINK row.  We re-drain until BOTH has_peer_active_leases() (no
            # peer still holds an in-flight lease that could become PENDING_SINK) and
            # has_scheduled_work() (no undrained PENDING_SINK row) are false.  The
            # wait is bounded by the same liveness-multiple deadline as 4b-pre and
            # honours shutdown/deposition each iteration; on timeout we stop draining
            # and rely on complete_run's quiescence arm as the backstop (a residual
            # PENDING_SINK row makes the run FAIL loudly — the correct, resumable
            # exactly-once fail-direction — never a silent lost row).
            if not loop_result.interrupted:

                def _drain_and_flush() -> bool:
                    # The drain->clear->accumulate->flush coupling stays HERE (not in the
                    # drain coordinator): write_pending_to_sinks does NOT consume
                    # pending_tokens entries, so the leader's already-written tokens
                    # remain. Accumulating follower results on top of them would re-write
                    # every leader token -> the node_states UNIQUE constraint. Returns
                    # True when this pass drained+flushed follower work so the coordinator
                    # re-checks immediately without sleeping.
                    follower_results = loop_ctx.processor.drain_scheduled_work(loop_ctx.ctx)
                    if not follower_results:
                        return False
                    for _sink_list in loop_ctx.pending_tokens.values():
                        _sink_list.clear()
                    accumulate_row_outcomes(follower_results, loop_ctx.counters, loop_ctx.pending_tokens)
                    self._sink_flush.flush_and_write_sinks(
                        factory,
                        run_id,
                        loop_ctx,
                        artifacts.sink_id_map,
                        artifacts.edge_map,
                        interrupted_by_shutdown=False,
                        on_token_written_factory=self._checkpoints.make_checkpoint_after_sink_factory(run_id, run_ctx.processor),
                        scheduler_terminalizer=run_ctx.processor,
                    )
                    return True

                _leader_follower_drain.drain_pending_sink_work(_drain_and_flush)

            # ADR-019 Phase 4: deferred cross-table invariant sweep.
            #
            # AUDIT-TRAIL DURABILITY CONTRACT:
            # 1. The run is still RUNNING here; successful terminal finalization
            #    has not executed.
            # 2. If this raises AuditIntegrityError, the exception propagates to
            #    the public run() failure ceremony, which finalizes the run as
            #    FAILED and re-raises the original exception.
            # 3. The offending token_outcomes/batch rows are evidence and are
            #    not deleted by the sweep.
            # 4. GracefulShutdownError skips this naturally because sink flush
            #    raises before this post-sink call site.
            factory.data_flow.sweep_deferred_invariants_or_crash(run_id)

            # 5. Final progress + PROCESS phase completion — AFTER sink writes
            # so these events reflect concrete, durable results. On shutdown,
            # flush_and_write_sinks raises GracefulShutdownError before we
            # reach here — matching the pre-extraction behavior where the
            # shutdown raise prevented progress/PhaseCompleted emission.
            progress_interval = 100
            current_time = time.perf_counter()
            time_since_last_progress = current_time - loop_result.last_progress_time
            if loop_ctx.counters.rows_processed % progress_interval != 0 or time_since_last_progress >= 1.0:
                elapsed = current_time - loop_result.start_time
                self._events.emit(
                    ProgressEvent(
                        rows_processed=loop_ctx.counters.rows_processed,
                        # elspeth-5069612f3c — rows_routed split. See the
                        # earlier emitter in source_iteration.py for the full
                        # rationale; this final-progress emission must match so
                        # the last streaming snapshot before terminal events
                        # agrees with the forthcoming CompletedData payload.
                        rows_succeeded=loop_ctx.counters.rows_succeeded,
                        rows_failed=loop_ctx.counters.rows_failed,
                        rows_quarantined=loop_ctx.counters.rows_quarantined,
                        rows_routed_success=loop_ctx.counters.rows_routed_success,
                        rows_routed_failure=loop_ctx.counters.rows_routed_failure,
                        elapsed_seconds=elapsed,
                    )
                )

            self._events.emit(PhaseCompleted(phase=PipelinePhase.PROCESS, duration_seconds=current_time - loop_result.phase_start))
        except GracefulShutdownError:
            raise
        except Exception as exc:
            raise _RunFailedWithPartialResultError(
                original_error=exc,
                partial_result=loop_ctx.counters.to_run_result(run_id, status=RunStatus.FAILED),
            ) from exc

        finally:
            cleanup_plugins(config, run_ctx.ctx, include_source=True)

        self._checkpoints.set_active_graph(None)
        return loop_ctx.counters.to_run_result(run_id, status=RunStatus.RUNNING)
