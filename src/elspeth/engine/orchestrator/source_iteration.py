"""SourceIterationDriver: source iteration and row-loop coordinator.

Extracted from Orchestrator (core.py); this driver coordinates the source
loading, per-row dispatch, and post-loop finalization phases of a pipeline
run. Two focused concerns live in collaborators the driver delegates to:

- ``QuarantineRouter`` (``quarantine_router.py``) — routes a validation-failed
  source row directly to its configured sink.
- ``SourceLifecycleRecorder`` (``source_lifecycle_recorder.py``) — records the
  source field-resolution mapping and run_source lifecycle evidence.

Idle-timeout aggregation/coalesce flushing runs on an ``IdleTimeoutPump``
(``idle_timeout_pump.py``); the driver binds the run-scoped flush closure into
it. The driver keeps the main row loop, progress emission, source-scoped
context restoration, and end-of-input finalization.

Dependencies held by this driver:
- ``_events``: EventBusProtocol for emitting progress and phase lifecycle events
- ``_span_factory``: SpanFactory for source-load spans
- ``_ceremony``: RunCeremony for telemetry and phase-error emission
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from elspeth.contracts import SourceRow
from elspeth.contracts.cli import ProgressEvent
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.events import (
    PhaseAction,
    PhaseChanged,
    PhaseCompleted,
    PhaseStarted,
    PipelinePhase,
)
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import RunSourceLifecycleState
from elspeth.core.operations import track_operation
from elspeth.engine.orchestrator.aggregation import (
    check_aggregation_timeouts,
    run_end_of_input_barrier_flush,
)
from elspeth.engine.orchestrator.ceremony import RunCeremony
from elspeth.engine.orchestrator.idle_timeout_pump import IdleTimeoutPump
from elspeth.engine.orchestrator.landscape_registration import record_schema_contract
from elspeth.engine.orchestrator.outcomes import (
    accumulate_row_outcomes,
    handle_coalesce_timeouts,
)
from elspeth.engine.orchestrator.quarantine_router import QuarantineRouter
from elspeth.engine.orchestrator.source_lifecycle_recorder import SourceLifecycleRecorder
from elspeth.engine.orchestrator.types import (
    AggNodeEntry,
    ExecutionCounters,
    LoopContext,
    LoopResult,
    PipelineConfig,
)
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from elspeth.contracts import SourceProtocol
    from elspeth.core.events import EventBusProtocol


class SourceIterationDriver:
    """Coordinate the source-iteration / row-loop cluster of an Orchestrator run.

    Owns source loading, the per-row dispatch loop, progress emission, and
    post-loop finalization. Quarantine routing and source lifecycle/field
    recording are delegated to the ``QuarantineRouter`` and
    ``SourceLifecycleRecorder`` collaborators built in ``__init__``.
    """

    _PROGRESS_ROW_INTERVAL = 100
    _PROGRESS_TIME_INTERVAL = 5.0  # seconds
    _SOURCE_IDLE_POLL_INTERVAL_SECONDS = 0.01

    def __init__(
        self,
        *,
        events: EventBusProtocol,
        span_factory: SpanFactory,
        ceremony: RunCeremony,
    ) -> None:
        self._events = events
        self._span_factory = span_factory
        self._ceremony = ceremony
        self._quarantine_router = QuarantineRouter(ceremony=ceremony)
        self._lifecycle_recorder = SourceLifecycleRecorder(ceremony=ceremony)

    def restore_source_iteration_context(
        self,
        ctx: PluginContext,
        *,
        source_id: NodeID,
        source_operation_id: str,
    ) -> None:
        """Restore source-scoped context before source generator code resumes.

        Source plugins run partly in `load(ctx)` setup and partly on each
        generator `next()` call. Transform execution mutates the shared
        PluginContext with transform-scoped node/state identity, so we must
        restore the source identity before the next generator step or any
        source-side validation/error recording will be misattributed.
        """
        ctx.node_id = source_id
        ctx.operation_id = source_operation_id

    def _requires_idle_aggregation_polling(self, config: PipelineConfig) -> bool:
        """Return True when the pipeline has time-sensitive buffered work."""
        has_aggregation_timeout = any(
            settings.trigger.has_timeout or settings.trigger.has_condition for settings in config.aggregation_settings.values()
        )
        has_coalesce_timeout = any(settings.timeout_seconds is not None for settings in config.coalesce_settings)
        return has_aggregation_timeout or has_coalesce_timeout

    def _idle_timeout_context(self, source_ctx: PluginContext) -> PluginContext:
        """Create an isolated context for idle timeout work.

        Source generators keep executing on the orchestrator/source thread
        while idle timeout checks run in a helper thread. The timeout path
        must therefore never clear or overwrite the source context's
        node/operation identity.
        """
        return PluginContext(
            run_id=source_ctx.run_id,
            config=source_ctx.config,
            landscape=source_ctx.landscape,
            payload_store=source_ctx.payload_store,
            rate_limit_registry=source_ctx.rate_limit_registry,
            concurrency_config=source_ctx.concurrency_config,
            shutdown_event=source_ctx.shutdown_event,
            contract=source_ctx.contract,
            telemetry_emit=source_ctx.telemetry_emit,
        )

    def _process_idle_timeout_flushes(
        self,
        loop_ctx: LoopContext,
        *,
        agg_transform_lookup: Mapping[str, AggNodeEntry],
        coalesce_node_map: Mapping[CoalesceName, NodeID],
        source_id: NodeID,
        source_operation_id: str,
    ) -> None:
        """Flush time-sensitive aggregation/coalesce state while no source row is ready."""
        ctx = self._idle_timeout_context(loop_ctx.ctx)
        # Match the existing pre-row timeout path: a source item has not been
        # handed to transforms, so transform state should be established by the
        # transform executor instead of inheriting the source operation id.
        ctx.operation_id = None
        timeout_result = check_aggregation_timeouts(
            config=loop_ctx.config,
            processor=loop_ctx.processor,
            ctx=ctx,
            pending_tokens=loop_ctx.pending_tokens,
            agg_transform_lookup=dict(agg_transform_lookup),
        )
        loop_ctx.counters.accumulate_flush_result(timeout_result)

        if loop_ctx.coalesce_executor is not None:
            handle_coalesce_timeouts(
                coalesce_executor=loop_ctx.coalesce_executor,
                coalesce_node_map=dict(coalesce_node_map),
                processor=loop_ctx.processor,
                ctx=ctx,
                counters=loop_ctx.counters,
                pending_tokens=loop_ctx.pending_tokens,
            )

    def _build_idle_timeout_pump(
        self,
        loop_ctx: LoopContext,
        *,
        agg_transform_lookup: Mapping[str, AggNodeEntry],
        coalesce_node_map: Mapping[CoalesceName, NodeID],
        source_id: NodeID,
        source_operation_id: str,
    ) -> IdleTimeoutPump:
        """Bind the idle-flush closure for this run/source into a pump.

        The poll interval is read at build time (inside the run / delegate
        call), NOT at import time, so per-instance overrides of
        ``_SOURCE_IDLE_POLL_INTERVAL_SECONDS`` in tests are honoured.
        """

        def flush_idle_timeouts() -> None:
            self._process_idle_timeout_flushes(
                loop_ctx,
                agg_transform_lookup=agg_transform_lookup,
                coalesce_node_map=coalesce_node_map,
                source_id=source_id,
                source_operation_id=source_operation_id,
            )

        return IdleTimeoutPump(
            flush=flush_idle_timeouts,
            poll_interval=self._SOURCE_IDLE_POLL_INTERVAL_SECONDS,
        )

    def _next_source_item_with_idle_timeout_flushes(
        self,
        source_iterator: Iterator[SourceRow],
        loop_ctx: LoopContext,
        *,
        agg_transform_lookup: Mapping[str, AggNodeEntry],
        coalesce_node_map: Mapping[CoalesceName, NodeID],
        source_id: NodeID,
        source_operation_id: str,
        pump: IdleTimeoutPump | None = None,
    ) -> SourceRow:
        """Fetch the next source row while periodically flushing idle timeouts.

        SourceProtocol lifecycle and iterator advancement stay on this caller
        thread. Only timeout/coalesce maintenance runs on the pump's persistent
        worker thread while the caller is blocked inside
        ``next(source_iterator)``; the pump's end-of-fetch park handshake
        serializes those flushes against this thread before it touches loop
        state again (see ``idle_timeout_pump.py`` for the invariants).

        ``pump`` is the run-scoped ``IdleTimeoutPump`` started by
        ``run_main_processing_loop``. When ``None`` (direct calls), a one-shot
        pump is built, started, and stopped around this single fetch —
        behaviourally identical to the historical thread-per-fetch poller.
        """
        self.restore_source_iteration_context(
            loop_ctx.ctx,
            source_id=source_id,
            source_operation_id=source_operation_id,
        )

        source_row: SourceRow | None
        if pump is not None:
            source_row = pump.fetch(lambda: next(source_iterator))
        else:
            one_shot = self._build_idle_timeout_pump(
                loop_ctx,
                agg_transform_lookup=agg_transform_lookup,
                coalesce_node_map=coalesce_node_map,
                source_id=source_id,
                source_operation_id=source_operation_id,
            )
            one_shot.start()
            try:
                source_row = one_shot.fetch(lambda: next(source_iterator))
            finally:
                one_shot.stop()

        if source_row is None:
            raise OrchestrationInvariantError("Source iterator returned no row, no StopIteration, and no exception.")
        return source_row

    def maybe_emit_progress(
        self,
        counters: ExecutionCounters,
        start_time: float,
        last_progress_time: float,
    ) -> float:
        """Emit a ProgressEvent if row count or time threshold is met.

        Hybrid timing: emit on first row, every 100 rows, or every 5 seconds.
        Used in both quarantine and valid-row paths.

        Returns:
            Updated last_progress_time (unchanged if no emission).
        """
        progress_interval = self._PROGRESS_ROW_INTERVAL
        progress_time_interval = self._PROGRESS_TIME_INTERVAL
        current_time = time.perf_counter()
        time_since_last_progress = current_time - last_progress_time
        should_emit = (
            counters.rows_processed == 1  # First row - immediate feedback
            or counters.rows_processed % progress_interval == 0  # Every N rows
            or time_since_last_progress >= progress_time_interval  # Every M seconds
        )
        if should_emit:
            elapsed = current_time - start_time
            self._events.emit(
                ProgressEvent(
                    rows_processed=counters.rows_processed,
                    # elspeth-5069612f3c — rows_routed split. Each terminal
                    # bucket is emitted on its own field so downstream
                    # consumers (web ProgressData → SSE → frontend, CLI
                    # progress formatter) can mirror the terminal-state
                    # taxonomy. The pre-split fold (rows_succeeded +=
                    # rows_routed_success) silently conflated MOVE-routed
                    # rows into rows_succeeded and dropped DIVERT entirely,
                    # leaving the in-flight signal incompatible with the
                    # terminal Pydantic schemas (CompletedData, etc.).
                    rows_succeeded=counters.rows_succeeded,
                    rows_failed=counters.rows_failed,
                    rows_quarantined=counters.rows_quarantined,
                    rows_routed_success=counters.rows_routed_success,
                    rows_routed_failure=counters.rows_routed_failure,
                    elapsed_seconds=elapsed,
                )
            )
            return current_time
        return last_progress_time

    def finalize_source_iteration(
        self,
        loop_ctx: LoopContext,
        factory: RecorderFactory,
        run_id: str,
        source_id: NodeID,
        active_source_name: str,
        source_operation_id: str,
        recorded_field_resolution: tuple[Mapping[str, str], str | None] | None,
        schema_contract_recorded: bool,
        *,
        source_exhausted: bool,
        interrupted_by_shutdown: bool,
        flush_end_of_input: bool,
        active_source: SourceProtocol,
    ) -> None:
        """Post-loop work after source iteration completes or is interrupted.

        Restores operation_id and records source-local deferred field resolution
        / schema contract. Aggregation and coalesce flushes are run-global
        end-of-input work: in multi-source runs they must execute only after the
        last source is exhausted, not after each individual source.

        On graceful shutdown we intentionally skip end-of-source flushes. A
        shutdown stops after the current row; it must not synthesize
        END_OF_SOURCE aggregation outputs or force pending coalesces to resolve.
        """
        config = loop_ctx.config
        ctx = loop_ctx.ctx
        processor = loop_ctx.processor
        counters = loop_ctx.counters
        pending_tokens = loop_ctx.pending_tokens
        coalesce_executor = loop_ctx.coalesce_executor
        coalesce_node_map = dict(loop_ctx.coalesce_node_map)

        # CRITICAL: Restore source-scoped identity before post-loop flushes.
        # On normal loop exit, the restore at end-of-iteration ensures
        # node_id == source_id and operation_id == source_operation_id.
        # On shutdown break, that restore is SKIPPED — both fields still
        # hold transform-scoped values. Aggregation and coalesce flushes
        # can trigger transforms that make external calls — those must be
        # attributed to source_load, not orphaned or misattributed.
        # Idempotent on normal exit; essential on shutdown-break path.
        self.restore_source_iteration_context(
            ctx,
            source_id=source_id,
            source_operation_id=source_operation_id,
        )

        # Record deferred source metadata before EOF engine work. A crash in
        # aggregation/coalesce flushing after StopIteration must be resumable as
        # source-exhausted engine work, not indistinguishable from mid-source
        # interruption.
        #
        # Unconditional re-record (elspeth-fb108a77c9): sparse sources extend
        # the field-resolution union on later rows, so the first-row write is
        # provisional; the overwrite UPDATE here lands the final union — also
        # on shutdown interruption (the audit reflects fields observed up to
        # that point). Skipped internally when the mapping is unchanged, so
        # fixed-header sources see no second write or telemetry event.
        self._lifecycle_recorder.record_field_resolution(
            factory,
            run_id,
            active_source=active_source,
            previously_recorded=recorded_field_resolution,
        )

        if not schema_contract_recorded:
            record_schema_contract(factory, run_id, source_id, ctx, active_source=active_source)

        if source_exhausted and not interrupted_by_shutdown:
            self._lifecycle_recorder.record_run_source_lifecycle(
                factory,
                run_id,
                source_id,
                active_source_name,
                active_source,
                RunSourceLifecycleState.EXHAUSTED,
            )

        if not interrupted_by_shutdown and flush_end_of_input:
            # CRITICAL: Flush remaining barriers only at true end-of-input.
            # Multi-source runs may feed shared downstream queues, aggregations,
            # and coalesce barriers. Per-source completion is not a global EOF.
            # A graceful shutdown is resumable and must preserve buffered state
            # instead of forcing an END_OF_SOURCE flush.
            #
            # ADR-030 §D steps 2-3 (slice 3): the flush is gated on journal
            # quiescence and runs as an intake -> trigger evaluation -> flush
            # loop until no BLOCKED barrier holds remain.
            # NOTE: Aggregation-flushed tokens are NOT checkpointed here.
            # They go into pending_tokens and are checkpointed only after
            # SinkExecutor.write() achieves sink durability, via the
            # checkpoint_after_sink callback. Coalesce flushing happens only
            # when all configured sources are exhausted — per-source flushing
            # would resolve shared barriers before later source roots have had
            # a chance to contribute.
            run_end_of_input_barrier_flush(
                config=config,
                processor=processor,
                ctx=ctx,
                counters=counters,
                pending_tokens=pending_tokens,
                coalesce_executor=coalesce_executor,
                coalesce_node_map=coalesce_node_map,
            )

    def load_source_with_events(
        self,
        run_id: str,
        ctx: PluginContext,
        *,
        active_source: SourceProtocol,
    ) -> Iterator[SourceRow]:
        """Execute SOURCE phase: emit lifecycle events, load source, handle errors.

        Source iteration is lazy so the caller can wrap the first ``next()``
        in idle-timeout polling. Errors during load() (file not found, auth
        failure) are emitted as PhaseError before re-raising.
        """

        def _source_iter() -> Iterator[SourceRow]:
            phase_start = time.perf_counter()
            self._events.emit(PhaseStarted(phase=PipelinePhase.SOURCE, action=PhaseAction.INITIALIZING, target=active_source.name))
            self._ceremony.emit_telemetry(
                PhaseChanged(
                    timestamp=datetime.now(UTC),
                    run_id=run_id,
                    phase=PipelinePhase.SOURCE,
                    action=PhaseAction.INITIALIZING,
                )
            )

            try:
                with self._span_factory.source_span(active_source.name):
                    source_iterator = iter(active_source.load(ctx))
                    try:
                        first_row = next(source_iterator)
                    except StopIteration:
                        self._events.emit(PhaseCompleted(phase=PipelinePhase.SOURCE, duration_seconds=time.perf_counter() - phase_start))
                        return
            except Exception as e:
                self._ceremony.emit_phase_error(PipelinePhase.SOURCE, e, target=active_source.name)
                raise

            self._events.emit(PhaseCompleted(phase=PipelinePhase.SOURCE, duration_seconds=time.perf_counter() - phase_start))
            yield first_row
            yield from source_iterator

        return _source_iter()

    def run_main_processing_loop(
        self,
        loop_ctx: LoopContext,
        factory: RecorderFactory,
        run_id: str,
        source_id: NodeID,
        edge_map: Mapping[tuple[NodeID, str], str],
        *,
        active_source_name: str,
        active_source: SourceProtocol,
        shutdown_event: threading.Event | None = None,
        flush_end_of_input: bool = True,
        check_coordination_latch: Callable[[], None] | None = None,
    ) -> LoopResult:
        """Run the main processing loop: source iteration, quarantine, transform, flush.

        Owns the track_operation(source_load) context — everything inside executes
        within source_load operation attribution. Sink writes happen OUTSIDE this
        method in _flush_and_write_sinks() (separate track_operation per sink).

        Final progress emission and PhaseCompleted(PROCESS) are emitted by the
        caller AFTER sink writes, using the timing state in LoopResult.

        Parameters
        ----------
        check_coordination_latch:
            Optional zero-argument callable that raises
            :class:`~elspeth.contracts.errors.RunWorkerEvictedError` if the
            heartbeat thread has detected seat deposition or registry eviction.
            Called at the same boundary as the ``shutdown_event`` check — once
            per row, after row processing completes.  Pass
            ``RunHeartbeatThread.check_and_raise`` here.  ``None`` (the
            default) disables latch polling (e.g. non-coordinated single-shot
            runs or tests that do not start a heartbeat thread).
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

        start_time = time.perf_counter()
        last_progress_time = start_time

        # source_load operation covers the entire source consumption lifecycle
        with track_operation(
            recorder=factory.execution,
            run_id=run_id,
            node_id=source_id,
            operation_type="source_load",
            ctx=ctx,
            input_data={"source_plugin": active_source.name},
        ) as source_op_handle:
            # Generator-based sources execute on next() — restore operation_id
            # before each iteration so external calls are attributed to source_load
            source_operation_id = source_op_handle.operation.operation_id

            self.restore_source_iteration_context(
                ctx,
                source_id=source_id,
                source_operation_id=source_operation_id,
            )
            self._lifecycle_recorder.record_run_source_lifecycle(
                factory,
                run_id,
                source_id,
                active_source_name,
                active_source,
                RunSourceLifecycleState.LOADING,
            )

            source_iterator = self.load_source_with_events(run_id, ctx, active_source=active_source)
            use_idle_polling = self._requires_idle_aggregation_polling(config)
            idle_pump: IdleTimeoutPump | None = None
            if use_idle_polling:
                # ONE persistent idle-flush worker for the whole run; the two
                # fetch call sites below hand it each per-row next() and its
                # stop() is guaranteed on every exit path by the finally
                # bracketing the loop (elspeth-735df9576d).
                idle_pump = self._build_idle_timeout_pump(
                    loop_ctx,
                    agg_transform_lookup=agg_transform_lookup,
                    coalesce_node_map=coalesce_node_map,
                    source_id=source_id,
                    source_operation_id=source_operation_id,
                )
                idle_pump.start()
            try:
                source_exhausted = False
                pending_source_item: SourceRow | None = None
                try:
                    if use_idle_polling:
                        pending_source_item = self._next_source_item_with_idle_timeout_flushes(
                            source_iterator,
                            loop_ctx,
                            agg_transform_lookup=agg_transform_lookup,
                            coalesce_node_map=coalesce_node_map,
                            source_id=source_id,
                            source_operation_id=source_operation_id,
                            pump=idle_pump,
                        )
                    else:
                        pending_source_item = next(source_iterator)
                except StopIteration:
                    source_exhausted = True

                # Deferred recording flags — field resolution after first iteration,
                # schema contract after first VALID row. Always start false so
                # source-scoped run_sources contracts are backfilled even when the
                # legacy run-level singleton already exists.
                field_resolution_recorded = False
                recorded_field_resolution: tuple[Mapping[str, str], str | None] | None = None
                schema_contract_recorded = False

                # PROCESS phase
                phase_start = time.perf_counter()
                self._events.emit(PhaseStarted(phase=PipelinePhase.PROCESS, action=PhaseAction.PROCESSING))
                self._ceremony.emit_telemetry(
                    PhaseChanged(
                        timestamp=datetime.now(UTC),
                        run_id=run_id,
                        phase=PipelinePhase.PROCESS,
                        action=PhaseAction.PROCESSING,
                    )
                )

                interrupted_by_shutdown = False
                try:
                    source_row_index = 0
                    while True:
                        if pending_source_item is not None:
                            source_item = pending_source_item
                            pending_source_item = None
                        else:
                            try:
                                if use_idle_polling:
                                    source_item = self._next_source_item_with_idle_timeout_flushes(
                                        source_iterator,
                                        loop_ctx,
                                        agg_transform_lookup=agg_transform_lookup,
                                        coalesce_node_map=coalesce_node_map,
                                        source_id=source_id,
                                        source_operation_id=source_operation_id,
                                        pump=idle_pump,
                                    )
                                else:
                                    source_item = next(source_iterator)
                            except StopIteration:
                                source_exhausted = True
                                break

                        current_source_row_index = source_row_index
                        source_row_index += 1
                        if source_item.source_row_index is None:
                            raise OrchestrationInvariantError(
                                f"Source '{active_source.name}' yielded SourceRow without source_row_index at "
                                f"loop row_index={current_source_row_index}. Source row identity must be source-authored."
                            )
                        source_identity_index = source_item.source_row_index
                        ingest_sequence = counters.rows_processed
                        counters.rows_processed += 1

                        # Record field resolution on first iteration (generators execute body on first next()).
                        # Provisional: the finalizer re-records the union at EOF (elspeth-fb108a77c9).
                        if not field_resolution_recorded:
                            field_resolution_recorded = True
                            recorded_field_resolution = self._lifecycle_recorder.record_field_resolution(
                                factory, run_id, active_source=active_source
                            )

                        # Quarantine path — route directly to sink, skip normal processing
                        if source_item.is_quarantined:
                            self._quarantine_router.route(
                                factory,
                                run_id,
                                source_id,
                                source_item,
                                current_source_row_index,
                                source_identity_index,
                                ingest_sequence,
                                edge_map,
                                loop_ctx,
                                active_source=active_source,
                            )
                            last_progress_time = self.maybe_emit_progress(
                                counters,
                                start_time,
                                last_progress_time,
                            )
                            self.restore_source_iteration_context(
                                ctx,
                                source_id=source_id,
                                source_operation_id=source_operation_id,
                            )
                            if check_coordination_latch is not None:
                                check_coordination_latch()
                            if shutdown_event is not None and shutdown_event.is_set():
                                interrupted_by_shutdown = True
                                break
                            continue

                        # Record schema contract on first VALID row (quarantined rows don't populate contract)
                        if not schema_contract_recorded and record_schema_contract(
                            factory,
                            run_id,
                            source_id,
                            ctx,
                            active_source=active_source,
                        ):
                            schema_contract_recorded = True

                        # Clear operation_id — source item is fetched, transforms set their own state_id
                        ctx.operation_id = None

                        # Check aggregation timeouts BEFORE processing (flush OLD batch first)
                        timeout_result = check_aggregation_timeouts(
                            config=config,
                            processor=processor,
                            ctx=ctx,
                            pending_tokens=pending_tokens,
                            agg_transform_lookup=agg_transform_lookup,
                        )
                        counters.accumulate_flush_result(timeout_result)

                        results = processor.process_row(
                            row_index=current_source_row_index,
                            source_row=source_item,
                            transforms=config.transforms,
                            ctx=ctx,
                            source_node_id=source_id,
                            source_plugin=active_source,
                            source_on_success=active_source.on_success,
                            source_row_index=source_identity_index,
                            ingest_sequence=ingest_sequence,
                        )
                        accumulate_row_outcomes(results, counters, pending_tokens)

                        # Check coalesce timeouts after each row
                        if coalesce_executor is not None:
                            handle_coalesce_timeouts(
                                coalesce_executor=coalesce_executor,
                                coalesce_node_map=coalesce_node_map,
                                processor=processor,
                                ctx=ctx,
                                counters=counters,
                                pending_tokens=pending_tokens,
                            )

                        last_progress_time = self.maybe_emit_progress(
                            counters,
                            start_time,
                            last_progress_time,
                        )

                        # ADR-030 §A.3 / §C.2: latch check — raises RunWorkerEvictedError
                        # if the heartbeat thread detected seat deposition or registry
                        # eviction.  Checked at the same per-row boundary as the
                        # shutdown_event so the drain loop exits cleanly without
                        # emitting further work.  The latch is an optimization on top of
                        # the epoch/membership fences (§C.2 last sentence); the fences
                        # independently refuse any write a deposed leader attempts, but
                        # the latch surfaces the condition proactively between writes.
                        if check_coordination_latch is not None:
                            check_coordination_latch()

                        # Graceful shutdown — current row fully processed, safe to stop
                        if shutdown_event is not None and shutdown_event.is_set():
                            interrupted_by_shutdown = True
                            break

                        # Restore operation_id for next iteration (generators execute on next())
                        self.restore_source_iteration_context(
                            ctx,
                            source_id=source_id,
                            source_operation_id=source_operation_id,
                        )

                    # Post-loop: restore operation_id, flush aggregation/coalesce, record deferred state
                    self.finalize_source_iteration(
                        loop_ctx,
                        factory,
                        run_id,
                        source_id,
                        active_source_name,
                        source_operation_id,
                        recorded_field_resolution,
                        schema_contract_recorded,
                        source_exhausted=source_exhausted,
                        interrupted_by_shutdown=interrupted_by_shutdown,
                        flush_end_of_input=flush_end_of_input,
                        active_source=active_source,
                    )
                    if interrupted_by_shutdown:
                        self._lifecycle_recorder.record_run_source_lifecycle(
                            factory,
                            run_id,
                            source_id,
                            active_source_name,
                            active_source,
                            RunSourceLifecycleState.INTERRUPTED,
                        )
                    elif not source_exhausted:
                        self._lifecycle_recorder.record_run_source_lifecycle(
                            factory,
                            run_id,
                            source_id,
                            active_source_name,
                            active_source,
                            RunSourceLifecycleState.LOADED,
                        )

                except Exception as e:
                    self._ceremony.emit_phase_error(PipelinePhase.PROCESS, e, target=active_source.name)
                    raise
            finally:
                if idle_pump is not None:
                    idle_pump.stop()

        return LoopResult(
            interrupted=interrupted_by_shutdown,
            start_time=start_time,
            phase_start=phase_start,
            last_progress_time=last_progress_time,
        )
