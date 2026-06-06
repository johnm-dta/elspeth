"""SourceIterationDriver: source iteration and row-loop cluster.

Extracted from Orchestrator (core.py) — these methods own the source
loading, per-row dispatching, quarantine handling, field resolution
recording, and post-loop finalization phases of a pipeline run.

Dependencies held by this driver:
- ``_events``: EventBusProtocol for emitting progress and phase lifecycle events
- ``_span_factory``: SpanFactory for source-load spans
- ``_ceremony``: RunCeremony for telemetry and phase-error emission
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from itertools import chain
from typing import TYPE_CHECKING

from elspeth.contracts import PendingOutcome, SourceRow
from elspeth.contracts.cli import ProgressEvent
from elspeth.contracts.enums import NodeStateStatus, RoutingMode, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import (
    ExecutionError,
    OrchestrationInvariantError,
    SourceQuarantineReason,
)
from elspeth.contracts.events import (
    FieldResolutionApplied,
    PhaseAction,
    PhaseChanged,
    PhaseCompleted,
    PhaseStarted,
    PipelinePhase,
    RowCreated,
)
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.types import NodeID
from elspeth.core.canonical import sanitize_for_canonical, stable_hash
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.operations import track_operation
from elspeth.engine.orchestrator.aggregation import (
    check_aggregation_timeouts,
    flush_remaining_aggregation_buffers,
)
from elspeth.engine.orchestrator.ceremony import RunCeremony
from elspeth.engine.orchestrator.landscape_registration import record_schema_contract
from elspeth.engine.orchestrator.outcomes import (
    accumulate_row_outcomes,
    flush_coalesce_pending,
    handle_coalesce_timeouts,
)
from elspeth.engine.orchestrator.types import (
    ExecutionCounters,
    LoopContext,
    LoopResult,
    PipelineConfig,
    RouteValidationError,
)
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from elspeth.core.events import EventBusProtocol


class SourceIterationDriver:
    """Owns the source-iteration / row-loop cluster of an Orchestrator run.

    Extracted from Orchestrator — holds the 7 methods that span source loading,
    per-row dispatching, quarantine handling, field resolution recording, and
    post-loop finalization.
    """

    _PROGRESS_ROW_INTERVAL = 100
    _PROGRESS_TIME_INTERVAL = 5.0  # seconds

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

    def handle_quarantine_row(
        self,
        factory: RecorderFactory,
        run_id: str,
        source_id: NodeID,
        source_item: SourceRow,
        row_index: int,
        edge_map: Mapping[tuple[NodeID, str], str],
        loop_ctx: LoopContext,
    ) -> None:
        """Handle a quarantined source row: route directly to configured sink.

        Accesses loop_ctx.processor for token creation and loop_ctx.counters
        for incrementing quarantine count. Appends to loop_ctx.pending_tokens.

        This method performs the complete quarantine workflow:
        1. Validate quarantine destination exists
        2. Sanitize data for canonical JSON
        3. Create quarantine token
        4. Record source node_state (FAILED)
        5. Record DIVERT routing_event
        6. Emit telemetry
        7. Compute error_hash
        8. Append to pending_tokens with PendingOutcome
        """

        config = loop_ctx.config
        counters = loop_ctx.counters
        processor = loop_ctx.processor
        pending_tokens = loop_ctx.pending_tokens

        # Route quarantined row to configured sink
        # Per CLAUDE.md: plugin bugs must crash, no silent drops
        quarantine_sink = source_item.quarantine_destination

        # Validate destination exists - crash on plugin bug
        if not quarantine_sink:
            raise RouteValidationError(
                f"Source '{config.source.name}' yielded quarantined row "
                f"(row_index={row_index}) with missing quarantine_destination. "
                f"This is a plugin bug: quarantined rows MUST specify a destination. "
                f"Use SourceRow.quarantined(row, error, destination) factory method."
            )
        if quarantine_sink not in config.sinks:
            raise RouteValidationError(
                f"Source '{config.source.name}' yielded quarantined row "
                f"(row_index={row_index}) with invalid quarantine_destination='{quarantine_sink}'. "
                f"No sink named '{quarantine_sink}' exists. "
                f"Available sinks: {sorted(config.sinks.keys())}. "
                f"This is a plugin bug: quarantine_destination must match "
                f"source._on_validation_failure='{config.source._on_validation_failure}'."
            )

        # Destination validated. Source quarantine is a FAILURE lifecycle with
        # a quarantine reporting subset, so bump both counters.
        counters.rows_quarantined += 1
        counters.rows_failed += 1
        validation_error_id = loop_ctx.ctx.pop_pending_quarantine_validation_error_id(source_item.row)
        # Sanitize quarantine data at Tier-3 boundary: replace non-finite
        # floats (NaN, Infinity) with None so downstream canonical JSON
        # and stable_hash operations succeed. The quarantine_error records
        # what was originally wrong with the data.
        # SourceRow is frozen — create a new instance with sanitized row data.
        source_item = replace(source_item, row=sanitize_for_canonical(source_item.row))

        # Create a token for the quarantined row using specialized method
        # (quarantine rows don't have contracts - they failed validation)
        quarantine_token = processor.token_manager.create_quarantine_token(
            run_id=run_id,
            source_node_id=source_id,
            row_index=row_index,
            source_row=source_item,
            validation_error_id=validation_error_id,
        )

        # Record source node_state (step_index=0) for quarantine audit lineage.
        # Status is FAILED because the source validation rejected this row.
        quarantine_data = source_item.row if isinstance(source_item.row, dict) else {"_raw": source_item.row}
        quarantine_error_msg = source_item.quarantine_error or "unknown_validation_error"
        source_state = factory.execution.begin_node_state(
            token_id=quarantine_token.token_id,
            node_id=source_id,
            run_id=run_id,
            step_index=0,
            input_data=quarantine_data,
            quarantined=True,
        )
        factory.execution.complete_node_state(
            state_id=source_state.state_id,
            status=NodeStateStatus.FAILED,
            duration_ms=0,
            error=ExecutionError(
                exception=quarantine_error_msg,
                exception_type="ValidationError",
            ),
        )

        # Record DIVERT routing_event for the quarantine edge.
        # The __quarantine__ edge MUST exist — DAG creates it in
        # the source quarantine edge block of from_plugin_instances().
        quarantine_edge_key = (source_id, "__quarantine__")
        try:
            quarantine_edge_id = edge_map[quarantine_edge_key]
        except KeyError as exc:
            raise OrchestrationInvariantError(
                f"Quarantine row reached orchestrator but no __quarantine__ "
                f"DIVERT edge exists in DAG for source '{source_id}'. "
                f"This is a DAG construction bug — "
                f"on_validation_failure should have created a DIVERT edge "
                f"in from_plugin_instances()."
            ) from exc
        factory.execution.record_routing_event(
            state_id=source_state.state_id,
            edge_id=quarantine_edge_id,
            mode=RoutingMode.DIVERT,
            reason=SourceQuarantineReason(
                quarantine_error=quarantine_error_msg,
            ),
        )

        # Emit RowCreated telemetry AFTER Landscape recording succeeds.
        # source_item.row was already sanitized for Tier-3 non-canonical values
        # (NaN/Infinity -> None) above, so stable_hash gives a single deterministic
        # semantics for content_hash. No repr_hash fallback: after sanitization the
        # only residual stable_hash failure is a structurally non-serializable type,
        # which is a plugin-contract violation that must surface, not be masked by a
        # second, divergent hash function recorded under the same field name.
        quarantine_content_hash = stable_hash(source_item.row)
        self._ceremony.emit_telemetry(
            RowCreated(
                timestamp=datetime.now(UTC),
                run_id=run_id,
                row_id=quarantine_token.row_id,
                token_id=quarantine_token.token_id,
                content_hash=quarantine_content_hash,
            )
        )

        # Compute error_hash for QUARANTINED outcome audit trail
        # Per CLAUDE.md: every row must reach exactly one terminal state
        # Do NOT record outcome here — record after sink durability in SinkExecutor.write()
        quarantine_error_hash = hashlib.sha256(quarantine_error_msg.encode()).hexdigest()[:16]

        # Pass PendingOutcome with error_hash - outcome recorded after sink durability
        pending_tokens[quarantine_sink].append(
            (
                quarantine_token,
                PendingOutcome(
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.QUARANTINED_AT_SOURCE,
                    error_hash=quarantine_error_hash,
                ),
            )
        )

    def record_field_resolution(
        self,
        factory: RecorderFactory,
        run_id: str,
        config: PipelineConfig,
    ) -> bool:
        """Record source field resolution mapping if available.

        Called once per run — on first iteration (after generator body executes)
        or post-loop for empty sources (header-only files where the loop never
        executes but the source computed field resolution).

        Returns:
            True if field resolution was recorded, False otherwise.
        """
        field_resolution = config.source.get_field_resolution()
        if field_resolution is None:
            return False

        resolution_mapping, normalization_version = field_resolution
        factory.run_lifecycle.record_source_field_resolution(
            run_id=run_id,
            resolution_mapping=resolution_mapping,
            normalization_version=normalization_version,
        )
        # Emit telemetry AFTER Landscape succeeds
        self._ceremony.emit_telemetry(
            FieldResolutionApplied(
                timestamp=datetime.now(UTC),
                run_id=run_id,
                source_plugin=config.source.name,
                field_count=len(resolution_mapping),
                normalization_version=normalization_version,
                resolution_mapping=resolution_mapping,
            )
        )
        return True

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
        source_operation_id: str,
        field_resolution_recorded: bool,
        schema_contract_recorded: bool,
        *,
        interrupted_by_shutdown: bool,
    ) -> None:
        """Post-loop work after source iteration completes or is interrupted.

        Restores operation_id, optionally flushes end-of-source aggregation and
        coalesce state, and records deferred field resolution / schema contract.

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

        if not interrupted_by_shutdown:
            # CRITICAL: Flush remaining aggregation buffers only at true end-of-source.
            # A graceful shutdown is resumable and must preserve buffered state
            # instead of forcing an END_OF_SOURCE flush.
            if config.aggregation_settings:
                # NOTE: Aggregation-flushed tokens are NOT checkpointed here.
                # They go into pending_tokens and are checkpointed only after
                # SinkExecutor.write() achieves sink durability, via the
                # checkpoint_after_sink callback.
                flush_result = flush_remaining_aggregation_buffers(
                    config=config,
                    processor=processor,
                    ctx=ctx,
                    pending_tokens=pending_tokens,
                )
                counters.accumulate_flush_result(flush_result)

                # TERMINAL GUARANTEE: After end-of-source flush, all aggregation
                # buffers must be empty. Any remaining tokens would be silently
                # lost — never reaching a terminal state in the audit trail.
                for agg_node_id_str in config.aggregation_settings:
                    remaining = processor.get_aggregation_buffer_count(NodeID(agg_node_id_str))
                    if remaining > 0:
                        raise OrchestrationInvariantError(
                            f"Aggregation buffer for node '{agg_node_id_str}' still has "
                            f"{remaining} tokens after end-of-source flush. "
                            f"These tokens would never reach a terminal state."
                        )

            # Flush pending coalesce operations only when the source is actually exhausted.
            if coalesce_executor is not None:
                flush_coalesce_pending(
                    coalesce_executor=coalesce_executor,
                    coalesce_node_map=coalesce_node_map,
                    processor=processor,
                    ctx=ctx,
                    counters=counters,
                    pending_tokens=pending_tokens,
                )

        # Record field resolution for empty sources (header-only files).
        # For sources with rows, this was recorded inside the loop on first iteration.
        if not field_resolution_recorded:
            self.record_field_resolution(factory, run_id, config)

        # Record schema contract for runs with no valid source rows.
        # In-loop recording happens on first VALID row. For all-invalid
        # or empty inputs, that branch never executes.
        if not schema_contract_recorded:
            record_schema_contract(factory, run_id, source_id, config, ctx)

    def load_source_with_events(
        self,
        config: PipelineConfig,
        run_id: str,
        ctx: PluginContext,
    ) -> Iterator[SourceRow]:
        """Execute SOURCE phase: emit lifecycle events, load source, handle errors.

        SOURCE phase is complete when this method returns. Errors during load()
        (file not found, auth failure) are emitted as PhaseError before re-raising.
        """

        phase_start = time.perf_counter()
        self._events.emit(PhaseStarted(phase=PipelinePhase.SOURCE, action=PhaseAction.INITIALIZING, target=config.source.name))
        self._ceremony.emit_telemetry(
            PhaseChanged(
                timestamp=datetime.now(UTC),
                run_id=run_id,
                phase=PipelinePhase.SOURCE,
                action=PhaseAction.INITIALIZING,
            )
        )

        try:
            with self._span_factory.source_span(config.source.name):
                source_iterator = iter(config.source.load(ctx))
                try:
                    first_row = next(source_iterator)
                except StopIteration:
                    self._events.emit(PhaseCompleted(phase=PipelinePhase.SOURCE, duration_seconds=time.perf_counter() - phase_start))
                    return iter(())
        except Exception as e:
            self._ceremony.emit_phase_error(PipelinePhase.SOURCE, e, target=config.source.name)
            raise

        self._events.emit(PhaseCompleted(phase=PipelinePhase.SOURCE, duration_seconds=time.perf_counter() - phase_start))
        return chain((first_row,), source_iterator)

    def run_main_processing_loop(
        self,
        loop_ctx: LoopContext,
        factory: RecorderFactory,
        run_id: str,
        source_id: NodeID,
        edge_map: Mapping[tuple[NodeID, str], str],
        *,
        shutdown_event: threading.Event | None = None,
    ) -> LoopResult:
        """Run the main processing loop: source iteration, quarantine, transform, flush.

        Owns the track_operation(source_load) context — everything inside executes
        within source_load operation attribution. Sink writes happen OUTSIDE this
        method in _flush_and_write_sinks() (separate track_operation per sink).

        Final progress emission and PhaseCompleted(PROCESS) are emitted by the
        caller AFTER sink writes, using the timing state in LoopResult.
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
            input_data={"source_plugin": config.source.name},
        ) as source_op_handle:
            # Generator-based sources execute on next() — restore operation_id
            # before each iteration so external calls are attributed to source_load
            source_operation_id = source_op_handle.operation.operation_id

            source_iterator = self.load_source_with_events(config, run_id, ctx)
            self.restore_source_iteration_context(
                ctx,
                source_id=source_id,
                source_operation_id=source_operation_id,
            )

            # Deferred recording flags — field resolution after first iteration,
            # schema contract after first VALID row. If begin_run already stored
            # a contract (FIXED mode), skip re-recording.
            field_resolution_recorded = False
            schema_contract_recorded = factory.run_lifecycle.get_run_contract(run_id) is not None

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
                for row_index, source_item in enumerate(source_iterator):
                    counters.rows_processed += 1

                    # Record field resolution on first iteration (generators execute body on first next())
                    if not field_resolution_recorded:
                        field_resolution_recorded = True
                        self.record_field_resolution(factory, run_id, config)

                    # Quarantine path — route directly to sink, skip normal processing
                    if source_item.is_quarantined:
                        self.handle_quarantine_row(
                            factory,
                            run_id,
                            source_id,
                            source_item,
                            row_index,
                            edge_map,
                            loop_ctx,
                        )
                        quarantine_sink = source_item.quarantine_destination
                        if quarantine_sink is not None and loop_ctx.pending_tokens[quarantine_sink]:
                            loop_ctx.last_token_id = loop_ctx.pending_tokens[quarantine_sink][-1][0].token_id
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
                        if shutdown_event is not None and shutdown_event.is_set():
                            interrupted_by_shutdown = True
                            break
                        continue

                    # Record schema contract on first VALID row (quarantined rows don't populate contract)
                    if not schema_contract_recorded and record_schema_contract(factory, run_id, source_id, config, ctx):
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
                        row_index=row_index,
                        source_row=source_item,
                        transforms=config.transforms,
                        ctx=ctx,
                    )
                    if results:
                        loop_ctx.last_token_id = results[-1].token.token_id
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
                    source_operation_id,
                    field_resolution_recorded,
                    schema_contract_recorded,
                    interrupted_by_shutdown=interrupted_by_shutdown,
                )

            except Exception as e:
                self._ceremony.emit_phase_error(PipelinePhase.PROCESS, e, target=config.source.name)
                raise

        return LoopResult(
            interrupted=interrupted_by_shutdown,
            start_time=start_time,
            phase_start=phase_start,
            last_progress_time=last_progress_time,
        )
