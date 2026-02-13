# src/elspeth/engine/executors/sink.py
"""SinkExecutor - wraps sink.write() with artifact recording."""

import logging
import time
from collections.abc import Callable

from elspeth.contracts import (
    Artifact,
    ExecutionError,
    NodeStateOpen,
    PendingOutcome,
    TokenInfo,
)
from elspeth.contracts.enums import NodeStateStatus
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.core.landscape import LandscapeRecorder
from elspeth.core.operations import track_operation
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.protocols import SinkProtocol

logger = logging.getLogger(__name__)


class SinkExecutor:
    """Executes sinks with artifact recording.

    Wraps sink.write() to:
    1. Create node_state for EACH token - this is how COMPLETED terminal state is derived
    2. Time the operation
    3. Record artifact produced by sink
    4. Complete all token states
    5. Emit OpenTelemetry span

    CRITICAL: Every token reaching a sink gets a node_state. This is the audit
    proof that the row reached its terminal state. The COMPLETED terminal state
    is DERIVED from having a completed node_state at a sink node.

    Note: Unlike TransformExecutor/GateExecutor/AggregationExecutor, SinkExecutor
    does NOT use StepResolver. Sinks are not DAG processing nodes — their step is
    always max(processing_steps) + 1, computed by RowProcessor.resolve_sink_step()
    and passed as step_in_pipeline by the orchestrator. This is intentional: sinks
    exist after all processing nodes and have a fixed, deterministic step position.

    Example:
        executor = SinkExecutor(recorder, span_factory, run_id)
        artifact = executor.write(
            sink=my_sink,
            tokens=tokens_to_write,
            ctx=ctx,
            step_in_pipeline=5,
        )
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        run_id: str,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            run_id: Run identifier for artifact registration
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id

    def _complete_states_failed(
        self,
        *,
        states: list[tuple[TokenInfo, NodeStateOpen]],
        duration_ms: float,
        error: ExecutionError,
    ) -> None:
        """Complete all opened sink states as FAILED."""
        per_token_ms = duration_ms / len(states)
        for _, state in states:
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.FAILED,
                duration_ms=per_token_ms,
                error=error,
            )

    def write(
        self,
        sink: SinkProtocol,
        tokens: list[TokenInfo],
        ctx: PluginContext,
        step_in_pipeline: int,
        *,
        sink_name: str,
        pending_outcome: PendingOutcome,
        on_token_written: Callable[[TokenInfo], None] | None = None,
    ) -> Artifact | None:
        """Write tokens to sink with artifact recording.

        CRITICAL: Creates a node_state for EACH token written AND records
        token outcomes. Both records are created AFTER sink.flush()
        to ensure they only exist when data is durably written.

        This is the ONLY place terminal outcomes should be recorded for sink-bound
        tokens. Recording here (not in the orchestrator processing loop) ensures the
        token outcome contract is honored:
        - Invariant 3: "COMPLETED/ROUTED implies the token has a completed sink node_state"
        - Invariant 4: "Completed sink node_state implies a terminal token_outcome"

        Fix: P1-2026-01-31-quarantine-outcome-before-durability
        Uses PendingOutcome to carry error_hash for QUARANTINED outcomes through to
        recording, ensuring outcomes are only recorded after sink durability.

        Args:
            sink: Sink plugin to write to
            tokens: Tokens to write (may be empty)
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)
            sink_name: Name of the sink (for token_outcome recording)
            pending_outcome: PendingOutcome containing outcome and optional error_hash.
                    Required - all sink-bound tokens must have their outcome recorded.
            on_token_written: Optional callback called for each token after successful write.
                             Used for post-sink checkpointing.

        Returns:
            Artifact if tokens were written, None if empty

        Raises:
            Exception: Re-raised from sink.write() after recording failure
        """
        if not tokens:
            return None

        # Extract dicts from PipelineRow for sink write
        # Sinks serialize data to external formats - they receive plain dicts
        # Contract metadata is preserved in Landscape audit trail, not sink output
        rows = [t.row_data.to_dict() for t in tokens]

        # Create node_state for EACH token - this is how we derive COMPLETED terminal state
        # Sink must have node_id assigned by orchestrator before execution
        if sink.node_id is None:
            raise OrchestrationInvariantError(f"Sink '{sink.name}' executed without node_id - orchestrator bug")
        sink_node_id: str = sink.node_id

        states: list[tuple[TokenInfo, NodeStateOpen]] = []
        for token in tokens:
            # Extract dict from PipelineRow for Landscape recording
            # Landscape stores raw dicts, not PipelineRow objects
            input_dict = token.row_data.to_dict()
            state = self._recorder.begin_node_state(
                token_id=token.token_id,
                node_id=sink_node_id,
                run_id=ctx.run_id,
                step_index=step_in_pipeline,
                input_data=input_dict,
            )
            states.append((token, state))
        # Synchronize context contract to the sink-bound tokens.
        # Sinks (e.g., headers: original) lazily capture ctx.contract during write().
        # For mixed batches, merge contracts to preserve all available header lineage.
        contract_merge_start = time.perf_counter()
        try:
            batch_contract = tokens[0].row_data.contract
            for token in tokens[1:]:
                batch_contract = batch_contract.merge(token.row_data.contract)
        except Exception as e:
            merge_duration_ms = (time.perf_counter() - contract_merge_start) * 1000
            merge_error: ExecutionError = {
                "exception": str(e),
                "type": type(e).__name__,
                "phase": "contract_merge",
            }
            self._complete_states_failed(
                states=states,
                duration_ms=merge_duration_ms,
                error=merge_error,
            )
            raise
        ctx.contract = batch_contract

        # CRITICAL: Clear state_id before entering operation context.
        # The ctx.state_id may still be set from the last transform that processed
        # these tokens. Sinks use operation_id for call attribution, and having both
        # state_id AND operation_id set would trigger the XOR constraint violation.
        ctx.state_id = None
        # Note: operation call_index is handled by LandscapeRecorder.allocate_operation_call_index()

        # Wrap sink I/O in operation for external call tracking
        # External calls during sink.write() are attributed to the operation (not token states)
        # The track_operation context manager sets ctx.operation_id automatically
        with track_operation(
            recorder=self._recorder,
            run_id=self._run_id,
            node_id=sink_node_id,
            operation_type="sink_write",
            ctx=ctx,
            input_data={"sink_plugin": sink.name, "row_count": len(tokens)},
        ) as handle:
            # Execute sink write with timing and span
            # P2-2026-01-21: Pass all token_ids being written for accurate attribution
            # P2-2026-01-21: Pass node_id for disambiguation when multiple sinks exist
            sink_token_ids = [t.token_id for t in tokens]
            with self._spans.sink_span(
                sink.name,
                node_id=sink_node_id,
                token_ids=sink_token_ids,
            ):
                start = time.perf_counter()
                try:
                    artifact_info = sink.write(rows, ctx)
                    duration_ms = (time.perf_counter() - start) * 1000
                except Exception as e:
                    duration_ms = (time.perf_counter() - start) * 1000
                    error: ExecutionError = {
                        "exception": str(e),
                        "type": type(e).__name__,
                    }
                    self._complete_states_failed(
                        states=states,
                        duration_ms=duration_ms,
                        error=error,
                    )
                    raise

            # CRITICAL: Flush sink to ensure durability BEFORE checkpointing
            # If this fails, we want to crash - can't checkpoint non-durable data
            # But first we must complete node_states as FAILED to maintain audit integrity
            try:
                sink.flush()
            except Exception as e:
                # Flush failed - complete all node_states as FAILED before crashing
                # Without this, states remain OPEN permanently (audit integrity violation)
                flush_error: ExecutionError = {
                    "exception": str(e),
                    "type": type(e).__name__,
                    "phase": "flush",
                }
                flush_duration_ms = (time.perf_counter() - start) * 1000
                self._complete_states_failed(
                    states=states,
                    duration_ms=flush_duration_ms,
                    error=flush_error,
                )
                raise

            # Set output data on operation handle for audit trail
            handle.output_data = {
                "artifact_path": artifact_info.path_or_uri,
                "content_hash": artifact_info.content_hash,
            }

        # Complete all token states - status=NodeStateStatus.COMPLETED means they reached terminal
        # Output is the row data that was written to the sink, plus artifact reference
        # Amortize batch write time across tokens so aggregation math is correct:
        # sum(per-token duration) ~ actual batch time, not N * batch time
        per_token_ms = duration_ms / len(tokens)
        for token, state in states:
            # Extract dict from PipelineRow for Landscape output recording
            output_dict = token.row_data.to_dict()
            sink_output = {
                "row": output_dict,
                "artifact_path": artifact_info.path_or_uri,
                "content_hash": artifact_info.content_hash,
            }
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.COMPLETED,
                output_data=sink_output,
                duration_ms=per_token_ms,
            )

        # Register artifact (linked to first state for audit lineage)
        first_state = states[0][1]

        artifact = self._recorder.register_artifact(
            run_id=self._run_id,
            state_id=first_state.state_id,
            sink_node_id=sink_node_id,  # Already validated above
            artifact_type=artifact_info.artifact_type,
            path=artifact_info.path_or_uri,
            content_hash=artifact_info.content_hash,
            size_bytes=artifact_info.size_bytes,
        )

        # Record token outcomes AFTER sink durability is achieved
        # This is the ONLY correct place to record outcomes for sink-bound tokens - after:
        # 1. sink.write() succeeded
        # 2. sink.flush() succeeded (data is durable)
        # 3. node_states are marked COMPLETED
        # 4. artifact is registered
        # Recording here ensures Invariant 3: "COMPLETED/ROUTED implies completed sink node_state"
        #
        # Fix: P1-2026-01-31 - PendingOutcome carries error_hash for QUARANTINED outcomes
        # pending_outcome is REQUIRED - all sink-bound tokens must have outcomes recorded
        for token, _ in states:
            self._recorder.record_token_outcome(
                run_id=self._run_id,
                token_id=token.token_id,
                outcome=pending_outcome.outcome,
                error_hash=pending_outcome.error_hash,
                sink_name=sink_name,
            )

        # Call checkpoint callback for each token after successful write + flush
        # CRITICAL: Sink write + flush are durable - we CANNOT roll them back.
        # If checkpoint creation fails, we log the error but don't raise.
        # The sink artifact exists, but no checkpoint record → resume will replay
        # these rows → duplicate writes (acceptable for RC-1, see Bug #10 docs).
        if on_token_written is not None:
            for token in tokens:
                try:
                    on_token_written(token)
                except Exception as e:
                    # Sink write is durable, can't undo. Log error and continue.
                    # Operator must manually clean up checkpoint inconsistency.
                    logger.error(
                        "Checkpoint failed after durable sink write for token %s. "
                        "Sink artifact exists but no checkpoint record created. "
                        "Resume will replay this row (duplicate write). "
                        "Manual cleanup may be required. Error: %s",
                        token.token_id,
                        e,
                        exc_info=True,
                    )
                    # Don't raise - we can't undo the sink write

        return artifact
