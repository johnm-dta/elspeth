# src/elspeth/engine/executors/aggregation.py
"""AggregationExecutor - manages batch lifecycle with audit recording."""

import logging
import time
from typing import TYPE_CHECKING, Any

import structlog

from elspeth.contracts import (
    BatchPendingError,
    ExecutionError,
    PipelineRow,
    SchemaContract,
    TokenInfo,
)
from elspeth.contracts.enums import (
    BatchStatus,
    NodeStateStatus,
    TriggerType,
)
from elspeth.contracts.errors import OrchestrationInvariantError, PluginContractViolation
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.types import NodeID, StepResolver
from elspeth.core.canonical import stable_hash
from elspeth.core.config import AggregationSettings
from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.clock import DEFAULT_CLOCK
from elspeth.engine.spans import SpanFactory
from elspeth.engine.triggers import TriggerEvaluator
from elspeth.plugins.protocols import BatchTransformProtocol
from elspeth.plugins.results import TransformResult

if TYPE_CHECKING:
    from elspeth.engine.clock import Clock

logger = logging.getLogger(__name__)
slog = structlog.get_logger(__name__)

AGGREGATION_CHECKPOINT_VERSION = "3.0"


class AggregationExecutor:
    """Executes aggregations with batch tracking and audit recording.

    Manages the lifecycle of batches:
    1. Create batch on first accept (if _batch_id is None)
    2. Track batch members as rows are accepted
    3. Transition batch through states: draft -> executing -> completed/failed
    4. Reset _batch_id after flush for next batch

    CRITICAL: Terminal state CONSUMED_IN_BATCH is DERIVED from batch_members table,
    NOT stored in node_states.status (which is always "completed" for successful accepts).

    Example:
        executor = AggregationExecutor(recorder, span_factory, step_resolver, run_id)

        # Accept rows into batch
        result = executor.buffer_row(node_id, token)
        # Engine uses TriggerEvaluator to decide when to flush (WP-06)
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        step_resolver: StepResolver,
        run_id: str,
        *,
        aggregation_settings: dict[NodeID, AggregationSettings] | None = None,
        clock: "Clock | None" = None,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            step_resolver: Resolves NodeID to 1-indexed audit step position
            run_id: Run identifier for batch creation
            aggregation_settings: Map of node_id -> AggregationSettings for trigger evaluation
            clock: Optional clock for time access. Defaults to system clock.
                   Inject MockClock for deterministic testing.
        """
        self._recorder = recorder
        self._spans = span_factory
        self._step_resolver = step_resolver
        self._run_id = run_id
        self._clock = clock if clock is not None else DEFAULT_CLOCK
        self._member_counts: dict[str, int] = {}  # batch_id -> count for ordinals
        self._batch_ids: dict[NodeID, str | None] = {}  # node_id -> current batch_id
        self._aggregation_settings: dict[NodeID, AggregationSettings] = aggregation_settings or {}
        self._trigger_evaluators: dict[NodeID, TriggerEvaluator] = {}
        self._restored_states: dict[NodeID, dict[str, Any]] = {}  # node_id -> state

        # Engine-owned row buffers (node_id -> list of row dicts)
        self._buffers: dict[NodeID, list[dict[str, Any]]] = {}
        # Token tracking for audit trail (node_id -> list of TokenInfo)
        self._buffer_tokens: dict[NodeID, list[TokenInfo]] = {}

        # Create trigger evaluators for each configured aggregation
        for node_id, settings in self._aggregation_settings.items():
            self._trigger_evaluators[node_id] = TriggerEvaluator(settings.trigger, clock=self._clock)
            self._buffers[node_id] = []
            self._buffer_tokens[node_id] = []

    def buffer_row(
        self,
        node_id: NodeID,
        token: TokenInfo,
    ) -> None:
        """Buffer a row for aggregation.

        The engine owns the buffer. When trigger fires, buffered rows
        are passed to a batch-aware Transform.

        Args:
            node_id: Aggregation node ID
            token: Token with row data to buffer

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
                This prevents silent data loss where rows are buffered but no
                trigger evaluator exists to determine when to flush.
        """
        # Validate node is a configured aggregation (P2-2026-02-02: whitelist-reduction)
        # Without this check, rows could be buffered without a trigger evaluator,
        # meaning they'd sit in the buffer forever with no way to flush.
        if node_id not in self._aggregation_settings:
            raise OrchestrationInvariantError(
                f"buffer_row called for node '{node_id}' which is not in aggregation_settings. "
                f"Only configured aggregation nodes can buffer rows. "
                f"Configured nodes: {list(self._aggregation_settings.keys())}"
            )

        # Create batch on first row if needed
        # Note: We use node_id directly since we've validated it exists in aggregation_settings,
        # which means it was initialized in __init__ with buffers and trigger evaluator.
        if node_id not in self._batch_ids or self._batch_ids[node_id] is None:
            batch = self._recorder.create_batch(
                run_id=self._run_id,
                aggregation_node_id=node_id,
            )
            self._batch_ids[node_id] = batch.batch_id
            self._member_counts[batch.batch_id] = 0

        batch_id = self._batch_ids[node_id]
        assert batch_id is not None  # We just created it if it was None

        # Buffer the row - store dict (JSON-serializable for checkpoints)
        # TokenInfo.row_data is PipelineRow, extract dict for buffer
        self._buffers[node_id].append(token.row_data.to_dict())
        self._buffer_tokens[node_id].append(token)

        # Record batch membership for audit trail
        ordinal = self._member_counts[batch_id]
        self._recorder.add_batch_member(
            batch_id=batch_id,
            token_id=token.token_id,
            ordinal=ordinal,
        )
        self._member_counts[batch_id] = ordinal + 1

        # Update trigger evaluator - direct access since we validated node_id exists
        # in aggregation_settings, which guarantees a trigger evaluator was created
        self._trigger_evaluators[node_id].record_accept()

    def get_buffered_rows(self, node_id: NodeID) -> list[dict[str, Any]]:
        """Get currently buffered rows (does not clear buffer).

        Args:
            node_id: Aggregation node ID

        Returns:
            List of buffered row dicts (empty if no rows buffered yet)

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        # Validate node_id is a configured aggregation (P2-2026-02-02: whitelist-reduction)
        # This distinguishes "valid node, no rows yet" from "invalid node".
        if node_id not in self._aggregation_settings:
            raise OrchestrationInvariantError(
                f"get_buffered_rows called for node '{node_id}' which is not in aggregation_settings. "
                f"Configured nodes: {list(self._aggregation_settings.keys())}"
            )
        # Return empty list if no rows buffered yet (valid state for configured node)
        return list(self._buffers.get(node_id, []))

    def get_buffered_tokens(self, node_id: NodeID) -> list[TokenInfo]:
        """Get currently buffered tokens (does not clear buffer).

        Args:
            node_id: Aggregation node ID

        Returns:
            List of buffered TokenInfo objects (empty if no rows buffered yet)

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        # Validate node_id is a configured aggregation (P2-2026-02-02: whitelist-reduction)
        if node_id not in self._aggregation_settings:
            raise OrchestrationInvariantError(
                f"get_buffered_tokens called for node '{node_id}' which is not in aggregation_settings. "
                f"Configured nodes: {list(self._aggregation_settings.keys())}"
            )
        return list(self._buffer_tokens.get(node_id, []))

    def _get_buffered_data(self, node_id: NodeID) -> tuple[list[dict[str, Any]], list[TokenInfo]]:
        """Internal: Get buffered rows and tokens without clearing.

        IMPORTANT: This method does NOT record audit trail. Production code
        should use execute_flush() instead. This method is exposed for:
        - Testing buffer contents without triggering flush

        Args:
            node_id: Aggregation node ID

        Returns:
            Tuple of (buffered_rows, buffered_tokens)

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        # Validate node_id is a configured aggregation (P2-2026-02-02: whitelist-reduction)
        if node_id not in self._aggregation_settings:
            raise OrchestrationInvariantError(
                f"_get_buffered_data called for node '{node_id}' which is not in aggregation_settings. "
                f"Configured nodes: {list(self._aggregation_settings.keys())}"
            )
        rows = list(self._buffers.get(node_id, []))
        tokens = list(self._buffer_tokens.get(node_id, []))
        return rows, tokens

    def execute_flush(
        self,
        node_id: NodeID,
        transform: BatchTransformProtocol,
        ctx: PluginContext,
        trigger_type: TriggerType,
    ) -> tuple[TransformResult, list[TokenInfo], str]:
        """Execute a batch flush with full audit recording.

        This method:
        1. Transitions batch to "executing" with trigger reason
        2. Records node_state for the flush operation
        3. Executes the batch-aware transform
        4. Transitions batch to "completed" or "failed"
        5. Resets batch_id for next batch

        The step position in the DAG is resolved internally via StepResolver
        using node_id, rather than being passed as a parameter.

        Args:
            node_id: Aggregation node ID
            transform: Batch-aware transform plugin (must implement BatchTransformProtocol)
            ctx: Plugin context
            trigger_type: What triggered the flush (COUNT, TIMEOUT, END_OF_SOURCE, etc.)

        Returns:
            Tuple of (TransformResult with audit fields, list of consumed tokens, batch_id)

        Raises:
            Exception: Re-raised from transform.process() after recording failure
        """
        # Get batch_id - must exist if we're flushing
        batch_id = self._batch_ids.get(node_id)
        if batch_id is None:
            raise RuntimeError(f"No batch exists for node {node_id} - cannot flush")

        # Get buffered data - use direct access since batch existence implies buffers exist
        # (batches are created on first row buffered, so if batch_id exists, buffers must too)
        # If KeyError here, that indicates internal state corruption in buffer_row.
        try:
            buffered_rows = list(self._buffers[node_id])
            buffered_tokens = list(self._buffer_tokens[node_id])
        except KeyError as e:
            raise RuntimeError(
                f"Internal state corruption: batch_id exists for node '{node_id}' "
                f"but buffer is missing. batch_id={batch_id}, missing_key={e}"
            ) from e

        if not buffered_rows:
            raise RuntimeError(f"Cannot flush empty buffer for node {node_id}")

        # Defensive validation: buffer and tokens must be same length
        # This should never happen (checkpoint restore ensures they stay in sync)
        # but crashes explicitly if internal state is corrupted
        if len(buffered_rows) != len(buffered_tokens):
            raise RuntimeError(
                f"Internal state corruption in AggregationExecutor node '{node_id}': "
                f"buffer has {len(buffered_rows)} rows but tokens has {len(buffered_tokens)} entries. "
                f"These must always match. This indicates a bug in checkpoint "
                f"restore or buffer management."
            )

        # Use first token for node_state (represents the batch operation)
        representative_token = buffered_tokens[0]

        # Reconstruct PipelineRow objects from buffered dicts for transform execution
        # buffered_rows are plain dicts (for checkpoint serialization), but batch transforms
        # expect list[PipelineRow]. Reconstruct using contracts from buffered_tokens.
        # Fix for P1: AttributeError when transforms call .to_dict() on dict objects
        pipeline_rows: list[PipelineRow] = []
        for row_dict, token in zip(buffered_rows, buffered_tokens, strict=True):
            contract = token.row_data.contract
            if contract is None:
                raise RuntimeError(
                    f"Token {token.token_id} has no contract - cannot reconstruct PipelineRow. "
                    f"This indicates a bug in buffer_row() or checkpoint restore."
                )
            pipeline_rows.append(PipelineRow(row_dict, contract))

        # Step 1: Transition batch to "executing"
        self._recorder.update_batch_status(
            batch_id=batch_id,
            status=BatchStatus.EXECUTING,
            trigger_type=trigger_type,
        )

        # Step 2: Begin node state for flush operation
        # Wrap batch rows in a dict for node_state recording
        batch_input: dict[str, Any] = {"batch_rows": buffered_rows}

        # Compute input hash AFTER wrapping (must match what begin_node_state records)
        # See: P2-2026-01-21-aggregation-input-hash-mismatch
        input_hash = stable_hash(batch_input)

        # Resolve step position from node_id (injected StepResolver)
        step = self._step_resolver(node_id)

        state = self._recorder.begin_node_state(
            token_id=representative_token.token_id,
            node_id=node_id,
            run_id=ctx.run_id,
            step_index=step,
            input_data=batch_input,
            attempt=0,
        )

        # Set state_id and node_id on context for external call recording
        # and batch checkpoint lookup (node_id required for _batch_checkpoints keying)
        ctx.state_id = state.state_id
        ctx.node_id = node_id
        # Note: call_index allocation handled by LandscapeRecorder.allocate_call_index()

        # Expose per-row token identity for batch transforms. This allows transforms
        # like OpenRouterBatchLLMTransform to pass the correct token_id to audited
        # clients, ensuring per-token telemetry correlation in multi-token batches.
        batch_token_ids = [t.token_id for t in buffered_tokens]
        ctx.batch_token_ids = batch_token_ids
        with self._spans.aggregation_span(
            transform.name,
            node_id=node_id,
            input_hash=input_hash,
            batch_id=batch_id,
            token_ids=batch_token_ids,
        ):
            start = time.perf_counter()
            try:
                # Pass reconstructed PipelineRow objects to batch-aware transform
                result = transform.process(pipeline_rows, ctx)
                duration_ms = (time.perf_counter() - start) * 1000
            except BatchPendingError:
                # BatchPendingError is a CONTROL-FLOW SIGNAL, not an error.
                # The batch has been submitted but isn't complete yet.
                # Complete node_state with PENDING status and link batch for audit trail, then re-raise.
                duration_ms = (time.perf_counter() - start) * 1000

                # Close node_state with "pending" status - the submission succeeded
                # but the result isn't available yet. This prevents orphaned OPEN states.
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status=NodeStateStatus.PENDING,
                    duration_ms=duration_ms,
                )

                # Link batch to the aggregation state for traceability.
                # Keep status as "executing" but set aggregation_state_id.
                self._recorder.update_batch_status(
                    batch_id=batch_id,
                    status=BatchStatus.EXECUTING,
                    state_id=state.state_id,
                )

                # Clear batch_token_ids before re-raise to prevent stale IDs
                # leaking to subsequent calls (PluginContext is reused).
                ctx.batch_token_ids = None

                # Re-raise for orchestrator to schedule retry.
                # The batch remains in "executing" status, checkpoint is preserved.
                raise
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000

                # Record failure in node_state
                error: ExecutionError = {
                    "exception": str(e),
                    "type": type(e).__name__,
                }
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status=NodeStateStatus.FAILED,
                    duration_ms=duration_ms,
                    error=error,
                )

                # Transition batch to failed
                self._recorder.complete_batch(
                    batch_id=batch_id,
                    status=BatchStatus.FAILED,
                    trigger_type=trigger_type,
                    state_id=state.state_id,
                )

                # Reset for next batch
                self._reset_batch_state(node_id)

                # Clear batch_token_ids before re-raise to prevent stale IDs
                # leaking to subsequent calls (PluginContext is reused).
                ctx.batch_token_ids = None
                raise

        # Step 4: Populate audit fields on result
        # Wrap stable_hash calls to convert canonicalization errors to PluginContractViolation.
        # stable_hash calls canonical_json which rejects NaN, Infinity, non-serializable types.
        # Per CLAUDE.md: plugin bugs must crash with clear error messages.
        result.input_hash = input_hash
        try:
            if result.row is not None:
                result.output_hash = stable_hash(result.row)
            elif result.rows is not None:
                result.output_hash = stable_hash(result.rows)
            else:
                result.output_hash = None
        except (TypeError, ValueError) as e:
            raise PluginContractViolation(
                f"Aggregation transform '{transform.name}' emitted non-canonical data: {e}. "
                f"Ensure output contains only JSON-serializable types. "
                f"Use None instead of NaN for missing values."
            ) from e
        result.duration_ms = duration_ms

        # Step 5: Complete node state
        if result.status == "success":
            # Extract dicts for audit trail (Tier 1: full trust - store plain dicts)
            output_data: dict[str, Any] | list[dict[str, Any]]
            if result.row is not None:
                output_data = result.row.to_dict()
            elif result.rows is not None:
                output_data = [r.to_dict() for r in result.rows]
            else:
                # Contract violation: success status requires output data
                raise RuntimeError(
                    f"Aggregation transform '{transform.name}' returned success status but "
                    f"neither row nor rows contains data. Batch-aware transforms must return "
                    f"output via TransformResult.success(row) or TransformResult.success_multi(rows). "
                    f"This is a plugin bug."
                )

            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.COMPLETED,
                output_data=output_data,
                duration_ms=duration_ms,
                success_reason=result.success_reason,
            )

            # Transition batch to completed
            self._recorder.complete_batch(
                batch_id=batch_id,
                status=BatchStatus.COMPLETED,
                trigger_type=trigger_type,
                state_id=state.state_id,
            )
        else:
            # Transform returned error status
            error_info: ExecutionError = {
                "exception": str(result.reason) if result.reason else "Transform returned error",
                "type": "TransformError",
            }
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status=NodeStateStatus.FAILED,
                duration_ms=duration_ms,
                error=error_info,
            )

            # Transition batch to failed
            self._recorder.complete_batch(
                batch_id=batch_id,
                status=BatchStatus.FAILED,
                trigger_type=trigger_type,
                state_id=state.state_id,
            )

        # Step 6: Save batch_id before reset (needed by caller for CONSUMED_IN_BATCH)
        # Note: batch_id was validated at the start of this method
        flushed_batch_id = batch_id

        # Reset for next batch and clear buffers
        self._reset_batch_state(node_id)
        self._buffers[node_id] = []
        self._buffer_tokens[node_id] = []

        # Reset trigger evaluator for next batch
        # Direct access since buffer_row validates node_id is in aggregation_settings,
        # which guarantees a trigger evaluator exists
        self._trigger_evaluators[node_id].reset()

        # Clear batch_token_ids to prevent stale data leaking to next batch
        ctx.batch_token_ids = None

        return result, buffered_tokens, flushed_batch_id

    def _reset_batch_state(self, node_id: NodeID) -> None:
        """Reset batch tracking state for next batch.

        INTERNAL: Only called from execute_flush() which has already validated
        that batch_id exists. Direct access ensures we detect state corruption.

        Args:
            node_id: Aggregation node ID
        """
        # Direct access - execute_flush validated batch_id exists before calling us
        batch_id = self._batch_ids[node_id]
        # Type narrowing: execute_flush validates batch_id is not None before calling
        assert batch_id is not None, f"_reset_batch_state invariant violation: batch_id is None for {node_id}"
        del self._batch_ids[node_id]
        # member_counts is keyed by batch_id (not node_id) - direct access
        del self._member_counts[batch_id]

    def get_buffer_count(self, node_id: NodeID) -> int:
        """Get the number of rows currently buffered for an aggregation.

        Args:
            node_id: Aggregation node ID

        Returns:
            Number of buffered rows (0 if no rows buffered yet)

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        # Validate node_id is a configured aggregation (P2-2026-02-02: whitelist-reduction)
        if node_id not in self._aggregation_settings:
            raise OrchestrationInvariantError(
                f"get_buffer_count called for node '{node_id}' which is not in aggregation_settings. "
                f"Configured nodes: {list(self._aggregation_settings.keys())}"
            )
        return len(self._buffers.get(node_id, []))

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Return checkpoint state for persistence.

        Stores complete TokenInfo objects (not just IDs) to enable restoration
        without database queries. Validates size to prevent pathological growth.

        Returns:
            dict[str, Any]: Checkpoint state with format:
                {
                    "node_id_1": {
                        "tokens": [
                            {
                                "token_id": str,
                                "row_id": str,
                                "branch_name": str | None,
                                "fork_group_id": str | None,
                                "join_group_id": str | None,
                                "expand_group_id": str | None,
                                "row_data": dict[str, Any]
                            },
                            ...
                        ]
                    },
                    ...
                }

        Raises:
            RuntimeError: If checkpoint exceeds 10MB size limit
        """
        from elspeth.core.checkpoint.serialization import checkpoint_dumps

        # Build checkpoint state from all buffers
        state: dict[str, Any] = {}
        for node_id, tokens in self._buffer_tokens.items():
            if not tokens:  # Only include non-empty buffers
                continue

            # Get trigger state for preservation (Bug #6 + P2-2026-02-01)
            # If tokens exist for a node, a trigger evaluator MUST exist (created in __init__)
            # Direct access crashes if evaluator is missing, revealing configuration bugs.
            evaluator = self._trigger_evaluators[node_id]
            elapsed_age_seconds = evaluator.get_age_seconds()
            # P2-2026-02-01: Preserve fire time offsets for "first to fire wins" ordering
            count_fire_offset = evaluator.get_count_fire_offset()
            condition_fire_offset = evaluator.get_condition_fire_offset()

            if node_id not in self._batch_ids or self._batch_ids[node_id] is None:
                raise RuntimeError(
                    f"AggregationExecutor checkpoint missing batch_id for node {node_id}. "
                    "Buffered tokens exist without an active batch_id - internal state corruption."
                )

            batch_id = self._batch_ids[node_id]

            # Store full TokenInfo as dicts (not just IDs)
            # Include all lineage fields to preserve fork/join/expand metadata
            #
            # PipelineRow Migration (v2.0), hash-width bump (v2.1):
            # - row_data is stored as dict via to_dict() for JSON serialization
            # - contract is stored once per node (not per token) for efficiency
            # - contract_version links tokens to their contract for restoration
            #
            # Get contract from first token (all tokens in buffer share same contract)
            # Per CLAUDE.md Tier 1: tokens exist, so first token MUST exist
            first_token_contract = tokens[0].row_data.contract
            state[node_id] = {
                "tokens": [
                    {
                        "token_id": t.token_id,
                        "row_id": t.row_id,
                        "branch_name": t.branch_name,
                        "fork_group_id": t.fork_group_id,
                        "join_group_id": t.join_group_id,
                        "expand_group_id": t.expand_group_id,
                        "row_data": t.row_data.to_dict(),  # Extract dict for JSON serialization
                        "contract_version": t.row_data.contract.version_hash(),
                    }
                    for t in tokens
                ],
                "batch_id": batch_id,
                "elapsed_age_seconds": elapsed_age_seconds,  # Bug #6: Preserve timeout window
                # P2-2026-02-01: Preserve trigger fire time offsets
                "count_fire_offset": count_fire_offset,
                "condition_fire_offset": condition_fire_offset,
                # Store contract once per node (all buffered tokens share the same contract)
                "contract": first_token_contract.to_checkpoint_format(),
            }

        # Checkpoint format version
        # v1.0: Initial format with elapsed_age_seconds
        # v1.1: Added count_fire_offset/condition_fire_offset for trigger ordering (P2-2026-02-01)
        # v2.0: PipelineRow migration - row_data will be PipelineRow with contract
        # v2.1: Contract version_hash width changed (16 -> 32 hex chars)
        # v3.0: Phase 2 traversal refactor checkpoint break (no backwards compatibility)
        state["_version"] = AGGREGATION_CHECKPOINT_VERSION

        # Size validation (on serialized checkpoint)
        # Use checkpoint_dumps to handle datetime (P1-2026-02-05 fix)
        serialized = checkpoint_dumps(state)
        size_mb = len(serialized) / 1_000_000
        total_rows = sum(len(b) for b in self._buffer_tokens.values())

        if size_mb > 1:
            logger.warning(f"Large checkpoint: {size_mb:.1f}MB for {total_rows} buffered rows across {len(state)} nodes")

        if size_mb > 10:
            raise RuntimeError(
                f"Checkpoint size {size_mb:.1f}MB exceeds 10MB limit. "
                f"Buffer contains {total_rows} total rows across {len(state)} nodes. "
                f"Solutions: (1) Reduce aggregation count trigger to <5000 rows, "
                f"(2) Reduce row_data payload size, or (3) Implement checkpoint retention "
                f"policy"
            )

        return state

    def restore_from_checkpoint(self, state: dict[str, Any]) -> None:
        """Restore executor state from checkpoint.

        Reconstructs full TokenInfo objects from checkpoint data, eliminating
        database queries during restoration. Expects format from get_checkpoint_state().

        Args:
            state: Checkpoint state with format:
                {
                    "_version": "3.0",
                    "node_id": {
                        "tokens": [{"token_id", "row_id", "branch_name", "row_data", ...}],
                        "batch_id": str,
                        "elapsed_age_seconds": float,
                        "count_fire_offset": float | None,
                        "condition_fire_offset": float | None
                    }
                }

        Raises:
            ValueError: If checkpoint format is invalid (per CLAUDE.md - our data, full trust)
        """
        # Validate checkpoint version (Bug #12 fix)
        # v1.1: Pre-PipelineRow migration format
        # v2.0: PipelineRow migration - row_data will be PipelineRow with contract
        # v2.1: Contract version_hash width changed (16 -> 32 hex chars)
        # v3.0: Phase 2 traversal refactor checkpoint break (no backwards compatibility)
        checkpoint_version = AGGREGATION_CHECKPOINT_VERSION
        version = state.get("_version")

        if version != checkpoint_version:
            # Log checkpoint rejection for observability
            slog.warning(
                "checkpoint_version_rejected",
                found_version=version,
                expected_version=checkpoint_version,
                reason="incompatible_checkpoint_version",
            )
            raise ValueError(
                f"Incompatible checkpoint version: {version!r}. "
                f"Expected: {checkpoint_version!r}. "
                f"Cannot resume from incompatible checkpoint format. "
                f"This checkpoint may be from a different ELSPETH version."
            )

        for node_id_str, node_state in state.items():
            # Skip version metadata field
            if node_id_str == "_version":
                continue
            # Convert to typed NodeID for dictionary access
            node_id = NodeID(node_id_str)
            # Validate checkpoint format (OUR DATA - crash on mismatch, don't hide with .get())
            if "tokens" not in node_state:
                raise ValueError(
                    f"Invalid checkpoint format for node {node_id}: missing 'tokens' key. "
                    f"Found keys: {list(node_state.keys())}. "
                    f"Expected format: {{'tokens': [...], 'batch_id': str|None}}. "
                    f"This checkpoint may be from an incompatible ELSPETH version."
                )

            tokens_data = node_state["tokens"]

            # Validate tokens is a list
            if not isinstance(tokens_data, list):
                raise ValueError(f"Invalid checkpoint format for node {node_id}: 'tokens' must be a list, got {type(tokens_data).__name__}")

            # Restore contract from checkpoint (stored once per node)
            # Per CLAUDE.md Tier 1: contract MUST exist if tokens exist
            if "contract" not in node_state:
                raise ValueError(
                    f"Invalid checkpoint format for node {node_id}: missing 'contract' key. "
                    f"Checkpoint format {checkpoint_version} requires contract for PipelineRow restoration."
                )
            restored_contract = SchemaContract.from_checkpoint(node_state["contract"])

            # Reconstruct TokenInfo objects directly from checkpoint
            reconstructed_tokens = []
            for t in tokens_data:
                # Validate required fields (crash on missing - per CLAUDE.md)
                # All these fields are required in current checkpoint format (values can be None)
                required_fields = {
                    "token_id",
                    "row_id",
                    "row_data",
                    "branch_name",
                    "fork_group_id",
                    "join_group_id",
                    "expand_group_id",
                    "contract_version",  # v2.0: required for contract reference
                }
                missing = required_fields - set(t.keys())
                if missing:
                    raise ValueError(
                        f"Checkpoint token missing required fields: {missing}. "
                        f"Required in checkpoint format {checkpoint_version}: {required_fields}. Found: {set(t.keys())}"
                    )

                # Validate contract_version matches restored contract
                # Per CLAUDE.md Tier 1: integrity check on our data
                if t["contract_version"] != restored_contract.version_hash():
                    raise ValueError(
                        f"Contract version mismatch for token {t['token_id']}: "
                        f"expected {restored_contract.version_hash()}, got {t['contract_version']}. "
                        f"Checkpoint may be corrupted."
                    )

                # Reconstruct PipelineRow from checkpoint data
                row_data = PipelineRow(t["row_data"], restored_contract)

                # Reconstruct TokenInfo from checkpoint data
                # NOTE: These fields CAN be None (valid state for unforked tokens), but they
                # are ALWAYS present in current checkpoint format - use direct access to detect
                # corruption/missing fields. The difference between "field is None" and
                # "field is missing" matters: the former is valid, the latter is corruption.
                reconstructed_tokens.append(
                    TokenInfo(
                        row_id=t["row_id"],
                        token_id=t["token_id"],
                        row_data=row_data,  # PipelineRow, not dict
                        branch_name=t["branch_name"],
                        fork_group_id=t["fork_group_id"],
                        join_group_id=t["join_group_id"],
                        expand_group_id=t["expand_group_id"],
                    )
                )

            # Restore buffer state
            # _buffer_tokens stores TokenInfo with PipelineRow
            # _buffers stores dicts (JSON-serializable for future checkpoints)
            self._buffer_tokens[node_id] = reconstructed_tokens
            self._buffers[node_id] = [t.row_data.to_dict() for t in reconstructed_tokens]

            if "batch_id" not in node_state:
                raise ValueError(
                    f"Invalid checkpoint format for node {node_id}: missing 'batch_id' key. "
                    f"Found keys: {list(node_state.keys())}. "
                    "Checkpoint entries with tokens must include batch_id."
                )
            batch_id = node_state["batch_id"]
            if batch_id is None:
                raise ValueError(
                    f"Invalid checkpoint format for node {node_id}: 'batch_id' is None. "
                    "Checkpoint entries with tokens must include a batch_id."
                )
            self._batch_ids[node_id] = batch_id
            self._member_counts[batch_id] = len(reconstructed_tokens)

            # Restore trigger evaluator state (Bug #6 + P2-2026-02-01)
            # If tokens exist for a node, a trigger evaluator MUST exist (created in __init__)
            # Direct access crashes if evaluator is missing, revealing configuration bugs.
            evaluator = self._trigger_evaluators[node_id]

            # P2-2026-02-01: Use dedicated restore API that preserves fire time ordering
            # The old approach called record_accept() which set fire times to current time,
            # then rewound _first_accept_time, causing incorrect "first to fire wins" ordering.
            # NOTE: All fields are required in current checkpoint format - no backwards compat
            elapsed_seconds = node_state["elapsed_age_seconds"]
            count_fire_offset = node_state["count_fire_offset"]
            condition_fire_offset = node_state["condition_fire_offset"]

            evaluator.restore_from_checkpoint(
                batch_count=len(reconstructed_tokens),
                elapsed_age_seconds=elapsed_seconds,
                count_fire_offset=count_fire_offset,
                condition_fire_offset=condition_fire_offset,
            )

            # Log successful checkpoint restoration for observability
            slog.info(
                "checkpoint_restored",
                node_id=str(node_id),
                token_count=len(reconstructed_tokens),
                checkpoint_version=checkpoint_version,
            )

    def get_batch_id(self, node_id: NodeID) -> str | None:
        """Get current batch ID for an aggregation node.

        Primarily for testing - production code accesses this via checkpoint state.

        Note: Does not validate against aggregation_settings since this is a
        testing/inspection method that needs to work with partial setups.

        Args:
            node_id: Aggregation node ID

        Returns:
            Batch ID if a batch is in progress, None otherwise
        """
        return self._batch_ids.get(node_id)

    def should_flush(self, node_id: NodeID) -> bool:
        """Check if the aggregation should flush based on trigger config.

        Args:
            node_id: Aggregation node ID

        Returns:
            True if trigger condition is met, False otherwise

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        # Validate node_id is a configured aggregation (P2-2026-02-02: whitelist-reduction)
        # Trigger evaluators are created in __init__ for all configured nodes.
        if node_id not in self._aggregation_settings:
            raise OrchestrationInvariantError(
                f"should_flush called for node '{node_id}' which is not in aggregation_settings. "
                f"Configured nodes: {list(self._aggregation_settings.keys())}"
            )
        # Direct access - evaluator must exist if node is in aggregation_settings
        return self._trigger_evaluators[node_id].should_trigger()

    def get_trigger_type(self, node_id: NodeID) -> "TriggerType | None":
        """Get the TriggerType for the trigger that fired.

        Args:
            node_id: Aggregation node ID

        Returns:
            TriggerType enum if a trigger fired, None otherwise

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        # Validate node_id is a configured aggregation (P2-2026-02-02: whitelist-reduction)
        if node_id not in self._aggregation_settings:
            raise OrchestrationInvariantError(
                f"get_trigger_type called for node '{node_id}' which is not in aggregation_settings. "
                f"Configured nodes: {list(self._aggregation_settings.keys())}"
            )
        return self._trigger_evaluators[node_id].get_trigger_type()

    def check_flush_status(self, node_id: NodeID) -> tuple[bool, "TriggerType | None"]:
        """Check flush status and get trigger type in a single operation.

        This is an optimized method that combines should_flush() and get_trigger_type()
        with a single dict lookup instead of two. Used in the hot path where
        timeout checks happen before every row is processed.

        Args:
            node_id: Aggregation node ID

        Returns:
            Tuple of (should_flush, trigger_type):
            - should_flush: True if trigger condition is met
            - trigger_type: The type of trigger that fired, or None

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        # Validate node_id is a configured aggregation (P2-2026-02-02: whitelist-reduction)
        if node_id not in self._aggregation_settings:
            raise OrchestrationInvariantError(
                f"check_flush_status called for node '{node_id}' which is not in aggregation_settings. "
                f"Configured nodes: {list(self._aggregation_settings.keys())}"
            )
        # Direct access - evaluator must exist if node is in aggregation_settings
        evaluator = self._trigger_evaluators[node_id]
        should_flush = evaluator.should_trigger()
        trigger_type = evaluator.get_trigger_type() if should_flush else None
        return (should_flush, trigger_type)

    def restore_state(self, node_id: NodeID, state: dict[str, Any]) -> None:
        """Restore aggregation state from checkpoint.

        Called during recovery to restore plugin state. The state is stored
        for the aggregation plugin to access via get_restored_state().

        Args:
            node_id: Aggregation node ID
            state: Deserialized aggregation_state from checkpoint
        """
        self._restored_states[node_id] = state

    def get_restored_state(self, node_id: NodeID) -> dict[str, Any] | None:
        """Get restored state for an aggregation node.

        Used by aggregation plugins during recovery to restore their
        internal state from checkpoint.

        Note: This is a simple key-value lookup in _restored_states, which is
        populated by restore_state() during crash recovery. It does NOT validate
        against aggregation_settings because state restoration happens before
        full configuration is available, and the method is designed to return
        None for nodes without restored state.

        Args:
            node_id: Aggregation node ID

        Returns:
            Restored state dict, or None if no state was restored
        """
        return self._restored_states.get(node_id)

    def restore_batch(self, batch_id: str) -> None:
        """Restore a batch as the current in-progress batch.

        Called during recovery to resume a batch that was in progress
        when the crash occurred.

        Args:
            batch_id: The batch to restore as current

        Raises:
            ValueError: If batch not found
        """
        batch = self._recorder.get_batch(batch_id)
        if batch is None:
            raise ValueError(f"Batch not found: {batch_id}")

        node_id = NodeID(batch.aggregation_node_id)
        self._batch_ids[node_id] = batch_id

        # Restore member count from database
        members = self._recorder.get_batch_members(batch_id)
        self._member_counts[batch_id] = len(members)

    # NOTE: The old accept() and flush() methods that took AggregationProtocol
    # were DELETED in the aggregation structural cleanup.
    # Aggregation is now fully structural:
    # - Use buffer_row() to buffer rows
    # - Use should_flush() to check trigger
    # - Use execute_flush() to flush with full audit recording
    # - _get_buffered_data() is internal-only (for testing)
