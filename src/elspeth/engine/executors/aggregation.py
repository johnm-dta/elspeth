"""AggregationExecutor - manages batch lifecycle with audit recording."""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

import elspeth.contracts.errors as contract_errors
from elspeth.contracts import (
    BatchTransformProtocol,
    ExecutionError,
    PipelineRow,
    SchemaContract,
    TokenInfo,
    TransformResult,
)
from elspeth.contracts.aggregation_checkpoint import (
    AggregationCheckpointState,
    AggregationNodeCheckpoint,
    AggregationTokenCheckpoint,
)
from elspeth.contracts.enums import (
    BatchStatus,
    NodeStateStatus,
    TriggerType,
)
from elspeth.contracts.errors import (
    AuditIntegrityError,
    OrchestrationInvariantError,
    PluginContractViolation,
)
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.node_state_context import AggregationFlushContext
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.types import NodeID, StepResolver
from elspeth.core.canonical import stable_hash
from elspeth.core.checkpoint.serialization import checkpoint_dumps
from elspeth.core.config import AggregationSettings
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.engine.clock import DEFAULT_CLOCK
from elspeth.engine.executors.state_guard import NodeStateGuard
from elspeth.engine.spans import SpanFactory
from elspeth.engine.triggers import TriggerEvaluator

if TYPE_CHECKING:
    from elspeth.engine.clock import Clock

logger = logging.getLogger(__name__)
slog = structlog.get_logger(__name__)

AGGREGATION_CHECKPOINT_VERSION = "4.0"


@dataclass(slots=True)
class _AggregationNodeState:
    """Per-node aggregation state.

    Groups settings, trigger evaluator, batch tracking, and row buffers
    that were previously scattered across six parallel dicts keyed by NodeID,
    plus a member_count that was in a separate dict keyed by batch_id.

    Mutable because buffers grow during processing and batch_id/member_count
    change across batch lifecycles.  Not frozen (unlike _BranchEntry in
    coalesce_executor.py) because the fields are updated in-place.
    """

    settings: AggregationSettings
    trigger: TriggerEvaluator
    batch_id: str | None = None
    member_count: int = 0
    buffers: list[dict[str, Any]] = field(default_factory=list)
    tokens: list[TokenInfo] = field(default_factory=list)
    restored_state: AggregationCheckpointState | None = None


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
        executor = AggregationExecutor(execution, span_factory, step_resolver, run_id)

        # Accept rows into batch
        result = executor.buffer_row(node_id, token)
        # Engine uses TriggerEvaluator to decide when to flush
    """

    def __init__(
        self,
        execution: ExecutionRepository,
        span_factory: SpanFactory,
        step_resolver: StepResolver,
        run_id: str,
        *,
        aggregation_settings: dict[NodeID, AggregationSettings] | None = None,
        clock: "Clock | None" = None,
    ) -> None:
        """Initialize executor.

        Args:
            execution: Execution repository for audit trail
            span_factory: Span factory for tracing
            step_resolver: Resolves NodeID to 1-indexed audit step position
            run_id: Run identifier for batch creation
            aggregation_settings: Map of node_id -> AggregationSettings for trigger evaluation
            clock: Optional clock for time access. Defaults to system clock.
                   Inject MockClock for deterministic testing.
        """
        self._execution = execution
        self._spans = span_factory
        self._step_resolver = step_resolver
        self._run_id = run_id
        self._clock = clock if clock is not None else DEFAULT_CLOCK

        # Single consolidated dict replaces 7 parallel dicts:
        # _aggregation_settings, _trigger_evaluators, _batch_ids,
        # _member_counts, _buffers, _buffer_tokens, _restored_states
        self._nodes: dict[NodeID, _AggregationNodeState] = {}
        for node_id, settings in (aggregation_settings or {}).items():
            self._nodes[node_id] = _AggregationNodeState(
                settings=settings,
                trigger=TriggerEvaluator(settings.trigger, clock=self._clock),
            )

    def _get_node(self, node_id: NodeID, caller: str = "") -> _AggregationNodeState:
        """Look up a configured aggregation node, crash on unknown.

        This is the single validation point for all node access. Unknown
        node_id is always a bug — either in the caller or in checkpoint
        data (Tier 1 corruption).
        """
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise OrchestrationInvariantError(
                f"{caller or 'AggregationExecutor'} called for node '{node_id}' "
                f"which is not in aggregation_settings. "
                f"Configured nodes: {list(self._nodes.keys())}"
            ) from exc

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
        node = self._get_node(node_id, "buffer_row")

        # Create batch on first row if needed
        if node.batch_id is None:
            batch = self._execution.create_batch(
                run_id=self._run_id,
                aggregation_node_id=node_id,
            )
            node.batch_id = batch.batch_id
            node.member_count = 0

        batch_id = node.batch_id
        if batch_id is None:
            raise OrchestrationInvariantError(f"batch_id is None after creation for node {node_id}")

        # Buffer the row - store dict (JSON-serializable for checkpoints)
        # TokenInfo.row_data is PipelineRow, extract dict for buffer
        node.buffers.append(token.row_data.to_dict())
        node.tokens.append(token)

        # Record batch membership for audit trail
        ordinal = node.member_count
        self._execution.add_batch_member(
            batch_id=batch_id,
            token_id=token.token_id,
            ordinal=ordinal,
        )
        node.member_count = ordinal + 1

        # Update trigger evaluator
        node.trigger.record_accept()

    def get_buffered_rows(self, node_id: NodeID) -> list[dict[str, Any]]:
        """Get currently buffered rows (does not clear buffer).

        Args:
            node_id: Aggregation node ID

        Returns:
            List of buffered row dicts (empty if no rows buffered yet)

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        return list(self._get_node(node_id, "get_buffered_rows").buffers)

    def get_buffered_tokens(self, node_id: NodeID) -> list[TokenInfo]:
        """Get currently buffered tokens (does not clear buffer).

        Args:
            node_id: Aggregation node ID

        Returns:
            List of buffered TokenInfo objects (empty if no rows buffered yet)

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        return list(self._get_node(node_id, "get_buffered_tokens").tokens)

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
        node = self._get_node(node_id, "_get_buffered_data")
        return list(node.buffers), list(node.tokens)

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
        node = self._get_node(node_id, "execute_flush")
        batch_id = node.batch_id
        if batch_id is None:
            raise OrchestrationInvariantError(f"No batch exists for node {node_id} - cannot flush")

        # Snapshot buffered data (consolidated state eliminates KeyError risk —
        # buffers and tokens are structurally tied to the same node entry)
        buffered_rows = list(node.buffers)
        buffered_tokens = list(node.tokens)

        if not buffered_rows:
            raise OrchestrationInvariantError(f"Cannot flush empty buffer for node {node_id}")

        # Defensive validation: buffer and tokens must be same length
        # This should never happen (checkpoint restore ensures they stay in sync)
        # but crashes explicitly if internal state is corrupted
        if len(buffered_rows) != len(buffered_tokens):
            raise OrchestrationInvariantError(
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
                raise OrchestrationInvariantError(
                    f"Token {token.token_id} has no contract - cannot reconstruct PipelineRow. "
                    f"This indicates a bug in buffer_row() or checkpoint restore."
                )
            pipeline_rows.append(PipelineRow(row_dict, contract))

        # Step 1: Transition batch to "executing"
        self._execution.update_batch_status(
            batch_id=batch_id,
            status=BatchStatus.EXECUTING,
            trigger_type=trigger_type,
        )

        # Step 2: Prepare node state inputs
        # Wrap batch rows in a dict for node_state recording
        batch_input: dict[str, Any] = {"batch_rows": buffered_rows}

        # Compute input hash AFTER wrapping (must match what begin_node_state records)
        # Bug fix: hash must be computed after wrapping to match begin_node_state
        input_hash = stable_hash(batch_input)

        # Resolve step position from node_id (injected StepResolver)
        step = self._step_resolver(node_id)

        # NodeStateGuard guarantees the node state reaches terminal status.
        # If any post-processing step (output hashing, batch completion) raises
        # before the state is explicitly completed, the guard auto-completes
        # it as FAILED.  Batch lifecycle cleanup is handled separately below.
        with NodeStateGuard(
            self._execution,
            token_id=representative_token.token_id,
            node_id=node_id,
            run_id=ctx.run_id,
            step_index=step,
            input_data=batch_input,
            attempt=0,
        ) as guard:
            # Set state_id and node_id on context for external call recording.
            ctx.state_id = guard.state_id
            ctx.node_id = node_id
            # Note: call_index allocation handled by ExecutionRepository.allocate_call_index()

            # Expose per-row token identity for batch transforms. This allows transforms
            # like OpenRouterBatchLLMTransform to pass the correct token_id to audited
            # clients, ensuring per-token telemetry correlation in multi-token batches.
            batch_token_ids = tuple(t.token_id for t in buffered_tokens)
            ctx.batch_token_ids = batch_token_ids

            # Track whether the batch was finalized (COMPLETED or FAILED).
            # Used by the outer except to decide whether to fail the batch.
            batch_finalized = False

            try:
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
                    except contract_errors.TIER_1_ERRORS:
                        raise  # Tier 1 errors must crash — never record as row FAILED
                    except Exception as e:
                        duration_ms = (time.perf_counter() - start) * 1000

                        # Record failure in node_state
                        error = ExecutionError(
                            exception=str(e),
                            exception_type=type(e).__name__,
                        )
                        guard.complete(
                            NodeStateStatus.FAILED,
                            duration_ms=duration_ms,
                            error=error,
                        )
                        raise  # Batch cleanup in outer except

                # -- Post-processing (GUARDED by NodeStateGuard) --
                # If any of the following steps raise before guard.complete()
                # is called, the guard auto-completes the state as FAILED.

                # Populate audit fields on result
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

                # Complete node state and batch
                if result.status == "success":
                    # Extract dicts for audit trail (Tier 1: full trust - store plain dicts)
                    output_data: dict[str, Any] | list[dict[str, Any]]
                    if result.row is not None:
                        output_data = result.row.to_dict()
                    elif result.rows is not None:
                        output_data = [r.to_dict() for r in result.rows]
                    else:
                        # Contract violation: success status requires output data
                        raise PluginContractViolation(
                            f"Aggregation transform '{transform.name}' returned success status but "
                            f"neither row nor rows contains data. Batch-aware transforms must return "
                            f"output via TransformResult.success(row) or TransformResult.success_multi(rows)."
                        )

                    flush_context = AggregationFlushContext(
                        trigger_type=trigger_type,
                        buffer_size=len(buffered_rows),
                        batch_id=batch_id,
                    )
                    guard.complete(
                        NodeStateStatus.COMPLETED,
                        output_data=output_data,
                        duration_ms=duration_ms,
                        success_reason=result.success_reason,
                        context_after=flush_context,
                    )

                    # Transition batch to completed
                    self._execution.complete_batch(
                        batch_id=batch_id,
                        status=BatchStatus.COMPLETED,
                        trigger_type=trigger_type,
                        state_id=guard.state_id,
                    )
                    batch_finalized = True
                else:
                    # Transform returned error status
                    error_info = ExecutionError(
                        exception=str(result.reason) if result.reason else "Transform returned error",
                        exception_type="TransformError",
                    )
                    guard.complete(
                        NodeStateStatus.FAILED,
                        duration_ms=duration_ms,
                        error=error_info,
                    )

                    # Transition batch to failed
                    self._execution.complete_batch(
                        batch_id=batch_id,
                        status=BatchStatus.FAILED,
                        trigger_type=trigger_type,
                        state_id=guard.state_id,
                    )
                    batch_finalized = True

            except contract_errors.TIER_1_ERRORS:
                raise  # Tier 1 errors must crash — skip batch cleanup
            except Exception:
                # Batch cleanup on ANY failure (guard handles node state).
                # Only attempt to fail the batch if it wasn't already finalized
                # (avoids double-write if complete_batch itself raised).
                if not batch_finalized:
                    try:
                        self._execution.complete_batch(
                            batch_id=batch_id,
                            status=BatchStatus.FAILED,
                            trigger_type=trigger_type,
                            state_id=guard.state_id,
                        )
                    except contract_errors.TIER_1_ERRORS:
                        raise  # Tier 1 errors must crash immediately
                    except (TypeError, AttributeError, KeyError, NameError):
                        raise  # Programming errors in recorder — crash to surface the bug
                    except Exception as cleanup_err:
                        # Batch cleanup failure leaves batch in non-terminal state (DRAFT/EXECUTING)
                        # permanently — orphaned in the audit trail with no recovery path.
                        # Per Tier 1 rules: crash rather than leave corrupted audit state.
                        raise AuditIntegrityError(
                            f"Failed to mark batch {batch_id} as FAILED during error cleanup — "
                            f"batch would remain in non-terminal state (audit trail corruption). "
                            f"Cleanup error: {cleanup_err}"
                        ) from cleanup_err
                # Full cleanup: reset batch state, clear buffers, reset trigger
                self._reset_batch_state(node_id)
                node.buffers.clear()
                node.tokens.clear()
                node.trigger.reset()
                ctx.batch_token_ids = None
                raise

        # Success cleanup: save batch_id before reset (needed by caller for CONSUMED_IN_BATCH)
        flushed_batch_id = batch_id

        # Reset for next batch and clear buffers
        self._reset_batch_state(node_id)
        node.buffers.clear()
        node.tokens.clear()

        # Reset trigger evaluator for next batch
        node.trigger.reset()

        # Clear batch_token_ids to prevent stale data leaking to next batch
        ctx.batch_token_ids = None

        return result, buffered_tokens, flushed_batch_id

    def _reset_batch_state(self, node_id: NodeID) -> None:
        """Reset batch tracking state for next batch.

        INTERNAL: Only called from execute_flush() which has already validated
        that batch_id exists. Uses _get_node() for consistent validation.

        Args:
            node_id: Aggregation node ID
        """
        node = self._get_node(node_id, "_reset_batch_state")
        if node.batch_id is None:
            raise OrchestrationInvariantError(f"_reset_batch_state invariant violation: batch_id is None for {node_id}")
        node.batch_id = None
        node.member_count = 0

    def get_buffer_count(self, node_id: NodeID) -> int:
        """Get the number of rows currently buffered for an aggregation.

        Args:
            node_id: Aggregation node ID

        Returns:
            Number of buffered rows (0 if no rows buffered yet)

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        return len(self._get_node(node_id, "get_buffer_count").buffers)

    def get_checkpoint_state(self) -> AggregationCheckpointState:
        """Return checkpoint state for persistence.

        Stores complete TokenInfo objects (not just IDs) to enable restoration
        without database queries. Validates size to prevent pathological growth.

        Returns:
            AggregationCheckpointState with all buffered aggregation data.

        Raises:
            RuntimeError: If checkpoint exceeds 10MB size limit
        """
        # Build checkpoint state from all nodes
        nodes: dict[str, AggregationNodeCheckpoint] = {}
        for node_id, node in self._nodes.items():
            if not node.tokens:  # Only include non-empty buffers
                continue

            # Get trigger state for preservation
            elapsed_age_seconds = node.trigger.get_age_seconds()
            # Preserve fire time offsets for "first to fire wins" ordering
            count_fire_offset = node.trigger.get_count_fire_offset()
            condition_fire_offset = node.trigger.get_condition_fire_offset()

            if node.batch_id is None:
                raise OrchestrationInvariantError(
                    f"AggregationExecutor checkpoint missing batch_id for node {node_id}. "
                    "Buffered tokens exist without an active batch_id - internal state corruption."
                )

            batch_id = node.batch_id

            token_checkpoints = tuple(
                AggregationTokenCheckpoint(
                    token_id=t.token_id,
                    row_id=t.row_id,
                    branch_name=t.branch_name,
                    fork_group_id=t.fork_group_id,
                    join_group_id=t.join_group_id,
                    expand_group_id=t.expand_group_id,
                    row_data=t.row_data.to_dict(),
                    contract_version=t.row_data.contract.version_hash(),
                    contract=t.row_data.contract.to_checkpoint_format(),
                )
                for t in node.tokens
            )

            nodes[node_id] = AggregationNodeCheckpoint(
                tokens=token_checkpoints,
                batch_id=batch_id,
                elapsed_age_seconds=elapsed_age_seconds,
                count_fire_offset=count_fire_offset,
                condition_fire_offset=condition_fire_offset,
            )

        checkpoint = AggregationCheckpointState(
            version=AGGREGATION_CHECKPOINT_VERSION,
            nodes=nodes,
        )

        # Size validation (on serialized checkpoint)
        # Use checkpoint_dumps to handle datetime
        serialized = checkpoint_dumps(checkpoint.to_dict())
        size_mb = len(serialized) / 1_000_000
        total_rows = sum(len(n.tokens) for n in self._nodes.values())

        if size_mb > 1:
            logger.warning(f"Large checkpoint: {size_mb:.1f}MB for {total_rows} buffered rows across {len(nodes)} nodes")

        if size_mb > 10:
            raise OrchestrationInvariantError(
                f"Checkpoint size {size_mb:.1f}MB exceeds 10MB limit. "
                f"Buffer contains {total_rows} total rows across {len(nodes)} nodes. "
                f"Solutions: (1) Reduce aggregation count trigger to <5000 rows, "
                f"(2) Reduce row_data payload size, or (3) Implement checkpoint retention "
                f"policy"
            )

        return checkpoint

    def restore_from_checkpoint(self, state: AggregationCheckpointState) -> None:
        """Restore executor state from checkpoint.

        Reconstructs full TokenInfo objects from typed checkpoint data,
        eliminating database queries during restoration.

        Args:
            state: Typed checkpoint state from ``AggregationCheckpointState.from_dict()``.

        Raises:
            AuditIntegrityError: If checkpoint version is incompatible or data is invalid
                (per CLAUDE.md — our data, full trust)
        """
        # Validate checkpoint version
        checkpoint_version = AGGREGATION_CHECKPOINT_VERSION

        if state.version != checkpoint_version:
            raise AuditIntegrityError(
                f"Incompatible checkpoint version: {state.version!r}. "
                f"Expected: {checkpoint_version!r}. "
                f"Cannot resume from incompatible checkpoint format. "
                f"This checkpoint may be from a different ELSPETH version."
            )

        for node_id_str, node_checkpoint in state.nodes.items():
            node_id = NodeID(node_id_str)
            node = self._get_node(node_id, "restore_from_checkpoint")

            # Reconstruct TokenInfo objects directly from typed checkpoint
            reconstructed_tokens = []
            for t in node_checkpoint.tokens:
                restored_contract = SchemaContract.from_checkpoint(dict(t.contract))

                # Per CLAUDE.md Tier 1: integrity check on our data
                if t.contract_version != restored_contract.version_hash():
                    raise AuditIntegrityError(
                        f"Contract version mismatch for token {t.token_id}: "
                        f"expected {restored_contract.version_hash()}, got {t.contract_version}. "
                        f"Checkpoint may be corrupted."
                    )

                # deep_thaw() recursively converts MappingProxyType->dict and tuple->list,
                # preventing frozen nested containers from surviving into restored rows.
                row_data = PipelineRow(deep_thaw(t.row_data), restored_contract)

                reconstructed_tokens.append(
                    TokenInfo(
                        row_id=t.row_id,
                        token_id=t.token_id,
                        row_data=row_data,
                        branch_name=t.branch_name,
                        fork_group_id=t.fork_group_id,
                        join_group_id=t.join_group_id,
                        expand_group_id=t.expand_group_id,
                    )
                )

            persisted_member_count = self._reconcile_checkpoint_batch_members(
                node_id=node_id,
                batch_id=node_checkpoint.batch_id,
                checkpoint_tokens=reconstructed_tokens,
            )

            # Restore buffer state on consolidated node
            node.tokens = reconstructed_tokens
            node.buffers = [t.row_data.to_dict() for t in reconstructed_tokens]
            node.batch_id = node_checkpoint.batch_id
            node.member_count = persisted_member_count

            # Restore trigger evaluator state using dedicated API that preserves fire time ordering
            node.trigger.restore_from_checkpoint(
                batch_count=len(reconstructed_tokens),
                elapsed_age_seconds=node_checkpoint.elapsed_age_seconds,
                count_fire_offset=node_checkpoint.count_fire_offset,
                condition_fire_offset=node_checkpoint.condition_fire_offset,
            )

            # Log successful checkpoint restoration for observability
            slog.info(
                "checkpoint_restored",
                node_id=str(node_id),
                token_count=len(reconstructed_tokens),
                checkpoint_version=checkpoint_version,
            )

    def _reconcile_checkpoint_batch_members(
        self,
        *,
        node_id: NodeID,
        batch_id: str,
        checkpoint_tokens: list[TokenInfo],
    ) -> int:
        """Ensure persisted batch membership exactly matches the checkpoint snapshot.

        Crash recovery cannot safely continue an in-progress batch when the
        audit trail already contains members that are absent from the latest
        checkpoint. Resuming that batch would reuse persisted ordinals and mix
        replayed tokens with pre-crash membership.
        """
        batch = self._execution.get_batch(batch_id)
        if batch is None:
            raise AuditIntegrityError(f"Batch not found in audit trail: {batch_id}")

        if batch.aggregation_node_id != node_id:
            raise AuditIntegrityError(
                f"Checkpoint batch {batch_id!r} belongs to aggregation node "
                f"{batch.aggregation_node_id!r}, but restore is running for node "
                f"{node_id!r}. Checkpoint state or audit trail is corrupted."
            )

        persisted_members = self._execution.get_batch_members(batch_id)
        persisted_token_ids = tuple(member.token_id for member in persisted_members)
        checkpoint_token_ids = tuple(token.token_id for token in checkpoint_tokens)

        if persisted_token_ids != checkpoint_token_ids:
            raise AuditIntegrityError(
                f"Aggregation batch {batch_id!r} for node {node_id!r} advanced beyond the latest checkpoint. "
                f"Checkpoint tokens={list(checkpoint_token_ids)!r}; "
                f"persisted batch_members={list(persisted_token_ids)!r}. "
                "Cannot safely resume this batch in place."
            )

        return len(persisted_members)

    def get_batch_id(self, node_id: NodeID) -> str | None:
        """Get current batch ID for an aggregation node.

        Args:
            node_id: Aggregation node ID

        Returns:
            Batch ID if a batch is in progress, None otherwise

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        return self._get_node(node_id, "get_batch_id").batch_id

    def should_flush(self, node_id: NodeID) -> bool:
        """Check if the aggregation should flush based on trigger config.

        Args:
            node_id: Aggregation node ID

        Returns:
            True if trigger condition is met, False otherwise

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        return self._get_node(node_id, "should_flush").trigger.should_trigger()

    def get_trigger_type(self, node_id: NodeID) -> "TriggerType | None":
        """Get the TriggerType for the trigger that fired.

        Args:
            node_id: Aggregation node ID

        Returns:
            TriggerType enum if a trigger fired, None otherwise

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        return self._get_node(node_id, "get_trigger_type").trigger.get_trigger_type()

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
        node = self._get_node(node_id, "check_flush_status")
        should_flush = node.trigger.should_trigger()
        trigger_type = node.trigger.get_trigger_type() if should_flush else None
        return (should_flush, trigger_type)

    def restore_state(self, node_id: NodeID, state: AggregationCheckpointState) -> None:
        """Restore aggregation state from checkpoint.

        Called during recovery to restore plugin state. The state is stored
        on the node's consolidated state for plugin access via get_restored_state().

        Args:
            node_id: Aggregation node ID
            state: Typed aggregation checkpoint state

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
                This is Tier 1 checkpoint data — unknown node_id indicates corruption
                or configuration mismatch between the checkpoint and current pipeline.
        """
        node = self._get_node(node_id, "restore_state")
        node.restored_state = state

    def get_restored_state(self, node_id: NodeID) -> AggregationCheckpointState | None:
        """Get restored state for an aggregation node.

        Used by aggregation plugins during recovery to restore their
        internal state from checkpoint.

        Args:
            node_id: Aggregation node ID

        Returns:
            Typed checkpoint state, or None if no state was restored

        Raises:
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        return self._get_node(node_id, "get_restored_state").restored_state

    def restore_batch(self, batch_id: str) -> None:
        """Restore a batch as the current in-progress batch.

        Called during recovery to resume a batch that was in progress
        when the crash occurred.

        Args:
            batch_id: The batch to restore as current

        Raises:
            AuditIntegrityError: If batch not found in audit trail
        """
        batch = self._execution.get_batch(batch_id)
        if batch is None:
            raise AuditIntegrityError(f"Batch not found in audit trail: {batch_id}")

        node_id = NodeID(batch.aggregation_node_id)
        node = self._get_node(node_id, "restore_batch")
        node.batch_id = batch_id

        # Restore member count from database
        members = self._execution.get_batch_members(batch_id)
        node.member_count = len(members)
