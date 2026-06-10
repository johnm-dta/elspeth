"""AggregationExecutor - manages batch lifecycle with audit recording."""

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

import elspeth.contracts.errors as contract_errors
from elspeth.contracts import (
    BatchTransformProtocol,
    ExecutionError,
    PipelineRow,
    TokenInfo,
    TransformResult,
)
from elspeth.contracts.barrier_scalars import AggregationNodeScalars
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
from elspeth.contracts.node_state_context import AggregationBatchContext, AggregationFlushContext
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.scheduler import TokenWorkItem
from elspeth.contracts.types import NodeID, StepResolver
from elspeth.core.canonical import stable_hash
from elspeth.core.config import AggregationSettings
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.engine.clock import DEFAULT_CLOCK
from elspeth.engine.executors.state_guard import NodeStateGuard
from elspeth.engine.spans import SpanFactory
from elspeth.engine.triggers import TriggerEvaluator

if TYPE_CHECKING:
    from elspeth.engine.clock import Clock

slog = structlog.get_logger(__name__)


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
    # Durable counters used to derive AggregationBatchContext pagination metadata.
    # NOT persisted in the checkpoint row (F1 design D3): they derive from audit
    # tables at restore time and arrive via restore_from_journal.
    accepted_count_total: int = 0
    completed_flush_count: int = 0


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

        # Single consolidated dict replaces 6 parallel dicts:
        # _aggregation_settings, _trigger_evaluators, _batch_ids,
        # _member_counts, _buffers, _buffer_tokens
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
        # Durable cumulative counter that drives AggregationBatchContext.rows_seen_total.
        # Incremented exactly once per accepted row; persisted in the checkpoint.
        node.accepted_count_total += 1

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

            # Compute durable batch-context pagination metadata from the executor's
            # counters. accepted_count_total has already been incremented for every
            # row in buffered_rows (see buffer_row()), so:
            #   row_end   = accepted_count_total
            #   row_start = accepted_count_total - len(buffered_rows) + 1
            #   flush_index = completed_flush_count + 1 (this is the (N+1)-th flush)
            # completed_flush_count is only incremented on the success path below
            # so a retry of a failed flush gets the same flush_index.
            batch_size = len(buffered_rows)
            rows_seen_total = node.accepted_count_total
            row_start = rows_seen_total - batch_size + 1
            row_end = rows_seen_total
            flush_index = node.completed_flush_count + 1
            is_end_of_source = trigger_type is TriggerType.END_OF_SOURCE
            batch_context = AggregationBatchContext(
                trigger_type=trigger_type.value,
                batch_id=batch_id,
                batch_size=batch_size,
                flush_index=flush_index,
                rows_seen_total=rows_seen_total,
                row_start=row_start,
                row_end=row_end,
                is_end_of_source=is_end_of_source,
            )
            ctx.aggregation_batch = batch_context

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

                    # Normalize trigger_type to its str value so AggregationFlushContext
                    # (annotated as str) matches AggregationBatchContext exactly and a
                    # future non-StrEnum refactor cannot silently produce divergent
                    # serializations between the two contexts.
                    flush_context = AggregationFlushContext(
                        trigger_type=trigger_type.value,
                        buffer_size=batch_size,
                        batch_id=batch_id,
                        flush_index=flush_index,
                        rows_seen_total=rows_seen_total,
                        row_start=row_start,
                        row_end=row_end,
                        is_end_of_source=is_end_of_source,
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
                    # Durable counter — only advances on a successfully completed
                    # flush. A retry of a failed flush therefore receives the
                    # same flush_index, so the audit pagination is stable
                    # across retries.
                    node.completed_flush_count += 1
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
                ctx.aggregation_batch = None
                raise

        # Success cleanup: save batch_id before reset (needed by caller for CONSUMED_IN_BATCH)
        flushed_batch_id = batch_id

        # Reset for next batch and clear buffers
        self._reset_batch_state(node_id)
        node.buffers.clear()
        node.tokens.clear()

        # Reset trigger evaluator for next batch
        node.trigger.reset()

        # Clear batch_token_ids and aggregation_batch to prevent stale data
        # leaking to subsequent transforms on the shared context.
        ctx.batch_token_ids = None
        ctx.aggregation_batch = None

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

    def get_barrier_scalars(self) -> dict[NodeID, AggregationNodeScalars]:
        """Return the underivable trigger-latch scalars for the checkpoint row.

        F1 design D3: the checkpoint persists ONLY scalar barrier metadata —
        buffered tokens live in journal BLOCKED rows and counters derive from
        audit tables at restore time. The only underivable aggregation state
        is the pair of trigger fire-time latches, read live from each node's
        TriggerEvaluator.

        Emission choice: only nodes with at least one non-None fire offset are
        emitted. The checkpoint writer serializes None when no scalars exist
        (``BarrierScalars.has_state``), and restore treats a missing entry as
        ``(None, None)`` — so emitting unlatched/counter-only nodes would add
        bytes without information. This matches the Task 1.2 transitional
        projection, which skipped counter-only nodes.

        Returns:
            Mapping of node_id -> AggregationNodeScalars for latched nodes only.
        """
        scalars: dict[NodeID, AggregationNodeScalars] = {}
        for node_id, node in self._nodes.items():
            count_fire_offset = node.trigger.get_count_fire_offset()
            condition_fire_offset = node.trigger.get_condition_fire_offset()
            if count_fire_offset is None and condition_fire_offset is None:
                continue
            scalars[node_id] = AggregationNodeScalars(
                count_fire_offset=count_fire_offset,
                condition_fire_offset=condition_fire_offset,
            )
        return scalars

    def restore_from_journal(
        self,
        *,
        node_id: NodeID,
        items: Sequence[TokenWorkItem],
        member_order: Sequence[str],
        batch_id: str | None,
        accepted_count_total: int,
        completed_flush_count: int,
        scalars: AggregationNodeScalars,
        attempt_offsets: Mapping[str, int],
        resume_checkpoint_id: str,
        now: datetime,
    ) -> None:
        """Rebuild one node's buffers from journal BLOCKED rows (F1 resume path).

        Replaces the checkpoint-blob restore: the journal (token_work_items
        BLOCKED rows) is authoritative for buffered token payloads; the caller
        (processor, Task 3.1) partitions journal items by barrier_key and
        derives batch_id / member_order / counters / attempt offsets from
        audit tables.

        Args:
            node_id: Aggregation node being restored.
            items: BLOCKED journal rows for this node's barrier_key.
            member_order: Token ids in batch_members.ordinal order — the
                authoritative accept order for buffer reconstruction.
            batch_id: The in-progress batch id (None for a counter-only node).
            accepted_count_total: Audit-derived cumulative accept counter.
            completed_flush_count: Audit-derived completed-flush counter.
            scalars: Trigger fire-time latches from the checkpoint row.
            attempt_offsets: Per-token resume attempt offset (max_attempt + 1).
            resume_checkpoint_id: Checkpoint id stamped on restored tokens
                (resume provenance).
            now: Current wall-clock time (tz-aware) — trigger age derives from
                ``now - min(barrier_blocked_at)``, not from an offset blob.

        Raises:
            AuditIntegrityError: On any journal/audit disagreement — NULL
                barrier_blocked_at, duplicate journal rows, membership
                mismatch, missing attempt offset, batch_id/items inconsistency.
            OrchestrationInvariantError: If node_id is not a configured aggregation.
        """
        node = self._get_node(node_id, "restore_from_journal")

        tokens_by_id: dict[str, TokenInfo] = {}
        oldest_blocked_at: datetime | None = None
        for item in items:
            if item.barrier_blocked_at is None:
                # Every post-epoch-20 BLOCKED row is stamped by mark_blocked.
                raise AuditIntegrityError(
                    f"BLOCKED journal row for token {item.token_id!r} at aggregation node "
                    f"{node_id!r} has NULL barrier_blocked_at — journal corruption "
                    "(every BLOCKED row is stamped at mark_blocked time)."
                )
            if item.token_id in tokens_by_id:
                raise AuditIntegrityError(
                    f"Duplicate BLOCKED journal rows for token {item.token_id!r} at aggregation node {node_id!r} — journal corruption."
                )
            attempt_offset = attempt_offsets.get(item.token_id)
            if attempt_offset is None:
                raise AuditIntegrityError(
                    f"No entry in attempt_offsets for journal token {item.token_id!r} at "
                    f"aggregation node {node_id!r} — audit-derived offsets must cover "
                    "every BLOCKED journal row."
                )

            row_data = TokenSchedulerRepository.deserialize_row_payload(item.row_payload_json)
            tokens_by_id[item.token_id] = TokenInfo(
                row_id=item.row_id,
                token_id=item.token_id,
                row_data=row_data,
                branch_name=item.branch_name,
                fork_group_id=item.fork_group_id,
                join_group_id=item.join_group_id,
                expand_group_id=item.expand_group_id,
                resume_attempt_offset=attempt_offset,
                resume_checkpoint_id=resume_checkpoint_id,
            )
            if oldest_blocked_at is None or item.barrier_blocked_at < oldest_blocked_at:
                oldest_blocked_at = item.barrier_blocked_at

        self._reconcile_journal_batch_members(
            node_id=node_id,
            journal_token_ids=tokens_by_id.keys(),
            member_order=member_order,
        )

        # batch_id/items must agree: buffered journal rows imply an in-progress
        # batch; a batch_id with zero BLOCKED rows means batch membership
        # advanced past the journal (or vice versa) — corruption either way.
        if items and batch_id is None:
            raise AuditIntegrityError(
                f"Aggregation node {node_id!r} has {len(items)} BLOCKED journal rows but no "
                "batch_id — buffered tokens always belong to an in-progress batch."
            )
        if not items and batch_id is not None:
            raise AuditIntegrityError(
                f"Aggregation node {node_id!r} has batch_id {batch_id!r} but no BLOCKED "
                "journal rows — an in-progress batch must have blocked members."
            )

        # Counter sanity: the cumulative accept counter covers every currently
        # buffered row, so accepted_count_total < len(items) (or any negative
        # counter) is impossible audit state. Restoring it would silently emit
        # row_start <= 0 in the next flush's pagination metadata.
        if completed_flush_count < 0 or accepted_count_total < len(items):
            raise AuditIntegrityError(
                f"Aggregation node {node_id!r}: audit-derived counters are impossible "
                f"(accepted_count_total={accepted_count_total}, "
                f"completed_flush_count={completed_flush_count}, buffered={len(items)}). "
                "accepted_count_total must cover every buffered row and counters must "
                "be non-negative."
            )

        ordered_tokens = [tokens_by_id[token_id] for token_id in member_order]

        node.tokens = ordered_tokens
        node.buffers = [t.row_data.to_dict() for t in ordered_tokens]
        node.batch_id = batch_id
        node.member_count = len(ordered_tokens)
        node.accepted_count_total = accepted_count_total
        node.completed_flush_count = completed_flush_count

        # Trigger age derives from the absolute blocked-at stamp of the OLDEST
        # buffered row (first accept of the in-progress batch), clamped at 0
        # against clock skew. Counter-only nodes restore an empty trigger with
        # the scalars passed through (mirrors the old counters-only arm).
        if oldest_blocked_at is not None:
            elapsed_age_seconds = max(0.0, (now - oldest_blocked_at).total_seconds())
        else:
            elapsed_age_seconds = 0.0
        node.trigger.restore_from_checkpoint(
            batch_count=len(ordered_tokens),
            elapsed_age_seconds=elapsed_age_seconds,
            count_fire_offset=scalars.count_fire_offset,
            condition_fire_offset=scalars.condition_fire_offset,
        )

        slog.info(
            "aggregation_journal_restored",
            node_id=str(node_id),
            token_count=len(ordered_tokens),
            batch_id=batch_id,
            accepted_count_total=accepted_count_total,
            completed_flush_count=completed_flush_count,
            elapsed_age_seconds=elapsed_age_seconds,
        )

    @staticmethod
    def _reconcile_journal_batch_members(
        *,
        node_id: NodeID,
        journal_token_ids: Iterable[str],
        member_order: Sequence[str],
    ) -> None:
        """Ensure journal BLOCKED rows and persisted batch_members agree as SETS.

        This is the F1 descendant of the old checkpoint-vs-batch_members
        reconcile. It degenerates to set equality because ``member_order`` IS
        the batch_members.ordinal ordering (derived by the caller) — comparing
        ordered tuples would compare batch_members against itself, proving
        nothing. The real cross-check left is membership: a token with a
        BLOCKED journal row but no batch_members row (or vice versa) means the
        journal and the audit trail disagree about the in-progress batch.
        """
        journal_set = set(journal_token_ids)
        member_set = set(member_order)
        if len(member_set) != len(member_order):
            raise AuditIntegrityError(
                f"Duplicate token ids in batch_members order for aggregation node {node_id!r}: "
                f"{list(member_order)!r} — audit trail corruption."
            )
        if journal_set != member_set:
            missing_from_journal = sorted(member_set - journal_set)
            missing_from_members = sorted(journal_set - member_set)
            raise AuditIntegrityError(
                f"Aggregation node {node_id!r}: journal BLOCKED rows and persisted "
                f"batch_members disagree about batch membership. "
                f"In batch_members but not journal: {missing_from_journal!r}; "
                f"in journal but not batch_members: {missing_from_members!r}. "
                "Cannot safely resume this batch."
            )

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
