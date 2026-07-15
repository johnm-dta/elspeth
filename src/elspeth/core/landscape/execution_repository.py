"""Execution recording repository — compatibility facade.

The behaviour lives in the cohesive components under
:mod:`elspeth.core.landscape.execution` (filigree elspeth-c227effc89):
node states and routing events (:class:`NodeStateRepository`), the external
call audit trail with thread-safe call index allocation
(:class:`CallAuditRepository`), source/sink operation lifecycle
(:class:`OperationRepository`), aggregation batches
(:class:`BatchRepository`), and sink artifacts
(:class:`ArtifactRepository`). :class:`ExecutionRepository` composes them
behind the historical surface so call sites can migrate incrementally —
new code should prefer the component attributes (``.node_states``,
``.calls``, ``.operations``, ``.batches``, ``.artifacts``) over the flat
delegators.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, overload

from sqlalchemy.engine import Connection

from elspeth.contracts import (
    Artifact,
    Batch,
    BatchMember,
    BatchStatus,
    Call,
    CallStatus,
    CallType,
    CoalesceFailureReason,
    NodeState,
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStatePending,
    NodeStateStatus,
    Operation,
    OperationType,
    RoutingEvent,
    RoutingMode,
    RoutingReason,
    RoutingSpec,
    TriggerType,
)
from elspeth.contracts.call_data import CallPayload
from elspeth.contracts.errors import ExecutionError, TransformErrorReason
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution import (
    ArtifactRepository,
    BatchRepository,
    CallAuditRepository,
    NodeStateRepository,
    OperationRepository,
)
from elspeth.core.landscape.model_loaders import (
    ArtifactLoader,
    BatchLoader,
    BatchMemberLoader,
    CallLoader,
    NodeStateLoader,
    OperationLoader,
    RoutingEventLoader,
)
from elspeth.core.landscape.row_data import CallDataResult

if TYPE_CHECKING:
    from elspeth.contracts.errors import TransformSuccessReason
    from elspeth.contracts.node_state_context import NodeStateContext
    from elspeth.contracts.payload_store import PayloadStore

__all__ = ["ExecutionRepository"]


class ExecutionRepository:
    """Node state recording, external call tracking, and batch management.

    Compatibility facade over the execution components: every historical
    verb delegates to exactly one component. The components share the same
    :class:`LandscapeDB` and :class:`DatabaseOps` instances, so test seams
    that patch ``repo._db`` / ``repo._ops`` attributes remain effective.
    """

    def __init__(
        self,
        db: LandscapeDB,
        ops: DatabaseOps,
        *,
        node_state_loader: NodeStateLoader,
        routing_event_loader: RoutingEventLoader,
        call_loader: CallLoader,
        operation_loader: OperationLoader,
        batch_loader: BatchLoader,
        batch_member_loader: BatchMemberLoader,
        artifact_loader: ArtifactLoader,
        payload_store: PayloadStore | None = None,
    ) -> None:
        self._db = db
        self._ops = ops
        self.node_states = NodeStateRepository(
            db,
            ops,
            node_state_loader=node_state_loader,
            routing_event_loader=routing_event_loader,
            payload_store=payload_store,
        )
        self.calls = CallAuditRepository(db, ops, call_loader=call_loader, payload_store=payload_store)
        self.operations = OperationRepository(db, ops, operation_loader=operation_loader, payload_store=payload_store)
        self.batches = BatchRepository(db, ops, batch_loader=batch_loader, batch_member_loader=batch_member_loader)
        self.artifacts = ArtifactRepository(ops, artifact_loader=artifact_loader)

    # ── Node state recording (NodeStateRepository) ─────────────────────

    def begin_node_state(
        self,
        token_id: str,
        node_id: str,
        run_id: str,
        step_index: int,
        input_data: Mapping[str, object],
        *,
        state_id: str | None = None,
        attempt: int = 0,
        quarantined: bool = False,
        resume_checkpoint_id: str | None = None,
    ) -> NodeStateOpen:
        """Begin recording a node state (token visiting a node)."""
        return self.node_states.begin_node_state(
            token_id,
            node_id,
            run_id,
            step_index,
            input_data,
            state_id=state_id,
            attempt=attempt,
            quarantined=quarantined,
            resume_checkpoint_id=resume_checkpoint_id,
        )

    def record_completed_node_state(
        self,
        token_id: str,
        node_id: str,
        run_id: str,
        step_index: int,
        input_data: Mapping[str, object],
        output_data: Mapping[str, object] | list[Mapping[str, object]],
        duration_ms: float,
        *,
        state_id: str | None = None,
        attempt: int = 0,
        quarantined: bool = False,
        success_reason: TransformSuccessReason | None = None,
        context_after: NodeStateContext | None = None,
    ) -> NodeStateCompleted:
        """Insert an immediately completed node state in one audit transaction."""
        return self.node_states.record_completed_node_state(
            token_id,
            node_id,
            run_id,
            step_index,
            input_data,
            output_data,
            duration_ms,
            state_id=state_id,
            attempt=attempt,
            quarantined=quarantined,
            success_reason=success_reason,
            context_after=context_after,
        )

    def begin_node_states_many(
        self,
        entries: Sequence[tuple[str, str, str, int, Mapping[str, object]]],
    ) -> list[NodeStateOpen]:
        """Begin many node states in one audit transaction."""
        return self.node_states.begin_node_states_many(entries)

    @overload
    def complete_node_state(
        self,
        state_id: str,
        status: Literal[NodeStateStatus.PENDING],
        *,
        output_data: Mapping[str, object] | list[Mapping[str, object]] | None = None,
        duration_ms: float | None = None,
        error: ExecutionError | TransformErrorReason | CoalesceFailureReason | None = None,
        context_after: NodeStateContext | None = None,
    ) -> NodeStatePending: ...

    @overload
    def complete_node_state(
        self,
        state_id: str,
        status: Literal[NodeStateStatus.COMPLETED],
        *,
        output_data: Mapping[str, object] | list[Mapping[str, object]] | None = None,
        duration_ms: float | None = None,
        error: ExecutionError | TransformErrorReason | CoalesceFailureReason | None = None,
        success_reason: TransformSuccessReason | None = None,
        context_after: NodeStateContext | None = None,
    ) -> NodeStateCompleted: ...

    @overload
    def complete_node_state(
        self,
        state_id: str,
        status: Literal[NodeStateStatus.FAILED],
        *,
        output_data: Mapping[str, object] | list[Mapping[str, object]] | None = None,
        duration_ms: float | None = None,
        error: ExecutionError | TransformErrorReason | CoalesceFailureReason | None = None,
        context_after: NodeStateContext | None = None,
    ) -> NodeStateFailed: ...

    def complete_node_state(
        self,
        state_id: str,
        status: NodeStateStatus,
        *,
        output_data: Mapping[str, object] | list[Mapping[str, object]] | None = None,
        duration_ms: float | None = None,
        error: ExecutionError | TransformErrorReason | CoalesceFailureReason | None = None,
        success_reason: TransformSuccessReason | None = None,
        context_after: NodeStateContext | None = None,
    ) -> NodeStatePending | NodeStateCompleted | NodeStateFailed:
        """Complete a node state (PENDING, COMPLETED, or FAILED)."""
        return self.node_states.complete_node_state(
            state_id,
            status,
            output_data=output_data,
            duration_ms=duration_ms,
            error=error,
            success_reason=success_reason,
            context_after=context_after,
        )

    def complete_node_states_completed_many(
        self,
        completions: Sequence[tuple[str, Mapping[str, object], float]],
        *,
        conn: Connection | None = None,
    ) -> None:
        """Complete many node states as COMPLETED in one audit transaction."""
        return self.node_states.complete_node_states_completed_many(completions, conn=conn)

    def get_node_state(self, state_id: str) -> NodeState | None:
        """Get a node state by ID."""
        return self.node_states.get_node_state(state_id)

    def get_max_node_state_attempts(self, run_id: str, token_ids: Sequence[str], *, step_index: int | None = None) -> dict[str, int]:
        """Max ``node_states.attempt`` per token (F1 resume attempt-offset derivation)."""
        return self.node_states.get_max_node_state_attempts(run_id, token_ids, step_index=step_index)

    def get_open_node_state_ids(
        self,
        run_id: str,
        *,
        node_ids: Sequence[str],
        token_ids: Sequence[str],
    ) -> dict[str, str]:
        """Outstanding (OPEN) node_state hold ids per token at the given nodes."""
        return self.node_states.get_open_node_state_ids(run_id, node_ids=node_ids, token_ids=token_ids)

    def get_completed_row_ids_for_nodes(
        self,
        run_id: str,
        node_ids: frozenset[str],
    ) -> set[tuple[str, str]]:
        """Get (node_id, row_id) pairs where a node_state has been completed."""
        return self.node_states.get_completed_row_ids_for_nodes(run_id, node_ids)

    def has_completed_row_for_node(self, *, run_id: str, node_id: str, row_id: str) -> bool:
        """Return whether one row completed at one node in one run."""
        return self.node_states.has_completed_row_for_node(run_id=run_id, node_id=node_id, row_id=row_id)

    def record_routing_event(
        self,
        state_id: str,
        edge_id: str,
        mode: RoutingMode,
        reason: RoutingReason | None = None,
        *,
        event_id: str | None = None,
        routing_group_id: str | None = None,
        ordinal: int = 0,
        reason_ref: str | None = None,
    ) -> RoutingEvent:
        """Record a single routing event."""
        return self.node_states.record_routing_event(
            state_id,
            edge_id,
            mode,
            reason,
            event_id=event_id,
            routing_group_id=routing_group_id,
            ordinal=ordinal,
            reason_ref=reason_ref,
        )

    def record_routing_events(
        self,
        state_id: str,
        routes: list[RoutingSpec],
        reason: RoutingReason | None = None,
    ) -> list[RoutingEvent]:
        """Record multiple routing events (fork/multi-destination)."""
        return self.node_states.record_routing_events(state_id, routes, reason)

    # ── Call recording (CallAuditRepository) ───────────────────────────

    def allocate_call_index(self, state_id: str) -> int:
        """Allocate next call index for a state_id (thread-safe)."""
        return self.calls.allocate_call_index(state_id)

    def record_call(
        self,
        state_id: str,
        call_index: int,
        call_type: CallType,
        status: CallStatus,
        request_data: CallPayload,
        response_data: CallPayload | None = None,
        error: CallPayload | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> Call:
        """Record an external call for a node state."""
        return self.calls.record_call(
            state_id,
            call_index,
            call_type,
            status,
            request_data,
            response_data,
            error,
            latency_ms,
            request_ref=request_ref,
            response_ref=response_ref,
            resolved_prompt_template_hash=resolved_prompt_template_hash,
        )

    # === Operations (Source/Sink I/O) ===

    def begin_operation(
        self,
        run_id: str,
        node_id: str,
        operation_type: OperationType,
        *,
        input_data: Mapping[str, object] | None = None,
    ) -> Operation:
        """Begin an operation for source/sink I/O."""
        return self.operations.begin_operation(run_id, node_id, operation_type, input_data=input_data)

    def complete_operation(
        self,
        operation_id: str,
        status: Literal["completed", "failed"],
        *,
        output_data: Mapping[str, object] | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Complete an operation."""
        return self.operations.complete_operation(
            operation_id,
            status,
            output_data=output_data,
            error=error,
            duration_ms=duration_ms,
        )

    def allocate_operation_call_index(self, operation_id: str) -> int:
        """Allocate next call index for an operation_id (thread-safe)."""
        return self.calls.allocate_operation_call_index(operation_id)

    def record_operation_call(
        self,
        operation_id: str,
        call_type: CallType,
        status: CallStatus,
        request_data: CallPayload,
        response_data: CallPayload | None = None,
        error: CallPayload | None = None,
        latency_ms: float | None = None,
        *,
        call_index: int | None = None,
        request_ref: str | None = None,
        response_ref: str | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> Call:
        """Record an external call made during an operation."""
        return self.calls.record_operation_call(
            operation_id,
            call_type,
            status,
            request_data,
            response_data,
            error,
            latency_ms,
            call_index=call_index,
            request_ref=request_ref,
            response_ref=response_ref,
            resolved_prompt_template_hash=resolved_prompt_template_hash,
        )

    def get_operation(self, operation_id: str) -> Operation | None:
        """Get an operation by ID."""
        return self.operations.get_operation(operation_id)

    def get_operation_calls(self, operation_id: str) -> list[Call]:
        """Get external calls for an operation."""
        return self.calls.get_operation_calls(operation_id)

    def get_operations_for_run(self, run_id: str) -> list[Operation]:
        """Get all operations for a run."""
        return self.operations.get_operations_for_run(run_id)

    def get_all_operation_calls_for_run(self, run_id: str) -> list[Call]:
        """Get all operation-parented calls for a run (batch query)."""
        return self.calls.get_all_operation_calls_for_run(run_id)

    def find_call_by_request_hash(
        self,
        run_id: str,
        call_type: CallType,
        request_hash: str,
        *,
        sequence_index: int = 0,
    ) -> Call | None:
        """Find a call by its request hash within a run (replay lookup)."""
        return self.calls.find_call_by_request_hash(run_id, call_type, request_hash, sequence_index=sequence_index)

    def get_call_response_data(self, call_id: str) -> CallDataResult:
        """Retrieve the response data for a call with explicit state."""
        return self.calls.get_call_response_data(call_id)

    # ── Batch recording (BatchRepository) ──────────────────────────────

    def create_batch(
        self,
        run_id: str,
        aggregation_node_id: str,
        *,
        batch_id: str | None = None,
        attempt: int = 0,
    ) -> Batch:
        """Create a new batch for aggregation."""
        return self.batches.create_batch(run_id, aggregation_node_id, batch_id=batch_id, attempt=attempt)

    def add_batch_member(
        self,
        batch_id: str,
        token_id: str,
        ordinal: int,
        *,
        conn: Connection | None = None,
    ) -> BatchMember:
        """Add a token to a batch."""
        return self.batches.add_batch_member(batch_id, token_id, ordinal, conn=conn)

    def update_batch_status(
        self,
        batch_id: str,
        status: BatchStatus,
        *,
        trigger_type: TriggerType | None = None,
        trigger_reason: str | None = None,
        state_id: str | None = None,
    ) -> None:
        """Update batch status."""
        return self.batches.update_batch_status(
            batch_id,
            status,
            trigger_type=trigger_type,
            trigger_reason=trigger_reason,
            state_id=state_id,
        )

    def complete_batch(
        self,
        batch_id: str,
        status: BatchStatus,
        *,
        trigger_type: TriggerType | None = None,
        trigger_reason: str | None = None,
        state_id: str | None = None,
    ) -> Batch:
        """Complete a batch."""
        return self.batches.complete_batch(
            batch_id,
            status,
            trigger_type=trigger_type,
            trigger_reason=trigger_reason,
            state_id=state_id,
        )

    def get_batch(self, batch_id: str) -> Batch | None:
        """Get a batch by ID."""
        return self.batches.get_batch(batch_id)

    def get_batches(
        self,
        run_id: str,
        *,
        status: BatchStatus | None = None,
        node_id: str | None = None,
    ) -> list[Batch]:
        """Get batches for a run."""
        return self.batches.get_batches(run_id, status=status, node_id=node_id)

    def get_incomplete_batches(self, run_id: str) -> list[Batch]:
        """Get batches that need recovery (draft, executing, or failed)."""
        return self.batches.get_incomplete_batches(run_id)

    def get_batch_members(self, batch_id: str) -> list[BatchMember]:
        """Get all members of a batch."""
        return self.batches.get_batch_members(batch_id)

    def get_all_batch_members_for_run(self, run_id: str) -> list[BatchMember]:
        """Get all batch members for a run (batch query)."""
        return self.batches.get_all_batch_members_for_run(run_id)

    def retry_batch(self, batch_id: str) -> Batch:
        """Create a new batch attempt from a failed batch (idempotent)."""
        return self.batches.retry_batch(batch_id)

    # === Artifact Registration (ArtifactRepository) ===

    def register_artifact(
        self,
        run_id: str,
        state_id: str,
        sink_node_id: str,
        artifact_type: str,
        path: str,
        content_hash: str,
        size_bytes: int,
        *,
        artifact_id: str | None = None,
        idempotency_key: str | None = None,
        conn: Connection | None = None,
    ) -> Artifact:
        """Register an artifact produced by a sink."""
        return self.artifacts.register_artifact(
            run_id,
            state_id,
            sink_node_id,
            artifact_type,
            path,
            content_hash,
            size_bytes,
            artifact_id=artifact_id,
            idempotency_key=idempotency_key,
            conn=conn,
        )

    def get_artifacts(
        self,
        run_id: str,
        *,
        sink_node_id: str | None = None,
    ) -> list[Artifact]:
        """Get artifacts for a run."""
        return self.artifacts.get_artifacts(run_id, sink_node_id=sink_node_id)
