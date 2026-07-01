"""ExecutionRepository: node state recording, call tracking, and batch management.

Owns thread-safe call index allocation (Lock + per-state and per-operation dicts).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, overload
from uuid import uuid4

from sqlalchemy import bindparam, func, select
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts import (
    Artifact,
    Batch,
    BatchMember,
    BatchStatus,
    Call,
    CallStatus,
    CallType,
    CoalesceFailureReason,
    FrameworkBugError,
    NodeState,
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStatePending,
    NodeStateStatus,
    Operation,
    RoutingEvent,
    RoutingMode,
    RoutingReason,
    RoutingSpec,
    TriggerType,
)
from elspeth.contracts.audit import validate_resolved_prompt_template_hash
from elspeth.contracts.call_data import CallPayload
from elspeth.contracts.errors import AuditIntegrityError, ExecutionError, TransformErrorReason
from elspeth.contracts.hashing import repr_hash
from elspeth.contracts.payload_store import IntegrityError as PayloadIntegrityError
from elspeth.contracts.payload_store import PayloadNotFoundError
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import generate_id, now
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapePostCommitError, LandscapeRecordError
from elspeth.core.landscape.model_loaders import (
    ArtifactLoader,
    BatchLoader,
    BatchMemberLoader,
    CallLoader,
    NodeStateLoader,
    OperationLoader,
    RoutingEventLoader,
)
from elspeth.core.landscape.row_data import CallDataResult, CallDataState
from elspeth.core.landscape.schema import (
    artifacts_table,
    batch_members_table,
    batches_table,
    calls_table,
    node_states_table,
    operations_table,
    routing_events_table,
    tokens_table,
)

if TYPE_CHECKING:
    from elspeth.contracts.errors import TransformSuccessReason
    from elspeth.contracts.node_state_context import NodeStateContext
    from elspeth.contracts.payload_store import PayloadStore

_TERMINAL_BATCH_STATUSES = frozenset({BatchStatus.COMPLETED, BatchStatus.FAILED})
_TERMINAL_NODE_STATE_STATUSES = frozenset({NodeStateStatus.COMPLETED, NodeStateStatus.FAILED})
_COMPLETABLE_OPERATION_STATUSES = frozenset({"completed", "failed"})
# IN-clause chunk size for token-id lookups — stays under SQLite's default
# 999 bound-parameter ceiling with headroom for the fixed predicates.
_TOKEN_ID_CHUNK_SIZE = 500


def _validate_transform_success_reason(success_reason: object) -> None:
    """Validate TransformSuccessReason shape before Tier 1 audit serialization."""
    if success_reason is None:
        return
    if not isinstance(success_reason, Mapping):
        raise ValueError(f"success_reason must be a mapping, got {type(success_reason).__name__}")
    action = success_reason.get("action")
    if not isinstance(action, str):
        raise ValueError("success_reason['action'] must be a str")


class _PreparedCallData(NamedTuple):
    """Intermediate result from _prepare_call_payloads.

    Carries stable hashes plus any payload bytes that still need post-insert
    materialization into the payload store.
    """

    request_hash: str
    request_ref: str | None
    response_hash: str | None
    response_ref: str | None
    error_json: str | None
    request_bytes: bytes | None
    response_bytes: bytes | None


class ExecutionRepository:
    """Node state recording, external call tracking, and batch management.

    Consolidates three former mixins into a composed repository with
    explicit constructor injection of all dependencies.
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
        self._node_state_loader = node_state_loader
        self._routing_event_loader = routing_event_loader
        self._call_loader = call_loader
        self._operation_loader = operation_loader
        self._batch_loader = batch_loader
        self._batch_member_loader = batch_member_loader
        self._artifact_loader = artifact_loader
        self._payload_store = payload_store

        # Thread-safe call index allocation (internal state, not injected)
        self._call_indices: dict[str, int] = {}
        self._call_index_lock = Lock()
        self._operation_call_indices: dict[str, int] = {}

    # ── Node state recording ────────────────────────────────────────────

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
        """Begin recording a node state (token visiting a node).

        Args:
            token_id: Token being processed
            node_id: Node processing the token
            run_id: Run ID for composite FK to nodes table
            step_index: Position in token's execution path
            input_data: Input data for hashing
            state_id: Optional state ID (generated if not provided)
            attempt: Attempt number (0 for first attempt)
            quarantined: If True, input_data is Tier-3 external data that may
                contain non-canonical values (NaN, Infinity). Uses repr_hash fallback.
            resume_checkpoint_id: Checkpoint ID this node_state was re-driven from
                during a resume operation. NULL for every original-run write; set when
                re-driving an incomplete token so explain() can distinguish resume
                re-drives from run-1 tenacity retries (epoch 11).

        Returns:
            NodeStateOpen model with status=OPEN
        """
        state_id = state_id or generate_id()
        if quarantined:
            try:
                input_hash = stable_hash(input_data)
            except (ValueError, TypeError):
                input_hash = repr_hash(input_data)
        else:
            input_hash = stable_hash(input_data)
        timestamp = now()

        state = NodeStateOpen(
            state_id=state_id,
            token_id=token_id,
            node_id=node_id,
            step_index=step_index,
            attempt=attempt,
            status=NodeStateStatus.OPEN,
            input_hash=input_hash,
            context_before_json=None,
            started_at=timestamp,
        )

        self._ops.execute_insert(
            node_states_table.insert().values(
                state_id=state.state_id,
                token_id=state.token_id,
                node_id=state.node_id,
                run_id=run_id,  # Added for composite FK to nodes
                step_index=state.step_index,
                attempt=state.attempt,
                status=state.status,
                input_hash=state.input_hash,
                started_at=state.started_at,
                resume_checkpoint_id=resume_checkpoint_id,
            )
        )

        return state

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
        """Insert an immediately completed node state in one audit transaction.

        Source nodes are observed before processor traversal begins, so the
        common successful source path has no useful OPEN interval. This method
        preserves the same completed-row invariants as ``complete_node_state``
        without writing a transient OPEN row first.
        """
        state_id = state_id or generate_id()
        if quarantined:
            try:
                input_hash = stable_hash(input_data)
            except (ValueError, TypeError):
                input_hash = repr_hash(input_data)
        else:
            input_hash = stable_hash(input_data)
        output_hash = stable_hash(output_data)
        timestamp = now()
        _validate_transform_success_reason(success_reason)
        success_reason_json = canonical_json(success_reason) if success_reason is not None else None
        context_json = canonical_json(context_after.to_dict()) if context_after is not None else None

        try:
            with self._db.write_connection() as conn:
                result = conn.execute(
                    node_states_table.insert().values(
                        state_id=state_id,
                        token_id=token_id,
                        node_id=node_id,
                        run_id=run_id,
                        step_index=step_index,
                        attempt=attempt,
                        status=NodeStateStatus.COMPLETED.value,
                        input_hash=input_hash,
                        output_hash=output_hash,
                        duration_ms=duration_ms,
                        error_json=None,
                        success_reason_json=success_reason_json,
                        context_after_json=context_json,
                        started_at=timestamp,
                        completed_at=timestamp,
                    )
                )
                if result.rowcount == 0:
                    raise LandscapeRecordError(
                        f"record_completed_node_state: zero rows affected for state_id={state_id} — audit write failed"
                    )
                row = conn.execute(select(node_states_table).where(node_states_table.c.state_id == state_id)).fetchone()
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"record_completed_node_state failed for state_id={state_id} — database rejected audit write: {type(exc).__name__}: {exc}"
            ) from exc

        if row is None:
            raise LandscapeRecordError(f"NodeState {state_id} not found after insert — database corruption or transaction failure")
        try:
            loaded = self._node_state_loader.load(row)
        except AuditIntegrityError as exc:
            raise LandscapePostCommitError(f"NodeState {state_id} became unreadable immediately after insert: {exc}") from exc
        if loaded.status is not NodeStateStatus.COMPLETED:
            raise LandscapePostCommitError(f"NodeState {state_id} should be COMPLETED after atomic insert but has status {loaded.status}")
        return loaded

    def begin_node_states_many(
        self,
        entries: Sequence[tuple[str, str, str, int, Mapping[str, object]]],
    ) -> list[NodeStateOpen]:
        """Begin many node states in one audit transaction.

        This is intentionally narrower than ``begin_node_state``: it supports
        the high-volume sink success path where every entry is a normal
        non-quarantined first attempt. Each token still receives its own
        node_states row; only the database round trips are batched.
        """
        if not entries:
            return []

        timestamp = now()
        states: list[NodeStateOpen] = []
        values: list[dict[str, object]] = []
        for token_id, node_id, run_id, step_index, input_data in entries:
            input_hash = stable_hash(input_data)
            state = NodeStateOpen(
                state_id=generate_id(),
                token_id=token_id,
                node_id=node_id,
                step_index=step_index,
                attempt=0,
                status=NodeStateStatus.OPEN,
                input_hash=input_hash,
                context_before_json=None,
                started_at=timestamp,
            )
            states.append(state)
            values.append(
                {
                    "state_id": state.state_id,
                    "token_id": state.token_id,
                    "node_id": state.node_id,
                    "run_id": run_id,
                    "step_index": state.step_index,
                    "attempt": state.attempt,
                    "status": state.status,
                    "input_hash": state.input_hash,
                    "started_at": state.started_at,
                }
            )

        try:
            with self._db.write_connection() as conn:
                result = conn.execute(node_states_table.insert(), values)
                if result.rowcount not in (-1, len(values)):
                    raise LandscapeRecordError(
                        f"begin_node_states_many affected {result.rowcount} rows for {len(values)} states; "
                        "expected one audit row per token."
                    )
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"begin_node_states_many failed for {len(values)} states — database rejected audit write: {type(exc).__name__}: {exc}"
            ) from exc
        return states

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
        """Complete a node state.

        Args:
            state_id: State to complete
            status: NodeStateStatus (PENDING, COMPLETED, or FAILED)
            output_data: Output data for hashing (if success)
            duration_ms: Processing duration (required)
            error: Error details (if failed)
            context_after: Optional context snapshot after processing

        Returns:
            NodeStatePending if status is pending, NodeStateCompleted if completed, NodeStateFailed if failed

        Raises:
            ValueError: If status is OPEN (not a valid terminal status)
            ValueError: If duration_ms is not provided
        """
        if status == NodeStateStatus.OPEN:
            raise ValueError("Cannot complete a node state with status OPEN")

        if duration_ms is None:
            raise ValueError("duration_ms is required when completing a node state")

        # Required fields per status
        if status == NodeStateStatus.COMPLETED and output_data is None:
            raise ValueError("COMPLETED node state requires output_data (output_hash would be NULL)")

        if status == NodeStateStatus.FAILED and error is None:
            raise ValueError("FAILED node state requires error details")

        # Forbidden fields per status — prevent writing impossible states to Tier 1 data.
        # These mirror the read-side checks in NodeStateLoader.load().
        if status == NodeStateStatus.PENDING:
            if output_data is not None:
                raise ValueError("PENDING node state must not have output_data")
            if error is not None:
                raise ValueError("PENDING node state must not have error")
            if success_reason is not None:
                raise ValueError("PENDING node state must not have success_reason")

        if status == NodeStateStatus.COMPLETED and error is not None:
            raise ValueError("COMPLETED node state must not have error (contradicts success)")

        if status == NodeStateStatus.FAILED and success_reason is not None:
            raise ValueError("FAILED node state must not have success_reason (contradicts failure)")

        timestamp = now()
        output_hash = stable_hash(output_data) if output_data is not None else None
        # ExecutionError and CoalesceFailureReason are frozen dataclasses with
        # to_dict(); TransformErrorReason is a TypedDict (already a dict).
        if error is not None:
            error_data = error.to_dict() if isinstance(error, (ExecutionError, CoalesceFailureReason)) else error
            error_json = canonical_json(error_data)
        else:
            error_json = None
        context_json = canonical_json(context_after.to_dict()) if context_after is not None else None
        # Serialize success reason if provided (use canonical_json for audit consistency)
        _validate_transform_success_reason(success_reason)
        success_reason_json = canonical_json(success_reason) if success_reason is not None else None

        # Single transaction: UPDATE + SELECT-back for atomicity.
        # Prevents a concurrent reader from seeing the row between states.
        # Atomic conditional UPDATE: guard against already-terminal status in the
        # WHERE clause (same TOCTOU-safe pattern as complete_batch).
        terminal_values = [s.value for s in _TERMINAL_NODE_STATE_STATUSES]
        try:
            with self._db.write_connection() as conn:
                update_result = conn.execute(
                    node_states_table.update()
                    .where(node_states_table.c.state_id == state_id)
                    .where(node_states_table.c.status.notin_(terminal_values))
                    .values(
                        status=status,
                        output_hash=output_hash,
                        duration_ms=duration_ms,
                        error_json=error_json,
                        success_reason_json=success_reason_json,
                        context_after_json=context_json,
                        completed_at=timestamp,
                    )
                )
                if update_result.rowcount == 0:
                    # Distinguish "not found" from "already terminal".
                    existing = conn.execute(select(node_states_table.c.status).where(node_states_table.c.state_id == state_id)).fetchone()
                    if existing is not None:
                        raise LandscapeRecordError(
                            f"Cannot complete node state {state_id}: current status {existing.status!r} is already terminal. "
                            f"Terminal node states are immutable."
                        )
                    raise LandscapeRecordError(
                        f"complete_node_state: zero rows affected for state_id={state_id} — target row does not exist (audit data corruption)"
                    )

                row = conn.execute(select(node_states_table).where(node_states_table.c.state_id == state_id)).fetchone()
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"complete_node_state failed for state_id={state_id} — database rejected audit update: {type(exc).__name__}: {exc}"
            ) from exc

        if row is None:
            raise LandscapeRecordError(f"NodeState {state_id} not found after update — database corruption or transaction failure")
        try:
            result = self._node_state_loader.load(row)
        except AuditIntegrityError as exc:
            raise LandscapePostCommitError(f"NodeState {state_id} became unreadable immediately after completion: {exc}") from exc
        # Type narrowing: result is guaranteed to be terminal (PENDING/COMPLETED/FAILED)
        if result.status is NodeStateStatus.OPEN:
            raise LandscapePostCommitError(f"NodeState {state_id} should be terminal after completion but has status OPEN")
        return result

    def complete_node_states_completed_many(
        self,
        completions: Sequence[tuple[str, Mapping[str, object], float]],
    ) -> None:
        """Complete many node states as COMPLETED in one audit transaction.

        The method preserves the single-row immutability and post-write loader
        validation contract from ``complete_node_state`` while avoiding one
        transaction per sink token in high-volume writes.
        """
        if not completions:
            return

        timestamp = now()
        terminal_values = [status.value for status in _TERMINAL_NODE_STATE_STATUSES]
        state_ids = [state_id for state_id, _output_data, _duration_ms in completions]

        params: list[dict[str, object]] = []
        for state_id, output_data, duration_ms in completions:
            if duration_ms is None:
                raise ValueError("duration_ms is required when completing a node state")
            if output_data is None:
                raise ValueError("COMPLETED node state requires output_data (output_hash would be NULL)")
            params.append(
                {
                    "batch_state_id": state_id,
                    "batch_status": NodeStateStatus.COMPLETED.value,
                    "batch_output_hash": stable_hash(output_data),
                    "batch_duration_ms": duration_ms,
                    "batch_completed_at": timestamp,
                }
            )

        stmt = (
            node_states_table.update()
            .where(node_states_table.c.state_id == bindparam("batch_state_id"))
            .where(node_states_table.c.status != NodeStateStatus.COMPLETED.value)
            .where(node_states_table.c.status != NodeStateStatus.FAILED.value)
            .values(
                status=bindparam("batch_status"),
                output_hash=bindparam("batch_output_hash"),
                duration_ms=bindparam("batch_duration_ms"),
                error_json=None,
                success_reason_json=None,
                context_after_json=None,
                completed_at=bindparam("batch_completed_at"),
            )
        )
        try:
            with self._db.write_connection() as conn:
                before_rows = conn.execute(
                    select(node_states_table.c.state_id, node_states_table.c.status).where(node_states_table.c.state_id.in_(state_ids))
                ).fetchall()
                before_by_id = {row.state_id: row.status for row in before_rows}
                missing = [state_id for state_id in state_ids if state_id not in before_by_id]
                if missing:
                    raise LandscapeRecordError(
                        f"complete_node_states_completed_many: target rows do not exist (state_ids={missing!r}) — audit data corruption"
                    )
                terminal = [(state_id, before_by_id[state_id]) for state_id in state_ids if before_by_id[state_id] in terminal_values]
                if terminal:
                    first_state_id, first_status = terminal[0]
                    raise LandscapeRecordError(
                        f"Cannot complete node state {first_state_id}: current status {first_status!r} is already terminal. "
                        "Terminal node states are immutable."
                    )

                result = conn.execute(stmt, params)
                if result.rowcount not in (-1, len(params)):
                    raise LandscapeRecordError(
                        f"complete_node_states_completed_many affected {result.rowcount} rows for {len(params)} states; "
                        "expected one audit update per token."
                    )
                after_rows = conn.execute(select(node_states_table).where(node_states_table.c.state_id.in_(state_ids))).fetchall()
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"complete_node_states_completed_many failed for {len(params)} states — database rejected audit update: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if len(after_rows) != len(params):
            raise LandscapePostCommitError(
                f"complete_node_states_completed_many loaded {len(after_rows)} states after update; expected {len(params)}."
            )
        for row in after_rows:
            try:
                loaded = self._node_state_loader.load(row)
            except AuditIntegrityError as exc:
                raise LandscapePostCommitError(f"NodeState {row.state_id} became unreadable immediately after completion: {exc}") from exc
            if loaded.status is not NodeStateStatus.COMPLETED:
                raise LandscapePostCommitError(
                    f"NodeState {row.state_id} should be COMPLETED after batch completion but has status {loaded.status}"
                )

    def get_node_state(self, state_id: str) -> NodeState | None:
        """Get a node state by ID.

        Args:
            state_id: State ID to retrieve

        Returns:
            NodeState (union of Open, Pending, Completed, or Failed) or None
        """
        query = select(node_states_table).where(node_states_table.c.state_id == state_id)
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None
        return self._node_state_loader.load(row)

    def get_max_node_state_attempts(self, run_id: str, token_ids: Sequence[str], *, step_index: int | None = None) -> dict[str, int]:
        """Max ``node_states.attempt`` per token (F1 resume attempt-offset derivation).

        The resume restore path stamps every journal-restored token with
        ``resume_attempt_offset = max_attempt + 1`` so re-driven node_states
        never collide with attempts already recorded in the audit trail.
        Tokens with no node_states rows are absent from the result (callers
        treat absence as max_attempt = -1, i.e. offset 0).

        Args:
            run_id: Run ID to scope the query.
            token_ids: Tokens to look up (chunked internally).
            step_index: Optional step scope. The node_states uniqueness key is
                ``(token_id, step_index, attempt)``, so a re-drive that only
                writes ONE step (the PENDING_SINK sink write) derives its
                offset from that step alone — a max over all steps would
                over-bump for tokens whose earlier transform steps recorded
                attempts the sink step never saw.

        Returns:
            Mapping of token_id -> max attempt observed in node_states.
        """
        result: dict[str, int] = {}
        for i in range(0, len(token_ids), _TOKEN_ID_CHUNK_SIZE):
            chunk = list(token_ids[i : i + _TOKEN_ID_CHUNK_SIZE])
            query = (
                select(node_states_table.c.token_id, func.max(node_states_table.c.attempt).label("max_attempt"))
                .where(node_states_table.c.run_id == run_id)
                .where(node_states_table.c.token_id.in_(chunk))
                .group_by(node_states_table.c.token_id)
            )
            if step_index is not None:
                query = query.where(node_states_table.c.step_index == step_index)
            for row in self._ops.execute_fetchall(query):
                result[row.token_id] = int(row.max_attempt)
        return result

    def get_open_node_state_ids(
        self,
        run_id: str,
        *,
        node_ids: Sequence[str],
        token_ids: Sequence[str],
    ) -> dict[str, str]:
        """Outstanding (OPEN) node_state hold ids per token at the given nodes.

        F1 resume derivation for coalesce ``state_ids``: a held branch's
        node_state is written by ``begin_node_state`` at accept() time (status
        OPEN — the "pending hold") and is completed only when the coalesce
        resolves, so the un-completed OPEN row at the coalesce node IS the
        hold whose ``state_id`` the restored ``_BranchEntry`` must carry.

        If a token somehow has multiple OPEN states at the queried nodes,
        the highest attempt wins (rows are scanned in attempt order and the
        last write per token survives).

        Args:
            run_id: Run ID to scope the query.
            node_ids: Node IDs to match (e.g. the coalesce node ids).
            token_ids: Tokens to look up (chunked internally).

        Returns:
            Mapping of token_id -> state_id of the outstanding hold.
        """
        if not node_ids:
            return {}
        result: dict[str, str] = {}
        for i in range(0, len(token_ids), _TOKEN_ID_CHUNK_SIZE):
            chunk = list(token_ids[i : i + _TOKEN_ID_CHUNK_SIZE])
            query = (
                select(node_states_table.c.token_id, node_states_table.c.state_id)
                .where(node_states_table.c.run_id == run_id)
                .where(node_states_table.c.node_id.in_(list(node_ids)))
                .where(node_states_table.c.token_id.in_(chunk))
                .where(node_states_table.c.status == NodeStateStatus.OPEN.value)
                .order_by(node_states_table.c.token_id, node_states_table.c.attempt)
            )
            for row in self._ops.execute_fetchall(query):
                result[row.token_id] = row.state_id
        return result

    def get_completed_row_ids_for_nodes(
        self,
        run_id: str,
        node_ids: frozenset[str],
    ) -> set[tuple[str, str]]:
        """Get (node_id, row_id) pairs where a node_state has been completed.

        Used to reconstruct coalesce late-arrival detection state from the
        Landscape rather than from checkpoint data. The Landscape is the source
        of truth — checkpoint-based completed_keys is a cache optimization.

        Args:
            run_id: Run ID to scope the query
            node_ids: Set of node IDs to query (e.g., coalesce node IDs)

        Returns:
            Set of (node_id, row_id) tuples where the node_state has a
            completed_at timestamp (meaning the coalesce resolved — success
            or failure).
        """
        if not node_ids:
            return set()

        query = (
            select(node_states_table.c.node_id, tokens_table.c.row_id)
            .select_from(
                node_states_table.join(
                    tokens_table,
                    node_states_table.c.token_id == tokens_table.c.token_id,
                )
            )
            .where(
                node_states_table.c.run_id == run_id,
                node_states_table.c.node_id.in_(node_ids),
                node_states_table.c.completed_at.isnot(None),
            )
            .distinct()
        )
        rows = self._ops.execute_fetchall(query)
        return {(row.node_id, row.row_id) for row in rows}

    def has_completed_row_for_node(self, *, run_id: str, node_id: str, row_id: str) -> bool:
        """Return whether one row completed at one node in one run.

        This is the point lookup used by coalesce late-arrival detection.
        It avoids materializing every completed row for a coalesce node on
        ordinary cache misses.
        """
        query = (
            select(node_states_table.c.state_id)
            .select_from(
                node_states_table.join(
                    tokens_table,
                    (node_states_table.c.token_id == tokens_table.c.token_id) & (node_states_table.c.run_id == tokens_table.c.run_id),
                )
            )
            .where(
                node_states_table.c.run_id == run_id,
                node_states_table.c.node_id == node_id,
                node_states_table.c.completed_at.isnot(None),
                tokens_table.c.run_id == run_id,
                tokens_table.c.row_id == row_id,
            )
            .limit(1)
        )
        return self._ops.execute_fetchone(query) is not None

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
        """Record a single routing event.

        Args:
            state_id: Node state that made the routing decision
            edge_id: Edge that was taken
            mode: RoutingMode enum (MOVE or COPY)
            reason: Reason for this routing decision
            event_id: Optional event ID
            routing_group_id: Group ID (for multi-destination routing)
            ordinal: Position in routing group
            reason_ref: Optional payload store reference

        Returns:
            RoutingEvent model
        """
        event_id = event_id or generate_id()
        routing_group_id = routing_group_id or generate_id()
        reason_hash = stable_hash(reason) if reason else None
        timestamp = now()
        auto_reason_bytes = None
        if reason is not None and reason_ref is None and self._payload_store is not None:
            auto_reason_bytes = canonical_json(reason).encode("utf-8")

        event = RoutingEvent(
            event_id=event_id,
            state_id=state_id,
            edge_id=edge_id,
            routing_group_id=routing_group_id,
            ordinal=ordinal,
            mode=mode,
            reason_hash=reason_hash,
            reason_ref=reason_ref if auto_reason_bytes is None else None,
            created_at=timestamp,
        )

        self._ops.execute_insert(
            routing_events_table.insert().values(
                event_id=event.event_id,
                state_id=event.state_id,
                edge_id=event.edge_id,
                routing_group_id=event.routing_group_id,
                ordinal=event.ordinal,
                mode=event.mode,
                reason_hash=event.reason_hash,
                reason_ref=event.reason_ref,
                created_at=event.created_at,
            )
        )

        if auto_reason_bytes is not None:
            materialized_reason_ref = self._materialize_routing_reason_ref_after_insert(
                reason_bytes=auto_reason_bytes,
                event_id=event.event_id,
                expected_rows=1,
            )
            return RoutingEvent(
                event_id=event.event_id,
                state_id=event.state_id,
                edge_id=event.edge_id,
                routing_group_id=event.routing_group_id,
                ordinal=event.ordinal,
                mode=event.mode,
                reason_hash=event.reason_hash,
                reason_ref=materialized_reason_ref,
                created_at=event.created_at,
            )

        return event

    def record_routing_events(
        self,
        state_id: str,
        routes: list[RoutingSpec],
        reason: RoutingReason | None = None,
    ) -> list[RoutingEvent]:
        """Record multiple routing events (fork/multi-destination).

        All events share the same routing_group_id.

        Args:
            state_id: Node state that made the routing decision
            routes: List of RoutingSpec objects specifying edge_id and mode
            reason: Shared reason for all routes

        Returns:
            List of RoutingEvent models
        """
        if not routes:
            return []

        routing_group_id = generate_id()
        reason_hash = stable_hash(reason) if reason else None
        timestamp = now()
        auto_reason_bytes = None
        if reason is not None and self._payload_store is not None:
            auto_reason_bytes = canonical_json(reason).encode("utf-8")

        inserted_events: list[RoutingEvent] = []
        try:
            with self._db.write_connection() as conn:
                for ordinal, route in enumerate(routes):
                    event_id = generate_id()
                    event = RoutingEvent(
                        event_id=event_id,
                        state_id=state_id,
                        edge_id=route.edge_id,
                        routing_group_id=routing_group_id,
                        ordinal=ordinal,
                        mode=route.mode,  # Already RoutingMode enum from RoutingSpec
                        reason_hash=reason_hash,
                        reason_ref=None,
                        created_at=timestamp,
                    )

                    result = conn.execute(
                        routing_events_table.insert().values(
                            event_id=event.event_id,
                            state_id=event.state_id,
                            edge_id=event.edge_id,
                            routing_group_id=event.routing_group_id,
                            ordinal=event.ordinal,
                            mode=event.mode,
                            reason_hash=event.reason_hash,
                            reason_ref=event.reason_ref,
                            created_at=event.created_at,
                        )
                    )
                    if result.rowcount == 0:
                        raise AuditIntegrityError(f"Failed to insert routing event {event_id} for state {state_id} - zero rows affected")

                    inserted_events.append(event)
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"record_routing_events failed for state_id={state_id} — database rejected audit write: {type(exc).__name__}: {exc}"
            ) from exc

        reason_ref = None
        if auto_reason_bytes is not None:
            reason_ref = self._materialize_routing_reason_ref_after_insert(
                reason_bytes=auto_reason_bytes,
                routing_group_id=routing_group_id,
                expected_rows=len(inserted_events),
            )

        return [
            RoutingEvent(
                event_id=event.event_id,
                state_id=event.state_id,
                edge_id=event.edge_id,
                routing_group_id=event.routing_group_id,
                ordinal=event.ordinal,
                mode=event.mode,
                reason_hash=event.reason_hash,
                reason_ref=reason_ref,
                created_at=event.created_at,
            )
            for event in inserted_events
        ]

    # ── Call recording ─────────────────────────────────────────────────

    def allocate_call_index(self, state_id: str) -> int:
        """Allocate next call index for a state_id (thread-safe).

        Provides centralized call index allocation ensuring UNIQUE(state_id, call_index)
        across all client types (HTTP, LLM) and retry attempts.

        This is the single source of truth for call numbering. All AuditedClient
        instances MUST delegate to this method rather than maintaining their own counters.

        Thread Safety:
            Uses a lock to prevent race conditions when multiple threads allocate
            indices concurrently. Safe for pooled execution scenarios.

        Persistence:
            Counter seeds from the database on first access per state_id,
            so it survives recorder recreation (e.g., on resume). Subsequent
            allocations are pure in-memory for performance.

        Args:
            state_id: Node state ID to allocate index for

        Returns:
            Sequential call index (0-based), unique within this state_id

        Example:
            # Two different client types, same state_id
            http_client = AuditedHTTPClient(recorder, state_id="state-001")
            llm_client = AuditedLLMClient(recorder, state_id="state-001")

            # Both delegate to same recorder - indices coordinate correctly
            http_client.post(...)  # allocates index 0
            llm_client.query(...)   # allocates index 1 (not 0!)
        """
        with self._call_index_lock:
            if state_id not in self._call_indices:
                # Slow path (once per state_id): seed from database to survive
                # recorder recreation on resume. Without this, a new recorder
                # would restart indices at 0 for any state_id that already has
                # recorded calls, causing UNIQUE(state_id, call_index) violations.
                # The DB query is serialized under the lock — acceptable because
                # it only fires once per state_id per recorder lifetime. All
                # subsequent allocations hit the fast path (no DB access).
                row = self._ops.execute_fetchone(select(func.max(calls_table.c.call_index)).where(calls_table.c.state_id == state_id))
                existing_max = row[0] if row is not None and row[0] is not None else -1
                self._call_indices[state_id] = existing_max + 1
            # Fast path: allocate from in-memory counter (no DB access)
            idx = self._call_indices[state_id]
            self._call_indices[state_id] += 1
            return idx

    def _prepare_call_payloads(
        self,
        request_data: CallPayload,
        response_data: CallPayload | None,
        error: CallPayload | None,
        request_ref: str | None,
        response_ref: str | None,
    ) -> _PreparedCallData:
        """Serialize, hash, and stage call payloads for post-insert storage.

        Shared logic for record_call() and record_operation_call(). Converts
        CallPayload objects to dicts, computes stable hashes, prepares any
        payload bytes that still need storing after the call row exists, and
        serializes the error payload.
        """
        request_dict = request_data.to_dict()
        response_dict = response_data.to_dict() if response_data is not None else None

        request_hash = stable_hash(request_dict)
        response_hash = stable_hash(response_dict) if response_dict is not None else None

        request_bytes = None
        response_bytes = None
        if request_ref is None and self._payload_store is not None:
            request_bytes = canonical_json(request_dict).encode("utf-8")
        if response_dict is not None and response_ref is None and self._payload_store is not None:
            response_bytes = canonical_json(response_dict).encode("utf-8")

        error_json = canonical_json(error.to_dict()) if error is not None else None

        return _PreparedCallData(
            request_hash=request_hash,
            request_ref=request_ref,
            response_hash=response_hash,
            response_ref=response_ref,
            error_json=error_json,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
        )

    def _materialize_call_ref_after_insert(
        self,
        *,
        call_id: str,
        column_name: str,
        payload_bytes: bytes | None,
    ) -> str | None:
        """Store one call payload and update the already-recorded call row."""
        if payload_bytes is None:
            return None
        if self._payload_store is None:
            raise FrameworkBugError(f"_materialize_call_ref_after_insert({call_id!r}, {column_name!r}) requires a payload store")

        try:
            payload_ref = self._payload_store.store(payload_bytes)
            self._ops.execute_update(
                calls_table.update().where(calls_table.c.call_id == call_id).values(**{column_name: payload_ref}),
                context=f"calls.{column_name} for {call_id}",
            )
        except Exception as exc:
            raise LandscapePostCommitError(
                f"Call {call_id} was recorded, but {column_name} materialization failed: {type(exc).__name__}: {exc}"
            ) from exc
        return payload_ref

    def _materialize_routing_reason_ref_after_insert(
        self,
        *,
        reason_bytes: bytes,
        expected_rows: int,
        event_id: str | None = None,
        routing_group_id: str | None = None,
    ) -> str:
        """Store a routing reason after the event rows already exist."""
        if self._payload_store is None:
            raise FrameworkBugError("_materialize_routing_reason_ref_after_insert() requires a payload store")
        if (event_id is None) == (routing_group_id is None):
            raise FrameworkBugError("Routing reason materialization requires exactly one of event_id or routing_group_id")
        target = event_id if event_id is not None else routing_group_id

        try:
            reason_ref = self._payload_store.store(reason_bytes)
            stmt = routing_events_table.update().values(reason_ref=reason_ref)
            if event_id is not None:
                stmt = stmt.where(routing_events_table.c.event_id == event_id)
            else:
                stmt = stmt.where(routing_events_table.c.routing_group_id == routing_group_id)

            with self._db.write_connection() as conn:
                result = conn.execute(stmt)
                if result.rowcount != expected_rows:
                    raise AuditIntegrityError(
                        f"Routing reason ref update for {target} affected {result.rowcount} rows; expected {expected_rows}"
                    )
        except Exception as exc:
            raise LandscapePostCommitError(
                f"Routing event(s) {target} were recorded, but reason materialization failed: {type(exc).__name__}: {exc}"
            ) from exc
        return reason_ref

    def _materialize_call_refs_after_insert(self, call_id: str, prepared: _PreparedCallData) -> tuple[str | None, str | None]:
        """Store staged call payloads after the call row has been committed."""
        request_ref = prepared.request_ref
        response_ref = prepared.response_ref

        materialized_request_ref = self._materialize_call_ref_after_insert(
            call_id=call_id,
            column_name="request_ref",
            payload_bytes=prepared.request_bytes,
        )
        if materialized_request_ref is not None:
            request_ref = materialized_request_ref

        materialized_response_ref = self._materialize_call_ref_after_insert(
            call_id=call_id,
            column_name="response_ref",
            payload_bytes=prepared.response_bytes,
        )
        if materialized_response_ref is not None:
            response_ref = materialized_response_ref

        return request_ref, response_ref

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
        """Record an external call for a node state.

        Args:
            state_id: The node_state this call belongs to
            call_index: 0-based index of this call within the state
            call_type: Type of external call (LLM, HTTP, SQL, FILESYSTEM)
            status: Outcome of the call (SUCCESS, ERROR)
            request_data: Request payload (CallPayload — serialized internally)
            response_data: Response payload (CallPayload — serialized internally, optional for errors)
            error: Error payload if status is ERROR (CallPayload — serialized internally)
            latency_ms: Call duration in milliseconds
            request_ref: Optional payload store reference for request
            response_ref: Optional payload store reference for response
            resolved_prompt_template_hash: Cross-DB hash anchor (Phase 5b Task 9).
                When this LLM-transform call is downstream of an interpretation
                event the L3 plugin forwards the SHA-256 of the resolved prompt
                template string here; the value MUST equal
                ``interpretation_events.resolved_prompt_template_hash`` in the
                session audit DB for the same resolved string. ``None`` for
                non-LLM calls or for LLM transforms not downstream of an
                interpretation event.

        Returns:
            The recorded Call model

        Note:
            Duplicate (state_id, call_index) will raise IntegrityError from SQLAlchemy.
            Invalid state_id will raise IntegrityError due to foreign key constraint.
            Call indices should be allocated via allocate_call_index() for coordination.
        """
        # Validate the cross-DB prompt-hash anchor BEFORE inserting. The Call
        # dataclass enforces the same invariant in __post_init__, but that runs
        # AFTER execute_insert — a bad hash would commit to `calls` and only then
        # raise. Checking here keeps the audit trail pristine (Tier 1): a bad
        # hash leaves zero rows (elspeth-a94e626a36).
        validate_resolved_prompt_template_hash(call_type, resolved_prompt_template_hash)

        call_id = generate_id()
        timestamp = now()
        prepared = self._prepare_call_payloads(
            request_data,
            response_data,
            error,
            request_ref,
            response_ref,
        )

        values = {
            "call_id": call_id,
            "state_id": state_id,
            "operation_id": None,  # State call, not operation call
            "call_index": call_index,
            "call_type": call_type,
            "status": status,
            "request_hash": prepared.request_hash,
            "request_ref": prepared.request_ref,
            "response_hash": prepared.response_hash,
            "response_ref": prepared.response_ref,
            "resolved_prompt_template_hash": resolved_prompt_template_hash,
            "error_json": prepared.error_json,
            "latency_ms": latency_ms,
            "created_at": timestamp,
        }

        self._ops.execute_insert(calls_table.insert().values(**values))
        request_ref, response_ref = self._materialize_call_refs_after_insert(call_id, prepared)

        return Call(
            call_id=call_id,
            call_index=call_index,
            call_type=call_type,
            status=status,
            request_hash=prepared.request_hash,
            created_at=timestamp,
            state_id=state_id,
            operation_id=None,
            request_ref=request_ref,
            response_hash=prepared.response_hash,
            response_ref=response_ref,
            error_json=prepared.error_json,
            latency_ms=latency_ms,
            resolved_prompt_template_hash=resolved_prompt_template_hash,
        )

    # === Operations (Source/Sink I/O) ===

    def begin_operation(
        self,
        run_id: str,
        node_id: str,
        operation_type: Literal["source_load", "sink_write", "runtime_preflight"],
        *,
        input_data: Mapping[str, object] | None = None,
    ) -> Operation:
        """Begin an operation for source/sink I/O.

        Operations are the source/sink equivalent of node_states - they provide
        a parent context for external calls made during load() or write().

        Args:
            run_id: Run this operation belongs to
            node_id: Source or sink node performing the operation
            operation_type: Type of operation
            input_data: Optional input context (stored via payload store)

        Returns:
            Operation with operation_id for call attribution
        """
        # Use pure UUID for operation_id - run_id + node_id can exceed 64 chars
        # (run_id=36 + node_id=45 + prefixes would be 94+ chars)
        operation_id = f"op_{uuid4().hex}"  # "op_" + 32 hex = 35 chars, well under 64

        input_ref = None
        input_hash = None
        if input_data is not None:
            input_hash = stable_hash(input_data)
            if self._payload_store is not None:
                input_bytes = canonical_json(input_data).encode("utf-8")
                input_ref = self._payload_store.store(input_bytes)

        timestamp = now()
        operation = Operation(
            operation_id=operation_id,
            run_id=run_id,
            node_id=node_id,
            operation_type=operation_type,
            started_at=timestamp,
            status="open",
            input_data_ref=input_ref,
            input_data_hash=input_hash,
        )

        self._ops.execute_insert(operations_table.insert().values(**operation.to_dict()))
        return operation

    def complete_operation(
        self,
        operation_id: str,
        status: Literal["completed", "failed"],
        *,
        output_data: Mapping[str, object] | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Complete an operation.

        Args:
            operation_id: Operation to complete
            status: Final status ('completed' or 'failed')
            output_data: Optional output context
            error: Error message if failed
            duration_ms: Operation duration

        Raises:
            FrameworkBugError: If operation doesn't exist or is already completed
        """
        # Validate lifecycle invariants before persisting — matches Operation.__post_init__.
        # These are Tier 1 guards: impossible states in the audit trail are framework bugs.
        if status not in _COMPLETABLE_OPERATION_STATUSES:
            raise FrameworkBugError(
                f"complete_operation({operation_id!r}): unsupported status {status!r}; "
                f"expected one of {sorted(_COMPLETABLE_OPERATION_STATUSES)}"
            )
        if duration_ms is None:
            raise FrameworkBugError(f"complete_operation({operation_id!r}): status={status!r} but duration_ms is None")
        if status == "completed" and error is not None:
            raise FrameworkBugError(f"complete_operation({operation_id!r}): status='completed' but error is set")
        if status == "failed" and error is None:
            raise FrameworkBugError(f"complete_operation({operation_id!r}): status='failed' but error is None")
        if status == "failed" and error == "":
            raise FrameworkBugError(f"complete_operation({operation_id!r}): status='failed' but error must not be empty")

        # Atomic check-and-update: WHERE constrains both identity and status
        # to eliminate the TOCTOU race between separate SELECT and UPDATE.
        # Payload storage is deferred until AFTER the status check succeeds
        # to avoid orphaned blobs on duplicate-completion races or invalid IDs.
        timestamp = now()
        output_hash = stable_hash(output_data) if output_data is not None else None
        row = None
        stmt = (
            operations_table.update()
            .where((operations_table.c.operation_id == operation_id) & (operations_table.c.status == "open"))
            .values(
                completed_at=timestamp,
                status=status,
                error_message=error,
                duration_ms=duration_ms,
                output_data_hash=output_hash,
            )
        )
        with self._db.write_connection() as conn:
            result = conn.execute(stmt)
            if result.rowcount == 0:
                # Distinguish "doesn't exist" from "already completed" for diagnostics
                check = conn.execute(select(operations_table.c.status).where(operations_table.c.operation_id == operation_id)).fetchone()
                if check is None:
                    raise FrameworkBugError(f"Completing non-existent operation: {operation_id}")
                raise FrameworkBugError(
                    f"Completing already-completed operation {operation_id}: current status={check.status}, new status={status}"
                )

            # Store payload only after confirming the operation row was updated
            if output_data is not None and self._payload_store is not None:
                output_bytes = canonical_json(output_data).encode("utf-8")
                output_ref = self._payload_store.store(output_bytes)
                ref_result = conn.execute(
                    operations_table.update().where(operations_table.c.operation_id == operation_id).values(output_data_ref=output_ref)
                )
                if ref_result.rowcount == 0:
                    raise AuditIntegrityError(
                        f"complete_operation: output_data_ref UPDATE affected zero rows for "
                        f"operation {operation_id} — row disappeared between status update "
                        f"and ref update (database corruption)"
                    )
            row = conn.execute(select(operations_table).where(operations_table.c.operation_id == operation_id)).fetchone()

        if row is None:
            raise LandscapePostCommitError(
                f"Operation {operation_id} not found after completion — database corruption or transaction failure"
            )
        try:
            self._operation_loader.load(row)
        except AuditIntegrityError as exc:
            raise LandscapePostCommitError(f"Operation {operation_id} became unreadable immediately after completion: {exc}") from exc

    def allocate_operation_call_index(self, operation_id: str) -> int:
        """Allocate next call index for an operation_id (thread-safe).

        Provides centralized call index allocation ensuring unique call numbering
        within each operation. Parallel to allocate_call_index() for node_states.

        Args:
            operation_id: Operation ID to allocate index for

        Returns:
            Sequential call index (0-based), unique within this operation_id
        """
        with self._call_index_lock:  # Reuse existing lock
            if operation_id not in self._operation_call_indices:
                # Slow path (once per operation_id): seed from database to survive
                # recorder recreation on resume. Serialized under lock — acceptable
                # because it fires only once per operation_id per recorder lifetime.
                row = self._ops.execute_fetchone(
                    select(func.max(calls_table.c.call_index)).where(calls_table.c.operation_id == operation_id)
                )
                existing_max = row[0] if row is not None and row[0] is not None else -1
                self._operation_call_indices[operation_id] = existing_max + 1
            # Fast path: allocate from in-memory counter (no DB access)
            idx = self._operation_call_indices[operation_id]
            self._operation_call_indices[operation_id] += 1
            return idx

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
        """Record an external call made during an operation.

        This is the operation equivalent of record_call() - attributes calls
        to operations instead of node_states.

        Args:
            operation_id: The operation this call belongs to
            call_type: Type of external call (LLM, HTTP, SQL, FILESYSTEM)
            status: Outcome of the call (SUCCESS, ERROR)
            request_data: Request payload (CallPayload — serialized internally)
            response_data: Response payload (CallPayload — serialized internally, optional for errors)
            error: Error details if status is ERROR (stored as JSON)
            latency_ms: Call duration in milliseconds
            request_ref: Optional payload store reference for request
            response_ref: Optional payload store reference for response

        Returns:
            The recorded Call model
        """
        # Validate the cross-DB prompt-hash anchor BEFORE inserting (see the
        # state-parented record_call for rationale): a bad hash must leave zero
        # `calls` rows rather than commit and then raise from Call.__post_init__
        # (elspeth-a94e626a36).
        validate_resolved_prompt_template_hash(call_type, resolved_prompt_template_hash)

        if call_index is None:
            call_index = self.allocate_operation_call_index(operation_id)
        call_id = f"call_{operation_id}_{call_index}"
        timestamp = now()
        prepared = self._prepare_call_payloads(
            request_data,
            response_data,
            error,
            request_ref,
            response_ref,
        )

        values = {
            "call_id": call_id,
            "state_id": None,  # NOT a node_state call
            "operation_id": operation_id,  # Operation call
            "call_index": call_index,
            "call_type": call_type,
            "status": status,
            "request_hash": prepared.request_hash,
            "request_ref": prepared.request_ref,
            "response_hash": prepared.response_hash,
            "response_ref": prepared.response_ref,
            "resolved_prompt_template_hash": resolved_prompt_template_hash,
            "error_json": prepared.error_json,
            "latency_ms": latency_ms,
            "created_at": timestamp,
        }

        self._ops.execute_insert(calls_table.insert().values(**values))
        request_ref, response_ref = self._materialize_call_refs_after_insert(call_id, prepared)

        return Call(
            call_id=call_id,
            call_index=call_index,
            call_type=call_type,
            status=status,
            request_hash=prepared.request_hash,
            created_at=timestamp,
            state_id=None,
            operation_id=operation_id,
            request_ref=request_ref,
            response_hash=prepared.response_hash,
            response_ref=response_ref,
            error_json=prepared.error_json,
            latency_ms=latency_ms,
            resolved_prompt_template_hash=resolved_prompt_template_hash,
        )

    def get_operation(self, operation_id: str) -> Operation | None:
        """Get an operation by ID.

        Args:
            operation_id: Operation ID to retrieve

        Returns:
            Operation model or None if not found
        """
        query = select(operations_table).where(operations_table.c.operation_id == operation_id)
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None

        return self._operation_loader.load(row)

    def get_operation_calls(self, operation_id: str) -> list[Call]:
        """Get external calls for an operation.

        Args:
            operation_id: Operation ID

        Returns:
            List of Call models, ordered by call_index
        """
        query = select(calls_table).where(calls_table.c.operation_id == operation_id).order_by(calls_table.c.call_index)
        db_rows = self._ops.execute_fetchall(query)
        return [self._call_loader.load(r) for r in db_rows]

    def get_operations_for_run(self, run_id: str) -> list[Operation]:
        """Get all operations for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Operation models, ordered by started_at
        """
        query = select(operations_table).where(operations_table.c.run_id == run_id).order_by(operations_table.c.started_at)
        db_rows = self._ops.execute_fetchall(query)
        return [self._operation_loader.load(row) for row in db_rows]

    def get_all_operation_calls_for_run(self, run_id: str) -> list[Call]:
        """Get all operation-parented calls for a run (batch query).

        Fetches all calls where operation_id is NOT NULL and the operation
        belongs to the given run. This replaces per-operation get_operation_calls()
        loops in the exporter.

        Args:
            run_id: Run ID

        Returns:
            List of Call models, ordered by operation_id then call_index
        """
        query = (
            select(calls_table)
            .join(operations_table, calls_table.c.operation_id == operations_table.c.operation_id)
            .where(operations_table.c.run_id == run_id)
            .order_by(calls_table.c.operation_id, calls_table.c.call_index)
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._call_loader.load(r) for r in db_rows]

    def find_call_by_request_hash(
        self,
        run_id: str,
        call_type: CallType,
        request_hash: str,
        *,
        sequence_index: int = 0,
    ) -> Call | None:
        """Find a call by its request hash within a run.

        Used for replay mode to look up previously recorded calls by
        the hash of their request data.

        Args:
            run_id: Run ID to search within
            call_type: Type of call (llm, http, etc.)
            request_hash: SHA-256 hash of the canonical request data
            sequence_index: 0-based index for duplicate request hashes.
                When the same request is made multiple times in a run
                (e.g., retries, loops), use sequence_index to get the
                Nth occurrence (0=first, 1=second, etc.).

        Returns:
            Call model if found, None otherwise

        Note:
            Calls are ordered chronologically by created_at. The sequence_index
            parameter allows disambiguation when the same request was made
            multiple times (each returning a different response).
        """
        # Join to node_states to filter by run_id
        # NOTE: Use node_states.run_id directly (denormalized column) instead of
        # joining through nodes table. The nodes table has composite PK (node_id, run_id),
        # so joining on node_id alone would be ambiguous when node_id is reused across runs.
        query = (
            select(calls_table)
            .join(
                node_states_table,
                calls_table.c.state_id == node_states_table.c.state_id,
            )
            .where(node_states_table.c.run_id == run_id)
            .where(calls_table.c.call_type == call_type)
            .where(calls_table.c.request_hash == request_hash)
            .order_by(calls_table.c.created_at)
            .limit(1)
            .offset(sequence_index)
        )
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None
        return self._call_loader.load(row)

    def get_call_response_data(self, call_id: str) -> CallDataResult:
        """Retrieve the response data for a call with explicit state.

        Returns a CallDataResult with explicit state indicating why data
        may be unavailable. Callers match on state instead of guessing
        why the previous `None` return occurred.

        Args:
            call_id: The call ID to get response data for

        Returns:
            CallDataResult with state and data (if available)
        """
        # Get the call record first
        query = select(calls_table).where(calls_table.c.call_id == call_id)
        row = self._ops.execute_fetchone(query)

        if row is None:
            return CallDataResult(state=CallDataState.CALL_NOT_FOUND, data=None)

        if row.response_ref is None:
            if row.response_hash is not None:
                return CallDataResult(state=CallDataState.HASH_ONLY, data=None)
            return CallDataResult(state=CallDataState.NEVER_STORED, data=None)

        if self._payload_store is None:
            return CallDataResult(state=CallDataState.STORE_NOT_CONFIGURED, data=None)

        # Retrieve from payload store — PayloadNotFoundError means purged by
        # retention policy, PayloadIntegrityError means hash mismatch
        # (corruption/tampering), OSError means storage backend failure
        # (permissions, disk, etc.). All non-purge paths translate to
        # AuditIntegrityError with context, matching
        # query_repository._retrieve_and_parse_payload().
        try:
            payload_bytes = self._payload_store.retrieve(row.response_ref)
        except PayloadNotFoundError:
            return CallDataResult(state=CallDataState.PURGED, data=None)
        except PayloadIntegrityError as e:
            raise AuditIntegrityError(f"Payload integrity check failed for call_id={call_id} (ref={row.response_ref}): {e}") from e
        except OSError as e:
            raise AuditIntegrityError(
                f"Payload retrieval failed for call_id={call_id} (ref={row.response_ref}): {type(e).__name__}: {e}"
            ) from e

        # Everything below is Tier 1: our data, crash on anomaly
        try:
            decoded = json.loads(payload_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise AuditIntegrityError(f"Corrupt call response payload for call_id={call_id} (ref={row.response_ref}): {e}") from e
        if type(decoded) is not dict:
            raise AuditIntegrityError(
                f"Corrupt call response payload for call_id={call_id} (ref={row.response_ref}): "
                f"expected JSON object, got {type(decoded).__name__}"
            )
        return CallDataResult(state=CallDataState.AVAILABLE, data=decoded)

    # ── Batch recording ────────────────────────────────────────────────

    def create_batch(
        self,
        run_id: str,
        aggregation_node_id: str,
        *,
        batch_id: str | None = None,
        attempt: int = 0,
    ) -> Batch:
        """Create a new batch for aggregation.

        Args:
            run_id: Run this batch belongs to
            aggregation_node_id: Aggregation node collecting tokens
            batch_id: Optional batch ID (generated if not provided)
            attempt: Attempt number (0 for first attempt)

        Returns:
            Batch model with status="draft"
        """
        batch_id = batch_id or generate_id()
        timestamp = now()

        batch = Batch(
            batch_id=batch_id,
            run_id=run_id,
            aggregation_node_id=aggregation_node_id,
            attempt=attempt,
            status=BatchStatus.DRAFT,  # Strict: enum type
            created_at=timestamp,
        )

        self._ops.execute_insert(
            batches_table.insert().values(
                batch_id=batch.batch_id,
                run_id=batch.run_id,
                aggregation_node_id=batch.aggregation_node_id,
                attempt=batch.attempt,
                status=batch.status,
                created_at=batch.created_at,
            )
        )

        return batch

    def add_batch_member(
        self,
        batch_id: str,
        token_id: str,
        ordinal: int,
    ) -> BatchMember:
        """Add a token to a batch.

        Args:
            batch_id: Batch to add to
            token_id: Token to add
            ordinal: Order in batch

        Returns:
            BatchMember model
        """
        batch_row = self._ops.execute_fetchone(select(batches_table.c.run_id).where(batches_table.c.batch_id == batch_id))
        if batch_row is None:
            raise AuditIntegrityError(f"Cannot add batch member: batch {batch_id} not found")

        member = BatchMember(
            batch_id=batch_id,
            run_id=batch_row.run_id,
            token_id=token_id,
            ordinal=ordinal,
        )

        self._ops.execute_insert(
            batch_members_table.insert().values(
                batch_id=member.batch_id,
                run_id=member.run_id,
                token_id=member.token_id,
                ordinal=member.ordinal,
            ),
            context="batch_members",
        )

        return member

    def update_batch_status(
        self,
        batch_id: str,
        status: BatchStatus,
        *,
        trigger_type: TriggerType | None = None,
        trigger_reason: str | None = None,
        state_id: str | None = None,
    ) -> None:
        """Update batch status.

        Args:
            batch_id: Batch to update
            status: New BatchStatus
            trigger_type: TriggerType enum value
            trigger_reason: Human-readable reason for the trigger
            state_id: Node state for the flush operation

        Raises:
            AuditIntegrityError: If batch not found or current status is terminal
        """
        updates: dict[str, Any] = {"status": status}

        if trigger_type is not None:
            updates["trigger_type"] = trigger_type
        if trigger_reason is not None:
            updates["trigger_reason"] = trigger_reason
        if state_id is not None:
            updates["aggregation_state_id"] = state_id
        if status in (BatchStatus.COMPLETED, BatchStatus.FAILED):
            updates["completed_at"] = now()

        # Atomic conditional UPDATE: constrain current status to non-terminal in the
        # WHERE clause so the check-and-set is a single statement, eliminating the
        # TOCTOU race between the old get_batch() read and the subsequent update.
        terminal_values = [s.value for s in _TERMINAL_BATCH_STATUSES]
        try:
            with self._db.write_connection() as conn:
                result = conn.execute(
                    batches_table.update()
                    .where(batches_table.c.batch_id == batch_id)
                    .where(batches_table.c.status.notin_(terminal_values))
                    .values(**updates)
                )
                if result.rowcount == 0:
                    # Distinguish "not found" from "already terminal".
                    existing = conn.execute(select(batches_table.c.status).where(batches_table.c.batch_id == batch_id)).fetchone()
                    if existing is not None:
                        raise AuditIntegrityError(
                            f"Cannot transition batch {batch_id} from terminal status {existing.status!r} "
                            f"to {status.value!r}. Terminal batches are immutable."
                        )
                    raise AuditIntegrityError(f"Cannot update batch status: batch {batch_id} not found")
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"update_batch_status failed for batch_id={batch_id} — database rejected audit update: {type(exc).__name__}: {exc}"
            ) from exc

    def complete_batch(
        self,
        batch_id: str,
        status: BatchStatus,
        *,
        trigger_type: TriggerType | None = None,
        trigger_reason: str | None = None,
        state_id: str | None = None,
    ) -> Batch:
        """Complete a batch.

        Args:
            batch_id: Batch to complete
            status: Final BatchStatus (COMPLETED or FAILED)
            trigger_type: TriggerType enum value
            trigger_reason: Human-readable reason for the trigger
            state_id: Optional node state for the aggregation

        Returns:
            Updated Batch model

        Raises:
            AuditIntegrityError: If status is not a terminal batch status
        """
        if status not in _TERMINAL_BATCH_STATUSES:
            raise AuditIntegrityError(
                f"complete_batch() requires terminal status, got {status.value!r}. "
                f"Valid terminal statuses: {sorted(s.value for s in _TERMINAL_BATCH_STATUSES)}"
            )

        timestamp = now()

        # Atomic conditional UPDATE: guard against already-terminal status in the
        # WHERE clause (same TOCTOU-safe pattern as update_batch_status).
        terminal_values = [s.value for s in _TERMINAL_BATCH_STATUSES]
        try:
            with self._db.write_connection() as conn:
                update_result = conn.execute(
                    batches_table.update()
                    .where(batches_table.c.batch_id == batch_id)
                    .where(batches_table.c.status.notin_(terminal_values))
                    .values(
                        status=status,
                        trigger_type=trigger_type,
                        trigger_reason=trigger_reason,
                        aggregation_state_id=state_id,
                        completed_at=timestamp,
                    )
                )
                if update_result.rowcount == 0:
                    # Distinguish "not found" from "already terminal".
                    existing = conn.execute(select(batches_table.c.status).where(batches_table.c.batch_id == batch_id)).fetchone()
                    if existing is not None:
                        raise AuditIntegrityError(
                            f"Cannot complete batch {batch_id}: current status {existing.status!r} is already terminal. "
                            f"Terminal batches are immutable."
                        )
                    raise AuditIntegrityError(
                        f"complete_batch: zero rows affected for batch_id={batch_id} — target row does not exist (audit data corruption)"
                    )

                row = conn.execute(select(batches_table).where(batches_table.c.batch_id == batch_id)).fetchone()
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"complete_batch failed for batch_id={batch_id} — database rejected audit update: {type(exc).__name__}: {exc}"
            ) from exc

        if row is None:
            raise AuditIntegrityError(f"Batch {batch_id} not found after update — database corruption or transaction failure")
        return self._batch_loader.load(row)

    def get_batch(self, batch_id: str) -> Batch | None:
        """Get a batch by ID.

        Args:
            batch_id: Batch ID to retrieve

        Returns:
            Batch model or None
        """
        query = select(batches_table).where(batches_table.c.batch_id == batch_id)
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None
        return self._batch_loader.load(row)

    def get_batches(
        self,
        run_id: str,
        *,
        status: BatchStatus | None = None,
        node_id: str | None = None,
    ) -> list[Batch]:
        """Get batches for a run.

        Args:
            run_id: Run ID
            status: Optional BatchStatus filter
            node_id: Optional aggregation node filter

        Returns:
            List of Batch models, ordered by created_at then batch_id
            for deterministic export signatures.
        """
        query = select(batches_table).where(batches_table.c.run_id == run_id)

        if status is not None:
            query = query.where(batches_table.c.status == status)
        if node_id is not None:
            query = query.where(batches_table.c.aggregation_node_id == node_id)

        # Order for deterministic export signatures
        query = query.order_by(batches_table.c.created_at, batches_table.c.batch_id)
        rows = self._ops.execute_fetchall(query)
        return [self._batch_loader.load(row) for row in rows]

    def get_incomplete_batches(self, run_id: str) -> list[Batch]:
        """Get batches that need recovery (draft, executing, or failed).

        Used during crash recovery to find batches that were:
        - draft: Still collecting rows when crash occurred
        - executing: Mid-flush when crash occurred
        - failed: Flush failed and needs retry

        Args:
            run_id: The run to query

        Returns:
            List of Batch objects with status in (draft, executing, failed),
            ordered by created_at ascending (oldest first for deterministic recovery)
        """
        query = (
            select(batches_table)
            .where(batches_table.c.run_id == run_id)
            .where(batches_table.c.status.in_([BatchStatus.DRAFT, BatchStatus.EXECUTING, BatchStatus.FAILED]))
            .order_by(batches_table.c.created_at.asc())
        )
        result = self._ops.execute_fetchall(query)
        return [self._batch_loader.load(row) for row in result]

    def get_batch_members(self, batch_id: str) -> list[BatchMember]:
        """Get all members of a batch.

        Args:
            batch_id: Batch ID

        Returns:
            List of BatchMember models (ordered by ordinal)
        """
        query = select(batch_members_table).where(batch_members_table.c.batch_id == batch_id).order_by(batch_members_table.c.ordinal)
        rows = self._ops.execute_fetchall(query)
        return [self._batch_member_loader.load(row) for row in rows]

    def get_all_batch_members_for_run(self, run_id: str) -> list[BatchMember]:
        """Get all batch members for a run (batch query).

        Fetches all members for all batches in a run in one query,
        replacing per-batch get_batch_members() loops in the exporter.

        Args:
            run_id: Run ID

        Returns:
            List of BatchMember models, ordered by batch_id then ordinal
        """
        query = (
            select(batch_members_table)
            .join(
                batches_table,
                (batch_members_table.c.batch_id == batches_table.c.batch_id) & (batch_members_table.c.run_id == batches_table.c.run_id),
            )
            .where(batches_table.c.run_id == run_id)
            .order_by(batch_members_table.c.batch_id, batch_members_table.c.ordinal)
        )
        rows = self._ops.execute_fetchall(query)
        return [self._batch_member_loader.load(row) for row in rows]

    def retry_batch(self, batch_id: str) -> Batch:
        """Create a new batch attempt from a failed batch (idempotent).

        Copies batch metadata and members to a new batch with
        incremented attempt counter and draft status. If a retry batch
        already exists for this attempt, returns it without creating
        a duplicate.

        All operations (lookup, create, copy members, read-back) happen
        in a single transaction for atomicity.

        Args:
            batch_id: The failed batch to retry

        Returns:
            New or existing Batch with attempt = original.attempt + 1

        Raises:
            ValueError: If original batch not found or not in failed status
        """
        with self._db.write_connection() as conn:
            # 1. Get original batch
            original_row = conn.execute(select(batches_table).where(batches_table.c.batch_id == batch_id)).fetchone()
            if original_row is None:
                raise AuditIntegrityError(f"retry_batch: batch {batch_id} not found — audit data corruption")
            original = self._batch_loader.load(original_row)
            if original.status != BatchStatus.FAILED:
                raise AuditIntegrityError(f"retry_batch: can only retry failed batches, batch {batch_id} has status {original.status!r}")

            next_attempt = original.attempt + 1

            # 2. Idempotency: check if a retry batch already exists for this attempt
            existing_row = conn.execute(
                select(batches_table)
                .where(batches_table.c.run_id == original.run_id)
                .where(batches_table.c.retry_of_batch_id == original.batch_id)
            ).fetchone()
            if existing_row is not None:
                return self._batch_loader.load(existing_row)

            # 3. Create new batch
            new_batch_id = generate_id()
            timestamp = now()
            result = conn.execute(
                batches_table.insert().values(
                    batch_id=new_batch_id,
                    run_id=original.run_id,
                    aggregation_node_id=original.aggregation_node_id,
                    attempt=next_attempt,
                    retry_of_batch_id=original.batch_id,
                    status=BatchStatus.DRAFT,
                    created_at=timestamp,
                )
            )
            if result.rowcount == 0:
                raise AuditIntegrityError(f"retry_batch: INSERT for new batch affected zero rows (batch_id={new_batch_id})")

            # 4. Copy members from original batch
            member_rows = conn.execute(
                select(batch_members_table).where(batch_members_table.c.batch_id == batch_id).order_by(batch_members_table.c.ordinal)
            ).fetchall()
            for member_row in member_rows:
                member_result = conn.execute(
                    batch_members_table.insert().values(
                        batch_id=new_batch_id,
                        run_id=original.run_id,
                        token_id=member_row.token_id,
                        ordinal=member_row.ordinal,
                    )
                )
                if member_result.rowcount == 0:
                    raise AuditIntegrityError(
                        f"retry_batch: member INSERT affected zero rows (batch={new_batch_id}, token={member_row.token_id})"
                    )

            # 5. Read back the new batch for return
            new_row = conn.execute(select(batches_table).where(batches_table.c.batch_id == new_batch_id)).fetchone()

        if new_row is None:
            raise AuditIntegrityError(f"retry_batch: new batch {new_batch_id} not found after INSERT")
        return self._batch_loader.load(new_row)

    # === Artifact Registration ===

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
    ) -> Artifact:
        """Register an artifact produced by a sink.

        Args:
            run_id: Run that produced this artifact
            state_id: Node state that produced this artifact
            sink_node_id: Sink node that wrote the artifact
            artifact_type: Type of artifact (csv, json, etc.)
            path: File path or URI
            content_hash: Hash of artifact content
            size_bytes: Size of artifact in bytes
            artifact_id: Optional artifact ID
            idempotency_key: Optional key for retry deduplication

        Returns:
            Artifact model
        """
        artifact_id = artifact_id or generate_id()
        timestamp = now()

        artifact = Artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            produced_by_state_id=state_id,
            sink_node_id=sink_node_id,
            artifact_type=artifact_type,
            path_or_uri=path,
            content_hash=content_hash,
            size_bytes=size_bytes,
            created_at=timestamp,
            idempotency_key=idempotency_key,
        )

        self._ops.execute_insert(
            artifacts_table.insert().values(
                artifact_id=artifact.artifact_id,
                run_id=artifact.run_id,
                produced_by_state_id=artifact.produced_by_state_id,
                sink_node_id=artifact.sink_node_id,
                artifact_type=artifact.artifact_type,
                path_or_uri=artifact.path_or_uri,
                content_hash=artifact.content_hash,
                size_bytes=artifact.size_bytes,
                idempotency_key=artifact.idempotency_key,
                created_at=artifact.created_at,
            )
        )

        return artifact

    def get_artifacts(
        self,
        run_id: str,
        *,
        sink_node_id: str | None = None,
    ) -> list[Artifact]:
        """Get artifacts for a run.

        Args:
            run_id: Run ID
            sink_node_id: Optional filter by sink

        Returns:
            List of Artifact models
        """
        query = select(artifacts_table).where(artifacts_table.c.run_id == run_id)

        if sink_node_id is not None:
            query = query.where(artifacts_table.c.sink_node_id == sink_node_id)

        # Order for deterministic export signatures
        query = query.order_by(artifacts_table.c.created_at, artifacts_table.c.artifact_id)
        rows = self._ops.execute_fetchall(query)
        return [self._artifact_loader.load(row) for row in rows]
