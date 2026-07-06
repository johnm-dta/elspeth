"""Node-state and routing-event persistence (split from ``ExecutionRepository``).

Owns the ``node_states`` and ``routing_events`` audit aggregates: begin /
complete lifecycle writes (single and batched), resume-derivation reads,
and routing-event recording with post-insert reason materialization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from sqlalchemy import bindparam, func, select
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts import (
    CoalesceFailureReason,
    FrameworkBugError,
    NodeState,
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStatePending,
    NodeStateStatus,
    RoutingEvent,
    RoutingMode,
    RoutingReason,
    RoutingSpec,
)
from elspeth.contracts.errors import AuditIntegrityError, ExecutionError, TransformErrorReason
from elspeth.contracts.hashing import repr_hash
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.ids import generate_id
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import now
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapePostCommitError, LandscapeRecordError
from elspeth.core.landscape.model_loaders import NodeStateLoader, RoutingEventLoader
from elspeth.core.landscape.schema import edges_table, node_states_table, routing_events_table, tokens_table

if TYPE_CHECKING:
    from elspeth.contracts.errors import TransformSuccessReason
    from elspeth.contracts.node_state_context import NodeStateContext
    from elspeth.contracts.payload_store import PayloadStore

_TERMINAL_NODE_STATE_STATUSES = frozenset({NodeStateStatus.COMPLETED, NodeStateStatus.FAILED})
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


class NodeStateRepository:
    """Node-state lifecycle and routing-event recording for one audit boundary."""

    def __init__(
        self,
        db: LandscapeDB,
        ops: DatabaseOps,
        *,
        node_state_loader: NodeStateLoader,
        routing_event_loader: RoutingEventLoader,
        payload_store: PayloadStore | None = None,
    ) -> None:
        self._db = db
        self._ops = ops
        self._node_state_loader = node_state_loader
        self._routing_event_loader = routing_event_loader
        self._payload_store = payload_store

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

        The per-status ``@overload`` narrowing lives on the
        ``ExecutionRepository`` facade; this implementation accepts the broad
        status and returns the terminal union.

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
        run_id = self._routing_event_run_id(
            state_id=state_id,
            edge_id=edge_id,
            owner="record_routing_event",
        )
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
                run_id=run_id,
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
                route_run_ids = [
                    self._routing_event_run_id(
                        state_id=state_id,
                        edge_id=route.edge_id,
                        owner="record_routing_events",
                        conn=conn,
                    )
                    for route in routes
                ]
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
                            run_id=route_run_ids[ordinal],
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

    def _routing_event_run_id(
        self,
        *,
        state_id: str,
        edge_id: str,
        owner: str,
        conn: Connection | None = None,
    ) -> str:
        """Return the shared run_id for a routing state/edge pair or reject."""
        query = (
            select(
                node_states_table.c.run_id.label("state_run_id"),
                edges_table.c.run_id.label("edge_run_id"),
            )
            .select_from(node_states_table.join(edges_table, edges_table.c.edge_id == edge_id))
            .where(node_states_table.c.state_id == state_id)
        )
        row = conn.execute(query).fetchone() if conn is not None else self._ops.execute_fetchone(query)
        if row is None:
            raise LandscapeRecordError(f"{owner} requires existing state_id={state_id!r} and edge_id={edge_id!r} in the same run")
        if row.state_run_id != row.edge_run_id:
            raise LandscapeRecordError(f"{owner} requires state_id={state_id!r} and edge_id={edge_id!r} to belong to the same run")
        return str(row.state_run_id)

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
