"""QueryRepository: read-only queries for audit trail entities.

Provides the external read-only API used by MCP server, exporter, CLI,
and TUI. Does NOT need LandscapeDB — only read-only database ops for queries.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from sqlalchemy import select

from elspeth.contracts import (
    Call,
    NodeState,
    RoutingEvent,
    Row,
    RowLineage,
    Token,
    TokenOutcome,
    TokenParent,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.payload_store import IntegrityError as PayloadIntegrityError
from elspeth.contracts.payload_store import PayloadNotFoundError, PayloadStore
from elspeth.core.landscape._database_ops import ReadOnlyDatabaseOps
from elspeth.core.landscape.model_loaders import (
    CallLoader,
    NodeStateLoader,
    RoutingEventLoader,
    RowLoader,
    TokenLoader,
    TokenOutcomeLoader,
    TokenParentLoader,
)
from elspeth.core.landscape.row_data import RowDataResult, RowDataState
from elspeth.core.landscape.schema import (
    calls_table,
    node_states_table,
    routing_events_table,
    rows_table,
    token_outcomes_table,
    token_parents_table,
    tokens_table,
)

logger = structlog.get_logger(__name__)


class QueryRepository:
    """Read-only query repository for audit trail entities."""

    _QUERY_CHUNK_SIZE = 500

    def __init__(
        self,
        ops: ReadOnlyDatabaseOps,
        *,
        row_loader: RowLoader,
        token_loader: TokenLoader,
        token_parent_loader: TokenParentLoader,
        node_state_loader: NodeStateLoader,
        routing_event_loader: RoutingEventLoader,
        call_loader: CallLoader,
        token_outcome_loader: TokenOutcomeLoader,
        payload_store: PayloadStore | None = None,
    ) -> None:
        self._ops = ops
        self._row_loader = row_loader
        self._token_loader = token_loader
        self._token_parent_loader = token_parent_loader
        self._node_state_loader = node_state_loader
        self._routing_event_loader = routing_event_loader
        self._call_loader = call_loader
        self._token_outcome_loader = token_outcome_loader
        self._payload_store = payload_store

    def get_rows(self, run_id: str) -> list[Row]:
        """Get all rows for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Row models, ordered by row_index
        """
        query = select(rows_table).where(rows_table.c.run_id == run_id).order_by(rows_table.c.row_index)
        db_rows = self._ops.execute_fetchall(query)
        return [self._row_loader.load(r) for r in db_rows]

    def get_tokens(self, row_id: str) -> list[Token]:
        """Get all tokens for a row.

        Args:
            row_id: Row ID

        Returns:
            List of Token models, ordered by created_at then token_id
            for deterministic export signatures.
        """
        query = select(tokens_table).where(tokens_table.c.row_id == row_id).order_by(tokens_table.c.created_at, tokens_table.c.token_id)
        db_rows = self._ops.execute_fetchall(query)
        return [self._token_loader.load(r) for r in db_rows]

    def get_node_states_for_token(self, token_id: str) -> list[NodeState]:
        """Get all node states for a token.

        Args:
            token_id: Token ID

        Returns:
            List of NodeState models (discriminated union), ordered by (step_index, attempt)
        """
        # Order by (step_index, attempt) for deterministic ordering across retries
        query = (
            select(node_states_table)
            .where(node_states_table.c.token_id == token_id)
            .order_by(node_states_table.c.step_index, node_states_table.c.attempt)
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._node_state_loader.load(r) for r in db_rows]

    def get_row(self, row_id: str) -> Row | None:
        """Get a row by ID.

        Args:
            row_id: Row ID

        Returns:
            Row model or None if not found
        """
        query = select(rows_table).where(rows_table.c.row_id == row_id)
        r = self._ops.execute_fetchone(query)
        if r is None:
            return None
        return self._row_loader.load(r)

    def _retrieve_and_parse_payload(self, row_id: str, source_data_ref: str) -> dict[str, object]:
        """Retrieve and parse a payload, returning the validated dict.

        Shared by get_row_data() and explain_row() to eliminate duplication
        of retrieval + JSON parse + dict validation + error wrapping.

        Args:
            row_id: Row ID (for error context)
            source_data_ref: Payload store reference key

        Returns:
            Parsed dict from the payload store

        Raises:
            PayloadNotFoundError: Payload was purged by retention policy (caller decides handling)
            AuditIntegrityError: Payload is corrupt, fails integrity check,
                or cannot be retrieved due to infrastructure failure
        """
        if self._payload_store is None:
            raise ValueError("Cannot retrieve payload: payload store not configured")

        # PayloadIntegrityError = hash mismatch (corruption/tampering),
        # OSError = storage backend failure. Both translate to
        # AuditIntegrityError with context, matching
        # execution_repository.get_call_response_data().
        try:
            payload_bytes = self._payload_store.retrieve(source_data_ref)
        except PayloadIntegrityError as e:
            raise AuditIntegrityError(f"Payload integrity check failed for row {row_id} (ref={source_data_ref}): {e}") from e
        except OSError as e:
            raise AuditIntegrityError(f"Payload retrieval failed for row {row_id} (ref={source_data_ref}): {type(e).__name__}: {e}") from e

        try:
            decoded_data = json.loads(payload_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise AuditIntegrityError(f"Corrupt payload for row {row_id} (ref={source_data_ref}): {e}") from e

        match decoded_data:
            case dict() as data:
                return data
            case _:
                actual_type = type(decoded_data).__name__
                raise AuditIntegrityError(
                    f"Corrupt payload for row {row_id} (ref={source_data_ref}): expected JSON object, got {actual_type}"
                )

    def get_row_data(self, row_id: str) -> RowDataResult:
        """Get the payload data for a row with explicit state.

        Returns a RowDataResult with explicit state indicating why data
        may be unavailable. This replaces the previous ambiguous None return.

        Args:
            row_id: Row ID

        Returns:
            RowDataResult with state and data (if available)
        """
        row = self.get_row(row_id)
        if row is None:
            return RowDataResult(state=RowDataState.ROW_NOT_FOUND, data=None)

        if row.source_data_ref is None:
            return RowDataResult(state=RowDataState.NEVER_STORED, data=None)

        if self._payload_store is None:
            return RowDataResult(state=RowDataState.STORE_NOT_CONFIGURED, data=None)

        try:
            data = self._retrieve_and_parse_payload(row_id, row.source_data_ref)
            # Detect repr-fallback sentinel: quarantined data that couldn't be
            # canonically serialized is stored as {"_repr": repr(data)}.
            # Callers must know this is a lossy snapshot, not the real payload.
            if set(data.keys()) == {"_repr"}:
                return RowDataResult(state=RowDataState.REPR_FALLBACK, data=data)
            return RowDataResult(state=RowDataState.AVAILABLE, data=data)
        except PayloadNotFoundError as exc:
            logger.debug("Payload purged, returning PURGED state", content_hash=exc.content_hash)
            return RowDataResult(state=RowDataState.PURGED, data=None)

    def get_token(self, token_id: str) -> Token | None:
        """Get a token by ID.

        Args:
            token_id: Token ID

        Returns:
            Token model or None if not found
        """
        query = select(tokens_table).where(tokens_table.c.token_id == token_id)
        r = self._ops.execute_fetchone(query)
        if r is None:
            return None
        return self._token_loader.load(r)

    def get_token_parents(self, token_id: str) -> list[TokenParent]:
        """Get parent relationships for a token (backward lineage).

        Args:
            token_id: Token ID (the child)

        Returns:
            List of TokenParent models (ordered by ordinal)
        """
        query = select(token_parents_table).where(token_parents_table.c.token_id == token_id).order_by(token_parents_table.c.ordinal)
        db_rows = self._ops.execute_fetchall(query)
        return [self._token_parent_loader.load(r) for r in db_rows]

    def get_token_children(self, parent_token_id: str) -> list[TokenParent]:
        """Get child relationships for a token (forward lineage).

        Enables forward lineage queries: "what tokens were created from this parent?"
        This closes the audit trail gap where COALESCED tokens store join_group_id
        but forward traversal required reading node state output_data.

        Args:
            parent_token_id: Token ID (the parent)

        Returns:
            List of TokenParent models where this token is the parent.
            Ordered by child token_id for deterministic results.
            Note: ordinal represents the parent's position in the child's merge,
            not a child ordering (which doesn't exist semantically).
        """
        query = (
            select(token_parents_table)
            .where(token_parents_table.c.parent_token_id == parent_token_id)
            .order_by(token_parents_table.c.token_id)
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._token_parent_loader.load(r) for r in db_rows]

    def get_routing_events(self, state_id: str) -> list[RoutingEvent]:
        """Get routing events for a node state.

        Args:
            state_id: State ID

        Returns:
            List of RoutingEvent models, ordered by ordinal then event_id
            for deterministic export signatures.
        """
        query = (
            select(routing_events_table)
            .where(routing_events_table.c.state_id == state_id)
            .order_by(routing_events_table.c.ordinal, routing_events_table.c.event_id)
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._routing_event_loader.load(r) for r in db_rows]

    def get_calls(self, state_id: str) -> list[Call]:
        """Get external calls for a node state.

        Args:
            state_id: State ID

        Returns:
            List of Call models, ordered by call_index
        """
        query = select(calls_table).where(calls_table.c.state_id == state_id).order_by(calls_table.c.call_index)
        db_rows = self._ops.execute_fetchall(query)
        return [self._call_loader.load(r) for r in db_rows]

    # === Batch Query Methods for State Sets (ech8: N+1 query fix for lineage) ===
    #
    # These methods fetch entities for a set of state IDs in a single query,
    # replacing the N+1 pattern where per-state queries nested inside loops.

    def get_routing_events_for_states(self, state_ids: list[str]) -> list[RoutingEvent]:
        """Get routing events for multiple states in one query.

        Chunks state_ids to stay within SQLite's SQLITE_MAX_VARIABLE_NUMBER
        limit (default 999).

        Note: Each chunk is a separate query. For completed runs this is safe.
        For in-progress runs, concurrent writes between chunks could produce
        inconsistent results. Query only completed runs for reliable results.

        Args:
            state_ids: List of state IDs to query

        Returns:
            List of RoutingEvent models, ordered by execution order
            (step_index, attempt, ordinal, event_id)
        """
        if not state_ids:
            return []

        all_db_rows = []
        for offset in range(0, len(state_ids), self._QUERY_CHUNK_SIZE):
            chunk = state_ids[offset : offset + self._QUERY_CHUNK_SIZE]
            query = (
                select(
                    routing_events_table,
                    node_states_table.c.step_index,
                    node_states_table.c.attempt,
                )
                .join(node_states_table, routing_events_table.c.state_id == node_states_table.c.state_id)
                .where(routing_events_table.c.state_id.in_(chunk))
            )
            all_db_rows.extend(self._ops.execute_fetchall(query))

        # Sort with total ordering: state_id breaks ties when multiple tokens
        # share the same step_index/attempt (e.g., forked paths at the same step).
        all_db_rows.sort(key=lambda r: (r.step_index, r.attempt, r.state_id, r.ordinal, r.event_id))
        return [self._routing_event_loader.load(r) for r in all_db_rows]

    def get_calls_for_states(self, state_ids: list[str]) -> list[Call]:
        """Get external calls for multiple states in one query.

        Chunks state_ids to stay within SQLite's SQLITE_MAX_VARIABLE_NUMBER
        limit (default 999).

        Note: Each chunk is a separate query. For completed runs this is safe.
        For in-progress runs, concurrent writes between chunks could produce
        inconsistent results. Query only completed runs for reliable results.

        Args:
            state_ids: List of state IDs to query

        Returns:
            List of Call models, ordered by execution order
            (step_index, attempt, call_index)
        """
        if not state_ids:
            return []

        all_db_rows = []
        for offset in range(0, len(state_ids), self._QUERY_CHUNK_SIZE):
            chunk = state_ids[offset : offset + self._QUERY_CHUNK_SIZE]
            query = (
                select(
                    calls_table,
                    node_states_table.c.step_index,
                    node_states_table.c.attempt,
                )
                .join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id)
                .where(calls_table.c.state_id.in_(chunk))
            )
            all_db_rows.extend(self._ops.execute_fetchall(query))

        # Sort with total ordering: state_id breaks ties when multiple tokens
        # share the same step_index/attempt (e.g., forked paths at the same step).
        all_db_rows.sort(key=lambda r: (r.step_index, r.attempt, r.state_id, r.call_index))
        return [self._call_loader.load(r) for r in all_db_rows]

    # === Batch Query Methods (Bug 76r: N+1 query fix for exporter) ===
    #
    # These methods fetch all entities for a run in a single query,
    # replacing the N+1 pattern where per-entity queries nested inside loops.

    def get_all_tokens_for_run(self, run_id: str) -> list[Token]:
        """Get all tokens for a run (batch query).

        Args:
            run_id: Run ID

        Returns:
            List of Token models, ordered by row_id then created_at
        """
        # JOIN through rows table to filter by run_id
        query = (
            select(tokens_table)
            .join(rows_table, tokens_table.c.row_id == rows_table.c.row_id)
            .where(rows_table.c.run_id == run_id)
            .order_by(tokens_table.c.row_id, tokens_table.c.created_at, tokens_table.c.token_id)
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._token_loader.load(r) for r in db_rows]

    def get_all_node_states_for_run(self, run_id: str) -> list[NodeState]:
        """Get all node states for a run (batch query).

        Args:
            run_id: Run ID

        Returns:
            List of NodeState models, ordered by token_id then step_index then attempt
        """
        # node_states has run_id denormalized (per CLAUDE.md composite FK pattern)
        query = (
            select(node_states_table)
            .where(node_states_table.c.run_id == run_id)
            .order_by(
                node_states_table.c.token_id,
                node_states_table.c.step_index,
                node_states_table.c.attempt,
            )
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._node_state_loader.load(r) for r in db_rows]

    def get_all_routing_events_for_run(self, run_id: str) -> list[RoutingEvent]:
        """Get all routing events for a run (batch query).

        Args:
            run_id: Run ID

        Returns:
            List of RoutingEvent models, ordered by execution order
            (step_index, attempt, ordinal, event_id)
        """
        # JOIN through node_states to filter by run_id
        query = (
            select(routing_events_table)
            .join(node_states_table, routing_events_table.c.state_id == node_states_table.c.state_id)
            .where(node_states_table.c.run_id == run_id)
            .order_by(
                node_states_table.c.step_index,
                node_states_table.c.attempt,
                node_states_table.c.state_id,
                routing_events_table.c.ordinal,
                routing_events_table.c.event_id,
            )
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._routing_event_loader.load(r) for r in db_rows]

    def get_all_calls_for_run(self, run_id: str) -> list[Call]:
        """Get all calls (state-parented) for a run (batch query).

        Note: Operation-parented calls are fetched separately via get_operation_calls.
        This method only returns calls parented by node_states.

        Args:
            run_id: Run ID

        Returns:
            List of Call models, ordered by execution order
            (step_index, attempt, call_index)
        """
        # JOIN through node_states to filter by run_id
        # Only state-parented calls (state_id IS NOT NULL)
        query = (
            select(calls_table)
            .join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id)
            .where(node_states_table.c.run_id == run_id)
            .order_by(
                node_states_table.c.step_index,
                node_states_table.c.attempt,
                node_states_table.c.state_id,
                calls_table.c.call_index,
            )
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._call_loader.load(r) for r in db_rows]

    def get_all_token_parents_for_run(self, run_id: str) -> list[TokenParent]:
        """Get all token parent relationships for a run (batch query).

        Args:
            run_id: Run ID

        Returns:
            List of TokenParent models, ordered by token_id then ordinal
        """
        # JOIN through tokens and rows to filter by run_id
        query = (
            select(token_parents_table)
            .join(tokens_table, token_parents_table.c.token_id == tokens_table.c.token_id)
            .join(rows_table, tokens_table.c.row_id == rows_table.c.row_id)
            .where(rows_table.c.run_id == run_id)
            .order_by(token_parents_table.c.token_id, token_parents_table.c.ordinal)
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._token_parent_loader.load(r) for r in db_rows]

    def get_all_token_outcomes_for_run(self, run_id: str) -> list[TokenOutcome]:
        """Get all token outcomes for a run (batch query).

        Args:
            run_id: Run ID

        Returns:
            List of TokenOutcome models, ordered by token_id then recorded_at
        """
        query = (
            select(token_outcomes_table)
            .where(token_outcomes_table.c.run_id == run_id)
            .order_by(token_outcomes_table.c.token_id, token_outcomes_table.c.recorded_at)
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._token_outcome_loader.load(r) for r in db_rows]

    # === Explain Methods (Graceful Degradation) ===

    def explain_row(self, run_id: str, row_id: str) -> RowLineage | None:
        """Get lineage for a row, gracefully handling purged payloads.

        This method returns row lineage information even when the actual
        payload data has been purged by retention policies. The hash is
        always preserved, ensuring audit integrity can be verified.

        Args:
            run_id: Run this row belongs to
            row_id: Row ID to explain

        Returns:
            RowLineage with hash and optionally source data, or None if row not found

        Raises:
            AuditIntegrityError: If row exists but belongs to a different run,
                payload data is corrupt, fails integrity check,
                or cannot be retrieved due to infrastructure failure
        """
        row = self.get_row(row_id)
        if row is None:
            return None

        # Validate row belongs to the specified run — cross-run mismatch is a
        # caller bug or data corruption, not a normal "not found" case
        if row.run_id != run_id:
            raise AuditIntegrityError(f"Row {row_id} belongs to run {row.run_id}, not {run_id}")

        # Try to load payload
        source_data: dict[str, Any] | None = None
        payload_available = False

        if row.source_data_ref is not None and self._payload_store is not None:
            try:
                source_data = self._retrieve_and_parse_payload(row_id, row.source_data_ref)
                payload_available = True
            except PayloadNotFoundError as exc:
                logger.debug("Payload purged, continuing without source data", content_hash=exc.content_hash)

        return RowLineage(
            row_id=row.row_id,
            run_id=row.run_id,
            source_node_id=row.source_node_id,
            row_index=row.row_index,
            source_data_hash=row.source_data_hash,
            created_at=row.created_at,
            source_data=source_data,
            payload_available=payload_available,
        )
