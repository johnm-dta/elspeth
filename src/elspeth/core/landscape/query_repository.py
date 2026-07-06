"""QueryRepository: read-only queries for audit trail entities.

Provides the external read-only API used by MCP server, exporter, CLI,
and TUI. Does NOT need LandscapeDB — only read-only database ops for queries.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

import structlog
from sqlalchemy import and_, func, select

from elspeth.contracts import (
    Call,
    NodeState,
    NodeStateStatus,
    NodeType,
    RoutingEvent,
    Row,
    RowLineage,
    SchedulerEvent,
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
    SchedulerEventLoader,
    TokenLoader,
    TokenOutcomeLoader,
    TokenParentLoader,
)
from elspeth.core.landscape.row_data import RowDataResult, RowDataState
from elspeth.core.landscape.schema import (
    calls_table,
    node_states_table,
    nodes_table,
    routing_events_table,
    rows_table,
    scheduler_events_table,
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
        scheduler_event_loader: SchedulerEventLoader | None = None,
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
        self._scheduler_event_loader = scheduler_event_loader or SchedulerEventLoader()
        self._payload_store = payload_store

    def get_rows(self, run_id: str) -> list[Row]:
        """Get all rows for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Row models, ordered by row_index
        """
        order_column = rows_table.c.ingest_sequence if "ingest_sequence" in rows_table.c else rows_table.c.row_index
        query = select(rows_table).where(rows_table.c.run_id == run_id).order_by(order_column)
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

    def _load_payload_if_present(self, row_id: str, source_data_ref: str) -> tuple[dict[str, Any] | None, bool]:
        """Load a payload, recording absence when it was purged by retention.

        Returns ``(data, True)`` when the payload is present, or ``(None, False)``
        when the payload store reports it was purged (``PayloadNotFoundError`` —
        a documented retention outcome, not an error). The absence is recorded
        explicitly in the returned tuple rather than swallowed; corruption and
        infrastructure failures continue to propagate as ``AuditIntegrityError``
        from ``_retrieve_and_parse_payload``.

        Args:
            row_id: Row ID (for error context)
            source_data_ref: Payload store reference key

        Returns:
            ``(parsed_payload, payload_available)`` — ``payload_available`` is
            ``False`` exactly when the payload was purged.

        Raises:
            AuditIntegrityError: Payload is corrupt, fails integrity check,
                or cannot be retrieved due to infrastructure failure
        """
        try:
            return self._retrieve_and_parse_payload(row_id, source_data_ref), True
        except PayloadNotFoundError as exc:
            logger.debug("Payload purged, continuing without source data", content_hash=exc.content_hash)
            return None, False

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

    def get_scheduler_events(
        self,
        *,
        run_id: str,
        token_id: str | None = None,
        work_item_id: str | None = None,
    ) -> list[SchedulerEvent]:
        """Get scheduler transition events for a run, optionally narrowed to a token or work item."""
        predicates = [scheduler_events_table.c.run_id == run_id]
        if token_id is not None:
            predicates.append(scheduler_events_table.c.token_id == token_id)
        if work_item_id is not None:
            predicates.append(scheduler_events_table.c.work_item_id == work_item_id)
        query = (
            select(scheduler_events_table)
            .where(*predicates)
            .order_by(
                scheduler_events_table.c.recorded_at,
                scheduler_events_table.c.event_id,
            )
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._scheduler_event_loader.load(r) for r in db_rows]

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
                .join(
                    node_states_table,
                    and_(
                        routing_events_table.c.state_id == node_states_table.c.state_id,
                        routing_events_table.c.run_id == node_states_table.c.run_id,
                    ),
                )
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
            .join(
                node_states_table,
                and_(
                    routing_events_table.c.state_id == node_states_table.c.state_id,
                    routing_events_table.c.run_id == node_states_table.c.run_id,
                ),
            )
            .where(routing_events_table.c.run_id == run_id)
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

    # === Chunked Export Read APIs (elspeth-3ae79a4775: bounded-memory export) ===
    #
    # These methods let the exporter stream a run's row family in bounded
    # batches instead of preloading every child collection for the run.
    # Contract: grouping a set-scoped result by parent id yields exactly the
    # same per-parent sequences as grouping the corresponding full-run getter.
    # Each parent's children always land in a single IN-chunk (chunks partition
    # the parent ids) ordered by the same ORDER BY, so per-parent order is
    # exact. The flat cross-parent order follows input-chunk order — callers
    # must group by parent id, not rely on the flat sequence.
    #
    # Note: Each chunk/page is a separate query. For completed runs this is
    # safe. For in-progress runs, concurrent writes between queries could
    # produce inconsistent results. Query only completed runs.

    def iter_rows_for_run(self, run_id: str, *, batch_size: int = _QUERY_CHUNK_SIZE) -> Iterator[list[Row]]:
        """Iterate rows for a run in bounded batches (keyset pagination).

        Yields lists of Row models in the same global order as
        :meth:`get_rows` (``ingest_sequence`` ascending), loading at most
        ``batch_size`` rows per query. ``ingest_sequence`` is ``NOT NULL``
        and unique per run (``UniqueConstraint("run_id", "ingest_sequence")``),
        so it is an exact keyset cursor: no row can be skipped or duplicated
        between pages.

        Args:
            run_id: Run ID
            batch_size: Maximum rows per yielded batch (must be >= 1)

        Yields:
            Non-empty lists of Row models, ordered by ingest_sequence

        Raises:
            ValueError: If batch_size < 1
            AuditIntegrityError: If a row has NULL ingest_sequence — the
                schema declares the column NOT NULL, so this is Tier-1
                audit-database corruption
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        last_ingest_sequence: int | None = None
        while True:
            query = select(rows_table).where(rows_table.c.run_id == run_id)
            if last_ingest_sequence is not None:
                query = query.where(rows_table.c.ingest_sequence > last_ingest_sequence)
            query = query.order_by(rows_table.c.ingest_sequence).limit(batch_size)
            db_rows = self._ops.execute_fetchall(query)
            if not db_rows:
                return
            last_ingest_sequence = db_rows[-1].ingest_sequence
            if last_ingest_sequence is None:
                raise AuditIntegrityError(
                    f"Row '{db_rows[-1].row_id}' in run {run_id!r} has NULL ingest_sequence — the schema "
                    f"declares it NOT NULL, so this is a Tier-1 audit-database integrity violation."
                )
            yield [self._row_loader.load(r) for r in db_rows]
            if len(db_rows) < batch_size:
                return

    def get_tokens_for_rows(self, run_id: str, row_ids: Sequence[str]) -> list[Token]:
        """Get tokens for a set of rows (chunked batch query).

        Within each row, tokens are ordered by (created_at, token_id) — the
        same per-row ordering as :meth:`get_tokens` and
        :meth:`get_all_tokens_for_run`. Chunks row_ids to stay within
        SQLite's SQLITE_MAX_VARIABLE_NUMBER limit (default 999).

        Args:
            run_id: Run ID (guards against cross-run contamination)
            row_ids: Row IDs to fetch tokens for

        Returns:
            List of Token models; group by row_id for per-row sequences
        """
        if not row_ids:
            return []
        tokens: list[Token] = []
        for offset in range(0, len(row_ids), self._QUERY_CHUNK_SIZE):
            chunk = row_ids[offset : offset + self._QUERY_CHUNK_SIZE]
            query = (
                select(tokens_table)
                .where(tokens_table.c.run_id == run_id)
                .where(tokens_table.c.row_id.in_(chunk))
                .order_by(tokens_table.c.row_id, tokens_table.c.created_at, tokens_table.c.token_id)
            )
            tokens.extend(self._token_loader.load(r) for r in self._ops.execute_fetchall(query))
        return tokens

    def get_token_parents_for_tokens(self, token_ids: Sequence[str]) -> list[TokenParent]:
        """Get parent relationships for a set of tokens (chunked batch query).

        Within each token, parents are ordered by ordinal — the same
        per-token ordering as :meth:`get_token_parents` and
        :meth:`get_all_token_parents_for_run`. Chunks token_ids to stay
        within SQLite's SQLITE_MAX_VARIABLE_NUMBER limit (default 999).

        Args:
            token_ids: Child token IDs to fetch parent links for

        Returns:
            List of TokenParent models; group by token_id for per-token sequences
        """
        if not token_ids:
            return []
        parents: list[TokenParent] = []
        for offset in range(0, len(token_ids), self._QUERY_CHUNK_SIZE):
            chunk = token_ids[offset : offset + self._QUERY_CHUNK_SIZE]
            query = (
                select(token_parents_table)
                .where(token_parents_table.c.token_id.in_(chunk))
                .order_by(token_parents_table.c.token_id, token_parents_table.c.ordinal)
            )
            parents.extend(self._token_parent_loader.load(r) for r in self._ops.execute_fetchall(query))
        return parents

    def get_node_states_for_tokens(self, run_id: str, token_ids: Sequence[str]) -> list[NodeState]:
        """Get node states for a set of tokens (chunked batch query).

        Within each token, states are ordered by (step_index, attempt) — the
        same per-token ordering as :meth:`get_node_states_for_token` and
        :meth:`get_all_node_states_for_run`. Chunks token_ids to stay within
        SQLite's SQLITE_MAX_VARIABLE_NUMBER limit (default 999).

        Args:
            run_id: Run ID (guards against cross-run contamination)
            token_ids: Token IDs to fetch states for

        Returns:
            List of NodeState models; group by token_id for per-token sequences
        """
        if not token_ids:
            return []
        states: list[NodeState] = []
        for offset in range(0, len(token_ids), self._QUERY_CHUNK_SIZE):
            chunk = token_ids[offset : offset + self._QUERY_CHUNK_SIZE]
            query = (
                select(node_states_table)
                .where(node_states_table.c.run_id == run_id)
                .where(node_states_table.c.token_id.in_(chunk))
                .order_by(
                    node_states_table.c.token_id,
                    node_states_table.c.step_index,
                    node_states_table.c.attempt,
                )
            )
            states.extend(self._node_state_loader.load(r) for r in self._ops.execute_fetchall(query))
        return states

    def get_token_outcomes_for_tokens(self, run_id: str, token_ids: Sequence[str]) -> list[TokenOutcome]:
        """Get token outcomes for a set of tokens (chunked batch query).

        Within each token, outcomes are ordered by recorded_at — the same
        per-token ordering as :meth:`get_all_token_outcomes_for_run`. Chunks
        token_ids to stay within SQLite's SQLITE_MAX_VARIABLE_NUMBER limit
        (default 999).

        Args:
            run_id: Run ID (guards against cross-run contamination)
            token_ids: Token IDs to fetch outcomes for

        Returns:
            List of TokenOutcome models; group by token_id for per-token sequences
        """
        if not token_ids:
            return []
        outcomes: list[TokenOutcome] = []
        for offset in range(0, len(token_ids), self._QUERY_CHUNK_SIZE):
            chunk = token_ids[offset : offset + self._QUERY_CHUNK_SIZE]
            query = (
                select(token_outcomes_table)
                .where(token_outcomes_table.c.run_id == run_id)
                .where(token_outcomes_table.c.token_id.in_(chunk))
                .order_by(token_outcomes_table.c.token_id, token_outcomes_table.c.recorded_at)
            )
            outcomes.extend(self._token_outcome_loader.load(r) for r in self._ops.execute_fetchall(query))
        return outcomes

    def get_scheduler_events_for_tokens(self, run_id: str, token_ids: Sequence[str]) -> list[SchedulerEvent]:
        """Get scheduler transition events for a set of tokens (chunked batch query).

        Within each token, events are ordered by (recorded_at, event_id) —
        the same per-token ordering as :meth:`get_scheduler_events`. Chunks
        token_ids to stay within SQLite's SQLITE_MAX_VARIABLE_NUMBER limit
        (default 999).

        Args:
            run_id: Run ID (guards against cross-run contamination)
            token_ids: Token IDs to fetch scheduler events for

        Returns:
            List of SchedulerEvent models; group by token_id for per-token sequences
        """
        if not token_ids:
            return []
        events: list[SchedulerEvent] = []
        for offset in range(0, len(token_ids), self._QUERY_CHUNK_SIZE):
            chunk = token_ids[offset : offset + self._QUERY_CHUNK_SIZE]
            query = (
                select(scheduler_events_table)
                .where(scheduler_events_table.c.run_id == run_id)
                .where(scheduler_events_table.c.token_id.in_(chunk))
                .order_by(
                    scheduler_events_table.c.recorded_at,
                    scheduler_events_table.c.event_id,
                )
            )
            events.extend(self._scheduler_event_loader.load(r) for r in self._ops.execute_fetchall(query))
        return events

    def count_distinct_source_rows_with_terminal_outcome(self, run_id: str) -> int:
        """Count the distinct source rows that reached a terminal outcome.

        ``rows_processed`` semantics — the canonical definition (F2,
        elspeth-resume-fork-reemit) — is "one per *source row*", NOT one per
        terminal token.  The live processing loops increment ``rows_processed``
        exactly once per source row pulled from the source iterator
        (``resume.py`` ``run_resume_processing_loop`` and the main
        ``_run_main_processing_loop``), and structural fan-out
        (fork / expand) or fan-in (aggregation / coalesce) never moves that
        counter.  Reconstructing it from the audit trail therefore CANNOT be a
        per-token tally: a 1-source-row fork emits two leaf tokens, a
        3-source-row aggregation emits one result token, a 1-source-row expand
        emits N children — yet each contributes exactly its *source rows* to
        ``rows_processed`` (1, 3, 1 respectively).

        ``row_id`` is the stable source-row identity (CLAUDE.md DAG model):
        fork and expand children inherit their parent's ``row_id``
        (``tokens.expand_token`` / ``fork_token`` pass ``row_id=parent.row_id``),
        and aggregation's ``BATCH_CONSUMED`` tokens retain their own source
        ``row_id`` while the synthetic result token reuses one of them.  So the
        faithful reconstruction is the count of DISTINCT ``row_id`` among tokens
        that reached a *terminal* outcome (``completed = 1``) — this counts each
        source row once regardless of how many tokens it spawned, and matches an
        uninterrupted run field-for-field (verified across fork / aggregation /
        expand archetypes).

        ``completed = 1`` is the terminal boundary: it includes structural
        TRANSIENT parents (``FORK_PARENT`` / ``EXPAND_PARENT``) and
        ``BATCH_CONSUMED`` tokens (all terminal), and excludes non-terminal
        ``BUFFERED`` rows (``completed = 0``) — a row whose only audit record is
        ``BUFFERED`` has not yet been processed to a terminal state, so it must
        not inflate ``rows_processed``.

        Args:
            run_id: Run ID

        Returns:
            Distinct source-row count among terminal token outcomes.

        Raises:
            AuditIntegrityError: If the count query returns no row — a
                ``COUNT`` aggregate always returns exactly one row, so a NULL
                result indicates Tier-1 audit-database corruption.
        """
        query = (
            select(func.count(func.distinct(tokens_table.c.row_id)))
            .select_from(
                token_outcomes_table.join(
                    tokens_table,
                    (token_outcomes_table.c.token_id == tokens_table.c.token_id) & (token_outcomes_table.c.run_id == tokens_table.c.run_id),
                )
            )
            .where(token_outcomes_table.c.run_id == run_id)
            .where(token_outcomes_table.c.completed == 1)
        )
        r = self._ops.execute_fetchone(query)
        if r is None:
            raise AuditIntegrityError(
                f"count_distinct_source_rows_with_terminal_outcome returned no row for run {run_id!r} — "
                f"a COUNT aggregate must always return exactly one row; a NULL result is a Tier-1 "
                f"audit-database integrity violation."
            )
        return int(r[0])

    def count_failed_coalesce_barrier_rows(self, run_id: str) -> int:
        """Count distinct (coalesce node, source row) join barriers that FAILED.

        ``rows_coalesce_failed`` semantics (elspeth-7294de558e): the counter is
        per failed *barrier* — one pending key ``(coalesce_name, row_id)`` that
        failed to merge — NOT per branch token.  The durable evidence is the
        family of FAILED ``node_states`` that
        ``CoalesceExecutor._fail_pending`` writes at the coalesce node: one
        FAILED state per *arrived branch token*, all sharing the same
        ``(node_id, row_id)``.  A naive count of FAILED states (or of the
        per-branch ``(FAILURE, UNROUTED)`` ``token_outcomes``, which carry no
        node attribution at all) over-reports a 2-branch barrier failure as 2;
        the faithful reconstruction is the count of DISTINCT
        ``(node_id, row_id)`` pairs.

        ANCHOR CHOICE (pinned by
        ``tests/unit/core/landscape/test_query_methods.py::TestCountFailedCoalesceBarrierRows``):
        the query anchors on ``node_states.status = 'failed'`` joined to
        ``nodes.node_type = 'coalesce'`` — both indexed, structural columns —
        rather than on the ``failure_reason`` strings inside ``error_json``
        (stringly, unindexed, and ambiguous: ``all_branches_lost`` is written
        by two different resolution paths).  ``row_id`` comes from the
        ``tokens`` join (branch tokens of one barrier inherit the same source
        ``row_id`` — the pending key IS ``(coalesce_name, row_id)``).

        ONE exclusion, applied Python-side on the parsed error payload: a
        ``late_arrival_after_merge`` state is a straggler token rejected AFTER
        the barrier already resolved — it is not itself a barrier failure.
        After a *failed* merge the pair is already counted via the barrier's
        own ``_fail_pending`` states (the DISTINCT collapse absorbs the
        straggler); after a *successful* merge the pair must not be counted at
        all, which only the reason exclusion guarantees.

        DELIBERATE breadth: arrival-time barrier failures (branch-lost
        cascades via ``_evaluate_after_loss``, immediate merge failures such
        as ``select_branch_not_arrived``) ARE counted here even though the
        live accumulator only increments ``rows_coalesce_failed`` for barriers
        resolved by the timeout/flush sweeps (``outcomes.py``) — those
        arrival-time failures are real failed barriers and the durable record
        is the broader truth.  Conversely zero-arrival timeout failures
        (``best_effort_timeout_no_arrivals`` or ``first_timeout_no_arrivals``)
        consume no tokens and leave no node_states, so they are invisible here
        by construction.  Reconciling the live accumulator with this durable
        breadth is tracked:
        elspeth-ff6d48c180.

        Cumulativity: resume re-drives record under the SAME ``run_id``
        (resume provenance lives in ``resume_checkpoint_id``), so a single
        run-scoped query covers run-1 failures AND resumed-run failures, and
        the DISTINCT collapse dedupes a barrier that recorded states in both.

        Args:
            run_id: Run ID

        Returns:
            Distinct failed-barrier count for the run (run-1 + all resumes).

        Raises:
            AuditIntegrityError: If a FAILED coalesce node_state carries no
                parseable ``error_json`` — the write side requires an error
                payload for FAILED states, so its absence is Tier-1
                audit-database corruption.
        """
        query = (
            select(
                node_states_table.c.node_id,
                tokens_table.c.row_id,
                node_states_table.c.error_json,
            )
            .select_from(
                node_states_table.join(
                    nodes_table,
                    (node_states_table.c.node_id == nodes_table.c.node_id) & (node_states_table.c.run_id == nodes_table.c.run_id),
                ).join(
                    tokens_table,
                    (node_states_table.c.token_id == tokens_table.c.token_id) & (node_states_table.c.run_id == tokens_table.c.run_id),
                )
            )
            .where(node_states_table.c.run_id == run_id)
            .where(node_states_table.c.status == NodeStateStatus.FAILED.value)
            .where(nodes_table.c.node_type == NodeType.COALESCE.value)
        )
        failed_barriers: set[tuple[str, str]] = set()
        for db_row in self._ops.execute_fetchall(query):
            if db_row.error_json is None:
                raise AuditIntegrityError(
                    f"FAILED coalesce node_state for node {db_row.node_id!r} / row {db_row.row_id!r} in run "
                    f"{run_id!r} has no error_json — the write side requires an error payload for FAILED "
                    f"states, so this is a Tier-1 audit-database integrity violation."
                )
            try:
                error_payload = json.loads(db_row.error_json)
            except json.JSONDecodeError as exc:
                raise AuditIntegrityError(
                    f"FAILED coalesce node_state for node {db_row.node_id!r} / row {db_row.row_id!r} in run "
                    f"{run_id!r} has unparseable error_json — Tier-1 audit-database integrity violation: {exc}"
                ) from exc
            # error_json for a FAILED coalesce node_state is polymorphic across
            # two legitimate writers: coalesce_executor records a
            # CoalesceFailureReason (a dict WITH a required failure_reason) for
            # accept/merge failure outcomes, and the merge-cleanup handler
            # records an ExecutionError (a dict WITHOUT failure_reason, keys
            # {exception,type,phase}) for merge-time exceptions. Both are valid
            # FAILED barrier rows. A non-dict parsed payload is producible by
            # NEITHER writer (both .to_dict() to objects), so it is Tier-1 audit
            # corruption — crash with provenance, consistent with the null /
            # unparseable guards above.
            if not isinstance(error_payload, dict):
                raise AuditIntegrityError(
                    f"FAILED coalesce node_state for node {db_row.node_id!r} / row {db_row.row_id!r} in run "
                    f"{run_id!r} has a non-object error_json payload (got {type(error_payload).__name__}) — the "
                    f"write side serializes a CoalesceFailureReason or ExecutionError object, so this is a "
                    f"Tier-1 audit-database integrity violation."
                )
            # Exclude only the benign late-arrival-after-merge case (a
            # CoalesceFailureReason discriminator). Every other FAILED payload —
            # including ExecutionError shapes with no failure_reason — is a real
            # failed barrier. .get() reads the OPTIONAL discriminator across the
            # two valid payload shapes without crashing on the keyless one.
            if error_payload.get("failure_reason") == "late_arrival_after_merge":
                continue
            failed_barriers.add((db_row.node_id, db_row.row_id))
        return len(failed_barriers)

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

        # Try to load payload — purged payloads (retention) are recorded as
        # absence (source_data=None, payload_available=False), not an error.
        source_data: dict[str, Any] | None = None
        payload_available = False

        if row.source_data_ref is not None and self._payload_store is not None:
            source_data, payload_available = self._load_payload_if_present(row_id, row.source_data_ref)

        return RowLineage(
            row_id=row.row_id,
            run_id=row.run_id,
            source_node_id=row.source_node_id,
            row_index=row.row_index,
            source_row_index=row.source_row_index,
            ingest_sequence=row.ingest_sequence,
            source_data_hash=row.source_data_hash,
            created_at=row.created_at,
            source_data=source_data,
            payload_available=payload_available,
        )
