# src/elspeth/core/landscape/_query_methods.py
"""Read-only query methods for LandscapeRecorder."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from elspeth.contracts import (
    Call,
    NodeState,
    Row,
    RowLineage,
    Token,
    TokenParent,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.row_data import RowDataResult, RowDataState
from elspeth.core.landscape.schema import (
    calls_table,
    node_states_table,
    routing_events_table,
    rows_table,
    token_parents_table,
    tokens_table,
)

if TYPE_CHECKING:
    from elspeth.contracts import RoutingEvent
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.landscape._database_ops import DatabaseOps
    from elspeth.core.landscape.database import LandscapeDB
    from elspeth.core.landscape.repositories import (
        CallRepository,
        NodeStateRepository,
        RoutingEventRepository,
        RowRepository,
        TokenParentRepository,
        TokenRepository,
    )


class QueryMethodsMixin:
    """Read-only query methods. Mixed into LandscapeRecorder."""

    # Shared state annotations (set by LandscapeRecorder.__init__)
    _db: LandscapeDB
    _ops: DatabaseOps
    _row_repo: RowRepository
    _token_repo: TokenRepository
    _token_parent_repo: TokenParentRepository
    _call_repo: CallRepository
    _node_state_repo: NodeStateRepository
    _routing_event_repo: RoutingEventRepository
    _payload_store: PayloadStore | None

    def get_rows(self, run_id: str) -> list[Row]:
        """Get all rows for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Row models, ordered by row_index
        """
        query = select(rows_table).where(rows_table.c.run_id == run_id).order_by(rows_table.c.row_index)
        db_rows = self._ops.execute_fetchall(query)
        return [self._row_repo.load(r) for r in db_rows]

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
        return [self._token_repo.load(r) for r in db_rows]

    def get_node_states_for_token(self, token_id: str) -> list[NodeState]:
        """Get all node states for a token.

        Args:
            token_id: Token ID

        Returns:
            List of NodeState models (discriminated union), ordered by (step_index, attempt)
        """
        # Order by (step_index, attempt) for deterministic ordering across retries
        # Bug fix: P2-2026-01-19-node-state-ordering-missing-attempt
        query = (
            select(node_states_table)
            .where(node_states_table.c.token_id == token_id)
            .order_by(node_states_table.c.step_index, node_states_table.c.attempt)
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._node_state_repo.load(r) for r in db_rows]

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
        return self._row_repo.load(r)

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
            payload_bytes = self._payload_store.retrieve(row.source_data_ref)
            decoded_data = json.loads(payload_bytes.decode("utf-8"))
            match decoded_data:
                case dict() as data:
                    pass
                case _:
                    actual_type = type(decoded_data).__name__
                    raise AuditIntegrityError(
                        f"Corrupt payload for row {row_id} (ref={row.source_data_ref}): expected JSON object, got {actual_type}"
                    )
            return RowDataResult(state=RowDataState.AVAILABLE, data=data)
        except KeyError:
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
        return self._token_repo.load(r)

    def get_token_parents(self, token_id: str) -> list[TokenParent]:
        """Get parent relationships for a token.

        Args:
            token_id: Token ID

        Returns:
            List of TokenParent models (ordered by ordinal)
        """
        query = select(token_parents_table).where(token_parents_table.c.token_id == token_id).order_by(token_parents_table.c.ordinal)
        db_rows = self._ops.execute_fetchall(query)
        return [self._token_parent_repo.load(r) for r in db_rows]

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
        return [self._routing_event_repo.load(r) for r in db_rows]

    def get_calls(self, state_id: str) -> list[Call]:
        """Get external calls for a node state.

        Args:
            state_id: State ID

        Returns:
            List of Call models, ordered by call_index
        """
        query = select(calls_table).where(calls_table.c.state_id == state_id).order_by(calls_table.c.call_index)
        db_rows = self._ops.execute_fetchall(query)
        return [self._call_repo.load(r) for r in db_rows]

    # === Batch Query Methods for State Sets (ech8: N+1 query fix for lineage) ===
    #
    # These methods fetch entities for a set of state IDs in a single query,
    # replacing the N+1 pattern where per-state queries nested inside loops.

    def get_routing_events_for_states(self, state_ids: list[str]) -> list[RoutingEvent]:
        """Get routing events for multiple states in one query.

        Args:
            state_ids: List of state IDs to query

        Returns:
            List of RoutingEvent models, ordered by execution order
            (step_index, attempt, ordinal, event_id)
        """
        if not state_ids:
            return []
        query = (
            select(routing_events_table)
            .join(node_states_table, routing_events_table.c.state_id == node_states_table.c.state_id)
            .where(routing_events_table.c.state_id.in_(state_ids))
            .order_by(
                node_states_table.c.step_index,
                node_states_table.c.attempt,
                routing_events_table.c.ordinal,
                routing_events_table.c.event_id,
            )
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._routing_event_repo.load(r) for r in db_rows]

    def get_calls_for_states(self, state_ids: list[str]) -> list[Call]:
        """Get external calls for multiple states in one query.

        Args:
            state_ids: List of state IDs to query

        Returns:
            List of Call models, ordered by execution order
            (step_index, attempt, call_index)
        """
        if not state_ids:
            return []
        query = (
            select(calls_table)
            .join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id)
            .where(calls_table.c.state_id.in_(state_ids))
            .order_by(
                node_states_table.c.step_index,
                node_states_table.c.attempt,
                calls_table.c.call_index,
            )
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._call_repo.load(r) for r in db_rows]

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
        return [self._token_repo.load(r) for r in db_rows]

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
        return [self._node_state_repo.load(r) for r in db_rows]

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
                routing_events_table.c.ordinal,
                routing_events_table.c.event_id,
            )
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._routing_event_repo.load(r) for r in db_rows]

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
                calls_table.c.call_index,
            )
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._call_repo.load(r) for r in db_rows]

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
        return [self._token_parent_repo.load(r) for r in db_rows]

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
            or if row doesn't belong to the specified run
        """
        row = self.get_row(row_id)
        if row is None:
            return None

        # Validate row belongs to the specified run - audit systems must be strict
        if row.run_id != run_id:
            return None

        # Try to load payload
        source_data: dict[str, Any] | None = None
        payload_available = False

        if row.source_data_ref and self._payload_store:
            try:
                payload_bytes = self._payload_store.retrieve(row.source_data_ref)
                decoded_source_data = json.loads(payload_bytes.decode("utf-8"))
                match decoded_source_data:
                    case dict() as source_data_dict:
                        source_data = source_data_dict
                    case _:
                        actual_type = type(decoded_source_data).__name__
                        raise AuditIntegrityError(
                            f"Corrupt payload for row {row_id} (ref={row.source_data_ref}): expected JSON object, got {actual_type}"
                        )
                payload_available = True
            except KeyError:
                # Payload purged by retention policy — expected, continue without data
                pass
            except json.JSONDecodeError as e:
                # Tier 1 violation: payload store data is OUR data — corruption is catastrophic
                raise AuditIntegrityError(f"Corrupt payload for row {row_id} (ref={row.source_data_ref}): {e}") from e
            except OSError as e:
                # Infrastructure issue (NFS timeout, disk full) — payload unavailable
                logging.getLogger(__name__).warning(
                    "Payload retrieval failed for row %s (ref=%s): %s: %s",
                    row_id,
                    row.source_data_ref,
                    type(e).__name__,
                    e,
                )

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
