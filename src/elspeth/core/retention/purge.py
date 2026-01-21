# src/elspeth/core/retention/purge.py
"""Purge manager for PayloadStore content based on retention policy.

Identifies payloads eligible for deletion based on run completion time
and retention period. Deletes blobs while preserving hashes in Landscape
for audit integrity.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

from sqlalchemy import and_, select, union

from elspeth.core.landscape.schema import (
    calls_table,
    node_states_table,
    nodes_table,
    routing_events_table,
    rows_table,
    runs_table,
)

if TYPE_CHECKING:
    from elspeth.core.landscape.database import LandscapeDB


class PayloadStoreProtocol(Protocol):
    """Protocol for PayloadStore to avoid circular imports.

    Defines the minimal interface required by PurgeManager.
    """

    def exists(self, content_hash: str) -> bool:
        """Check if content exists."""
        ...

    def delete(self, content_hash: str) -> bool:
        """Delete content by hash. Returns True if deleted."""
        ...


@dataclass
class PurgeResult:
    """Result of a purge operation.

    Note: bytes_freed is always 0 in the current implementation because
    PayloadStoreProtocol.delete() does not return the size of deleted content.
    This field is retained for future compatibility when PayloadStore provides
    size information on deletion.
    """

    deleted_count: int
    bytes_freed: int
    skipped_count: int  # Refs that didn't exist (already purged/never stored)
    failed_refs: list[str]  # Refs that existed but failed to delete
    duration_seconds: float


class PurgeManager:
    """Manages payload purging based on retention policy.

    Identifies expired payloads from completed runs and deletes them
    from the PayloadStore while preserving audit hashes in Landscape.
    """

    def __init__(self, db: "LandscapeDB", payload_store: PayloadStoreProtocol) -> None:
        """Initialize PurgeManager.

        Args:
            db: Landscape database connection
            payload_store: PayloadStore instance for blob operations
        """
        self._db = db
        self._payload_store = payload_store

    def find_expired_row_payloads(
        self,
        retention_days: int,
        as_of: datetime | None = None,
    ) -> list[str]:
        """Find row payloads eligible for deletion based on retention policy.

        Args:
            retention_days: Number of days to retain payloads after run completion
            as_of: Reference datetime for cutoff calculation (defaults to now)

        Returns:
            List of source_data_ref values for expired payloads
        """
        if as_of is None:
            as_of = datetime.now(UTC)

        cutoff = as_of - timedelta(days=retention_days)

        # Query rows from completed runs older than cutoff
        # Only return non-null source_data_ref values
        # Use distinct() because multiple rows can reference the same payload
        # (content-addressed storage means identical content shares one blob)
        query = (
            select(rows_table.c.source_data_ref)
            .distinct()
            .select_from(rows_table.join(runs_table, rows_table.c.run_id == runs_table.c.run_id))
            .where(
                and_(
                    runs_table.c.status == "completed",
                    runs_table.c.completed_at.isnot(None),
                    runs_table.c.completed_at < cutoff,
                    rows_table.c.source_data_ref.isnot(None),
                )
            )
        )

        with self._db.connection() as conn:
            result = conn.execute(query)
            refs = [row[0] for row in result]

        return refs

    def find_expired_payload_refs(
        self,
        retention_days: int,
        as_of: datetime | None = None,
    ) -> list[str]:
        """Find all payload refs eligible for deletion based on retention policy.

        This includes payloads from:
        - rows.source_data_ref (source row payloads)
        - calls.request_ref and calls.response_ref (external call payloads)
        - routing_events.reason_ref (routing reason payloads)

        Args:
            retention_days: Number of days to retain payloads after run completion
            as_of: Reference datetime for cutoff calculation (defaults to now)

        Returns:
            Deduplicated list of payload refs for expired payloads
        """
        if as_of is None:
            as_of = datetime.now(UTC)

        cutoff = as_of - timedelta(days=retention_days)

        # Common conditions for completed runs older than cutoff
        run_expired_condition = and_(
            runs_table.c.status == "completed",
            runs_table.c.completed_at.isnot(None),
            runs_table.c.completed_at < cutoff,
        )

        # 1. Row payloads: rows → runs
        row_query = (
            select(rows_table.c.source_data_ref)
            .select_from(rows_table.join(runs_table, rows_table.c.run_id == runs_table.c.run_id))
            .where(and_(run_expired_condition, rows_table.c.source_data_ref.isnot(None)))
        )

        # 2. Call payloads: calls → node_states → nodes → runs
        # Join path to get run_id for calls
        call_join = (
            calls_table.join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id)
            .join(nodes_table, node_states_table.c.node_id == nodes_table.c.node_id)
            .join(runs_table, nodes_table.c.run_id == runs_table.c.run_id)
        )

        # Request refs
        call_request_query = (
            select(calls_table.c.request_ref)
            .select_from(call_join)
            .where(and_(run_expired_condition, calls_table.c.request_ref.isnot(None)))
        )

        # Response refs
        call_response_query = (
            select(calls_table.c.response_ref)
            .select_from(call_join)
            .where(and_(run_expired_condition, calls_table.c.response_ref.isnot(None)))
        )

        # 3. Routing payloads: routing_events → node_states → nodes → runs
        routing_join = (
            routing_events_table.join(node_states_table, routing_events_table.c.state_id == node_states_table.c.state_id)
            .join(nodes_table, node_states_table.c.node_id == nodes_table.c.node_id)
            .join(runs_table, nodes_table.c.run_id == runs_table.c.run_id)
        )

        routing_query = (
            select(routing_events_table.c.reason_ref)
            .select_from(routing_join)
            .where(and_(run_expired_condition, routing_events_table.c.reason_ref.isnot(None)))
        )

        # Combine all queries with UNION (automatically deduplicates)
        combined_query = union(row_query, call_request_query, call_response_query, routing_query)

        with self._db.connection() as conn:
            result = conn.execute(combined_query)
            refs = [row[0] for row in result]

        return refs

    def purge_payloads(self, refs: list[str]) -> PurgeResult:
        """Purge payloads from the PayloadStore.

        Deletes each payload by reference, tracking successes and failures.
        Hashes in Landscape rows are preserved - only blobs are deleted.

        Args:
            refs: List of payload references (content hashes) to delete

        Returns:
            PurgeResult with deletion statistics. Note that skipped_count
            tracks refs that didn't exist (already purged or never stored),
            while failed_refs tracks refs that existed but failed to delete.
        """
        start_time = perf_counter()

        deleted_count = 0
        skipped_count = 0
        bytes_freed = 0  # Not tracked by current PayloadStore protocol
        failed_refs: list[str] = []

        for ref in refs:
            if self._payload_store.exists(ref):
                deleted = self._payload_store.delete(ref)
                if deleted:
                    deleted_count += 1
                else:
                    failed_refs.append(ref)
            else:
                # Ref doesn't exist - already purged or never stored
                # This is not a failure, just skip it
                skipped_count += 1

        duration_seconds = perf_counter() - start_time

        return PurgeResult(
            deleted_count=deleted_count,
            bytes_freed=bytes_freed,
            skipped_count=skipped_count,
            failed_refs=failed_refs,
            duration_seconds=duration_seconds,
        )
