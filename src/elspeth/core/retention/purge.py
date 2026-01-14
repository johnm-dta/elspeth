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

from sqlalchemy import and_, select

from elspeth.core.landscape.schema import rows_table, runs_table

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
    """Result of a purge operation."""

    deleted_count: int
    bytes_freed: int
    failed_refs: list[str]
    duration_seconds: float


class PurgeManager:
    """Manages payload purging based on retention policy.

    Identifies expired payloads from completed runs and deletes them
    from the PayloadStore while preserving audit hashes in Landscape.
    """

    def __init__(
        self, db: "LandscapeDB", payload_store: PayloadStoreProtocol
    ) -> None:
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
        query = (
            select(rows_table.c.source_data_ref)
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

    def purge_payloads(self, refs: list[str]) -> PurgeResult:
        """Purge payloads from the PayloadStore.

        Deletes each payload by reference, tracking successes and failures.
        Hashes in Landscape rows are preserved - only blobs are deleted.

        Args:
            refs: List of payload references (content hashes) to delete

        Returns:
            PurgeResult with deletion statistics
        """
        start_time = perf_counter()

        deleted_count = 0
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
                # Ref doesn't exist in store - record as failed
                failed_refs.append(ref)

        duration_seconds = perf_counter() - start_time

        return PurgeResult(
            deleted_count=deleted_count,
            bytes_freed=bytes_freed,
            failed_refs=failed_refs,
            duration_seconds=duration_seconds,
        )
