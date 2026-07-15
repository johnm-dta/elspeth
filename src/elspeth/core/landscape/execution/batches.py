"""Aggregation batch persistence (split from ``ExecutionRepository``).

Owns the ``batches`` and ``batch_members`` audit aggregates: batch lifecycle
(draft → executing → terminal), membership, recovery reads, and idempotent
retry-batch creation.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import insert, literal, select
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts import Batch, BatchMember, BatchStatus, TriggerType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.ids import generate_id
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import now
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.model_loaders import BatchLoader, BatchMemberLoader
from elspeth.core.landscape.schema import batch_members_table, batches_table, tokens_table

_TERMINAL_BATCH_STATUSES = frozenset({BatchStatus.COMPLETED, BatchStatus.FAILED})
_MEMBERSHIP_MUTABLE_BATCH_STATUSES = frozenset({BatchStatus.DRAFT})
_MEMBERSHIP_MUTABLE_BATCH_STATUS_VALUES = tuple(status.value for status in _MEMBERSHIP_MUTABLE_BATCH_STATUSES)


def add_batch_member_guarded(
    conn: Connection,
    *,
    batch_id: str,
    token_id: str,
    ordinal: int,
    expected_run_id: str | None = None,
) -> BatchMember:
    """Insert one member through the shared DRAFT-only transaction primitive.

    SQLite deliberately executes the conditional INSERT as its first
    snapshot-establishing statement.  That lets a caller-owned DEFERRED
    transaction acquire write intent before any predicate read, eliminating
    the read/peer-write/write ``SQLITE_BUSY_SNAPSHOT`` window.  PostgreSQL
    keeps the batch-row ``FOR UPDATE`` lock first so status transitions and
    membership additions retain their established lock ordering.

    ``expected_run_id`` binds scheduler callers to their already-fenced run;
    ordinary batch repository callers rely on the batch/token run join.
    """
    batch_row = None
    if conn.dialect.name != "sqlite":
        batch_row = conn.execute(
            select(batches_table.c.run_id, batches_table.c.status).where(batches_table.c.batch_id == batch_id).with_for_update()
        ).fetchone()
        if batch_row is None:
            raise AuditIntegrityError(f"Cannot add batch member: batch {batch_id} not found")
        if expected_run_id is not None and batch_row.run_id != expected_run_id:
            raise AuditIntegrityError(
                f"Cannot add batch member: batch {batch_id} belongs to run {batch_row.run_id!r}, not expected run {expected_run_id!r}"
            )
        if batch_row.status not in _MEMBERSHIP_MUTABLE_BATCH_STATUS_VALUES:
            raise AuditIntegrityError(
                f"Cannot add batch member: batch {batch_id} has status {batch_row.status!r}; "
                "batch membership is immutable outside status 'draft'."
            )

    source = (
        select(
            literal(batch_id),
            batches_table.c.run_id,
            tokens_table.c.token_id,
            literal(ordinal),
        )
        .select_from(
            batches_table.join(
                tokens_table,
                tokens_table.c.run_id == batches_table.c.run_id,
            )
        )
        .where(batches_table.c.batch_id == batch_id)
        .where(batches_table.c.status.in_(_MEMBERSHIP_MUTABLE_BATCH_STATUS_VALUES))
        .where(tokens_table.c.token_id == token_id)
    )
    if expected_run_id is not None:
        source = source.where(batches_table.c.run_id == expected_run_id)

    inserted = conn.execute(
        insert(batch_members_table)
        .from_select(
            ["batch_id", "run_id", "token_id", "ordinal"],
            source,
        )
        .returning(
            batch_members_table.c.batch_id,
            batch_members_table.c.run_id,
            batch_members_table.c.token_id,
            batch_members_table.c.ordinal,
        )
    ).fetchone()
    if inserted is not None:
        return BatchMember(
            batch_id=str(inserted.batch_id),
            run_id=str(inserted.run_id),
            token_id=str(inserted.token_id),
            ordinal=int(inserted.ordinal),
        )

    if batch_row is None:
        batch_row = conn.execute(
            select(batches_table.c.run_id, batches_table.c.status).where(batches_table.c.batch_id == batch_id)
        ).fetchone()
    if batch_row is None:
        raise AuditIntegrityError(f"Cannot add batch member: batch {batch_id} not found")
    if expected_run_id is not None and batch_row.run_id != expected_run_id:
        raise AuditIntegrityError(
            f"Cannot add batch member: batch {batch_id} belongs to run {batch_row.run_id!r}, not expected run {expected_run_id!r}"
        )
    if batch_row.status not in _MEMBERSHIP_MUTABLE_BATCH_STATUS_VALUES:
        raise AuditIntegrityError(
            f"Cannot add batch member: batch {batch_id} has status {batch_row.status!r}; "
            "batch membership is immutable outside status 'draft'."
        )

    token_run_id = conn.execute(select(tokens_table.c.run_id).where(tokens_table.c.token_id == token_id)).scalar_one_or_none()
    if token_run_id is None:
        raise AuditIntegrityError(f"Cannot add batch member: token {token_id} not found")
    if token_run_id != batch_row.run_id:
        raise AuditIntegrityError(
            f"Cannot add batch member: token {token_id} belongs to run {token_run_id!r}, "
            f"but batch {batch_id} belongs to run {batch_row.run_id!r}"
        )
    raise AuditIntegrityError(
        f"Cannot add batch member: conditional INSERT returned no row for batch {batch_id} and token {token_id}; expected exactly one"
    )


class BatchRepository:
    """Aggregation batch lifecycle and membership for the audit trail."""

    def __init__(
        self,
        db: LandscapeDB,
        ops: DatabaseOps,
        *,
        batch_loader: BatchLoader,
        batch_member_loader: BatchMemberLoader,
    ) -> None:
        self._db = db
        self._ops = ops
        self._batch_loader = batch_loader
        self._batch_member_loader = batch_member_loader

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
        *,
        conn: Connection | None = None,
    ) -> BatchMember:
        """Add a token to a DRAFT batch in one atomic database boundary.

        Args:
            batch_id: Batch to add to
            token_id: Token to add
            ordinal: Order in batch
            conn: Optional caller-owned write transaction

        Returns:
            BatchMember model
        """

        try:
            if conn is not None:
                return add_batch_member_guarded(conn, batch_id=batch_id, token_id=token_id, ordinal=ordinal)
            with self._db.write_connection() as active_conn:
                return add_batch_member_guarded(active_conn, batch_id=batch_id, token_id=token_id, ordinal=ordinal)
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"add_batch_member failed (batch_members) — database rejected audit write: {type(exc).__name__}"
            ) from exc

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
            AuditIntegrityError: If status is terminal, batch not found, or current status is terminal
        """
        if status in _TERMINAL_BATCH_STATUSES:
            raise AuditIntegrityError(f"update_batch_status() cannot write terminal status {status.value!r}; use complete_batch().")

        updates: dict[str, Any] = {"status": status}

        if trigger_type is not None:
            updates["trigger_type"] = trigger_type
        if trigger_reason is not None:
            updates["trigger_reason"] = trigger_reason
        if state_id is not None:
            updates["aggregation_state_id"] = state_id

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
