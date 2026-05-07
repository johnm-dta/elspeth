"""Purge manager for PayloadStore content based on retention policy.

Identifies payloads eligible for deletion based on run completion time
and retention period. Deletes blobs while preserving hashes in Landscape
for audit integrity.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import ColumnElement, CompoundSelect, FromClause, and_, or_, select, union
from sqlalchemy.engine import Connection

import elspeth.contracts.errors as contract_errors
from elspeth.contracts import RunStatus
from elspeth.contracts.payload_store import PayloadStore
from elspeth.core.landscape.reproducibility import update_grade_after_purge
from elspeth.core.landscape.schema import (
    calls_table,
    node_states_table,
    operations_table,
    routing_events_table,
    rows_table,
    runs_table,
)

if TYPE_CHECKING:
    from elspeth.core.landscape.database import LandscapeDB


@dataclass(frozen=True, slots=True)
class PurgeResult:
    """Result of a purge operation."""

    deleted_count: int
    skipped_count: int  # Refs that didn't exist (already purged/never stored)
    failed_refs: tuple[str, ...]  # Refs whose delete operation raised
    grade_update_failures: tuple[str, ...]  # Run IDs whose grade update failed after deletion
    duration_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "failed_refs", tuple(self.failed_refs))
        object.__setattr__(self, "grade_update_failures", tuple(self.grade_update_failures))
        if self.deleted_count < 0:
            raise ValueError(f"deleted_count must be non-negative, got {self.deleted_count}")
        if self.skipped_count < 0:
            raise ValueError(f"skipped_count must be non-negative, got {self.skipped_count}")
        if self.duration_seconds < 0:
            raise ValueError(f"duration_seconds must be non-negative, got {self.duration_seconds}")


logger = structlog.get_logger()


class PurgeManager:
    """Manages payload purging based on retention policy.

    Identifies expired payloads from completed runs and deletes them
    from the PayloadStore while preserving audit hashes in Landscape.
    """

    def __init__(self, db: "LandscapeDB", payload_store: PayloadStore) -> None:
        """Initialize PurgeManager.

        Args:
            db: Landscape database connection
            payload_store: PayloadStore instance for blob operations
        """
        self._db = db
        self._payload_store = payload_store

    def _validate_run_lifecycle_rows(self, conn: Connection) -> None:
        """Crash on impossible Tier-1 run lifecycle rows before purge queries."""
        query = select(runs_table.c.run_id, runs_table.c.status, runs_table.c.completed_at)
        result = conn.execute(query)

        for row in result:
            try:
                status = RunStatus(row.status)
            except ValueError as exc:
                raise contract_errors.AuditIntegrityError(
                    f"Invalid run status '{row.status}' for run {row.run_id} — expected one of {[status.value for status in RunStatus]}"
                ) from exc

            if status == RunStatus.RUNNING and row.completed_at is not None:
                raise contract_errors.AuditIntegrityError(
                    f"Run {row.run_id} has status='running' but completed_at is set — "
                    f"audit integrity violation (running runs must not have completed_at)"
                )
            if status == RunStatus.COMPLETED and row.completed_at is None:
                raise contract_errors.AuditIntegrityError(
                    f"Run {row.run_id} has status='completed' but completed_at is NULL — "
                    f"audit integrity violation (completed runs must have completed_at)"
                )

    def _build_ref_union_query(
        self,
        run_condition: ColumnElement[bool],
        *,
        rows_join: FromClause,
        operation_join: FromClause,
        call_state_join: FromClause,
        call_op_join: FromClause,
        routing_join: FromClause,
    ) -> CompoundSelect[Any]:
        """Build a UNION of all 8 payload ref sub-queries for a given run condition.

        Each sub-query selects a single ref column from a different table/join,
        filtered by the run condition and a NOT NULL guard on the ref column.

        Args:
            run_condition: SQLAlchemy WHERE clause for run filtering
                (e.g. expired condition or active condition)
            rows_join: Pre-built join for rows → runs
            operation_join: Pre-built join for operations → runs
            call_state_join: Pre-built join for calls → node_states → runs
            call_op_join: Pre-built join for calls → operations → runs
            routing_join: Pre-built join for routing_events → node_states → runs

        Returns:
            UNION of all 8 sub-queries
        """
        return union(
            # 1. Row payloads
            select(rows_table.c.source_data_ref)
            .select_from(rows_join)
            .where(and_(run_condition, rows_table.c.source_data_ref.isnot(None))),
            # 2. Operation input payloads
            select(operations_table.c.input_data_ref)
            .select_from(operation_join)
            .where(and_(run_condition, operations_table.c.input_data_ref.isnot(None))),
            # 3. Operation output payloads
            select(operations_table.c.output_data_ref)
            .select_from(operation_join)
            .where(and_(run_condition, operations_table.c.output_data_ref.isnot(None))),
            # 4. Call request payloads (transform calls via state_id)
            select(calls_table.c.request_ref)
            .select_from(call_state_join)
            .where(and_(run_condition, calls_table.c.request_ref.isnot(None))),
            # 5. Call response payloads (transform calls via state_id)
            select(calls_table.c.response_ref)
            .select_from(call_state_join)
            .where(and_(run_condition, calls_table.c.response_ref.isnot(None))),
            # 6. Call request payloads (source/sink calls via operation_id)
            select(calls_table.c.request_ref).select_from(call_op_join).where(and_(run_condition, calls_table.c.request_ref.isnot(None))),
            # 7. Call response payloads (source/sink calls via operation_id)
            select(calls_table.c.response_ref).select_from(call_op_join).where(and_(run_condition, calls_table.c.response_ref.isnot(None))),
            # 8. Routing reason payloads
            select(routing_events_table.c.reason_ref)
            .select_from(routing_join)
            .where(and_(run_condition, routing_events_table.c.reason_ref.isnot(None))),
        )

    def find_expired_payload_refs(
        self,
        retention_days: int,
        as_of: datetime | None = None,
    ) -> list[str]:
        """Find all payload refs eligible for deletion based on retention policy.

        This includes payloads from:
        - rows.source_data_ref (source row payloads)
        - operations.input_data_ref and operations.output_data_ref (source/sink operation payloads)
        - calls.request_ref and calls.response_ref (external call payloads)
        - routing_events.reason_ref (routing reason payloads)

        IMPORTANT: Because payloads are content-addressable, the same hash can
        appear in multiple runs. We must exclude refs that are still used by
        non-expired runs to avoid breaking replay/explain for active runs.

        Args:
            retention_days: Number of days to retain payloads after run completion
            as_of: Reference datetime for cutoff calculation (defaults to now)

        Returns:
            Deduplicated list of payload refs for expired payloads that are NOT
            used by any non-expired or incomplete runs
        """
        if as_of is None:
            as_of = datetime.now(UTC)

        cutoff = as_of - timedelta(days=retention_days)

        # Condition for expired runs: finished (not running) AND older than cutoff
        # Both "completed" and "failed" runs are eligible for purge once they're
        # past the retention period. Only "running" runs are excluded.
        run_expired_condition = and_(
            runs_table.c.status.in_(("completed", "failed")),
            runs_table.c.completed_at.isnot(None),
            runs_table.c.completed_at < cutoff,
        )

        # Condition for active runs: NOT expired (recent or still running)
        # A run is "active" if any of:
        # - completed_at >= cutoff (recent, within retention period)
        # - completed_at IS NULL (still running, hasn't finished yet)
        # - status == "running" (explicitly marked as running)
        run_active_condition = or_(
            runs_table.c.completed_at >= cutoff,
            runs_table.c.completed_at.is_(None),
            runs_table.c.status == "running",
            runs_table.c.status == "interrupted",  # Interrupted runs are recovery candidates
        )

        # === Build joins (shared between expired and active queries) ===
        # NOTE: Use node_states.run_id directly (denormalized column) instead of
        # joining through nodes table. The nodes table has composite PK (node_id, run_id),
        # so joining on node_id alone would be ambiguous when node_id is reused across runs.
        rows_join = rows_table.join(runs_table, rows_table.c.run_id == runs_table.c.run_id)
        operation_join = operations_table.join(runs_table, operations_table.c.run_id == runs_table.c.run_id)
        call_state_join = calls_table.join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id).join(
            runs_table, node_states_table.c.run_id == runs_table.c.run_id
        )
        # XOR constraint: calls have either state_id OR operation_id, not both
        call_op_join = calls_table.join(operations_table, calls_table.c.operation_id == operations_table.c.operation_id).join(
            runs_table, operations_table.c.run_id == runs_table.c.run_id
        )
        routing_join = routing_events_table.join(node_states_table, routing_events_table.c.state_id == node_states_table.c.state_id).join(
            runs_table, node_states_table.c.run_id == runs_table.c.run_id
        )

        expired_refs_query = self._build_ref_union_query(
            run_expired_condition,
            rows_join=rows_join,
            operation_join=operation_join,
            call_state_join=call_state_join,
            call_op_join=call_op_join,
            routing_join=routing_join,
        )
        active_refs_query = self._build_ref_union_query(
            run_active_condition,
            rows_join=rows_join,
            operation_join=operation_join,
            call_state_join=call_state_join,
            call_op_join=call_op_join,
            routing_join=routing_join,
        )

        # === Execute both queries and compute set difference ===
        # We use Python set difference rather than SQL EXCEPT because:
        # 1. SQLite's EXCEPT can have performance issues with complex UNIONs
        # 2. The result sets are typically small enough for in-memory operation
        # 3. Python set operations are clearer for this anti-join pattern

        with self._db.connection() as conn:
            self._validate_run_lifecycle_rows(conn)

            # Get all refs from expired runs
            expired_result = conn.execute(expired_refs_query)
            expired_refs = {row[0] for row in expired_result}

            # Get all refs from active runs
            active_result = conn.execute(active_refs_query)
            active_refs = {row[0] for row in active_result}

        # Return refs that are ONLY in expired runs (not in any active run)
        safe_to_delete = expired_refs - active_refs
        return list(safe_to_delete)

    # SQLite default SQLITE_MAX_VARIABLE_NUMBER is 999. Chunk IN clauses
    # to stay well under this limit (8 queries x chunk_size variables each).
    _PURGE_CHUNK_SIZE = 100

    def _find_affected_run_ids(self, refs: list[str]) -> set[str]:
        """Find run IDs that have payloads in the given refs list.

        Queries all payload reference columns to find which runs are affected
        by purging the specified refs. Used to update reproducibility grades
        after purge completes.

        Large ref lists are chunked to avoid exceeding SQLite's bind variable
        limit (SQLITE_MAX_VARIABLE_NUMBER, default 999).

        Args:
            refs: List of payload references (content hashes) being purged

        Returns:
            Set of run_ids that have at least one payload in the refs list
        """
        if not refs:
            return set()

        refs_list = list(set(refs))
        all_run_ids: set[str] = set()

        for offset in range(0, len(refs_list), self._PURGE_CHUNK_SIZE):
            chunk = refs_list[offset : offset + self._PURGE_CHUNK_SIZE]
            all_run_ids |= self._find_affected_run_ids_chunk(chunk)

        return all_run_ids

    def _find_affected_run_ids_chunk(self, refs_chunk: list[str]) -> set[str]:
        """Find run IDs affected by a single chunk of refs."""
        # 1. From rows.source_data_ref
        row_runs_query = select(rows_table.c.run_id).distinct().where(rows_table.c.source_data_ref.in_(refs_chunk))

        # 2. From operations.input_data_ref and operations.output_data_ref
        operation_input_runs_query = select(operations_table.c.run_id).distinct().where(operations_table.c.input_data_ref.in_(refs_chunk))
        operation_output_runs_query = select(operations_table.c.run_id).distinct().where(operations_table.c.output_data_ref.in_(refs_chunk))

        # 3. From calls.request_ref and calls.response_ref (transform calls via state_id)
        # Use node_states.run_id directly (denormalized column)
        call_state_join = calls_table.join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id)

        call_state_request_runs_query = (
            select(node_states_table.c.run_id).distinct().select_from(call_state_join).where(calls_table.c.request_ref.in_(refs_chunk))
        )

        call_state_response_runs_query = (
            select(node_states_table.c.run_id).distinct().select_from(call_state_join).where(calls_table.c.response_ref.in_(refs_chunk))
        )

        # 4. From calls.request_ref and calls.response_ref (source/sink calls via operation_id)
        # XOR constraint: calls have either state_id OR operation_id, not both
        call_op_join = calls_table.join(operations_table, calls_table.c.operation_id == operations_table.c.operation_id)

        call_op_request_runs_query = (
            select(operations_table.c.run_id).distinct().select_from(call_op_join).where(calls_table.c.request_ref.in_(refs_chunk))
        )

        call_op_response_runs_query = (
            select(operations_table.c.run_id).distinct().select_from(call_op_join).where(calls_table.c.response_ref.in_(refs_chunk))
        )

        # 5. From routing_events.reason_ref
        routing_join = routing_events_table.join(node_states_table, routing_events_table.c.state_id == node_states_table.c.state_id)

        routing_runs_query = (
            select(node_states_table.c.run_id).distinct().select_from(routing_join).where(routing_events_table.c.reason_ref.in_(refs_chunk))
        )

        # Union all run_id queries
        all_runs_query = union(
            row_runs_query,
            operation_input_runs_query,
            operation_output_runs_query,
            call_state_request_runs_query,
            call_state_response_runs_query,
            call_op_request_runs_query,
            call_op_response_runs_query,
            routing_runs_query,
        )

        with self._db.connection() as conn:
            result = conn.execute(all_runs_query)
            return {row[0] for row in result}

    def purge_payloads(self, refs: list[str]) -> PurgeResult:
        """Purge payloads from the PayloadStore.

        Deletes each payload by reference, tracking successes and failures.
        Hashes in Landscape rows are preserved - only blobs are deleted.

        After deletion, updates reproducibility_grade for affected runs:
        - REPLAY_REPRODUCIBLE -> ATTRIBUTABLE_ONLY (payloads needed for replay are gone)
        - FULL_REPRODUCIBLE -> unchanged (doesn't depend on payloads)
        - ATTRIBUTABLE_ONLY -> unchanged (already at lowest grade)

        Grade updates only occur for runs whose payloads were actually deleted.
        Runs that only had failed deletions retain their grade (payloads still exist).

        Args:
            refs: List of payload references (content hashes) to delete

        Returns:
            PurgeResult with deletion statistics. Note that skipped_count
            tracks refs that didn't exist (already purged or never stored),
            while failed_refs tracks refs whose delete operation raised.
        """
        start_time = perf_counter()

        # Step 1: Delete the payloads, tracking which refs were actually deleted
        deleted_count = 0
        skipped_count = 0
        failed_refs: list[str] = []
        deleted_refs: list[str] = []

        for ref in refs:
            try:
                deleted = self._payload_store.delete(ref)
            except OSError as e:
                logger.warning(
                    "payload_deletion_failed",
                    ref=ref,
                    error_type=type(e).__name__,
                    error=str(e),
                )
                failed_refs.append(ref)
                continue

            if deleted:
                deleted_count += 1
                deleted_refs.append(ref)
            else:
                # Ref doesn't exist - already purged or never stored
                # This is not a failure, just skip it
                skipped_count += 1

        # Step 2: Find runs affected by ONLY the successfully deleted refs
        # Runs with only failed refs still have their payloads and should not be downgraded
        affected_run_ids = self._find_affected_run_ids(deleted_refs)

        # Step 3: Update reproducibility grades for affected runs
        # This degrades REPLAY_REPRODUCIBLE -> ATTRIBUTABLE_ONLY since
        # nondeterministic runs can no longer be replayed without payloads.
        # Each update is wrapped individually because payloads are already
        # irreversibly deleted — a failure for one run must not prevent
        # grade updates for the remaining runs.
        grade_update_failures: list[str] = []
        for run_id in sorted(affected_run_ids):
            try:
                update_grade_after_purge(self._db, run_id, deleted_refs=deleted_refs)
            except contract_errors.TIER_1_ERRORS:
                raise  # Tier 1 errors must crash — never swallow
            except Exception as exc:
                logger.warning(
                    "grade_update_failed",
                    run_id=run_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    msg="Payloads already deleted but grade update failed — run may have stale reproducibility grade",
                )
                grade_update_failures.append(run_id)

        duration_seconds = perf_counter() - start_time

        return PurgeResult(
            deleted_count=deleted_count,
            skipped_count=skipped_count,
            failed_refs=tuple(failed_refs),
            grade_update_failures=tuple(grade_update_failures),
            duration_seconds=duration_seconds,
        )
