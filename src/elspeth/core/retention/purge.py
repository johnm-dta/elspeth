# src/elspeth/core/retention/purge.py
"""Purge manager for PayloadStore content based on retention policy.

Identifies payloads eligible for deletion based on run completion time
and retention period. Deletes blobs while preserving hashes in Landscape
for audit integrity.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import TYPE_CHECKING

from sqlalchemy import and_, or_, select, union

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

    def __init__(self, db: "LandscapeDB", payload_store: PayloadStore) -> None:
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

        # Query rows from finished runs (completed OR failed) older than cutoff
        # Only return non-null source_data_ref values
        # Use distinct() because multiple rows can reference the same payload
        # (content-addressed storage means identical content shares one blob)
        #
        # Note: Both completed and failed runs are eligible for purge. Only
        # running runs (status="running") are excluded - they haven't finished
        # and their payloads may still be needed.
        query = (
            select(rows_table.c.source_data_ref)
            .distinct()
            .select_from(rows_table.join(runs_table, rows_table.c.run_id == runs_table.c.run_id))
            .where(
                and_(
                    runs_table.c.status != "running",
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
            runs_table.c.status != "running",
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
        )

        # === Build queries for refs from EXPIRED runs ===

        # 1. Row payloads from expired runs
        row_expired_query = (
            select(rows_table.c.source_data_ref)
            .select_from(rows_table.join(runs_table, rows_table.c.run_id == runs_table.c.run_id))
            .where(and_(run_expired_condition, rows_table.c.source_data_ref.isnot(None)))
        )

        # 2. Operation input/output payloads from expired runs
        operation_expired_join = operations_table.join(runs_table, operations_table.c.run_id == runs_table.c.run_id)

        operation_input_expired_query = (
            select(operations_table.c.input_data_ref)
            .select_from(operation_expired_join)
            .where(and_(run_expired_condition, operations_table.c.input_data_ref.isnot(None)))
        )

        operation_output_expired_query = (
            select(operations_table.c.output_data_ref)
            .select_from(operation_expired_join)
            .where(and_(run_expired_condition, operations_table.c.output_data_ref.isnot(None)))
        )

        # 3. Call payloads from expired runs (transform calls via state_id)
        # NOTE: Use node_states.run_id directly (denormalized column) instead of
        # joining through nodes table. The nodes table has composite PK (node_id, run_id),
        # so joining on node_id alone would be ambiguous when node_id is reused across runs.
        call_state_join = calls_table.join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id).join(
            runs_table, node_states_table.c.run_id == runs_table.c.run_id
        )

        call_state_request_expired_query = (
            select(calls_table.c.request_ref)
            .select_from(call_state_join)
            .where(and_(run_expired_condition, calls_table.c.request_ref.isnot(None)))
        )

        call_state_response_expired_query = (
            select(calls_table.c.response_ref)
            .select_from(call_state_join)
            .where(and_(run_expired_condition, calls_table.c.response_ref.isnot(None)))
        )

        # 4. Call payloads from expired runs (source/sink calls via operation_id)
        # XOR constraint: calls have either state_id OR operation_id, not both
        call_op_join = calls_table.join(operations_table, calls_table.c.operation_id == operations_table.c.operation_id).join(
            runs_table, operations_table.c.run_id == runs_table.c.run_id
        )

        call_op_request_expired_query = (
            select(calls_table.c.request_ref)
            .select_from(call_op_join)
            .where(and_(run_expired_condition, calls_table.c.request_ref.isnot(None)))
        )

        call_op_response_expired_query = (
            select(calls_table.c.response_ref)
            .select_from(call_op_join)
            .where(and_(run_expired_condition, calls_table.c.response_ref.isnot(None)))
        )

        # 5. Routing payloads from expired runs
        # NOTE: Same pattern as call_state_join - use node_states.run_id directly
        routing_join = routing_events_table.join(node_states_table, routing_events_table.c.state_id == node_states_table.c.state_id).join(
            runs_table, node_states_table.c.run_id == runs_table.c.run_id
        )

        routing_expired_query = (
            select(routing_events_table.c.reason_ref)
            .select_from(routing_join)
            .where(and_(run_expired_condition, routing_events_table.c.reason_ref.isnot(None)))
        )

        # Combine all expired refs
        expired_refs_query = union(
            row_expired_query,
            operation_input_expired_query,
            operation_output_expired_query,
            call_state_request_expired_query,
            call_state_response_expired_query,
            call_op_request_expired_query,
            call_op_response_expired_query,
            routing_expired_query,
        )

        # === Build queries for refs from ACTIVE runs (to exclude) ===

        # 1. Row payloads from active runs
        row_active_query = (
            select(rows_table.c.source_data_ref)
            .select_from(rows_table.join(runs_table, rows_table.c.run_id == runs_table.c.run_id))
            .where(and_(run_active_condition, rows_table.c.source_data_ref.isnot(None)))
        )

        # 2. Operation input/output payloads from active runs
        operation_active_join = operations_table.join(runs_table, operations_table.c.run_id == runs_table.c.run_id)

        operation_input_active_query = (
            select(operations_table.c.input_data_ref)
            .select_from(operation_active_join)
            .where(and_(run_active_condition, operations_table.c.input_data_ref.isnot(None)))
        )

        operation_output_active_query = (
            select(operations_table.c.output_data_ref)
            .select_from(operation_active_join)
            .where(and_(run_active_condition, operations_table.c.output_data_ref.isnot(None)))
        )

        # 3. Call payloads from active runs (transform calls via state_id)
        call_state_request_active_query = (
            select(calls_table.c.request_ref)
            .select_from(call_state_join)
            .where(and_(run_active_condition, calls_table.c.request_ref.isnot(None)))
        )

        call_state_response_active_query = (
            select(calls_table.c.response_ref)
            .select_from(call_state_join)
            .where(and_(run_active_condition, calls_table.c.response_ref.isnot(None)))
        )

        # 4. Call payloads from active runs (source/sink calls via operation_id)
        call_op_request_active_query = (
            select(calls_table.c.request_ref)
            .select_from(call_op_join)
            .where(and_(run_active_condition, calls_table.c.request_ref.isnot(None)))
        )

        call_op_response_active_query = (
            select(calls_table.c.response_ref)
            .select_from(call_op_join)
            .where(and_(run_active_condition, calls_table.c.response_ref.isnot(None)))
        )

        # 5. Routing payloads from active runs
        routing_active_query = (
            select(routing_events_table.c.reason_ref)
            .select_from(routing_join)
            .where(and_(run_active_condition, routing_events_table.c.reason_ref.isnot(None)))
        )

        # Combine all active refs
        active_refs_query = union(
            row_active_query,
            operation_input_active_query,
            operation_output_active_query,
            call_state_request_active_query,
            call_state_response_active_query,
            call_op_request_active_query,
            call_op_response_active_query,
            routing_active_query,
        )

        # === Execute both queries and compute set difference ===
        # We use Python set difference rather than SQL EXCEPT because:
        # 1. SQLite's EXCEPT can have performance issues with complex UNIONs
        # 2. The result sets are typically small enough for in-memory operation
        # 3. Python set operations are clearer for this anti-join pattern

        with self._db.connection() as conn:
            # Get all refs from expired runs
            expired_result = conn.execute(expired_refs_query)
            expired_refs = {row[0] for row in expired_result}

            # Get all refs from active runs
            active_result = conn.execute(active_refs_query)
            active_refs = {row[0] for row in active_result}

        # Return refs that are ONLY in expired runs (not in any active run)
        safe_to_delete = expired_refs - active_refs
        return list(safe_to_delete)

    def _find_affected_run_ids(self, refs: list[str]) -> set[str]:
        """Find run IDs that have payloads in the given refs list.

        Queries all payload reference columns to find which runs are affected
        by purging the specified refs. Used to update reproducibility grades
        after purge completes.

        Args:
            refs: List of payload references (content hashes) being purged

        Returns:
            Set of run_ids that have at least one payload in the refs list
        """
        if not refs:
            return set()

        refs_set = set(refs)

        # Query all run_ids that have any of these refs
        # 1. From rows.source_data_ref
        row_runs_query = select(rows_table.c.run_id).distinct().where(rows_table.c.source_data_ref.in_(refs_set))

        # 2. From operations.input_data_ref and operations.output_data_ref
        operation_input_runs_query = select(operations_table.c.run_id).distinct().where(operations_table.c.input_data_ref.in_(refs_set))
        operation_output_runs_query = select(operations_table.c.run_id).distinct().where(operations_table.c.output_data_ref.in_(refs_set))

        # 3. From calls.request_ref and calls.response_ref (transform calls via state_id)
        # Use node_states.run_id directly (denormalized column)
        call_state_join = calls_table.join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id)

        call_state_request_runs_query = (
            select(node_states_table.c.run_id).distinct().select_from(call_state_join).where(calls_table.c.request_ref.in_(refs_set))
        )

        call_state_response_runs_query = (
            select(node_states_table.c.run_id).distinct().select_from(call_state_join).where(calls_table.c.response_ref.in_(refs_set))
        )

        # 4. From calls.request_ref and calls.response_ref (source/sink calls via operation_id)
        # XOR constraint: calls have either state_id OR operation_id, not both
        call_op_join = calls_table.join(operations_table, calls_table.c.operation_id == operations_table.c.operation_id)

        call_op_request_runs_query = (
            select(operations_table.c.run_id).distinct().select_from(call_op_join).where(calls_table.c.request_ref.in_(refs_set))
        )

        call_op_response_runs_query = (
            select(operations_table.c.run_id).distinct().select_from(call_op_join).where(calls_table.c.response_ref.in_(refs_set))
        )

        # 5. From routing_events.reason_ref
        routing_join = routing_events_table.join(node_states_table, routing_events_table.c.state_id == node_states_table.c.state_id)

        routing_runs_query = (
            select(node_states_table.c.run_id).distinct().select_from(routing_join).where(routing_events_table.c.reason_ref.in_(refs_set))
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
            while failed_refs tracks refs that existed but failed to delete.
        """
        start_time = perf_counter()

        # Step 1: Delete the payloads, tracking which refs were actually deleted
        deleted_count = 0
        skipped_count = 0
        bytes_freed = 0  # Not tracked by current PayloadStore protocol
        failed_refs: list[str] = []
        deleted_refs: list[str] = []

        for ref in refs:
            try:
                exists = self._payload_store.exists(ref)
            except OSError:
                # I/O error checking existence - record as failure, continue with others
                failed_refs.append(ref)
                continue

            if exists:
                try:
                    deleted = self._payload_store.delete(ref)
                except OSError:
                    # I/O error during deletion - record as failure, continue with others
                    failed_refs.append(ref)
                    continue

                if deleted:
                    deleted_count += 1
                    deleted_refs.append(ref)
                else:
                    failed_refs.append(ref)
            else:
                # Ref doesn't exist - already purged or never stored
                # This is not a failure, just skip it
                skipped_count += 1

        # Step 2: Find runs affected by ONLY the successfully deleted refs
        # Runs with only failed refs still have their payloads and should not be downgraded
        affected_run_ids = self._find_affected_run_ids(deleted_refs)

        # Step 3: Update reproducibility grades for affected runs
        # This degrades REPLAY_REPRODUCIBLE -> ATTRIBUTABLE_ONLY since
        # nondeterministic runs can no longer be replayed without payloads
        for run_id in affected_run_ids:
            update_grade_after_purge(self._db, run_id)

        duration_seconds = perf_counter() - start_time

        return PurgeResult(
            deleted_count=deleted_count,
            bytes_freed=bytes_freed,
            skipped_count=skipped_count,
            failed_refs=failed_refs,
            duration_seconds=duration_seconds,
        )
