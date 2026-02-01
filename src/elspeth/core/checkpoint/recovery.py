"""Recovery protocol for resuming failed runs.

Provides the API for determining if and how a failed run can be resumed:
- can_resume(run_id) - Check if run can be resumed (failed status + checkpoint exists)
- get_resume_point(run_id) - Get checkpoint info for resuming

The actual resume logic (Orchestrator.resume()) is implemented separately.
"""

import json
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.engine import Row

from elspeth.contracts import PayloadStore, PluginSchema, ResumeCheck, ResumePoint, RowOutcome, RunStatus
from elspeth.core.checkpoint.compatibility import CheckpointCompatibilityValidator
from elspeth.core.checkpoint.manager import CheckpointManager, IncompatibleCheckpointError
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    rows_table,
    runs_table,
    token_outcomes_table,
    tokens_table,
)

if TYPE_CHECKING:
    from elspeth.core.dag import ExecutionGraph

__all__ = [
    "RecoveryManager",
    "ResumeCheck",  # Re-exported from contracts for convenience
    "ResumePoint",  # Re-exported from contracts for convenience
]


class RecoveryManager:
    """Manages recovery of failed runs from checkpoints.

    Recovery protocol:
    1. Check if run can be resumed (failed status + checkpoint exists)
    2. Load checkpoint and aggregation state
    3. Identify unprocessed rows (sequence > checkpoint.sequence)
    4. Resume processing from checkpoint position

    Usage:
        recovery = RecoveryManager(db, checkpoint_manager)

        check = recovery.can_resume(run_id)
        if check.can_resume:
            resume_point = recovery.get_resume_point(run_id)
            # Pass resume_point to Orchestrator.resume()
    """

    def __init__(self, db: LandscapeDB, checkpoint_manager: CheckpointManager) -> None:
        """Initialize with Landscape database and checkpoint manager.

        Args:
            db: LandscapeDB instance for querying run status
            checkpoint_manager: CheckpointManager for loading checkpoints
        """
        self._db = db
        self._checkpoint_manager = checkpoint_manager

    def can_resume(self, run_id: str, graph: "ExecutionGraph") -> ResumeCheck:
        """Check if a run can be resumed.

        A run can be resumed if:
        - It exists in the database
        - Its status is "failed" (not "completed" or "running")
        - At least one checkpoint exists for recovery
        - The checkpoint's upstream topology is compatible with current graph

        Args:
            run_id: The run to check
            graph: The current execution graph to validate against

        Returns:
            ResumeCheck with can_resume=True if resumable,
            or can_resume=False with reason explaining why not.
        """
        run = self._get_run(run_id)
        if run is None:
            return ResumeCheck(can_resume=False, reason=f"Run {run_id} not found")

        if run.status == RunStatus.COMPLETED:
            return ResumeCheck(can_resume=False, reason="Run already completed successfully")

        if run.status == RunStatus.RUNNING:
            return ResumeCheck(can_resume=False, reason="Run is still in progress")

        try:
            checkpoint = self._checkpoint_manager.get_latest_checkpoint(run_id)
        except IncompatibleCheckpointError as e:
            # Return ResumeCheck instead of propagating exception (API contract)
            return ResumeCheck(can_resume=False, reason=str(e))
        if checkpoint is None:
            return ResumeCheck(can_resume=False, reason="No checkpoint found for recovery")

        # Validate topological compatibility
        validator = CheckpointCompatibilityValidator()
        return validator.validate(checkpoint, graph)

    def get_resume_point(self, run_id: str, graph: "ExecutionGraph") -> ResumePoint | None:
        """Get the resume point for a failed run.

        Returns all information needed to resume processing:
        - The checkpoint itself (for audit trail)
        - Token ID to resume from
        - Node ID where processing stopped
        - Sequence number for ordering
        - Deserialized aggregation state (if any)

        Args:
            run_id: The run to get resume point for
            graph: The current execution graph to validate against

        Returns:
            ResumePoint if run can be resumed, None otherwise
        """
        check = self.can_resume(run_id, graph)
        if not check.can_resume:
            return None

        checkpoint = self._checkpoint_manager.get_latest_checkpoint(run_id)
        if checkpoint is None:
            return None

        agg_state = None
        if checkpoint.aggregation_state_json:
            agg_state = json.loads(checkpoint.aggregation_state_json)

        return ResumePoint(
            checkpoint=checkpoint,
            token_id=checkpoint.token_id,
            node_id=checkpoint.node_id,
            sequence_number=checkpoint.sequence_number,
            aggregation_state=agg_state,
        )

    def get_unprocessed_row_data(
        self,
        run_id: str,
        payload_store: PayloadStore,
        *,
        source_schema_class: type[PluginSchema],
    ) -> list[tuple[str, int, dict[str, Any]]]:
        """Get row data for unprocessed rows with type fidelity preservation.

        Retrieves actual row data (not just IDs) for rows that need
        processing during resume. Returns tuples of (row_id, row_index, row_data)
        ordered by row_index for deterministic processing.

        IMPORTANT: Type Fidelity Preservation (REQUIRED)
        -------------------------------------------------
        Payloads are stored via canonical_json(), which normalizes non-JSON types:
        - datetime → ISO string ("2024-01-01T00:00:00+00:00")
        - Decimal → string ("42.50")
        - pandas/numpy scalars → primitives

        On resume, json.loads() returns degraded types (all strings). To restore
        type fidelity, this method REQUIRES source_schema_class to re-validate rows
        through the source's Pydantic schema, which re-coerces strings back to typed values.

        Without schema validation, transforms would receive wrong types (str instead of
        datetime/Decimal), violating the Tier 2 pipeline data trust model from CLAUDE.md.

        Args:
            run_id: The run to get unprocessed rows for
            payload_store: PayloadStore for retrieving row data
            source_schema_class: Pydantic schema class for type restoration (REQUIRED).
                Resume cannot guarantee type fidelity without schema validation.
                The schema must have allow_coercion=True to handle string→typed conversions.

        Returns:
            List of (row_id, row_index, row_data) tuples, ordered by row_index.
            Empty list if run cannot be resumed or all rows were processed.

        Raises:
            ValueError: If row data cannot be retrieved (payload purged or missing),
                or if schema validation fails (indicates data corruption or schema mismatch)
        """
        row_ids = self.get_unprocessed_rows(run_id)
        if not row_ids:
            return []

        result: list[tuple[str, int, dict[str, Any]]] = []

        with self._db.engine.connect() as conn:
            for row_id in row_ids:
                # Get row metadata
                row_result = conn.execute(
                    select(rows_table.c.row_index, rows_table.c.source_data_ref).where(rows_table.c.row_id == row_id)
                ).fetchone()

                if row_result is None:
                    raise ValueError(f"Row {row_id} not found in database")

                row_index = row_result.row_index
                source_data_ref = row_result.source_data_ref

                if source_data_ref is None:
                    raise ValueError(f"Row {row_id} has no source_data_ref - cannot resume without payload")

                # Retrieve from payload store
                try:
                    payload_bytes = payload_store.retrieve(source_data_ref)
                    degraded_data = json.loads(payload_bytes.decode("utf-8"))
                except KeyError:
                    raise ValueError(f"Row {row_id} payload has been purged - cannot resume") from None

                # TYPE FIDELITY RESTORATION:
                # Re-validate through source schema to restore types.
                # This is critical for datetime, Decimal, and other coerced types
                # that canonical_json normalizes to strings.
                # Schema is now REQUIRED - no fallback to degraded types.
                validated = source_schema_class.model_validate(degraded_data)
                row_data = validated.to_row()

                # DEFENSE-IN-DEPTH: Detect silent data loss from empty schemas
                # If source data has fields but restored data is empty, the schema is losing data.
                # This catches bugs like NullSourceSchema (no fields) being used for resume.
                if degraded_data and not row_data:
                    raise ValueError(
                        f"Resume failed for row {row_id}: Schema validation returned empty data "
                        f"but source had {len(degraded_data)} fields. "
                        f"Schema class '{source_schema_class.__name__}' appears to have no fields defined. "
                        f"Cannot resume - this would silently discard all row data. "
                        f"The source plugin's schema must declare fields matching the stored row structure."
                    )

                result.append((row_id, row_index, row_data))

        return result

    def get_unprocessed_rows(self, run_id: str) -> list[str]:
        """Get row IDs that were not processed before the run failed.

        Uses token outcomes to determine which rows need processing:
        - Rows with terminal outcomes (COMPLETED, ROUTED, QUARANTINED, FAILED) are done
        - Rows whose tokens lack terminal outcomes need reprocessing

        This correctly handles multi-sink scenarios where rows are routed to
        different sinks in interleaved order. The previous row_index boundary
        approach would skip rows routed to a failed sink if a later row
        succeeded on a different sink.

        Args:
            run_id: The run to get unprocessed rows for

        Returns:
            List of row_id strings for rows that need processing.
            Empty list if run cannot be resumed or all rows were processed.
        """
        checkpoint = self._checkpoint_manager.get_latest_checkpoint(run_id)
        if checkpoint is None:
            return []

        with self._db.engine.connect() as conn:
            # CORRECT SEMANTICS FOR FORK/AGGREGATION/COALESCE RECOVERY:
            #
            # A row is "complete" when ALL its "leaf" tokens have terminal outcomes.
            # "Leaf" tokens = tokens that are NOT delegation markers.
            #
            # Delegation markers (excluded from completion check):
            # - FORKED: Fork parent, children carry completion status
            # - EXPANDED: Deaggregation parent, expanded children carry status
            #
            # Terminal outcomes (indicate row processing is done):
            # - COMPLETED: Reached output sink successfully
            # - ROUTED: Sent to named sink by gate
            # - QUARANTINED: Failed validation, stored for investigation
            # - FAILED: Processing failed, not recoverable
            # - CONSUMED_IN_BATCH: Absorbed into aggregation (batch recovery handles batch)
            # - COALESCED: Merged in join (merged token carries forward)
            #
            # A row is "incomplete" (needs reprocessing) if ANY of:
            # 1. No tokens at all (never started processing)
            # 2. Any non-delegation token lacks terminal outcome
            # 3. Has tokens but NONE have terminal outcomes (delegation marker only)
            #
            # BUG FIX (P2-recovery-skips-forked-rows):
            # Previous approach: "row has ANY terminal token → complete"
            # Failed: If child A completed but child B crashed, row marked done.
            # Fix: "ALL non-delegation tokens must have terminal outcomes"

            # Subquery: Tokens that are delegation markers (FORKED or EXPANDED)
            # These delegate completion to their children, so exclude from completion check
            delegation_tokens = (
                select(token_outcomes_table.c.token_id)
                .where(token_outcomes_table.c.run_id == run_id)
                .where(
                    token_outcomes_table.c.outcome.in_(
                        [
                            RowOutcome.FORKED.value,
                            RowOutcome.EXPANDED.value,
                        ]
                    )
                )
            ).scalar_subquery()

            # Terminal outcomes that indicate row processing is complete
            # (excludes FORKED and EXPANDED which are delegation markers)
            terminal_outcome_values = [
                RowOutcome.COMPLETED.value,
                RowOutcome.ROUTED.value,
                RowOutcome.QUARANTINED.value,
                RowOutcome.FAILED.value,
                RowOutcome.CONSUMED_IN_BATCH.value,
                RowOutcome.COALESCED.value,
            ]

            # Subquery: Tokens with terminal outcomes
            terminal_tokens = (
                select(token_outcomes_table.c.token_id)
                .where(token_outcomes_table.c.run_id == run_id)
                .where(token_outcomes_table.c.is_terminal == 1)
                .where(token_outcomes_table.c.outcome.in_(terminal_outcome_values))
            ).scalar_subquery()

            # Subquery: Rows that have at least one terminal outcome
            rows_with_terminal = (
                select(tokens_table.c.row_id)
                .distinct()
                .select_from(
                    tokens_table.join(
                        token_outcomes_table,
                        tokens_table.c.token_id == token_outcomes_table.c.token_id,
                    )
                )
                .where(token_outcomes_table.c.run_id == run_id)
                .where(token_outcomes_table.c.is_terminal == 1)
                .where(token_outcomes_table.c.outcome.in_(terminal_outcome_values))
            ).scalar_subquery()

            # Main query: Find incomplete rows
            # Row is incomplete if:
            # - Case 1: No tokens at all
            # - Case 2: Has non-delegation token without terminal outcome
            # - Case 3: Has tokens but none have terminal outcomes (delegation only)
            #
            # NOTE: PostgreSQL requires ORDER BY columns to be in SELECT when using DISTINCT.
            # We select both row_id and row_index, then extract just row_id from results.
            query = (
                select(rows_table.c.row_id, rows_table.c.row_index)
                .select_from(rows_table)
                .outerjoin(
                    tokens_table,
                    rows_table.c.row_id == tokens_table.c.row_id,
                )
                .where(rows_table.c.run_id == run_id)
                .where(
                    # Case 1: No tokens at all
                    (tokens_table.c.token_id.is_(None))
                    |
                    # Case 2: Non-delegation token without terminal outcome
                    ((~tokens_table.c.token_id.in_(delegation_tokens)) & (~tokens_table.c.token_id.in_(terminal_tokens)))
                    |
                    # Case 3: Has tokens but no terminal outcomes (fork parent only)
                    (~rows_table.c.row_id.in_(rows_with_terminal))
                )
                .order_by(rows_table.c.row_index)
                .distinct()
            )

            unprocessed = [row.row_id for row in conn.execute(query).fetchall()]

        return unprocessed

    def _get_run(self, run_id: str) -> Row[Any] | None:
        """Get run metadata from the database.

        Args:
            run_id: The run to fetch

        Returns:
            Row result with run data, or None if not found
        """
        with self._db.engine.connect() as conn:
            result = conn.execute(select(runs_table).where(runs_table.c.run_id == run_id)).fetchone()

        return result
