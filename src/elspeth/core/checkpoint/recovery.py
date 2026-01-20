"""Recovery protocol for resuming failed runs.

Provides the API for determining if and how a failed run can be resumed:
- can_resume(run_id) - Check if run can be resumed (failed status + checkpoint exists)
- get_resume_point(run_id) - Get checkpoint info for resuming

The actual resume logic (Orchestrator.resume()) is implemented separately.
"""

import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Row

from elspeth.contracts import Checkpoint, RunStatus
from elspeth.core.checkpoint.manager import CheckpointManager
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import rows_table, runs_table, tokens_table
from elspeth.core.payload_store import PayloadStore


@dataclass(frozen=True)
class ResumeCheck:
    """Result of checking if a run can be resumed.

    Replaces tuple[bool, str | None] return type from can_resume().
    """

    can_resume: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.can_resume and self.reason is not None:
            raise ValueError("can_resume=True should not have a reason")
        if not self.can_resume and self.reason is None:
            raise ValueError("can_resume=False must have a reason explaining why")


@dataclass
class ResumePoint:
    """Information needed to resume a run.

    Contains all the data needed by Orchestrator.resume() to continue
    processing from where a failed run left off.
    """

    checkpoint: Checkpoint
    token_id: str
    node_id: str
    sequence_number: int
    aggregation_state: dict[str, Any] | None


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

    def can_resume(self, run_id: str) -> ResumeCheck:
        """Check if a run can be resumed.

        A run can be resumed if:
        - It exists in the database
        - Its status is "failed" (not "completed" or "running")
        - At least one checkpoint exists for recovery

        Args:
            run_id: The run to check

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

        checkpoint = self._checkpoint_manager.get_latest_checkpoint(run_id)
        if checkpoint is None:
            return ResumeCheck(can_resume=False, reason="No checkpoint found for recovery")

        return ResumeCheck(can_resume=True)

    def get_resume_point(self, run_id: str) -> ResumePoint | None:
        """Get the resume point for a failed run.

        Returns all information needed to resume processing:
        - The checkpoint itself (for audit trail)
        - Token ID to resume from
        - Node ID where processing stopped
        - Sequence number for ordering
        - Deserialized aggregation state (if any)

        Args:
            run_id: The run to get resume point for

        Returns:
            ResumePoint if run can be resumed, None otherwise
        """
        check = self.can_resume(run_id)
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
    ) -> list[tuple[str, int, dict[str, Any]]]:
        """Get row data for unprocessed rows.

        Retrieves actual row data (not just IDs) for rows that need
        processing during resume. Returns tuples of (row_id, row_index, row_data)
        ordered by row_index for deterministic processing.

        Args:
            run_id: The run to get unprocessed rows for
            payload_store: PayloadStore for retrieving row data

        Returns:
            List of (row_id, row_index, row_data) tuples, ordered by row_index.
            Empty list if run cannot be resumed or all rows were processed.

        Raises:
            ValueError: If row data cannot be retrieved (payload purged or missing)
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
                    row_data = json.loads(payload_bytes.decode("utf-8"))
                except KeyError:
                    raise ValueError(f"Row {row_id} payload has been purged - cannot resume") from None

                result.append((row_id, row_index, row_data))

        return result

    def get_unprocessed_rows(self, run_id: str) -> list[str]:
        """Get row IDs that were not processed before the run failed.

        Derives the row boundary from token lineage:
        checkpoint.token_id -> tokens.row_id -> rows.row_index

        This is correct even when sequence_number != row_index (e.g., forks
        where one row produces multiple tokens, or failures where sequence
        doesn't advance).

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
            # Step 1: Find the row_index of the checkpointed token's source row
            # Join: checkpoint.token_id -> tokens.row_id -> rows.row_index
            checkpointed_row_index_query = (
                select(rows_table.c.row_index)
                .select_from(
                    tokens_table.join(
                        rows_table,
                        tokens_table.c.row_id == rows_table.c.row_id,
                    )
                )
                .where(tokens_table.c.token_id == checkpoint.token_id)
            )
            checkpointed_row_result = conn.execute(checkpointed_row_index_query).fetchone()

            if checkpointed_row_result is None:
                raise RuntimeError(
                    f"Checkpoint references non-existent token: {checkpoint.token_id}. "
                    "This indicates database corruption or a bug in checkpoint creation."
                )

            checkpointed_row_index = checkpointed_row_result.row_index

            # Step 2: Find all rows with row_index > checkpointed_row_index
            result = conn.execute(
                select(rows_table.c.row_id)
                .where(rows_table.c.run_id == run_id)
                .where(rows_table.c.row_index > checkpointed_row_index)
                .order_by(rows_table.c.row_index)
            ).fetchall()

        return [row.row_id for row in result]

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
