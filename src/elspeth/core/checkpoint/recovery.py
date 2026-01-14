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

from elspeth.core.checkpoint.manager import CheckpointManager
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.models import Checkpoint
from elspeth.core.landscape.schema import rows_table, runs_table


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

        can_resume, reason = recovery.can_resume(run_id)
        if can_resume:
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

    def can_resume(self, run_id: str) -> tuple[bool, str | None]:
        """Check if a run can be resumed.

        A run can be resumed if:
        - It exists in the database
        - Its status is "failed" (not "completed" or "running")
        - At least one checkpoint exists for recovery

        Args:
            run_id: The run to check

        Returns:
            Tuple of (can_resume, reason_if_not).
            If can_resume is True, reason is None.
            If can_resume is False, reason explains why.
        """
        run = self._get_run(run_id)
        if run is None:
            return False, f"Run {run_id} not found"

        if run.status == "completed":
            return False, "Run already completed successfully"

        if run.status == "running":
            return False, "Run is still in progress"

        checkpoint = self._checkpoint_manager.get_latest_checkpoint(run_id)
        if checkpoint is None:
            return False, "No checkpoint found for recovery"

        return True, None

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
        can_resume, _ = self.can_resume(run_id)
        if not can_resume:
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

    def get_unprocessed_rows(self, run_id: str) -> list[str]:
        """Get row IDs that were not processed before the run failed.

        Returns rows with row_index greater than the checkpoint's sequence_number,
        representing rows that still need processing after recovery.

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
            result = conn.execute(
                select(rows_table.c.row_id)
                .where(rows_table.c.run_id == run_id)
                .where(rows_table.c.row_index > checkpoint.sequence_number)
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
            result = conn.execute(
                select(runs_table).where(runs_table.c.run_id == run_id)
            ).fetchone()

        return result
