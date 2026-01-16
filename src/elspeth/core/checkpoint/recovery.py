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
from elspeth.core.landscape.schema import rows_table, runs_table


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
            return ResumeCheck(
                can_resume=False, reason="Run already completed successfully"
            )

        if run.status == RunStatus.RUNNING:
            return ResumeCheck(can_resume=False, reason="Run is still in progress")

        checkpoint = self._checkpoint_manager.get_latest_checkpoint(run_id)
        if checkpoint is None:
            return ResumeCheck(
                can_resume=False, reason="No checkpoint found for recovery"
            )

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
