# src/elspeth/plugins/llm/batch_errors.py
"""Batch processing control flow errors.

These errors are NOT failures - they're control flow signals that tell
the engine to schedule retry checks later. The two-phase checkpoint
approach uses these to pause processing while batches complete.
"""

from __future__ import annotations


class BatchPendingError(Exception):
    """Raised when batch is submitted but not yet complete.

    This is NOT an error condition - it's a control flow signal
    telling the engine to schedule a retry check later.

    The engine catches BatchPendingError, checkpoints the batch state,
    and schedules a retry after check_after_seconds.

    Attributes:
        batch_id: Azure batch job ID
        status: Current batch status (e.g., "submitted", "in_progress")
        check_after_seconds: When to check again (default 300s = 5 min)

    Example:
        # Phase 1: Submit batch
        batch_id = client.batches.create(...)
        ctx.update_checkpoint({"batch_id": batch_id})
        raise BatchPendingError(batch_id, "submitted", check_after_seconds=300)

        # Engine catches this, checkpoints, schedules retry

        # Phase 2: Resume and check
        checkpoint = ctx.get_checkpoint()
        if checkpoint.get("batch_id"):
            status = client.batches.retrieve(batch_id).status
            if status == "in_progress":
                raise BatchPendingError(batch_id, "in_progress")
            elif status == "completed":
                # Download results and return
    """

    def __init__(
        self,
        batch_id: str,
        status: str,
        *,
        check_after_seconds: int = 300,
    ) -> None:
        """Initialize BatchPendingError.

        Args:
            batch_id: Azure batch job ID
            status: Current batch status
            check_after_seconds: Seconds until next check (default 300)
        """
        self.batch_id = batch_id
        self.status = status
        self.check_after_seconds = check_after_seconds
        super().__init__(
            f"Batch {batch_id} is {status}, check after {check_after_seconds}s"
        )
