# src/elspeth/plugins/llm/batch_errors.py
"""Batch processing control flow errors.

These errors are NOT failures - they're control flow signals that tell
the engine to schedule retry checks later. The two-phase checkpoint
approach uses these to pause processing while batches complete.
"""

from __future__ import annotations

from typing import Any


class BatchPendingError(Exception):
    """Raised when batch is submitted but not yet complete.

    This is NOT an error condition - it's a control flow signal
    telling the engine to schedule a retry check later.

    The exception carries the checkpoint state so the caller can persist
    it and restore it when scheduling a retry. This enables crash recovery
    and correct resume behavior.

    Attributes:
        batch_id: Azure batch job ID
        status: Current batch status (e.g., "submitted", "in_progress")
        check_after_seconds: When to check again (default 300s = 5 min)
        checkpoint: Checkpoint data to persist for retry (batch_id, row_mapping, etc.)
        node_id: Transform node ID that raised this (for checkpoint keying)

    Example:
        # Phase 1: Submit batch
        batch_id = client.batches.create(...)
        checkpoint_data = {"batch_id": batch_id, "row_mapping": {...}}
        ctx.update_checkpoint(checkpoint_data)
        raise BatchPendingError(
            batch_id, "submitted",
            check_after_seconds=300,
            checkpoint=checkpoint_data,
            node_id=self.node_id,
        )

        # Caller catches, persists checkpoint, schedules retry

        # Phase 2: Resume and check (caller passes checkpoint back via orchestrator)
        checkpoint = ctx.get_checkpoint()
        if checkpoint.get("batch_id"):
            status = client.batches.retrieve(batch_id).status
            if status == "in_progress":
                raise BatchPendingError(batch_id, "in_progress", checkpoint=checkpoint)
            elif status == "completed":
                # Download results and return
    """

    def __init__(
        self,
        batch_id: str,
        status: str,
        *,
        check_after_seconds: int = 300,
        checkpoint: dict[str, Any] | None = None,
        node_id: str | None = None,
    ) -> None:
        """Initialize BatchPendingError.

        Args:
            batch_id: Azure batch job ID
            status: Current batch status
            check_after_seconds: Seconds until next check (default 300)
            checkpoint: Checkpoint data for retry (caller should persist this)
            node_id: Transform node ID (for checkpoint keying)
        """
        self.batch_id = batch_id
        self.status = status
        self.check_after_seconds = check_after_seconds
        self.checkpoint = checkpoint
        self.node_id = node_id
        super().__init__(f"Batch {batch_id} is {status}, check after {check_after_seconds}s")
