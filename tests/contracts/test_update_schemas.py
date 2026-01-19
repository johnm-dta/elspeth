"""Tests for update schema TypedDicts."""

from datetime import UTC, datetime


class TestUpdateSchemas:
    """Verify update TypedDicts are importable and usable."""

    def test_export_status_update_importable(self) -> None:
        """ExportStatusUpdate should be importable from contracts."""
        from elspeth.contracts import ExportStatus, ExportStatusUpdate

        # Should accept all valid fields
        update: ExportStatusUpdate = {
            "export_status": ExportStatus.COMPLETED,
            "exported_at": datetime.now(UTC),
        }
        assert "export_status" in update

    def test_batch_status_update_importable(self) -> None:
        """BatchStatusUpdate should be importable from contracts."""
        from elspeth.contracts import BatchStatus, BatchStatusUpdate

        update: BatchStatusUpdate = {
            "status": BatchStatus.EXECUTING,
            "trigger_reason": "count_reached",
        }
        assert "status" in update

    def test_export_status_update_partial(self) -> None:
        """ExportStatusUpdate should allow partial updates."""
        from elspeth.contracts import ExportStatus, ExportStatusUpdate

        # Only status field
        update: ExportStatusUpdate = {"export_status": ExportStatus.PENDING}
        assert len(update) == 1

    def test_batch_status_update_with_state_id(self) -> None:
        """BatchStatusUpdate should accept aggregation_state_id for aggregation linking."""
        from elspeth.contracts import BatchStatus, BatchStatusUpdate

        update: BatchStatusUpdate = {
            "status": BatchStatus.COMPLETED,
            "completed_at": datetime.now(UTC),
            "aggregation_state_id": "state-123",
        }
        assert "aggregation_state_id" in update
