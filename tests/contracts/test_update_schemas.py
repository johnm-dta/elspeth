"""Tests for update schema TypedDicts."""

from datetime import UTC, datetime
from typing import get_type_hints


class TestUpdateSchemas:
    """Verify update TypedDicts are importable and usable."""

    def test_export_status_update_schema(self) -> None:
        """ExportStatusUpdate has correct schema definition."""
        from elspeth.contracts import ExportStatus
        from elspeth.contracts.audit import ExportStatusUpdate

        assert ExportStatusUpdate.__required_keys__ == set()
        assert ExportStatusUpdate.__optional_keys__ == {
            "export_status",
            "exported_at",
            "export_error",
            "export_format",
            "export_sink",
        }
        hints = get_type_hints(ExportStatusUpdate)
        assert hints["export_status"] is ExportStatus

    def test_export_status_update_importable(self) -> None:
        """ExportStatusUpdate should be importable from contracts."""
        from elspeth.contracts import ExportStatus, ExportStatusUpdate

        update: ExportStatusUpdate = {
            "export_status": ExportStatus.COMPLETED,
            "exported_at": datetime.now(UTC),
        }
        assert update["export_status"] == ExportStatus.COMPLETED

    def test_batch_status_update_schema(self) -> None:
        """BatchStatusUpdate has correct schema definition."""
        from elspeth.contracts import BatchStatus
        from elspeth.contracts.audit import BatchStatusUpdate

        assert BatchStatusUpdate.__required_keys__ == set()
        assert BatchStatusUpdate.__optional_keys__ == {
            "status",
            "completed_at",
            "trigger_reason",
            "aggregation_state_id",
        }
        hints = get_type_hints(BatchStatusUpdate)
        assert hints["status"] is BatchStatus

    def test_batch_status_update_importable(self) -> None:
        """BatchStatusUpdate should be importable from contracts."""
        from elspeth.contracts import BatchStatus, BatchStatusUpdate

        update: BatchStatusUpdate = {
            "status": BatchStatus.EXECUTING,
            "trigger_reason": "count_reached",
        }
        assert update["status"] == BatchStatus.EXECUTING

    def test_export_status_update_partial(self) -> None:
        """ExportStatusUpdate should allow partial updates."""
        from elspeth.contracts import ExportStatus, ExportStatusUpdate

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
        assert update["aggregation_state_id"] == "state-123"
