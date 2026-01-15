"""Tests for RunStatus enum."""

from elspeth.engine.orchestrator import RunResult, RunStatus


class TestRunStatus:
    """Tests for RunStatus enum."""

    def test_status_values(self) -> None:
        """RunStatus should have expected values."""
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.FAILED.value == "failed"

    def test_run_result_uses_enum(self) -> None:
        """RunResult.status should be RunStatus, not str."""
        result = RunResult(
            run_id="test",
            status=RunStatus.COMPLETED,
            rows_processed=10,
            rows_succeeded=10,
            rows_failed=0,
            rows_routed=0,
        )
        assert isinstance(result.status, RunStatus)
        assert result.status == RunStatus.COMPLETED
