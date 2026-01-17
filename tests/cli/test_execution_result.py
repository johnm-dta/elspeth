"""Tests for ExecutionResult TypedDict."""


class TestExecutionResult:
    """Verify ExecutionResult TypedDict works correctly."""

    def test_execution_result_importable(self) -> None:
        """ExecutionResult should be importable from contracts."""
        from elspeth.contracts import ExecutionResult

        result: ExecutionResult = {
            "run_id": "run-123",
            "status": "completed",
            "rows_processed": 100,
        }
        assert result["run_id"] == "run-123"

    def test_execution_result_full(self) -> None:
        """ExecutionResult should accept all fields."""
        from elspeth.contracts import ExecutionResult

        result: ExecutionResult = {
            "run_id": "run-456",
            "status": "completed",
            "rows_processed": 1000,
            "rows_succeeded": 990,
            "rows_failed": 10,
            "duration_seconds": 45.5,
        }
        assert result["rows_succeeded"] == 990
