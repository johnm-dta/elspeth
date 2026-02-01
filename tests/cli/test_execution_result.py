"""Tests for ExecutionResult TypedDict.

Validates the ExecutionResult contract including required/optional keys.
"""

from typing import get_type_hints

from elspeth.contracts import ExecutionResult, RunStatus


class TestExecutionResult:
    """Verify ExecutionResult TypedDict works correctly."""

    def test_execution_result_importable(self) -> None:
        """ExecutionResult should be importable from contracts."""
        result: ExecutionResult = {
            "run_id": "run-123",
            "status": "completed",
            "rows_processed": 100,
        }
        # Assert all required fields
        assert result["run_id"] == "run-123"
        assert result["status"] == "completed"
        assert result["rows_processed"] == 100

    def test_execution_result_full(self) -> None:
        """ExecutionResult should accept all fields."""
        result: ExecutionResult = {
            "run_id": "run-456",
            "status": "completed",
            "rows_processed": 1000,
            "rows_succeeded": 990,
            "rows_failed": 10,
            "duration_seconds": 45.5,
        }
        # Assert ALL fields are set correctly
        assert result["run_id"] == "run-456"
        assert result["status"] == "completed"
        assert result["rows_processed"] == 1000
        assert result["rows_succeeded"] == 990
        assert result["rows_failed"] == 10
        assert result["duration_seconds"] == 45.5


class TestExecutionResultContract:
    """Verify ExecutionResult TypedDict contract is locked."""

    def test_required_keys_are_locked(self) -> None:
        """ExecutionResult must have exactly these required keys."""
        expected_required = {"run_id", "status", "rows_processed"}
        actual_required = ExecutionResult.__required_keys__
        assert actual_required == expected_required, f"Required keys changed! Expected {expected_required}, got {actual_required}"

    def test_optional_keys_are_locked(self) -> None:
        """ExecutionResult must have exactly these optional keys."""
        expected_optional = {"rows_succeeded", "rows_failed", "duration_seconds"}
        actual_optional = ExecutionResult.__optional_keys__
        assert actual_optional == expected_optional, f"Optional keys changed! Expected {expected_optional}, got {actual_optional}"

    def test_field_types_are_correct(self) -> None:
        """ExecutionResult field types must match contract."""
        hints = get_type_hints(ExecutionResult)
        # Required fields
        assert hints["run_id"] is str
        assert hints["status"] is RunStatus
        assert hints["rows_processed"] is int
        # Optional fields
        assert hints["rows_succeeded"] is int
        assert hints["rows_failed"] is int
        assert hints["duration_seconds"] is float

    def test_no_extra_keys_allowed_by_contract(self) -> None:
        """ExecutionResult contract is complete - all keys are accounted for."""
        all_keys = ExecutionResult.__required_keys__ | ExecutionResult.__optional_keys__
        expected_all = {
            "run_id",
            "status",
            "rows_processed",
            "rows_succeeded",
            "rows_failed",
            "duration_seconds",
        }
        assert all_keys == expected_all, f"ExecutionResult keys changed! Expected {expected_all}, got {all_keys}"


class TestExecutionResultEdgeCases:
    """Edge case tests for ExecutionResult."""

    def test_minimal_required_fields_only(self) -> None:
        """ExecutionResult with only required fields is valid."""
        result: ExecutionResult = {
            "run_id": "minimal-run",
            "status": "completed",
            "rows_processed": 0,
        }
        assert result["run_id"] == "minimal-run"
        assert result["rows_processed"] == 0

    def test_failed_status_result(self) -> None:
        """ExecutionResult can represent failed runs."""
        result: ExecutionResult = {
            "run_id": "failed-run",
            "status": "failed",
            "rows_processed": 50,
            "rows_succeeded": 45,
            "rows_failed": 5,
        }
        assert result["status"] == "failed"
        assert result["rows_failed"] == 5

    def test_zero_duration_is_valid(self) -> None:
        """Zero duration is valid (fast pipeline)."""
        result: ExecutionResult = {
            "run_id": "fast-run",
            "status": "completed",
            "rows_processed": 1,
            "duration_seconds": 0.0,
        }
        assert result["duration_seconds"] == 0.0
