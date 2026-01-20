# src/elspeth/contracts/cli.py
"""CLI-related type contracts."""

from typing import NotRequired, TypedDict


class ExecutionResult(TypedDict):
    """Result from pipeline execution.

    Returned by _execute_pipeline() in cli.py.

    Required fields:
        run_id: Unique identifier for this pipeline run.
        status: Execution status (e.g., "completed", "failed").
        rows_processed: Total number of rows processed.

    Optional fields (may be added for detailed reporting):
        rows_succeeded: Number of rows that completed successfully.
        rows_failed: Number of rows that failed processing.
        duration_seconds: Total execution time in seconds.
    """

    run_id: str
    status: str
    rows_processed: int
    rows_succeeded: NotRequired[int]
    rows_failed: NotRequired[int]
    duration_seconds: NotRequired[float]
