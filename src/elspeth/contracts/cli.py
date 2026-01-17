# src/elspeth/contracts/cli.py
"""CLI-related type contracts."""

from typing import TypedDict


class ExecutionResult(TypedDict, total=False):
    """Result from pipeline execution.

    Returned by _execute_pipeline() in cli.py.

    Required fields (always present in practice):
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
    rows_succeeded: int
    rows_failed: int
    duration_seconds: float
