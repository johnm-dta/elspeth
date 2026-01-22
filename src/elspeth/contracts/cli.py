# src/elspeth/contracts/cli.py
"""CLI-related type contracts."""

from dataclasses import dataclass
from typing import NotRequired, TypedDict


@dataclass(frozen=True)
class ProgressEvent:
    """Progress event emitted during pipeline execution.

    Emitted every N rows (default 100) to provide visibility into long-running
    pipelines. The CLI subscribes to these events and renders progress output.

    Attributes:
        rows_processed: Total rows processed so far.
        rows_succeeded: Rows that completed successfully.
        rows_failed: Rows that failed processing.
        rows_quarantined: Rows that were quarantined for investigation.
        elapsed_seconds: Time elapsed since run started.
    """

    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    rows_quarantined: int
    elapsed_seconds: float


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
