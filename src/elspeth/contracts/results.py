"""Operation outcomes and results.

These types answer: "What did an operation produce?"

IMPORTANT:
- TransformResult.status uses Literal["success", "error"], NOT an enum
- TransformResult and GateResult KEEP audit fields (input_hash, output_hash, duration_ms)
- ArtifactDescriptor matches architecture schema (artifact_type, content_hash REQUIRED, size_bytes REQUIRED)
- FailureInfo provides type-safe error details for RowResult
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from elspeth.engine.retry import MaxRetriesExceeded

from elspeth.contracts.enums import RowOutcome
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.routing import RoutingAction


@dataclass
class FailureInfo:
    """Type-safe error details for RowResult.

    Captures structured failure information for FAILED outcomes.
    Use factory methods for common error types.

    Fields:
        exception_type: The exception class name (required)
        message: Human-readable error message (required)
        attempts: Number of retry attempts (optional, for retry failures)
        last_error: The underlying error message (optional)
    """

    exception_type: str
    message: str
    attempts: int | None = None
    last_error: str | None = None

    @classmethod
    def from_max_retries_exceeded(cls, e: "MaxRetriesExceeded") -> "FailureInfo":
        """Create FailureInfo from MaxRetriesExceeded exception.

        Args:
            e: The MaxRetriesExceeded exception

        Returns:
            FailureInfo with all retry details
        """
        return cls(
            exception_type="MaxRetriesExceeded",
            message=str(e),
            attempts=e.attempts,
            last_error=str(e.last_error),
        )


@dataclass
class TransformResult:
    """Result of a transform operation.

    Use the factory methods to create instances.

    IMPORTANT: status uses Literal["success", "error"], NOT enum, per architecture.
    Audit fields (input_hash, output_hash, duration_ms) are populated by executors.

    Multi-row output:
    - Single-row: success(row) sets row=row, rows=None
    - Multi-row: success_multi(rows) sets row=None, rows=rows
    - Use is_multi_row property to distinguish
    - Use has_output_data property to check if ANY output exists
    """

    status: Literal["success", "error"]
    row: dict[str, Any] | None
    reason: dict[str, Any] | None
    retryable: bool = False
    rows: list[dict[str, Any]] | None = None

    # Audit fields - set by executor, not by plugin
    input_hash: str | None = field(default=None, repr=False)
    output_hash: str | None = field(default=None, repr=False)
    duration_ms: float | None = field(default=None, repr=False)

    @property
    def is_multi_row(self) -> bool:
        """True if this result contains multiple output rows."""
        return self.rows is not None

    @property
    def has_output_data(self) -> bool:
        """True if this result has any output data (row or rows)."""
        return self.row is not None or self.rows is not None

    @classmethod
    def success(cls, row: dict[str, Any]) -> "TransformResult":
        """Create successful result with single output row."""
        return cls(status="success", row=row, reason=None, rows=None)

    @classmethod
    def success_multi(cls, rows: list[dict[str, Any]]) -> "TransformResult":
        """Create successful result with multiple output rows.

        Args:
            rows: List of output rows (must not be empty)

        Returns:
            TransformResult with status="success", row=None, rows=rows

        Raises:
            ValueError: If rows is empty
        """
        if not rows:
            raise ValueError("success_multi requires at least one row")
        return cls(status="success", row=None, reason=None, rows=rows)

    @classmethod
    def error(
        cls,
        reason: dict[str, Any],
        *,
        retryable: bool = False,
    ) -> "TransformResult":
        """Create error result with reason."""
        return cls(
            status="error",
            row=None,
            reason=reason,
            retryable=retryable,
            rows=None,
        )


@dataclass
class GateResult:
    """Result of a gate evaluation.

    Contains the (possibly modified) row and routing action.
    Audit fields are populated by GateExecutor, not by plugin.
    """

    row: dict[str, Any]
    action: RoutingAction

    # Audit fields - set by executor, not by plugin
    input_hash: str | None = field(default=None, repr=False)
    output_hash: str | None = field(default=None, repr=False)
    duration_ms: float | None = field(default=None, repr=False)


# NOTE: AcceptResult was deleted in aggregation structural cleanup.
# Aggregation is now engine-controlled via batch-aware transforms.
# The engine buffers rows and decides when to flush via TriggerEvaluator.


@dataclass
class RowResult:
    """Final result of processing a row through the pipeline.

    Uses RowOutcome enum. The outcome is derived at query time
    from node_states/routing_events/batch_members, but this type
    is used to communicate the result during processing.

    Fields:
        token: Token identity for this row instance
        final_data: Final row data after processing (may be original if failed early)
        outcome: Terminal state (COMPLETED, FAILED, QUARANTINED, etc.)
        sink_name: For ROUTED outcomes, the destination sink name
        error: For FAILED outcomes, type-safe error details for audit
    """

    token: TokenInfo
    final_data: dict[str, Any]
    outcome: RowOutcome
    sink_name: str | None = None
    error: FailureInfo | None = None

    @property
    def token_id(self) -> str:
        """Token ID for backwards compatibility."""
        return self.token.token_id

    @property
    def row_id(self) -> str:
        """Row ID for backwards compatibility."""
        return self.token.row_id


@dataclass(frozen=True)
class ArtifactDescriptor:
    """Descriptor for an artifact written by a sink.

    Matches architecture artifacts table schema:
    - artifact_type: NOT NULL (matches DB column name)
    - content_hash: NOT NULL (REQUIRED for audit integrity)
    - size_bytes: NOT NULL (REQUIRED for verification)

    Factory methods provide convenient construction for each artifact type.
    """

    artifact_type: Literal["file", "database", "webhook"]
    path_or_uri: str
    content_hash: str  # REQUIRED - audit integrity
    size_bytes: int  # REQUIRED - verification
    metadata: dict[str, object] | None = None

    @classmethod
    def for_file(
        cls,
        path: str,
        content_hash: str,
        size_bytes: int,
    ) -> "ArtifactDescriptor":
        """Create descriptor for file-based artifacts."""
        return cls(
            artifact_type="file",
            path_or_uri=f"file://{path}",
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    @classmethod
    def for_database(
        cls,
        url: str,
        table: str,
        content_hash: str,
        payload_size: int,
        row_count: int,
    ) -> "ArtifactDescriptor":
        """Create descriptor for database artifacts."""
        return cls(
            artifact_type="database",
            path_or_uri=f"db://{table}@{url}",
            content_hash=content_hash,
            size_bytes=payload_size,
            metadata={"table": table, "row_count": row_count},
        )

    @classmethod
    def for_webhook(
        cls,
        url: str,
        content_hash: str,
        request_size: int,
        response_code: int,
    ) -> "ArtifactDescriptor":
        """Create descriptor for webhook artifacts."""
        return cls(
            artifact_type="webhook",
            path_or_uri=f"webhook://{url}",
            content_hash=content_hash,
            size_bytes=request_size,
            metadata={"response_code": response_code},
        )


@dataclass
class SourceRow:
    """Result from source loading - either valid data or quarantined invalid data.

    ALL rows from sources MUST be wrapped in SourceRow:
    - Valid rows: SourceRow.valid(row_dict)
    - Invalid rows: SourceRow.quarantined(row_dict, error, destination)

    This makes source outcomes first-class engine concepts:
    - All rows get proper token_id for lineage
    - Metrics include both valid and quarantine counts
    - Audit trail shows complete source output
    - Quarantine sinks receive invalid data for investigation

    Example usage in a source:
        try:
            validated = schema.model_validate(row)
            yield SourceRow.valid(validated.to_row())
        except ValidationError as e:
            if on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row=row,
                    error=str(e),
                    destination=on_validation_failure,
                )
            # else: don't yield, row is intentionally discarded
    """

    row: dict[str, Any]
    is_quarantined: bool
    quarantine_error: str | None = None
    quarantine_destination: str | None = None

    @classmethod
    def valid(cls, row: dict[str, Any]) -> "SourceRow":
        """Create a valid row result.

        Args:
            row: The validated row data to pass to the engine.
        """
        return cls(row=row, is_quarantined=False)

    @classmethod
    def quarantined(
        cls,
        row: dict[str, Any],
        error: str,
        destination: str,
    ) -> "SourceRow":
        """Create a quarantined row result.

        Args:
            row: The original row data (before validation)
            error: The validation error message
            destination: The sink name to route this row to
        """
        return cls(
            row=row,
            is_quarantined=True,
            quarantine_error=error,
            quarantine_destination=destination,
        )
