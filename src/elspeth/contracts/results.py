"""Operation outcomes and results.

These types answer: "What did an operation produce?"

IMPORTANT:
- TransformResult.status uses Literal["success", "error"], NOT an enum
- TransformResult and GateResult KEEP audit fields (input_hash, output_hash, duration_ms)
- ArtifactDescriptor matches architecture schema (artifact_type, content_hash REQUIRED, size_bytes REQUIRED)
- FailureInfo provides type-safe error details for RowResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from elspeth.contracts.url import SanitizedDatabaseUrl, SanitizedWebhookUrl

if TYPE_CHECKING:
    from elspeth.engine.retry import MaxRetriesExceeded

from elspeth.contracts.enums import RowOutcome
from elspeth.contracts.errors import TransformErrorReason, TransformSuccessReason
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.routing import RoutingAction


@dataclass
class ExceptionResult:
    """Wrapper for exceptions that should propagate through async pattern.

    When a worker thread encounters an uncaught exception (plugin bug),
    it wraps the exception in this container. The waiter then re-raises
    the original exception in the orchestrator thread, ensuring plugin
    bugs crash the pipeline as intended.

    Used by:
    - engine/batch_adapter.py: Wraps exceptions in worker threads
    - plugins/batching/mixin.py: Creates ExceptionResult on worker failure
    - plugins/batching/ports.py: Type hint in BatchOutputPort protocol
    """

    exception: BaseException
    traceback: str


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
    def from_max_retries_exceeded(cls, e: MaxRetriesExceeded) -> FailureInfo:
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
    reason: TransformErrorReason | None
    retryable: bool = False
    rows: list[dict[str, Any]] | None = None

    # Success metadata - REQUIRED for success results, None for error results
    # Invariant: status="success" implies success_reason is not None
    success_reason: TransformSuccessReason | None = None

    # Audit fields - set by executor, not by plugin
    input_hash: str | None = field(default=None, repr=False)
    output_hash: str | None = field(default=None, repr=False)
    duration_ms: float | None = field(default=None, repr=False)

    # Context snapshot for audit trail (optional)
    # Contains operational metadata like pool stats, ordering info
    # P3-2026-02-02: Enables pool metadata to flow to context_after_json
    context_after: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate invariants - success results MUST have success_reason."""
        if self.status == "success" and self.success_reason is None:
            raise ValueError(
                "TransformResult with status='success' MUST provide success_reason. "
                "Use TransformResult.success(row, success_reason={'action': '...'}) "
                "to create success results. Missing success_reason is a plugin bug."
            )

    @property
    def is_multi_row(self) -> bool:
        """True if this result contains multiple output rows."""
        return self.rows is not None

    @property
    def has_output_data(self) -> bool:
        """True if this result has any output data (row or rows)."""
        return self.row is not None or self.rows is not None

    @classmethod
    def success(
        cls,
        row: dict[str, Any],
        *,
        success_reason: TransformSuccessReason,
        context_after: dict[str, Any] | None = None,
    ) -> TransformResult:
        """Create successful result with single output row.

        Args:
            row: The transformed row data
            success_reason: REQUIRED metadata about what the transform did.
                           Must include at least 'action' field.
                           See TransformSuccessReason for available fields.
            context_after: Optional operational metadata for audit trail
                          (e.g., pool stats, ordering info).

        Returns:
            TransformResult with status="success" and the provided row.

        Example:
            return TransformResult.success(
                row,
                success_reason={"action": "processed", "fields_modified": ["amount"]}
            )
        """
        return cls(
            status="success",
            row=row,
            reason=None,
            rows=None,
            success_reason=success_reason,
            context_after=context_after,
        )

    @classmethod
    def success_multi(
        cls,
        rows: list[dict[str, Any]],
        *,
        success_reason: TransformSuccessReason,
        context_after: dict[str, Any] | None = None,
    ) -> TransformResult:
        """Create successful result with multiple output rows.

        Args:
            rows: List of output rows (must not be empty)
            success_reason: REQUIRED metadata about what the transform did.
                           Must include at least 'action' field.
                           See TransformSuccessReason for available fields.
            context_after: Optional operational metadata for audit trail
                          (e.g., pool stats, ordering info).

        Returns:
            TransformResult with status="success", row=None, rows=rows

        Raises:
            ValueError: If rows is empty

        Example:
            return TransformResult.success_multi(
                rows,
                success_reason={"action": "split", "fields_added": ["row_index"]}
            )
        """
        if not rows:
            raise ValueError("success_multi requires at least one row")
        return cls(
            status="success",
            row=None,
            reason=None,
            rows=rows,
            success_reason=success_reason,
            context_after=context_after,
        )

    @classmethod
    def error(
        cls,
        reason: TransformErrorReason,
        *,
        retryable: bool = False,
        context_after: dict[str, Any] | None = None,
    ) -> TransformResult:
        """Create error result with structured reason.

        Args:
            reason: Error details with required 'reason' field from
                    TransformErrorCategory (compile-time validated).
                    See TransformErrorReason for all available context fields.
            retryable: Whether the error is transient and should be retried.
            context_after: Optional operational metadata for audit trail
                          (e.g., pool stats from partial execution).

        Returns:
            TransformResult with status="error" and the provided reason.
        """
        return cls(
            status="error",
            row=None,
            reason=reason,
            retryable=retryable,
            rows=None,
            context_after=context_after,
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

    Uses RowOutcome enum, which is explicitly recorded in the token_outcomes
    table (AUD-001) at determination time for complete audit traceability.

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
    ) -> ArtifactDescriptor:
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
        url: SanitizedDatabaseUrl,
        table: str,
        content_hash: str,
        payload_size: int,
        row_count: int,
    ) -> ArtifactDescriptor:
        """Create descriptor for database artifacts.

        URL must be pre-sanitized using SanitizedDatabaseUrl.from_raw_url().
        This ensures credentials are never stored in the audit trail.
        """
        # Type safety: enforce SanitizedDatabaseUrl, not duck-typed objects
        if not isinstance(url, SanitizedDatabaseUrl):
            raise TypeError(
                "url must be a SanitizedDatabaseUrl instance. Use SanitizedDatabaseUrl.from_raw_url(url) to sanitize raw database URLs."
            )

        metadata: dict[str, object] = {"table": table, "row_count": row_count}
        if url.fingerprint:
            metadata["url_fingerprint"] = url.fingerprint

        return cls(
            artifact_type="database",
            path_or_uri=f"db://{table}@{url.sanitized_url}",
            content_hash=content_hash,
            size_bytes=payload_size,
            metadata=metadata,
        )

    @classmethod
    def for_webhook(
        cls,
        url: SanitizedWebhookUrl,
        content_hash: str,
        request_size: int,
        response_code: int,
    ) -> ArtifactDescriptor:
        """Create descriptor for webhook artifacts.

        URL must be pre-sanitized using SanitizedWebhookUrl.from_raw_url().
        This ensures tokens are never stored in the audit trail.
        """
        # Type safety: enforce SanitizedWebhookUrl, not duck-typed objects
        if not isinstance(url, SanitizedWebhookUrl):
            raise TypeError(
                "url must be a SanitizedWebhookUrl instance. Use SanitizedWebhookUrl.from_raw_url(url) to sanitize raw webhook URLs."
            )

        metadata: dict[str, object] = {"response_code": response_code}
        if url.fingerprint:
            metadata["url_fingerprint"] = url.fingerprint

        return cls(
            artifact_type="webhook",
            path_or_uri=f"webhook://{url.sanitized_url}",
            content_hash=content_hash,
            size_bytes=request_size,
            metadata=metadata,
        )


@dataclass
class SourceRow:
    """Result from source loading - either valid data or quarantined invalid data.

    ALL rows from sources MUST be wrapped in SourceRow:
    - Valid rows: SourceRow.valid(row_dict)
    - Invalid rows: SourceRow.quarantined(row_data, error, destination)

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

    # Note: row is Any (not dict) because quarantined rows from external data
    # may not be dicts (e.g., JSON arrays containing primitives like numbers).
    # Valid rows are always dicts (they passed schema validation).
    row: Any
    is_quarantined: bool
    quarantine_error: str | None = None
    quarantine_destination: str | None = None

    @classmethod
    def valid(cls, row: dict[str, Any]) -> SourceRow:
        """Create a valid row result.

        Args:
            row: The validated row data to pass to the engine.
        """
        return cls(row=row, is_quarantined=False)

    @classmethod
    def quarantined(
        cls,
        row: Any,
        error: str,
        destination: str,
    ) -> SourceRow:
        """Create a quarantined row result.

        Args:
            row: The original row data (before validation). May be non-dict
                 for malformed external data (e.g., JSON primitives).
            error: The validation error message
            destination: The sink name to route this row to
        """
        return cls(
            row=row,
            is_quarantined=True,
            quarantine_error=error,
            quarantine_destination=destination,
        )
