"""Operation outcomes and results.

These types answer: "What did an operation produce?"

IMPORTANT:
- TransformResult.status uses Literal["success", "error"], NOT an enum
- TransformResult and GateResult KEEP audit fields (input_hash, output_hash, duration_ms)
- ArtifactDescriptor matches architecture schema (artifact_type, content_hash REQUIRED, size_bytes REQUIRED)
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from elspeth.contracts.enums import RowOutcome
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.routing import RoutingAction


@dataclass
class TransformResult:
    """Result of a transform operation.

    Use the factory methods to create instances.

    IMPORTANT: status uses Literal["success", "error"], NOT enum, per architecture.
    Audit fields (input_hash, output_hash, duration_ms) are populated by executors.
    """

    status: Literal["success", "error"]
    row: dict[str, Any] | None
    reason: dict[str, Any] | None
    retryable: bool = False

    # Audit fields - set by executor, not by plugin
    input_hash: str | None = field(default=None, repr=False)
    output_hash: str | None = field(default=None, repr=False)
    duration_ms: float | None = field(default=None, repr=False)

    @classmethod
    def success(cls, row: dict[str, Any]) -> "TransformResult":
        """Create successful result with output row."""
        return cls(status="success", row=row, reason=None)

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


@dataclass
class AcceptResult:
    """Result of aggregation accept check.

    Indicates whether the row was accepted into a batch.
    """

    accepted: bool
    trigger: bool
    batch_id: str | None = field(default=None, repr=False)


@dataclass
class RowResult:
    """Final result of processing a row through the pipeline.

    Uses RowOutcome enum. The outcome is derived at query time
    from node_states/routing_events/batch_members, but this type
    is used to communicate the result during processing.
    """

    token: TokenInfo
    final_data: dict[str, Any]
    outcome: RowOutcome
    sink_name: str | None = None

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
