"""RetrievalProvider protocol and RetrievalError exception."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from elspeth.contracts.errors import PluginRetryableError
from elspeth.contracts.probes import CollectionReadinessResult
from elspeth.plugins.infrastructure.clients.retrieval.types import RetrievalChunk


class RetrievalError(PluginRetryableError):
    """Base exception for retrieval provider errors.

    Raised by providers for transient failures (retryable=True) to trigger
    engine retry, or for permanent failures (retryable=False) to be caught
    by the transform and converted to TransformResult.error().

    Attributes:
        retryable: Whether the error is transient and should be retried.
        status_code: HTTP status code if applicable (for audit context).
    """

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None) -> None:
        super().__init__(message, retryable=retryable, status_code=status_code)


@runtime_checkable
class RetrievalProvider(Protocol):
    """Search backend interface for RAG retrieval.

    Implementations handle search execution, score normalization, and
    resource lifecycle. The protocol is deliberately minimal — no
    provider-specific query objects leak into the transform.
    """

    last_skipped_count: int
    last_skipped_reasons: list[dict[str, Any]]

    def search(
        self,
        query: str,
        top_k: int,
        min_score: float,
        *,
        state_id: str,
        token_id: str | None,
    ) -> list[RetrievalChunk]:
        """Execute a search query and return ranked results.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance score threshold (0.0-1.0).
            state_id: Per-row audit identity for AuditedHTTPClient scoping.
            token_id: Pipeline token identity for audit correlation.

        Returns:
            List of RetrievalChunk, ordered by descending relevance score.
            May be empty if no results meet the min_score threshold.

        Raises:
            RetrievalError: On search failures (retryable or permanent).
        """
        ...

    def check_readiness(self) -> CollectionReadinessResult:
        """Check that the target collection exists and has documents.

        Single-attempt, no retry. Called during on_start() — transient
        failures crash the pipeline startup.
        """
        ...

    def close(self) -> None:
        """Release provider resources (connections, clients)."""
        ...
