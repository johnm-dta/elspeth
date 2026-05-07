"""Retrieval type dataclasses.

These types represent the output of a retrieval provider search operation.
RetrievalChunk enforces two invariants at construction time:
1. Score is normalized to [0.0, 1.0]
2. Metadata is JSON-serializable
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True)
class RetrievalChunk:
    """A single retrieved document chunk.

    Attributes:
        content: The retrieved text content.
        score: Relevance score, normalized to 0.0-1.0.
        source_id: Document/chunk identifier (for audit traceability).
        metadata: Provider-specific metadata (page, section, index name, etc.).
            Must be JSON-serializable — providers must coerce non-primitive types
            (datetime -> ISO 8601 str, UUID -> str) at the Tier 3 boundary.
    """

    content: str
    score: float
    source_id: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"Score must be normalized to [0.0, 1.0], got {self.score!r}. "
                f"Provider score normalization bug — check the provider implementation."
            )
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"metadata must be JSON-serializable (got {type(exc).__name__}: {exc}). "
                f"Provider must coerce non-primitive types (datetime -> ISO 8601 str, "
                f"UUID -> str, etc.) at the Tier 3 boundary before constructing RetrievalChunk."
            ) from exc
        freeze_fields(self, "metadata")
