# src/elspeth/engine/artifacts.py
"""Unified artifact descriptors for all sink types.

Different sink types (file, database, webhook) produce different kinds of artifacts.
ArtifactDescriptor provides a unified interface that SinkExecutor uses to register
artifacts in the Landscape audit trail.

This solves the bug where database sinks crashed with KeyError because
SinkExecutor expected 'path', 'content_hash', 'size_bytes' keys that only
file-based artifacts provided.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ArtifactDescriptor:
    """Unified artifact descriptor for all sink types.

    All artifact types require a content hash and size for audit integrity.
    For non-file sinks, the hash is computed from the serialized payload
    before sending.

    Attributes:
        artifact_type: Type of artifact (file, database, webhook)
        path_or_uri: Unified location identifier with scheme prefix
        content_hash: SHA-256 hash of artifact content
        size_bytes: Size of content in bytes
        metadata: Optional type-specific metadata
    """

    artifact_type: Literal["file", "database", "webhook"]
    path_or_uri: str
    content_hash: str
    size_bytes: int
    metadata: dict[str, object] | None = None

    @classmethod
    def for_file(
        cls,
        path: str,
        content_hash: str,
        size_bytes: int,
    ) -> "ArtifactDescriptor":
        """Create descriptor for file-based artifacts.

        Args:
            path: File path (absolute or relative)
            content_hash: SHA-256 hash of file contents
            size_bytes: File size in bytes

        Returns:
            ArtifactDescriptor with file:// URI scheme
        """
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
        """Create descriptor for database artifacts.

        The content_hash should be computed from the canonical JSON
        of the rows before inserting into the database.

        Args:
            url: Database connection URL
            table: Target table name
            content_hash: Hash of serialized row payload
            payload_size: Size of serialized payload in bytes
            row_count: Number of rows written

        Returns:
            ArtifactDescriptor with db:// URI scheme
        """
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
        """Create descriptor for webhook artifacts.

        The content_hash should be computed from the canonical JSON
        of the request payload before sending.

        Args:
            url: Webhook URL
            content_hash: Hash of request payload
            request_size: Size of request payload in bytes
            response_code: HTTP response code received

        Returns:
            ArtifactDescriptor with webhook:// URI scheme
        """
        return cls(
            artifact_type="webhook",
            path_or_uri=f"webhook://{url}",
            content_hash=content_hash,
            size_bytes=request_size,
            metadata={"response_code": response_code},
        )
