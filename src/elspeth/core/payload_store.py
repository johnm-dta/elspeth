# src/elspeth/core/payload_store.py
"""
Payload store for separating large blobs from audit tables.

Uses content-addressable storage (hash-based) for:
- Automatic deduplication of identical content
- Integrity verification on retrieval
- Efficient storage of large payloads referenced by multiple rows
"""

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class PayloadStore(Protocol):
    """Protocol for payload storage backends.

    All implementations must provide content-addressable storage
    where payloads are stored by their SHA-256 hash.
    """

    def store(self, content: bytes) -> str:
        """Store content and return its hash.

        Args:
            content: Raw bytes to store

        Returns:
            SHA-256 hex digest of content
        """
        ...

    def retrieve(self, content_hash: str) -> bytes:
        """Retrieve content by hash.

        Args:
            content_hash: SHA-256 hex digest

        Returns:
            Original content bytes

        Raises:
            KeyError: If content not found
        """
        ...

    def exists(self, content_hash: str) -> bool:
        """Check if content exists.

        Args:
            content_hash: SHA-256 hex digest

        Returns:
            True if content exists
        """
        ...

    def delete(self, content_hash: str) -> bool:
        """Delete content by hash.

        Args:
            content_hash: SHA-256 hex digest

        Returns:
            True if content was deleted, False if not found
        """
        ...


class FilesystemPayloadStore:
    """Filesystem-based payload store.

    Stores payloads in a directory structure using first 2 characters
    of hash as subdirectory for better file distribution.

    Structure: base_path/ab/abcdef123...
    """

    def __init__(self, base_path: Path) -> None:
        """Initialize filesystem store.

        Args:
            base_path: Root directory for payload storage
        """
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path_for_hash(self, content_hash: str) -> Path:
        """Get filesystem path for content hash."""
        # Use first 2 chars as subdirectory
        return self.base_path / content_hash[:2] / content_hash

    def store(self, content: bytes) -> str:
        """Store content and return its hash."""
        content_hash = hashlib.sha256(content).hexdigest()
        path = self._path_for_hash(content_hash)

        # Idempotent: skip if already exists
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        return content_hash

    def retrieve(self, content_hash: str) -> bytes:
        """Retrieve content by hash."""
        path = self._path_for_hash(content_hash)
        if not path.exists():
            raise KeyError(f"Payload not found: {content_hash}")
        return path.read_bytes()

    def exists(self, content_hash: str) -> bool:
        """Check if content exists."""
        return self._path_for_hash(content_hash).exists()

    def delete(self, content_hash: str) -> bool:
        """Delete content by hash.

        Returns:
            True if content was deleted, False if not found
        """
        path = self._path_for_hash(content_hash)
        if not path.exists():
            return False
        path.unlink()
        return True
