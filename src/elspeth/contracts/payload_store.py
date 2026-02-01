# src/elspeth/contracts/payload_store.py
"""PayloadStore protocol for content-addressable blob storage.

This protocol defines the interface for payload storage backends used by:
- core/payload_store.py (FilesystemPayloadStore implementation)
- core/retention/purge.py (PurgeManager for retention policy)

Consolidated here to avoid circular imports and provide single source of truth.
"""

from typing import Protocol, runtime_checkable


class IntegrityError(Exception):
    """Raised when payload content doesn't match expected hash.

    This indicates either filesystem corruption, tampering, or a bug.
    For an audit system, this is a critical failure that must be
    investigated - we never silently return corrupted data.
    """

    pass


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
        """Retrieve content by hash with integrity verification.

        Args:
            content_hash: SHA-256 hex digest

        Returns:
            Original content bytes

        Raises:
            KeyError: If content not found
            IntegrityError: If content doesn't match expected hash
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
