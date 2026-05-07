"""Tests for payload storage through PayloadStore.

Previously tested ExecutionRepository.store_payload(), which was a thin
wrapper over PayloadStore.store(). The recorder facade has been eliminated;
payload storage is now accessed directly through PayloadStore.
"""

import tempfile
from pathlib import Path

from elspeth.core.payload_store import FilesystemPayloadStore


class TestStorePayload:
    def test_stores_content_and_returns_sha256_hex(self):
        """store() returns a 64-char hex string (SHA-256)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FilesystemPayloadStore(Path(tmpdir))

            content = b"test processed content for audit"
            result = store.store(content)

            # SHA-256 hex digest is 64 characters
            assert isinstance(result, str)
            assert len(result) == 64
            assert all(c in "0123456789abcdef" for c in result)

            # Content is retrievable
            retrieved = store.retrieve(result)
            assert retrieved == content

    def test_store_payload_empty_bytes(self):
        """Empty content is valid — SHA-256 of empty is well-defined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FilesystemPayloadStore(Path(tmpdir))

            result = store.store(b"")
            assert len(result) == 64  # SHA-256 hex

    def test_same_content_returns_same_hash(self):
        """Same content always returns the same hash (content-addressed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FilesystemPayloadStore(Path(tmpdir))

            content = b"identical content"
            hash1 = store.store(content)
            hash2 = store.store(content)

            assert hash1 == hash2
