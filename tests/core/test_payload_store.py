# tests/core/test_payload_store.py
"""Tests for payload store protocol and implementations."""

from pathlib import Path

import pytest


class TestPayloadStoreProtocol:
    """Test PayloadStore protocol definition."""

    def test_protocol_has_required_methods(self) -> None:
        from elspeth.core.payload_store import PayloadStore

        # Protocol should define these methods
        assert hasattr(PayloadStore, "store")
        assert hasattr(PayloadStore, "retrieve")
        assert hasattr(PayloadStore, "exists")
        assert hasattr(PayloadStore, "delete")


class TestFilesystemPayloadStore:
    """Test filesystem-based payload store."""

    def test_store_returns_content_hash(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)
        content = b"hello world"
        content_hash = store.store(content)

        # Should be SHA-256 hex
        assert len(content_hash) == 64
        assert all(c in "0123456789abcdef" for c in content_hash)

    def test_retrieve_by_hash(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)
        content = b"hello world"
        content_hash = store.store(content)

        retrieved = store.retrieve(content_hash)
        assert retrieved == content

    def test_exists_returns_true_for_stored(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)
        content = b"test content"
        content_hash = store.store(content)

        assert store.exists(content_hash) is True
        assert store.exists("nonexistent" * 4) is False

    def test_retrieve_nonexistent_raises(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)

        with pytest.raises(KeyError):
            store.retrieve("nonexistent" * 4)

    def test_store_is_idempotent(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)
        content = b"duplicate content"

        hash1 = store.store(content)
        hash2 = store.store(content)

        assert hash1 == hash2

    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)
        content = b"test"
        content_hash = store.store(content)

        # Should use first 2 chars as subdirectory for distribution
        expected_dir = tmp_path / content_hash[:2]
        expected_file = expected_dir / content_hash

        assert expected_dir.exists()
        assert expected_file.exists()

    def test_delete_removes_content(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)
        content = b"content to delete"
        content_hash = store.store(content)

        assert store.exists(content_hash)
        result = store.delete(content_hash)
        assert result is True
        assert store.exists(content_hash) is False

    def test_delete_nonexistent_returns_false(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)
        result = store.delete("nonexistent" * 4)
        assert result is False

    def test_retrieve_corrupted_file_raises_integrity_error(self, tmp_path: Path) -> None:
        """Corrupted content must raise IntegrityError, never return silently."""
        from elspeth.core.payload_store import FilesystemPayloadStore, IntegrityError

        store = FilesystemPayloadStore(base_path=tmp_path)
        content = b"original content"
        content_hash = store.store(content)

        # Corrupt the file directly
        file_path = tmp_path / content_hash[:2] / content_hash
        file_path.write_bytes(b"corrupted content")

        with pytest.raises(IntegrityError) as exc_info:
            store.retrieve(content_hash)

        assert "integrity check failed" in str(exc_info.value)
        assert content_hash in str(exc_info.value)

    def test_retrieve_truncated_file_raises_integrity_error(self, tmp_path: Path) -> None:
        """Truncated files are also integrity violations."""
        from elspeth.core.payload_store import FilesystemPayloadStore, IntegrityError

        store = FilesystemPayloadStore(base_path=tmp_path)
        content = b"this is some longer content that will be truncated"
        content_hash = store.store(content)

        # Truncate the file
        file_path = tmp_path / content_hash[:2] / content_hash
        file_path.write_bytes(content[:10])

        with pytest.raises(IntegrityError):
            store.retrieve(content_hash)

    def test_integrity_error_includes_actual_hash(self, tmp_path: Path) -> None:
        """IntegrityError message should include actual hash for investigation."""
        import hashlib

        from elspeth.core.payload_store import FilesystemPayloadStore, IntegrityError

        store = FilesystemPayloadStore(base_path=tmp_path)
        original = b"original"
        content_hash = store.store(original)

        # Replace with different content
        corrupted = b"different"
        file_path = tmp_path / content_hash[:2] / content_hash
        file_path.write_bytes(corrupted)

        corrupted_hash = hashlib.sha256(corrupted).hexdigest()

        with pytest.raises(IntegrityError) as exc_info:
            store.retrieve(content_hash)

        # Error message should contain both hashes for debugging
        assert content_hash in str(exc_info.value)  # expected
        assert corrupted_hash in str(exc_info.value)  # actual
