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
