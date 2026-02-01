# tests/core/test_payload_store.py
"""Tests for payload store protocol and implementations."""

from pathlib import Path

import pytest


class TestPayloadStoreProtocol:
    """Test PayloadStore protocol definition."""

    def test_protocol_has_required_methods(self) -> None:
        from elspeth.contracts.payload_store import PayloadStore

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
        # Use valid hex format that doesn't exist in store
        assert store.exists("a" * 64) is False

    def test_retrieve_nonexistent_raises(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)

        # Use valid hex format that doesn't exist in store
        with pytest.raises(KeyError):
            store.retrieve("b" * 64)

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
        # Use valid hex format that doesn't exist in store
        result = store.delete("c" * 64)
        assert result is False

    def test_retrieve_corrupted_file_raises_integrity_error(self, tmp_path: Path) -> None:
        """Corrupted content must raise IntegrityError, never return silently."""
        from elspeth.contracts.payload_store import IntegrityError
        from elspeth.core.payload_store import FilesystemPayloadStore

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
        from elspeth.contracts.payload_store import IntegrityError
        from elspeth.core.payload_store import FilesystemPayloadStore

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

        from elspeth.contracts.payload_store import IntegrityError
        from elspeth.core.payload_store import FilesystemPayloadStore

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


class TestPayloadStoreSecurityValidation:
    """Security tests for content_hash validation and path containment.

    These tests verify that FilesystemPayloadStore rejects malformed hashes
    and path traversal attempts. Per CLAUDE.md Tier 1 rules, invalid data
    from the audit trail must crash immediately - never silently fail.
    """

    def test_retrieve_rejects_path_traversal(self, tmp_path: Path) -> None:
        """Path traversal attempts must raise ValueError, not access files."""
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path / "payloads")

        # Create a sensitive file outside the payload store
        sensitive_file = tmp_path / "sensitive.txt"
        sensitive_file.write_text("secret data")

        # Path traversal attempt - should raise, not read the file
        with pytest.raises(ValueError, match="Invalid content_hash"):
            store.retrieve("../sensitive.txt")

    def test_exists_rejects_path_traversal(self, tmp_path: Path) -> None:
        """exists() must not probe files outside base_path."""
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path / "payloads")

        # Create a file outside the payload store
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("data")

        # Should raise, not return True/False for external file
        with pytest.raises(ValueError, match="Invalid content_hash"):
            store.exists("../outside.txt")

    def test_delete_rejects_path_traversal(self, tmp_path: Path) -> None:
        """delete() must not delete files outside base_path."""
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path / "payloads")

        # Create a file that should NOT be deletable
        protected_file = tmp_path / "protected.txt"
        protected_file.write_text("important")

        # Should raise, not delete the file
        with pytest.raises(ValueError, match="Invalid content_hash"):
            store.delete("../protected.txt")

        # File must still exist
        assert protected_file.exists(), "Path traversal deleted file outside base_path!"

    def test_rejects_non_hex_characters(self, tmp_path: Path) -> None:
        """content_hash with non-hex characters must be rejected."""
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)

        # Contains 'g' which is not hex
        invalid_hash = "g" * 64

        with pytest.raises(ValueError, match="Invalid content_hash"):
            store.retrieve(invalid_hash)

    def test_rejects_uppercase_hex(self, tmp_path: Path) -> None:
        """content_hash must be lowercase hex (SHA-256 hexdigest convention)."""
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)

        # Uppercase hex - should be rejected for consistency
        uppercase_hash = "A" * 64

        with pytest.raises(ValueError, match="Invalid content_hash"):
            store.retrieve(uppercase_hash)

    def test_rejects_wrong_length(self, tmp_path: Path) -> None:
        """content_hash must be exactly 64 characters (SHA-256)."""
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)

        # Too short
        with pytest.raises(ValueError, match="Invalid content_hash"):
            store.retrieve("abcd1234")

        # Too long
        with pytest.raises(ValueError, match="Invalid content_hash"):
            store.retrieve("a" * 65)

    def test_rejects_empty_hash(self, tmp_path: Path) -> None:
        """Empty content_hash must be rejected."""
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)

        with pytest.raises(ValueError, match="Invalid content_hash"):
            store.retrieve("")

    def test_accepts_valid_sha256_hash(self, tmp_path: Path) -> None:
        """Valid SHA-256 hex digest should work normally."""
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)

        # Store content and verify round-trip still works
        content = b"valid content"
        content_hash = store.store(content)

        # Valid hash should work
        assert store.exists(content_hash)
        assert store.retrieve(content_hash) == content

    def test_path_containment_after_resolution(self, tmp_path: Path) -> None:
        """Resolved path must be under base_path even with valid-looking hash."""
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path / "payloads")

        # Create a file that could be accessed via symlink trickery
        # (if symlinks were followed, which they shouldn't be for this)
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "target.txt"
        target.write_text("should not be accessible")

        # Valid hex format but designed to break out
        # This tests the containment check after path resolution
        # Note: "2e2e" is hex for ".." - but the path construction
        # would create base_path/2e/2e2e... which is safe.
        # The real risk is ".." literal in first 2 chars.
        traversal_hash = ".." + "a" * 62  # Starts with ..

        with pytest.raises(ValueError, match="Invalid content_hash"):
            store.exists(traversal_hash)
