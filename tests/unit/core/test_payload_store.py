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

    def test_payload_not_found_error_is_not_a_keyerror(self) -> None:
        """PayloadNotFoundError must NOT be catchable by except KeyError.

        This is the whole point of the domain exception — callers that
        catch KeyError (e.g. for dict lookups) must not accidentally
        swallow a missing-payload condition.
        """
        from elspeth.contracts.payload_store import PayloadNotFoundError

        assert not issubclass(PayloadNotFoundError, KeyError)
        assert not issubclass(PayloadNotFoundError, LookupError)


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

    def test_retrieve_nonexistent_raises_payload_not_found_error(self, tmp_path: Path) -> None:
        from elspeth.contracts.payload_store import PayloadNotFoundError
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)
        fake_hash = "b" * 64

        with pytest.raises(PayloadNotFoundError) as exc_info:
            store.retrieve(fake_hash)

        assert exc_info.value.content_hash == fake_hash
        assert fake_hash in str(exc_info.value)

    def test_store_is_idempotent(self, tmp_path: Path) -> None:
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)
        content = b"duplicate content"

        hash1 = store.store(content)
        hash2 = store.store(content)

        assert hash1 == hash2

    def test_store_detects_corrupted_existing_file(self, tmp_path: Path) -> None:
        """BUG #5: store() must verify existing files match expected hash.

        If a file exists but is corrupted (bit rot, tampering, previous write failure),
        store() must detect the mismatch and raise IntegrityError. Without this check,
        the audit trail would reference a hash that doesn't match the actual content,
        violating Tier-1 integrity.

        This is the symmetric requirement to retrieve() which already verifies integrity.
        """
        from elspeth.contracts.payload_store import IntegrityError
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)
        original_content = b"original content"
        content_hash = store.store(original_content)

        # Corrupt the existing file (simulating bit rot, tampering, etc.)
        file_path = tmp_path / content_hash[:2] / content_hash
        corrupted_content = b"corrupted by bit rot"
        file_path.write_bytes(corrupted_content)

        # Attempt to store the original content again
        # Since the file exists, store() does early return WITHOUT verification.
        # This is the bug - it should verify the existing file matches the expected hash.
        with pytest.raises(IntegrityError) as exc_info:
            store.store(original_content)

        # Error should indicate the mismatch
        assert "integrity check failed" in str(exc_info.value).lower()
        assert content_hash in str(exc_info.value)

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


class TestPayloadStoreConcurrency:
    """Bug 7.7: Concurrent writes to the same hash must not race on temp files."""

    def test_concurrent_writes_same_hash(self, tmp_path: Path) -> None:
        """Multiple threads writing the same content should all succeed."""
        import concurrent.futures

        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)
        content = b"concurrent content for deduplication test"
        errors: list[Exception] = []

        def _write() -> str:
            try:
                return store.store(content)
            except Exception as e:
                errors.append(e)
                raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_write) for _ in range(20)]
            results = [f.result() for f in futures]

        # All writes should succeed and return the same hash
        assert len(errors) == 0
        assert all(r == results[0] for r in results)

        # Content should be retrievable
        assert store.retrieve(results[0]) == content

    def test_concurrent_writes_different_hashes(self, tmp_path: Path) -> None:
        """Concurrent writes of different content should all succeed."""
        import concurrent.futures

        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)
        errors: list[Exception] = []

        def _write(i: int) -> str:
            try:
                return store.store(f"content_{i}".encode())
            except Exception as e:
                errors.append(e)
                raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_write, i) for i in range(20)]
            results = [f.result() for f in futures]

        assert len(errors) == 0
        # All hashes should be unique
        assert len(set(results)) == 20


class TestPayloadStoreCleanup:
    """Mutation-killing tests for crash recovery in FilesystemPayloadStore.store().

    Targets surviving mutants on lines 110-132 of payload_store.py:
    - Lines 117/129: except BaseException cleanup must catch real exceptions
    - Lines 118/130: temp file cleanup must not be inverted
    - Line 110: parents=True must create nested hash prefix directories
    """

    def test_store_cleans_up_temp_on_write_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject fsync failure after write, verify no orphaned .tmp files.

        Kills mutants:
        - Line 117: except BaseException narrowed — cleanup stops catching OSError
        - Line 118: if temp_path.exists() inverted — orphaned temp file left behind
        """
        import os

        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(tmp_path / "payloads")

        # Inject fsync failure — this triggers the write-phase cleanup (lines 117-120)
        def failing_fsync(fd: int) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(os, "fsync", failing_fsync)

        with pytest.raises(OSError, match="disk full"):
            store.store(b"will fail on fsync")

        # If cleanup is inverted (mutant: `if not temp_path.exists()`), temp file remains
        # If exception type is narrowed (mutant: except Exception), BaseException-derived
        # errors won't clean up — but OSError IS an Exception subclass, so we also test
        # with KeyboardInterrupt below.
        orphaned = list(tmp_path.rglob("*.tmp"))
        assert orphaned == [], f"Orphaned temp files found: {orphaned}"

    def test_store_cleans_up_temp_on_keyboard_interrupt_during_write(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """KeyboardInterrupt during write must still clean up temp file.

        Kills mutant on line 117: except BaseException → except Exception
        KeyboardInterrupt is a BaseException but NOT an Exception subclass.
        """
        import os

        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(tmp_path / "payloads")

        def interrupted_fsync(fd: int) -> None:
            raise KeyboardInterrupt()

        monkeypatch.setattr(os, "fsync", interrupted_fsync)

        with pytest.raises(KeyboardInterrupt):
            store.store(b"interrupted during write")

        orphaned = list(tmp_path.rglob("*.tmp"))
        assert orphaned == [], f"Orphaned temp files after KeyboardInterrupt: {orphaned}"

    def test_store_cleans_up_temp_on_rename_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject os.replace failure, verify temp file cleaned up.

        Kills mutants:
        - Line 129: except BaseException narrowed — cleanup stops catching OSError
        - Line 130: if temp_path.exists() inverted — orphaned temp file left behind
        """
        import os

        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(tmp_path / "payloads")

        def failing_replace(src: str, dst: str) -> None:
            raise OSError("rename failed")

        monkeypatch.setattr(os, "replace", failing_replace)

        with pytest.raises(OSError, match="rename failed"):
            store.store(b"will fail on rename")

        orphaned = list(tmp_path.rglob("*.tmp"))
        assert orphaned == [], f"Orphaned temp files after rename failure: {orphaned}"

    def test_store_cleans_up_temp_on_keyboard_interrupt_during_rename(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """KeyboardInterrupt during rename must still clean up temp file.

        Kills mutant on line 129: except BaseException → except Exception
        """
        import os

        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(tmp_path / "payloads")

        def interrupted_replace(src: str, dst: str) -> None:
            raise KeyboardInterrupt()

        monkeypatch.setattr(os, "replace", interrupted_replace)

        with pytest.raises(KeyboardInterrupt):
            store.store(b"interrupted during rename")

        orphaned = list(tmp_path.rglob("*.tmp"))
        assert orphaned == [], f"Orphaned temp files after KeyboardInterrupt on rename: {orphaned}"

    def test_store_creates_nested_hash_prefix_directory(self, tmp_path: Path) -> None:
        """Line 110: parents=True creates intermediate directories on first write.

        Kills mutant: parents=True → parents=False
        When parents=False, mkdir fails if the hash-prefix subdirectory's parent
        doesn't exist. The base_path is created in __init__, but the 2-char hash
        prefix subdirectory (e.g., base_path/ab/) is created on first store().
        We verify by using a base_path that exists but has no subdirectories yet.
        """
        from elspeth.core.payload_store import FilesystemPayloadStore

        # base_path is created by __init__, but hash prefix dirs are not
        store = FilesystemPayloadStore(tmp_path / "payloads")
        content_hash = store.store(b"first write creates hash prefix dir")

        # Verify the content was stored and is retrievable
        assert store.exists(content_hash)
        assert store.retrieve(content_hash) == b"first write creates hash prefix dir"

        # Verify the nested directory structure was created
        hash_prefix_dir = tmp_path / "payloads" / content_hash[:2]
        assert hash_prefix_dir.is_dir()


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

    def test_rejects_trailing_newline(self, tmp_path: Path) -> None:
        """content_hash with a trailing newline must be rejected.

        Python's ``re.match(r"^...$", s)`` treats ``$`` as "end of string
        OR just before a final newline", so a 64-hex hash followed by a
        single ``\\n`` would pass a naive ``^[a-f0-9]{64}$`` check and
        then mis-key the on-disk filename — every subsequent lookup
        would fail with an opaque FileNotFoundError instead of a clean
        validation error. ``fullmatch`` (and equivalently ``\\A...\\Z``)
        is the correct anchor: it requires the whole string to match,
        newline and all.

        A valid SHA-256 digest from ``hashlib.sha256().hexdigest()``
        never contains ``\\n``, so any value that does is either
        externally-sourced (Tier 3 boundary — reject) or corrupt Tier-1
        data (reject).
        """
        from elspeth.core.payload_store import FilesystemPayloadStore

        store = FilesystemPayloadStore(base_path=tmp_path)

        trailing_newline_hash = "a" * 64 + "\n"

        with pytest.raises(ValueError, match="Invalid content_hash"):
            store.retrieve(trailing_newline_hash)
        with pytest.raises(ValueError, match="Invalid content_hash"):
            store.exists(trailing_newline_hash)

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
