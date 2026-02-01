# tests/property/core/test_payload_store_properties.py
"""Property-based tests for payload store (content-addressable storage).

These tests verify the foundational properties of ELSPETH's payload store:
- Content integrity: retrieved content matches stored content exactly
- Determinism: same content always produces same hash
- Idempotence: storing same content multiple times is safe

Per ELSPETH's architecture, the payload store is Tier 1 (full trust) -
any anomaly in storage/retrieval is a catastrophic failure.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings

from elspeth.core.payload_store import FilesystemPayloadStore
from tests.property.conftest import binary_content, nonempty_binary, small_binary


class TestStoreRetrieveProperties:
    """Property tests for store/retrieve round-trip."""

    @given(content=binary_content)
    @settings(max_examples=300)
    def test_store_retrieve_roundtrip(self, content: bytes) -> None:
        """Property: retrieve(store(content)) == content.

        This is THE fundamental property of content-addressable storage.
        Stored content must be retrieved exactly, byte-for-byte.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemPayloadStore(Path(tmp_dir) / "payloads")

            content_hash = store.store(content)
            retrieved = store.retrieve(content_hash)

            assert retrieved == content, (
                f"Content changed during store/retrieve! Stored {len(content)} bytes, retrieved {len(retrieved)} bytes"
            )

    @given(content=nonempty_binary)
    @settings(max_examples=300)
    def test_exists_after_store(self, content: bytes) -> None:
        """Property: After store(), exists() returns True."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemPayloadStore(Path(tmp_dir) / "payloads")

            content_hash = store.store(content)

            assert store.exists(content_hash), "exists() returned False immediately after store()"


class TestHashDeterminismProperties:
    """Property tests for hash determinism."""

    @given(content=nonempty_binary)
    @settings(max_examples=500)
    def test_store_hash_is_deterministic(self, content: bytes) -> None:
        """Property: Same content always produces same hash.

        If hashing is non-deterministic, the audit trail becomes meaningless
        because we can't verify content integrity.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemPayloadStore(Path(tmp_dir) / "payloads")

            hash1 = store.store(content)
            hash2 = store.store(content)

            assert hash1 == hash2, f"Same content produced different hashes: {hash1} vs {hash2}"

    @given(content=nonempty_binary)
    @settings(max_examples=300)
    def test_store_hash_matches_sha256(self, content: bytes) -> None:
        """Property: Store returns standard SHA-256 hash.

        The payload store uses SHA-256 for content addressing.
        This verifies the hash matches direct SHA-256 computation.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemPayloadStore(Path(tmp_dir) / "payloads")

            store_hash = store.store(content)
            direct_hash = hashlib.sha256(content).hexdigest()

            assert store_hash == direct_hash, f"Store hash {store_hash} doesn't match direct SHA-256 {direct_hash}"

    @given(content1=small_binary, content2=small_binary)
    @settings(max_examples=200)
    def test_different_content_different_hash(self, content1: bytes, content2: bytes) -> None:
        """Property: Different content produces different hashes (with high probability).

        This is a statistical property - hash collisions are possible but
        astronomically unlikely for SHA-256.
        """
        if content1 == content2:
            # Skip when Hypothesis generates identical content
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemPayloadStore(Path(tmp_dir) / "payloads")

            hash1 = store.store(content1)
            hash2 = store.store(content2)

            assert hash1 != hash2, (
                f"Different content produced same hash! This is extremely unlikely ({len(content1)} bytes vs {len(content2)} bytes)"
            )


class TestStorageIdempotenceProperties:
    """Property tests for storage idempotence."""

    @given(content=nonempty_binary)
    @settings(max_examples=200)
    def test_store_is_idempotent(self, content: bytes) -> None:
        """Property: Storing same content multiple times doesn't create duplicates.

        Content-addressable storage deduplicates automatically.
        Multiple stores of same content should return same hash
        and not increase storage usage.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemPayloadStore(Path(tmp_dir) / "payloads")

            # First store
            hash1 = store.store(content)

            # Count files before second store
            files_before = list(Path(tmp_dir).rglob("*"))
            file_count_before = len([f for f in files_before if f.is_file()])

            # Second store of same content
            hash2 = store.store(content)

            # Count files after
            files_after = list(Path(tmp_dir).rglob("*"))
            file_count_after = len([f for f in files_after if f.is_file()])

            assert hash1 == hash2, "Same content produced different hashes"
            assert file_count_before == file_count_after, f"Duplicate storage created new file: {file_count_before} -> {file_count_after}"

    @given(content=nonempty_binary)
    @settings(max_examples=100)
    def test_store_then_delete_then_store(self, content: bytes) -> None:
        """Property: Delete then re-store produces same hash.

        After deleting content, storing it again should work
        and produce the same hash.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemPayloadStore(Path(tmp_dir) / "payloads")

            # Store, delete, store again
            hash1 = store.store(content)
            deleted = store.delete(hash1)
            hash2 = store.store(content)

            assert deleted, "Delete should return True for existing content"
            assert hash1 == hash2, "Re-stored content has different hash"
            assert store.exists(hash2), "Re-stored content should exist"


class TestIntegrityVerificationProperties:
    """Property tests for integrity verification."""

    @given(content=nonempty_binary)
    @settings(max_examples=100)
    def test_retrieve_nonexistent_raises_keyerror(self, content: bytes) -> None:
        """Property: Retrieving non-existent content raises KeyError.

        The store should not return garbage or empty bytes for
        content that doesn't exist.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemPayloadStore(Path(tmp_dir) / "payloads")

            # Generate hash without storing
            fake_hash = hashlib.sha256(content).hexdigest()

            with pytest.raises(KeyError):
                store.retrieve(fake_hash)

    @given(content=nonempty_binary)
    @settings(max_examples=50)
    def test_corrupted_content_detected(self, content: bytes) -> None:
        """Property: Corrupted content is detected on retrieval.

        If file content is modified after storage, integrity check
        should fail with IntegrityError.
        """
        from elspeth.contracts.payload_store import IntegrityError

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemPayloadStore(Path(tmp_dir) / "payloads")

            # Store content
            content_hash = store.store(content)

            # Corrupt the file
            file_path = store._path_for_hash(content_hash)
            corrupted = content + b"CORRUPTED"
            file_path.write_bytes(corrupted)

            # Retrieval should detect corruption
            with pytest.raises(IntegrityError):
                store.retrieve(content_hash)
