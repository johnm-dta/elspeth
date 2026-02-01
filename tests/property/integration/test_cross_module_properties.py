# tests/property/integration/test_cross_module_properties.py
"""Cross-module property tests for ELSPETH audit pipeline consistency.

These tests verify that invariants hold ACROSS module boundaries:
- Field normalization + canonical hash consistency
- Canonical JSON + payload store hash consistency
- End-to-end determinism of the audit pipeline

These are "glue" tests that catch integration bugs that unit tests miss.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from hypothesis import assume, given, settings

from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.plugins.sources.field_normalization import normalize_field_name
from tests.property.conftest import messy_headers, row_data


class TestFieldNormalizationCanonicalHashProperties:
    """Tests that field normalization + canonical hash work together correctly."""

    @given(raw=messy_headers)
    @settings(max_examples=300)
    def test_normalized_fields_hash_consistently(self, raw: str) -> None:
        """Property: Same raw header always produces same canonical hash.

        This bridges field normalization (trust boundary) with canonical JSON (audit trail).
        The same input must ALWAYS produce the same audit trail entry.
        """
        try:
            normalized = normalize_field_name(raw)
        except ValueError:
            assume(False)
            return

        # Hash a dict containing the normalized field
        hash1 = stable_hash({"field": normalized})
        hash2 = stable_hash({"field": normalized})

        assert hash1 == hash2, "Normalized field name produced inconsistent hash"

    @given(raw=messy_headers)
    @settings(max_examples=200)
    def test_normalization_canonical_idempotence(self, raw: str) -> None:
        """Property: Normalize → hash == normalize → normalize → hash.

        If normalization is idempotent (which it is), the hash should be
        identical whether we normalize once or twice.
        """
        try:
            normalized_once = normalize_field_name(raw)
            normalized_twice = normalize_field_name(normalized_once)
        except ValueError:
            assume(False)
            return

        hash_once = stable_hash({"field": normalized_once})
        hash_twice = stable_hash({"field": normalized_twice})

        assert hash_once == hash_twice, f"Idempotence broken in hash chain: '{raw}' -> '{normalized_once}' -> '{normalized_twice}'"


class TestCanonicalJsonPayloadStoreProperties:
    """Tests that canonical JSON + payload store work together correctly."""

    @given(data=row_data)
    @settings(max_examples=200)
    def test_canonical_payload_hash_consistency(self, data: dict[str, Any]) -> None:
        """Property: Canonical JSON of data hashes same way as payload store.

        The payload store uses SHA-256 for content addressing.
        Canonical JSON is the serialization format.
        These must produce consistent hashes.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemPayloadStore(Path(tmp_dir) / "payloads")

            # Canonical JSON of the data
            canonical = canonical_json(data)
            canonical_bytes = canonical.encode("utf-8")

            # Store returns SHA-256 hash
            payload_hash = store.store(canonical_bytes)

            # Direct SHA-256 should match
            direct_hash = hashlib.sha256(canonical_bytes).hexdigest()

            assert payload_hash == direct_hash, f"Hash mismatch: payload_store={payload_hash}, direct={direct_hash}"

    @given(data=row_data)
    @settings(max_examples=200)
    def test_canonical_stable_hash_matches_payload_store(self, data: dict[str, Any]) -> None:
        """Property: stable_hash() matches payload store hash of canonical JSON.

        stable_hash() is a convenience wrapper that should produce the
        same hash as manually storing canonical JSON in the payload store.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemPayloadStore(Path(tmp_dir) / "payloads")

            # stable_hash uses canonical_json internally
            stable = stable_hash(data)

            # Manually compute via payload store
            canonical_bytes = canonical_json(data).encode("utf-8")
            payload_hash = store.store(canonical_bytes)

            assert stable == payload_hash, f"stable_hash() doesn't match payload store: {stable} vs {payload_hash}"

    @given(data=row_data)
    @settings(max_examples=100)
    def test_payload_store_preserves_canonical_json_exactly(self, data: dict[str, Any]) -> None:
        """Property: Stored canonical JSON retrieves identically.

        The canonical JSON bytes stored must be exactly what we retrieve.
        This is critical for audit trail integrity.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemPayloadStore(Path(tmp_dir) / "payloads")

            # Store canonical JSON
            original = canonical_json(data)
            original_bytes = original.encode("utf-8")
            content_hash = store.store(original_bytes)

            # Retrieve and compare
            retrieved_bytes = store.retrieve(content_hash)
            retrieved = retrieved_bytes.decode("utf-8")

            assert retrieved == original, (
                f"Canonical JSON changed during storage! Original: {original[:100]}..., Retrieved: {retrieved[:100]}..."
            )


class TestEndToEndAuditPipelineProperties:
    """Tests for end-to-end audit pipeline determinism."""

    @given(data=row_data)
    @settings(max_examples=200)
    def test_full_pipeline_determinism(self, data: dict[str, Any]) -> None:
        """Property: Full audit pipeline is deterministic.

        Data → canonical JSON → hash → store → retrieve
        must produce byte-identical content.

        Note: We verify BYTE identity, not Python object round-trip,
        because that's what matters for audit integrity. The stored
        bytes must be exactly what we retrieve.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemPayloadStore(Path(tmp_dir) / "payloads")

            # Full pipeline
            canonical = canonical_json(data)
            canonical_bytes = canonical.encode("utf-8")
            content_hash = store.store(canonical_bytes)
            retrieved_bytes = store.retrieve(content_hash)

            # Byte-identical comparison (this is what audit integrity requires)
            assert retrieved_bytes == canonical_bytes, "Full pipeline round-trip changed the bytes!"

            # Also verify the hash is reproducible
            recalculated_hash = hashlib.sha256(retrieved_bytes).hexdigest()
            assert recalculated_hash == content_hash, "Hash changed after retrieve!"

    @given(data=row_data)
    @settings(max_examples=100)
    def test_hash_proves_integrity(self, data: dict[str, Any]) -> None:
        """Property: Hash can verify data hasn't been tampered with.

        If we store data and later retrieve it with the same hash,
        we can prove the data is unchanged.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemPayloadStore(Path(tmp_dir) / "payloads")

            # Store original data
            original = canonical_json(data)
            original_hash = store.store(original.encode("utf-8"))

            # Recompute hash from retrieved data
            retrieved = store.retrieve(original_hash)
            recomputed_hash = hashlib.sha256(retrieved).hexdigest()

            assert recomputed_hash == original_hash, "Integrity verification failed! Data may have been tampered with."
