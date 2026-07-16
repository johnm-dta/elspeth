# tests/property/core/test_exporter_properties.py
"""Property-based tests for landscape exporter signing and record integrity.

The LandscapeExporter exports audit trails for compliance review. When signing
is enabled, each record gets an HMAC-SHA256 signature and a final manifest
contains a running hash chain.

Properties tested:
- Signing determinism: Same data → same signatures
- Hash chain integrity: final_hash is the SHA-256 of concatenated signatures
- Record count accuracy: manifest.record_count matches actual records
- Manifest ordering: manifest is always the last yielded record
- No signing without key: sign=True without key always raises
- Signature format: Always 64-char lowercase hex
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts.audit_export import AuditExportDerivationConfig
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.exporter import LandscapeExporter
from tests.fixtures.landscape import make_landscape_db
from tests.strategies.json import json_primitives

# =============================================================================
# Strategies
# =============================================================================

# Signing keys (non-empty bytes)
signing_keys = st.binary(min_size=16, max_size=64)

# Simple record dicts (the exporter signs these)
# Uses json_primitives to stay within RFC 8785 safe integer range
record_dicts = st.dictionaries(
    st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz_"),
    json_primitives,
    min_size=1,
    max_size=5,
)


# =============================================================================
# _sign_record Properties
# =============================================================================


class TestSignRecordDeterminism:
    """_sign_record must be deterministic for the same input."""

    @given(key=signing_keys, record=record_dicts)
    @settings(max_examples=200)
    def test_same_record_same_signature(self, key: bytes, record: dict[str, Any]) -> None:
        """Property: Signing the same record with the same key produces identical output."""
        exporter = _make_exporter(signing_key=key)
        sig1 = exporter._sign_record(record)
        sig2 = exporter._sign_record(record)
        assert sig1 == sig2

    @given(key=signing_keys, record=record_dicts)
    @settings(max_examples=200)
    def test_signature_is_64_char_hex(self, key: bytes, record: dict[str, Any]) -> None:
        """Property: Signature is always a 64-character lowercase hex string (SHA-256)."""
        exporter = _make_exporter(signing_key=key)
        sig = exporter._sign_record(record)
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    @given(key=signing_keys, record=record_dicts)
    @settings(max_examples=100)
    def test_signature_matches_manual_hmac(self, key: bytes, record: dict[str, Any]) -> None:
        """Property: Signature matches a manually computed HMAC-SHA256."""
        exporter = _make_exporter(signing_key=key)
        sig = exporter._sign_record(record)

        canonical = canonical_json(record)
        expected = hmac.new(key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
        assert sig == expected

    @given(
        key1=signing_keys,
        key2=signing_keys,
        record=record_dicts,
    )
    @settings(max_examples=100)
    def test_different_keys_different_signatures(self, key1: bytes, key2: bytes, record: dict[str, Any]) -> None:
        """Property: HMAC-inequivalent signing keys produce different signatures.

        HMAC (RFC 2104) zero-pads keys shorter than the hash block size, so
        keys that differ only by trailing zero bytes are the SAME effective
        key and legitimately produce identical signatures. Skip those.
        """
        # SHA-256 block size is 64 bytes; generated keys are <= 64 bytes,
        # so the HMAC-effective key is the zero-padded 64-byte form.
        if key1.ljust(64, b"\x00") == key2.ljust(64, b"\x00"):
            return  # Skip when keys are HMAC-equivalent

        exporter1 = _make_exporter(signing_key=key1)
        exporter2 = _make_exporter(signing_key=key2)
        sig1 = exporter1._sign_record(record)
        sig2 = exporter2._sign_record(record)
        assert sig1 != sig2

    def test_no_signing_key_raises(self) -> None:
        """Property: Signing without a key raises ValueError."""
        exporter = _make_exporter(signing_key=None)
        with pytest.raises(ValueError, match="Signing key not configured"):
            exporter._sign_record({"record_type": "test"})


# =============================================================================
# Export Signing Properties
# =============================================================================


class TestExportSigningProperties:
    """Signed exports must have correct hash chains and manifests."""

    @given(key=signing_keys)
    @settings(max_examples=50)
    def test_manifest_is_last_record(self, key: bytes) -> None:
        """Property: When sign=True, the manifest is always the last record."""
        exporter = _make_exporter_with_data(signing_key=key, num_records=3)
        records = list(exporter.export_run("run-1", sign=True))
        assert records[-1]["record_type"] == "manifest"

    @given(key=signing_keys)
    @settings(max_examples=50)
    def test_manifest_record_count_accurate(self, key: bytes) -> None:
        """Property: manifest.record_count matches actual non-manifest records."""
        n = 3
        exporter = _make_exporter_with_data(signing_key=key, num_records=n)
        records = list(exporter.export_run("run-1", sign=True))

        manifest = records[-1]
        non_manifest = [r for r in records if r["record_type"] != "manifest"]
        assert manifest["record_count"] == len(non_manifest)

    @given(key=signing_keys)
    @settings(max_examples=50)
    def test_hash_chain_integrity(self, key: bytes) -> None:
        """Property: final_hash is SHA-256 of concatenated record signatures."""
        exporter = _make_exporter_with_data(signing_key=key, num_records=3)
        records = list(exporter.export_run("run-1", sign=True))

        manifest = records[-1]
        non_manifest = [r for r in records if r["record_type"] != "manifest"]

        # Manually compute the running hash
        running_hash = hashlib.sha256()
        for record in non_manifest:
            assert "signature" in record, f"Record missing signature: {record}"
            running_hash.update(record["signature"].encode())

        assert manifest["final_hash"] == running_hash.hexdigest()

    @given(key=signing_keys)
    @settings(max_examples=50)
    def test_all_records_have_signatures(self, key: bytes) -> None:
        """Property: When sign=True, every record (including manifest) has a signature."""
        exporter = _make_exporter_with_data(signing_key=key, num_records=2)
        records = list(exporter.export_run("run-1", sign=True))

        for record in records:
            assert "signature" in record
            assert len(record["signature"]) == 64

    @given(key=signing_keys)
    @settings(max_examples=30)
    def test_signing_is_deterministic_for_data_records(self, key: bytes) -> None:
        """Property: Non-manifest records have deterministic signatures across invocations.

        The manifest includes `exported_at` (a timestamp), so its signature
        differs between invocations. But data records are pure functions of
        their content, so their signatures must be identical.
        """
        records1 = list(_make_exporter_with_data(signing_key=key, num_records=2).export_run("run-1", sign=True))
        records2 = list(_make_exporter_with_data(signing_key=key, num_records=2).export_run("run-1", sign=True))

        # Compare non-manifest records (manifest has timestamp, so skip it)
        data1 = [r for r in records1 if r["record_type"] != "manifest"]
        data2 = [r for r in records2 if r["record_type"] != "manifest"]
        for r1, r2 in zip(data1, data2, strict=True):
            assert r1["signature"] == r2["signature"]

    def test_sign_true_without_key_raises(self) -> None:
        """Property: export_run(sign=True) without signing_key raises ValueError."""
        exporter = _make_exporter_with_data(signing_key=None, num_records=1)
        with pytest.raises(ValueError, match="no signing_key"):
            list(exporter.export_run("run-1", sign=True))


class TestExportUnsignedProperties:
    """Unsigned exports retain data-record shape and one null-signed manifest."""

    def test_unsigned_records_have_no_signature(self) -> None:
        """Property: When sign=False, records have no 'signature' field."""
        exporter = _make_exporter_with_data(signing_key=None, num_records=2)
        records = list(exporter.export_run("run-1", sign=False))

        for record in records[:-1]:
            assert "signature" not in record
        assert records[-1]["signature"] is None

    def test_unsigned_has_one_manifest_last(self) -> None:
        """Property: unsigned v2 still emits exactly one final manifest."""
        exporter = _make_exporter_with_data(signing_key=None, num_records=2)
        records = list(exporter.export_run("run-1", sign=False))

        assert [r.get("record_type") for r in records].count("manifest") == 1
        assert records[-1]["record_type"] == "manifest"


# =============================================================================
# Manifest Metadata Properties
# =============================================================================


class TestManifestMetadata:
    """Manifest must contain all required metadata fields."""

    def test_manifest_has_required_fields(self) -> None:
        """Property: Manifest contains all required metadata."""
        exporter = _make_exporter_with_data(signing_key=b"test-key", num_records=2)
        records = list(exporter.export_run("run-1", sign=True))
        manifest = records[-1]

        assert manifest["record_type"] == "manifest"
        assert manifest["run_id"] == "run-1"
        assert "record_count" in manifest
        assert "final_hash" in manifest
        assert manifest["hash_algorithm"] == "sha256"
        assert manifest["signature_algorithm"] == "hmac_sha256"
        assert manifest["schema"] == "elspeth.audit-export-manifest.v2"
        assert "exported_at" in manifest
        assert "signature" in manifest


# =============================================================================
# Helpers
# =============================================================================


_SHARED_DB: LandscapeDB | None = None


def _shared_landscape_db() -> LandscapeDB:
    """Return a real (in-memory, schema-initialized) LandscapeDB, built once.

    The exporter's repository wiring enforces Tier-1 engine discipline (PRAGMA
    foreign_keys / journal_mode) at construction, so a mock DB is rejected.
    These tests never write through the DB; one shared read-only instance keeps
    the Hypothesis examples fast.
    """
    global _SHARED_DB
    if _SHARED_DB is None:
        _SHARED_DB = make_landscape_db()
    return _SHARED_DB


def _make_exporter(signing_key: bytes | None = None) -> LandscapeExporter:
    """Create an exporter over a real empty DB (for _sign_record tests only)."""
    return LandscapeExporter(
        _shared_landscape_db(),
        signing_key=signing_key,
        signer_key_id="property-signer-v1" if signing_key is not None else None,
    )


def _make_exporter_with_data(
    signing_key: bytes | None = None,
    num_records: int = 3,
) -> LandscapeExporter:
    """Create an exporter that yields deterministic test records.

    Patches _iter_records to yield simple test records; the signing/manifest
    pipeline under test is independent of how records are sourced.
    """
    test_records = [{"record_type": f"test_{i}", "data": f"value_{i}", "index": i} for i in range(num_records)]

    signing_mode = "hmac_sha256" if signing_key is not None else "unsigned"
    exporter = LandscapeExporter(
        _shared_landscape_db(),
        signing_key=signing_key,
        signer_key_id="property-signer-v1" if signing_key is not None else None,
        derivation_config=AuditExportDerivationConfig(
            source_run_id="run-1",
            source_status="completed",
            source_completed_at="2026-01-15T13:00:00.000000Z",
            export_format="json",
            exporter_version="landscape-exporter-v1",
            serialization_version="audit-export-v2",
            chunking_algorithm_version="record-framing-v1",
            include_raw_error_rows=False,
            per_chunk_byte_limit=1_048_576,
            per_chunk_record_limit=1_000,
            signing_mode=signing_mode,
            signer_key_id="property-signer-v1" if signing_key is not None else "UNSIGNED",
            signing_key=signing_key,
        ),
    )

    # Patch _iter_records to return our test data
    def _mock_iter(run_id: str):
        yield from test_records

    exporter._iter_records = _mock_iter  # type: ignore[method-assign]
    return exporter
