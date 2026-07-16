"""Independent vectors for the closed audit-export derivation boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

import pytest

from elspeth.contracts.audit_export import (
    AUDIT_EXPORT_DERIVATION_VERSION,
    REF,
    AuditExportContentDescriptor,
    AuditExportContentStoreResolver,
    C,
    H,
    derive_public_export_config_hash,
    derive_registry_key_hash,
)

PUBLIC_CONFIG = {
    "chunking_algorithm_version": "record-framing-v1",
    "export_format": "json",
    "exporter_version": "landscape-exporter-v1",
    "include_raw_error_rows": False,
    "per_chunk_byte_limit": 1048576,
    "per_chunk_record_limit": 1000,
    "serialization_version": "audit-export-v2",
    "signer_key_id": "UNSIGNED",
    "signing_mode": "unsigned",
}

PUBLIC_CONFIG_BYTES = (
    b'{"payload":{"chunking_algorithm_version":"record-framing-v1","export_format":"json",'
    b'"exporter_version":"landscape-exporter-v1","include_raw_error_rows":false,'
    b'"per_chunk_byte_limit":1048576,"per_chunk_record_limit":1000,'
    b'"serialization_version":"audit-export-v2","signer_key_id":"UNSIGNED",'
    b'"signing_mode":"unsigned"},"schema":"audit-export-public-config-v1"}'
)
PUBLIC_CONFIG_HASH = "a9b4481a9c8c16eed4d98109ac63bf0021811da52e22dfba8a603e54ab266362"

REGISTRY_KEY = {
    "export_format": "json",
    "exporter_version": "landscape-exporter-v1",
    "public_export_config_hash": PUBLIC_CONFIG_HASH,
    "serialization_version": "audit-export-v2",
    "signer_key_id": "UNSIGNED",
    "signing_mode": "unsigned",
    "source_run_id": "run-golden-001",
}
REGISTRY_KEY_BYTES = (
    b'{"payload":{"export_format":"json","exporter_version":"landscape-exporter-v1",'
    b'"public_export_config_hash":"a9b4481a9c8c16eed4d98109ac63bf0021811da52e22dfba8a603e54ab266362",'
    b'"serialization_version":"audit-export-v2","signer_key_id":"UNSIGNED",'
    b'"signing_mode":"unsigned","source_run_id":"run-golden-001"},'
    b'"schema":"audit-export-registry-key-v1"}'
)
REGISTRY_KEY_HASH = "5726ec70871545ac8f28360b2da7da568cbb45e68a7c3ac6d41873a6bdfda06d"


def test_literal_public_config_and_registry_key_vectors() -> None:
    assert AUDIT_EXPORT_DERIVATION_VERSION == "audit-export-derivation-v1"
    assert C("audit-export-public-config-v1", PUBLIC_CONFIG) == PUBLIC_CONFIG_BYTES
    assert H(PUBLIC_CONFIG_BYTES) == PUBLIC_CONFIG_HASH
    assert REF(PUBLIC_CONFIG_HASH) == f"sha256:{PUBLIC_CONFIG_HASH}"
    assert derive_public_export_config_hash(PUBLIC_CONFIG) == PUBLIC_CONFIG_HASH

    assert C("audit-export-registry-key-v1", REGISTRY_KEY) == REGISTRY_KEY_BYTES
    assert H(REGISTRY_KEY_BYTES) == REGISTRY_KEY_HASH
    assert derive_registry_key_hash(REGISTRY_KEY) == REGISTRY_KEY_HASH


@pytest.mark.parametrize(
    "payload",
    [
        {**PUBLIC_CONFIG, "unknown": None},
        {key: value for key, value in PUBLIC_CONFIG.items() if key != "export_format"},
        {**PUBLIC_CONFIG, "per_chunk_record_limit": 1.0},
        {**PUBLIC_CONFIG, "per_chunk_record_limit": 9007199254740992},
        {**PUBLIC_CONFIG, "export_format": ("json",)},
        {**PUBLIC_CONFIG, "signing_mode": {"unsigned"}},
        {**PUBLIC_CONFIG, 1: "non-string-key"},
        {**PUBLIC_CONFIG, "export_format": b"json"},
        {**PUBLIC_CONFIG, "exporter_version": datetime(2026, 7, 16, tzinfo=UTC)},
    ],
)
def test_closed_public_config_rejects_non_schema_values(payload: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        C("audit-export-public-config-v1", payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("public_export_config_hash", "A" * 64),
        ("public_export_config_hash", "0" * 63),
        ("signing_mode", "HMAC_SHA256"),
        ("export_format", "xml"),
    ],
)
def test_registry_key_rejects_noncanonical_identity_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        C("audit-export-registry-key-v1", {**REGISTRY_KEY, field: value})


def test_c_rejects_unknown_tags_and_h_never_canonicalizes() -> None:
    with pytest.raises(ValueError, match="unknown audit-export schema tag"):
        C("audit-export-not-real-v1", {})
    with pytest.raises(TypeError, match="bytes"):
        H(PUBLIC_CONFIG)  # type: ignore[arg-type]


class _Format(StrEnum):
    JSON = "json"


def test_c_rejects_implicit_enum_conversion() -> None:
    with pytest.raises(TypeError, match="enum"):
        C("audit-export-public-config-v1", {**PUBLIC_CONFIG, "export_format": _Format.JSON})


def test_snapshot_stage_validates_nested_closed_shapes_and_canonical_timestamps() -> None:
    snapshot_seal = {
        "chunk_count": 1,
        "chunking_algorithm_version": "record-framing-v1",
        "exported_at": "2026-07-16T12:00:00.000001Z",
        "last_chunk_seal_hash": "1" * 64,
        "manifest_hash": "2" * 64,
        "per_chunk_byte_limit": 1024,
        "per_chunk_record_limit": 10,
        "record_count": 1,
        "registry_key_hash": "3" * 64,
        "snapshot_hash": "4" * 64,
        "snapshot_id": "5" * 64,
        "source_completed_at": "2026-07-16T11:59:59.999999Z",
        "source_run_id": "run-1",
        "source_status": "completed",
        "total_bytes": 12,
    }
    assert C("audit-export-snapshot-seal-v1", snapshot_seal).startswith(b'{"payload":')
    with pytest.raises(ValueError, match="timestamp"):
        C(
            "audit-export-snapshot-seal-v1",
            {**snapshot_seal, "exported_at": "2026-07-16T12:00:00Z"},
        )


def test_content_descriptor_requires_exact_sha256_reference() -> None:
    descriptor = AuditExportContentDescriptor(
        content_ref=f"sha256:{'a' * 64}",
        content_hash="a" * 64,
        size_bytes=7,
        object_kind="data_chunk",
    )
    assert descriptor.content_ref == f"sha256:{'a' * 64}"
    with pytest.raises(ValueError, match="must equal"):
        AuditExportContentDescriptor(
            content_ref=f"sha256:{'b' * 64}",
            content_hash="a" * 64,
            size_bytes=7,
            object_kind="data_chunk",
        )


class _Store:
    content_store_id = "archive-primary-v1"
    namespace = "audit-export"

    def is_durable(self) -> bool:
        return True

    def put_immutable(self, content: bytes, *, candidate_id: str, object_kind: str) -> str:
        del content, candidate_id, object_kind
        return f"sha256:{'a' * 64}"

    def open_registered(self, registration: object) -> object:
        return registration

    def mark_candidate_orphans(self, candidate_id: str, descriptors: tuple[object, ...]) -> None:
        del candidate_id, descriptors

    def garbage_collect_candidate(self, request: object) -> bool:
        del request
        return False


def test_store_resolver_keeps_content_store_id_stable_and_rejects_reinterpretation() -> None:
    resolver = AuditExportContentStoreResolver()
    store = _Store()
    resolver.register(store)
    assert resolver.resolve("archive-primary-v1") is store
    resolver.register(store)  # same exact instance is idempotent
    with pytest.raises(ValueError, match="already registered"):
        resolver.register(_Store())
    with pytest.raises(LookupError, match="unresolvable"):
        resolver.resolve("retired-store")
