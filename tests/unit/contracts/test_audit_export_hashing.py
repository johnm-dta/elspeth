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
    final_manifest_identity_payload,
    hash_final_manifest_identity_payload,
)
from elspeth.contracts.sink_effects import AuditExportSignedManifestInput, AuditExportSigningMode

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


def _manifest_descriptor() -> AuditExportSignedManifestInput:
    return AuditExportSignedManifestInput(
        content_ref=f"sha256:{'a' * 64}",
        content_hash="a" * 64,
        size_bytes=512,
        manifest_schema="elspeth.audit-export-manifest.v2",
        derivation_version="audit-export-derivation-v1",
        signature_algorithm=AuditExportSigningMode.UNSIGNED,
        signature_key_id="UNSIGNED",
        record_chain_algorithm="sha256_concat_record_sha256_v1",
        final_hash="b" * 64,
        signature=None,
    )


@pytest.mark.parametrize(
    "field",
    [
        "content_hash",
        "content_ref",
        "size_bytes",
        "derivation_version",
        "manifest_schema",
        "signature_algorithm",
        "signature_key_id",
        "record_chain_algorithm",
        "final_hash",
        "signature",
    ],
)
def test_final_manifest_identity_component_binds_every_serialized_field(field: str) -> None:
    baseline = final_manifest_identity_payload(_manifest_descriptor())
    changed = dict(baseline)
    replacements: dict[str, object] = {
        "content_hash": "c" * 64,
        "content_ref": f"sha256:{'c' * 64}",
        "size_bytes": 513,
        "derivation_version": "alternate-version",
        "manifest_schema": "alternate-schema",
        "signature_algorithm": "hmac_sha256",
        "signature_key_id": "operator-key-2",
        "record_chain_algorithm": "sha256_concat_hmac_sha256_signatures_v1",
        "final_hash": "d" * 64,
        "signature": "e" * 64,
    }
    changed[field] = replacements[field]
    assert hash_final_manifest_identity_payload(changed, validate=False) != hash_final_manifest_identity_payload(baseline, validate=False)


def test_literal_final_manifest_and_export_effect_identity_vectors() -> None:
    final_payload = final_manifest_identity_payload(_manifest_descriptor())
    final_bytes = (
        b'{"payload":{"content_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"content_ref":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"derivation_version":"audit-export-derivation-v1",'
        b'"final_hash":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        b'"manifest_schema":"elspeth.audit-export-manifest.v2",'
        b'"record_chain_algorithm":"sha256_concat_record_sha256_v1","signature":null,'
        b'"signature_algorithm":"unsigned","signature_key_id":"UNSIGNED","size_bytes":512},'
        b'"schema":"sink-effect-audit-export-final-manifest-v1"}'
    )
    assert C("sink-effect-audit-export-final-manifest-v1", final_payload) == final_bytes
    assert H(final_bytes) == "e2df9077ce2b991cbc0c682fdfdb86e8aa353685a336e4bfcfa5da92bfb732a2"

    effect_payload = {
        "export_format": "json",
        "final_manifest_identity_hash": "e2df9077ce2b991cbc0c682fdfdb86e8aa353685a336e4bfcfa5da92bfb732a2",
        "input_kind": "audit_export_snapshot",
        "manifest_hash": "1" * 64,
        "protocol_version": "sink-effect-v1",
        "registry_key_hash": "2" * 64,
        "role": "primary",
        "serialization_version": "audit-export-v2",
        "signer_key_id": "UNSIGNED",
        "signing_mode": "unsigned",
        "sink_node_id": "audit-export",
        "snapshot_hash": "3" * 64,
        "snapshot_id": "4" * 64,
        "source_run_id": "run-golden-001",
        "target_config_hash": "9" * 64,
    }
    effect_bytes = (
        b'{"payload":{"export_format":"json",'
        b'"final_manifest_identity_hash":"e2df9077ce2b991cbc0c682fdfdb86e8aa353685a336e4bfcfa5da92bfb732a2",'
        b'"input_kind":"audit_export_snapshot",'
        b'"manifest_hash":"1111111111111111111111111111111111111111111111111111111111111111",'
        b'"protocol_version":"sink-effect-v1",'
        b'"registry_key_hash":"2222222222222222222222222222222222222222222222222222222222222222",'
        b'"role":"primary","serialization_version":"audit-export-v2","signer_key_id":"UNSIGNED",'
        b'"signing_mode":"unsigned","sink_node_id":"audit-export",'
        b'"snapshot_hash":"3333333333333333333333333333333333333333333333333333333333333333",'
        b'"snapshot_id":"4444444444444444444444444444444444444444444444444444444444444444",'
        b'"source_run_id":"run-golden-001",'
        b'"target_config_hash":"9999999999999999999999999999999999999999999999999999999999999999"},'
        b'"schema":"sink-effect-audit-export-effect-v1"}'
    )
    assert C("sink-effect-audit-export-effect-v1", effect_payload) == effect_bytes
    assert H(effect_bytes) == "d09466d2cbbb27205cf84de5a6bfddcda6c6c47f42428e1c0e3349dcf39b38bb"
