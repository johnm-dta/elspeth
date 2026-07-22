"""Independent vectors for the closed audit-export derivation boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from io import BytesIO

import pytest

from elspeth.contracts.audit_export import (
    AUDIT_EXPORT_DERIVATION_VERSION,
    REF,
    AuditExportContentDescriptor,
    AuditExportContentStoreResolver,
    AuditExportDerivationConfig,
    C,
    H,
    derive_audit_export_bundle,
    derive_audit_export_bundle_to_spool,
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


def _golden_config(
    *,
    signing_mode: str = "unsigned",
    signer_key_id: str = "UNSIGNED",
    signing_key: bytes | None = None,
    per_chunk_record_limit: int = 1_000,
) -> AuditExportDerivationConfig:
    return AuditExportDerivationConfig(
        source_run_id="run-golden-001",
        source_status="completed",
        source_completed_at="2026-07-16T12:00:00.000001Z",
        export_format="json",
        exporter_version="landscape-exporter-v1",
        serialization_version="audit-export-v2",
        chunking_algorithm_version="record-framing-v1",
        include_raw_error_rows=False,
        per_chunk_byte_limit=1_048_576,
        per_chunk_record_limit=per_chunk_record_limit,
        signing_mode=signing_mode,  # type: ignore[arg-type]
        signer_key_id=signer_key_id,
        signing_key=signing_key,
    )


def _golden_record() -> dict[str, object]:
    return {
        "completed_at": "2026-07-16T12:00:00.000001Z",
        "record_type": "run",
        "run_id": "run-golden-001",
        "status": "completed",
    }


def test_unsigned_end_to_end_literal_derivation_vector() -> None:
    completed_at = "2026-07-16T12:00:00.000001Z"
    bundle = derive_audit_export_bundle((_golden_record(),), _golden_config())

    assert bundle.exported_at == completed_at
    assert bundle.source_completed_at == completed_at
    assert bundle.public_export_config_bytes == PUBLIC_CONFIG_BYTES
    assert bundle.public_export_config_hash == PUBLIC_CONFIG_HASH
    assert bundle.registry_key_bytes == REGISTRY_KEY_BYTES
    assert bundle.registry_key_hash == REGISTRY_KEY_HASH
    assert bundle.unsigned_record_bytes == (
        b'{"completed_at":"2026-07-16T12:00:00.000001Z","record_type":"run","run_id":"run-golden-001","status":"completed"}',
    )
    assert bundle.record_frames == (
        b'{"completed_at":"2026-07-16T12:00:00.000001Z","record_type":"run","run_id":"run-golden-001","status":"completed"}\n',
    )
    assert bundle.chunks[0].descriptor == AuditExportContentDescriptor(
        content_ref="sha256:85883c12e6e3754e1861b2fcdf93a2ee6ceb15945dbf584d71e795b07bdd8d1e",
        content_hash="85883c12e6e3754e1861b2fcdf93a2ee6ceb15945dbf584d71e795b07bdd8d1e",
        size_bytes=114,
        object_kind="data_chunk",
    )
    assert bundle.snapshot_content_bytes == (
        b'{"payload":{"chunking_algorithm_version":"record-framing-v1","chunks":['
        b'{"content_hash":"85883c12e6e3754e1861b2fcdf93a2ee6ceb15945dbf584d71e795b07bdd8d1e",'
        b'"cumulative_bytes":114,"cumulative_records":1,"ordinal":0,"record_count":1,"size_bytes":114}],'
        b'"record_count":1,"serialization_version":"audit-export-v2","total_bytes":114},'
        b'"schema":"audit-export-snapshot-content-v1"}'
    )
    assert bundle.snapshot_hash == "1b22e09b5a4ece941519c9c438181b395deeb94bdb7cbaa0fe851d75e788d5ff"
    assert bundle.snapshot_id_bytes == (
        b'{"payload":{"registry_key_hash":"5726ec70871545ac8f28360b2da7da568cbb45e68a7c3ac6d41873a6bdfda06d",'
        b'"snapshot_hash":"1b22e09b5a4ece941519c9c438181b395deeb94bdb7cbaa0fe851d75e788d5ff"},'
        b'"schema":"audit-export-snapshot-id-v1"}'
    )
    assert bundle.snapshot_id == "f6c08911e8469ea5408deae72b61454a891eb1b7271d3bbb1404baf57f03e605"
    assert bundle.chunks[0].chunk_seal_bytes == (
        b'{"payload":{"chunking_algorithm_version":"record-framing-v1",'
        b'"content_hash":"85883c12e6e3754e1861b2fcdf93a2ee6ceb15945dbf584d71e795b07bdd8d1e",'
        b'"content_ref":"sha256:85883c12e6e3754e1861b2fcdf93a2ee6ceb15945dbf584d71e795b07bdd8d1e",'
        b'"cumulative_bytes":114,"cumulative_records":1,"derivation_version":"audit-export-derivation-v1",'
        b'"ordinal":0,"predecessor":{"kind":"genesis"},"record_count":1,'
        b'"serialization_version":"audit-export-v2","size_bytes":114,'
        b'"snapshot_id":"f6c08911e8469ea5408deae72b61454a891eb1b7271d3bbb1404baf57f03e605"},'
        b'"schema":"audit-export-chunk-seal-v1"}'
    )
    assert bundle.chunks[0].chunk_seal_hash == "25d81e6f65274cd41c9b7ec7c6dedfff650696d8cde9902e012f1b1d3c8fb1c4"
    assert bundle.chunk_manifest_bytes == (
        b'{"payload":{"chunk_count":1,"chunks":['
        b'{"chunk_seal_hash":"25d81e6f65274cd41c9b7ec7c6dedfff650696d8cde9902e012f1b1d3c8fb1c4",'
        b'"content_hash":"85883c12e6e3754e1861b2fcdf93a2ee6ceb15945dbf584d71e795b07bdd8d1e",'
        b'"content_ref":"sha256:85883c12e6e3754e1861b2fcdf93a2ee6ceb15945dbf584d71e795b07bdd8d1e",'
        b'"cumulative_bytes":114,"cumulative_records":1,"ordinal":0,"predecessor_seal_hash":null,'
        b'"record_count":1,"size_bytes":114}],"record_count":1,'
        b'"snapshot_id":"f6c08911e8469ea5408deae72b61454a891eb1b7271d3bbb1404baf57f03e605",'
        b'"total_bytes":114},"schema":"audit-export-manifest-v1"}'
    )
    assert bundle.manifest_hash == "ae602e41dad5d16af48df9b918346602cb16d52fcdb8f1a704332ce317435ed6"
    assert bundle.snapshot_seal_bytes == (
        b'{"payload":{"chunk_count":1,"chunking_algorithm_version":"record-framing-v1",'
        b'"exported_at":"2026-07-16T12:00:00.000001Z",'
        b'"last_chunk_seal_hash":"25d81e6f65274cd41c9b7ec7c6dedfff650696d8cde9902e012f1b1d3c8fb1c4",'
        b'"manifest_hash":"ae602e41dad5d16af48df9b918346602cb16d52fcdb8f1a704332ce317435ed6",'
        b'"per_chunk_byte_limit":1048576,"per_chunk_record_limit":1000,"record_count":1,'
        b'"registry_key_hash":"5726ec70871545ac8f28360b2da7da568cbb45e68a7c3ac6d41873a6bdfda06d",'
        b'"snapshot_hash":"1b22e09b5a4ece941519c9c438181b395deeb94bdb7cbaa0fe851d75e788d5ff",'
        b'"snapshot_id":"f6c08911e8469ea5408deae72b61454a891eb1b7271d3bbb1404baf57f03e605",'
        b'"source_completed_at":"2026-07-16T12:00:00.000001Z","source_run_id":"run-golden-001",'
        b'"source_status":"completed","total_bytes":114},"schema":"audit-export-snapshot-seal-v1"}'
    )
    assert bundle.snapshot_seal_hash == "2c1fac75acf1a2a74e8c90f7f9eaad6bb68e2cb02ff391237b4c356d53776040"
    assert bundle.final_hash == "23ad15b5ddb5391a65605774122036f6dd889e9230d0e167ce1d527902a54589"
    assert bundle.signing_body == (
        b'{"payload":{"chunk_count":1,"derivation_version":"audit-export-derivation-v1",'
        b'"export_format":"json","exported_at":"2026-07-16T12:00:00.000001Z",'
        b'"final_hash":"23ad15b5ddb5391a65605774122036f6dd889e9230d0e167ce1d527902a54589",'
        b'"hash_algorithm":"sha256",'
        b'"last_chunk_seal_hash":"25d81e6f65274cd41c9b7ec7c6dedfff650696d8cde9902e012f1b1d3c8fb1c4",'
        b'"manifest_hash":"ae602e41dad5d16af48df9b918346602cb16d52fcdb8f1a704332ce317435ed6",'
        b'"record_chain_algorithm":"sha256_concat_record_sha256_v1","record_count":1,'
        b'"record_type":"manifest",'
        b'"registry_key_hash":"5726ec70871545ac8f28360b2da7da568cbb45e68a7c3ac6d41873a6bdfda06d",'
        b'"run_id":"run-golden-001","schema":"elspeth.audit-export-manifest.v2",'
        b'"signature_algorithm":"unsigned","signature_key_id":"UNSIGNED",'
        b'"snapshot_hash":"1b22e09b5a4ece941519c9c438181b395deeb94bdb7cbaa0fe851d75e788d5ff",'
        b'"snapshot_id":"f6c08911e8469ea5408deae72b61454a891eb1b7271d3bbb1404baf57f03e605",'
        b'"snapshot_seal_hash":"2c1fac75acf1a2a74e8c90f7f9eaad6bb68e2cb02ff391237b4c356d53776040",'
        b'"source_completed_at":"2026-07-16T12:00:00.000001Z","source_status":"completed",'
        b'"total_bytes":114},"schema":"audit-export-final-manifest-signing-body-v2"}'
    )
    assert bundle.signed_manifest_bytes == (
        b'{"chunk_count":1,"derivation_version":"audit-export-derivation-v1","export_format":"json",'
        b'"exported_at":"2026-07-16T12:00:00.000001Z",'
        b'"final_hash":"23ad15b5ddb5391a65605774122036f6dd889e9230d0e167ce1d527902a54589",'
        b'"hash_algorithm":"sha256",'
        b'"last_chunk_seal_hash":"25d81e6f65274cd41c9b7ec7c6dedfff650696d8cde9902e012f1b1d3c8fb1c4",'
        b'"manifest_hash":"ae602e41dad5d16af48df9b918346602cb16d52fcdb8f1a704332ce317435ed6",'
        b'"record_chain_algorithm":"sha256_concat_record_sha256_v1","record_count":1,'
        b'"record_type":"manifest",'
        b'"registry_key_hash":"5726ec70871545ac8f28360b2da7da568cbb45e68a7c3ac6d41873a6bdfda06d",'
        b'"run_id":"run-golden-001","schema":"elspeth.audit-export-manifest.v2","signature":null,'
        b'"signature_algorithm":"unsigned","signature_key_id":"UNSIGNED",'
        b'"snapshot_hash":"1b22e09b5a4ece941519c9c438181b395deeb94bdb7cbaa0fe851d75e788d5ff",'
        b'"snapshot_id":"f6c08911e8469ea5408deae72b61454a891eb1b7271d3bbb1404baf57f03e605",'
        b'"snapshot_seal_hash":"2c1fac75acf1a2a74e8c90f7f9eaad6bb68e2cb02ff391237b4c356d53776040",'
        b'"source_completed_at":"2026-07-16T12:00:00.000001Z","source_status":"completed",'
        b'"total_bytes":114}'
    )
    assert bundle.signed_manifest.content_hash == "6ece028d51231229a870823bd4c775b5e8086d12e7f6d72462088ea88f19273d"
    assert bundle.signed_manifest.content_ref == ("sha256:6ece028d51231229a870823bd4c775b5e8086d12e7f6d72462088ea88f19273d")
    assert bundle.signed_manifest.size_bytes == 1100
    assert bundle.json_target_bytes == b"".join(bundle.chunk_bytes) + bundle.signed_manifest_bytes
    assert bundle.final_manifest["signature"] is None


def test_hmac_end_to_end_literal_derivation_vector() -> None:
    bundle = derive_audit_export_bundle(
        (_golden_record(),),
        _golden_config(
            signing_mode="hmac_sha256",
            signer_key_id="operator-key-v1",
            signing_key=b"golden-key",
        ),
    )

    assert bundle.public_export_config_hash == "da0ad18d78cea923e62f5e4db2681f9f329839e5e104ed569087f2d51b5f5144"
    assert bundle.registry_key_hash == "97e57909a1f03821db3d98ad0abf88e281f3f9b21a8678f04860d1bbdd61b835"
    assert bundle.unsigned_record_bytes == (
        b'{"completed_at":"2026-07-16T12:00:00.000001Z","record_type":"run","run_id":"run-golden-001","status":"completed"}',
    )
    assert bundle.record_objects[0]["signature"] == "103133ca66b940d225b799e5311d253a822c5a1469a0931cca2a114e07a2543b"
    assert bundle.record_frames == (
        b'{"completed_at":"2026-07-16T12:00:00.000001Z","record_type":"run",'
        b'"run_id":"run-golden-001",'
        b'"signature":"103133ca66b940d225b799e5311d253a822c5a1469a0931cca2a114e07a2543b",'
        b'"status":"completed"}\n',
    )
    assert bundle.chunks[0].descriptor.content_hash == "df943880fa1bdfa03b300a13dcd67fd59cc6921e28756025dd97520e837fa70a"
    assert bundle.snapshot_hash == "7e2769b75d78420c87ad3c941517d19f23a710f1a793e6652612b3fb0592bd8e"
    assert bundle.snapshot_id == "9b7d0994fdeccf13283750dc2281943128bd7136c3b90aec54e1b194cd4c52cb"
    assert bundle.chunks[0].chunk_seal_hash == "6e5b1c39c90af166ac272f388a2bcf9dc8863aa80df98432d18d72c3e3003fd4"
    assert bundle.manifest_hash == "0dea48a8a0077a7c8218e0a6e8970ac8a54c38640b748c788d0170bd12d65e83"
    assert bundle.snapshot_seal_hash == "069c42e3d02b558a323e57a960177e8f1d760c151e61e09aff2e70006d2330fa"
    assert bundle.final_hash == "061084b57b7b3d40295e5699a22e685d86ef95c2f492c34aad2fdb62564c426e"
    assert bundle.final_manifest["signature"] == "4db2df3c8d6aed706956d9a5bd471054025bd44483b388d632105bb8f2f45d83"
    assert bundle.signed_manifest_bytes == (
        b'{"chunk_count":1,"derivation_version":"audit-export-derivation-v1","export_format":"json",'
        b'"exported_at":"2026-07-16T12:00:00.000001Z",'
        b'"final_hash":"061084b57b7b3d40295e5699a22e685d86ef95c2f492c34aad2fdb62564c426e",'
        b'"hash_algorithm":"sha256",'
        b'"last_chunk_seal_hash":"6e5b1c39c90af166ac272f388a2bcf9dc8863aa80df98432d18d72c3e3003fd4",'
        b'"manifest_hash":"0dea48a8a0077a7c8218e0a6e8970ac8a54c38640b748c788d0170bd12d65e83",'
        b'"record_chain_algorithm":"sha256_concat_hmac_sha256_signatures_v1","record_count":1,'
        b'"record_type":"manifest",'
        b'"registry_key_hash":"97e57909a1f03821db3d98ad0abf88e281f3f9b21a8678f04860d1bbdd61b835",'
        b'"run_id":"run-golden-001","schema":"elspeth.audit-export-manifest.v2",'
        b'"signature":"4db2df3c8d6aed706956d9a5bd471054025bd44483b388d632105bb8f2f45d83",'
        b'"signature_algorithm":"hmac_sha256","signature_key_id":"operator-key-v1",'
        b'"snapshot_hash":"7e2769b75d78420c87ad3c941517d19f23a710f1a793e6652612b3fb0592bd8e",'
        b'"snapshot_id":"9b7d0994fdeccf13283750dc2281943128bd7136c3b90aec54e1b194cd4c52cb",'
        b'"snapshot_seal_hash":"069c42e3d02b558a323e57a960177e8f1d760c151e61e09aff2e70006d2330fa",'
        b'"source_completed_at":"2026-07-16T12:00:00.000001Z","source_status":"completed",'
        b'"total_bytes":193}'
    )
    assert bundle.signed_manifest.content_hash == "cd9ed845731e32330b38b45ccd387d85aed5fcddf0ca9b800c32253ebb8eaa89"
    assert bundle.signed_manifest.content_ref == ("sha256:cd9ed845731e32330b38b45ccd387d85aed5fcddf0ca9b800c32253ebb8eaa89")
    assert bundle.signed_manifest.size_bytes == 1181


@pytest.mark.parametrize("signed", [False, True])
def test_spooled_derivation_writes_completed_chunks_before_consuming_all_records(signed: bool) -> None:
    spool = BytesIO()
    config = (
        _golden_config(
            signing_mode="hmac_sha256",
            signer_key_id="audit-key-v1",
            signing_key=b"golden-test-key",
            per_chunk_record_limit=1,
        )
        if signed
        else _golden_config(per_chunk_record_limit=1)
    )
    second = {**_golden_record(), "run_id": "run-golden-002"}

    def records():
        yield _golden_record()
        assert spool.tell() > 0
        yield second

    spooled = derive_audit_export_bundle_to_spool(
        records(),
        config,
        spool,
        max_total_records=2,
        max_total_bytes=4096,
        max_chunks=2,
    )
    materialized = derive_audit_export_bundle((_golden_record(), second), config)

    assert spooled.snapshot_id == materialized.snapshot_id
    assert spooled.snapshot_hash == materialized.snapshot_hash
    assert spooled.manifest_hash == materialized.manifest_hash
    assert spooled.snapshot_seal_hash == materialized.snapshot_seal_hash
    assert spooled.final_hash == materialized.final_hash
    assert spooled.record_chain_algorithm == materialized.record_chain_algorithm
    assert spooled.signing_body == materialized.signing_body
    assert spooled.signed_manifest_bytes == materialized.signed_manifest_bytes
    assert tuple(spool.getvalue()[offset : offset + size] for offset, size in spooled.chunk_offsets) == materialized.chunk_bytes
    assert [chunk.chunk_seal_hash for chunk in spooled.chunks] == [chunk.chunk_seal_hash for chunk in materialized.chunks]
    assert "record_frames" not in type(spooled).__dataclass_fields__
    assert "content" not in type(spooled.chunks[0]).__dataclass_fields__
