"""Closed audit-export derivation and durable content-store contracts.

This module is deliberately L0-only.  It defines the exact RFC 8785 tagged
payload boundary used by snapshot/effect identity and the capabilities a
durable store must expose.  It never resolves credentials or opens arbitrary
content references on behalf of an adapter.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Final, Literal, Protocol, runtime_checkable

from elspeth.contracts.hashing import canonical_json

AUDIT_EXPORT_DERIVATION_VERSION: Final = "audit-export-derivation-v1"
AUDIT_EXPORT_SERIALIZATION_VERSION: Final = "audit-export-v2"
AUDIT_EXPORT_MANIFEST_SCHEMA: Final = "elspeth.audit-export-manifest.v2"
AUDIT_EXPORT_MAX_CHUNKS: Final = 100_000
AUDIT_EXPORT_MAX_CHUNK_BYTES: Final = 64 * 1024 * 1024
AUDIT_EXPORT_MAX_CHUNK_RECORDS: Final = 1_000_000
AUDIT_EXPORT_MAX_TOTAL_BYTES: Final = 1024 * 1024 * 1024 * 1024
AUDIT_EXPORT_MAX_TOTAL_RECORDS: Final = 100_000_000
MAX_AUDIT_EXPORT_SIGNED_MANIFEST_BYTES: Final = 64 * 1024
AUDIT_EXPORT_SAFE_INTEGER_MAX: Final = 9_007_199_254_740_991

type ClosedAuditExportScalar = str | bool | int | None
type ClosedAuditExportJSON = ClosedAuditExportScalar | list[ClosedAuditExportJSON] | dict[str, ClosedAuditExportJSON]
type AuditExportObjectKind = Literal["data_chunk", "final_manifest"]

_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_UTC_MICROSECOND_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}\Z")


def _object(value: object, *, fields: frozenset[str], path: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{path} must be an exact string-keyed object")
    result = value
    assert isinstance(result, dict)
    if any(type(key) is not str for key in result):
        raise TypeError(f"{path} requires string keys")
    keys = frozenset(result)
    if keys != fields:
        missing = sorted(fields - keys)
        unknown = sorted(str(key) for key in keys - fields)
        raise ValueError(f"{path} fields are not exhaustive: missing={missing}, unknown={unknown}")
    return result


def _list(value: object, path: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{path} must be an ordered list")
    result = value
    assert isinstance(result, list)
    return result


def _string(value: object, path: str, *, allowed: frozenset[str] | None = None) -> str:
    if isinstance(value, Enum):
        raise TypeError(f"{path} must be converted from enum to an exact string before C")
    if type(value) is not str or not value:
        raise TypeError(f"{path} must be a non-empty exact string")
    result = value
    assert isinstance(result, str)
    if allowed is not None and result not in allowed:
        raise ValueError(f"{path} must be one of {sorted(allowed)}")
    return result


def _integer(value: object, path: str, *, minimum: int = 0, maximum: int = AUDIT_EXPORT_SAFE_INTEGER_MAX) -> int:
    if type(value) is not int:
        if isinstance(value, float):
            raise TypeError(f"{path} floats are forbidden")
        raise TypeError(f"{path} must be an exact integer")
    result = value
    assert isinstance(result, int)
    if result < minimum or result > maximum:
        raise ValueError(f"{path} integer must be within [{minimum}, {maximum}]")
    return result


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{path} must be an exact boolean")
    return bool(value)


def _hash(value: object, path: str) -> str:
    result = _string(value, path)
    if _LOWER_HEX_64.fullmatch(result) is None:
        raise ValueError(f"{path} must be lowercase 64-character hexadecimal")
    return result


def _ref(value: object, path: str, *, content_hash: str | None = None) -> str:
    result = _string(value, path)
    suffix = result.removeprefix("sha256:")
    if not result.startswith("sha256:") or _LOWER_HEX_64.fullmatch(suffix) is None:
        raise ValueError(f"{path} must be sha256:<lowercase-64-hex>")
    if content_hash is not None and result != f"sha256:{content_hash}":
        raise ValueError(f"{path} must equal 'sha256:{content_hash}'")
    return result


def _timestamp(value: object, path: str) -> str:
    result = _string(value, path)
    if _UTC_MICROSECOND_TIMESTAMP.fullmatch(result) is None:
        raise ValueError(f"{path} must be an exact UTC microsecond timestamp")
    try:
        datetime.strptime(result, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"{path} must be a valid UTC timestamp") from exc
    return result


def _validate_public_config(payload: object) -> None:
    fields = frozenset(
        {
            "chunking_algorithm_version",
            "export_format",
            "exporter_version",
            "include_raw_error_rows",
            "per_chunk_byte_limit",
            "per_chunk_record_limit",
            "serialization_version",
            "signer_key_id",
            "signing_mode",
        }
    )
    obj = _object(payload, fields=fields, path="public config")
    _string(obj["chunking_algorithm_version"], "chunking_algorithm_version")
    _string(obj["export_format"], "export_format", allowed=frozenset({"json", "csv"}))
    _string(obj["exporter_version"], "exporter_version")
    _boolean(obj["include_raw_error_rows"], "include_raw_error_rows")
    _integer(obj["per_chunk_byte_limit"], "per_chunk_byte_limit", minimum=1, maximum=AUDIT_EXPORT_MAX_CHUNK_BYTES)
    _integer(
        obj["per_chunk_record_limit"],
        "per_chunk_record_limit",
        minimum=1,
        maximum=AUDIT_EXPORT_MAX_CHUNK_RECORDS,
    )
    _string(obj["serialization_version"], "serialization_version")
    signer = _string(obj["signer_key_id"], "signer_key_id")
    mode = _string(obj["signing_mode"], "signing_mode", allowed=frozenset({"unsigned", "hmac_sha256"}))
    _validate_signer(mode, signer, None, allow_signature_none=True)


def _validate_registry_key(payload: object) -> None:
    fields = frozenset(
        {
            "export_format",
            "exporter_version",
            "public_export_config_hash",
            "serialization_version",
            "signer_key_id",
            "signing_mode",
            "source_run_id",
        }
    )
    obj = _object(payload, fields=fields, path="registry key")
    _string(obj["export_format"], "export_format", allowed=frozenset({"json", "csv"}))
    _string(obj["exporter_version"], "exporter_version")
    _hash(obj["public_export_config_hash"], "public_export_config_hash")
    _string(obj["serialization_version"], "serialization_version")
    signer = _string(obj["signer_key_id"], "signer_key_id")
    mode = _string(obj["signing_mode"], "signing_mode", allowed=frozenset({"unsigned", "hmac_sha256"}))
    _validate_signer(mode, signer, None, allow_signature_none=True)
    _string(obj["source_run_id"], "source_run_id")


def _validate_snapshot_content(payload: object) -> None:
    obj = _object(
        payload,
        fields=frozenset({"chunking_algorithm_version", "chunks", "record_count", "serialization_version", "total_bytes"}),
        path="snapshot content",
    )
    _string(obj["chunking_algorithm_version"], "chunking_algorithm_version")
    for index, raw in enumerate(_list(obj["chunks"], "chunks")):
        chunk = _object(
            raw,
            fields=frozenset({"content_hash", "cumulative_bytes", "cumulative_records", "ordinal", "record_count", "size_bytes"}),
            path=f"chunks[{index}]",
        )
        _hash(chunk["content_hash"], f"chunks[{index}].content_hash")
        for field in ("cumulative_bytes", "cumulative_records", "record_count", "size_bytes"):
            _integer(chunk[field], f"chunks[{index}].{field}", minimum=1)
        _integer(chunk["ordinal"], f"chunks[{index}].ordinal")
    _integer(obj["record_count"], "record_count", minimum=1)
    _string(obj["serialization_version"], "serialization_version")
    _integer(obj["total_bytes"], "total_bytes", minimum=1)


def _validate_snapshot_id(payload: object) -> None:
    obj = _object(payload, fields=frozenset({"registry_key_hash", "snapshot_hash"}), path="snapshot ID")
    _hash(obj["registry_key_hash"], "registry_key_hash")
    _hash(obj["snapshot_hash"], "snapshot_hash")


def _validate_chunk_seal(payload: object) -> None:
    obj = _object(
        payload,
        fields=frozenset(
            {
                "chunking_algorithm_version",
                "content_hash",
                "content_ref",
                "cumulative_bytes",
                "cumulative_records",
                "derivation_version",
                "ordinal",
                "predecessor",
                "record_count",
                "serialization_version",
                "size_bytes",
                "snapshot_id",
            }
        ),
        path="chunk seal",
    )
    _string(obj["chunking_algorithm_version"], "chunking_algorithm_version")
    content_hash = _hash(obj["content_hash"], "content_hash")
    _ref(obj["content_ref"], "content_ref", content_hash=content_hash)
    for field in ("cumulative_bytes", "cumulative_records", "record_count", "size_bytes"):
        _integer(obj[field], field, minimum=1)
    _string(obj["derivation_version"], "derivation_version", allowed=frozenset({AUDIT_EXPORT_DERIVATION_VERSION}))
    _integer(obj["ordinal"], "ordinal")
    predecessor = obj["predecessor"]
    if type(predecessor) is not dict:
        raise TypeError("predecessor must be an exact object")
    assert isinstance(predecessor, dict)
    if predecessor.get("kind") == "genesis":
        _object(predecessor, fields=frozenset({"kind"}), path="predecessor")
    else:
        pred = _object(predecessor, fields=frozenset({"hash", "kind"}), path="predecessor")
        _string(pred["kind"], "predecessor.kind", allowed=frozenset({"chunk_seal"}))
        _hash(pred["hash"], "predecessor.hash")
    _string(obj["serialization_version"], "serialization_version")
    _hash(obj["snapshot_id"], "snapshot_id")


def _validate_chunk_manifest(payload: object) -> None:
    obj = _object(
        payload,
        fields=frozenset({"chunk_count", "chunks", "record_count", "snapshot_id", "total_bytes"}),
        path="chunk manifest",
    )
    _integer(obj["chunk_count"], "chunk_count", minimum=1)
    for index, raw in enumerate(_list(obj["chunks"], "chunks")):
        chunk = _object(
            raw,
            fields=frozenset(
                {
                    "chunk_seal_hash",
                    "content_hash",
                    "content_ref",
                    "cumulative_bytes",
                    "cumulative_records",
                    "ordinal",
                    "predecessor_seal_hash",
                    "record_count",
                    "size_bytes",
                }
            ),
            path=f"chunks[{index}]",
        )
        _hash(chunk["chunk_seal_hash"], f"chunks[{index}].chunk_seal_hash")
        content_hash = _hash(chunk["content_hash"], f"chunks[{index}].content_hash")
        _ref(chunk["content_ref"], f"chunks[{index}].content_ref", content_hash=content_hash)
        for field in ("cumulative_bytes", "cumulative_records", "record_count", "size_bytes"):
            _integer(chunk[field], f"chunks[{index}].{field}", minimum=1)
        _integer(chunk["ordinal"], f"chunks[{index}].ordinal")
        if chunk["predecessor_seal_hash"] is not None:
            _hash(chunk["predecessor_seal_hash"], f"chunks[{index}].predecessor_seal_hash")
    _integer(obj["record_count"], "record_count", minimum=1)
    _hash(obj["snapshot_id"], "snapshot_id")
    _integer(obj["total_bytes"], "total_bytes", minimum=1)


def _validate_snapshot_seal(payload: object) -> None:
    obj = _object(
        payload,
        fields=frozenset(
            {
                "chunk_count",
                "chunking_algorithm_version",
                "exported_at",
                "last_chunk_seal_hash",
                "manifest_hash",
                "per_chunk_byte_limit",
                "per_chunk_record_limit",
                "record_count",
                "registry_key_hash",
                "snapshot_hash",
                "snapshot_id",
                "source_completed_at",
                "source_run_id",
                "source_status",
                "total_bytes",
            }
        ),
        path="snapshot seal",
    )
    for field in ("chunk_count", "per_chunk_byte_limit", "per_chunk_record_limit", "record_count", "total_bytes"):
        _integer(obj[field], field, minimum=1)
    _string(obj["chunking_algorithm_version"], "chunking_algorithm_version")
    _timestamp(obj["exported_at"], "exported_at")
    for field in ("last_chunk_seal_hash", "manifest_hash", "registry_key_hash", "snapshot_hash", "snapshot_id"):
        _hash(obj[field], field)
    _timestamp(obj["source_completed_at"], "source_completed_at")
    _string(obj["source_run_id"], "source_run_id")
    _string(obj["source_status"], "source_status", allowed=frozenset({"completed", "completed_with_failures", "empty"}))


def _validate_signer(mode: str, signer: str, signature: object, *, allow_signature_none: bool = False) -> None:
    if mode == "unsigned":
        if signer != "UNSIGNED":
            raise ValueError("unsigned signer_key_id must equal 'UNSIGNED'")
        if signature is not None and not allow_signature_none:
            raise ValueError("unsigned signature must be null")
    else:
        if signer == "UNSIGNED":
            raise ValueError("HMAC signer_key_id must not use reserved 'UNSIGNED'")
        validate_credential_free_identifier(signer, "signer_key_id")
        if signature is not None:
            _hash(signature, "signature")


def _validate_final_manifest_identity(payload: object) -> None:
    obj = _object(
        payload,
        fields=frozenset(
            {
                "content_hash",
                "content_ref",
                "derivation_version",
                "final_hash",
                "manifest_schema",
                "record_chain_algorithm",
                "signature",
                "signature_algorithm",
                "signature_key_id",
                "size_bytes",
            }
        ),
        path="final manifest identity",
    )
    content_hash = _hash(obj["content_hash"], "content_hash")
    _ref(obj["content_ref"], "content_ref", content_hash=content_hash)
    _string(obj["derivation_version"], "derivation_version", allowed=frozenset({AUDIT_EXPORT_DERIVATION_VERSION}))
    _hash(obj["final_hash"], "final_hash")
    _string(obj["manifest_schema"], "manifest_schema", allowed=frozenset({AUDIT_EXPORT_MANIFEST_SCHEMA}))
    mode = _string(obj["signature_algorithm"], "signature_algorithm", allowed=frozenset({"unsigned", "hmac_sha256"}))
    expected_chain = "sha256_concat_record_sha256_v1" if mode == "unsigned" else "sha256_concat_hmac_sha256_signatures_v1"
    _string(obj["record_chain_algorithm"], "record_chain_algorithm", allowed=frozenset({expected_chain}))
    signer = _string(obj["signature_key_id"], "signature_key_id")
    _validate_signer(mode, signer, obj["signature"])
    _integer(obj["size_bytes"], "size_bytes", minimum=1, maximum=MAX_AUDIT_EXPORT_SIGNED_MANIFEST_BYTES)


def _validate_effect_identity(payload: object) -> None:
    obj = _object(
        payload,
        fields=frozenset(
            {
                "export_format",
                "final_manifest_identity_hash",
                "input_kind",
                "manifest_hash",
                "protocol_version",
                "registry_key_hash",
                "role",
                "serialization_version",
                "signer_key_id",
                "signing_mode",
                "sink_node_id",
                "snapshot_hash",
                "snapshot_id",
                "source_run_id",
                "target_config_hash",
            }
        ),
        path="audit export effect identity",
    )
    _string(obj["export_format"], "export_format", allowed=frozenset({"json", "csv"}))
    for field in (
        "final_manifest_identity_hash",
        "manifest_hash",
        "registry_key_hash",
        "snapshot_hash",
        "snapshot_id",
        "target_config_hash",
    ):
        _hash(obj[field], field)
    _string(obj["input_kind"], "input_kind", allowed=frozenset({"audit_export_snapshot"}))
    _string(obj["protocol_version"], "protocol_version", allowed=frozenset({"sink-effect-v1"}))
    _string(obj["role"], "role", allowed=frozenset({"primary", "failsink"}))
    _string(obj["serialization_version"], "serialization_version")
    signer = _string(obj["signer_key_id"], "signer_key_id")
    mode = _string(obj["signing_mode"], "signing_mode", allowed=frozenset({"unsigned", "hmac_sha256"}))
    _validate_signer(mode, signer, None, allow_signature_none=True)
    _string(obj["sink_node_id"], "sink_node_id")
    _string(obj["source_run_id"], "source_run_id")


_SCHEMA_VALIDATORS: Final[dict[str, Callable[[object], None]]] = {
    "audit-export-public-config-v1": _validate_public_config,
    "audit-export-registry-key-v1": _validate_registry_key,
    "audit-export-snapshot-content-v1": _validate_snapshot_content,
    "audit-export-snapshot-id-v1": _validate_snapshot_id,
    "audit-export-chunk-seal-v1": _validate_chunk_seal,
    "audit-export-manifest-v1": _validate_chunk_manifest,
    "audit-export-snapshot-seal-v1": _validate_snapshot_seal,
    "sink-effect-audit-export-final-manifest-v1": _validate_final_manifest_identity,
    "sink-effect-audit-export-effect-v1": _validate_effect_identity,
}


def validate_closed_stage_payload(tag: str, payload: ClosedAuditExportJSON) -> None:
    """Validate one exhaustive tagged audit-export payload without coercion."""
    if type(tag) is not str:
        raise TypeError("audit-export schema tag must be an exact string")
    try:
        validator = _SCHEMA_VALIDATORS[tag]
    except KeyError as exc:
        raise ValueError(f"unknown audit-export schema tag: {tag!r}") from exc
    validator(payload)


def C(tag: str, payload: ClosedAuditExportJSON) -> bytes:
    """Return exact RFC 8785 bytes for one closed, tagged payload."""
    validate_closed_stage_payload(tag, payload)
    return canonical_json({"payload": payload, "schema": tag}).encode("utf-8")


def H(data: bytes) -> str:
    """Hash exact supplied bytes; never normalize or canonicalize them."""
    if type(data) is not bytes:
        raise TypeError("H data must be exact bytes")
    return hashlib.sha256(data).hexdigest()


def REF(content_hash: str) -> str:
    """Return the global credential-free reference for an exact digest."""
    return f"sha256:{_hash(content_hash, 'content_hash')}"


def derive_public_export_config_hash(payload: ClosedAuditExportJSON) -> str:
    return H(C("audit-export-public-config-v1", payload))


def derive_registry_key_hash(payload: ClosedAuditExportJSON) -> str:
    return H(C("audit-export-registry-key-v1", payload))


def validate_credential_free_identifier(value: str, field_name: str) -> str:
    """Validate an operator-visible identifier that cannot carry credentials."""
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a bounded credential-free identifier")
    return value


def validate_content_namespace(value: str) -> str:
    if type(value) is not str or _NAMESPACE.fullmatch(value) is None or ".." in value.split("/"):
        raise ValueError("namespace must be a bounded credential-free relative namespace")
    return value


@dataclass(frozen=True, slots=True)
class AuditExportContentDescriptor:
    """Immutable globally addressed audit-export object descriptor."""

    content_ref: str
    content_hash: str
    size_bytes: int
    object_kind: AuditExportObjectKind

    def __post_init__(self) -> None:
        content_hash = _hash(self.content_hash, "content_hash")
        _ref(self.content_ref, "content_ref", content_hash=content_hash)
        _integer(self.size_bytes, "size_bytes", minimum=1, maximum=AUDIT_EXPORT_MAX_CHUNK_BYTES)
        if self.object_kind not in {"data_chunk", "final_manifest"}:
            raise ValueError("object_kind must be data_chunk or final_manifest")
        if self.object_kind == "final_manifest" and self.size_bytes > MAX_AUDIT_EXPORT_SIGNED_MANIFEST_BYTES:
            raise ValueError("final manifest exceeds the code-owned 64 KiB maximum")


@dataclass(frozen=True, slots=True)
class RegisteredAuditExportContent:
    """Landscape-authorized descriptor that a store may open."""

    snapshot_id: str
    content_store_id: str
    namespace: str
    descriptor: AuditExportContentDescriptor

    def __post_init__(self) -> None:
        _hash(self.snapshot_id, "snapshot_id")
        validate_credential_free_identifier(self.content_store_id, "content_store_id")
        validate_content_namespace(self.namespace)
        if type(self.descriptor) is not AuditExportContentDescriptor:
            raise TypeError("descriptor must be AuditExportContentDescriptor")


@dataclass(frozen=True, slots=True)
class AuditExportOrphanCollectionRequest:
    """Candidate-scoped, reference-checked garbage collection authority."""

    candidate_id: str
    namespace: str
    descriptors: tuple[AuditExportContentDescriptor, ...]
    marked_at: datetime
    grace_period_seconds: int
    fresh_winner_reference_check: Callable[[str], bool]

    def __post_init__(self) -> None:
        validate_credential_free_identifier(self.candidate_id, "candidate_id")
        validate_content_namespace(self.namespace)
        if not self.descriptors or any(type(item) is not AuditExportContentDescriptor for item in self.descriptors):
            raise ValueError("descriptors must be a non-empty exact descriptor tuple")
        if self.marked_at.tzinfo is None or self.marked_at.utcoffset() is None:
            raise ValueError("marked_at must be timezone-aware")
        _integer(self.grace_period_seconds, "grace_period_seconds", minimum=1)
        if not callable(self.fresh_winner_reference_check):
            raise TypeError("fresh_winner_reference_check must be callable")


@runtime_checkable
class BoundAuditExportContentReader(Protocol):
    """No-arbitrary-ref reader returned only for registered content."""

    def read(self) -> bytes: ...


@runtime_checkable
class AuditExportContentStore(Protocol):
    """Durable immutable store used by audit-export snapshot winners."""

    @property
    def content_store_id(self) -> str: ...

    @property
    def namespace(self) -> str: ...

    def is_durable(self) -> bool: ...

    def put_immutable(self, content: bytes, *, candidate_id: str, object_kind: AuditExportObjectKind) -> str: ...

    def open_registered(self, registration: RegisteredAuditExportContent) -> BoundAuditExportContentReader: ...

    def mark_candidate_orphans(self, candidate_id: str, descriptors: tuple[AuditExportContentDescriptor, ...]) -> None: ...

    def garbage_collect_candidate(self, request: AuditExportOrphanCollectionRequest) -> bool: ...


class AuditExportContentStoreResolver:
    """Stable resolver for immutable winning ``content_store_id`` provenance."""

    def __init__(self) -> None:
        self._stores: dict[str, AuditExportContentStore] = {}

    def register(self, store: AuditExportContentStore) -> None:
        if not isinstance(store, AuditExportContentStore):
            raise TypeError("store must implement AuditExportContentStore")
        store_id = validate_credential_free_identifier(store.content_store_id, "content_store_id")
        validate_content_namespace(store.namespace)
        if not store.is_durable():
            raise ValueError("audit-export content store must prove durability")
        current = self._stores.get(store_id)
        if current is not None and current is not store:
            raise ValueError(f"content_store_id {store_id!r} is already registered to a different store")
        self._stores[store_id] = store

    def resolve(self, content_store_id: str) -> AuditExportContentStore:
        store_id = validate_credential_free_identifier(content_store_id, "content_store_id")
        try:
            return self._stores[store_id]
        except KeyError as exc:
            raise LookupError(f"winning content_store_id {store_id!r} is unresolvable") from exc


class IterableBoundAuditExportContentReader:
    """Small exact reader useful to store implementations and tests."""

    __slots__ = ("_content",)

    def __init__(self, content: bytes) -> None:
        if type(content) is not bytes:
            raise TypeError("content must be bytes")
        self._content = content

    def read(self) -> bytes:
        return self._content

    def __iter__(self) -> Iterator[bytes]:
        yield self.read()
