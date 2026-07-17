"""Closed audit-export derivation and durable content-store contracts.

This module is deliberately L0-only.  It defines the exact RFC 8785 tagged
payload boundary used by snapshot/effect identity and the capabilities a
durable store must expose.  It never resolves credentials or opens arbitrary
content references on behalf of an adapter.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, BinaryIO, Final, Literal, Protocol, cast, runtime_checkable

from elspeth.contracts.audit import AuditExportSnapshot, AuditExportSnapshotChunk
from elspeth.contracts.enums import RunStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import freeze_fields
from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.sink_effects import AuditExportFormat, AuditExportSigningMode

if TYPE_CHECKING:
    from elspeth.contracts.sink_effects import AuditExportSignedManifestInput

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
_EXPORT_TERMINAL_STATUSES: Final = frozenset({"completed", "completed_with_failures", "empty"})
_EXPORT_TERMINAL: Final = frozenset({RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_FAILURES, RunStatus.EMPTY})
_UNSIGNED_RECORD_CHAIN: Final = "sha256_concat_record_sha256_v1"
_HMAC_RECORD_CHAIN: Final = "sha256_concat_hmac_sha256_signatures_v1"
_FINAL_MANIFEST_CORE_FIELDS: Final = frozenset(
    {
        "chunk_count",
        "derivation_version",
        "export_format",
        "exported_at",
        "final_hash",
        "hash_algorithm",
        "last_chunk_seal_hash",
        "manifest_hash",
        "record_chain_algorithm",
        "record_count",
        "record_type",
        "registry_key_hash",
        "run_id",
        "schema",
        "signature_algorithm",
        "signature_key_id",
        "snapshot_hash",
        "snapshot_id",
        "snapshot_seal_hash",
        "source_completed_at",
        "source_status",
        "total_bytes",
    }
)


def _require_positive_bounded(value: int, field_name: str, maximum: int) -> None:
    if type(value) is not int or value < 1 or value > maximum:
        raise ValueError(f"{field_name} must be an exact integer within [1, {maximum}]")


@dataclass(frozen=True, slots=True)
class AuditExportTerminalWitness:
    """Immutable run tuple that pins every snapshot timestamp and status."""

    source_run_id: str
    source_status: RunStatus
    source_completed_at: datetime

    def __post_init__(self) -> None:
        if type(self.source_run_id) is not str or not self.source_run_id:
            raise ValueError("source_run_id must be a non-empty exact string")
        if type(self.source_status) is not RunStatus or self.source_status not in _EXPORT_TERMINAL:
            raise AuditIntegrityError("audit export requires an immutable export-terminal run status")
        if self.source_completed_at.tzinfo is None or self.source_completed_at.utcoffset() is None:
            raise AuditIntegrityError("audit export terminal completed_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class AuditExportSnapshotRegistryKey:
    """Exact persisted lookup key plus every separately stored shaping field."""

    source_run_id: str
    exporter_version: str
    serialization_version: str
    export_format: AuditExportFormat
    signing_mode: AuditExportSigningMode
    signer_key_id: str
    public_export_config_hash: str
    registry_key_hash: str
    chunking_algorithm_version: str
    per_chunk_record_limit: int
    per_chunk_byte_limit: int

    def __post_init__(self) -> None:
        for field_name in (
            "source_run_id",
            "exporter_version",
            "serialization_version",
            "signer_key_id",
            "chunking_algorithm_version",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or not value:
                raise ValueError(f"{field_name} must be a non-empty exact string")
        if type(self.export_format) is not AuditExportFormat:
            raise TypeError("export_format must be exact AuditExportFormat")
        if type(self.signing_mode) is not AuditExportSigningMode:
            raise TypeError("signing_mode must be exact AuditExportSigningMode")
        for field_name in ("public_export_config_hash", "registry_key_hash"):
            value = getattr(self, field_name)
            if type(value) is not str or _LOWER_HEX_64.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be lowercase SHA-256 hex")
        _require_positive_bounded(self.per_chunk_record_limit, "per_chunk_record_limit", AUDIT_EXPORT_MAX_CHUNK_RECORDS)
        _require_positive_bounded(self.per_chunk_byte_limit, "per_chunk_byte_limit", AUDIT_EXPORT_MAX_CHUNK_BYTES)

    @classmethod
    def from_snapshot(cls, snapshot: AuditExportSnapshot) -> AuditExportSnapshotRegistryKey:
        if type(snapshot) is not AuditExportSnapshot:
            raise TypeError("snapshot must be exact AuditExportSnapshot")
        return cls(
            source_run_id=snapshot.source_run_id,
            exporter_version=snapshot.exporter_version,
            serialization_version=snapshot.serialization_version,
            export_format=snapshot.export_format,
            signing_mode=snapshot.signing_mode,
            signer_key_id=snapshot.signer_key_id,
            public_export_config_hash=snapshot.public_export_config_hash,
            registry_key_hash=snapshot.registry_key_hash,
            chunking_algorithm_version=snapshot.chunking_algorithm_version,
            per_chunk_record_limit=snapshot.per_chunk_record_limit,
            per_chunk_byte_limit=snapshot.per_chunk_byte_limit,
        )


def _validate_snapshot_bundle(snapshot: AuditExportSnapshot, chunks: tuple[AuditExportSnapshotChunk, ...]) -> None:
    if type(snapshot) is not AuditExportSnapshot:
        raise TypeError("snapshot must be exact AuditExportSnapshot")
    if not chunks or any(type(chunk) is not AuditExportSnapshotChunk for chunk in chunks):
        raise ValueError("chunks must be a non-empty exact AuditExportSnapshotChunk tuple")
    if len(chunks) != snapshot.chunk_count:
        raise AuditIntegrityError("snapshot chunk_count does not match its exact chunk tuple")
    cumulative_records = 0
    cumulative_bytes = 0
    predecessor: str | None = None
    for ordinal, chunk in enumerate(chunks):
        if chunk.snapshot_id != snapshot.snapshot_id or chunk.ordinal != ordinal:
            raise AuditIntegrityError("snapshot chunks must bind one snapshot with dense ordered ordinals")
        if chunk.predecessor_seal_hash != predecessor:
            raise AuditIntegrityError("snapshot chunk predecessor seal chain is not exact")
        cumulative_records += chunk.record_count
        cumulative_bytes += chunk.size_bytes
        if chunk.cumulative_records != cumulative_records or chunk.cumulative_bytes != cumulative_bytes:
            raise AuditIntegrityError("snapshot chunk cumulative totals are not exact")
        predecessor = chunk.chunk_seal_hash
    terminal = chunks[-1]
    if (
        terminal.ordinal != snapshot.terminal_chunk_ordinal
        or terminal.chunk_seal_hash != snapshot.last_chunk_seal_hash
        or cumulative_records != snapshot.record_count
        or cumulative_bytes != snapshot.total_bytes
    ):
        raise AuditIntegrityError("snapshot terminal chunk descriptor does not seal its registry totals")


@dataclass(frozen=True, slots=True)
class AuditExportSnapshotCandidate:
    """Fully materialized and durably stored candidate ready for registry CAS."""

    snapshot: AuditExportSnapshot
    chunks: tuple[AuditExportSnapshotChunk, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunks", tuple(self.chunks))
        _validate_snapshot_bundle(self.snapshot, self.chunks)


@dataclass(frozen=True, slots=True)
class AuditExportSnapshotWinner:
    """Immutable registry winner and its exact sealed data descriptors."""

    snapshot: AuditExportSnapshot
    chunks: tuple[AuditExportSnapshotChunk, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunks", tuple(self.chunks))
        _validate_snapshot_bundle(self.snapshot, self.chunks)


@dataclass(frozen=True, slots=True)
class AuditExportSnapshotReadLimits:
    """Current acceptance limits, deliberately absent from snapshot identity."""

    max_total_bytes: int = AUDIT_EXPORT_MAX_TOTAL_BYTES
    max_total_records: int = AUDIT_EXPORT_MAX_TOTAL_RECORDS
    max_chunks: int = AUDIT_EXPORT_MAX_CHUNKS
    max_chunk_bytes: int = AUDIT_EXPORT_MAX_CHUNK_BYTES
    max_chunk_records: int = AUDIT_EXPORT_MAX_CHUNK_RECORDS

    def __post_init__(self) -> None:
        for field_name, maximum in (
            ("max_total_bytes", AUDIT_EXPORT_MAX_TOTAL_BYTES),
            ("max_total_records", AUDIT_EXPORT_MAX_TOTAL_RECORDS),
            ("max_chunks", AUDIT_EXPORT_MAX_CHUNKS),
            ("max_chunk_bytes", AUDIT_EXPORT_MAX_CHUNK_BYTES),
            ("max_chunk_records", AUDIT_EXPORT_MAX_CHUNK_RECORDS),
        ):
            _require_positive_bounded(getattr(self, field_name), field_name, maximum)


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


def _validate_final_manifest_core(payload: object) -> None:
    obj = _object(payload, fields=_FINAL_MANIFEST_CORE_FIELDS, path="final manifest signing body")
    for field in ("chunk_count", "record_count", "total_bytes"):
        _integer(obj[field], field, minimum=1)
    _string(obj["derivation_version"], "derivation_version", allowed=frozenset({AUDIT_EXPORT_DERIVATION_VERSION}))
    _string(obj["export_format"], "export_format", allowed=frozenset({"json", "csv"}))
    _timestamp(obj["exported_at"], "exported_at")
    for field in (
        "final_hash",
        "last_chunk_seal_hash",
        "manifest_hash",
        "registry_key_hash",
        "snapshot_hash",
        "snapshot_id",
        "snapshot_seal_hash",
    ):
        _hash(obj[field], field)
    _string(obj["hash_algorithm"], "hash_algorithm", allowed=frozenset({"sha256"}))
    _string(obj["record_type"], "record_type", allowed=frozenset({"manifest"}))
    _string(obj["run_id"], "run_id")
    _string(obj["schema"], "schema", allowed=frozenset({AUDIT_EXPORT_MANIFEST_SCHEMA}))
    mode = _string(obj["signature_algorithm"], "signature_algorithm", allowed=frozenset({"unsigned", "hmac_sha256"}))
    signer = _string(obj["signature_key_id"], "signature_key_id")
    _validate_signer(mode, signer, None, allow_signature_none=True)
    expected_chain = _UNSIGNED_RECORD_CHAIN if mode == "unsigned" else _HMAC_RECORD_CHAIN
    _string(obj["record_chain_algorithm"], "record_chain_algorithm", allowed=frozenset({expected_chain}))
    _timestamp(obj["source_completed_at"], "source_completed_at")
    _string(obj["source_status"], "source_status", allowed=_EXPORT_TERMINAL_STATUSES)


_SCHEMA_VALIDATORS: Final[dict[str, Callable[[object], None]]] = {
    "audit-export-public-config-v1": _validate_public_config,
    "audit-export-registry-key-v1": _validate_registry_key,
    "audit-export-snapshot-content-v1": _validate_snapshot_content,
    "audit-export-snapshot-id-v1": _validate_snapshot_id,
    "audit-export-chunk-seal-v1": _validate_chunk_seal,
    "audit-export-manifest-v1": _validate_chunk_manifest,
    "audit-export-snapshot-seal-v1": _validate_snapshot_seal,
    "audit-export-final-manifest-signing-body-v2": _validate_final_manifest_core,
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


def final_manifest_identity_payload(descriptor: AuditExportSignedManifestInput) -> dict[str, ClosedAuditExportJSON]:
    """Detach every immutable signed-manifest descriptor field for identity."""
    try:
        signature_algorithm = descriptor.signature_algorithm.value
        payload: dict[str, ClosedAuditExportJSON] = {
            "content_hash": descriptor.content_hash,
            "content_ref": descriptor.content_ref,
            "derivation_version": descriptor.derivation_version,
            "final_hash": descriptor.final_hash,
            "manifest_schema": descriptor.manifest_schema,
            "record_chain_algorithm": descriptor.record_chain_algorithm,
            "signature": descriptor.signature,
            "signature_algorithm": signature_algorithm,
            "signature_key_id": descriptor.signature_key_id,
            "size_bytes": descriptor.size_bytes,
        }
    except AttributeError as exc:
        raise TypeError("descriptor must expose the complete signed-manifest identity contract") from exc
    validate_closed_stage_payload("sink-effect-audit-export-final-manifest-v1", payload)
    return payload


def hash_final_manifest_identity_payload(
    payload: dict[str, ClosedAuditExportJSON],
    *,
    validate: bool = True,
) -> str:
    """Hash the exact complete component, optionally bypassing tuple validity.

    ``validate=False`` exists for scalar binding proofs: an isolated field
    mutation can be intentionally invalid as a whole tuple while still proving
    that the serialized key participates in the formula.
    """
    if type(payload) is not dict or any(type(key) is not str for key in payload):
        raise TypeError("final manifest identity payload must be an exact string-keyed object")
    if validate:
        return H(C("sink-effect-audit-export-final-manifest-v1", payload))
    return H(canonical_json({"payload": payload, "schema": "sink-effect-audit-export-final-manifest-v1"}).encode("utf-8"))


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


@dataclass(frozen=True, slots=True)
class AuditExportDerivationConfig:
    """Complete immutable authority for one canonical export derivation."""

    source_run_id: str
    source_status: str
    source_completed_at: str
    export_format: Literal["json", "csv"]
    exporter_version: str
    serialization_version: str
    chunking_algorithm_version: str
    include_raw_error_rows: bool
    per_chunk_byte_limit: int
    per_chunk_record_limit: int
    signing_mode: Literal["unsigned", "hmac_sha256"]
    signer_key_id: str
    signing_key: bytes | None

    def __post_init__(self) -> None:
        _string(self.source_run_id, "source_run_id")
        _string(self.source_status, "source_status", allowed=_EXPORT_TERMINAL_STATUSES)
        _timestamp(self.source_completed_at, "source_completed_at")
        _string(self.export_format, "export_format", allowed=frozenset({"json", "csv"}))
        _string(self.exporter_version, "exporter_version")
        if self.serialization_version != AUDIT_EXPORT_SERIALIZATION_VERSION:
            raise ValueError(f"serialization_version must equal {AUDIT_EXPORT_SERIALIZATION_VERSION!r}")
        _string(self.chunking_algorithm_version, "chunking_algorithm_version")
        _boolean(self.include_raw_error_rows, "include_raw_error_rows")
        _integer(self.per_chunk_byte_limit, "per_chunk_byte_limit", minimum=1, maximum=AUDIT_EXPORT_MAX_CHUNK_BYTES)
        _integer(
            self.per_chunk_record_limit,
            "per_chunk_record_limit",
            minimum=1,
            maximum=AUDIT_EXPORT_MAX_CHUNK_RECORDS,
        )
        _string(self.signing_mode, "signing_mode", allowed=frozenset({"unsigned", "hmac_sha256"}))
        _validate_signer(self.signing_mode, self.signer_key_id, None, allow_signature_none=True)
        if self.signing_mode == "unsigned":
            if self.signing_key is not None:
                raise ValueError("unsigned derivation forbids a signing key")
        elif type(self.signing_key) is not bytes or not self.signing_key:
            raise ValueError("hmac_sha256 derivation requires non-empty exact signing-key bytes")

    @property
    def exported_at(self) -> str:
        return self.source_completed_at


@dataclass(frozen=True, slots=True)
class AuditExportDerivedChunk:
    ordinal: int
    content: bytes
    descriptor: AuditExportContentDescriptor
    record_count: int
    cumulative_records: int
    cumulative_bytes: int
    predecessor_seal_hash: str | None
    chunk_seal_bytes: bytes
    chunk_seal_hash: str


@dataclass(frozen=True, slots=True)
class AuditExportDerivedBundle:
    """Exact canonical records, chunks, seals, and detached final manifest."""

    config: AuditExportDerivationConfig
    public_export_config_bytes: bytes
    public_export_config_hash: str
    registry_key_bytes: bytes
    registry_key_hash: str
    unsigned_record_bytes: tuple[bytes, ...]
    snapshot_content_bytes: bytes
    snapshot_hash: str
    snapshot_id_bytes: bytes
    snapshot_id: str
    chunk_manifest_bytes: bytes
    manifest_hash: str
    snapshot_seal_bytes: bytes
    snapshot_seal_hash: str
    last_chunk_seal_hash: str
    record_chain_algorithm: str
    final_hash: str
    record_objects: tuple[Mapping[str, ClosedAuditExportJSON], ...]
    record_frames: tuple[bytes, ...]
    chunks: tuple[AuditExportDerivedChunk, ...]
    signed_manifest: AuditExportSignedManifestInput
    final_manifest: Mapping[str, ClosedAuditExportJSON]
    signing_body: bytes
    signed_manifest_bytes: bytes

    def __post_init__(self) -> None:
        freeze_fields(self, "record_objects", "final_manifest")

    @property
    def source_completed_at(self) -> str:
        return self.config.source_completed_at

    @property
    def exported_at(self) -> str:
        return self.config.exported_at

    @property
    def chunk_bytes(self) -> tuple[bytes, ...]:
        return tuple(chunk.content for chunk in self.chunks)

    @property
    def json_target_bytes(self) -> bytes:
        return b"".join(self.chunk_bytes) + self.signed_manifest_bytes


@dataclass(frozen=True, slots=True)
class AuditExportSpooledChunk:
    ordinal: int
    descriptor: AuditExportContentDescriptor
    record_count: int
    cumulative_records: int
    cumulative_bytes: int
    predecessor_seal_hash: str | None
    chunk_seal_bytes: bytes
    chunk_seal_hash: str


@dataclass(frozen=True, slots=True)
class AuditExportSpooledBundle:
    """Bounded-memory derivation result whose data bytes live in one spool."""

    config: AuditExportDerivationConfig
    public_export_config_bytes: bytes
    public_export_config_hash: str
    registry_key_bytes: bytes
    registry_key_hash: str
    snapshot_content_bytes: bytes
    snapshot_hash: str
    snapshot_id_bytes: bytes
    snapshot_id: str
    chunk_manifest_bytes: bytes
    manifest_hash: str
    snapshot_seal_bytes: bytes
    snapshot_seal_hash: str
    last_chunk_seal_hash: str
    record_chain_algorithm: str
    final_hash: str
    chunks: tuple[AuditExportSpooledChunk, ...]
    signed_manifest: AuditExportSignedManifestInput
    final_manifest: Mapping[str, ClosedAuditExportJSON]
    signing_body: bytes
    signed_manifest_bytes: bytes
    chunk_offsets: tuple[tuple[int, int], ...]
    signed_manifest_offset: tuple[int, int]

    def __post_init__(self) -> None:
        freeze_fields(self, "final_manifest")

    @property
    def record_count(self) -> int:
        return self.chunks[-1].cumulative_records

    @property
    def total_bytes(self) -> int:
        return self.chunks[-1].cumulative_bytes


def _detached_record(record: Mapping[str, object]) -> dict[str, ClosedAuditExportJSON]:
    if type(record) is not dict or any(type(key) is not str for key in record):
        raise TypeError("audit export records must be exact string-keyed dictionaries")
    if "signature" in record:
        raise ValueError("audit export input records must not predeclare the reserved signature field")
    if record.get("record_type") == "manifest":
        raise ValueError("audit export input records must not contain a manifest")
    # Round-tripping through the committed canonical encoder proves the value
    # tree is detached JSON data and rejects datetimes/bytes/custom authority.
    try:
        encoded = canonical_json(record)
        value = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TypeError("audit export record is not closed canonical JSON") from exc
    if type(value) is not dict:
        raise TypeError("audit export record must canonicalize to an object")
    return value


def derive_audit_export_bundle(
    records: Iterable[Mapping[str, object]],
    config: AuditExportDerivationConfig,
) -> AuditExportDerivedBundle:
    """Derive the exact v2 snapshot and final-manifest byte graph.

    This pure boundary is intentionally target/store independent. Persistence
    code may spool the returned complete-frame chunks before registering the
    immutable winner; no formula here includes a target, effect, or store ID.
    """
    if type(config) is not AuditExportDerivationConfig:
        raise TypeError("config must be exact AuditExportDerivationConfig")
    from elspeth.contracts.sink_effects import AuditExportSignedManifestInput

    public_config: dict[str, ClosedAuditExportJSON] = {
        "chunking_algorithm_version": config.chunking_algorithm_version,
        "export_format": config.export_format,
        "exporter_version": config.exporter_version,
        "include_raw_error_rows": config.include_raw_error_rows,
        "per_chunk_byte_limit": config.per_chunk_byte_limit,
        "per_chunk_record_limit": config.per_chunk_record_limit,
        "serialization_version": config.serialization_version,
        "signer_key_id": config.signer_key_id,
        "signing_mode": config.signing_mode,
    }
    public_export_config_bytes = C("audit-export-public-config-v1", public_config)
    public_export_config_hash = H(public_export_config_bytes)
    registry_key: dict[str, ClosedAuditExportJSON] = {
        "export_format": config.export_format,
        "exporter_version": config.exporter_version,
        "public_export_config_hash": public_export_config_hash,
        "serialization_version": config.serialization_version,
        "signer_key_id": config.signer_key_id,
        "signing_mode": config.signing_mode,
        "source_run_id": config.source_run_id,
    }
    registry_key_bytes = C("audit-export-registry-key-v1", registry_key)
    registry_key_hash = H(registry_key_bytes)

    record_objects: list[Mapping[str, ClosedAuditExportJSON]] = []
    unsigned_record_bytes: list[bytes] = []
    record_frames: list[bytes] = []
    chain = hashlib.sha256()
    for raw in records:
        unsigned = _detached_record(raw)
        unsigned_bytes = canonical_json(unsigned).encode("utf-8")
        emitted = dict(unsigned)
        if config.signing_mode == "hmac_sha256":
            assert config.signing_key is not None
            record_signature = hmac.new(config.signing_key, unsigned_bytes, hashlib.sha256).hexdigest()
            emitted["signature"] = record_signature
            chain.update(record_signature.encode("ascii"))
        else:
            chain.update(hashlib.sha256(unsigned_bytes).hexdigest().encode("ascii"))
        frame = canonical_json(emitted).encode("utf-8") + b"\n"
        if len(frame) > config.per_chunk_byte_limit:
            raise ValueError("one audit export record frame exceeds per_chunk_byte_limit")
        record_objects.append(MappingProxyType(emitted))
        unsigned_record_bytes.append(unsigned_bytes)
        record_frames.append(frame)
        if len(record_frames) > AUDIT_EXPORT_MAX_TOTAL_RECORDS:
            raise ValueError("audit export record count exceeds the code-owned maximum")
    if not record_frames:
        raise ValueError("audit export requires at least the run record")

    chunk_material: list[tuple[bytes, int]] = []
    current: list[bytes] = []
    current_bytes = 0
    for frame in record_frames:
        if current and (len(current) == config.per_chunk_record_limit or current_bytes + len(frame) > config.per_chunk_byte_limit):
            chunk_material.append((b"".join(current), len(current)))
            current = []
            current_bytes = 0
        current.append(frame)
        current_bytes += len(frame)
    if current:
        chunk_material.append((b"".join(current), len(current)))
    if len(chunk_material) > AUDIT_EXPORT_MAX_CHUNKS:
        raise ValueError("audit export chunk count exceeds the code-owned maximum")

    snapshot_chunks: list[dict[str, ClosedAuditExportJSON]] = []
    cumulative_records = 0
    cumulative_bytes = 0
    for ordinal, (content, record_count) in enumerate(chunk_material):
        cumulative_records += record_count
        cumulative_bytes += len(content)
        if cumulative_bytes > AUDIT_EXPORT_MAX_TOTAL_BYTES:
            raise ValueError("audit export total bytes exceed the code-owned maximum")
        snapshot_chunks.append(
            {
                "content_hash": H(content),
                "cumulative_bytes": cumulative_bytes,
                "cumulative_records": cumulative_records,
                "ordinal": ordinal,
                "record_count": record_count,
                "size_bytes": len(content),
            }
        )
    snapshot_content: dict[str, ClosedAuditExportJSON] = {
        "chunking_algorithm_version": config.chunking_algorithm_version,
        "chunks": cast(list[ClosedAuditExportJSON], snapshot_chunks),
        "record_count": len(record_frames),
        "serialization_version": config.serialization_version,
        "total_bytes": cumulative_bytes,
    }
    snapshot_content_bytes = C("audit-export-snapshot-content-v1", snapshot_content)
    snapshot_hash = H(snapshot_content_bytes)
    snapshot_id_bytes = C(
        "audit-export-snapshot-id-v1",
        {"registry_key_hash": registry_key_hash, "snapshot_hash": snapshot_hash},
    )
    snapshot_id = H(snapshot_id_bytes)

    derived_chunks: list[AuditExportDerivedChunk] = []
    manifest_chunks: list[dict[str, ClosedAuditExportJSON]] = []
    predecessor: str | None = None
    for snapshot_chunk, (content, record_count) in zip(snapshot_chunks, chunk_material, strict=True):
        content_hash = snapshot_chunk["content_hash"]
        cumulative_bytes_value = snapshot_chunk["cumulative_bytes"]
        cumulative_records_value = snapshot_chunk["cumulative_records"]
        ordinal_value = snapshot_chunk["ordinal"]
        assert type(content_hash) is str
        assert type(cumulative_bytes_value) is int
        assert type(cumulative_records_value) is int
        assert type(ordinal_value) is int
        content_ref = REF(content_hash)
        predecessor_object: dict[str, ClosedAuditExportJSON] = (
            {"kind": "genesis"} if predecessor is None else {"hash": predecessor, "kind": "chunk_seal"}
        )
        chunk_seal_payload: dict[str, ClosedAuditExportJSON] = {
            "chunking_algorithm_version": config.chunking_algorithm_version,
            "content_hash": content_hash,
            "content_ref": content_ref,
            "cumulative_bytes": cumulative_bytes_value,
            "cumulative_records": cumulative_records_value,
            "derivation_version": AUDIT_EXPORT_DERIVATION_VERSION,
            "ordinal": ordinal_value,
            "predecessor": predecessor_object,
            "record_count": record_count,
            "serialization_version": config.serialization_version,
            "size_bytes": len(content),
            "snapshot_id": snapshot_id,
        }
        chunk_seal_bytes = C("audit-export-chunk-seal-v1", chunk_seal_payload)
        seal_hash = H(chunk_seal_bytes)
        descriptor = AuditExportContentDescriptor(content_ref, content_hash, len(content), "data_chunk")
        derived_chunks.append(
            AuditExportDerivedChunk(
                ordinal=ordinal_value,
                content=content,
                descriptor=descriptor,
                record_count=record_count,
                cumulative_records=cumulative_records_value,
                cumulative_bytes=cumulative_bytes_value,
                predecessor_seal_hash=predecessor,
                chunk_seal_bytes=chunk_seal_bytes,
                chunk_seal_hash=seal_hash,
            )
        )
        manifest_chunks.append(
            {
                "chunk_seal_hash": seal_hash,
                "content_hash": content_hash,
                "content_ref": content_ref,
                "cumulative_bytes": cumulative_bytes_value,
                "cumulative_records": cumulative_records_value,
                "ordinal": ordinal_value,
                "predecessor_seal_hash": predecessor,
                "record_count": record_count,
                "size_bytes": len(content),
            }
        )
        predecessor = seal_hash
    assert predecessor is not None
    manifest_payload: dict[str, ClosedAuditExportJSON] = {
        "chunk_count": len(derived_chunks),
        "chunks": cast(list[ClosedAuditExportJSON], manifest_chunks),
        "record_count": len(record_frames),
        "snapshot_id": snapshot_id,
        "total_bytes": cumulative_bytes,
    }
    chunk_manifest_bytes = C("audit-export-manifest-v1", manifest_payload)
    manifest_hash = H(chunk_manifest_bytes)
    snapshot_seal_payload: dict[str, ClosedAuditExportJSON] = {
        "chunk_count": len(derived_chunks),
        "chunking_algorithm_version": config.chunking_algorithm_version,
        "exported_at": config.exported_at,
        "last_chunk_seal_hash": predecessor,
        "manifest_hash": manifest_hash,
        "per_chunk_byte_limit": config.per_chunk_byte_limit,
        "per_chunk_record_limit": config.per_chunk_record_limit,
        "record_count": len(record_frames),
        "registry_key_hash": registry_key_hash,
        "snapshot_hash": snapshot_hash,
        "snapshot_id": snapshot_id,
        "source_completed_at": config.source_completed_at,
        "source_run_id": config.source_run_id,
        "source_status": config.source_status,
        "total_bytes": cumulative_bytes,
    }
    snapshot_seal_bytes = C("audit-export-snapshot-seal-v1", snapshot_seal_payload)
    snapshot_seal_hash = H(snapshot_seal_bytes)
    record_chain_algorithm = _UNSIGNED_RECORD_CHAIN if config.signing_mode == "unsigned" else _HMAC_RECORD_CHAIN
    final_hash = chain.hexdigest()
    final_core: dict[str, ClosedAuditExportJSON] = {
        "chunk_count": len(derived_chunks),
        "derivation_version": AUDIT_EXPORT_DERIVATION_VERSION,
        "export_format": config.export_format,
        "exported_at": config.exported_at,
        "final_hash": final_hash,
        "hash_algorithm": "sha256",
        "last_chunk_seal_hash": predecessor,
        "manifest_hash": manifest_hash,
        "record_chain_algorithm": record_chain_algorithm,
        "record_count": len(record_frames),
        "record_type": "manifest",
        "registry_key_hash": registry_key_hash,
        "run_id": config.source_run_id,
        "schema": AUDIT_EXPORT_MANIFEST_SCHEMA,
        "signature_algorithm": config.signing_mode,
        "signature_key_id": config.signer_key_id,
        "snapshot_hash": snapshot_hash,
        "snapshot_id": snapshot_id,
        "snapshot_seal_hash": snapshot_seal_hash,
        "source_completed_at": config.source_completed_at,
        "source_status": config.source_status,
        "total_bytes": cumulative_bytes,
    }
    signing_body = C("audit-export-final-manifest-signing-body-v2", final_core)
    signature: str | None = None
    if config.signing_mode == "hmac_sha256":
        assert config.signing_key is not None
        signature = hmac.new(config.signing_key, signing_body, hashlib.sha256).hexdigest()
    final_manifest = {**final_core, "signature": signature}
    signed_manifest_bytes = canonical_json(final_manifest).encode("utf-8")
    signed_manifest_hash = H(signed_manifest_bytes)
    signed_manifest = AuditExportSignedManifestInput(
        content_ref=REF(signed_manifest_hash),
        content_hash=signed_manifest_hash,
        size_bytes=len(signed_manifest_bytes),
        manifest_schema=AUDIT_EXPORT_MANIFEST_SCHEMA,
        derivation_version=AUDIT_EXPORT_DERIVATION_VERSION,
        signature_algorithm=AuditExportSigningMode(config.signing_mode),
        signature_key_id=config.signer_key_id,
        record_chain_algorithm=record_chain_algorithm,
        final_hash=final_hash,
        signature=signature,
    )
    return AuditExportDerivedBundle(
        config=config,
        public_export_config_bytes=public_export_config_bytes,
        public_export_config_hash=public_export_config_hash,
        registry_key_bytes=registry_key_bytes,
        registry_key_hash=registry_key_hash,
        unsigned_record_bytes=tuple(unsigned_record_bytes),
        snapshot_content_bytes=snapshot_content_bytes,
        snapshot_hash=snapshot_hash,
        snapshot_id_bytes=snapshot_id_bytes,
        snapshot_id=snapshot_id,
        chunk_manifest_bytes=chunk_manifest_bytes,
        manifest_hash=manifest_hash,
        snapshot_seal_bytes=snapshot_seal_bytes,
        snapshot_seal_hash=snapshot_seal_hash,
        last_chunk_seal_hash=predecessor,
        record_chain_algorithm=record_chain_algorithm,
        final_hash=final_hash,
        record_objects=tuple(record_objects),
        record_frames=tuple(record_frames),
        chunks=tuple(derived_chunks),
        signed_manifest=signed_manifest,
        final_manifest=MappingProxyType(final_manifest),
        signing_body=signing_body,
        signed_manifest_bytes=signed_manifest_bytes,
    )


def _write_spooled(spool: BinaryIO, content: bytes | bytearray) -> tuple[int, int]:
    offset = spool.tell()
    view = memoryview(content)
    written = 0
    while written < len(view):
        count = spool.write(view[written:])
        if count is None or count <= 0:
            raise OSError("audit export spool write made no progress")
        written += count
    return offset, written


def derive_audit_export_bundle_to_spool(
    records: Iterable[Mapping[str, object]],
    config: AuditExportDerivationConfig,
    spool: BinaryIO,
    *,
    max_total_records: int,
    max_total_bytes: int,
    max_chunks: int,
) -> AuditExportSpooledBundle:
    """Derive one exact graph while retaining only one bounded chunk in RAM."""

    stream = stream_audit_export_bundle_to_spool(
        records,
        config,
        spool,
        max_total_records=max_total_records,
        max_total_bytes=max_total_bytes,
        max_chunks=max_chunks,
    )
    while True:
        try:
            next(stream)
        except StopIteration as stop:
            bundle = stop.value
            if type(bundle) is not AuditExportSpooledBundle:
                raise AuditIntegrityError("audit export stream terminated without a spooled bundle") from None
            return bundle


def stream_audit_export_bundle_to_spool(
    records: Iterable[Mapping[str, object]],
    config: AuditExportDerivationConfig,
    spool: BinaryIO,
    *,
    max_total_records: int,
    max_total_bytes: int,
    max_chunks: int,
) -> Generator[dict[str, ClosedAuditExportJSON], None, AuditExportSpooledBundle]:
    """Stream emitted export records while deriving the exact spooled graph.

    Yields each emitted record (carrying its per-record signature when the
    config signs) as soon as its frame is sealed into the running chunk, so
    a consumer observes records on a bounded-memory path: only one chunk is
    retained in RAM at a time, exactly as in
    :func:`derive_audit_export_bundle_to_spool`. The complete
    ``AuditExportSpooledBundle`` — including the final manifest — is the
    generator's return value (``StopIteration.value``); the emitted byte
    graph is identical to the non-streaming derivation.
    """
    if type(config) is not AuditExportDerivationConfig:
        raise TypeError("config must be exact AuditExportDerivationConfig")
    _integer(max_total_records, "max_total_records", minimum=1, maximum=AUDIT_EXPORT_MAX_TOTAL_RECORDS)
    _integer(max_total_bytes, "max_total_bytes", minimum=1, maximum=AUDIT_EXPORT_MAX_TOTAL_BYTES)
    _integer(max_chunks, "max_chunks", minimum=1, maximum=AUDIT_EXPORT_MAX_CHUNKS)
    return _stream_audit_export_bundle_to_spool(
        records,
        config,
        spool,
        max_total_records=max_total_records,
        max_total_bytes=max_total_bytes,
        max_chunks=max_chunks,
    )


def _stream_audit_export_bundle_to_spool(
    records: Iterable[Mapping[str, object]],
    config: AuditExportDerivationConfig,
    spool: BinaryIO,
    *,
    max_total_records: int,
    max_total_bytes: int,
    max_chunks: int,
) -> Generator[dict[str, ClosedAuditExportJSON], None, AuditExportSpooledBundle]:
    """Generator body for :func:`stream_audit_export_bundle_to_spool`."""
    from elspeth.contracts.sink_effects import AuditExportSignedManifestInput, AuditExportSigningMode

    public_config: dict[str, ClosedAuditExportJSON] = {
        "chunking_algorithm_version": config.chunking_algorithm_version,
        "export_format": config.export_format,
        "exporter_version": config.exporter_version,
        "include_raw_error_rows": config.include_raw_error_rows,
        "per_chunk_byte_limit": config.per_chunk_byte_limit,
        "per_chunk_record_limit": config.per_chunk_record_limit,
        "serialization_version": config.serialization_version,
        "signer_key_id": config.signer_key_id,
        "signing_mode": config.signing_mode,
    }
    public_export_config_bytes = C("audit-export-public-config-v1", public_config)
    public_export_config_hash = H(public_export_config_bytes)
    registry_key: dict[str, ClosedAuditExportJSON] = {
        "export_format": config.export_format,
        "exporter_version": config.exporter_version,
        "public_export_config_hash": public_export_config_hash,
        "serialization_version": config.serialization_version,
        "signer_key_id": config.signer_key_id,
        "signing_mode": config.signing_mode,
        "source_run_id": config.source_run_id,
    }
    registry_key_bytes = C("audit-export-registry-key-v1", registry_key)
    registry_key_hash = H(registry_key_bytes)

    chain = hashlib.sha256()
    current = bytearray()
    current_records = 0
    record_count = 0
    total_bytes = 0
    chunk_offsets: list[tuple[int, int]] = []
    snapshot_chunks: list[dict[str, ClosedAuditExportJSON]] = []

    def finish_chunk() -> None:
        nonlocal current, current_records
        if not current:
            return
        if len(snapshot_chunks) >= max_chunks:
            raise ValueError("audit export chunk count exceeds max_chunks")
        ordinal = len(snapshot_chunks)
        content_hash = hashlib.sha256(current).hexdigest()
        chunk_offsets.append(_write_spooled(spool, current))
        snapshot_chunks.append(
            {
                "content_hash": content_hash,
                "cumulative_bytes": total_bytes,
                "cumulative_records": record_count,
                "ordinal": ordinal,
                "record_count": current_records,
                "size_bytes": len(current),
            }
        )
        current = bytearray()
        current_records = 0

    for raw in records:
        unsigned = _detached_record(raw)
        unsigned_bytes = canonical_json(unsigned).encode("utf-8")
        emitted = dict(unsigned)
        if config.signing_mode == "hmac_sha256":
            assert config.signing_key is not None
            record_signature = hmac.new(config.signing_key, unsigned_bytes, hashlib.sha256).hexdigest()
            emitted["signature"] = record_signature
            chain.update(record_signature.encode("ascii"))
        else:
            chain.update(hashlib.sha256(unsigned_bytes).hexdigest().encode("ascii"))
        frame = canonical_json(emitted).encode("utf-8") + b"\n"
        if len(frame) > config.per_chunk_byte_limit:
            raise ValueError("one audit export record frame exceeds per_chunk_byte_limit")
        if record_count >= max_total_records:
            raise ValueError("audit export record count exceeds max_total_records")
        if total_bytes + len(frame) > max_total_bytes:
            raise ValueError("audit export total bytes exceed max_total_bytes")
        if current and (current_records >= config.per_chunk_record_limit or len(current) + len(frame) > config.per_chunk_byte_limit):
            finish_chunk()
        current.extend(frame)
        current_records += 1
        record_count += 1
        total_bytes += len(frame)
        if current_records == config.per_chunk_record_limit or len(current) == config.per_chunk_byte_limit:
            finish_chunk()
        yield emitted
    finish_chunk()
    if not snapshot_chunks:
        raise ValueError("audit export requires at least the run record")

    snapshot_content: dict[str, ClosedAuditExportJSON] = {
        "chunking_algorithm_version": config.chunking_algorithm_version,
        "chunks": cast(list[ClosedAuditExportJSON], snapshot_chunks),
        "record_count": record_count,
        "serialization_version": config.serialization_version,
        "total_bytes": total_bytes,
    }
    snapshot_content_bytes = C("audit-export-snapshot-content-v1", snapshot_content)
    snapshot_hash = H(snapshot_content_bytes)
    snapshot_id_bytes = C(
        "audit-export-snapshot-id-v1",
        {"registry_key_hash": registry_key_hash, "snapshot_hash": snapshot_hash},
    )
    snapshot_id = H(snapshot_id_bytes)

    derived_chunks: list[AuditExportSpooledChunk] = []
    manifest_chunks: list[dict[str, ClosedAuditExportJSON]] = []
    predecessor: str | None = None
    for chunk in snapshot_chunks:
        content_hash = cast(str, chunk["content_hash"])
        cumulative_bytes = cast(int, chunk["cumulative_bytes"])
        cumulative_records = cast(int, chunk["cumulative_records"])
        ordinal = cast(int, chunk["ordinal"])
        chunk_record_count = cast(int, chunk["record_count"])
        size_bytes = cast(int, chunk["size_bytes"])
        content_ref = REF(content_hash)
        predecessor_object: dict[str, ClosedAuditExportJSON] = (
            {"kind": "genesis"} if predecessor is None else {"hash": predecessor, "kind": "chunk_seal"}
        )
        seal_payload: dict[str, ClosedAuditExportJSON] = {
            "chunking_algorithm_version": config.chunking_algorithm_version,
            "content_hash": content_hash,
            "content_ref": content_ref,
            "cumulative_bytes": cumulative_bytes,
            "cumulative_records": cumulative_records,
            "derivation_version": AUDIT_EXPORT_DERIVATION_VERSION,
            "ordinal": ordinal,
            "predecessor": predecessor_object,
            "record_count": chunk_record_count,
            "serialization_version": config.serialization_version,
            "size_bytes": size_bytes,
            "snapshot_id": snapshot_id,
        }
        seal_bytes = C("audit-export-chunk-seal-v1", seal_payload)
        seal_hash = H(seal_bytes)
        descriptor = AuditExportContentDescriptor(content_ref, content_hash, size_bytes, "data_chunk")
        derived_chunks.append(
            AuditExportSpooledChunk(
                ordinal=ordinal,
                descriptor=descriptor,
                record_count=chunk_record_count,
                cumulative_records=cumulative_records,
                cumulative_bytes=cumulative_bytes,
                predecessor_seal_hash=predecessor,
                chunk_seal_bytes=seal_bytes,
                chunk_seal_hash=seal_hash,
            )
        )
        manifest_chunks.append(
            {
                "chunk_seal_hash": seal_hash,
                "content_hash": content_hash,
                "content_ref": content_ref,
                "cumulative_bytes": cumulative_bytes,
                "cumulative_records": cumulative_records,
                "ordinal": ordinal,
                "predecessor_seal_hash": predecessor,
                "record_count": chunk_record_count,
                "size_bytes": size_bytes,
            }
        )
        predecessor = seal_hash
    assert predecessor is not None
    manifest_payload: dict[str, ClosedAuditExportJSON] = {
        "chunk_count": len(derived_chunks),
        "chunks": cast(list[ClosedAuditExportJSON], manifest_chunks),
        "record_count": record_count,
        "snapshot_id": snapshot_id,
        "total_bytes": total_bytes,
    }
    chunk_manifest_bytes = C("audit-export-manifest-v1", manifest_payload)
    manifest_hash = H(chunk_manifest_bytes)
    snapshot_seal_payload: dict[str, ClosedAuditExportJSON] = {
        "chunk_count": len(derived_chunks),
        "chunking_algorithm_version": config.chunking_algorithm_version,
        "exported_at": config.exported_at,
        "last_chunk_seal_hash": predecessor,
        "manifest_hash": manifest_hash,
        "per_chunk_byte_limit": config.per_chunk_byte_limit,
        "per_chunk_record_limit": config.per_chunk_record_limit,
        "record_count": record_count,
        "registry_key_hash": registry_key_hash,
        "snapshot_hash": snapshot_hash,
        "snapshot_id": snapshot_id,
        "source_completed_at": config.source_completed_at,
        "source_run_id": config.source_run_id,
        "source_status": config.source_status,
        "total_bytes": total_bytes,
    }
    snapshot_seal_bytes = C("audit-export-snapshot-seal-v1", snapshot_seal_payload)
    snapshot_seal_hash = H(snapshot_seal_bytes)
    record_chain_algorithm = _UNSIGNED_RECORD_CHAIN if config.signing_mode == "unsigned" else _HMAC_RECORD_CHAIN
    final_hash = chain.hexdigest()
    final_core: dict[str, ClosedAuditExportJSON] = {
        "chunk_count": len(derived_chunks),
        "derivation_version": AUDIT_EXPORT_DERIVATION_VERSION,
        "export_format": config.export_format,
        "exported_at": config.exported_at,
        "final_hash": final_hash,
        "hash_algorithm": "sha256",
        "last_chunk_seal_hash": predecessor,
        "manifest_hash": manifest_hash,
        "record_chain_algorithm": record_chain_algorithm,
        "record_count": record_count,
        "record_type": "manifest",
        "registry_key_hash": registry_key_hash,
        "run_id": config.source_run_id,
        "schema": AUDIT_EXPORT_MANIFEST_SCHEMA,
        "signature_algorithm": config.signing_mode,
        "signature_key_id": config.signer_key_id,
        "snapshot_hash": snapshot_hash,
        "snapshot_id": snapshot_id,
        "snapshot_seal_hash": snapshot_seal_hash,
        "source_completed_at": config.source_completed_at,
        "source_status": config.source_status,
        "total_bytes": total_bytes,
    }
    signing_body = C("audit-export-final-manifest-signing-body-v2", final_core)
    signature: str | None = None
    if config.signing_mode == "hmac_sha256":
        assert config.signing_key is not None
        signature = hmac.new(config.signing_key, signing_body, hashlib.sha256).hexdigest()
    final_manifest = {**final_core, "signature": signature}
    signed_manifest_bytes = canonical_json(final_manifest).encode("utf-8")
    signed_manifest_hash = H(signed_manifest_bytes)
    signed_manifest = AuditExportSignedManifestInput(
        content_ref=REF(signed_manifest_hash),
        content_hash=signed_manifest_hash,
        size_bytes=len(signed_manifest_bytes),
        manifest_schema=AUDIT_EXPORT_MANIFEST_SCHEMA,
        derivation_version=AUDIT_EXPORT_DERIVATION_VERSION,
        signature_algorithm=AuditExportSigningMode(config.signing_mode),
        signature_key_id=config.signer_key_id,
        record_chain_algorithm=record_chain_algorithm,
        final_hash=final_hash,
        signature=signature,
    )
    signed_manifest_offset = _write_spooled(spool, signed_manifest_bytes)
    return AuditExportSpooledBundle(
        config=config,
        public_export_config_bytes=public_export_config_bytes,
        public_export_config_hash=public_export_config_hash,
        registry_key_bytes=registry_key_bytes,
        registry_key_hash=registry_key_hash,
        snapshot_content_bytes=snapshot_content_bytes,
        snapshot_hash=snapshot_hash,
        snapshot_id_bytes=snapshot_id_bytes,
        snapshot_id=snapshot_id,
        chunk_manifest_bytes=chunk_manifest_bytes,
        manifest_hash=manifest_hash,
        snapshot_seal_bytes=snapshot_seal_bytes,
        snapshot_seal_hash=snapshot_seal_hash,
        last_chunk_seal_hash=predecessor,
        record_chain_algorithm=record_chain_algorithm,
        final_hash=final_hash,
        chunks=tuple(derived_chunks),
        signed_manifest=signed_manifest,
        final_manifest=MappingProxyType(final_manifest),
        signing_body=signing_body,
        signed_manifest_bytes=signed_manifest_bytes,
        chunk_offsets=tuple(chunk_offsets),
        signed_manifest_offset=signed_manifest_offset,
    )
