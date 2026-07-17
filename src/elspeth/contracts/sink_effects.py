"""Closed, immutable contracts for recoverable sink publication effects."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NoReturn, final
from urllib.parse import parse_qsl, urlsplit

from elspeth.contracts.enums import CallType, TerminalOutcome, TerminalPath
from elspeth.contracts.freeze import deep_freeze, deep_thaw, freeze_fields, require_int
from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.results import ArtifactDescriptor, require_no_artifact_uri_credentials
from elspeth.contracts.secret_scrub import scrub_payload_for_audit, scrub_text_for_audit
from elspeth.contracts.url import SENSITIVE_PARAMS

if TYPE_CHECKING:
    from elspeth.contracts.audit import Artifact, ArtifactPublicationEvidenceKind, SinkEffect

SINK_EFFECT_PROTOCOL_VERSION: Final = "sink-effect-v1"
_SINK_EFFECT_EVIDENCE_MAX_BYTES: Final = 64 * 1024
_AUDIT_EXPORT_MANIFEST_MAX_BYTES: Final = 64 * 1024
_AUDIT_EXPORT_MAX_CHUNKS: Final = 100_000
_AUDIT_EXPORT_MAX_CHUNK_BYTES: Final = 64 * 1024 * 1024
_AUDIT_EXPORT_MAX_CHUNK_RECORDS: Final = 1_000_000
_AUDIT_EXPORT_MAX_TOTAL_BYTES: Final = 1024 * 1024 * 1024 * 1024
_AUDIT_EXPORT_MAX_TOTAL_RECORDS: Final = 100_000_000
_AUDIT_EXPORT_MANIFEST_SCHEMA: Final = "elspeth.audit-export-manifest.v2"
_AUDIT_EXPORT_DERIVATION_VERSION: Final = "audit-export-derivation-v1"
_AUDIT_EXPORT_SERIALIZATION_VERSION: Final = "audit-export-v2"
_UNSIGNED_RECORD_CHAIN: Final = "sha256_concat_record_sha256_v1"
_HMAC_RECORD_CHAIN: Final = "sha256_concat_hmac_sha256_signatures_v1"
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_EMPTY_IDENTITY_HASH: Final = sha256(b"{}").hexdigest()
_UTC_MICROSECOND_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z\Z")
_V2_MANIFEST_FIELDS: Final = frozenset(
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
        "signature",
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
_EVIDENCE_PERFORMED: Final[dict[str, bool]] = {
    "returned": True,
    "reconciled": True,
    "inherited": False,
    "virtual": False,
}


class SinkEffectRole(StrEnum):
    PRIMARY = "primary"
    FAILSINK = "failsink"


class SinkEffectState(StrEnum):
    RESERVED = "reserved"
    PREPARED = "prepared"
    IN_FLIGHT = "in_flight"
    FINALIZED = "finalized"


class SinkEffectDescriptorMode(StrEnum):
    PRECOMPUTED = "precomputed"
    RESULT_DERIVED = "result_derived"
    NO_PUBLICATION = "no_publication"


class SinkEffectInspectionMode(StrEnum):
    INSPECTED = "inspected"
    NO_INSPECTION_REQUIRED = "no_inspection_required"


class SinkEffectInputKind(StrEnum):
    PIPELINE_MEMBERS = "pipeline_members"
    AUDIT_EXPORT_SNAPSHOT = "audit_export_snapshot"


class SinkEffectExecutionPurpose(StrEnum):
    FRESH = "fresh"
    RESUME = "resume"
    FOLLOWER = "follower"
    AUDIT_EXPORT = "audit_export"


@dataclass(frozen=True, slots=True)
class ResolvedSinkEffectMode:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("Resolved sink effect mode must be a non-empty string")


@dataclass(frozen=True, slots=True, repr=False)
class SinkEffectRuntimeBinding:
    sink_name: str
    sink: object
    sink_type: type[object]
    config_fingerprint: str
    purpose: SinkEffectExecutionPurpose
    effect_mode: ResolvedSinkEffectMode | None

    def __post_init__(self) -> None:
        if not isinstance(self.sink_name, str) or not self.sink_name.strip():
            raise ValueError("Sink runtime binding name must be non-empty")
        if self.sink_type is not type(self.sink):
            raise TypeError("Sink runtime binding type must exactly match its sink instance")
        if not isinstance(self.config_fingerprint, str) or _LOWER_HEX_64.fullmatch(self.config_fingerprint) is None:
            raise ValueError("Sink runtime binding config fingerprint must be a lowercase SHA-256 digest")
        if not isinstance(self.purpose, SinkEffectExecutionPurpose):
            raise TypeError("Sink runtime binding purpose must be exact SinkEffectExecutionPurpose")
        if self.effect_mode is not None and type(self.effect_mode) is not ResolvedSinkEffectMode:
            raise TypeError("Sink runtime binding mode must be ResolvedSinkEffectMode or None")


class AuditExportFormat(StrEnum):
    JSON = "json"
    CSV = "csv"


class AuditExportSigningMode(StrEnum):
    UNSIGNED = "unsigned"
    HMAC_SHA256 = "hmac_sha256"


class SinkEffectReconcileKind(StrEnum):
    NOT_APPLIED = "not_applied"
    APPLIED_WITH_EXACT_DESCRIPTOR = "applied_with_exact_descriptor"
    UNKNOWN = "unknown"


class SinkEffectAttemptAction(StrEnum):
    INSPECT = "inspect"
    COMMIT = "commit"
    RECONCILE = "reconcile"


class SinkEffectAttemptState(StrEnum):
    INTENT = "intent"
    RETURNED = "returned"
    RESPONSE_LOST = "response_lost"
    ERROR = "error"


def _require_exact_enum[EnumT: StrEnum](value: object, enum_type: type[EnumT], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{field_name} must be {enum_type.__name__}, got {type(value).__name__}")


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_lower_hex_64(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _LOWER_HEX_64.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be lowercase 64-character hexadecimal")


def _require_utc_microsecond_timestamp(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _UTC_MICROSECOND_TIMESTAMP.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be an exact UTC microsecond timestamp")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid UTC timestamp") from exc


def _require_bounded_positive_int(value: object, field_name: str, maximum: int) -> None:
    require_int(value, field_name)
    assert isinstance(value, int)
    if value < 1:
        raise ValueError(f"{field_name} must be strictly positive")
    if value > maximum:
        raise ValueError(f"{field_name} exceeds the code-owned maximum {maximum}")


def _validate_content_descriptor(
    *,
    content_ref: object,
    content_hash: object,
    size_bytes: object,
    field_prefix: str,
    max_size_bytes: int,
) -> None:
    _require_lower_hex_64(content_hash, f"{field_prefix}content_hash")
    assert isinstance(content_hash, str)
    expected_ref = f"sha256:{content_hash}"
    if content_ref != expected_ref:
        raise ValueError(f"{field_prefix}content_ref must equal {expected_ref!r}")
    assert isinstance(content_ref, str)
    _reject_credential_bearing_reference(content_ref, f"{field_prefix}content_ref")
    _require_bounded_positive_int(size_bytes, f"{field_prefix}size_bytes", max_size_bytes)


def _base_param_name(key: str) -> str:
    """Normalize query/fragment key suffix syntax like the URL contracts."""
    bracket = key.find("[")
    if bracket != -1:
        key = key[:bracket]
    dot = key.find(".")
    if dot != -1:
        key = key[:dot]
    return key


def _reject_credential_bearing_reference(value: str, field_name: str) -> None:
    """Reject userinfo, credential parameters, and known secret forms."""
    _require_nonempty_string(value, field_name)
    parsed = urlsplit(value)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must be credential-free (URI userinfo is forbidden)")
    for raw_params in (parsed.query, parsed.fragment):
        if any(_base_param_name(key.lower()) in SENSITIVE_PARAMS for key, _value in parse_qsl(raw_params, keep_blank_values=True)):
            raise ValueError(f"{field_name} must be credential-free (sensitive URI parameters are forbidden)")
    if scrub_text_for_audit(value) != value:
        raise ValueError(f"{field_name} must be credential-free (known secret form detected)")


def _freeze_bounded_evidence(evidence: Mapping[str, object], field_name: str) -> Mapping[str, object]:
    if not isinstance(evidence, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    if _contains_restricted_reader(evidence):
        raise TypeError(f"{field_name} cannot contain a RestrictedAuditExportSnapshotReader")
    frozen = deep_freeze(evidence)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by input check
        raise TypeError(f"{field_name} must be a mapping")
    detached = deep_thaw(frozen)
    canonical = canonical_json(detached)
    if len(canonical.encode("utf-8")) > _SINK_EFFECT_EVIDENCE_MAX_BYTES:
        raise ValueError(f"{field_name} canonical JSON exceeds the 64 KiB limit")
    if scrub_payload_for_audit(detached) != detached:
        raise ValueError(f"{field_name} must be credential-free (known secret form detected)")
    return frozen


def _freeze_canonical_row_value(value: object, path: str) -> object:
    """Detach one JSON/RFC-8785-safe value tree and reject authority objects."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise ValueError(f"{path} integer exceeds the canonical JSON safe range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} float must be finite")
        return value
    if isinstance(value, Mapping):
        frozen_items: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mappings require string keys")
            frozen_items[key] = _freeze_canonical_row_value(item, f"{path}.{key}")
        return MappingProxyType(frozen_items)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_canonical_row_value(item, f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, RestrictedAuditExportSnapshotReader):
        raise TypeError(f"{path} cannot contain a restricted audit export reader")
    raise TypeError(f"{path} contains unsupported non-canonical value {type(value).__name__}")


def _contains_restricted_reader(value: object) -> bool:
    reader_type = globals().get("RestrictedAuditExportSnapshotReader")
    if isinstance(reader_type, type) and isinstance(value, reader_type):
        return True
    if isinstance(value, Mapping):
        return any(_contains_restricted_reader(key) or _contains_restricted_reader(item) for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_restricted_reader(item) for item in value)
    return False


def _freeze_member_sequence(members_input: Sequence[SinkEffectMember], field_name: str) -> tuple[SinkEffectMember, ...]:
    members = tuple(members_input)
    ordinals: list[int] = []
    for member in members:
        if not isinstance(member, SinkEffectMember):
            raise TypeError(f"{field_name} entries must be SinkEffectMember")
        ordinals.append(member.ordinal)
    if len(ordinals) != len(set(ordinals)):
        raise ValueError(f"{field_name} ordinals must be unique")
    if ordinals != list(range(len(ordinals))):
        raise ValueError(f"{field_name} ordinals must be dense and ordered from zero")
    return members


def _freeze_ordinal_sequence(ordinals_input: Sequence[int], field_name: str) -> tuple[int, ...]:
    ordinals = tuple(ordinals_input)
    for ordinal in ordinals:
        require_int(ordinal, f"{field_name} ordinal")
        if ordinal < 0:
            raise ValueError(f"{field_name} ordinals must be non-negative")
    if len(ordinals) != len(set(ordinals)):
        raise ValueError(f"{field_name} ordinals must be unique")
    return ordinals


@dataclass(frozen=True, slots=True)
class SinkEffectMemberCandidate:
    """One exact sink-boundary row awaiting durable lineage ordering."""

    token_id: str
    row: Mapping[str, object]
    pending_identity: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        _require_nonempty_string(self.token_id, "token_id")
        frozen_row = _freeze_canonical_row_value(self.row, "row")
        if not isinstance(frozen_row, Mapping):
            raise TypeError("row must be a mapping")
        object.__setattr__(self, "row", deep_freeze(frozen_row))
        object.__setattr__(self, "pending_identity", _freeze_bounded_evidence(self.pending_identity, "pending_identity"))
        freeze_fields(self, "pending_identity")


@dataclass(frozen=True, slots=True)
class SinkEffectMember:
    ordinal: int
    token_id: str
    row_id: str
    ingest_sequence: int
    lineage_json: str
    lineage_hash: str
    payload_hash: str
    row: Mapping[str, object]
    pending_identity_hash: str = field(default=_EMPTY_IDENTITY_HASH)
    member_effect_id: str | None = None

    def __post_init__(self) -> None:
        require_int(self.ordinal, "ordinal", min_value=0)
        require_int(self.ingest_sequence, "ingest_sequence", min_value=0)
        for field_name in ("token_id", "row_id", "lineage_json"):
            _require_nonempty_string(getattr(self, field_name), field_name)
        for field_name in ("lineage_hash", "payload_hash", "pending_identity_hash"):
            _require_lower_hex_64(getattr(self, field_name), field_name)
        try:
            lineage = json.loads(self.lineage_json)
        except json.JSONDecodeError as exc:
            raise ValueError("lineage_json must be canonical JSON") from exc
        if type(lineage) is not list or canonical_json(lineage) != self.lineage_json:
            raise ValueError("lineage_json must be an exact canonical ordered list")
        if len(self.lineage_json.encode("utf-8")) > _SINK_EFFECT_EVIDENCE_MAX_BYTES:
            raise ValueError("lineage_json exceeds the 64 KiB limit")
        if sha256(self.lineage_json.encode("utf-8")).hexdigest() != self.lineage_hash:
            raise ValueError("lineage_hash must bind exact lineage_json")
        if self.member_effect_id is not None:
            _require_lower_hex_64(self.member_effect_id, "member_effect_id")
        frozen_row = _freeze_canonical_row_value(self.row, "row")
        if not isinstance(frozen_row, Mapping):
            raise TypeError("row must be a mapping")
        if sha256(canonical_json(deep_thaw(frozen_row)).encode("utf-8")).hexdigest() != self.payload_hash:
            raise ValueError("payload_hash must bind the exact canonical row")
        object.__setattr__(self, "row", deep_freeze(frozen_row))


@dataclass(frozen=True, slots=True)
class SinkEffectReservationRequest:
    """Complete, credential-free authority required to reserve one effect."""

    run_id: str
    sink_node_id: str
    role: SinkEffectRole
    input_kind: SinkEffectInputKind
    requested_target_hash: str
    members: Sequence[SinkEffectMember]
    audit_export_snapshot_id: str | None
    config_hash: str
    replacing_target: bool
    primary_effect_id: str | None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.run_id, "run_id")
        _require_nonempty_string(self.sink_node_id, "sink_node_id")
        if type(self.role) is not SinkEffectRole:
            raise TypeError("role must be exact SinkEffectRole")
        if type(self.input_kind) is not SinkEffectInputKind:
            raise TypeError("input_kind must be exact SinkEffectInputKind")
        _require_lower_hex_64(self.requested_target_hash, "requested_target_hash")
        _require_lower_hex_64(self.config_hash, "config_hash")
        if self.primary_effect_id is not None:
            _require_lower_hex_64(self.primary_effect_id, "primary_effect_id")
        if type(self.replacing_target) is not bool:
            raise TypeError("replacing_target must be exact bool")

        members = tuple(self.members)
        if any(type(member) is not SinkEffectMember for member in members):
            raise TypeError("members must contain exact SinkEffectMember values")
        if len({member.token_id for member in members}) != len(members):
            raise ValueError("members must contain unique token IDs")
        ordinals = [member.ordinal for member in members]
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("members must carry unique source ordinals")
        members = tuple(
            replace(member, ordinal=ordinal, member_effect_id=None)
            for ordinal, member in enumerate(sorted(members, key=lambda member: member.ordinal))
        )
        object.__setattr__(self, "members", members)
        freeze_fields(self, "members")

        if self.role is SinkEffectRole.PRIMARY and self.primary_effect_id is not None:
            raise ValueError("primary effects cannot refer to another primary effect")
        if self.role is SinkEffectRole.FAILSINK and self.primary_effect_id is None:
            raise ValueError("failsink effects require primary_effect_id")

        if self.input_kind is SinkEffectInputKind.PIPELINE_MEMBERS:
            if not members:
                raise ValueError("pipeline reservation requires at least one member")
            if self.audit_export_snapshot_id is not None:
                raise ValueError("pipeline reservation cannot carry an audit export snapshot")
        else:
            if members:
                raise ValueError("audit export reservation cannot carry pipeline members")
            _require_lower_hex_64(self.audit_export_snapshot_id, "audit_export_snapshot_id")
            if self.config_hash != self.requested_target_hash:
                raise ValueError("audit export config_hash must equal requested_target_hash")


@dataclass(frozen=True, slots=True)
class SinkEffectAttemptRequest:
    effect_id: str
    member_ordinal: int | None
    generation: int
    action: SinkEffectAttemptAction
    call_kind: CallType
    request_hash: str

    def __post_init__(self) -> None:
        _require_lower_hex_64(self.effect_id, "effect_id")
        _require_lower_hex_64(self.request_hash, "request_hash")
        if self.member_ordinal is not None and (type(self.member_ordinal) is not int or self.member_ordinal < 0):
            raise ValueError("member_ordinal must be a non-negative exact int or None")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative exact int")
        if type(self.action) is not SinkEffectAttemptAction:
            raise TypeError("action must be exact SinkEffectAttemptAction")
        if type(self.call_kind) is not CallType:
            raise TypeError("call_kind must be exact CallType")


@dataclass(frozen=True, slots=True)
class SinkEffectAttemptResult:
    attempt_id: str
    evidence: Mapping[str, object]
    latency_ms: float

    def __post_init__(self) -> None:
        _require_lower_hex_64(self.attempt_id, "attempt_id")
        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, int | float):
            raise TypeError("latency_ms must be a finite number")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        object.__setattr__(self, "latency_ms", float(self.latency_ms))
        object.__setattr__(self, "evidence", _freeze_bounded_evidence(self.evidence, "evidence"))
        freeze_fields(self, "evidence")


@dataclass(frozen=True, slots=True)
class SinkEffectLease:
    effect_id: str
    owner: str
    generation: int
    expires_at: datetime

    def __post_init__(self) -> None:
        _require_lower_hex_64(self.effect_id, "effect_id")
        if not isinstance(self.owner, str) or not self.owner.strip():
            raise ValueError("lease owner must be non-empty")
        if len(self.owner) > 128:
            raise ValueError("lease owner exceeds 128 characters")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("lease generation must be a positive exact int")
        if not isinstance(self.expires_at, datetime) or self.expires_at.tzinfo is None:
            raise ValueError("lease expires_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class SinkEffectFinalizationMember:
    """One member's state and terminal-outcome writes."""

    ordinal: int
    output_data: Mapping[str, object]
    duration_ms: float
    outcome: TerminalOutcome
    path: TerminalPath
    sink_name: str | None = None
    batch_id: str | None = None
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None
    error_hash: str | None = None
    context: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("ordinal must be a non-negative exact int")
        output_data = self.output_data
        if not isinstance(output_data, Mapping):
            raise TypeError("output_data must be a mapping")
        if isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, int | float):
            raise TypeError("duration_ms must be a finite non-negative number")
        if not math.isfinite(self.duration_ms) or self.duration_ms < 0:
            raise ValueError("duration_ms must be finite and non-negative")
        if type(self.outcome) is not TerminalOutcome or type(self.path) is not TerminalPath:
            raise TypeError("outcome and path must be exact terminal enums")
        object.__setattr__(self, "duration_ms", float(self.duration_ms))
        object.__setattr__(self, "output_data", deep_freeze(self.output_data))
        if self.context is not None:
            object.__setattr__(self, "context", _freeze_bounded_evidence(self.context, "evidence"))
        freeze_fields(self, "output_data", "context")


@dataclass(frozen=True, slots=True)
class SinkEffectFinalizeRequest:
    """Complete one exact, generation-fenced effect winner."""

    effect_id: str
    lease_owner: str | None
    generation: int
    descriptor: ArtifactDescriptor
    publication_performed: bool
    publication_evidence_kind: ArtifactPublicationEvidenceKind
    accepted_ordinals: Sequence[int]
    diverted_ordinals: Sequence[int]
    evidence: Mapping[str, object]
    members: Sequence[SinkEffectFinalizationMember]
    attempt_id: str | None = None
    reconcile_kind: SinkEffectReconcileKind | None = None
    operation_duration_ms: float = 0.0

    def __post_init__(self) -> None:
        _require_lower_hex_64(self.effect_id, "effect_id")
        if self.lease_owner is not None and (not isinstance(self.lease_owner, str) or not self.lease_owner.strip()):
            raise ValueError("lease_owner must be non-empty or None")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative exact int")
        if type(self.descriptor) is not ArtifactDescriptor:
            raise TypeError("descriptor must be exact ArtifactDescriptor")
        require_no_artifact_uri_credentials(self.descriptor.path_or_uri)
        if type(self.publication_performed) is not bool:
            raise TypeError("publication_performed must be exact bool")
        expected_performed = _EVIDENCE_PERFORMED.get(self.publication_evidence_kind)
        if expected_performed is None or expected_performed is not self.publication_performed:
            raise ValueError("publication evidence kind contradicts publication_performed")
        accepted = tuple(self.accepted_ordinals)
        diverted = tuple(self.diverted_ordinals)
        for field_name, values in (("accepted_ordinals", accepted), ("diverted_ordinals", diverted)):
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"{field_name} must contain non-negative exact ints")
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} must be unique and ascending")
        if set(accepted) & set(diverted):
            raise ValueError("accepted and diverted ordinals must be disjoint")
        finalization_members = tuple(self.members)
        if any(type(member) is not SinkEffectFinalizationMember for member in finalization_members):
            raise TypeError("members must contain exact SinkEffectFinalizationMember values")
        member_ordinals = tuple(member.ordinal for member in finalization_members)
        if member_ordinals != accepted:
            raise ValueError("finalization member ordinals must exactly equal accepted_ordinals")
        if self.attempt_id is not None:
            _require_lower_hex_64(self.attempt_id, "attempt_id")
        if self.reconcile_kind is not None and type(self.reconcile_kind) is not SinkEffectReconcileKind:
            raise TypeError("reconcile_kind must be exact SinkEffectReconcileKind or None")
        if isinstance(self.operation_duration_ms, bool) or not isinstance(self.operation_duration_ms, int | float):
            raise TypeError("operation_duration_ms must be a finite non-negative number")
        if not math.isfinite(self.operation_duration_ms) or self.operation_duration_ms < 0:
            raise ValueError("operation_duration_ms must be finite and non-negative")
        object.__setattr__(self, "accepted_ordinals", accepted)
        object.__setattr__(self, "diverted_ordinals", diverted)
        object.__setattr__(self, "members", finalization_members)
        object.__setattr__(self, "evidence", _freeze_bounded_evidence(self.evidence, "evidence"))
        object.__setattr__(self, "operation_duration_ms", float(self.operation_duration_ms))
        freeze_fields(self, "accepted_ordinals", "diverted_ordinals", "evidence", "members")


@dataclass(frozen=True, slots=True)
class SinkEffectFinalizationResult:
    effect: SinkEffect
    artifact: Artifact
    state_ids: tuple[str, ...]
    outcome_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SinkEffectIdentity:
    """Credential-free deterministic identities computed before reservation."""

    effect_id: str
    artifact_id: str
    artifact_idempotency_key: str
    stream_id: str
    config_hash: str
    requested_target_hash: str
    membership_or_manifest_hash: str
    group_payload_hash: str
    input_kind: SinkEffectInputKind
    members: Sequence[SinkEffectMember]
    member_ids: Sequence[str]
    snapshot_hash: str | None = None
    final_manifest_identity_hash: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "effect_id",
            "artifact_id",
            "artifact_idempotency_key",
            "stream_id",
            "config_hash",
            "requested_target_hash",
            "membership_or_manifest_hash",
            "group_payload_hash",
        ):
            _require_lower_hex_64(getattr(self, field_name), field_name)
        for field_name in ("snapshot_hash", "final_manifest_identity_hash"):
            value = getattr(self, field_name)
            if value is not None:
                _require_lower_hex_64(value, field_name)
        _require_exact_enum(self.input_kind, SinkEffectInputKind, "input_kind")
        members = tuple(self.members)
        member_ids = tuple(self.member_ids)
        if any(type(member) is not SinkEffectMember for member in members):
            raise TypeError("members must contain exact SinkEffectMember values")
        if any(not isinstance(member_id, str) or _LOWER_HEX_64.fullmatch(member_id) is None for member_id in member_ids):
            raise ValueError("member_ids must contain lowercase SHA-256 digests")
        if self.input_kind is SinkEffectInputKind.PIPELINE_MEMBERS:
            if not members or member_ids != tuple(member.member_effect_id for member in members):
                raise ValueError("pipeline identity requires exact non-empty member IDs")
            if self.snapshot_hash is not None or self.final_manifest_identity_hash is not None:
                raise ValueError("pipeline identity cannot carry audit-export hashes")
        elif members or member_ids or self.snapshot_hash is None or self.final_manifest_identity_hash is None:
            raise ValueError("audit-export identity requires zero members and complete snapshot hashes")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "member_ids", member_ids)
        freeze_fields(self, "members", "member_ids")


@dataclass(frozen=True, slots=True)
class SinkEffectPipelineMembersInput:
    members: Sequence[SinkEffectMember]
    target_snapshot_members: Sequence[SinkEffectMember]

    def __post_init__(self) -> None:
        members = _freeze_member_sequence(self.members, "members")
        if not members:
            raise ValueError("members must be non-empty")
        target_snapshot_members = _freeze_member_sequence(self.target_snapshot_members, "target_snapshot_members")
        object.__setattr__(self, "members", tuple(members))
        object.__setattr__(self, "target_snapshot_members", tuple(target_snapshot_members))

    @property
    def input_kind(self) -> SinkEffectInputKind:
        return SinkEffectInputKind.PIPELINE_MEMBERS


@dataclass(frozen=True, slots=True)
class AuditExportSnapshotChunkInput:
    ordinal: int
    content_ref: str
    content_hash: str
    size_bytes: int
    record_count: int

    def __post_init__(self) -> None:
        require_int(self.ordinal, "ordinal", min_value=0)
        _validate_content_descriptor(
            content_ref=self.content_ref,
            content_hash=self.content_hash,
            size_bytes=self.size_bytes,
            field_prefix="",
            max_size_bytes=_AUDIT_EXPORT_MAX_CHUNK_BYTES,
        )
        _require_bounded_positive_int(self.record_count, "record_count", _AUDIT_EXPORT_MAX_CHUNK_RECORDS)


@dataclass(frozen=True, slots=True)
class AuditExportSignedManifestInput:
    content_ref: str
    content_hash: str
    size_bytes: int
    manifest_schema: str
    derivation_version: str
    signature_algorithm: AuditExportSigningMode
    signature_key_id: str
    record_chain_algorithm: str
    final_hash: str
    signature: str | None

    def __post_init__(self) -> None:
        _validate_content_descriptor(
            content_ref=self.content_ref,
            content_hash=self.content_hash,
            size_bytes=self.size_bytes,
            field_prefix="signed_manifest_",
            max_size_bytes=_AUDIT_EXPORT_MANIFEST_MAX_BYTES,
        )
        if self.manifest_schema != _AUDIT_EXPORT_MANIFEST_SCHEMA:
            raise ValueError(f"manifest_schema must equal {_AUDIT_EXPORT_MANIFEST_SCHEMA!r}")
        if self.derivation_version != _AUDIT_EXPORT_DERIVATION_VERSION:
            raise ValueError(f"derivation_version must equal {_AUDIT_EXPORT_DERIVATION_VERSION!r}")
        _require_exact_enum(self.signature_algorithm, AuditExportSigningMode, "signature_algorithm")
        _require_lower_hex_64(self.final_hash, "final_hash")
        _require_nonempty_string(self.signature_key_id, "signature_key_id")
        _reject_credential_bearing_reference(self.signature_key_id, "signature_key_id")
        if self.signature_algorithm is AuditExportSigningMode.UNSIGNED:
            if self.signature_key_id != "UNSIGNED":
                raise ValueError("unsigned signature_key_id must equal 'UNSIGNED'")
            if self.record_chain_algorithm != _UNSIGNED_RECORD_CHAIN:
                raise ValueError(f"unsigned record_chain_algorithm must equal {_UNSIGNED_RECORD_CHAIN!r}")
            if self.signature is not None:
                raise ValueError("unsigned signature must be None")
        else:
            if self.signature_key_id == "UNSIGNED":
                raise ValueError("HMAC signature_key_id must not equal 'UNSIGNED'")
            if self.record_chain_algorithm != _HMAC_RECORD_CHAIN:
                raise ValueError(f"HMAC record_chain_algorithm must equal {_HMAC_RECORD_CHAIN!r}")
            _require_lower_hex_64(self.signature, "signature")


@dataclass(frozen=True, slots=True)
class _AuditExportReaderBinding:
    snapshot_id: str
    source_run_id: str
    registry_key_hash: str
    manifest_hash: str
    snapshot_hash: str
    export_format: AuditExportFormat
    signing_mode: AuditExportSigningMode
    signer_key_id: str
    record_count: int
    total_bytes: int
    serialization_version: str
    exported_at: str
    source_completed_at: str
    source_status: str
    last_chunk_seal_hash: str
    snapshot_seal_hash: str


@dataclass(frozen=True, slots=True)
class _AuditExportReaderStoreAccess:
    resolve: Callable[[str], bytes]
    count_records: Callable[[bytes], int]


@final
class RestrictedAuditExportSnapshotReader:
    """Factory-created bound capability with no arbitrary-reference API."""

    __slots__ = ("__binding", "__chunks", "__limits", "__signed_manifest", "__store_resolver")
    __binding: _AuditExportReaderBinding
    __chunks: tuple[AuditExportSnapshotChunkInput, ...]
    __signed_manifest: AuditExportSignedManifestInput
    __limits: tuple[int, int, int, int, int]
    __store_resolver: _AuditExportReaderStoreAccess

    def __init__(self) -> None:
        raise TypeError("RestrictedAuditExportSnapshotReader is factory-created only")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("RestrictedAuditExportSnapshotReader is immutable")

    def __delattr__(self, _name: str) -> None:
        raise TypeError("RestrictedAuditExportSnapshotReader is immutable")

    @property
    def snapshot_id(self) -> str:
        return self.__binding.snapshot_id

    @property
    def manifest_hash(self) -> str:
        return self.__binding.manifest_hash

    @property
    def chunk_count(self) -> int:
        return len(self.__chunks)

    def iter_verified_chunks(self) -> Iterator[bytes]:
        cumulative_bytes = 0
        cumulative_records = 0
        max_total_bytes, max_total_records, max_chunks, max_chunk_bytes, max_chunk_records = self.__limits
        if len(self.__chunks) > max_chunks:
            raise ValueError("reader chunk_count exceeds its configured bound")
        for expected_ordinal, descriptor in enumerate(self.__chunks):
            if descriptor.ordinal != expected_ordinal:
                raise ValueError("reader chunk ordinals are not dense and ordered")
            if descriptor.size_bytes > max_chunk_bytes:
                raise ValueError("reader chunk size_bytes exceeds max_chunk_bytes")
            if descriptor.record_count > max_chunk_records:
                raise ValueError("reader chunk record_count exceeds max_chunk_records")
            content = self.__store_resolver.resolve(descriptor.content_ref)
            if len(content) > max_chunk_bytes:
                raise ValueError("reader observed chunk bytes exceed max_chunk_bytes")
            _verify_content_bytes(content, descriptor.content_hash, descriptor.size_bytes, "chunk")
            observed_records = self.__store_resolver.count_records(content)
            require_int(observed_records, "observed chunk record_count", min_value=0)
            if observed_records > max_chunk_records:
                raise ValueError("reader observed chunk records exceed max_chunk_records")
            if observed_records != descriptor.record_count:
                raise ValueError("reader chunk record_count does not match its bound descriptor")
            cumulative_bytes += len(content)
            cumulative_records += observed_records
            if cumulative_bytes > max_total_bytes or cumulative_records > max_total_records:
                raise ValueError("reader cumulative data exceeds its configured bound")
            yield content
        if cumulative_bytes != self.__binding.total_bytes:
            raise ValueError("reader cumulative total_bytes does not match its binding")
        if cumulative_records != self.__binding.record_count:
            raise ValueError("reader cumulative record_count does not match its binding")

    def read_verified_signed_manifest(self) -> bytes:
        content = self.__store_resolver.resolve(self.__signed_manifest.content_ref)
        _verify_content_bytes(
            content,
            self.__signed_manifest.content_hash,
            self.__signed_manifest.size_bytes,
            "signed manifest",
        )
        _verify_signed_manifest_bytes(content, self.__binding, self.__signed_manifest, len(self.__chunks))
        return content

    def _binding_matches(
        self,
        *,
        snapshot_id: str,
        source_run_id: str,
        registry_key_hash: str,
        manifest_hash: str,
        snapshot_hash: str,
        export_format: AuditExportFormat,
        signing_mode: AuditExportSigningMode,
        signer_key_id: str,
        record_count: int,
        total_bytes: int,
        serialization_version: str,
        chunks: tuple[AuditExportSnapshotChunkInput, ...],
        signed_manifest: AuditExportSignedManifestInput,
    ) -> bool:
        binding = self.__binding
        return (
            binding.snapshot_id == snapshot_id
            and binding.source_run_id == source_run_id
            and binding.registry_key_hash == registry_key_hash
            and binding.manifest_hash == manifest_hash
            and binding.snapshot_hash == snapshot_hash
            and binding.export_format is export_format
            and binding.signing_mode is signing_mode
            and binding.signer_key_id == signer_key_id
            and binding.record_count == record_count
            and binding.total_bytes == total_bytes
            and binding.serialization_version == serialization_version
            and self.__chunks == chunks
            and self.__signed_manifest == signed_manifest
        )

    def __deepcopy__(self, _memo: dict[int, object]) -> NoReturn:
        raise TypeError("RestrictedAuditExportSnapshotReader cannot be serialized")

    def __reduce__(self) -> NoReturn:
        raise TypeError("RestrictedAuditExportSnapshotReader cannot be serialized")


def _verify_content_bytes(content: object, expected_hash: str, expected_size: int, label: str) -> None:
    if not isinstance(content, bytes):
        raise TypeError(f"{label} resolver must return bytes")
    if len(content) != expected_size:
        raise ValueError(f"{label} size does not match its bound descriptor")
    if sha256(content).hexdigest() != expected_hash:
        raise ValueError(f"{label} hash does not match its bound descriptor")


def _verify_signed_manifest_bytes(
    content: bytes,
    binding: _AuditExportReaderBinding,
    descriptor: AuditExportSignedManifestInput,
    chunk_count: int,
) -> None:
    try:
        decoded = content.decode("utf-8")
        manifest = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("signed manifest must be canonical JSON UTF-8") from exc
    if not isinstance(manifest, dict) or canonical_json(manifest).encode("utf-8") != content:
        raise ValueError("signed manifest must use exact canonical JSON bytes")
    if set(manifest) != _V2_MANIFEST_FIELDS:
        raise ValueError("signed manifest must contain the exact v2 field set")
    for field_name in ("chunk_count", "record_count", "total_bytes"):
        if type(manifest[field_name]) is not int:
            raise TypeError(f"signed manifest {field_name} must be exact int")
        if manifest[field_name] < 1:
            raise ValueError(f"signed manifest {field_name} must be strictly positive")
    for field_name in _V2_MANIFEST_FIELDS - {"chunk_count", "record_count", "signature", "total_bytes"}:
        if type(manifest[field_name]) is not str:
            raise TypeError(f"signed manifest {field_name} must be exact str")
    if manifest["signature"] is not None and type(manifest["signature"]) is not str:
        raise TypeError("signed manifest signature must be exact str or None")
    for field_name in (
        "final_hash",
        "last_chunk_seal_hash",
        "manifest_hash",
        "registry_key_hash",
        "snapshot_hash",
        "snapshot_id",
        "snapshot_seal_hash",
    ):
        _require_lower_hex_64(manifest[field_name], f"signed manifest {field_name}")
    for field_name in ("exported_at", "source_completed_at"):
        _require_utc_microsecond_timestamp(manifest[field_name], f"signed manifest {field_name}")
    if manifest["hash_algorithm"] != "sha256":
        raise ValueError("signed manifest hash_algorithm must equal 'sha256'")
    if manifest["record_type"] != "manifest":
        raise ValueError("signed manifest record_type must equal 'manifest'")
    if manifest["source_status"] not in {"completed", "completed_with_failures", "empty"}:
        raise ValueError("signed manifest source_status is not export-terminal")
    expected_fields: dict[str, object] = {
        "schema": descriptor.manifest_schema,
        "derivation_version": descriptor.derivation_version,
        "signature_algorithm": descriptor.signature_algorithm.value,
        "signature_key_id": descriptor.signature_key_id,
        "record_chain_algorithm": descriptor.record_chain_algorithm,
        "final_hash": descriptor.final_hash,
        "signature": descriptor.signature,
        "snapshot_id": binding.snapshot_id,
        "snapshot_hash": binding.snapshot_hash,
        "manifest_hash": binding.manifest_hash,
        "registry_key_hash": binding.registry_key_hash,
        "run_id": binding.source_run_id,
        "export_format": binding.export_format.value,
        "record_count": binding.record_count,
        "total_bytes": binding.total_bytes,
        "chunk_count": chunk_count,
        "exported_at": binding.exported_at,
        "source_completed_at": binding.source_completed_at,
        "source_status": binding.source_status,
        "last_chunk_seal_hash": binding.last_chunk_seal_hash,
        "snapshot_seal_hash": binding.snapshot_seal_hash,
    }
    for field_name, expected in expected_fields.items():
        if manifest[field_name] != expected:
            raise ValueError(f"signed manifest {field_name} does not match its bound descriptor")


def _create_restricted_audit_export_snapshot_reader(
    *,
    snapshot_id: str,
    source_run_id: str,
    registry_key_hash: str,
    manifest_hash: str,
    snapshot_hash: str,
    export_format: AuditExportFormat,
    signing_mode: AuditExportSigningMode,
    signer_key_id: str,
    record_count: int,
    total_bytes: int,
    serialization_version: str,
    exported_at: str,
    source_completed_at: str,
    source_status: str,
    last_chunk_seal_hash: str,
    snapshot_seal_hash: str,
    chunks: Sequence[AuditExportSnapshotChunkInput],
    signed_manifest: AuditExportSignedManifestInput,
    store_resolver: Callable[[str], bytes],
    record_counter: Callable[[bytes], int],
    signed_manifest_verifier: Callable[[bytes, AuditExportSignedManifestInput], None],
    max_total_bytes: int = _AUDIT_EXPORT_MAX_TOTAL_BYTES,
    max_total_records: int = _AUDIT_EXPORT_MAX_TOTAL_RECORDS,
    max_chunks: int = _AUDIT_EXPORT_MAX_CHUNKS,
    max_chunk_bytes: int = _AUDIT_EXPORT_MAX_CHUNK_BYTES,
    max_chunk_records: int = _AUDIT_EXPORT_MAX_CHUNK_RECORDS,
) -> RestrictedAuditExportSnapshotReader:
    """Internal trusted factory used after registry/store verification."""
    chunk_tuple = tuple(chunks)
    _require_bounded_positive_int(max_total_bytes, "max_total_bytes", _AUDIT_EXPORT_MAX_TOTAL_BYTES)
    _require_bounded_positive_int(max_total_records, "max_total_records", _AUDIT_EXPORT_MAX_TOTAL_RECORDS)
    _require_bounded_positive_int(max_chunks, "max_chunks", _AUDIT_EXPORT_MAX_CHUNKS)
    _require_bounded_positive_int(max_chunk_bytes, "max_chunk_bytes", _AUDIT_EXPORT_MAX_CHUNK_BYTES)
    _require_bounded_positive_int(max_chunk_records, "max_chunk_records", _AUDIT_EXPORT_MAX_CHUNK_RECORDS)
    _require_utc_microsecond_timestamp(exported_at, "exported_at")
    _require_utc_microsecond_timestamp(source_completed_at, "source_completed_at")
    if not isinstance(source_status, str) or source_status not in {"completed", "completed_with_failures", "empty"}:
        raise ValueError("source_status is not export-terminal")
    _require_lower_hex_64(last_chunk_seal_hash, "last_chunk_seal_hash")
    _require_lower_hex_64(snapshot_seal_hash, "snapshot_seal_hash")
    if serialization_version != _AUDIT_EXPORT_SERIALIZATION_VERSION:
        raise ValueError(f"serialization_version must equal {_AUDIT_EXPORT_SERIALIZATION_VERSION!r}")
    if total_bytes > max_total_bytes or record_count > max_total_records or len(chunk_tuple) > max_chunks:
        raise ValueError("snapshot exceeds configured reader limits")
    for chunk in chunk_tuple:
        if chunk.size_bytes > max_chunk_bytes:
            raise ValueError("snapshot chunk size_bytes exceeds max_chunk_bytes")
        if chunk.record_count > max_chunk_records:
            raise ValueError("snapshot chunk record_count exceeds max_chunk_records")
    binding = _AuditExportReaderBinding(
        snapshot_id=snapshot_id,
        source_run_id=source_run_id,
        registry_key_hash=registry_key_hash,
        manifest_hash=manifest_hash,
        snapshot_hash=snapshot_hash,
        export_format=export_format,
        signing_mode=signing_mode,
        signer_key_id=signer_key_id,
        record_count=record_count,
        total_bytes=total_bytes,
        serialization_version=serialization_version,
        exported_at=exported_at,
        source_completed_at=source_completed_at,
        source_status=source_status,
        last_chunk_seal_hash=last_chunk_seal_hash,
        snapshot_seal_hash=snapshot_seal_hash,
    )
    manifest_bytes = store_resolver(signed_manifest.content_ref)
    _verify_content_bytes(manifest_bytes, signed_manifest.content_hash, signed_manifest.size_bytes, "signed manifest")
    _verify_signed_manifest_bytes(manifest_bytes, binding, signed_manifest, len(chunk_tuple))
    signed_manifest_verifier(manifest_bytes, signed_manifest)

    reader = object.__new__(RestrictedAuditExportSnapshotReader)
    object.__setattr__(reader, "_RestrictedAuditExportSnapshotReader__binding", binding)
    object.__setattr__(reader, "_RestrictedAuditExportSnapshotReader__chunks", chunk_tuple)
    object.__setattr__(reader, "_RestrictedAuditExportSnapshotReader__signed_manifest", signed_manifest)
    object.__setattr__(
        reader,
        "_RestrictedAuditExportSnapshotReader__limits",
        (max_total_bytes, max_total_records, max_chunks, max_chunk_bytes, max_chunk_records),
    )
    object.__setattr__(
        reader,
        "_RestrictedAuditExportSnapshotReader__store_resolver",
        _AuditExportReaderStoreAccess(resolve=store_resolver, count_records=record_counter),
    )
    return reader


@dataclass(frozen=True, slots=True)
class SinkEffectAuditExportSnapshotInput:
    snapshot_id: str
    source_run_id: str
    registry_key_hash: str
    manifest_hash: str
    snapshot_hash: str
    serialization_version: str
    export_format: AuditExportFormat
    signing_mode: AuditExportSigningMode
    signer_key_id: str
    record_count: int
    total_bytes: int
    chunk_count: int
    chunks: Sequence[AuditExportSnapshotChunkInput]
    signed_manifest: AuditExportSignedManifestInput
    reader: RestrictedAuditExportSnapshotReader = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        _require_lower_hex_64(self.snapshot_id, "snapshot_id")
        _require_nonempty_string(self.source_run_id, "source_run_id")
        for field_name in ("registry_key_hash", "manifest_hash", "snapshot_hash"):
            _require_lower_hex_64(getattr(self, field_name), field_name)
        if self.serialization_version != _AUDIT_EXPORT_SERIALIZATION_VERSION:
            raise ValueError(f"serialization_version must equal {_AUDIT_EXPORT_SERIALIZATION_VERSION!r}")
        _require_exact_enum(self.export_format, AuditExportFormat, "export_format")
        _require_exact_enum(self.signing_mode, AuditExportSigningMode, "signing_mode")
        _require_nonempty_string(self.signer_key_id, "signer_key_id")
        _reject_credential_bearing_reference(self.signer_key_id, "signer_key_id")
        _require_bounded_positive_int(self.record_count, "record_count", _AUDIT_EXPORT_MAX_TOTAL_RECORDS)
        _require_bounded_positive_int(self.total_bytes, "total_bytes", _AUDIT_EXPORT_MAX_TOTAL_BYTES)
        _require_bounded_positive_int(self.chunk_count, "chunk_count", _AUDIT_EXPORT_MAX_CHUNKS)

        chunks = tuple(self.chunks)
        if any(not isinstance(chunk, AuditExportSnapshotChunkInput) for chunk in chunks):
            raise TypeError("chunks entries must be AuditExportSnapshotChunkInput")
        ordinals = [chunk.ordinal for chunk in chunks]
        if ordinals != list(range(len(chunks))):
            raise ValueError("chunks ordinals must be dense and ordered from zero")
        if len(chunks) != self.chunk_count:
            raise ValueError("chunk_count must equal len(chunks)")
        if sum(chunk.record_count for chunk in chunks) != self.record_count:
            raise ValueError("record_count must equal the exact chunk record_count sum")
        if sum(chunk.size_bytes for chunk in chunks) != self.total_bytes:
            raise ValueError("total_bytes must equal the exact chunk size_bytes sum")
        object.__setattr__(self, "chunks", tuple(chunks))

        if not isinstance(self.signed_manifest, AuditExportSignedManifestInput):
            raise TypeError("signed_manifest must be AuditExportSignedManifestInput")
        if self.signed_manifest.content_ref in {chunk.content_ref for chunk in chunks}:
            raise ValueError("signed manifest must be separate from data chunks")
        if self.signed_manifest.signature_algorithm is not self.signing_mode:
            raise ValueError("signed_manifest signature_algorithm must equal signing_mode")
        if self.signed_manifest.signature_key_id != self.signer_key_id:
            raise ValueError("signed_manifest signature_key_id must equal signer_key_id")
        if type(self.reader) is not RestrictedAuditExportSnapshotReader:
            raise TypeError("reader must be a factory-created RestrictedAuditExportSnapshotReader")
        if not self.reader._binding_matches(
            snapshot_id=self.snapshot_id,
            source_run_id=self.source_run_id,
            registry_key_hash=self.registry_key_hash,
            manifest_hash=self.manifest_hash,
            snapshot_hash=self.snapshot_hash,
            export_format=self.export_format,
            signing_mode=self.signing_mode,
            signer_key_id=self.signer_key_id,
            record_count=self.record_count,
            total_bytes=self.total_bytes,
            serialization_version=self.serialization_version,
            chunks=chunks,
            signed_manifest=self.signed_manifest,
        ):
            raise ValueError("reader binding must exactly match the audit export snapshot input")

    @property
    def input_kind(self) -> SinkEffectInputKind:
        return SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT


@dataclass(frozen=True, slots=True)
class SinkEffectInspectionRequest:
    effect_id: str
    target: str
    predecessor_descriptor: ArtifactDescriptor | None
    input_kind: SinkEffectInputKind = SinkEffectInputKind.PIPELINE_MEMBERS

    def __post_init__(self) -> None:
        _require_nonempty_string(self.effect_id, "effect_id")
        _reject_credential_bearing_reference(self.target, "target")
        if self.predecessor_descriptor is not None and not isinstance(self.predecessor_descriptor, ArtifactDescriptor):
            raise TypeError("predecessor_descriptor must be ArtifactDescriptor or None")
        _require_exact_enum(self.input_kind, SinkEffectInputKind, "input_kind")


@dataclass(frozen=True, slots=True)
class SinkEffectInspection:
    mode: SinkEffectInspectionMode
    reference: str
    evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_exact_enum(self.mode, SinkEffectInspectionMode, "mode")
        _reject_credential_bearing_reference(self.reference, "reference")
        frozen_evidence = _freeze_bounded_evidence(self.evidence, "evidence")
        object.__setattr__(self, "evidence", deep_freeze(frozen_evidence))


@dataclass(frozen=True, slots=True)
class SinkEffectPrepareRequest:
    effect_id: str
    effect_input: SinkEffectPipelineMembersInput | SinkEffectAuditExportSnapshotInput
    inspection: SinkEffectInspection

    def __post_init__(self) -> None:
        _require_nonempty_string(self.effect_id, "effect_id")
        if not isinstance(self.effect_input, (SinkEffectPipelineMembersInput, SinkEffectAuditExportSnapshotInput)):
            raise TypeError("effect_input must be a member of the closed sink effect input union")
        if not isinstance(self.inspection, SinkEffectInspection):
            raise TypeError("inspection must be SinkEffectInspection")

    @property
    def input_kind(self) -> SinkEffectInputKind:
        return self.effect_input.input_kind

    def validate_plan(self, plan: SinkEffectPlan) -> None:
        if not isinstance(plan, SinkEffectPlan):
            raise TypeError("plan must be SinkEffectPlan")
        if plan.effect_id != self.effect_id:
            raise ValueError("plan effect_id must equal request effect_id")
        if plan.input_kind is not self.input_kind:
            raise ValueError("plan input kind must equal the request-derived input kind")


@dataclass(frozen=True, slots=True)
class SinkEffectPlan:
    effect_id: str
    protocol_version: str
    input_kind: SinkEffectInputKind
    descriptor_mode: SinkEffectDescriptorMode
    inspection_mode: SinkEffectInspectionMode
    target: str
    plan_hash: str
    payload_hash: str
    expected_descriptor: ArtifactDescriptor | None
    safe_evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        for field_name in ("effect_id", "plan_hash", "payload_hash"):
            _require_nonempty_string(getattr(self, field_name), field_name)
        if self.protocol_version != SINK_EFFECT_PROTOCOL_VERSION:
            raise ValueError(f"protocol_version must equal {SINK_EFFECT_PROTOCOL_VERSION!r}")
        _require_exact_enum(self.input_kind, SinkEffectInputKind, "input_kind")
        _require_exact_enum(self.descriptor_mode, SinkEffectDescriptorMode, "descriptor_mode")
        _require_exact_enum(self.inspection_mode, SinkEffectInspectionMode, "inspection_mode")
        _reject_credential_bearing_reference(self.target, "target")
        if self.descriptor_mode in {
            SinkEffectDescriptorMode.PRECOMPUTED,
            SinkEffectDescriptorMode.NO_PUBLICATION,
        }:
            if not isinstance(self.expected_descriptor, ArtifactDescriptor):
                raise ValueError(f"{self.descriptor_mode.value} descriptor_mode requires an exact expected_descriptor")
        elif self.expected_descriptor is not None:
            raise ValueError(f"{self.descriptor_mode.value} descriptor_mode must not claim an expected_descriptor")
        frozen_evidence = _freeze_bounded_evidence(self.safe_evidence, "safe_evidence")
        if self.descriptor_mode is SinkEffectDescriptorMode.NO_PUBLICATION:
            publication_kind = frozen_evidence.get("publication_kind")
            if publication_kind not in {"inherited", "virtual"}:
                raise ValueError("NO_PUBLICATION safe_evidence requires publication_kind inherited or virtual")
        object.__setattr__(self, "safe_evidence", deep_freeze(frozen_evidence))


@dataclass(frozen=True, slots=True)
class SinkEffectCommitResult:
    descriptor: ArtifactDescriptor
    evidence: Mapping[str, object]
    accepted_ordinals: Sequence[int]
    diverted_ordinals: Sequence[int]

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, ArtifactDescriptor):
            raise TypeError("descriptor must be ArtifactDescriptor")
        frozen_evidence = _freeze_bounded_evidence(self.evidence, "evidence")
        object.__setattr__(self, "evidence", deep_freeze(frozen_evidence))
        accepted = _freeze_ordinal_sequence(self.accepted_ordinals, "accepted_ordinals")
        diverted = _freeze_ordinal_sequence(self.diverted_ordinals, "diverted_ordinals")
        object.__setattr__(self, "accepted_ordinals", tuple(accepted))
        object.__setattr__(self, "diverted_ordinals", tuple(diverted))
        if set(accepted) & set(diverted):
            raise ValueError("accepted_ordinals and diverted_ordinals must not overlap")


@dataclass(frozen=True, slots=True)
class SinkEffectReconcileResult:
    kind: SinkEffectReconcileKind
    descriptor: ArtifactDescriptor | None = None
    evidence: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    accepted_ordinals: Sequence[int] | None = None
    diverted_ordinals: Sequence[int] | None = None

    def __post_init__(self) -> None:
        _require_exact_enum(self.kind, SinkEffectReconcileKind, "kind")
        if self.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR:
            if not isinstance(self.descriptor, ArtifactDescriptor):
                raise ValueError("APPLIED_WITH_EXACT_DESCRIPTOR requires an exact descriptor")
        elif self.descriptor is not None:
            raise ValueError(f"{self.kind.value} must not carry a descriptor")
        if (self.accepted_ordinals is None) != (self.diverted_ordinals is None):
            raise ValueError("accepted_ordinals and diverted_ordinals must both be present or both be absent")
        if self.accepted_ordinals is not None and self.diverted_ordinals is not None:
            if self.kind is not SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR:
                raise ValueError(f"{self.kind.value} must not carry a result-derived ordinal partition")
            accepted = _freeze_ordinal_sequence(self.accepted_ordinals, "accepted_ordinals")
            diverted = _freeze_ordinal_sequence(self.diverted_ordinals, "diverted_ordinals")
            if set(accepted) & set(diverted):
                raise ValueError("accepted_ordinals and diverted_ordinals must not overlap")
            object.__setattr__(self, "accepted_ordinals", tuple(accepted))
            object.__setattr__(self, "diverted_ordinals", tuple(diverted))
        frozen_evidence = _freeze_bounded_evidence(self.evidence, "evidence")
        object.__setattr__(self, "evidence", deep_freeze(frozen_evidence))

    @property
    def may_commit(self) -> bool:
        return self.kind is SinkEffectReconcileKind.NOT_APPLIED

    @classmethod
    def applied(
        cls,
        descriptor: ArtifactDescriptor,
        *,
        evidence: Mapping[str, object],
        accepted_ordinals: Sequence[int] | None = None,
        diverted_ordinals: Sequence[int] | None = None,
    ) -> SinkEffectReconcileResult:
        return cls(
            kind=SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR,
            descriptor=descriptor,
            evidence=evidence,
            accepted_ordinals=accepted_ordinals,
            diverted_ordinals=diverted_ordinals,
        )

    @classmethod
    def not_applied(cls, *, evidence: Mapping[str, object]) -> SinkEffectReconcileResult:
        return cls(kind=SinkEffectReconcileKind.NOT_APPLIED, evidence=evidence)

    @classmethod
    def unknown(cls, *, evidence: Mapping[str, object]) -> SinkEffectReconcileResult:
        return cls(kind=SinkEffectReconcileKind.UNKNOWN, evidence=evidence)


@dataclass(frozen=True, slots=True)
class RestrictedSinkEffectContext:
    run_id: str
    run_started_at: datetime
    operation_id: str
    sink_node_id: str

    def __post_init__(self) -> None:
        for field_name in ("run_id", "operation_id", "sink_node_id"):
            _require_nonempty_string(getattr(self, field_name), field_name)
        if not isinstance(self.run_started_at, datetime):
            raise TypeError("run_started_at must be datetime")


__all__ = [
    "SINK_EFFECT_PROTOCOL_VERSION",
    "AuditExportFormat",
    "AuditExportSignedManifestInput",
    "AuditExportSigningMode",
    "AuditExportSnapshotChunkInput",
    "ResolvedSinkEffectMode",
    "RestrictedAuditExportSnapshotReader",
    "RestrictedSinkEffectContext",
    "SinkEffectAttemptAction",
    "SinkEffectAttemptRequest",
    "SinkEffectAttemptResult",
    "SinkEffectAttemptState",
    "SinkEffectAuditExportSnapshotInput",
    "SinkEffectCommitResult",
    "SinkEffectDescriptorMode",
    "SinkEffectExecutionPurpose",
    "SinkEffectFinalizationMember",
    "SinkEffectFinalizationResult",
    "SinkEffectFinalizeRequest",
    "SinkEffectIdentity",
    "SinkEffectInputKind",
    "SinkEffectInspection",
    "SinkEffectInspectionMode",
    "SinkEffectInspectionRequest",
    "SinkEffectLease",
    "SinkEffectMember",
    "SinkEffectMemberCandidate",
    "SinkEffectPipelineMembersInput",
    "SinkEffectPlan",
    "SinkEffectPrepareRequest",
    "SinkEffectReconcileKind",
    "SinkEffectReconcileResult",
    "SinkEffectReservationRequest",
    "SinkEffectRole",
    "SinkEffectRuntimeBinding",
    "SinkEffectState",
]
