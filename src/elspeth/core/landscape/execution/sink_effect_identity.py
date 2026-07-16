"""Bounded canonical lineage and deterministic sink-effect identities."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from enum import Enum
from hashlib import sha256
from typing import Final, Protocol, cast

from elspeth.contracts.audit_export import (
    C,
    ClosedAuditExportJSON,
    H,
    final_manifest_identity_payload,
    hash_final_manifest_identity_payload,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import canonical_json, stable_hash
from elspeth.contracts.secret_scrub import scrub_payload_for_audit
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    AuditExportSigningMode,
    SinkEffectAuditExportSnapshotInput,
    SinkEffectIdentity,
    SinkEffectInputKind,
    SinkEffectMember,
    SinkEffectMemberCandidate,
    SinkEffectRole,
)
from elspeth.core.canonical import stable_hash as type_faithful_stable_hash

MAX_LINEAGE_DEPTH: Final = 256
MAX_LINEAGE_NODES_PER_MEMBER: Final = 4_096
MAX_LINEAGE_PARENTS: Final = 1_024
MAX_LINEAGE_EVIDENCE_BYTES: Final = 64 * 1024
_SAFE_INTEGER_MAX: Final = 9_007_199_254_740_991
_LINEAGE_LIMITS: Final = {
    "depth": MAX_LINEAGE_DEPTH,
    "evidence_bytes": MAX_LINEAGE_EVIDENCE_BYTES,
    "nodes_per_member": MAX_LINEAGE_NODES_PER_MEMBER,
    "parents": MAX_LINEAGE_PARENTS,
}

type LineageStructure = tuple[tuple[int, LineageStructure], ...]


class _Token(Protocol):
    token_id: str
    row_id: str
    run_id: str
    fork_group_id: str | None
    join_group_id: str | None
    expand_group_id: str | None


class _Row(Protocol):
    row_id: str
    run_id: str
    ingest_sequence: int


class _Parent(Protocol):
    token_id: str
    parent_token_id: str
    ordinal: int


class _LineageQuery(Protocol):
    def get_token(self, token_id: str) -> _Token | None: ...
    def get_tokens_by_ids(self, token_ids: Sequence[str]) -> list[_Token]: ...
    def get_token_parents(self, token_id: str) -> list[_Parent]: ...
    def get_row(self, row_id: str) -> _Row | None: ...


class _LineageSource(Protocol):
    query: _LineageQuery


def _audit_error(message: str) -> AuditIntegrityError:
    return AuditIntegrityError(f"Sink effect lineage audit integrity violation: {message}")


def _structure_as_json_value(structure: LineageStructure) -> list[object]:
    return [[ordinal, _structure_as_json_value(parent)] for ordinal, parent in structure]


def _closed_safe_value(value: object, path: str) -> object:
    if value is None or type(value) in (str, bool):
        return value
    if type(value) is int:
        if abs(value) > _SAFE_INTEGER_MAX:
            raise ValueError(f"{path} integer exceeds the canonical safe range")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} float must be finite")
        return value
    if isinstance(value, Enum):
        raise TypeError(f"{path} enums must be converted to exact wire strings")
    if type(value) is list or type(value) is tuple:
        return [_closed_safe_value(item, f"{path}[]") for item in cast(Sequence[object], value)]
    if type(value) is dict or isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in cast(Mapping[object, object], value).items():
            if type(key) is not str:
                raise TypeError(f"{path} requires exact string keys")
            result[key] = _closed_safe_value(item, f"{path}.{key}")
        return result
    raise TypeError(f"{path} contains unsupported identity value {type(value).__name__}")


def _credential_free_hash(tag: str, value: Mapping[str, object], field_name: str) -> str:
    closed = _closed_safe_value(value, field_name)
    if not isinstance(closed, dict):  # pragma: no cover - Mapping input guarantees this
        raise TypeError(f"{field_name} must be an object")
    if scrub_payload_for_audit(closed) != closed:
        raise ValueError(f"{field_name} must be credential-free")
    return _labeled_hash(tag, closed)


def _labeled_hash(tag: str, payload: object) -> str:
    return sha256(canonical_json({"payload": payload, "schema": tag}).encode("utf-8")).hexdigest()


class _BoundedLineageResolver:
    """Per-member resolver state; never shared across identity candidates."""

    def __init__(self, query: _LineageQuery, expected_run: str, expected_row: str) -> None:
        self._query = query
        self._expected_run = expected_run
        self._expected_row = expected_row
        self._memo: dict[str, LineageStructure] = {}
        self._visiting: set[str] = set()
        self._visited: set[str] = set()

    def resolve(self, root: _Token) -> LineageStructure:
        return self._walk(root, 0)

    def _walk(self, token: _Token, depth: int) -> LineageStructure:
        if depth > MAX_LINEAGE_DEPTH:
            raise _audit_error(f"depth exceeds {MAX_LINEAGE_DEPTH}")
        if token.token_id in self._visiting:
            raise _audit_error(f"cycle detected at token {token.token_id!r}")
        if token.token_id in self._memo:
            return self._memo[token.token_id]
        if token.run_id != self._expected_run:
            raise _audit_error(f"cross-run parent {token.token_id!r} belongs to {token.run_id!r}, expected {self._expected_run!r}")
        if token.row_id != self._expected_row:
            raise _audit_error(f"lineage token {token.token_id!r} belongs to row {token.row_id!r}, expected {self._expected_row!r}")
        row = self._query.get_row(token.row_id)
        if row is None or row.row_id != token.row_id or row.run_id != self._expected_run:
            raise _audit_error(f"lineage token {token.token_id!r} has a missing or cross-run row")
        self._visited.add(token.token_id)
        if len(self._visited) > MAX_LINEAGE_NODES_PER_MEMBER:
            raise _audit_error(f"node count exceeds {MAX_LINEAGE_NODES_PER_MEMBER}")
        self._visiting.add(token.token_id)
        parents = list(self._query.get_token_parents(token.token_id))
        if len(parents) > MAX_LINEAGE_PARENTS:
            raise _audit_error(f"fan-in exceeds {MAX_LINEAGE_PARENTS}")
        if any(parent.token_id != token.token_id for parent in parents):
            raise _audit_error(f"token {token.token_id!r} has a relation child mismatch")
        if not parents and any(value is not None for value in (token.fork_group_id, token.join_group_id, token.expand_group_id)):
            raise _audit_error(f"token {token.token_id!r} claims lineage metadata without a parent relation")
        ordinals = [parent.ordinal for parent in parents]
        if any(type(ordinal) is not int or ordinal < 0 for ordinal in ordinals):
            raise _audit_error(f"token {token.token_id!r} has a non-canonical parent ordinal")
        # Fork/expand encode the child's durable sibling position on their
        # sole relation, so a singleton ordinal need not be zero.  A
        # multi-parent join is the shape whose parent slots must be dense.
        if len(parents) > 1 and sorted(ordinals) != list(range(len(parents))):
            raise _audit_error(f"token {token.token_id!r} has duplicate or non-dense parent ordinals")
        parent_ids = [parent.parent_token_id for parent in parents]
        if len(parent_ids) != len(set(parent_ids)):
            raise _audit_error(f"token {token.token_id!r} repeats a parent token")
        hydrated = {item.token_id: item for item in self._query.get_tokens_by_ids(tuple(parent_ids))}
        if set(hydrated) != set(parent_ids):
            missing = sorted(set(parent_ids) - set(hydrated))
            raise _audit_error(f"token {token.token_id!r} references missing parents {missing}")
        entries = tuple(
            (relation.ordinal, self._walk(hydrated[relation.parent_token_id], depth + 1))
            for relation in sorted(parents, key=lambda parent: parent.ordinal)
        )
        self._visiting.remove(token.token_id)
        self._memo[token.token_id] = entries
        return entries


def resolve_sink_effect_members(
    source: _LineageSource,
    candidates: Iterable[SinkEffectMemberCandidate],
) -> tuple[SinkEffectMember, ...]:
    """Resolve, validate, bound, and densely order exact sink-boundary rows."""
    query = source.query
    candidate_tuple = tuple(candidates)
    if not candidate_tuple:
        raise ValueError("sink effect members must be non-empty")
    if any(type(candidate) is not SinkEffectMemberCandidate for candidate in candidate_tuple):
        raise TypeError("candidates must contain exact SinkEffectMemberCandidate values")
    token_ids = tuple(candidate.token_id for candidate in candidate_tuple)
    if len(set(token_ids)) != len(token_ids):
        raise _audit_error("candidate token IDs must be unique")

    ordered_inputs: list[tuple[int, LineageStructure, str, str, str, str, Mapping[str, object], str]] = []
    for candidate in candidate_tuple:
        root = query.get_token(candidate.token_id)
        if root is None:
            raise _audit_error(f"member token {candidate.token_id!r} is missing")
        expected_run = root.run_id
        structure = _BoundedLineageResolver(query, expected_run, root.row_id).resolve(root)
        row = query.get_row(root.row_id)
        if row is None:
            raise _audit_error(f"member token {root.token_id!r} references missing row {root.row_id!r}")
        if row.run_id != expected_run or row.row_id != root.row_id:
            raise _audit_error(f"member token {root.token_id!r} has a cross-run or divergent row link")
        if type(row.ingest_sequence) is not int or row.ingest_sequence < 0:
            raise _audit_error(f"row {row.row_id!r} has invalid ingest_sequence")
        lineage_json = canonical_json(_structure_as_json_value(structure))
        if len(lineage_json.encode("utf-8")) > MAX_LINEAGE_EVIDENCE_BYTES:
            raise _audit_error(f"lineage evidence exceeds {MAX_LINEAGE_EVIDENCE_BYTES} bytes")
        try:
            payload_hash = type_faithful_stable_hash(candidate.row)
            pending_identity_hash = stable_hash(candidate.pending_identity)
        except (TypeError, ValueError) as exc:
            raise _audit_error(f"member {root.token_id!r} has non-canonical sink-boundary input") from exc
        ordered_inputs.append(
            (
                row.ingest_sequence,
                structure,
                root.token_id,
                root.row_id,
                lineage_json,
                payload_hash,
                candidate.row,
                pending_identity_hash,
            )
        )

    ordered_inputs.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(
        SinkEffectMember(
            ordinal=ordinal,
            token_id=item[2],
            row_id=item[3],
            ingest_sequence=item[0],
            lineage_json=item[4],
            lineage_hash=sha256(item[4].encode("utf-8")).hexdigest(),
            payload_hash=item[5],
            row=item[6],
            pending_identity_hash=item[7],
        )
        for ordinal, item in enumerate(ordered_inputs)
    )


def _derived_ids(effect_id: str, member_count: int) -> tuple[str, str, tuple[str, ...]]:
    artifact_id = _labeled_hash("sink-effect-artifact-v1", {"effect_id": effect_id})
    idempotency_key = _labeled_hash("sink-effect-artifact-idempotency-v1", {"effect_id": effect_id})
    member_ids = tuple(
        _labeled_hash("sink-effect-member-v1", {"effect_id": effect_id, "ordinal": ordinal}) for ordinal in range(member_count)
    )
    return artifact_id, idempotency_key, member_ids


def compute_pipeline_effect_identity(
    *,
    run_id: str,
    sink_node_id: str,
    role: SinkEffectRole,
    sink_config: Mapping[str, object],
    target_config: Mapping[str, object],
    members: Sequence[SinkEffectMember],
) -> SinkEffectIdentity:
    """Compute the complete pipeline-member identity without attempt state."""
    if type(role) is not SinkEffectRole:
        raise TypeError("role must be exact SinkEffectRole")
    if not run_id or not sink_node_id:
        raise ValueError("run_id and sink_node_id must be non-empty")
    member_tuple = tuple(members)
    if not member_tuple or any(type(member) is not SinkEffectMember for member in member_tuple):
        raise ValueError("pipeline identity requires exact non-empty members")
    if [member.ordinal for member in member_tuple] != list(range(len(member_tuple))):
        raise ValueError("pipeline members must be dense and ordered")
    config_hash = _credential_free_hash("sink-effect-config-v1", sink_config, "sink_config")
    requested_target_hash = _credential_free_hash("sink-effect-target-v1", target_config, "target_config")
    membership = [
        {
            "ingest_sequence": member.ingest_sequence,
            "lineage_hash": member.lineage_hash,
            "ordinal": member.ordinal,
            "payload_hash": member.payload_hash,
            "pending_identity_hash": member.pending_identity_hash,
            "row_id": member.row_id,
            "token_id": member.token_id,
        }
        for member in member_tuple
    ]
    membership_hash = _labeled_hash("sink-effect-membership-v1", membership)
    group_payload_hash = _labeled_hash("sink-effect-group-payload-v1", [member.payload_hash for member in member_tuple])
    effect_id = _labeled_hash(
        "sink-effect-pipeline-v1",
        {
            "config_hash": config_hash,
            "input_kind": SinkEffectInputKind.PIPELINE_MEMBERS.value,
            "limits": _LINEAGE_LIMITS,
            "membership": membership,
            "protocol_version": SINK_EFFECT_PROTOCOL_VERSION,
            "requested_target_hash": requested_target_hash,
            "role": role.value,
            "run_id": run_id,
            "sink_node_id": sink_node_id,
        },
    )
    stream_id = _labeled_hash(
        "sink-effect-stream-v1",
        {
            "requested_target_hash": requested_target_hash,
            "role": role.value,
            "run_id": run_id,
            "sink_node_id": sink_node_id,
        },
    )
    artifact_id, idempotency_key, member_ids = _derived_ids(effect_id, len(member_tuple))
    bound_members = tuple(replace(member, member_effect_id=member_ids[member.ordinal]) for member in member_tuple)
    return SinkEffectIdentity(
        effect_id=effect_id,
        artifact_id=artifact_id,
        artifact_idempotency_key=idempotency_key,
        stream_id=stream_id,
        config_hash=config_hash,
        requested_target_hash=requested_target_hash,
        membership_or_manifest_hash=membership_hash,
        group_payload_hash=group_payload_hash,
        input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        members=bound_members,
        member_ids=member_ids,
    )


compute_effect_identity = compute_pipeline_effect_identity


def compute_audit_export_effect_identity(
    snapshot: SinkEffectAuditExportSnapshotInput,
    target_config: Mapping[str, object],
    *,
    sink_node_id: str,
    role: SinkEffectRole,
) -> SinkEffectIdentity:
    """Compute the zero-member audit-export effect identity."""
    if type(snapshot) is not SinkEffectAuditExportSnapshotInput:
        raise TypeError("snapshot must be exact SinkEffectAuditExportSnapshotInput")
    if type(role) is not SinkEffectRole:
        raise TypeError("role must be exact SinkEffectRole")
    descriptor = snapshot.signed_manifest
    if descriptor.signature_algorithm is not snapshot.signing_mode:
        raise ValueError("signed manifest signature algorithm does not match snapshot signing mode")
    if descriptor.signature_key_id != snapshot.signer_key_id:
        raise ValueError("signed manifest signer does not match snapshot signer")
    expected_chain = (
        "sha256_concat_record_sha256_v1"
        if snapshot.signing_mode is AuditExportSigningMode.UNSIGNED
        else "sha256_concat_hmac_sha256_signatures_v1"
    )
    if descriptor.record_chain_algorithm != expected_chain:
        raise ValueError("signed manifest record-chain algorithm does not match snapshot signing mode")
    target_config_hash = _credential_free_hash("sink-effect-target-v1", target_config, "target_config")
    manifest_component = final_manifest_identity_payload(descriptor)
    final_manifest_hash = hash_final_manifest_identity_payload(manifest_component)
    effect_payload: dict[str, ClosedAuditExportJSON] = {
        "export_format": snapshot.export_format.value,
        "final_manifest_identity_hash": final_manifest_hash,
        "input_kind": SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT.value,
        "manifest_hash": snapshot.manifest_hash,
        "protocol_version": SINK_EFFECT_PROTOCOL_VERSION,
        "registry_key_hash": snapshot.registry_key_hash,
        "role": role.value,
        "serialization_version": snapshot.serialization_version,
        "signer_key_id": snapshot.signer_key_id,
        "signing_mode": snapshot.signing_mode.value,
        "sink_node_id": sink_node_id,
        "snapshot_hash": snapshot.snapshot_hash,
        "snapshot_id": snapshot.snapshot_id,
        "source_run_id": snapshot.source_run_id,
        "target_config_hash": target_config_hash,
    }
    effect_id = H(C("sink-effect-audit-export-effect-v1", effect_payload))
    stream_id = _labeled_hash(
        "sink-effect-stream-v1",
        {
            "requested_target_hash": target_config_hash,
            "role": role.value,
            "run_id": snapshot.source_run_id,
            "sink_node_id": sink_node_id,
        },
    )
    artifact_id, idempotency_key, member_ids = _derived_ids(effect_id, 0)
    return SinkEffectIdentity(
        effect_id=effect_id,
        artifact_id=artifact_id,
        artifact_idempotency_key=idempotency_key,
        stream_id=stream_id,
        config_hash=target_config_hash,
        requested_target_hash=target_config_hash,
        membership_or_manifest_hash=snapshot.manifest_hash,
        group_payload_hash=snapshot.snapshot_hash,
        input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT,
        members=(),
        member_ids=member_ids,
        snapshot_hash=snapshot.snapshot_hash,
        final_manifest_identity_hash=final_manifest_hash,
    )


__all__ = [
    "MAX_LINEAGE_DEPTH",
    "MAX_LINEAGE_EVIDENCE_BYTES",
    "MAX_LINEAGE_NODES_PER_MEMBER",
    "MAX_LINEAGE_PARENTS",
    "compute_audit_export_effect_identity",
    "compute_effect_identity",
    "compute_pipeline_effect_identity",
    "resolve_sink_effect_members",
]
