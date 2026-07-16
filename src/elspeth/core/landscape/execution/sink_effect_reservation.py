"""Atomic reservation of durable sink effects and replacing-target streams."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, Final

from sqlalchemy import Row, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection

from elspeth.contracts.audit import SinkEffect
from elspeth.contracts.audit_export import C, H, final_manifest_identity_payload, hash_final_manifest_identity_payload
from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    AuditExportSignedManifestInput,
    AuditExportSigningMode,
    SinkEffectInputKind,
    SinkEffectMember,
    SinkEffectRole,
    SinkEffectState,
)
from elspeth.core.landscape._helpers import now
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.model_loaders import SinkEffectLoader
from elspeth.core.landscape.schema import (
    audit_export_snapshots_table,
    node_states_table,
    operations_table,
    rows_table,
    sink_effect_export_snapshots_table,
    sink_effect_members_table,
    sink_effect_streams_table,
    sink_effects_table,
    tokens_table,
)

_LOWER_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")
_EMPTY_TARGET_JSON: Final = "{}"


def _require_nonempty(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_hash(value: object, field_name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or _LOWER_HEX_64.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


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
        _require_nonempty(self.run_id, "run_id")
        _require_nonempty(self.sink_node_id, "sink_node_id")
        if type(self.role) is not SinkEffectRole:
            raise TypeError("role must be exact SinkEffectRole")
        if type(self.input_kind) is not SinkEffectInputKind:
            raise TypeError("input_kind must be exact SinkEffectInputKind")
        _require_hash(self.requested_target_hash, "requested_target_hash")
        _require_hash(self.config_hash, "config_hash")
        _require_hash(self.primary_effect_id, "primary_effect_id", optional=True)
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
            _require_hash(self.audit_export_snapshot_id, "audit_export_snapshot_id")
            if self.config_hash != self.requested_target_hash:
                raise ValueError("audit export config_hash must equal requested_target_hash")


@dataclass(frozen=True, slots=True)
class SinkEffectReservationResult:
    """Partition of requested work after one atomic reservation attempt."""

    finalized_effect_ids: Sequence[str]
    open_effect_ids: Sequence[str]
    new_effect: SinkEffect | None

    def __post_init__(self) -> None:
        finalized = tuple(self.finalized_effect_ids)
        opened = tuple(self.open_effect_ids)
        if any(_LOWER_HEX_64.fullmatch(effect_id) is None for effect_id in (*finalized, *opened)):
            raise ValueError("reservation result effect IDs must be lowercase SHA-256 digests")
        if set(finalized) & set(opened):
            raise ValueError("reservation result cannot classify one effect as both finalized and open")
        object.__setattr__(self, "finalized_effect_ids", finalized)
        object.__setattr__(self, "open_effect_ids", opened)


@dataclass(frozen=True, slots=True)
class _EffectIdentity:
    effect_id: str
    artifact_id: str
    artifact_idempotency_key: str
    stream_id: str
    membership_or_manifest_hash: str
    group_payload_hash: str
    members: tuple[SinkEffectMember, ...]


@dataclass(frozen=True, slots=True)
class _PipelineWitness:
    state_ids_by_token: Mapping[str, str]


def _labeled_hash(tag: str, payload: object) -> str:
    return sha256(canonical_json({"payload": payload, "schema": tag}).encode("utf-8")).hexdigest()


def _derived_ids(effect_id: str, member_count: int) -> tuple[str, str, tuple[str, ...]]:
    artifact_id = _labeled_hash("sink-effect-artifact-v1", {"effect_id": effect_id})
    idempotency_key = _labeled_hash("sink-effect-artifact-idempotency-v1", {"effect_id": effect_id})
    member_ids = tuple(
        _labeled_hash("sink-effect-member-v1", {"effect_id": effect_id, "ordinal": ordinal}) for ordinal in range(member_count)
    )
    return artifact_id, idempotency_key, member_ids


def _pipeline_identity(request: SinkEffectReservationRequest, members: Sequence[SinkEffectMember]) -> _EffectIdentity:
    dense = tuple(replace(member, ordinal=ordinal, member_effect_id=None) for ordinal, member in enumerate(members))
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
        for member in dense
    ]
    limits = {"depth": 256, "evidence_bytes": 64 * 1024, "nodes_per_member": 4_096, "parents": 1_024}
    membership_hash = _labeled_hash("sink-effect-membership-v1", membership)
    group_payload_hash = _labeled_hash("sink-effect-group-payload-v1", [member.payload_hash for member in dense])
    effect_id = _labeled_hash(
        "sink-effect-pipeline-v1",
        {
            "config_hash": request.config_hash,
            "input_kind": request.input_kind.value,
            "limits": limits,
            "membership": membership,
            "protocol_version": SINK_EFFECT_PROTOCOL_VERSION,
            "requested_target_hash": request.requested_target_hash,
            "role": request.role.value,
            "run_id": request.run_id,
            "sink_node_id": request.sink_node_id,
        },
    )
    stream_id = _labeled_hash(
        "sink-effect-stream-v1",
        {
            "requested_target_hash": request.requested_target_hash,
            "role": request.role.value,
            "run_id": request.run_id,
            "sink_node_id": request.sink_node_id,
        },
    )
    artifact_id, idempotency_key, member_ids = _derived_ids(effect_id, len(dense))
    bound = tuple(replace(member, member_effect_id=member_ids[member.ordinal]) for member in dense)
    return _EffectIdentity(
        effect_id=effect_id,
        artifact_id=artifact_id,
        artifact_idempotency_key=idempotency_key,
        stream_id=stream_id,
        membership_or_manifest_hash=membership_hash,
        group_payload_hash=group_payload_hash,
        members=bound,
    )


def _export_identity(request: SinkEffectReservationRequest, snapshot: Row[Any]) -> _EffectIdentity:
    descriptor = AuditExportSignedManifestInput(
        content_ref=snapshot.signed_manifest_ref,
        content_hash=snapshot.signed_manifest_hash,
        size_bytes=snapshot.signed_manifest_size_bytes,
        manifest_schema=snapshot.signed_manifest_schema,
        derivation_version=snapshot.derivation_version,
        signature_algorithm=AuditExportSigningMode(snapshot.signing_mode),
        signature_key_id=snapshot.signer_key_id,
        record_chain_algorithm=snapshot.record_chain_algorithm,
        final_hash=snapshot.final_hash,
        signature=snapshot.signature_hex,
    )
    final_manifest_hash = hash_final_manifest_identity_payload(final_manifest_identity_payload(descriptor))
    effect_payload = {
        "export_format": snapshot.export_format,
        "final_manifest_identity_hash": final_manifest_hash,
        "input_kind": request.input_kind.value,
        "manifest_hash": snapshot.manifest_hash,
        "protocol_version": SINK_EFFECT_PROTOCOL_VERSION,
        "registry_key_hash": snapshot.registry_key_hash,
        "role": request.role.value,
        "serialization_version": snapshot.serialization_version,
        "signer_key_id": snapshot.signer_key_id,
        "signing_mode": snapshot.signing_mode,
        "sink_node_id": request.sink_node_id,
        "snapshot_hash": snapshot.snapshot_hash,
        "snapshot_id": snapshot.snapshot_id,
        "source_run_id": snapshot.source_run_id,
        "target_config_hash": request.config_hash,
    }
    effect_id = H(C("sink-effect-audit-export-effect-v1", effect_payload))
    stream_id = _labeled_hash(
        "sink-effect-stream-v1",
        {
            "requested_target_hash": request.requested_target_hash,
            "role": request.role.value,
            "run_id": request.run_id,
            "sink_node_id": request.sink_node_id,
        },
    )
    artifact_id, idempotency_key, _member_ids = _derived_ids(effect_id, 0)
    return _EffectIdentity(
        effect_id=effect_id,
        artifact_id=artifact_id,
        artifact_idempotency_key=idempotency_key,
        stream_id=stream_id,
        membership_or_manifest_hash=snapshot.manifest_hash,
        group_payload_hash=snapshot.snapshot_hash,
        members=(),
    )


def _conflict_safe_insert(conn: Connection, table: Any, values: Mapping[str, object], *, index_elements: Sequence[str]) -> bool:
    if conn.dialect.name == "sqlite":
        statement = (
            sqlite_insert(table)
            .values(**values)
            .on_conflict_do_nothing(index_elements=list(index_elements))
            .returning(*table.primary_key.columns)
        )
        return conn.execute(statement).fetchone() is not None
    if conn.dialect.name == "postgresql":
        statement = (
            postgresql_insert(table)
            .values(**values)
            .on_conflict_do_nothing(index_elements=list(index_elements))
            .returning(*table.primary_key.columns)
        )
        return conn.execute(statement).fetchone() is not None
    raise RuntimeError(f"unsupported Landscape backend {conn.dialect.name!r}")  # pragma: no cover


class SinkEffectReservation:
    """Token-first transaction coordinator for sink-effect reservation."""

    def __init__(self, db: LandscapeDB, *, effect_loader: SinkEffectLoader) -> None:
        self._db = db
        self._effect_loader = effect_loader

    def reserve(self, request: SinkEffectReservationRequest) -> SinkEffectReservationResult:
        if type(request) is not SinkEffectReservationRequest:
            raise TypeError("request must be exact SinkEffectReservationRequest")
        if request.input_kind is SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT:
            return self._reserve_export(request)

        witness = self._resolve_pipeline_witness(request)
        with self._db.write_connection() as conn:
            self._lock_and_validate_pipeline_witness(conn, request, witness)
            return self._reserve_pipeline_locked(conn, request)

    def _resolve_pipeline_witness(self, request: SinkEffectReservationRequest) -> _PipelineWitness:
        token_ids = tuple(member.token_id for member in request.members)
        with self._db.read_only_connection() as conn:
            token_rows = conn.execute(
                select(
                    tokens_table.c.token_id,
                    tokens_table.c.row_id,
                    tokens_table.c.run_id,
                    rows_table.c.ingest_sequence,
                )
                .join(rows_table, rows_table.c.row_id == tokens_table.c.row_id)
                .where(tokens_table.c.token_id.in_(token_ids))
            ).fetchall()
            self._validate_token_rows(request, token_rows)
            state_ids = self._current_state_ids(conn, request, token_ids)
        return _PipelineWitness(state_ids_by_token=state_ids)

    def _lock_and_validate_pipeline_witness(
        self,
        conn: Connection,
        request: SinkEffectReservationRequest,
        witness: _PipelineWitness,
    ) -> None:
        token_ids = tuple(sorted(member.token_id for member in request.members))
        locked_tokens: list[Row[Any]] = []
        for token_id in token_ids:
            row = conn.execute(select(tokens_table).where(tokens_table.c.token_id == token_id).with_for_update()).fetchone()
            if row is None:
                raise ValueError(f"sink effect member token {token_id!r} disappeared")
            locked_tokens.append(row)
            self._after_token_lock(self._backend_pid(conn), tuple(item.token_id for item in locked_tokens))

        token_rows = conn.execute(
            select(
                tokens_table.c.token_id,
                tokens_table.c.row_id,
                tokens_table.c.run_id,
                rows_table.c.ingest_sequence,
            )
            .join(rows_table, rows_table.c.row_id == tokens_table.c.row_id)
            .where(tokens_table.c.token_id.in_(token_ids))
        ).fetchall()
        self._validate_token_rows(request, token_rows)

        state_ids = tuple(sorted(set(witness.state_ids_by_token.values())))
        locked_states: list[Row[Any]] = []
        for state_id in state_ids:
            row = conn.execute(select(node_states_table).where(node_states_table.c.state_id == state_id).with_for_update()).fetchone()
            if row is None:
                raise ValueError(f"sink effect current state {state_id!r} disappeared")
            locked_states.append(row)
            self._after_state_lock(self._backend_pid(conn), tuple(item.state_id for item in locked_states))

        current = self._current_state_ids(conn, request, token_ids)
        if current != witness.state_ids_by_token:
            raise ValueError("sink effect current-state witness changed during reservation")
        member_by_token = {member.token_id: member for member in request.members}
        for row in locked_states:
            member = member_by_token.get(row.token_id)
            if (
                member is None
                or row.run_id != request.run_id
                or row.node_id != request.sink_node_id
                or row.input_hash != member.payload_hash
            ):
                raise ValueError("sink effect current-state witness is divergent")
        self._after_witness_locks(self._backend_pid(conn), token_ids, state_ids)

    def _current_state_ids(
        self,
        conn: Connection,
        request: SinkEffectReservationRequest,
        token_ids: Sequence[str],
    ) -> dict[str, str]:
        rows = conn.execute(
            select(
                node_states_table.c.state_id,
                node_states_table.c.token_id,
                node_states_table.c.input_hash,
            )
            .where(
                node_states_table.c.run_id == request.run_id,
                node_states_table.c.node_id == request.sink_node_id,
                node_states_table.c.token_id.in_(tuple(token_ids)),
            )
            .order_by(
                node_states_table.c.token_id,
                node_states_table.c.attempt.desc(),
                node_states_table.c.started_at.desc(),
                node_states_table.c.state_id.desc(),
            )
        ).fetchall()
        current: dict[str, str] = {}
        member_by_token = {member.token_id: member for member in request.members}
        for row in rows:
            if row.token_id in current:
                continue
            member = member_by_token[row.token_id]
            if row.input_hash != member.payload_hash:
                raise ValueError(f"sink effect member {row.token_id!r} payload is divergent from its current state")
            current[row.token_id] = row.state_id
        missing = sorted(set(token_ids) - set(current))
        if missing:
            raise ValueError(f"sink effect members have no current sink-node state: {missing!r}")
        return current

    @staticmethod
    def _validate_token_rows(request: SinkEffectReservationRequest, rows: Sequence[Row[Any]]) -> None:
        by_token = {row.token_id: row for row in rows}
        missing = sorted({member.token_id for member in request.members} - set(by_token))
        if missing:
            raise ValueError(f"sink effect member tokens are missing: {missing!r}")
        for member in request.members:
            row = by_token[member.token_id]
            if row.run_id != request.run_id or row.row_id != member.row_id or row.ingest_sequence != member.ingest_sequence:
                raise ValueError(f"sink effect member {member.token_id!r} row/run identity is divergent")

    def _reserve_pipeline_locked(
        self,
        conn: Connection,
        request: SinkEffectReservationRequest,
    ) -> SinkEffectReservationResult:
        token_ids = tuple(member.token_id for member in request.members)
        bindings = self._member_bindings(conn, request, token_ids)
        unbound = tuple(member for member in request.members if member.token_id not in bindings)
        existing_effect_ids = tuple(sorted({row.effect_id for row in bindings.values()}))

        expected_stream_id = _pipeline_identity(request, request.members[:1]).stream_id
        optimistic_effects = self._effect_rows(conn, existing_effect_ids, lock=False)
        self._validate_existing_stream_shape(request, optimistic_effects, expected_stream_id)
        stream = self._lock_stream(
            conn,
            request,
            expected_stream_id,
            create=bool(unbound) and request.replacing_target,
        )
        effects = self._effect_rows(conn, existing_effect_ids, lock=True)
        self._validate_existing_stream_shape(request, effects, expected_stream_id)

        # Re-read after the complete mutable lock set and exact-compare every
        # durable member field. Membership is never rewritten to fit rebatching.
        bindings = self._member_bindings(conn, request, token_ids)
        self._validate_existing_members(request, bindings)
        unbound = tuple(member for member in request.members if member.token_id not in bindings)
        finalized, opened = self._partition_effects(effects)
        if not unbound:
            return SinkEffectReservationResult(finalized, opened, None)

        identity = _pipeline_identity(request, unbound)
        sequence: int | None = None
        predecessor: str | None = None
        if request.replacing_target:
            if stream is None:
                raise ValueError("replacing sink effect stream disappeared")
            sequence = stream.next_sequence
            predecessor = stream.tail_effect_id

        inserted, effect = self._insert_or_compare_effect(
            conn,
            request,
            identity,
            stream_id=expected_stream_id if request.replacing_target else None,
            stream_sequence=sequence,
            predecessor_effect_id=predecessor,
        )
        if not inserted:
            # Token locks make this reachable only after an independently
            # committed exact winner. Treat it as existing open/final work.
            category = (effect.effect_id,)
            if effect.state is SinkEffectState.FINALIZED:
                return SinkEffectReservationResult(tuple(sorted(set(finalized) | set(category))), opened, None)
            return SinkEffectReservationResult(finalized, tuple(sorted(set(opened) | set(category))), None)

        self._insert_members(conn, request, identity)
        self._insert_or_compare_operation(conn, request, effect)
        if stream is not None:
            assert sequence is not None
            result = conn.execute(
                sink_effect_streams_table.update()
                .where(
                    sink_effect_streams_table.c.stream_id == stream.stream_id,
                    sink_effect_streams_table.c.next_sequence == sequence,
                    sink_effect_streams_table.c.tail_effect_id.is_(predecessor)
                    if predecessor is None
                    else sink_effect_streams_table.c.tail_effect_id == predecessor,
                )
                .values(next_sequence=sequence + 1, tail_effect_id=effect.effect_id)
            )
            if result.rowcount != 1:
                raise ValueError("sink effect stream tail changed during reservation")
        return SinkEffectReservationResult(finalized, opened, effect)

    def _reserve_export(self, request: SinkEffectReservationRequest) -> SinkEffectReservationResult:
        assert request.audit_export_snapshot_id is not None
        with self._db.read_only_connection() as conn:
            optimistic = conn.execute(
                select(audit_export_snapshots_table).where(audit_export_snapshots_table.c.snapshot_id == request.audit_export_snapshot_id)
            ).fetchone()
        if optimistic is None:
            raise ValueError("audit export snapshot does not exist")
        self._validate_export_snapshot(request, optimistic)
        optimistic_values = dict(optimistic._mapping)

        with self._db.write_connection() as conn:
            snapshot = conn.execute(
                select(audit_export_snapshots_table).where(audit_export_snapshots_table.c.snapshot_id == request.audit_export_snapshot_id)
            ).fetchone()
            if snapshot is None or dict(snapshot._mapping) != optimistic_values:
                raise ValueError("audit export snapshot registry winner is divergent")
            self._validate_export_snapshot(request, snapshot)
            identity = _export_identity(request, snapshot)
            stream = self._lock_stream(
                conn,
                request,
                identity.stream_id,
                create=request.replacing_target,
            )
            sequence = stream.next_sequence if stream is not None else None
            predecessor = stream.tail_effect_id if stream is not None else None
            inserted, effect = self._insert_or_compare_effect(
                conn,
                request,
                identity,
                stream_id=identity.stream_id if request.replacing_target else None,
                stream_sequence=sequence,
                predecessor_effect_id=predecessor,
            )
            self._insert_or_compare_export_association(conn, effect.effect_id, request.audit_export_snapshot_id)
            self._insert_or_compare_operation(conn, request, effect)
            if inserted and stream is not None:
                assert sequence is not None
                result = conn.execute(
                    sink_effect_streams_table.update()
                    .where(
                        sink_effect_streams_table.c.stream_id == stream.stream_id,
                        sink_effect_streams_table.c.next_sequence == sequence,
                        sink_effect_streams_table.c.tail_effect_id.is_(predecessor)
                        if predecessor is None
                        else sink_effect_streams_table.c.tail_effect_id == predecessor,
                    )
                    .values(next_sequence=sequence + 1, tail_effect_id=effect.effect_id)
                )
                if result.rowcount != 1:
                    raise ValueError("audit export stream tail changed during reservation")
            if inserted:
                return SinkEffectReservationResult((), (), effect)
            if effect.state is SinkEffectState.FINALIZED:
                return SinkEffectReservationResult((effect.effect_id,), (), None)
            return SinkEffectReservationResult((), (effect.effect_id,), None)

    @staticmethod
    def _validate_export_snapshot(request: SinkEffectReservationRequest, snapshot: Row[Any]) -> None:
        if snapshot.source_run_id != request.run_id:
            raise ValueError("audit export snapshot belongs to a different run")
        if snapshot.snapshot_id != request.audit_export_snapshot_id:
            raise ValueError("audit export snapshot registry winner identity is divergent")
        # Constructing the descriptor validates the immutable manifest/signing
        # cross-map before any stream/effect mutation.
        _export_identity(request, snapshot)

    def _lock_stream(
        self,
        conn: Connection,
        request: SinkEffectReservationRequest,
        stream_id: str,
        *,
        create: bool,
    ) -> Row[Any] | None:
        if not request.replacing_target:
            return None
        if create:
            _conflict_safe_insert(
                conn,
                sink_effect_streams_table,
                {
                    "stream_id": stream_id,
                    "run_id": request.run_id,
                    "sink_node_id": request.sink_node_id,
                    "role": request.role.value,
                    "requested_target_hash": request.requested_target_hash,
                    "resolved_target": None,
                    "next_sequence": 0,
                    "tail_effect_id": None,
                    "head_effect_id": None,
                    "head_descriptor_hash": None,
                },
                index_elements=("run_id", "sink_node_id", "role", "requested_target_hash"),
            )
        row = conn.execute(
            select(sink_effect_streams_table)
            .where(
                sink_effect_streams_table.c.run_id == request.run_id,
                sink_effect_streams_table.c.sink_node_id == request.sink_node_id,
                sink_effect_streams_table.c.role == request.role.value,
                sink_effect_streams_table.c.requested_target_hash == request.requested_target_hash,
            )
            .with_for_update()
        ).fetchone()
        if row is None:
            raise ValueError("replacing sink effect stream is missing")
        if row.stream_id != stream_id:
            raise ValueError("replacing sink effect stream winner is divergent")
        return row

    @staticmethod
    def _member_bindings(
        conn: Connection,
        request: SinkEffectReservationRequest,
        token_ids: Sequence[str],
    ) -> dict[str, Row[Any]]:
        rows = conn.execute(
            select(sink_effect_members_table).where(
                sink_effect_members_table.c.run_id == request.run_id,
                sink_effect_members_table.c.sink_node_id == request.sink_node_id,
                sink_effect_members_table.c.role == request.role.value,
                sink_effect_members_table.c.token_id.in_(tuple(token_ids)),
            )
        ).fetchall()
        return {row.token_id: row for row in rows}

    @staticmethod
    def _validate_existing_members(request: SinkEffectReservationRequest, bindings: Mapping[str, Row[Any]]) -> None:
        for member in request.members:
            row = bindings.get(member.token_id)
            if row is None:
                continue
            expected = {
                "run_id": request.run_id,
                "sink_node_id": request.sink_node_id,
                "role": request.role.value,
                "input_kind": SinkEffectInputKind.PIPELINE_MEMBERS.value,
                "token_id": member.token_id,
                "row_id": member.row_id,
                "ingest_sequence": member.ingest_sequence,
                "lineage_json": member.lineage_json,
                "lineage_hash": member.lineage_hash,
                "payload_hash": member.payload_hash,
            }
            if any(getattr(row, field_name) != value for field_name, value in expected.items()):
                raise ValueError(f"sink effect member {member.token_id!r} has divergent immutable membership")

    @staticmethod
    def _effect_rows(conn: Connection, effect_ids: Sequence[str], *, lock: bool) -> tuple[Row[Any], ...]:
        if not effect_ids:
            return ()
        statement = (
            select(sink_effects_table).where(sink_effects_table.c.effect_id.in_(tuple(effect_ids))).order_by(sink_effects_table.c.effect_id)
        )
        if lock:
            statement = statement.with_for_update()
        rows = tuple(conn.execute(statement).fetchall())
        if len(rows) != len(effect_ids):
            raise ValueError("sink effect membership references a missing effect")
        return rows

    @staticmethod
    def _validate_existing_stream_shape(
        request: SinkEffectReservationRequest,
        effects: Sequence[Row[Any]],
        expected_stream_id: str,
    ) -> None:
        for effect in effects:
            if request.replacing_target:
                if effect.stream_id != expected_stream_id:
                    raise ValueError("existing sink effect has divergent replacing-target stream membership")
            elif effect.stream_id is not None:
                raise ValueError("existing sink effect is stream-bound but request is not replacing")

    @staticmethod
    def _partition_effects(effects: Sequence[Row[Any]]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        finalized = tuple(sorted(row.effect_id for row in effects if row.state == SinkEffectState.FINALIZED.value))
        opened = tuple(sorted(row.effect_id for row in effects if row.state != SinkEffectState.FINALIZED.value))
        return finalized, opened

    def _insert_or_compare_effect(
        self,
        conn: Connection,
        request: SinkEffectReservationRequest,
        identity: _EffectIdentity,
        *,
        stream_id: str | None,
        stream_sequence: int | None,
        predecessor_effect_id: str | None,
    ) -> tuple[bool, SinkEffect]:
        timestamp = now()
        values: dict[str, object] = {
            "effect_id": identity.effect_id,
            "run_id": request.run_id,
            "sink_node_id": request.sink_node_id,
            "role": request.role.value,
            "state": SinkEffectState.RESERVED.value,
            "protocol_version": SINK_EFFECT_PROTOCOL_VERSION,
            "input_kind": request.input_kind.value,
            "required_member_ordinal": 0 if request.input_kind is SinkEffectInputKind.PIPELINE_MEMBERS else None,
            "required_snapshot_slot": 0 if request.input_kind is SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT else None,
            "config_hash": request.config_hash,
            "membership_or_manifest_hash": identity.membership_or_manifest_hash,
            "group_payload_hash": identity.group_payload_hash,
            "artifact_id": identity.artifact_id,
            "artifact_idempotency_key": identity.artifact_idempotency_key,
            "target_json": _EMPTY_TARGET_JSON,
            "inspection_mode": None,
            "inspection_attempt_id": None,
            "plan_json": None,
            "plan_hash": None,
            "descriptor_mode": None,
            "expected_descriptor_hash": None,
            "precondition_hash": None,
            "prepared_at": None,
            "lease_owner": None,
            "generation": 0,
            "lease_expires_at": None,
            "lease_heartbeat_at": None,
            "reconcile_kind": None,
            "reconcile_evidence_hash": None,
            "result_descriptor_hash": None,
            "publication_performed": None,
            "publication_evidence_kind": None,
            "primary_effect_id": request.primary_effect_id,
            "stream_id": stream_id,
            "stream_sequence": stream_sequence,
            "predecessor_effect_id": predecessor_effect_id,
            "created_at": timestamp,
            "updated_at": timestamp,
            "finalized_at": None,
        }
        inserted = _conflict_safe_insert(conn, sink_effects_table, values, index_elements=("effect_id",))
        row = conn.execute(
            select(sink_effects_table).where(sink_effects_table.c.effect_id == identity.effect_id).with_for_update()
        ).fetchone()
        if row is None:
            raise ValueError("sink effect winner disappeared")
        immutable = {
            "run_id": request.run_id,
            "sink_node_id": request.sink_node_id,
            "role": request.role.value,
            "protocol_version": SINK_EFFECT_PROTOCOL_VERSION,
            "input_kind": request.input_kind.value,
            "required_member_ordinal": values["required_member_ordinal"],
            "required_snapshot_slot": values["required_snapshot_slot"],
            "config_hash": request.config_hash,
            "membership_or_manifest_hash": identity.membership_or_manifest_hash,
            "group_payload_hash": identity.group_payload_hash,
            "artifact_id": identity.artifact_id,
            "artifact_idempotency_key": identity.artifact_idempotency_key,
            "primary_effect_id": request.primary_effect_id,
            "stream_id": stream_id,
            "stream_sequence": stream_sequence,
            "predecessor_effect_id": predecessor_effect_id,
        }
        if any(getattr(row, field_name) != value for field_name, value in immutable.items()):
            raise ValueError("sink effect identity winner is divergent")
        if inserted and row.target_json != _EMPTY_TARGET_JSON:
            raise ValueError("new sink effect did not preserve its empty target sentinel")
        return inserted, self._effect_loader.load(row)

    @staticmethod
    def _insert_members(conn: Connection, request: SinkEffectReservationRequest, identity: _EffectIdentity) -> None:
        for member in identity.members:
            inserted = _conflict_safe_insert(
                conn,
                sink_effect_members_table,
                {
                    "effect_id": identity.effect_id,
                    "input_kind": request.input_kind.value,
                    "ordinal": member.ordinal,
                    "run_id": request.run_id,
                    "sink_node_id": request.sink_node_id,
                    "role": request.role.value,
                    "token_id": member.token_id,
                    "row_id": member.row_id,
                    "ingest_sequence": member.ingest_sequence,
                    "lineage_json": member.lineage_json,
                    "lineage_hash": member.lineage_hash,
                    "payload_hash": member.payload_hash,
                    "prepared_disposition": None,
                    "reason_hash": None,
                    "member_effect_id": member.member_effect_id,
                    "member_state": None,
                    "descriptor_hash": None,
                    "evidence_hash": None,
                },
                index_elements=("effect_id", "ordinal"),
            )
            if not inserted:
                raise ValueError("sink effect member winner already exists during new reservation")

    @staticmethod
    def _insert_or_compare_export_association(conn: Connection, effect_id: str, snapshot_id: str) -> None:
        _conflict_safe_insert(
            conn,
            sink_effect_export_snapshots_table,
            {
                "effect_id": effect_id,
                "input_kind": SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT.value,
                "slot": 0,
                "snapshot_id": snapshot_id,
            },
            index_elements=("effect_id", "slot"),
        )
        row = conn.execute(
            select(sink_effect_export_snapshots_table).where(
                sink_effect_export_snapshots_table.c.effect_id == effect_id,
                sink_effect_export_snapshots_table.c.slot == 0,
            )
        ).fetchone()
        if row is None or row.input_kind != SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT.value or row.snapshot_id != snapshot_id:
            raise ValueError("audit export effect snapshot association winner is divergent")

    @staticmethod
    def _insert_or_compare_operation(conn: Connection, request: SinkEffectReservationRequest, effect: SinkEffect) -> None:
        operation_id = _labeled_hash("sink-effect-operation-v1", {"effect_id": effect.effect_id})
        _conflict_safe_insert(
            conn,
            operations_table,
            {
                "operation_id": operation_id,
                "run_id": request.run_id,
                "node_id": request.sink_node_id,
                "operation_type": "sink_write",
                "sink_effect_id": effect.effect_id,
                "started_at": effect.created_at,
                "completed_at": None,
                "status": "open",
                "input_data_ref": None,
                "input_data_hash": None,
                "output_data_ref": None,
                "output_data_hash": None,
                "error_message": None,
                "duration_ms": None,
            },
            index_elements=("sink_effect_id",),
        )
        row = conn.execute(select(operations_table).where(operations_table.c.sink_effect_id == effect.effect_id)).fetchone()
        if (
            row is None
            or row.operation_id != operation_id
            or row.run_id != request.run_id
            or row.node_id != request.sink_node_id
            or row.operation_type != "sink_write"
        ):
            raise ValueError("sink effect operation winner is divergent")

    @staticmethod
    def _backend_pid(conn: Connection) -> int:
        if conn.dialect.name == "postgresql":
            return int(conn.exec_driver_sql("SELECT pg_backend_pid()").scalar_one())
        return 0

    def _after_token_lock(self, _pid: int, _token_ids: tuple[str, ...]) -> None:
        """Deterministic contention seam; production implementation is inert."""

    def _after_state_lock(self, _pid: int, _state_ids: tuple[str, ...]) -> None:
        """Deterministic contention seam; production implementation is inert."""

    def _after_witness_locks(self, _pid: int, _token_ids: tuple[str, ...], _state_ids: tuple[str, ...]) -> None:
        """Proof seam reached only after the complete ordered witness set."""


__all__ = [
    "SinkEffectReservation",
    "SinkEffectReservationRequest",
    "SinkEffectReservationResult",
]
