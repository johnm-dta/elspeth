"""Immutable audit-export snapshot registry persistence and capability binding."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import and_, or_, select
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from elspeth.contracts.audit import AuditExportSnapshot, AuditExportSnapshotChunk
from elspeth.contracts.audit_export import (
    AUDIT_EXPORT_MAX_CHUNK_BYTES,
    AUDIT_EXPORT_MAX_CHUNK_RECORDS,
    AUDIT_EXPORT_MAX_CHUNKS,
    AUDIT_EXPORT_MAX_TOTAL_BYTES,
    AUDIT_EXPORT_MAX_TOTAL_RECORDS,
    AuditExportContentDescriptor,
    AuditExportContentStoreResolver,
    RegisteredAuditExportContent,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.sink_effects import (
    AuditExportFormat,
    AuditExportSignedManifestInput,
    AuditExportSigningMode,
    AuditExportSnapshotChunkInput,
    SinkEffectAuditExportSnapshotInput,
    _create_restricted_audit_export_snapshot_reader,
)
from elspeth.core.landscape.model_loaders import AuditExportSnapshotChunkLoader, _AuditExportSnapshotRowLoader
from elspeth.core.landscape.schema import audit_export_snapshot_chunks_table, audit_export_snapshots_table

if TYPE_CHECKING:
    from collections.abc import Callable

_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")


def _require_positive_bounded(value: int, field_name: str, maximum: int) -> None:
    if type(value) is not int or value < 1 or value > maximum:
        raise ValueError(f"{field_name} must be an exact integer within [1, {maximum}]")


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
class AuditExportSnapshotRegistration:
    winner: AuditExportSnapshotWinner
    inserted: bool

    def __post_init__(self) -> None:
        if type(self.winner) is not AuditExportSnapshotWinner:
            raise TypeError("winner must be exact AuditExportSnapshotWinner")
        if type(self.inserted) is not bool:
            raise TypeError("inserted must be exact bool")


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


def _snapshot_values(snapshot: AuditExportSnapshot) -> dict[str, object]:
    return {field.name: getattr(snapshot, field.name) for field in fields(AuditExportSnapshot)} | {
        "source_status": snapshot.source_status.value,
        "export_format": snapshot.export_format.value,
        "signing_mode": snapshot.signing_mode.value,
    }


def _chunk_values(chunk: AuditExportSnapshotChunk) -> dict[str, object]:
    return {field.name: getattr(chunk, field.name) for field in fields(AuditExportSnapshotChunk)}


def _timestamp(value: datetime) -> str:
    # SQLite's DB-API returns a naive datetime for ``DateTime(timezone=True)``;
    # Landscape timestamps are UTC by contract, so restore that lost carrier
    # metadata at the row-decoding boundary. PostgreSQL retains the offset.
    normalized = value.replace(tzinfo=UTC) if value.tzinfo is None or value.utcoffset() is None else value.astimezone(UTC)
    return normalized.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _snapshot_comparison_values(snapshot: AuditExportSnapshot) -> tuple[object, ...]:
    values: list[object] = []
    for field in fields(AuditExportSnapshot):
        value = getattr(snapshot, field.name)
        if isinstance(value, datetime):
            value = _timestamp(value)
        values.append(value)
    return tuple(values)


class AuditExportSnapshotRepository:
    """Exact CAS registry and bound-winner construction surface.

    Transaction ownership stays with the caller. Long source reads and short
    winner CAS transactions therefore cannot accidentally span content-store
    I/O.
    """

    def __init__(self) -> None:
        self._snapshot_loader = _AuditExportSnapshotRowLoader()
        self._chunk_loader = AuditExportSnapshotChunkLoader()

    @staticmethod
    def winner_from_candidate(candidate: AuditExportSnapshotCandidate) -> AuditExportSnapshotWinner:
        if type(candidate) is not AuditExportSnapshotCandidate:
            raise TypeError("candidate must be exact AuditExportSnapshotCandidate")
        return AuditExportSnapshotWinner(snapshot=candidate.snapshot, chunks=candidate.chunks)

    @staticmethod
    def _identity_predicate(key: AuditExportSnapshotRegistryKey):  # type: ignore[no-untyped-def]
        table = audit_export_snapshots_table.c
        return and_(
            table.source_run_id == key.source_run_id,
            table.exporter_version == key.exporter_version,
            table.serialization_version == key.serialization_version,
            table.export_format == key.export_format.value,
            table.signing_mode == key.signing_mode.value,
            table.signer_key_id == key.signer_key_id,
            table.public_export_config_hash == key.public_export_config_hash,
        )

    @staticmethod
    def _assert_exact_key(snapshot: AuditExportSnapshot, key: AuditExportSnapshotRegistryKey) -> None:
        if AuditExportSnapshotRegistryKey.from_snapshot(snapshot) != key:
            raise AuditIntegrityError("audit-export registry key matched a winner with divergent exact shaping fields")

    def find_winner(
        self,
        connection: Connection,
        key: AuditExportSnapshotRegistryKey,
    ) -> AuditExportSnapshotWinner | None:
        if not isinstance(connection, Connection):
            raise TypeError("connection must be a SQLAlchemy Connection")
        if type(key) is not AuditExportSnapshotRegistryKey:
            raise TypeError("key must be exact AuditExportSnapshotRegistryKey")
        query = select(audit_export_snapshots_table).where(
            or_(
                audit_export_snapshots_table.c.registry_key_hash == key.registry_key_hash,
                self._identity_predicate(key),
            )
        )
        rows = connection.execute(query).fetchmany(2)
        if len(rows) > 1:
            raise AuditIntegrityError("audit-export registry key resolves to multiple immutable winners")
        if not rows:
            return None
        snapshot = self._snapshot_loader.load(rows[0])
        self._assert_exact_key(snapshot, key)
        chunk_rows = connection.execute(
            select(audit_export_snapshot_chunks_table)
            .where(audit_export_snapshot_chunks_table.c.snapshot_id == snapshot.snapshot_id)
            .order_by(audit_export_snapshot_chunks_table.c.ordinal)
        ).fetchall()
        chunks = tuple(self._chunk_loader.load(row) for row in chunk_rows)
        return AuditExportSnapshotWinner(snapshot=snapshot, chunks=chunks)

    @staticmethod
    def _assert_candidate_equals_winner(
        candidate: AuditExportSnapshotCandidate,
        winner: AuditExportSnapshotWinner,
    ) -> None:
        if (
            _snapshot_comparison_values(candidate.snapshot) != _snapshot_comparison_values(winner.snapshot)
            or candidate.chunks != winner.chunks
        ):
            raise AuditIntegrityError("same audit-export registry key produced a divergent snapshot candidate")

    def register_candidate(
        self,
        connection: Connection,
        candidate: AuditExportSnapshotCandidate,
    ) -> AuditExportSnapshotRegistration:
        if type(candidate) is not AuditExportSnapshotCandidate:
            raise TypeError("candidate must be exact AuditExportSnapshotCandidate")
        key = AuditExportSnapshotRegistryKey.from_snapshot(candidate.snapshot)
        existing = self.find_winner(connection, key)
        if existing is not None:
            self._assert_candidate_equals_winner(candidate, existing)
            return AuditExportSnapshotRegistration(winner=existing, inserted=False)

        conflict: IntegrityError | None = None
        try:
            with connection.begin_nested():
                connection.execute(
                    audit_export_snapshot_chunks_table.insert(),
                    [_chunk_values(chunk) for chunk in candidate.chunks],
                )
                connection.execute(audit_export_snapshots_table.insert().values(**_snapshot_values(candidate.snapshot)))
        except IntegrityError as exc:
            conflict = exc

        winner = self.find_winner(connection, key)
        if winner is None:
            if conflict is None:
                raise AuditIntegrityError("audit-export registry CAS insert completed without a visible winner")
            raise AuditIntegrityError("audit-export registry CAS collided without an exact reusable winner") from conflict
        self._assert_candidate_equals_winner(candidate, winner)
        return AuditExportSnapshotRegistration(winner=winner, inserted=conflict is None)

    def bind_winner(
        self,
        winner: AuditExportSnapshotWinner,
        *,
        content_store_resolver: AuditExportContentStoreResolver,
        limits: AuditExportSnapshotReadLimits,
        signed_manifest_verifier: Callable[[bytes, AuditExportSignedManifestInput], None],
    ) -> SinkEffectAuditExportSnapshotInput:
        if type(winner) is not AuditExportSnapshotWinner:
            raise TypeError("winner must be exact AuditExportSnapshotWinner")
        if type(content_store_resolver) is not AuditExportContentStoreResolver:
            raise TypeError("content_store_resolver must be exact AuditExportContentStoreResolver")
        if type(limits) is not AuditExportSnapshotReadLimits:
            raise TypeError("limits must be exact AuditExportSnapshotReadLimits")
        if not callable(signed_manifest_verifier):
            raise TypeError("signed_manifest_verifier must be callable")

        snapshot = winner.snapshot
        store = content_store_resolver.resolve(snapshot.content_store_id)
        chunk_inputs = tuple(
            AuditExportSnapshotChunkInput(
                ordinal=chunk.ordinal,
                content_ref=chunk.content_ref,
                content_hash=chunk.content_hash,
                size_bytes=chunk.size_bytes,
                record_count=chunk.record_count,
            )
            for chunk in winner.chunks
        )
        signed_manifest = AuditExportSignedManifestInput(
            content_ref=snapshot.signed_manifest_ref,
            content_hash=snapshot.signed_manifest_hash,
            size_bytes=snapshot.signed_manifest_size_bytes,
            manifest_schema=snapshot.signed_manifest_schema,
            derivation_version=snapshot.derivation_version,
            signature_algorithm=snapshot.signing_mode,
            signature_key_id=snapshot.signer_key_id,
            record_chain_algorithm=snapshot.record_chain_algorithm,
            final_hash=snapshot.final_hash,
            signature=snapshot.signature_hex,
        )
        descriptors = {
            chunk.content_ref: AuditExportContentDescriptor(
                content_ref=chunk.content_ref,
                content_hash=chunk.content_hash,
                size_bytes=chunk.size_bytes,
                object_kind="data_chunk",
            )
            for chunk in winner.chunks
        }
        descriptors[snapshot.signed_manifest_ref] = AuditExportContentDescriptor(
            content_ref=snapshot.signed_manifest_ref,
            content_hash=snapshot.signed_manifest_hash,
            size_bytes=snapshot.signed_manifest_size_bytes,
            object_kind="final_manifest",
        )

        def resolve_registered(content_ref: str) -> bytes:
            try:
                descriptor = descriptors[content_ref]
            except KeyError as exc:  # pragma: no cover - restricted reader never supplies arbitrary refs
                raise AuditIntegrityError("restricted snapshot reader requested an unregistered content reference") from exc
            registration = RegisteredAuditExportContent(
                snapshot_id=snapshot.snapshot_id,
                content_store_id=snapshot.content_store_id,
                namespace=store.namespace,
                descriptor=descriptor,
            )
            content = store.open_registered(registration).read()
            if type(content) is not bytes:
                raise TypeError("bound audit-export content reader must return exact bytes")
            return content

        reader = _create_restricted_audit_export_snapshot_reader(
            snapshot_id=snapshot.snapshot_id,
            source_run_id=snapshot.source_run_id,
            registry_key_hash=snapshot.registry_key_hash,
            manifest_hash=snapshot.manifest_hash,
            snapshot_hash=snapshot.snapshot_hash,
            export_format=snapshot.export_format,
            signing_mode=snapshot.signing_mode,
            signer_key_id=snapshot.signer_key_id,
            record_count=snapshot.record_count,
            total_bytes=snapshot.total_bytes,
            serialization_version=snapshot.serialization_version,
            exported_at=_timestamp(snapshot.exported_at),
            source_completed_at=_timestamp(snapshot.source_completed_at),
            source_status=snapshot.source_status.value,
            last_chunk_seal_hash=snapshot.last_chunk_seal_hash,
            snapshot_seal_hash=snapshot.snapshot_seal_hash,
            chunks=chunk_inputs,
            signed_manifest=signed_manifest,
            store_resolver=resolve_registered,
            record_counter=lambda content: content.count(b"\n"),
            signed_manifest_verifier=signed_manifest_verifier,
            max_total_bytes=limits.max_total_bytes,
            max_total_records=limits.max_total_records,
            max_chunks=limits.max_chunks,
            max_chunk_bytes=limits.max_chunk_bytes,
            max_chunk_records=limits.max_chunk_records,
        )
        return SinkEffectAuditExportSnapshotInput(
            snapshot_id=snapshot.snapshot_id,
            source_run_id=snapshot.source_run_id,
            registry_key_hash=snapshot.registry_key_hash,
            manifest_hash=snapshot.manifest_hash,
            snapshot_hash=snapshot.snapshot_hash,
            serialization_version=snapshot.serialization_version,
            export_format=snapshot.export_format,
            signing_mode=snapshot.signing_mode,
            signer_key_id=snapshot.signer_key_id,
            record_count=snapshot.record_count,
            total_bytes=snapshot.total_bytes,
            chunk_count=snapshot.chunk_count,
            chunks=chunk_inputs,
            signed_manifest=signed_manifest,
            reader=reader,
        )


__all__ = [
    "AuditExportSnapshotCandidate",
    "AuditExportSnapshotReadLimits",
    "AuditExportSnapshotRegistration",
    "AuditExportSnapshotRegistryKey",
    "AuditExportSnapshotRepository",
    "AuditExportSnapshotWinner",
]
