"""Immutable audit-export snapshot registry persistence and capability binding."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from sqlalchemy import and_, or_, select
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from elspeth.contracts.audit import AuditExportSnapshot, AuditExportSnapshotChunk
from elspeth.contracts.audit_export import (
    AUDIT_EXPORT_DERIVATION_VERSION,
    AuditExportContentDescriptor,
    AuditExportContentStoreResolver,
    AuditExportSnapshotCandidate,
    AuditExportSnapshotReadLimits,
    AuditExportSnapshotRegistryKey,
    AuditExportSnapshotWinner,
    C,
    ClosedAuditExportJSON,
    H,
    RegisteredAuditExportContent,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.sink_effects import (
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


def _expect_graph_value(observed: object, expected: object, field_name: str) -> None:
    if observed != expected:
        raise AuditIntegrityError(f"audit-export snapshot {field_name} is not derived from its registered bytes")


def _verify_snapshot_graph(
    snapshot: AuditExportSnapshot,
    chunks: tuple[AuditExportSnapshotChunk, ...],
    *,
    resolve_registered: Callable[[str], bytes],
    signed_manifest_verifier: Callable[[bytes, AuditExportSignedManifestInput], None],
    record_signature_verifier: Callable[[bytes, str], None] | None,
) -> None:
    """Recompute the complete cryptographic graph from registered bytes."""

    snapshot_chunks: list[dict[str, ClosedAuditExportJSON]] = []
    chain = hashlib.sha256()
    cumulative_records = 0
    cumulative_bytes = 0
    for chunk in chunks:
        content = resolve_registered(chunk.content_ref)
        content_hash = H(content)
        _expect_graph_value(content_hash, chunk.content_hash, f"chunk {chunk.ordinal} content_hash")
        _expect_graph_value(len(content), chunk.size_bytes, f"chunk {chunk.ordinal} size_bytes")
        if not content.endswith(b"\n"):
            raise AuditIntegrityError(f"audit-export snapshot chunk {chunk.ordinal} does not end on a complete record frame")
        record_count = 0
        for frame in content.splitlines(keepends=True):
            if not frame.endswith(b"\n") or frame == b"\n":
                raise AuditIntegrityError(f"audit-export snapshot chunk {chunk.ordinal} contains an incomplete record frame")
            try:
                emitted = json.loads(frame[:-1])
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AuditIntegrityError(f"audit-export snapshot chunk {chunk.ordinal} contains invalid JSON") from exc
            if type(emitted) is not dict or emitted.get("record_type") == "manifest":
                raise AuditIntegrityError(f"audit-export snapshot chunk {chunk.ordinal} contains an invalid data record")
            if canonical_json(emitted).encode("utf-8") + b"\n" != frame:
                raise AuditIntegrityError(f"audit-export snapshot chunk {chunk.ordinal} contains non-canonical record bytes")
            signature = emitted.pop("signature", None)
            unsigned_bytes = canonical_json(emitted).encode("utf-8")
            if snapshot.signing_mode is AuditExportSigningMode.UNSIGNED:
                if signature is not None:
                    raise AuditIntegrityError("unsigned audit-export snapshot contains a record signature")
                chain.update(hashlib.sha256(unsigned_bytes).hexdigest().encode("ascii"))
            else:
                if type(signature) is not str or _LOWER_HEX_64.fullmatch(signature) is None:
                    raise AuditIntegrityError("signed audit-export snapshot contains an invalid record signature")
                if record_signature_verifier is None:
                    raise AuditIntegrityError("signed audit-export snapshot requires a record-signature verifier")
                try:
                    record_signature_verifier(unsigned_bytes, signature)
                except AuditIntegrityError:
                    raise
                except Exception as exc:
                    raise AuditIntegrityError("audit-export snapshot record HMAC verification failed") from exc
                chain.update(signature.encode("ascii"))
            record_count += 1
        _expect_graph_value(record_count, chunk.record_count, f"chunk {chunk.ordinal} record_count")
        cumulative_records += record_count
        cumulative_bytes += len(content)
        _expect_graph_value(cumulative_records, chunk.cumulative_records, f"chunk {chunk.ordinal} cumulative_records")
        _expect_graph_value(cumulative_bytes, chunk.cumulative_bytes, f"chunk {chunk.ordinal} cumulative_bytes")
        snapshot_chunks.append(
            {
                "content_hash": content_hash,
                "cumulative_bytes": cumulative_bytes,
                "cumulative_records": cumulative_records,
                "ordinal": chunk.ordinal,
                "record_count": record_count,
                "size_bytes": len(content),
            }
        )

    public_key_payload: dict[str, ClosedAuditExportJSON] = {
        "export_format": snapshot.export_format.value,
        "exporter_version": snapshot.exporter_version,
        "public_export_config_hash": snapshot.public_export_config_hash,
        "serialization_version": snapshot.serialization_version,
        "signer_key_id": snapshot.signer_key_id,
        "signing_mode": snapshot.signing_mode.value,
        "source_run_id": snapshot.source_run_id,
    }
    registry_key_hash = H(C("audit-export-registry-key-v1", public_key_payload))
    _expect_graph_value(registry_key_hash, snapshot.registry_key_hash, "registry_key_hash")
    snapshot_content: dict[str, ClosedAuditExportJSON] = {
        "chunking_algorithm_version": snapshot.chunking_algorithm_version,
        "chunks": cast(list[ClosedAuditExportJSON], snapshot_chunks),
        "record_count": cumulative_records,
        "serialization_version": snapshot.serialization_version,
        "total_bytes": cumulative_bytes,
    }
    snapshot_hash = H(C("audit-export-snapshot-content-v1", snapshot_content))
    snapshot_id_payload: dict[str, ClosedAuditExportJSON] = {
        "registry_key_hash": registry_key_hash,
        "snapshot_hash": snapshot_hash,
    }
    snapshot_id = H(C("audit-export-snapshot-id-v1", snapshot_id_payload))
    _expect_graph_value(snapshot_hash, snapshot.snapshot_hash, "snapshot_hash")
    _expect_graph_value(snapshot_id, snapshot.snapshot_id, "snapshot_id")

    manifest_chunks: list[dict[str, ClosedAuditExportJSON]] = []
    predecessor: str | None = None
    for chunk, summary in zip(chunks, snapshot_chunks, strict=True):
        predecessor_object: dict[str, ClosedAuditExportJSON] = (
            {"kind": "genesis"} if predecessor is None else {"hash": predecessor, "kind": "chunk_seal"}
        )
        seal_payload: dict[str, ClosedAuditExportJSON] = {
            "chunking_algorithm_version": snapshot.chunking_algorithm_version,
            "content_hash": summary["content_hash"],
            "content_ref": chunk.content_ref,
            "cumulative_bytes": summary["cumulative_bytes"],
            "cumulative_records": summary["cumulative_records"],
            "derivation_version": AUDIT_EXPORT_DERIVATION_VERSION,
            "ordinal": chunk.ordinal,
            "predecessor": predecessor_object,
            "record_count": summary["record_count"],
            "serialization_version": snapshot.serialization_version,
            "size_bytes": summary["size_bytes"],
            "snapshot_id": snapshot_id,
        }
        seal_hash = H(C("audit-export-chunk-seal-v1", seal_payload))
        _expect_graph_value(chunk.predecessor_seal_hash, predecessor, f"chunk {chunk.ordinal} predecessor seal")
        _expect_graph_value(chunk.chunk_seal_hash, seal_hash, f"chunk {chunk.ordinal} chunk seal")
        manifest_chunks.append(
            {
                "chunk_seal_hash": seal_hash,
                "content_hash": summary["content_hash"],
                "content_ref": chunk.content_ref,
                "cumulative_bytes": summary["cumulative_bytes"],
                "cumulative_records": summary["cumulative_records"],
                "ordinal": chunk.ordinal,
                "predecessor_seal_hash": predecessor,
                "record_count": summary["record_count"],
                "size_bytes": summary["size_bytes"],
            }
        )
        predecessor = seal_hash
    assert predecessor is not None
    manifest_payload: dict[str, ClosedAuditExportJSON] = {
        "chunk_count": len(chunks),
        "chunks": cast(list[ClosedAuditExportJSON], manifest_chunks),
        "record_count": cumulative_records,
        "snapshot_id": snapshot_id,
        "total_bytes": cumulative_bytes,
    }
    manifest_hash = H(C("audit-export-manifest-v1", manifest_payload))
    _expect_graph_value(manifest_hash, snapshot.manifest_hash, "manifest_hash")
    _expect_graph_value(predecessor, snapshot.last_chunk_seal_hash, "last_chunk_seal_hash")
    snapshot_seal_payload: dict[str, ClosedAuditExportJSON] = {
        "chunk_count": len(chunks),
        "chunking_algorithm_version": snapshot.chunking_algorithm_version,
        "exported_at": _timestamp(snapshot.exported_at),
        "last_chunk_seal_hash": predecessor,
        "manifest_hash": manifest_hash,
        "per_chunk_byte_limit": snapshot.per_chunk_byte_limit,
        "per_chunk_record_limit": snapshot.per_chunk_record_limit,
        "record_count": cumulative_records,
        "registry_key_hash": registry_key_hash,
        "snapshot_hash": snapshot_hash,
        "snapshot_id": snapshot_id,
        "source_completed_at": _timestamp(snapshot.source_completed_at),
        "source_run_id": snapshot.source_run_id,
        "source_status": snapshot.source_status.value,
        "total_bytes": cumulative_bytes,
    }
    snapshot_seal_hash = H(C("audit-export-snapshot-seal-v1", snapshot_seal_payload))
    _expect_graph_value(snapshot_seal_hash, snapshot.snapshot_seal_hash, "snapshot_seal_hash")
    final_hash = chain.hexdigest()
    _expect_graph_value(final_hash, snapshot.final_hash, "final_hash")
    expected_chain = (
        "sha256_concat_record_sha256_v1"
        if snapshot.signing_mode is AuditExportSigningMode.UNSIGNED
        else "sha256_concat_hmac_sha256_signatures_v1"
    )
    _expect_graph_value(snapshot.record_chain_algorithm, expected_chain, "record_chain_algorithm")
    final_core: dict[str, ClosedAuditExportJSON] = {
        "chunk_count": len(chunks),
        "derivation_version": AUDIT_EXPORT_DERIVATION_VERSION,
        "export_format": snapshot.export_format.value,
        "exported_at": _timestamp(snapshot.exported_at),
        "final_hash": final_hash,
        "hash_algorithm": "sha256",
        "last_chunk_seal_hash": predecessor,
        "manifest_hash": manifest_hash,
        "record_chain_algorithm": expected_chain,
        "record_count": cumulative_records,
        "record_type": "manifest",
        "registry_key_hash": registry_key_hash,
        "run_id": snapshot.source_run_id,
        "schema": snapshot.signed_manifest_schema,
        "signature_algorithm": snapshot.signing_mode.value,
        "signature_key_id": snapshot.signer_key_id,
        "snapshot_hash": snapshot_hash,
        "snapshot_id": snapshot_id,
        "snapshot_seal_hash": snapshot_seal_hash,
        "source_completed_at": _timestamp(snapshot.source_completed_at),
        "source_status": snapshot.source_status.value,
        "total_bytes": cumulative_bytes,
    }
    manifest_bytes = resolve_registered(snapshot.signed_manifest_ref)
    expected_manifest_bytes = canonical_json({**final_core, "signature": snapshot.signature_hex}).encode("utf-8")
    _expect_graph_value(manifest_bytes, expected_manifest_bytes, "signed manifest bytes")
    _expect_graph_value(H(manifest_bytes), snapshot.signed_manifest_hash, "signed_manifest_hash")
    _expect_graph_value(len(manifest_bytes), snapshot.signed_manifest_size_bytes, "signed_manifest_size_bytes")
    descriptor = AuditExportSignedManifestInput(
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
    try:
        signed_manifest_verifier(manifest_bytes, descriptor)
    except AuditIntegrityError:
        raise
    except Exception as exc:
        raise AuditIntegrityError("audit-export snapshot final-manifest signature verification failed") from exc


def _validate_snapshot_limits(
    snapshot: AuditExportSnapshot,
    chunks: tuple[AuditExportSnapshotChunk, ...],
    limits: AuditExportSnapshotReadLimits,
) -> None:
    if snapshot.total_bytes > limits.max_total_bytes:
        raise ValueError("snapshot exceeds configured reader limits: total_bytes")
    if snapshot.record_count > limits.max_total_records:
        raise ValueError("snapshot exceeds configured reader limits: record_count")
    if snapshot.chunk_count > limits.max_chunks:
        raise ValueError("snapshot exceeds configured reader limits: chunk_count")
    for chunk in chunks:
        if chunk.size_bytes > limits.max_chunk_bytes:
            raise ValueError("snapshot exceeds configured reader limits: chunk size_bytes")
        if chunk.record_count > limits.max_chunk_records:
            raise ValueError("snapshot exceeds configured reader limits: chunk record_count")


def _registered_content_resolver(
    snapshot: AuditExportSnapshot,
    chunks: tuple[AuditExportSnapshotChunk, ...],
    content_store_resolver: AuditExportContentStoreResolver,
) -> Callable[[str], bytes]:
    store = content_store_resolver.resolve(snapshot.content_store_id)
    descriptors = {
        chunk.content_ref: AuditExportContentDescriptor(
            content_ref=chunk.content_ref,
            content_hash=chunk.content_hash,
            size_bytes=chunk.size_bytes,
            object_kind="data_chunk",
        )
        for chunk in chunks
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
        except KeyError as exc:  # pragma: no cover - callers use registered refs only
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

    return resolve_registered


@dataclass(frozen=True, slots=True)
class AuditExportSnapshotRegistration:
    winner: AuditExportSnapshotWinner
    inserted: bool

    def __post_init__(self) -> None:
        if type(self.winner) is not AuditExportSnapshotWinner:
            raise TypeError("winner must be exact AuditExportSnapshotWinner")
        if type(self.inserted) is not bool:
            raise TypeError("inserted must be exact bool")


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
        # Store provenance belongs to the CAS winner, not to the canonical
        # byte graph. A rotated writer may derive and store the exact same
        # graph under a new store ID and must reuse the prior winner.
        if field.name == "content_store_id":
            continue
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

    def find_lineage_signer_key_ids(
        self,
        connection: Connection,
        key: AuditExportSnapshotRegistryKey,
    ) -> tuple[str, ...]:
        """Return signer identities already sealed for this export lineage."""
        if not isinstance(connection, Connection):
            raise TypeError("connection must be a SQLAlchemy Connection")
        if type(key) is not AuditExportSnapshotRegistryKey:
            raise TypeError("key must be exact AuditExportSnapshotRegistryKey")
        table = audit_export_snapshots_table.c
        rows = connection.execute(
            select(table.signer_key_id)
            .where(
                table.source_run_id == key.source_run_id,
                table.exporter_version == key.exporter_version,
                table.serialization_version == key.serialization_version,
                table.export_format == key.export_format.value,
            )
            .distinct()
            .order_by(table.signer_key_id)
        ).scalars()
        return tuple(rows)

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
        *,
        content_store_resolver: AuditExportContentStoreResolver,
        limits: AuditExportSnapshotReadLimits,
        signed_manifest_verifier: Callable[[bytes, AuditExportSignedManifestInput], None],
        record_signature_verifier: Callable[[bytes, str], None] | None = None,
    ) -> AuditExportSnapshotRegistration:
        if type(candidate) is not AuditExportSnapshotCandidate:
            raise TypeError("candidate must be exact AuditExportSnapshotCandidate")
        if type(content_store_resolver) is not AuditExportContentStoreResolver:
            raise TypeError("content_store_resolver must be exact AuditExportContentStoreResolver")
        if type(limits) is not AuditExportSnapshotReadLimits:
            raise TypeError("limits must be exact AuditExportSnapshotReadLimits")
        if not callable(signed_manifest_verifier):
            raise TypeError("signed_manifest_verifier must be callable")
        _validate_snapshot_limits(candidate.snapshot, candidate.chunks, limits)
        candidate_resolver = _registered_content_resolver(candidate.snapshot, candidate.chunks, content_store_resolver)
        _verify_snapshot_graph(
            candidate.snapshot,
            candidate.chunks,
            resolve_registered=candidate_resolver,
            signed_manifest_verifier=signed_manifest_verifier,
            record_signature_verifier=record_signature_verifier,
        )
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
        record_signature_verifier: Callable[[bytes, str], None] | None = None,
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
        _validate_snapshot_limits(snapshot, winner.chunks, limits)
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
        resolve_registered = _registered_content_resolver(snapshot, winner.chunks, content_store_resolver)

        _verify_snapshot_graph(
            snapshot,
            winner.chunks,
            resolve_registered=resolve_registered,
            signed_manifest_verifier=signed_manifest_verifier,
            record_signature_verifier=record_signature_verifier,
        )

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
