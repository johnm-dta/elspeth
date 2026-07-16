"""Durable audit-export snapshot registry and capability tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from elspeth.contracts.audit import AuditExportSnapshot, AuditExportSnapshotChunk
from elspeth.contracts.audit_export import (
    AuditExportContentDescriptor,
    AuditExportContentStoreResolver,
    IterableBoundAuditExportContentReader,
    RegisteredAuditExportContent,
)
from elspeth.contracts.enums import RunStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.sink_effects import AuditExportFormat, AuditExportSigningMode
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution.audit_export_snapshots import (
    AuditExportSnapshotCandidate,
    AuditExportSnapshotReadLimits,
    AuditExportSnapshotRegistryKey,
    AuditExportSnapshotRepository,
)
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import audit_export_snapshot_chunks_table, audit_export_snapshots_table, runs_table

COMPLETED_AT = datetime(2026, 7, 16, 1, 2, 3, 456789, tzinfo=UTC)
COMPLETED_AT_TEXT = "2026-07-16T01:02:03.456789Z"


class _MemoryContentStore:
    content_store_id = "durable-store"
    namespace = "audit/export"

    def __init__(self) -> None:
        self.content: dict[str, bytes] = {}
        self.opened: list[RegisteredAuditExportContent] = []
        self.orphans: list[tuple[str, tuple[AuditExportContentDescriptor, ...]]] = []

    def is_durable(self) -> bool:
        return True

    def put_immutable(self, content: bytes, *, candidate_id: str, object_kind: str) -> str:
        del candidate_id, object_kind
        content_ref = f"sha256:{sha256(content).hexdigest()}"
        self.content.setdefault(content_ref, content)
        return content_ref

    def open_registered(self, registration: RegisteredAuditExportContent) -> IterableBoundAuditExportContentReader:
        assert registration.content_store_id == self.content_store_id
        assert registration.namespace == self.namespace
        self.opened.append(registration)
        return IterableBoundAuditExportContentReader(self.content[registration.descriptor.content_ref])

    def mark_candidate_orphans(
        self,
        candidate_id: str,
        descriptors: tuple[AuditExportContentDescriptor, ...],
    ) -> None:
        self.orphans.append((candidate_id, descriptors))

    def garbage_collect_candidate(self, request: object) -> bool:
        del request
        return False


def _insert_terminal_run(db: LandscapeDB, run_id: str = "run-export") -> None:
    with db.engine.begin() as connection:
        connection.execute(
            runs_table.insert().values(
                run_id=run_id,
                started_at=COMPLETED_AT,
                completed_at=COMPLETED_AT,
                config_hash="0" * 64,
                settings_json="{}",
                canonical_version="v1",
                status="completed",
                openrouter_catalog_sha256="1" * 64,
                openrouter_catalog_source="bundled",
            )
        )


def _registry_key(*, signer_key_id: str = "UNSIGNED", registry_key_hash: str = "5" * 64) -> AuditExportSnapshotRegistryKey:
    return AuditExportSnapshotRegistryKey(
        source_run_id="run-export",
        exporter_version="landscape-exporter-v2",
        serialization_version="audit-export-v2",
        export_format=AuditExportFormat.JSON,
        signing_mode=AuditExportSigningMode.UNSIGNED,
        signer_key_id=signer_key_id,
        public_export_config_hash="6" * 64,
        registry_key_hash=registry_key_hash,
        chunking_algorithm_version="complete-frame-v1",
        per_chunk_record_limit=100,
        per_chunk_byte_limit=1024,
    )


def _candidate(
    store: _MemoryContentStore,
    *,
    key: AuditExportSnapshotRegistryKey | None = None,
    snapshot_id: str = "2" * 64,
    manifest_hash: str = "7" * 64,
) -> AuditExportSnapshotCandidate:
    key = key or _registry_key()
    chunk_bytes = b'{"record_type":"run"}\n'
    chunk_hash = sha256(chunk_bytes).hexdigest()
    chunk_ref = store.put_immutable(chunk_bytes, candidate_id=snapshot_id, object_kind="data_chunk")
    chunk_seal_hash = "4" * 64
    snapshot_hash = "8" * 64
    snapshot_seal_hash = "9" * 64
    final_hash = "a" * 64
    manifest = {
        "chunk_count": 1,
        "derivation_version": "audit-export-derivation-v1",
        "export_format": "json",
        "exported_at": COMPLETED_AT_TEXT,
        "final_hash": final_hash,
        "hash_algorithm": "sha256",
        "last_chunk_seal_hash": chunk_seal_hash,
        "manifest_hash": manifest_hash,
        "record_chain_algorithm": "sha256_concat_record_sha256_v1",
        "record_count": 1,
        "record_type": "manifest",
        "registry_key_hash": key.registry_key_hash,
        "run_id": key.source_run_id,
        "schema": "elspeth.audit-export-manifest.v2",
        "signature": None,
        "signature_algorithm": "unsigned",
        "signature_key_id": key.signer_key_id,
        "snapshot_hash": snapshot_hash,
        "snapshot_id": snapshot_id,
        "snapshot_seal_hash": snapshot_seal_hash,
        "source_completed_at": COMPLETED_AT_TEXT,
        "source_status": "completed",
        "total_bytes": len(chunk_bytes),
    }
    manifest_bytes = canonical_json(manifest).encode("utf-8")
    signed_manifest_hash = sha256(manifest_bytes).hexdigest()
    signed_manifest_ref = store.put_immutable(manifest_bytes, candidate_id=snapshot_id, object_kind="final_manifest")
    snapshot = AuditExportSnapshot(
        snapshot_id=snapshot_id,
        source_run_id=key.source_run_id,
        source_status=RunStatus.COMPLETED,
        source_completed_at=COMPLETED_AT,
        exported_at=COMPLETED_AT,
        registry_key_hash=key.registry_key_hash,
        exporter_version=key.exporter_version,
        serialization_version=key.serialization_version,
        export_format=key.export_format,
        signing_mode=key.signing_mode,
        signer_key_id=key.signer_key_id,
        derivation_version="audit-export-derivation-v1",
        public_export_config_hash=key.public_export_config_hash,
        chunking_algorithm_version=key.chunking_algorithm_version,
        per_chunk_record_limit=key.per_chunk_record_limit,
        per_chunk_byte_limit=key.per_chunk_byte_limit,
        record_count=1,
        total_bytes=len(chunk_bytes),
        chunk_count=1,
        terminal_chunk_ordinal=0,
        content_store_id=store.content_store_id,
        manifest_hash=manifest_hash,
        last_chunk_seal_hash=chunk_seal_hash,
        snapshot_hash=snapshot_hash,
        snapshot_seal_hash=snapshot_seal_hash,
        signature_hex=None,
        record_chain_algorithm="sha256_concat_record_sha256_v1",
        final_hash=final_hash,
        signed_manifest_schema="elspeth.audit-export-manifest.v2",
        signed_manifest_hash=signed_manifest_hash,
        signed_manifest_ref=signed_manifest_ref,
        signed_manifest_size_bytes=len(manifest_bytes),
    )
    chunk = AuditExportSnapshotChunk(
        snapshot_id=snapshot_id,
        ordinal=0,
        content_ref=chunk_ref,
        content_hash=chunk_hash,
        size_bytes=len(chunk_bytes),
        record_count=1,
        predecessor_seal_hash=None,
        cumulative_records=1,
        cumulative_bytes=len(chunk_bytes),
        chunk_seal_hash=chunk_seal_hash,
    )
    return AuditExportSnapshotCandidate(snapshot=snapshot, chunks=(chunk,))


def _repository() -> AuditExportSnapshotRepository:
    return AuditExportSnapshotRepository()


def test_writable_factory_exposes_one_snapshot_repository_capability() -> None:
    db = LandscapeDB.in_memory()
    try:
        factory = RecorderFactory(db)
        assert type(factory.audit_export_snapshots) is AuditExportSnapshotRepository
        assert factory.write_repositories().audit_export_snapshots is factory.audit_export_snapshots
    finally:
        db.close()


def test_registry_cas_inserts_once_and_reuses_exact_winner() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    candidate = _candidate(store)
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            first = _repository().register_candidate(connection, candidate)
        with db.engine.begin() as connection:
            second = _repository().register_candidate(connection, candidate)

        assert first.inserted is True
        assert second.inserted is False
        assert first.winner == second.winner
        with db.engine.connect() as connection:
            assert connection.scalar(select(func.count()).select_from(audit_export_snapshots_table)) == 1
            assert connection.scalar(select(func.count()).select_from(audit_export_snapshot_chunks_table)) == 1
    finally:
        db.close()


def test_registry_lookup_exact_checks_every_shaping_field_and_signer_id() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    candidate = _candidate(store)
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            _repository().register_candidate(connection, candidate)
        with db.engine.connect() as connection:
            assert _repository().find_winner(connection, _registry_key()) is not None
            with pytest.raises(AuditIntegrityError, match="registry key"):
                _repository().find_winner(connection, replace(_registry_key(), signer_key_id="rotated-key"))
    finally:
        db.close()


def test_same_registry_key_with_divergent_candidate_fails_closed() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    first = _candidate(store)
    divergent = _candidate(store, snapshot_id="d" * 64, manifest_hash="e" * 64)
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            _repository().register_candidate(connection, first)
        with db.engine.begin() as connection, pytest.raises(AuditIntegrityError, match="divergent"):
            _repository().register_candidate(connection, divergent)
        with db.engine.connect() as connection:
            assert connection.scalar(select(func.count()).select_from(audit_export_snapshots_table)) == 1
            assert connection.scalar(select(func.count()).select_from(audit_export_snapshot_chunks_table)) == 1
    finally:
        db.close()


def test_bound_winner_reader_resolves_only_registered_store_content() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    candidate = _candidate(store)
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            winner = _repository().register_candidate(connection, candidate).winner

        effect_input = _repository().bind_winner(
            winner,
            content_store_resolver=resolver,
            limits=AuditExportSnapshotReadLimits(),
            signed_manifest_verifier=lambda _content, _descriptor: None,
        )

        assert list(effect_input.reader.iter_verified_chunks()) == [b'{"record_type":"run"}\n']
        manifest = effect_input.reader.read_verified_signed_manifest()
        assert manifest.endswith(b"}") and not manifest.endswith(b"\n")
        assert {item.descriptor.object_kind for item in store.opened} == {"data_chunk", "final_manifest"}
    finally:
        db.close()


def test_winner_store_resolution_failure_never_substitutes_current_store() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    candidate = _candidate(store)
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            winner = _repository().register_candidate(connection, candidate).winner
        with pytest.raises(LookupError, match="unresolvable"):
            _repository().bind_winner(
                winner,
                content_store_resolver=AuditExportContentStoreResolver(),
                limits=AuditExportSnapshotReadLimits(),
                signed_manifest_verifier=lambda _content, _descriptor: None,
            )
    finally:
        db.close()


def test_current_acceptance_limits_are_enforced_without_changing_identity() -> None:
    store = _MemoryContentStore()
    candidate = _candidate(store)
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    winner = _repository().winner_from_candidate(candidate)

    with pytest.raises(ValueError, match="configured reader limits"):
        _repository().bind_winner(
            winner,
            content_store_resolver=resolver,
            limits=AuditExportSnapshotReadLimits(max_total_bytes=candidate.snapshot.total_bytes - 1),
            signed_manifest_verifier=lambda _content, _descriptor: None,
        )


def test_reader_rechecks_content_bytes_before_yielding_any_bad_chunk() -> None:
    store = _MemoryContentStore()
    candidate = _candidate(store)
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    winner = _repository().winner_from_candidate(candidate)
    reader = (
        _repository()
        .bind_winner(
            winner,
            content_store_resolver=resolver,
            limits=AuditExportSnapshotReadLimits(),
            signed_manifest_verifier=lambda _content, _descriptor: None,
        )
        .reader
    )
    store.content[candidate.chunks[0].content_ref] = b"tampered\n"

    with pytest.raises(ValueError, match=r"chunk (size|hash)"):
        list(reader.iter_verified_chunks())


def test_sealed_registry_rows_remain_database_immutable() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            _repository().register_candidate(connection, _candidate(store))
        with pytest.raises(IntegrityError), db.engine.begin() as connection:
            connection.execute(update(audit_export_snapshots_table).values(content_store_id="replacement"))
        with pytest.raises(IntegrityError), db.engine.begin() as connection:
            connection.execute(update(audit_export_snapshot_chunks_table).values(size_bytes=999))
    finally:
        db.close()
