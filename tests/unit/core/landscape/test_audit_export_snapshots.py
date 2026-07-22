"""Durable audit-export snapshot registry and capability tests."""

from __future__ import annotations

import hmac
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from hashlib import sha256

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from elspeth.contracts.audit import AuditExportSnapshot, AuditExportSnapshotChunk
from elspeth.contracts.audit_export import (
    AuditExportContentDescriptor,
    AuditExportContentStoreResolver,
    AuditExportDerivationConfig,
    C,
    IterableBoundAuditExportContentReader,
    RegisteredAuditExportContent,
    derive_audit_export_bundle,
)
from elspeth.contracts.enums import RunStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.sink_effects import AuditExportFormat, AuditExportSignedManifestInput, AuditExportSigningMode
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution.audit_export_snapshots import (
    AuditExportSnapshotCandidate,
    AuditExportSnapshotReadLimits,
    AuditExportSnapshotRegistryKey,
    AuditExportSnapshotRepository,
    VerifiedAuditExportCandidate,
)
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import audit_export_snapshot_chunks_table, audit_export_snapshots_table, runs_table

COMPLETED_AT = datetime(2026, 7, 16, 1, 2, 3, 456789, tzinfo=UTC)
COMPLETED_AT_TEXT = "2026-07-16T01:02:03.456789Z"
HMAC_KEY = b"audit-export-test-key"


class _MemoryContentStore:
    def __init__(self, content_store_id: str = "durable-store") -> None:
        self.content_store_id = content_store_id
        self.namespace = "audit/export"
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


def _derivation_config(*, signed: bool = False) -> AuditExportDerivationConfig:
    return AuditExportDerivationConfig(
        source_run_id="run-export",
        source_status="completed",
        source_completed_at=COMPLETED_AT_TEXT,
        export_format="json",
        exporter_version="landscape-exporter-v2",
        serialization_version="audit-export-v2",
        chunking_algorithm_version="complete-frame-v1",
        include_raw_error_rows=False,
        per_chunk_record_limit=100,
        per_chunk_byte_limit=1024,
        signing_mode="hmac_sha256" if signed else "unsigned",
        signer_key_id="audit-key-v1" if signed else "UNSIGNED",
        signing_key=HMAC_KEY if signed else None,
    )


def _registry_key() -> AuditExportSnapshotRegistryKey:
    bundle = derive_audit_export_bundle([{"record_type": "run"}], _derivation_config())
    return AuditExportSnapshotRegistryKey(
        source_run_id=bundle.config.source_run_id,
        exporter_version=bundle.config.exporter_version,
        serialization_version=bundle.config.serialization_version,
        export_format=AuditExportFormat(bundle.config.export_format),
        signing_mode=AuditExportSigningMode(bundle.config.signing_mode),
        signer_key_id=bundle.config.signer_key_id,
        public_export_config_hash=bundle.public_export_config_hash,
        registry_key_hash=bundle.registry_key_hash,
        chunking_algorithm_version=bundle.config.chunking_algorithm_version,
        per_chunk_record_limit=bundle.config.per_chunk_record_limit,
        per_chunk_byte_limit=bundle.config.per_chunk_byte_limit,
    )


def _candidate(
    store: _MemoryContentStore,
    *,
    records: list[dict[str, object]] | None = None,
    signed: bool = False,
) -> AuditExportSnapshotCandidate:
    bundle = derive_audit_export_bundle(records or [{"record_type": "run"}], _derivation_config(signed=signed))
    for chunk in bundle.chunks:
        assert (
            store.put_immutable(
                chunk.content,
                candidate_id=bundle.snapshot_id,
                object_kind="data_chunk",
            )
            == chunk.descriptor.content_ref
        )
    assert (
        store.put_immutable(
            bundle.signed_manifest_bytes,
            candidate_id=bundle.snapshot_id,
            object_kind="final_manifest",
        )
        == bundle.signed_manifest.content_ref
    )
    snapshot = AuditExportSnapshot(
        snapshot_id=bundle.snapshot_id,
        source_run_id=bundle.config.source_run_id,
        source_status=RunStatus.COMPLETED,
        source_completed_at=COMPLETED_AT,
        exported_at=COMPLETED_AT,
        registry_key_hash=bundle.registry_key_hash,
        exporter_version=bundle.config.exporter_version,
        serialization_version=bundle.config.serialization_version,
        export_format=AuditExportFormat(bundle.config.export_format),
        signing_mode=AuditExportSigningMode(bundle.config.signing_mode),
        signer_key_id=bundle.config.signer_key_id,
        derivation_version="audit-export-derivation-v1",
        public_export_config_hash=bundle.public_export_config_hash,
        chunking_algorithm_version=bundle.config.chunking_algorithm_version,
        per_chunk_record_limit=bundle.config.per_chunk_record_limit,
        per_chunk_byte_limit=bundle.config.per_chunk_byte_limit,
        record_count=len(bundle.record_frames),
        total_bytes=sum(chunk.descriptor.size_bytes for chunk in bundle.chunks),
        chunk_count=len(bundle.chunks),
        terminal_chunk_ordinal=bundle.chunks[-1].ordinal,
        content_store_id=store.content_store_id,
        manifest_hash=bundle.manifest_hash,
        last_chunk_seal_hash=bundle.last_chunk_seal_hash,
        snapshot_hash=bundle.snapshot_hash,
        snapshot_seal_hash=bundle.snapshot_seal_hash,
        signature_hex=bundle.signed_manifest.signature,
        record_chain_algorithm=bundle.record_chain_algorithm,
        final_hash=bundle.final_hash,
        signed_manifest_schema="elspeth.audit-export-manifest.v2",
        signed_manifest_hash=bundle.signed_manifest.content_hash,
        signed_manifest_ref=bundle.signed_manifest.content_ref,
        signed_manifest_size_bytes=bundle.signed_manifest.size_bytes,
    )
    chunks = tuple(
        AuditExportSnapshotChunk(
            snapshot_id=bundle.snapshot_id,
            ordinal=chunk.ordinal,
            content_ref=chunk.descriptor.content_ref,
            content_hash=chunk.descriptor.content_hash,
            size_bytes=chunk.descriptor.size_bytes,
            record_count=chunk.record_count,
            predecessor_seal_hash=chunk.predecessor_seal_hash,
            cumulative_records=chunk.cumulative_records,
            cumulative_bytes=chunk.cumulative_bytes,
            chunk_seal_hash=chunk.chunk_seal_hash,
        )
        for chunk in bundle.chunks
    )
    return AuditExportSnapshotCandidate(snapshot=snapshot, chunks=chunks)


def _repository() -> AuditExportSnapshotRepository:
    return AuditExportSnapshotRepository()


def _signed_manifest_verifier(content: bytes, descriptor: AuditExportSignedManifestInput) -> None:
    manifest = json.loads(content)
    signature = manifest.pop("signature")
    expected = hmac.new(HMAC_KEY, C("audit-export-final-manifest-signing-body-v2", manifest), sha256).hexdigest()
    if not hmac.compare_digest(signature, expected) or descriptor.signature != signature:
        raise ValueError("invalid final manifest HMAC")


def _record_signature_verifier(unsigned_bytes: bytes, signature: str) -> None:
    expected = hmac.new(HMAC_KEY, unsigned_bytes, sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("invalid record HMAC")


def _register_candidate(connection: object, candidate: AuditExportSnapshotCandidate, store: _MemoryContentStore):
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    return _repository().register_candidate(
        connection,  # type: ignore[arg-type]
        candidate,
        content_store_resolver=resolver,
        limits=AuditExportSnapshotReadLimits(),
        signed_manifest_verifier=lambda _content, _descriptor: None,
    )


def _insert_unverified_candidate(connection: object, candidate: AuditExportSnapshotCandidate) -> None:
    chunk_values = [asdict(chunk) for chunk in candidate.chunks]
    snapshot_values = asdict(candidate.snapshot) | {
        "source_status": candidate.snapshot.source_status.value,
        "export_format": candidate.snapshot.export_format.value,
        "signing_mode": candidate.snapshot.signing_mode.value,
    }
    connection.execute(audit_export_snapshot_chunks_table.insert(), chunk_values)  # type: ignore[attr-defined]
    connection.execute(audit_export_snapshots_table.insert().values(**snapshot_values))  # type: ignore[attr-defined]


def _forge_chunk_seal(
    store: _MemoryContentStore,
    candidate: AuditExportSnapshotCandidate,
) -> AuditExportSnapshotCandidate:
    forged_seal = "f" * 64
    assert forged_seal != candidate.chunks[-1].chunk_seal_hash
    manifest = json.loads(store.content[candidate.snapshot.signed_manifest_ref])
    manifest["last_chunk_seal_hash"] = forged_seal
    manifest_bytes = canonical_json(manifest).encode("utf-8")
    manifest_hash = sha256(manifest_bytes).hexdigest()
    manifest_ref = store.put_immutable(
        manifest_bytes,
        candidate_id=candidate.snapshot.snapshot_id,
        object_kind="final_manifest",
    )
    return AuditExportSnapshotCandidate(
        snapshot=replace(
            candidate.snapshot,
            last_chunk_seal_hash=forged_seal,
            signed_manifest_hash=manifest_hash,
            signed_manifest_ref=manifest_ref,
            signed_manifest_size_bytes=len(manifest_bytes),
        ),
        chunks=(*candidate.chunks[:-1], replace(candidate.chunks[-1], chunk_seal_hash=forged_seal)),
    )


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
            first = _register_candidate(connection, candidate, store)
        with db.engine.begin() as connection:
            second = _register_candidate(connection, candidate, store)

        assert first.inserted is True
        assert second.inserted is False
        assert first.winner == second.winner
        with db.engine.connect() as connection:
            assert connection.scalar(select(func.count()).select_from(audit_export_snapshots_table)) == 1
            assert connection.scalar(select(func.count()).select_from(audit_export_snapshot_chunks_table)) == 1
    finally:
        db.close()


def test_registry_cas_reuses_byte_identical_winner_from_prior_content_store() -> None:
    db = LandscapeDB.in_memory()
    old_store = _MemoryContentStore("durable-store-v1")
    new_store = _MemoryContentStore("durable-store-v2")
    resolver = AuditExportContentStoreResolver()
    resolver.register(old_store)
    resolver.register(new_store)
    old_candidate = _candidate(old_store)
    new_candidate = _candidate(new_store)
    repository = _repository()
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            first = repository.register_candidate(
                connection,
                old_candidate,
                content_store_resolver=resolver,
                limits=AuditExportSnapshotReadLimits(),
                signed_manifest_verifier=lambda _content, _descriptor: None,
            )
        with db.engine.begin() as connection:
            second = repository.register_candidate(
                connection,
                new_candidate,
                content_store_resolver=resolver,
                limits=AuditExportSnapshotReadLimits(),
                signed_manifest_verifier=lambda _content, _descriptor: None,
            )

        assert first.inserted is True
        assert second.inserted is False
        assert second.winner.snapshot.content_store_id == old_store.content_store_id
        assert second.winner.snapshot.snapshot_id == new_candidate.snapshot.snapshot_id
    finally:
        db.close()


def test_registry_lookup_exact_checks_every_shaping_field_and_signer_id() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    candidate = _candidate(store)
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            _register_candidate(connection, candidate, store)
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
    divergent = _candidate(store, records=[{"record_type": "run", "value": "divergent"}])
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            _register_candidate(connection, first, store)
        with db.engine.begin() as connection, pytest.raises(AuditIntegrityError, match="divergent"):
            _register_candidate(connection, divergent, store)
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
            winner = _register_candidate(connection, candidate, store).winner

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


def test_bound_winner_rejects_chunk_seal_not_derived_from_registered_bytes() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    forged = _forge_chunk_seal(store, _candidate(store))
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            _insert_unverified_candidate(connection, forged)
            winner = _repository().find_winner(connection, AuditExportSnapshotRegistryKey.from_snapshot(forged.snapshot))
            assert winner is not None

        with pytest.raises(AuditIntegrityError, match="chunk seal"):
            _repository().bind_winner(
                winner,
                content_store_resolver=resolver,
                limits=AuditExportSnapshotReadLimits(),
                signed_manifest_verifier=lambda _content, _descriptor: None,
            )
    finally:
        db.close()


def test_verified_candidate_carrier_cannot_be_hand_built() -> None:
    store = _MemoryContentStore()
    candidate = _candidate(store)
    with pytest.raises(TypeError, match="verify_candidate"):
        VerifiedAuditExportCandidate(candidate=candidate)
    with pytest.raises(TypeError, match="verify_candidate"):
        VerifiedAuditExportCandidate(candidate=candidate, _proof=object())


def test_register_verified_candidate_registers_without_content_store_reads() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    candidate = _candidate(store)
    repository = _repository()
    try:
        _insert_terminal_run(db)
        verified = repository.verify_candidate(
            candidate,
            content_store_resolver=resolver,
            limits=AuditExportSnapshotReadLimits(),
            signed_manifest_verifier=lambda _content, _descriptor: None,
        )
        reads_after_verification = len(store.opened)

        with db.engine.begin() as connection:
            registration = repository.register_verified_candidate(connection, verified)

        assert registration.inserted is True
        assert len(store.opened) == reads_after_verification
        with pytest.raises(TypeError, match="VerifiedAuditExportCandidate"), db.engine.begin() as connection:
            repository.register_verified_candidate(connection, candidate)  # type: ignore[arg-type]
    finally:
        db.close()


def test_registration_rejects_forged_graph_before_registry_insert() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    forged = _forge_chunk_seal(store, _candidate(store))
    try:
        _insert_terminal_run(db)
        with pytest.raises(AuditIntegrityError, match="chunk seal"), db.engine.begin() as connection:
            _repository().register_candidate(
                connection,
                forged,
                content_store_resolver=resolver,
                limits=AuditExportSnapshotReadLimits(),
                signed_manifest_verifier=lambda _content, _descriptor: None,
            )
        with db.engine.connect() as connection:
            assert connection.scalar(select(func.count()).select_from(audit_export_snapshots_table)) == 0
            assert connection.scalar(select(func.count()).select_from(audit_export_snapshot_chunks_table)) == 0
    finally:
        db.close()


@pytest.mark.parametrize(
    ("field_name", "message"),
    (
        ("public_export_config_hash", "registry_key_hash"),
        ("registry_key_hash", "registry_key_hash"),
        ("snapshot_hash", "snapshot_hash"),
        ("snapshot_id", "snapshot_id"),
        ("manifest_hash", "manifest_hash"),
        ("snapshot_seal_hash", "snapshot_seal_hash"),
        ("final_hash", "final_hash"),
    ),
)
def test_bound_winner_recomputes_persisted_graph_hashes_from_registered_bytes(field_name: str, message: str) -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    candidate = _candidate(store)
    observed = getattr(candidate.snapshot, field_name)
    forged_hash = "f" * 64 if observed != "f" * 64 else "e" * 64
    if field_name == "public_export_config_hash":
        forged_snapshot = replace(candidate.snapshot, public_export_config_hash=forged_hash)
    elif field_name == "registry_key_hash":
        forged_snapshot = replace(candidate.snapshot, registry_key_hash=forged_hash)
    elif field_name == "snapshot_hash":
        forged_snapshot = replace(candidate.snapshot, snapshot_hash=forged_hash)
    elif field_name == "snapshot_id":
        forged_snapshot = replace(candidate.snapshot, snapshot_id=forged_hash)
    elif field_name == "manifest_hash":
        forged_snapshot = replace(candidate.snapshot, manifest_hash=forged_hash)
    elif field_name == "snapshot_seal_hash":
        forged_snapshot = replace(candidate.snapshot, snapshot_seal_hash=forged_hash)
    else:
        assert field_name == "final_hash"
        forged_snapshot = replace(candidate.snapshot, final_hash=forged_hash)
    forged_chunks = candidate.chunks
    if field_name == "snapshot_id":
        forged_chunks = tuple(replace(chunk, snapshot_id=forged_hash) for chunk in candidate.chunks)
    forged = AuditExportSnapshotCandidate(snapshot=forged_snapshot, chunks=forged_chunks)
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            _insert_unverified_candidate(connection, forged)
            winner = _repository().find_winner(connection, AuditExportSnapshotRegistryKey.from_snapshot(forged.snapshot))
            assert winner is not None

        with pytest.raises(AuditIntegrityError, match=message):
            _repository().bind_winner(
                winner,
                content_store_resolver=resolver,
                limits=AuditExportSnapshotReadLimits(),
                signed_manifest_verifier=lambda _content, _descriptor: None,
            )
    finally:
        db.close()


def test_bound_winner_rejects_noncanonical_registered_record_bytes() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    candidate = _candidate(store)
    noncanonical = b'{"record_type": "run"}\n'
    content_ref = store.put_immutable(noncanonical, candidate_id="forged", object_kind="data_chunk")
    chunk = replace(
        candidate.chunks[0],
        content_ref=content_ref,
        content_hash=sha256(noncanonical).hexdigest(),
        size_bytes=len(noncanonical),
        cumulative_bytes=len(noncanonical),
    )
    forged = AuditExportSnapshotCandidate(
        snapshot=replace(candidate.snapshot, total_bytes=len(noncanonical)),
        chunks=(chunk,),
    )
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            _insert_unverified_candidate(connection, forged)
            winner = _repository().find_winner(connection, AuditExportSnapshotRegistryKey.from_snapshot(forged.snapshot))
            assert winner is not None

        with pytest.raises(AuditIntegrityError, match="non-canonical record bytes"):
            _repository().bind_winner(
                winner,
                content_store_resolver=resolver,
                limits=AuditExportSnapshotReadLimits(),
                signed_manifest_verifier=lambda _content, _descriptor: None,
            )
    finally:
        db.close()


def test_bound_winner_rejects_record_hmac_not_derived_from_unsigned_record_bytes() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    candidate = _candidate(store, signed=True)
    emitted = json.loads(store.content[candidate.chunks[0].content_ref])
    emitted["signature"] = "0" * 64
    forged_content = canonical_json(emitted).encode("utf-8") + b"\n"
    assert len(forged_content) == candidate.chunks[0].size_bytes
    content_ref = store.put_immutable(forged_content, candidate_id="forged", object_kind="data_chunk")
    chunk = replace(
        candidate.chunks[0],
        content_ref=content_ref,
        content_hash=sha256(forged_content).hexdigest(),
    )
    forged = AuditExportSnapshotCandidate(snapshot=candidate.snapshot, chunks=(chunk,))
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            _insert_unverified_candidate(connection, forged)
            winner = _repository().find_winner(connection, AuditExportSnapshotRegistryKey.from_snapshot(forged.snapshot))
            assert winner is not None

        with pytest.raises(AuditIntegrityError, match="record HMAC"):
            _repository().bind_winner(
                winner,
                content_store_resolver=resolver,
                limits=AuditExportSnapshotReadLimits(),
                signed_manifest_verifier=_signed_manifest_verifier,
                record_signature_verifier=_record_signature_verifier,
            )
    finally:
        db.close()


def test_bound_winner_rejects_final_manifest_bytes_not_derived_from_graph() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    candidate = _candidate(store)
    original = store.content[candidate.snapshot.signed_manifest_ref]
    forged = original.replace(b'"hash_algorithm":"sha256"', b'"hash_algorithm":"sha257"')
    assert forged != original and len(forged) == len(original)
    store.content[candidate.snapshot.signed_manifest_ref] = forged
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            _insert_unverified_candidate(connection, candidate)
            winner = _repository().find_winner(connection, AuditExportSnapshotRegistryKey.from_snapshot(candidate.snapshot))
            assert winner is not None

        with pytest.raises(AuditIntegrityError, match="signed manifest bytes"):
            _repository().bind_winner(
                winner,
                content_store_resolver=resolver,
                limits=AuditExportSnapshotReadLimits(),
                signed_manifest_verifier=lambda _content, _descriptor: None,
            )
    finally:
        db.close()


def test_bound_winner_rejects_final_manifest_signature_metadata_not_derived_from_bytes() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    candidate = _candidate(store, signed=True)
    assert candidate.snapshot.signature_hex != "f" * 64
    forged = AuditExportSnapshotCandidate(
        snapshot=replace(candidate.snapshot, signature_hex="f" * 64),
        chunks=candidate.chunks,
    )
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            _insert_unverified_candidate(connection, forged)
            winner = _repository().find_winner(connection, AuditExportSnapshotRegistryKey.from_snapshot(forged.snapshot))
            assert winner is not None

        with pytest.raises(AuditIntegrityError, match="signed manifest bytes"):
            _repository().bind_winner(
                winner,
                content_store_resolver=resolver,
                limits=AuditExportSnapshotReadLimits(),
                signed_manifest_verifier=_signed_manifest_verifier,
                record_signature_verifier=_record_signature_verifier,
            )
    finally:
        db.close()


def test_winner_store_resolution_failure_never_substitutes_current_store() -> None:
    db = LandscapeDB.in_memory()
    store = _MemoryContentStore()
    candidate = _candidate(store)
    try:
        _insert_terminal_run(db)
        with db.engine.begin() as connection:
            winner = _register_candidate(connection, candidate, store).winner
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
            _register_candidate(connection, _candidate(store), store)
        with pytest.raises(IntegrityError), db.engine.begin() as connection:
            connection.execute(update(audit_export_snapshots_table).values(content_store_id="replacement"))
        with pytest.raises(IntegrityError), db.engine.begin() as connection:
            connection.execute(update(audit_export_snapshot_chunks_table).values(size_bytes=999))
    finally:
        db.close()
