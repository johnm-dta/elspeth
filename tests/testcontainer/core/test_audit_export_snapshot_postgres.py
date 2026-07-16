"""Real PostgreSQL snapshot CAS and repeatable-read proof."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256

import pytest
from sqlalchemy import func, select
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from elspeth.contracts.audit import AuditExportSnapshot, AuditExportSnapshotChunk
from elspeth.contracts.audit_export import (
    AuditExportContentDescriptor,
    AuditExportContentStoreResolver,
    AuditExportDerivationConfig,
    IterableBoundAuditExportContentReader,
    RegisteredAuditExportContent,
    derive_audit_export_bundle,
)
from elspeth.contracts.enums import RunStatus
from elspeth.contracts.sink_effects import AuditExportFormat, AuditExportSigningMode
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution.audit_export_snapshots import (
    AuditExportSnapshotCandidate,
    AuditExportSnapshotReadLimits,
    AuditExportSnapshotRepository,
)
from elspeth.core.landscape.export_read_model import open_export_read_transaction
from elspeth.core.landscape.schema import audit_export_snapshots_table, run_attributions_table, runs_table

pytestmark = pytest.mark.testcontainer
COMPLETED_AT = datetime(2026, 7, 16, 3, 4, 5, 678901, tzinfo=UTC)
COMPLETED_AT_TEXT = "2026-07-16T03:04:05.678901Z"


@pytest.fixture
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


@pytest.fixture
def postgres_db(postgres_url: str) -> Iterator[LandscapeDB]:
    db = LandscapeDB(postgres_url)
    try:
        yield db
    finally:
        db.close()


def _seed_run(db: LandscapeDB) -> None:
    with db.engine.begin() as connection:
        connection.execute(
            runs_table.insert().values(
                run_id="pg-export",
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


class _MemoryContentStore:
    content_store_id = "durable-store"
    namespace = "audit/export"

    def __init__(self) -> None:
        self.content: dict[str, bytes] = {}

    def is_durable(self) -> bool:
        return True

    def put_immutable(self, content: bytes, *, candidate_id: str, object_kind: str) -> str:
        del candidate_id, object_kind
        content_ref = f"sha256:{sha256(content).hexdigest()}"
        self.content.setdefault(content_ref, content)
        return content_ref

    def open_registered(self, registration: RegisteredAuditExportContent) -> IterableBoundAuditExportContentReader:
        return IterableBoundAuditExportContentReader(self.content[registration.descriptor.content_ref])

    def mark_candidate_orphans(
        self,
        candidate_id: str,
        descriptors: tuple[AuditExportContentDescriptor, ...],
    ) -> None:
        del candidate_id, descriptors

    def garbage_collect_candidate(self, request: object) -> bool:
        del request
        return False


def _candidate(store: _MemoryContentStore) -> AuditExportSnapshotCandidate:
    bundle = derive_audit_export_bundle(
        [{"record_type": "run"}],
        AuditExportDerivationConfig(
            source_run_id="pg-export",
            source_status="completed",
            source_completed_at=COMPLETED_AT_TEXT,
            export_format="json",
            exporter_version="landscape-exporter-v2",
            serialization_version="audit-export-v2",
            chunking_algorithm_version="complete-frame-v1",
            include_raw_error_rows=False,
            per_chunk_record_limit=100,
            per_chunk_byte_limit=1024,
            signing_mode="unsigned",
            signer_key_id="UNSIGNED",
            signing_key=None,
        ),
    )
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
        source_run_id="pg-export",
        source_status=RunStatus.COMPLETED,
        source_completed_at=COMPLETED_AT,
        exported_at=COMPLETED_AT,
        registry_key_hash=bundle.registry_key_hash,
        exporter_version="landscape-exporter-v2",
        serialization_version="audit-export-v2",
        export_format=AuditExportFormat.JSON,
        signing_mode=AuditExportSigningMode.UNSIGNED,
        signer_key_id="UNSIGNED",
        derivation_version="audit-export-derivation-v1",
        public_export_config_hash=bundle.public_export_config_hash,
        chunking_algorithm_version="complete-frame-v1",
        per_chunk_record_limit=100,
        per_chunk_byte_limit=1024,
        record_count=1,
        total_bytes=sum(chunk.descriptor.size_bytes for chunk in bundle.chunks),
        chunk_count=1,
        terminal_chunk_ordinal=0,
        content_store_id="durable-store",
        manifest_hash=bundle.manifest_hash,
        last_chunk_seal_hash=bundle.last_chunk_seal_hash,
        snapshot_hash=bundle.snapshot_hash,
        snapshot_seal_hash=bundle.snapshot_seal_hash,
        signature_hex=None,
        record_chain_algorithm="sha256_concat_record_sha256_v1",
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


def test_postgres_concurrent_registry_cas_uses_distinct_backends_and_one_winner(postgres_db: LandscapeDB) -> None:
    _seed_run(postgres_db)
    store = _MemoryContentStore()
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    candidate = _candidate(store)
    barrier = threading.Barrier(2)
    pids: list[int] = []

    def contender() -> tuple[bool, str]:
        with postgres_db.engine.begin() as connection:
            pids.append(int(connection.exec_driver_sql("SELECT pg_backend_pid()").scalar_one()))
            barrier.wait(timeout=20)
            registration = AuditExportSnapshotRepository().register_candidate(
                connection,
                candidate,
                content_store_resolver=resolver,
                limits=AuditExportSnapshotReadLimits(),
                signed_manifest_verifier=lambda _content, _descriptor: None,
            )
            return registration.inserted, registration.winner.snapshot.snapshot_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=30) for future in (pool.submit(contender), pool.submit(contender))]

    assert len(set(pids)) == 2
    assert sorted(inserted for inserted, _snapshot_id in results) == [False, True]
    assert {snapshot_id for _inserted, snapshot_id in results} == {candidate.snapshot.snapshot_id}
    with postgres_db.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(audit_export_snapshots_table)) == 1


def test_postgres_export_read_model_is_repeatable_read_and_excludes_later_insert(postgres_db: LandscapeDB) -> None:
    _seed_run(postgres_db)
    with open_export_read_transaction(postgres_db.engine) as model:
        reader_pid = int(model.connection.exec_driver_sql("SELECT pg_backend_pid()").scalar_one())
        isolation = str(model.connection.exec_driver_sql("SHOW transaction_isolation").scalar_one())
        assert isolation == "repeatable read"
        assert model.get_export_terminal_witness("pg-export").source_completed_at == COMPLETED_AT
        assert model.get_run_attribution("pg-export") is None
        with postgres_db.engine.begin() as writer:
            writer_pid = int(writer.exec_driver_sql("SELECT pg_backend_pid()").scalar_one())
            writer.execute(
                run_attributions_table.insert().values(
                    run_id="pg-export",
                    recorded_at=COMPLETED_AT,
                    initiated_by_user_id="later-user",
                    auth_provider_type="local",
                )
            )
        assert reader_pid != writer_pid
        assert model.get_run_attribution("pg-export") is None
