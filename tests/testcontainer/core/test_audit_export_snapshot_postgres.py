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
from elspeth.contracts.enums import RunStatus
from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.sink_effects import AuditExportFormat, AuditExportSigningMode
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution.audit_export_snapshots import (
    AuditExportSnapshotCandidate,
    AuditExportSnapshotRepository,
)
from elspeth.core.landscape.export_read_model import open_export_read_transaction
from elspeth.core.landscape.schema import audit_export_snapshots_table, run_attributions_table, runs_table

pytestmark = pytest.mark.testcontainer
COMPLETED_AT = datetime(2026, 7, 16, 3, 4, 5, 678901, tzinfo=UTC)
COMPLETED_AT_TEXT = "2026-07-16T03:04:05.678901Z"


@pytest.fixture(scope="module")
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


def _candidate() -> AuditExportSnapshotCandidate:
    snapshot_id = "2" * 64
    chunk_bytes = b'{"record_type":"run"}\n'
    chunk_hash = sha256(chunk_bytes).hexdigest()
    manifest_hash = "7" * 64
    snapshot_hash = "8" * 64
    snapshot_seal_hash = "9" * 64
    chunk_seal_hash = "4" * 64
    registry_key_hash = "5" * 64
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
        "registry_key_hash": registry_key_hash,
        "run_id": "pg-export",
        "schema": "elspeth.audit-export-manifest.v2",
        "signature": None,
        "signature_algorithm": "unsigned",
        "signature_key_id": "UNSIGNED",
        "snapshot_hash": snapshot_hash,
        "snapshot_id": snapshot_id,
        "snapshot_seal_hash": snapshot_seal_hash,
        "source_completed_at": COMPLETED_AT_TEXT,
        "source_status": "completed",
        "total_bytes": len(chunk_bytes),
    }
    manifest_bytes = canonical_json(manifest).encode()
    snapshot = AuditExportSnapshot(
        snapshot_id=snapshot_id,
        source_run_id="pg-export",
        source_status=RunStatus.COMPLETED,
        source_completed_at=COMPLETED_AT,
        exported_at=COMPLETED_AT,
        registry_key_hash=registry_key_hash,
        exporter_version="landscape-exporter-v2",
        serialization_version="audit-export-v2",
        export_format=AuditExportFormat.JSON,
        signing_mode=AuditExportSigningMode.UNSIGNED,
        signer_key_id="UNSIGNED",
        derivation_version="audit-export-derivation-v1",
        public_export_config_hash="6" * 64,
        chunking_algorithm_version="complete-frame-v1",
        per_chunk_record_limit=100,
        per_chunk_byte_limit=1024,
        record_count=1,
        total_bytes=len(chunk_bytes),
        chunk_count=1,
        terminal_chunk_ordinal=0,
        content_store_id="durable-store",
        manifest_hash=manifest_hash,
        last_chunk_seal_hash=chunk_seal_hash,
        snapshot_hash=snapshot_hash,
        snapshot_seal_hash=snapshot_seal_hash,
        signature_hex=None,
        record_chain_algorithm="sha256_concat_record_sha256_v1",
        final_hash=final_hash,
        signed_manifest_schema="elspeth.audit-export-manifest.v2",
        signed_manifest_hash=sha256(manifest_bytes).hexdigest(),
        signed_manifest_ref=f"sha256:{sha256(manifest_bytes).hexdigest()}",
        signed_manifest_size_bytes=len(manifest_bytes),
    )
    chunk = AuditExportSnapshotChunk(
        snapshot_id=snapshot_id,
        ordinal=0,
        content_ref=f"sha256:{chunk_hash}",
        content_hash=chunk_hash,
        size_bytes=len(chunk_bytes),
        record_count=1,
        predecessor_seal_hash=None,
        cumulative_records=1,
        cumulative_bytes=len(chunk_bytes),
        chunk_seal_hash=chunk_seal_hash,
    )
    return AuditExportSnapshotCandidate(snapshot=snapshot, chunks=(chunk,))


def test_postgres_concurrent_registry_cas_uses_distinct_backends_and_one_winner(postgres_db: LandscapeDB) -> None:
    _seed_run(postgres_db)
    candidate = _candidate()
    barrier = threading.Barrier(2)
    pids: list[int] = []

    def contender() -> tuple[bool, str]:
        with postgres_db.engine.begin() as connection:
            pids.append(int(connection.exec_driver_sql("SELECT pg_backend_pid()").scalar_one()))
            barrier.wait(timeout=20)
            registration = AuditExportSnapshotRepository().register_candidate(connection, candidate)
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
