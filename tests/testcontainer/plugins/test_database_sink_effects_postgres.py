"""Real PostgreSQL proof for atomic Database sink effect markers."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256

import pytest
from sqlalchemy import Column, Integer, MetaData, Table, Text, create_engine, func, insert, select
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]
from tests.fixtures.base_classes import inject_write_failure

from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.sink_effects import (
    RestrictedSinkEffectContext,
    SinkEffectInspectionRequest,
    SinkEffectMember,
    SinkEffectPipelineMembersInput,
    SinkEffectPrepareRequest,
    SinkEffectReconcileKind,
)
from elspeth.plugins.sinks.database_sink import DatabaseSink, database_effect_ledger_table

pytestmark = pytest.mark.testcontainer

_CTX = RestrictedSinkEffectContext(
    run_id="run-postgres-effect",
    run_started_at=datetime(2026, 7, 16, tzinfo=UTC),
    operation_id="operation-postgres-effect",
    sink_node_id="sink-postgres-effect",
)


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


def _member(ordinal: int, row: dict[str, object]) -> SinkEffectMember:
    payload = canonical_json(row).encode()
    return SinkEffectMember(
        ordinal=ordinal,
        token_id=f"token-pg-{ordinal}",
        row_id=f"row-pg-{ordinal}",
        ingest_sequence=ordinal,
        lineage_json="[]",
        lineage_hash=sha256(b"[]").hexdigest(),
        payload_hash=sha256(payload).hexdigest(),
        row=row,
        member_effect_id=sha256(f"member-pg-{ordinal}".encode()).hexdigest(),
    )


def test_postgres_marker_rows_and_result_partition_commit_atomically(postgres_url: str) -> None:
    engine = create_engine(postgres_url)
    metadata = MetaData()
    target = Table(
        "database_effect_output",
        metadata,
        Column("id", Integer, nullable=False, unique=True),
        Column("name", Text, nullable=False),
    )
    ledger = database_effect_ledger_table(metadata, "_elspeth_sink_effects")
    metadata.drop_all(engine, checkfirst=True)
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(insert(target), [{"id": 2, "name": "existing"}])

    config = {
        "url": postgres_url,
        "table": "database_effect_output",
        "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
        "if_exists": "append",
        "effect_ledger": {
            "table": "_elspeth_sink_effects",
            "schema_version": 1,
            "permissions": ["select", "insert"],
        },
    }
    sink = inject_write_failure(DatabaseSink(config))
    members = tuple(
        _member(ordinal, row)
        for ordinal, row in enumerate(({"id": 1, "name": "one"}, {"id": 2, "name": "duplicate"}, {"id": 3, "name": "three"}))
    )
    inspection = sink.inspect_effect(
        SinkEffectInspectionRequest(effect_id="d" * 64, target="{}", predecessor_descriptor=None),
        _CTX,
    )
    plan = sink.prepare_effect(
        SinkEffectPrepareRequest(
            effect_id="d" * 64,
            effect_input=SinkEffectPipelineMembersInput(members=members, target_snapshot_members=members),
            inspection=inspection,
        ),
        _CTX,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(inject_write_failure(DatabaseSink(config)).commit_effect, plan, _CTX) for _ in range(2)]
        committed, concurrent_replay = (future.result(timeout=10) for future in futures)
    recovered = inject_write_failure(DatabaseSink(config)).reconcile_effect(plan, _CTX)

    assert concurrent_replay == committed
    assert committed.accepted_ordinals == (0, 2)
    assert committed.diverted_ordinals == (1,)
    assert recovered.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR
    assert recovered.descriptor == committed.descriptor
    assert recovered.accepted_ordinals == committed.accepted_ordinals
    assert recovered.diverted_ordinals == committed.diverted_ordinals
    with engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(target)) == 3
        assert conn.scalar(select(func.count()).select_from(ledger)) == 1
    engine.dispose()
