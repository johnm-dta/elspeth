"""Real PostgreSQL proofs for run-scoped audit foreign keys."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]
from tests.fixtures.landscape import make_factory, register_test_node

from elspeth.contracts import NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import token_parents_table, validation_errors_table

pytestmark = pytest.mark.testcontainer


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


@pytest.fixture
def postgres_db(postgres_url: str) -> Iterator[LandscapeDB]:
    db = LandscapeDB.from_url(postgres_url)
    try:
        yield db
    finally:
        db.close()


@pytest.mark.timeout(120)
def test_postgres_rejects_cross_run_token_parent(postgres_db: LandscapeDB) -> None:
    factory = make_factory(postgres_db)
    child_run = factory.run_lifecycle.begin_run(
        config={},
        canonical_version="v1",
        run_id="token-parent-child-run",
        openrouter_catalog_sha256="0" * 64,
        openrouter_catalog_source="bundled",
    )
    child_source = register_test_node(
        factory.data_flow,
        child_run.run_id,
        "token-parent-child-source",
        node_type=NodeType.SOURCE,
        plugin_name="source",
    )
    child_row = factory.data_flow.create_row(
        run_id=child_run.run_id,
        source_node_id=child_source,
        row_index=0,
        source_row_index=0,
        ingest_sequence=0,
        data={"side": "child"},
    )
    child = factory.data_flow.create_token(child_row.row_id)

    parent_run = factory.run_lifecycle.begin_run(
        config={},
        canonical_version="v1",
        run_id="token-parent-parent-run",
        openrouter_catalog_sha256="0" * 64,
        openrouter_catalog_source="bundled",
    )
    parent_source = register_test_node(
        factory.data_flow,
        parent_run.run_id,
        "token-parent-parent-source",
        node_type=NodeType.SOURCE,
        plugin_name="source",
    )
    parent_row = factory.data_flow.create_row(
        run_id=parent_run.run_id,
        source_node_id=parent_source,
        row_index=0,
        source_row_index=0,
        ingest_sequence=0,
        data={"side": "parent"},
    )
    parent = factory.data_flow.create_token(parent_row.row_id)

    values: dict[str, object] = {
        "token_id": child.token_id,
        "parent_token_id": parent.token_id,
        "ordinal": 0,
    }
    if "run_id" in token_parents_table.c:
        values["run_id"] = child_run.run_id

    with pytest.raises(IntegrityError), postgres_db.write_connection() as conn:
        conn.execute(token_parents_table.insert().values(**values))


def _seed_cross_run_validation_row(factory: RecorderFactory, *, suffix: str) -> tuple[str, str, str]:
    run_a = f"validation-{suffix}-run-a"
    run_b = f"validation-{suffix}-run-b"
    for run_id in (run_a, run_b):
        factory.run_lifecycle.begin_run(
            config={},
            canonical_version="v1",
            run_id=run_id,
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
        )
    node_a = register_test_node(
        factory.data_flow,
        run_a,
        f"validation-{suffix}-node-a",
        node_type=NodeType.SOURCE,
        plugin_name="source",
    )
    node_b = register_test_node(
        factory.data_flow,
        run_b,
        f"validation-{suffix}-node-b",
        node_type=NodeType.SOURCE,
        plugin_name="source",
    )
    row_b = factory.data_flow.create_row(
        run_id=run_b,
        source_node_id=node_b,
        row_index=0,
        source_row_index=0,
        ingest_sequence=0,
        data={"invalid": True},
    )
    return run_a, node_a, row_b.row_id


@pytest.mark.timeout(120)
def test_postgres_rejects_raw_cross_run_validation_error_row_link(postgres_db: LandscapeDB) -> None:
    factory = make_factory(postgres_db)
    run_a, node_a, row_b = _seed_cross_run_validation_row(factory, suffix="raw")
    error_id = "verr_pg_cross_run_raw"

    with pytest.raises(IntegrityError), postgres_db.write_connection() as conn:
        conn.execute(
            validation_errors_table.insert().values(
                error_id=error_id,
                run_id=run_a,
                node_id=node_a,
                row_id=row_b,
                row_hash="0" * 64,
                row_data_json="{}",
                error="invalid row",
                schema_mode="fixed",
                destination="discard",
                created_at=datetime.now(UTC),
            )
        )

    with postgres_db.read_only_connection() as conn:
        assert (
            conn.execute(select(validation_errors_table.c.error_id).where(validation_errors_table.c.error_id == error_id)).fetchone()
            is None
        )


@pytest.mark.timeout(120)
def test_postgres_public_writer_rejects_cross_run_validation_error_row_link(postgres_db: LandscapeDB) -> None:
    factory = make_factory(postgres_db)
    run_a, node_a, row_b = _seed_cross_run_validation_row(factory, suffix="writer")

    with pytest.raises(AuditIntegrityError, match="cross-run contamination"):
        factory.data_flow.record_validation_error(
            run_id=run_a,
            node_id=node_a,
            row_id=row_b,
            row_data={"invalid": True},
            error="invalid row",
            schema_mode="fixed",
            destination="discard",
        )

    with postgres_db.read_only_connection() as conn:
        assert conn.execute(select(validation_errors_table.c.error_id).where(validation_errors_table.c.run_id == run_a)).fetchone() is None


@pytest.mark.timeout(120)
def test_postgres_validation_error_link_is_compare_and_set(
    postgres_db: LandscapeDB,
    postgres_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent same-run linkers cannot silently overwrite row lineage."""
    first_factory = make_factory(postgres_db)
    run_id = "validation-link-cas-run"
    first_factory.run_lifecycle.begin_run(
        config={},
        canonical_version="v1",
        run_id=run_id,
        openrouter_catalog_sha256="0" * 64,
        openrouter_catalog_source="bundled",
    )
    node_id = register_test_node(
        first_factory.data_flow,
        run_id,
        "validation-link-cas-node",
        node_type=NodeType.SOURCE,
        plugin_name="source",
    )
    rows = [
        first_factory.data_flow.create_row(
            run_id=run_id,
            source_node_id=node_id,
            row_index=index,
            source_row_index=index,
            ingest_sequence=index,
            data={"candidate": index},
        )
        for index in range(2)
    ]
    error_id = first_factory.data_flow.record_validation_error(
        run_id=run_id,
        node_id=node_id,
        row_data={"invalid": True},
        error="invalid row",
        schema_mode="fixed",
        destination="discard",
    )

    second_db = LandscapeDB.from_url(postgres_url)
    second_factory = make_factory(second_db)
    old_read_barrier = threading.Barrier(2)

    # Deterministically reproduces the old split read/update implementation:
    # both workers observe validation_errors.row_id=NULL before either writes.
    for factory in (first_factory, second_factory):
        ops = factory.data_flow.errors._ops
        original_fetchone = ops.execute_fetchone

        def synchronized_fetchone(query: Any, *, _original: Any = original_fetchone) -> Any:
            result = _original(query)
            selected = tuple(query.selected_columns.keys())
            if selected == ("run_id", "row_id"):
                old_read_barrier.wait(timeout=5)
            return result

        monkeypatch.setattr(ops, "execute_fetchone", synchronized_fetchone)

    def link(candidate: tuple[RecorderFactory, str]) -> str:
        factory, row_id = candidate
        try:
            factory.data_flow.link_validation_error_to_row(run_id=run_id, error_id=error_id, row_id=row_id)
        except AuditIntegrityError as exc:
            assert "already linked to row" in str(exc)
            return "rejected"
        return "committed"

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(link, ((first_factory, rows[0].row_id), (second_factory, rows[1].row_id))))
    finally:
        second_db.close()

    assert sorted(results) == ["committed", "rejected"]
    with postgres_db.read_only_connection() as conn:
        linked_row_id = conn.execute(
            select(validation_errors_table.c.row_id).where(validation_errors_table.c.error_id == error_id)
        ).scalar_one()
    assert linked_row_id in {rows[0].row_id, rows[1].row_id}
