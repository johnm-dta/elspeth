"""Real PostgreSQL contention proof for node output-contract evolution."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest
from sqlalchemy import select
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]
from tests.fixtures.landscape import make_factory

from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import nodes_table

pytestmark = pytest.mark.testcontainer

_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


def _contract(name: str, python_type: type) -> SchemaContract:
    return SchemaContract(
        mode="OBSERVED",
        fields=(
            FieldContract(
                normalized_name=name,
                original_name=name,
                python_type=python_type,
                required=True,
                source="inferred",
            ),
        ),
        locked=True,
    )


@pytest.mark.timeout(120)
def test_concurrent_postgres_writers_lock_and_merge_disjoint_contracts(postgres_url: str) -> None:
    first_db = LandscapeDB.from_url(postgres_url)
    second_db = LandscapeDB.from_url(postgres_url)
    try:
        first = make_factory(first_db)
        second = make_factory(second_db)
        first.run_lifecycle.begin_run(
            config={},
            canonical_version="v1",
            run_id="run-1",
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
        )
        first.data_flow.register_node(
            run_id="run-1",
            plugin_name="mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm",
            output_contract=_contract("base", int),
            schema_config=_SCHEMA,
        )
        barrier = threading.Barrier(2)
        failures: list[BaseException] = []

        def update(factory: RecorderFactory, contract: SchemaContract) -> None:
            try:
                barrier.wait(timeout=10)
                factory.data_flow.update_node_output_contract("run-1", "xfm", contract)
            except BaseException as exc:  # pragma: no cover - asserted below
                failures.append(exc)

        threads = (
            threading.Thread(target=update, args=(first, _contract("left", str))),
            threading.Thread(target=update, args=(second, _contract("right", float))),
        )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not any(thread.is_alive() for thread in threads)
        assert failures == []
        _, stored = first.data_flow.get_node_contracts("run-1", "xfm")
        assert stored is not None
        assert {field.normalized_name for field in stored.fields} == {"base", "left", "right"}
        with first_db.read_only_connection() as conn:
            stored_hash = conn.execute(
                select(nodes_table.c.output_contract_hash).where((nodes_table.c.run_id == "run-1") & (nodes_table.c.node_id == "xfm"))
            ).scalar_one()
        assert stored_hash == stored.version_hash()
    finally:
        second_db.close()
        first_db.close()
