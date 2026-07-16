"""Real PostgreSQL proofs for durable coalesce-effect contention."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import Connection
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]
from tests.fixtures.landscape import register_test_node

from elspeth.contracts import NodeType
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import coalesce_effects_table, token_parents_table, tokens_table
from elspeth.core.payload_store import FilesystemPayloadStore

pytestmark = pytest.mark.testcontainer

_CONTRACT = SchemaContract(mode="OBSERVED", fields=(), locked=True)


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


@pytest.mark.timeout(120)
def test_identical_coalesce_writers_serialize_on_parents_and_reuse_one_effect(
    postgres_url: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loser blocks on parent authority, then reads the winner's receipt."""
    first_db = LandscapeDB.from_url(postgres_url)
    second_db = LandscapeDB.from_url(postgres_url)
    payload_root = tmp_path / "payloads"
    first = RecorderFactory(first_db, payload_store=FilesystemPayloadStore(payload_root))
    second = RecorderFactory(second_db, payload_store=FilesystemPayloadStore(payload_root))
    run = first.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source_id = register_test_node(first.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
    coalesce_id = register_test_node(
        first.data_flow,
        run.run_id,
        "coalesce",
        node_type=NodeType.COALESCE,
        plugin_name="coalesce",
    )
    row = first.data_flow.create_row(
        run_id=run.run_id,
        source_node_id=source_id,
        row_index=0,
        source_row_index=0,
        ingest_sequence=0,
        data={"source": True},
    )
    parents = [first.data_flow.create_token(row.row_id) for _ in range(2)]
    refs = tuple(TokenRef(token_id=token.token_id, run_id=run.run_id) for token in parents)
    state_ids = tuple(
        first.execution.begin_node_state(
            token_id=ref.token_id,
            node_id=coalesce_id,
            run_id=run.run_id,
            step_index=4,
            input_data={"ordinal": ordinal},
        ).state_id
        for ordinal, ref in enumerate(refs)
    )

    winner_has_authority = threading.Event()
    loser_attempting_authority = threading.Event()
    loser_has_authority = threading.Event()
    release_winner = threading.Event()
    backend_pids: dict[str, int] = {}
    original_first_lock = first.data_flow.tokens._lock_coalesce_dependencies
    original_second_lock = second.data_flow.tokens._lock_coalesce_dependencies

    def pause_winner(
        conn: Connection,
        *,
        parent_refs,
        parent_state_ids,
        coalesce_node_id,
    ) -> None:
        original_first_lock(
            conn,
            parent_refs=parent_refs,
            parent_state_ids=parent_state_ids,
            coalesce_node_id=coalesce_node_id,
        )
        backend_pids["winner"] = int(conn.exec_driver_sql("SELECT pg_backend_pid()").scalar_one())
        winner_has_authority.set()
        assert release_winner.wait(timeout=10)

    def observe_loser(
        conn: Connection,
        *,
        parent_refs,
        parent_state_ids,
        coalesce_node_id,
    ) -> None:
        backend_pids["loser"] = int(conn.exec_driver_sql("SELECT pg_backend_pid()").scalar_one())
        loser_attempting_authority.set()
        original_second_lock(
            conn,
            parent_refs=parent_refs,
            parent_state_ids=parent_state_ids,
            coalesce_node_id=coalesce_node_id,
        )
        loser_has_authority.set()

    monkeypatch.setattr(first.data_flow.tokens, "_lock_coalesce_dependencies", pause_winner)
    monkeypatch.setattr(second.data_flow.tokens, "_lock_coalesce_dependencies", observe_loser)
    results: dict[str, object] = {}

    def materialize(name: str, factory: RecorderFactory) -> None:
        try:
            results[name] = factory.data_flow.coalesce_tokens(
                parent_refs=list(refs),
                row_id=row.row_id,
                coalesce_node_id=coalesce_id,
                parent_state_ids=state_ids,
                merged_payload={"merged": True},
                merged_contract=_CONTRACT,
                step_in_pipeline=4,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            results[name] = exc

    threads = [
        threading.Thread(target=materialize, name="coalesce-winner", args=("winner", first)),
        threading.Thread(target=materialize, name="coalesce-loser", args=("loser", second)),
    ]
    try:
        threads[0].start()
        assert winner_has_authority.wait(timeout=10)
        threads[1].start()
        assert loser_attempting_authority.wait(timeout=10)
        assert not loser_has_authority.wait(timeout=0.25)
        release_winner.set()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive()

        assert set(results) == {"winner", "loser"}
        assert not any(isinstance(result, BaseException) for result in results.values())
        winner = results["winner"]
        loser = results["loser"]
        assert winner == loser
        assert backend_pids["winner"] != backend_pids["loser"]

        with first_db.connection() as conn:
            assert conn.execute(select(func.count()).select_from(coalesce_effects_table)).scalar_one() == 1
            assert conn.execute(select(func.count()).select_from(tokens_table)).scalar_one() == 3
            assert conn.execute(select(func.count()).select_from(token_parents_table)).scalar_one() == 2
    finally:
        release_winner.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=30)
        first_db.close()
        second_db.close()
