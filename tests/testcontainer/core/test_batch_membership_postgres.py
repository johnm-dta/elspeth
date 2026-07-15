"""PostgreSQL contention proof for atomic batch membership closure."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest
from sqlalchemy import event
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from elspeth.contracts import BatchStatus, NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory

pytestmark = pytest.mark.testcontainer

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


def _seed(factory: RecorderFactory) -> tuple[str, str]:
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
    source = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id="source-1",
        schema_config=_DYNAMIC_SCHEMA,
    )
    aggregation = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="aggregation",
        node_type=NodeType.AGGREGATION,
        plugin_version="1.0",
        config={},
        node_id="agg-1",
        schema_config=_DYNAMIC_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run.run_id,
        source.node_id,
        0,
        {"value": 1},
        row_id="row-1",
        source_row_index=0,
        ingest_sequence=0,
    )
    token = factory.data_flow.create_token(row.row_id, token_id="token-1")
    batch = factory.execution.create_batch(run.run_id, aggregation.node_id, batch_id="batch-1")
    return batch.batch_id, token.token_id


@pytest.mark.timeout(120)
def test_postgres_transition_lock_orders_racing_membership(postgres_url: str) -> None:
    """A status UPDATE that wins the row lock makes the racing add fail closed."""
    db = LandscapeDB.from_url(postgres_url)
    factory = RecorderFactory(db)
    batch_id, token_id = _seed(factory)

    transition_holds_lock = threading.Event()
    release_transition = threading.Event()
    member_reached_batch_read = threading.Event()
    outcomes: dict[str, str] = {}
    outcomes_lock = threading.Lock()

    def pause_after_transition(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:
        if statement.lstrip().upper().startswith("UPDATE BATCHES"):
            transition_holds_lock.set()
            if not release_transition.wait(timeout=30):
                raise TimeoutError("test did not release PostgreSQL batch transition")

    def observe_member_read(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:
        normalized = " ".join(statement.upper().split())
        if normalized.startswith("SELECT") and "FROM BATCHES" in normalized:
            member_reached_batch_read.set()

    event.listen(db.engine, "after_cursor_execute", pause_after_transition)
    event.listen(db.engine, "before_cursor_execute", observe_member_read)

    def transition() -> None:
        try:
            factory.execution.update_batch_status(batch_id, BatchStatus.EXECUTING)
        except BaseException as exc:  # pragma: no cover - asserted below
            result = f"{type(exc).__name__}: {exc}"
        else:
            result = "committed"
        with outcomes_lock:
            outcomes["transition"] = result

    def add_member() -> None:
        try:
            factory.execution.add_batch_member(batch_id, token_id, ordinal=0)
        except AuditIntegrityError as exc:
            result = f"refused: {exc}"
        except BaseException as exc:  # pragma: no cover - asserted below
            result = f"{type(exc).__name__}: {exc}"
        else:
            result = "inserted"
        with outcomes_lock:
            outcomes["member"] = result

    transition_thread = threading.Thread(target=transition, name="batch-transition")
    member_thread = threading.Thread(target=add_member, name="batch-member")
    try:
        transition_thread.start()
        assert transition_holds_lock.wait(timeout=30), "transition never reached its locked UPDATE"
        member_thread.start()
        assert member_reached_batch_read.wait(timeout=30), "member never reached its batch predicate read"
        release_transition.set()
        transition_thread.join(timeout=30)
        member_thread.join(timeout=30)
        assert not transition_thread.is_alive()
        assert not member_thread.is_alive()

        assert outcomes["transition"] == "committed"
        assert outcomes["member"].startswith("refused: ")
        assert "status 'executing'" in outcomes["member"]
        batch = factory.execution.get_batch(batch_id)
        assert batch is not None
        assert batch.status is BatchStatus.EXECUTING
        assert factory.execution.get_batch_members(batch_id) == []
    finally:
        release_transition.set()
        if transition_thread.ident is not None:
            transition_thread.join(timeout=30)
        if member_thread.ident is not None:
            member_thread.join(timeout=30)
        event.remove(db.engine, "before_cursor_execute", observe_member_read)
        event.remove(db.engine, "after_cursor_execute", pause_after_transition)
        db.close()
