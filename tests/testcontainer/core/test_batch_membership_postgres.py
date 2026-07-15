"""PostgreSQL contention proof for atomic batch membership closure."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

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


def _seed(factory: RecorderFactory, *, suffix: str) -> tuple[str, str]:
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=f"run-{suffix}")
    source = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id=f"source-{suffix}",
        schema_config=_DYNAMIC_SCHEMA,
    )
    aggregation = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="aggregation",
        node_type=NodeType.AGGREGATION,
        plugin_version="1.0",
        config={},
        node_id=f"agg-{suffix}",
        schema_config=_DYNAMIC_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run.run_id,
        source.node_id,
        0,
        {"value": 1},
        row_id=f"row-{suffix}",
        source_row_index=0,
        ingest_sequence=0,
    )
    token = factory.data_flow.create_token(row.row_id, token_id=f"token-{suffix}")
    batch = factory.execution.create_batch(run.run_id, aggregation.node_id, batch_id=f"batch-{suffix}")
    return batch.batch_id, token.token_id


def _physical_postgres_connection(conn: Any) -> tuple[int, int]:
    """Return process-local DBAPI identity plus server backend PID."""
    driver_connection = conn.connection.driver_connection
    return id(driver_connection), int(driver_connection.info.backend_pid)


@pytest.mark.parametrize(
    "target_status",
    [BatchStatus.EXECUTING, BatchStatus.COMPLETED, BatchStatus.FAILED],
)
@pytest.mark.timeout(120)
def test_postgres_transition_lock_orders_racing_membership(
    postgres_url: str,
    target_status: BatchStatus,
) -> None:
    """Every membership-closing transition orders before a racing add."""
    db = LandscapeDB.from_url(postgres_url)
    factory = RecorderFactory(db)
    batch_id, token_id = _seed(factory, suffix=target_status.value)

    transition_holds_lock = threading.Event()
    release_transition = threading.Event()
    member_reached_batch_read = threading.Event()
    outcomes: dict[str, str] = {}
    physical_connections: dict[str, tuple[int, int]] = {}
    outcomes_lock = threading.Lock()

    def pause_after_transition(conn, _cursor, statement, _parameters, _context, _executemany) -> None:  # type: ignore[no-untyped-def]
        if statement.lstrip().upper().startswith("UPDATE BATCHES"):
            physical_connections["transition"] = _physical_postgres_connection(conn)
            transition_holds_lock.set()
            if not release_transition.wait(timeout=30):
                raise TimeoutError("test did not release PostgreSQL batch transition")

    def observe_member_read(conn, _cursor, statement, _parameters, _context, _executemany) -> None:  # type: ignore[no-untyped-def]
        normalized = " ".join(statement.upper().split())
        if threading.current_thread().name == "batch-member" and normalized.startswith("SELECT") and "FROM BATCHES" in normalized:
            physical_connections.setdefault("member", _physical_postgres_connection(conn))
            member_reached_batch_read.set()

    event.listen(db.engine, "after_cursor_execute", pause_after_transition)
    event.listen(db.engine, "before_cursor_execute", observe_member_read)

    def transition() -> None:
        try:
            if target_status is BatchStatus.EXECUTING:
                factory.execution.update_batch_status(batch_id, target_status)
            else:
                factory.execution.complete_batch(batch_id, target_status)
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
        assert f"status {target_status.value!r}" in outcomes["member"]
        assert set(physical_connections) == {"transition", "member"}
        transition_connection, member_connection = physical_connections["transition"], physical_connections["member"]
        assert transition_connection[0] != member_connection[0], "threads must use distinct physical DBAPI connections"
        assert transition_connection[1] != member_connection[1], "threads must use distinct PostgreSQL backend processes"
        batch = factory.execution.get_batch(batch_id)
        assert batch is not None
        assert batch.status is target_status
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
