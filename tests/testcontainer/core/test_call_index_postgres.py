"""PostgreSQL proof for database-owned call-index collision recovery."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any, Literal

import pytest
from sqlalchemy import event
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from elspeth.contracts import CallStatus, CallType, NodeType
from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory

pytestmark = pytest.mark.testcontainer

_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


def _seed_parent(factory: RecorderFactory, parent_kind: Literal["state", "operation"]) -> str:
    run_id = f"call-race-{parent_kind}"
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id=f"source-{parent_kind}",
        schema_config=_SCHEMA,
    )
    if parent_kind == "operation":
        return factory.execution.begin_operation(run_id, f"source-{parent_kind}", "source_load").operation_id

    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="transform",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        node_id="transform-state",
        schema_config=_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run_id,
        "source-state",
        0,
        {"value": 1},
        source_row_index=0,
        ingest_sequence=0,
    )
    token = factory.data_flow.create_token(row.row_id)
    return factory.execution.begin_node_state(
        token.token_id,
        "transform-state",
        run_id,
        0,
        {"value": 1},
    ).state_id


def _physical_connection(conn: Any) -> tuple[int, int]:
    driver_connection = conn.connection.driver_connection
    return id(driver_connection), int(driver_connection.info.backend_pid)


@pytest.mark.parametrize("parent_kind", ["state", "operation"])
@pytest.mark.timeout(120)
def test_postgres_completed_effects_survive_call_index_collision(
    postgres_url: str,
    parent_kind: Literal["state", "operation"],
) -> None:
    db = LandscapeDB.from_url(postgres_url)
    first = RecorderFactory(db)
    second = RecorderFactory(db)
    parent_id = _seed_parent(first, parent_kind)
    proposed = [
        first.execution.allocate_call_index(parent_id)
        if parent_kind == "state"
        else first.execution.allocate_operation_call_index(parent_id),
        second.execution.allocate_call_index(parent_id)
        if parent_kind == "state"
        else second.execution.allocate_operation_call_index(parent_id),
    ]
    assert proposed == [0, 0]

    at_insert = threading.Barrier(2)
    physical: dict[str, tuple[int, int]] = {}
    effects: list[str] = []
    outcomes: dict[str, tuple[str, int | str]] = {}
    lock = threading.Lock()

    def synchronize_inserts(conn, _cursor, statement, _parameters, _context, _executemany) -> None:  # type: ignore[no-untyped-def]
        normalized = " ".join(statement.upper().split())
        if normalized.startswith("INSERT INTO CALLS"):
            thread_name = threading.current_thread().name
            with lock:
                first_attempt = thread_name not in physical
                physical.setdefault(thread_name, _physical_connection(conn))
            if first_attempt:
                at_insert.wait(timeout=30)

    event.listen(db.engine, "before_cursor_execute", synchronize_inserts)

    def worker(name: str, factory: RecorderFactory) -> None:
        # The observable effect is complete before the contended audit insert.
        with lock:
            effects.append(name)
        try:
            if parent_kind == "state":
                call = factory.execution.record_call(
                    parent_id,
                    0,
                    CallType.HTTP,
                    CallStatus.SUCCESS,
                    request_data=RawCallPayload({"worker": name}),
                    response_data=RawCallPayload({"ok": True}),
                )
            else:
                call = factory.execution.record_operation_call(
                    parent_id,
                    CallType.HTTP,
                    CallStatus.SUCCESS,
                    request_data=RawCallPayload({"worker": name}),
                    response_data=RawCallPayload({"ok": True}),
                    call_index=0,
                )
        except BaseException as exc:  # pragma: no cover - asserted below
            result: tuple[str, int | str] = (type(exc).__name__, str(exc))
        else:
            result = ("ok", call.call_index)
        with lock:
            outcomes[name] = result

    threads = [
        threading.Thread(target=worker, name="call-first", args=("first", first)),
        threading.Thread(target=worker, name="call-second", args=("second", second)),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive()

        assert sorted(effects) == ["first", "second"]
        assert sorted(outcomes.values(), key=lambda item: str(item[1])) == [("ok", 0), ("ok", 1)]
        assert set(physical) == {"call-first", "call-second"}
        first_connection, second_connection = physical.values()
        assert first_connection[0] != second_connection[0]
        assert first_connection[1] != second_connection[1]
        calls = first.query.get_calls(parent_id) if parent_kind == "state" else first.execution.get_operation_calls(parent_id)
        assert [call.call_index for call in calls] == [0, 1]
    finally:
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=30)
        event.remove(db.engine, "before_cursor_execute", synchronize_inserts)
        db.close()
