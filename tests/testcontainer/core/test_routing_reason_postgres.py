"""PostgreSQL contention proofs for atomic routing decisions and reasons."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from elspeth.contracts import NodeType, RoutingEvent, RoutingMode, RoutingSpec
from elspeth.contracts.errors import AuditIntegrityError, ConfigGateReason
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.payload_store import FilesystemPayloadStore

pytestmark = pytest.mark.testcontainer

_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


def _seed_routing_state(factory: RecorderFactory, *, suffix: str) -> tuple[str, str, str]:
    run_id = f"routing-reason-{suffix}"
    source_id = f"source-{suffix}"
    gate_id = f"gate-{suffix}"
    sink_id = f"sink-{suffix}"
    state_id = f"state-{suffix}"
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id=source_id,
        schema_config=_SCHEMA,
    )
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="gate",
        node_type=NodeType.GATE,
        plugin_version="1.0",
        config={},
        node_id=gate_id,
        schema_config=_SCHEMA,
    )
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="sink",
        node_type=NodeType.SINK,
        plugin_version="1.0",
        config={},
        node_id=sink_id,
        schema_config=_SCHEMA,
    )
    edge_ids: list[str] = []
    for label in ("accepted", "rejected"):
        edge_ids.append(
            factory.data_flow.register_edge(
                run_id=run_id,
                from_node_id=gate_id,
                to_node_id=sink_id,
                label=label,
                mode=RoutingMode.MOVE,
                edge_id=f"edge-{label}-{suffix}",
            ).edge_id
        )
    row = factory.data_flow.create_row(
        run_id,
        source_id,
        0,
        {"route": "accepted"},
        row_id=f"row-{suffix}",
        source_row_index=0,
        ingest_sequence=0,
    )
    token = factory.data_flow.create_token(row.row_id, token_id=f"token-{suffix}")
    factory.execution.begin_node_state(
        token.token_id,
        gate_id,
        run_id,
        0,
        {"route": "accepted"},
        state_id=state_id,
    )
    return state_id, edge_ids[0], edge_ids[1]


def _seed_additional_state(factory: RecorderFactory, *, suffix: str) -> str:
    """Add another state in an existing seeded run for PK-collision proof."""
    run_id = f"routing-reason-{suffix}"
    row = factory.data_flow.create_row(
        run_id,
        f"source-{suffix}",
        1,
        {"route": "accepted"},
        row_id=f"row-{suffix}-second",
        source_row_index=1,
        ingest_sequence=1,
    )
    token = factory.data_flow.create_token(row.row_id, token_id=f"token-{suffix}-second")
    return factory.execution.begin_node_state(
        token.token_id,
        f"gate-{suffix}",
        run_id,
        0,
        {"route": "accepted"},
        state_id=f"state-{suffix}-second",
    ).state_id


def _physical_connection(conn: Any) -> tuple[int, int]:
    driver_connection = conn.connection.driver_connection
    return id(driver_connection), int(driver_connection.info.backend_pid)


def _install_state_authority_pause(
    db: LandscapeDB,
    *,
    winner_thread: str,
) -> tuple[threading.Event, threading.Event, threading.Event, dict[str, tuple[int, int]], Any]:
    winner_has_authority = threading.Event()
    loser_reached_lock = threading.Event()
    release_winner = threading.Event()
    physical: dict[str, tuple[int, int]] = {}
    lock = threading.Lock()

    def synchronize(conn, _cursor, statement, _parameters, _context, _executemany) -> None:  # type: ignore[no-untyped-def]
        normalized = " ".join(statement.upper().split())
        if normalized.startswith("SELECT") and "FROM NODE_STATES" in normalized and "FOR UPDATE" in normalized:
            thread_name = threading.current_thread().name
            with lock:
                physical.setdefault(thread_name, _physical_connection(conn))
            if thread_name != winner_thread:
                loser_reached_lock.set()

    def pause_winner(conn, _cursor, statement, _parameters, _context, _executemany) -> None:  # type: ignore[no-untyped-def]
        normalized = " ".join(statement.upper().split())
        if (
            threading.current_thread().name == winner_thread
            and normalized.startswith("SELECT")
            and "FROM NODE_STATES" in normalized
            and "FOR UPDATE" in normalized
            and not winner_has_authority.is_set()
        ):
            winner_has_authority.set()
            if not release_winner.wait(timeout=30):
                raise TimeoutError("test did not release PostgreSQL routing authority")

    event.listen(db.engine, "before_cursor_execute", synchronize)
    event.listen(db.engine, "after_cursor_execute", pause_winner)
    return winner_has_authority, loser_reached_lock, release_winner, physical, (synchronize, pause_winner)


@pytest.mark.timeout(120)
def test_postgres_identical_writers_return_one_exact_event(postgres_url: str, tmp_path: Path) -> None:
    """Distinct PostgreSQL backends converge through ON CONFLICT readback."""
    db = LandscapeDB.from_url(postgres_url)
    store_a = FilesystemPayloadStore(tmp_path / "payloads")
    store_b = FilesystemPayloadStore(tmp_path / "payloads")
    first = RecorderFactory(db, payload_store=store_a)
    second = RecorderFactory(db, payload_store=store_b)
    state_id, edge_id, _ = _seed_routing_state(first, suffix="identical")
    reason: ConfigGateReason = {"condition": "route == accepted", "result": "true"}
    winner_has_authority, loser_reached_lock, release_winner, physical, hooks = _install_state_authority_pause(
        db,
        winner_thread="routing-first",
    )
    outcomes: dict[str, RoutingEvent | BaseException] = {}
    outcomes_lock = threading.Lock()

    def worker(name: str, factory: RecorderFactory) -> None:
        try:
            result: RoutingEvent | BaseException = factory.execution.record_routing_event(
                state_id,
                edge_id,
                RoutingMode.MOVE,
                reason=reason,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            result = exc
        with outcomes_lock:
            outcomes[name] = result

    threads = [
        threading.Thread(target=worker, name="routing-first", args=("first", first)),
        threading.Thread(target=worker, name="routing-second", args=("second", second)),
    ]
    try:
        threads[0].start()
        assert winner_has_authority.wait(timeout=10)
        threads[1].start()
        assert loser_reached_lock.wait(timeout=10)
        release_winner.set()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive()

        assert set(outcomes) == {"first", "second"}
        assert all(isinstance(result, RoutingEvent) for result in outcomes.values())
        first_event = outcomes["first"]
        second_event = outcomes["second"]
        assert isinstance(first_event, RoutingEvent)
        assert isinstance(second_event, RoutingEvent)
        assert first_event == second_event
        assert set(physical) == {"routing-first", "routing-second"}
        first_connection, second_connection = physical.values()
        assert first_connection[0] != second_connection[0]
        assert first_connection[1] != second_connection[1]

        durable = first.query.get_routing_events(state_id)
        assert durable == [first_event]
        assert first_event.reason_hash == stable_hash(reason)
        assert first_event.reason_ref == first_event.reason_hash
        assert store_a.retrieve(first_event.reason_ref) == canonical_json(reason).encode("utf-8")
    finally:
        release_winner.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=30)
        event.remove(db.engine, "before_cursor_execute", hooks[0])
        event.remove(db.engine, "after_cursor_execute", hooks[1])
        db.close()


@pytest.mark.timeout(120)
def test_postgres_divergent_contender_fails_closed_without_mixed_group(postgres_url: str, tmp_path: Path) -> None:
    """One divergent contender loses without adding or mutating a route."""
    db = LandscapeDB.from_url(postgres_url)
    store_a = FilesystemPayloadStore(tmp_path / "payloads")
    store_b = FilesystemPayloadStore(tmp_path / "payloads")
    first = RecorderFactory(db, payload_store=store_a)
    second = RecorderFactory(db, payload_store=store_b)
    state_id, accepted_edge, rejected_edge = _seed_routing_state(first, suffix="divergent")
    accepted_reason: ConfigGateReason = {"condition": "route == accepted", "result": "true"}
    rejected_reason: ConfigGateReason = {"condition": "route == rejected", "result": "true"}
    winner_has_authority, loser_reached_lock, release_winner, physical, hooks = _install_state_authority_pause(
        db,
        winner_thread="routing-accepted",
    )
    outcomes: dict[str, RoutingEvent | BaseException] = {}
    outcomes_lock = threading.Lock()

    def worker(name: str, factory: RecorderFactory, edge_id: str, reason: ConfigGateReason) -> None:
        try:
            result: RoutingEvent | BaseException = factory.execution.record_routing_event(
                state_id,
                edge_id,
                RoutingMode.MOVE,
                reason=reason,
            )
        except BaseException as exc:
            result = exc
        with outcomes_lock:
            outcomes[name] = result

    threads = [
        threading.Thread(
            target=worker,
            name="routing-accepted",
            args=("accepted", first, accepted_edge, accepted_reason),
        ),
        threading.Thread(
            target=worker,
            name="routing-rejected",
            args=("rejected", second, rejected_edge, rejected_reason),
        ),
    ]
    try:
        threads[0].start()
        assert winner_has_authority.wait(timeout=10)
        threads[1].start()
        assert loser_reached_lock.wait(timeout=10)
        release_winner.set()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive()

        winners = [result for result in outcomes.values() if isinstance(result, RoutingEvent)]
        losers = [result for result in outcomes.values() if isinstance(result, AuditIntegrityError)]
        assert len(winners) == 1
        assert len(losers) == 1
        assert "durable event differs" in str(losers[0])
        assert set(physical) == {"routing-accepted", "routing-rejected"}
        first_connection, second_connection = physical.values()
        assert first_connection[0] != second_connection[0]
        assert first_connection[1] != second_connection[1]

        durable = first.query.get_routing_events(state_id)
        assert durable == winners
        assert len({event.routing_group_id for event in durable}) == 1
        winner = winners[0]
        expected_reason = accepted_reason if winner.edge_id == accepted_edge else rejected_reason
        assert winner.reason_hash == stable_hash(expected_reason)
        assert winner.reason_ref == winner.reason_hash
        assert store_a.retrieve(winner.reason_ref) == canonical_json(expected_reason).encode("utf-8")
        assert store_a.exists(stable_hash(accepted_reason))
        assert store_a.exists(stable_hash(rejected_reason))
    finally:
        release_winner.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=30)
        event.remove(db.engine, "before_cursor_execute", hooks[0])
        event.remove(db.engine, "after_cursor_execute", hooks[1])
        db.close()


@pytest.mark.parametrize("winner_kind", ["single", "group"])
@pytest.mark.timeout(120)
def test_postgres_single_group_race_has_one_complete_winner(
    postgres_url: str,
    tmp_path: Path,
    winner_kind: str,
) -> None:
    """Parent-row authority prevents partial extension in either lock order."""
    suffix = f"cardinality-{winner_kind}"
    db = LandscapeDB.from_url(postgres_url)
    first = RecorderFactory(db, payload_store=FilesystemPayloadStore(tmp_path / "payloads"))
    second = RecorderFactory(db, payload_store=FilesystemPayloadStore(tmp_path / "payloads"))
    state_id, accepted_edge, rejected_edge = _seed_routing_state(first, suffix=suffix)
    loser_kind = "group" if winner_kind == "single" else "single"
    winner_thread_name = f"routing-{winner_kind}"
    winner_has_authority, loser_reached_lock, release_winner, physical, hooks = _install_state_authority_pause(
        db,
        winner_thread=winner_thread_name,
    )
    outcomes: dict[str, list[RoutingEvent] | BaseException] = {}
    outcomes_lock = threading.Lock()

    def worker(name: str, factory: RecorderFactory) -> None:
        try:
            if name == "single":
                result: list[RoutingEvent] | BaseException = [
                    factory.execution.record_routing_event(
                        state_id,
                        accepted_edge,
                        RoutingMode.MOVE,
                        reason={"condition": "route chosen", "result": "true"},
                    )
                ]
            else:
                result = factory.execution.record_routing_events(
                    state_id,
                    [
                        RoutingSpec(edge_id=accepted_edge, mode=RoutingMode.MOVE),
                        RoutingSpec(edge_id=rejected_edge, mode=RoutingMode.MOVE),
                    ],
                    reason={"condition": "route chosen", "result": "true"},
                )
        except BaseException as exc:
            result = exc
        with outcomes_lock:
            outcomes[name] = result

    threads = [
        threading.Thread(target=worker, name=winner_thread_name, args=(winner_kind, first)),
        threading.Thread(target=worker, name=f"routing-{loser_kind}", args=(loser_kind, second)),
    ]
    try:
        threads[0].start()
        assert winner_has_authority.wait(timeout=10)
        threads[1].start()
        assert loser_reached_lock.wait(timeout=10)
        release_winner.set()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive()

        assert isinstance(outcomes[winner_kind], list)
        assert isinstance(outcomes[loser_kind], AuditIntegrityError)
        assert len(outcomes[winner_kind]) == (1 if winner_kind == "single" else 2)  # type: ignore[arg-type]
        assert set(physical) == {winner_thread_name, f"routing-{loser_kind}"}
        connection_a, connection_b = physical.values()
        assert connection_a[0] != connection_b[0]
        assert connection_a[1] != connection_b[1]
        durable = first.query.get_routing_events(state_id)
        assert len(durable) == (1 if winner_kind == "single" else 2)
        assert len({event.routing_group_id for event in durable}) == 1
    finally:
        release_winner.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=30)
        event.remove(db.engine, "before_cursor_execute", hooks[0])
        event.remove(db.engine, "after_cursor_execute", hooks[1])
        db.close()


@pytest.mark.timeout(120)
def test_postgres_unrelated_event_id_collision_remains_landscape_record_error(
    postgres_url: str,
    tmp_path: Path,
) -> None:
    """Natural-key retry handling must not swallow primary-key violations."""
    db = LandscapeDB.from_url(postgres_url)
    store = FilesystemPayloadStore(tmp_path / "payloads")
    factory = RecorderFactory(db, payload_store=store)
    first_state, edge_id, _ = _seed_routing_state(factory, suffix="event-pk")
    second_state = _seed_additional_state(factory, suffix="event-pk")
    factory.execution.record_routing_event(
        first_state,
        edge_id,
        RoutingMode.MOVE,
        reason={"condition": "first", "result": "true"},
        event_id="supplied-event-id-collision",
        routing_group_id="supplied-first-group",
    )

    try:
        with pytest.raises(LandscapeRecordError, match="database rejected audit write: IntegrityError"):
            factory.execution.record_routing_event(
                second_state,
                edge_id,
                RoutingMode.MOVE,
                reason={"condition": "second", "result": "true"},
                event_id="supplied-event-id-collision",
                routing_group_id="supplied-second-group",
            )
        assert factory.query.get_routing_events(second_state) == []
    finally:
        db.close()


@pytest.mark.timeout(120)
def test_postgres_foreign_key_violation_remains_landscape_record_error(
    postgres_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Natural-key retry handling must not swallow foreign-key violations."""
    db = LandscapeDB.from_url(postgres_url)
    factory = RecorderFactory(db, payload_store=FilesystemPayloadStore(tmp_path / "payloads"))
    state_id, edge_id, _ = _seed_routing_state(factory, suffix="event-fk")
    original_values = factory.execution.node_states._routing_event_values

    def values_with_missing_edge(routing_event: RoutingEvent, *, run_id: str) -> dict[str, object]:
        values = original_values(routing_event, run_id=run_id)
        values["edge_id"] = "missing-edge-injected-after-validation"
        return values

    monkeypatch.setattr(factory.execution.node_states, "_routing_event_values", values_with_missing_edge)
    try:
        with pytest.raises(LandscapeRecordError, match="database rejected audit write: IntegrityError"):
            factory.execution.record_routing_event(
                state_id,
                edge_id,
                RoutingMode.MOVE,
                reason={"condition": "foreign key", "result": "true"},
            )
        assert factory.query.get_routing_events(state_id) == []
    finally:
        db.close()
