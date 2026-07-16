"""Crash, contention, and restore proofs for atomic routing reasons."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy import select

from elspeth.contracts import NodeType, RoutingMode, RoutingSpec
from elspeth.contracts.errors import AuditIntegrityError, ConfigGateReason
from elspeth.contracts.payload_store import PayloadStore
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.exporter import LandscapeExporter
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import routing_events_table
from elspeth.core.payload_store import FilesystemPayloadStore

RUN_ID = "routing-atomic-run"
SOURCE_ID = "source-0"
GATE_ID = "gate-0"
SINK_ID = "sink-0"
ROW_ID = "row-0"
TOKEN_ID = "token-0"
STATE_ID = "state-0"
STATE_B_ID = "state-1"
EDGE_ID = "edge-0"
EDGE_B_ID = "edge-1"
SHARED_GROUP_ID = "caller-shared-cross-state-group"
REASON: ConfigGateReason = {"condition": "row['route'] == 'accepted'", "result": "true"}
SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _seed_routing_state(db_url: str) -> None:
    with LandscapeDB.from_url(db_url) as db:
        factory = RecorderFactory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=RUN_ID)
        factory.data_flow.register_node(
            run_id=RUN_ID,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id=SOURCE_ID,
            schema_config=SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=RUN_ID,
            plugin_name="gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            node_id=GATE_ID,
            schema_config=SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=RUN_ID,
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id=SINK_ID,
            schema_config=SCHEMA,
        )
        factory.data_flow.register_edge(
            run_id=RUN_ID,
            from_node_id=GATE_ID,
            to_node_id=SINK_ID,
            label="accepted",
            mode=RoutingMode.MOVE,
            edge_id=EDGE_ID,
        )
        factory.data_flow.register_edge(
            run_id=RUN_ID,
            from_node_id=GATE_ID,
            to_node_id=SINK_ID,
            label="rejected",
            mode=RoutingMode.MOVE,
            edge_id=EDGE_B_ID,
        )
        factory.data_flow.create_row(
            RUN_ID,
            SOURCE_ID,
            0,
            {"route": "accepted"},
            row_id=ROW_ID,
            source_row_index=0,
            ingest_sequence=0,
        )
        factory.data_flow.create_token(ROW_ID, token_id=TOKEN_ID)
        factory.execution.begin_node_state(
            TOKEN_ID,
            GATE_ID,
            RUN_ID,
            0,
            {"route": "accepted"},
            state_id=STATE_ID,
        )


def _seed_additional_routing_state(db_url: str) -> None:
    """Add a second state in the seeded run for cross-state ownership races."""
    with LandscapeDB.from_url(db_url, create_tables=False) as db:
        factory = RecorderFactory(db)
        factory.data_flow.create_row(
            RUN_ID,
            SOURCE_ID,
            1,
            {"route": "accepted"},
            row_id="row-1",
            source_row_index=1,
            ingest_sequence=1,
        )
        factory.data_flow.create_token("row-1", token_id="token-1")
        factory.execution.begin_node_state(
            "token-1",
            GATE_ID,
            RUN_ID,
            0,
            {"route": "accepted"},
            state_id=STATE_B_ID,
        )


class _CrashAfterStore(PayloadStore):
    """Persist through the real filesystem store, then lose the process."""

    def __init__(self, payload_dir: str) -> None:
        self._inner = FilesystemPayloadStore(Path(payload_dir))

    def store(self, content: bytes) -> str:
        self._inner.store(content)
        os._exit(73)

    def retrieve(self, content_hash: str) -> bytes:
        return self._inner.retrieve(content_hash)

    def exists(self, content_hash: str) -> bool:
        return self._inner.exists(content_hash)

    def delete(self, content_hash: str) -> bool:
        return self._inner.delete(content_hash)


def _crash_between_reason_store_and_event_insert(db_url: str, payload_dir: str) -> None:
    with LandscapeDB.from_url(db_url, create_tables=False) as db:
        factory = RecorderFactory(db, payload_store=_CrashAfterStore(payload_dir))
        factory.execution.record_routing_event(STATE_ID, EDGE_ID, RoutingMode.MOVE, reason=REASON)
    os._exit(74)


def _record_concurrently(db_url: str, payload_dir: str, barrier: Any, results: Any) -> None:
    try:
        with LandscapeDB.from_url(db_url, create_tables=False) as db:
            store = FilesystemPayloadStore(Path(payload_dir))
            factory = RecorderFactory(db, payload_store=store)
            barrier.wait(timeout=30)
            event = factory.execution.record_routing_event(STATE_ID, EDGE_ID, RoutingMode.MOVE, reason=REASON)
            results.put(
                (
                    "ok",
                    event.event_id,
                    event.routing_group_id,
                    event.reason_hash,
                    event.reason_ref,
                )
            )
    except BaseException as exc:
        results.put(("error", type(exc).__name__, str(exc)))
        raise


def _record_decision_with_sqlite_lock_pause(
    db_url: str,
    payload_dir: str,
    decision_kind: str,
    pause_after_lock: bool,
    begin_attempted: Any,
    write_lock_acquired: Any,
    release_lock: Any,
    result_ready: Any,
    results: Any,
    state_id: str = STATE_ID,
    routing_group_id: str | None = None,
    ordinal: int = 0,
    event_id: str | None = None,
) -> None:
    """Record one single/group decision, optionally pausing with write authority."""
    with LandscapeDB.from_url(db_url, create_tables=False) as db:

        def observe_begin_attempt(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:  # type: ignore[no-untyped-def]
            if statement.strip().upper() == "BEGIN IMMEDIATE":
                begin_attempted.set()

        def observe_lock_acquired(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:  # type: ignore[no-untyped-def]
            if statement.strip().upper() == "BEGIN IMMEDIATE":
                write_lock_acquired.set()
                if pause_after_lock and not release_lock.wait(timeout=30):
                    raise TimeoutError("test did not release SQLite routing authority")

        sqlalchemy_event.listen(db.engine, "before_cursor_execute", observe_begin_attempt)
        sqlalchemy_event.listen(db.engine, "after_cursor_execute", observe_lock_acquired)

        store = FilesystemPayloadStore(Path(payload_dir))
        factory = RecorderFactory(db, payload_store=store)
        try:
            if decision_kind == "group":
                recorded = factory.execution.record_routing_events(
                    state_id,
                    [
                        RoutingSpec(edge_id=EDGE_ID, mode=RoutingMode.MOVE),
                        RoutingSpec(edge_id=EDGE_B_ID, mode=RoutingMode.MOVE),
                    ],
                    reason=REASON,
                )
            else:
                recorded = [
                    factory.execution.record_routing_event(
                        state_id,
                        EDGE_ID,
                        RoutingMode.MOVE,
                        reason=REASON,
                        event_id=event_id,
                        routing_group_id=routing_group_id,
                        ordinal=ordinal,
                    )
                ]
        except BaseException as exc:
            results.put((decision_kind, "error", os.getpid(), type(exc).__name__, str(exc)))
        else:
            results.put((decision_kind, "ok", os.getpid(), len(recorded), tuple(event.event_id for event in recorded)))
        finally:
            result_ready.set()


def _insert_legacy_routing_group(
    db_url: str,
    *,
    group_id: str,
    event_prefix: str,
    edge_ids: tuple[str, ...],
    reason_hash: str,
    reason_ref: str | None,
) -> None:
    """Insert the pre-change random-ID row shape directly."""
    with LandscapeDB.from_url(db_url, create_tables=False) as db, db.write_connection() as conn:
        for ordinal, edge_id in enumerate(edge_ids):
            conn.execute(
                routing_events_table.insert().values(
                    event_id=f"{event_prefix}-{ordinal}",
                    state_id=STATE_ID,
                    edge_id=edge_id,
                    run_id=RUN_ID,
                    routing_group_id=group_id,
                    ordinal=ordinal,
                    mode=RoutingMode.MOVE,
                    reason_hash=reason_hash,
                    reason_ref=reason_ref,
                    created_at=datetime(2026, 7, 1, tzinfo=UTC),
                )
            )


def test_crash_after_reason_store_restarts_to_exact_event(tmp_path: Path) -> None:
    """A hard process loss after fsync leaves no event and retry converges."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    payload_dir = tmp_path / "payloads"
    _seed_routing_state(db_url)

    process = multiprocessing.get_context("spawn").Process(
        target=_crash_between_reason_store_and_event_insert,
        args=(db_url, str(payload_dir)),
    )
    process.start()
    process.join(timeout=30)
    assert not process.is_alive()
    assert process.exitcode == 73

    expected_ref = stable_hash(REASON)
    store = FilesystemPayloadStore(payload_dir)
    assert store.exists(expected_ref)
    with LandscapeDB.from_url(db_url, create_tables=False) as db:
        with db.read_only_connection() as conn:
            assert conn.execute(select(routing_events_table.c.event_id)).all() == []

        restarted = RecorderFactory(db, payload_store=store)
        event = restarted.execution.record_routing_event(STATE_ID, EDGE_ID, RoutingMode.MOVE, reason=REASON)
        assert event.reason_hash == expected_ref
        assert event.reason_ref == expected_ref
        assert store.retrieve(expected_ref) == b'{"condition":"row[\'route\'] == \'accepted\'","result":"true"}'

        with db.read_only_connection() as conn:
            rows = conn.execute(select(routing_events_table)).all()
        assert len(rows) == 1
        assert rows[0].reason_hash == expected_ref
        assert rows[0].reason_ref == expected_ref


def test_spawned_identical_writers_converge_and_export_exact_reason(tmp_path: Path) -> None:
    """Independent engines and stores converge on one authoritative event."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    payload_dir = tmp_path / "payloads"
    _seed_routing_state(db_url)

    context = multiprocessing.get_context("spawn")
    writer_count = 4
    barrier = context.Barrier(writer_count)
    results = context.Queue()
    processes = [
        context.Process(target=_record_concurrently, args=(db_url, str(payload_dir), barrier, results)) for _ in range(writer_count)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=45)
        assert not process.is_alive()
        assert process.exitcode == 0

    worker_results = [results.get(timeout=10) for _ in range(writer_count)]
    assert all(result[0] == "ok" for result in worker_results)
    assert len(set(worker_results)) == 1

    expected_ref = stable_hash(REASON)
    store = FilesystemPayloadStore(payload_dir)
    with LandscapeDB.from_url(db_url, create_tables=False) as db:
        with db.read_only_connection() as conn:
            rows = conn.execute(select(routing_events_table)).all()
        assert len(rows) == 1
        assert rows[0].reason_hash == expected_ref
        assert rows[0].reason_ref == expected_ref
        assert store.retrieve(rows[0].reason_ref) == b'{"condition":"row[\'route\'] == \'accepted\'","result":"true"}'

        routing_exports = [record for record in LandscapeExporter(db).export_run(RUN_ID) if record["record_type"] == "routing_event"]
        assert len(routing_exports) == 1
        assert routing_exports[0]["event_id"] == rows[0].event_id
        assert routing_exports[0]["reason_hash"] == expected_ref
        assert routing_exports[0]["reason_ref"] == expected_ref


def test_multi_then_single_is_refused_without_shrinking_decision(tmp_path: Path) -> None:
    """A complete two-route decision cannot be retried as one route."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    payload_dir = tmp_path / "payloads"
    _seed_routing_state(db_url)
    with LandscapeDB.from_url(db_url, create_tables=False) as db:
        factory = RecorderFactory(db, payload_store=FilesystemPayloadStore(payload_dir))
        routes = [
            RoutingSpec(edge_id=EDGE_ID, mode=RoutingMode.MOVE),
            RoutingSpec(edge_id=EDGE_B_ID, mode=RoutingMode.MOVE),
        ]
        original = factory.execution.record_routing_events(STATE_ID, routes, reason=REASON)

        with pytest.raises(AuditIntegrityError, match="complete routing decision"):
            factory.execution.record_routing_event(STATE_ID, EDGE_ID, RoutingMode.MOVE, reason=REASON)

        assert factory.query.get_routing_events(STATE_ID) == original


def test_single_then_multi_is_refused_without_extending_decision(tmp_path: Path) -> None:
    """A complete one-route decision cannot be extended by a fork retry."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    payload_dir = tmp_path / "payloads"
    _seed_routing_state(db_url)
    with LandscapeDB.from_url(db_url, create_tables=False) as db:
        factory = RecorderFactory(db, payload_store=FilesystemPayloadStore(payload_dir))
        original = factory.execution.record_routing_event(STATE_ID, EDGE_ID, RoutingMode.MOVE, reason=REASON)

        with pytest.raises(AuditIntegrityError, match="complete routing decision"):
            factory.execution.record_routing_events(
                STATE_ID,
                [
                    RoutingSpec(edge_id=EDGE_ID, mode=RoutingMode.MOVE),
                    RoutingSpec(edge_id=EDGE_B_ID, mode=RoutingMode.MOVE),
                ],
                reason=REASON,
            )

        assert factory.query.get_routing_events(STATE_ID) == [original]


@pytest.mark.parametrize("winner_kind", ["single", "group"])
def test_spawned_single_group_race_has_one_complete_winner(tmp_path: Path, winner_kind: str) -> None:
    """SQLite write authority makes either preselected decision win wholly."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    payload_dir = tmp_path / "payloads"
    _seed_routing_state(db_url)
    loser_kind = "group" if winner_kind == "single" else "single"
    context = multiprocessing.get_context("spawn")
    winner_begin_attempted = context.Event()
    winner_write_lock_acquired = context.Event()
    release_winner = context.Event()
    winner_result_ready = context.Event()
    loser_begin_attempted = context.Event()
    loser_write_lock_acquired = context.Event()
    release_loser = context.Event()
    loser_result_ready = context.Event()
    results = context.Queue()
    winner = context.Process(
        target=_record_decision_with_sqlite_lock_pause,
        args=(
            db_url,
            str(payload_dir),
            winner_kind,
            True,
            winner_begin_attempted,
            winner_write_lock_acquired,
            release_winner,
            winner_result_ready,
            results,
        ),
    )
    loser = context.Process(
        target=_record_decision_with_sqlite_lock_pause,
        args=(
            db_url,
            str(payload_dir),
            loser_kind,
            False,
            loser_begin_attempted,
            loser_write_lock_acquired,
            release_loser,
            loser_result_ready,
            results,
        ),
    )
    try:
        winner.start()
        assert winner_begin_attempted.wait(timeout=15)
        assert winner_write_lock_acquired.wait(timeout=15)
        assert not winner_result_ready.is_set()

        loser.start()
        assert loser_begin_attempted.wait(timeout=15)
        assert not loser_write_lock_acquired.wait(timeout=1)
        assert not loser_result_ready.is_set()
        assert not winner_result_ready.is_set()

        release_winner.set()
        assert winner_result_ready.wait(timeout=15)
        assert loser_write_lock_acquired.wait(timeout=15)
        assert loser_result_ready.wait(timeout=15)
    finally:
        release_winner.set()
        release_loser.set()
        for process in (winner, loser):
            if process.pid is not None:
                process.join(timeout=45)
                assert not process.is_alive()
                assert process.exitcode == 0

    outcomes = {result[0]: result[1:] for result in (results.get(timeout=10), results.get(timeout=10))}
    assert outcomes[winner_kind][0] == "ok"
    assert outcomes[loser_kind][0] == "error"
    assert outcomes[loser_kind][2] == "AuditIntegrityError"
    assert outcomes[winner_kind][1] == winner.pid
    assert outcomes[loser_kind][1] == loser.pid
    assert winner.pid != loser.pid
    assert os.getpid() not in {winner.pid, loser.pid}
    with LandscapeDB.from_url(db_url, create_tables=False) as db:
        durable = RecorderFactory(db).query.get_routing_events(STATE_ID)
    assert len(durable) == (1 if winner_kind == "single" else 2)
    assert len({event.routing_group_id for event in durable}) == 1


def test_spawned_cross_state_shared_group_has_one_durable_owner(tmp_path: Path) -> None:
    """SQLite write authority serializes ownership of an absent shared group."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    payload_dir = tmp_path / "payloads"
    _seed_routing_state(db_url)
    _seed_additional_routing_state(db_url)
    context = multiprocessing.get_context("spawn")
    winner_begin_attempted = context.Event()
    winner_write_lock_acquired = context.Event()
    release_winner = context.Event()
    winner_result_ready = context.Event()
    loser_begin_attempted = context.Event()
    loser_write_lock_acquired = context.Event()
    release_loser = context.Event()
    loser_result_ready = context.Event()
    results = context.Queue()
    winner = context.Process(
        target=_record_decision_with_sqlite_lock_pause,
        args=(
            db_url,
            str(payload_dir),
            "state-first",
            True,
            winner_begin_attempted,
            winner_write_lock_acquired,
            release_winner,
            winner_result_ready,
            results,
            STATE_ID,
            SHARED_GROUP_ID,
            0,
            "cross-state-event-0",
        ),
    )
    loser = context.Process(
        target=_record_decision_with_sqlite_lock_pause,
        args=(
            db_url,
            str(payload_dir),
            "state-second",
            False,
            loser_begin_attempted,
            loser_write_lock_acquired,
            release_loser,
            loser_result_ready,
            results,
            STATE_B_ID,
            SHARED_GROUP_ID,
            1,
            "cross-state-event-1",
        ),
    )
    try:
        winner.start()
        assert winner_begin_attempted.wait(timeout=15)
        assert winner_write_lock_acquired.wait(timeout=15)
        assert not winner_result_ready.is_set()

        loser.start()
        assert loser_begin_attempted.wait(timeout=15)
        assert not loser_write_lock_acquired.wait(timeout=1)
        assert not loser_result_ready.is_set()
        assert not winner_result_ready.is_set()

        release_winner.set()
        assert winner_result_ready.wait(timeout=15)
        assert loser_write_lock_acquired.wait(timeout=15)
        assert loser_result_ready.wait(timeout=15)
    finally:
        release_winner.set()
        release_loser.set()
        for process in (winner, loser):
            if process.pid is not None:
                process.join(timeout=45)
                assert not process.is_alive()
                assert process.exitcode == 0

    outcomes = {result[0]: result[1:] for result in (results.get(timeout=10), results.get(timeout=10))}
    assert outcomes["state-first"][0] == "ok"
    assert outcomes["state-second"][0] == "error"
    assert outcomes["state-second"][2] == "AuditIntegrityError"
    assert outcomes["state-first"][1] == winner.pid
    assert outcomes["state-second"][1] == loser.pid
    assert winner.pid != loser.pid
    assert os.getpid() not in {winner.pid, loser.pid}

    with LandscapeDB.from_url(db_url, create_tables=False) as db, db.read_only_connection() as conn:
        durable = list(
            conn.execute(
                select(
                    routing_events_table.c.event_id,
                    routing_events_table.c.state_id,
                    routing_events_table.c.ordinal,
                ).where(routing_events_table.c.routing_group_id == SHARED_GROUP_ID)
            ).fetchall()
        )
    assert len(durable) == 1
    assert durable[0].state_id == STATE_ID
    assert durable[0].ordinal == 0


def test_legacy_single_default_retry_returns_random_id_row(tmp_path: Path) -> None:
    """A pre-change single decision is retried without appending new IDs."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    payload_dir = tmp_path / "payloads"
    _seed_routing_state(db_url)
    store = FilesystemPayloadStore(payload_dir)
    reason_ref = store.store(b'{"condition":"row[\'route\'] == \'accepted\'","result":"true"}')
    _insert_legacy_routing_group(
        db_url,
        group_id="legacy-random-single-group",
        event_prefix="legacy-random-single-event",
        edge_ids=(EDGE_ID,),
        reason_hash=stable_hash(REASON),
        reason_ref=reason_ref,
    )

    with LandscapeDB.from_url(db_url, create_tables=False) as reopened:
        factory = RecorderFactory(reopened, payload_store=store)
        retried = factory.execution.record_routing_event(STATE_ID, EDGE_ID, RoutingMode.MOVE, reason=REASON)
        assert retried.event_id == "legacy-random-single-event-0"
        assert retried.routing_group_id == "legacy-random-single-group"
        assert factory.query.get_routing_events(STATE_ID) == [retried]


def test_legacy_group_only_retry_returns_random_event_id_without_append(tmp_path: Path) -> None:
    """A caller-stable legacy group does not imply a deterministic event ID."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    payload_dir = tmp_path / "payloads"
    _seed_routing_state(db_url)
    store = FilesystemPayloadStore(payload_dir)
    reason_ref = store.store(b'{"condition":"row[\'route\'] == \'accepted\'","result":"true"}')
    _insert_legacy_routing_group(
        db_url,
        group_id="caller-stable-legacy-group",
        event_prefix="legacy-random-event",
        edge_ids=(EDGE_ID,),
        reason_hash=stable_hash(REASON),
        reason_ref=reason_ref,
    )

    with LandscapeDB.from_url(db_url, create_tables=False) as reopened:
        factory = RecorderFactory(reopened, payload_store=store)
        retried = factory.execution.record_routing_event(
            STATE_ID,
            EDGE_ID,
            RoutingMode.MOVE,
            reason=REASON,
            routing_group_id="caller-stable-legacy-group",
        )
        assert retried.event_id == "legacy-random-event-0"
        assert retried.routing_group_id == "caller-stable-legacy-group"
        assert factory.query.get_routing_events(STATE_ID) == [retried]


def test_default_retry_rejects_random_event_id_in_current_deterministic_group(tmp_path: Path) -> None:
    """A current deterministic group cannot silently bless a corrupt event ID."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    payload_dir = tmp_path / "payloads"
    _seed_routing_state(db_url)
    store = FilesystemPayloadStore(payload_dir)
    reason_ref = store.store(b'{"condition":"row[\'route\'] == \'accepted\'","result":"true"}')
    deterministic_group_id = stable_hash({"kind": "routing_group", "state_id": STATE_ID})
    _insert_legacy_routing_group(
        db_url,
        group_id=deterministic_group_id,
        event_prefix="random-event-in-current-group",
        edge_ids=(EDGE_ID,),
        reason_hash=stable_hash(REASON),
        reason_ref=reason_ref,
    )

    with LandscapeDB.from_url(db_url, create_tables=False) as reopened:
        factory = RecorderFactory(reopened, payload_store=store)
        with pytest.raises(AuditIntegrityError, match="durable event differs"):
            factory.execution.record_routing_event(STATE_ID, EDGE_ID, RoutingMode.MOVE, reason=REASON)
        durable = factory.query.get_routing_events(STATE_ID)
        assert len(durable) == 1
        assert durable[0].event_id == "random-event-in-current-group-0"


def test_legacy_retry_with_explicit_event_id_remains_strict(tmp_path: Path) -> None:
    """An explicit event identity cannot alias a different durable legacy ID."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    payload_dir = tmp_path / "payloads"
    _seed_routing_state(db_url)
    store = FilesystemPayloadStore(payload_dir)
    reason_ref = store.store(b'{"condition":"row[\'route\'] == \'accepted\'","result":"true"}')
    _insert_legacy_routing_group(
        db_url,
        group_id="caller-stable-strict-group",
        event_prefix="legacy-durable-event",
        edge_ids=(EDGE_ID,),
        reason_hash=stable_hash(REASON),
        reason_ref=reason_ref,
    )

    with LandscapeDB.from_url(db_url, create_tables=False) as reopened:
        factory = RecorderFactory(reopened, payload_store=store)
        with pytest.raises(AuditIntegrityError, match="durable event differs"):
            factory.execution.record_routing_event(
                STATE_ID,
                EDGE_ID,
                RoutingMode.MOVE,
                reason=REASON,
                event_id="explicit-different-event-id",
                routing_group_id="caller-stable-strict-group",
            )
        durable = factory.query.get_routing_events(STATE_ID)
        assert len(durable) == 1
        assert durable[0].event_id == "legacy-durable-event-0"


def test_legacy_multi_default_retry_returns_random_id_group(tmp_path: Path) -> None:
    """A pre-change fork is retried without appending deterministic IDs."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    payload_dir = tmp_path / "payloads"
    _seed_routing_state(db_url)
    store = FilesystemPayloadStore(payload_dir)
    reason_ref = store.store(b'{"condition":"row[\'route\'] == \'accepted\'","result":"true"}')
    _insert_legacy_routing_group(
        db_url,
        group_id="legacy-random-multi-group",
        event_prefix="legacy-random-multi-event",
        edge_ids=(EDGE_ID, EDGE_B_ID),
        reason_hash=stable_hash(REASON),
        reason_ref=reason_ref,
    )

    with LandscapeDB.from_url(db_url, create_tables=False) as reopened:
        factory = RecorderFactory(reopened, payload_store=store)
        retried = factory.execution.record_routing_events(
            STATE_ID,
            [
                RoutingSpec(edge_id=EDGE_ID, mode=RoutingMode.MOVE),
                RoutingSpec(edge_id=EDGE_B_ID, mode=RoutingMode.MOVE),
            ],
            reason=REASON,
        )
        assert [event.event_id for event in retried] == ["legacy-random-multi-event-0", "legacy-random-multi-event-1"]
        assert {event.routing_group_id for event in retried} == {"legacy-random-multi-group"}
        assert factory.query.get_routing_events(STATE_ID) == retried


def test_multiple_legacy_groups_fail_closed_without_append(tmp_path: Path) -> None:
    """Ambiguous pre-change decision groups cannot be guessed or merged."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    payload_dir = tmp_path / "payloads"
    _seed_routing_state(db_url)
    store = FilesystemPayloadStore(payload_dir)
    reason_ref = store.store(b'{"condition":"row[\'route\'] == \'accepted\'","result":"true"}')
    for index, edge_id in enumerate((EDGE_ID, EDGE_B_ID)):
        _insert_legacy_routing_group(
            db_url,
            group_id=f"legacy-ambiguous-group-{index}",
            event_prefix=f"legacy-ambiguous-event-{index}",
            edge_ids=(edge_id,),
            reason_hash=stable_hash(REASON),
            reason_ref=reason_ref,
        )

    with LandscapeDB.from_url(db_url, create_tables=False) as reopened:
        factory = RecorderFactory(reopened, payload_store=store)
        with pytest.raises(AuditIntegrityError, match="multiple durable routing groups"):
            factory.execution.record_routing_event(STATE_ID, EDGE_ID, RoutingMode.MOVE, reason=REASON)
        assert len(factory.query.get_routing_events(STATE_ID)) == 2


def test_mismatched_legacy_reason_ref_is_rejected_without_append(tmp_path: Path) -> None:
    """Store-first retry never bypasses a dangling legacy explanation ref."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    payload_dir = tmp_path / "payloads"
    _seed_routing_state(db_url)
    store = FilesystemPayloadStore(payload_dir)
    _insert_legacy_routing_group(
        db_url,
        group_id="legacy-dangling-group",
        event_prefix="legacy-dangling-event",
        edge_ids=(EDGE_ID,),
        reason_hash=stable_hash(REASON),
        reason_ref="f" * 64,
    )

    with LandscapeDB.from_url(db_url, create_tables=False) as reopened:
        factory = RecorderFactory(reopened, payload_store=store)
        with pytest.raises(AuditIntegrityError, match="durable event differs"):
            factory.execution.record_routing_event(STATE_ID, EDGE_ID, RoutingMode.MOVE, reason=REASON)
        durable = factory.query.get_routing_events(STATE_ID)
        assert len(durable) == 1
        assert durable[0].event_id == "legacy-dangling-event-0"
        assert durable[0].reason_ref == "f" * 64


def test_journal_captures_final_reason_ref_on_insert_without_update(tmp_path: Path) -> None:
    """Backup journal sees one self-contained event write, never a repair UPDATE."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    journal_path = tmp_path / "journal.jsonl"
    payload_dir = tmp_path / "payloads"
    _seed_routing_state(db_url)

    with LandscapeDB.from_url(
        db_url,
        create_tables=False,
        dump_to_jsonl=True,
        dump_to_jsonl_path=str(journal_path),
    ) as db:
        factory = RecorderFactory(db, payload_store=FilesystemPayloadStore(payload_dir))
        event = factory.execution.record_routing_event(STATE_ID, EDGE_ID, RoutingMode.MOVE, reason=REASON)

    records = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    routing_writes = [record for record in records if "ROUTING_EVENTS" in record["statement"].upper()]
    assert len(routing_writes) == 1
    assert routing_writes[0]["statement"].lstrip().upper().startswith("INSERT INTO ROUTING_EVENTS")
    assert not any(record["statement"].lstrip().upper().startswith("UPDATE ROUTING_EVENTS") for record in records)
    assert event.reason_ref is not None
    assert event.reason_ref in json.dumps(routing_writes[0]["parameters"])


def test_sqlite_backup_restore_preserves_exported_reason_and_retry_identity(tmp_path: Path) -> None:
    """A real database+payload restore preserves exact evidence and idempotency."""
    source_db_path = tmp_path / "source.db"
    source_payload_dir = tmp_path / "source-payloads"
    db_url = f"sqlite:///{source_db_path}"
    _seed_routing_state(db_url)
    source_store = FilesystemPayloadStore(source_payload_dir)

    with LandscapeDB.from_url(db_url, create_tables=False) as db:
        event = RecorderFactory(db, payload_store=source_store).execution.record_routing_event(
            STATE_ID,
            EDGE_ID,
            RoutingMode.MOVE,
            reason=REASON,
        )
        source_export = [record for record in LandscapeExporter(db).export_run(RUN_ID) if record["record_type"] == "routing_event"]

    backup_path = tmp_path / "audit.backup.db"
    with sqlite3.connect(source_db_path) as source, sqlite3.connect(backup_path) as backup:
        source.backup(backup)

    restored_db_path = tmp_path / "restored.db"
    restored_payload_dir = tmp_path / "restored-payloads"
    shutil.copy2(backup_path, restored_db_path)
    shutil.copytree(source_payload_dir, restored_payload_dir)

    restored_store = FilesystemPayloadStore(restored_payload_dir)
    with LandscapeDB.from_url(f"sqlite:///{restored_db_path}", create_tables=False) as restored_db:
        restored_export = [
            record for record in LandscapeExporter(restored_db).export_run(RUN_ID) if record["record_type"] == "routing_event"
        ]
        assert restored_export == source_export
        assert event.reason_ref is not None
        assert restored_store.retrieve(event.reason_ref) == b'{"condition":"row[\'route\'] == \'accepted\'","result":"true"}'

        retried = RecorderFactory(restored_db, payload_store=restored_store).execution.record_routing_event(
            STATE_ID,
            EDGE_ID,
            RoutingMode.MOVE,
            reason=REASON,
        )
        assert retried.event_id == event.event_id
        assert retried.reason_ref == event.reason_ref
        with restored_db.read_only_connection() as conn:
            assert len(conn.execute(select(routing_events_table)).all()) == 1


def test_without_payload_store_records_hash_only(tmp_path: Path) -> None:
    """No payload backend is represented honestly as hash-only evidence."""
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    _seed_routing_state(db_url)

    with LandscapeDB.from_url(db_url, create_tables=False) as db:
        event = RecorderFactory(db).execution.record_routing_event(STATE_ID, EDGE_ID, RoutingMode.MOVE, reason=REASON)
        assert event.reason_hash == stable_hash(REASON)
        assert event.reason_ref is None
