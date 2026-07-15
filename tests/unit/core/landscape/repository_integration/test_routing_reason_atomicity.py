"""Crash, contention, and restore proofs for atomic routing reasons."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from sqlalchemy import select

from elspeth.contracts import NodeType, RoutingMode
from elspeth.contracts.errors import ConfigGateReason
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
EDGE_ID = "edge-0"
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
