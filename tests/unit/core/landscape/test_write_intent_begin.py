"""Write-intent BEGIN IMMEDIATE discipline (option-c slice 1, ADR-030 §D5).

Pins the transaction-begin mechanism installed by
``LandscapeDB._configure_sqlite`` on writable SQLite engines:

- Transactions carrying the :data:`WRITE_INTENT_OPTION` execution option
  (``LandscapeDB.write_connection()`` / ``begin_write(engine)``) begin with
  ``BEGIN IMMEDIATE`` — the WAL write lock is taken AT BEGIN, so cross-process
  read-then-write transactions can never abort mid-flight with the
  non-retryable ``SQLITE_BUSY_SNAPSHOT``.
- Read transactions (``connection()``) and direct ``engine.begin()`` keep
  DEFERRED semantics: an explicit lock-free ``BEGIN``, never IMMEDIATE.
  ``connection()`` additionally rejects DML; callers must use
  ``write_connection()`` when they intend to write.
- Read-only engines (``from_url(read_only=True)``) keep stock pysqlite
  autocommit-read behaviour and never emit any BEGIN at all (F10 closure:
  dashboard reads never contend for the write lock).

Statement-level proof goes through the raw sqlite3 trace callback (what the
DBAPI actually executed, including anything pysqlite might emit on its own)
and, for the in-memory/SQLCipher variants, a SQLAlchemy
``before_cursor_execute`` spy.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event, insert, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from elspeth.contracts import NodeType
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import (
    WRITE_INTENT_OPTION,
    LandscapeDB,
    begin_write,
    verify_sqlite_tier1_pragmas,
)
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    SQLITE_SCHEMA_EPOCH,
    nodes_table,
    rows_table,
    runs_table,
    tokens_table,
)

BASE = datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC)
RUN_ID = "run-write-intent"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attach_trace(engine: Engine) -> list[str]:
    """Attach a raw sqlite3 trace callback to the engine's pooled connection.

    Checks out the (single, sequential-use) pooled DBAPI connection, installs
    the trace callback, and returns it to the pool. Subsequent checkouts in
    this thread reuse the same DBAPI connection, so every statement SQLite
    actually executes lands in the returned list.
    """
    captured: list[str] = []
    raw = engine.raw_connection()
    try:
        driver_conn = raw.driver_connection
        assert driver_conn is not None
        driver_conn.set_trace_callback(captured.append)
    finally:
        raw.close()
    return captured


def _attach_statement_spy(engine: Engine) -> list[str]:
    """SQLAlchemy-boundary spy: collect statements via before_cursor_execute."""
    captured: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _spy(
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        captured.append(statement)

    return captured


def _run_values(run_id: str = RUN_ID) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": BASE,
        "config_hash": "config",
        "settings_json": "{}",
        "canonical_version": "v1",
        "status": "running",
        "openrouter_catalog_sha256": "0" * 64,
        "openrouter_catalog_source": "bundled",
    }


def _open_file_db(tmp_path: Path, **kwargs: Any) -> LandscapeDB:
    return LandscapeDB.from_url(f"sqlite:///{tmp_path / 'landscape.db'}", **kwargs)


def _begin_statements(trace: list[str]) -> list[str]:
    return [s.strip() for s in trace if s.strip().upper().startswith("BEGIN")]


# ---------------------------------------------------------------------------
# (a) Begin-mode proofs at the SQLite boundary
# ---------------------------------------------------------------------------


class TestBeginMode:
    def test_write_connection_begins_with_begin_immediate(self, tmp_path: Path) -> None:
        db = _open_file_db(tmp_path)
        try:
            trace = _attach_trace(db.engine)
            with db.write_connection() as conn:
                conn.execute(insert(runs_table).values(**_run_values()))

            begins = _begin_statements(trace)
            # Exactly ONE begin — the listener's BEGIN IMMEDIATE.  A second
            # BEGIN would mean pysqlite's implicit begin survived the
            # isolation_level=None takeover ("cannot start a transaction
            # within a transaction" hazard).
            assert begins == ["BEGIN IMMEDIATE"]
            # BEGIN IMMEDIATE executed before the INSERT, and the txn committed.
            immediate_idx = next(i for i, s in enumerate(trace) if s.strip() == "BEGIN IMMEDIATE")
            insert_idx = next(i for i, s in enumerate(trace) if s.lstrip().upper().startswith("INSERT"))
            assert immediate_idx < insert_idx
            assert any(s.strip().upper() == "COMMIT" for s in trace)
        finally:
            db.close()

    def test_plain_connection_emits_deferred_begin_never_immediate(self, tmp_path: Path) -> None:
        db = _open_file_db(tmp_path)
        try:
            trace = _attach_trace(db.engine)
            with db.connection() as conn:
                conn.execute(select(runs_table.c.run_id)).fetchall()

            begins = _begin_statements(trace)
            # Owned slice-1 delta: reads on writable engines now begin with an
            # explicit DEFERRED BEGIN (lock-free) where pysqlite previously
            # emitted nothing.  Never IMMEDIATE.
            assert begins == ["BEGIN"]
        finally:
            db.close()

    def test_plain_connection_rejects_writes_on_writable_handle(self, tmp_path: Path) -> None:
        db = _open_file_db(tmp_path)
        try:
            with pytest.raises(OperationalError, match="attempt to write a readonly database"), db.connection() as conn:
                conn.execute(insert(runs_table).values(**_run_values()))
        finally:
            db.close()

    def test_begin_write_engine_helper_parity(self, tmp_path: Path) -> None:
        db = _open_file_db(tmp_path)
        try:
            trace = _attach_trace(db.engine)
            with begin_write(db.engine) as conn:
                conn.execute(insert(runs_table).values(**_run_values()))

            assert _begin_statements(trace) == ["BEGIN IMMEDIATE"]
            # Committed and visible on a fresh transaction.
            with db.connection() as conn:
                rows = conn.execute(select(runs_table.c.run_id)).fetchall()
            assert [row[0] for row in rows] == [RUN_ID]
        finally:
            db.close()

    def test_in_memory_static_pool_write_connection(self) -> None:
        db = LandscapeDB.in_memory()
        try:
            statements = _attach_statement_spy(db.engine)
            with db.write_connection() as conn:
                conn.execute(insert(runs_table).values(**_run_values()))

            assert "BEGIN IMMEDIATE" in statements
            # StaticPool path: the commit stuck and the row is visible.
            with db.connection() as conn:
                rows = conn.execute(select(runs_table.c.run_id)).fetchall()
            assert [row[0] for row in rows] == [RUN_ID]
        finally:
            db.close()

    def test_scheduler_claim_ready_emits_begin_immediate(self, tmp_path: Path) -> None:
        """Smoke the design's named BUSY_SNAPSHOT hazard verb end-to-end."""
        db = _open_file_db(tmp_path)
        try:
            with db.write_connection() as conn:
                conn.execute(insert(runs_table).values(**_run_values()))
                conn.execute(
                    insert(nodes_table).values(
                        run_id=RUN_ID,
                        node_id="normalize",
                        plugin_name="identity",
                        node_type=NodeType.TRANSFORM.value,
                        plugin_version="1.0",
                        determinism="deterministic",
                        config_hash="config",
                        config_json="{}",
                        registered_at=BASE,
                    )
                )
                conn.execute(
                    insert(rows_table).values(
                        row_id="row-0",
                        run_id=RUN_ID,
                        source_node_id="normalize",
                        row_index=0,
                        source_row_index=0,
                        ingest_sequence=0,
                        source_data_hash="hash-row-0",
                        created_at=BASE,
                    )
                )
                conn.execute(insert(tokens_table).values(token_id="token-0", row_id="row-0", run_id=RUN_ID, created_at=BASE))

            repo = TokenSchedulerRepository(db.engine)
            payload = TokenSchedulerRepository.serialize_row_payload(
                PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
            )
            repo.enqueue_ready(
                run_id=RUN_ID,
                token_id="token-0",
                row_id="row-0",
                node_id="normalize",
                step_index=1,
                ingest_sequence=0,
                available_at=BASE,
                row_payload_json=payload,
            )

            trace = _attach_trace(db.engine)
            claimed = repo.claim_ready(run_id=RUN_ID, lease_owner="worker-1", lease_seconds=30, now=BASE)
            assert claimed is not None
            assert "BEGIN IMMEDIATE" in _begin_statements(trace)

            # heartbeat_lease (manual conn.begin() shape) carries intent too.
            trace2 = _attach_trace(db.engine)
            repo.heartbeat_lease(
                run_id=RUN_ID,
                work_item_id=claimed.work_item_id,
                lease_owner="worker-1",
                lease_seconds=30,
                now=BASE,
                membership_fenced=False,
            )
            assert "BEGIN IMMEDIATE" in _begin_statements(trace2)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# (b) Lock semantics: write lock taken AT BEGIN, reads never hold it
# ---------------------------------------------------------------------------


class TestLockAtBegin:
    def test_write_lock_taken_at_begin_before_any_statement(self, tmp_path: Path) -> None:
        db = _open_file_db(tmp_path)
        db_path = tmp_path / "landscape.db"
        try:
            with db.write_connection():
                # ZERO statements executed in this transaction — the write
                # lock must already be held because IMMEDIATE was emitted at
                # begin, not lazily at the first write.
                probe = sqlite3.connect(str(db_path), isolation_level=None)
                try:
                    probe.execute("PRAGMA busy_timeout=0")
                    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                        probe.execute("BEGIN IMMEDIATE")
                finally:
                    probe.close()
        finally:
            db.close()

    def test_plain_read_transaction_does_not_hold_write_lock(self, tmp_path: Path) -> None:
        db = _open_file_db(tmp_path)
        db_path = tmp_path / "landscape.db"
        try:
            with db.connection() as conn:
                conn.execute(select(runs_table.c.run_id)).fetchall()
                # Mid-read-transaction, an external writer can take the write
                # lock immediately: DEFERRED BEGIN took no lock, and WAL
                # readers never block the writer.
                probe = sqlite3.connect(str(db_path), isolation_level=None)
                try:
                    probe.execute("PRAGMA busy_timeout=0")
                    probe.execute("BEGIN IMMEDIATE")
                    probe.execute("ROLLBACK")
                finally:
                    probe.close()
        finally:
            db.close()

    def test_two_write_intent_transactions_serialize(self, tmp_path: Path) -> None:
        """Two concurrent write-intent transactions serialize at BEGIN.

        Neither errors with BUSY_SNAPSHOT (impossible by construction — the
        snapshot is taken with the write lock already held) nor with
        "database is locked" (the second BEGIN IMMEDIATE polls inside the
        5000 ms busy_timeout window until the first commits).
        """
        db = _open_file_db(tmp_path)
        first_in_txn = threading.Event()
        order: list[str] = []
        errors: list[BaseException] = []

        def second_writer() -> None:
            try:
                assert first_in_txn.wait(timeout=10)
                with db.write_connection() as conn:
                    order.append("second-begin")
                    conn.execute(insert(runs_table).values(**_run_values("run-second")))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=second_writer)
        thread.start()
        try:
            with db.write_connection() as conn:
                conn.execute(insert(runs_table).values(**_run_values("run-first")))
                first_in_txn.set()
                # Give the second writer time to block on its BEGIN IMMEDIATE
                # while we still hold the write lock.
                time.sleep(0.3)
                order.append("first-commit")
            thread.join(timeout=10)
            assert not thread.is_alive()
            assert errors == []
            assert order == ["first-commit", "second-begin"]
            with db.connection() as conn:
                rows = conn.execute(select(runs_table.c.run_id).order_by(runs_table.c.run_id)).fetchall()
            assert [row[0] for row in rows] == ["run-first", "run-second"]
        finally:
            thread.join(timeout=10)
            db.close()

    def test_deferred_read_then_write_hazard_is_what_immediate_eliminates(self, tmp_path: Path) -> None:
        """Document the hazard (raw DEFERRED control): BUSY_SNAPSHOT is
        immediate and non-retryable — busy_timeout does NOT apply to it.

        A raw DEFERRED transaction reads, a peer commits a write, then the
        reader tries to upgrade to writer: SQLite returns BUSY_SNAPSHOT
        (surfaced as "database is locked") without consulting the busy
        handler. ``write_connection()`` makes this shape impossible because
        the write lock is held from BEGIN.
        """
        db = _open_file_db(tmp_path)
        db_path = str(tmp_path / "landscape.db")
        try:
            with db.write_connection() as conn:
                conn.execute(insert(runs_table).values(**_run_values()))

            reader = sqlite3.connect(db_path, isolation_level=None)
            peer = sqlite3.connect(db_path, isolation_level=None)
            try:
                reader.execute("PRAGMA busy_timeout=5000")
                reader.execute("BEGIN")  # DEFERRED
                reader.execute("SELECT count(*) FROM runs").fetchone()  # snapshot established

                peer.execute("UPDATE runs SET status = 'completed'")  # autocommit write

                started = time.monotonic()
                with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                    reader.execute("UPDATE runs SET status = 'failed'")
                # Non-retryable: returned immediately, NOT after the 5 s
                # busy_timeout poll.
                assert time.monotonic() - started < 2.0
                reader.execute("ROLLBACK")
            finally:
                reader.close()
                peer.close()
        finally:
            db.close()


# ---------------------------------------------------------------------------
# (b2) StaticPool shared-connection serialization
# ---------------------------------------------------------------------------


class TestStaticPoolConnectionSerialization:
    """StaticPool engines share ONE DBAPI connection across all threads.

    ``LandscapeDB.in_memory()`` (tests only) uses ``StaticPool`` +
    ``check_same_thread=False``: every thread that opens a connection drives the
    SAME underlying SQLite connection.  The idle-timeout aggregation poller
    (``source_iteration.py``) runs audit writes on a helper thread CONCURRENTLY
    with the main source thread, which may itself write audit rows during
    ``next()`` (e.g. ``ctx.record_call``).  Without app-level serialization the
    two threads drive one connection at once and SQLite raises "recursive use of
    cursors not allowed" / "cannot start a transaction within a transaction".

    File-backed production engines use the default ``QueuePool`` (one connection
    per thread) + WAL/BEGIN IMMEDIATE/busy_timeout, so they are already safe and
    take the no-op (lock-free) path; this serialization engages ONLY for
    StaticPool engines.
    """

    def test_concurrent_write_connections_on_static_pool_do_not_collide(self) -> None:
        """Two threads driving write_connection() on one StaticPool engine must
        serialize, not collide on the shared connection.

        Each thread holds its transaction open across a short sleep to force the
        windows to overlap.  Pre-fix the second thread drives the shared
        connection while the first is mid-transaction -> ProgrammingError
        ("recursive use of cursors" / nested transaction).  Post-fix the
        per-engine StaticPool lock serializes them so both commit cleanly.
        """
        db = LandscapeDB.in_memory()
        start_barrier = threading.Barrier(2)
        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        def writer(run_id: str) -> None:
            try:
                start_barrier.wait(timeout=10)
                with db.write_connection() as conn:
                    conn.execute(insert(runs_table).values(**_run_values(run_id)))
                    # Hold the transaction open to widen the overlap window so the
                    # shared-connection collision is deterministic without the lock.
                    time.sleep(0.05)
            except BaseException as exc:
                with errors_lock:
                    errors.append(exc)

        try:
            threads = [threading.Thread(target=writer, args=(f"run-static-{i}",)) for i in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
                assert not thread.is_alive()

            assert errors == [], f"concurrent StaticPool writes collided: {errors!r}"
            with db.connection() as conn:
                rows = conn.execute(select(runs_table.c.run_id).order_by(runs_table.c.run_id)).fetchall()
            assert [row[0] for row in rows] == ["run-static-0", "run-static-1"]
        finally:
            db.close()

    def test_tier1_pragma_probe_waits_for_static_pool_write_transaction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tier-1 verification must not drive a shared connection mid-write."""
        from elspeth.core.landscape import database as landscape_database

        db = LandscapeDB.in_memory()
        write_started = threading.Event()
        release_write = threading.Event()
        probe_lock_contended = threading.Event()
        probe_finished = threading.Event()
        probe_progressed = threading.Event()
        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        def record_error(exc: BaseException) -> None:
            with errors_lock:
                errors.append(exc)

        def writer() -> None:
            try:
                with db.write_connection():
                    write_started.set()
                    assert release_write.wait(timeout=10)
            except BaseException as exc:
                record_error(exc)

        def probe() -> None:
            try:
                verify_sqlite_tier1_pragmas(db.engine, owner="concurrent probe")
            except BaseException as exc:
                record_error(exc)
            finally:
                probe_finished.set()
                probe_progressed.set()

        writer_thread = threading.Thread(target=writer)
        probe_thread = threading.Thread(target=probe)
        writer_thread_started = False
        probe_thread_started = False
        try:
            writer_thread.start()
            writer_thread_started = True
            assert write_started.wait(timeout=10)

            real_lock = landscape_database._shared_connection_lock(db.engine)
            assert real_lock is not None

            class RecordingLock:
                def __enter__(self) -> None:
                    acquired = real_lock.acquire(blocking=False)
                    if acquired:
                        real_lock.release()
                        raise AssertionError("writer did not hold the StaticPool lock")
                    probe_lock_contended.set()
                    probe_progressed.set()
                    real_lock.acquire()

                def __exit__(self, *_args: Any) -> None:
                    real_lock.release()

            original_shared_connection_lock = landscape_database._shared_connection_lock

            def recording_shared_connection_lock(engine: Engine) -> Any:
                if engine is db.engine:
                    return RecordingLock()
                return original_shared_connection_lock(engine)

            monkeypatch.setattr(landscape_database, "_shared_connection_lock", recording_shared_connection_lock)

            probe_thread.start()
            probe_thread_started = True
            assert probe_progressed.wait(timeout=10)
            assert probe_lock_contended.is_set(), f"Tier-1 probe bypassed the held StaticPool lock: {errors!r}"
            assert not probe_finished.is_set()

            release_write.set()
        finally:
            release_write.set()
            if writer_thread_started:
                writer_thread.join(timeout=10)
            if probe_thread_started:
                probe_thread.join(timeout=10)
            db.close()

        assert not writer_thread.is_alive()
        assert not probe_thread.is_alive()
        assert probe_finished.is_set()
        assert errors == []

    def test_file_backed_engine_takes_no_static_pool_lock(self, tmp_path: Path) -> None:
        """Production (file-backed QueuePool) engines must NOT be given the
        StaticPool serialization lock — they rely on per-thread connections +
        WAL and must keep the epoch-21 multi-writer concurrency unchanged.
        """
        from elspeth.core.landscape.database import _shared_connection_lock

        db = _open_file_db(tmp_path)
        try:
            assert _shared_connection_lock(db.engine) is None
        finally:
            db.close()

    def test_in_memory_engine_has_a_static_pool_lock(self) -> None:
        """In-memory StaticPool engines DO get a (stable, per-engine) lock."""
        from elspeth.core.landscape.database import _shared_connection_lock

        db = LandscapeDB.in_memory()
        try:
            lock = _shared_connection_lock(db.engine)
            assert lock is not None
            # Stable across calls — the same engine yields the same lock object.
            assert _shared_connection_lock(db.engine) is lock
        finally:
            db.close()


# ---------------------------------------------------------------------------
# (c) Read-only engines: F10 closure
# ---------------------------------------------------------------------------


class TestReadOnlyEngine:
    def test_read_only_engine_never_emits_any_begin(self, tmp_path: Path) -> None:
        writable = _open_file_db(tmp_path)
        with writable.write_connection() as conn:
            conn.execute(insert(runs_table).values(**_run_values()))
        writable.close()

        ro = _open_file_db(tmp_path, read_only=True, create_tables=False)
        try:
            trace = _attach_trace(ro.engine)
            # PRAGMA query_only=ON applied by the connect hook.
            with ro.engine.connect() as conn:
                assert conn.exec_driver_sql("PRAGMA query_only").scalar_one() == 1
            with ro.connection() as conn:  # auto-routes to read_only_connection
                rows = conn.execute(select(runs_table.c.run_id)).fetchall()
            assert [row[0] for row in rows] == [RUN_ID]
            # On read-only engines the finally block must NOT disarm
            # query_only: it is the only write barrier for SQLCipher
            # read-only opens, which have no mode=ro file backstop.
            with ro.engine.connect() as conn:
                assert conn.exec_driver_sql("PRAGMA query_only").scalar_one() == 1
            # Stock pysqlite autocommit reads: provably no BEGIN of any kind.
            assert _begin_statements(trace) == []
        finally:
            ro.close()

    def test_write_connection_on_read_only_handle_raises(self, tmp_path: Path) -> None:
        writable = _open_file_db(tmp_path)
        writable.close()
        ro = _open_file_db(tmp_path, read_only=True, create_tables=False)
        try:
            with pytest.raises(RuntimeError, match="read-only"), ro.write_connection():
                pass  # pragma: no cover — must not be reached
        finally:
            ro.close()

    def test_recorder_factory_on_read_only_handle_skips_scheduler(self, tmp_path: Path) -> None:
        """Read-only handles get no scheduler repository (pure write surface).

        The scheduler constructor's Tier-1 WAL probe must never run against a
        read-only open: immutable snapshot opens legitimately report
        ``journal_mode=delete``, and a read-only engine could not satisfy any
        scheduler write verb anyway. Access fails loudly instead.
        """
        from elspeth.core.landscape.factory import RecorderFactory

        writable = _open_file_db(tmp_path)
        writable.close()
        ro = _open_file_db(tmp_path, read_only=True, create_tables=False)
        try:
            assert ro.is_read_only is True
            factory = RecorderFactory(ro)
            with pytest.raises(RuntimeError, match="read-only"):
                _ = factory.scheduler
        finally:
            ro.close()

        # Control: a writable handle still constructs the scheduler eagerly.
        writable = _open_file_db(tmp_path)
        try:
            assert writable.is_read_only is False
            assert RecorderFactory(writable).scheduler is not None
        finally:
            writable.close()

    def test_recorder_factory_read_only_surface_excludes_write_repositories(self, tmp_path: Path) -> None:
        """Read-only construction exposes only read-capable repository ports."""
        from elspeth.core.landscape.factory import LandscapeReadRepositories, RecorderFactory

        writable = _open_file_db(tmp_path)
        writable.close()
        ro = _open_file_db(tmp_path, read_only=True, create_tables=False)
        try:
            read_repos = RecorderFactory.read_only(ro)
            assert isinstance(read_repos, LandscapeReadRepositories)
            assert read_repos.run_lifecycle.get_run("missing-run") is None
            assert read_repos.query.get_rows("missing-run") == []
            for write_attr in ("auth_audit", "scheduler", "run_coordination", "plugin_audit_writer"):
                assert not hasattr(read_repos, write_attr)
        finally:
            ro.close()

    def test_recorder_factory_read_only_constructor_does_not_build_write_ops(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Read-only construction must not route through the writable ops helper."""
        from elspeth.core.landscape.factory import RecorderFactory

        writable = _open_file_db(tmp_path)
        writable.close()
        ro = _open_file_db(tmp_path, read_only=True, create_tables=False)

        def fail_database_ops(*args: object, **kwargs: object) -> object:
            raise AssertionError("read_only() must not construct DatabaseOps")

        monkeypatch.setattr("elspeth.core.landscape.factory.DatabaseOps", fail_database_ops)
        try:
            read_repos = RecorderFactory.read_only(ro)
            assert read_repos.query.get_rows("missing-run") == []
        finally:
            ro.close()

    def test_recorder_factory_writable_constructor_rejects_read_only_handle(self, tmp_path: Path) -> None:
        """Writable construction is explicit and fails before exposing write ports."""
        from elspeth.core.landscape.factory import LandscapeWriteRepositories, RecorderFactory

        writable = _open_file_db(tmp_path)
        writable.close()
        ro = _open_file_db(tmp_path, read_only=True, create_tables=False)
        try:
            with pytest.raises(RuntimeError, match="read-only"):
                RecorderFactory.writable(ro)
        finally:
            ro.close()

        writable = _open_file_db(tmp_path)
        try:
            write_repos = RecorderFactory.writable(writable)
            assert isinstance(write_repos, LandscapeWriteRepositories)
            assert write_repos.scheduler is not None
            assert write_repos.run_coordination is not None
        finally:
            writable.close()


# ---------------------------------------------------------------------------
# (d) No-behavior-change checks under the isolation_level takeover
# ---------------------------------------------------------------------------


class TestNoBehaviorChange:
    def test_write_connection_rolls_back_on_exception(self, tmp_path: Path) -> None:
        db = _open_file_db(tmp_path)
        try:
            with pytest.raises(RuntimeError, match="boom-write"), db.write_connection() as conn:
                conn.execute(insert(runs_table).values(**_run_values("run-rollback-w")))
                raise RuntimeError("boom-write")
            with db.connection() as conn:
                rows = conn.execute(select(runs_table.c.run_id)).fetchall()
            assert rows == []
        finally:
            db.close()

    def test_journal_records_writes_only_never_transaction_control(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "landscape.journal.jsonl"
        db = _open_file_db(tmp_path, dump_to_jsonl=True, dump_to_jsonl_path=str(journal_path))
        try:
            with db.write_connection() as conn:
                conn.execute(insert(runs_table).values(**_run_values()))
        finally:
            db.close()

        import json

        records = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
        assert records, "journal must contain the committed INSERT"
        statements = [record["statement"].lstrip().upper() for record in records]
        assert all(s.startswith(("INSERT", "UPDATE", "DELETE", "REPLACE")) for s in statements)
        assert not any(s.startswith(("BEGIN", "COMMIT", "ROLLBACK")) for s in statements)
        assert any(s.startswith("INSERT") for s in statements)

    def test_pragma_invariants_probe_passes_post_takeover(self, tmp_path: Path) -> None:
        url = f"sqlite:///{tmp_path / 'landscape.db'}"
        db = LandscapeDB.from_url(url)
        try:
            # from_url already ran the probe at open; re-run explicitly to pin
            # that PRAGMA reads work inside the explicit-BEGIN regime.
            LandscapeDB._verify_sqlite_pragmas(db.engine, url)
        finally:
            db.close()

    def test_schema_epoch_stamp_round_trips(self, tmp_path: Path) -> None:
        db = _open_file_db(tmp_path)
        try:
            assert db._get_sqlite_schema_epoch() == SQLITE_SCHEMA_EPOCH
            # PRAGMA user_version write inside an explicit IMMEDIATE txn is
            # legal and transactional.
            db._set_sqlite_schema_epoch(SQLITE_SCHEMA_EPOCH)
            assert db._get_sqlite_schema_epoch() == SQLITE_SCHEMA_EPOCH
        finally:
            db.close()

    def test_read_only_connection_toggle_still_returns_writable_connection(self, tmp_path: Path) -> None:
        db = _open_file_db(tmp_path)
        try:
            with db.read_only_connection() as conn, pytest.raises(Exception, match=r"query_only|readonly|attempt to write"):
                conn.execute(insert(runs_table).values(**_run_values("run-denied")))
            # Pooled connection came back writable (query_only reset OFF).
            with db.write_connection() as conn:
                conn.execute(insert(runs_table).values(**_run_values()))
            with db.connection() as conn:
                rows = conn.execute(select(runs_table.c.run_id)).fetchall()
            assert [row[0] for row in rows] == [RUN_ID]
        finally:
            db.close()

    def test_scheduler_constructor_probe_passes(self, tmp_path: Path) -> None:
        db = _open_file_db(tmp_path)
        try:
            # The Tier-1 PRAGMA probe in the constructor runs plain reads
            # under the explicit-DEFERRED-BEGIN regime; must still pass.
            TokenSchedulerRepository(db.engine)
        finally:
            db.close()

    def test_write_intent_option_constant_is_stable(self) -> None:
        # Slice-2+ surfaces (RunCoordinationRepository) key off this exact
        # string; renaming it silently would strand callers on DEFERRED.
        assert WRITE_INTENT_OPTION == "elspeth_write_intent"


class TestSQLCipherParity:
    def test_sqlcipher_write_intent_begin_immediate(self, tmp_path: Path) -> None:
        pytest.importorskip("sqlcipher3", reason="sqlcipher3 not installed (install with: uv pip install 'elspeth[security]')")
        db = LandscapeDB.from_url(f"sqlite:///{tmp_path / 'encrypted.db'}", passphrase="test-passphrase")
        try:
            statements = _attach_statement_spy(db.engine)
            with db.write_connection() as conn:
                conn.execute(insert(runs_table).values(**_run_values()))
            assert "BEGIN IMMEDIATE" in statements

            # Rollback parity on the sqlcipher3 driver (isolation_level
            # attribute mirrors pysqlite).
            with pytest.raises(RuntimeError, match="boom-cipher"), db.write_connection() as conn:
                conn.execute(insert(runs_table).values(**_run_values("run-cipher-rollback")))
                raise RuntimeError("boom-cipher")
            with db.connection() as conn:
                rows = conn.execute(select(runs_table.c.run_id)).fetchall()
            assert [row[0] for row in rows] == [RUN_ID]
        finally:
            db.close()
