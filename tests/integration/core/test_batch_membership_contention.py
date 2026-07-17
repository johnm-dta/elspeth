"""Cross-process atomicity proof for aggregation batch membership."""

from __future__ import annotations

import multiprocessing
import queue
import threading
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError

from elspeth.contracts import BatchStatus, NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.factory import RecorderFactory

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _transition_batch_worker(
    db_url: str,
    batch_id: str,
    target_status_value: str,
    transition_holds_lock: Any,
    release_transition: Any,
    results: Any,
) -> None:
    """Hold the transition transaction open after its UPDATE executes."""
    db = LandscapeDB.from_url(db_url, create_tables=False)
    factory = RecorderFactory(db)

    def pause_after_transition(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:
        if statement.lstrip().upper().startswith("UPDATE BATCHES"):
            transition_holds_lock.set()
            if not release_transition.wait(timeout=20):
                raise TimeoutError("parent did not release batch transition")

    event.listen(db.engine, "after_cursor_execute", pause_after_transition)
    try:
        target_status = BatchStatus(target_status_value)
        if target_status is BatchStatus.EXECUTING:
            factory.execution.update_batch_status(batch_id, target_status)
        else:
            factory.execution.complete_batch(batch_id, target_status)
        results.put(("transition", "committed"))
    except BaseException as exc:  # pragma: no cover - asserted in parent
        results.put(("transition", f"{type(exc).__name__}: {exc}"))
    finally:
        event.remove(db.engine, "after_cursor_execute", pause_after_transition)
        db.close()


def _add_member_worker(
    db_url: str,
    batch_id: str,
    token_id: str,
    add_reached_database: Any,
    results: Any,
) -> None:
    """Attempt membership through a separately opened process connection."""
    db = LandscapeDB.from_url(db_url, create_tables=False)
    factory = RecorderFactory(db)

    def observe_attempt(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:
        normalized = " ".join(statement.upper().split())
        if normalized.startswith("BEGIN IMMEDIATE") or (normalized.startswith("SELECT") and "FROM BATCHES" in normalized):
            add_reached_database.set()

    event.listen(db.engine, "before_cursor_execute", observe_attempt)
    try:
        factory.execution.add_batch_member(batch_id, token_id, ordinal=0)
    except AuditIntegrityError as exc:
        results.put(("member", f"refused: {exc}"))
    except BaseException as exc:  # pragma: no cover - asserted in parent
        results.put(("member", f"{type(exc).__name__}: {exc}"))
    else:
        results.put(("member", "inserted"))
    finally:
        event.remove(db.engine, "before_cursor_execute", observe_attempt)
        db.close()


def _seed_file_database(db_path: Path) -> tuple[str, str, str]:
    db_url = f"sqlite:///{db_path}"
    db = LandscapeDB.from_url(db_url)
    factory = RecorderFactory(db)
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
    db.close()
    return db_url, batch.batch_id, token.token_id


@pytest.mark.parametrize(
    "target_status",
    [BatchStatus.EXECUTING, BatchStatus.COMPLETED, BatchStatus.FAILED],
)
@pytest.mark.timeout(60)
def test_batch_transition_wins_cross_process_race_without_post_closure_membership(
    tmp_path: Path,
    target_status: BatchStatus,
) -> None:
    """Every SQLite membership-closing transition wins before a waiting add.

    The transition worker pauses after ``UPDATE batches`` while still holding
    its BEGIN IMMEDIATE transaction. The member process begins its repository
    call during that pause. Once released, it must observe the new immutable
    status and refuse the insert; the final audit image has no member row.

    EXECUTING uses ``update_batch_status``. COMPLETED and FAILED use the real
    ``complete_batch`` production path, not a raw fixture update.
    """
    db_url, batch_id, token_id = _seed_file_database(tmp_path / f"audit-{target_status.value}.db")
    ctx = multiprocessing.get_context("spawn")
    transition_holds_lock = ctx.Event()
    release_transition = ctx.Event()
    add_reached_database = ctx.Event()
    results = ctx.Queue()

    transition = ctx.Process(
        target=_transition_batch_worker,
        args=(db_url, batch_id, target_status.value, transition_holds_lock, release_transition, results),
    )
    member = ctx.Process(
        target=_add_member_worker,
        args=(db_url, batch_id, token_id, add_reached_database, results),
    )
    transition.start()
    try:
        assert transition_holds_lock.wait(timeout=20), "transition never reached its locked UPDATE"
        member.start()
        assert add_reached_database.wait(timeout=20), "member process never reached its database write boundary"
    finally:
        release_transition.set()
        for process in (transition, member):
            if process.pid is None:
                continue
            process.join(timeout=20)
            if process.is_alive():  # pragma: no cover - failure cleanup
                process.terminate()
                process.join(timeout=5)

    assert not transition.is_alive()
    assert not member.is_alive()
    assert transition.exitcode == 0
    assert member.exitcode == 0

    observed: dict[str, str] = {}
    for _ in range(2):
        try:
            actor, outcome = results.get(timeout=5)
        except queue.Empty as exc:  # pragma: no cover - assertion diagnostic
            raise AssertionError(f"worker result missing; observed={observed!r}") from exc
        observed[actor] = outcome
    assert observed["transition"] == "committed"
    assert observed["member"].startswith("refused: ")
    assert f"status {target_status.value!r}" in observed["member"]

    verify_db = LandscapeDB.from_url(db_url, create_tables=False)
    verify = RecorderFactory(verify_db)
    try:
        batch = verify.execution.get_batch(batch_id)
        assert batch is not None
        assert batch.status is target_status
        assert verify.execution.get_batch_members(batch_id) == []
    finally:
        verify_db.close()


@pytest.mark.timeout(30)
def test_public_read_connection_cannot_bypass_write_transaction_policy_during_closure(tmp_path: Path) -> None:
    """A caller-owned public read connection cannot become a deferred writer.

    The peer closes the batch immediately before the guarded insert.  The
    caller then attempts to pass ``LandscapeDB.connection()`` into the write
    repository directly, reproducing the former escape hatch.  The read-only
    connection must reject the write at the database boundary and leave no
    membership row; callers that need to compose writes must explicitly use
    ``write_connection()`` and its eager SQLite write intent.
    """
    db_url, batch_id, token_id = _seed_file_database(tmp_path / "deferred.db")
    caller_db = LandscapeDB.from_url(db_url, create_tables=False)
    peer_db = LandscapeDB.from_url(db_url, create_tables=False)
    caller = RecorderFactory(caller_db)
    peer = RecorderFactory(peer_db)
    start_transition = threading.Event()
    transition_done = threading.Event()
    coordinated = threading.Event()
    peer_errors: list[BaseException] = []

    def transition() -> None:
        if not start_transition.wait(timeout=10):
            peer_errors.append(TimeoutError("membership attempt did not request transition"))
            transition_done.set()
            return
        try:
            peer.execution.update_batch_status(batch_id, BatchStatus.EXECUTING)
        except BaseException as exc:  # pragma: no cover - asserted below
            peer_errors.append(exc)
        finally:
            transition_done.set()

    def commit_peer_transition() -> None:
        if coordinated.is_set():
            return
        coordinated.set()
        start_transition.set()
        if not transition_done.wait(timeout=10):
            raise TimeoutError("peer batch transition did not commit")

    def after_caller_statement(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:
        normalized = " ".join(statement.upper().split())
        if normalized.startswith("SELECT") and "FROM BATCHES" in normalized:
            commit_peer_transition()

    def before_caller_statement(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:
        normalized = " ".join(statement.upper().split())
        if normalized.startswith("INSERT INTO BATCH_MEMBERS"):
            commit_peer_transition()

    worker = threading.Thread(target=transition, name="deferred-membership-peer")
    event.listen(caller_db.engine, "after_cursor_execute", after_caller_statement)
    event.listen(caller_db.engine, "before_cursor_execute", before_caller_statement)
    worker.start()
    try:
        with (
            caller_db.connection() as conn,
            pytest.raises(
                LandscapeRecordError,
                match=r"database rejected audit write: OperationalError",
            ) as exc_info,
        ):
            caller.execution.add_batch_member(batch_id, token_id, ordinal=0, conn=conn)
    finally:
        start_transition.set()
        worker.join(timeout=10)
        event.remove(caller_db.engine, "before_cursor_execute", before_caller_statement)
        event.remove(caller_db.engine, "after_cursor_execute", after_caller_statement)

    assert isinstance(exc_info.value.__cause__, OperationalError)
    assert "attempt to write a readonly database" in str(exc_info.value.__cause__)
    assert not worker.is_alive()
    assert peer_errors == []
    assert coordinated.is_set(), "test did not exercise the synchronized closure window"
    batch = caller.execution.get_batch(batch_id)
    assert batch is not None and batch.status is BatchStatus.EXECUTING
    assert caller.execution.get_batch_members(batch_id) == []
    caller_db.close()
    peer_db.close()
