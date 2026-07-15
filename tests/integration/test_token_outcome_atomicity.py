"""Real SQLite concurrency proofs for token-outcome invariant recording."""

from __future__ import annotations

import multiprocessing
import sqlite3
from pathlib import Path
from queue import Empty
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.engine import Connection

from elspeth.contracts import ExecutionError, NodeStateStatus, NodeType
from elspeth.contracts.audit import DISCARD_SINK_NAME, TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import node_states_table, token_outcomes_table
from tests.fixtures.landscape import register_test_node


def _try_complete_node_state(
    db_path: str,
    state_id: str,
    start: Any,
    result: Any,
) -> None:
    """Try the old validation-window mutation from a fresh child connection."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=200")
    try:
        if not start.wait(timeout=5):
            result.put("start-timeout")
            return
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE node_states SET status = ? WHERE state_id = ?",
                (NodeStateStatus.COMPLETED.value, state_id),
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            if conn.in_transaction:
                conn.rollback()
            if "locked" not in str(exc).lower():
                result.put(f"unexpected:{type(exc).__name__}:{exc}")
                return
            result.put("blocked")
            return
        result.put("mutated")
    finally:
        conn.close()


def _build_discard_candidate(db_path: Path) -> tuple[LandscapeDB, RecorderFactory, str, str, str]:
    db = LandscapeDB(f"sqlite:///{db_path}")
    factory = RecorderFactory(db)
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
    sink_id = register_test_node(factory.data_flow, run.run_id, "sink", node_type=NodeType.SINK, plugin_name="sink")
    row = factory.data_flow.create_row(
        run_id=run.run_id,
        source_node_id=source_id,
        row_index=0,
        data={"value": 1},
        source_row_index=0,
        ingest_sequence=0,
    )
    token = factory.data_flow.create_token(row.row_id)
    state = factory.execution.begin_node_state(
        token_id=token.token_id,
        node_id=sink_id,
        run_id=run.run_id,
        step_index=0,
        input_data={"value": 1},
    )
    factory.execution.complete_node_state(
        state_id=state.state_id,
        status=NodeStateStatus.FAILED,
        error=ExecutionError(exception="discard", exception_type="TestDiscard", phase="sink_write"),
        duration_ms=1.0,
    )
    return db, factory, run.run_id, token.token_id, state.state_id


def test_sqlite_process_cannot_mutate_node_state_between_validation_and_insert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "outcome-atomicity.db"
    db, factory, run_id, token_id, state_id = _build_discard_candidate(db_path)
    outcomes = factory.data_flow.outcomes
    original_invariants = outcomes._validate_cross_table_invariants
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    result = ctx.Queue()
    process = ctx.Process(target=_try_complete_node_state, args=(str(db_path), state_id, start, result))
    process.start()

    def pause_after_validation(
        ref: TokenRef,
        outcome: TerminalOutcome | None,
        path: TerminalPath,
        *,
        sink_name: str | None,
        sink_node_id: str | None,
        artifact_id: str | None,
        conn: Connection | None = None,
    ) -> None:
        original_invariants(
            ref,
            outcome,
            path,
            sink_name=sink_name,
            sink_node_id=sink_node_id,
            artifact_id=artifact_id,
            conn=conn,
        )
        start.set()
        try:
            mutation_result = result.get(timeout=5)
        except Empty as exc:
            raise AssertionError("child mutation did not report before timeout") from exc
        assert mutation_result == "blocked"

    monkeypatch.setattr(outcomes, "_validate_cross_table_invariants", pause_after_validation)
    try:
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id=run_id),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.SINK_DISCARDED,
            sink_name=DISCARD_SINK_NAME,
            error_hash="discard-error",
        )
    finally:
        start.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0

    with db.read_only_connection() as conn:
        status = conn.execute(select(node_states_table.c.status).where(node_states_table.c.state_id == state_id)).scalar_one()
        outcomes_count = conn.execute(select(token_outcomes_table.c.outcome_id).where(token_outcomes_table.c.token_id == token_id)).all()
    assert status == NodeStateStatus.FAILED.value
    assert len(outcomes_count) == 1


def test_sqlite_writer_lock_contention_uses_landscape_error_taxonomy(tmp_path: Path) -> None:
    db_path = tmp_path / "outcome-lock-contention.db"
    db, factory, run_id, token_id, _state_id = _build_discard_candidate(db_path)

    # The PRAGMA is connection-local and persists when SQLAlchemy returns this
    # DBAPI connection to the pool, keeping the regression fast while a raw
    # connection holds SQLite's single writer slot.
    with db.engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA busy_timeout=50")

    holder = sqlite3.connect(db_path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(LandscapeRecordError, match=r"transaction boundary.*OperationalError") as exc_info:
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.SINK_DISCARDED,
                sink_name=DISCARD_SINK_NAME,
                error_hash="discard-error",
            )
    finally:
        holder.rollback()
        holder.close()

    assert "database is locked" in str(exc_info.value.__cause__).lower()
    assert factory.data_flow.get_token_outcome(token_id) is None
