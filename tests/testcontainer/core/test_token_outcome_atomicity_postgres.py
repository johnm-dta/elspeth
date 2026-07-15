"""PostgreSQL row-lock proofs for atomic token-outcome validation."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, insert, select, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError
from sqlalchemy.sql import Executable
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]
from tests.fixtures.landscape import register_test_node

from elspeth.contracts import ExecutionError, NodeStateStatus, NodeType
from elspeth.contracts.audit import DISCARD_SINK_NAME, TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import artifacts_table, node_states_table, token_outcomes_table

pytestmark = pytest.mark.testcontainer


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


@pytest.fixture
def postgres_factory(postgres_url: str) -> Iterator[tuple[LandscapeDB, RecorderFactory]]:
    db = LandscapeDB(postgres_url)
    try:
        yield db, RecorderFactory(db)
    finally:
        db.engine.dispose()


def _build_token(factory: RecorderFactory) -> tuple[str, str, str]:
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
    return run.run_id, token.token_id, sink_id


def _lock_timeout_result(db: LandscapeDB, start: threading.Event, statement: Executable) -> str:
    if not start.wait(timeout=5):
        return "start-timeout"
    try:
        with db.engine.begin() as conn:
            conn.exec_driver_sql("SET LOCAL lock_timeout = '200ms'")
            conn.execute(statement)
    except DBAPIError as exc:
        if "lock timeout" not in str(exc).lower():
            return f"unexpected:{type(exc).__name__}:{exc}"
        return "blocked"
    return "mutated"


def _record_while_mutation_contends(
    *,
    db: LandscapeDB,
    factory: RecorderFactory,
    monkeypatch: pytest.MonkeyPatch,
    ref: TokenRef,
    mutation: Executable,
    outcome: TerminalOutcome,
    path: TerminalPath,
    sink_name: str,
    error_hash: str,
    sink_node_id: str | None = None,
    artifact_id: str | None = None,
) -> None:
    """Pause after invariant evaluation and prove the competing write blocks."""
    start = threading.Event()
    outcomes = factory.data_flow.outcomes
    original_invariants = outcomes._validate_cross_table_invariants

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_lock_timeout_result, db, start, mutation)

        def pause_after_validation(
            checked_ref: TokenRef,
            checked_outcome: TerminalOutcome | None,
            checked_path: TerminalPath,
            *,
            sink_name: str | None,
            sink_node_id: str | None,
            artifact_id: str | None,
            conn: Connection | None = None,
        ) -> None:
            original_invariants(
                checked_ref,
                checked_outcome,
                checked_path,
                sink_name=sink_name,
                sink_node_id=sink_node_id,
                artifact_id=artifact_id,
                conn=conn,
            )
            start.set()
            assert future.result(timeout=5) == "blocked"

        monkeypatch.setattr(outcomes, "_validate_cross_table_invariants", pause_after_validation)
        factory.data_flow.record_token_outcome(
            ref=ref,
            outcome=outcome,
            path=path,
            sink_name=sink_name,
            sink_node_id=sink_node_id,
            artifact_id=artifact_id,
            error_hash=error_hash,
        )


def test_postgres_locks_discard_node_states_until_outcome_insert(
    postgres_factory: tuple[LandscapeDB, RecorderFactory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, factory = postgres_factory
    run_id, token_id, sink_id = _build_token(factory)
    state = factory.execution.begin_node_state(
        token_id=token_id,
        node_id=sink_id,
        run_id=run_id,
        step_index=0,
        input_data={"value": 1},
    )
    factory.execution.complete_node_state(
        state_id=state.state_id,
        status=NodeStateStatus.FAILED,
        error=ExecutionError(exception="discard", exception_type="TestDiscard", phase="sink_write"),
        duration_ms=1.0,
    )
    mutation = (
        update(node_states_table).where(node_states_table.c.state_id == state.state_id).values(status=NodeStateStatus.COMPLETED.value)
    )
    _record_while_mutation_contends(
        db=db,
        factory=factory,
        monkeypatch=monkeypatch,
        ref=TokenRef(token_id=token_id, run_id=run_id),
        mutation=mutation,
        outcome=TerminalOutcome.FAILURE,
        path=TerminalPath.SINK_DISCARDED,
        sink_name=DISCARD_SINK_NAME,
        error_hash="discard-error",
    )

    with db.read_only_connection() as conn:
        assert conn.execute(select(node_states_table.c.status).where(node_states_table.c.state_id == state.state_id)).scalar_one() == (
            NodeStateStatus.FAILED.value
        )
        assert len(conn.execute(select(token_outcomes_table.c.outcome_id).where(token_outcomes_table.c.token_id == token_id)).all()) == 1


def test_postgres_token_lock_blocks_phantom_node_state_until_outcome_insert(
    postgres_factory: tuple[LandscapeDB, RecorderFactory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, factory = postgres_factory
    run_id, token_id, sink_id = _build_token(factory)
    mutation = insert(node_states_table).values(
        state_id="phantom-completed-state",
        token_id=token_id,
        run_id=run_id,
        node_id=sink_id,
        step_index=0,
        attempt=0,
        status=NodeStateStatus.COMPLETED.value,
        input_hash="0" * 64,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    _record_while_mutation_contends(
        db=db,
        factory=factory,
        monkeypatch=monkeypatch,
        ref=TokenRef(token_id=token_id, run_id=run_id),
        mutation=mutation,
        outcome=TerminalOutcome.FAILURE,
        path=TerminalPath.SINK_DISCARDED,
        sink_name=DISCARD_SINK_NAME,
        error_hash="discard-error",
    )

    with db.read_only_connection() as conn:
        assert (
            conn.execute(select(node_states_table.c.state_id).where(node_states_table.c.state_id == "phantom-completed-state")).all() == []
        )
        assert len(conn.execute(select(token_outcomes_table.c.outcome_id).where(token_outcomes_table.c.token_id == token_id)).all()) == 1


def test_postgres_locks_failsink_artifact_witness_until_outcome_insert(
    postgres_factory: tuple[LandscapeDB, RecorderFactory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, factory = postgres_factory
    run_id, token_id, sink_id = _build_token(factory)
    state = factory.execution.begin_node_state(
        token_id=token_id,
        node_id=sink_id,
        run_id=run_id,
        step_index=0,
        input_data={"value": 1},
    )
    factory.execution.complete_node_state(
        state_id=state.state_id,
        status=NodeStateStatus.COMPLETED,
        output_data={"written": True},
        duration_ms=1.0,
    )
    artifact = factory.execution.register_artifact(
        run_id=run_id,
        state_id=state.state_id,
        sink_node_id=sink_id,
        artifact_type="test",
        path="memory://failsink/artifact",
        content_hash="deadbeef" * 8,
        size_bytes=0,
    )
    mutation = delete(artifacts_table).where(artifacts_table.c.artifact_id == artifact.artifact_id)
    _record_while_mutation_contends(
        db=db,
        factory=factory,
        monkeypatch=monkeypatch,
        ref=TokenRef(token_id=token_id, run_id=run_id),
        mutation=mutation,
        outcome=TerminalOutcome.TRANSIENT,
        path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
        sink_name="failsink",
        sink_node_id=sink_id,
        artifact_id=artifact.artifact_id,
        error_hash="failsink-error",
    )

    with db.read_only_connection() as conn:
        assert (
            conn.execute(select(artifacts_table.c.artifact_id).where(artifacts_table.c.artifact_id == artifact.artifact_id)).scalar_one()
            == artifact.artifact_id
        )
        assert len(conn.execute(select(token_outcomes_table.c.outcome_id).where(token_outcomes_table.c.token_id == token_id)).all()) == 1
