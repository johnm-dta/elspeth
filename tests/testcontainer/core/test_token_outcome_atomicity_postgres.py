"""PostgreSQL row-lock proofs for atomic token-outcome validation."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import delete, insert, select, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError
from sqlalchemy.sql import Executable
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]
from tests.fixtures.base_classes import create_observed_contract
from tests.fixtures.landscape import register_test_node

from elspeth.contracts import ExecutionError, NodeStateStatus, NodeType, PendingOutcome, TokenInfo
from elspeth.contracts.audit import DISCARD_SINK_NAME, TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import artifacts_table, node_states_table, token_outcomes_table
from elspeth.engine.executors.sink import SinkExecutor
from elspeth.engine.spans import SpanFactory

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


def test_primary_sink_prelocks_tokens_before_states_so_concurrent_discard_cannot_deadlock(
    postgres_factory: tuple[LandscapeDB, RecorderFactory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composed sink path obeys tokens -> states -> artifact for every token."""
    db, factory = postgres_factory
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source_id = register_test_node(factory.data_flow, run.run_id, "source-batch", node_type=NodeType.SOURCE, plugin_name="source")
    sink_id = register_test_node(factory.data_flow, run.run_id, "sink-batch", node_type=NodeType.SINK, plugin_name="sink")
    primary_states = []
    for row_index in range(2):
        data = {"value": row_index}
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source_id,
            row_index=row_index,
            data=data,
            source_row_index=row_index,
            ingest_sequence=row_index,
        )
        token = factory.data_flow.create_token(row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=sink_id,
            run_id=run.run_id,
            step_index=0,
            input_data=data,
        )
        token_info = TokenInfo(
            row_id=row.row_id,
            token_id=token.token_id,
            row_data=PipelineRow(data=data, contract=create_observed_contract(data)),
        )
        primary_states.append((token_info, state))

    # Deliberately reverse caller order. The prelock primitive must impose a
    # deterministic token-id order independently of batch arrival order.
    primary_states.sort(key=lambda item: item[0].token_id, reverse=True)
    token_refs = tuple(TokenRef(token_id=token.token_id, run_id=run.run_id) for token, _state in primary_states)
    expected_lock_order = tuple(sorted(ref.token_id for ref in token_refs))
    sink_tokens_locked = threading.Event()
    discard_entered_token_lock = threading.Event()
    lock_orders: list[tuple[str, ...]] = []
    original_prelock = factory.data_flow.lock_token_outcome_dependencies
    original_token_lock = factory.data_flow.outcomes.lock_token_outcome_dependencies
    original_lock_query = factory.data_flow.outcomes._execute_lock_query

    def synchronized_sink_prelock(refs: tuple[TokenRef, ...], *, conn: Connection) -> None:
        original_prelock(refs, conn=conn)
        sink_tokens_locked.set()
        assert discard_entered_token_lock.wait(timeout=5)

    def observe_token_lock(refs: tuple[TokenRef, ...], *, conn: Connection) -> None:
        if threading.current_thread().name == "discard-contender":
            discard_entered_token_lock.set()
        original_token_lock(refs, conn=conn)

    def capture_lock_order(conn: Connection, query: Executable, *, operation: str) -> list[Any]:
        rows = original_lock_query(conn, query, operation=operation)
        if operation == "record_token_outcome token lock" and len(rows) > 1:
            lock_orders.append(tuple(row.token_id for row in rows))
        return rows

    monkeypatch.setattr(factory.data_flow, "lock_token_outcome_dependencies", synchronized_sink_prelock)
    monkeypatch.setattr(factory.data_flow.outcomes, "lock_token_outcome_dependencies", observe_token_lock)
    monkeypatch.setattr(factory.data_flow.outcomes, "_execute_lock_query", capture_lock_order)

    executor = SinkExecutor(factory.execution, factory.data_flow, SpanFactory(), run.run_id)
    artifact_info = ArtifactDescriptor.for_file(path="/tmp/atomic-sink", content_hash="a" * 64, size_bytes=2)

    def complete_primary() -> None:
        executor._complete_primary(
            primary_states=primary_states,
            divert_states=[],
            artifact_info=artifact_info,
            total_token_count=2,
            duration_ms=2.0,
            pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
            sink_name="output",
            sink_node_id=sink_id,
            on_token_written=None,
        )

    def record_discard() -> None:
        assert sink_tokens_locked.wait(timeout=5)
        factory.data_flow.record_token_outcome(
            ref=token_refs[0],
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.SINK_DISCARDED,
            sink_name=DISCARD_SINK_NAME,
            error_hash="concurrent-discard",
        )

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="atomic-outcome") as pool:
        sink_future = pool.submit(complete_primary)

        # ThreadPoolExecutor's generated name is not stable enough for the
        # lock hook, so name this worker for the synchronized observation.
        def named_discard() -> None:
            threading.current_thread().name = "discard-contender"
            record_discard()

        discard_future = pool.submit(named_discard)
        sink_future.result(timeout=10)
        with pytest.raises(AuditIntegrityError, match="I3 violation"):
            discard_future.result(timeout=10)

    assert lock_orders == [expected_lock_order]
    with db.read_only_connection() as conn:
        outcomes = conn.execute(
            select(token_outcomes_table.c.token_id, token_outcomes_table.c.path)
            .where(token_outcomes_table.c.run_id == run.run_id)
            .order_by(token_outcomes_table.c.token_id)
        ).all()
        states = (
            conn.execute(
                select(node_states_table.c.status).where(node_states_table.c.run_id == run.run_id).order_by(node_states_table.c.state_id)
            )
            .scalars()
            .all()
        )
    assert outcomes == [(token_id, TerminalPath.DEFAULT_FLOW.value) for token_id in expected_lock_order]
    assert states == [NodeStateStatus.COMPLETED.value, NodeStateStatus.COMPLETED.value]
