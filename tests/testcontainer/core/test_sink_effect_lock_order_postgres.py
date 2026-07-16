"""Real PostgreSQL proof for reservation lock order and conflict safety."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import Connection
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]
from tests.fixtures.landscape import make_factory, register_test_node

from elspeth.contracts import NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.audit import DISCARD_SINK_NAME, TokenRef
from elspeth.contracts.sink_effects import SinkEffectInputKind, SinkEffectMemberCandidate, SinkEffectRole
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution.sink_effect_identity import compute_pipeline_effect_identity, resolve_sink_effect_members
from elspeth.core.landscape.execution.sink_effect_reservation import SinkEffectReservationRequest
from elspeth.core.landscape.schema import sink_effect_members_table, sink_effects_table, token_outcomes_table

pytestmark = pytest.mark.testcontainer


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


@pytest.fixture(scope="module")
def postgres_db(postgres_url: str) -> Iterator[LandscapeDB]:
    db = LandscapeDB(postgres_url)
    try:
        yield db
    finally:
        db.close()


def test_concurrent_reservation_reverse_arrival_uses_ascending_locks_and_one_effect(
    postgres_db: LandscapeDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = postgres_db
    factory = make_factory(db)
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
    sink = register_test_node(factory.data_flow, run.run_id, "sink", node_type=NodeType.SINK, plugin_name="sink")
    candidates: list[SinkEffectMemberCandidate] = []
    for ordinal in range(2):
        payload = {"ordinal": ordinal}
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source,
            row_index=ordinal,
            data=payload,
            source_row_index=ordinal,
            ingest_sequence=ordinal,
        )
        token = factory.data_flow.create_token(row.row_id)
        factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=sink,
            run_id=run.run_id,
            step_index=0,
            input_data=payload,
        )
        candidates.append(SinkEffectMemberCandidate(token_id=token.token_id, row=payload))
    members = resolve_sink_effect_members(factory, candidates)
    identity = compute_pipeline_effect_identity(
        run_id=run.run_id,
        sink_node_id=sink,
        role=SinkEffectRole.PRIMARY,
        sink_config={"name": "sink"},
        target_config={"path": "out"},
        members=members,
    )
    request = SinkEffectReservationRequest(
        run_id=run.run_id,
        sink_node_id=sink,
        role=SinkEffectRole.PRIMARY,
        input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        requested_target_hash=identity.requested_target_hash,
        members=members,
        audit_export_snapshot_id=None,
        config_hash=identity.config_hash,
        replacing_target=True,
        primary_effect_id=None,
    )
    reverse_request = replace(request, members=tuple(reversed(members)))
    first_locked = threading.Event()
    release_first = threading.Event()
    observations: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []

    def pause(pid: int, token_ids: tuple[str, ...], state_ids: tuple[str, ...]) -> None:
        observations.append((pid, token_ids, state_ids))
        if not first_locked.is_set():
            first_locked.set()
            assert release_first.wait(timeout=5)

    monkeypatch.setattr(factory.execution.sink_effects._reservation, "_after_witness_locks", pause)
    second_factory = make_factory(db)
    monkeypatch.setattr(second_factory.execution.sink_effects._reservation, "_after_witness_locks", pause)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(factory.execution.sink_effects.reserve, request)
        assert first_locked.wait(timeout=5)
        second = pool.submit(second_factory.execution.sink_effects.reserve, reverse_request)
        release_first.set()
        results = (first.result(timeout=10), second.result(timeout=10))

    assert len({pid for pid, _tokens, _states in observations}) == 2
    assert all(list(tokens) == sorted(tokens) and list(states) == sorted(states) for _pid, tokens, states in observations)
    assert sum(result.new_effect is not None for result in results) == 1
    with db.read_only_connection() as conn:
        assert conn.scalar(select(func.count()).select_from(sink_effects_table)) == 1


def test_concurrent_disjoint_reservations_form_one_stream_predecessor_chain(
    postgres_db: LandscapeDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = postgres_db
    first_factory = make_factory(db)
    second_factory = make_factory(db)
    run = first_factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source = register_test_node(
        first_factory.data_flow,
        run.run_id,
        "disjoint-source",
        node_type=NodeType.SOURCE,
        plugin_name="source",
    )
    sink = register_test_node(
        first_factory.data_flow,
        run.run_id,
        "disjoint-sink",
        node_type=NodeType.SINK,
        plugin_name="sink",
    )
    candidates: list[SinkEffectMemberCandidate] = []
    for ordinal in range(2):
        payload = {"ordinal": ordinal}
        row = first_factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source,
            row_index=ordinal,
            data=payload,
            source_row_index=ordinal,
            ingest_sequence=ordinal,
        )
        token = first_factory.data_flow.create_token(row.row_id)
        first_factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=sink,
            run_id=run.run_id,
            step_index=0,
            input_data=payload,
        )
        candidates.append(SinkEffectMemberCandidate(token_id=token.token_id, row=payload))
    members = resolve_sink_effect_members(first_factory, candidates)

    requests: list[SinkEffectReservationRequest] = []
    for member in members:
        dense_member = member if member.ordinal == 0 else replace(member, ordinal=0)
        identity = compute_pipeline_effect_identity(
            run_id=run.run_id,
            sink_node_id=sink,
            role=SinkEffectRole.PRIMARY,
            sink_config={"name": "sink"},
            target_config={"path": "same-output"},
            members=(dense_member,),
        )
        requests.append(
            SinkEffectReservationRequest(
                run_id=run.run_id,
                sink_node_id=sink,
                role=SinkEffectRole.PRIMARY,
                input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
                requested_target_hash=identity.requested_target_hash,
                members=(member,),
                audit_export_snapshot_id=None,
                config_hash=identity.config_hash,
                replacing_target=True,
                primary_effect_id=None,
            )
        )

    witness_barrier = threading.Barrier(2)
    backend_pids: set[int] = set()
    backend_guard = threading.Lock()

    def await_both_witnesses(pid: int, token_ids: tuple[str, ...], state_ids: tuple[str, ...]) -> None:
        assert token_ids == tuple(sorted(token_ids))
        assert state_ids == tuple(sorted(state_ids))
        with backend_guard:
            backend_pids.add(pid)
        witness_barrier.wait(timeout=5)

    monkeypatch.setattr(first_factory.execution.sink_effects._reservation, "_after_witness_locks", await_both_witnesses)
    monkeypatch.setattr(second_factory.execution.sink_effects._reservation, "_after_witness_locks", await_both_witnesses)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (
            pool.submit(first_factory.execution.sink_effects.reserve, requests[0]),
            pool.submit(second_factory.execution.sink_effects.reserve, requests[1]),
        )
        effects = tuple(future.result(timeout=10).new_effect for future in futures)

    assert len(backend_pids) == 2
    assert all(effect is not None for effect in effects)
    ordered = sorted((effect for effect in effects if effect is not None), key=lambda effect: effect.stream_sequence or 0)
    assert [effect.stream_sequence for effect in ordered] == [0, 1]
    assert ordered[0].predecessor_effect_id is None
    assert ordered[1].predecessor_effect_id == ordered[0].effect_id


def test_reservation_vs_outcome_uses_token_first_order_without_deadlock(postgres_db: LandscapeDB, monkeypatch: pytest.MonkeyPatch) -> None:
    db = postgres_db
    reservation_factory = make_factory(db)
    outcome_factory = make_factory(db)
    run = reservation_factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source = register_test_node(
        reservation_factory.data_flow,
        run.run_id,
        "outcome-race-source",
        node_type=NodeType.SOURCE,
        plugin_name="source",
    )
    sink = register_test_node(
        reservation_factory.data_flow,
        run.run_id,
        "outcome-race-sink",
        node_type=NodeType.SINK,
        plugin_name="sink",
    )
    candidates: list[SinkEffectMemberCandidate] = []
    for ordinal in range(2):
        payload = {"ordinal": ordinal}
        row = reservation_factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source,
            row_index=ordinal,
            data=payload,
            source_row_index=ordinal,
            ingest_sequence=ordinal,
        )
        token = reservation_factory.data_flow.create_token(row.row_id)
        reservation_factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=sink,
            run_id=run.run_id,
            step_index=0,
            input_data=payload,
        )
        candidates.append(SinkEffectMemberCandidate(token_id=token.token_id, row=payload))
    members = resolve_sink_effect_members(reservation_factory, candidates)
    identity = compute_pipeline_effect_identity(
        run_id=run.run_id,
        sink_node_id=sink,
        role=SinkEffectRole.PRIMARY,
        sink_config={"name": "sink"},
        target_config={"path": "outcome-race-output"},
        members=members,
    )
    request = SinkEffectReservationRequest(
        run_id=run.run_id,
        sink_node_id=sink,
        role=SinkEffectRole.PRIMARY,
        input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        requested_target_hash=identity.requested_target_hash,
        members=members,
        audit_export_snapshot_id=None,
        config_hash=identity.config_hash,
        replacing_target=True,
        primary_effect_id=None,
    )

    first_token_locked = threading.Event()
    outcome_entered = threading.Event()
    reservation_pid: list[int] = []
    outcome_pid: list[int] = []
    complete_witnesses: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def pause_after_first_token(pid: int, token_ids: tuple[str, ...]) -> None:
        if len(token_ids) == 1 and not first_token_locked.is_set():
            reservation_pid.append(pid)
            first_token_locked.set()
            assert outcome_entered.wait(timeout=5)

    def capture_complete_witnesses(pid: int, token_ids: tuple[str, ...], state_ids: tuple[str, ...]) -> None:
        if not reservation_pid:
            reservation_pid.append(pid)
        complete_witnesses.append((token_ids, state_ids))

    original_outcome_locks = outcome_factory.data_flow.outcomes.lock_token_outcome_dependencies

    def enter_outcome_lock(refs: tuple[TokenRef, ...], *, conn: Connection) -> None:
        outcome_pid.append(int(conn.exec_driver_sql("SELECT pg_backend_pid()").scalar_one()))
        outcome_entered.set()
        original_outcome_locks(refs, conn=conn)

    monkeypatch.setattr(reservation_factory.execution.sink_effects._reservation, "_after_token_lock", pause_after_first_token)
    monkeypatch.setattr(reservation_factory.execution.sink_effects._reservation, "_after_witness_locks", capture_complete_witnesses)
    monkeypatch.setattr(outcome_factory.data_flow.outcomes, "lock_token_outcome_dependencies", enter_outcome_lock)

    first_token_id = min(member.token_id for member in members)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reservation_future = pool.submit(reservation_factory.execution.sink_effects.reserve, request)
        assert first_token_locked.wait(timeout=5)
        outcome_future = pool.submit(
            outcome_factory.data_flow.record_token_outcome,
            TokenRef(token_id=first_token_id, run_id=run.run_id),
            TerminalOutcome.FAILURE,
            TerminalPath.SINK_DISCARDED,
            sink_name=DISCARD_SINK_NAME,
            error_hash="outcome-race",
        )
        reservation = reservation_future.result(timeout=10)
        outcome_future.result(timeout=10)

    assert reservation.new_effect is not None
    assert reservation_pid and outcome_pid and reservation_pid[0] != outcome_pid[0]
    assert complete_witnesses == [
        (
            tuple(sorted(member.token_id for member in members)),
            tuple(sorted(state_id for state_id in request_witness_state_ids(db, run.run_id, sink))),
        )
    ]
    with db.read_only_connection() as conn:
        assert (
            conn.scalar(
                select(func.count())
                .select_from(sink_effect_members_table)
                .where(sink_effect_members_table.c.effect_id == reservation.new_effect.effect_id)
            )
            == 2
        )
        assert (
            conn.scalar(select(func.count()).select_from(token_outcomes_table).where(token_outcomes_table.c.token_id == first_token_id))
            == 1
        )


def request_witness_state_ids(db: LandscapeDB, run_id: str, sink_node_id: str) -> tuple[str, ...]:
    """Read the race's complete state set after both transactions finish."""
    from elspeth.core.landscape.schema import node_states_table

    with db.read_only_connection() as conn:
        return tuple(
            conn.execute(
                select(node_states_table.c.state_id).where(
                    node_states_table.c.run_id == run_id,
                    node_states_table.c.node_id == sink_node_id,
                )
            ).scalars()
        )
