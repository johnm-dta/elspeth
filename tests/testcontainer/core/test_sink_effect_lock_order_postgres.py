"""Real PostgreSQL proof for reservation lock order and conflict safety."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.engine import Connection
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]
from tests.fixtures.landscape import make_factory, register_test_node

from elspeth.contracts import NodeStateStatus, NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.audit import DISCARD_SINK_NAME, TokenRef
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    SinkEffectAttemptAction,
    SinkEffectDescriptorMode,
    SinkEffectFinalizationResult,
    SinkEffectInputKind,
    SinkEffectInspectionMode,
    SinkEffectMemberCandidate,
    SinkEffectPlan,
    SinkEffectRole,
)
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.execution.sink_effect_finalization import SinkEffectFinalizationMember, SinkEffectFinalizeRequest
from elspeth.core.landscape.execution.sink_effect_identity import compute_pipeline_effect_identity, resolve_sink_effect_members
from elspeth.core.landscape.execution.sink_effect_lifecycle import SinkEffectAttemptRequest, SinkEffectAttemptResult
from elspeth.core.landscape.execution.sink_effect_reservation import SinkEffectReservationRequest
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import (
    artifacts_table,
    node_states_table,
    sink_effect_members_table,
    sink_effects_table,
    token_outcomes_table,
)

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


def test_finalization_vs_outcome_mutation_uses_distinct_backends_and_token_first_order(
    postgres_db: LandscapeDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = postgres_db
    finalizer_factory = make_factory(db)
    outcome_factory = make_factory(db)
    run = finalizer_factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source = register_test_node(finalizer_factory.data_flow, run.run_id, "finalize-source", node_type=NodeType.SOURCE, plugin_name="source")
    sink = register_test_node(finalizer_factory.data_flow, run.run_id, "finalize-sink", node_type=NodeType.SINK, plugin_name="sink")
    payload = {"ordinal": 0}
    row = finalizer_factory.data_flow.create_row(
        run_id=run.run_id,
        source_node_id=source,
        row_index=0,
        data=payload,
        source_row_index=0,
        ingest_sequence=0,
    )
    token = finalizer_factory.data_flow.create_token(row.row_id)
    finalizer_factory.execution.begin_node_state(
        token_id=token.token_id,
        node_id=sink,
        run_id=run.run_id,
        step_index=0,
        input_data=payload,
    )
    members = resolve_sink_effect_members(finalizer_factory, [SinkEffectMemberCandidate(token_id=token.token_id, row=payload)])
    identity = compute_pipeline_effect_identity(
        run_id=run.run_id,
        sink_node_id=sink,
        role=SinkEffectRole.PRIMARY,
        sink_config={"name": "sink"},
        target_config={"path": "finalize.jsonl"},
        members=members,
    )
    effect = finalizer_factory.execution.sink_effects.reserve(
        SinkEffectReservationRequest(
            run_id=run.run_id,
            sink_node_id=sink,
            role=SinkEffectRole.PRIMARY,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            requested_target_hash=identity.requested_target_hash,
            members=members,
            audit_export_snapshot_id=None,
            config_hash=identity.config_hash,
            replacing_target=False,
            primary_effect_id=None,
        )
    ).new_effect
    assert effect is not None
    descriptor = ArtifactDescriptor(
        artifact_type="file",
        path_or_uri="file:///tmp/finalize.jsonl",
        content_hash="d" * 64,
        size_bytes=12,
    )
    claim = finalizer_factory.execution.sink_effects.claim_preparation(
        effect.effect_id,
        owner="worker-a",
        ttl=timedelta(seconds=30),
    )
    finalizer_factory.execution.sink_effects.complete_plan(
        effect.effect_id,
        SinkEffectPlan(
            effect_id=effect.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            descriptor_mode=SinkEffectDescriptorMode.PRECOMPUTED,
            inspection_mode=SinkEffectInspectionMode.NO_INSPECTION_REQUIRED,
            target=descriptor.path_or_uri,
            plan_hash="a" * 64,
            payload_hash="b" * 64,
            expected_descriptor=descriptor,
            safe_evidence={"inspection_reference": "no-inspection-required:v1"},
        ),
        claim=claim,
    )
    lease = finalizer_factory.execution.sink_effects.acquire_lease(
        effect.effect_id,
        owner="worker-a",
        ttl=timedelta(seconds=30),
    )
    attempt = finalizer_factory.execution.sink_effects.begin_attempt(
        SinkEffectAttemptRequest(
            effect_id=effect.effect_id,
            member_ordinal=None,
            generation=lease.generation,
            action=SinkEffectAttemptAction.COMMIT,
            request_hash="a" * 64,
        )
    )
    finalizer_factory.execution.sink_effects.record_attempt_result(
        SinkEffectAttemptResult(attempt_id=attempt.attempt_id, evidence={"result": "exact"}, latency_ms=1.0)
    )
    request = SinkEffectFinalizeRequest(
        effect_id=effect.effect_id,
        lease_owner=lease.owner,
        generation=lease.generation,
        descriptor=descriptor,
        publication_performed=True,
        publication_evidence_kind="returned",
        accepted_ordinals=(0,),
        diverted_ordinals=(),
        evidence={"result": "exact"},
        members=(
            SinkEffectFinalizationMember(
                ordinal=0,
                output_data={"row": payload},
                duration_ms=1.0,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="sink",
            ),
        ),
        attempt_id=attempt.attempt_id,
    )
    finalizer_holds_token = threading.Event()
    release_finalizer = threading.Event()
    outcome_approached_token = threading.Event()
    backend_pids: dict[str, int] = {}

    def after_token_locks(pid: int, token_ids: tuple[str, ...]) -> None:
        assert token_ids == tuple(sorted(token_ids))
        backend_pids["finalizer"] = pid
        finalizer_holds_token.set()
        assert release_finalizer.wait(timeout=5)

    original_outcome_lock = outcome_factory.data_flow.outcomes.lock_token_outcome_dependencies

    def outcome_lock(refs: tuple[TokenRef, ...], *, conn: Connection) -> None:
        backend_pids["outcome"] = int(conn.exec_driver_sql("SELECT pg_backend_pid()").scalar_one())
        outcome_approached_token.set()
        original_outcome_lock(refs, conn=conn)

    monkeypatch.setattr(finalizer_factory.execution.sink_effects._finalization, "_after_token_locks", after_token_locks)
    monkeypatch.setattr(outcome_factory.data_flow.outcomes, "lock_token_outcome_dependencies", outcome_lock)

    def competing_outcome() -> str:
        return outcome_factory.data_flow.record_token_outcome(
            TokenRef(token_id=token.token_id, run_id=run.run_id),
            TerminalOutcome.SUCCESS,
            TerminalPath.DEFAULT_FLOW,
            sink_name="sink",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        finalization = pool.submit(finalizer_factory.execution.sink_effects.finalize, request)
        assert finalizer_holds_token.wait(timeout=5)
        outcome = pool.submit(competing_outcome)
        assert outcome_approached_token.wait(timeout=5)
        release_finalizer.set()
        winner = finalization.result(timeout=10)
        with pytest.raises(LandscapeRecordError):
            outcome.result(timeout=10)

    assert backend_pids["finalizer"] != backend_pids["outcome"]
    assert winner.effect.state.value == "finalized"
    with db.read_only_connection() as conn:
        assert (
            conn.scalar(select(func.count()).select_from(token_outcomes_table).where(token_outcomes_table.c.token_id == token.token_id))
            == 1
        )


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

    with db.read_only_connection() as conn:
        return tuple(
            conn.execute(
                select(node_states_table.c.state_id).where(
                    node_states_table.c.run_id == run_id,
                    node_states_table.c.node_id == sink_node_id,
                )
            ).scalars()
        )


@dataclass(frozen=True)
class _InFlightEffect:
    """A leased effect with a returned commit attempt, ready to finalize."""

    run_id: str
    token_id: str
    sink_node_id: str
    effect_id: str
    lease_owner: str
    generation: int
    request: SinkEffectFinalizeRequest


def _build_in_flight_effect(factory: RecorderFactory, *, name_prefix: str, owner: str = "worker-a") -> _InFlightEffect:
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source = register_test_node(factory.data_flow, run.run_id, f"{name_prefix}-source", node_type=NodeType.SOURCE, plugin_name="source")
    sink = register_test_node(factory.data_flow, run.run_id, f"{name_prefix}-sink", node_type=NodeType.SINK, plugin_name="sink")
    payload = {"ordinal": 0}
    row = factory.data_flow.create_row(
        run_id=run.run_id,
        source_node_id=source,
        row_index=0,
        data=payload,
        source_row_index=0,
        ingest_sequence=0,
    )
    token = factory.data_flow.create_token(row.row_id)
    factory.execution.begin_node_state(
        token_id=token.token_id,
        node_id=sink,
        run_id=run.run_id,
        step_index=0,
        input_data=payload,
    )
    members = resolve_sink_effect_members(factory, [SinkEffectMemberCandidate(token_id=token.token_id, row=payload)])
    identity = compute_pipeline_effect_identity(
        run_id=run.run_id,
        sink_node_id=sink,
        role=SinkEffectRole.PRIMARY,
        sink_config={"name": "sink"},
        target_config={"path": f"{name_prefix}.jsonl"},
        members=members,
    )
    effect = factory.execution.sink_effects.reserve(
        SinkEffectReservationRequest(
            run_id=run.run_id,
            sink_node_id=sink,
            role=SinkEffectRole.PRIMARY,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            requested_target_hash=identity.requested_target_hash,
            members=members,
            audit_export_snapshot_id=None,
            config_hash=identity.config_hash,
            replacing_target=False,
            primary_effect_id=None,
        )
    ).new_effect
    assert effect is not None
    descriptor = ArtifactDescriptor(
        artifact_type="file",
        path_or_uri=f"file:///tmp/{name_prefix}.jsonl",
        content_hash="d" * 64,
        size_bytes=12,
    )
    claim = factory.execution.sink_effects.claim_preparation(effect.effect_id, owner=owner, ttl=timedelta(seconds=30))
    factory.execution.sink_effects.complete_plan(
        effect.effect_id,
        SinkEffectPlan(
            effect_id=effect.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            descriptor_mode=SinkEffectDescriptorMode.PRECOMPUTED,
            inspection_mode=SinkEffectInspectionMode.NO_INSPECTION_REQUIRED,
            target=descriptor.path_or_uri,
            plan_hash="a" * 64,
            payload_hash="b" * 64,
            expected_descriptor=descriptor,
            safe_evidence={"inspection_reference": "no-inspection-required:v1"},
        ),
        claim=claim,
    )
    lease = factory.execution.sink_effects.acquire_lease(effect.effect_id, owner=owner, ttl=timedelta(seconds=30))
    attempt = factory.execution.sink_effects.begin_attempt(
        SinkEffectAttemptRequest(
            effect_id=effect.effect_id,
            member_ordinal=None,
            generation=lease.generation,
            action=SinkEffectAttemptAction.COMMIT,
            request_hash="a" * 64,
        )
    )
    factory.execution.sink_effects.record_attempt_result(
        SinkEffectAttemptResult(attempt_id=attempt.attempt_id, evidence={"result": "exact"}, latency_ms=1.0)
    )
    request = SinkEffectFinalizeRequest(
        effect_id=effect.effect_id,
        lease_owner=lease.owner,
        generation=lease.generation,
        descriptor=descriptor,
        publication_performed=True,
        publication_evidence_kind="returned",
        accepted_ordinals=(0,),
        diverted_ordinals=(),
        evidence={"result": "exact"},
        members=(
            SinkEffectFinalizationMember(
                ordinal=0,
                output_data={"row": payload},
                duration_ms=1.0,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="sink",
            ),
        ),
        attempt_id=attempt.attempt_id,
    )
    return _InFlightEffect(
        run_id=run.run_id,
        token_id=token.token_id,
        sink_node_id=sink,
        effect_id=effect.effect_id,
        lease_owner=lease.owner,
        generation=lease.generation,
        request=request,
    )


def test_takeover_vs_finalization_generation_fences_stale_finalizer(postgres_db: LandscapeDB, monkeypatch: pytest.MonkeyPatch) -> None:
    """Takeover paused after its effect lock beats a stale finalizer approaching
    through sorted token/state locks: the generation fence rejects the stale
    finalization, both transactions complete bounded, and no artifact or
    outcome from the loser survives (design scenario: takeover versus
    finalization/head CAS)."""
    db = postgres_db
    finalizer_factory = make_factory(db)
    takeover_factory = make_factory(db)
    built = _build_in_flight_effect(finalizer_factory, name_prefix="takeover-fence")

    # Expire the lease directly so takeover is legal while the original owner
    # still believes it holds the effect — the crash-recovery race the design
    # fences with generation CAS. A sleep-based expiry would be flaky here.
    with db.engine.begin() as conn:
        conn.execute(
            update(sink_effects_table)
            .where(sink_effects_table.c.effect_id == built.effect_id)
            .values(
                lease_heartbeat_at=datetime.now(UTC) - timedelta(hours=2),
                lease_expires_at=datetime.now(UTC) - timedelta(hours=1),
            )
        )

    takeover_locked = threading.Event()
    release_takeover = threading.Event()
    finalizer_states_locked = threading.Event()
    backend_pids: dict[str, int] = {}

    def pause_after_effect_lock(pid: int, effect_id: str) -> None:
        assert effect_id == built.effect_id
        backend_pids["takeover"] = pid
        takeover_locked.set()
        assert release_takeover.wait(timeout=5)

    def capture_token_locks(pid: int, token_ids: tuple[str, ...]) -> None:
        assert token_ids == tuple(sorted(token_ids))
        backend_pids["finalizer"] = pid

    def signal_state_locks(_pid: int, state_ids: tuple[str, ...]) -> None:
        assert state_ids == tuple(sorted(state_ids))
        finalizer_states_locked.set()

    monkeypatch.setattr(takeover_factory.execution.sink_effects._lifecycle, "_after_effect_lock", pause_after_effect_lock)
    monkeypatch.setattr(finalizer_factory.execution.sink_effects._finalization, "_after_token_locks", capture_token_locks)
    monkeypatch.setattr(finalizer_factory.execution.sink_effects._finalization, "_after_state_locks", signal_state_locks)

    with ThreadPoolExecutor(max_workers=2) as pool:
        takeover = pool.submit(
            takeover_factory.execution.sink_effects.takeover_expired,
            built.effect_id,
            owner="worker-b",
            ttl=timedelta(seconds=30),
        )
        assert takeover_locked.wait(timeout=5)
        finalization = pool.submit(finalizer_factory.execution.sink_effects.finalize, built.request)
        assert finalizer_states_locked.wait(timeout=5)
        release_takeover.set()
        new_lease = takeover.result(timeout=10)
        with pytest.raises(LandscapeRecordError, match="stale lease owner"):
            finalization.result(timeout=10)

    assert new_lease.owner == "worker-b"
    assert new_lease.generation == built.generation + 1
    assert backend_pids["takeover"] != backend_pids["finalizer"]
    with db.read_only_connection() as conn:
        effect_row = conn.execute(select(sink_effects_table).where(sink_effects_table.c.effect_id == built.effect_id)).one()
        assert effect_row.state == "in_flight"
        assert effect_row.lease_owner == "worker-b"
        assert int(effect_row.generation) == built.generation + 1
        assert (
            conn.scalar(select(func.count()).select_from(artifacts_table).where(artifacts_table.c.sink_effect_id == built.effect_id)) == 0
        )
        assert (
            conn.scalar(select(func.count()).select_from(token_outcomes_table).where(token_outcomes_table.c.token_id == built.token_id))
            == 0
        )
        statuses = list(conn.execute(select(node_states_table.c.status).where(node_states_table.c.token_id == built.token_id)).scalars())
        assert statuses == [NodeStateStatus.OPEN.value]


def test_takeover_blocked_by_finalization_observes_finalized_effect(postgres_db: LandscapeDB, monkeypatch: pytest.MonkeyPatch) -> None:
    """The other legal winner: a finalizer paused holding its effect locks
    commits first, and the concurrent takeover — blocked on the same effect
    row — must observe FINALIZED and be rejected instead of stealing the
    lease of a completed effect."""
    db = postgres_db
    finalizer_factory = make_factory(db)
    takeover_factory = make_factory(db)
    built = _build_in_flight_effect(finalizer_factory, name_prefix="takeover-loses")

    finalizer_holds_effect = threading.Event()
    release_finalizer = threading.Event()
    takeover_approached = threading.Event()
    backend_pids: dict[str, int] = {}

    def pause_after_effect_locks(pid: int, effect_ids: tuple[str, ...]) -> None:
        assert effect_ids == tuple(sorted(effect_ids))
        backend_pids["finalizer"] = pid
        finalizer_holds_effect.set()
        assert release_finalizer.wait(timeout=5)

    def capture_takeover_pid(pid: int, _effect_id: str) -> None:
        backend_pids["takeover"] = pid

    lifecycle = takeover_factory.execution.sink_effects._lifecycle
    original_lock_effect = lifecycle._lock_effect

    def approaching_lock_effect(conn: Connection, effect_id: str, *, include_stream: bool) -> object:
        takeover_approached.set()
        return original_lock_effect(conn, effect_id, include_stream=include_stream)

    monkeypatch.setattr(finalizer_factory.execution.sink_effects._finalization, "_after_effect_locks", pause_after_effect_locks)
    monkeypatch.setattr(lifecycle, "_after_effect_lock", capture_takeover_pid)
    monkeypatch.setattr(lifecycle, "_lock_effect", approaching_lock_effect)

    with ThreadPoolExecutor(max_workers=2) as pool:
        finalization = pool.submit(finalizer_factory.execution.sink_effects.finalize, built.request)
        assert finalizer_holds_effect.wait(timeout=5)
        takeover = pool.submit(
            takeover_factory.execution.sink_effects.takeover_expired,
            built.effect_id,
            owner="worker-b",
            ttl=timedelta(seconds=30),
        )
        assert takeover_approached.wait(timeout=5)
        release_finalizer.set()
        winner = finalization.result(timeout=10)
        with pytest.raises(LandscapeRecordError, match="finalized sink effect cannot be taken over"):
            takeover.result(timeout=10)

    assert winner.effect.state.value == "finalized"
    assert backend_pids["takeover"] != backend_pids["finalizer"]
    with db.read_only_connection() as conn:
        effect_row = conn.execute(select(sink_effects_table).where(sink_effects_table.c.effect_id == built.effect_id)).one()
        assert effect_row.state == "finalized"
        assert effect_row.lease_owner is None
        assert int(effect_row.generation) == built.generation
        assert (
            conn.scalar(select(func.count()).select_from(artifacts_table).where(artifacts_table.c.sink_effect_id == built.effect_id)) == 1
        )


def test_concurrent_finalization_retries_converge_on_winner_under_effect_lock(
    postgres_db: LandscapeDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Effect-linked artifact path: two crash-recovery retries of an already
    finalized effect serialize on the effect row and only then take the
    artifact FOR UPDATE — effect before artifact, never the reverse — and
    both converge on the identical winner with exactly one artifact row."""
    db = postgres_db
    setup_factory = make_factory(db)
    built = _build_in_flight_effect(setup_factory, name_prefix="retry-converge")
    first = setup_factory.execution.sink_effects.finalize(built.request)
    assert first.effect.state.value == "finalized"

    retry_a_factory = make_factory(db)
    retry_b_factory = make_factory(db)

    a_holds_effect = threading.Event()
    release_a = threading.Event()
    b_approached = threading.Event()
    backend_pids: dict[str, int] = {}

    def pause_a_after_effect_locks(pid: int, effect_ids: tuple[str, ...]) -> None:
        assert effect_ids == tuple(sorted(effect_ids))
        backend_pids["retry_a"] = pid
        a_holds_effect.set()
        assert release_a.wait(timeout=5)

    def capture_b_effect_locks(pid: int, _effect_ids: tuple[str, ...]) -> None:
        backend_pids["retry_b"] = pid

    b_finalization = retry_b_factory.execution.sink_effects._finalization
    original_b_lock = b_finalization._lock_stream_and_effects

    def approaching_b_lock(conn: Connection, optimistic_effect: object, linked_effect_ids: tuple[str, ...]) -> dict[str, object]:
        b_approached.set()
        return original_b_lock(conn, optimistic_effect, linked_effect_ids)

    monkeypatch.setattr(retry_a_factory.execution.sink_effects._finalization, "_after_effect_locks", pause_a_after_effect_locks)
    monkeypatch.setattr(b_finalization, "_after_effect_locks", capture_b_effect_locks)
    monkeypatch.setattr(b_finalization, "_lock_stream_and_effects", approaching_b_lock)

    with ThreadPoolExecutor(max_workers=2) as pool:
        retry_a = pool.submit(retry_a_factory.execution.sink_effects.finalize, built.request)
        assert a_holds_effect.wait(timeout=5)
        retry_b = pool.submit(retry_b_factory.execution.sink_effects.finalize, built.request)
        assert b_approached.wait(timeout=5)
        # Retry B cannot finish while retry A holds the effect row lock: the
        # artifact winner is only readable behind the effect lock class.
        assert not retry_b.done()
        release_a.set()
        winners: list[SinkEffectFinalizationResult] = [retry_a.result(timeout=10), retry_b.result(timeout=10)]

    assert backend_pids["retry_a"] != backend_pids["retry_b"]
    assert {winner.effect.effect_id for winner in winners} == {built.effect_id}
    assert {winner.artifact.artifact_id for winner in winners} == {first.artifact.artifact_id}
    assert all(winner.effect.state.value == "finalized" for winner in winners)
    assert all(winner.artifact.content_hash == built.request.descriptor.content_hash for winner in winners)
    with db.read_only_connection() as conn:
        assert (
            conn.scalar(select(func.count()).select_from(artifacts_table).where(artifacts_table.c.sink_effect_id == built.effect_id)) == 1
        )


def test_legacy_state_linked_artifact_outcome_contention_single_winner(postgres_db: LandscapeDB, monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy state-linked artifact path: two composed outcome writers for the
    same token follow token, state, artifact lock order; they serialize at the
    token class, exactly one outcome wins, the loser is rejected without
    deadlock, and the artifact witness row survives untouched."""
    db = postgres_db
    winner_factory = make_factory(db)
    loser_factory = make_factory(db)
    run = winner_factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source = register_test_node(
        winner_factory.data_flow, run.run_id, "legacy-artifact-source", node_type=NodeType.SOURCE, plugin_name="source"
    )
    failsink = register_test_node(
        winner_factory.data_flow, run.run_id, "legacy-artifact-failsink", node_type=NodeType.SINK, plugin_name="failsink"
    )
    payload = {"value": 1}
    row = winner_factory.data_flow.create_row(
        run_id=run.run_id,
        source_node_id=source,
        row_index=0,
        data=payload,
        source_row_index=0,
        ingest_sequence=0,
    )
    token = winner_factory.data_flow.create_token(row.row_id)
    state = winner_factory.execution.begin_node_state(
        token_id=token.token_id,
        node_id=failsink,
        run_id=run.run_id,
        step_index=0,
        input_data=payload,
    )
    winner_factory.execution.complete_node_state(
        state_id=state.state_id,
        status=NodeStateStatus.COMPLETED,
        output_data={"written": True},
        duration_ms=1.0,
    )
    artifact = winner_factory.execution.register_artifact(
        run_id=run.run_id,
        state_id=state.state_id,
        sink_node_id=failsink,
        artifact_type="test",
        path="memory://legacy/fallback-artifact",
        content_hash="ab" * 32,
        size_bytes=0,
    )

    winner_locked = threading.Event()
    release_winner = threading.Event()
    loser_approached = threading.Event()
    backend_pids: dict[str, int] = {}

    original_winner_lock = winner_factory.data_flow.outcomes.lock_token_outcome_dependencies

    def winner_lock(refs: tuple[TokenRef, ...], *, conn: Connection) -> None:
        original_winner_lock(refs, conn=conn)
        backend_pids["winner"] = int(conn.exec_driver_sql("SELECT pg_backend_pid()").scalar_one())
        winner_locked.set()
        assert release_winner.wait(timeout=5)

    original_loser_lock = loser_factory.data_flow.outcomes.lock_token_outcome_dependencies

    def loser_lock(refs: tuple[TokenRef, ...], *, conn: Connection) -> None:
        backend_pids["loser"] = int(conn.exec_driver_sql("SELECT pg_backend_pid()").scalar_one())
        loser_approached.set()
        original_loser_lock(refs, conn=conn)

    monkeypatch.setattr(winner_factory.data_flow.outcomes, "lock_token_outcome_dependencies", winner_lock)
    monkeypatch.setattr(loser_factory.data_flow.outcomes, "lock_token_outcome_dependencies", loser_lock)

    def record_fallback(factory: RecorderFactory) -> str:
        return factory.data_flow.record_token_outcome(
            TokenRef(token_id=token.token_id, run_id=run.run_id),
            TerminalOutcome.TRANSIENT,
            TerminalPath.SINK_FALLBACK_TO_FAILSINK,
            sink_name="failsink",
            sink_node_id=failsink,
            artifact_id=artifact.artifact_id,
            error_hash="fallback-error",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        winner = pool.submit(record_fallback, winner_factory)
        assert winner_locked.wait(timeout=5)
        loser = pool.submit(record_fallback, loser_factory)
        assert loser_approached.wait(timeout=5)
        release_winner.set()
        outcome_id = winner.result(timeout=10)
        with pytest.raises(LandscapeRecordError, match="database rejected audit write"):
            loser.result(timeout=10)

    assert outcome_id
    assert backend_pids["winner"] != backend_pids["loser"]
    with db.read_only_connection() as conn:
        outcomes = conn.execute(select(token_outcomes_table).where(token_outcomes_table.c.token_id == token.token_id)).fetchall()
        assert len(outcomes) == 1
        assert outcomes[0].outcome == TerminalOutcome.TRANSIENT.value
        assert outcomes[0].path == TerminalPath.SINK_FALLBACK_TO_FAILSINK.value
        artifact_row = conn.execute(select(artifacts_table).where(artifacts_table.c.artifact_id == artifact.artifact_id)).one()
        assert artifact_row.produced_by_state_id == state.state_id
        assert artifact_row.content_hash == "ab" * 32
