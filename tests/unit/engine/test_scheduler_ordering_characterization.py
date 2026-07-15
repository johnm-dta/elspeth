"""Characterization tests for durable token-scheduler admission, lease, and parking semantics.

These tests pin the CURRENT behavior of ``TokenSchedulerRepository`` (the
SQLite-backed work queue behind the durable token scheduler) at the repository
level, against a real Tier-1 SQLite engine:

1. Claim admission follows the documented deterministic ORDER BY policy
   ``(ingest_sequence, step_index, created_at, work_item_id)`` — including
   the ``work_item_id`` last-resort tiebreaker for exact same-tick collisions.
2. Lease semantics: a live lease cannot be re-claimed; an expired lease is
   recovered exactly once (with attempt bump + work_item_id rotation) and a
   recovery sweep never reaps the caller's own lease (self-steal guard, G1).
3. PENDING_SINK parking: a token parked at PENDING_SINK is non-terminal but
   does not block other tokens from being claimed and driven to terminal;
   the parked token is drained afterward via ``claim_pending_sink``. The
   durable scheduler_events timeline proves the overlapping lifecycles.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, insert, select

from elspeth.contracts import NodeType
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    metadata,
    nodes_table,
    rows_table,
    run_coordination_table,
    runs_table,
    scheduler_events_table,
    token_work_items_table,
    tokens_table,
)

RUN_ID = "run-order"
LEADER_WORKER_ID = "test-leader"
# Epoch-1 token; _insert_run_and_nodes seeds the matching seat.
COORD_TOKEN = CoordinationToken(run_id=RUN_ID, worker_id=LEADER_WORKER_ID, leader_epoch=1)


def _make_scheduler_engine() -> Tier1Engine:
    engine = create_engine("sqlite:///:memory:", echo=False)
    LandscapeDB._configure_sqlite(engine)
    LandscapeDB._verify_sqlite_pragmas(engine, "sqlite:///:memory:")
    metadata.create_all(engine)
    return Tier1Engine(engine)


def _row_payload_json() -> str:
    return TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))


def _insert_run_and_nodes(engine: Tier1Engine, *, now: datetime) -> None:
    """Insert the run plus two source nodes and one transform node."""
    with engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=RUN_ID,
                started_at=now,
                config_hash="config",
                settings_json="{}",
                canonical_version="v1",
                status="running",
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        for node_id, node_type, plugin in (
            ("source-a", NodeType.SOURCE, "csv"),
            ("source-b", NodeType.SOURCE, "csv"),
            ("normalize", NodeType.TRANSFORM, "identity"),
        ):
            conn.execute(
                insert(nodes_table).values(
                    run_id=RUN_ID,
                    node_id=node_id,
                    plugin_name=plugin,
                    node_type=node_type.value,
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="config",
                    config_json="{}",
                    registered_at=now,
                )
            )
        # Epoch-1 coordination seat for mark_pending_sink_terminal (slice-4 REQUIRED).
        conn.execute(
            insert(run_coordination_table).values(
                run_id=RUN_ID,
                leader_worker_id=LEADER_WORKER_ID,
                leader_epoch=1,
                leader_heartbeat_expires_at=now + timedelta(hours=1),
                updated_at=now,
            )
        )


def _insert_row_with_tokens(
    engine: Tier1Engine,
    *,
    row_id: str,
    source_node_id: str,
    source_row_index: int,
    ingest_sequence: int,
    token_ids: tuple[str, ...],
    now: datetime,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            insert(rows_table).values(
                row_id=row_id,
                run_id=RUN_ID,
                source_node_id=source_node_id,
                row_index=ingest_sequence,
                source_row_index=source_row_index,
                ingest_sequence=ingest_sequence,
                source_data_hash=f"hash-{row_id}",
                created_at=now,
            )
        )
        for token_id in token_ids:
            conn.execute(
                insert(tokens_table).values(
                    token_id=token_id,
                    row_id=row_id,
                    run_id=RUN_ID,
                    created_at=now,
                )
            )


def _claim_tokens_in_order(repo: TokenSchedulerRepository, *, lease_owner: str, now: datetime) -> list[str]:
    """Claim every READY item without terminalizing; return token_ids in claim order."""
    claimed: list[str] = []
    while True:
        item = repo.claim_ready(run_id=RUN_ID, lease_owner=lease_owner, lease_seconds=300, now=now)
        if item is None:
            return claimed
        claimed.append(item.token_id)


def _events(engine: Tier1Engine) -> list[dict[str, object]]:
    with engine.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                select(scheduler_events_table)
                .where(scheduler_events_table.c.run_id == RUN_ID)
                .order_by(scheduler_events_table.c.recorded_at, scheduler_events_table.c.event_id)
            ).mappings()
        ]


def test_claim_ready_order_is_global_ingest_sequence_not_enqueue_order() -> None:
    """claim_ready admits work in global ingest_sequence order across sources, regardless of enqueue order."""
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _row_payload_json()
    _insert_run_and_nodes(engine, now=now)

    # Two sources, per-source row indices restarting at 0, global ingest_sequence 0..3.
    rows = (
        ("row-a0", "source-a", 0, 0, "token-a0"),
        ("row-a1", "source-a", 1, 1, "token-a1"),
        ("row-b0", "source-b", 0, 2, "token-b0"),
        ("row-b1", "source-b", 1, 3, "token-b1"),
    )
    for row_id, source_node_id, source_row_index, ingest_sequence, token_id in rows:
        _insert_row_with_tokens(
            engine,
            row_id=row_id,
            source_node_id=source_node_id,
            source_row_index=source_row_index,
            ingest_sequence=ingest_sequence,
            token_ids=(token_id,),
            now=now,
        )

    # Enqueue deliberately scrambled relative to ingest order.
    for row_id, ingest_sequence, token_id in (
        ("row-b1", 3, "token-b1"),
        ("row-a0", 0, "token-a0"),
        ("row-b0", 2, "token-b0"),
        ("row-a1", 1, "token-a1"),
    ):
        repo.enqueue_ready(
            run_id=RUN_ID,
            token_id=token_id,
            row_id=row_id,
            node_id="normalize",
            step_index=1,
            ingest_sequence=ingest_sequence,
            available_at=now,
            row_payload_json=payload,
        )

    claimed = _claim_tokens_in_order(repo, lease_owner="worker-a", now=now + timedelta(seconds=1))

    assert claimed == ["token-a0", "token-a1", "token-b0", "token-b1"]


def test_claim_ready_ties_resolve_by_step_index_then_created_at_then_work_item_id() -> None:
    """Within one ingest_sequence, claim order is step_index, then created_at, then the
    deterministic work_item_id last-resort tiebreaker (filigree elspeth-6cb89db535)."""
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _row_payload_json()
    _insert_run_and_nodes(engine, now=now)

    # Four sibling tokens on one row (fork-family shape): all share ingest_sequence 0.
    _insert_row_with_tokens(
        engine,
        row_id="row-0",
        source_node_id="source-a",
        source_row_index=0,
        ingest_sequence=0,
        token_ids=("token-w", "token-x", "token-y", "token-z"),
        now=now,
    )

    items = {}
    # token-w: later step_index, earliest created_at — must claim LAST.
    # token-x: step 1 but later created_at — claims after the step-1/t0 pair.
    # token-y / token-z: exact (step_index, created_at) collision — tiebreak on work_item_id.
    for token_id, step_index, available_at in (
        ("token-w", 2, now),
        ("token-x", 1, now + timedelta(seconds=5)),
        ("token-y", 1, now),
        ("token-z", 1, now),
    ):
        items[token_id] = repo.enqueue_ready(
            run_id=RUN_ID,
            token_id=token_id,
            row_id="row-0",
            node_id="normalize",
            step_index=step_index,
            ingest_sequence=0,
            available_at=available_at,
            row_payload_json=payload,
        )

    claimed = _claim_tokens_in_order(repo, lease_owner="worker-a", now=now + timedelta(seconds=10))

    tied_pair = sorted(("token-y", "token-z"), key=lambda token_id: items[token_id].work_item_id)
    assert claimed == [*tied_pair, "token-x", "token-w"]


def test_live_lease_is_not_reclaimable_and_expired_lease_recovers_exactly_once() -> None:
    """A live lease blocks claim and recovery; recovery never reaps the caller's own
    lease (G1 self-steal guard); an expired lease is recovered exactly once with an
    attempt bump and work_item_id rotation, then is claimable exactly once."""
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _row_payload_json()
    _insert_run_and_nodes(engine, now=now)
    _insert_row_with_tokens(
        engine,
        row_id="row-0",
        source_node_id="source-a",
        source_row_index=0,
        ingest_sequence=0,
        token_ids=("token-1",),
        now=now,
    )
    original = repo.enqueue_ready(
        run_id=RUN_ID,
        token_id="token-1",
        row_id="row-0",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )

    claimed = repo.claim_ready(run_id=RUN_ID, lease_owner="worker-a", lease_seconds=30, now=now)
    assert claimed is not None
    assert claimed.attempt == 1

    # Live lease: not claimable by a peer, not recoverable by anyone.
    assert repo.claim_ready(run_id=RUN_ID, lease_owner="worker-b", lease_seconds=30, now=now + timedelta(seconds=1)) is None
    assert repo.recover_expired_leases(run_id=RUN_ID, now=now + timedelta(seconds=10), caller_owner="worker-b") == 0

    # Expired lease: the holder's own recovery sweep must NOT reap it (self-steal guard).
    expired_at = now + timedelta(seconds=31)
    assert repo.recover_expired_leases(run_id=RUN_ID, now=expired_at, caller_owner="worker-a") == 0

    # A peer recovers it exactly once.
    assert repo.recover_expired_leases(run_id=RUN_ID, now=expired_at, caller_owner="worker-b") == 1
    assert repo.recover_expired_leases(run_id=RUN_ID, now=expired_at, caller_owner="worker-b") == 0

    reclaimed = repo.claim_ready(run_id=RUN_ID, lease_owner="worker-b", lease_seconds=30, now=expired_at)
    assert reclaimed is not None
    assert reclaimed.token_id == "token-1"
    assert reclaimed.attempt == 2
    assert reclaimed.work_item_id != original.work_item_id
    # Exactly one recovered continuation exists.
    assert repo.claim_ready(run_id=RUN_ID, lease_owner="worker-b", lease_seconds=30, now=expired_at) is None

    recovery_events = [event for event in _events(engine) if event["event_type"] == SchedulerEventType.RECOVER_EXPIRED_LEASE.value]
    assert len(recovery_events) == 1
    assert recovery_events[0]["from_attempt"] == 1
    assert recovery_events[0]["to_attempt"] == 2
    assert recovery_events[0]["caller_owner"] == "worker-b"


def test_pending_sink_parked_token_does_not_block_later_tokens_and_drains_afterward() -> None:
    """A token parked at PENDING_SINK stays non-terminal while a later token is
    claimed and driven to TERMINAL past it; the parked token is then drained via
    claim_pending_sink. The scheduler_events timeline proves the overlap: token-2
    is admitted, advanced, and terminalized strictly between token-1's
    MARK_PENDING_SINK and token-1's MARK_PENDING_SINK_TERMINAL."""
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _row_payload_json()
    _insert_run_and_nodes(engine, now=now)
    for row_id, ingest_sequence, token_id in (("row-0", 0, "token-1"), ("row-1", 1, "token-2")):
        _insert_row_with_tokens(
            engine,
            row_id=row_id,
            source_node_id="source-a",
            source_row_index=ingest_sequence,
            ingest_sequence=ingest_sequence,
            token_ids=(token_id,),
            now=now,
        )
        repo.enqueue_ready(
            run_id=RUN_ID,
            token_id=token_id,
            row_id=row_id,
            node_id="normalize",
            step_index=1,
            ingest_sequence=ingest_sequence,
            available_at=now,
            row_payload_json=payload,
        )

    first = repo.claim_ready(run_id=RUN_ID, lease_owner="worker-a", lease_seconds=300, now=now + timedelta(seconds=1))
    assert first is not None
    assert first.token_id == "token-1"
    repo.mark_pending_sink(
        work_item_id=first.work_item_id,
        row_payload_json=payload,
        sink_name="sink-a",
        outcome="success",
        path="default_flow",
        error_hash=None,
        error_message=None,
        now=now + timedelta(seconds=2),
        expected_lease_owner="worker-a",
    )

    # token-1 is parked non-terminal; token-2 is still claimable past it.
    second = repo.claim_ready(run_id=RUN_ID, lease_owner="worker-a", lease_seconds=300, now=now + timedelta(seconds=3))
    assert second is not None
    assert second.token_id == "token-2"

    with engine.connect() as conn:
        parked_status = conn.execute(
            select(token_work_items_table.c.status).where(token_work_items_table.c.token_id == "token-1")
        ).scalar_one()
    assert parked_status == TokenWorkStatus.PENDING_SINK.value

    repo.mark_terminal(work_item_id=second.work_item_id, now=now + timedelta(seconds=4), expected_lease_owner="worker-a")

    # The parked token drains afterward.
    drained = repo.claim_pending_sink(run_id=RUN_ID, lease_owner="worker-a", lease_seconds=300, now=now + timedelta(seconds=5))
    assert drained is not None
    assert drained.token_id == "token-1"
    assert drained.work_item_id == first.work_item_id
    assert (
        repo.mark_pending_sink_terminal(
            run_id=RUN_ID,
            token_id="token-1",
            now=now + timedelta(seconds=6),
            expected_lease_owner="worker-a",
            coordination_token=COORD_TOKEN,
        )
        == 1
    )

    timeline = [(event["event_type"], event["token_id"]) for event in _events(engine)]
    parked = timeline.index((SchedulerEventType.MARK_PENDING_SINK.value, "token-1"))
    overlap_claim = timeline.index((SchedulerEventType.CLAIM_READY.value, "token-2"))
    overlap_terminal = timeline.index((SchedulerEventType.MARK_TERMINAL.value, "token-2"))
    drained_terminal = timeline.index((SchedulerEventType.MARK_PENDING_SINK_TERMINAL.value, "token-1"))
    assert parked < overlap_claim < overlap_terminal < drained_terminal

    with engine.connect() as conn:
        final_statuses = dict(conn.execute(select(token_work_items_table.c.token_id, token_work_items_table.c.status)).all())
    assert final_statuses == {
        "token-1": TokenWorkStatus.TERMINAL.value,
        "token-2": TokenWorkStatus.TERMINAL.value,
    }
