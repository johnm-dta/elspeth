"""Unit tests for evict_worker housekeeping sweep (§C.2 path 1, slice 4).

Design §C.2 :226-236 specifies the INDIVIDUAL (not bulk) eviction of dead
non-leader workers by the leader maintenance sweep:

1. evict_worker is called per-member, not in bulk.
2. The grace_seconds CAS guards inside evict_worker make it idempotent
   (benign skip when the worker heartbeated between the dead-list read and the
   evict write, or when it already holds live item leases).
3. Ordering: eviction BEFORE reap so reaped items' owners are already
   status='evicted' (owner_registry_dead arm b) — no worker_stalled emitted
   for already-evicted members.
4. The processor's _run_scheduler_maintenance wires (i) dead_non_leader_workers
   read, then (ii) evict_worker per dead member, then (iii) recover_expired_leases.

Tests:
A. test_individual_not_bulk: two dead followers evicted by separate calls.
B. test_eviction_before_reap_ordering: dead member with expired item lease —
   housekeeping evicts THEN reaps; assert status='evicted' at reap time (arm b),
   no worker_stalled emitted.
C. test_evict_worker_grace_cas_miss_idempotent: CAS miss (worker heartbeated
   between dead-list scan and evict) returns False, no eviction event.
D. test_evict_worker_live_lease_blocks_eviction: no-unexpired-leases
   precondition (evict_worker returns False when target holds a live item lease).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, insert, select

from elspeth.contracts.coordination import (
    DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
    CoordinationToken,
)
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    metadata,
    nodes_table,
    rows_table,
    run_coordination_events_table,
    run_coordination_table,
    run_workers_table,
    runs_table,
    token_work_items_table,
    tokens_table,
)

RUN_ID = "run-evict-housekeeping"
NOW = datetime(2026, 6, 13, 10, 0, 0, tzinfo=UTC)
WINDOW = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS  # 80 s
GRACE = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS


def _make_engine() -> Tier1Engine:
    engine = create_engine("sqlite:///:memory:", echo=False)
    LandscapeDB._configure_sqlite(engine)
    LandscapeDB._verify_sqlite_pragmas(engine, "sqlite:///:memory:")
    metadata.create_all(engine)
    return Tier1Engine(engine)


def _seed_run(engine: Tier1Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=RUN_ID,
                started_at=NOW,
                config_hash="config",
                settings_json="{}",
                canonical_version="v1",
                status="running",
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        for node_id, node_type, plugin in (
            ("source-a", "source", "csv"),
            ("transform-1", "transform", "identity"),
        ):
            conn.execute(
                insert(nodes_table).values(
                    run_id=RUN_ID,
                    node_id=node_id,
                    plugin_name=plugin,
                    node_type=node_type,
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="config",
                    config_json="{}",
                    registered_at=NOW,
                )
            )


def _seed_leader(engine: Tier1Engine, *, leader_id: str, now: datetime = NOW) -> CoordinationToken:
    """Mint the run_coordination seat for leader_id and return a token."""
    with engine.begin() as conn:
        conn.execute(
            insert(run_coordination_table).values(
                run_id=RUN_ID,
                leader_worker_id=leader_id,
                leader_epoch=1,
                leader_heartbeat_expires_at=now + timedelta(seconds=WINDOW),
                updated_at=now,
            )
        )
        conn.execute(
            insert(run_workers_table).values(
                worker_id=leader_id,
                run_id=RUN_ID,
                role="leader",
                status="active",
                registered_at=now,
                heartbeat_expires_at=now + timedelta(seconds=WINDOW),
            )
        )
    return CoordinationToken(run_id=RUN_ID, worker_id=leader_id, leader_epoch=1)


def _seed_follower(
    engine: Tier1Engine,
    *,
    worker_id: str,
    now: datetime = NOW,
    heartbeat_expires_at: datetime,
) -> None:
    """Seed an active follower row."""
    with engine.begin() as conn:
        conn.execute(
            insert(run_workers_table).values(
                worker_id=worker_id,
                run_id=RUN_ID,
                role="follower",
                status="active",
                registered_at=now,
                heartbeat_expires_at=heartbeat_expires_at,
            )
        )


def _worker_status(engine: Tier1Engine, worker_id: str) -> str | None:
    with engine.connect() as conn:
        row = conn.execute(
            select(run_workers_table.c.status).where(run_workers_table.c.worker_id == worker_id)
        ).one_or_none()
    return None if row is None else str(row[0])


def _coordination_events(engine: Tier1Engine, event_type: str) -> list[dict[str, object]]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(run_coordination_events_table)
            .where(run_coordination_events_table.c.run_id == RUN_ID)
            .where(run_coordination_events_table.c.event_type == event_type)
            .order_by(run_coordination_events_table.c.seq)
        ).mappings().all()
    return [dict(r) for r in rows]


def _seed_leased_item(
    engine: Tier1Engine,
    *,
    token_id: str,
    lease_owner: str,
    now: datetime,
    lease_seconds: int = 300,
) -> str:
    """Seed a LEASED item row directly (bypassing claim_ready for simplicity)."""
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract

    row_id = f"row-{token_id}"
    with engine.begin() as conn:
        conn.execute(insert(rows_table).values(
            row_id=row_id,
            run_id=RUN_ID,
            source_node_id="source-a",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            source_data_hash=f"hash-{token_id}",
            created_at=now,
        ))
        conn.execute(insert(tokens_table).values(
            token_id=token_id,
            row_id=row_id,
            run_id=RUN_ID,
            created_at=now,
        ))
    repo = TokenSchedulerRepository(engine)
    payload = TokenSchedulerRepository.serialize_row_payload(
        PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
    )
    repo.enqueue_ready(
        run_id=RUN_ID,
        token_id=token_id,
        row_id=row_id,
        node_id="transform-1",
        step_index=1,
        ingest_sequence=0,
        row_payload_json=payload,
        available_at=now,
    )
    item = repo.claim_ready(run_id=RUN_ID, lease_owner=lease_owner, lease_seconds=lease_seconds, now=now)
    assert item is not None
    return item.work_item_id


class TestEvictWorkerHousekeepingIndividualNotBulk:
    """§C.2 :233: individual eviction, not bulk. Two dead followers evicted
    by separate ``evict_worker`` calls in the housekeeping sweep."""

    def test_two_dead_followers_each_evicted_by_separate_call(self) -> None:
        """dead_non_leader_workers returns both expired followers; each is
        evicted by a separate evict_worker call and produces a worker_evict
        event with reason=liveness_expired."""
        engine = _make_engine()
        coord = RunCoordinationRepository(engine)
        _seed_run(engine)

        sweep_at = NOW + timedelta(seconds=200)  # well past grace threshold
        leader_id = "leader-w"
        token = _seed_leader(engine, leader_id=leader_id, now=NOW)

        # Two followers with expired heartbeats.
        follower_a, follower_b = "follower-a", "follower-b"
        expired_hb = NOW - timedelta(seconds=GRACE + 10)
        _seed_follower(engine, worker_id=follower_a, now=NOW, heartbeat_expires_at=expired_hb)
        _seed_follower(engine, worker_id=follower_b, now=NOW, heartbeat_expires_at=expired_hb)

        dead = coord.dead_non_leader_workers(
            run_id=RUN_ID,
            leader_worker_id=leader_id,
            now=sweep_at,
            grace_seconds=GRACE,
        )
        assert set(dead) == {follower_a, follower_b}

        # Evict each individually.
        evicted_a = coord.evict_worker(
            token=token, target_worker_id=follower_a, now=sweep_at,
            grace_seconds=GRACE, window_seconds=WINDOW,
        )
        evicted_b = coord.evict_worker(
            token=token, target_worker_id=follower_b, now=sweep_at,
            grace_seconds=GRACE, window_seconds=WINDOW,
        )
        assert evicted_a is True
        assert evicted_b is True

        assert _worker_status(engine, follower_a) == "evicted"
        assert _worker_status(engine, follower_b) == "evicted"

        events = _coordination_events(engine, "worker_evict")
        assert len(events) == 2
        evicted_ids = {json.loads(str(e["context_json"]))["evicted_by_worker_id"] for e in events}
        assert evicted_ids == {leader_id}
        reasons = {json.loads(str(e["context_json"]))["reason"] for e in events}
        assert reasons == {"liveness_expired"}

    def test_dead_non_leader_workers_excludes_leader_and_live_followers(self) -> None:
        """dead_non_leader_workers must NOT include the leader itself or any
        follower with a fresh heartbeat."""
        engine = _make_engine()
        coord = RunCoordinationRepository(engine)
        _seed_run(engine)

        leader_id = "leader-w"
        _seed_leader(engine, leader_id=leader_id, now=NOW)

        # One live follower, one dead follower.
        live_follower = "follower-live"
        dead_follower = "follower-dead"
        _seed_follower(engine, worker_id=live_follower, now=NOW,
                       heartbeat_expires_at=NOW + timedelta(hours=1))
        _seed_follower(engine, worker_id=dead_follower, now=NOW,
                       heartbeat_expires_at=NOW - timedelta(seconds=GRACE + 10))

        sweep_at = NOW + timedelta(seconds=200)
        dead = coord.dead_non_leader_workers(
            run_id=RUN_ID,
            leader_worker_id=leader_id,
            now=sweep_at,
            grace_seconds=GRACE,
        )
        assert dead == (dead_follower,)  # tuple, deterministic by registered_at

    def test_evict_worker_cas_miss_on_fresh_heartbeat_returns_false(self) -> None:
        """If the target worker heartbeated between the dead-list scan and the
        evict write, the heartbeat_expires_at CAS misses and evict_worker
        returns False (benign skip). No eviction event is written."""
        engine = _make_engine()
        coord = RunCoordinationRepository(engine)
        _seed_run(engine)

        leader_id = "leader-w"
        token = _seed_leader(engine, leader_id=leader_id, now=NOW)

        # Follower has a FRESH heartbeat — the grace CAS will miss.
        fresh_follower = "follower-fresh"
        _seed_follower(engine, worker_id=fresh_follower, now=NOW,
                       heartbeat_expires_at=NOW + timedelta(hours=1))

        sweep_at = NOW + timedelta(seconds=200)
        result = coord.evict_worker(
            token=token, target_worker_id=fresh_follower, now=sweep_at,
            grace_seconds=GRACE, window_seconds=WINDOW,
        )
        assert result is False
        assert _worker_status(engine, fresh_follower) == "active"
        assert _coordination_events(engine, "worker_evict") == []

    def test_evict_worker_live_lease_blocks_eviction(self) -> None:
        """evict_worker's no-unexpired-leases precondition: a worker that still
        holds an unexpired item lease must NOT be evicted (its lease is still
        protecting live in-flight work)."""
        engine = _make_engine()
        coord = RunCoordinationRepository(engine)
        _seed_run(engine)

        leader_id = "leader-w"
        token = _seed_leader(engine, leader_id=leader_id, now=NOW)

        target = "worker-with-lease"
        # Expired heartbeat but holds an UNEXPIRED item lease.
        expired_hb = NOW - timedelta(seconds=GRACE + 10)
        _seed_follower(engine, worker_id=target, now=NOW, heartbeat_expires_at=expired_hb)
        _seed_leased_item(engine, token_id="token-held", lease_owner=target, now=NOW, lease_seconds=600)

        sweep_at = NOW + timedelta(seconds=200)
        result = coord.evict_worker(
            token=token, target_worker_id=target, now=sweep_at,
            grace_seconds=GRACE, window_seconds=WINDOW,
        )
        assert result is False, "evict_worker must refuse when target holds an unexpired lease"
        assert _worker_status(engine, target) == "active"
        assert _coordination_events(engine, "worker_evict") == []


class TestEvictionBeforeReapOrdering:
    """§C.2 :232: evict dead members BEFORE reaping their expired item leases.

    After eviction the member's registry row is status='evicted' (arm b of
    owner_registry_dead), so the subsequent reap is a DEAD-owner reap:
    no worker_stalled event is emitted (that arm is silent on already-evicted).
    """

    def test_eviction_before_reap_yields_silent_dead_arm(self) -> None:
        """Sequence: seed dead follower + expired item lease → evict → reap.
        At reap time the owner's row is status='evicted' (arm b), so the item
        is reaped silently — no worker_stalled coordination event."""
        engine = _make_engine()
        coord = RunCoordinationRepository(engine)
        _seed_run(engine)
        scheduler = TokenSchedulerRepository(engine)

        leader_id = "leader-w"
        token = _seed_leader(engine, leader_id=leader_id, now=NOW)

        dead_member = "dead-follower"
        expired_hb = NOW - timedelta(seconds=GRACE + 10)
        _seed_follower(engine, worker_id=dead_member, now=NOW, heartbeat_expires_at=expired_hb)

        # Seed and expire an item lease under dead_member.
        from sqlalchemy import update

        _seed_leased_item(engine, token_id="token-dead-owned", lease_owner=dead_member, now=NOW, lease_seconds=10)
        # Force-expire the lease.
        sweep_at = NOW + timedelta(seconds=200)
        with engine.begin() as conn:
            conn.execute(
                update(token_work_items_table)
                .where(token_work_items_table.c.run_id == RUN_ID)
                .where(token_work_items_table.c.lease_owner == dead_member)
                .values(lease_expires_at=NOW - timedelta(seconds=1))
            )

        # Step 1: evict before reap.
        evicted = coord.evict_worker(
            token=token, target_worker_id=dead_member, now=sweep_at,
            grace_seconds=GRACE, window_seconds=WINDOW,
        )
        assert evicted is True
        assert _worker_status(engine, dead_member) == "evicted"

        # Step 2: reap. The owner is now status='evicted' → owner_registry_dead (arm b).
        reaped = scheduler.recover_expired_leases(
            run_id=RUN_ID,
            now=sweep_at,
            caller_owner=leader_id,
            coordination_token=token,
            grace_seconds=GRACE,
            stall_budget_seconds=1.0,  # very short budget to force reap regardless
        )
        assert reaped == 1, "expired item lease of evicted member must be reaped"

        # No worker_stalled event: owner was dead (evicted) at reap time.
        assert _coordination_events(engine, "worker_stalled") == []
