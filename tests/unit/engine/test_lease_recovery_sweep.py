"""Characterization tests for multi-item expired-lease recovery sweeps.

These tests pin the CURRENT behavior of ``TokenSchedulerRepository.
recover_expired_leases`` against a real Tier-1 SQLite engine when the sweep
faces a POPULATION of leases rather than a single item (filigree
elspeth-0bae6d8a52):

1. A sweep by a fresh ``lease_owner`` recovers every expired lease in one
   call — exactly once each, with an attempt bump and ``work_item_id``
   rotation — and never touches a live (unexpired) lease.
2. Recovery order is the deterministic 3-key ORDER BY
   ``(ingest_sequence, step_index, work_item_id)`` — including the
   ``work_item_id`` last-resort tiebreaker for exact same-key collisions
   (the same determinism contract as ``claim_ready``, filigree
   elspeth-6cb89db535).
3. The G1 self-steal guard extends PAST lease expiry: an expired lease is
   invisible to its own holder's sweep, even while that same sweep recovers
   other owners' expired leases. Recovery therefore requires a DIFFERENT
   ``lease_owner`` — the resume-sweep path, not in-run self-recovery.

Slice-4 liveness-aware reap tests (§A.5/§C.1, design :140/221-224):

4. A registry-LIVE owner's expired item lease is REVIVED, not reaped —
   the N=1 improvement that makes long LLM calls safe against racing sweeps.
5. A registry-DEAD owner's expired item lease IS reaped under all three
   dead-owner arms: (a) absent row, (b) status='evicted'/'departed',
   (c) status='active' + stale heartbeat.
6. The stall budget arm reaps a live-heartbeat-but-wedged owner and emits
   ``worker_stalled`` in the same transaction.
7. The explicitly named legacy adapter (no run_workers rows) behaves
   identically to pre-slice-4 — legacy reap, no new restrictions.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, insert, select, text, update

from elspeth.contracts import NodeType
from elspeth.contracts.coordination import (
    DEFAULT_ITEM_STALL_BUDGET_SECONDS,
    DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
    CoordinationToken,
)
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkItem, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    metadata,
    nodes_table,
    rows_table,
    run_coordination_events_table,
    run_workers_table,
    runs_table,
    scheduler_events_table,
    token_work_items_table,
    tokens_table,
)

RUN_ID = "run-lease-sweep"


def _make_scheduler_engine() -> Tier1Engine:
    engine = create_engine("sqlite:///:memory:", echo=False)
    LandscapeDB._configure_sqlite(engine)
    LandscapeDB._verify_sqlite_pragmas(engine, "sqlite:///:memory:")
    metadata.create_all(engine)
    return Tier1Engine(engine)


def _row_payload_json() -> str:
    return TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))


def _insert_run_and_nodes(engine: Tier1Engine, *, now: datetime) -> None:
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


def _insert_row_with_tokens(
    engine: Tier1Engine,
    *,
    row_id: str,
    ingest_sequence: int,
    token_ids: tuple[str, ...],
    now: datetime,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            insert(rows_table).values(
                row_id=row_id,
                run_id=RUN_ID,
                source_node_id="source-a",
                row_index=ingest_sequence,
                source_row_index=ingest_sequence,
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


def _enqueue_single_token_rows(
    repo: TokenSchedulerRepository,
    engine: Tier1Engine,
    token_ids: tuple[str, ...],
    *,
    now: datetime,
) -> dict[str, TokenWorkItem]:
    """One row + one token per entry, ingest_sequence in tuple order."""
    payload = _row_payload_json()
    items: dict[str, TokenWorkItem] = {}
    for ingest_sequence, token_id in enumerate(token_ids):
        row_id = f"row-{ingest_sequence}"
        _insert_row_with_tokens(engine, row_id=row_id, ingest_sequence=ingest_sequence, token_ids=(token_id,), now=now)
        items[token_id] = repo.enqueue_ready(
            run_id=RUN_ID,
            token_id=token_id,
            row_id=row_id,
            node_id="normalize",
            step_index=1,
            ingest_sequence=ingest_sequence,
            available_at=now,
            row_payload_json=payload,
        )
    return items


def _work_item_states(engine: Tier1Engine) -> dict[str, dict[str, object]]:
    with engine.connect() as conn:
        return {
            row["token_id"]: dict(row)
            for row in conn.execute(
                select(
                    token_work_items_table.c.token_id,
                    token_work_items_table.c.work_item_id,
                    token_work_items_table.c.status,
                    token_work_items_table.c.attempt,
                    token_work_items_table.c.lease_owner,
                ).where(token_work_items_table.c.run_id == RUN_ID)
            ).mappings()
        }


def _recovery_events(engine: Tier1Engine) -> list[dict[str, object]]:
    """Recovery events in insertion order.

    All events of one sweep share a single ``recorded_at`` and ``event_id`` is
    a non-monotonic opaque id, so SQLite's rowid is the only durable witness
    of the sweep's per-item iteration order.
    """
    with engine.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                select(scheduler_events_table)
                .where(scheduler_events_table.c.run_id == RUN_ID)
                .where(scheduler_events_table.c.event_type == SchedulerEventType.RECOVER_EXPIRED_LEASE.value)
                .order_by(text("rowid"))
            ).mappings()
        ]


def test_sweep_recovers_every_expired_lease_exactly_once_and_never_live_leases() -> None:
    """A fresh-owner sweep over 3 expired + 2 live leases recovers exactly the
    3 expired items in one call — each exactly once with attempt bump and
    work_item_id rotation — leaves the live leases untouched, and is
    idempotent across repeated calls."""
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    _insert_run_and_nodes(engine, now=now)

    token_ids = ("token-0", "token-1", "token-2", "token-3", "token-4")
    originals = _enqueue_single_token_rows(repo, engine, token_ids, now=now)

    # claim_ready admits in ingest_sequence order, so the Nth claim leases
    # token-N. Tokens 0/2/4 get a 30s lease (expired at sweep time); tokens
    # 1/3 get a 3600s lease (still live at sweep time).
    expired_tokens = ("token-0", "token-2", "token-4")
    live_tokens = ("token-1", "token-3")
    lease_seconds_by_token = {"token-0": 30, "token-1": 3600, "token-2": 30, "token-3": 3600, "token-4": 30}
    for token_id in token_ids:
        claimed = repo.claim_ready(run_id=RUN_ID, lease_owner="worker-a", lease_seconds=lease_seconds_by_token[token_id], now=now)
        assert claimed is not None
        assert claimed.token_id == token_id

    sweep_at = now + timedelta(seconds=60)
    assert repo.recover_expired_leases_legacy_unfenced(run_id=RUN_ID, now=sweep_at, caller_owner="resume-sweeper") == 3

    states = _work_item_states(engine)
    for token_id in expired_tokens:
        assert states[token_id]["status"] == TokenWorkStatus.READY.value
        assert states[token_id]["attempt"] == 2
        assert states[token_id]["lease_owner"] is None
        assert states[token_id]["work_item_id"] != originals[token_id].work_item_id
    for token_id in live_tokens:
        assert states[token_id]["status"] == TokenWorkStatus.LEASED.value
        assert states[token_id]["attempt"] == 1
        assert states[token_id]["lease_owner"] == "worker-a"
        assert states[token_id]["work_item_id"] == originals[token_id].work_item_id

    # Exactly one recovery event per expired item, each bumping 1 -> 2.
    events = _recovery_events(engine)
    assert sorted(str(event["token_id"]) for event in events) == sorted(expired_tokens)
    assert all(event["from_attempt"] == 1 and event["to_attempt"] == 2 for event in events)
    assert all(event["caller_owner"] == "resume-sweeper" for event in events)

    # Idempotent: a second sweep finds nothing left to recover.
    assert repo.recover_expired_leases_legacy_unfenced(run_id=RUN_ID, now=sweep_at, caller_owner="resume-sweeper") == 0
    assert len(_recovery_events(engine)) == 3

    # The recovered continuations are claimable in ingest order; the live
    # leases still block their own tokens.
    reclaimed: list[str] = []
    while True:
        item = repo.claim_ready(run_id=RUN_ID, lease_owner="resume-sweeper", lease_seconds=300, now=sweep_at)
        if item is None:
            break
        assert item.attempt == 2
        reclaimed.append(item.token_id)
    assert reclaimed == list(expired_tokens)


def test_sweep_recovery_order_is_ingest_sequence_then_step_index_then_work_item_id() -> None:
    """Multi-item recovery walks expired leases in the deterministic 3-key
    order (ingest_sequence, step_index, work_item_id) — the work_item_id
    last-resort tiebreaker resolves exact same-key collisions, mirroring the
    claim_ready determinism contract (elspeth-6cb89db535)."""
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _row_payload_json()
    _insert_run_and_nodes(engine, now=now)

    # Fork-family shape: three sibling tokens on row-0 (ingest_sequence 0) —
    # token-y/token-z collide exactly on (ingest_sequence=0, step_index=1),
    # token-w trails at step_index=2 — plus token-c on row-1 (ingest_sequence 1).
    _insert_row_with_tokens(engine, row_id="row-0", ingest_sequence=0, token_ids=("token-w", "token-y", "token-z"), now=now)
    _insert_row_with_tokens(engine, row_id="row-1", ingest_sequence=1, token_ids=("token-c",), now=now)
    items: dict[str, TokenWorkItem] = {}
    for token_id, row_id, step_index, ingest_sequence in (
        ("token-w", "row-0", 2, 0),
        ("token-y", "row-0", 1, 0),
        ("token-z", "row-0", 1, 0),
        ("token-c", "row-1", 1, 1),
    ):
        items[token_id] = repo.enqueue_ready(
            run_id=RUN_ID,
            token_id=token_id,
            row_id=row_id,
            node_id="normalize",
            step_index=step_index,
            ingest_sequence=ingest_sequence,
            available_at=now,
            row_payload_json=payload,
        )

    for _ in range(4):
        assert repo.claim_ready(run_id=RUN_ID, lease_owner="worker-a", lease_seconds=30, now=now) is not None

    sweep_at = now + timedelta(seconds=60)
    assert repo.recover_expired_leases_legacy_unfenced(run_id=RUN_ID, now=sweep_at, caller_owner="resume-sweeper") == 4

    tied_pair = sorted(("token-y", "token-z"), key=lambda token_id: items[token_id].work_item_id)
    recovery_order = [event["token_id"] for event in _recovery_events(engine)]
    assert recovery_order == [*tied_pair, "token-w", "token-c"]


def test_expired_lease_is_invisible_to_its_own_holders_sweep() -> None:
    """G1 self-steal guard extends past expiry: a sweep recovers other owners'
    expired leases but NEVER the caller's own — even when the caller's lease is
    itself expired. The wedged item is recovered only when a DIFFERENT
    lease_owner (the resume-sweep path) runs the sweep."""
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    _insert_run_and_nodes(engine, now=now)

    _enqueue_single_token_rows(repo, engine, ("token-0", "token-1"), now=now)

    # claim_ready admits in ingest order: worker-a leases token-0, worker-b
    # leases token-1. Both leases expire before the sweep.
    claimed_a = repo.claim_ready(run_id=RUN_ID, lease_owner="worker-a", lease_seconds=30, now=now)
    claimed_b = repo.claim_ready(run_id=RUN_ID, lease_owner="worker-b", lease_seconds=30, now=now)
    assert claimed_a is not None and claimed_a.token_id == "token-0"
    assert claimed_b is not None and claimed_b.token_id == "token-1"

    sweep_at = now + timedelta(seconds=60)

    # worker-a's sweep recovers ONLY worker-b's expired lease; its own expired
    # lease stays LEASED under worker-a (invisible to its own holder).
    assert repo.recover_expired_leases_legacy_unfenced(run_id=RUN_ID, now=sweep_at, caller_owner="worker-a") == 1
    states = _work_item_states(engine)
    assert states["token-1"]["status"] == TokenWorkStatus.READY.value
    assert states["token-1"]["attempt"] == 2
    assert states["token-0"]["status"] == TokenWorkStatus.LEASED.value
    assert states["token-0"]["attempt"] == 1
    assert states["token-0"]["lease_owner"] == "worker-a"

    # Repeating its own sweep never reaps it.
    assert repo.recover_expired_leases_legacy_unfenced(run_id=RUN_ID, now=sweep_at, caller_owner="worker-a") == 0

    # A different lease_owner — the resume-sweep identity — recovers it.
    assert repo.recover_expired_leases_legacy_unfenced(run_id=RUN_ID, now=sweep_at, caller_owner="resume-sweeper") == 1
    states = _work_item_states(engine)
    assert states["token-0"]["status"] == TokenWorkStatus.READY.value
    assert states["token-0"]["attempt"] == 2
    assert states["token-0"]["lease_owner"] is None

    # No recovery event was ever attributed to the lease's own holder.
    events = _recovery_events(engine)
    assert [(event["token_id"], event["caller_owner"], event["from_lease_owner"]) for event in events] == [
        ("token-1", "worker-a", "worker-b"),
        ("token-0", "resume-sweeper", "worker-a"),
    ]


# =============================================================================
# Slice-4 liveness-aware reap tests (§A.5/§C.1)
# =============================================================================

_LEASE_SECONDS = 30
_SWEEP_GRACE = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS  # 80 s
_STALL_BUDGET = DEFAULT_ITEM_STALL_BUDGET_SECONDS  # 600 s
_WINDOW = 80.0
_EPOCH = 1


def _make_coord_token(engine: Tier1Engine, *, leader_worker_id: str, run_id: str, now: datetime) -> CoordinationToken:
    """Mint the run_coordination seat and return a CoordinationToken for it.

    The fenced recover_expired_leases path calls verify_and_extend_leader_fence
    as its first statement. That requires a run_coordination row with a
    matching leader_worker_id and leader_epoch. We insert a live seat here
    so the fence doesn't refuse before the reap logic runs.
    """
    from elspeth.core.landscape.schema import run_coordination_table

    with engine.begin() as conn:
        conn.execute(
            insert(run_coordination_table).values(
                run_id=run_id,
                leader_worker_id=leader_worker_id,
                leader_epoch=_EPOCH,
                leader_heartbeat_expires_at=now + timedelta(seconds=_WINDOW),
                updated_at=now,
            )
        )
    return CoordinationToken(run_id=run_id, worker_id=leader_worker_id, leader_epoch=_EPOCH)


def _seed_active_run_worker(
    engine: Tier1Engine,
    *,
    worker_id: str,
    run_id: str,
    status: str,
    heartbeat_expires_at: datetime,
    now: datetime,
) -> None:
    """Seed a run_workers row with the given liveness state."""
    with engine.begin() as conn:
        conn.execute(
            insert(run_workers_table).values(
                worker_id=worker_id,
                run_id=run_id,
                role="follower",
                status=status,
                registered_at=now,
                heartbeat_expires_at=heartbeat_expires_at,
                evicted_at=now if status == "evicted" else None,
                evicted_by_worker_id="test-evictor" if status == "evicted" else None,
            )
        )


def _claim_and_expire(
    repo: TokenSchedulerRepository,
    engine: Tier1Engine,
    *,
    run_id: str,
    lease_owner: str,
    now: datetime,
) -> str:
    """Claim the first READY item and expire its lease; return token_id."""
    item = repo.claim_ready(run_id=run_id, lease_owner=lease_owner, lease_seconds=_LEASE_SECONDS, now=now)
    assert item is not None
    # Force-expire the lease by back-dating lease_expires_at on the DB row.
    with engine.begin() as conn:
        conn.execute(
            update(token_work_items_table)
            .where(token_work_items_table.c.work_item_id == item.work_item_id)
            .values(lease_expires_at=now - timedelta(seconds=1))
        )
    return item.token_id


def _coordination_events(engine: Tier1Engine, *, run_id: str, event_type: str) -> list[dict[str, object]]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(run_coordination_events_table)
            .where(run_coordination_events_table.c.run_id == run_id)
            .where(run_coordination_events_table.c.event_type == event_type)
            .order_by(run_coordination_events_table.c.seq)
        ).mappings()
    return [dict(r) for r in rows]


def test_live_registered_owner_expired_lease_is_revived_not_reaped() -> None:
    """§A.5 / §C.1 — N=1 WIN: a registry-LIVE owner's expired item lease is
    left LEASED (owner_registry_dead is False) so the owner's next
    heartbeat_lease call can revive it. Long LLM calls are no longer reapable
    by a racing maintenance sweep.

    Setup: leader sweeper (token), peer worker-alive has a FRESH heartbeat.
    Item is leased under worker-alive with an expired lease_expires_at.
    Sweep by leader must return 0 (not reaped).
    """
    now = datetime.now(UTC)
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    _insert_run_and_nodes(engine, now=now)

    leader_id = "leader-sweeper"
    live_owner = "worker-alive"

    # Leader mints the coordination seat and gets a fencing token.
    token = _make_coord_token(engine, leader_worker_id=leader_id, run_id=RUN_ID, now=now)

    # live_owner has a FRESH run_workers row — heartbeat expires well after grace.
    _seed_active_run_worker(
        engine,
        worker_id=live_owner,
        run_id=RUN_ID,
        status="active",
        heartbeat_expires_at=now + timedelta(hours=1),  # clearly fresh
        now=now,
    )

    # Enqueue and have live_owner claim the item.
    _enqueue_single_token_rows(repo, engine, ("token-live",), now=now)
    token_id = _claim_and_expire(repo, engine, run_id=RUN_ID, lease_owner=live_owner, now=now)

    sweep_at = now + timedelta(seconds=_LEASE_SECONDS + 10)

    # Leader's sweep: recover_expired_leases should NOT reap the item because
    # live_owner's heartbeat is fresh (owner_registry_dead is False) and the
    # lease has NOT passed the stall budget (only 40 s past lease_expires_at).
    reaped = repo.recover_expired_leases(
        now=sweep_at,
        coordination_token=token,
        grace_seconds=_SWEEP_GRACE,
        stall_budget_seconds=_STALL_BUDGET,
    )
    assert reaped == 0, "live-owner expired lease must NOT be reaped by a peer sweep"

    # Row still LEASED under live_owner.
    states = _work_item_states(engine)
    assert states[token_id]["status"] == TokenWorkStatus.LEASED.value
    assert states[token_id]["lease_owner"] == live_owner


@pytest.mark.parametrize(
    ("owner_status", "heartbeat_fresh"),
    [
        ("absent", True),  # arm (a): no run_workers row at all
        ("evicted", True),  # arm (b): status='evicted'
        ("departed", True),  # arm (b): status='departed'
        ("active", False),  # arm (c): status='active' + stale heartbeat
    ],
    ids=["absent-row", "evicted", "departed", "active-stale-heartbeat"],
)
def test_dead_registered_owner_expired_lease_is_reaped(owner_status: str, heartbeat_fresh: bool) -> None:
    """§A.5 / §C.1 — dead-owner arms: (a) absent row, (b) non-active status,
    (c) active-but-stale heartbeat. All three are reaped by a leader sweep.
    Attempt rotation is pinned (attempt 1 → 2, work_item_id rotated for READY).
    """
    now = datetime.now(UTC)
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    _insert_run_and_nodes(engine, now=now)

    leader_id = "leader-sweeper"
    dead_owner = "worker-dead"

    token = _make_coord_token(engine, leader_worker_id=leader_id, run_id=RUN_ID, now=now)

    if owner_status != "absent":
        # heartbeat_fresh=False means heartbeat already expired past grace threshold.
        hb_expires_at = now + timedelta(hours=1) if heartbeat_fresh else now - timedelta(seconds=_SWEEP_GRACE + 1)
        if owner_status in ("evicted", "departed"):
            # Slice-4 membership fence: claim_ready refuses non-active workers.
            # Real lifecycle: worker registers ACTIVE, claims, then gets evicted/
            # departed. Seed as 'active' so claim works, then transition below.
            _seed_active_run_worker(
                engine,
                worker_id=dead_owner,
                run_id=RUN_ID,
                status="active",
                heartbeat_expires_at=hb_expires_at,
                now=now,
            )
        else:
            _seed_active_run_worker(
                engine,
                worker_id=dead_owner,
                run_id=RUN_ID,
                status=owner_status,
                heartbeat_expires_at=hb_expires_at,
                now=now,
            )
    # absent: no run_workers row at all

    _enqueue_single_token_rows(repo, engine, ("token-dead",), now=now)
    token_id = _claim_and_expire(repo, engine, run_id=RUN_ID, lease_owner=dead_owner, now=now)

    # For evicted/departed: transition the worker to the final dead state AFTER
    # claiming. This mirrors the real lifecycle where eviction happens post-claim.
    if owner_status in ("evicted", "departed"):
        with engine.begin() as conn:
            conn.execute(
                update(run_workers_table)
                .where(run_workers_table.c.worker_id == dead_owner)
                .values(
                    status=owner_status,
                    evicted_at=now if owner_status == "evicted" else None,
                    evicted_by_worker_id="test-evictor" if owner_status == "evicted" else None,
                )
            )

    sweep_at = now + timedelta(seconds=_LEASE_SECONDS + 10)

    reaped = repo.recover_expired_leases(
        now=sweep_at,
        coordination_token=token,
        grace_seconds=_SWEEP_GRACE,
        stall_budget_seconds=_STALL_BUDGET,
    )
    assert reaped == 1, f"dead-owner arm={owner_status!r} must be reaped"

    states = _work_item_states(engine)
    assert states[token_id]["status"] == TokenWorkStatus.READY.value
    assert states[token_id]["attempt"] == 2
    assert states[token_id]["lease_owner"] is None

    # No worker_stalled event for a dead-owner reap.
    stalled_events = _coordination_events(engine, run_id=RUN_ID, event_type="worker_stalled")
    assert stalled_events == [], "dead-owner reap must NOT emit worker_stalled"


def test_stall_budget_reaps_live_owner_and_emits_worker_stalled() -> None:
    """§A.5 :145 — stall arm: a registry-LIVE owner (fresh heartbeat) but drain
    loop is wedged. The item has been expired past stall_budget_seconds.
    The reap must succeed AND emit worker_stalled in the same transaction.
    """
    now = datetime.now(UTC)
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    _insert_run_and_nodes(engine, now=now)

    leader_id = "leader-sweeper"
    live_but_wedged = "worker-wedged"

    token = _make_coord_token(engine, leader_worker_id=leader_id, run_id=RUN_ID, now=now)

    # Wedged owner has a FRESH heartbeat — it is registry-LIVE.
    _seed_active_run_worker(
        engine,
        worker_id=live_but_wedged,
        run_id=RUN_ID,
        status="active",
        heartbeat_expires_at=now + timedelta(hours=1),
        now=now,
    )

    _enqueue_single_token_rows(repo, engine, ("token-stalled",), now=now)
    token_id = _claim_and_expire(repo, engine, run_id=RUN_ID, lease_owner=live_but_wedged, now=now)

    # Advance past stall_budget_seconds so the stall arm triggers.
    stall_budget = 60.0  # short custom budget for the test
    sweep_at = now + timedelta(seconds=_LEASE_SECONDS + stall_budget + 10)

    reaped = repo.recover_expired_leases(
        now=sweep_at,
        coordination_token=token,
        grace_seconds=_SWEEP_GRACE,
        stall_budget_seconds=stall_budget,
    )
    assert reaped == 1, "stall-budget arm must reap the item"

    states = _work_item_states(engine)
    assert states[token_id]["status"] == TokenWorkStatus.READY.value
    assert states[token_id]["attempt"] == 2
    assert states[token_id]["lease_owner"] is None

    # worker_stalled event must be emitted for the live-but-wedged owner.
    stalled_events = _coordination_events(engine, run_id=RUN_ID, event_type="worker_stalled")
    assert len(stalled_events) == 1
    ctx = json.loads(str(stalled_events[0]["context_json"]))
    assert ctx["reason"] == "item_stall_budget"
    assert stalled_events[0]["worker_id"] == live_but_wedged
    assert stalled_events[0]["leader_epoch"] == _EPOCH


def test_named_legacy_adapter_without_registry_preserves_reap_semantics() -> None:
    """§C.1 named legacy adapter re-pin: it deliberately does not consult
    run_workers, so all expired leases not owned by its explicit caller are
    rotated regardless of registry liveness. This re-pins the baseline
    contract for direct slice 1-3 repository harnesses.
    """
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    _insert_run_and_nodes(engine, now=now)

    token_ids = ("token-u0", "token-u1", "token-u2")
    _enqueue_single_token_rows(repo, engine, token_ids, now=now)

    # Claim all three under three different owners — NO run_workers rows.
    for token_id, owner in zip(token_ids, ("owner-a", "owner-b", "owner-c"), strict=True):
        item = repo.claim_ready(run_id=RUN_ID, lease_owner=owner, lease_seconds=_LEASE_SECONDS, now=now)
        assert item is not None and item.token_id == token_id

    sweep_at = now + timedelta(seconds=60)

    # Explicit legacy sweep: all three must be reaped.
    reaped = repo.recover_expired_leases_legacy_unfenced(
        run_id=RUN_ID,
        now=sweep_at,
        caller_owner="resume-sweeper",
    )
    assert reaped == 3, "named legacy adapter must reap all expired items not owned by its explicit caller"

    states = _work_item_states(engine)
    for token_id in token_ids:
        assert states[token_id]["status"] == TokenWorkStatus.READY.value
        assert states[token_id]["attempt"] == 2
        assert states[token_id]["lease_owner"] is None
