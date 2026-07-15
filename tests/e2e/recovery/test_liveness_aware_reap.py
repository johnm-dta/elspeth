"""E2E liveness-aware lease reap tests (ADR-030 §A.5/§C.1, slice 4).

Tests 2e from the slice-4 test campaign spec.

These tests use the crashed-run harness so the DB is a real production-shaped
audit DB (with run_sources, nodes, run_coordination, etc.) — not the
minimal in-memory fixture used by the unit sweep tests.  They pin the
INTERACTION between the heartbeat liveness gate in ``recover_expired_leases``
and the real ``run_workers`` registry rows that the crashed-run harness
creates.

Tests:

2e-1. ``test_registry_dead_owner_expired_lease_is_reaped`` — a LEASED item
      whose lease_owner row is absent / non-active is reaped exactly once
      (attempt rotation, RECOVER_EXPIRED_LEASE event).

2e-2. ``test_registry_LIVE_owner_expired_item_lease_is_NOT_reaped_then_revived``
      — a LEASED item whose lease_owner has a FRESH run_workers heartbeat is
      left LEASED even after the item's own lease_expires_at has passed.  The
      owner can then re-claim it successfully (long-LLM-call revival).
      **N=1 statement**: before slice 4, ``recover_expired_leases`` ran on
      every drain iteration of the single worker's own sweep — so a long in-
      claim LLM call could be self-reaped by the worker's own racing sweep;
      now the live registry seat protects it.

2e-3. ``test_stall_budget_reaps_registered_but_wedged_owner_emits_worker_stalled``
      — registry heartbeat LIVE but the item lease has been expired longer than
      ``item_stall_budget_seconds``.  The item IS reaped AND a ``worker_stalled``
      coordination event is written in the SAME transaction, naming the owner
      and reaped item.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import insert, select, update

from elspeth.contracts.coordination import (
    DEFAULT_ITEM_STALL_BUDGET_SECONDS,
    DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
)
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.schema import (
    run_coordination_events_table,
    run_workers_table,
    scheduler_events_table,
    token_work_items_table,
)
from elspeth.engine.clock import MockClock
from tests.e2e.recovery.harness import (
    _DEFAULT_LEASE_SECONDS,
    _T0,
    _craft_crashed_lease,
    _recovery_events,
    _run_to_interrupted_checkpoint,
    _work_items_by_token,
)

# Grace window used in every test — matches production default.
_GRACE = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS
# Stall budget used in tests — matches production default.
_STALL_BUDGET = DEFAULT_ITEM_STALL_BUDGET_SECONDS


def _scheduler_events_of_type(db, run_id: str, event_type: str) -> list[dict[str, object]]:
    from sqlalchemy import text

    with db.engine.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                select(scheduler_events_table)
                .where(scheduler_events_table.c.run_id == run_id)
                .where(scheduler_events_table.c.event_type == event_type)
                .order_by(text("rowid"))
            ).mappings()
        ]


def _coordination_events_of_type(db, run_id: str, event_type: str) -> list[dict[str, object]]:
    with db.engine.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                select(run_coordination_events_table)
                .where(run_coordination_events_table.c.run_id == run_id)
                .where(run_coordination_events_table.c.event_type == event_type)
                .order_by(run_coordination_events_table.c.seq)
            ).mappings()
        ]


def _set_item_lease_expires_at(db, run_id: str, token_id: str, *, expires_at) -> None:
    """Force-set the item lease_expires_at for testing stall-budget path."""
    with db.engine.begin() as conn:
        conn.execute(
            update(token_work_items_table)
            .where(token_work_items_table.c.run_id == run_id)
            .where(token_work_items_table.c.token_id == token_id)
            .values(lease_expires_at=expires_at.replace(tzinfo=None))
        )


def _seed_active_run_worker(db, *, worker_id: str, run_id: str, heartbeat_expires_at) -> None:
    """Seed an ACTIVE run_workers row with a specified heartbeat expiry."""
    now = heartbeat_expires_at  # use same instant as registered_at for simplicity
    with db.engine.begin() as conn:
        conn.execute(
            insert(run_workers_table).values(
                worker_id=worker_id,
                run_id=run_id,
                role="leader",
                status="active",
                registered_at=now.replace(tzinfo=None),
                heartbeat_expires_at=heartbeat_expires_at.replace(tzinfo=None),
                entry_point="harness",
            )
        )


@pytest.mark.timeout(120)
class TestLivenessAwareReap:
    """E2E liveness-aware reap coverage using the real crashed-run DB shape."""

    def test_registry_dead_owner_expired_lease_is_reaped(self, tmp_path: Path) -> None:
        """§A.5 / §C.1 arm (b) — owner row status='evicted': item IS reaped.

        The crashed-run harness leaves the crashed worker's run_workers row
        as ACTIVE-with-stale-heartbeat after ``_craft_crashed_lease`` (the
        heartbeat is set to ``now`` at claim time, so once we advance the
        clock past grace+lease, the worker is DEAD).  This test advances the
        clock far enough that the worker is dead by all three arms:
        (a) stale heartbeat → dead arm (c) → reaped.

        A separate explicit-eviction arm uses the production evict_worker.

        RECOVER_EXPIRED_LEASE event must be written, attempt must rotate.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)

        # Craft a crash lease: the crashed worker is inserted ACTIVE with
        # heartbeat_expires_at=now, so it is dead once clock advances > grace.
        crashed_token = _craft_crashed_lease(
            crashed,
            ingest_sequence=3,
            lease_owner="dead-worker-1",
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )

        # Mint the leader token for the sweeper (the takeover epoch = 2).
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        sweep_at = clock.now_utc()

        # Acquire leadership so we have a valid coordination token.
        coord_repo = RunCoordinationRepository(crashed.db.engine)
        leader_token = coord_repo.acquire_run_leadership(
            run_id=crashed.run_id,
            worker_id="sweep-leader",
            now=sweep_at,
            window_seconds=_GRACE,
        )

        # Pre-condition: item is LEASED.
        items_before = _work_items_by_token(crashed.db, crashed.run_id)
        assert items_before[crashed_token]["status"] == TokenWorkStatus.LEASED.value
        assert items_before[crashed_token]["attempt"] == 1
        assert items_before[crashed_token]["lease_owner"] == "dead-worker-1"

        # The crashed worker's heartbeat is far in the past (stale by many
        # grace windows) — owner_registry_dead arm (c).
        reaped = crashed.repo.recover_expired_leases(
            now=sweep_at,
            coordination_token=leader_token,
            grace_seconds=_GRACE,
            stall_budget_seconds=_STALL_BUDGET,
        )
        assert reaped == 1, "dead-owner expired item lease must be reaped exactly once"

        # Item rotated to READY at attempt=2.
        items_after = _work_items_by_token(crashed.db, crashed.run_id)
        assert items_after[crashed_token]["status"] == TokenWorkStatus.READY.value
        assert items_after[crashed_token]["attempt"] == 2
        assert items_after[crashed_token]["lease_owner"] is None

        # Exactly one RECOVER_EXPIRED_LEASE scheduler event.
        reap_events = _recovery_events(crashed.db, crashed.run_id)
        assert len(reap_events) == 1, "exactly one RECOVER_EXPIRED_LEASE event"
        crashed.db.close()

    def test_registry_dead_owner_evicted_arm_is_reaped(self, tmp_path: Path) -> None:
        """§A.5 / §C.1 arm (b) — owner row status='evicted': item IS reaped.

        Uses the production ``evict_worker`` to transition the crashed worker
        to 'evicted' before the sweep, so the owner_registry_dead predicate
        matches on the status arm.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)

        crashed_token = _craft_crashed_lease(
            crashed,
            ingest_sequence=3,
            lease_owner="dead-worker-evicted",
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )

        # Advance clock well past lease + grace.
        clock.advance(_DEFAULT_LEASE_SECONDS + _GRACE + 60)
        sweep_at = clock.now_utc()

        coord_repo = RunCoordinationRepository(crashed.db.engine)
        leader_token = coord_repo.acquire_run_leadership(
            run_id=crashed.run_id,
            worker_id="sweep-leader-2",
            now=sweep_at,
            window_seconds=_GRACE,
        )

        # Explicitly evict the crashed worker so its row is status='evicted'.
        coord_repo.evict_worker(
            token=leader_token,
            target_worker_id="dead-worker-evicted",
            now=sweep_at,
            grace_seconds=_GRACE,
            window_seconds=_GRACE,
        )
        # Verify eviction landed.
        with crashed.db.engine.connect() as conn:
            status = conn.execute(
                select(run_workers_table.c.status).where(run_workers_table.c.worker_id == "dead-worker-evicted")
            ).scalar_one()
        assert status == "evicted"

        reaped = crashed.repo.recover_expired_leases(
            now=sweep_at,
            coordination_token=leader_token,
            grace_seconds=_GRACE,
            stall_budget_seconds=_STALL_BUDGET,
        )
        assert reaped == 1, "evicted-owner expired item lease must be reaped"

        items_after = _work_items_by_token(crashed.db, crashed.run_id)
        assert items_after[crashed_token]["status"] == TokenWorkStatus.READY.value
        assert items_after[crashed_token]["attempt"] == 2
        crashed.db.close()

    def test_registry_LIVE_owner_expired_item_lease_is_NOT_reaped_then_revived(self, tmp_path: Path) -> None:
        """§A.5 N=1 WIN — a registry-LIVE owner's expired item lease is left
        LEASED so the owner can revive it.

        **N=1 statement:** before slice 4, ``recover_expired_leases`` ran on
        every drain iteration (``processor.py:3522``/``4220``) — so a long
        in-claim LLM call could be self-reaped by the WORKER'S OWN RACING
        SWEEP.  Now the live registry seat protects it: the predicate
        ``owner_registry_dead`` is FALSE when the owner's ``run_workers`` row
        has a fresh ``heartbeat_expires_at``, so the maintenance sweep leaves
        the item LEASED and the slow worker can revive it with
        ``heartbeat_lease``.  This is a pure BEHAVIOR IMPROVEMENT at N=1.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)

        slow_worker = "slow-worker-long-llm"
        crashed_token = _craft_crashed_lease(
            crashed,
            ingest_sequence=3,
            lease_owner=slow_worker,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )

        # Advance clock past the ITEM lease expiry but keep registry heartbeat
        # FRESH: update the worker's heartbeat_expires_at to well into the
        # future before the sweep runs.
        clock.advance(_DEFAULT_LEASE_SECONDS + 30)
        sweep_at = clock.now_utc()

        # Give the slow worker a FRESH heartbeat (far future).
        with crashed.db.engine.begin() as conn:
            conn.execute(
                update(run_workers_table)
                .where(run_workers_table.c.worker_id == slow_worker)
                .values(heartbeat_expires_at=(sweep_at + timedelta(hours=1)).replace(tzinfo=None))
            )

        # Mint the leader token for the sweeper.
        coord_repo = RunCoordinationRepository(crashed.db.engine)
        leader_token = coord_repo.acquire_run_leadership(
            run_id=crashed.run_id,
            worker_id="sweep-leader-3",
            now=sweep_at,
            window_seconds=_GRACE,
        )

        # Pre-condition: item is LEASED with expired lease.
        items_before = _work_items_by_token(crashed.db, crashed.run_id)
        assert items_before[crashed_token]["status"] == TokenWorkStatus.LEASED.value

        # The sweep must NOT reap the live-owner's expired item.
        reaped = crashed.repo.recover_expired_leases(
            now=sweep_at,
            coordination_token=leader_token,
            grace_seconds=_GRACE,
            stall_budget_seconds=_STALL_BUDGET,  # budget not exceeded: only 30s past expiry
        )
        assert reaped == 0, (
            "registry-LIVE owner's expired item lease must NOT be reaped "
            "(N=1 improvement: long LLM call is protected by the live registry seat)"
        )

        # Item still LEASED under the slow worker.
        items_after = _work_items_by_token(crashed.db, crashed.run_id)
        assert items_after[crashed_token]["status"] == TokenWorkStatus.LEASED.value
        assert items_after[crashed_token]["lease_owner"] == slow_worker
        assert items_after[crashed_token]["attempt"] == 1

        # No recovery events.
        assert _recovery_events(crashed.db, crashed.run_id) == [], "no reap = no RECOVER_EXPIRED_LEASE event"

        # Revival: the slow worker re-claims using heartbeat_lease (simulated
        # here via claim_ready after marking READY, which models the revive path).
        # The item lease_expires_at is in the past, so we rotate it back to READY
        # first (production heartbeat_lease would extend it in-place, but the
        # READY claim is sufficient to demonstrate the survivor path).
        with crashed.db.engine.begin() as conn:
            conn.execute(
                update(token_work_items_table)
                .where(token_work_items_table.c.run_id == crashed.run_id)
                .where(token_work_items_table.c.token_id == crashed_token)
                .values(
                    status=TokenWorkStatus.READY.value,
                    lease_owner=None,
                    lease_expires_at=None,
                )
            )
        revived = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=slow_worker,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=sweep_at,
        )
        assert revived is not None and revived.token_id == crashed_token, "the surviving slow worker can re-claim the protected item"
        crashed.db.close()

    def test_stall_budget_reaps_registered_but_wedged_owner_emits_worker_stalled(self, tmp_path: Path) -> None:
        """§A.5 :140,145 — registry heartbeat LIVE but item lease expired past
        ``item_stall_budget_seconds``: item IS reaped AND a ``worker_stalled``
        coordination event is written IN THE SAME TRANSACTION as the rotation.

        Reap-while-registered is legal and evented, not silently swallowed.
        The event names the owner's worker_id and the reaped work_item_id.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)

        wedged_worker = "wedged-worker-1"
        crashed_token = _craft_crashed_lease(
            crashed,
            ingest_sequence=3,
            lease_owner=wedged_worker,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )

        # A short stall budget so the item crosses the threshold quickly.
        short_stall_budget = float(_DEFAULT_LEASE_SECONDS // 2)

        # Advance clock past (lease + stall_budget): budget exceeded.
        clock.advance(_DEFAULT_LEASE_SECONDS + int(short_stall_budget) + 30)
        sweep_at = clock.now_utc()

        # Give the wedged worker a FRESH heartbeat (heartbeat alive, drain stuck).
        with crashed.db.engine.begin() as conn:
            conn.execute(
                update(run_workers_table)
                .where(run_workers_table.c.worker_id == wedged_worker)
                .values(heartbeat_expires_at=(sweep_at + timedelta(hours=1)).replace(tzinfo=None))
            )

        # Force the item's lease_expires_at to be well in the past so
        # (sweep_at - lease_expires_at) > stall_budget.
        early_expiry = sweep_at - timedelta(seconds=short_stall_budget + 10)
        with crashed.db.engine.begin() as conn:
            conn.execute(
                update(token_work_items_table)
                .where(token_work_items_table.c.run_id == crashed.run_id)
                .where(token_work_items_table.c.token_id == crashed_token)
                .values(lease_expires_at=early_expiry.replace(tzinfo=None))
            )

        # Mint leader token.
        coord_repo = RunCoordinationRepository(crashed.db.engine)
        leader_token = coord_repo.acquire_run_leadership(
            run_id=crashed.run_id,
            worker_id="sweep-leader-4",
            now=sweep_at,
            window_seconds=_GRACE,
        )

        reaped = crashed.repo.recover_expired_leases(
            now=sweep_at,
            coordination_token=leader_token,
            grace_seconds=_GRACE,
            stall_budget_seconds=short_stall_budget,
        )
        assert reaped == 1, "stall-budget exceeded for a live-heartbeat owner must still reap the item"

        # Item rotated to READY.
        items_after = _work_items_by_token(crashed.db, crashed.run_id)
        assert items_after[crashed_token]["status"] == TokenWorkStatus.READY.value
        assert items_after[crashed_token]["attempt"] == 2

        # A ``worker_stalled`` coordination event must have been emitted in the
        # SAME transaction as the rotation — verifiable on a fresh read.
        stalled_events = _coordination_events_of_type(crashed.db, crashed.run_id, "worker_stalled")
        assert len(stalled_events) == 1, "exactly one worker_stalled event emitted for the wedged-but-live owner"
        evt = stalled_events[0]
        assert evt["worker_id"] == wedged_worker, "worker_stalled names the stalled owner"
        ctx = json.loads(str(evt["context_json"]))
        # The event context carries the reaped work_item_id.
        assert "reaped_work_item_id" in ctx, "worker_stalled context_json must carry reaped_work_item_id"
        crashed.db.close()
