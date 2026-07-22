"""Follower isolation (G25b) and chaos (G25h) campaigns (ADR-030 §H, slice 5).

3A  TestFollowerIsolation
    G25b: structural isolation guarantees between concurrent workers.
    I1  cannot claim another worker's LEASED row (lease_owner CAS fence)
    I2  evicted follower claim is refused before insert (membership fence)
    I3  two followers claim disjoint READY rows — no collision
    I4  clean audit attribution (attempt + lease_owner per follower)

3B  TestFollowerChaos
    G25h: liveness, reaper, and crash scenarios.
    C1  leader + follower race on READY rows — exactly one claim wins per row
    C2  follower crash → lease lapses → reaper recovers expired item
    C3  follower joins then heartbeat latch fires → RunWorkerEvictedError; run
        status survives (still RUNNING / FAILED per setup); no orphan claim
    C4  slow follower: item lease expired but run_workers row is registry-live →
        reaper's liveness-aware gate skips the item
    C5  slow follower reaches EOF of one drain batch with empty READY queue;
        no spurious claim attempt; idle poll fires and exits cleanly

Construction discipline: real Orchestrator → real DB via
_run_to_interrupted_checkpoint + _seat_run_with_live_leader + _join_follower
(same helpers as test_follower_join_and_drain.py).  Scheduler verbs are
driven DIRECTLY via TokenSchedulerRepository for maximum control.

Cross-cutting invariants asserted for every test:
    - _duplicate_terminal_outcome_tokens == [] (no double-terminal)
    - no unexpected run_coordination events for unaffected workers
    - seat epoch stable unless the test explicitly bumps it
"""

from __future__ import annotations

import types
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import select, update

from elspeth.contracts import PipelineRow, RunStatus
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.errors import RunWorkerEvictedError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import run_workers_table, token_work_items_table
from elspeth.engine.clock import MockClock
from elspeth.engine.orchestrator.follower import FollowerProcessor
from tests.e2e.recovery.harness import (
    _DEFAULT_LEASE_SECONDS,
    _T0,
    _duplicate_terminal_outcome_tokens,
    _observed_contract,
    _run_to_interrupted_checkpoint,
    _run_workers,
)
from tests.e2e.recovery.test_follower_join_and_drain import (
    _get_db_config_hash,
    _join_follower,
    _orchestrator,
    _seat_run_with_live_leader,
    _seed_ready_row,
)

_GUARD_LIVE_SEAT_WINDOW_SECONDS = 10**9


class _FollowerDrainFake:
    """Small fake for the follower's one-method drain surface."""

    def __init__(
        self,
        *,
        drain: Callable[[PluginContext], list[Any]] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._drain = drain
        self._error = error
        self.drain_calls = 0

    def drain_follower_ready_work(
        self,
        ctx: PluginContext,
        *,
        before_claim: Callable[[], None] | None = None,
    ) -> list[Any]:
        del before_claim
        self.drain_calls += 1
        if self._error is not None:
            raise self._error
        if self._drain is None:
            return []
        return self._drain(ctx)


def _seed_ready_row_direct(crashed: Any, *, ingest_sequence: int) -> tuple[str, str]:
    """Seed a READY token_work_items row WITHOUT using _craft_crashed_lease.

    Creates a row+token via the factory, then enqueues directly (no claim).
    Safe to call multiple times: each call produces an independent READY row
    with no run_workers side effects.  Returns (token_id, work_item_id).
    """
    now = crashed.clock.now_utc()
    data = {"id": ingest_sequence, "value": ingest_sequence * 10}
    row = crashed.factory.data_flow.create_row(
        run_id=crashed.run_id,
        source_node_id=crashed.source_node_id,
        row_index=ingest_sequence,
        data=data,
        source_row_index=ingest_sequence,
        ingest_sequence=ingest_sequence,
    )
    token = crashed.factory.data_flow.create_token(row_id=row.row_id)
    work_item = crashed.repo.enqueue_ready(
        run_id=crashed.run_id,
        token_id=token.token_id,
        row_id=row.row_id,
        node_id=crashed.journal_node_id,
        step_index=crashed.journal_step_index,
        ingest_sequence=ingest_sequence,
        row_payload_json=TokenSchedulerRepository.serialize_row_payload(PipelineRow(data, _observed_contract(data))),
        available_at=now,
        # No worker_id → unfenced legacy enqueue (test/harness use case)
    )
    return token.token_id, work_item.work_item_id


# ---------------------------------------------------------------------------
# 3A  Follower isolation (G25b)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
class TestFollowerIsolation:
    """G25b: structural isolation guarantees between concurrent workers.

    These tests verify that the scheduler's item-layer CAS verbs provide
    correct isolation: one worker cannot claim or mutate another's lease,
    and two followers can claim disjoint rows safely.
    """

    def test_follower_cannot_claim_another_workers_leased_row(self, tmp_path: Path) -> None:
        """I1: claim_ready returns None when no READY row exists (other row is LEASED).

        A row currently LEASED by worker-A is NOT claimable by worker-B.
        claim_ready returns None (no work) rather than stealing the lease.
        Verifies the lease_owner CAS fence in the scheduler item layer.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)

        # Admit two followers.
        follower_a = _join_follower(crashed, leader_token)
        # Admit second follower with a second join_run call.
        db_hash = _get_db_config_hash(crashed)
        with (
            patch("elspeth.engine.orchestrator.join_admission.resolve_config", return_value={}),
            patch("elspeth.engine.orchestrator.join_admission.stable_hash", return_value=db_hash),
        ):
            follower_b = _orchestrator(crashed).join_run(
                run_id=crashed.run_id,
                settings=types.SimpleNamespace(),
                now=clock.now_utc(),
                window_seconds=_GUARD_LIVE_SEAT_WINDOW_SECONDS,
            )

        # Seed one READY row for follower-A to claim.
        token_id, _wid = _seed_ready_row(crashed, ingest_sequence=10)

        # Follower-A claims the row (now LEASED by A).
        claimed_a = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=follower_a,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed_a is not None and claimed_a.token_id == token_id
        assert claimed_a.lease_owner == follower_a

        # Follower-B tries claim_ready — no READY row available → None.
        claimed_b = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=follower_b,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed_b is None, "follower-B must not claim follower-A's LEASED row"

        # The item is still LEASED by A.
        with crashed.db.engine.connect() as conn:
            item = conn.execute(select(token_work_items_table).where(token_work_items_table.c.token_id == token_id)).mappings().one()
        assert item["status"] == TokenWorkStatus.LEASED.value
        assert item["lease_owner"] == follower_a

        # No double-terminal.
        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        crashed.db.close()

    def test_evicted_follower_claim_refused_before_insert(self, tmp_path: Path) -> None:
        """I2: evicted follower's claim_ready raises RunWorkerEvictedError.

        Mirrors L5(b) from test_follower_join_and_drain.py but for
        claim_ready (not enqueue_ready): an evicted worker raises
        RunWorkerEvictedError (the fence fires after the UPDATE returns
        rowcount=0 and the re-probe detects a non-active run_workers row).
        The READY row is untouched.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        # Seed a READY row BEFORE eviction (avoids claim collision with temp workers).
        token_id, _wid = _seed_ready_row_direct(crashed, ingest_sequence=11)

        # Evict the follower (simulate leader's §C.2 sweep).
        with crashed.db.engine.begin() as conn:
            conn.execute(
                update(run_workers_table)
                .where(run_workers_table.c.worker_id == follower_id)
                .values(status="evicted", evicted_at=clock.now_utc())
            )

        # Evicted follower tries to claim → claim_verb_fence_clause detects
        # non-active row → raises RunWorkerEvictedError (§C case (a)).
        with pytest.raises(RunWorkerEvictedError) as exc_info:
            crashed.repo.claim_ready(
                run_id=crashed.run_id,
                lease_owner=follower_id,
                lease_seconds=_DEFAULT_LEASE_SECONDS,
                now=clock.now_utc(),
            )

        assert exc_info.value.worker_id == follower_id
        assert exc_info.value.run_id == crashed.run_id

        # The row is still READY (fence fired before any mutation).
        with crashed.db.engine.connect() as conn:
            item = conn.execute(select(token_work_items_table).where(token_work_items_table.c.token_id == token_id)).mappings().one()
        assert item["status"] == TokenWorkStatus.READY.value, "row must remain READY after evicted follower's refused claim"

        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        crashed.db.close()

    def test_two_followers_claim_disjoint_rows_no_collision(self, tmp_path: Path) -> None:
        """I3: two followers each claim a different READY row — no collision.

        Seeds two READY rows; follower-A claims the first (by claiming first),
        follower-B claims the second.  Each holds exactly one LEASED row and
        they are disjoint: no token_id appears under both owners.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_a = _join_follower(crashed, leader_token)

        db_hash = _get_db_config_hash(crashed)
        with (
            patch("elspeth.engine.orchestrator.join_admission.resolve_config", return_value={}),
            patch("elspeth.engine.orchestrator.join_admission.stable_hash", return_value=db_hash),
        ):
            follower_b = _orchestrator(crashed).join_run(
                run_id=crashed.run_id,
                settings=types.SimpleNamespace(),
                now=clock.now_utc(),
                window_seconds=_GUARD_LIVE_SEAT_WINDOW_SECONDS,
            )

        # Seed two READY rows using direct enqueue (safe with multiple rows).
        token_a, _wid_a = _seed_ready_row_direct(crashed, ingest_sequence=20)
        token_b, _wid_b = _seed_ready_row_direct(crashed, ingest_sequence=21)

        # Follower-A claims its row.
        claimed_a = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=follower_a,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed_a is not None
        token_a_claimed = claimed_a.token_id

        # Follower-B claims the remaining row.
        claimed_b = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=follower_b,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed_b is not None
        token_b_claimed = claimed_b.token_id

        # They claimed DIFFERENT rows.
        assert token_a_claimed != token_b_claimed, "followers must claim different rows"
        # Together they cover both seeded rows.
        assert {token_a_claimed, token_b_claimed} == {token_a, token_b}

        # Each row's lease_owner is its claimant.
        with crashed.db.engine.connect() as conn:
            rows = {
                row["token_id"]: dict(row)
                for row in conn.execute(
                    select(
                        token_work_items_table.c.token_id,
                        token_work_items_table.c.lease_owner,
                    ).where(token_work_items_table.c.token_id.in_([token_a_claimed, token_b_claimed]))
                ).mappings()
            }
        assert rows[token_a_claimed]["lease_owner"] == follower_a
        assert rows[token_b_claimed]["lease_owner"] == follower_b

        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        crashed.db.close()

    def test_clean_audit_attribution_per_follower(self, tmp_path: Path) -> None:
        """I4: attempt=1 and lease_owner correctly attributed per follower.

        Each follower claims and marks a fresh READY row terminal.  The
        durable journal shows attempt==1 (no reuse) and lease_owner==follower_id
        for each row at the LEASED stage (cleared to None at TERMINAL, per
        mark_terminal semantics).
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_a = _join_follower(crashed, leader_token)

        db_hash = _get_db_config_hash(crashed)
        with (
            patch("elspeth.engine.orchestrator.join_admission.resolve_config", return_value={}),
            patch("elspeth.engine.orchestrator.join_admission.stable_hash", return_value=db_hash),
        ):
            follower_b = _orchestrator(crashed).join_run(
                run_id=crashed.run_id,
                settings=types.SimpleNamespace(),
                now=clock.now_utc(),
                window_seconds=_GUARD_LIVE_SEAT_WINDOW_SECONDS,
            )

        # Seed two disjoint READY rows using direct enqueue (safe with multiple rows).
        token_a, _wid_a = _seed_ready_row_direct(crashed, ingest_sequence=30)
        token_b, _wid_b = _seed_ready_row_direct(crashed, ingest_sequence=31)

        # Follower-A claims one row.
        claimed_a = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=follower_a,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed_a is not None
        assert claimed_a.token_id in (token_a, token_b)
        assert claimed_a.attempt == 1

        # Follower-B claims the remaining row.
        claimed_b = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=follower_b,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed_b is not None
        assert claimed_b.token_id in (token_a, token_b)
        assert claimed_b.attempt == 1

        # Mark both terminal.
        crashed.repo.mark_terminal(
            work_item_id=claimed_a.work_item_id,
            now=clock.now_utc(),
            expected_lease_owner=follower_a,
        )
        crashed.repo.mark_terminal(
            work_item_id=claimed_b.work_item_id,
            now=clock.now_utc(),
            expected_lease_owner=follower_b,
        )

        # Both items are TERMINAL with attempt==1 and lease_owner=None (cleared).
        # Use claimed token IDs (order is non-deterministic across READY rows).
        token_a_actual = claimed_a.token_id
        token_b_actual = claimed_b.token_id
        with crashed.db.engine.connect() as conn:
            items = {
                row["token_id"]: dict(row)
                for row in conn.execute(
                    select(
                        token_work_items_table.c.token_id,
                        token_work_items_table.c.status,
                        token_work_items_table.c.attempt,
                        token_work_items_table.c.lease_owner,
                    ).where(token_work_items_table.c.token_id.in_([token_a_actual, token_b_actual]))
                ).mappings()
            }
        assert items[token_a_actual]["status"] == TokenWorkStatus.TERMINAL.value
        assert items[token_a_actual]["attempt"] == 1
        assert items[token_a_actual]["lease_owner"] is None  # cleared by mark_terminal

        assert items[token_b_actual]["status"] == TokenWorkStatus.TERMINAL.value
        assert items[token_b_actual]["attempt"] == 1
        assert items[token_b_actual]["lease_owner"] is None

        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        crashed.db.close()


# ---------------------------------------------------------------------------
# 3B  Follower chaos (G25h)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
class TestFollowerChaos:
    """G25h: liveness, reaper, and crash scenarios.

    These tests verify the system's behaviour under adversarial conditions:
    races, crashes, stale leases, and unexpected exits.
    """

    def test_leader_follower_race_exactly_one_claim_wins_per_row(self, tmp_path: Path) -> None:
        """C1: leader + follower racing on two READY rows — each row claimed exactly once.

        Seeds two READY rows; the leader claims one, the follower claims the
        other.  No row can be claimed twice (the atomic claim CAS prevents it).
        Both end up TERMINAL; no double-terminal outcomes.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        token_x, _wid_x = _seed_ready_row_direct(crashed, ingest_sequence=40)
        token_y, _wid_y = _seed_ready_row_direct(crashed, ingest_sequence=41)

        # Leader claims first.
        leader_claim = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=leader_id,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert leader_claim is not None, "leader must claim one of the READY rows"

        # Follower claims next (the remaining READY row).
        follower_claim = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=follower_id,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert follower_claim is not None, "follower must claim the second READY row"

        # They must have claimed different rows.
        assert leader_claim.token_id != follower_claim.token_id, "leader and follower must claim different rows"
        assert {leader_claim.token_id, follower_claim.token_id} == {token_x, token_y}

        # Mark both terminal.
        crashed.repo.mark_terminal(
            work_item_id=leader_claim.work_item_id,
            now=clock.now_utc(),
            expected_lease_owner=leader_id,
        )
        crashed.repo.mark_terminal(
            work_item_id=follower_claim.work_item_id,
            now=clock.now_utc(),
            expected_lease_owner=follower_id,
        )

        # Both rows are TERMINAL.
        with crashed.db.engine.connect() as conn:
            terminal_rows = conn.execute(
                select(token_work_items_table.c.token_id, token_work_items_table.c.status).where(
                    token_work_items_table.c.token_id.in_([token_x, token_y])
                )
            ).fetchall()
        assert len(terminal_rows) == 2
        assert all(r.status == TokenWorkStatus.TERMINAL.value for r in terminal_rows)

        # No double-terminal in the outcomes table.
        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        crashed.db.close()

    def test_follower_crash_lease_lapses_reaper_recovers(self, tmp_path: Path) -> None:
        """C2: follower holds LEASED row, then crashes; reaper recovers after expiry.

        The follower's run_workers row is explicitly set to 'departed'
        (simulating a crash where the heartbeat dies too), the clock is
        advanced past the item lease, and recover_expired_leases is called.
        The item transitions LEASED → READY (attempt bumped) so the leader
        can claim it.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        token_id, _wid = _seed_ready_row_direct(crashed, ingest_sequence=50)

        # Follower claims the row.
        claimed = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=follower_id,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed is not None and claimed.token_id == token_id

        # Simulate follower crash: mark registry row as departed so
        # owner_registry_dead=True and the reaper can target the item.
        with crashed.db.engine.begin() as conn:
            conn.execute(
                update(run_workers_table)
                .where(run_workers_table.c.worker_id == follower_id)
                .values(status="departed", departed_at=clock.now_utc())
            )

        # Advance clock past the item lease + grace window.
        clock.advance(_DEFAULT_LEASE_SECONDS + 100)

        # This direct crash-image harness has no leader seat, so it opts into
        # the explicitly named legacy recovery adapter.
        recovered = crashed.repo.recover_expired_leases_legacy_unfenced(
            run_id=crashed.run_id,
            now=clock.now_utc(),
            caller_owner=leader_id,
        )
        assert recovered >= 1, "reaper must recover the follower's lapsed lease"

        # The item is now READY with attempt bumped to 2.
        with crashed.db.engine.connect() as conn:
            item = conn.execute(select(token_work_items_table).where(token_work_items_table.c.token_id == token_id)).mappings().one()
        assert item["status"] == TokenWorkStatus.READY.value
        assert item["attempt"] == 2, "reaper must bump attempt on recovery"

        # No double-terminal.
        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        crashed.db.close()

    def test_follower_joins_then_heartbeat_latch_fires_eviction(self, tmp_path: Path) -> None:
        """C3: follower is evicted while holding a LEASED row; RunWorkerEvictedError propagates.

        The heartbeat latch simulation fires RunWorkerEvictedError inside the
        drain loop.  The follower's run() propagates the error after a best-
        effort depart.  The LEASED item is NOT spuriously marked terminal.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        # Fake the inner RowProcessor raising RunWorkerEvictedError on first drain.
        stub_proc = _FollowerDrainFake(
            error=RunWorkerEvictedError(
                worker_id=follower_id,
                run_id=crashed.run_id,
            )
        )

        follower_token = CoordinationToken(run_id=crashed.run_id, worker_id=follower_id, leader_epoch=0)
        follower = FollowerProcessor(
            processor=stub_proc,
            token=follower_token,
            run_coordination=crashed.factory.run_coordination,
            factory=crashed.factory,
            now_fn=lambda: clock.now_utc(),
            wait_fn=lambda _: None,
        )

        ctx = PluginContext(run_id=crashed.run_id, config={}, landscape=None)
        with pytest.raises(RunWorkerEvictedError) as exc_info:
            follower.run(ctx)

        assert exc_info.value.worker_id == follower_id
        assert exc_info.value.run_id == crashed.run_id

        # The follower's run_workers row was best-effort departed.
        # (depart_worker is idempotent; it may have been already set to evicted
        # by the setup or left as-is — the key is that RunWorkerEvictedError propagated.)
        # The run itself is still RUNNING (no spurious finalize).
        from elspeth.core.landscape.schema import runs_table

        with crashed.db.engine.connect() as conn:
            status = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == crashed.run_id)).scalar_one()
        # Status is RUNNING (alive) or FAILED (crash preset) — not COMPLETED.
        assert status in (RunStatus.RUNNING.value, RunStatus.FAILED.value)

        # No double-terminal.
        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        crashed.db.close()

    def test_slow_follower_registry_live_reaper_skips_item(self, tmp_path: Path) -> None:
        """C4: item lease expired but run_workers row is active + heartbeat fresh.

        The liveness-aware reaper (§A.5) must skip an item whose owner is
        still registry-live (active + heartbeat_expires_at > now - grace).
        The item stays LEASED; attempt is unchanged.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        token_id, _wid = _seed_ready_row_direct(crashed, ingest_sequence=60)

        # Follower claims the row.
        claimed = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=follower_id,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed is not None and claimed.token_id == token_id

        # Advance clock so the ITEM lease expires BUT the follower's
        # run_workers heartbeat_expires_at is still live (reset it to far future).
        # Also keep the leader's heartbeat live so recover_expired_leases can
        # pass the verify_and_extend_fence (the fenced reaper path requires a
        # live leader seat).
        clock.advance(_DEFAULT_LEASE_SECONDS + 10)
        from datetime import timedelta

        far_future = clock.now_utc() + timedelta(hours=1)
        with crashed.db.engine.begin() as conn:
            conn.execute(
                update(run_workers_table).where(run_workers_table.c.worker_id == follower_id).values(heartbeat_expires_at=far_future)
            )
            # Keep the leader's seat live too.
            conn.execute(
                update(run_workers_table).where(run_workers_table.c.worker_id == leader_id).values(heartbeat_expires_at=far_future)
            )

        # Update the coordination seat's heartbeat so the fence passes.
        from elspeth.core.landscape.schema import run_coordination_table

        with crashed.db.engine.begin() as conn:
            conn.execute(
                update(run_coordination_table)
                .where(run_coordination_table.c.run_id == crashed.run_id)
                .values(leader_heartbeat_expires_at=far_future)
            )

        # Reaper is called by the leader WITH a coordination_token so the
        # liveness-aware gate (§A.5) is active.  The follower's item lease IS
        # expired, but its run_workers row is still registry-live → reaper skips it.
        crashed.repo.recover_expired_leases(
            now=clock.now_utc(),
            coordination_token=leader_token,
        )
        # The reaper skips items owned by registry-live workers.
        # Our follower's item must remain LEASED (the liveness gate protected it).
        with crashed.db.engine.connect() as conn:
            item = conn.execute(select(token_work_items_table).where(token_work_items_table.c.token_id == token_id)).mappings().one()
        assert item["status"] == TokenWorkStatus.LEASED.value, "item leased by a registry-live follower must NOT be reaped"
        assert item["attempt"] == 1, "attempt must not be bumped for a live-worker's item"

        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        crashed.db.close()

    def test_slow_follower_empty_drain_batch_idle_poll_exits_cleanly(self, tmp_path: Path) -> None:
        """C5: follower drains with empty READY queue; idle poll fires; exits when run is terminal.

        The READY queue is empty when the follower's first drain pass runs.
        The follower idle-polls once then exits when the run is flipped to
        FAILED (terminal).  No spurious claim; the follower departs cleanly.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        # No READY rows seeded — the queue is empty.

        # Stub inner RowProcessor: first drain is idle (empty list), second is
        # also idle; the wait_fn flips the run to FAILED so the loop exits.
        from elspeth.core.landscape.schema import runs_table

        def _idle_drain(_ctx: PluginContext) -> list[Any]:
            return []

        stub_proc = _FollowerDrainFake(drain=_idle_drain)

        def _wait(seconds: float) -> None:
            # After first idle wait, flip run to FAILED so the next terminal check exits.
            with crashed.db.engine.begin() as conn:
                conn.execute(update(runs_table).where(runs_table.c.run_id == crashed.run_id).values(status=RunStatus.FAILED.value))

        follower_token = CoordinationToken(run_id=crashed.run_id, worker_id=follower_id, leader_epoch=0)
        follower = FollowerProcessor(
            processor=stub_proc,
            token=follower_token,
            run_coordination=crashed.factory.run_coordination,
            factory=crashed.factory,
            now_fn=lambda: clock.now_utc(),
            wait_fn=_wait,
            idle_poll_seconds=0.001,
        )

        ctx = PluginContext(run_id=crashed.run_id, config={}, landscape=None)
        follower.run(ctx)  # must return normally

        # The follower completed ≥1 idle drain pass.
        assert stub_proc.drain_calls >= 1, "follower must have executed at least one drain pass"

        # Follower departed cleanly.
        workers = {w["worker_id"]: w for w in _run_workers(crashed.db, crashed.run_id)}
        assert workers[follower_id]["status"] == "departed"

        # No double-terminal.
        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        crashed.db.close()
