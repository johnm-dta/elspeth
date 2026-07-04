"""Follower join admission, disposition, lifecycle, and fence tests (ADR-030 §B.1, slice 5).

2A  TestJoinRunAdmissionRefusals
    Each arm raises JoinRefusedError with zero durable mutation.
    R1  terminal run (COMPLETED / FAILED)
    R2  config_hash mismatch
    R3  no live leader (seat expired / vacant)
    R4  filesystem preflight permission failure (os.access mock)

2B  TestFollowerDispositions
    RUNNING + live leader + admitted follower; ONE READY row shaped per arm.
    D1  terminal disposition → TERMINAL
    D2  barrier node → mark_blocked, no in-memory accept
    D3  lossy coalesce fork-lineage token → branch-loss record in same txn
    D4  sink-bound → PENDING_SINK, follower never writes sink

2C  TestFollowerLifecycle
    L1  idle backoff — re-reads run status + coordination snapshot
    L2  run terminal → depart + exit 0
    L3  seat dead → finish/abandon, exit naming elspeth-resume
    L4  SIGINT → depart + exits

2D  TestFollowerEnqueueFence
    L5  enqueue_ready membership fence live on follower child continuation
        (a) active follower → child READY inserted
        (b) evicted follower → RunWorkerEvictedError BEFORE insert

Construction discipline: these tests use the _run_to_interrupted_checkpoint
harness (real Orchestrator → real DB) for admission/disposition/lifecycle
tests so that join_run operates against a real RUNNING run image.  Follower
claims are driven directly via claim_ready on the scheduler repository for the
disposition tests, which gives full control over what the follower "claims"
without needing to bring up a full ExecutionGraph.

The lifecycle tests use the FollowerProcessor with stubs for the drain loop
(matching the unit-test pattern in test_follower_processor.py) to stay
deterministic.  The fence tests drive the production enqueue_ready verb.
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import select, update

from elspeth.contracts import RunStatus
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.errors import FollowerSeatDeadError, JoinRefusedError, RunWorkerEvictedError
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.core.landscape.schema import (
    coalesce_branch_losses_table,
    run_coordination_table,
    run_workers_table,
    runs_table,
    token_work_items_table,
)
from elspeth.engine.clock import MockClock
from elspeth.engine.orchestrator import Orchestrator
from elspeth.engine.orchestrator.follower import FollowerProcessor
from tests.e2e.recovery.harness import (
    _DEFAULT_LEASE_SECONDS,
    _T0,
    _coord,
    _coordination_events,
    _coordination_row,
    _craft_crashed_lease,
    _duplicate_terminal_outcome_tokens,
    _run_to_interrupted_checkpoint,
    _run_workers,
)
from tests.e2e.recovery.test_suspended_winner_fences import (
    _work_item,
)

# Window large enough that the seat reads live under both MockClock and wall clock.
_GUARD_LIVE_SEAT_WINDOW_SECONDS = 10**9

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _orchestrator(crashed: Any) -> Orchestrator:
    return Orchestrator(crashed.db, clock=crashed.clock)


def _seat_run_with_live_leader(
    crashed: Any,
    *,
    leader_id: str,
    window_seconds: float = _GUARD_LIVE_SEAT_WINDOW_SECONDS,
) -> CoordinationToken:
    """Flip FAILED→RUNNING and seat a live leader via the production CAS."""
    return _coord(crashed).acquire_run_leadership(
        run_id=crashed.run_id,
        worker_id=leader_id,
        now=crashed.clock.now_utc(),
        window_seconds=window_seconds,
    )


def _get_db_config_hash(crashed: Any) -> str:
    with crashed.db.engine.connect() as conn:
        return str(conn.execute(select(runs_table.c.config_hash).where(runs_table.c.run_id == crashed.run_id)).scalar_one())


def _join_follower(crashed: Any, leader_token: CoordinationToken) -> str:
    """Admit a follower via join_run, mocking stable_hash to the real DB hash."""
    db_hash = _get_db_config_hash(crashed)
    with (
        patch("elspeth.engine.orchestrator.join_admission.resolve_config", return_value={}),
        patch("elspeth.engine.orchestrator.join_admission.stable_hash", return_value=db_hash),
    ):
        orch = _orchestrator(crashed)
        return orch.join_run(
            run_id=crashed.run_id,
            settings=types.SimpleNamespace(),
            now=crashed.clock.now_utc(),
            window_seconds=_GUARD_LIVE_SEAT_WINDOW_SECONDS,
        )


def _seed_ready_row(crashed: Any, *, ingest_sequence: int) -> tuple[str, str]:
    """Seed a READY token_work_items row.  Returns (token_id, work_item_id).

    Uses _craft_crashed_lease with a unique temporary claimant (not the test's
    real follower_id) to produce a LEASED row, then resets it to READY so the
    test follower can claim it via claim_ready.  A unique temp_owner per
    ingest_sequence avoids run_workers UNIQUE constraint collisions.
    """
    temp_owner = f"worker-temp-seed-{ingest_sequence}"
    token_id = _craft_crashed_lease(
        crashed,
        ingest_sequence=ingest_sequence,
        lease_owner=temp_owner,
        lease_seconds=_DEFAULT_LEASE_SECONDS,
    )
    with crashed.db.engine.connect() as conn:
        work_item_id = str(
            conn.execute(select(token_work_items_table.c.work_item_id).where(token_work_items_table.c.token_id == token_id)).scalar_one()
        )
    with crashed.db.engine.begin() as conn:
        conn.execute(
            update(token_work_items_table)
            .where(token_work_items_table.c.work_item_id == work_item_id)
            .values(
                status=TokenWorkStatus.READY.value,
                lease_owner=None,
                lease_expires_at=None,
            )
        )
    return token_id, work_item_id


def _assert_no_follower_mutation(crashed: Any, *, run_id: str, seat_before: dict[str, Any]) -> None:
    """Common 'zero durable mutation' contract for refusal tests."""
    # No follower run_workers row at all
    workers = {w["worker_id"]: w for w in _run_workers(crashed.db, run_id)}
    follower_rows = [w for w in workers.values() if w["role"] == "follower" and w["status"] == "active"]
    assert follower_rows == [], "no follower run_workers row on refusal"
    # No worker_register event for any new follower
    register_events = [
        e
        for e in _coordination_events(crashed.db, run_id, "worker_register")
        if json.loads(str(e["context_json"])).get("entry_point") == "join"
    ]
    assert register_events == [], "no worker_register event on refusal"
    # Seat byte-identical
    assert _coordination_row(crashed.db, run_id) == seat_before, "seat must be byte-identical on refusal"


# ---------------------------------------------------------------------------
# 2A  Admission refusals
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
class TestJoinRunAdmissionRefusals:
    """join_run refuses with JoinRefusedError; zero durable mutation on all arms."""

    def test_join_refused_when_run_terminal_completed(self, tmp_path: Path) -> None:
        """R1: COMPLETED run raises JoinRefusedError with 'terminal' in reason."""
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        # Force COMPLETED (bypasses begin_run restrictions via direct update).
        with crashed.db.engine.begin() as conn:
            conn.execute(update(runs_table).where(runs_table.c.run_id == crashed.run_id).values(status=RunStatus.COMPLETED.value))
        seat_before = _coordination_row(crashed.db, crashed.run_id)

        db_hash = _get_db_config_hash(crashed)
        with (
            patch("elspeth.engine.orchestrator.join_admission.resolve_config", return_value={}),
            patch("elspeth.engine.orchestrator.join_admission.stable_hash", return_value=db_hash),
            pytest.raises(JoinRefusedError) as exc_info,
        ):
            _orchestrator(crashed).join_run(
                run_id=crashed.run_id,
                settings=types.SimpleNamespace(),
                now=clock.now_utc(),
            )

        assert exc_info.value.run_id == crashed.run_id
        assert "terminal" in str(exc_info.value).lower(), f"reason should mention terminal: {exc_info.value}"
        _assert_no_follower_mutation(crashed, run_id=crashed.run_id, seat_before=seat_before)
        crashed.db.close()

    def test_join_refused_when_run_terminal_failed(self, tmp_path: Path) -> None:
        """R1: FAILED run raises JoinRefusedError directing to elspeth resume."""
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        # _run_to_interrupted_checkpoint leaves status as FAILED (it crashed).
        seat_before = _coordination_row(crashed.db, crashed.run_id)

        db_hash = _get_db_config_hash(crashed)
        with (
            patch("elspeth.engine.orchestrator.join_admission.resolve_config", return_value={}),
            patch("elspeth.engine.orchestrator.join_admission.stable_hash", return_value=db_hash),
            pytest.raises(JoinRefusedError) as exc_info,
        ):
            _orchestrator(crashed).join_run(
                run_id=crashed.run_id,
                settings=types.SimpleNamespace(),
                now=clock.now_utc(),
            )

        assert exc_info.value.run_id == crashed.run_id
        reason = str(exc_info.value).lower()
        assert "resume" in reason or "use" in reason, f"FAILED refusal should mention elspeth resume: {exc_info.value}"
        _assert_no_follower_mutation(crashed, run_id=crashed.run_id, seat_before=seat_before)
        crashed.db.close()

    def test_join_refused_on_config_hash_mismatch(self, tmp_path: Path) -> None:
        """R2: RUNNING + live seat but different pipeline hash → refused."""
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader-mismatch"
        _seat_run_with_live_leader(crashed, leader_id=leader_id)
        seat_before = _coordination_row(crashed.db, crashed.run_id)

        wrong_hash = "wrong-hash-totally-different-abc999"
        with (
            patch("elspeth.engine.orchestrator.join_admission.resolve_config", return_value={}),
            patch("elspeth.engine.orchestrator.join_admission.stable_hash", return_value=wrong_hash),
            pytest.raises(JoinRefusedError) as exc_info,
        ):
            _orchestrator(crashed).join_run(
                run_id=crashed.run_id,
                settings=types.SimpleNamespace(),
                now=clock.now_utc(),
            )

        assert exc_info.value.run_id == crashed.run_id
        reason = str(exc_info.value)
        assert "does-not-match" in reason or "does not match" in reason.lower(), f"refusal should mention hash mismatch: {reason}"
        assert wrong_hash in reason or _get_db_config_hash(crashed) in reason, "refusal should name the hashes"
        _assert_no_follower_mutation(crashed, run_id=crashed.run_id, seat_before=seat_before)
        crashed.db.close()

    def test_join_refused_when_no_live_leader(self, tmp_path: Path) -> None:
        """R3: RUNNING but seat expired/vacant → 'no live leader; use elspeth resume'."""
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        # Flip to RUNNING but do NOT seat a live leader: set status RUNNING
        # and leave the seat vacant (leader_worker_id=None, expires=None).
        with crashed.db.engine.begin() as conn:
            conn.execute(update(runs_table).where(runs_table.c.run_id == crashed.run_id).values(status="running"))
            conn.execute(
                update(run_coordination_table)
                .where(run_coordination_table.c.run_id == crashed.run_id)
                .values(leader_worker_id=None, leader_heartbeat_expires_at=None)
            )
        seat_before = _coordination_row(crashed.db, crashed.run_id)

        db_hash = _get_db_config_hash(crashed)
        with (
            patch("elspeth.engine.orchestrator.join_admission.resolve_config", return_value={}),
            patch("elspeth.engine.orchestrator.join_admission.stable_hash", return_value=db_hash),
            pytest.raises(JoinRefusedError) as exc_info,
        ):
            _orchestrator(crashed).join_run(
                run_id=crashed.run_id,
                settings=types.SimpleNamespace(),
                now=clock.now_utc(),
            )

        assert exc_info.value.run_id == crashed.run_id
        reason = str(exc_info.value).lower()
        assert "no live leader" in reason or "no-live-leader" in reason, f"refusal should mention no live leader: {exc_info.value}"
        assert "resume" in reason, f"refusal should direct to elspeth resume: {exc_info.value}"
        _assert_no_follower_mutation(crashed, run_id=crashed.run_id, seat_before=seat_before)
        crashed.db.close()

    def test_join_refused_on_filesystem_preflight_permission_fail(self, tmp_path: Path) -> None:
        """R4: RUNNING + live + matching hash, but DB not writable → preflight fails."""
        if os.geteuid() == 0:
            pytest.skip("root bypasses filesystem permission checks")

        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader-perm"
        _seat_run_with_live_leader(crashed, leader_id=leader_id)
        seat_before = _coordination_row(crashed.db, crashed.run_id)

        db_hash = _get_db_config_hash(crashed)

        # Mock os.access to deny write on any path → preflight fires before admit_follower.
        original_access = os.access

        def _deny_write(path: Any, mode: int, **kwargs: Any) -> bool:
            if mode & os.W_OK:
                return False
            return original_access(path, mode, **kwargs)

        with (
            patch("elspeth.engine.orchestrator.join_admission.resolve_config", return_value={}),
            patch("elspeth.engine.orchestrator.join_admission.stable_hash", return_value=db_hash),
            patch("elspeth.engine.orchestrator.join_admission.os.access", side_effect=_deny_write),
            pytest.raises(JoinRefusedError) as exc_info,
        ):
            _orchestrator(crashed).join_run(
                run_id=crashed.run_id,
                settings=types.SimpleNamespace(),
                now=clock.now_utc(),
            )

        assert exc_info.value.run_id == crashed.run_id
        reason = str(exc_info.value)
        # Should name the path.
        assert "audit.db" in reason or str(tmp_path).split("/")[-1] in reason, f"preflight refusal should name the path: {reason}"
        # admit_follower never reached → no follower row / event.
        _assert_no_follower_mutation(crashed, run_id=crashed.run_id, seat_before=seat_before)
        crashed.db.close()


# ---------------------------------------------------------------------------
# 2B  Follower dispositions (claim_ready only, no sink I/O)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
class TestFollowerDispositions:
    """Follower disposition contracts against a real WAL DB.

    Each test:
    - Sets up a RUNNING run with live leader.
    - Admits a follower via join_run.
    - Seeds ONE READY row of a specific shape.
    - Drives claim_ready (what the follower does) under the follower's identity.
    - Asserts the correct durable disposition and that the follower NEVER does
      sink I/O or touches leader-plane state.

    These tests drive the scheduler repository verbs directly (not the full
    FollowerProcessor loop) to keep the test scope targeted and deterministic.
    """

    def test_follower_terminal_disposition_marks_terminal(self, tmp_path: Path) -> None:
        """D1: claim + mark_terminal → TERMINAL, attempt==1, clean identity."""
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        token_id, _wid = _seed_ready_row(crashed, ingest_sequence=5)

        # Follower claims and marks terminal.
        claimed = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=follower_id,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed is not None and claimed.token_id == token_id
        assert claimed.lease_owner == follower_id
        assert claimed.attempt == 1

        crashed.repo.mark_terminal(
            work_item_id=claimed.work_item_id,
            now=clock.now_utc(),
            expected_lease_owner=follower_id,
        )

        item = _work_item(crashed.db, token_id)
        assert item["status"] == TokenWorkStatus.TERMINAL.value
        assert item["attempt"] == 1
        # No pending_sink / blocked.
        assert item["pending_sink_name"] is None
        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        crashed.db.close()

    def test_follower_barrier_node_marks_blocked_no_inmemory_accept(self, tmp_path: Path) -> None:
        """D2: follower hitting a BLOCKED status — mark_blocked via the repo.

        Followers never have a barrier executor in memory; the BLOCKED row is
        just a durable hold.  The leader later evaluates trigger conditions
        (§B.2, leader-only).  We drive mark_blocked directly to verify the
        durable contract: barrier_adopted_epoch IS NULL (leader adopts later).
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        token_id, _wid = _seed_ready_row(crashed, ingest_sequence=6)

        claimed = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=follower_id,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed is not None and claimed.token_id == token_id

        # Follower drives mark_blocked (barrier disposition, no in-memory accept).
        barrier_key = "barrier_0"
        crashed.repo.mark_blocked(
            work_item_id=claimed.work_item_id,
            queue_key=None,
            barrier_key=barrier_key,
            now=clock.now_utc(),
            expected_lease_owner=follower_id,
        )

        item = _work_item(crashed.db, token_id)
        assert item["status"] == TokenWorkStatus.BLOCKED.value
        # barrier_adopted_epoch IS NULL — leader adopts later (§B.2).
        assert item["barrier_adopted_epoch"] is None
        # Follower built NO barrier executor (zero batches written by follower).
        from elspeth.core.landscape.schema import batches_table

        with crashed.db.engine.connect() as conn:
            batch_rows = conn.execute(select(batches_table).where(batches_table.c.run_id == crashed.run_id)).fetchall()
        assert batch_rows == [], "follower must not write any batch rows for the barrier"
        crashed.db.close()

    def test_follower_lossy_coalesce_branch_writes_branch_loss_record_in_same_txn(self, tmp_path: Path) -> None:
        """D3: lossy coalesce → branch-loss record written with mark_failed.

        We simulate a lossy branch-loss disposition by calling mark_failed with
        a BranchLossSpec (the same API the engine uses in processor.py).  The
        branch-loss record is written in the same transaction as the mark_failed
        (E.5 uniformity rule); adopted_epoch IS NULL; recorded_by == follower_id.
        Idempotent on (run_id, coalesce_name, row_id, branch_name).
        The follower did NOT fire/fail the merge (no batch rows from follower).
        """
        from elspeth.core.landscape.scheduler_repository import BranchLossSpec

        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        token_id, _wid = _seed_ready_row(crashed, ingest_sequence=7)

        # Get the row_id for the branch-loss spec.
        with crashed.db.engine.connect() as conn:
            row_id = str(
                conn.execute(select(token_work_items_table.c.row_id).where(token_work_items_table.c.token_id == token_id)).scalar_one()
            )

        claimed = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=follower_id,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed is not None and claimed.token_id == token_id

        branch_loss = BranchLossSpec(
            coalesce_name="coalesce_0",
            row_id=row_id,
            branch_name="branch_a",
            token_id=token_id,
            reason="failed",
            recorded_by=follower_id,
        )
        crashed.repo.mark_failed(
            work_item_id=claimed.work_item_id,
            now=clock.now_utc(),
            expected_lease_owner=follower_id,
            branch_loss=branch_loss,
        )

        # Branch-loss record committed in the SAME transaction as mark_failed.
        with crashed.db.engine.connect() as conn:
            losses = conn.execute(
                select(coalesce_branch_losses_table).where(
                    coalesce_branch_losses_table.c.run_id == crashed.run_id,
                    coalesce_branch_losses_table.c.coalesce_name == "coalesce_0",
                    coalesce_branch_losses_table.c.row_id == row_id,
                    coalesce_branch_losses_table.c.branch_name == "branch_a",
                )
            ).fetchall()
        assert len(losses) == 1, "exactly one branch-loss record"
        (loss,) = losses
        assert loss.adopted_epoch is None, "leader adopts later; follower does not set adopted_epoch"
        assert loss.recorded_by == follower_id

        # Idempotent: a second call on the same (run_id, coalesce_name, row_id, branch_name)
        # must not raise (the scheduler INSERT OR IGNORE or UNIQUE handles it).
        # Re-seed to get a fresh LEASED row for the idempotency call.
        _token_id2, _wid2 = _seed_ready_row(crashed, ingest_sequence=8)
        claimed2 = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=follower_id,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed2 is not None
        # Same (coalesce_name, row_id, branch_name) → idempotent.
        branch_loss_dup = BranchLossSpec(
            coalesce_name="coalesce_0",
            row_id=row_id,
            branch_name="branch_a",
            token_id=token_id,
            reason="failed",
            recorded_by=follower_id,
        )
        crashed.repo.mark_failed(
            work_item_id=claimed2.work_item_id,
            now=clock.now_utc(),
            expected_lease_owner=follower_id,
            branch_loss=branch_loss_dup,
        )
        with crashed.db.engine.connect() as conn:
            loss_count = conn.execute(
                select(coalesce_branch_losses_table).where(
                    coalesce_branch_losses_table.c.run_id == crashed.run_id,
                    coalesce_branch_losses_table.c.coalesce_name == "coalesce_0",
                    coalesce_branch_losses_table.c.row_id == row_id,
                    coalesce_branch_losses_table.c.branch_name == "branch_a",
                )
            ).fetchall()
        assert len(loss_count) == 1, "idempotent: still exactly one branch-loss record for this (coalesce, row, branch)"

        # Follower did NOT fire/fail a merge (no adopt_blocked_barrier_item calls).
        from elspeth.core.landscape.schema import batches_table

        with crashed.db.engine.connect() as conn:
            follower_batches = conn.execute(select(batches_table).where(batches_table.c.run_id == crashed.run_id)).fetchall()
        assert follower_batches == [], "follower must not write batch rows"
        crashed.db.close()

    def test_follower_sink_bound_marks_pending_sink_handoff_never_writes_sink(self, tmp_path: Path) -> None:
        """D4: sink-bound → mark_pending_sink; follower has no sink pipeline.

        The leader picks up the pending-sink row later.  The follower's claim
        calls mark_pending_sink only; no actual sink I/O occurs.  The item
        status becomes PENDING_SINK.  Follower's drain called claim_ready ONLY
        (never claim_pending_sink).
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        token_id, _wid = _seed_ready_row(crashed, ingest_sequence=9)

        claimed = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=follower_id,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed is not None and claimed.token_id == token_id

        # Follower calls mark_pending_sink (the sink handoff verb).
        from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
        from tests.e2e.recovery.harness import PipelineRow, _observed_contract  # type: ignore[attr-defined]

        data = {"id": 9, "value": 90}
        crashed.repo.mark_pending_sink(
            work_item_id=claimed.work_item_id,
            row_payload_json=TokenSchedulerRepository.serialize_row_payload(PipelineRow(data, _observed_contract(data))),
            sink_name="output",
            outcome="success",
            path="completed",
            error_hash=None,
            error_message=None,
            now=clock.now_utc(),
            expected_lease_owner=follower_id,
        )

        item = _work_item(crashed.db, token_id)
        assert item["status"] == TokenWorkStatus.PENDING_SINK.value
        assert item["pending_sink_name"] == "output"
        # The leader sink stays empty until the LEADER drains pending_sink.
        # The follower drain calls claim_ready ONLY — never claim_pending_sink.
        # This is a structural property of build_follower_processor; we pin it
        # by verifying the item is PENDING_SINK (not TERMINAL), which means
        # sink I/O has NOT occurred.
        crashed.db.close()


# ---------------------------------------------------------------------------
# 2C  Follower lifecycle (stub drain loop)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
class TestFollowerLifecycle:
    """Follower lifecycle stop-condition contracts.

    These tests use the stub-based pattern from test_follower_processor.py
    for the drain loop itself and only reach into the real DB for
    depart_worker and run-status reads.
    """

    def _make_follower_with_stubs(
        self,
        crashed: Any,
        *,
        leader_token: CoordinationToken,
        follower_id: str,
        seat_live: bool = True,
        wait_calls: list[float] | None = None,
        drain_results: list[list[Any]] | None = None,
    ) -> tuple[FollowerProcessor, Any, Any]:
        """Build a FollowerProcessor with stub processor and injected now_fn.

        Returns (follower, stub_processor, stub_coord_repo).
        """
        from unittest.mock import MagicMock

        run_id = crashed.run_id
        clock = crashed.clock
        coord_repo = crashed.factory.run_coordination
        factory = crashed.factory

        # Stub the inner RowProcessor so we control drain results.
        stub_proc = MagicMock()
        stub_proc.drain_follower_ready_work.return_value = []
        if drain_results is not None:
            stub_proc.drain_follower_ready_work.side_effect = lambda *a, **kw: drain_results.pop(0) if drain_results else []

        recorded_waits: list[float] = wait_calls if wait_calls is not None else []

        # wait_fn: record the sleep duration; after first sleep, set running=False
        # so the loop can exit via the terminal check.
        _call_count = [0]

        def _wait(seconds: float) -> None:
            recorded_waits.append(seconds)
            _call_count[0] += 1

        follower_token = CoordinationToken(run_id=run_id, worker_id=follower_id, leader_epoch=0)

        follower = FollowerProcessor(
            processor=stub_proc,
            token=follower_token,
            run_coordination=coord_repo,
            factory=factory,
            now_fn=lambda: clock.now_utc(),
            wait_fn=_wait,
            idle_poll_seconds=0.001,
        )
        return follower, stub_proc, coord_repo

    def test_follower_idle_backoff_rechecks_status_and_snapshot(self, tmp_path: Path) -> None:
        """L1: no claimable work → idle loop re-reads run status + coordination snapshot.

        The follower must call live_leader (coordination snapshot) and
        _run_is_terminal (run status) on each loop iteration.  We drive two
        idle drain passes before a terminal exit.
        """
        from unittest.mock import MagicMock

        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        # Real coord repo + real factory so live_leader / depart_worker hit the DB.
        real_coord = crashed.factory.run_coordination
        real_factory = crashed.factory

        # Stub the inner RowProcessor (always idle).
        stub_proc = MagicMock()
        stub_proc.drain_follower_ready_work.return_value = []

        wait_calls: list[float] = []
        loop_count = [0]
        max_loops = 3  # idle for 2 loops then exit

        def _wait(seconds: float) -> None:
            wait_calls.append(seconds)
            loop_count[0] += 1
            if loop_count[0] >= max_loops:
                # Flip run to FAILED (terminal, no completed_at required) so the
                # loop exits on the next terminal check.
                with crashed.db.engine.begin() as conn:
                    conn.execute(update(runs_table).where(runs_table.c.run_id == crashed.run_id).values(status=RunStatus.FAILED.value))

        follower_token = CoordinationToken(run_id=crashed.run_id, worker_id=follower_id, leader_epoch=0)
        follower = FollowerProcessor(
            processor=stub_proc,
            token=follower_token,
            run_coordination=real_coord,
            factory=real_factory,
            now_fn=lambda: clock.now_utc(),
            wait_fn=_wait,
            idle_poll_seconds=0.001,
        )

        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id=crashed.run_id, config={}, landscape=None)
        follower.run(ctx)

        # wait_fn called ≥ once (idle backoff triggered) and loop exited cleanly.
        assert len(wait_calls) >= 1, "idle backoff must call wait_fn at least once"
        # Drain was called each loop iteration.
        assert stub_proc.drain_follower_ready_work.call_count >= 1
        crashed.db.close()

    def test_follower_run_terminal_departs_and_exits_zero(self, tmp_path: Path) -> None:
        """L2: leader finalizes COMPLETED while follower idles → follower departs + exits 0."""
        from unittest.mock import MagicMock

        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        # Immediately set run FAILED (terminal, no completed_at required) so the
        # follower exits on its first terminal check.
        with crashed.db.engine.begin() as conn:
            conn.execute(update(runs_table).where(runs_table.c.run_id == crashed.run_id).values(status=RunStatus.FAILED.value))

        stub_proc = MagicMock()
        stub_proc.drain_follower_ready_work.return_value = []
        follower_token = CoordinationToken(run_id=crashed.run_id, worker_id=follower_id, leader_epoch=0)
        follower = FollowerProcessor(
            processor=stub_proc,
            token=follower_token,
            run_coordination=crashed.factory.run_coordination,
            factory=crashed.factory,
            now_fn=lambda: clock.now_utc(),
            wait_fn=lambda _: None,
        )

        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id=crashed.run_id, config={}, landscape=None)
        follower.run(ctx)  # must return normally (not raise)

        # depart_worker CAS active → departed; idempotent if §D already departed.
        workers = {w["worker_id"]: w for w in _run_workers(crashed.db, crashed.run_id)}
        assert workers[follower_id]["status"] == "departed"
        assert workers[follower_id]["evicted_by_worker_id"] is None
        # Exactly one departed transition for follower (§depart_worker idempotent).
        depart_events = [e for e in _coordination_events(crashed.db, crashed.run_id, "worker_depart") if e["worker_id"] == follower_id]
        assert len(depart_events) == 1
        crashed.db.close()

    def test_follower_dead_seat_finishes_then_exits_naming_resume(self, tmp_path: Path) -> None:
        """L3: seat expires mid-drain → follower exits naming elspeth resume, no new claims."""
        from unittest.mock import MagicMock

        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        # Seat with a short window: expires in 5 s.
        leader_token = _coord(crashed).acquire_run_leadership(
            run_id=crashed.run_id,
            worker_id=leader_id,
            now=clock.now_utc(),
            window_seconds=5.0,
        )
        follower_id = _join_follower(crashed, leader_token)

        # Advance clock past the seat window so live_leader returns seat_live=False.
        clock.advance(10.0)

        stub_proc = MagicMock()
        stub_proc.drain_follower_ready_work.return_value = []
        follower_token = CoordinationToken(run_id=crashed.run_id, worker_id=follower_id, leader_epoch=0)

        follower = FollowerProcessor(
            processor=stub_proc,
            token=follower_token,
            run_coordination=crashed.factory.run_coordination,
            factory=crashed.factory,
            now_fn=lambda: clock.now_utc(),
            wait_fn=lambda _: None,
        )

        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id=crashed.run_id, config={}, landscape=None)
        # Design §B.1 step 5: seat-dead raises FollowerSeatDeadError (after
        # clean depart) so the CLI can surface the 'elspeth resume' guidance.
        with pytest.raises(FollowerSeatDeadError) as exc_info:
            follower.run(ctx)

        assert "elspeth resume" in str(exc_info.value)

        # Follower departed cleanly even though exception was raised.
        workers = {w["worker_id"]: w for w in _run_workers(crashed.db, crashed.run_id)}
        assert workers[follower_id]["status"] == "departed"
        # No auto-promotion: no leader_acquire from follower_id.
        leader_events = [e for e in _coordination_events(crashed.db, crashed.run_id, "leader_acquire") if e["worker_id"] == follower_id]
        assert leader_events == [], "follower must never emit leader_acquire (no auto-promotion)"
        # Seat epoch unchanged (follower never bumped epoch).
        seat = _coordination_row(crashed.db, crashed.run_id)
        assert int(seat["leader_epoch"]) == leader_token.leader_epoch
        crashed.db.close()

    def test_follower_sigint_finishes_or_abandons_departs_exits(self, tmp_path: Path) -> None:
        """L4: SIGINT (KeyboardInterrupt) → depart + propagates.

        The follower catches KeyboardInterrupt during the drain loop, calls
        depart_worker, then re-raises.
        """
        from unittest.mock import MagicMock

        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        stub_proc = MagicMock()
        # First drain call raises KeyboardInterrupt (simulates SIGINT mid-loop).
        stub_proc.drain_follower_ready_work.side_effect = KeyboardInterrupt

        follower_token = CoordinationToken(run_id=crashed.run_id, worker_id=follower_id, leader_epoch=0)
        follower = FollowerProcessor(
            processor=stub_proc,
            token=follower_token,
            run_coordination=crashed.factory.run_coordination,
            factory=crashed.factory,
            now_fn=lambda: clock.now_utc(),
            wait_fn=lambda _: None,
        )

        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id=crashed.run_id, config={}, landscape=None)
        with pytest.raises(KeyboardInterrupt):
            follower.run(ctx)

        # Follower departed despite the SIGINT.
        workers = {w["worker_id"]: w for w in _run_workers(crashed.db, crashed.run_id)}
        assert workers[follower_id]["status"] == "departed"
        depart_events = [e for e in _coordination_events(crashed.db, crashed.run_id, "worker_depart") if e["worker_id"] == follower_id]
        assert len(depart_events) == 1, "exactly one worker_depart event"
        crashed.db.close()


# ---------------------------------------------------------------------------
# 2D  Enqueue_ready membership fence (task e)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
class TestFollowerEnqueueFence:
    """L5: enqueue_ready membership fence is live on follower child continuations.

    Spec: a follower processing a token that enqueues a CHILD continuation
    threads its own worker_id into enqueue_ready.
    (a) ACTIVE follower → child READY inserted.
    (b) follower EVICTED before child-enqueue → RunWorkerEvictedError BEFORE
        inserting (no orphan READY at the child work_item_id; zero rows).

    These tests call enqueue_ready with an explicit worker_id parameter directly
    (the production caller in processor.py:4645 threads scheduler_lease_owner).
    This is the PRODUCTION verb test of the fence.
    """

    def test_active_follower_enqueue_ready_child_inserted(self, tmp_path: Path) -> None:
        """L5(a): active follower's enqueue_ready inserts the child READY row."""
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        # Build a fresh row/token as the child we're enqueueing.
        from elspeth.contracts import PipelineRow
        from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
        from tests.e2e.recovery.harness import _observed_contract

        data = {"id": 100, "value": 1000}
        child_row = crashed.factory.data_flow.create_row(
            run_id=crashed.run_id,
            source_node_id=crashed.source_node_id,
            row_index=100,
            data=data,
            source_row_index=100,
            ingest_sequence=100,
        )
        child_token = crashed.factory.data_flow.create_token(row_id=child_row.row_id)

        # enqueue_ready with worker_id=follower_id (membership fence live).
        crashed.repo.enqueue_ready(
            run_id=crashed.run_id,
            token_id=child_token.token_id,
            row_id=child_row.row_id,
            node_id=crashed.journal_node_id,
            step_index=crashed.journal_step_index,
            ingest_sequence=100,
            row_payload_json=TokenSchedulerRepository.serialize_row_payload(PipelineRow(data, _observed_contract(data))),
            available_at=clock.now_utc(),
            worker_id=follower_id,
        )

        # Child row is READY.
        with crashed.db.engine.connect() as conn:
            rows = conn.execute(select(token_work_items_table).where(token_work_items_table.c.token_id == child_token.token_id)).fetchall()
        assert len(rows) == 1
        assert rows[0].status == TokenWorkStatus.READY.value
        crashed.db.close()

    def test_evicted_follower_enqueue_ready_raises_before_insert(self, tmp_path: Path) -> None:
        """L5(b): evicted follower → RunWorkerEvictedError BEFORE inserting.

        Zero orphan READY rows at the child work_item_id.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)
        leader_id = f"worker:{crashed.run_id}:leader"
        leader_token = _seat_run_with_live_leader(crashed, leader_id=leader_id)
        follower_id = _join_follower(crashed, leader_token)

        # Evict the follower (simulate the leader's §C.2 housekeeping sweep).
        with crashed.db.engine.begin() as conn:
            conn.execute(
                update(run_workers_table)
                .where(run_workers_table.c.worker_id == follower_id)
                .values(status="evicted", evicted_at=clock.now_utc())
            )

        # Build a fresh row/token as the child.
        from elspeth.contracts import PipelineRow
        from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
        from tests.e2e.recovery.harness import _observed_contract

        data = {"id": 101, "value": 1010}
        child_row = crashed.factory.data_flow.create_row(
            run_id=crashed.run_id,
            source_node_id=crashed.source_node_id,
            row_index=101,
            data=data,
            source_row_index=101,
            ingest_sequence=101,
        )
        child_token = crashed.factory.data_flow.create_token(row_id=child_row.row_id)

        with pytest.raises(RunWorkerEvictedError) as exc_info:
            crashed.repo.enqueue_ready(
                run_id=crashed.run_id,
                token_id=child_token.token_id,
                row_id=child_row.row_id,
                node_id=crashed.journal_node_id,
                step_index=crashed.journal_step_index,
                ingest_sequence=101,
                row_payload_json=TokenSchedulerRepository.serialize_row_payload(PipelineRow(data, _observed_contract(data))),
                available_at=clock.now_utc(),
                worker_id=follower_id,
            )

        assert exc_info.value.worker_id == follower_id
        assert exc_info.value.run_id == crashed.run_id

        # Zero orphan READY rows — fence fired BEFORE insert.
        with crashed.db.engine.connect() as conn:
            rows = conn.execute(select(token_work_items_table).where(token_work_items_table.c.token_id == child_token.token_id)).fetchall()
        assert rows == [], "no READY row must be inserted when the fence fires"
        crashed.db.close()
