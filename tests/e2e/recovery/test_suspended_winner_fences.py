"""Deterministic suspended-winner fence suite (ADR-030 §H, slice-2 campaign §5).

The surface ticket elspeth-40886ef9f8 said "cannot be driven deterministically
in-process": a leader suspended (SIGSTOP / VM pause / long GC) across a
takeover wakes up holding a STALE coordination token and tries to keep
writing. Design §C.4's closing line makes it an ordinary deterministic test:
"every fence is a DB CAS against an injectable stale token."

Construction per test: build the verb's run state on a REAL crashed run
(:mod:`tests.e2e.recovery.harness` — production writers only), acquire a REAL
``CoordinationToken`` for ``worker-old`` through the production takeover CAS
(epoch E), then ``_usurp_seat`` — the in-DB image of a takeover (bump
``leader_epoch`` directly to E+1 under a usurper identity) — and call the
fenced verb with the stale epoch-E token.

Common contract asserted for EVERY fenced verb:

(i)   the stale call raises :class:`RunLeadershipLostError` (the Tier-2
      sibling of ``SchedulerLeaseLostError``) — and NOT ``AuditIntegrityError``;
(ii)  exactly one new ``run_coordination_events`` row with
      ``event_type='fence_refusal'``, ``worker_id='worker-old'``,
      ``leader_epoch=E`` (the stale epoch presented) and ``context_json``
      naming the verb — written on a FRESH connection despite the payload
      rollback (best-effort by design; deterministic in-process);
(iii) ZERO payload mutation (verb-specific snapshot equality) AND the seat's
      ``leader_heartbeat_expires_at`` NOT extended by the refusal;
(iv)  positive control: the SAME verb under a current-epoch token succeeds —
      proving the refusal was the fence, not broken setup — and the seat
      expiry moves forward (verify-AND-EXTEND, design :246-255).

Slice scope: ``adopt_blocked_barrier_item`` and
``adopt_coalesce_branch_losses`` joined this matrix in slice 3 (design :490);
the membership-fence arms (``claim_ready`` / ``claim_pending_sink`` /
``enqueue_ready`` refusing ``RunWorkerEvictedError``) are slice 4.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, insert, select, update

from elspeth.contracts import PipelineRow, RunStatus
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.errors import (
    AuditIntegrityError,
    OrchestrationInvariantError,
    RunLeadershipLostError,
    RunWorkerEvictedError,
)
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.database import begin_write
from elspeth.core.landscape.scheduler_repository import (
    BatchMembershipSpec,
    BufferedOutcomeSpec,
    TokenSchedulerRepository,
    record_coalesce_branch_loss,
)
from elspeth.core.landscape.schema import (
    batch_members_table,
    batches_table,
    coalesce_branch_losses_table,
    run_coordination_table,
    run_workers_table,
    runs_table,
    token_outcomes_table,
    token_work_items_table,
)
from elspeth.engine.clock import MockClock
from tests.e2e.recovery.harness import (
    _DEFAULT_LEASE_SECONDS,
    _T0,
    _checkpoint_count_and_max_seq,
    _coord,
    _coordination_events,
    _coordination_row,
    _craft_crashed_lease,
    _CrashedRun,
    _observed_contract,
    _recovery_events,
    _rows_and_tokens_at,
    _run_to_interrupted_checkpoint,
    _run_workers,
    _usurp_seat,
)
from tests.helpers.checkpoint import create_checkpoint

WORKER_OLD = "worker-old"
USURPER = "worker-usurper"


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def _takeover_image(tmp_path: Path) -> tuple[_CrashedRun, CoordinationToken]:
    """Crashed run + ``worker-old`` seated as the REAL takeover leader (epoch 2)."""
    clock = MockClock(start=_T0)
    crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
    clock.advance(_DEFAULT_LEASE_SECONDS + 60)
    token_old = _coord(crashed).acquire_run_leadership(
        run_id=crashed.run_id,
        worker_id=WORKER_OLD,
        now=clock.now_utc(),
        window_seconds=80.0,
    )
    return crashed, token_old


def _usurp(crashed: _CrashedRun) -> CoordinationToken:
    """Depose ``worker-old`` (the in-DB takeover image); return a CURRENT token."""
    new_epoch = _usurp_seat(crashed.db, crashed.run_id, usurper=USURPER, now=crashed.clock.now_utc())
    return CoordinationToken(run_id=crashed.run_id, worker_id=USURPER, leader_epoch=new_epoch)


def _seed_journal_row(crashed: _CrashedRun, *, ingest_sequence: int) -> tuple[str, str, str]:
    """READY journal row at ``ingest_sequence`` claimed-LEASED by worker-old.

    Production writers only (same discipline as ``_craft_crashed_lease``).
    Returns (token_id, row_id, work_item_id).
    """
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
    crashed.repo.enqueue_ready(
        run_id=crashed.run_id,
        token_id=token.token_id,
        row_id=row.row_id,
        node_id=crashed.journal_node_id,
        step_index=crashed.journal_step_index,
        ingest_sequence=ingest_sequence,
        row_payload_json=TokenSchedulerRepository.serialize_row_payload(PipelineRow(data, _observed_contract(data))),
        available_at=crashed.clock.now_utc(),
    )
    claimed = crashed.repo.claim_ready(
        run_id=crashed.run_id,
        lease_owner=WORKER_OLD,
        lease_seconds=_DEFAULT_LEASE_SECONDS,
        now=crashed.clock.now_utc(),
    )
    assert claimed is not None and claimed.token_id == token.token_id
    return token.token_id, row.row_id, claimed.work_item_id


def _parked_pending_sink(crashed: _CrashedRun, *, ingest_sequence: int, claim_back: bool) -> str:
    """PENDING_SINK handoff owned by worker-old; optionally claimed-LEASED back."""
    token_id, _row_id, work_item_id = _seed_journal_row(crashed, ingest_sequence=ingest_sequence)
    crashed.repo.mark_pending_sink(
        work_item_id=work_item_id,
        row_payload_json=TokenSchedulerRepository.serialize_row_payload(
            PipelineRow({"id": ingest_sequence}, _observed_contract({"id": ingest_sequence}))
        ),
        sink_name="output",
        outcome="success",
        path="completed",
        error_hash=None,
        error_message=None,
        now=crashed.clock.now_utc(),
        expected_lease_owner=WORKER_OLD,
    )
    if claim_back:
        reclaimed = crashed.repo.claim_pending_sink(
            run_id=crashed.run_id,
            lease_owner=WORKER_OLD,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=crashed.clock.now_utc(),
        )
        assert reclaimed is not None and reclaimed.token_id == token_id
    return token_id


def _seed_active_follower(crashed: _CrashedRun, *, worker_id: str = "worker-follower") -> str:
    """Raw active follower registry row (the §D departure-hygiene witness)."""
    now = crashed.clock.now_utc()
    with crashed.db.engine.begin() as conn:
        conn.execute(
            insert(run_workers_table).values(
                worker_id=worker_id,
                run_id=crashed.run_id,
                role="follower",
                status="active",
                registered_at=now,
                heartbeat_expires_at=now + timedelta(hours=1),
                entry_point="join",
            )
        )
    return worker_id


def _work_item(db: LandscapeDB, token_id: str) -> dict[str, Any]:
    with db.engine.connect() as conn:
        row = conn.execute(select(token_work_items_table).where(token_work_items_table.c.token_id == token_id)).mappings().one()
    return dict(row)


def _table_count(db: LandscapeDB, table: Any, run_id: str) -> int:
    with db.engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(table).where(table.c.run_id == run_id)).scalar_one())


def _run_status_and_completed_at(db: LandscapeDB, run_id: str) -> tuple[str, Any]:
    with db.engine.connect() as conn:
        row = conn.execute(select(runs_table.c.status, runs_table.c.completed_at).where(runs_table.c.run_id == run_id)).one()
    return str(row.status), row.completed_at


def _fence_refusals(crashed: _CrashedRun, verb: str) -> list[dict[str, Any]]:
    return [
        event
        for event in _coordination_events(crashed.db, crashed.run_id, "fence_refusal")
        if json.loads(str(event["context_json"])).get("verb") == verb
    ]


def _assert_refusal_contract(
    crashed: _CrashedRun,
    *,
    verb: str,
    stale_epoch: int,
    seat_before: dict[str, Any],
) -> None:
    """Common arms (ii) + (iii)-seat for every fenced verb."""
    refusals = _fence_refusals(crashed, verb)
    assert len(refusals) == 1, f"exactly one fence_refusal for {verb!r}, got {len(refusals)}"
    (refusal,) = refusals
    assert refusal["worker_id"] == WORKER_OLD
    assert refusal["leader_epoch"] == stale_epoch, "the refusal records the STALE epoch presented"
    assert _coordination_row(crashed.db, crashed.run_id) == seat_before, (
        "the stale refusal must NOT extend the seat (verify-and-extend only fires on a fence HIT)"
    )


def _assert_seat_extended(crashed: _CrashedRun, seat_before: dict[str, Any]) -> None:
    seat_after = _coordination_row(crashed.db, crashed.run_id)
    before = seat_before["leader_heartbeat_expires_at"]
    after = seat_after["leader_heartbeat_expires_at"]
    assert after > before, "the positive control must verify-AND-EXTEND the seat"


@pytest.mark.timeout(120)
class TestSuspendedWinnerFences:
    """One test per slice-2 fenced verb: raise + event + zero mutation + control."""

    def test_stale_complete_run_refused(self, tmp_path: Path) -> None:
        crashed, token_old = _takeover_image(tmp_path)
        follower = _seed_active_follower(crashed)
        current = _usurp(crashed)
        seat_before = _coordination_row(crashed.db, crashed.run_id)

        with pytest.raises(RunLeadershipLostError) as exc_info:
            crashed.factory.run_lifecycle.complete_run(crashed.run_id, RunStatus.COMPLETED, token=token_old)
        assert not isinstance(exc_info.value, AuditIntegrityError)

        status, completed_at = _run_status_and_completed_at(crashed.db, crashed.run_id)
        assert status == RunStatus.RUNNING.value, "a deposed leader cannot finalize out from under the new one"
        assert completed_at is None
        finalize_statuses = [
            json.loads(str(event["context_json"])).get("status") for event in _coordination_events(crashed.db, crashed.run_id, "finalize")
        ]
        assert RunStatus.COMPLETED.value not in finalize_statuses
        follower_row = next(worker for worker in _run_workers(crashed.db, crashed.run_id) if worker["worker_id"] == follower)
        assert follower_row["status"] == "active", "the §D departure-hygiene UPDATE rolled back with the fence"
        _assert_refusal_contract(crashed, verb="complete_run", stale_epoch=token_old.leader_epoch, seat_before=seat_before)

        # Positive control: the current leader finalizes; expiry extends;
        # the seeded follower is departed by the §D hygiene arm.
        crashed.factory.run_lifecycle.complete_run(crashed.run_id, RunStatus.COMPLETED, token=current)
        status, _ = _run_status_and_completed_at(crashed.db, crashed.run_id)
        assert status == RunStatus.COMPLETED.value
        _assert_seat_extended(crashed, seat_before)
        follower_row = next(worker for worker in _run_workers(crashed.db, crashed.run_id) if worker["worker_id"] == follower)
        assert follower_row["status"] == "departed"
        crashed.db.close()

    def test_stale_update_run_status_refused(self, tmp_path: Path) -> None:
        """Design §C.4 row 4 / crash-walk step 6: a deposed leader cannot
        stamp FAILED over the new leader's progress."""
        crashed, token_old = _takeover_image(tmp_path)
        current = _usurp(crashed)
        seat_before = _coordination_row(crashed.db, crashed.run_id)

        with pytest.raises(RunLeadershipLostError) as exc_info:
            crashed.factory.run_lifecycle.update_run_status(crashed.run_id, RunStatus.FAILED, token=token_old)
        assert not isinstance(exc_info.value, AuditIntegrityError)

        status, _ = _run_status_and_completed_at(crashed.db, crashed.run_id)
        assert status == RunStatus.RUNNING.value
        _assert_refusal_contract(crashed, verb="update_run_status", stale_epoch=token_old.leader_epoch, seat_before=seat_before)

        # Positive control.
        crashed.factory.run_lifecycle.update_run_status(crashed.run_id, RunStatus.FAILED, token=current)
        status, _ = _run_status_and_completed_at(crashed.db, crashed.run_id)
        assert status == RunStatus.FAILED.value
        _assert_seat_extended(crashed, seat_before)
        crashed.db.close()

    def test_stale_checkpoint_write_refused(self, tmp_path: Path) -> None:
        """Design §C.4 row 5: refused BEFORE the duplicate-sequence probe and
        BEFORE the UNIQUE constraint — proven by the positive control writing
        the SAME sequence_number successfully afterwards."""
        crashed, token_old = _takeover_image(tmp_path)
        current = _usurp(crashed)
        seat_before = _coordination_row(crashed.db, crashed.run_id)
        count_before, max_seq_before = _checkpoint_count_and_max_seq(crashed.db, crashed.run_id)
        assert max_seq_before is not None, "the real run checkpointed every row"
        contested_seq = max_seq_before + 1

        with pytest.raises(RunLeadershipLostError) as exc_info:
            create_checkpoint(
                crashed.checkpoint_mgr,
                run_id=crashed.run_id,
                sequence_number=contested_seq,
                barrier_scalars=None,
                graph=crashed.graph,
                coordination_token=token_old,
            )
        assert not isinstance(exc_info.value, OrchestrationInvariantError), "the fence fires before the duplicate-sequence probe"

        assert _checkpoint_count_and_max_seq(crashed.db, crashed.run_id) == (count_before, max_seq_before)
        _assert_refusal_contract(crashed, verb="create_checkpoint", stale_epoch=token_old.leader_epoch, seat_before=seat_before)

        # Positive control: the SAME sequence number is genuinely uncontested.
        create_checkpoint(
            crashed.checkpoint_mgr,
            run_id=crashed.run_id,
            sequence_number=contested_seq,
            barrier_scalars=None,
            graph=crashed.graph,
            coordination_token=current,
        )
        assert _checkpoint_count_and_max_seq(crashed.db, crashed.run_id) == (count_before + 1, contested_seq)
        _assert_seat_extended(crashed, seat_before)
        crashed.db.close()

    def test_stale_complete_barrier_refused_before_any_journal_mutation(self, tmp_path: Path) -> None:
        """Design §C.4 row 6. Exactly ONE BLOCKED row exists and the stale
        call names exactly it, so the bidirectional exhaustiveness validation
        can never be what fires — the refusal is the fence."""
        crashed, token_old = _takeover_image(tmp_path)
        token_id, _row_id, work_item_id = _seed_journal_row(crashed, ingest_sequence=3)
        crashed.repo.mark_blocked(
            work_item_id=work_item_id,
            queue_key=None,
            barrier_key="barrier-1",
            now=crashed.clock.now_utc(),
            expected_lease_owner=WORKER_OLD,
        )
        current = _usurp(crashed)
        seat_before = _coordination_row(crashed.db, crashed.run_id)
        item_before = _work_item(crashed.db, token_id)
        assert item_before["status"] == TokenWorkStatus.BLOCKED.value
        batches_before = _table_count(crashed.db, batches_table, crashed.run_id)
        members_before = _table_count(crashed.db, batch_members_table, crashed.run_id)
        outcomes_before = _table_count(crashed.db, token_outcomes_table, crashed.run_id)

        with pytest.raises(RunLeadershipLostError) as exc_info:
            crashed.repo.complete_barrier(
                run_id=crashed.run_id,
                barrier_key="barrier-1",
                consumed_token_ids=(token_id,),
                emitted_pending_sink=(),
                emitted_ready=(),
                now=crashed.clock.now_utc(),
                coordination_token=token_old,
            )
        assert not isinstance(exc_info.value, AuditIntegrityError)

        item_after = _work_item(crashed.db, token_id)
        assert item_after == item_before, "BLOCKED row intact: status, barrier_key, barrier_blocked_at all untouched"
        assert _table_count(crashed.db, batches_table, crashed.run_id) == batches_before
        assert _table_count(crashed.db, batch_members_table, crashed.run_id) == members_before
        assert _table_count(crashed.db, token_outcomes_table, crashed.run_id) == outcomes_before, "no BUFFERED outcome appended"
        _assert_refusal_contract(crashed, verb="complete_barrier", stale_epoch=token_old.leader_epoch, seat_before=seat_before)

        # Positive control: the current leader completes the same barrier.
        crashed.repo.complete_barrier(
            run_id=crashed.run_id,
            barrier_key="barrier-1",
            consumed_token_ids=(token_id,),
            emitted_pending_sink=(),
            emitted_ready=(),
            now=crashed.clock.now_utc(),
            coordination_token=current,
        )
        assert _work_item(crashed.db, token_id)["status"] == TokenWorkStatus.TERMINAL.value
        _assert_seat_extended(crashed, seat_before)
        crashed.db.close()

    def test_stale_adopt_blocked_barrier_item_refused_zero_rows_in_all_three_tables(self, tmp_path: Path) -> None:
        """Slice 3 (design §C.4 row 6a / §E.2): the journal-first adoption verb.

        A deposed leader's adoption must leave ZERO trace in all three
        tables the verb writes — no ``barrier_adopted_epoch`` CAS marker, no
        ``batch_members`` row, no BUFFERED ``token_outcomes`` row. The CAS is
        the ONLY double-BUFFERED guard, so a fence leak here IS the F2
        duplicate-acceptance regression."""
        crashed, token_old = _takeover_image(tmp_path)
        token_id, _row_id, work_item_id = _seed_journal_row(crashed, ingest_sequence=3)
        crashed.repo.mark_blocked(
            work_item_id=work_item_id,
            queue_key=None,
            barrier_key="agg-1",
            now=crashed.clock.now_utc(),
            expected_lease_owner=WORKER_OLD,
        )
        batch = crashed.factory.execution.create_batch(crashed.run_id, crashed.journal_node_id)
        current = _usurp(crashed)
        seat_before = _coordination_row(crashed.db, crashed.run_id)
        item_before = _work_item(crashed.db, token_id)
        assert item_before["status"] == TokenWorkStatus.BLOCKED.value
        assert item_before["barrier_adopted_epoch"] is None
        members_before = _table_count(crashed.db, batch_members_table, crashed.run_id)
        outcomes_before = _table_count(crashed.db, token_outcomes_table, crashed.run_id)

        def _adopt(coordination_token: CoordinationToken) -> Any:
            return crashed.repo.adopt_blocked_barrier_item(
                run_id=crashed.run_id,
                work_item_id=work_item_id,
                token_id=token_id,
                barrier_key="agg-1",
                membership=BatchMembershipSpec(batch_id=batch.batch_id, ordinal=0),
                buffered_outcome=BufferedOutcomeSpec(batch_id=batch.batch_id),
                now=crashed.clock.now_utc(),
                coordination_token=coordination_token,
            )

        with pytest.raises(RunLeadershipLostError) as exc_info:
            _adopt(token_old)
        assert not isinstance(exc_info.value, AuditIntegrityError)

        item_after = _work_item(crashed.db, token_id)
        assert item_after == item_before, "BLOCKED row intact: barrier_adopted_epoch still NULL"
        assert _table_count(crashed.db, batch_members_table, crashed.run_id) == members_before, "zero batch_members rows"
        assert _table_count(crashed.db, token_outcomes_table, crashed.run_id) == outcomes_before, "zero BUFFERED outcomes"
        _assert_refusal_contract(crashed, verb="adopt_blocked_barrier_item", stale_epoch=token_old.leader_epoch, seat_before=seat_before)

        # Positive control: the current leader adopts the same hold.
        result = _adopt(current)
        assert result.adopted is True
        assert result.barrier_adopted_epoch == current.leader_epoch
        adopted_item = _work_item(crashed.db, token_id)
        assert adopted_item["status"] == TokenWorkStatus.BLOCKED.value, "adoption marks, it does not transition"
        assert adopted_item["barrier_adopted_epoch"] == current.leader_epoch
        assert _table_count(crashed.db, batch_members_table, crashed.run_id) == members_before + 1
        assert _table_count(crashed.db, token_outcomes_table, crashed.run_id) == outcomes_before + 1
        _assert_seat_extended(crashed, seat_before)

        # Idempotency: a second current-token call on the already-adopted row
        # is a success-SKIP — no second batch_members row, NO second BUFFERED
        # outcome (the adoption CAS is the ONLY double-BUFFERED guard).
        readopt = _adopt(current)
        assert readopt.adopted is False
        assert readopt.barrier_adopted_epoch == current.leader_epoch
        assert _table_count(crashed.db, batch_members_table, crashed.run_id) == members_before + 1
        assert _table_count(crashed.db, token_outcomes_table, crashed.run_id) == outcomes_before + 1
        with crashed.db.engine.connect() as conn:
            live_buffered = conn.execute(
                select(func.count())
                .select_from(token_outcomes_table)
                .where(token_outcomes_table.c.token_id == token_id)
                # ADR-019 BUFFERED shape: (outcome=NULL, path='buffered'), non-terminal.
                .where(token_outcomes_table.c.path == "buffered")
                .where(token_outcomes_table.c.completed == 0)
            ).scalar_one()
        assert live_buffered == 1, "exactly one live non-terminal BUFFERED acceptance"
        crashed.db.close()

    def test_stale_adopt_coalesce_branch_losses_refused(self, tmp_path: Path) -> None:
        """Slice 3 (design §E.5): a deposed leader cannot move the branch-loss
        replay cursor (``adopted_epoch`` stays NULL for the real leader's
        intake replay)."""
        crashed, token_old = _takeover_image(tmp_path)
        with begin_write(crashed.db.engine) as conn:
            assert record_coalesce_branch_loss(
                conn,
                run_id=crashed.run_id,
                coalesce_name="merge",
                row_id="row-3",
                branch_name="left",
                token_id="token-left",
                reason="failed",
                recorded_by=WORKER_OLD,
                now=crashed.clock.now_utc(),
            )
        (loss,) = crashed.repo.list_unadopted_coalesce_branch_losses(run_id=crashed.run_id)
        current = _usurp(crashed)
        seat_before = _coordination_row(crashed.db, crashed.run_id)

        with pytest.raises(RunLeadershipLostError) as exc_info:
            crashed.repo.adopt_coalesce_branch_losses(
                run_id=crashed.run_id,
                loss_ids=(loss.loss_id,),
                now=crashed.clock.now_utc(),
                coordination_token=token_old,
            )
        assert not isinstance(exc_info.value, AuditIntegrityError)

        with crashed.db.engine.connect() as conn:
            adopted_epoch = conn.execute(
                select(coalesce_branch_losses_table.c.adopted_epoch).where(coalesce_branch_losses_table.c.loss_id == loss.loss_id)
            ).scalar_one()
        assert adopted_epoch is None, "the replay cursor did not move"
        _assert_refusal_contract(crashed, verb="adopt_coalesce_branch_losses", stale_epoch=token_old.leader_epoch, seat_before=seat_before)

        # Positive control: the current leader marks it under its own epoch.
        marked = crashed.repo.adopt_coalesce_branch_losses(
            run_id=crashed.run_id,
            loss_ids=(loss.loss_id,),
            now=crashed.clock.now_utc(),
            coordination_token=current,
        )
        assert marked == 1
        assert crashed.repo.list_unadopted_coalesce_branch_losses(run_id=crashed.run_id) == []
        full = crashed.repo.list_coalesce_branch_losses(run_id=crashed.run_id)
        assert [loss_row.adopted_epoch for loss_row in full] == [current.leader_epoch]
        _assert_seat_extended(crashed, seat_before)
        crashed.db.close()

    def test_stale_ingest_rolls_back_atomically_no_orphan_rows_row(self, tmp_path: Path) -> None:
        """The woken-mid-ingest interleaving (design §C.4 row 9, crash-walk
        step 8): the stale ingest's rows insert rolls back with everything
        else, and the current leader then ingests at the SAME
        ingest_sequence — UNIQUE(run_id, ingest_sequence) genuinely
        uncontested; "B's recovered iteration re-inserts nothing it didn't
        author"."""
        crashed, token_old = _takeover_image(tmp_path)
        current = _usurp(crashed)
        seat_before = _coordination_row(crashed.db, crashed.run_id)
        contested_sequence = 10
        payload = TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 10}, _observed_contract({"id": 10})))

        def _insert_for(row_id: str, token_id: str) -> Any:
            def _do(conn: Any) -> Any:
                return crashed.factory.data_flow.insert_row_with_token_on(
                    conn,
                    run_id=crashed.run_id,
                    source_node_id=crashed.source_node_id,
                    row_index=contested_sequence,
                    data={"id": 10},
                    source_row_index=contested_sequence,
                    ingest_sequence=contested_sequence,
                    row_id=row_id,
                    token_id=token_id,
                )

            return _do

        with pytest.raises(RunLeadershipLostError):
            crashed.repo.ingest_row_with_initial_claim(
                coordination_token=token_old,
                now=crashed.clock.now_utc(),
                insert_row_and_token=_insert_for("row-stale", "token-stale"),
                token_id="token-stale",
                row_id="row-stale",
                node_id=crashed.journal_node_id,
                step_index=crashed.journal_step_index,
                ingest_sequence=contested_sequence,
                row_payload_json=payload,
                lease_owner=WORKER_OLD,
                lease_seconds=_DEFAULT_LEASE_SECONDS,
            )

        assert _rows_and_tokens_at(crashed.db, crashed.run_id, contested_sequence) == ([], []), (
            "no orphan rows/tokens row at the contested sequence"
        )
        with crashed.db.engine.connect() as conn:
            stale_items = conn.execute(
                select(token_work_items_table.c.work_item_id).where(token_work_items_table.c.token_id == "token-stale")
            ).all()
        assert stale_items == []
        _assert_refusal_contract(crashed, verb="ingest_row_with_initial_claim", stale_epoch=token_old.leader_epoch, seat_before=seat_before)

        # Positive control: the current leader ingests at the SAME slot.
        _row, _token, work_item = crashed.repo.ingest_row_with_initial_claim(
            coordination_token=current,
            now=crashed.clock.now_utc(),
            insert_row_and_token=_insert_for("row-current", "token-current"),
            token_id="token-current",
            row_id="row-current",
            node_id=crashed.journal_node_id,
            step_index=crashed.journal_step_index,
            ingest_sequence=contested_sequence,
            row_payload_json=payload,
            lease_owner=USURPER,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )
        assert _rows_and_tokens_at(crashed.db, crashed.run_id, contested_sequence) == (["row-current"], ["token-current"])
        assert work_item.status == TokenWorkStatus.LEASED
        assert work_item.lease_owner == USURPER, "the surviving journal entry belongs to the current leader's call"
        _assert_seat_extended(crashed, seat_before)
        crashed.db.close()

    def test_stale_recover_expired_leases_refused(self, tmp_path: Path) -> None:
        """Design §C.4 row 8: a deposed leader cannot rotate attempts under
        the new one."""
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        crashed_token = _craft_crashed_lease(
            crashed,
            ingest_sequence=3,
            lease_owner="crashed-worker-1",
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)  # the crafted lease is expired
        token_old = _coord(crashed).acquire_run_leadership(
            run_id=crashed.run_id,
            worker_id=WORKER_OLD,
            now=clock.now_utc(),
            window_seconds=80.0,
        )
        current = _usurp(crashed)
        seat_before = _coordination_row(crashed.db, crashed.run_id)
        item_before = _work_item(crashed.db, crashed_token)

        with pytest.raises(RunLeadershipLostError):
            crashed.repo.recover_expired_leases(
                run_id=crashed.run_id,
                now=clock.now_utc(),
                caller_owner=WORKER_OLD,
                coordination_token=token_old,
            )

        item_after = _work_item(crashed.db, crashed_token)
        assert item_after["status"] == TokenWorkStatus.LEASED.value
        assert item_after["attempt"] == item_before["attempt"] == 1
        assert item_after["lease_owner"] == "crashed-worker-1"
        assert _recovery_events(crashed.db, crashed.run_id) == [], "no RECOVER_EXPIRED_LEASE scheduler event"
        _assert_refusal_contract(crashed, verb="recover_expired_leases", stale_epoch=token_old.leader_epoch, seat_before=seat_before)

        # Positive control: the current leader's sweep recovers it.
        assert (
            crashed.repo.recover_expired_leases(
                run_id=crashed.run_id,
                now=clock.now_utc(),
                caller_owner=USURPER,
                coordination_token=current,
            )
            == 1
        )
        item_recovered = _work_item(crashed.db, crashed_token)
        assert item_recovered["status"] == TokenWorkStatus.READY.value
        assert item_recovered["attempt"] == 2
        events = _recovery_events(crashed.db, crashed.run_id)
        assert len(events) == 1 and events[0]["token_id"] == crashed_token
        _assert_seat_extended(crashed, seat_before)
        crashed.db.close()

    def test_stale_mark_pending_sink_terminal_many_refused(self, tmp_path: Path) -> None:
        """§C.4 row 7 ("per terminalization batch" verify-UPDATE): the epoch
        fence governs the batch terminalization, ON TOP of the strict owner
        CAS — a deposed leader with a perfectly MATCHING owner is still
        refused. (The owner-strictness arms — required kwarg, NULL-park
        refusal, mismatch refusal — are unit-pinned in
        tests/unit/core/landscape/test_leader_fence_stale_token.py.)"""
        crashed, token_old = _takeover_image(tmp_path)
        token_id = _parked_pending_sink(crashed, ingest_sequence=3, claim_back=True)
        current = _usurp(crashed)
        seat_before = _coordination_row(crashed.db, crashed.run_id)
        item_before = _work_item(crashed.db, token_id)
        assert item_before["status"] == TokenWorkStatus.LEASED.value
        assert item_before["pending_sink_name"] == "output"
        outcomes_before = _table_count(crashed.db, token_outcomes_table, crashed.run_id)

        with pytest.raises(RunLeadershipLostError) as exc_info:
            crashed.repo.mark_pending_sink_terminal_many(
                run_id=crashed.run_id,
                token_ids=(token_id,),
                now=crashed.clock.now_utc(),
                expected_lease_owner=WORKER_OLD,
                coordination_token=token_old,
            )
        assert not isinstance(exc_info.value, AuditIntegrityError)

        item_after = _work_item(crashed.db, token_id)
        assert item_after == item_before, "still LEASED with its pending_sink_name — zero mutation"
        assert _table_count(crashed.db, token_outcomes_table, crashed.run_id) == outcomes_before, "zero terminal outcomes appended"
        _assert_refusal_contract(
            crashed, verb="mark_pending_sink_terminal_many", stale_epoch=token_old.leader_epoch, seat_before=seat_before
        )

        # Positive control: same batch, same owner, current epoch.
        terminalized = crashed.repo.mark_pending_sink_terminal_many(
            run_id=crashed.run_id,
            token_ids=(token_id,),
            now=crashed.clock.now_utc(),
            expected_lease_owner=WORKER_OLD,
            coordination_token=current,
        )
        assert terminalized == 1
        assert _work_item(crashed.db, token_id)["status"] == TokenWorkStatus.TERMINAL.value
        _assert_seat_extended(crashed, seat_before)
        crashed.db.close()

    def test_stale_repair_sweep_refused(self, tmp_path: Path) -> None:
        """``terminalize_pending_sinks_with_terminal_outcomes`` — design §C.4
        row 8's repair sweep, fenced in slice 2. A durable terminal outcome
        witness is seeded so that WITHOUT the fence the row WOULD be
        repaired — proving the refusal is the fence, not a missing match."""
        crashed, token_old = _takeover_image(tmp_path)
        token_id = _parked_pending_sink(crashed, ingest_sequence=3, claim_back=False)
        with crashed.db.engine.begin() as conn:
            conn.execute(
                insert(token_outcomes_table).values(
                    outcome_id="outcome-repair-1",
                    run_id=crashed.run_id,
                    token_id=token_id,
                    outcome="success",
                    path="completed",
                    completed=1,
                    recorded_at=crashed.clock.now_utc(),
                    sink_name="output",
                )
            )
        current = _usurp(crashed)
        seat_before = _coordination_row(crashed.db, crashed.run_id)

        with pytest.raises(RunLeadershipLostError):
            crashed.repo.terminalize_pending_sinks_with_terminal_outcomes(
                run_id=crashed.run_id,
                now=crashed.clock.now_utc(),
                caller_owner=WORKER_OLD,
                coordination_token=token_old,
            )

        assert _work_item(crashed.db, token_id)["status"] == TokenWorkStatus.PENDING_SINK.value, "the repairable row is untouched"
        _assert_refusal_contract(
            crashed,
            verb="terminalize_pending_sinks_with_terminal_outcomes",
            stale_epoch=token_old.leader_epoch,
            seat_before=seat_before,
        )

        # Positive control: the current leader's sweep repairs it.
        repaired = crashed.repo.terminalize_pending_sinks_with_terminal_outcomes(
            run_id=crashed.run_id,
            now=crashed.clock.now_utc(),
            caller_owner=USURPER,
            coordination_token=current,
        )
        assert repaired == 1
        assert _work_item(crashed.db, token_id)["status"] == TokenWorkStatus.TERMINAL.value
        _assert_seat_extended(crashed, seat_before)
        crashed.db.close()

    def test_takeover_identity_evicts_deposed_leader_even_with_fresh_worker_heartbeat(self, tmp_path: Path) -> None:
        """§H :470 pin, real-CAS variant (not ``_usurp_seat``): eviction is by
        IDENTITY, never liveness (design §B.4 correction 1) — the deposed
        leader's registry row is evicted even though its worker-row heartbeat
        is FRESH; the seeded active follower is NOT evicted (correction 2:
        no bulk follower eviction in the takeover transaction)."""
        crashed, token_old = _takeover_image(tmp_path)
        follower = _seed_active_follower(crashed)
        clock = crashed.clock

        # The fresh-heartbeat image: worker-old's REGISTRY clock far in the
        # future; only the SEAT clock expires.
        with crashed.db.engine.begin() as conn:
            conn.execute(
                update(run_workers_table)
                .where(run_workers_table.c.worker_id == WORKER_OLD)
                .values(heartbeat_expires_at=clock.now_utc() + timedelta(days=365))
            )
            conn.execute(
                update(run_coordination_table)
                .where(run_coordination_table.c.run_id == crashed.run_id)
                .values(leader_heartbeat_expires_at=clock.now_utc() - timedelta(seconds=1))
            )

        token_new = _coord(crashed).acquire_run_leadership(
            run_id=crashed.run_id,
            worker_id="worker-new",
            now=clock.now_utc(),
            window_seconds=80.0,
        )
        assert token_new.leader_epoch == token_old.leader_epoch + 1, "epoch monotonic bump E -> E+1"

        workers = {worker["worker_id"]: worker for worker in _run_workers(crashed.db, crashed.run_id)}
        deposed = workers[WORKER_OLD]
        assert deposed["status"] == "evicted", "evicted by IDENTITY despite the fresh worker-row heartbeat"
        assert deposed["evicted_at"] is not None
        assert deposed["evicted_by_worker_id"] == "worker-new"
        assert workers[follower]["status"] == "active", "no follower bulk-eviction at takeover"

        # worker_evict + (worker_register +) leader_acquire committed in ONE
        # transaction: strictly consecutive seq values in that order.
        events = _coordination_events(crashed.db, crashed.run_id)
        evict_events = [event for event in events if event["event_type"] == "worker_evict"]
        assert len(evict_events) == 1
        (evict_event,) = evict_events
        assert evict_event["worker_id"] == WORKER_OLD
        assert json.loads(str(evict_event["context_json"])) == {
            "evicted_by_worker_id": "worker-new",
            "reason": "deposed_leader_takeover",
        }
        evict_seq = int(evict_event["seq"])
        by_seq = {int(event["seq"]): event for event in events}
        assert by_seq[evict_seq + 1]["event_type"] == "worker_register"
        assert by_seq[evict_seq + 1]["worker_id"] == "worker-new"
        assert by_seq[evict_seq + 2]["event_type"] == "leader_acquire"
        assert by_seq[evict_seq + 2]["worker_id"] == "worker-new"
        assert by_seq[evict_seq + 2]["leader_epoch"] == token_new.leader_epoch
        crashed.db.close()

    # ── Slice-4 membership fence tests (design §G verb table, :491) ──────────
    #
    # The fence tests inject EVICTED/ABSENT workers directly via raw inserts
    # to give explicit control over registry status.  To avoid _seed_journal_row
    # and _craft_crashed_lease calling claim_ready with an unregistered owner,
    # a fictional crashed worker is always registered as ACTIVE before the
    # harness claim and then marked EVICTED before the fence assertion.

    def _seed_ready_item_for_fence(self, crashed: _CrashedRun, *, ingest_sequence: int) -> tuple[str, str]:
        """Seed a READY token_work_items row using _craft_crashed_lease with a
        temporary active worker, then return (token_id, LEASED-to-READY) so
        fence tests can claim with a different (evicted) identity.

        Returns (token_id, registerd_lease_owner) — the owner is left ACTIVE
        so callers can evict it explicitly for the negative test arm.
        """
        temp_owner = f"worker-temp-active-{ingest_sequence}"
        token_id = _craft_crashed_lease(
            crashed,
            ingest_sequence=ingest_sequence,
            lease_owner=temp_owner,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )
        # _craft_crashed_lease leaves the row LEASED.  Return it to READY.
        with crashed.db.engine.connect() as conn:
            work_item_id = conn.execute(
                select(token_work_items_table.c.work_item_id).where(token_work_items_table.c.token_id == token_id)
            ).scalar_one()
        with crashed.db.engine.begin() as conn:
            conn.execute(
                update(token_work_items_table)
                .where(token_work_items_table.c.work_item_id == work_item_id)
                .values(status=TokenWorkStatus.READY.value, lease_owner=None, lease_expires_at=None)
            )
        return token_id, temp_owner

    def test_evicted_worker_claim_ready_raises_membership_fence(self, tmp_path: Path) -> None:
        """Slice-4 (design §G): an EVICTED worker calling claim_ready is refused
        with RunWorkerEvictedError and leaves the READY row untouched.

        Contract: (i) RunWorkerEvictedError not AuditIntegrityError; (ii) zero
        payload mutation — the row stays READY; (iii) positive control: an
        ACTIVE worker claims it successfully.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        token_id, evicted_id = self._seed_ready_item_for_fence(crashed, ingest_sequence=5)
        # Mark the temp owner as EVICTED so the fence fires.
        with crashed.db.engine.begin() as conn:
            conn.execute(
                update(run_workers_table)
                .where(run_workers_table.c.worker_id == evicted_id)
                .values(status="evicted", evicted_at=clock.now_utc())
            )

        item_before = _work_item(crashed.db, token_id)
        assert item_before["status"] == TokenWorkStatus.READY.value

        # EVICTED worker refused.
        with pytest.raises(RunWorkerEvictedError) as exc_info:
            crashed.repo.claim_ready(
                run_id=crashed.run_id,
                lease_owner=evicted_id,
                lease_seconds=_DEFAULT_LEASE_SECONDS,
                now=clock.now_utc(),
            )
        assert not isinstance(exc_info.value, AuditIntegrityError)
        assert exc_info.value.worker_id == evicted_id
        assert exc_info.value.run_id == crashed.run_id

        # Zero mutation: row is still READY.
        item_after = _work_item(crashed.db, token_id)
        assert item_after == item_before, "claim_ready with evicted worker left the READY row mutated"

        # Positive control: an ACTIVE worker claims it successfully.
        _seed_active_follower(crashed, worker_id=USURPER)
        claimed = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=USURPER,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed is not None and claimed.token_id == token_id
        assert claimed.lease_owner == USURPER
        crashed.db.close()

    def test_evicted_worker_claim_pending_sink_raises_membership_fence(self, tmp_path: Path) -> None:
        """Slice-4 (design §G): an EVICTED worker calling claim_pending_sink is
        refused with RunWorkerEvictedError and leaves the PENDING_SINK row
        untouched.

        Contract: same (i)-(iii) as claim_ready above.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        # Use _takeover_image (which registers WORKER_OLD as ACTIVE) then run
        # the production acquire_run_leadership with a second worker to evict it.
        crashed2, _token_old = _takeover_image(tmp_path)
        # Seed the PENDING_SINK row while WORKER_OLD is still ACTIVE.
        token_id = _parked_pending_sink(crashed2, ingest_sequence=6, claim_back=False)
        # Now run a real takeover CAS: evicts WORKER_OLD in run_workers.
        clock2 = crashed2.clock
        clock2.advance(_DEFAULT_LEASE_SECONDS + 10)  # seat is expired for the takeover
        _coord(crashed2).acquire_run_leadership(
            run_id=crashed2.run_id,
            worker_id=USURPER,
            now=clock2.now_utc(),
            window_seconds=80.0,
        )
        workers = {w["worker_id"]: w for w in _run_workers(crashed2.db, crashed2.run_id)}
        assert workers[WORKER_OLD]["status"] == "evicted", "precondition: production takeover evicted worker-old"
        assert workers[USURPER]["status"] == "active"

        item_before = _work_item(crashed2.db, token_id)
        assert item_before["status"] == TokenWorkStatus.PENDING_SINK.value

        # EVICTED worker refused.
        with pytest.raises(RunWorkerEvictedError) as exc_info:
            crashed2.repo.claim_pending_sink(
                run_id=crashed2.run_id,
                lease_owner=WORKER_OLD,
                lease_seconds=_DEFAULT_LEASE_SECONDS,
                now=clock2.now_utc(),
            )
        assert not isinstance(exc_info.value, AuditIntegrityError)
        assert exc_info.value.worker_id == WORKER_OLD
        assert exc_info.value.run_id == crashed2.run_id

        # Zero mutation.
        item_after = _work_item(crashed2.db, token_id)
        assert item_after == item_before, "claim_pending_sink with evicted worker mutated the row"

        # Positive control: USURPER (now the active leader) claims it.
        claimed = crashed2.repo.claim_pending_sink(
            run_id=crashed2.run_id,
            lease_owner=USURPER,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock2.now_utc(),
        )
        assert claimed is not None and claimed.token_id == token_id
        crashed.db.close()
        crashed2.db.close()

    def test_evicted_worker_enqueue_ready_raises_membership_fence(self, tmp_path: Path) -> None:
        """Slice-4 (design §G): an EVICTED worker calling enqueue_ready with its
        worker_id raises RunWorkerEvictedError BEFORE inserting any row.

        The fence fires on the pre-INSERT membership probe; the dedup path
        (existing work_item_id) is not reached.  An ABSENT worker (no registry
        row) is also refused.  An ACTIVE worker succeeds.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)

        # Seed an evicted worker row explicitly.
        evicted_id = "worker-evicted-enq-slice4"
        with crashed.db.engine.begin() as conn:
            conn.execute(
                insert(run_workers_table).values(
                    worker_id=evicted_id,
                    run_id=crashed.run_id,
                    role="leader",
                    status="evicted",
                    registered_at=clock.now_utc(),
                    heartbeat_expires_at=clock.now_utc(),
                    entry_point="resume",
                    evicted_at=clock.now_utc(),
                )
            )

        def _try_enqueue(worker_id: str, seq: int) -> None:
            data = {"id": seq}
            row = crashed.factory.data_flow.create_row(
                run_id=crashed.run_id,
                source_node_id=crashed.source_node_id,
                row_index=seq,
                data=data,
                source_row_index=seq,
                ingest_sequence=seq,
            )
            token = crashed.factory.data_flow.create_token(row_id=row.row_id)
            crashed.repo.enqueue_ready(
                run_id=crashed.run_id,
                token_id=token.token_id,
                row_id=row.row_id,
                node_id=crashed.journal_node_id,
                step_index=crashed.journal_step_index,
                ingest_sequence=seq,
                row_payload_json=TokenSchedulerRepository.serialize_row_payload(PipelineRow(data, _observed_contract(data))),
                available_at=clock.now_utc(),
                worker_id=worker_id,
            )

        # Evicted worker refused.
        with pytest.raises(RunWorkerEvictedError) as exc_info:
            _try_enqueue(evicted_id, seq=20)
        assert not isinstance(exc_info.value, AuditIntegrityError)
        assert exc_info.value.worker_id == evicted_id

        # Absent worker (no registry row) also refused.
        with pytest.raises(RunWorkerEvictedError) as exc_info2:
            _try_enqueue("worker-phantom", seq=21)
        assert exc_info2.value.worker_id == "worker-phantom"

        # Positive control: USURPER (active) enqueues successfully.
        _seed_active_follower(crashed, worker_id=USURPER)
        _try_enqueue(USURPER, seq=22)  # must not raise
        crashed.db.close()
