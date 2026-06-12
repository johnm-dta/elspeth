"""E2E crash + concurrent-resume proofs over the durable scheduler journal.

Ticket elspeth-40886ef9f8: the old ``test_concurrent_resume.py`` only tested
resume *rejection* (now ``test_resume_rejection.py``). This file is the real
coverage for the three gaps that ticket names:

1. **Mid-claim crash, then resume** — a worker dies holding a LEASED
   ``token_work_items`` row; a later resume's recovery sweep reclaims it
   after lease expiry (attempt bump + work_item_id rotation), the run
   completes, and the audit DB shows exactly-once terminal token outcomes.
2. **Expired-lease reclaim under contention** — a population of crashed
   workers' leases with different expiries: the drain refuses atomically
   while ANY peer lease is still live (ADR-026 Precondition #9), then
   reclaims everything once all leases age out, with per-item attempt
   offsets visible in both the journal ``attempt`` column and the
   ``node_states`` attempt identity.
3. **Two resume() calls racing on the same run_id** — DESIGNED CONTRACTS at
   epoch 21 (ADR-030, the §H test-#1 flip; the characterization died here):
   *loser-during* is refused by the seat-acquisition CAS
   (``acquire_run_leadership`` rowcount 0 ⇒
   ``NonResumableRunError("run leadership is held by …")``) BEFORE any
   durable write — a designed coordination outcome, not an
   immutability-guard side effect; *loser-after* (winner COMPLETED) is
   refused at the resume() entry guard with "Run is terminal", the durable
   immutability guards retained beneath as the backstop (independently
   pinned in tests/unit/core/landscape/). The mid-flight interleaving is the
   entry guard's RUNNING arm (elspeth-2f23292372, option b), which at slice 2
   gained §B.3 live-seat precision: a RUNNING run under a LIVE seat is
   refused naming the incumbent and directing to ``elspeth join``.

Durability-unification (F1) survival contract: every ASSERTION below reads
PUBLIC, durable surfaces only — the ``token_work_items`` journal columns
(status/attempt/lease_owner), ``scheduler_events``, ``token_outcomes``,
``node_states``, ``runs``, ``run_coordination``/``run_workers``/
``run_coordination_events``, terminal RunStatus, and the public
``Orchestrator.resume()`` API. Nothing asserts on checkpoint internals
(the blob layer is deleted — Task 4.1), and resume points are treated as
opaque handles (never dereferenced).

The construction technique and the checkpoint-coupled harness seams live in
``tests/e2e/recovery/harness.py`` (extracted verbatim from this file so the
slice-2 fence suites reuse them); see its module docstring for the honest
construction notes and the epoch-21 seat assumption.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select, update

from elspeth.contracts import RunStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.core.checkpoint.recovery import NonResumableRunError
from elspeth.core.landscape.schema import run_coordination_table, runs_table
from elspeth.engine.clock import MockClock
from tests.e2e.recovery.harness import (
    _DEFAULT_LEASE_SECONDS,
    _SOURCE_ROWS,
    _T0,
    _build_pipeline,
    _completed_outcome_tokens,
    _coord,
    _coordination_events,
    _coordination_row,
    _craft_crashed_lease,
    _duplicate_terminal_outcome_tokens,
    _node_state_identities,
    _recovery_events,
    _recovery_manager,
    _resume,
    _resume_point,
    _run_to_interrupted_checkpoint,
    _run_workers,
    _work_items_by_token,
)

# The resume() entry guard and can_resume() evaluate seat liveness with the
# WALL clock (datetime.now(UTC)), while the harness MockClock lives at the
# fixed _T0 epoch (June 2025). A production-sized 80 s window minted in the
# MockClock domain therefore reads as long-expired to the guard. The
# live-seat tests mint the winner's seat with this deliberately huge window
# so "live" holds under both clock domains; the expired-seat companion arm
# stamps an explicit past expiry instead.
_GUARD_LIVE_SEAT_WINDOW_SECONDS = 10**9


@pytest.mark.timeout(120)
class TestMidClaimCrashResume:
    """Ticket item 1: mid-claim crash -> lease expiry -> sweep -> completion."""

    def test_mid_claim_crash_resume_recovers_leased_item_exactly_once(self, tmp_path: Path) -> None:
        """A LEASED journal row left by a dead worker is recovered exactly once.

        Invariants pinned (all durable surfaces):
        - the public resume claims the crashed item only AFTER lease expiry,
          via the recovery sweep: the journal row rotates to attempt=2 and
          reaches 'terminal'; exactly one RECOVER_EXPIRED_LEASE event exists
          (from_attempt=1, to_attempt=2, from_lease_owner=the dead worker,
          caller_owner=a different identity — the G1 self-steal guard);
        - the run completes COMPLETED with cumulative rows_processed=4 and
          the recovered row reaches the sink exactly once;
        - exactly-once token outcomes: every token has exactly one completed
          terminal outcome — including the crashed token, under its ORIGINAL
          token_id (the journal re-drive does not mint a replacement token);
        - the re-drive records the transform node_state at attempt=1 (the
          journal attempt offset, claimed.attempt-1), with no attempt-0 row
          for the crashed token (the dead worker never got that far) and no
          duplicate (token_id, node_id, attempt) identity anywhere;
        - no source replay: the resume-side source plugin is never load()ed.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        crashed_token = _craft_crashed_lease(
            crashed,
            ingest_sequence=3,
            lease_owner="crashed-worker-1",
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )

        # The dead worker's lease must age out before recovery is possible.
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)

        result, resume_sink, resume_source = _resume(crashed)

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 4
        assert resume_sink.results == [{"id": 3, "value": 30}]
        assert resume_source.load_invocations == 0, "scheduler-drain resume must not replay the source"

        items = _work_items_by_token(crashed.db, crashed.run_id)
        assert len(items) == 4
        assert all(item["status"] == TokenWorkStatus.TERMINAL.value for item in items.values())
        assert items[crashed_token]["attempt"] == 2
        assert all(item["attempt"] == 1 for token_id, item in items.items() if token_id != crashed_token)

        events = _recovery_events(crashed.db, crashed.run_id)
        assert len(events) == 1
        (event,) = events
        assert event["token_id"] == crashed_token
        assert event["from_attempt"] == 1
        assert event["to_attempt"] == 2
        assert event["from_lease_owner"] == "crashed-worker-1"
        assert event["caller_owner"] != "crashed-worker-1"

        # Exactly-once terminal outcomes, for the original token identity.
        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        completed_tokens = _completed_outcome_tokens(crashed.db, crashed.run_id)
        assert crashed_token in completed_tokens
        assert len(completed_tokens) == 4

        # node_states attempt identity: re-drive at the journal attempt offset.
        identities = _node_state_identities(crashed.db, crashed.run_id)
        assert len(identities) == len(set(identities)), "duplicate node_states identity"
        crashed_attempts = {(node_id, attempt) for token_id, node_id, attempt in identities if token_id == crashed_token}
        transform_attempts = {attempt for node_id, attempt in crashed_attempts if node_id.startswith("transform_")}
        assert transform_attempts == {1}, f"expected re-drive at attempt offset 1, got {sorted(transform_attempts)}"

        # Full completion is already durably proven above (terminal
        # RunStatus, all-TERMINAL journal, exactly-once outcomes); the
        # checkpoint store's delete-on-completion lifecycle is deliberately
        # NOT asserted — that layer dissolves with F1.
        crashed.db.close()


@pytest.mark.timeout(120)
class TestExpiredLeaseReclaimUnderContention:
    """Ticket item 2: a population of crashed leases with attempt offsets."""

    def test_contended_reclaim_is_atomic_and_bumps_attempts_with_offset_identity(self, tmp_path: Path) -> None:
        """Two crashed workers' leases, different expiries and attempt history.

        Construction (every journal transition written by the production
        repository): worker-a dies holding token-3 at attempt=1 (300s lease).
        token-4 was already recovered once during the prior process's own
        maintenance sweep (a REAL recover_expired_leases call) and re-claimed
        by worker-b at attempt=2 with a long 7200s lease before the kill.

        Invariants pinned:
        - PARTIAL expiry refuses ATOMICALLY: a resume at a time when token-3's
          lease has expired but worker-b's is still live raises
          AuditIntegrityError naming the live peer (ADR-026 Precondition #9),
          and the refusal touches NEITHER journal row — worker-a's expired
          lease is NOT half-recovered (still LEASED/attempt=1/owner intact);
        - after all leases expire, one resume recovers BOTH: attempts bump
          1->2 (token-3) and 2->3 (token-4), recovery events carry the dead
          owners' identities, and recovery order follows ingest_sequence;
        - attempt offsets land in the audit DB: the re-driven transform
          node_state is recorded at attempt = journal_attempt - 1 (1 for
          token-3, 2 for token-4) with no duplicate node_states identities;
        - exactly-once terminal outcomes across all 5 tokens, run COMPLETED.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)

        token_3 = _craft_crashed_lease(
            crashed,
            ingest_sequence=3,
            lease_owner="crashed-worker-a",
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )
        token_4 = _craft_crashed_lease(
            crashed,
            ingest_sequence=4,
            lease_owner="crashed-worker-b",
            lease_seconds=60,
        )

        # Production attempt history for token-4: its first lease expires and
        # the prior process's own maintenance sweep (a different owner — the
        # G1 self-steal guard requires it) recovers it, then worker-b
        # re-claims at attempt=2 with a long lease... and dies holding it.
        clock.advance(120)  # token-4's 60s lease is expired; token-3's 300s is not
        assert (
            crashed.repo.recover_expired_leases(
                run_id=crashed.run_id,
                now=clock.now_utc(),
                caller_owner="row-processor:prior-attempt-sweeper",
            )
            == 1
        )
        reclaimed = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner="crashed-worker-b",
            lease_seconds=7200,
            now=clock.now_utc(),
        )
        assert reclaimed is not None and reclaimed.token_id == token_4 and reclaimed.attempt == 2

        # ---- Resume #1: token-3 expired, worker-b's 7200s lease still live ----
        clock.advance(3600)
        resume_point = _resume_point(crashed)
        assert resume_point is not None
        config_1, graph_1, sink_1, _source_1 = _build_pipeline(_SOURCE_ROWS)
        with pytest.raises(AuditIntegrityError, match=r"peer worker\(s\).*crashed-worker-b"):
            crashed.resume_orchestrator().resume(resume_point, config_1, graph_1, payload_store=crashed.payload_store)
        assert sink_1.results == []

        # Atomicity: the refused drain recovered NOTHING — not even the
        # already-expired worker-a lease.
        items = _work_items_by_token(crashed.db, crashed.run_id)
        assert items[token_3]["status"] == TokenWorkStatus.LEASED.value
        assert items[token_3]["attempt"] == 1
        assert items[token_3]["lease_owner"] == "crashed-worker-a"
        assert items[token_4]["status"] == TokenWorkStatus.LEASED.value
        assert items[token_4]["attempt"] == 2
        assert items[token_4]["lease_owner"] == "crashed-worker-b"

        # The failure ceremony re-finalized FAILED, so the run stays resumable.
        with crashed.db.engine.connect() as conn:
            status = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == crashed.run_id)).scalar_one()
        assert status == RunStatus.FAILED.value

        # ---- Resume #2: every crashed lease has aged out ----
        clock.advance(7200)
        result, resume_sink, _resume_source = _resume(crashed)

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 5
        assert resume_sink.results == [{"id": 3, "value": 30}, {"id": 4, "value": 40}]

        items = _work_items_by_token(crashed.db, crashed.run_id)
        assert all(item["status"] == TokenWorkStatus.TERMINAL.value for item in items.values())
        assert items[token_3]["attempt"] == 2
        assert items[token_4]["attempt"] == 3

        # Recovery journal: the crafted prior-process sweep (token-4 1->2),
        # then the final resume's sweep in ingest_sequence order.
        events = _recovery_events(crashed.db, crashed.run_id)
        assert [(event["token_id"], event["from_attempt"], event["to_attempt"], event["from_lease_owner"]) for event in events] == [
            (token_4, 1, 2, "crashed-worker-b"),
            (token_3, 1, 2, "crashed-worker-a"),
            (token_4, 2, 3, "crashed-worker-b"),
        ]
        final_sweep_callers = {event["caller_owner"] for event in events[1:]}
        assert len(final_sweep_callers) == 1
        assert final_sweep_callers.isdisjoint({"crashed-worker-a", "crashed-worker-b"})

        # Attempt offsets in the audit DB: node_states attempt identity is
        # journal_attempt - 1 for the re-driven transform segment.
        identities = _node_state_identities(crashed.db, crashed.run_id)
        assert len(identities) == len(set(identities)), "duplicate node_states identity"

        def _transform_attempts(token_id: str) -> set[int]:
            return {attempt for tid, node_id, attempt in identities if tid == token_id and node_id.startswith("transform_")}

        assert _transform_attempts(token_3) == {1}
        assert _transform_attempts(token_4) == {2}

        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        assert len(_completed_outcome_tokens(crashed.db, crashed.run_id)) == 5
        crashed.db.close()


@pytest.mark.timeout(120)
class TestTwoResumesSameRunId:
    """Ticket item 3 — the two deterministic interleavings of a resume race.

    Epoch 21 (ADR-030, the §H test-#1 flip — the characterization is DEAD,
    both interleavings are ENFORCED DESIGNED CONTRACTS):

    - loser-DURING: the seat-acquisition CAS ``acquire_run_leadership`` is
      resume()'s first durable act and the race arbiter — the second
      acquisition's rowcount-0 raises
      ``NonResumableRunError("run leadership is held by …")`` BEFORE any
      durable write (§B.4; this is what CLOSED the documented TOCTOU);
    - loser-AFTER-winner: refused at the resume() ENTRY GUARD with
      "Run is terminal" — the immutable-success durable backstops
      (``acquire_run_leadership``'s in-CAS arm and the run_lifecycle
      conditional UPDATEs) are retained BENEATH and independently pinned in
      tests/unit/core/landscape/;
    - loser-while-RUNNING: the entry guard (elspeth-2f23292372, option b),
      which at slice 2 gained the §B.3 live-seat precision arm.
    """

    def test_two_resumes_loser_during_refused_at_seat_cas(self, tmp_path: Path) -> None:
        """The loser-during interleaving: refused AT the seat CAS, zero mutation.

        HONESTY NOTE (why this drives the repository verb, not public
        resume()): after the winner's atomic acquire the run is RUNNING, so
        a loser entering through public ``resume()`` is refused at the entry
        guard (that arm is
        ``test_entry_guard_refuses_resume_while_run_status_running``). The
        seat CAS and the FAILED→RUNNING status flip being ONE transaction is
        exactly WHY the TOCTOU is closed; this test pins the closure — the
        CAS-loss arm itself — not a re-creation of the window. The winner's
        first durable resume act (``acquire_run_leadership``) is executed
        for real; the loser presents at the same arbiter.

        The winner's subsequent completion is likewise driven through the
        production verbs UNDER ITS TOKEN (recover → claim → mark_terminal →
        fenced complete_run): public resume() cannot legally re-enter a
        RUNNING run in slice 2 (the dead-leader takeover admission is the
        slice-4 guard flip), so "the seat loss did not poison the winner" is
        proven on the journal + runs surfaces rather than a sink list.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        crashed_token = _craft_crashed_lease(
            crashed,
            ingest_sequence=3,
            lease_owner="crashed-worker-1",
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)

        coord = _coord(crashed)
        winner_id = f"worker:{crashed.run_id}:winner"
        loser_id = f"worker:{crashed.run_id}:loser"

        # The winner's FIRST durable resume act, executed for real: one
        # IMMEDIATE transaction = seat takeover + FAILED→RUNNING flip.
        winner_token = coord.acquire_run_leadership(
            run_id=crashed.run_id,
            worker_id=winner_id,
            now=clock.now_utc(),
            window_seconds=80.0,
        )
        assert winner_token.leader_epoch == 2, "takeover of the begin_run epoch-1 seat"

        # Snapshot EVERY durable surface the refused loser must not touch.
        seat_before = _coordination_row(crashed.db, crashed.run_id)
        events_before = _coordination_events(crashed.db, crashed.run_id)
        workers_before = _run_workers(crashed.db, crashed.run_id)
        items_before = _work_items_by_token(crashed.db, crashed.run_id)
        outcomes_before = _completed_outcome_tokens(crashed.db, crashed.run_id)
        identities_before = sorted(_node_state_identities(crashed.db, crashed.run_id))

        # ---- the loser arrives at the arbiter ----
        with pytest.raises(NonResumableRunError, match=r"run leadership is held by") as exc_info:
            coord.acquire_run_leadership(
                run_id=crashed.run_id,
                worker_id=loser_id,
                now=clock.now_utc(),
                window_seconds=80.0,
            )
        assert winner_id in str(exc_info.value), "the refusal names the incumbent"

        # Zero mutation ("BEFORE any durable write", design §B.4 :196).
        assert _coordination_row(crashed.db, crashed.run_id) == seat_before, (
            "seat byte-identical: leader still winner, epoch unchanged, expiry NOT extended"
        )
        assert _run_workers(crashed.db, crashed.run_id) == workers_before
        assert all(worker["worker_id"] != loser_id for worker in _run_workers(crashed.db, crashed.run_id))
        events_after = _coordination_events(crashed.db, crashed.run_id)
        assert events_after == events_before, "no second leader_acquire, no fence_refusal — CAS-loss at acquisition is not a fence refusal"
        with crashed.db.engine.connect() as conn:
            status = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == crashed.run_id)).scalar_one()
        assert status == RunStatus.RUNNING.value
        items_after = _work_items_by_token(crashed.db, crashed.run_id)
        assert items_after == items_before, "journal rows untouched"
        assert items_after[crashed_token]["status"] == TokenWorkStatus.LEASED.value
        assert items_after[crashed_token]["attempt"] == 1
        assert items_after[crashed_token]["lease_owner"] == "crashed-worker-1"
        assert _recovery_events(crashed.db, crashed.run_id) == []
        assert _completed_outcome_tokens(crashed.db, crashed.run_id) == outcomes_before
        assert sorted(_node_state_identities(crashed.db, crashed.run_id)) == identities_before

        # ---- the winner completes normally under its token ----
        assert (
            crashed.repo.recover_expired_leases(
                run_id=crashed.run_id,
                now=clock.now_utc(),
                caller_owner=winner_id,
                coordination_token=winner_token,
            )
            == 1
        )
        claimed = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner=winner_id,
            lease_seconds=_DEFAULT_LEASE_SECONDS,
            now=clock.now_utc(),
        )
        assert claimed is not None and claimed.token_id == crashed_token and claimed.attempt == 2
        crashed.repo.mark_terminal(work_item_id=claimed.work_item_id, now=clock.now_utc(), expected_lease_owner=winner_id)
        crashed.factory.run_lifecycle.complete_run(crashed.run_id, RunStatus.COMPLETED, token=winner_token)
        coord.release_seat(token=winner_token, now=clock.now_utc())

        with crashed.db.engine.connect() as conn:
            final_status = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == crashed.run_id)).scalar_one()
        assert final_status == RunStatus.COMPLETED.value, "the seat loss did not poison the winner"
        final_items = _work_items_by_token(crashed.db, crashed.run_id)
        assert all(item["status"] == TokenWorkStatus.TERMINAL.value for item in final_items.values())
        assert final_items[crashed_token]["attempt"] == 2
        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        seat_final = _coordination_row(crashed.db, crashed.run_id)
        assert seat_final["leader_worker_id"] is None, "graceful release after finalize"
        crashed.db.close()

    def test_two_resumes_loser_after_winner_refused_at_entry_guard(self, tmp_path: Path) -> None:
        """THE FLIP (design §H :467): the loser-after refusal is the designed
        entry guard's "Run is terminal" — a clean operator-facing
        NonResumableRunError — NOT AuditIntegrityError("...from COMPLETED
        ...immutable") surfacing from the durable backstop. The backstop
        itself is independently pinned (tests/unit/core/landscape/
        test_run_lifecycle_repository.py and
        test_run_coordination_repository.py).
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        crashed_token = _craft_crashed_lease(
            crashed,
            ingest_sequence=3,
            lease_owner="crashed-worker-1",
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)

        # Both racing operators fetch equally-valid resume points from the
        # same durable FAILED state — the cross-process race decision point.
        resume_point_winner = _resume_point(crashed)
        resume_point_loser = _resume_point(crashed)
        assert resume_point_winner is not None and resume_point_loser is not None

        config_w, graph_w, sink_w, _source_w = _build_pipeline(_SOURCE_ROWS)
        result_winner = crashed.resume_orchestrator().resume(resume_point_winner, config_w, graph_w, payload_store=crashed.payload_store)
        assert result_winner.status == RunStatus.COMPLETED
        assert sink_w.results == [{"id": 3, "value": 30}]

        with crashed.db.engine.connect() as conn:
            completed_at_after_winner = conn.execute(
                select(runs_table.c.completed_at).where(runs_table.c.run_id == crashed.run_id)
            ).scalar_one()
        items_after_winner = _work_items_by_token(crashed.db, crashed.run_id)
        events_after_winner = _coordination_events(crashed.db, crashed.run_id)
        seat_after_winner = _coordination_row(crashed.db, crashed.run_id)

        # The loser arrives with its (now stale) resume point.
        clock.advance(10)
        config_l, graph_l, sink_l, _source_l = _build_pipeline(_SOURCE_ROWS)
        with pytest.raises(NonResumableRunError, match=r"terminal") as exc_info:
            crashed.resume_orchestrator().resume(resume_point_loser, config_l, graph_l, payload_store=crashed.payload_store)
        assert exc_info.value.run_id == crashed.run_id

        # Audit integrity after the race: the loser changed NOTHING.
        assert sink_l.results == []
        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        assert len(_completed_outcome_tokens(crashed.db, crashed.run_id)) == 4
        identities = _node_state_identities(crashed.db, crashed.run_id)
        assert len(identities) == len(set(identities))
        with crashed.db.engine.connect() as conn:
            status = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == crashed.run_id)).scalar_one()
            completed_at_after_loser = conn.execute(
                select(runs_table.c.completed_at).where(runs_table.c.run_id == crashed.run_id)
            ).scalar_one()
        assert status == RunStatus.COMPLETED.value
        assert completed_at_after_loser == completed_at_after_winner
        assert _work_items_by_token(crashed.db, crashed.run_id) == items_after_winner
        assert items_after_winner[crashed_token]["attempt"] == 2

        # Coordination surfaces: exactly one COMPLETED finalize event (the
        # winner's — the original run's graceful-shutdown ceremony also
        # finalized, as INTERRUPTED, before the harness crafted the crash
        # image); the loser appended NO coordination event of any type; the
        # seat row is unchanged from the post-winner image.
        finalize_events = _coordination_events(crashed.db, crashed.run_id, "finalize")
        completed_finalizes = [event for event in finalize_events if '"completed"' in str(event["context_json"])]
        assert len(completed_finalizes) == 1
        assert _coordination_events(crashed.db, crashed.run_id) == events_after_winner
        assert _coordination_row(crashed.db, crashed.run_id) == seat_after_winner
        crashed.db.close()

    def test_entry_guard_refuses_resume_while_run_status_running(self, tmp_path: Path) -> None:
        """Arm (a) of the entry guard (design §B.3, re-pinned at slice 2).

        The winner's mid-flight image is now produced by the PRODUCTION
        takeover CAS — ``acquire_run_leadership`` atomically seats the
        winner and flips FAILED→RUNNING — instead of the old bare
        ``update_run_status(RUNNING)``. The guard (whose status+seat check
        lives in the SHARED ``check_run_status_resumable``, so can_resume
        and resume() can never drift) then refuses the competing loser
        NAMING the live leader, the seat expiry, and the ``elspeth join``
        direction.

        The old KNOWN RESIDUAL TOCTOU (two resumes both observing FAILED at
        the guard) is CLOSED at epoch 21 by the seat CAS — exactly one racer
        commits ``acquire_run_leadership``; the guard now only closes the
        caller-convention gap, and check-then-act here is acceptable because
        the leadership CAS is the arbiter (ADR-030 §B.3).

        Slice ownership of the remaining arms: (b) ``elspeth join`` admits a
        follower — slice 5; (c) RUNNING + EXPIRED seat becomes resumable
        (dead-leader takeover; the guard learns seat liveness) — slice 4. In
        slice 2 RUNNING + expired seat is STILL REFUSED (flat reason) — the
        companion assertion at the bottom pins that this slice does not
        accidentally open the takeover arm early.

        Durable surfaces pinned — the refused loser changed NOTHING:
        - the crashed worker's LEASED journal row is untouched
          (status/attempt/lease_owner intact), no recovery events appended;
        - no new token outcomes, no new node_states identities;
        - the runs row still says RUNNING (the winner's state);
        - the coordination surfaces are untouched (winner still seated, epoch
          unchanged, expiry NOT extended; no loser run_workers row; zero
          fence_refusal events — the guard refusal is pre-CAS, pre-fence);
        - the loser's sink receives nothing.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        crashed_token = _craft_crashed_lease(
            crashed,
            ingest_sequence=3,
            lease_owner="crashed-worker-1",
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)

        resume_point_loser = _resume_point(crashed)
        assert resume_point_loser is not None

        # The winner's first durable resume write — the PRODUCTION image:
        # one IMMEDIATE transaction = live seat + FAILED→RUNNING.
        winner_id = f"worker:{crashed.run_id}:winner"
        _coord(crashed).acquire_run_leadership(
            run_id=crashed.run_id,
            worker_id=winner_id,
            now=clock.now_utc(),
            window_seconds=_GUARD_LIVE_SEAT_WINDOW_SECONDS,
        )

        # Advisory surface refuses, naming all three components: the live
        # leader, an ISO-8601 seat expiry, and the join direction.
        check = _recovery_manager(crashed).can_resume(crashed.run_id, crashed.graph)
        assert not check.can_resume
        assert check.reason is not None
        assert winner_id in check.reason
        assert re.search(r"seat expires \d{4}-\d{2}-\d{2}T\d{2}:\d{2}", check.reason)
        assert "elspeth join" in check.reason

        # Snapshot every durable surface the refused loser must not touch.
        items_before = _work_items_by_token(crashed.db, crashed.run_id)
        outcomes_before = _completed_outcome_tokens(crashed.db, crashed.run_id)
        identities_before = sorted(_node_state_identities(crashed.db, crashed.run_id))
        seat_before = _coordination_row(crashed.db, crashed.run_id)
        workers_before = _run_workers(crashed.db, crashed.run_id)
        events_before = _coordination_events(crashed.db, crashed.run_id)

        # ...and the public resume() refuses with the SAME reason (the
        # elspeth-2f23292372 parity contract), at entry, before any mutation.
        config_l, graph_l, sink_l, _source_l = _build_pipeline(_SOURCE_ROWS)
        with pytest.raises(NonResumableRunError, match=r"in progress under live leader") as exc_info:
            crashed.resume_orchestrator().resume(resume_point_loser, config_l, graph_l, payload_store=crashed.payload_store)
        assert exc_info.value.run_id == crashed.run_id
        assert exc_info.value.reason == check.reason

        # The refused loser changed NOTHING durable.
        assert sink_l.results == []
        items_after = _work_items_by_token(crashed.db, crashed.run_id)
        assert items_after == items_before
        assert items_after[crashed_token]["status"] == TokenWorkStatus.LEASED.value
        assert items_after[crashed_token]["attempt"] == 1
        assert items_after[crashed_token]["lease_owner"] == "crashed-worker-1"
        assert _recovery_events(crashed.db, crashed.run_id) == []
        assert _completed_outcome_tokens(crashed.db, crashed.run_id) == outcomes_before
        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        assert sorted(_node_state_identities(crashed.db, crashed.run_id)) == identities_before
        with crashed.db.engine.connect() as conn:
            status = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == crashed.run_id)).scalar_one()
        assert status == RunStatus.RUNNING.value
        assert _coordination_row(crashed.db, crashed.run_id) == seat_before, (
            "winner still seated, epoch unchanged, expiry not extended by the refused loser"
        )
        assert _run_workers(crashed.db, crashed.run_id) == workers_before, "no run_workers row for the loser"
        assert _coordination_events(crashed.db, crashed.run_id) == events_before
        assert _coordination_events(crashed.db, crashed.run_id, "fence_refusal") == [], "guard refusal is pre-CAS, pre-fence"

        # COMPANION ARM: seat usurped-then-EXPIRED ⇒ slice 2 still refuses
        # (flat reason — the guard learns seat liveness only in slice 4; the
        # takeover arm must not open early).
        with crashed.db.engine.begin() as conn:
            conn.execute(
                update(run_coordination_table)
                .where(run_coordination_table.c.run_id == crashed.run_id)
                .values(leader_heartbeat_expires_at=datetime(2020, 1, 1, tzinfo=UTC))
            )
        config_l2, graph_l2, sink_l2, _source_l2 = _build_pipeline(_SOURCE_ROWS)
        with pytest.raises(NonResumableRunError) as exc_info_expired:
            crashed.resume_orchestrator().resume(resume_point_loser, config_l2, graph_l2, payload_store=crashed.payload_store)
        assert exc_info_expired.value.reason == "Run is still in progress"
        assert sink_l2.results == []
        crashed.db.close()
