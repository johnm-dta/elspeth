# tests/integration/pipeline/test_barrier_intake_dispositions.py
"""ADR-030 slice 3 — intake dispositions over REAL executors (§E.3a / §E.5).

The unit twin (tests/unit/engine/test_adr030_slice3_intake.py) pins the
processor wiring against a mocked CoalesceExecutor; this file drives the
SAME dispositions through the PRODUCTION CoalesceExecutor over real
repositories, so the completed-keys machinery, the merge/fail policies and
the audit writes are all real:

* §E.3a late-branch release — a branch arriving after its group completed is
  adopted, rejected by the executor's completed-keys check, and
  journal-released with ``late_arrival`` context in the SAME drain iteration;
  the §D quiescence predicate is then satisfied (run can finalize COMPLETED,
  no stranded BLOCKED row). Both frames: the completing leader's own intake
  AND a takeover leader inheriting the unreleased late row (§E.4).
* §E.5 branch-loss replay — a follower-recorded durable loss
  (``adopted_epoch IS NULL``) is adopted journal-first and replayed through
  ``notify_branch_lost`` in ONE intake pass: a must-fail (require_all) group
  FAILS within that iteration (not at a timeout), and a best-effort merge
  fires carrying ``branches_lost`` audit content equal to the table row.
* §E.5 takeover survival — a loss recorded by a dead leader is re-derived
  from the ledger at restore (checkpoint scalar ABSENT — proving the table,
  not the D3 scalar, is the truth), and the group reaches the same terminal
  disposition as the no-crash run.

Implementation note (recon-spec deviation, pinned deliberately): the §E.3a
release happens AFTER the fenced adoption CAS — design §E.3a's rule fires
"when intake adoption hits a completed key", so the released row carries a
stamped ``barrier_adopted_epoch``, not NULL. The durable trace reads:
adopted by epoch E, then released as late.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select, update

from elspeth.contracts import TokenInfo
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.core.config import CoalesceSettings
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.database import begin_write
from elspeth.core.landscape.scheduler_repository import record_coalesce_branch_loss
from elspeth.core.landscape.schema import (
    coalesce_branch_losses_table,
    node_states_table,
    run_coordination_table,
    scheduler_events_table,
    token_outcomes_table,
    token_work_items_table,
)
from elspeth.engine.clock import MockClock
from elspeth.engine.coalesce_executor import CoalesceExecutor
from elspeth.engine.orchestrator.aggregation import run_end_of_input_barrier_flush
from elspeth.engine.orchestrator.types import ExecutionCounters, PipelineConfig
from elspeth.engine.processor import BarrierJournalRestoreContext, _LiveBarrierHold
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenManager
from elspeth.testing import make_row
from tests.fixtures.factories import make_context
from tests.unit.engine.test_processor import (
    _make_factory,
    _make_processor,
    _persist_blocked_scheduler_work,
)

_T0 = 1_750_000_000.0
RUN_ID = "test-run"
USURPER = "worker-usurper"
COALESCE_NODE = NodeID("coalesce::merge")
MERGE = CoalesceName("merge")


def _usurp_seat(db: LandscapeDB, clock: MockClock) -> None:
    """In-DB takeover image (harness ``_usurp_seat``): epoch bump, live expiry."""
    now = clock.now_utc()
    with begin_write(db.engine) as conn:
        conn.execute(
            update(run_coordination_table)
            .where(run_coordination_table.c.run_id == RUN_ID)
            .values(
                leader_worker_id=USURPER,
                leader_epoch=run_coordination_table.c.leader_epoch + 1,
                leader_heartbeat_expires_at=now + timedelta(seconds=300),
                updated_at=now,
            )
        )


def _real_coalesce_executor(factory: Any, clock: MockClock, *, policy: str) -> CoalesceExecutor:
    token_manager = TokenManager(factory.data_flow, step_resolver=lambda node_id: 2)
    executor = CoalesceExecutor(
        execution=factory.execution,
        span_factory=SpanFactory(),
        token_manager=token_manager,
        run_id=RUN_ID,
        step_resolver=lambda node_id: 2,
        clock=clock,
        data_flow=factory.data_flow,
    )
    executor.register_coalesce(
        CoalesceSettings(
            name="merge",
            branches=["a", "b"],
            policy=policy,
            merge="union",
            on_success="out",
            # best_effort validation requires a timeout; it is deliberately
            # enormous — these tests prove dispositions fire WITHOUT it.
            timeout_seconds=3600.0 if policy == "best_effort" else None,
        ),
        COALESCE_NODE,
        output_schema=SchemaContract(mode="OBSERVED", fields=(), locked=False),
    )
    return executor


def _coalesce_processor(
    factory: Any,
    executor: CoalesceExecutor,
    clock: MockClock,
    *,
    barrier_restore: Any = None,
    stamp_blocked_rows_adopted: bool = True,
) -> Any:
    return _make_processor(
        factory,
        coalesce_executor=executor,
        coalesce_node_ids={MERGE: COALESCE_NODE},
        node_step_map={COALESCE_NODE: 2},
        coalesce_on_success_map={MERGE: "out"},
        sink_names=frozenset({"out"}),
        clock=clock,
        barrier_restore=barrier_restore,
        stamp_blocked_rows_adopted=stamp_blocked_rows_adopted,
    )


def _branch_token(branch: str, *, token_id: str | None = None, row_id: str = "row-1") -> TokenInfo:
    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"tok-branch-{branch}",
        row_data=make_row({f"field_{branch}": 1}),
        branch_name=branch,
    )


def _arrive_via_intake(factory: Any, processor: Any, token: TokenInfo, *, ingest_sequence: int = 0) -> list[Any]:
    """One branch arrival on the live path: BLOCKED deposit + stash + intake."""
    ctx = make_context(landscape=factory.plugin_audit_writer())
    _persist_blocked_scheduler_work(
        factory,
        processor,
        token,
        node_id=COALESCE_NODE,
        barrier_key="merge",
        adopted=False,
        ingest_sequence=ingest_sequence,
        coalesce_name="merge",
    )
    processor._live_barrier_holds[token.token_id] = _LiveBarrierHold(token=token, barrier_key="merge")
    return processor.run_barrier_intake(ctx)


def _work_item_row(db: LandscapeDB, token_id: str) -> dict[str, Any]:
    with db.connection() as conn:
        row = conn.execute(select(token_work_items_table).where(token_work_items_table.c.token_id == token_id)).mappings().one()
    return dict(row)


def _release_events(db: LandscapeDB, token_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        return [
            dict(row)
            for row in conn.execute(
                select(scheduler_events_table)
                .where(scheduler_events_table.c.token_id == token_id)
                .where(scheduler_events_table.c.event_type == SchedulerEventType.MARK_BLOCKED_BARRIER_TERMINAL.value)
            ).mappings()
        ]


def _record_foreign_loss(db: LandscapeDB, clock: MockClock, *, branch: str, token_id: str, reason: str) -> None:
    """A follower-recorded durable loss (production verb, ``adopted_epoch IS NULL``)."""
    with begin_write(db.engine) as conn:
        assert record_coalesce_branch_loss(
            conn,
            run_id=RUN_ID,
            coalesce_name="merge",
            row_id="row-1",
            branch_name=branch,
            token_id=token_id,
            reason=reason,
            recorded_by="worker-follower",
            now=clock.now_utc(),
        )


def _loss_rows(db: LandscapeDB) -> list[dict[str, Any]]:
    with db.connection() as conn:
        return [dict(row) for row in conn.execute(select(coalesce_branch_losses_table)).mappings()]


def _assert_quiescent_and_finalize_ready(processor: Any) -> None:
    """The §D predicate: no BLOCKED holds, zero READY/LEASED journal work —
    a run in this state finalizes COMPLETED once PENDING_SINK rows deliver."""
    assert processor.has_blocked_barrier_work() is False
    assert processor.count_unquiesced_scheduler_work() == 0


@pytest.mark.timeout(120)
class TestLateBranchRelease:
    """§E.3a: the late branch is adopted, rejected and journal-released."""

    def test_late_branch_released_in_same_iteration_run_finalize_ready(self) -> None:
        clock = MockClock(start=_T0)
        db, factory = _make_factory()
        executor = _real_coalesce_executor(factory, clock, policy="first")
        processor = _coalesce_processor(factory, executor, clock)

        # Group (merge, row-1) completes with branch b outstanding.
        results = _arrive_via_intake(factory, processor, _branch_token("a"))
        assert len(results) == 1
        merged_token_id = str(results[0].token.token_id)
        assert results[0].scheduler_pending_sink is True
        merged_row = _work_item_row(db, merged_token_id)
        assert merged_row["status"] == TokenWorkStatus.PENDING_SINK.value
        assert merged_row["pending_sink_name"] == "out"
        assert _work_item_row(db, "tok-branch-a")["status"] == TokenWorkStatus.TERMINAL.value

        # Branch b arrives AFTER completion (the slow-claim-past-merge shape).
        clock.advance(2.0)
        late_results = _arrive_via_intake(factory, processor, _branch_token("b"), ingest_sequence=1)

        # (1)/(2) NOT adopted into a batch/group — released to TERMINAL with
        # late_arrival context in the SAME drain iteration.
        assert [(r.outcome, r.path) for r in late_results] == [(TerminalOutcome.FAILURE, TerminalPath.UNROUTED)]
        late_row = _work_item_row(db, "tok-branch-b")
        assert late_row["status"] == TokenWorkStatus.TERMINAL.value
        events = _release_events(db, "tok-branch-b")
        assert len(events) == 1
        context = json.loads(str(events[0]["context_json"]))
        assert context["late_arrival"] is True
        assert context["reason"] == "late_arrival_after_merge"
        assert context["released_by"] == processor._scheduler_lease_owner
        assert context["scope_row_id"] == "row-1"

        # (3) — implemented §E.3a shape (supersedes the recon-spec NULL
        # expectation): adoption PRECEDES the completed-key discovery, so the
        # released row carries the adopting leader's epoch.
        assert late_row["barrier_adopted_epoch"] == 1

        # The executor recorded the late arrival's FAILURE outcome durably.
        with db.connection() as conn:
            outcome_row = (
                conn.execute(
                    select(token_outcomes_table)
                    .where(token_outcomes_table.c.token_id == "tok-branch-b")
                    .where(token_outcomes_table.c.completed == 1)
                )
                .mappings()
                .one()
            )
        assert outcome_row["outcome"] == "failure"

        # (4) Run finalize-ready: no stranded BLOCKED row, journal quiesced.
        _assert_quiescent_and_finalize_ready(processor)

    def test_fresh_leader_releases_follower_blocked_late_branch(self) -> None:
        """A normal-run leader adopts a follower-blocked row from the journal."""
        clock = MockClock(start=_T0)
        db, factory = _make_factory()
        executor = _real_coalesce_executor(factory, clock, policy="first")
        processor = _coalesce_processor(factory, executor, clock)

        results = _arrive_via_intake(factory, processor, _branch_token("a"))
        assert len(results) == 1 and results[0].scheduler_pending_sink is True

        # Branch b is deposited by another live worker. The leader has no
        # process-local _live_barrier_holds entry and is not a resume leader.
        clock.advance(2.0)
        _persist_blocked_scheduler_work(
            factory,
            processor,
            _branch_token("b"),
            node_id=COALESCE_NODE,
            barrier_key="merge",
            adopted=False,
            ingest_sequence=1,
            coalesce_name="merge",
        )

        ctx = make_context(landscape=factory.plugin_audit_writer())
        late_results = processor.run_barrier_intake(ctx)

        assert [(r.outcome, r.path) for r in late_results] == [(TerminalOutcome.FAILURE, TerminalPath.UNROUTED)]
        late_row = _work_item_row(db, "tok-branch-b")
        assert late_row["status"] == TokenWorkStatus.TERMINAL.value
        assert late_row["barrier_adopted_epoch"] == 1
        events = _release_events(db, "tok-branch-b")
        assert len(events) == 1
        context = json.loads(str(events[0]["context_json"]))
        assert context["late_arrival"] is True
        assert context["scope_row_id"] == "row-1"
        _assert_quiescent_and_finalize_ready(processor)

    def test_takeover_leader_releases_inherited_late_branch(self) -> None:
        """§E.4 inherited disposition: the group completed under leader A; b's
        BLOCKED row arrived but A died before releasing it. Leader B's intake
        (journal rehydration under resume provenance + landscape-reconstructed
        completed keys) performs the SAME release."""
        clock = MockClock(start=_T0)
        db, factory = _make_factory()
        executor_a = _real_coalesce_executor(factory, clock, policy="first")
        processor_a = _coalesce_processor(factory, executor_a, clock)

        results = _arrive_via_intake(factory, processor_a, _branch_token("a"))
        assert len(results) == 1 and results[0].scheduler_pending_sink is True

        # b's BLOCKED row lands; A dies before its intake runs (no release,
        # the live stash dies with the process).
        clock.advance(2.0)
        _persist_blocked_scheduler_work(
            factory,
            processor_a,
            _branch_token("b"),
            node_id=COALESCE_NODE,
            barrier_key="merge",
            adopted=False,
            ingest_sequence=1,
            coalesce_name="merge",
        )

        _usurp_seat(db, clock)
        executor_b = _real_coalesce_executor(factory, clock, policy="first")
        processor_b = _coalesce_processor(
            factory,
            executor_b,
            clock,
            barrier_restore=BarrierJournalRestoreContext(
                resume_checkpoint_id="ckpt-takeover",
                barrier_scalars=None,
                batch_id_remap={},
            ),
            # b's row is the dead leader's UNADOPTED deposit — it must stay
            # intake-pending through the restore (the journal-first intake,
            # not the restore, dispositions it).
            stamp_blocked_rows_adopted=False,
        )
        # Restore left the intake-pending row for the journal-first intake
        # and reconstructed the completed key from the Landscape.
        assert processor_b.has_blocked_barrier_work() is True
        assert ("merge", "row-1") in executor_b._completed_keys

        ctx = make_context(landscape=factory.plugin_audit_writer())
        late_results = processor_b.run_barrier_intake(ctx)

        assert [(r.outcome, r.path) for r in late_results] == [(TerminalOutcome.FAILURE, TerminalPath.UNROUTED)]
        late_row = _work_item_row(db, "tok-branch-b")
        assert late_row["status"] == TokenWorkStatus.TERMINAL.value
        assert late_row["barrier_adopted_epoch"] == 2, "adopted by the takeover leader's epoch, then released"
        events = _release_events(db, "tok-branch-b")
        assert len(events) == 1
        context = json.loads(str(events[0]["context_json"]))
        assert context["late_arrival"] is True
        assert context["scope_row_id"] == "row-1"
        _assert_quiescent_and_finalize_ready(processor_b)

    def test_restore_releases_adopted_holdless_late_row_against_completed_key(self) -> None:
        """§E.3a + §E.4 crash-window: adopted CAS committed, late release did NOT run.

        This is the crash window between _intake_adopt_coalesce_row step (1) and
        step (3): adopt_blocked_barrier_item committed (barrier_adopted_epoch stamped),
        CoalesceExecutor.accept() identified the late arrival and wrote the FAILED
        node_state + FAILURE outcome (step 2 completed), but mark_blocked_barrier_terminal
        (step 3) never ran because the leader died.

        On takeover, leader B's restore reconcile (§E.3a at restore) must detect the
        adopted BLOCKED row whose key is Landscape-completed and journal-release it,
        so the §D quiescence predicate is satisfied and the run can finalize.
        """
        clock = MockClock(start=_T0)
        db, factory = _make_factory()
        executor_a = _real_coalesce_executor(factory, clock, policy="first")
        processor_a = _coalesce_processor(factory, executor_a, clock)

        # Group (merge, row-1) completes with branch a.
        results = _arrive_via_intake(factory, processor_a, _branch_token("a"))
        assert len(results) == 1 and results[0].scheduler_pending_sink is True

        # b's BLOCKED row arrives and is ADOPTED (adoption CAS committed),
        # but leader A crashes before mark_blocked_barrier_terminal runs.
        # Simulate this by persisting b's row with adopted=True (post-adoption image),
        # but skipping the late-release step.
        clock.advance(2.0)
        _persist_blocked_scheduler_work(
            factory,
            processor_a,
            _branch_token("b"),
            node_id=COALESCE_NODE,
            barrier_key="merge",
            adopted=True,  # CAS committed — barrier_adopted_epoch stamped
            ingest_sequence=1,
            coalesce_name="merge",
        )
        # Confirm b's row is BLOCKED (not yet released).
        b_row_before = _work_item_row(db, "tok-branch-b")
        assert b_row_before["status"] == TokenWorkStatus.BLOCKED.value
        assert b_row_before["barrier_adopted_epoch"] == 1

        # Takeover: leader B restores. The §E.3a restore reconcile must
        # detect the adopted row against the completed key and release it.
        _usurp_seat(db, clock)
        executor_b = _real_coalesce_executor(factory, clock, policy="first")
        # stamp_blocked_rows_adopted=False because b's row is ALREADY adopted;
        # the helper must NOT re-stamp (would increment epoch, confusing the check).
        processor_b = _coalesce_processor(
            factory,
            executor_b,
            clock,
            barrier_restore=BarrierJournalRestoreContext(
                resume_checkpoint_id="ckpt-takeover",
                barrier_scalars=None,
                batch_id_remap={},
            ),
            stamp_blocked_rows_adopted=False,
        )

        # The restore reconcile ran and released the adopted late row.
        # The §D predicate is satisfied immediately — no intake pass needed.
        b_row_after = _work_item_row(db, "tok-branch-b")
        assert b_row_after["status"] == TokenWorkStatus.TERMINAL.value, (
            "restore §E.3a reconcile must terminal-release adopted-holdless late row"
        )
        events = _release_events(db, "tok-branch-b")
        assert len(events) == 1
        context = json.loads(str(events[0]["context_json"]))
        assert context["late_arrival"] is True
        assert context["restore_reconcile"] is True
        assert context["scope_row_id"] == "row-1"

        _assert_quiescent_and_finalize_ready(processor_b)

    def test_restore_recovers_adopted_holdless_row_against_incomplete_key(self) -> None:
        """§E.3 crash-window: adopted CAS committed, accept() never ran, key NOT completed.

        The crash sequence: adopt_blocked_barrier_item committed (barrier_adopted_epoch
        stamped), then the leader died before CoalesceExecutor.accept() wrote the PENDING
        hold node_state.  Key is still pending (no merge happened yet).

        On takeover, leader B's restore reconcile resets barrier_adopted_epoch to NULL so
        the row becomes intake-pending again.  The first journal-first intake adopts it
        afresh and runs the full accept + trigger path, which may fire a merge — correctly
        producing RowResults that the intake caller commits to the journal.

        The processor MUST be created successfully.  After one intake pass, the merge fires
        (both branches now present for require_all) and quiescence is achieved.
        """
        clock = MockClock(start=_T0)
        db, factory = _make_factory()

        executor_a = _real_coalesce_executor(factory, clock, policy="require_all")
        processor_a = _coalesce_processor(factory, executor_a, clock)

        # Branch a: full live arrival (adopt + accept → held).
        held_results = _arrive_via_intake(factory, processor_a, _branch_token("a"))
        assert held_results == []  # held, not completed

        # Branch b: ADOPTED (CAS stamped) but accept() never ran.
        # Simulate: deposit the BLOCKED row and stamp epoch — no executor.accept call.
        _persist_blocked_scheduler_work(
            factory,
            processor_a,
            _branch_token("b"),
            node_id=COALESCE_NODE,
            barrier_key="merge",
            adopted=True,  # adoption CAS committed
            ingest_sequence=1,
            coalesce_name="merge",
        )
        b_row_before = _work_item_row(db, "tok-branch-b")
        assert b_row_before["status"] == TokenWorkStatus.BLOCKED.value
        assert b_row_before["barrier_adopted_epoch"] == 1

        # Takeover: leader B restores. The crash-window recovery resets b's epoch
        # to NULL (intake-pending) and restore_from_journal sees only branch a.
        _usurp_seat(db, clock)
        executor_b = _real_coalesce_executor(factory, clock, policy="require_all")
        processor_b = _coalesce_processor(
            factory,
            executor_b,
            clock,
            barrier_restore=BarrierJournalRestoreContext(
                resume_checkpoint_id="ckpt-takeover",
                barrier_scalars=None,
                batch_id_remap={},
            ),
            stamp_blocked_rows_adopted=False,  # b's row is already adopted; reset is done by restore reconcile
        )

        # After restore: branch a is in _pending; branch b's row is back to
        # intake-pending (barrier_adopted_epoch == NULL) after the reset.
        b_row_after_restore = _work_item_row(db, "tok-branch-b")
        assert b_row_after_restore["barrier_adopted_epoch"] is None, (
            "crash-window recovery must reset adopted epoch to NULL for intake re-processing"
        )
        key = ("merge", "row-1")
        assert key in executor_b._pending, "branch a must be in _pending from normal restore"
        assert "a" in executor_b._pending[key].branches

        # has_blocked_barrier_work must be True: branch b's reset row is still BLOCKED.
        assert processor_b.has_blocked_barrier_work() is True

        # One intake pass: branch b is re-adopted and accepted → merge fires (require_all).
        ctx = make_context(landscape=factory.plugin_audit_writer())
        merge_results = processor_b.run_barrier_intake(ctx)
        assert len(merge_results) == 1
        merged = merge_results[0]
        assert merged.outcome is TerminalOutcome.SUCCESS
        assert merged.scheduler_pending_sink is True

        _assert_quiescent_and_finalize_ready(processor_b)


@pytest.mark.timeout(120)
class TestBranchLossReplay:
    """§E.5: follower-recorded losses replay journal-first in ONE intake pass."""

    def test_must_fail_group_fails_within_the_replay_iteration(self) -> None:
        clock = MockClock(start=_T0)
        db, factory = _make_factory()
        executor = _real_coalesce_executor(factory, clock, policy="require_all")
        processor = _coalesce_processor(factory, executor, clock)

        # Branch a held (live arrival); branch b's loss recorded follower-style.
        held_results = _arrive_via_intake(factory, processor, _branch_token("a"))
        assert held_results == []
        _record_foreign_loss(db, clock, branch="b", token_id="tok-branch-b", reason="quarantined:boom")

        # ONE leader intake pass: adopt-the-loss (journal-first) -> replay
        # through notify_branch_lost -> require_all fails the group NOW,
        # not at any timeout (none is configured — failure here proves it).
        ctx = make_context(landscape=factory.plugin_audit_writer())
        results = processor.run_barrier_intake(ctx)

        assert len(results) == 1
        assert (results[0].outcome, results[0].path) == (TerminalOutcome.FAILURE, TerminalPath.UNROUTED)
        assert results[0].token.token_id == "tok-branch-a"
        assert _work_item_row(db, "tok-branch-a")["status"] == TokenWorkStatus.TERMINAL.value

        # (iv) the replay cursor is stamped under the adopting epoch.
        (loss_row,) = _loss_rows(db)
        assert loss_row["adopted_epoch"] == 1
        assert loss_row["reason"] == "quarantined:boom"
        _assert_quiescent_and_finalize_ready(processor)

    def test_best_effort_merge_carries_branches_lost_from_the_table(self) -> None:
        clock = MockClock(start=_T0)
        db, factory = _make_factory()
        executor = _real_coalesce_executor(factory, clock, policy="best_effort")
        processor = _coalesce_processor(factory, executor, clock)

        held_results = _arrive_via_intake(factory, processor, _branch_token("a"))
        assert held_results == []
        _record_foreign_loss(db, clock, branch="b", token_id="tok-branch-b", reason="quarantined:boom")

        ctx = make_context(landscape=factory.plugin_audit_writer())
        results = processor.run_barrier_intake(ctx)

        # The loss completes the best-effort group: merged child emitted as
        # PENDING_SINK in the same completion (terminal coalesce).
        assert len(results) == 1
        merged_result = results[0]
        assert merged_result.outcome is TerminalOutcome.SUCCESS
        assert merged_result.scheduler_pending_sink is True
        merged_row = _work_item_row(db, str(merged_result.token.token_id))
        assert merged_row["status"] == TokenWorkStatus.PENDING_SINK.value
        assert _work_item_row(db, "tok-branch-a")["status"] == TokenWorkStatus.TERMINAL.value

        # branches_lost audit content == the durable table row, not memory.
        with db.connection() as conn:
            contexts = (
                conn.execute(
                    select(node_states_table.c.context_after_json)
                    .where(node_states_table.c.node_id == str(COALESCE_NODE))
                    .where(node_states_table.c.context_after_json.is_not(None))
                )
                .scalars()
                .all()
            )
        merge_contexts = [c for c in contexts if c is not None and "branches_lost" in c]
        assert merge_contexts, "the merge audit record carries branches_lost"
        assert any(json.loads(c).get("branches_lost") == {"b": "quarantined:boom"} for c in merge_contexts)

        (loss_row,) = _loss_rows(db)
        assert loss_row["adopted_epoch"] == 1
        _assert_quiescent_and_finalize_ready(processor)

    def test_loss_survives_takeover_and_group_reaches_same_disposition(self) -> None:
        """§E.5 takeover survival over the REAL executor: the dead leader
        recorded the loss (rode its disposition txn) but never replayed it.
        The new leader re-derives ``lost_branches`` from the LEDGER (the
        checkpoint scalar is absent — ``barrier_scalars=None`` — proving the
        table is the truth), the replay cursor is stamped without a duplicate
        notify, and the must-fail group reaches the SAME terminal disposition
        as the no-crash run: branch a FAILURE/UNROUTED, journal quiescent."""
        clock = MockClock(start=_T0)
        db, factory = _make_factory()
        executor_a = _real_coalesce_executor(factory, clock, policy="require_all")
        processor_a = _coalesce_processor(factory, executor_a, clock)

        # Leader A: branch a held + adopted; b's loss recorded; A dies
        # BEFORE the replay (adopted_epoch stays NULL).
        held_results = _arrive_via_intake(factory, processor_a, _branch_token("a"))
        assert held_results == []
        _record_foreign_loss(db, clock, branch="b", token_id="tok-branch-b", reason="quarantined:boom")
        (loss_row,) = _loss_rows(db)
        assert loss_row["adopted_epoch"] is None

        _usurp_seat(db, clock)
        executor_b = _real_coalesce_executor(factory, clock, policy="require_all")
        processor_b = _coalesce_processor(
            factory,
            executor_b,
            clock,
            barrier_restore=BarrierJournalRestoreContext(
                resume_checkpoint_id="ckpt-takeover",
                barrier_scalars=None,  # the D3 scalar is GONE; the ledger must carry the loss
                batch_id_remap={},
            ),
        )

        # Restore seeded lost_branches from the durable ledger.
        pending = executor_b._pending[("merge", "row-1")]
        assert dict(pending.lost_branches) == {"b": "quarantined:boom"}
        assert set(pending.branches) == {"a"}

        # §D EOF arm: intake (stamps the replay cursor; the restored loss is
        # already in memory, so replay dedups — no duplicate-notify crash),
        # then the coalesce flush resolves the must-fail group.
        config = MagicMock(spec=PipelineConfig)
        config.aggregation_settings = {}
        ctx = make_context(landscape=factory.plugin_audit_writer())
        counters = ExecutionCounters()
        run_end_of_input_barrier_flush(
            config=config,
            processor=processor_b,
            ctx=ctx,
            counters=counters,
            pending_tokens={},
            coalesce_executor=executor_b,
            coalesce_node_map={MERGE: COALESCE_NODE},
        )

        # Same disposition as the no-crash run: held branch a FAILED out.
        assert _work_item_row(db, "tok-branch-a")["status"] == TokenWorkStatus.TERMINAL.value
        with db.connection() as conn:
            outcome_row = (
                conn.execute(
                    select(token_outcomes_table)
                    .where(token_outcomes_table.c.token_id == "tok-branch-a")
                    .where(token_outcomes_table.c.completed == 1)
                )
                .mappings()
                .one()
            )
        assert outcome_row["outcome"] == "failure"
        (loss_after,) = _loss_rows(db)
        assert loss_after["adopted_epoch"] == 2, "the takeover leader stamped the replay cursor under ITS epoch"
        assert counters.rows_coalesce_failed == 1
        _assert_quiescent_and_finalize_ready(processor_b)
