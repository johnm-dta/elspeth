# tests/unit/engine/test_adr030_slice3_intake.py
"""ADR-030 slice 3 — journal-first barrier acceptance (§E.2/§E.3/§E.3a/§E.5, §D, §H).

The §H test battery owed by the slice:

* §H 476 — backdated accept timing: a batch's timeout fire time is a pure
  function of durable state (``barrier_blocked_at(oldest member) +
  timeout_seconds``), invariant whether acceptance happened at the same
  leader's next-iteration intake or at a takeover leader's delayed intake.
* §H 477 — per-firing-group snapshot algebra lives in
  tests/unit/core/landscape/test_scheduler_repository_complete_barrier.py
  (slice-3 step 1); the engine-side wiring (snapshot == buffered batch /
  fired group) is exercised here through the intake-fired flush paths.
* §H 478 — late-branch §E.3a release: a branch arriving after its group
  completed is journal-released by the intake with a ``late_arrival``
  release context and surfaces a FAILURE RowResult.
* §H 479 — §E.5 branch-loss hand-off: the durable loss record commits with
  the branch's own disposition, the must-fail group fails within ONE drain
  step of the loss (record-then-notify), and the loss survives takeover
  (restore seeds executor memory from the ledger).
* §H 480 — §D step-2 EOF gating: the final flush is refused while the
  journal still holds READY/LEASED work that could deposit a new arrival; a
  slow in-flight row joins the SINGLE final batch.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from sqlalchemy import select

from elspeth.contracts import TokenInfo, TransformResult
from elspeth.contracts.enums import TerminalOutcome, TerminalPath, TriggerType
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.core.config import AggregationSettings
from elspeth.core.landscape.schema import (
    coalesce_branch_losses_table,
    scheduler_events_table,
    token_work_items_table,
)
from elspeth.engine.clock import MockClock
from elspeth.engine.coalesce_executor import CoalesceExecutor, CoalesceOutcome
from elspeth.engine.orchestrator.leader_drain import run_end_of_input_barrier_flush
from elspeth.engine.orchestrator.types import ExecutionCounters, PipelineConfig
from elspeth.engine.processor import _LiveBarrierHold
from elspeth.testing import make_row, make_token_info
from tests.fixtures.factories import make_context
from tests.unit.engine.test_processor import (
    BarrierJournalRestoreContext,
    _make_factory,
    _make_mock_transform,
    _make_processor,
    _make_source_row,
    _persist_blocked_scheduler_work,
)

AGG_NODE = NodeID("agg-1")


def _agg_processor(
    factory: Any,
    *,
    trigger: dict[str, Any],
    transform: Any,
    clock: Any = None,
    barrier_restore: Any = None,
    output_mode: str = "passthrough",
) -> Any:
    return _make_processor(
        factory,
        node_step_map={NodeID("source-0"): 0, AGG_NODE: 1},
        node_to_next={NodeID("source-0"): AGG_NODE, AGG_NODE: None},
        node_to_plugin={AGG_NODE: transform},
        aggregation_settings={
            AGG_NODE: AggregationSettings(
                name="batch_agg",
                plugin="agg-transform",
                input="default",
                on_error="discard",
                trigger=trigger,
                output_mode=output_mode,
            ),
        },
        clock=clock,
        barrier_restore=barrier_restore,
    )


def _passthrough_flush_transform() -> Mock:
    transform = _make_mock_transform(
        node_id=str(AGG_NODE),
        name="agg-transform",
        is_batch_aware=True,
        on_success="agg_sink",
    )

    def _process(rows: list[Any], ctx: Any) -> TransformResult:
        # success_multi requires ONE shared contract instance across rows.
        from elspeth.contracts.schema_contract import PipelineRow

        shared_contract = rows[0].contract
        return TransformResult.success_multi(
            [PipelineRow(row.to_dict(), shared_contract) for row in rows],
            success_reason={"action": "passthrough"},
        )

    transform.process.side_effect = _process
    return transform


class TestBackdatedAcceptTiming:
    """§H 476 — the trigger anchor is a pure function of durable state."""

    def test_timeout_anchor_is_blocked_at_even_when_adoption_is_delayed(self) -> None:
        """A takeover leader adopting a row 5 s after it blocked inherits the original anchor.

        The intake-pending row blocked at T0; the (resume-shaped) leader's
        first intake runs at T0+5. With live-clock anchoring the timeout
        (10 s) would fire at T0+15; the §E.2 backdated accept anchors it at
        T0, so it fires at T0+10 — invariant under takeover (§H 476).
        """
        clock = MockClock(start=1_700_000_000.0)
        _db, factory = _make_factory()
        transform = _passthrough_flush_transform()
        processor = _agg_processor(
            factory,
            trigger={"timeout_seconds": 10},
            transform=transform,
            clock=clock,
            barrier_restore=BarrierJournalRestoreContext(
                resume_checkpoint_id="ckpt-takeover",
                barrier_scalars=None,
                batch_id_remap={},
            ),
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        # T0: a prior leader deposited the BLOCKED row and crashed before
        # adoption (barrier_adopted_epoch stays NULL).
        token = make_token_info(row_id="row-1", token_id="tok-1", data={"value": 1})
        _persist_blocked_scheduler_work(factory, processor, token, node_id=AGG_NODE, barrier_key=str(AGG_NODE), adopted=False)

        # T0+5: the new leader's first intake adopts with the backdated anchor.
        clock.advance(5.0)
        intake_results = processor.run_barrier_intake(ctx)
        assert intake_results == []
        assert processor.get_aggregation_buffer_count(AGG_NODE) == 1

        # T0+9.5: not yet — anchored at T0, not at adoption (T0+5).
        clock.advance(4.5)
        should_flush, _ = processor.check_aggregation_timeout(AGG_NODE)
        assert should_flush is False

        # T0+10.5: fires. (Live-clock anchoring would not fire until T0+15.)
        clock.advance(1.0)
        should_flush, trigger_type = processor.check_aggregation_timeout(AGG_NODE)
        assert should_flush is True
        assert trigger_type is TriggerType.TIMEOUT

    def test_buffered_outcome_recorded_at_is_backdated_to_blocked_at(self) -> None:
        """The adoption verb's BUFFERED outcome carries the durable arrival instant."""
        clock = MockClock(start=1_700_000_000.0)
        _db, factory = _make_factory()
        transform = _passthrough_flush_transform()
        processor = _agg_processor(factory, trigger={"count": 5}, transform=transform, clock=clock)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        results = processor.process_row(
            row_index=0,
            source_row=_make_source_row({"value": 1}),
            transforms=[transform],
            ctx=ctx,
            source_row_index=0,
            ingest_sequence=0,
        )
        assert [(r.outcome, r.path) for r in results] == [(None, TerminalPath.BUFFERED)]
        token_id = results[0].token.token_id

        outcome = factory.data_flow.get_token_outcome(token_id)
        assert outcome is not None
        with factory.scheduler._engine.connect() as conn:
            blocked_at = conn.execute(
                select(token_work_items_table.c.barrier_blocked_at).where(token_work_items_table.c.token_id == token_id)
            ).scalar_one()
        assert outcome.recorded_at.replace(tzinfo=None) == blocked_at.replace(tzinfo=None)


class TestLateArrivalRelease:
    """§H 478 / §E.3a — late branches are journal-released by the intake."""

    def test_late_arrival_is_released_with_late_arrival_context(self) -> None:
        late_token = TokenInfo(row_id="row-1", token_id="tok-late", row_data=make_row({}), branch_name="path_b")
        coalesce = Mock(spec=CoalesceExecutor)
        coalesce.accept.return_value = CoalesceOutcome(
            held=False,
            failure_reason="late_arrival_after_merge",
            consumed_tokens=(late_token,),
            outcomes_recorded=True,
            late_arrival=True,
        )
        db, factory = _make_factory()
        processor = _make_processor(
            factory,
            coalesce_executor=coalesce,
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 2},
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        _persist_blocked_scheduler_work(
            factory, processor, late_token, node_id=NodeID("coalesce::merge"), barrier_key="merge", adopted=False
        )
        processor._live_barrier_holds[late_token.token_id] = _LiveBarrierHold(token=late_token, barrier_key="merge")

        results, child_items = processor._run_barrier_intake_pass(ctx)

        assert child_items == []
        assert len(results) == 1
        assert (results[0].outcome, results[0].path) == (TerminalOutcome.FAILURE, TerminalPath.UNROUTED)

        with db.connection() as conn:
            status = conn.execute(
                select(token_work_items_table.c.status).where(token_work_items_table.c.token_id == "tok-late")
            ).scalar_one()
            release_contexts = (
                conn.execute(
                    select(scheduler_events_table.c.context_json)
                    .where(scheduler_events_table.c.token_id == "tok-late")
                    .where(scheduler_events_table.c.event_type == "mark_blocked_barrier_terminal")
                )
                .scalars()
                .all()
            )
        # Run-completable: the late row reached TERMINAL in the same drain
        # iteration (§E.3a), so finalize's unresolved-work invariant passes.
        assert status == TokenWorkStatus.TERMINAL.value
        assert len(release_contexts) == 1
        assert '"late_arrival":true' in release_contexts[0]
        assert '"reason":"late_arrival_after_merge"' in release_contexts[0]
        assert '"scope_row_id":"row-1"' in release_contexts[0]
        assert processor.has_unresolved_scheduler_work() is False


class TestBranchLossHandOff:
    """§H 479 / §E.5 — record-then-notify, one-drain-step must-fail, takeover."""

    def _forked_processor(self, factory: Any, coalesce_executor: Any, *, barrier_restore: Any = None) -> Any:
        return _make_processor(
            factory,
            coalesce_executor=coalesce_executor,
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            branch_to_coalesce={"path_a": CoalesceName("merge"), "path_b": CoalesceName("merge")},
            node_step_map={NodeID("coalesce::merge"): 2},
            barrier_restore=barrier_restore,
        )

    def test_loss_record_rides_the_failing_branch_disposition(self) -> None:
        """A failing branch's durable loss commits with its mark_failed, and the
        require_all must-fail consequence surfaces in the SAME drain step."""
        held_token = TokenInfo(row_id="row-1", token_id="tok-held", row_data=make_row({}), branch_name="path_a")
        coalesce = Mock(spec=CoalesceExecutor)
        # The in-claim notify (retained at N=1) fails the group immediately.
        coalesce.notify_branch_lost.return_value = CoalesceOutcome(
            held=False,
            failure_reason="branch_lost:path_b",
            consumed_tokens=(held_token,),
            outcomes_recorded=True,
        )
        db, factory = _make_factory()
        processor = self._forked_processor(factory, coalesce)
        ctx = make_context(landscape=factory.plugin_audit_writer())
        # Sibling A is durably held (post-adoption image).
        _persist_blocked_scheduler_work(
            factory, processor, held_token, node_id=NodeID("coalesce::merge"), barrier_key="merge", adopted=True
        )

        # Branch B's claim ends in a lossy FAILURE disposition.
        losing_token = TokenInfo(row_id="row-1", token_id="tok-lost", row_data=make_row({}), branch_name="path_b")
        sibling_results = processor._notify_coalesce_of_lost_branch(losing_token, "quarantined:boom", [])
        assert len(sibling_results) == 1  # must-fail within the same drain step
        assert sibling_results[0].token.token_id == "tok-held"

        # The staged loss rides the claim's own disposition transaction.
        spec = processor._take_claim_branch_loss("tok-lost")
        assert spec is not None
        assert (spec.coalesce_name, spec.row_id, spec.branch_name, spec.token_id) == ("merge", "row-1", "path_b", "tok-lost")
        assert spec.recorded_by == processor._scheduler_lease_owner

        # Drive the disposition the drain would issue and prove atomic commit.
        from tests.unit.engine.test_processor import _persist_token_for_scheduler

        _persist_token_for_scheduler(factory, losing_token, ingest_sequence=0)
        item = factory.scheduler.enqueue_ready_claimed(
            run_id=processor.run_id,
            token_id="tok-lost",
            row_id="row-1",
            node_id="coalesce::merge",
            step_index=2,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(losing_token.row_data),
            available_at=processor._clock.now_utc(),
            lease_owner="test-harness",
            lease_seconds=60,
            now=processor._clock.now_utc(),
        )
        factory.scheduler.mark_failed(
            work_item_id=item.work_item_id,
            now=processor._clock.now_utc(),
            expected_lease_owner="test-harness",
            branch_loss=spec,
        )
        with db.connection() as conn:
            loss_rows = conn.execute(select(coalesce_branch_losses_table)).mappings().all()
        assert len(loss_rows) == 1
        assert loss_rows[0]["branch_name"] == "path_b"
        assert loss_rows[0]["adopted_epoch"] is None  # intake-pending replay cursor

        # The next intake marks the loss adopted; the in-memory replay dedups
        # via has_recorded_branch_loss (record-then-notify already ran).
        coalesce.has_recorded_branch_loss.return_value = True
        results, child_items = processor._run_barrier_intake_pass(ctx)
        assert results == [] and child_items == []
        coalesce.notify_branch_lost.assert_called_once()  # the in-claim call only
        with db.connection() as conn:
            adopted_epoch = conn.execute(select(coalesce_branch_losses_table.c.adopted_epoch)).scalar_one()
        assert adopted_epoch == 1

    def test_loss_survives_takeover_via_restore_ledger_seed(self) -> None:
        """§E.4/§E.5: the new leader rebuilds lost_branches from the durable ledger."""
        _db, factory = _make_factory()
        bootstrap = self._forked_processor(factory, Mock(spec=CoalesceExecutor))
        # Durable image left by the dead leader: branch A held+adopted at the
        # coalesce, branch B's loss recorded with its disposition.
        held_token = TokenInfo(row_id="row-1", token_id="tok-held", row_data=make_row({}), branch_name="path_a")
        _persist_blocked_scheduler_work(
            factory, bootstrap, held_token, node_id=NodeID("coalesce::merge"), barrier_key="merge", adopted=True
        )
        # Simulate the OPEN node_state hold that accept() writes (§E.2: the
        # adoption CAS committed AND accept() ran before the crash).  Without
        # this the restore reconcile classifies the row as a crash-window
        # holdless item and resets it to intake-pending instead of restoring it.
        factory.execution.begin_node_state(
            token_id="tok-held",
            node_id="coalesce::merge",
            run_id="test-run",
            step_index=2,
            input_data={},
            attempt=0,
            resume_checkpoint_id=None,
        )
        losing_token = TokenInfo(row_id="row-1", token_id="tok-lost", row_data=make_row({}), branch_name="path_b")
        from tests.unit.engine.test_processor import _persist_token_for_scheduler

        _persist_token_for_scheduler(factory, losing_token, ingest_sequence=0)
        item = factory.scheduler.enqueue_ready_claimed(
            run_id=bootstrap.run_id,
            token_id="tok-lost",
            row_id="row-1",
            node_id="coalesce::merge",
            step_index=2,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(losing_token.row_data),
            available_at=bootstrap._clock.now_utc(),
            lease_owner="dead-leader",
            lease_seconds=60,
            now=bootstrap._clock.now_utc(),
        )
        from elspeth.core.landscape.scheduler_repository import BranchLossSpec

        factory.scheduler.mark_failed(
            work_item_id=item.work_item_id,
            now=bootstrap._clock.now_utc(),
            expected_lease_owner="dead-leader",
            branch_loss=BranchLossSpec(
                coalesce_name="merge",
                row_id="row-1",
                branch_name="path_b",
                token_id="tok-lost",
                reason="quarantined:boom",
                recorded_by="dead-leader",
            ),
        )

        takeover_coalesce = Mock(spec=CoalesceExecutor)
        self._forked_processor(
            factory,
            takeover_coalesce,
            barrier_restore=BarrierJournalRestoreContext(
                resume_checkpoint_id="ckpt-takeover",
                barrier_scalars=None,
                batch_id_remap={},
            ),
        )
        # restore_from_journal received the ledger-seeded scalars for the
        # still-pending key: the loss did NOT die with the dead leader.
        restore_call = takeover_coalesce.restore_from_journal.call_args
        assert restore_call is not None
        seeded = restore_call.kwargs["scalars"]
        assert ("merge", "row-1") in seeded
        assert dict(seeded[("merge", "row-1")].lost_branches) == {"path_b": "quarantined:boom"}
        assert [i.token_id for i in restore_call.kwargs["items"]] == ["tok-held"]


class TestEofGating:
    """§H 480 / §D steps 2-3 — journal quiescence gates the final flush."""

    def test_eof_flush_refused_while_journal_has_in_flight_work(self) -> None:
        _db, factory = _make_factory()
        transform = _passthrough_flush_transform()
        processor = _agg_processor(factory, trigger={"count": 50}, transform=transform)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        # A slow worker's in-flight row: deposited READY, never claimed.
        slow_token = make_token_info(row_id="row-slow", token_id="tok-slow", data={"value": 9})
        from tests.unit.engine.test_processor import _persist_token_for_scheduler

        _persist_token_for_scheduler(factory, slow_token, ingest_sequence=3)
        factory.scheduler.enqueue_ready(
            run_id=processor.run_id,
            token_id="tok-slow",
            row_id="row-slow",
            node_id=str(AGG_NODE),
            step_index=1,
            ingest_sequence=3,
            row_payload_json=factory.scheduler.serialize_row_payload(slow_token.row_data),
            available_at=processor._clock.now_utc(),
        )

        config = MagicMock(spec=PipelineConfig)
        config.aggregation_settings = {str(AGG_NODE): MagicMock(spec=AggregationSettings)}
        with pytest.raises(OrchestrationInvariantError, match="End-of-input barrier flush refused"):
            run_end_of_input_barrier_flush(
                config=config,
                processor=processor,
                ctx=ctx,
                counters=ExecutionCounters(),
                pending_tokens={},
                coalesce_executor=None,
                coalesce_node_map={},
            )

    def test_in_flight_row_joins_the_single_final_batch(self) -> None:
        """§D step 3: once the journal quiesces, EOF flushes exactly ONE batch
        containing every member — the slow row is not split into its own batch."""
        _db, factory = _make_factory()
        transform = _passthrough_flush_transform()
        processor = _agg_processor(factory, trigger={"count": 50}, transform=transform)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        for index in range(2):
            results = processor.process_row(
                row_index=index,
                source_row=_make_source_row({"value": index}),
                transforms=[transform],
                ctx=ctx,
                source_row_index=index,
                ingest_sequence=index,
            )
            assert [(r.outcome, r.path) for r in results] == [(None, TerminalPath.BUFFERED)]

        # Journal quiesced; both arrivals adopted into ONE in-progress batch.
        assert processor.count_unquiesced_scheduler_work() == 0
        assert processor.get_aggregation_buffer_count(AGG_NODE) == 2

        flush_results, child_items = processor.handle_timeout_flush(
            node_id=AGG_NODE,
            transform=transform,
            ctx=ctx,
            trigger_type=TriggerType.END_OF_SOURCE,
        )
        assert child_items == []
        assert len(flush_results) == 2
        assert all(r.scheduler_pending_sink for r in flush_results)

        batches = factory.execution.get_batches(processor.run_id)
        completed = [batch for batch in batches if batch.status.value == "completed"]
        assert len(completed) == 1, "the in-flight arrivals join a SINGLE final batch"
        member_ids = {m.token_id for m in factory.execution.get_batch_members(completed[0].batch_id)}
        assert len(member_ids) == 2
        assert processor.has_blocked_barrier_work() is False


class TestIntakeFiredCountFlush:
    """§E.2 owned change — the count trigger fires from the intake step."""

    def test_count_flush_fires_in_the_triggering_arrivals_drain(self) -> None:
        _db, factory = _make_factory()
        transform = _passthrough_flush_transform()
        processor = _agg_processor(factory, trigger={"count": 2}, transform=transform)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        first = processor.process_row(
            row_index=0,
            source_row=_make_source_row({"value": 1}),
            transforms=[transform],
            ctx=ctx,
            source_row_index=0,
            ingest_sequence=0,
        )
        assert [(r.outcome, r.path) for r in first] == [(None, TerminalPath.BUFFERED)]

        second = processor.process_row(
            row_index=1,
            source_row=_make_source_row({"value": 2}),
            transforms=[transform],
            ctx=ctx,
            source_row_index=1,
            ingest_sequence=1,
        )
        # The triggering arrival surfaces as BUFFERED; the flush fires from
        # the NEXT drain iteration's intake — still within the same
        # process_row call (one extra BLOCKED→consumed transition, §E.2).
        outcomes = [(r.outcome, r.path) for r in second]
        assert outcomes[0] == (None, TerminalPath.BUFFERED)
        sink_bound = [r for r in second if r.path is TerminalPath.DEFAULT_FLOW]
        assert len(sink_bound) == 2, "terminal passthrough flush emits both members"
        assert all(r.scheduler_pending_sink for r in sink_bound)

        # Every member — the trigger arrival included — was a consumed BLOCKED
        # row; nothing rode a claim.
        with _db.connection() as conn:
            statuses = (
                conn.execute(select(token_work_items_table.c.status).where(token_work_items_table.c.barrier_key == str(AGG_NODE)))
                .scalars()
                .all()
            )
        # Passthrough terminal flush hands both members BLOCKED -> PENDING_SINK
        # in the ONE atomic completion; post-sink terminalization runs later.
        assert sorted(statuses) == [TokenWorkStatus.PENDING_SINK.value, TokenWorkStatus.PENDING_SINK.value]
