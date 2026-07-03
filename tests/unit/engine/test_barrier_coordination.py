"""BarrierIntakeCoordinator boundary tests (elspeth-e76a186916).

Barrier adoption used to be choreography spread across RowProcessor,
the scheduler repository, and the aggregation/coalesce executors, with the
crash-window ordering (open batch -> fenced adopt -> feed memory -> evaluate
trigger) preserved only by caller convention and docstring prose. The
coordinator owns that ordered sequence behind one intake contract that
returns typed dispositions.

These tests pin the boundary with hand-rolled recording fakes:

* the disposition taxonomy (held / terminal / pending-sink /
  ready-continuation / flush-fired);
* the ordering invariants — batch membership opens BEFORE the fenced
  adoption, executor memory is fed ONLY on the adopted=True arm, and the
  aggregation trigger is evaluated from the same intake step as the
  triggering arrival's adoption;
* fail-closed on orphan barrier keys.

Behavioral (repository-backed) coverage lives in
tests/unit/engine/test_adr030_slice3_intake.py and
tests/integration/pipeline/test_barrier_intake_dispositions.py — this file
is the coordinator-level contract net.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from elspeth.contracts import TokenInfo, TransformProtocol
from elspeth.contracts.enums import TerminalOutcome, TerminalPath, TriggerType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.results import RowResult
from elspeth.contracts.scheduler import TokenWorkItem, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.engine.barrier_coordination import (
    BarrierIntakeCoordinator,
    BarrierIntakeDispositionKind,
    _LiveBarrierHold,
)
from elspeth.engine.clock import MockClock
from elspeth.engine.coalesce_executor import CoalesceOutcome
from elspeth.engine.dag_navigator import WorkItem

_CONTRACT = SchemaContract(mode="OBSERVED", fields=(), locked=True)
_NOW = datetime(2026, 7, 3, 12, 0, 0, tzinfo=UTC)
_AGG_NODE = NodeID("agg-node")
_COALESCE = CoalesceName("merge")


def _payload() -> str:
    return TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, _CONTRACT))


def _token(token_id: str = "tok-1", row_id: str = "row-1") -> TokenInfo:
    return TokenInfo(row_id=row_id, token_id=token_id, row_data=PipelineRow({"id": 1}, _CONTRACT))


def _blocked_row(*, barrier_key: str, token_id: str = "tok-1", row_id: str = "row-1") -> TokenWorkItem:
    return TokenWorkItem(
        work_item_id=f"wi-{token_id}",
        run_id="run-1",
        token_id=token_id,
        row_id=row_id,
        node_id=str(_AGG_NODE),
        step_index=1,
        ingest_sequence=1,
        row_payload_json=_payload(),
        status=TokenWorkStatus.BLOCKED,
        attempt=1,
        available_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
        barrier_key=barrier_key,
        barrier_blocked_at=_NOW,
        coalesce_name=barrier_key if barrier_key == str(_COALESCE) else None,
    )


class RecordingScheduler:
    """Scheduler fake recording the fenced-verb call sequence."""

    def __init__(self, *, pending: list[TokenWorkItem], adopted: bool = True) -> None:
        self.pending = pending
        self.adopted = adopted
        self.calls: list[str] = []
        self.release_contexts: list[dict[str, object]] = []

    def list_pending_blocked_barrier_items(self, *, run_id: str) -> list[TokenWorkItem]:
        self.calls.append("list_pending")
        return list(self.pending)

    def adopt_blocked_barrier_item(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append("adopt")
        return SimpleNamespace(adopted=self.adopted)

    def mark_blocked_barrier_terminal(self, *, token_ids, release_context=None, **kwargs: object) -> int:
        self.calls.append("release")
        if release_context is not None:
            self.release_contexts.append(dict(release_context))
        return len(tuple(token_ids))

    def list_unadopted_coalesce_branch_losses(self, *, run_id: str) -> list[object]:
        return []

    def adopt_coalesce_branch_losses(self, **kwargs: object) -> None:
        self.calls.append("adopt_losses")


class RecordingAggregationExecutor:
    def __init__(self, *, should_flush: bool = False) -> None:
        self.should_flush = should_flush
        self.calls: list[str] = []
        self.accepted: list[TokenInfo] = []

    def open_batch_membership(self, node_id: NodeID) -> tuple[str, int]:
        self.calls.append("open_batch")
        return ("batch-1", 0)

    def accept_adopted_row(self, node_id: NodeID, token: TokenInfo, *, accept_time: float) -> None:
        self.calls.append("accept")
        self.accepted.append(token)

    def check_flush_status(self, node_id: NodeID) -> tuple[bool, TriggerType | None]:
        self.calls.append("check_flush")
        return (self.should_flush, TriggerType.COUNT if self.should_flush else None)


class RecordingCoalesceExecutor:
    def __init__(self, outcome: CoalesceOutcome) -> None:
        self.outcome = outcome
        self.accepted: list[str] = []

    def accept(self, *, token: TokenInfo, coalesce_name: str, arrival_time: float) -> CoalesceOutcome:
        self.accepted.append(token.token_id)
        return self.outcome

    def has_recorded_branch_loss(self, coalesce_name: str, row_id: str, branch_name: str) -> bool:
        return True

    def notify_branch_lost(self, **kwargs: object) -> CoalesceOutcome | None:
        return None


_DEFAULT_NEXT_NODE = NodeID("after-merge")


class FakeNav:
    def __init__(self, *, next_node: NodeID | None = _DEFAULT_NEXT_NODE, transform: object | None = None) -> None:
        self.next_node = next_node
        self.transform = transform

    def resolve_plugin_for_node(self, node_id: NodeID) -> object | None:
        return self.transform

    def resolve_next_node(self, node_id: NodeID) -> NodeID | None:
        return self.next_node

    def create_work_item(self, **kwargs: object) -> WorkItem:
        return WorkItem(
            token=kwargs["token"],
            current_node_id=kwargs.get("current_node_id"),
            coalesce_node_id=kwargs.get("coalesce_node_id"),
            coalesce_name=kwargs.get("coalesce_name"),
            on_success_sink=kwargs.get("on_success_sink"),
        )


def _batch_aware_transform() -> Mock:
    """Specced protocol mock — satisfies the runtime TransformProtocol check."""
    transform = Mock(spec=TransformProtocol)
    transform.is_batch_aware = True
    return transform


def _make_coordinator(
    *,
    scheduler: RecordingScheduler,
    aggregation_executor: RecordingAggregationExecutor | None = None,
    coalesce_executor: RecordingCoalesceExecutor | None = None,
    nav: FakeNav | None = None,
    live_holds: dict[str, _LiveBarrierHold] | None = None,
    flush_calls: list[tuple[NodeID, TriggerType]] | None = None,
    fire_calls: list[dict[str, object]] | None = None,
) -> BarrierIntakeCoordinator:
    def _flush_batch(node_id: NodeID, transform: object, ctx: object, trigger_type: TriggerType):
        if flush_calls is not None:
            flush_calls.append((node_id, trigger_type))
        flush_token = _token(token_id="tok-flush", row_id="row-flush")
        result = RowResult(
            token=flush_token,
            final_data=flush_token.row_data,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="default",
        )
        return (result,), [WorkItem(token=flush_token, current_node_id=NodeID("after-merge"))]

    def _complete_coalesce_fire(**kwargs: object) -> None:
        if fire_calls is not None:
            fire_calls.append(dict(kwargs))

    def _terminal_coalesce_row_result(token: TokenInfo, coalesce_name: CoalesceName, *, context: str) -> RowResult:
        return RowResult(
            token=token,
            final_data=token.row_data,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.COALESCED,
            sink_name="merged_sink",
        )

    return BarrierIntakeCoordinator(
        run_id="run-1",
        scheduler=scheduler,
        data_flow=SimpleNamespace(record_token_outcome=lambda **kwargs: None),
        execution=SimpleNamespace(get_max_node_state_attempts=lambda run_id, token_ids: {}),
        aggregation_executor=aggregation_executor or RecordingAggregationExecutor(),
        coalesce_executor=coalesce_executor,
        nav=nav or FakeNav(transform=_batch_aware_transform()),
        clock=MockClock(start=100.0),
        aggregation_settings={_AGG_NODE: object()} if aggregation_executor is not None else {},
        coalesce_node_ids={_COALESCE: NodeID("coalesce-node")} if coalesce_executor is not None else {},
        coordination_token=SimpleNamespace(worker_id="leader-1", epoch=1),
        scheduler_lease_owner="leader-1",
        live_barrier_holds=live_holds if live_holds is not None else {},
        resume_checkpoint_id=None,
        flush_batch=_flush_batch,
        complete_coalesce_fire=_complete_coalesce_fire,
        terminal_coalesce_row_result=_terminal_coalesce_row_result,
        emit_token_completed=lambda token, *, outcome, path, sink_name=None: None,
        mark_coalesce_consumed_terminal=lambda *, coalesce_name, consumed_tokens: None,
    )


def _ctx() -> SimpleNamespace:
    return SimpleNamespace()


class TestAggregationIntakeOrdering:
    def test_held_arrival_opens_batch_before_adopt_and_accepts_after(self) -> None:
        row = _blocked_row(barrier_key=str(_AGG_NODE))
        # Share ONE call log between the scheduler and executor fakes so the
        # cross-object ordering (the invariant this ticket moves behind the
        # coordinator boundary) is directly assertable.
        combined: list[str] = []
        scheduler = RecordingScheduler(pending=[row])
        scheduler.calls = combined
        agg = RecordingAggregationExecutor(should_flush=False)
        agg.calls = combined
        holds = {row.token_id: _LiveBarrierHold(token=_token(), barrier_key=str(_AGG_NODE))}
        coordinator = _make_coordinator(scheduler=scheduler, aggregation_executor=agg, live_holds=holds)

        outcome = coordinator.run_intake_pass(_ctx())

        assert [d.kind for d in outcome.dispositions] == [BarrierIntakeDispositionKind.HELD]
        assert outcome.results == []
        assert outcome.child_items == []
        # Ordering by construction: open batch -> fenced adopt -> feed memory
        # -> trigger evaluation, all within one intake step.
        assert combined.index("open_batch") < combined.index("adopt") < combined.index("accept") < combined.index("check_flush")

    def test_idempotent_skip_arm_does_not_feed_memory(self) -> None:
        row = _blocked_row(barrier_key=str(_AGG_NODE))
        scheduler = RecordingScheduler(pending=[row], adopted=False)
        agg = RecordingAggregationExecutor()
        coordinator = _make_coordinator(scheduler=scheduler, aggregation_executor=agg)

        outcome = coordinator.run_intake_pass(_ctx())

        assert outcome.dispositions == ()
        assert "accept" not in agg.calls
        assert "check_flush" not in agg.calls

    def test_count_trigger_fires_flush_in_same_intake_step(self) -> None:
        row = _blocked_row(barrier_key=str(_AGG_NODE))
        scheduler = RecordingScheduler(pending=[row])
        agg = RecordingAggregationExecutor(should_flush=True)
        holds = {row.token_id: _LiveBarrierHold(token=_token(), barrier_key=str(_AGG_NODE))}
        flush_calls: list[tuple[NodeID, TriggerType]] = []
        coordinator = _make_coordinator(
            scheduler=scheduler,
            aggregation_executor=agg,
            live_holds=holds,
            flush_calls=flush_calls,
        )

        outcome = coordinator.run_intake_pass(_ctx())

        assert [d.kind for d in outcome.dispositions] == [BarrierIntakeDispositionKind.FLUSH_FIRED]
        assert flush_calls == [(_AGG_NODE, TriggerType.COUNT)]
        assert len(outcome.results) == 1
        assert len(outcome.child_items) == 1


class TestCoalesceIntakeTaxonomy:
    def test_held_arrival(self) -> None:
        row = _blocked_row(barrier_key=str(_COALESCE))
        scheduler = RecordingScheduler(pending=[row])
        coalesce = RecordingCoalesceExecutor(CoalesceOutcome(held=True))
        holds = {row.token_id: _LiveBarrierHold(token=_token(), barrier_key=str(_COALESCE))}
        coordinator = _make_coordinator(scheduler=scheduler, coalesce_executor=coalesce, live_holds=holds)

        outcome = coordinator.run_intake_pass(_ctx())

        assert [d.kind for d in outcome.dispositions] == [BarrierIntakeDispositionKind.HELD]
        assert coalesce.accepted == [row.token_id]

    def test_late_arrival_releases_row_and_returns_terminal(self) -> None:
        row = _blocked_row(barrier_key=str(_COALESCE))
        scheduler = RecordingScheduler(pending=[row])
        coalesce = RecordingCoalesceExecutor(
            CoalesceOutcome(
                held=False,
                failure_reason="late_arrival_after_merge",
                outcomes_recorded=True,
                late_arrival=True,
            )
        )
        holds = {row.token_id: _LiveBarrierHold(token=_token(), barrier_key=str(_COALESCE))}
        coordinator = _make_coordinator(scheduler=scheduler, coalesce_executor=coalesce, live_holds=holds)

        outcome = coordinator.run_intake_pass(_ctx())

        assert [d.kind for d in outcome.dispositions] == [BarrierIntakeDispositionKind.TERMINAL]
        assert len(outcome.results) == 1
        assert outcome.results[0].outcome is TerminalOutcome.FAILURE
        assert scheduler.release_contexts and scheduler.release_contexts[0]["late_arrival"] is True

    def test_nonterminal_merge_returns_ready_continuation(self) -> None:
        row = _blocked_row(barrier_key=str(_COALESCE))
        scheduler = RecordingScheduler(pending=[row])
        merged = _token(token_id="tok-merged", row_id="row-1")
        consumed = (_token(token_id="tok-a"), _token(token_id="tok-b"))
        coalesce = RecordingCoalesceExecutor(CoalesceOutcome(held=False, merged_token=merged, consumed_tokens=consumed))
        holds = {row.token_id: _LiveBarrierHold(token=_token(), barrier_key=str(_COALESCE))}
        fire_calls: list[dict[str, object]] = []
        coordinator = _make_coordinator(
            scheduler=scheduler,
            coalesce_executor=coalesce,
            live_holds=holds,
            fire_calls=fire_calls,
        )

        outcome = coordinator.run_intake_pass(_ctx())

        assert [d.kind for d in outcome.dispositions] == [BarrierIntakeDispositionKind.READY_CONTINUATION]
        assert len(outcome.child_items) == 1
        assert outcome.child_items[0].token.token_id == "tok-merged"
        assert len(fire_calls) == 1
        assert fire_calls[0]["merged_item"] is outcome.child_items[0]

    def test_terminal_merge_returns_pending_sink(self) -> None:
        row = _blocked_row(barrier_key=str(_COALESCE))
        scheduler = RecordingScheduler(pending=[row])
        merged = _token(token_id="tok-merged", row_id="row-1")
        coalesce = RecordingCoalesceExecutor(CoalesceOutcome(held=False, merged_token=merged, consumed_tokens=(_token(token_id="tok-a"),)))
        holds = {row.token_id: _LiveBarrierHold(token=_token(), barrier_key=str(_COALESCE))}
        fire_calls: list[dict[str, object]] = []
        coordinator = _make_coordinator(
            scheduler=scheduler,
            coalesce_executor=coalesce,
            live_holds=holds,
            nav=FakeNav(next_node=None),
            fire_calls=fire_calls,
        )

        outcome = coordinator.run_intake_pass(_ctx())

        assert [d.kind for d in outcome.dispositions] == [BarrierIntakeDispositionKind.PENDING_SINK]
        assert len(outcome.results) == 1
        assert outcome.results[0].scheduler_pending_sink is True
        assert outcome.child_items == []
        assert len(fire_calls) == 1 and "merged_sink_result" in fire_calls[0]


class TestIntakeFailClosed:
    def test_orphan_barrier_key_raises(self) -> None:
        row = _blocked_row(barrier_key="not-a-barrier")
        scheduler = RecordingScheduler(pending=[row])
        coordinator = _make_coordinator(
            scheduler=scheduler,
            aggregation_executor=RecordingAggregationExecutor(),
        )

        with pytest.raises(AuditIntegrityError, match="orphan barrier_key"):
            coordinator.run_intake_pass(_ctx())
