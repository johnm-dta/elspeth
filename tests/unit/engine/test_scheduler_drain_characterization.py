# tests/unit/engine/test_scheduler_drain_characterization.py
"""Characterization net for the RowProcessor scheduler-drain subsystem.

Pins the CURRENT observable behavior of the durable scheduler claim/drain
loop at the RowProcessor surface BEFORE the SchedulerDrain extraction
(filigree elspeth-c49f33d6e4, component 3). Every test here must pass
unchanged against both the pre-move and post-move trees:

1. Recovery-drain ordering (``drain_scheduled_work``): durable PENDING_SINK
   rows left by a prior crashed worker are recovered — recover_expired_leases
   -> terminalize_pending_sinks_with_terminal_outcomes -> claim_pending_sink —
   BEFORE the first claim_ready, and the reconstructed RowResult carries
   ``scheduler_pending_sink=True`` with the parked outcome/path/sink.
2. Claim -> disposition mapping and fence arguments: a sink-bound result parks
   the claim PENDING_SINK (lease owner kept, membership-fence ``worker_id``
   threaded, returned result tagged); a claimed-token FAILURE marks FAILED;
   a non-sink terminal result marks TERMINAL; unregistered (legacy/N=0)
   builds thread ``worker_id=None`` (unfenced).
3. Clean abandonment: SchedulerLeaseLostError drops the in-flight result, and
   RunWorkerEvictedError propagates unchanged. Both clear staged §E.5 branch
   losses and issue NO disposition write against the abandoned claim.
4. Active-claim heartbeat: between claim and disposition,
   ``_heartbeat_active_claim`` refreshes the claimed lease at most once per
   heartbeat interval; outside an active claim it is a no-op.
5. Maintenance cadence: non-recovery drains run scheduler maintenance every
   SCHEDULER_MAINTENANCE_INTERVAL drains; the FOLLOWER-mode build skips lease
   recovery entirely (ADR-030 §C.3 — the explicit ProcessorMode from
   elspeth-577179bba1, replacing the old triple-None sentinel inference),
   while any LEADER-mode build requires a coordination token before recovery;
   direct pre-coordination repository harnesses use the separately named
   legacy adapter instead of production maintenance.

The observables are durable scheduler rows, returned RowResults, and the
verb order recorded by a delegating scheduler wrapper — not processor
internals — so the net survives the extraction. ``_drain_scheduler_claims``
and ``_heartbeat_active_claim`` are called by their private names on purpose:
those names are load-bearing (orchestrator/follower.py and
test_adr030_loosened_invariant_guard.py call them) and the extraction must
leave delegates behind.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from sqlalchemy import delete, insert, select

from elspeth.contracts import RowResult, TokenInfo
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import OrchestrationInvariantError, RunWorkerEvictedError, SchedulerLeaseLostError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.scheduler import BranchLossSpec, SchedulerEventType, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    run_coordination_events_table,
    run_workers_table,
    scheduler_events_table,
    token_parents_table,
    token_work_items_table,
)
from elspeth.engine.clock import MockClock
from elspeth.engine.processor import SCHEDULER_MAINTENANCE_INTERVAL, DAGTraversalContext, RowProcessor
from elspeth.engine.scheduler_drain import ProcessorMode
from elspeth.engine.spans import SpanFactory
from elspeth.engine.work_items import WorkItem
from tests.fixtures.landscape import RecorderSetup, leader_coordination_token, make_recorder_with_run, register_test_node

NODE_ID = "normalize"
LEADER_OWNER = "leader-a"
CRASHED_OWNER = "crashed-worker"

_CONTRACT = SchemaContract(mode="OBSERVED", fields=(), locked=True)
_PAYLOAD = TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, _CONTRACT))


class _RecordingScheduler:
    """Delegating wrapper over the real scheduler repository.

    Records ``(verb, kwargs)`` for the drain-relevant verbs in invocation
    order, then delegates to the real repository — the durable behavior is
    exactly the real repository's. Everything else passes straight through.
    """

    _RECORDED = frozenset(
        {
            "recover_expired_leases",
            "terminalize_pending_sinks_with_terminal_outcomes",
            "claim_pending_sink",
            "claim_ready",
            "mark_pending_sink",
            "mark_failed",
            "mark_terminal",
            "mark_blocked",
            "heartbeat_lease",
            "enqueue_ready_claimed",
            "enqueue_ready_claimed_legacy_unfenced",
        }
    )

    def __init__(self, inner: TokenSchedulerRepository) -> None:
        self._inner = inner
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name in self._RECORDED:

            def recorded(*args: Any, _attr: Any = attr, _name: str = name, **kwargs: Any) -> Any:
                self.calls.append((_name, dict(kwargs)))
                return _attr(*args, **kwargs)

            return recorded
        return attr

    def verbs(self) -> list[str]:
        return [name for name, _ in self.calls]

    def calls_for(self, verb: str) -> list[dict[str, Any]]:
        return [kwargs for name, kwargs in self.calls if name == verb]


def _build(
    *,
    lease_owner: str | None,
    register_leader: str | None = LEADER_OWNER,
    bind_leader_token: bool = False,
    heartbeat_seconds: int = 60,
    mode: ProcessorMode = ProcessorMode.LEADER,
) -> tuple[RowProcessor, _RecordingScheduler, RecorderSetup, MockClock]:
    """Real RowProcessor over a real scheduler DB behind the recording wrapper."""
    setup = make_recorder_with_run(
        run_id="run-drain-char",
        source_node_id="source-1",
        leader_worker_id=register_leader,
    )
    register_test_node(setup.data_flow, setup.run_id, NODE_ID)
    clock = MockClock(start=1_750_000_000.0)
    spy = _RecordingScheduler(setup.factory.scheduler)
    processor = RowProcessor(
        execution=setup.execution,
        data_flow=setup.data_flow,
        span_factory=SpanFactory(),
        run_id=setup.run_id,
        source_node_id=NodeID(setup.source_node_id),
        source_on_success="default",
        traversal=DAGTraversalContext(
            # A real step for NODE_ID so resolve_sink_step() (used by the
            # pending-sink reconstruction) resolves to max(steps) + 1.
            node_step_map={NodeID(setup.source_node_id): 0, NodeID(NODE_ID): 1},
            node_to_plugin={},
            node_to_next={},
            coalesce_node_map={},
        ),
        scheduler=cast(TokenSchedulerRepository, spy),
        scheduler_lease_owner=lease_owner,
        scheduler_heartbeat_seconds=heartbeat_seconds,
        coordination_token=(leader_coordination_token(setup.factory, setup.run_id) if bind_leader_token else None),
        clock=clock,
        mode=mode,
    )
    return processor, spy, setup, clock


def _ctx(setup: RecorderSetup) -> PluginContext:
    return PluginContext(run_id=setup.run_id, config={}, landscape=None)


def _register_worker(setup: RecorderSetup, worker_id: str) -> None:
    """Register a run_workers identity so the membership fence admits its claims."""
    registered_at = datetime.now(UTC)
    with setup.db.engine.begin() as conn:
        conn.execute(
            insert(run_workers_table).values(
                worker_id=worker_id,
                run_id=setup.run_id,
                role="follower",
                status="active",
                registered_at=registered_at,
                heartbeat_expires_at=registered_at + timedelta(hours=1),
            )
        )


def _enqueue_ready(
    setup: RecorderSetup,
    scheduler: _RecordingScheduler,
    clock: MockClock,
    *,
    sequence: int,
) -> tuple[str, TokenInfo]:
    """Enqueue a READY continuation; return (work_item_id, token)."""
    row, token = setup.data_flow.create_row_with_token(
        run_id=setup.run_id,
        source_node_id=setup.source_node_id,
        row_index=sequence,
        data={"id": sequence},
        source_row_index=sequence,
        ingest_sequence=sequence,
    )
    item = scheduler.enqueue_ready(
        run_id=setup.run_id,
        token_id=token.token_id,
        row_id=row.row_id,
        node_id=NODE_ID,
        step_index=1,
        ingest_sequence=sequence,
        available_at=clock.now_utc(),
        row_payload_json=_PAYLOAD,
    )
    token_info = TokenInfo(row_id=row.row_id, token_id=token.token_id, row_data=PipelineRow({"id": sequence}, _CONTRACT))
    return item.work_item_id, token_info


def _unscheduled_work_item(setup: RecorderSetup, *, sequence: int) -> WorkItem:
    row, token = setup.data_flow.create_row_with_token(
        run_id=setup.run_id,
        source_node_id=setup.source_node_id,
        row_index=sequence,
        data={"id": sequence},
        source_row_index=sequence,
        ingest_sequence=sequence,
    )
    return WorkItem(
        token=TokenInfo(row_id=row.row_id, token_id=token.token_id, row_data=PipelineRow({"id": sequence}, _CONTRACT)),
        current_node_id=NodeID(NODE_ID),
    )


def _park_pending_sink(
    setup: RecorderSetup,
    scheduler: _RecordingScheduler,
    clock: MockClock,
    *,
    sequence: int,
    owner: str = CRASHED_OWNER,
) -> tuple[str, TokenInfo]:
    """Claim a READY row under ``owner`` and park it PENDING_SINK (crash image)."""
    work_item_id, token = _enqueue_ready(setup, scheduler, clock, sequence=sequence)
    claimed = scheduler.claim_ready(run_id=setup.run_id, lease_owner=owner, lease_seconds=300, now=clock.now_utc())
    assert claimed is not None and claimed.work_item_id == work_item_id
    scheduler.mark_pending_sink(
        work_item_id=work_item_id,
        row_payload_json=_PAYLOAD,
        sink_name="default",
        outcome=TerminalOutcome.SUCCESS.value,
        path=TerminalPath.DEFAULT_FLOW.value,
        error_hash=None,
        error_message=None,
        now=clock.now_utc(),
        expected_lease_owner=owner,
    )
    return work_item_id, token


def _row_status(setup: RecorderSetup, work_item_id: str) -> tuple[str, str | None]:
    with setup.db.engine.connect() as conn:
        return conn.execute(  # type: ignore[return-value]
            select(token_work_items_table.c.status, token_work_items_table.c.lease_owner).where(
                token_work_items_table.c.work_item_id == work_item_id
            )
        ).one()


def _durable_claim_image(
    setup: RecorderSetup,
    work_item_id: str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Item plus both event planes a clean-abandon path must not mutate."""
    with setup.db.engine.connect() as conn:
        item = dict(
            conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id)).mappings().one()
        )
        scheduler_events = tuple(
            dict(row)
            for row in conn.execute(
                select(scheduler_events_table)
                .where(scheduler_events_table.c.run_id == setup.run_id)
                .order_by(scheduler_events_table.c.event_id)
            ).mappings()
        )
        coordination_events = tuple(
            dict(row)
            for row in conn.execute(
                select(run_coordination_events_table)
                .where(run_coordination_events_table.c.run_id == setup.run_id)
                .order_by(run_coordination_events_table.c.event_id)
            ).mappings()
        )
    return item, scheduler_events, coordination_events


def _sink_bound_result(token: TokenInfo) -> RowResult:
    return RowResult(
        token=token,
        final_data=token.row_data,
        outcome=TerminalOutcome.SUCCESS,
        path=TerminalPath.DEFAULT_FLOW,
        sink_name="default",
    )


def _dropped_result(token: TokenInfo) -> RowResult:
    """Non-sink terminal result: drives the mark_terminal disposition arm."""
    return RowResult(token=token, final_data=token.row_data, outcome=TerminalOutcome.SUCCESS, path=TerminalPath.FILTER_DROPPED)


def _failed_result(token: TokenInfo) -> RowResult:
    """Claimed-token FAILURE result: drives the mark_failed disposition arm."""
    return RowResult(token=token, final_data=token.row_data, outcome=TerminalOutcome.FAILURE, path=TerminalPath.UNROUTED)


def test_recovery_drain_recovers_parked_pending_sinks_before_claiming_ready() -> None:
    """drain_scheduled_work: recover -> terminalize -> claim_pending_sink run
    BEFORE the first claim_ready; the reconstructed result is pending-sink
    tagged and the row is re-leased (NOT terminalized) under this worker."""
    processor, spy, setup, clock = _build(lease_owner=LEADER_OWNER, bind_leader_token=True)
    _register_worker(setup, CRASHED_OWNER)
    parked_item_id, parked_token = _park_pending_sink(setup, spy, clock, sequence=0)
    ready_item_id, ready_token = _enqueue_ready(setup, spy, clock, sequence=1)

    def fake_process(**kwargs: Any) -> tuple[RowResult, list[Any]]:
        return _dropped_result(kwargs["token"]), []

    spy.calls.clear()
    with patch.object(processor, "_process_single_token", new=fake_process):
        results = processor.drain_scheduled_work(_ctx(setup))

    verbs = spy.verbs()
    first_claim_ready = verbs.index("claim_ready")
    first_pending_claim = verbs.index("claim_pending_sink")
    assert first_pending_claim < first_claim_ready, "pending-sink recovery must run before the claim_ready loop"
    assert verbs.index("recover_expired_leases") < first_pending_claim
    assert verbs.index("terminalize_pending_sinks_with_terminal_outcomes") < first_pending_claim

    by_token = {result.token.token_id: result for result in results}
    recovered = by_token[parked_token.token_id]
    assert recovered.scheduler_pending_sink is True
    assert recovered.outcome is TerminalOutcome.SUCCESS
    assert recovered.path is TerminalPath.DEFAULT_FLOW
    assert recovered.sink_name == "default"
    assert recovered.error is None

    # Re-leased under THIS worker; terminalization is deferred to the
    # post-sink callback (mark_sink_bound_scheduler_terminal), not the drain.
    status, owner = _row_status(setup, parked_item_id)
    assert status == TokenWorkStatus.LEASED.value
    assert owner == LEADER_OWNER

    # The READY row was processed and terminalized in the same drain.
    assert by_token[ready_token.token_id].path is TerminalPath.FILTER_DROPPED
    status, _owner = _row_status(setup, ready_item_id)
    assert status == TokenWorkStatus.TERMINAL.value


def test_recovery_drain_requires_token_before_lease_recovery() -> None:
    """A tokenless leader drain refuses before calling strict recovery."""
    processor, spy, setup, _clock = _build(lease_owner=None, register_leader=None)

    spy.calls.clear()
    with pytest.raises(OrchestrationInvariantError, match="coordination token"):
        processor.drain_scheduled_work(_ctx(setup))

    assert spy.calls_for("recover_expired_leases") == []


def test_sink_bound_result_parks_pending_sink_with_fenced_owner_and_tags_result() -> None:
    """A sink-bound success parks the claim PENDING_SINK: lease owner kept,
    membership fence worker_id threaded, parked metadata mirrors the result,
    and the returned result is tagged scheduler_pending_sink=True."""
    processor, spy, setup, clock = _build(lease_owner=LEADER_OWNER)
    work_item_id, _token = _enqueue_ready(setup, spy, clock, sequence=0)

    def fake_process(**kwargs: Any) -> tuple[RowResult, list[Any]]:
        return _sink_bound_result(kwargs["token"]), []

    spy.calls.clear()
    with patch.object(processor, "_process_single_token", new=fake_process):
        results = processor._drain_scheduler_claims(ctx=_ctx(setup), pending_items={}, recover_pending_sinks=False)

    assert len(results) == 1
    assert results[0].scheduler_pending_sink is True

    parks = spy.calls_for("mark_pending_sink")
    assert len(parks) == 1
    park = parks[0]
    assert park["work_item_id"] == work_item_id
    assert park["expected_lease_owner"] == LEADER_OWNER
    assert park["worker_id"] == LEADER_OWNER, "registered lease owner must thread the membership fence"
    assert park["sink_name"] == "default"
    assert park["outcome"] == TerminalOutcome.SUCCESS.value
    assert park["path"] == TerminalPath.DEFAULT_FLOW.value
    assert park["error_hash"] is None
    assert park["error_message"] is None

    assert spy.calls_for("mark_terminal") == []
    assert spy.calls_for("mark_failed") == []
    status, owner = _row_status(setup, work_item_id)
    assert status == TokenWorkStatus.PENDING_SINK.value
    assert owner == LEADER_OWNER, "mark_pending_sink keeps the claimant's lease_owner"


def test_claimed_token_failure_marks_failed_with_fence() -> None:
    """A claimed-token FAILURE (non-sink-bound) drives mark_failed with the
    proven lease owner and the membership-fence worker_id."""
    processor, spy, setup, clock = _build(lease_owner=LEADER_OWNER)
    work_item_id, _token = _enqueue_ready(setup, spy, clock, sequence=0)

    def fake_process(**kwargs: Any) -> tuple[RowResult, list[Any]]:
        return _failed_result(kwargs["token"]), []

    spy.calls.clear()
    with patch.object(processor, "_process_single_token", new=fake_process):
        results = processor._drain_scheduler_claims(ctx=_ctx(setup), pending_items={}, recover_pending_sinks=False)

    assert len(results) == 1
    assert results[0].outcome is TerminalOutcome.FAILURE

    fails = spy.calls_for("mark_failed")
    assert len(fails) == 1
    assert fails[0]["work_item_id"] == work_item_id
    assert fails[0]["expected_lease_owner"] == LEADER_OWNER
    assert fails[0]["worker_id"] == LEADER_OWNER
    assert spy.calls_for("mark_pending_sink") == []
    assert spy.calls_for("mark_terminal") == []
    status, _owner = _row_status(setup, work_item_id)
    assert status == TokenWorkStatus.FAILED.value


def test_non_sink_terminal_marks_terminal_and_unregistered_build_is_unfenced() -> None:
    """A non-sink terminal result drives mark_terminal; a build whose
    constructor was given NO lease owner threads worker_id=None — the
    unfenced disposition arm.

    The auto-minted owner is registered in run_workers ONLY so the claim
    fence (always active in these fixtures: begin_run registers a leader
    unconditionally) admits the claim; the pin is that the DRAIN still
    threads worker_id=None because the owner was not passed at construction.
    """
    processor, spy, setup, clock = _build(lease_owner=None, register_leader=None)
    _register_worker(setup, processor._scheduler_lease_owner)
    work_item_id, _token = _enqueue_ready(setup, spy, clock, sequence=0)

    def fake_process(**kwargs: Any) -> tuple[RowResult, list[Any]]:
        return _dropped_result(kwargs["token"]), []

    spy.calls.clear()
    with patch.object(processor, "_process_single_token", new=fake_process):
        results = processor._drain_scheduler_claims(ctx=_ctx(setup), pending_items={}, recover_pending_sinks=False)

    assert len(results) == 1
    terms = spy.calls_for("mark_terminal")
    assert len(terms) == 1
    assert terms[0]["work_item_id"] == work_item_id
    # The auto-minted owner is the proven lease owner; the fence identity is
    # None because the owner was never registered in run_workers.
    assert terms[0]["expected_lease_owner"] == processor._scheduler_lease_owner
    assert terms[0]["worker_id"] is None
    status, _owner = _row_status(setup, work_item_id)
    assert status == TokenWorkStatus.TERMINAL.value


def test_child_ready_enqueue_rolls_back_when_parent_terminal_event_fails() -> None:
    """TS-00 children and the parent TS-08 disposition are one commit.

    A process can die at any statement boundary.  Model the durable failure
    point by rejecting the parent's scheduler event after its row UPDATE.  No
    child READY row or ENQUEUE event may survive while the parent lease stays
    reprocessable.
    """
    processor, spy, setup, clock = _build(lease_owner=LEADER_OWNER)
    parent_work_item_id, _parent_token = _enqueue_ready(setup, spy, clock, sequence=0)
    child_item = _unscheduled_work_item(setup, sequence=1)

    def produce_child(**kwargs: Any) -> tuple[RowResult, list[WorkItem]]:
        return _dropped_result(kwargs["token"]), [child_item]

    real_record = setup.factory.scheduler.events.record

    def reject_parent_terminal_event(conn: Any, *, event_type: SchedulerEventType, **kwargs: Any) -> None:
        if event_type is SchedulerEventType.MARK_TERMINAL:
            raise RuntimeError("injected parent terminal event failure")
        real_record(conn, event_type=event_type, **kwargs)

    with (
        patch.object(processor, "_process_single_token", new=produce_child),
        patch.object(setup.factory.scheduler.events, "record", new=reject_parent_terminal_event),
        pytest.raises(RuntimeError, match="injected parent terminal event failure"),
    ):
        processor._drain_scheduler_claims(ctx=_ctx(setup), pending_items={}, recover_pending_sinks=False)

    with setup.db.engine.connect() as conn:
        parent_status = conn.execute(
            select(token_work_items_table.c.status).where(token_work_items_table.c.work_item_id == parent_work_item_id)
        ).scalar_one()
        child_rows = conn.execute(
            select(token_work_items_table.c.work_item_id).where(
                token_work_items_table.c.run_id == setup.run_id,
                token_work_items_table.c.token_id == child_item.token.token_id,
            )
        ).all()
        child_events = conn.execute(
            select(scheduler_events_table.c.event_id).where(
                scheduler_events_table.c.run_id == setup.run_id,
                scheduler_events_table.c.token_id == child_item.token.token_id,
            )
        ).all()

    assert parent_status == TokenWorkStatus.LEASED.value
    assert child_rows == []
    assert child_events == []


def test_child_and_parent_disposition_roll_back_when_branch_loss_write_fails() -> None:
    """The child, parent event, and §E.5 loss share one commit boundary."""
    processor, spy, setup, clock = _build(lease_owner=LEADER_OWNER)
    parent_work_item_id, parent_token = _enqueue_ready(setup, spy, clock, sequence=0)
    child_item = _unscheduled_work_item(setup, sequence=1)
    processor._pending_branch_losses.append(
        BranchLossSpec(
            coalesce_name="merge",
            row_id=parent_token.row_id,
            branch_name="left",
            token_id=parent_token.token_id,
            reason="test branch loss",
            recorded_by=LEADER_OWNER,
        )
    )

    def produce_child(**kwargs: Any) -> tuple[RowResult, list[WorkItem]]:
        return _dropped_result(kwargs["token"]), [child_item]

    with (
        patch.object(processor, "_process_single_token", new=produce_child),
        patch(
            "elspeth.core.landscape.scheduler.dispositions.record_coalesce_branch_loss",
            side_effect=RuntimeError("injected branch-loss write failure"),
        ),
        pytest.raises(RuntimeError, match="injected branch-loss write failure"),
    ):
        processor._drain_scheduler_claims(ctx=_ctx(setup), pending_items={}, recover_pending_sinks=False)

    with setup.db.engine.connect() as conn:
        parent_status = conn.execute(
            select(token_work_items_table.c.status).where(token_work_items_table.c.work_item_id == parent_work_item_id)
        ).scalar_one()
        child_rows = conn.execute(
            select(token_work_items_table.c.work_item_id).where(token_work_items_table.c.token_id == child_item.token.token_id)
        ).all()
        parent_terminal_events = conn.execute(
            select(scheduler_events_table.c.event_id).where(
                scheduler_events_table.c.token_id == parent_token.token_id,
                scheduler_events_table.c.event_type == SchedulerEventType.MARK_TERMINAL.value,
            )
        ).all()

    assert parent_status == TokenWorkStatus.LEASED.value
    assert child_rows == []
    assert parent_terminal_events == []


@pytest.mark.parametrize(
    ("parent_result", "rejected_event"),
    [
        pytest.param(_failed_result, SchedulerEventType.MARK_FAILED, id="failed"),
        pytest.param(_sink_bound_result, SchedulerEventType.MARK_PENDING_SINK, id="pending-sink"),
    ],
)
def test_child_ready_enqueue_rolls_back_when_nonterminal_parent_event_fails(
    parent_result: Any,
    rejected_event: SchedulerEventType,
) -> None:
    """TS-00 children share the TS-09/TS-10 parent commit boundary."""
    processor, spy, setup, clock = _build(lease_owner=LEADER_OWNER)
    parent_work_item_id, _parent_token = _enqueue_ready(setup, spy, clock, sequence=0)
    child_item = _unscheduled_work_item(setup, sequence=1)

    def produce_child(**kwargs: Any) -> tuple[RowResult, list[WorkItem]]:
        return parent_result(kwargs["token"]), [child_item]

    real_record = setup.factory.scheduler.events.record

    def reject_parent_event(conn: Any, *, event_type: SchedulerEventType, **kwargs: Any) -> None:
        if event_type is rejected_event:
            raise RuntimeError("injected parent disposition event failure")
        real_record(conn, event_type=event_type, **kwargs)

    with (
        patch.object(processor, "_process_single_token", new=produce_child),
        patch.object(setup.factory.scheduler.events, "record", new=reject_parent_event),
        pytest.raises(RuntimeError, match="injected parent disposition event failure"),
    ):
        processor._drain_scheduler_claims(ctx=_ctx(setup), pending_items={}, recover_pending_sinks=False)

    with setup.db.engine.connect() as conn:
        parent_status = conn.execute(
            select(token_work_items_table.c.status).where(token_work_items_table.c.work_item_id == parent_work_item_id)
        ).scalar_one()
        child_rows = conn.execute(
            select(token_work_items_table.c.work_item_id).where(
                token_work_items_table.c.run_id == setup.run_id,
                token_work_items_table.c.token_id == child_item.token.token_id,
            )
        ).all()
        child_events = conn.execute(
            select(scheduler_events_table.c.event_id).where(
                scheduler_events_table.c.run_id == setup.run_id,
                scheduler_events_table.c.token_id == child_item.token.token_id,
            )
        ).all()

    assert parent_status == TokenWorkStatus.LEASED.value
    assert child_rows == []
    assert child_events == []


def test_expansion_restart_reuses_children_and_delivers_each_once() -> None:
    """Crash/restart proof for the production expansion and scheduler seams."""
    processor, spy, setup, clock = _build(lease_owner=LEADER_OWNER, bind_leader_token=True)
    parent_work_item_id, parent_token = _enqueue_ready(setup, spy, clock, sequence=1)
    parent_calls = 0
    child_calls: dict[str, int] = {}
    expansion_ids_by_attempt: list[tuple[str, ...]] = []

    def expand_parent_or_finish_child(**kwargs: Any) -> tuple[RowResult, list[WorkItem]]:
        nonlocal parent_calls
        token = cast(TokenInfo, kwargs["token"])
        if token.token_id != parent_token.token_id:
            child_calls[token.token_id] = child_calls.get(token.token_id, 0) + 1
            return _dropped_result(token), []

        parent_calls += 1
        payloads = ({"item": 1}, {"item": 2})
        children, _expand_group_id = setup.data_flow.expand_token(
            parent_ref=TokenRef(token_id=token.token_id, run_id=setup.run_id),
            row_id=token.row_id,
            child_payloads=payloads,
            output_contract=_CONTRACT,
            step_in_pipeline=1,
        )
        expansion_ids_by_attempt.append(tuple(child.token_id for child in children))
        child_items = [
            WorkItem(
                token=TokenInfo(
                    row_id=child.row_id,
                    token_id=child.token_id,
                    row_data=PipelineRow(payload, _CONTRACT),
                    expand_group_id=child.expand_group_id,
                ),
                current_node_id=NodeID(NODE_ID),
            )
            for child, payload in zip(children, payloads, strict=True)
        ]
        return _dropped_result(token), child_items

    real_record = setup.factory.scheduler.events.record

    def reject_first_parent_event(conn: Any, *, event_type: SchedulerEventType, **kwargs: Any) -> None:
        if event_type is SchedulerEventType.MARK_TERMINAL:
            raise RuntimeError("injected crash before parent scheduler commit")
        real_record(conn, event_type=event_type, **kwargs)

    with (
        patch.object(processor, "_process_single_token", new=expand_parent_or_finish_child),
        patch.object(setup.factory.scheduler.events, "record", new=reject_first_parent_event),
        pytest.raises(RuntimeError, match="injected crash before parent scheduler commit"),
    ):
        processor._drain_scheduler_claims(ctx=_ctx(setup), pending_items={}, recover_pending_sinks=False)

    assert parent_calls == 1
    with setup.db.engine.connect() as conn:
        assert (
            conn.execute(
                select(token_work_items_table.c.status).where(token_work_items_table.c.work_item_id == parent_work_item_id)
            ).scalar_one()
            == TokenWorkStatus.LEASED.value
        )
        assert (
            conn.execute(
                select(token_work_items_table.c.token_id).where(token_work_items_table.c.token_id.in_(expansion_ids_by_attempt[0]))
            ).all()
            == []
        )
        assert (
            len(
                conn.execute(
                    select(token_parents_table.c.token_id).where(token_parents_table.c.parent_token_id == parent_token.token_id)
                ).all()
            )
            == 2
        )

    # Model the successor process's lease reap through the repository's named
    # crash-image adapter, then enter through the production recovery drain.
    clock.advance(1_000)
    recovered = setup.factory.scheduler.recover_expired_leases_legacy_unfenced(
        run_id=setup.run_id,
        now=clock.now_utc(),
        caller_owner="restart-reaper",
    )
    assert recovered == 1
    with patch.object(processor, "_process_single_token", new=expand_parent_or_finish_child):
        results = processor.drain_scheduled_work(_ctx(setup))

    assert parent_calls == 2
    assert expansion_ids_by_attempt[1] == expansion_ids_by_attempt[0]
    assert child_calls == dict.fromkeys(expansion_ids_by_attempt[0], 1)
    assert {result.token.token_id for result in results} == {parent_token.token_id, *expansion_ids_by_attempt[0]}

    with setup.db.engine.connect() as conn:
        final_rows = conn.execute(
            select(token_work_items_table.c.token_id, token_work_items_table.c.status).where(
                token_work_items_table.c.token_id.in_((parent_token.token_id, *expansion_ids_by_attempt[0]))
            )
        ).all()
        enqueue_counts = {
            token_id: len(
                conn.execute(
                    select(scheduler_events_table.c.event_id).where(
                        scheduler_events_table.c.token_id == token_id,
                        scheduler_events_table.c.event_type == SchedulerEventType.ENQUEUE.value,
                    )
                ).all()
            )
            for token_id in expansion_ids_by_attempt[0]
        }

    assert set(final_rows) == {
        (parent_token.token_id, TokenWorkStatus.TERMINAL.value),
        *((token_id, TokenWorkStatus.TERMINAL.value) for token_id in expansion_ids_by_attempt[0]),
    }
    assert enqueue_counts == dict.fromkeys(expansion_ids_by_attempt[0], 1)


def test_lease_lost_mid_processing_abandons_result_and_writes_no_disposition() -> None:
    """SchedulerLeaseLostError abandonment: already-proven results are
    returned (none here), staged branch losses are cleared, and NO
    mark_* disposition is issued for the abandoned claim."""
    processor, spy, setup, clock = _build(lease_owner=LEADER_OWNER)
    work_item_id, token = _enqueue_ready(setup, spy, clock, sequence=0)
    processor._pending_branch_losses.append(
        BranchLossSpec(
            coalesce_name="merge",
            row_id=token.row_id,
            branch_name="left",
            token_id=token.token_id,
            reason="staged before the lease was lost",
            recorded_by=LEADER_OWNER,
        )
    )

    def fake_process(**kwargs: Any) -> tuple[RowResult, list[Any]]:
        raise SchedulerLeaseLostError(work_item_id=work_item_id, lease_owner=LEADER_OWNER, run_id=setup.run_id)

    spy.calls.clear()
    with patch.object(processor, "_process_single_token", new=fake_process):
        results = processor._drain_scheduler_claims(ctx=_ctx(setup), pending_items={}, recover_pending_sinks=False)

    assert results == []
    assert processor._pending_branch_losses == [], "staged §E.5 losses are discarded with the abandoned claim"
    verbs = spy.verbs()
    assert "mark_failed" not in verbs
    assert "mark_terminal" not in verbs
    assert "mark_pending_sink" not in verbs
    # The row is left as the peer's rewrite left it — still LEASED here (the
    # test never actually reaped it); the drain must not touch it further.
    status, owner = _row_status(setup, work_item_id)
    assert status == TokenWorkStatus.LEASED.value
    assert owner == LEADER_OWNER


@pytest.mark.parametrize("surviving_peer", [False, True], ids=["sole-row-deletion", "n-greater-than-zero-absence"])
def test_heartbeat_membership_loss_propagates_without_failure_disposition(surviving_peer: bool) -> None:
    """A heartbeat membership refusal is not a plugin-processing failure.

    Deleting the claimant's registry row must propagate RunWorkerEvictedError
    directly whether the run registry becomes empty or another member remains.
    The claimed item and both event planes stay byte-for-byte at the image
    observed immediately after deletion; no mark_* disposition follows.
    """
    processor, spy, setup, clock = _build(lease_owner=LEADER_OWNER, heartbeat_seconds=60)
    if surviving_peer:
        _register_worker(setup, "worker-peer")
    work_item_id, token = _enqueue_ready(setup, spy, clock, sequence=0)
    processor._pending_branch_losses.append(
        BranchLossSpec(
            coalesce_name="merge",
            row_id=token.row_id,
            branch_name="left",
            token_id=token.token_id,
            reason="staged before membership loss",
            recorded_by=LEADER_OWNER,
        )
    )
    refused_image: list[tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]] = []

    def fake_process(**kwargs: Any) -> tuple[RowResult, list[Any]]:
        with setup.db.engine.begin() as conn:
            deleted = conn.execute(delete(run_workers_table).where(run_workers_table.c.worker_id == LEADER_OWNER))
        assert deleted.rowcount == 1
        refused_image.append(_durable_claim_image(setup, work_item_id))
        clock.advance(61.0)
        processor._heartbeat_active_claim()
        raise AssertionError("membership-refused heartbeat must not return")

    spy.calls.clear()
    with (
        patch.object(processor, "_process_single_token", new=fake_process),
        pytest.raises(RunWorkerEvictedError) as exc_info,
    ):
        processor._drain_scheduler_claims(ctx=_ctx(setup), pending_items={}, recover_pending_sinks=False)

    assert exc_info.value.worker_id == LEADER_OWNER
    assert exc_info.value.run_id == setup.run_id
    assert processor._pending_branch_losses == []
    assert len(refused_image) == 1
    assert _durable_claim_image(setup, work_item_id) == refused_image[0]
    assert len(spy.calls_for("heartbeat_lease")) == 1
    assert spy.calls_for("mark_failed") == []
    assert spy.calls_for("mark_terminal") == []
    assert spy.calls_for("mark_pending_sink") == []
    status, owner = _row_status(setup, work_item_id)
    assert status == TokenWorkStatus.LEASED.value
    assert owner == LEADER_OWNER


def test_heartbeat_refreshes_active_claim_once_per_interval_and_is_noop_outside() -> None:
    """Between claim and disposition, _heartbeat_active_claim writes at most
    one lease refresh per heartbeat interval; outside an active claim it is
    a no-op."""
    processor, spy, setup, clock = _build(lease_owner=LEADER_OWNER, heartbeat_seconds=60)
    work_item_id, _token = _enqueue_ready(setup, spy, clock, sequence=0)

    def fake_process(**kwargs: Any) -> tuple[RowResult, list[Any]]:
        processor._heartbeat_active_claim()  # within the interval: no write
        clock.advance(61.0)
        processor._heartbeat_active_claim()  # past the interval: exactly one write
        processor._heartbeat_active_claim()  # same instant: no second write
        return _dropped_result(kwargs["token"]), []

    spy.calls.clear()
    with patch.object(processor, "_process_single_token", new=fake_process):
        processor._drain_scheduler_claims(ctx=_ctx(setup), pending_items={}, recover_pending_sinks=False)

    beats = spy.calls_for("heartbeat_lease")
    assert len(beats) == 1
    assert beats[0]["work_item_id"] == work_item_id
    assert beats[0]["lease_owner"] == LEADER_OWNER
    assert beats[0]["membership_fenced"] is True

    # After the drain the claim is no longer active: even a long-overdue
    # heartbeat writes nothing.
    clock.advance(3600.0)
    processor._heartbeat_active_claim()
    assert len(spy.calls_for("heartbeat_lease")) == 1


def test_non_recovery_drains_run_maintenance_every_interval() -> None:
    """Scheduler maintenance (lease recovery) fires on every
    SCHEDULER_MAINTENANCE_INTERVAL-th non-recovery drain, fenced by the bound
    coordination token."""
    processor, spy, setup, _clock = _build(lease_owner=LEADER_OWNER, bind_leader_token=True)
    ctx = _ctx(setup)

    spy.calls.clear()
    for _ in range(SCHEDULER_MAINTENANCE_INTERVAL - 1):
        processor._drain_scheduler_claims(ctx=ctx, pending_items={}, recover_pending_sinks=False)
    assert spy.calls_for("recover_expired_leases") == [], "maintenance must not fire before the interval elapses"

    processor._drain_scheduler_claims(ctx=ctx, pending_items={}, recover_pending_sinks=False)
    recoveries = spy.calls_for("recover_expired_leases")
    assert len(recoveries) == 1
    assert set(recoveries[0]) == {"now", "coordination_token"}
    assert recoveries[0]["coordination_token"].worker_id == LEADER_OWNER


def test_immediate_enqueue_routes_registered_worker_to_strict_and_unregistered_to_explicit_legacy() -> None:
    registered, registered_spy, registered_setup, _clock = _build(lease_owner=LEADER_OWNER)
    registered_pending: dict[str, WorkItem] = {}
    registered_spy.calls.clear()

    registered_item = registered._scheduler_drain.enqueue_work_item(
        _unscheduled_work_item(registered_setup, sequence=20),
        registered_pending,
        claim_immediately=True,
    )

    assert registered_item.status is TokenWorkStatus.LEASED
    assert registered_spy.verbs() == ["enqueue_ready_claimed"]
    assert registered_item.work_item_id in registered_pending

    legacy, legacy_spy, legacy_setup, _clock = _build(lease_owner=None, register_leader=None)
    legacy_pending: dict[str, WorkItem] = {}
    legacy_spy.calls.clear()

    legacy_item = legacy._scheduler_drain.enqueue_work_item(
        _unscheduled_work_item(legacy_setup, sequence=21),
        legacy_pending,
        claim_immediately=True,
    )

    assert legacy_item.status is TokenWorkStatus.LEASED
    assert legacy_spy.verbs() == ["enqueue_ready_claimed_legacy_unfenced"]
    assert legacy_item.work_item_id in legacy_pending


def test_immediate_enqueue_routing_ast_and_legacy_production_references_are_pinned() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    drain_path = repo_root / "src/elspeth/engine/scheduler_drain.py"
    drain_tree = ast.parse(drain_path.read_text(encoding="utf-8"), filename=str(drain_path))
    enqueue_method = next(node for node in ast.walk(drain_tree) if isinstance(node, ast.FunctionDef) and node.name == "enqueue_work_item")
    route = next(
        node
        for node in ast.walk(enqueue_method)
        if isinstance(node, ast.IfExp)
        and isinstance(node.body, ast.Attribute)
        and node.body.attr == "enqueue_ready_claimed"
        and isinstance(node.orelse, ast.Attribute)
        and node.orelse.attr == "enqueue_ready_claimed_legacy_unfenced"
    )
    assert ast.unparse(route.test) == "self._scheduler_lease_owner_registered"

    legacy_references: list[tuple[str, str]] = []
    for source_path in (repo_root / "src/elspeth").rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr != "enqueue_ready_claimed_legacy_unfenced":
                continue
            cursor: ast.AST | None = node
            while cursor is not None and not isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cursor = parents.get(cursor)
            assert isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef))
            legacy_references.append((str(source_path.relative_to(repo_root)), cursor.name))

    assert sorted(legacy_references) == [
        ("src/elspeth/core/landscape/scheduler_repository.py", "enqueue_ready_claimed_legacy_unfenced"),
        ("src/elspeth/engine/scheduler_drain.py", "enqueue_work_item"),
    ]


def test_follower_build_skips_recovery_and_leader_without_token_refuses() -> None:
    """ADR-030 §C.3: the FOLLOWER-mode build must NOT run
    recover_expired_leases; a LEADER-mode build must hold a token.

    elspeth-577179bba1 made follower-ness an explicit ProcessorMode instead
    of the old triple-None inference (token=None + run_coordination=None +
    registered lease owner). The production follower build passes
    mode=FOLLOWER and returns before the strict token boundary. Direct
    pre-coordination tests recover through the named repository adapter, not
    by downgrading this production maintenance path."""
    follower, follower_spy, _fsetup, _fclock = _build(lease_owner="follower-1", mode=ProcessorMode.FOLLOWER)
    assert follower.reap_expired_peer_leases() == 0
    assert follower_spy.calls_for("recover_expired_leases") == [], "follower builds must never reap peer leases"

    legacy, legacy_spy, _lsetup, _lclock = _build(lease_owner=None, register_leader=None)
    with pytest.raises(OrchestrationInvariantError, match="coordination token"):
        legacy.reap_expired_peer_leases()
    assert legacy_spy.calls_for("recover_expired_leases") == []

    # The old sentinel shape WITHOUT the explicit mode remains a LEADER-mode
    # build, but it cannot select an unfenced transaction from missing authority.
    sentinel_shape, sentinel_spy, _ssetup, _sclock = _build(lease_owner="follower-1")
    with pytest.raises(OrchestrationInvariantError, match="coordination token"):
        sentinel_shape.reap_expired_peer_leases()
    assert sentinel_spy.calls_for("recover_expired_leases") == []
