"""Pin ``RowProcessor.has_unresolved_scheduler_work`` — the run-completion invariant.

The orchestrator refuses to complete a run (core.py post-source-loop) or a
resume (resume.py post-EOF-flush) while ``has_unresolved_scheduler_work()``
is true. The merge into the decomposed orchestrator refined this invariant
from PENDING_SINK-inclusive ``has_scheduled_work()`` to the unresolved-work
predicate (scheduler_repository.``_unresolved_work_predicate``): sink-bound
rows whose producer work is durably complete legitimately park until
post-invariant sink durability and must NOT block run completion.

These tests pin both edges of that predicate against the real scheduler
repository, driving every status through production verbs:

FIRES (work short of a durable sink handoff):
- READY (enqueued, unclaimed)
- LEASED without ``pending_sink_name`` (claimed transform work in flight)
- BLOCKED (parked at a barrier)

PASSES (resolved into a durable sink handoff, or finished):
- PENDING_SINK (durable handoff parked for the sink drain)
- LEASED with ``pending_sink_name`` set (handoff re-claimed by the drain)
- TERMINAL / FAILED

The PASSES-but-still-active cases also assert ``has_scheduled_work()`` is
True, pinning that the refinement is "active but resolved", not "gone".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from elspeth.contracts.scheduler import TokenWorkItem, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.engine.processor import DAGTraversalContext, RowProcessor
from elspeth.engine.spans import SpanFactory
from tests.fixtures.landscape import RecorderSetup, make_recorder_with_run, register_test_node

NODE_ID = "normalize"
LEASE_OWNER = "worker-a"
NOW = datetime(2026, 1, 1, tzinfo=UTC)

_PAYLOAD = TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))


def _build_processor() -> tuple[RowProcessor, TokenSchedulerRepository, RecorderSetup]:
    """Real RowProcessor over a real in-memory audit DB and scheduler repository."""
    setup = make_recorder_with_run(run_id="run-unresolved-pin", source_node_id="source-1")
    register_test_node(setup.data_flow, setup.run_id, NODE_ID)
    scheduler = setup.factory.scheduler
    processor = RowProcessor(
        execution=setup.execution,
        data_flow=setup.data_flow,
        span_factory=SpanFactory(),
        run_id=setup.run_id,
        source_node_id=NodeID(setup.source_node_id),
        source_on_success="default",
        traversal=DAGTraversalContext(node_step_map={}, node_to_plugin={}, node_to_next={}, coalesce_node_map={}),
        scheduler=scheduler,
        scheduler_lease_owner=LEASE_OWNER,
    )
    return processor, scheduler, setup


def _enqueue_ready_token(setup: RecorderSetup, scheduler: TokenSchedulerRepository, *, sequence: int = 0) -> TokenWorkItem:
    """Create a real row/token pair and enqueue its READY continuation."""
    row, token = setup.data_flow.create_row_with_token(
        run_id=setup.run_id,
        source_node_id=setup.source_node_id,
        row_index=sequence,
        data={"id": sequence},
        source_row_index=sequence,
        ingest_sequence=sequence,
    )
    return scheduler.enqueue_ready(
        run_id=setup.run_id,
        token_id=token.token_id,
        row_id=row.row_id,
        node_id=NODE_ID,
        step_index=1,
        ingest_sequence=sequence,
        available_at=NOW,
        row_payload_json=_PAYLOAD,
    )


def _claim(scheduler: TokenSchedulerRepository, run_id: str) -> TokenWorkItem:
    item = scheduler.claim_ready(run_id=run_id, lease_owner=LEASE_OWNER, lease_seconds=300, now=NOW)
    assert item is not None
    return item


def _mark_pending_sink(scheduler: TokenSchedulerRepository, work_item_id: str) -> TokenWorkItem:
    return scheduler.mark_pending_sink(
        work_item_id=work_item_id,
        row_payload_json=_PAYLOAD,
        sink_name="sink-a",
        outcome="success",
        path="completed",
        error_hash=None,
        error_message=None,
        now=NOW + timedelta(seconds=1),
        expected_lease_owner=LEASE_OWNER,
    )


def test_no_scheduler_work_does_not_fire() -> None:
    """Baseline: a run with no scheduler work has nothing unresolved."""
    processor, _scheduler, _setup = _build_processor()
    assert processor.has_unresolved_scheduler_work() is False
    assert processor.has_scheduled_work() is False


def test_fires_for_ready_work() -> None:
    """READY work (enqueued, never claimed) blocks run completion."""
    processor, scheduler, setup = _build_processor()
    item = _enqueue_ready_token(setup, scheduler)
    assert item.status is TokenWorkStatus.READY
    assert processor.has_unresolved_scheduler_work() is True


def test_fires_for_leased_transform_work() -> None:
    """LEASED work without a pending_sink_name (transform in flight) blocks run completion."""
    processor, scheduler, setup = _build_processor()
    _enqueue_ready_token(setup, scheduler)
    claimed = _claim(scheduler, setup.run_id)
    assert claimed.status is TokenWorkStatus.LEASED
    assert claimed.pending_sink_name is None
    assert processor.has_unresolved_scheduler_work() is True


def test_fires_for_blocked_work() -> None:
    """BLOCKED work (parked at a barrier) blocks run completion."""
    processor, scheduler, setup = _build_processor()
    _enqueue_ready_token(setup, scheduler)
    claimed = _claim(scheduler, setup.run_id)
    blocked = scheduler.mark_blocked(
        work_item_id=claimed.work_item_id,
        queue_key=None,
        barrier_key="barrier-1",
        now=NOW + timedelta(seconds=1),
        expected_lease_owner=LEASE_OWNER,
    )
    assert blocked.status is TokenWorkStatus.BLOCKED
    assert processor.has_unresolved_scheduler_work() is True


def test_passes_for_pending_sink_handoff() -> None:
    """A PENDING_SINK handoff is active-but-resolved: it must NOT block the
    pre-sink completion invariant, because it is terminalized only after sink
    durability (the merge-authored refinement over has_scheduled_work)."""
    processor, scheduler, setup = _build_processor()
    _enqueue_ready_token(setup, scheduler)
    claimed = _claim(scheduler, setup.run_id)
    parked = _mark_pending_sink(scheduler, claimed.work_item_id)
    assert parked.status is TokenWorkStatus.PENDING_SINK
    assert processor.has_unresolved_scheduler_work() is False
    assert processor.has_scheduled_work() is True


def test_passes_for_leased_pending_sink_reclaim() -> None:
    """A pending sink re-claimed by the drain (LEASED with pending_sink_name set)
    is still resolved: its producer work is durably complete."""
    processor, scheduler, setup = _build_processor()
    _enqueue_ready_token(setup, scheduler)
    claimed = _claim(scheduler, setup.run_id)
    _mark_pending_sink(scheduler, claimed.work_item_id)
    reclaimed = scheduler.claim_pending_sink(
        run_id=setup.run_id, lease_owner=LEASE_OWNER, lease_seconds=300, now=NOW + timedelta(seconds=2)
    )
    assert reclaimed is not None
    assert reclaimed.status is TokenWorkStatus.LEASED
    assert reclaimed.pending_sink_name == "sink-a"
    assert processor.has_unresolved_scheduler_work() is False
    assert processor.has_scheduled_work() is True


def test_passes_after_terminal_and_failed_work() -> None:
    """TERMINAL and FAILED work items leave nothing unresolved (and nothing active)."""
    processor, scheduler, setup = _build_processor()
    _enqueue_ready_token(setup, scheduler, sequence=0)
    _enqueue_ready_token(setup, scheduler, sequence=1)
    first = _claim(scheduler, setup.run_id)
    scheduler.mark_terminal(work_item_id=first.work_item_id, now=NOW + timedelta(seconds=1), expected_lease_owner=LEASE_OWNER)
    second = _claim(scheduler, setup.run_id)
    scheduler.mark_failed(work_item_id=second.work_item_id, now=NOW + timedelta(seconds=2), expected_lease_owner=LEASE_OWNER)
    assert processor.has_unresolved_scheduler_work() is False
    assert processor.has_scheduled_work() is False
