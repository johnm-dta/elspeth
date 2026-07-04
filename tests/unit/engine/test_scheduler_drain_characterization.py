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
3. SchedulerLeaseLostError abandonment: the in-flight result is dropped,
   staged §E.5 branch losses are cleared, and NO disposition write follows —
   the peer's rewrite owns the row now.
4. Active-claim heartbeat: between claim and disposition,
   ``_heartbeat_active_claim`` refreshes the claimed lease at most once per
   heartbeat interval; outside an active claim it is a no-op.
5. Maintenance cadence: non-recovery drains run scheduler maintenance every
   SCHEDULER_MAINTENANCE_INTERVAL drains; the follower build (no coordination
   token, no run_coordination, registered lease owner) skips lease recovery
   entirely (ADR-030 §C.3), while the legacy/unregistered build reaps via the
   unfenced arm.

The observables are durable scheduler rows, returned RowResults, and the
verb order recorded by a delegating scheduler wrapper — not processor
internals — so the net survives the extraction. ``_drain_scheduler_claims``
and ``_heartbeat_active_claim`` are called by their private names on purpose:
those names are load-bearing (orchestrator/follower.py and
test_adr030_loosened_invariant_guard.py call them) and the extraction must
leave delegates behind.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import patch

from sqlalchemy import insert, select

from elspeth.contracts import RowResult, TokenInfo
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import SchedulerLeaseLostError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.scheduler import BranchLossSpec, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import run_workers_table, token_work_items_table
from elspeth.engine.clock import MockClock
from elspeth.engine.processor import SCHEDULER_MAINTENANCE_INTERVAL, DAGTraversalContext, RowProcessor
from elspeth.engine.spans import SpanFactory
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
    assert recoveries[0]["caller_owner"] == LEADER_OWNER
    assert recoveries[0]["coordination_token"] is not None


def test_follower_build_skips_lease_recovery_and_legacy_build_reaps_unfenced() -> None:
    """ADR-030 §C.3: the follower build (no coordination token, no
    run_coordination, registered lease owner) must NOT run
    recover_expired_leases; the legacy/unregistered build reaps via the
    unfenced (coordination_token=None) arm."""
    follower, follower_spy, _fsetup, _fclock = _build(lease_owner="follower-1")
    assert follower.reap_expired_peer_leases() == 0
    assert follower_spy.calls_for("recover_expired_leases") == [], "follower builds must never reap peer leases"

    legacy, legacy_spy, _lsetup, _lclock = _build(lease_owner=None, register_leader=None)
    legacy.reap_expired_peer_leases()
    recoveries = legacy_spy.calls_for("recover_expired_leases")
    assert len(recoveries) == 1
    assert recoveries[0]["coordination_token"] is None
