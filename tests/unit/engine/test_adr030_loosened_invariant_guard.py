"""Pin the ADR-030 loosened OrchestrationInvariantError (M1 remediation).

The leader's drain loop (``RowProcessor._drain_scheduler_claims``) used to raise
``OrchestrationInvariantError`` whenever ``claim_ready`` returned None while
in-memory ``pending_items`` remained. The multi-worker change loosened this so a
leader can relinquish continuations a PEER follower has already claimed.

The M1 remediation tightened the discriminator. A leader relinquishes its
non-READY pending set ONLY when ALL THREE hold:

  (1) ``count_ready_in_set == 0`` — no pending item is still READY (a still-READY
      item is a genuine stranded continuation → raise);
  (2) ``count_failed_in_set == 0`` — no pending item is FAILED. FAILED is the
      ONLY status absent from BOTH backstops (``count_active_work`` /
      ``has_unresolved_scheduler_work`` AND ``complete_run``'s quiescence CAS,
      which cover READY/LEASED/BLOCKED/PENDING_SINK but NOT FAILED), so a silently
      relinquished self-FAILED stray would be lost with no backstop; and
  (3) a peer is/was carrying work — ``has_peer_owned_work()`` is True: some OTHER
      lease_owner holds a LEASED or PENDING_SINK row on this run. PENDING_SINK is
      included because mark_pending_sink KEEPS the claimant's lease_owner, so a
      follower that parked all its claims and let its active leases lapse is still
      detectable (the in-claim drain races against exactly that).

This closes the N=1 holes the bare loosening opened:

  - N=1 stranded-READY STILL raises — arm (1) fails (ready_count > 0);
  - N=1 self-FAILED (own pending item FAILED, NO peer) STILL raises — arms (2)
    AND (3) both fail; FAILED is the unbacked-stopped status, so this is the
    decisive M1 fix and it lands at N=1;
  - a MIXED set (one peer-carried continuation + one self-FAILED stray) STILL
    raises — arm (2) fails even though a peer IS present (the residual the coarse
    run-level "ready==0 AND some peer" interim gate left open);
  - N=1 self-BLOCKED STILL raises — arm (3) fails (own rows carry the leader's
    owner, BLOCKED carries none → no peer-owned work); BLOCKED is additionally
    caught by the run-level backstop;
  - the unregistered (legacy/test) path raises immediately — the
    ``_scheduler_lease_owner_registered`` guard is never entered.

The HAPPY path (N>=2 peer handoff) clears+breaks WITHOUT raising AND emits a
Relinquishing log line: ready==0, no FAILED pending, and a peer holds a lease —
the relinquish set is exactly "non-READY/non-FAILED continuations a peer is
carrying", every one covered by the run-level + quiescence backstops.

These tests drive the REAL ``_drain_scheduler_claims`` over a real scheduler DB
with a pre-populated ``pending_items`` dict, controlling the claim outcome by
shaping the durable rows. The self-FAILED and mixed cases FAIL under both the
bare pre-M1 loosening AND the coarse run-level interim gate.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import update

from elspeth.contracts import TokenInfo
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import token_work_items_table
from elspeth.engine.clock import MockClock
from elspeth.engine.dag_navigator import WorkItem
from elspeth.engine.processor import DAGTraversalContext, RowProcessor
from elspeth.engine.spans import SpanFactory
from tests.fixtures.landscape import RecorderSetup, make_recorder_with_run, register_test_node

NODE_ID = "normalize"
LEADER_OWNER = "leader-a"
PEER_OWNER = "follower-b"

_CONTRACT = SchemaContract(mode="OBSERVED", fields=(), locked=True)
_PAYLOAD = TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, _CONTRACT))


def _build_processor(*, scheduler_lease_owner: str | None) -> tuple[RowProcessor, TokenSchedulerRepository, RecorderSetup, MockClock]:
    """Real RowProcessor + real scheduler DB, with a deterministic MockClock.

    ``scheduler_lease_owner=None`` exercises the unregistered (legacy/test)
    path (``_scheduler_lease_owner_registered=False``).
    """
    setup = make_recorder_with_run(run_id="run-loosen-guard", source_node_id="source-1")
    register_test_node(setup.data_flow, setup.run_id, NODE_ID)
    clock = MockClock(start=1_750_000_000.0)
    processor = RowProcessor(
        execution=setup.execution,
        data_flow=setup.data_flow,
        span_factory=SpanFactory(),
        run_id=setup.run_id,
        source_node_id=NodeID(setup.source_node_id),
        source_on_success="default",
        traversal=DAGTraversalContext(node_step_map={}, node_to_plugin={}, node_to_next={}, coalesce_node_map={}),
        scheduler=setup.factory.scheduler,
        scheduler_lease_owner=scheduler_lease_owner,
        clock=clock,
    )
    return processor, setup.factory.scheduler, setup, clock


def _enqueue_ready(
    setup: RecorderSetup, scheduler: TokenSchedulerRepository, clock: MockClock, *, sequence: int, available_at_offset: float = 0.0
) -> tuple[str, TokenInfo]:
    """Enqueue a READY continuation; return (work_item_id, token).

    ``available_at_offset`` lets the row be READY-but-not-yet-claimable (a
    future available_at means claim_ready skips it, but it is still READY).
    """
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
        available_at=clock.now_utc() + timedelta(seconds=available_at_offset),
        row_payload_json=_PAYLOAD,
    )
    token_info = TokenInfo(row_id=row.row_id, token_id=token.token_id, row_data=PipelineRow({"id": sequence}, _CONTRACT))
    return item.work_item_id, token_info


def _pending_work_item(work_item_id: str, token: TokenInfo) -> dict[str, WorkItem]:
    return {work_item_id: WorkItem(token=token, current_node_id=NodeID(NODE_ID))}


def _seed_peer_leased_row(setup: RecorderSetup, scheduler: TokenSchedulerRepository, clock: MockClock, *, sequence: int) -> None:
    """Create an unrelated row and force it LEASED under a PEER owner.

    Direct SQL flip (the technique used across the e2e follower suites) so the
    leader's ``peer_active_leases()`` sees an unexpired peer-held lease.
    """
    work_item_id, _token = _enqueue_ready(setup, scheduler, clock, sequence=sequence)
    with setup.db.engine.begin() as conn:
        conn.execute(
            update(token_work_items_table)
            .where(token_work_items_table.c.work_item_id == work_item_id)
            .values(
                status=TokenWorkStatus.LEASED.value,
                lease_owner=PEER_OWNER,
                lease_expires_at=clock.now_utc() + timedelta(seconds=300),
            )
        )


def test_n1_stranded_ready_still_raises() -> None:
    """Test 1: a registered solo leader with a genuinely-stranded READY pending
    item (count_ready_in_set > 0) STILL raises — the loosening is a no-op while
    items remain READY.

    NOTE: this is a defensive positive-contract pin, not the M1-specific
    regression. It also raises under the PRE-M1 bare loosening (which only checked
    count_ready_in_set==0) because ready_count > 0 here (arm 1 fails). The
    M1-specific regression (the added FAILED arm + peer-existence arm) is pinned by
    ``test_n1_self_failed_still_raises`` and ``test_mixed_self_failed_and_peer_still_raises``.
    """
    processor, scheduler, setup, clock = _build_processor(scheduler_lease_owner=LEADER_OWNER)
    # READY but available_at in the future → claim_ready returns None, row stays READY.
    work_item_id, token = _enqueue_ready(setup, scheduler, clock, sequence=0, available_at_offset=3600.0)
    pending = _pending_work_item(work_item_id, token)

    assert scheduler.count_ready_in_set(run_id=setup.run_id, work_item_ids=[work_item_id]) == 1
    with pytest.raises(OrchestrationInvariantError, match="no READY work item could be claimed"):
        processor._drain_scheduler_claims(
            ctx=PluginContext(run_id=setup.run_id, config={}, landscape=None), pending_items=pending, recover_pending_sinks=False
        )


def test_n1_self_failed_still_raises() -> None:
    """Test 2 (the M1 residual): a registered solo leader whose OWN pending item
    is FAILED, with NO peer holding a lease, STILL raises.

    count_failed_in_set==1 (arm 2 fails) AND has_peer_owned_work() is False (arm 3
    fails) → the discriminator refuses to clear+break → raise. This is the case
    the bare loosening (pre-M1) silently cleared, losing a continuation that
    FAILED is absent from complete_run's quiescence predicate cannot catch.
    """
    processor, scheduler, setup, clock = _build_processor(scheduler_lease_owner=LEADER_OWNER)
    work_item_id, token = _enqueue_ready(setup, scheduler, clock, sequence=0)
    # Claim under the leader, then mark FAILED under the leader's OWN owner.
    claimed = scheduler.claim_ready(run_id=setup.run_id, lease_owner=LEADER_OWNER, lease_seconds=300, now=clock.now_utc())
    assert claimed is not None and claimed.work_item_id == work_item_id
    scheduler.mark_failed(work_item_id=work_item_id, now=clock.now_utc(), expected_lease_owner=LEADER_OWNER)

    pending = _pending_work_item(work_item_id, token)
    assert scheduler.count_ready_in_set(run_id=setup.run_id, work_item_ids=[work_item_id]) == 0
    assert scheduler.count_failed_in_set(run_id=setup.run_id, work_item_ids=[work_item_id]) == 1
    assert scheduler.has_peer_owned_work(run_id=setup.run_id, caller_owner=LEADER_OWNER) is False

    with pytest.raises(OrchestrationInvariantError, match="no READY work item could be claimed"):
        processor._drain_scheduler_claims(
            ctx=PluginContext(run_id=setup.run_id, config={}, landscape=None), pending_items=pending, recover_pending_sinks=False
        )


def test_n1_self_blocked_still_raises_and_backstop_counts_blocked() -> None:
    """Test 6: a registered solo leader whose OWN pending item is BLOCKED (a
    barrier hold) with NO peer STILL raises (the peer-existence gate refuses to
    relinquish a self-held BLOCKED row). And the run-level backstop
    has_unresolved_scheduler_work() counts BLOCKED — so even if it WERE
    relinquished to a peer, the BLOCKED row would not silently complete.

    NOTE: like test 1, this is a defensive positive-contract pin. Under the M1
    discriminator the self-BLOCKED-with-no-peer case raises because arm (3) fails
    (no peer). BLOCKED additionally has the run-level has_unresolved_scheduler_work
    backstop, making it less discriminating than the self-FAILED case, which has
    NO backstop and is the sharpest M1 pin.
    """
    processor, scheduler, setup, clock = _build_processor(scheduler_lease_owner=LEADER_OWNER)
    work_item_id, token = _enqueue_ready(setup, scheduler, clock, sequence=0)
    claimed = scheduler.claim_ready(run_id=setup.run_id, lease_owner=LEADER_OWNER, lease_seconds=300, now=clock.now_utc())
    assert claimed is not None and claimed.work_item_id == work_item_id
    scheduler.mark_blocked(
        work_item_id=work_item_id,
        queue_key=None,
        barrier_key="barrier-1",
        now=clock.now_utc(),
        expected_lease_owner=LEADER_OWNER,
    )

    pending = _pending_work_item(work_item_id, token)
    assert scheduler.count_ready_in_set(run_id=setup.run_id, work_item_ids=[work_item_id]) == 0
    # mark_blocked NULLs lease_owner, so the self-BLOCKED row is not peer-owned.
    assert scheduler.has_peer_owned_work(run_id=setup.run_id, caller_owner=LEADER_OWNER) is False
    # The run-level backstop counts BLOCKED as unresolved (the safety net).
    assert processor.has_unresolved_scheduler_work() is True

    with pytest.raises(OrchestrationInvariantError, match="no READY work item could be claimed"):
        processor._drain_scheduler_claims(
            ctx=PluginContext(run_id=setup.run_id, config={}, landscape=None), pending_items=pending, recover_pending_sinks=False
        )


def test_unregistered_owner_path_raises_immediately() -> None:
    """Test 3: scheduler_lease_owner=None (_scheduler_lease_owner_registered=False)
    → the loosening guard is never entered, so a non-empty pending set with no
    claimable READY raises immediately — even though a peer lease exists.

    NOTE: defensive positive-contract pin for the ``if
    self._scheduler_lease_owner_registered:`` guard. It raises under both pre- and
    post-M1 code (the unregistered path never enters the loosening at all)."""
    processor, scheduler, setup, clock = _build_processor(scheduler_lease_owner=None)
    # A peer lease exists, but the unregistered guard must short-circuit it.
    _seed_peer_leased_row(setup, scheduler, clock, sequence=1)
    work_item_id, token = _enqueue_ready(setup, scheduler, clock, sequence=0)
    # Move it out of READY so claim_ready returns None for our pending item.
    with setup.db.engine.begin() as conn:
        conn.execute(
            update(token_work_items_table)
            .where(token_work_items_table.c.work_item_id == work_item_id)
            .values(status=TokenWorkStatus.FAILED.value, lease_owner=None, lease_expires_at=None)
        )
    pending = _pending_work_item(work_item_id, token)

    with pytest.raises(OrchestrationInvariantError, match="no READY work item could be claimed"):
        processor._drain_scheduler_claims(
            ctx=PluginContext(run_id=setup.run_id, config={}, landscape=None), pending_items=pending, recover_pending_sinks=False
        )


def test_n2_peer_claim_handoff_clears_and_breaks_without_raising(caplog: pytest.LogCaptureFixture) -> None:
    """Test 4 (in-process arm): a registered leader whose pending item is no
    longer READY (a peer claimed it) AND a peer holds an active lease →
    clears+breaks WITHOUT raising. This is the loosening doing its job.

    We model the peer-claimed continuation as a FAILED/non-READY row under the
    leader (count_ready_in_set==0) while a SEPARATE peer-held LEASED row makes
    peer_active_leases() non-empty.

    SCOPE: this pins only the LEADER's relinquish decision (clear+break without
    raising) and the observability log line at that decision. The full custody
    chain — the peer's LEASED row reaching TERMINAL with exactly one sink
    emission, and the leader NOT stealing it — is covered end-to-end by
    ``tests/e2e/recovery/test_multi_worker_leader_finalize.py::
    test_end_to_end_exactly_once_with_follower_attributed_work``.
    """
    processor, scheduler, setup, clock = _build_processor(scheduler_lease_owner=LEADER_OWNER)
    # Peer holds an active lease on an unrelated row.
    _seed_peer_leased_row(setup, scheduler, clock, sequence=1)
    # The leader's pending continuation is no longer READY (peer took it).
    work_item_id, token = _enqueue_ready(setup, scheduler, clock, sequence=0)
    with setup.db.engine.begin() as conn:
        conn.execute(
            update(token_work_items_table)
            .where(token_work_items_table.c.work_item_id == work_item_id)
            .values(
                status=TokenWorkStatus.LEASED.value,
                lease_owner=PEER_OWNER,
                lease_expires_at=clock.now_utc() + timedelta(seconds=300),
            )
        )
    pending = _pending_work_item(work_item_id, token)

    assert scheduler.count_ready_in_set(run_id=setup.run_id, work_item_ids=[work_item_id]) == 0
    assert scheduler.count_failed_in_set(run_id=setup.run_id, work_item_ids=[work_item_id]) == 0
    assert scheduler.has_peer_owned_work(run_id=setup.run_id, caller_owner=LEADER_OWNER) is True

    # Must NOT raise; the drain clears the pending set and breaks.
    import logging

    with caplog.at_level(logging.INFO, logger="elspeth.engine.processor"):
        results = processor._drain_scheduler_claims(
            ctx=PluginContext(run_id=setup.run_id, config={}, landscape=None), pending_items=pending, recover_pending_sinks=False
        )
    assert results == [], "no rows processed by the leader; the peer owns the continuation"
    assert pending == {}, "the leader relinquished (cleared) its pending continuations to the peer"

    # The custody transfer is OBSERVABLE: a Relinquishing log line records the
    # run_id and the relinquished work_item_id (M1 spec).
    relinquish_records = [r for r in caplog.records if "Relinquishing" in r.getMessage()]
    assert relinquish_records, "the clear+break must emit a Relinquishing log line (custody-transfer observability)"
    msg = relinquish_records[0].getMessage()
    assert setup.run_id in msg
    assert work_item_id in msg


def test_mixed_self_failed_and_peer_still_raises() -> None:
    """The concurrency residual: a MIXED pending set — one item genuinely
    peer-claimed (LEASED under PEER_OWNER) AND one the leader's OWN self-FAILED
    stray — with a live peer present must STILL raise.

    The coarse run-level interim gate (ready_count==0 over ALL pending keys AND
    any peer holds a lease) would relinquish BOTH — including the self-FAILED
    stray, which is absent from complete_run's quiescence predicate and therefore
    has NO backstop. The shipped M1 discriminator adds the count_failed_in_set==0
    arm, so a single FAILED pending row keeps the leader raising:
    count_failed_in_set==1 here even though ready_count==0 and a peer is present.

    If this test FAILS (clears+breaks instead of raising), the FAILED arm has
    regressed and a self-FAILED leader continuation would be silently relinquished
    with no backstop — the exact lost-work hole the FAILED arm closes.
    """
    processor, scheduler, setup, clock = _build_processor(scheduler_lease_owner=LEADER_OWNER)

    # Item A: a genuine peer-claimed continuation (LEASED under PEER_OWNER).
    peer_item_id, peer_token = _enqueue_ready(setup, scheduler, clock, sequence=0)
    with setup.db.engine.begin() as conn:
        conn.execute(
            update(token_work_items_table)
            .where(token_work_items_table.c.work_item_id == peer_item_id)
            .values(
                status=TokenWorkStatus.LEASED.value,
                lease_owner=PEER_OWNER,
                lease_expires_at=clock.now_utc() + timedelta(seconds=300),
            )
        )

    # Item B: the leader's OWN self-FAILED stray (claimed by the leader, marked
    # FAILED under the leader's own owner — NO peer owns it).
    self_item_id, self_token = _enqueue_ready(setup, scheduler, clock, sequence=1)
    claimed = scheduler.claim_ready(run_id=setup.run_id, lease_owner=LEADER_OWNER, lease_seconds=300, now=clock.now_utc())
    assert claimed is not None and claimed.work_item_id == self_item_id
    scheduler.mark_failed(work_item_id=self_item_id, now=clock.now_utc(), expected_lease_owner=LEADER_OWNER)

    pending = {
        peer_item_id: WorkItem(token=peer_token, current_node_id=NodeID(NODE_ID)),
        self_item_id: WorkItem(token=self_token, current_node_id=NodeID(NODE_ID)),
    }
    # ready_count==0 over the whole set (A is LEASED, B is FAILED) and a peer DOES
    # own work (A) — so arms (1) and (3) would BOTH pass. The FAILED arm (2) is the
    # only thing keeping this loud: count_failed_in_set==1 (B). The SAFE contract:
    # the self-FAILED stray (B) must keep the leader raising.
    assert scheduler.count_ready_in_set(run_id=setup.run_id, work_item_ids=list(pending.keys())) == 0
    assert scheduler.count_failed_in_set(run_id=setup.run_id, work_item_ids=list(pending.keys())) == 1
    assert scheduler.has_peer_owned_work(run_id=setup.run_id, caller_owner=LEADER_OWNER) is True

    with pytest.raises(OrchestrationInvariantError, match="no READY work item could be claimed"):
        processor._drain_scheduler_claims(
            ctx=PluginContext(run_id=setup.run_id, config={}, landscape=None), pending_items=pending, recover_pending_sinks=False
        )
