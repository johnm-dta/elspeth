# tests/e2e/recovery/test_barrier_timing_invariance.py
"""§H 476 pinned doctrine — barrier timing is invariant under leader takeover.

ADR-030 slice 3 (§E.2 backdated adoption): a batch's timeout fire time is a
pure function of durable state — ``barrier_blocked_at(oldest member) +
timeout_seconds`` — never of WHEN a leader's intake adopted the row.  Both
anchoring frames are pinned against the SAME MockClock schedule:

* **Frame A (live path):** the leader that blocked the rows adopts them at
  its own drain-iteration intake; ``TriggerEvaluator._first_accept_time`` /
  coalesce ``first_arrival`` anchor to the clamped wall→monotonic transform
  of T_b (the durable ``barrier_blocked_at``), NOT to adoption time.
* **Frame B (takeover restore path):** the seat is usurped mid-window and a
  new leader restores via ``_restore_barriers_from_journal``; the restored
  anchor is the SAME transform of the SAME durable stamp.

In both frames the trigger must NOT fire at T_b+timeout-ε and MUST fire at
T_b+timeout+ε — the fire-instant difference across the takeover is exactly
zero under MockClock — and batch composition (the ``batch_members`` set) is
identical in both frames.

Construction: the unit-engine builders (real LandscapeDB, real scheduler
journal, real run_coordination seat minted by ``begin_run``) with a REAL
takeover image — the seat epoch is bumped under a usurper identity exactly
like :func:`tests.e2e.recovery.harness._usurp_seat`, and the takeover
processor binds the post-usurpation token.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import update

from elspeth.contracts import TokenInfo
from elspeth.contracts.enums import TerminalPath, TriggerType
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.core.config import CoalesceSettings
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.database import begin_write
from elspeth.core.landscape.schema import run_coordination_table
from elspeth.engine.clock import MockClock
from elspeth.engine.coalesce_executor import CoalesceExecutor
from elspeth.engine.processor import BarrierJournalRestoreContext, _LiveBarrierHold
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenManager
from elspeth.testing import make_row
from tests.fixtures.factories import make_context
from tests.unit.engine.test_adr030_slice3_intake import (
    AGG_NODE,
    _agg_processor,
    _passthrough_flush_transform,
)
from tests.unit.engine.test_processor import (
    _make_factory,
    _make_processor,
    _make_source_row,
    _persist_blocked_scheduler_work,
)

_T0 = 1_750_000_000.0
RUN_ID = "test-run"
USURPER = "worker-usurper"
TIMEOUT_SECONDS = 10.0
COALESCE_NODE = NodeID("coalesce::merge")


def _usurp_seat(db: LandscapeDB, run_id: str, clock: MockClock) -> None:
    """The in-DB image of a takeover (harness ``_usurp_seat``): epoch bump
    under a usurper identity. Expiry is kept live so the takeover processor's
    ``leader_coordination_token`` binding reads a current seat."""
    now = clock.now_utc()
    with begin_write(db.engine) as conn:
        conn.execute(
            update(run_coordination_table)
            .where(run_coordination_table.c.run_id == run_id)
            .values(
                leader_worker_id=USURPER,
                leader_epoch=run_coordination_table.c.leader_epoch + 1,
                leader_heartbeat_expires_at=now + timedelta(seconds=300),
                updated_at=now,
            )
        )


def _restore_context() -> BarrierJournalRestoreContext:
    return BarrierJournalRestoreContext(
        resume_checkpoint_id="ckpt-takeover",
        barrier_scalars=None,
        batch_id_remap={},
    )


def _real_coalesce_executor(factory: Any, clock: MockClock, *, policy: str = "best_effort") -> CoalesceExecutor:
    """The production executor over the REAL repositories (no mocks)."""
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
            timeout_seconds=TIMEOUT_SECONDS,
            on_success="default",
        ),
        COALESCE_NODE,
        output_schema=SchemaContract(mode="OBSERVED", fields=(), locked=False),
    )
    return executor


@pytest.mark.timeout(120)
class TestAggregationTimeoutInvariance:
    """Aggregation arm: ``TriggerEvaluator._first_accept_time`` anchors to T_b."""

    def test_fire_instant_and_composition_invariant_across_takeover(self) -> None:
        clock = MockClock(start=_T0)
        db, factory = _make_factory()
        transform = _passthrough_flush_transform()
        processor_a = _agg_processor(factory, trigger={"timeout_seconds": TIMEOUT_SECONDS}, transform=transform, clock=clock)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        # T_b: leader A blocks two rows; the SAME process_row calls run the
        # journal-first intake, so adoption happens on the live path.
        mono_at_tb = clock.monotonic()
        for index in range(2):
            results = processor_a.process_row(
                row_index=index,
                source_row=_make_source_row({"value": index}),
                transforms=[transform],
                ctx=ctx,
                source_row_index=index,
                ingest_sequence=index,
            )
            assert [(r.outcome, r.path) for r in results] == [(None, TerminalPath.BUFFERED)]
        assert processor_a.get_aggregation_buffer_count(AGG_NODE) == 2

        # (1) Frame A anchor == the clamped wall→monotonic transform of T_b.
        evaluator_a = processor_a._aggregation_executor._nodes[AGG_NODE].trigger
        assert evaluator_a._first_accept_time == pytest.approx(mono_at_tb)

        # Frame A batch composition (the open DRAFT batch A's adoptions filled).
        batches_a = factory.execution.get_batches(RUN_ID)
        assert len(batches_a) == 1
        members_a = {(m.token_id, m.ordinal) for m in factory.execution.get_batch_members(batches_a[0].batch_id)}
        assert len(members_a) == 2

        # (2) Frame A at T_b+timeout-ε: must not fire.
        clock.advance(TIMEOUT_SECONDS - 0.5)
        should_fire_a, _ = processor_a.check_aggregation_timeout(AGG_NODE)
        assert should_fire_a is False

        # ── Takeover mid-window: usurp the seat, restore as leader B. ──────
        _usurp_seat(db, RUN_ID, clock)
        processor_b = _agg_processor(
            factory,
            trigger={"timeout_seconds": TIMEOUT_SECONDS},
            transform=_passthrough_flush_transform(),
            clock=clock,
            barrier_restore=_restore_context(),
        )

        # (3) Composition identical: restore created NO new batch and NO new
        # members — the durable membership set is byte-identical.
        batches_b = factory.execution.get_batches(RUN_ID)
        assert [b.batch_id for b in batches_b] == [batches_a[0].batch_id]
        members_b = {(m.token_id, m.ordinal) for m in factory.execution.get_batch_members(batches_a[0].batch_id)}
        assert members_b == members_a
        assert processor_b.get_aggregation_buffer_count(AGG_NODE) == 2

        # (1) Frame B anchor == the SAME transform of the SAME durable stamp.
        evaluator_b = processor_b._aggregation_executor._nodes[AGG_NODE].trigger
        assert evaluator_b._first_accept_time == pytest.approx(mono_at_tb)
        assert evaluator_b.batch_count == 2

        # (2) Frame B at the SAME instant (T_b+timeout-ε): must not fire.
        should_fire_b, _ = processor_b.check_aggregation_timeout(AGG_NODE)
        assert should_fire_b is False

        # (4) T_b+timeout+ε: BOTH frames flip between the same two clock
        # readings — the fire-instant difference across the takeover is zero.
        clock.advance(1.0)
        should_fire_b, trigger_b = processor_b.check_aggregation_timeout(AGG_NODE)
        assert should_fire_b is True
        assert trigger_b is TriggerType.TIMEOUT
        should_fire_a, trigger_a = processor_a.check_aggregation_timeout(AGG_NODE)
        assert should_fire_a is True, "frame A fires at the SAME instant (pure read; A is deposed but its memory is frame evidence)"
        assert trigger_a is TriggerType.TIMEOUT


@pytest.mark.timeout(120)
class TestCoalesceTimeoutInvariance:
    """Coalesce mirror: ``first_arrival`` anchors to T_b in both frames."""

    def test_first_arrival_anchor_and_fire_instant_invariant_across_takeover(self) -> None:
        clock = MockClock(start=_T0)
        db, factory = _make_factory()
        executor_a = _real_coalesce_executor(factory, clock)
        processor_a = _make_processor(
            factory,
            coalesce_executor=executor_a,
            coalesce_node_ids={CoalesceName("merge"): COALESCE_NODE},
            node_step_map={COALESCE_NODE: 2},
            clock=clock,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        # T_b: branch-a's BLOCKED row is deposited (live hold stashed exactly
        # as the drain would) and adopted by leader A's intake in the same
        # clock instant.
        token_a = TokenInfo(row_id="row-1", token_id="tok-branch-a", row_data=make_row({"amount": 1}), branch_name="a")
        mono_at_tb = clock.monotonic()
        _persist_blocked_scheduler_work(
            factory, processor_a, token_a, node_id=COALESCE_NODE, barrier_key="merge", adopted=False, coalesce_name="merge"
        )
        processor_a._live_barrier_holds[token_a.token_id] = _LiveBarrierHold(token=token_a, barrier_key="merge")
        intake_results = processor_a.run_barrier_intake(ctx)
        assert intake_results == []

        # (1) Frame A anchor: the live-path accept was backdated to T_b.
        pending_a = executor_a._pending[("merge", "row-1")]
        assert pending_a.first_arrival == pytest.approx(mono_at_tb)

        # (2) Frame A at T_b+timeout-ε: no timeout fire (pure when not firing).
        clock.advance(TIMEOUT_SECONDS - 0.5)
        assert executor_a.check_timeouts("merge") == []

        # ── Takeover mid-window. ────────────────────────────────────────────
        _usurp_seat(db, RUN_ID, clock)
        executor_b = _real_coalesce_executor(factory, clock)
        processor_b = _make_processor(
            factory,
            coalesce_executor=executor_b,
            coalesce_node_ids={CoalesceName("merge"): COALESCE_NODE},
            node_step_map={COALESCE_NODE: 2},
            clock=clock,
            barrier_restore=_restore_context(),
        )
        assert processor_b.has_blocked_barrier_work() is True

        # (1) Frame B anchor: restored from the SAME durable barrier_blocked_at.
        pending_b = executor_b._pending[("merge", "row-1")]
        assert pending_b.first_arrival == pytest.approx(mono_at_tb)
        assert set(pending_b.branches) == {"a"}

        # (2) Frame B at the SAME instant: no fire.
        assert executor_b.check_timeouts("merge") == []

        # (4) T_b+timeout+ε: frame B fires — best_effort merges the arrived
        # branch. Frame A's fire predicate (now - first_arrival ≥ timeout)
        # flips at the same instant; asserted arithmetically because actually
        # firing BOTH executors would double-record the consumed branch's
        # terminal outcomes.
        clock.advance(1.0)
        fired = executor_b.check_timeouts("merge")
        assert len(fired) == 1
        assert fired[0].merged_token is not None
        assert {token.token_id for token in fired[0].consumed_tokens} == {"tok-branch-a"}
        assert clock.monotonic() - pending_a.first_arrival >= TIMEOUT_SECONDS, "frame A's fire instant is the same clock reading"
