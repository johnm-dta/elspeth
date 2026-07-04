# tests/unit/engine/test_journal_restore.py
"""Unit tests for the barrier journal-restore boundary (journal_restore.py).

The *JournalRestorer classes are the crash-resume hydration seam extracted
from CoalesceExecutor / AggregationExecutor. The executor-level suites
(test_coalesce_executor.py, test_executors.py) already pin the corruption
guards and facade behavior; this file pins the restorer boundary itself —
the PAYLOAD FLOW (journal rows → rehydrated tokens with resume provenance),
the monotonic-anchor math, the frozen-state contract, and the
validate-before-mutate discipline across the extraction seam.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.contracts.barrier_scalars import AggregationNodeScalars, CoalescePendingScalars
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.scheduler import TokenWorkItem, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.config import AggregationSettings, CoalesceSettings, TriggerConfig
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.engine.clock import MockClock
from elspeth.engine.coalesce_executor import CoalesceExecutor
from elspeth.engine.executors.aggregation import AggregationExecutor
from elspeth.engine.journal_restore import (
    AggregationJournalRestorer,
    CoalesceJournalRestorer,
    RestoredAggregationState,
    RestoredCoalesceState,
)
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenManager

# Reference instant for journal-restore tests (tz-aware, like barrier_blocked_at).
_JOURNAL_T0 = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)


def _journal_payload(data: dict[str, Any], contract: SchemaContract | None = None) -> str:
    """Build a REAL journal row payload via the serializer mark_blocked rows carry.

    Round-trip fidelity through serialize_row_payload/deserialize_row_payload is
    the property the restorers depend on — fixtures must not shortcut it.
    """
    if contract is None:
        contract = SchemaContract(mode="OBSERVED", fields=(), locked=True)
    return TokenSchedulerRepository.serialize_row_payload(PipelineRow(data, contract))


def _blocked_item(
    *,
    token_id: str,
    row_id: str,
    blocked_at: datetime | None,
    payload: str | None = None,
    branch_name: str | None = None,
    coalesce_name: str | None = None,
    barrier_key: str | None = None,
    node_id: str = "node-1",
    attempt: int = 0,
) -> TokenWorkItem:
    """Build a BLOCKED journal row as list_blocked_barrier_items returns them."""
    return TokenWorkItem(
        work_item_id=f"wi-{token_id}",
        run_id="run_1",
        token_id=token_id,
        row_id=row_id,
        node_id=node_id,
        step_index=0,
        ingest_sequence=0,
        row_payload_json=payload if payload is not None else _journal_payload({"amount": 100}),
        status=TokenWorkStatus.BLOCKED,
        attempt=attempt,
        available_at=_JOURNAL_T0,
        created_at=_JOURNAL_T0,
        updated_at=_JOURNAL_T0,
        barrier_key=barrier_key if barrier_key is not None else (coalesce_name or node_id),
        branch_name=branch_name,
        fork_group_id=None,
        join_group_id=None,
        expand_group_id=None,
        coalesce_node_id=node_id if coalesce_name is not None else None,
        coalesce_name=coalesce_name,
        barrier_blocked_at=blocked_at,
    )


def _coalesce_settings(name: str = "merge", branches: list[str] | None = None) -> CoalesceSettings:
    return CoalesceSettings(
        name=name,
        branches=branches if branches is not None else ["a", "b"],
        policy="require_all",
        merge="union",
        on_success="default",
    )


def _coalesce_restorer(
    *,
    settings: dict[str, CoalesceSettings] | None = None,
    node_ids: dict[str, NodeID] | None = None,
    completed_pairs: set[tuple[str, str]] | None = None,
    clock: MockClock | None = None,
) -> CoalesceJournalRestorer:
    execution = MagicMock(spec=ExecutionRepository)
    execution.get_completed_row_ids_for_nodes.return_value = completed_pairs if completed_pairs is not None else set()
    return CoalesceJournalRestorer(
        settings=settings if settings is not None else {"merge": _coalesce_settings()},
        node_ids=node_ids if node_ids is not None else {"merge": NodeID("co-1")},
        execution=execution,
        run_id="run_1",
        clock=clock if clock is not None else MockClock(start=100.0),
    )


class TestCoalesceJournalRestorer:
    """Payload flow and hydration math for the coalesce restore boundary."""

    def test_rehydrates_token_payloads_with_resume_provenance(self) -> None:
        """Journal payloads flow into restored tokens; audit-derived resume stamps land per token."""
        restorer = _coalesce_restorer()
        items = [
            _blocked_item(
                token_id="t_a",
                row_id="row_1",
                branch_name="a",
                coalesce_name="merge",
                blocked_at=_JOURNAL_T0,
                payload=_journal_payload({"amount": 1, "src": "branch_a"}),
            ),
            _blocked_item(
                token_id="t_b",
                row_id="row_1",
                branch_name="b",
                coalesce_name="merge",
                blocked_at=_JOURNAL_T0,
                payload=_journal_payload({"amount": 2, "src": "branch_b"}),
            ),
        ]

        restored = restorer.restore(
            items=items,
            scalars={},
            state_ids={"t_a": "s_a", "t_b": "s_b"},
            attempt_offsets={"t_a": 3, "t_b": 5},
            resume_checkpoint_id="cp-7",
            now=_JOURNAL_T0,
        )

        assert isinstance(restored, RestoredCoalesceState)
        assert restored.token_count == 2
        (group,) = restored.pending
        assert group.key == ("merge", "row_1")
        by_branch = {b.branch_name: b for b in group.branches}
        assert set(by_branch) == {"a", "b"}
        # Payloads survive the journal round-trip verbatim.
        assert by_branch["a"].token.row_data.to_dict() == {"amount": 1, "src": "branch_a"}
        assert by_branch["b"].token.row_data.to_dict() == {"amount": 2, "src": "branch_b"}
        # Lineage and audit-derived resume state land on the right token.
        assert by_branch["a"].token.token_id == "t_a"
        assert by_branch["a"].token.branch_name == "a"
        assert by_branch["a"].token.resume_attempt_offset == 3
        assert by_branch["b"].token.resume_attempt_offset == 5
        assert by_branch["a"].token.resume_checkpoint_id == "cp-7"
        assert by_branch["a"].state_id == "s_a"
        assert by_branch["b"].state_id == "s_b"

    def test_anchors_arrival_times_on_monotonic_scale_preserving_offsets(self) -> None:
        """first_arrival = monotonic_now - pending age; branch offsets preserve blocked-at deltas."""
        restorer = _coalesce_restorer(clock=MockClock(start=100.0))
        items = [
            _blocked_item(token_id="t_a", row_id="row_1", branch_name="a", coalesce_name="merge", blocked_at=_JOURNAL_T0),
            _blocked_item(
                token_id="t_b",
                row_id="row_1",
                branch_name="b",
                coalesce_name="merge",
                blocked_at=_JOURNAL_T0 + timedelta(seconds=10),
            ),
        ]

        restored = restorer.restore(
            items=items,
            scalars={},
            state_ids={"t_a": "s_a", "t_b": "s_b"},
            attempt_offsets={"t_a": 1, "t_b": 1},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0 + timedelta(seconds=30),
        )

        (group,) = restored.pending
        # Oldest branch blocked 30s before now: anchor rewinds 30s from monotonic 100.
        assert group.first_arrival == pytest.approx(70.0)
        by_branch = {b.branch_name: b for b in group.branches}
        assert by_branch["a"].arrival_time == pytest.approx(70.0)
        assert by_branch["b"].arrival_time == pytest.approx(80.0)

    def test_loss_only_scalars_become_zero_arrival_pending_group(self) -> None:
        """A non-completed lost-branches scalar with no journal rows restores as an empty-branch group."""
        restorer = _coalesce_restorer(clock=MockClock(start=100.0))

        restored = restorer.restore(
            items=[],
            scalars={("merge", "row_9"): CoalescePendingScalars(lost_branches={"a": "error_routed"})},
            state_ids={},
            attempt_offsets={},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0,
        )

        (group,) = restored.pending
        assert group.key == ("merge", "row_9")
        assert group.branches == ()
        assert dict(group.lost_branches) == {"a": "error_routed"}
        # Zero-arrival keys anchor at restore time, not a journal stamp.
        assert group.first_arrival == pytest.approx(100.0)
        assert restored.token_count == 0

    def test_journal_groups_and_loss_only_scalars_coexist(self) -> None:
        """A loss-only scalar for key B must AUGMENT journal-backed groups for key A, not replace them.

        Guards the accumulation seam: a refactor that rebinds or clears the
        pending collection in the scalar-only loop silently discards the
        rehydrated arrived-branch tokens on a real crash-resume where both
        populations coexist.
        """
        restorer = _coalesce_restorer(clock=MockClock(start=100.0))

        restored = restorer.restore(
            items=[
                _blocked_item(token_id="t_a", row_id="row_1", branch_name="a", coalesce_name="merge", blocked_at=_JOURNAL_T0),
            ],
            scalars={("merge", "row_9"): CoalescePendingScalars(lost_branches={"b": "error_routed"})},
            state_ids={"t_a": "s_a"},
            attempt_offsets={"t_a": 1},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0,
        )

        by_key = {group.key: group for group in restored.pending}
        assert set(by_key) == {("merge", "row_1"), ("merge", "row_9")}
        assert [b.branch_name for b in by_key[("merge", "row_1")].branches] == ["a"]
        assert by_key[("merge", "row_9")].branches == ()
        assert dict(by_key[("merge", "row_9")].lost_branches) == {"b": "error_routed"}
        assert restored.token_count == 1

    def test_branches_preserve_journal_item_order(self) -> None:
        """DTO branches keep journal grouping order so the executor's rebuilt dict iterates identically."""
        restorer = _coalesce_restorer(settings={"merge": _coalesce_settings(branches=["a", "b", "c"])})

        restored = restorer.restore(
            items=[
                _blocked_item(token_id="t_c", row_id="row_1", branch_name="c", coalesce_name="merge", blocked_at=_JOURNAL_T0),
                _blocked_item(token_id="t_a", row_id="row_1", branch_name="a", coalesce_name="merge", blocked_at=_JOURNAL_T0),
            ],
            scalars={},
            state_ids={"t_c": "s_c", "t_a": "s_a"},
            attempt_offsets={"t_c": 1, "t_a": 1},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0,
        )

        (group,) = restored.pending
        # Journal order (c before a), NOT alphabetical or settings order.
        assert [b.branch_name for b in group.branches] == ["c", "a"]

    def test_backward_skew_clamps_anchor_but_preserves_branch_offsets(self) -> None:
        """Multi-branch group under backward wall-clock skew: anchor clamps at monotonic_now, deltas survive."""
        restorer = _coalesce_restorer(clock=MockClock(start=100.0))

        restored = restorer.restore(
            items=[
                _blocked_item(
                    token_id="t_a",
                    row_id="row_1",
                    branch_name="a",
                    coalesce_name="merge",
                    blocked_at=_JOURNAL_T0 + timedelta(seconds=40),  # blocked AFTER "now"
                ),
                _blocked_item(
                    token_id="t_b",
                    row_id="row_1",
                    branch_name="b",
                    coalesce_name="merge",
                    blocked_at=_JOURNAL_T0 + timedelta(seconds=50),
                ),
            ],
            scalars={},
            state_ids={"t_a": "s_a", "t_b": "s_b"},
            attempt_offsets={"t_a": 1, "t_b": 1},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0,  # wall clock stepped backward past both stamps
        )

        (group,) = restored.pending
        # Clamp: the oldest stamp is in the monotonic future, so the anchor
        # pins to monotonic_now rather than rewinding into the future...
        assert group.first_arrival == pytest.approx(100.0)
        by_branch = {b.branch_name: b for b in group.branches}
        assert by_branch["a"].arrival_time == pytest.approx(100.0)
        # ...while the 10s blocked-at delta between branches is preserved.
        assert by_branch["b"].arrival_time == pytest.approx(110.0)

    def test_completed_keys_map_landscape_node_ids_to_coalesce_names(self) -> None:
        """Landscape (node_id, row_id) pairs come back as (coalesce_name, row_id); foreign nodes drop."""
        restorer = _coalesce_restorer(
            node_ids={"merge": NodeID("co-1")},
            completed_pairs={("co-1", "row_0"), ("co-1", "row_9"), ("other-node", "row_x")},
        )

        restored = restorer.restore(
            items=[],
            scalars={},
            state_ids={},
            attempt_offsets={},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0,
        )

        assert set(restored.completed_keys) == {("merge", "row_0"), ("merge", "row_9")}
        assert restored.pending == ()

    def test_restored_state_is_frozen(self) -> None:
        """The returned state object is immutable — fields and lost_branches reject mutation."""
        restorer = _coalesce_restorer()
        restored = restorer.restore(
            items=[
                _blocked_item(token_id="t_a", row_id="row_1", branch_name="a", coalesce_name="merge", blocked_at=_JOURNAL_T0),
            ],
            scalars={("merge", "row_1"): CoalescePendingScalars(lost_branches={"b": "lost"})},
            state_ids={"t_a": "s_a"},
            attempt_offsets={"t_a": 1},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0,
        )

        with pytest.raises(FrozenInstanceError):
            restored.token_count = 5  # type: ignore[misc]
        (group,) = restored.pending
        with pytest.raises(FrozenInstanceError):
            group.first_arrival = 0.0  # type: ignore[misc]
        with pytest.raises(TypeError):
            group.lost_branches["c"] = "mutated"  # type: ignore[index]


class TestCoalesceFacadeValidateBeforeMutate:
    """The extraction seam must preserve all-or-nothing restore on the executor."""

    def _make_executor(self, settings: CoalesceSettings | None = None) -> CoalesceExecutor:
        execution = MagicMock(spec=ExecutionRepository)
        execution.get_completed_row_ids_for_nodes.return_value = set()
        execution.has_completed_row_for_node.return_value = False
        executor = CoalesceExecutor(
            execution=execution,
            span_factory=MagicMock(spec=SpanFactory),
            token_manager=MagicMock(spec=TokenManager),
            run_id="run_1",
            step_resolver=lambda node_id: 1,
            data_flow=MagicMock(spec=DataFlowRepository),
            clock=MockClock(start=100.0),
        )
        executor.register_coalesce(settings if settings is not None else _coalesce_settings(), NodeID("co-1"))
        return executor

    def test_failed_restore_leaves_prior_executor_state_intact(self) -> None:
        """A corrupt journal row raises BEFORE the executor discards restored state."""
        executor = self._make_executor()
        executor.restore_from_journal(
            items=[_blocked_item(token_id="t_a", row_id="row_1", branch_name="a", coalesce_name="merge", blocked_at=_JOURNAL_T0)],
            scalars={},
            state_ids={"t_a": "s_a"},
            attempt_offsets={"t_a": 1},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0,
        )
        assert ("merge", "row_1") in executor._pending

        with pytest.raises(AuditIntegrityError, match="NULL barrier_blocked_at"):
            executor.restore_from_journal(
                items=[_blocked_item(token_id="t_b", row_id="row_2", branch_name="b", coalesce_name="merge", blocked_at=None)],
                scalars={},
                state_ids={"t_b": "s_b"},
                attempt_offsets={"t_b": 1},
                resume_checkpoint_id="cp-1",
                now=_JOURNAL_T0,
            )

        # The failed second restore must not have cleared the first.
        assert ("merge", "row_1") in executor._pending

    def test_facade_applies_journal_groups_and_loss_only_scalars_together(self) -> None:
        """Both populations land in executor._pending from one restore call."""
        executor = self._make_executor()
        executor.restore_from_journal(
            items=[_blocked_item(token_id="t_a", row_id="row_1", branch_name="a", coalesce_name="merge", blocked_at=_JOURNAL_T0)],
            scalars={("merge", "row_9"): CoalescePendingScalars(lost_branches={"b": "error_routed"})},
            state_ids={"t_a": "s_a"},
            attempt_offsets={"t_a": 1},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0,
        )

        assert set(executor._pending) == {("merge", "row_1"), ("merge", "row_9")}
        assert list(executor._pending[("merge", "row_1")].branches) == ["a"]
        assert executor._pending[("merge", "row_9")].branches == {}
        assert executor._pending[("merge", "row_9")].lost_branches == {"b": "error_routed"}

    def test_restored_lost_branches_stay_mutable_for_live_loss_notifications(self) -> None:
        """The facade must thaw the frozen DTO's lost_branches into a mutable dict.

        notify_branch_lost mutates pending.lost_branches in place; if the
        apply step ever hands the executor the restorer's MappingProxyType
        directly, the first post-resume branch loss on a restored key crashes
        with TypeError instead of recording the loss.
        """
        executor = self._make_executor(_coalesce_settings(branches=["a", "b", "c"]))
        executor.restore_from_journal(
            items=[_blocked_item(token_id="t_a", row_id="row_1", branch_name="a", coalesce_name="merge", blocked_at=_JOURNAL_T0)],
            scalars={("merge", "row_1"): CoalescePendingScalars(lost_branches={"b": "error_routed"})},
            state_ids={"t_a": "s_a"},
            attempt_offsets={"t_a": 1},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0,
        )

        # In-place mutation of the restored loss record must succeed; under
        # require_all the second loss makes the merge impossible, so the key
        # fails with BOTH losses recorded in the audit metadata.
        outcome = executor.notify_branch_lost("merge", "row_1", "c", "error_routed")

        assert outcome is not None
        assert outcome.held is False
        assert outcome.failure_reason is not None
        assert outcome.coalesce_metadata is not None
        assert dict(outcome.coalesce_metadata.branches_lost) == {"b": "error_routed", "c": "error_routed"}


def _agg_settings() -> AggregationSettings:
    return AggregationSettings(
        name="test_agg",
        plugin="batch_stats",
        input="default",
        on_error="discard",
        trigger=TriggerConfig(count=3),
    )


class TestAggregationJournalRestorer:
    """Payload flow and latch decisions for the aggregation restore boundary."""

    def test_rehydrates_tokens_in_member_order_with_latch(self) -> None:
        """Tokens restore in batch_members.ordinal order; the latch carries batch age and offsets."""
        restorer = AggregationJournalRestorer(run_id="run_1")
        node_id = NodeID("agg-1")
        items = [
            _blocked_item(token_id="t1", row_id="r1", node_id="agg-1", blocked_at=_JOURNAL_T0, payload=_journal_payload({"v": 1})),
            _blocked_item(
                token_id="t2",
                row_id="r2",
                node_id="agg-1",
                blocked_at=_JOURNAL_T0 + timedelta(seconds=5),
                payload=_journal_payload({"v": 2}),
            ),
        ]

        restored = restorer.restore(
            node_id=node_id,
            items=items,
            member_order=["t2", "t1"],  # authoritative accept order differs from journal order
            batch_id="batch-1",
            accepted_count_total=7,
            completed_flush_count=2,
            scalars=AggregationNodeScalars(count_fire_offset=1.5, condition_fire_offset=None),
            attempt_offsets={"t1": 2, "t2": 4},
            resume_checkpoint_id="cp-3",
            now=_JOURNAL_T0 + timedelta(seconds=20),
        )

        assert isinstance(restored, RestoredAggregationState)
        assert [t.token_id for t in restored.tokens] == ["t2", "t1"]
        assert [t.row_data.to_dict() for t in restored.tokens] == [{"v": 2}, {"v": 1}]
        assert [t.resume_attempt_offset for t in restored.tokens] == [4, 2]
        assert all(t.resume_checkpoint_id == "cp-3" for t in restored.tokens)
        assert restored.batch_id == "batch-1"
        assert restored.accepted_count_total == 7
        assert restored.completed_flush_count == 2
        latch = restored.trigger_latch
        assert latch is not None
        assert latch.batch_count == 2
        # Age derives from the OLDEST blocked row (t1 at T0), 20s before now.
        assert latch.elapsed_age_seconds == pytest.approx(20.0)
        assert latch.count_fire_offset == pytest.approx(1.5)
        assert latch.condition_fire_offset is None
        assert restored.elapsed_age_seconds == pytest.approx(20.0)

    def test_counter_only_node_drops_stale_scalars_without_latch(self) -> None:
        """No buffered rows: latches are batch-scoped, so checkpoint scalars are stale and dropped."""
        restorer = AggregationJournalRestorer(run_id="run_1")

        restored = restorer.restore(
            node_id=NodeID("agg-1"),
            items=[],
            member_order=[],
            batch_id=None,
            accepted_count_total=9,
            completed_flush_count=3,
            scalars=AggregationNodeScalars(count_fire_offset=2.0, condition_fire_offset=4.0),
            attempt_offsets={},
            resume_checkpoint_id="cp-3",
            now=_JOURNAL_T0,
        )

        assert restored.tokens == ()
        assert restored.batch_id is None
        assert restored.trigger_latch is None
        assert restored.elapsed_age_seconds == 0.0
        assert restored.accepted_count_total == 9
        assert restored.completed_flush_count == 3

    def test_clock_skew_clamps_batch_age_at_zero(self) -> None:
        """A wall-clock backward step must not produce a negative restored batch age."""
        restorer = AggregationJournalRestorer(run_id="run_1")

        restored = restorer.restore(
            node_id=NodeID("agg-1"),
            items=[
                _blocked_item(
                    token_id="t1",
                    row_id="r1",
                    node_id="agg-1",
                    blocked_at=_JOURNAL_T0 + timedelta(seconds=30),  # blocked AFTER "now"
                )
            ],
            member_order=["t1"],
            batch_id="batch-1",
            accepted_count_total=1,
            completed_flush_count=0,
            scalars=AggregationNodeScalars(count_fire_offset=None, condition_fire_offset=None),
            attempt_offsets={"t1": 1},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0,
        )

        latch = restored.trigger_latch
        assert latch is not None
        assert latch.elapsed_age_seconds == 0.0

    def test_restored_state_is_frozen(self) -> None:
        """The returned state object is immutable."""
        restorer = AggregationJournalRestorer(run_id="run_1")
        restored = restorer.restore(
            node_id=NodeID("agg-1"),
            items=[],
            member_order=[],
            batch_id=None,
            accepted_count_total=0,
            completed_flush_count=0,
            scalars=AggregationNodeScalars(count_fire_offset=None, condition_fire_offset=None),
            attempt_offsets={},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0,
        )

        with pytest.raises(FrozenInstanceError):
            restored.batch_id = "mutated"  # type: ignore[misc]


class TestAggregationFacadeValidateBeforeMutate:
    """The extraction seam must preserve validate-before-mutate on the executor."""

    def _make_executor(self) -> tuple[AggregationExecutor, NodeID]:
        node_id = NodeID("agg-1")
        execution = MagicMock(spec=ExecutionRepository)
        executor = AggregationExecutor(
            execution,
            MagicMock(spec=SpanFactory),
            lambda nid: 1,
            run_id="run_1",
            aggregation_settings={node_id: _agg_settings()},
            clock=MockClock(start=100.0),
        )
        return executor, node_id

    def test_failed_restore_leaves_node_state_intact(self) -> None:
        """A membership mismatch raises BEFORE any node buffers or counters change."""
        executor, node_id = self._make_executor()
        executor.restore_from_journal(
            node_id=node_id,
            items=[_blocked_item(token_id="t1", row_id="r1", node_id="agg-1", blocked_at=_JOURNAL_T0)],
            member_order=["t1"],
            batch_id="batch-1",
            accepted_count_total=1,
            completed_flush_count=0,
            scalars=AggregationNodeScalars(count_fire_offset=None, condition_fire_offset=None),
            attempt_offsets={"t1": 1},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0,
        )
        assert executor.get_buffer_count(node_id) == 1
        assert executor.get_batch_id(node_id) == "batch-1"

        with pytest.raises(AuditIntegrityError, match="disagree about batch membership"):
            executor.restore_from_journal(
                node_id=node_id,
                items=[_blocked_item(token_id="t2", row_id="r2", node_id="agg-1", blocked_at=_JOURNAL_T0)],
                member_order=["t2", "t_ghost"],  # batch_members disagrees with the journal
                batch_id="batch-2",
                accepted_count_total=2,
                completed_flush_count=0,
                scalars=AggregationNodeScalars(count_fire_offset=None, condition_fire_offset=None),
                attempt_offsets={"t2": 1},
                resume_checkpoint_id="cp-1",
                now=_JOURNAL_T0,
            )

        # The failed second restore must not have touched the applied state.
        assert executor.get_buffer_count(node_id) == 1
        assert executor.get_batch_id(node_id) == "batch-1"

    def test_facade_wires_buffered_count_not_accepted_total_into_trigger(self) -> None:
        """The trigger's restored batch_count is the BUFFERED row count, never the cumulative accept counter.

        accepted_count_total deliberately diverges from len(tokens) here
        (7 vs 2): with a count=3 trigger, wiring the cumulative counter into
        restore_from_checkpoint would latch a phantom flush (7 >= 3) for a
        batch that only has 2 buffered rows.
        """
        executor, node_id = self._make_executor()  # trigger count=3
        executor.restore_from_journal(
            node_id=node_id,
            items=[
                _blocked_item(token_id="t1", row_id="r1", node_id="agg-1", blocked_at=_JOURNAL_T0),
                _blocked_item(token_id="t2", row_id="r2", node_id="agg-1", blocked_at=_JOURNAL_T0),
            ],
            member_order=["t1", "t2"],
            batch_id="batch-1",
            accepted_count_total=7,
            completed_flush_count=2,
            scalars=AggregationNodeScalars(count_fire_offset=None, condition_fire_offset=None),
            attempt_offsets={"t1": 1, "t2": 1},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0,
        )
        assert executor.should_flush(node_id) is False

        # Control: three buffered rows genuinely meet the count trigger.
        executor2, node_id2 = self._make_executor()
        executor2.restore_from_journal(
            node_id=node_id2,
            items=[
                _blocked_item(token_id="t1", row_id="r1", node_id="agg-1", blocked_at=_JOURNAL_T0),
                _blocked_item(token_id="t2", row_id="r2", node_id="agg-1", blocked_at=_JOURNAL_T0),
                _blocked_item(token_id="t3", row_id="r3", node_id="agg-1", blocked_at=_JOURNAL_T0),
            ],
            member_order=["t1", "t2", "t3"],
            batch_id="batch-1",
            accepted_count_total=3,
            completed_flush_count=0,
            scalars=AggregationNodeScalars(count_fire_offset=None, condition_fire_offset=None),
            attempt_offsets={"t1": 1, "t2": 1, "t3": 1},
            resume_checkpoint_id="cp-0",
            now=_JOURNAL_T0,
        )
        assert executor2.should_flush(node_id2) is True
