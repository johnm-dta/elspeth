# tests/unit/engine/orchestrator/test_rc6_checkpoint_interrupted_edge.py
"""Shutdown-checkpoint anchor selection when aggregation buffers are flush-emptied.

``AggregationExecutor.get_checkpoint_state()`` includes counter-only nodes
(empty ``tokens``, non-zero ``accepted_count_total``/``completed_flush_count``)
so pagination metadata survives resume even immediately after a successful
flush. ``checkpoint_interrupted_progress`` must therefore never assume the
first aggregation node has buffered tokens:

- the checkpoint anchor must come from a node that actually holds tokens,
  falling through to the coalesce / pending-sink / last-token anchors when
  every aggregation buffer is empty;
- the full aggregation state (including counter-only nodes) must still be
  persisted whenever any node is present, because resume restores the state
  wholesale regardless of which token anchors the checkpoint row.
"""

from __future__ import annotations

from unittest.mock import Mock

import structlog.testing

from elspeth.contracts.aggregation_checkpoint import (
    AggregationCheckpointState,
    AggregationNodeCheckpoint,
    AggregationTokenCheckpoint,
)
from elspeth.contracts.types import NodeID
from elspeth.engine.orchestrator.checkpointing import CheckpointCoordinator


def _make_coordinator() -> CheckpointCoordinator:
    config = Mock()
    config.enabled = True
    coordinator = CheckpointCoordinator(checkpoint_manager=Mock(), checkpoint_config=config)
    coordinator.set_active_graph(Mock())
    return coordinator


def _counter_only_node(*, accepted: int = 5, flushes: int = 1) -> AggregationNodeCheckpoint:
    """Post-flush snapshot: empty buffer, durable counters only."""
    return AggregationNodeCheckpoint(
        tokens=(),
        batch_id=None,
        elapsed_age_seconds=0.0,
        count_fire_offset=None,
        condition_fire_offset=None,
        accepted_count_total=accepted,
        completed_flush_count=flushes,
    )


def _buffered_node(token_id: str) -> AggregationNodeCheckpoint:
    token = AggregationTokenCheckpoint(
        token_id=token_id,
        row_id=f"row-{token_id}",
        branch_name=None,
        fork_group_id=None,
        join_group_id=None,
        expand_group_id=None,
        row_data={"value": 1},
        contract_version="cv-1",
        contract={"fields": {}},
    )
    return AggregationNodeCheckpoint(
        tokens=(token,),
        batch_id="batch-1",
        elapsed_age_seconds=0.5,
        count_fire_offset=None,
        condition_fire_offset=None,
        accepted_count_total=1,
        completed_flush_count=0,
    )


def _loop_ctx(
    aggregation_state: AggregationCheckpointState,
    *,
    last_token_id: str | None,
    last_token_source_id: NodeID | None,
    pending_tokens: dict[str, list] | None = None,
) -> Mock:
    processor = Mock()
    processor.get_aggregation_checkpoint_state.return_value = aggregation_state
    processor.get_coalesce_checkpoint_state.return_value = None

    loop_ctx = Mock()
    loop_ctx.processor = processor
    loop_ctx.pending_tokens = pending_tokens if pending_tokens is not None else {"default": []}
    loop_ctx.last_token_id = last_token_id
    loop_ctx.last_token_source_id = last_token_source_id
    return loop_ctx


class TestFlushEmptiedAggregationAnchors:
    """checkpoint_interrupted_progress with counter-only aggregation nodes."""

    def test_counter_only_node_anchors_to_last_token_and_persists_counters(self) -> None:
        """A flush-emptied aggregation node must not crash the shutdown checkpoint.

        Anchor falls through to the last processed token; the counter-only
        aggregation state is still persisted so resume restores the durable
        counters (flush_index / rows_seen_total provenance).
        """
        coordinator = _make_coordinator()
        state = AggregationCheckpointState(version="5.0", nodes={"agg_emptied": _counter_only_node()})
        loop_ctx = _loop_ctx(state, last_token_id="tok-last", last_token_source_id=NodeID("source_2"))

        coordinator.checkpoint_interrupted_progress(
            run_id="run-counter-only",
            loop_ctx=loop_ctx,
            sink_id_map={},
            source_id=NodeID("source_1"),
        )

        coordinator._checkpoint_manager.create_checkpoint.assert_called_once()
        kwargs = coordinator._checkpoint_manager.create_checkpoint.call_args.kwargs
        assert kwargs["token_id"] == "tok-last"
        assert kwargs["node_id"] == "source_2"
        assert kwargs["aggregation_state"] is state

    def test_anchor_skips_flush_emptied_node_and_uses_node_with_tokens(self) -> None:
        """With multiple aggregation nodes, the anchor must come from a node
        that actually holds buffered tokens — not whichever node happens to
        iterate first (dict-order-dependent IndexError otherwise)."""
        coordinator = _make_coordinator()
        state = AggregationCheckpointState(
            version="5.0",
            nodes={
                "agg_emptied": _counter_only_node(),
                "agg_buffered": _buffered_node("tok-buffered"),
            },
        )
        loop_ctx = _loop_ctx(state, last_token_id="tok-last", last_token_source_id=NodeID("source_1"))

        coordinator.checkpoint_interrupted_progress(
            run_id="run-mixed-agg",
            loop_ctx=loop_ctx,
            sink_id_map={},
            source_id=NodeID("source_1"),
        )

        coordinator._checkpoint_manager.create_checkpoint.assert_called_once()
        kwargs = coordinator._checkpoint_manager.create_checkpoint.call_args.kwargs
        assert kwargs["token_id"] == "tok-buffered"
        assert kwargs["node_id"] == "agg_buffered"
        assert kwargs["aggregation_state"] is state

    def test_counter_only_node_falls_through_to_pending_sink_anchor(self) -> None:
        """When no aggregation node has tokens, pending sink tokens anchor the
        checkpoint (same precedence as the no-aggregation path)."""
        coordinator = _make_coordinator()
        state = AggregationCheckpointState(version="5.0", nodes={"agg_emptied": _counter_only_node()})

        pending_token = Mock()
        pending_token.token_id = "tok-pending"
        loop_ctx = _loop_ctx(
            state,
            last_token_id="tok-last",
            last_token_source_id=NodeID("source_1"),
            pending_tokens={"out": [(pending_token, Mock())]},
        )

        coordinator.checkpoint_interrupted_progress(
            run_id="run-pending-anchor",
            loop_ctx=loop_ctx,
            sink_id_map={"out": NodeID("sink_out")},
            source_id=NodeID("source_1"),
        )

        coordinator._checkpoint_manager.create_checkpoint.assert_called_once()
        kwargs = coordinator._checkpoint_manager.create_checkpoint.call_args.kwargs
        assert kwargs["token_id"] == "tok-pending"
        assert kwargs["node_id"] == "sink_out"
        assert kwargs["aggregation_state"] is state

    def test_counter_only_node_with_no_anchor_skips_with_warning(self) -> None:
        """No buffered tokens anywhere and no last token: skip with the
        structured warning, never IndexError."""
        coordinator = _make_coordinator()
        state = AggregationCheckpointState(version="5.0", nodes={"agg_emptied": _counter_only_node()})
        loop_ctx = _loop_ctx(state, last_token_id=None, last_token_source_id=None)

        with structlog.testing.capture_logs() as captured:
            coordinator.checkpoint_interrupted_progress(
                run_id="run-no-anchor",
                loop_ctx=loop_ctx,
                sink_id_map={},
                source_id=NodeID("source_1"),
            )

        coordinator._checkpoint_manager.create_checkpoint.assert_not_called()
        events = [entry["event"] for entry in captured]
        assert "shutdown_checkpoint_skipped" in events
