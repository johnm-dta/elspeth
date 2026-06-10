# tests/unit/engine/orchestrator/test_rc6_checkpoint_interrupted_edge.py
"""Shutdown-checkpoint behavior when aggregation buffers are flush-emptied.

``AggregationExecutor.get_checkpoint_state()`` includes counter-only nodes
(empty ``tokens``, non-zero ``accepted_count_total``/``completed_flush_count``)
so pagination metadata survives resume even immediately after a successful
flush.

History: the original RC6 edge here was the token-anchor selection chain
(commit f487d7b13 fixed a ``tokens[-1]`` IndexError on counter-only nodes).
The anchor itself was deleted as vestigial (F2, 2026-06-10) — the entire
fallback chain is gone, so the crash surface no longer exists. What survives,
and what these tests pin:

- the full aggregation state (including counter-only nodes) must still be
  persisted whenever any node is present, because resume restores the state
  wholesale;
- a shutdown with no buffered tokens anywhere still writes a recovery
  checkpoint (the former no-anchor skip arm is gone).
"""

from __future__ import annotations

from unittest.mock import Mock

from elspeth.contracts.aggregation_checkpoint import (
    AggregationCheckpointState,
    AggregationNodeCheckpoint,
    AggregationTokenCheckpoint,
)
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
    pending_tokens: dict[str, list] | None = None,
) -> Mock:
    processor = Mock()
    processor.get_aggregation_checkpoint_state.return_value = aggregation_state
    processor.get_coalesce_checkpoint_state.return_value = None

    loop_ctx = Mock()
    loop_ctx.processor = processor
    loop_ctx.pending_tokens = pending_tokens if pending_tokens is not None else {"default": []}
    return loop_ctx


class TestFlushEmptiedAggregationCheckpoints:
    """checkpoint_interrupted_progress with counter-only aggregation nodes."""

    def test_counter_only_node_persists_counters(self) -> None:
        """A flush-emptied aggregation node must not crash the shutdown checkpoint.

        The counter-only aggregation state is persisted so resume restores the
        durable counters (flush_index / rows_seen_total provenance).
        """
        coordinator = _make_coordinator()
        state = AggregationCheckpointState(version="5.0", nodes={"agg_emptied": _counter_only_node()})
        loop_ctx = _loop_ctx(state)

        coordinator.checkpoint_interrupted_progress(
            run_id="run-counter-only",
            loop_ctx=loop_ctx,
        )

        coordinator._checkpoint_manager.create_checkpoint.assert_called_once()
        kwargs = coordinator._checkpoint_manager.create_checkpoint.call_args.kwargs
        assert kwargs["aggregation_state"] is state

    def test_mixed_counter_only_and_buffered_nodes_persist_full_state(self) -> None:
        """With multiple aggregation nodes (some flush-emptied, some buffered),
        the full state — counter-only nodes included — is persisted wholesale."""
        coordinator = _make_coordinator()
        state = AggregationCheckpointState(
            version="5.0",
            nodes={
                "agg_emptied": _counter_only_node(),
                "agg_buffered": _buffered_node("tok-buffered"),
            },
        )
        loop_ctx = _loop_ctx(state)

        coordinator.checkpoint_interrupted_progress(
            run_id="run-mixed-agg",
            loop_ctx=loop_ctx,
        )

        coordinator._checkpoint_manager.create_checkpoint.assert_called_once()
        kwargs = coordinator._checkpoint_manager.create_checkpoint.call_args.kwargs
        assert kwargs["aggregation_state"] is state

    def test_no_buffered_tokens_anywhere_still_checkpoints(self) -> None:
        """No buffered tokens and no pending sink tokens: the shutdown
        checkpoint is still written (the former no-anchor skip arm is gone),
        and never IndexErrors on the counter-only node."""
        coordinator = _make_coordinator()
        state = AggregationCheckpointState(version="5.0", nodes={"agg_emptied": _counter_only_node()})
        loop_ctx = _loop_ctx(state)

        coordinator.checkpoint_interrupted_progress(
            run_id="run-no-anchor",
            loop_ctx=loop_ctx,
        )

        coordinator._checkpoint_manager.create_checkpoint.assert_called_once()
        kwargs = coordinator._checkpoint_manager.create_checkpoint.call_args.kwargs
        assert kwargs["run_id"] == "run-no-anchor"
        assert kwargs["aggregation_state"] is state
