# tests/unit/engine/orchestrator/test_rc6_checkpoint_interrupted_edge.py
"""Shutdown-checkpoint behavior for the scalar barrier projection.

History: the original RC6 edge here was the token-anchor selection chain
(commit f487d7b13 fixed a ``tokens[-1]`` IndexError on counter-only nodes).
The anchor itself was deleted as vestigial (F2, 2026-06-10) — the entire
fallback chain is gone, so the crash surface no longer exists. F1 Task 1.2
then retired the blob persistence (checkpoint carries only BarrierScalars;
counters derive from audit tables at restore, design D3), Task 2.1 retired
the executor blob state, and Task 2.4 unified the write path: the processor
surfaces ONE composed ``get_barrier_scalars() -> BarrierScalars`` and the
coordinator hands it to ``create_checkpoint`` unconditionally (the manager
serializes NULL when ``has_state`` is False). What these tests pin:

- unlatched / counter-only aggregation nodes contribute NO persisted scalars
  (their counters are derivable; their fire offsets are None);
- latched nodes contribute per-node trigger-latch scalars;
- a shutdown with no latched barriers anywhere still writes a recovery
  checkpoint (the former no-anchor skip arm is gone) whose empty
  BarrierScalars persists as NULL.
"""

from __future__ import annotations

from unittest.mock import Mock

from elspeth.contracts.barrier_scalars import AggregationNodeScalars, BarrierScalars
from elspeth.contracts.types import NodeID
from elspeth.engine.orchestrator.checkpointing import CheckpointCoordinator


def _make_coordinator() -> CheckpointCoordinator:
    config = Mock()
    config.enabled = True
    coordinator = CheckpointCoordinator(checkpoint_manager=Mock(), checkpoint_config=config)
    coordinator.set_active_graph(Mock())
    return coordinator


def _loop_ctx(
    aggregation_scalars: dict[NodeID, AggregationNodeScalars],
    *,
    pending_tokens: dict[str, list] | None = None,
) -> Mock:
    processor = Mock()
    processor.get_barrier_scalars.return_value = BarrierScalars(
        aggregation={str(node_id): scalars for node_id, scalars in aggregation_scalars.items()},
        coalesce={},
    )

    loop_ctx = Mock()
    loop_ctx.processor = processor
    loop_ctx.pending_tokens = pending_tokens if pending_tokens is not None else {"default": []}
    return loop_ctx


class TestFlushEmptiedAggregationCheckpoints:
    """checkpoint_interrupted_progress with unlatched/counter-only aggregation nodes."""

    def test_unlatched_nodes_contribute_no_scalars(self) -> None:
        """A flush-emptied or unlatched aggregation node must not crash the shutdown checkpoint.

        F1 design D3: durable counters (flush_index / rows_seen_total
        provenance) derive from audit tables at restore, and unlatched
        triggers have no fire offsets — the executor emits no entry for such
        nodes, so the checkpoint is still written with an empty BarrierScalars
        (``has_state`` False → the manager persists NULL).
        """
        coordinator = _make_coordinator()
        loop_ctx = _loop_ctx({})  # executor emitted nothing — no latched nodes

        coordinator.checkpoint_interrupted_progress(
            run_id="run-counter-only",
            loop_ctx=loop_ctx,
        )

        coordinator._checkpoint_manager.create_checkpoint.assert_called_once()
        kwargs = coordinator._checkpoint_manager.create_checkpoint.call_args.kwargs
        assert isinstance(kwargs["barrier_scalars"], BarrierScalars)
        assert kwargs["barrier_scalars"].has_state is False

    def test_latched_nodes_persist_their_scalars(self) -> None:
        """Latched aggregation nodes surface as per-node BarrierScalars entries.

        The executor's get_barrier_scalars() already filters to latched nodes;
        the projection must carry them through verbatim.
        """
        coordinator = _make_coordinator()
        loop_ctx = _loop_ctx(
            {NodeID("agg_latched"): AggregationNodeScalars(count_fire_offset=0.25, condition_fire_offset=None)},
        )

        coordinator.checkpoint_interrupted_progress(
            run_id="run-mixed-agg",
            loop_ctx=loop_ctx,
        )

        coordinator._checkpoint_manager.create_checkpoint.assert_called_once()
        kwargs = coordinator._checkpoint_manager.create_checkpoint.call_args.kwargs
        scalars = kwargs["barrier_scalars"]
        assert isinstance(scalars, BarrierScalars)
        assert set(scalars.aggregation) == {"agg_latched"}
        assert scalars.aggregation["agg_latched"].count_fire_offset == 0.25
        assert scalars.coalesce == {}

    def test_no_buffered_tokens_anywhere_still_checkpoints(self) -> None:
        """No latched barriers and no pending sink tokens: the shutdown
        checkpoint is still written (the former no-anchor skip arm is gone)."""
        coordinator = _make_coordinator()
        loop_ctx = _loop_ctx({})

        coordinator.checkpoint_interrupted_progress(
            run_id="run-no-anchor",
            loop_ctx=loop_ctx,
        )

        coordinator._checkpoint_manager.create_checkpoint.assert_called_once()
        kwargs = coordinator._checkpoint_manager.create_checkpoint.call_args.kwargs
        assert kwargs["run_id"] == "run-no-anchor"
        assert kwargs["barrier_scalars"].has_state is False
