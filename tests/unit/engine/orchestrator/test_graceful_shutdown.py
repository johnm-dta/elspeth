# tests/unit/engine/orchestrator/test_graceful_shutdown.py
"""Unit tests for graceful shutdown contracts and signal handler context.

These tests don't need database access — they verify:
- GracefulShutdownError contract
- INTERRUPTED enum values
- Signal handler context manager (install/restore)
"""

from __future__ import annotations

import signal
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import create_autospec, patch

from elspeth.contracts import RunStatus
from elspeth.contracts.barrier_scalars import BarrierScalars
from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
from elspeth.contracts.errors import GracefulShutdownError
from elspeth.contracts.events import RunCompletionStatus
from elspeth.core.checkpoint import CheckpointManager


@dataclass(slots=True)
class _BarrierScalarsProcessor:
    scalars: BarrierScalars
    terminalized_batches: list[tuple[str, ...]] = field(default_factory=list)

    def get_barrier_scalars(self) -> BarrierScalars:
        return self.scalars

    def mark_sink_bound_scheduler_terminal_many(self, token_ids: tuple[str, ...]) -> None:
        self.terminalized_batches.append(token_ids)


@dataclass(frozen=True, slots=True)
class _LoopContextSlice:
    processor: _BarrierScalarsProcessor
    pending_tokens: dict[str, list[object]]


@dataclass(frozen=True, slots=True)
class _SinkSlice:
    _on_write_failure: str | None = None


def _checkpoint_config(*, enabled: bool = True) -> RuntimeCheckpointConfig:
    return RuntimeCheckpointConfig(enabled=enabled, frequency=1, checkpoint_interval=None)


def _checkpoint_manager_mock() -> CheckpointManager:
    return create_autospec(CheckpointManager, instance=True)


class TestGracefulShutdownError:
    """Tests for GracefulShutdownError contract."""

    def test_error_attributes(self) -> None:
        """Error carries rows_processed and run_id."""
        err = GracefulShutdownError(rows_processed=42, run_id="run-abc")
        assert err.rows_processed == 42
        assert err.run_id == "run-abc"
        assert "42" in str(err)
        assert "run-abc" in str(err)

    def test_error_message_includes_resume_hint(self) -> None:
        """Error message includes resume command."""
        err = GracefulShutdownError(rows_processed=10, run_id="run-xyz")
        assert "elspeth resume run-xyz --execute" in str(err)

    def test_error_is_exception(self) -> None:
        """GracefulShutdownError is an Exception subclass."""
        err = GracefulShutdownError(rows_processed=0, run_id="run-0")
        assert isinstance(err, Exception)

    def test_error_carries_outcome_counters(self) -> None:
        """GracefulShutdownError carries real outcome counters for RunSummary.

        Regression: The graceful-shutdown handler hard-coded succeeded/failed/
        quarantined/routed to 0 in RunSummary because GracefulShutdownError
        only carried rows_processed. Now it must carry full counters.
        """
        err = GracefulShutdownError(
            rows_processed=100,
            run_id="run-abc",
            rows_succeeded=80,
            rows_failed=10,
            rows_quarantined=5,
            rows_routed_success=5,
            rows_routed_failure=0,
            routed_destinations={"archive": 3, "review": 2},
        )
        assert err.rows_succeeded == 80
        assert err.rows_failed == 10
        assert err.rows_quarantined == 5
        assert err.rows_routed_success == 5
        assert err.rows_routed_failure == 0
        assert err.routed_destinations == {"archive": 3, "review": 2}

    def test_error_counters_default_to_zero(self) -> None:
        """Backwards-compatible: omitting counters defaults to zero."""
        err = GracefulShutdownError(rows_processed=42, run_id="run-xyz")
        assert err.rows_succeeded == 0
        assert err.rows_failed == 0
        assert err.rows_quarantined == 0
        assert err.rows_routed_success == 0
        assert err.rows_routed_failure == 0
        assert err.routed_destinations == {}


class TestRunStatusInterrupted:
    """Tests for INTERRUPTED enum values."""

    def test_run_status_has_interrupted(self) -> None:
        assert RunStatus.INTERRUPTED == RunStatus.INTERRUPTED
        assert RunStatus.INTERRUPTED.value == "interrupted"

    def test_run_completion_status_has_interrupted(self) -> None:
        assert RunCompletionStatus.INTERRUPTED == RunCompletionStatus.INTERRUPTED
        assert RunCompletionStatus.INTERRUPTED.value == "interrupted"


class TestShutdownHandlerContext:
    """Tests for shutdown_handler_context() signal handler management.

    The context manager is a standalone module function (no orchestrator
    state) — tests exercise it directly, which is also the production call
    path (``core.py`` invokes ``shutdown_handler_context()`` directly).
    """

    def test_handler_restores_original_signals(self) -> None:
        """After context exits, signal handlers are restored."""
        from elspeth.engine.orchestrator.shutdown import shutdown_handler_context

        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        with shutdown_handler_context() as event:
            # Inside: handlers should be different from original
            current_sigint = signal.getsignal(signal.SIGINT)
            assert current_sigint != original_sigint
            assert not event.is_set()

        # After: handlers should be restored
        assert signal.getsignal(signal.SIGINT) == original_sigint
        assert signal.getsignal(signal.SIGTERM) == original_sigterm

    def test_context_yields_unset_event(self) -> None:
        """Context manager yields a threading.Event that starts unset."""
        from elspeth.engine.orchestrator.shutdown import shutdown_handler_context

        with shutdown_handler_context() as event:
            assert isinstance(event, threading.Event)
            assert not event.is_set()

    def test_handler_sets_event_on_signal(self) -> None:
        """Signal handler sets the event when invoked."""
        from elspeth.engine.orchestrator.shutdown import shutdown_handler_context

        with shutdown_handler_context() as event:
            assert not event.is_set()
            # Simulate signal by calling the handler directly
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler)
            handler(signal.SIGINT, None)
            assert event.is_set()

    def test_second_signal_restores_default_handler(self) -> None:
        """After first signal, SIGINT handler is restored to default (force-kill)."""
        from elspeth.engine.orchestrator.shutdown import shutdown_handler_context

        with shutdown_handler_context():
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler)
            handler(signal.SIGINT, None)

            # After first signal, SIGINT should now be default_int_handler
            assert signal.getsignal(signal.SIGINT) == signal.default_int_handler


class TestCheckpointInterruptedProgress:
    """Tests for checkpoint_interrupted_progress shutdown-checkpoint behavior."""

    def test_creates_checkpoint_even_with_no_buffered_state(self) -> None:
        """Shutdown always writes a recovery checkpoint; no state → NULL scalars.

        With the token anchor deleted (F2) and blob state retired (F1), a
        shutdown with no in-flight barrier state still produces a resumable
        checkpoint row, and the empty BarrierScalars from the live executors
        (``has_state`` False) persists ``barrier_scalars_json IS NULL``.
        """
        from sqlalchemy import select

        from elspeth.contracts.barrier_scalars import BarrierScalars
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.landscape.schema import checkpoints_table
        from elspeth.engine.orchestrator.checkpointing import CheckpointCoordinator
        from tests.fixtures.factories import make_graph_linear
        from tests.fixtures.landscape import leader_coordination_token, make_recorder_with_run

        setup = make_recorder_with_run(run_id="run-test-123")
        try:
            coordinator = CheckpointCoordinator(
                checkpoint_manager=CheckpointManager(setup.db),
                checkpoint_config=_checkpoint_config(),
            )
            coordinator.set_active_graph(make_graph_linear())
            # Checkpoint writes fail closed without the run's leader token
            # (elspeth-fab455790d); bind the seat begin_run minted.
            coordinator.bind_coordination(leader_coordination_token(setup.factory, "run-test-123"))

            # Live executors with no latched triggers / no recorded losses
            # compose an empty BarrierScalars.
            processor = _BarrierScalarsProcessor(BarrierScalars(aggregation={}, coalesce={}))

            loop_ctx = _LoopContextSlice(
                processor=processor,
                pending_tokens={"default": []},  # No pending tokens
            )

            coordinator.checkpoint_interrupted_progress(
                run_id="run-test-123",
                loop_ctx=loop_ctx,
            )

            with setup.db.engine.connect() as conn:
                rows = conn.execute(select(checkpoints_table)).fetchall()
            assert len(rows) == 1
            assert rows[0].run_id == "run-test-123"
            assert rows[0].sequence_number == 1
            assert rows[0].barrier_scalars_json is None  # empty scalars → NULL
        finally:
            setup.db.close()

    def test_latched_count_trigger_persists_count_fire_offset(self) -> None:
        """A latched count trigger writes count_fire_offset into the scalars.

        Companion to the NULL case: when the live aggregation executor reports
        a latched count trigger, the shutdown checkpoint row carries the fire
        offset — and ONLY the trigger latches. Counters are NOT in the scalars;
        they derive from audit tables at restore (design D3).
        """
        import json

        from sqlalchemy import select

        from elspeth.contracts.barrier_scalars import AggregationNodeScalars, BarrierScalars
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.landscape.schema import checkpoints_table
        from elspeth.engine.orchestrator.checkpointing import CheckpointCoordinator
        from tests.fixtures.factories import make_graph_linear
        from tests.fixtures.landscape import leader_coordination_token, make_recorder_with_run

        setup = make_recorder_with_run(run_id="run-latched")
        try:
            coordinator = CheckpointCoordinator(
                checkpoint_manager=CheckpointManager(setup.db),
                checkpoint_config=_checkpoint_config(),
            )
            coordinator.set_active_graph(make_graph_linear())
            coordinator.bind_coordination(leader_coordination_token(setup.factory, "run-latched"))

            processor = _BarrierScalarsProcessor(
                BarrierScalars(
                    aggregation={"agg-1": AggregationNodeScalars(count_fire_offset=2.5, condition_fire_offset=None)},
                    coalesce={},
                )
            )

            loop_ctx = _LoopContextSlice(processor=processor, pending_tokens={"default": []})

            coordinator.checkpoint_interrupted_progress(
                run_id="run-latched",
                loop_ctx=loop_ctx,
            )

            with setup.db.engine.connect() as conn:
                rows = conn.execute(select(checkpoints_table)).fetchall()
            assert len(rows) == 1
            assert rows[0].barrier_scalars_json is not None
            persisted = json.loads(rows[0].barrier_scalars_json)
            node_entry = persisted["aggregation"]["agg-1"]
            assert node_entry["count_fire_offset"] == 2.5
            assert node_entry["condition_fire_offset"] is None
            # D3: trigger latches ONLY — no counters in the persisted scalars.
            assert set(node_entry) == {"_version", "count_fire_offset", "condition_fire_offset"}
        finally:
            setup.db.close()

    def test_sink_factory_passes_live_barrier_scalars(self) -> None:
        """The checkpoint-PROGRESS callback hands live executor scalars through.

        The progress callback reads barrier_scalars_source.get_barrier_scalars()
        (single composed accessor, F1 Task 2.4) and passes the BarrierScalars
        verbatim to create_checkpoint — including coalesce lost-branch records.
        Scheduler terminalization is a SEPARATE callback composed in sink_flush
        (elspeth-107a29d02e); this progress callback must NOT terminalize. That
        split is covered by
        test_pending_sink_terminalization_uses_per_token_scheduler_handoff.
        """
        from elspeth.contracts.barrier_scalars import BarrierScalars, CoalescePendingScalars
        from elspeth.contracts.identity import TokenInfo
        from elspeth.engine.orchestrator import Orchestrator
        from elspeth.testing import make_row
        from tests.fixtures.factories import make_graph_linear
        from tests.fixtures.landscape import make_landscape_db

        db = make_landscape_db()
        try:
            orchestrator = Orchestrator(db=db)
            orchestrator._checkpoints._checkpoint_config = _checkpoint_config()
            orchestrator._checkpoints._checkpoint_manager = _checkpoint_manager_mock()
            orchestrator._checkpoints._active_graph = make_graph_linear()
            orchestrator._checkpoints._sequence_number = 0
            from elspeth.contracts.coordination import CoordinationToken

            orchestrator._checkpoints.bind_coordination(CoordinationToken(run_id="run-x", worker_id="test-leader", leader_epoch=1))

            scalars = BarrierScalars(
                aggregation={},
                coalesce={("merge_1", "row-1"): CoalescePendingScalars(lost_branches={"branch_b": "transform_failed"})},
            )

            processor = _BarrierScalarsProcessor(scalars)

            factory = orchestrator._checkpoints.make_checkpoint_after_sink_factory("run-x", processor)
            callback = factory("sink_0")

            token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=make_row({"value": 1}))
            callback(token)
            callback.flush()

            orchestrator._checkpoints._checkpoint_manager.create_checkpoint.assert_called_once()
            call_kwargs = orchestrator._checkpoints._checkpoint_manager.create_checkpoint.call_args.kwargs
            assert call_kwargs["barrier_scalars"] is scalars
            # Scheduler terminalization is no longer this callback's concern
            # (elspeth-107a29d02e): the progress callback only checkpoints.
            assert processor.terminalized_batches == []
        finally:
            db.close()

    def test_pending_sink_terminalization_uses_per_token_scheduler_handoff(self) -> None:
        """Generated sink tokens must not be terminalized as scheduler work.

        A sink batch may mix scheduler-backed tokens with tokens generated after
        a scheduler barrier. Only tokens with a recorded PENDING_SINK handoff
        should be closed by the post-sink callback. Post-split
        (elspeth-107a29d02e) this discrimination lives in write_pending_to_sinks,
        which composes a scheduler-terminalization callback only for groups whose
        pending outcome carries scheduler_pending_sink — so the caller must supply
        the scheduler_terminalizer for terminalization to occur at all.
        """
        from elspeth.contracts import PendingOutcome, TokenInfo
        from elspeth.contracts.barrier_scalars import BarrierScalars
        from elspeth.contracts.enums import TerminalOutcome, TerminalPath
        from elspeth.engine.executors.sink import DiversionCounts
        from elspeth.engine.orchestrator import Orchestrator
        from elspeth.engine.orchestrator.types import ExecutionCounters
        from elspeth.testing import make_row
        from tests.fixtures.landscape import make_landscape_db

        db = make_landscape_db()
        try:
            orchestrator = Orchestrator(db=db)
            orchestrator._checkpoints._checkpoint_config = _checkpoint_config(enabled=False)

            sink = _SinkSlice()
            config = SimpleNamespace(sinks={"output": sink})

            processor = _BarrierScalarsProcessor(BarrierScalars(aggregation={}, coalesce={}))
            on_token_written_factory = orchestrator._checkpoints.make_checkpoint_after_sink_factory("run-x", processor)

            scheduler_token = TokenInfo(row_id="row-1", token_id="tok-scheduler", row_data=make_row({"value": 1}))
            generated_token = TokenInfo(row_id="row-2", token_id="tok-generated", row_data=make_row({"value": 2}))
            pending_tokens = {
                "output": [
                    (
                        generated_token,
                        PendingOutcome(
                            outcome=TerminalOutcome.SUCCESS,
                            path=TerminalPath.COALESCED,
                            scheduler_pending_sink=False,
                        ),
                    ),
                    (
                        scheduler_token,
                        PendingOutcome(
                            outcome=TerminalOutcome.SUCCESS,
                            path=TerminalPath.COALESCED,
                            scheduler_pending_sink=True,
                        ),
                    ),
                ]
            }

            def write_side_effect(*, tokens, on_token_written, **_kwargs):
                for token in tokens:
                    on_token_written(token)
                return None, DiversionCounts()

            with patch("elspeth.engine.executors.sink.SinkExecutor", autospec=True) as sink_executor_cls:
                sink_executor_cls.return_value.write.side_effect = write_side_effect

                orchestrator._sink_flush.write_pending_to_sinks(
                    factory=SimpleNamespace(execution=object(), data_flow=object()),
                    run_id="run-x",
                    config=config,
                    ctx=object(),
                    counters=ExecutionCounters(),
                    pending_tokens=pending_tokens,
                    sink_id_map={"output": "sink-output"},
                    edge_map={},
                    sink_step=1,
                    on_token_written_factory=on_token_written_factory,
                    scheduler_terminalizer=processor,
                )

            assert processor.terminalized_batches == [("tok-scheduler",)]
        finally:
            db.close()

    def test_shutdown_passes_coalesce_lost_branch_scalars_through(self) -> None:
        """Shutdown hands the composed BarrierScalars through verbatim.

        checkpoint_interrupted_progress reads processor.get_barrier_scalars()
        (single composed accessor, F1 Task 2.4) and passes the result
        unconditionally to create_checkpoint — coalesce lost-branch records
        included. (The blob-era completed_keys concept is gone: late-arrival
        detection derives from audit tables at restore, design D3.)
        """
        from elspeth.contracts.barrier_scalars import BarrierScalars, CoalescePendingScalars
        from elspeth.engine.orchestrator import Orchestrator
        from tests.fixtures.factories import make_graph_linear
        from tests.fixtures.landscape import make_landscape_db

        db = make_landscape_db()
        try:
            orchestrator = Orchestrator(db=db)
            orchestrator._checkpoints._checkpoint_config = _checkpoint_config()
            orchestrator._checkpoints._checkpoint_manager = _checkpoint_manager_mock()
            orchestrator._checkpoints._active_graph = make_graph_linear()
            orchestrator._checkpoints._sequence_number = 0
            from elspeth.contracts.coordination import CoordinationToken

            orchestrator._checkpoints.bind_coordination(
                CoordinationToken(run_id="run-lost-branches", worker_id="test-leader", leader_epoch=1)
            )

            scalars = BarrierScalars(
                aggregation={},
                coalesce={
                    ("merge_1", "row-1"): CoalescePendingScalars(lost_branches={"branch_b": "gate_routed_away"}),
                    ("merge_2", "row-2"): CoalescePendingScalars(lost_branches={"branch_c": "transform_failed"}),
                },
            )

            processor = _BarrierScalarsProcessor(scalars)

            loop_ctx = _LoopContextSlice(processor=processor, pending_tokens={"default": []})

            orchestrator._checkpoints.checkpoint_interrupted_progress(
                run_id="run-lost-branches",
                loop_ctx=loop_ctx,
            )

            orchestrator._checkpoints._checkpoint_manager.create_checkpoint.assert_called_once()
            call_kwargs = orchestrator._checkpoints._checkpoint_manager.create_checkpoint.call_args.kwargs
            assert call_kwargs["barrier_scalars"] is scalars
            assert set(call_kwargs["barrier_scalars"].coalesce) == {("merge_1", "row-1"), ("merge_2", "row-2")}
        finally:
            db.close()
