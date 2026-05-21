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
from unittest.mock import Mock

import structlog.testing

from elspeth.contracts import RunStatus
from elspeth.contracts.errors import GracefulShutdownError
from elspeth.contracts.events import RunCompletionStatus


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
    """Tests for _checkpoint_interrupted_progress warning when no checkpoint is possible."""

    def test_logs_warning_when_no_token_available(self) -> None:
        """Shutdown checkpoint skip must emit a structured warning, not silently return."""
        from elspeth.contracts.types import NodeID
        from elspeth.engine.orchestrator import Orchestrator
        from tests.fixtures.landscape import make_landscape_db

        db = make_landscape_db()
        try:
            orchestrator = Orchestrator(db=db)
            # Enable checkpointing so the method doesn't early-return on config check
            orchestrator._checkpoint_config = Mock()
            orchestrator._checkpoint_config.enabled = True
            orchestrator._checkpoint_manager = Mock()
            orchestrator._current_graph = Mock()
            orchestrator._sequence_number = 0

            # Build a LoopContext where all token resolution paths return None:
            # - no aggregation nodes
            # - no coalesce pending entries
            # - no pending sink tokens
            # - last_token_id is None
            mock_processor = Mock()
            mock_agg_state = Mock()
            mock_agg_state.nodes = {}  # Empty — no aggregation buffers
            mock_processor.get_aggregation_checkpoint_state.return_value = mock_agg_state
            mock_processor.get_coalesce_checkpoint_state.return_value = None

            loop_ctx = Mock()
            loop_ctx.processor = mock_processor
            loop_ctx.pending_tokens = {"default": []}  # No pending tokens
            loop_ctx.last_token_id = None

            with structlog.testing.capture_logs() as captured:
                orchestrator._checkpoint_interrupted_progress(
                    run_id="run-test-123",
                    loop_ctx=loop_ctx,
                    sink_id_map={},
                    source_id=NodeID("source_0"),
                )

            # Verify NO checkpoint was created
            orchestrator._checkpoint_manager.create_checkpoint.assert_not_called()

            # Verify warning was emitted via structlog
            events = [entry["event"] for entry in captured]
            assert "shutdown_checkpoint_skipped" in events, (
                f"Expected 'shutdown_checkpoint_skipped' warning in structlog output, got: {events}"
            )
        finally:
            db.close()

    def test_sink_factory_persists_coalesce_completed_keys_only(self) -> None:
        """Regression: _make_checkpoint_after_sink_factory must persist completed_keys.

        The sink checkpoint callback filtered on coalesce_state.pending,
        dropping completed_keys needed for late-arrival detection on resume.
        """
        from elspeth.contracts.coalesce_checkpoint import CoalesceCheckpointState
        from elspeth.contracts.identity import TokenInfo
        from elspeth.engine.orchestrator import Orchestrator
        from tests.fixtures.landscape import make_landscape_db

        db = make_landscape_db()
        try:
            orchestrator = Orchestrator(db=db)
            orchestrator._checkpoint_config = Mock()
            orchestrator._checkpoint_config.enabled = True
            orchestrator._checkpoint_config.frequency = 1  # every_row
            orchestrator._checkpoint_manager = Mock()
            orchestrator._current_graph = Mock()
            orchestrator._sequence_number = 0

            coalesce_state = CoalesceCheckpointState(
                version="1.0",
                pending=(),
                completed_keys=(("merge_1", "row-1"),),
            )

            mock_processor = Mock()
            mock_agg_state = Mock()
            mock_processor.get_aggregation_checkpoint_state.return_value = mock_agg_state
            mock_processor.get_coalesce_checkpoint_state.return_value = coalesce_state

            factory = orchestrator._make_checkpoint_after_sink_factory("run-x", mock_processor)
            callback = factory("sink_0")

            token = Mock(spec=TokenInfo)
            token.token_id = "tok-1"
            callback(token)

            orchestrator._checkpoint_manager.create_checkpoint.assert_called_once()
            call_kwargs = orchestrator._checkpoint_manager.create_checkpoint.call_args.kwargs
            assert call_kwargs["coalesce_state"] is coalesce_state
        finally:
            db.close()

    def test_persists_coalesce_state_with_completed_keys_only(self) -> None:
        """Regression: coalesce state with completed_keys but no pending must be persisted.

        Bug: both _make_checkpoint_after_sink_factory and _checkpoint_interrupted_progress
        filtered on `coalesce_state.pending`, dropping completed_keys needed for
        late-arrival detection on resume.
        """
        from elspeth.contracts.coalesce_checkpoint import CoalesceCheckpointState
        from elspeth.contracts.types import NodeID
        from elspeth.engine.orchestrator import Orchestrator
        from tests.fixtures.landscape import make_landscape_db

        db = make_landscape_db()
        try:
            orchestrator = Orchestrator(db=db)
            orchestrator._checkpoint_config = Mock()
            orchestrator._checkpoint_config.enabled = True
            orchestrator._checkpoint_manager = Mock()
            orchestrator._current_graph = Mock()
            orchestrator._sequence_number = 0

            # Coalesce state with NO pending but HAS completed_keys
            coalesce_state = CoalesceCheckpointState(
                version="1.0",
                pending=(),
                completed_keys=(("merge_1", "row-1"), ("merge_2", "row-2")),
            )

            mock_processor = Mock()
            mock_agg_state = Mock()
            mock_agg_state.nodes = {}
            mock_processor.get_aggregation_checkpoint_state.return_value = mock_agg_state
            mock_processor.get_coalesce_checkpoint_state.return_value = coalesce_state

            loop_ctx = Mock()
            loop_ctx.processor = mock_processor
            loop_ctx.pending_tokens = {"default": []}
            loop_ctx.last_token_id = "token-99"

            source_id = NodeID("source_0")
            orchestrator._checkpoint_interrupted_progress(
                run_id="run-completed-keys",
                loop_ctx=loop_ctx,
                sink_id_map={},
                source_id=source_id,
            )

            orchestrator._checkpoint_manager.create_checkpoint.assert_called_once()
            call_kwargs = orchestrator._checkpoint_manager.create_checkpoint.call_args.kwargs
            assert call_kwargs["coalesce_state"] is coalesce_state
            assert call_kwargs["coalesce_state"].completed_keys == (
                ("merge_1", "row-1"),
                ("merge_2", "row-2"),
            )
        finally:
            db.close()

    def test_creates_checkpoint_when_last_token_available(self) -> None:
        """When last_token_id is set, the fallback path creates a checkpoint."""
        from elspeth.contracts.types import NodeID
        from elspeth.engine.orchestrator import Orchestrator
        from tests.fixtures.landscape import make_landscape_db

        db = make_landscape_db()
        try:
            orchestrator = Orchestrator(db=db)
            orchestrator._checkpoint_config = Mock()
            orchestrator._checkpoint_config.enabled = True
            orchestrator._checkpoint_manager = Mock()
            orchestrator._current_graph = Mock()
            orchestrator._sequence_number = 0

            mock_processor = Mock()
            mock_agg_state = Mock()
            mock_agg_state.nodes = {}
            mock_processor.get_aggregation_checkpoint_state.return_value = mock_agg_state
            mock_processor.get_coalesce_checkpoint_state.return_value = None

            loop_ctx = Mock()
            loop_ctx.processor = mock_processor
            loop_ctx.pending_tokens = {"default": []}
            loop_ctx.last_token_id = "token-42"  # Has a last token

            source_id = NodeID("source_0")
            orchestrator._checkpoint_interrupted_progress(
                run_id="run-test-456",
                loop_ctx=loop_ctx,
                sink_id_map={},
                source_id=source_id,
            )

            # Checkpoint SHOULD have been created with the fallback token
            orchestrator._checkpoint_manager.create_checkpoint.assert_called_once()
            call_kwargs = orchestrator._checkpoint_manager.create_checkpoint.call_args
            assert call_kwargs.kwargs["token_id"] == "token-42"
            assert call_kwargs.kwargs["node_id"] == str(source_id)
        finally:
            db.close()

    def test_last_token_checkpoint_uses_source_that_produced_token(self) -> None:
        """Multi-source shutdown checkpoints must not fall back to the first source."""
        from elspeth.contracts.types import NodeID
        from elspeth.engine.orchestrator import Orchestrator
        from tests.fixtures.landscape import make_landscape_db

        db = make_landscape_db()
        try:
            orchestrator = Orchestrator(db=db)
            orchestrator._checkpoint_config = Mock()
            orchestrator._checkpoint_config.enabled = True
            orchestrator._checkpoint_manager = Mock()
            orchestrator._current_graph = Mock()
            orchestrator._sequence_number = 0

            mock_processor = Mock()
            mock_agg_state = Mock()
            mock_agg_state.nodes = {}
            mock_processor.get_aggregation_checkpoint_state.return_value = mock_agg_state
            mock_processor.get_coalesce_checkpoint_state.return_value = None

            loop_ctx = Mock()
            loop_ctx.processor = mock_processor
            loop_ctx.pending_tokens = {"default": []}
            loop_ctx.last_token_id = "token-from-source-2"
            loop_ctx.last_token_source_id = NodeID("source_2")

            orchestrator._checkpoint_interrupted_progress(
                run_id="run-test-source-2",
                loop_ctx=loop_ctx,
                sink_id_map={},
                source_id=NodeID("source_1"),
            )

            orchestrator._checkpoint_manager.create_checkpoint.assert_called_once()
            call_kwargs = orchestrator._checkpoint_manager.create_checkpoint.call_args
            assert call_kwargs.kwargs["token_id"] == "token-from-source-2"
            assert call_kwargs.kwargs["node_id"] == "source_2"
        finally:
            db.close()
