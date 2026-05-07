"""Tests for telemetry.manager -- TelemetryManager event coordination."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from types import MappingProxyType
from unittest.mock import patch

import pytest

from elspeth.contracts.config.defaults import INTERNAL_DEFAULTS
from elspeth.contracts.enums import (
    BackpressureMode,
    CallStatus,
    CallType,
    RunStatus,
    TelemetryGranularity,
)
from elspeth.contracts.events import (
    ExternalCallCompleted,
    PhaseAction,
    PhaseChanged,
    PipelinePhase,
    RowCreated,
    RunFinished,
    RunStarted,
)
from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.manager import TelemetryManager
from tests.fixtures.telemetry import (
    FailingExporter,
    MockTelemetryConfig,
    TelemetryTestExporter,
)

# =============================================================================
# Constants and Factories
# =============================================================================

_NOW = datetime(2026, 1, 15, tzinfo=UTC)


def _lifecycle_event() -> RunStarted:
    return RunStarted(timestamp=_NOW, run_id="run-1", config_hash="h", source_plugin="csv")


def _run_finished_event() -> RunFinished:
    return RunFinished(
        timestamp=_NOW,
        run_id="run-1",
        status=RunStatus.COMPLETED,
        row_count=10,
        duration_ms=100.0,
    )


def _phase_changed_event() -> PhaseChanged:
    return PhaseChanged(
        timestamp=_NOW,
        run_id="run-1",
        phase=PipelinePhase.PROCESS,
        action=PhaseAction.PROCESSING,
    )


def _row_event() -> RowCreated:
    return RowCreated(
        timestamp=_NOW,
        run_id="run-1",
        row_id="r1",
        token_id="t1",
        content_hash="ch",
    )


def _external_call_event() -> ExternalCallCompleted:
    return ExternalCallCompleted(
        timestamp=_NOW,
        run_id="run-1",
        call_type=CallType.LLM,
        provider="test",
        status=CallStatus.SUCCESS,
        latency_ms=50.0,
        state_id="s1",
    )


def _wait_for_processing(manager: TelemetryManager, timeout: float = 5.0) -> None:
    """Wait for all queued events to be processed by the export thread."""
    manager._queue.join()


def _wait_until(condition: Callable[[], bool], *, timeout: float = 2.0, poll: float = 0.005) -> None:
    """Poll until condition is True, or raise after timeout."""
    import time

    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Condition not met within {timeout}s")
        time.sleep(poll)


# =============================================================================
# Initialization
# =============================================================================


class TestInitialization:
    def test_creates_with_empty_exporters(self) -> None:
        config = MockTelemetryConfig()
        manager = TelemetryManager(config, exporters=[])
        try:
            assert manager._exporters == []
        finally:
            manager.close()

    def test_creates_with_single_exporter(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            assert len(manager._exporters) == 1
            assert manager._exporters[0] is exporter
        finally:
            manager.close()

    def test_creates_with_multiple_exporters(self) -> None:
        config = MockTelemetryConfig()
        e1 = TelemetryTestExporter(name="a")
        e2 = TelemetryTestExporter(name="b")
        manager = TelemetryManager(config, exporters=[e1, e2])
        try:
            assert len(manager._exporters) == 2
        finally:
            manager.close()

    def test_export_thread_starts_and_is_ready(self) -> None:
        config = MockTelemetryConfig()
        manager = TelemetryManager(config, exporters=[])
        try:
            assert manager._export_thread.is_alive()
            assert manager._export_thread_ready.is_set()
        finally:
            manager.close()

    def test_queue_has_default_maxsize(self) -> None:
        config = MockTelemetryConfig()
        manager = TelemetryManager(config, exporters=[])
        try:
            assert manager._queue.maxsize == 1000
        finally:
            manager.close()

    def test_initial_health_metrics_are_zero(self) -> None:
        config = MockTelemetryConfig()
        manager = TelemetryManager(config, exporters=[])
        try:
            metrics = manager.health_metrics
            assert metrics["events_emitted"] == 0
            assert metrics["events_dropped"] == 0
            assert metrics["consecutive_total_failures"] == 0
        finally:
            manager.close()


# =============================================================================
# handle_event Basic Flow
# =============================================================================


class TestHandleEventBasic:
    def test_event_reaches_exporter(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert len(exporter.events) == 1
            assert isinstance(exporter.events[0], RunStarted)
        finally:
            manager.close()

    def test_multiple_events_processed_in_order(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            e1 = _lifecycle_event()
            e2 = _row_event()
            e3 = _external_call_event()
            manager.handle_event(e1)
            manager.handle_event(e2)
            manager.handle_event(e3)
            _wait_for_processing(manager)
            assert len(exporter.events) == 3
            assert exporter.events[0] is e1
            assert exporter.events[1] is e2
            assert exporter.events[2] is e3
        finally:
            manager.close()

    def test_event_delivered_to_multiple_exporters(self) -> None:
        config = MockTelemetryConfig()
        e1 = TelemetryTestExporter(name="a")
        e2 = TelemetryTestExporter(name="b")
        manager = TelemetryManager(config, exporters=[e1, e2])
        try:
            event = _lifecycle_event()
            manager.handle_event(event)
            _wait_for_processing(manager)
            assert len(e1.events) == 1
            assert len(e2.events) == 1
        finally:
            manager.close()

    def test_events_emitted_count_increments(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            manager.handle_event(_lifecycle_event())
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert manager.health_metrics["events_emitted"] == 2
        finally:
            manager.close()


# =============================================================================
# Granularity Filtering
# =============================================================================


class TestGranularityFiltering:
    def test_lifecycle_granularity_passes_lifecycle_events(self) -> None:
        config = MockTelemetryConfig(granularity=TelemetryGranularity.LIFECYCLE)
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert len(exporter.events) == 1
        finally:
            manager.close()

    def test_lifecycle_granularity_filters_row_events(self) -> None:
        config = MockTelemetryConfig(granularity=TelemetryGranularity.LIFECYCLE)
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            manager.handle_event(_row_event())
            # Send a lifecycle event after to confirm processing is working
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert len(exporter.events) == 1
            assert isinstance(exporter.events[0], RunStarted)
        finally:
            manager.close()

    def test_lifecycle_granularity_filters_external_call_events(self) -> None:
        config = MockTelemetryConfig(granularity=TelemetryGranularity.LIFECYCLE)
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            manager.handle_event(_external_call_event())
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert len(exporter.events) == 1
            assert isinstance(exporter.events[0], RunStarted)
        finally:
            manager.close()

    def test_rows_granularity_passes_lifecycle_and_row_events(self) -> None:
        config = MockTelemetryConfig(granularity=TelemetryGranularity.ROWS)
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            manager.handle_event(_lifecycle_event())
            manager.handle_event(_row_event())
            _wait_for_processing(manager)
            assert len(exporter.events) == 2
        finally:
            manager.close()

    def test_rows_granularity_filters_external_call_events(self) -> None:
        config = MockTelemetryConfig(granularity=TelemetryGranularity.ROWS)
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            manager.handle_event(_external_call_event())
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert len(exporter.events) == 1
            assert isinstance(exporter.events[0], RunStarted)
        finally:
            manager.close()

    def test_full_granularity_passes_all_events(self) -> None:
        config = MockTelemetryConfig(granularity=TelemetryGranularity.FULL)
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            manager.handle_event(_lifecycle_event())
            manager.handle_event(_row_event())
            manager.handle_event(_external_call_event())
            _wait_for_processing(manager)
            assert len(exporter.events) == 3
        finally:
            manager.close()


# =============================================================================
# Shutdown / Disabled Guards
# =============================================================================


class TestShutdownGuards:
    def test_after_close_handle_event_is_noop(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        manager.close()
        manager.handle_event(_lifecycle_event())
        # No events should be exported after close
        assert len(exporter.events) == 0

    def test_disabled_manager_handle_event_is_noop(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            manager._disabled = True
            manager.handle_event(_lifecycle_event())
            # Disabled manager rejects synchronously — no wait needed
            assert len(exporter.events) == 0
        finally:
            manager.close()

    def test_empty_exporters_handle_event_is_noop(self) -> None:
        config = MockTelemetryConfig()
        manager = TelemetryManager(config, exporters=[])
        try:
            manager.handle_event(_lifecycle_event())
            # Verify nothing was queued (queue should be empty since we skip)
            assert manager._queue.qsize() == 0
        finally:
            manager.close()

    def test_shutdown_event_prevents_new_events(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            manager._shutdown_event.set()
            manager.handle_event(_lifecycle_event())
            # Shutdown event set — handle_event rejects synchronously
            assert len(exporter.events) == 0
        finally:
            manager.close()


# =============================================================================
# DROP Backpressure Mode
# =============================================================================


def _small_queue_defaults() -> MappingProxyType[str, object]:
    """INTERNAL_DEFAULTS with telemetry queue_size=2 for backpressure tests."""
    return MappingProxyType(
        {
            **INTERNAL_DEFAULTS,
            "telemetry": MappingProxyType({**INTERNAL_DEFAULTS["telemetry"], "queue_size": 2}),
        }
    )


class TestDropBackpressure:
    def test_drop_mode_drops_oldest_when_queue_full(self) -> None:
        with patch("elspeth.telemetry.manager.INTERNAL_DEFAULTS", _small_queue_defaults()):
            config = MockTelemetryConfig(backpressure_mode=BackpressureMode.DROP)
            exporter = TelemetryTestExporter()
            manager = TelemetryManager(config, exporters=[exporter])
            try:
                # Block the export thread so it can't drain immediately
                blocker = threading.Event()
                original_dispatch = manager._dispatch_to_exporters

                def slow_dispatch(event):
                    blocker.wait(timeout=5.0)
                    original_dispatch(event)

                object.__setattr__(manager, "_dispatch_to_exporters", slow_dispatch)

                # First event is picked up by thread and blocks in dispatch
                e1 = RunStarted(timestamp=_NOW, run_id="run-1", config_hash="h", source_plugin="csv")
                e2 = RunStarted(timestamp=_NOW, run_id="run-2", config_hash="h", source_plugin="csv")
                e3 = RunStarted(timestamp=_NOW, run_id="run-3", config_hash="h", source_plugin="csv")
                e4 = RunStarted(timestamp=_NOW, run_id="run-4", config_hash="h", source_plugin="csv")

                manager.handle_event(e1)
                # Wait for export thread to dequeue e1 (blocks in slow_dispatch)
                _wait_until(lambda: manager._queue.empty(), timeout=2.0)

                # Fill queue to capacity, then overflow
                manager.handle_event(e2)
                manager.handle_event(e3)
                dropped_before = manager.health_metrics["events_dropped"]
                manager.handle_event(e4)
                dropped_after = manager.health_metrics["events_dropped"]
                assert dropped_after > dropped_before

                # Release exporter and verify oldest queued event (e2) was evicted
                blocker.set()
                object.__setattr__(manager, "_dispatch_to_exporters", original_dispatch)
                _wait_for_processing(manager)

                run_ids = [event.run_id for event in exporter.events]
                assert run_ids == ["run-1", "run-3", "run-4"]
            finally:
                if "blocker" in locals() and not blocker.is_set():
                    blocker.set()
                if "original_dispatch" in locals():
                    object.__setattr__(manager, "_dispatch_to_exporters", original_dispatch)
                manager.close()

    def test_drop_mode_increments_events_dropped(self) -> None:
        with patch("elspeth.telemetry.manager.INTERNAL_DEFAULTS", _small_queue_defaults()):
            config = MockTelemetryConfig(backpressure_mode=BackpressureMode.DROP)
            exporter = TelemetryTestExporter()
            manager = TelemetryManager(config, exporters=[exporter])
            try:
                blocker = threading.Event()
                original_dispatch = manager._dispatch_to_exporters

                def slow_dispatch(event):
                    blocker.wait(timeout=5.0)
                    original_dispatch(event)

                object.__setattr__(manager, "_dispatch_to_exporters", slow_dispatch)

                # First event occupies thread; queue fills with next two; fourth overflows
                manager.handle_event(_lifecycle_event())
                # Wait for export thread to dequeue first event
                _wait_until(lambda: manager._queue.empty(), timeout=2.0)
                manager.handle_event(_lifecycle_event())
                manager.handle_event(_lifecycle_event())

                initial_dropped = manager.health_metrics["events_dropped"]
                manager.handle_event(_lifecycle_event())
                assert manager.health_metrics["events_dropped"] > initial_dropped
            finally:
                if "blocker" in locals() and not blocker.is_set():
                    blocker.set()
                if "original_dispatch" in locals():
                    object.__setattr__(manager, "_dispatch_to_exporters", original_dispatch)
                manager.close()

    def test_drop_mode_replacement_keeps_join_blocked_until_replacement_queued(self) -> None:
        """Eviction must not let queue.join() observe a transient zero.

        Regression test for DROP-mode overflow replacement: if task_done() is
        called before put_nowait(replacement), a concurrent flush() can return
        before the replacement event is queued.
        """

        replacement_event = _row_event()

        class BlockingReplacementQueue(queue.Queue[object]):
            def __init__(self) -> None:
                super().__init__(maxsize=1)
                self.replacement_put_entered = threading.Event()
                self.allow_replacement_put = threading.Event()

            def put_nowait(self, item: object) -> None:
                if item is replacement_event:
                    self.replacement_put_entered.set()
                    assert self.allow_replacement_put.wait(timeout=2.0), "Timed out waiting to release replacement put"
                return super().put_nowait(item)

        class DummyManager:
            _LOG_INTERVAL = 100

            def __init__(self, q: BlockingReplacementQueue) -> None:
                self._queue = q
                self._dropped_lock = threading.Lock()
                self._events_dropped = 0
                self._last_logged_drop_count = 0

            def _log_drops_if_needed(self) -> None:
                if self._events_dropped - self._last_logged_drop_count >= self._LOG_INTERVAL:
                    self._last_logged_drop_count = self._events_dropped

        q = BlockingReplacementQueue()
        dummy = DummyManager(q)
        old_event = _lifecycle_event()
        q.put_nowait(old_event)

        flush_returned = threading.Event()

        def flush_waiter() -> None:
            q.join()
            flush_returned.set()

        flush_thread = threading.Thread(target=flush_waiter, name="flush-waiter")
        flush_thread.start()

        drop_thread = threading.Thread(
            target=TelemetryManager._drop_oldest_and_enqueue_newest,
            args=(dummy, replacement_event),
            name="drop-replace",
        )
        drop_thread.start()

        assert q.replacement_put_entered.wait(timeout=2.0), "Did not reach replacement put"

        # Critical assertion: join() must still be blocked while replacement
        # put is in progress. Old ordering (task_done then put) fails here.
        assert not flush_returned.is_set()

        q.allow_replacement_put.set()
        drop_thread.join(timeout=2.0)
        assert not drop_thread.is_alive()

        # Replacement is now queued and unfinished. join() must still block until
        # the replacement task is completed.
        assert not flush_returned.is_set()

        queued = q.get_nowait()
        assert queued is replacement_event
        q.task_done()

        flush_thread.join(timeout=2.0)
        assert flush_returned.is_set()

    def test_drop_mode_requeues_shutdown_sentinel_under_contention(self) -> None:
        """Shutdown sentinel must survive DROP-mode eviction races."""

        incoming_event = _row_event()
        interloper_event = _lifecycle_event()

        class SentinelContentionQueue(queue.Queue[object]):
            def __init__(self) -> None:
                super().__init__(maxsize=1)
                self._sentinel_requeue_attempts = 0

            def put_nowait(self, item: object) -> None:
                if item is None:
                    self._sentinel_requeue_attempts += 1
                    if self._sentinel_requeue_attempts == 1:
                        # Simulate another producer taking the slot between
                        # sentinel eviction and reinsertion.
                        super().put_nowait(interloper_event)
                        raise queue.Full
                return super().put_nowait(item)

        class DummyManager:
            _LOG_INTERVAL = 100

            def __init__(self, q: SentinelContentionQueue) -> None:
                self._queue = q
                self._dropped_lock = threading.Lock()
                self._events_dropped = 0
                self._last_logged_drop_count = 0

            def _log_drops_if_needed(self) -> None:
                if self._events_dropped - self._last_logged_drop_count >= self._LOG_INTERVAL:
                    self._last_logged_drop_count = self._events_dropped

        q = SentinelContentionQueue()
        dummy = DummyManager(q)
        queue.Queue.put_nowait(q, None)

        TelemetryManager._drop_oldest_and_enqueue_newest(dummy, incoming_event)  # type: ignore[arg-type]  # DummyManager duck-types TelemetryManager for isolated unit test

        queued = q.get_nowait()
        assert queued is None
        q.task_done()
        assert dummy._events_dropped == 2

    def test_drop_mode_degrades_gracefully_when_sentinel_requeue_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sentinel requeue failure degrades to dropped event, never raises."""

        incoming_event = _row_event()

        class DummyManager:
            _LOG_INTERVAL = 100

            def __init__(self, q: queue.Queue[object]) -> None:
                self._queue = q
                self._dropped_lock = threading.Lock()
                self._events_dropped = 0
                self._last_logged_drop_count = 0

            def _log_drops_if_needed(self) -> None:
                if self._events_dropped - self._last_logged_drop_count >= self._LOG_INTERVAL:
                    self._last_logged_drop_count = self._events_dropped

        q: queue.Queue[object] = queue.Queue(maxsize=1)
        dummy = DummyManager(q)
        q.put_nowait(None)

        def _raise_requeue_failure(_manager: object) -> None:
            raise RuntimeError("sentinel requeue failure")

        monkeypatch.setattr(TelemetryManager, "_requeue_shutdown_sentinel_or_raise", _raise_requeue_failure)

        # Must NOT raise — telemetry failures never propagate to callers
        TelemetryManager._drop_oldest_and_enqueue_newest(dummy, incoming_event)  # type: ignore[arg-type]  # DummyManager duck-types TelemetryManager for isolated unit test

        # Critical regression guard: queue task accounting must remain balanced.
        assert q.unfinished_tasks == 0
        q.join()

        # Event counted as dropped (not silently swallowed)
        assert dummy._events_dropped == 1


# =============================================================================
# Exporter Failure Isolation
# =============================================================================


class TestExporterFailureIsolation:
    def test_failing_exporter_does_not_block_healthy_exporter(self) -> None:
        config = MockTelemetryConfig()
        healthy = TelemetryTestExporter(name="healthy")
        failing = FailingExporter(name="failing")
        manager = TelemetryManager(config, exporters=[healthy, failing])
        try:
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert len(healthy.events) == 1
            assert failing.export_attempts == 1
        finally:
            manager.close()

    def test_partial_success_counts_as_emitted(self) -> None:
        config = MockTelemetryConfig()
        healthy = TelemetryTestExporter(name="healthy")
        failing = FailingExporter(name="failing")
        manager = TelemetryManager(config, exporters=[healthy, failing])
        try:
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert manager.health_metrics["events_emitted"] == 1
        finally:
            manager.close()

    def test_partial_failure_resets_consecutive_total_failures(self) -> None:
        config = MockTelemetryConfig()
        healthy = TelemetryTestExporter(name="healthy")
        failing = FailingExporter(name="failing")
        manager = TelemetryManager(config, exporters=[healthy, failing])
        try:
            # Artificially set some consecutive failures
            manager._consecutive_total_failures = 5
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert manager.health_metrics["consecutive_total_failures"] == 0
        finally:
            manager.close()

    def test_exporter_failure_tracked_per_name(self) -> None:
        config = MockTelemetryConfig()
        healthy = TelemetryTestExporter(name="healthy")
        failing = FailingExporter(name="bad-one")
        manager = TelemetryManager(config, exporters=[healthy, failing])
        try:
            manager.handle_event(_lifecycle_event())
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            failures = manager.health_metrics["exporter_failures"]
            assert failures.get("bad-one") == 2
            assert "healthy" not in failures
        finally:
            manager.close()


# =============================================================================
# Total Exporter Failure
# =============================================================================


class TestTotalExporterFailure:
    def test_all_exporters_fail_increments_events_dropped(self) -> None:
        config = MockTelemetryConfig()
        failing = FailingExporter(name="failing")
        manager = TelemetryManager(config, exporters=[failing])
        try:
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert manager.health_metrics["events_dropped"] == 1
            assert manager.health_metrics["events_emitted"] == 0
        finally:
            manager.close()

    def test_all_exporters_fail_increments_consecutive_total_failures(self) -> None:
        config = MockTelemetryConfig()
        failing = FailingExporter(name="failing")
        manager = TelemetryManager(config, exporters=[failing])
        try:
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert manager.health_metrics["consecutive_total_failures"] == 1
        finally:
            manager.close()

    def test_consecutive_total_failures_resets_on_success(self) -> None:
        config = MockTelemetryConfig()
        # Fails 2 times then succeeds
        exporter = FailingExporter(name="flaky", fail_count=2)
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            # Two failures
            manager.handle_event(_lifecycle_event())
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert manager.health_metrics["consecutive_total_failures"] == 2

            # Third call succeeds
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert manager.health_metrics["consecutive_total_failures"] == 0
        finally:
            manager.close()

    def test_multiple_failing_exporters_all_fail_counts_as_total_failure(self) -> None:
        config = MockTelemetryConfig()
        f1 = FailingExporter(name="failing-a")
        f2 = FailingExporter(name="failing-b")
        manager = TelemetryManager(config, exporters=[f1, f2])
        try:
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert manager.health_metrics["events_dropped"] == 1
            assert manager.health_metrics["consecutive_total_failures"] == 1
        finally:
            manager.close()

    def test_disabled_after_max_consecutive_failures_fail_on_total_false(self) -> None:
        # When all exporters fail max_consecutive_failures times, manager disables.
        # Circuit breaker threshold equals max_consecutive_failures, so they trip together.
        config = MockTelemetryConfig(fail_on_total_exporter_failure=False, max_consecutive_failures=5)
        failing = FailingExporter(name="failing")
        manager = TelemetryManager(config, exporters=[failing])
        try:
            # Send max_consecutive_failures events
            for _ in range(5):
                manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert manager._disabled is True
        finally:
            manager.close()

    def test_disabled_manager_stops_accepting_events(self) -> None:
        # Circuit breaker threshold equals max_consecutive_failures (both = 5 here)
        config = MockTelemetryConfig(fail_on_total_exporter_failure=False, max_consecutive_failures=5)
        failing = FailingExporter(name="failing")
        manager = TelemetryManager(config, exporters=[failing])
        try:
            # Trigger disable
            for _ in range(5):
                manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert manager._disabled is True

            dropped_before = manager.health_metrics["events_dropped"]
            manager.handle_event(_lifecycle_event())
            # Disabled manager rejects synchronously — no wait needed
            assert manager.health_metrics["events_dropped"] == dropped_before
        finally:
            manager.close()

    def test_telemetry_exporter_error_raised_on_flush_fail_on_total_true(self) -> None:
        # Circuit breaker threshold equals max_consecutive_failures (both = 5 here)
        config = MockTelemetryConfig(fail_on_total_exporter_failure=True, max_consecutive_failures=5)
        failing = FailingExporter(name="failing")
        manager = TelemetryManager(config, exporters=[failing])
        try:
            for _ in range(5):
                manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            # flush() will re-raise the stored exception
            with pytest.raises(TelemetryExporterError):
                manager.flush()
        finally:
            manager.close()

    def test_stored_exception_cleared_after_flush_reraise(self) -> None:
        # Circuit breaker threshold equals max_consecutive_failures (both = 5 here)
        config = MockTelemetryConfig(fail_on_total_exporter_failure=True, max_consecutive_failures=5)
        failing = FailingExporter(name="failing")
        manager = TelemetryManager(config, exporters=[failing])
        try:
            for _ in range(5):
                manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            with pytest.raises(TelemetryExporterError):
                manager.flush()
            # Second flush should not raise (exception was cleared)
            manager.flush()
        finally:
            manager.close()


# =============================================================================
# Health Metrics
# =============================================================================


class TestHealthMetrics:
    def test_returns_correct_structure(self) -> None:
        config = MockTelemetryConfig()
        manager = TelemetryManager(config, exporters=[])
        try:
            metrics = manager.health_metrics
            expected_keys = {
                "events_emitted",
                "events_dropped",
                "exporter_failures",
                "consecutive_total_failures",
                "queue_depth",
                "queue_maxsize",
                "circuit_breakers",
            }
            assert set(metrics.keys()) == expected_keys
        finally:
            manager.close()

    def test_events_emitted_accurate(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            for _ in range(5):
                manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert manager.health_metrics["events_emitted"] == 5
        finally:
            manager.close()

    def test_events_dropped_accurate_with_total_failure(self) -> None:
        config = MockTelemetryConfig()
        failing = FailingExporter(name="failing")
        manager = TelemetryManager(config, exporters=[failing])
        try:
            for _ in range(3):
                manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert manager.health_metrics["events_dropped"] == 3
        finally:
            manager.close()

    def test_queue_maxsize_matches_internal_default(self) -> None:
        config = MockTelemetryConfig()
        manager = TelemetryManager(config, exporters=[])
        try:
            assert manager.health_metrics["queue_maxsize"] == 1000
        finally:
            manager.close()

    def test_exporter_failures_per_name(self) -> None:
        config = MockTelemetryConfig()
        f1 = FailingExporter(name="exporter-a")
        f2 = FailingExporter(name="exporter-b")
        manager = TelemetryManager(config, exporters=[f1, f2])
        try:
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            failures = manager.health_metrics["exporter_failures"]
            assert failures["exporter-a"] == 1
            assert failures["exporter-b"] == 1
        finally:
            manager.close()

    def test_exporter_failures_is_copy(self) -> None:
        config = MockTelemetryConfig()
        manager = TelemetryManager(config, exporters=[])
        try:
            failures = manager.health_metrics["exporter_failures"]
            failures["injected"] = 999
            assert "injected" not in manager.health_metrics["exporter_failures"]
        finally:
            manager.close()

    def test_queue_depth_reflects_pending_items(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            # After processing, queue should be empty
            manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)
            assert manager.health_metrics["queue_depth"] == 0
        finally:
            manager.close()


# =============================================================================
# Flush
# =============================================================================


class TestFlush:
    def test_flush_waits_for_queue_drain(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            for _ in range(10):
                manager.handle_event(_lifecycle_event())
            manager.flush()
            assert len(exporter.events) == 10
        finally:
            manager.close()

    def test_flush_calls_flush_on_each_exporter(self) -> None:
        config = MockTelemetryConfig()
        e1 = TelemetryTestExporter(name="a")
        e2 = TelemetryTestExporter(name="b")
        manager = TelemetryManager(config, exporters=[e1, e2])
        try:
            manager.flush()
            assert e1.flush_count == 1
            assert e2.flush_count == 1
        finally:
            manager.close()

    def test_flush_exporter_failure_does_not_raise(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            # Make flush raise
            def bad_flush():
                raise ConnectionError("flush transport error")

            object.__setattr__(exporter, "flush", bad_flush)
            # Should not raise
            manager.flush()
        finally:
            manager.close()

    def test_flush_after_shutdown_skips_queue_join(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        manager.close()
        # flush() after close should not hang - it skips queue.join()
        manager.flush()
        # Exporter flush is still called even after shutdown
        assert exporter.flush_count == 1

    def test_flush_multiple_times_safe(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        try:
            manager.flush()
            manager.flush()
            manager.flush()
            assert exporter.flush_count == 3
        finally:
            manager.close()


# =============================================================================
# Close
# =============================================================================


class TestClose:
    def test_close_calls_close_on_each_exporter(self) -> None:
        config = MockTelemetryConfig()
        e1 = TelemetryTestExporter(name="a")
        e2 = TelemetryTestExporter(name="b")
        manager = TelemetryManager(config, exporters=[e1, e2])
        manager.close()
        assert e1.close_count == 1
        assert e2.close_count == 1

    def test_close_thread_exits(self) -> None:
        config = MockTelemetryConfig()
        manager = TelemetryManager(config, exporters=[])
        assert manager._export_thread.is_alive()
        manager.close()
        assert not manager._export_thread.is_alive()

    def test_close_safe_when_thread_already_dead(self) -> None:
        config = MockTelemetryConfig()
        manager = TelemetryManager(config, exporters=[])
        manager.close()
        # Second close should not raise or hang
        manager.close()

    def test_close_exporter_failure_does_not_raise(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()

        def bad_close():
            raise ConnectionError("close transport error")

        object.__setattr__(exporter, "close", bad_close)
        manager = TelemetryManager(config, exporters=[exporter])
        # Should not raise
        manager.close()

    def test_close_sets_shutdown_event(self) -> None:
        config = MockTelemetryConfig()
        manager = TelemetryManager(config, exporters=[])
        assert not manager._shutdown_event.is_set()
        manager.close()
        assert manager._shutdown_event.is_set()

    def test_close_processes_remaining_events_before_shutdown(self) -> None:
        config = MockTelemetryConfig()
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])
        for _ in range(5):
            manager.handle_event(_lifecycle_event())
        manager.close()
        # All events should have been processed before thread exited
        assert len(exporter.events) == 5


# =============================================================================
# Circuit Breaker Correctness
# =============================================================================


class TestCircuitBreakerCorrectness:
    """Tests for circuit breaker edge cases and correctness.

    Regression tests for:
    - P1: Open-breaker skips should NOT count toward total failure threshold
    - P2: Same-named exporter instances must have independent breakers
    """

    def test_same_named_exporters_have_independent_breakers(self) -> None:
        """Two exporter instances with the same name must have independent circuit breakers.

        Regression test: Before fix, both exporters shared a single breaker keyed by
        exporter.name, causing failures from one to affect the other.

        Real-world scenario: two OTLP endpoints (staging + prod), both named "otlp".
        """
        config = MockTelemetryConfig()

        # Create two exporters with the SAME name (simulating two otlp endpoints)
        exporter1 = TelemetryTestExporter(name="otlp")
        exporter2 = TelemetryTestExporter(name="otlp")

        manager = TelemetryManager(config, exporters=[exporter1, exporter2])
        try:
            # Verify each exporter has its own breaker
            assert len(manager._circuit_breakers) == 2, (
                f"Expected 2 independent breakers for 2 exporters, got {len(manager._circuit_breakers)}"
            )

            # The breakers should be keyed differently (not both "otlp")
            breaker_keys = list(manager._circuit_breakers.keys())
            assert breaker_keys[0] != breaker_keys[1], f"Breaker keys should be unique, but both are {breaker_keys}"
        finally:
            manager.close()

    def test_same_named_failing_exporter_does_not_affect_healthy_same_name(self) -> None:
        """Failures in one exporter should not trip another exporter's breaker.

        Even if both exporters have the same name, their circuit breakers
        must be independent.
        """
        config = MockTelemetryConfig()

        # One healthy, one failing — both named "otlp"
        healthy = TelemetryTestExporter(name="otlp")
        failing = FailingExporter(name="otlp")

        manager = TelemetryManager(config, exporters=[healthy, failing])
        try:
            # Emit enough events to trip the failing exporter's breaker
            for _ in range(10):
                manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)

            # Healthy exporter should have received all events
            assert len(healthy.events) == 10, f"Healthy exporter should receive all events, got {len(healthy.events)}"

            # The healthy exporter's breaker should still be CLOSED
            # Find the breaker for the healthy exporter
            for breaker in manager._circuit_breakers.values():
                # The healthy exporter's breaker should have 0 failures
                if breaker.failure_count == 0:
                    assert breaker.state.name == "CLOSED"
                    break
            else:
                pytest.fail("Could not find a breaker with 0 failures (healthy exporter)")
        finally:
            manager.close()

    def test_failing_exporter_isolated_by_circuit_breaker(self) -> None:
        """One failing exporter doesn't block healthy exporters.

        Circuit breakers isolate failures: when one exporter's breaker trips,
        events still flow to healthy exporters. This prevents one bad backend
        from bringing down all telemetry.

        Expected: After failing exporter's breaker trips, events continue to
        succeed via the healthy exporter, and consecutive_total_failures stays 0.
        """
        config = MockTelemetryConfig(
            fail_on_total_exporter_failure=True,
            max_consecutive_failures=5,
        )
        failing = FailingExporter(name="failing")
        healthy = TelemetryTestExporter(name="healthy")
        manager = TelemetryManager(config, exporters=[failing, healthy])
        try:
            # Emit 10 events — failing exporter trips after 5, healthy keeps working
            for _ in range(10):
                manager.handle_event(_lifecycle_event())
            _wait_for_processing(manager)

            # Healthy exporter succeeded on all 10 events
            assert healthy.event_count == 10, f"Expected 10 exports to healthy exporter, got {healthy.event_count}"
            # Failing exporter only saw 5 before breaker tripped
            assert failing.export_attempts == 5, f"Expected 5 exports to failing exporter (before breaker), got {failing.export_attempts}"
            # No total failures because healthy exporter always succeeded
            assert manager.health_metrics["consecutive_total_failures"] == 0, (
                "Expected 0 consecutive total failures (healthy exporter saved each event)"
            )
            # All 10 events emitted successfully
            assert manager.health_metrics["events_emitted"] == 10, "Expected 10 emitted events (healthy exporter succeeded)"
        finally:
            manager.close()
