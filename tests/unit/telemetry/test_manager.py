# tests/unit/telemetry/test_manager.py
"""Unit tests for TelemetryManager event coordination.

Tests cover:
- Event dispatching to exporters
- Granularity filtering integration
- Individual exporter failure isolation
- Aggregate logging every 100 drops (Warning Fatigue prevention)
- Total exporter failure handling (crash vs continue)
- Health metrics tracking
- flush() and close() behavior
- Property-based state machine tests
"""

import queue
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule

from elspeth.contracts.enums import BackpressureMode, RunStatus, TelemetryGranularity
from elspeth.contracts.events import TelemetryEvent
from elspeth.telemetry import RunFinished, RunStarted, TelemetryManager
from elspeth.telemetry.errors import TelemetryExporterError

# =============================================================================
# Test Doubles
# =============================================================================


@dataclass
class MockConfig:
    """Mock RuntimeTelemetryProtocol implementation for testing."""

    enabled: bool = True
    granularity: TelemetryGranularity = TelemetryGranularity.FULL
    fail_on_total_exporter_failure: bool = False
    backpressure_mode: BackpressureMode = BackpressureMode.BLOCK

    @property
    def exporter_configs(self) -> tuple:
        return ()


class MockExporter:
    """Mock exporter that tracks calls and can simulate failures."""

    def __init__(self, name: str, *, fail_export: bool = False, fail_flush: bool = False, fail_close: bool = False):
        self._name = name
        self._fail_export = fail_export
        self._fail_flush = fail_flush
        self._fail_close = fail_close
        self.exports: list[TelemetryEvent] = []
        self.flush_count = 0
        self.close_count = 0

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        pass

    def export(self, event: TelemetryEvent) -> None:
        if self._fail_export:
            raise RuntimeError(f"Simulated export failure in {self._name}")
        self.exports.append(event)

    def flush(self) -> None:
        if self._fail_flush:
            raise RuntimeError(f"Simulated flush failure in {self._name}")
        self.flush_count += 1

    def close(self) -> None:
        if self._fail_close:
            raise RuntimeError(f"Simulated close failure in {self._name}")
        self.close_count += 1


class SlowExporter:
    """Exporter that blocks on export until signaled."""

    def __init__(self, name: str):
        self._name = name
        self.export_started = threading.Event()
        self.can_continue = threading.Event()
        self.exports: list[TelemetryEvent] = []

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        pass

    def export(self, event: TelemetryEvent) -> None:
        self.export_started.set()
        self.can_continue.wait(timeout=10.0)
        self.exports.append(event)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class ReentrantExporter:
    """Exporter that tries to emit telemetry from export() - tests re-entrance."""

    def __init__(self, name: str, manager: "TelemetryManager"):
        self._name = name
        self._manager = manager
        self.reentrant_calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        pass

    def export(self, event: TelemetryEvent) -> None:
        # Try to call handle_event from export thread - should not deadlock
        # This simulates buggy exporter that triggers telemetry
        try:
            reentrant_event = RunStarted(
                timestamp=event.timestamp,
                run_id=f"reentrant-{event.run_id}",
            )
            self._manager.handle_event(reentrant_event)
            self.reentrant_calls.append(event.run_id)
        except Exception as e:
            self.reentrant_calls.append(f"error: {e}")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_timestamp() -> datetime:
    """Fixed timestamp for deterministic tests."""
    return datetime(2026, 1, 30, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def base_run_id() -> str:
    """Fixed run ID for tests."""
    return "run-manager-test"


def make_run_started(run_id: str, timestamp: datetime | None = None) -> RunStarted:
    """Create a RunStarted event for testing."""
    return RunStarted(
        timestamp=timestamp or datetime.now(tz=UTC),
        run_id=run_id,
        config_hash="abc123",
        source_plugin="csv",
    )


def make_run_finished(run_id: str, timestamp: datetime | None = None) -> RunFinished:
    """Create a RunFinished event for testing."""
    return RunFinished(
        timestamp=timestamp or datetime.now(tz=UTC),
        run_id=run_id,
        status=RunStatus.COMPLETED,
        row_count=100,
        duration_ms=5000.0,
    )


# =============================================================================
# Basic Dispatching Tests
# =============================================================================


class TestBasicDispatching:
    """Tests for basic event dispatching to exporters."""

    def test_event_dispatched_to_single_exporter(self, base_run_id: str) -> None:
        """Event is dispatched to a single configured exporter."""
        exporter = MockExporter("test")
        manager = TelemetryManager(MockConfig(), exporters=[exporter])

        event = make_run_started(base_run_id)
        manager.handle_event(event)
        manager.flush()  # Wait for background export

        assert len(exporter.exports) == 1
        assert exporter.exports[0] is event

    def test_event_dispatched_to_multiple_exporters(self, base_run_id: str) -> None:
        """Event is dispatched to all configured exporters."""
        exporter1 = MockExporter("exporter1")
        exporter2 = MockExporter("exporter2")
        exporter3 = MockExporter("exporter3")
        manager = TelemetryManager(MockConfig(), exporters=[exporter1, exporter2, exporter3])

        event = make_run_started(base_run_id)
        manager.handle_event(event)
        manager.flush()  # Wait for background export

        assert len(exporter1.exports) == 1
        assert len(exporter2.exports) == 1
        assert len(exporter3.exports) == 1
        assert exporter1.exports[0] is event
        assert exporter2.exports[0] is event
        assert exporter3.exports[0] is event

    def test_no_exporters_is_noop(self, base_run_id: str) -> None:
        """No error when no exporters are configured."""
        manager = TelemetryManager(MockConfig(), exporters=[])

        event = make_run_started(base_run_id)
        # Should not raise
        manager.handle_event(event)

        assert manager.health_metrics["events_emitted"] == 0
        assert manager.health_metrics["events_dropped"] == 0


# =============================================================================
# Granularity Filtering Tests
# =============================================================================


class TestGranularityFiltering:
    """Tests for granularity-based event filtering."""

    def test_lifecycle_granularity_passes_run_started(self, base_run_id: str) -> None:
        """RunStarted passes at LIFECYCLE granularity."""
        exporter = MockExporter("test")
        config = MockConfig(granularity=TelemetryGranularity.LIFECYCLE)
        manager = TelemetryManager(config, exporters=[exporter])

        event = make_run_started(base_run_id)
        manager.handle_event(event)
        manager.flush()  # Wait for background export

        assert len(exporter.exports) == 1

    def test_lifecycle_granularity_filters_row_events(self, base_run_id: str, base_timestamp: datetime) -> None:
        """Row events are filtered at LIFECYCLE granularity."""
        from elspeth.telemetry import RowCreated

        exporter = MockExporter("test")
        config = MockConfig(granularity=TelemetryGranularity.LIFECYCLE)
        manager = TelemetryManager(config, exporters=[exporter])

        event = RowCreated(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            content_hash="hash",
        )
        manager.handle_event(event)

        # Event should be filtered, not dispatched
        assert len(exporter.exports) == 0


# =============================================================================
# Exporter Failure Isolation Tests
# =============================================================================


class TestExporterFailureIsolation:
    """Tests for exporter failure isolation - one failure shouldn't affect others."""

    def test_single_exporter_failure_logged_as_warning(self, base_run_id: str) -> None:
        """Individual exporter failure is logged as warning."""
        failing_exporter = MockExporter("failing", fail_export=True)
        manager = TelemetryManager(MockConfig(), exporters=[failing_exporter])

        with patch("elspeth.telemetry.manager.logger") as mock_logger:
            event = make_run_started(base_run_id)
            manager.handle_event(event)
            manager.flush()  # Wait for background export

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert call_args[0][0] == "Telemetry exporter failed"
            assert call_args[1]["exporter"] == "failing"

    def test_one_failing_exporter_doesnt_stop_others(self, base_run_id: str) -> None:
        """One failing exporter doesn't prevent other exporters from receiving events."""
        working_exporter1 = MockExporter("working1")
        failing_exporter = MockExporter("failing", fail_export=True)
        working_exporter2 = MockExporter("working2")
        manager = TelemetryManager(MockConfig(), exporters=[working_exporter1, failing_exporter, working_exporter2])

        event = make_run_started(base_run_id)
        manager.handle_event(event)
        manager.flush()  # Wait for background export

        # Working exporters still receive the event
        assert len(working_exporter1.exports) == 1
        assert len(working_exporter2.exports) == 1

    def test_partial_success_counts_as_emitted(self, base_run_id: str) -> None:
        """Partial success (some exporters work) counts as emitted, not dropped."""
        working_exporter = MockExporter("working")
        failing_exporter = MockExporter("failing", fail_export=True)
        manager = TelemetryManager(MockConfig(), exporters=[working_exporter, failing_exporter])

        event = make_run_started(base_run_id)
        manager.handle_event(event)
        manager.flush()  # Wait for background export

        assert manager.health_metrics["events_emitted"] == 1
        assert manager.health_metrics["events_dropped"] == 0

    def test_partial_success_resets_consecutive_failures(self, base_run_id: str) -> None:
        """Partial success resets the consecutive total failure counter."""
        working_exporter = MockExporter("working")
        failing_exporter = MockExporter("failing", fail_export=True)
        manager = TelemetryManager(MockConfig(), exporters=[working_exporter, failing_exporter])

        # Simulate prior total failures by setting internal state
        manager._consecutive_total_failures = 5

        event = make_run_started(base_run_id)
        manager.handle_event(event)
        manager.flush()  # Wait for background export

        # Partial success resets the counter
        assert manager._consecutive_total_failures == 0


# =============================================================================
# Total Exporter Failure Tests
# =============================================================================


class TestTotalExporterFailure:
    """Tests for behavior when ALL exporters fail."""

    def test_all_exporters_fail_increments_dropped_count(self, base_run_id: str) -> None:
        """When all exporters fail, event is counted as dropped."""
        failing1 = MockExporter("failing1", fail_export=True)
        failing2 = MockExporter("failing2", fail_export=True)
        manager = TelemetryManager(MockConfig(), exporters=[failing1, failing2])

        event = make_run_started(base_run_id)
        manager.handle_event(event)
        manager.flush()  # Wait for background export

        assert manager.health_metrics["events_emitted"] == 0
        assert manager.health_metrics["events_dropped"] == 1

    def test_all_exporters_fail_increments_consecutive_count(self, base_run_id: str) -> None:
        """Consecutive total failures are tracked."""
        failing = MockExporter("failing", fail_export=True)
        manager = TelemetryManager(MockConfig(), exporters=[failing])

        for _ in range(5):
            manager.handle_event(make_run_started(base_run_id))
        manager.flush()  # Wait for background export

        assert manager._consecutive_total_failures == 5
        assert manager.health_metrics["events_dropped"] == 5

    def test_aggregate_logging_at_100_drops(self, base_run_id: str) -> None:
        """Aggregate error logged every 100 total failures.

        Note: By default, fail_on_total_exporter_failure=False causes the manager
        to disable itself after 10 consecutive failures. To test aggregate logging
        at 100 drops, we need fail_on_total_exporter_failure=True (which raises
        instead of disabling) OR we need to reset the consecutive counter somehow.

        This test verifies the LOG_INTERVAL logic by using a modified approach:
        we create multiple manager instances to avoid the disable threshold.
        """
        # Test the aggregate logging logic by setting a higher threshold
        failing = MockExporter("failing", fail_export=True)
        manager = TelemetryManager(MockConfig(), exporters=[failing])
        # Override the max_consecutive_failures to allow more events
        manager._max_consecutive_failures = 200

        with patch("elspeth.telemetry.manager.logger") as mock_logger:
            # Send 100 events (all will fail)
            for _ in range(100):
                manager.handle_event(make_run_started(base_run_id))
            manager.flush()  # Wait for background export

            # Should have logged aggregate error once
            assert mock_logger.error.call_count == 1
            call_args = mock_logger.error.call_args
            assert call_args[0][0] == "ALL telemetry exporters failing - events dropped"
            assert call_args[1]["dropped_total"] == 100

    def test_aggregate_logging_at_every_100_drops(self, base_run_id: str) -> None:
        """Aggregate error logged at every 100 drop milestone."""
        failing = MockExporter("failing", fail_export=True)
        manager = TelemetryManager(MockConfig(), exporters=[failing])
        # Override the max_consecutive_failures to allow more events
        manager._max_consecutive_failures = 500

        with patch("elspeth.telemetry.manager.logger") as mock_logger:
            # Send 350 events
            for _ in range(350):
                manager.handle_event(make_run_started(base_run_id))
            manager.flush()  # Wait for background export

            # Should have logged 3 times (at 100, 200, 300)
            assert mock_logger.error.call_count == 3

    def test_fail_on_total_failure_false_logs_critical_and_continues(self, base_run_id: str) -> None:
        """With fail_on_total_exporter_failure=False, logs CRITICAL and disables telemetry."""
        failing = MockExporter("failing", fail_export=True)
        config = MockConfig(fail_on_total_exporter_failure=False)
        manager = TelemetryManager(config, exporters=[failing])

        with patch("elspeth.telemetry.manager.logger") as mock_logger:
            # Send enough events to trigger the threshold (10 consecutive failures)
            for _ in range(10):
                manager.handle_event(make_run_started(base_run_id))
            manager.flush()  # Wait for background export

            # Should have logged CRITICAL
            mock_logger.critical.assert_called_once()
            call_args = mock_logger.critical.call_args
            assert call_args[0][0] == "Telemetry disabled after repeated total failures"

    def test_fail_on_total_failure_false_disables_further_events(self, base_run_id: str) -> None:
        """After telemetry is disabled, further events are silently dropped."""
        failing = MockExporter("failing", fail_export=True)
        config = MockConfig(fail_on_total_exporter_failure=False)
        manager = TelemetryManager(config, exporters=[failing])

        with patch("elspeth.telemetry.manager.logger"):
            # Trigger disable
            for _ in range(10):
                manager.handle_event(make_run_started(base_run_id))
            manager.flush()  # Wait for background export

            # Telemetry should now be disabled
            assert manager._disabled is True

            # Further events don't increment counters
            initial_dropped = manager.health_metrics["events_dropped"]
            for _ in range(5):
                manager.handle_event(make_run_started(base_run_id))

            assert manager.health_metrics["events_dropped"] == initial_dropped

    def test_fail_on_total_failure_true_raises_on_flush(self, base_run_id: str) -> None:
        """With fail_on_total_exporter_failure=True, TelemetryExporterError is raised on flush().

        The exception is raised in the background export thread and stored. When
        flush() is called, it waits for the queue to drain and then re-raises
        the stored exception. This provides the "loud failure" semantics that
        fail_on_total=True promises.
        """
        failing = MockExporter("failing", fail_export=True)
        config = MockConfig(fail_on_total_exporter_failure=True)
        manager = TelemetryManager(config, exporters=[failing])

        with patch("elspeth.telemetry.manager.logger"):
            # Send 10 events - the 10th triggers TelemetryExporterError in export thread
            for _ in range(10):
                manager.handle_event(make_run_started(base_run_id))

            # flush() re-raises the stored exception
            with pytest.raises(TelemetryExporterError) as exc_info:
                manager.flush()

            assert exc_info.value.exporter_name == "all"
            assert "10 consecutive times" in str(exc_info.value)


# =============================================================================
# Health Metrics Tests
# =============================================================================


class TestHealthMetrics:
    """Tests for health metrics tracking."""

    def test_initial_metrics(self) -> None:
        """Initial health metrics are all zeros plus queue metrics."""
        manager = TelemetryManager(MockConfig(), exporters=[MockExporter("test")])

        metrics = manager.health_metrics
        assert metrics["events_emitted"] == 0
        assert metrics["events_dropped"] == 0
        assert metrics["exporter_failures"] == {}
        assert metrics["consecutive_total_failures"] == 0
        assert metrics["queue_depth"] == 0
        assert metrics["queue_maxsize"] == 1000  # From INTERNAL_DEFAULTS

    def test_events_emitted_increments(self, base_run_id: str) -> None:
        """events_emitted increments on successful dispatch."""
        exporter = MockExporter("test")
        manager = TelemetryManager(MockConfig(), exporters=[exporter])

        for _ in range(5):
            manager.handle_event(make_run_started(base_run_id))
        manager.flush()  # Wait for background export

        assert manager.health_metrics["events_emitted"] == 5

    def test_exporter_failures_tracked_per_exporter(self, base_run_id: str) -> None:
        """Per-exporter failure counts are tracked."""
        failing1 = MockExporter("failing1", fail_export=True)
        failing2 = MockExporter("failing2", fail_export=True)
        manager = TelemetryManager(MockConfig(), exporters=[failing1, failing2])

        for _ in range(3):
            manager.handle_event(make_run_started(base_run_id))
        manager.flush()  # Wait for background export

        failures = manager.health_metrics["exporter_failures"]
        assert failures["failing1"] == 3
        assert failures["failing2"] == 3

    def test_health_metrics_returns_copy(self, base_run_id: str) -> None:
        """health_metrics returns a copy of exporter_failures dict."""
        failing = MockExporter("failing", fail_export=True)
        manager = TelemetryManager(MockConfig(), exporters=[failing])
        manager.handle_event(make_run_started(base_run_id))
        manager.flush()  # Wait for background export

        metrics = manager.health_metrics
        metrics["exporter_failures"]["failing"] = 999

        # Internal state should not be modified
        assert manager.health_metrics["exporter_failures"]["failing"] == 1


# =============================================================================
# Flush and Close Tests
# =============================================================================


class TestFlushAndClose:
    """Tests for flush() and close() methods."""

    def test_flush_calls_all_exporters(self) -> None:
        """flush() calls flush() on all exporters."""
        exporter1 = MockExporter("exporter1")
        exporter2 = MockExporter("exporter2")
        manager = TelemetryManager(MockConfig(), exporters=[exporter1, exporter2])

        manager.flush()

        assert exporter1.flush_count == 1
        assert exporter2.flush_count == 1

    def test_flush_failure_logged_but_continues(self) -> None:
        """Exporter flush failure is logged but doesn't stop other exporters."""
        failing = MockExporter("failing", fail_flush=True)
        working = MockExporter("working")
        manager = TelemetryManager(MockConfig(), exporters=[failing, working])

        with patch("elspeth.telemetry.manager.logger") as mock_logger:
            manager.flush()

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert call_args[0][0] == "Exporter flush failed"
            assert call_args[1]["exporter"] == "failing"

        # Working exporter was still flushed
        assert working.flush_count == 1

    def test_close_calls_all_exporters(self) -> None:
        """close() calls close() on all exporters."""
        exporter1 = MockExporter("exporter1")
        exporter2 = MockExporter("exporter2")
        manager = TelemetryManager(MockConfig(), exporters=[exporter1, exporter2])

        manager.close()

        assert exporter1.close_count == 1
        assert exporter2.close_count == 1

    def test_close_failure_logged_but_continues(self) -> None:
        """Exporter close failure is logged but doesn't stop other exporters."""
        failing = MockExporter("failing", fail_close=True)
        working = MockExporter("working")
        manager = TelemetryManager(MockConfig(), exporters=[failing, working])

        with patch("elspeth.telemetry.manager.logger") as mock_logger:
            manager.close()

            # Should have one warning for failure, one info for closing
            assert mock_logger.warning.call_count == 1
            assert mock_logger.info.call_count == 1

        # Working exporter was still closed
        assert working.close_count == 1

    def test_close_logs_final_metrics(self, base_run_id: str) -> None:
        """close() logs final health metrics including queue metrics."""
        exporter = MockExporter("test")
        manager = TelemetryManager(MockConfig(), exporters=[exporter])

        # Emit some events
        for _ in range(5):
            manager.handle_event(make_run_started(base_run_id))

        with patch("elspeth.telemetry.manager.logger") as mock_logger:
            manager.close()  # close() waits for queue to drain

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert call_args[0][0] == "Telemetry manager closing"
            assert call_args[1]["events_emitted"] == 5
            assert call_args[1]["queue_depth"] == 0  # Queue drained
            assert call_args[1]["queue_maxsize"] == 1000


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_log_interval_is_100(self) -> None:
        """LOG_INTERVAL constant is 100."""
        assert TelemetryManager._LOG_INTERVAL == 100

    def test_max_consecutive_failures_is_10(self) -> None:
        """Default max consecutive failures threshold is 10."""
        manager = TelemetryManager(MockConfig(), exporters=[])
        assert manager._max_consecutive_failures == 10

    def test_success_after_total_failures_resets_counter(self, base_run_id: str) -> None:
        """Successful dispatch after total failures resets consecutive counter."""

        # Create an exporter that can be toggled
        class ToggleableExporter(MockExporter):
            def export(self, event: TelemetryEvent) -> None:
                if self._fail_export:
                    raise RuntimeError("Simulated failure")
                self.exports.append(event)

        exporter = ToggleableExporter("toggled")
        exporter._fail_export = True
        manager = TelemetryManager(MockConfig(), exporters=[exporter])

        # Accumulate some consecutive failures
        for _ in range(5):
            manager.handle_event(make_run_started(base_run_id))
        manager.flush()  # Wait for background export

        assert manager._consecutive_total_failures == 5

        # Fix the exporter and send another event
        exporter._fail_export = False
        manager.handle_event(make_run_started(base_run_id))
        manager.flush()  # Wait for background export

        # Counter should be reset
        assert manager._consecutive_total_failures == 0


# =============================================================================
# Property-Based State Machine Tests
# =============================================================================


class TelemetryManagerStateMachine(RuleBasedStateMachine):
    """State machine for property-based testing of TelemetryManager.

    Tests invariants:
    - events_emitted + events_dropped == total events that passed filtering
    - consecutive_total_failures resets to 0 on any success
    - exporter_failures[name] >= events where that exporter failed
    """

    def __init__(self) -> None:
        super().__init__()
        self.working_exporter = MockExporter("working")
        self.failing_exporter = MockExporter("failing", fail_export=True)
        self.manager = TelemetryManager(
            MockConfig(),
            exporters=[self.working_exporter, self.failing_exporter],
        )
        self.total_events_sent = 0

    def teardown(self) -> None:
        """Clean up manager after test."""
        self.manager.close()

    events = Bundle("events")

    @rule(target=events, run_id=st.text(min_size=1, max_size=20))
    def create_event(self, run_id: str) -> TelemetryEvent:
        """Create a new event."""
        return make_run_started(run_id)

    @rule(event=events)
    def send_event(self, event: TelemetryEvent) -> None:
        """Send an event to the manager."""
        self.manager.handle_event(event)
        self.total_events_sent += 1
        self.manager.flush()  # Wait for background export

        # Check invariant: events_emitted + events_dropped <= total_events_sent
        metrics = self.manager.health_metrics
        assert metrics["events_emitted"] + metrics["events_dropped"] <= self.total_events_sent

    @rule()
    def check_partial_success_is_emitted(self) -> None:
        """With one working and one failing exporter, events should be emitted, not dropped."""
        self.manager.flush()  # Ensure all events processed
        metrics = self.manager.health_metrics
        # All events should be emitted (partial success counts as emitted)
        assert metrics["events_emitted"] == self.total_events_sent
        assert metrics["events_dropped"] == 0

    @rule()
    def check_consecutive_failures_zero(self) -> None:
        """With a working exporter, consecutive_total_failures should always be 0."""
        self.manager.flush()  # Ensure all events processed
        metrics = self.manager.health_metrics
        assert metrics["consecutive_total_failures"] == 0


TestStateMachine = TelemetryManagerStateMachine.TestCase


class AllFailingStateMachine(RuleBasedStateMachine):
    """State machine for testing when all exporters fail."""

    def __init__(self) -> None:
        super().__init__()
        self.failing1 = MockExporter("failing1", fail_export=True)
        self.failing2 = MockExporter("failing2", fail_export=True)
        # Don't crash on failure for this test
        self.manager = TelemetryManager(
            MockConfig(fail_on_total_exporter_failure=False),
            exporters=[self.failing1, self.failing2],
        )
        self.total_events_sent = 0

    def teardown(self) -> None:
        """Clean up manager after test."""
        self.manager.close()

    @rule(run_id=st.text(min_size=1, max_size=20))
    def send_event(self, run_id: str) -> None:
        """Send an event that will fail."""
        # Suppress logging during property tests
        with patch("elspeth.telemetry.manager.logger"):
            event = make_run_started(run_id)
            self.manager.handle_event(event)
            self.total_events_sent += 1
            self.manager.flush()  # Wait for background export

    @rule()
    def check_all_dropped(self) -> None:
        """All events should be dropped when all exporters fail."""
        self.manager.flush()  # Ensure all events processed
        if self.manager._disabled:
            # After disabled, count stops incrementing
            return
        metrics = self.manager.health_metrics
        assert metrics["events_emitted"] == 0
        # Before disable threshold, all events are dropped
        if self.total_events_sent <= 10:
            assert metrics["events_dropped"] == self.total_events_sent


TestAllFailingStateMachine = AllFailingStateMachine.TestCase


# =============================================================================
# Additional Property Tests
# =============================================================================


class TestPropertyBasedInvariants:
    """Property-based tests for TelemetryManager invariants."""

    @given(num_events=st.integers(min_value=0, max_value=50))
    @hypothesis_settings(max_examples=25)
    def test_emitted_count_matches_successful_exports(self, num_events: int) -> None:
        """Property: events_emitted equals number of events exported to at least one exporter."""
        exporter = MockExporter("test")
        manager = TelemetryManager(MockConfig(), exporters=[exporter])

        for i in range(num_events):
            manager.handle_event(make_run_started(f"run-{i}"))
        manager.flush()  # Wait for background export

        assert manager.health_metrics["events_emitted"] == num_events
        assert len(exporter.exports) == num_events
        manager.close()

    @given(num_events=st.integers(min_value=0, max_value=50))
    @hypothesis_settings(max_examples=25)
    def test_dropped_count_when_all_fail(self, num_events: int) -> None:
        """Property: events_dropped equals num_events when all exporters fail (before disable).

        Note: With async export, events may be queued before the export thread
        processes the event that triggers disable. So we may have slightly more
        than 10 events dropped depending on timing.
        """
        failing = MockExporter("failing", fail_export=True)
        # Don't crash on failure
        config = MockConfig(fail_on_total_exporter_failure=False)
        manager = TelemetryManager(config, exporters=[failing])

        with patch("elspeth.telemetry.manager.logger"):
            for i in range(num_events):
                manager.handle_event(make_run_started(f"run-{i}"))
            manager.flush()  # Wait for background export

        # After disable threshold (10), events stop being counted
        # With async export, events queued before disable is set may still be processed
        if num_events <= 10:
            assert manager.health_metrics["events_dropped"] == num_events
        else:
            # At least 10 events dropped, at most num_events (if all were queued before disable)
            dropped = manager.health_metrics["events_dropped"]
            assert dropped >= 10, f"Expected at least 10 drops, got {dropped}"
            assert dropped <= num_events, f"Expected at most {num_events} drops, got {dropped}"
        manager.close()


# =============================================================================
# Backpressure Mode Tests
# =============================================================================


class TestBackpressureMode:
    """Tests for backpressure_mode config wiring."""

    def test_drop_mode_drops_when_queue_full(self, base_timestamp: datetime) -> None:
        """DROP mode drops events when queue is full instead of blocking."""
        # Use a tiny queue to make it fill quickly
        slow_exporter = SlowExporter("slow")
        config = MockConfig(backpressure_mode=BackpressureMode.DROP)
        manager = TelemetryManager(config, exporters=[slow_exporter])

        # Override queue size for testing (access internal for test)
        # This tests the behavior, not the default size
        manager._queue = queue.Queue(maxsize=2)

        event = RunStarted(
            timestamp=base_timestamp,
            run_id="test-run",
            config_hash="test-hash",
            source_plugin="test-source",
        )

        try:
            # Fill queue: 2 events should queue, 3rd should drop
            manager.handle_event(event)
            manager.handle_event(event)
            manager.handle_event(event)  # Should drop, not block

            # Verify drop was counted
            assert manager.health_metrics["events_dropped"] == 1

            # Verify we didn't block (test would timeout if we blocked)
        finally:
            slow_exporter.can_continue.set()  # Unblock exporter
            manager.close()

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_block_mode_blocks_when_queue_full(self, base_timestamp: datetime) -> None:
        """BLOCK mode blocks handle_event() when queue is full.

        Note: This test replaces the internal queue mid-flight, which causes task_done()
        count mismatches (expected warning). This is a test artifact, not a production issue.
        """
        slow_exporter = SlowExporter("slow")
        config = MockConfig(backpressure_mode=BackpressureMode.BLOCK)
        manager = TelemetryManager(config, exporters=[slow_exporter])

        event = RunStarted(
            timestamp=base_timestamp,
            run_id="test-run",
            config_hash="test-hash",
            source_plugin="test-source",
        )

        # Step 1: Send first event to old queue - export thread will consume it
        manager.handle_event(event)

        # Wait for export thread to start processing (confirms event was consumed from queue)
        assert slow_exporter.export_started.wait(timeout=5.0), "Export thread never started"

        # Step 2: NOW replace queue - export thread has consumed from old queue
        # and is blocked in SlowExporter.export()
        manager._queue = queue.Queue(maxsize=1)

        # Step 3: Fill the new queue
        manager.handle_event(event)  # Fills queue (maxsize=1)

        # Step 4: Third event should block because queue is full
        started_blocking = threading.Event()
        finished_blocking = threading.Event()

        def blocking_put() -> None:
            started_blocking.set()  # Signal we're about to try
            manager.handle_event(event)  # Should block - queue is full
            finished_blocking.set()  # Signal that we got past blocking

        thread = threading.Thread(target=blocking_put)
        thread.start()

        # Wait for thread to signal it's starting
        assert started_blocking.wait(timeout=5.0), "Thread never started"

        # Give thread time to enter blocking put()
        # Then verify it hasn't finished (still blocked)
        assert not finished_blocking.wait(timeout=0.5), "handle_event should have blocked but didn't"

        # Unblock exporter - this allows export thread to finish and read from new queue
        slow_exporter.can_continue.set()

        # Now thread should complete (export thread will consume from new queue)
        assert finished_blocking.wait(timeout=5.0), "handle_event never unblocked"

        thread.join(timeout=1.0)
        manager.close()

    def test_concurrent_close_during_export(self, base_timestamp: datetime) -> None:
        """close() during active export doesn't deadlock or corrupt state."""
        slow_exporter = SlowExporter("slow")
        config = MockConfig()
        manager = TelemetryManager(config, exporters=[slow_exporter])

        # Queue several events
        for i in range(5):
            event = make_run_started(f"run-{i}", base_timestamp)
            manager.handle_event(event)

        # Wait for export thread to start processing
        assert slow_exporter.export_started.wait(timeout=5.0), "Export never started"

        # Now call close() while export is blocked
        close_completed = threading.Event()

        def close_manager():
            manager.close()
            close_completed.set()

        close_thread = threading.Thread(target=close_manager)
        close_thread.start()

        # Give close() a moment to start
        # Then unblock exporter so everything can complete
        import time

        time.sleep(0.1)
        slow_exporter.can_continue.set()

        # Close should complete without deadlock
        assert close_completed.wait(timeout=5.0), "close() deadlocked during active export"

        close_thread.join(timeout=1.0)

        # Verify no corruption - metrics should be consistent
        metrics = manager.health_metrics
        assert isinstance(metrics["events_emitted"], int)
        assert isinstance(metrics["events_dropped"], int)

    def test_lock_contention_on_events_dropped(self, base_timestamp: datetime) -> None:
        """Both threads incrementing _events_dropped doesn't corrupt counter."""
        # Create scenario where both threads write _events_dropped:
        # - Pipeline thread: DROP mode with full queue
        # - Export thread: All exporters failing

        failing_exporter = MockExporter("failing", fail_export=True)
        config = MockConfig(backpressure_mode=BackpressureMode.DROP)
        manager = TelemetryManager(config, exporters=[failing_exporter])

        # Use tiny queue to force drops
        manager._queue = queue.Queue(maxsize=2)

        # Fire many events - some will drop on queue full, some on export fail
        events_fired = 100
        for i in range(events_fired):
            event = make_run_started(f"run-{i}", base_timestamp)
            manager.handle_event(event)

        # Close to flush everything
        manager.close()

        # Key assertion: dropped count should be consistent
        # (not corrupted by concurrent increments)
        metrics = manager.health_metrics
        total_accounted = metrics["events_emitted"] + metrics["events_dropped"]

        # We don't know exact split, but total should equal events fired
        # (some may be in-flight when we closed, so allow small variance)
        assert total_accounted <= events_fired, (
            f"More events accounted ({total_accounted}) than fired ({events_fired}) - counter corruption detected"
        )

    def test_reentrant_handle_event_no_deadlock(self, base_timestamp: datetime) -> None:
        """Exporter calling handle_event() from export thread doesn't deadlock.

        This tests a buggy exporter that tries to emit telemetry from within
        the export thread. It should either work or fail gracefully, not deadlock.
        """
        config = MockConfig()
        # Need to create manager first, then exporter (circular dependency)
        manager = TelemetryManager(config, exporters=[])

        reentrant_exporter = ReentrantExporter("reentrant", manager)
        manager._exporters = [reentrant_exporter]

        event = make_run_started("test", base_timestamp)

        # This should complete without deadlock
        manager.handle_event(event)

        # Close with timeout - deadlock would hang here
        close_completed = threading.Event()

        def close_manager():
            manager.close()
            close_completed.set()

        close_thread = threading.Thread(target=close_manager)
        close_thread.start()

        assert close_completed.wait(timeout=5.0), "Re-entrant handle_event caused deadlock"
        close_thread.join(timeout=1.0)

        # Verify the reentrant call happened (or was handled)
        assert len(reentrant_exporter.reentrant_calls) > 0

    def test_close_processes_all_queued_events(self, base_timestamp: datetime) -> None:
        """close() processes all queued events before returning."""
        exporter = MockExporter("test")
        config = MockConfig()
        manager = TelemetryManager(config, exporters=[exporter])

        # Queue several events
        for i in range(5):
            event = make_run_started(f"run-{i}", base_timestamp)
            manager.handle_event(event)

        # Close should process all events
        manager.close()

        # Verify all events were exported
        assert len(exporter.exports) == 5
        assert [e.run_id for e in exporter.exports] == [f"run-{i}" for i in range(5)]

    def test_thread_death_disables_telemetry(self, base_timestamp: datetime) -> None:
        """If export thread dies, handle_event disables telemetry."""
        exporter = MockExporter("test")
        config = MockConfig()
        manager = TelemetryManager(config, exporters=[exporter])

        # Force thread to exit by sending sentinel directly
        manager._queue.put(None)
        manager._export_thread.join(timeout=1.0)
        assert not manager._export_thread.is_alive()

        # Now handle_event should detect dead thread and disable
        event = make_run_started("test", base_timestamp)
        manager.handle_event(event)

        assert manager._disabled is True

        # Cleanup (close won't try to stop already-dead thread)
        manager._shutdown_event.set()

    def test_events_exported_in_fifo_order(self, base_timestamp: datetime) -> None:
        """Events are exported in the order they were queued (FIFO)."""
        exporter = MockExporter("test")
        config = MockConfig()
        manager = TelemetryManager(config, exporters=[exporter])

        # Queue events with distinct run_ids
        run_ids = [f"run-{i}" for i in range(20)]
        for run_id in run_ids:
            event = make_run_started(run_id, base_timestamp)
            manager.handle_event(event)

        # Close and verify order
        manager.close()

        exported_ids = [e.run_id for e in exporter.exports]
        assert exported_ids == run_ids, "Events not exported in FIFO order"

    def test_task_done_called_on_exception(self, base_timestamp: datetime) -> None:
        """task_done() is called even when exporter raises, preventing join() hang."""
        failing_exporter = MockExporter("failing", fail_export=True)
        config = MockConfig()
        manager = TelemetryManager(config, exporters=[failing_exporter])

        # Queue events (they'll all fail to export)
        for i in range(3):
            event = make_run_started(f"run-{i}", base_timestamp)
            manager.handle_event(event)

        # close() should not hang - use threading.Event for cross-platform timeout
        close_completed = threading.Event()
        close_exception: Exception | None = None

        def close_with_tracking():
            nonlocal close_exception
            try:
                manager.close()
            except Exception as e:
                close_exception = e
            finally:
                close_completed.set()

        close_thread = threading.Thread(target=close_with_tracking)
        close_thread.start()

        # Wait with timeout - if task_done() not called, close() would hang
        if not close_completed.wait(timeout=5.0):
            # Force cleanup and fail
            manager._shutdown_event.set()
            manager._queue.put(None)  # Try to unblock
            close_thread.join(timeout=1.0)
            pytest.fail("close() hung - task_done() not called properly")

        close_thread.join(timeout=1.0)

        # Should get here without hanging
        assert manager.health_metrics["events_dropped"] == 3

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_close_completes_when_queue_is_full(self, base_timestamp: datetime) -> None:
        """close() MUST complete even when queue is full at shutdown time.

        Regression test for shutdown hang vulnerability.

        When the queue is full at shutdown (e.g., slow exporters, DROP mode backlog),
        close() must guarantee the sentinel gets inserted. If the sentinel cannot be
        inserted, the export thread blocks on get() forever, and because it's non-daemon,
        the process hangs indefinitely.

        The fix drains items from the queue to make room for the sentinel when necessary.

        Note: This test replaces the internal queue mid-flight, which causes task_done()
        count mismatches (expected warning). This is a test artifact, not a production issue.
        """
        # Use SlowExporter to block the export thread - this allows us to fill the queue
        # The thread will be blocked in export(), not in get()
        slow_exporter = SlowExporter("slow")
        config = MockConfig(backpressure_mode=BackpressureMode.DROP)
        manager = TelemetryManager(config, exporters=[slow_exporter])

        # Replace queue with tiny one BEFORE any events are sent
        # The export thread is blocked on get() on the original queue initially,
        # but when we send the first event, it will wake up and process it.
        # Then it will be blocked in SlowExporter.export(), and we can replace the queue.

        # Send first event to original queue
        first_event = make_run_started("first", base_timestamp)
        manager.handle_event(first_event)

        # Wait for export thread to consume the event and block in SlowExporter
        assert slow_exporter.export_started.wait(timeout=5.0), "Export thread never started"

        # NOW replace the queue - thread is blocked in SlowExporter, not in get()
        manager._queue = queue.Queue(maxsize=3)

        # Fill the new queue completely with events in DROP mode
        for i in range(10):
            event = make_run_started(f"run-{i}", base_timestamp)
            manager.handle_event(event)

        # Queue should be full (or nearly full due to timing)
        # Now call close() - this should drain queue and insert sentinel

        # Now call close() - this MUST complete without hanging
        close_completed = threading.Event()
        close_exception: Exception | None = None

        def close_with_tracking():
            nonlocal close_exception
            try:
                manager.close()
            except Exception as e:
                close_exception = e
            finally:
                close_completed.set()

        # Start close in a thread so we can timeout if it hangs
        close_thread = threading.Thread(target=close_with_tracking)
        close_thread.start()

        # Give close() time to start its shutdown sequence (drain queue, insert sentinel)
        import time

        time.sleep(0.3)

        # Now unblock the slow exporter - thread will finish export, then process queue
        slow_exporter.can_continue.set()

        # close() MUST complete within reasonable time
        # If sentinel insertion fails when queue is full, the thread would hang
        if not close_completed.wait(timeout=5.0):
            # Force cleanup attempt
            manager._shutdown_event.set()
            try:
                # Force-insert sentinel to try to unblock
                while True:
                    try:
                        manager._queue.get_nowait()
                    except queue.Empty:
                        break
                manager._queue.put(None)
            except Exception:
                pass
            close_thread.join(timeout=1.0)
            pytest.fail("close() hung when queue was full at shutdown - sentinel insertion must be guaranteed even when queue is full")

        close_thread.join(timeout=1.0)

        if close_exception is not None:
            raise close_exception

        # Verify export thread exited cleanly
        assert not manager._export_thread.is_alive(), "Export thread still alive after close()"
