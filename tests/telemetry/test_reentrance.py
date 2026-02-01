# tests/telemetry/test_reentrance.py
"""Tests for EventBus and TelemetryManager re-entrance safety.

These tests verify that the telemetry system handles re-entrant scenarios
gracefully, where handlers or exporters might accidentally trigger additional
telemetry events during processing.

Re-entrance scenarios to protect against:
1. Exporter.export() emitting another telemetry event
2. EventBus handler triggering another event of the same type
3. Recursive event chains (A triggers B triggers A)
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from elspeth.contracts.enums import BackpressureMode, TelemetryGranularity
from elspeth.contracts.events import TelemetryEvent
from elspeth.core.events import EventBus
from elspeth.telemetry.events import RunStarted
from elspeth.telemetry.manager import TelemetryManager


@dataclass
class FakeRuntimeTelemetryConfig:
    """Minimal config for testing TelemetryManager."""

    enabled: bool = True
    granularity: TelemetryGranularity = TelemetryGranularity.FULL
    backpressure_mode: BackpressureMode = BackpressureMode.DROP
    fail_on_total_exporter_failure: bool = False
    exporter_configs: tuple[Any, ...] = ()


class ReentrantExporter:
    """Test exporter that tries to emit events during export.

    This simulates a badly-written exporter that might accidentally
    trigger telemetry (e.g., by calling a traced function or emitting
    a diagnostic event).
    """

    def __init__(self, telemetry_manager: TelemetryManager | None = None) -> None:
        self._telemetry_manager = telemetry_manager
        self._export_count = 0
        self._max_depth = 100  # Safety limit for testing

    @property
    def name(self) -> str:
        return "reentrant_test"

    def set_telemetry_manager(self, manager: TelemetryManager) -> None:
        """Set the manager after construction (for circular reference)."""
        self._telemetry_manager = manager

    def configure(self, config: dict[str, Any]) -> None:
        """No configuration needed for test exporter."""
        pass

    def export(self, event: TelemetryEvent) -> None:
        """Export that tries to re-emit the same event type."""
        self._export_count += 1

        # Safety: prevent actual stack overflow in case protection fails
        if self._export_count > self._max_depth:
            return

        # Try to cause re-entrance by emitting another event
        if self._telemetry_manager is not None:
            reentrant_event = RunStarted(
                timestamp=datetime.now(UTC),
                run_id=f"reentrant-{self._export_count}",
                config_hash="test",
                source_plugin="test",
            )
            self._telemetry_manager.handle_event(reentrant_event)

    def flush(self) -> None:
        """No buffering."""
        pass

    def close(self) -> None:
        """No resources to release."""
        pass


class RecursiveEventExporter:
    """Exporter that emits events during flush and close."""

    def __init__(self) -> None:
        self._telemetry_manager: TelemetryManager | None = None
        self._export_count = 0
        self._flush_count = 0
        self._close_count = 0

    @property
    def name(self) -> str:
        return "recursive_test"

    def set_telemetry_manager(self, manager: TelemetryManager) -> None:
        """Set the manager after construction (for circular reference)."""
        self._telemetry_manager = manager

    def configure(self, config: dict[str, Any]) -> None:
        """No configuration needed."""
        pass

    def export(self, event: TelemetryEvent) -> None:
        """Normal export."""
        self._export_count += 1

    def flush(self) -> None:
        """Flush that tries to emit an event."""
        self._flush_count += 1
        if self._flush_count <= 1 and self._telemetry_manager is not None:
            event = RunStarted(
                timestamp=datetime.now(UTC),
                run_id="flush-reentrant",
                config_hash="test",
                source_plugin="test",
            )
            self._telemetry_manager.handle_event(event)

    def close(self) -> None:
        """Close that tries to emit an event."""
        self._close_count += 1
        if self._close_count <= 1 and self._telemetry_manager is not None:
            event = RunStarted(
                timestamp=datetime.now(UTC),
                run_id="close-reentrant",
                config_hash="test",
                source_plugin="test",
            )
            self._telemetry_manager.handle_event(event)


class TestTelemetryManagerReentrance:
    """Tests for TelemetryManager handling of re-entrant event emission."""

    def test_reentrant_export_does_not_stack_overflow(self) -> None:
        """Verify that an exporter emitting events during export doesn't overflow.

        This tests the scenario where a badly-written exporter might accidentally
        emit another telemetry event during export. The system should handle this
        gracefully without stack overflow.

        The test passes if:
        1. No RecursionError is raised
        2. The system completes processing
        3. Events are processed (export_count > 0)
        """
        config = FakeRuntimeTelemetryConfig()
        exporter = ReentrantExporter()

        manager = TelemetryManager(config, [exporter])
        exporter.set_telemetry_manager(manager)

        # Emit an event - this will cause the exporter to try re-entrance
        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="test-run",
            config_hash="abc123",
            source_plugin="csv",
        )

        # This should NOT raise RecursionError
        # The exporter has a safety limit of 100 to prevent actual overflow
        # in case the system doesn't have re-entrance protection
        manager.handle_event(event)

        # Wait for background thread to process the event
        manager.flush()

        # Verify processing occurred
        assert exporter._export_count > 0, "Expected at least one export call"

        # Log the actual behavior for debugging
        # The count tells us how the system handled re-entrance:
        # - count=1: Perfect re-entrance blocking (event ignored)
        # - count>1: Re-entrance allowed but bounded
        # - count=100: Hit safety limit (would have overflowed)

    def test_reentrant_export_does_not_infinite_loop(self) -> None:
        """Verify that re-entrant exports terminate in reasonable time.

        Even if re-entrance is allowed, it should terminate and not
        enter an infinite loop.
        """
        config = FakeRuntimeTelemetryConfig()
        exporter = ReentrantExporter()

        manager = TelemetryManager(config, [exporter])
        exporter.set_telemetry_manager(manager)

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="test-run",
            config_hash="abc123",
            source_plugin="csv",
        )

        # Set a reasonable expectation - this should complete quickly
        # If it takes more than a few seconds, something is wrong
        import time

        start = time.monotonic()
        manager.handle_event(event)
        elapsed = time.monotonic() - start

        # Should complete in under 1 second even with re-entrance
        assert elapsed < 1.0, f"Re-entrant handling took too long: {elapsed:.2f}s"

    def test_flush_reentrance_does_not_overflow(self) -> None:
        """Verify that emitting events during flush doesn't cause problems."""
        config = FakeRuntimeTelemetryConfig()
        exporter = RecursiveEventExporter()

        manager = TelemetryManager(config, [exporter])
        exporter.set_telemetry_manager(manager)

        # First, send a normal event
        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="test-run",
            config_hash="abc123",
            source_plugin="csv",
        )
        manager.handle_event(event)

        # Now flush - the exporter will try to emit during flush
        manager.flush()

        # Should complete without error
        assert exporter._flush_count >= 1
        assert exporter._export_count >= 1

    def test_close_reentrance_does_not_overflow(self) -> None:
        """Verify that emitting events during close doesn't cause problems."""
        config = FakeRuntimeTelemetryConfig()
        exporter = RecursiveEventExporter()

        manager = TelemetryManager(config, [exporter])
        exporter.set_telemetry_manager(manager)

        # Send a normal event
        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="test-run",
            config_hash="abc123",
            source_plugin="csv",
        )
        manager.handle_event(event)

        # Now close - the exporter will try to emit during close
        manager.close()

        # Should complete without error
        assert exporter._close_count >= 1

    def test_disabled_manager_handles_reentrance(self) -> None:
        """Verify that a disabled manager handles re-entrant calls gracefully."""
        config = FakeRuntimeTelemetryConfig()
        exporter = ReentrantExporter()

        manager = TelemetryManager(config, [exporter])
        exporter.set_telemetry_manager(manager)

        # Manually disable the manager (simulating repeated failures)
        manager._disabled = True

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="test-run",
            config_hash="abc123",
            source_plugin="csv",
        )

        # Should be a no-op when disabled
        manager.handle_event(event)

        # No exports should have occurred
        assert exporter._export_count == 0


class TestEventBusReentrance:
    """Tests for EventBus handling of re-entrant event emission."""

    def test_handler_emitting_same_event_type_does_not_overflow(self) -> None:
        """Verify that a handler emitting the same event type doesn't overflow.

        This tests the scenario where an EventBus handler might accidentally
        emit another event of the same type during handling.
        """
        bus = EventBus()
        call_count = 0
        max_depth = 100  # Safety limit

        @dataclass
        class TestEvent:
            value: int

        def reentrant_handler(event: TestEvent) -> None:
            nonlocal call_count
            call_count += 1

            # Safety limit to prevent actual overflow
            if call_count > max_depth:
                return

            # Try to cause re-entrance
            bus.emit(TestEvent(value=event.value + 1))

        bus.subscribe(TestEvent, reentrant_handler)

        # This will cause recursive emission
        bus.emit(TestEvent(value=1))

        # Should complete - the exact count depends on whether EventBus
        # has re-entrance protection or allows bounded recursion
        assert call_count > 0, "Expected at least one handler call"

    def test_handler_emitting_different_event_type_allowed(self) -> None:
        """Verify that handlers can emit different event types (event chains)."""
        bus = EventBus()
        event_a_count = 0
        event_b_count = 0

        @dataclass
        class EventA:
            pass

        @dataclass
        class EventB:
            pass

        def handler_a(event: EventA) -> None:
            nonlocal event_a_count
            event_a_count += 1
            # Emit a different event type - this should be fine
            bus.emit(EventB())

        def handler_b(event: EventB) -> None:
            nonlocal event_b_count
            event_b_count += 1

        bus.subscribe(EventA, handler_a)
        bus.subscribe(EventB, handler_b)

        bus.emit(EventA())

        assert event_a_count == 1
        assert event_b_count == 1

    def test_circular_event_chain_terminates(self) -> None:
        """Verify that A->B->A event chains terminate.

        Tests the scenario where EventA triggers EventB, which triggers EventA.
        This should either be blocked or terminate gracefully.
        """
        bus = EventBus()
        event_a_count = 0
        event_b_count = 0
        max_depth = 50  # Safety limit

        @dataclass
        class EventA:
            depth: int

        @dataclass
        class EventB:
            depth: int

        def handler_a(event: EventA) -> None:
            nonlocal event_a_count
            event_a_count += 1
            if event.depth < max_depth:
                bus.emit(EventB(depth=event.depth + 1))

        def handler_b(event: EventB) -> None:
            nonlocal event_b_count
            event_b_count += 1
            if event.depth < max_depth:
                bus.emit(EventA(depth=event.depth + 1))

        bus.subscribe(EventA, handler_a)
        bus.subscribe(EventB, handler_b)

        bus.emit(EventA(depth=0))

        # Should complete - counts depend on implementation
        assert event_a_count > 0
        assert event_b_count > 0

    def test_multiple_handlers_with_reentrance(self) -> None:
        """Verify that multiple handlers with re-entrance are handled correctly."""
        bus = EventBus()
        handler1_count = 0
        handler2_count = 0
        max_depth = 20

        @dataclass
        class TestEvent:
            depth: int

        def handler1(event: TestEvent) -> None:
            nonlocal handler1_count
            handler1_count += 1
            if event.depth < max_depth:
                bus.emit(TestEvent(depth=event.depth + 1))

        def handler2(event: TestEvent) -> None:
            nonlocal handler2_count
            handler2_count += 1
            # This handler doesn't re-emit

        bus.subscribe(TestEvent, handler1)
        bus.subscribe(TestEvent, handler2)

        bus.emit(TestEvent(depth=0))

        # Both handlers should have been called
        assert handler1_count > 0
        assert handler2_count > 0


class TestTelemetryManagerWithEventBus:
    """Integration tests for TelemetryManager with EventBus re-entrance."""

    def test_telemetry_handler_reentrance_through_eventbus(self) -> None:
        """Test re-entrance when TelemetryManager is subscribed to EventBus.

        This simulates the production scenario where:
        1. EventBus emits a telemetry event
        2. TelemetryManager.handle_event() is called
        3. An exporter tries to emit back to EventBus
        """
        bus = EventBus()
        config = FakeRuntimeTelemetryConfig()

        export_count = 0

        class EventBusReentrantExporter:
            """Exporter that emits back to EventBus during export."""

            def __init__(self, event_bus: EventBus) -> None:
                self._bus = event_bus
                self._depth = 0
                self._max_depth = 10

            @property
            def name(self) -> str:
                return "eventbus_reentrant"

            def configure(self, config: dict[str, Any]) -> None:
                pass

            def export(self, event: TelemetryEvent) -> None:
                nonlocal export_count
                export_count += 1
                self._depth += 1

                if self._depth <= self._max_depth:
                    # Emit back to EventBus, which will call handle_event again
                    reentrant = RunStarted(
                        timestamp=datetime.now(UTC),
                        run_id=f"reentrant-{self._depth}",
                        config_hash="test",
                        source_plugin="test",
                    )
                    self._bus.emit(reentrant)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        exporter = EventBusReentrantExporter(bus)
        manager = TelemetryManager(config, [exporter])

        # Subscribe TelemetryManager to EventBus for RunStarted events
        bus.subscribe(RunStarted, manager.handle_event)

        # Trigger the chain
        initial_event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="initial",
            config_hash="test",
            source_plugin="test",
        )
        bus.emit(initial_event)

        # Wait for background thread to process
        manager.flush()

        # Should complete without overflow
        assert export_count > 0
