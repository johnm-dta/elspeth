"""Tests for EventBus infrastructure."""

from dataclasses import dataclass

import pytest

from elspeth.core.events import EventBus, EventBusProtocol, NullEventBus


# Test event types
@dataclass(frozen=True)
class SampleEvent:
    """Sample event for EventBus tests."""

    value: str


@dataclass(frozen=True)
class AnotherEvent:
    """Another test event type."""

    count: int


class TestEventBus:
    """Tests for EventBus implementation."""

    def test_subscribe_and_emit(self) -> None:
        """Test basic subscribe and emit functionality."""
        bus = EventBus()
        received_events: list[SampleEvent] = []

        def handler(event: SampleEvent) -> None:
            received_events.append(event)

        bus.subscribe(SampleEvent, handler)
        bus.emit(SampleEvent(value="hello"))

        assert len(received_events) == 1
        assert received_events[0].value == "hello"

    def test_multiple_subscribers(self) -> None:
        """Test multiple subscribers to same event type."""
        bus = EventBus()
        handler1_events: list[SampleEvent] = []
        handler2_events: list[SampleEvent] = []

        def handler1(event: SampleEvent) -> None:
            handler1_events.append(event)

        def handler2(event: SampleEvent) -> None:
            handler2_events.append(event)

        bus.subscribe(SampleEvent, handler1)
        bus.subscribe(SampleEvent, handler2)

        event = SampleEvent(value="broadcast")
        bus.emit(event)

        # Both handlers should receive the event
        assert len(handler1_events) == 1
        assert len(handler2_events) == 1
        assert handler1_events[0] is event
        assert handler2_events[0] is event

    def test_handler_ordering(self) -> None:
        """Test handlers are called in subscription order."""
        bus = EventBus()
        call_order: list[int] = []

        def handler1(event: SampleEvent) -> None:
            call_order.append(1)

        def handler2(event: SampleEvent) -> None:
            call_order.append(2)

        def handler3(event: SampleEvent) -> None:
            call_order.append(3)

        bus.subscribe(SampleEvent, handler1)
        bus.subscribe(SampleEvent, handler2)
        bus.subscribe(SampleEvent, handler3)

        bus.emit(SampleEvent(value="test"))

        assert call_order == [1, 2, 3]

    def test_different_event_types_isolated(self) -> None:
        """Test different event types don't cross-contaminate."""
        bus = EventBus()
        test_events: list[SampleEvent] = []
        another_events: list[AnotherEvent] = []

        bus.subscribe(SampleEvent, lambda e: test_events.append(e))
        bus.subscribe(AnotherEvent, lambda e: another_events.append(e))

        bus.emit(SampleEvent(value="test"))
        bus.emit(AnotherEvent(count=42))

        assert len(test_events) == 1
        assert len(another_events) == 1
        assert test_events[0].value == "test"
        assert another_events[0].count == 42

    def test_no_subscribers_silent(self) -> None:
        """Test emitting event with no subscribers is silent (intentional)."""
        bus = EventBus()
        # Should not raise - events with no subscribers are ignored
        bus.emit(SampleEvent(value="nobody listening"))

    def test_handler_exception_propagates(self) -> None:
        """Test handler exceptions propagate to caller (crash on bug)."""
        bus = EventBus()

        def failing_handler(event: SampleEvent) -> None:
            raise ValueError("Handler bug")

        bus.subscribe(SampleEvent, failing_handler)

        # Handler exceptions should propagate - formatters are "our code"
        with pytest.raises(ValueError, match="Handler bug"):
            bus.emit(SampleEvent(value="test"))

    def test_handler_exception_stops_subsequent_handlers(self) -> None:
        """Test exception in handler stops subsequent handlers (fail-fast)."""
        bus = EventBus()
        handler2_called = False

        def handler1(event: SampleEvent) -> None:
            raise ValueError("Handler 1 bug")

        def handler2(event: SampleEvent) -> None:
            nonlocal handler2_called
            handler2_called = True

        bus.subscribe(SampleEvent, handler1)
        bus.subscribe(SampleEvent, handler2)

        with pytest.raises(ValueError):
            bus.emit(SampleEvent(value="test"))

        # Handler 2 should NOT be called - fail fast on handler 1 bug
        assert not handler2_called


class TestNullEventBus:
    """Tests for NullEventBus (no-op implementation)."""

    def test_subscribe_is_noop(self) -> None:
        """Test NullEventBus.subscribe() is a no-op."""
        bus = NullEventBus()
        handler_called = False

        def handler(event: SampleEvent) -> None:
            nonlocal handler_called
            handler_called = True

        # Subscribing should not raise
        bus.subscribe(SampleEvent, handler)

        # But handler won't be called
        bus.emit(SampleEvent(value="test"))
        assert not handler_called

    def test_emit_is_noop(self) -> None:
        """Test NullEventBus.emit() is a no-op."""
        bus = NullEventBus()
        # Should not raise even though no subscribers
        bus.emit(SampleEvent(value="test"))


class SampleEventBusProtocol:
    """Tests for EventBusProtocol compatibility."""

    def test_eventbus_satisfies_protocol(self) -> None:
        """Test EventBus satisfies EventBusProtocol."""

        def accepts_protocol(bus: EventBusProtocol) -> None:
            bus.subscribe(SampleEvent, lambda e: None)
            bus.emit(SampleEvent(value="test"))

        # Should not raise type errors
        accepts_protocol(EventBus())

    def test_nulleventbus_satisfies_protocol(self) -> None:
        """Test NullEventBus satisfies EventBusProtocol."""

        def accepts_protocol(bus: EventBusProtocol) -> None:
            bus.subscribe(SampleEvent, lambda e: None)
            bus.emit(SampleEvent(value="test"))

        # Should not raise type errors
        accepts_protocol(NullEventBus())


class SampleEventBusEdgeCases:
    """Edge case tests for EventBus."""

    def test_subscribe_same_handler_multiple_times(self) -> None:
        """Test subscribing same handler multiple times calls it multiple times."""
        bus = EventBus()
        call_count = 0

        def handler(event: SampleEvent) -> None:
            nonlocal call_count
            call_count += 1

        # Subscribe same handler twice
        bus.subscribe(SampleEvent, handler)
        bus.subscribe(SampleEvent, handler)

        bus.emit(SampleEvent(value="test"))

        # Handler called twice (once per subscription)
        assert call_count == 2

    def test_handler_can_emit_events(self) -> None:
        """Test handlers can emit other events (no deadlock)."""
        bus = EventBus()
        received: list[str] = []

        def handler1(event: SampleEvent) -> None:
            received.append(f"handler1:{event.value}")
            # Emit another event from within handler
            bus.emit(AnotherEvent(count=42))

        def handler2(event: AnotherEvent) -> None:
            received.append(f"handler2:{event.count}")

        bus.subscribe(SampleEvent, handler1)
        bus.subscribe(AnotherEvent, handler2)

        bus.emit(SampleEvent(value="start"))

        # Both events processed
        assert received == ["handler1:start", "handler2:42"]
