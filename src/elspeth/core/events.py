"""Event bus for pipeline observability.

Provides a simple synchronous event bus for emitting domain events from the
orchestrator to CLI formatters. Designed for clean separation between domain
logic (orchestrator) and presentation logic (CLI).
"""

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class EventBusProtocol(Protocol):
    """Protocol for event bus implementations.

    Allows both EventBus and NullEventBus to satisfy the interface
    without inheritance, preventing accidental substitution bugs.
    """

    def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Subscribe a handler to an event type."""
        ...

    def emit(self, event: T) -> None:
        """Emit an event to all subscribers."""
        ...


class EventBus:
    """Simple synchronous event bus for pipeline observability.

    Events are dispatched synchronously to all subscribers. Handler
    exceptions propagate to the caller - formatters are "our code"
    per CLAUDE.md, so bugs should crash immediately.

    Example:
        bus = EventBus()
        bus.subscribe(PhaseStarted, lambda e: print(f"[{e.phase}] Starting"))
        bus.emit(PhaseStarted(phase=PipelinePhase.CONFIG, action=PhaseAction.LOADING))
    """

    def __init__(self) -> None:
        self._subscribers: dict[type, list[Callable[[Any], None]]] = {}

    def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Subscribe a handler to an event type.

        Args:
            event_type: The event class to subscribe to
            handler: Callable that receives the event instance

        Example:
            def on_phase_started(event: PhaseStarted):
                print(f"Phase {event.phase} started")

            bus.subscribe(PhaseStarted, on_phase_started)
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def emit(self, event: T) -> None:
        """Emit an event to all subscribers.

        Handlers are called synchronously in subscription order.
        Exceptions propagate - handlers are system code, not user code.

        Events with no subscribers are silently ignored - this is
        intentional for decoupling. Formatters subscribe to events
        they care about.

        Args:
            event: The event instance to emit

        Example:
            bus.emit(PhaseStarted(
                phase=PipelinePhase.CONFIG,
                action=PhaseAction.LOADING,
                target="settings.yaml"
            ))
        """
        handlers = self._subscribers.get(type(event), [])
        for handler in handlers:
            handler(event)


class NullEventBus:
    """No-op event bus for library use where no CLI is present.

    IMPORTANT: Does NOT inherit from EventBus. Calling subscribe() on
    this is a no-op by design - use only when you genuinely don't want
    event observability (e.g., programmatic API usage, testing).

    Per CLAUDE.md: "A defective plugin that silently produces wrong
    results is worse than a crash." If someone subscribes expecting
    callbacks, inheritance would hide the bug. Protocol-based design
    makes the no-op behavior explicit.

    Example:
        # Library usage without CLI observability
        orchestrator = Orchestrator(db, event_bus=NullEventBus())
    """

    def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """No-op subscription - handler will never be called."""
        pass  # Intentional no-op - no subscribers expected

    def emit(self, event: T) -> None:
        """No-op emission - no handlers to call."""
        pass  # Intentional no-op
