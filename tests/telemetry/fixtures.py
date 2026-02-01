# tests/telemetry/fixtures.py
"""Reusable test fixtures for telemetry testing.

These fixtures provide:
1. TelemetryTestExporter - In-memory exporter that captures events for verification
2. MockTelemetryConfig - Simple config implementation for tests
3. pytest fixtures for common test patterns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts.enums import BackpressureMode, TelemetryGranularity
from elspeth.telemetry import TelemetryManager

if TYPE_CHECKING:
    from elspeth.contracts.events import TelemetryEvent


class TelemetryTestExporter:
    """In-memory exporter that captures events for test verification.

    This exporter implements ExporterProtocol and stores all events in memory
    for inspection during tests. It also provides assertion helpers.

    Example:
        exporter = TelemetryTestExporter()
        manager = TelemetryManager(config, exporters=[exporter])

        # ... run pipeline ...

        # Verify events
        exporter.assert_event_emitted("RunStarted")
        exporter.assert_event_emitted("ExternalCallCompleted", call_type="LLM")
    """

    def __init__(self, name: str = "test"):
        self._name = name
        self.events: list[TelemetryEvent] = []
        self.flush_count = 0
        self.close_count = 0
        self._configured = False

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the exporter (no-op for tests)."""
        self._configured = True

    def export(self, event: TelemetryEvent) -> None:
        """Capture event in memory."""
        self.events.append(event)

    def flush(self) -> None:
        """Track flush calls."""
        self.flush_count += 1

    def close(self) -> None:
        """Track close calls."""
        self.close_count += 1

    # =========================================================================
    # Assertion Helpers
    # =========================================================================

    def assert_event_emitted(
        self,
        event_type: str | type,
        **filters: Any,
    ) -> TelemetryEvent:
        """Assert that an event matching the criteria was emitted.

        Args:
            event_type: Event class or class name (e.g., "RunStarted" or RunStarted)
            **filters: Field values to match (e.g., call_type="LLM", status="SUCCESS")

        Returns:
            The first matching event

        Raises:
            AssertionError: If no matching event found
        """
        type_name = event_type if isinstance(event_type, str) else event_type.__name__

        matches = []
        for event in self.events:
            if type(event).__name__ != type_name:
                continue

            # Check all filters match
            all_match = True
            for key, expected in filters.items():
                actual = getattr(event, key, None)
                # Handle enum comparisons
                if hasattr(actual, "value"):
                    actual = actual.value
                if hasattr(expected, "value"):
                    expected = expected.value
                if actual != expected:
                    all_match = False
                    break

            if all_match:
                matches.append(event)

        assert matches, f"No {type_name} event found with filters {filters}. Events captured: {[type(e).__name__ for e in self.events]}"
        return matches[0]

    def assert_no_event_emitted(self, event_type: str | type) -> None:
        """Assert that no event of this type was emitted.

        Args:
            event_type: Event class or class name
        """
        type_name = event_type if isinstance(event_type, str) else event_type.__name__

        for event in self.events:
            if type(event).__name__ == type_name:
                raise AssertionError(f"Expected no {type_name} events, but found: {event}")

    def get_events_of_type(self, event_type: str | type) -> list[TelemetryEvent]:
        """Get all events of a specific type.

        Args:
            event_type: Event class or class name

        Returns:
            List of matching events (may be empty)
        """
        type_name = event_type if isinstance(event_type, str) else event_type.__name__
        return [e for e in self.events if type(e).__name__ == type_name]

    def clear(self) -> None:
        """Clear all captured events."""
        self.events.clear()

    @property
    def event_types(self) -> list[str]:
        """Get list of event type names captured (in order)."""
        return [type(e).__name__ for e in self.events]

    @property
    def event_count(self) -> int:
        """Get total number of events captured."""
        return len(self.events)


@dataclass
class MockTelemetryConfig:
    """Mock RuntimeTelemetryProtocol implementation for testing.

    Provides sensible defaults for test scenarios.
    """

    enabled: bool = True
    granularity: TelemetryGranularity = TelemetryGranularity.FULL
    fail_on_total_exporter_failure: bool = False
    backpressure_mode: BackpressureMode = BackpressureMode.BLOCK

    @property
    def exporter_configs(self) -> tuple:
        """Return empty tuple - exporters are passed directly in tests."""
        return ()


class FailingExporter:
    """Exporter that always fails export() calls.

    Use this to test failure isolation and error handling.
    """

    def __init__(self, name: str = "failing", *, fail_count: int | None = None):
        """Initialize failing exporter.

        Args:
            name: Exporter name for identification
            fail_count: If set, only fail this many times then succeed.
                       If None, always fail.
        """
        self._name = name
        self._fail_count = fail_count
        self._failures = 0
        self.export_attempts = 0

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        pass

    def export(self, event: TelemetryEvent) -> None:
        self.export_attempts += 1
        if self._fail_count is None or self._failures < self._fail_count:
            self._failures += 1
            raise RuntimeError(f"Simulated export failure in {self._name}")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


# =============================================================================
# pytest Fixtures
# =============================================================================


@pytest.fixture
def telemetry_test_exporter() -> TelemetryTestExporter:
    """Fixture providing a fresh test exporter."""
    return TelemetryTestExporter()


@pytest.fixture
def telemetry_config() -> MockTelemetryConfig:
    """Fixture providing default telemetry config."""
    return MockTelemetryConfig()


@pytest.fixture
def telemetry_manager_with_exporter(
    telemetry_test_exporter: TelemetryTestExporter,
    telemetry_config: MockTelemetryConfig,
) -> TelemetryManager:
    """Fixture providing TelemetryManager wired to test exporter.

    The manager is NOT closed automatically - tests should call close()
    if they need to verify flush/close behavior.
    """
    return TelemetryManager(telemetry_config, exporters=[telemetry_test_exporter])
