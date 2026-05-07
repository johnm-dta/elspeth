# tests/fixtures/telemetry.py
"""Telemetry test fixtures for tests.

Re-exports TelemetryTestExporter and MockTelemetryConfig from the
original test infrastructure. These are pure helper classes with no
dependency on old test conftest.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts.enums import BackpressureMode, TelemetryGranularity
from elspeth.telemetry import TelemetryManager

if TYPE_CHECKING:
    from elspeth.contracts.events import TelemetryEvent


class TelemetryTestExporter:
    """In-memory exporter that captures events for test verification.

    Implements ExporterProtocol and stores all events in memory
    for inspection during tests.
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

    def configure(self, config: Mapping[str, Any]) -> None:
        self._configured = True

    def export(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.close_count += 1

    def assert_event_emitted(
        self,
        event_type: str | type,
        **filters: Any,
    ) -> TelemetryEvent:
        """Assert that an event matching the criteria was emitted."""
        type_name = event_type if isinstance(event_type, str) else event_type.__name__

        matches = []
        for event in self.events:
            if type(event).__name__ != type_name:
                continue

            all_match = True
            for key, expected in filters.items():
                actual = getattr(event, key, None)
                if actual is not None and hasattr(actual, "value"):
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
        """Assert that no event of this type was emitted."""
        type_name = event_type if isinstance(event_type, str) else event_type.__name__

        for event in self.events:
            if type(event).__name__ == type_name:
                raise AssertionError(f"Expected no {type_name} events, but found: {event}")

    def get_events_of_type(self, event_type: str | type) -> list[TelemetryEvent]:
        """Get all events of a specific type."""
        type_name = event_type if isinstance(event_type, str) else event_type.__name__
        return [e for e in self.events if type(e).__name__ == type_name]

    def clear(self) -> None:
        self.events.clear()

    @property
    def event_types(self) -> list[str]:
        return [type(e).__name__ for e in self.events]

    @property
    def event_count(self) -> int:
        return len(self.events)


@dataclass
class MockTelemetryConfig:
    """Mock RuntimeTelemetryProtocol implementation for testing."""

    enabled: bool = True
    granularity: TelemetryGranularity = TelemetryGranularity.FULL
    fail_on_total_exporter_failure: bool = False
    backpressure_mode: BackpressureMode = BackpressureMode.BLOCK
    max_consecutive_failures: int = 10

    @property
    def exporter_configs(self) -> tuple[()]:
        return ()


class FailingExporter:
    """Exporter that always fails export() calls.

    Use this to test failure isolation and error handling.
    """

    def __init__(self, name: str = "failing", *, fail_count: int | None = None):
        self._name = name
        self._fail_count = fail_count
        self._failures = 0
        self.export_attempts = 0

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: Mapping[str, Any]) -> None:
        pass

    def export(self, event: TelemetryEvent) -> None:
        self.export_attempts += 1
        if self._fail_count is None or self._failures < self._fail_count:
            self._failures += 1
            raise ConnectionError(f"Simulated export failure in {self._name}")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


@contextmanager
def telemetry_manager_cleanup() -> Generator[None, None, None]:
    """Context manager that tracks and closes TelemetryManager instances.

    TelemetryManager starts a non-daemon background thread for async export.
    Without cleanup, these threads block pytest from exiting. This context
    manager tracks all instances created during its scope and closes them
    on exit.

    Usage in conftest.py:
        from tests.fixtures.telemetry import telemetry_manager_cleanup

        @pytest.fixture(autouse=True)
        def _auto_close_telemetry_managers():
            with telemetry_manager_cleanup():
                yield
    """
    import queue as queue_module

    created_managers: list[tuple[TelemetryManager, queue_module.Queue[Any]]] = []
    original_init = TelemetryManager.__init__

    def tracking_init(self: TelemetryManager, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        created_managers.append((self, self._queue))

    TelemetryManager.__init__ = tracking_init  # type: ignore[method-assign]

    try:
        yield
    finally:
        TelemetryManager.__init__ = original_init  # type: ignore[method-assign]
        cleanup_errors: list[str] = []

        for manager_index, (manager, original_queue) in enumerate(created_managers):
            try:
                manager._shutdown_event.set()

                current_queue = manager._queue
                if current_queue is not original_queue:
                    try:
                        original_queue.put_nowait(None)
                    except queue_module.Full:
                        try:
                            original_queue.get_nowait()
                            original_queue.put_nowait(None)
                        except (queue_module.Full, queue_module.Empty):
                            pass

                if manager._export_thread.is_alive():
                    manager.close()

                if manager._export_thread.is_alive():
                    manager._export_thread.join(timeout=1.0)
            except Exception as exc:
                cleanup_errors.append(f"manager[{manager_index}] {type(exc).__name__}: {exc}")

        if cleanup_errors:
            preview = "\n".join(cleanup_errors[:5])
            if len(cleanup_errors) > 5:
                preview += f"\n... and {len(cleanup_errors) - 5} more"
            raise RuntimeError(f"TelemetryManager cleanup failed — {len(cleanup_errors)} error(s).\n{preview}")


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
    """Fixture providing TelemetryManager wired to test exporter."""
    return TelemetryManager(telemetry_config, exporters=[telemetry_test_exporter])
