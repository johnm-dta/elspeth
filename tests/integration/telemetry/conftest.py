# tests/integration/telemetry/conftest.py
"""Telemetry integration test configuration.

Provides autouse cleanup for TelemetryManager thread leaks.
"""

from collections.abc import Iterator

import pytest

from tests.fixtures.telemetry import telemetry_manager_cleanup


@pytest.fixture(autouse=True)
def _auto_close_telemetry_managers() -> Iterator[None]:
    """Cleanup TelemetryManager instances after each test."""
    with telemetry_manager_cleanup():
        yield
