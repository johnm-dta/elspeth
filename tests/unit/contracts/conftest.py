# tests/unit/contracts/conftest.py
"""Contract test configuration.

Provides payload_store fixture (MockPayloadStore) needed by
orchestrator wiring tests in test_telemetry_contracts.py,
and autouse telemetry cleanup for TelemetryManager thread leaks.
"""

from collections.abc import Iterator

import pytest

from elspeth.contracts import tier_registry
from tests.fixtures.stores import payload_store  # noqa: F401
from tests.fixtures.telemetry import telemetry_manager_cleanup


@pytest.fixture(autouse=True)
def _auto_close_telemetry_managers() -> Iterator[None]:
    """Cleanup TelemetryManager instances after each test."""
    with telemetry_manager_cleanup():
        yield


@pytest.fixture
def reset_tier_registry() -> Iterator[None]:
    """Restore the Tier-1 registry after tests that mutate it."""
    before_registry = list(tier_registry._REGISTRY)
    before_reasons = dict(tier_registry._REASONS)
    before_frozen = tier_registry._FROZEN
    tier_registry._FROZEN = False
    yield
    tier_registry._REGISTRY[:] = before_registry
    tier_registry._REASONS.clear()
    tier_registry._REASONS.update(before_reasons)
    tier_registry._FROZEN = before_frozen
