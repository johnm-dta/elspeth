# tests/unit/conftest.py
"""Unit test configuration.

Unit tests verify component logic in isolation from external services.
In-memory SQLite (LandscapeDB.in_memory()) is permitted for audit-trail
verification — it is fast, deterministic, requires no external service,
and avoids the anti-pattern of testing mocks instead of behavior.

Provides shared in-memory fixtures (payload store, plugin manager).
"""

import pytest

from elspeth.contracts.payload_store import PayloadStore
from elspeth.plugins.infrastructure.manager import PluginManager
from tests.fixtures.stores import MockPayloadStore


@pytest.fixture
def payload_store() -> PayloadStore:
    """In-memory PayloadStore for tests that need artifact storage."""
    return MockPayloadStore()


@pytest.fixture
def plugin_manager() -> PluginManager:
    """Standard plugin manager with builtin plugins registered."""
    manager = PluginManager()
    manager.register_builtin_plugins()
    return manager
