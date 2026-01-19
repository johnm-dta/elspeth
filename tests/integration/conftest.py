"""Shared fixtures for integration tests."""

import pytest

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder


@pytest.fixture
def landscape_db() -> LandscapeDB:
    """Create a temporary in-memory landscape database for testing."""
    return LandscapeDB.in_memory()


@pytest.fixture
def recorder(landscape_db: LandscapeDB) -> LandscapeRecorder:
    """Create a LandscapeRecorder with the test database."""
    return LandscapeRecorder(landscape_db)
