"""Fixtures for repository-level audit persistence tests."""

from __future__ import annotations

import pytest
from tests.fixtures.landscape import make_factory, make_landscape_db

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory


@pytest.fixture
def landscape_db() -> LandscapeDB:
    """Function-scoped in-memory LandscapeDB for repository round-trip tests."""
    return make_landscape_db()


@pytest.fixture
def landscape_factory(landscape_db: LandscapeDB) -> RecorderFactory:
    """Function-scoped RecorderFactory bound to the test database."""
    return make_factory(landscape_db)
