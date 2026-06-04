# tests/e2e/conftest.py
"""E2E test configuration.

File-based SQLite databases and real payload stores.
No mocks except for external services.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from tests.fixtures.landscape import make_factory


@pytest.fixture
def system_landscape_db(tmp_path: Path) -> LandscapeDB:
    """Function-scoped file-based LandscapeDB for E2E tests."""
    db_path = tmp_path / "audit.db"
    return LandscapeDB(f"sqlite:///{db_path}")


@pytest.fixture
def system_landscape_factory(system_landscape_db: LandscapeDB) -> RecorderFactory:
    """Function-scoped recorder factory for E2E tests."""
    return make_factory(system_landscape_db)


@pytest.fixture
def payload_store_path(tmp_path: Path) -> Path:
    """Create a payload store directory for E2E tests."""
    store_path = tmp_path / "payloads"
    store_path.mkdir()
    return store_path


@pytest.fixture
def example_pipeline_dir() -> Path:
    """Locate the examples/ directory in the repository."""
    return Path(__file__).resolve().parents[2] / "examples"
