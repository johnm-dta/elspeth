"""Shared fixtures for system tests.

System tests run full pipelines end-to-end, so they need more comprehensive
fixtures than unit/integration tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def system_landscape_db(tmp_path: Path) -> Iterator[LandscapeDB]:
    """Create a file-based landscape database for system tests.

    Unlike integration tests that use in-memory DBs, system tests
    use file-based DBs to test persistence and recovery scenarios.
    """
    db_path = tmp_path / "landscape.db"
    db = LandscapeDB(f"sqlite:///{db_path}")
    yield db
    db.close()


@pytest.fixture
def system_recorder(system_landscape_db: LandscapeDB) -> LandscapeRecorder:
    """Create a LandscapeRecorder for system tests."""
    return LandscapeRecorder(system_landscape_db)


@pytest.fixture
def payload_store_path(tmp_path: Path) -> Path:
    """Create a temporary directory for payload storage."""
    store_path = tmp_path / "payloads"
    store_path.mkdir()
    return store_path
