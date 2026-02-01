"""Shared fixtures for landscape tests.

Fixture Scoping Strategy
========================
- landscape_db: Module-scoped for performance (schema creation is expensive)
- recorder: Function-scoped (lightweight wrapper, uses shared db)

Tests in this package should use unique run_ids to avoid data pollution.
The shared database contains data from all tests in the module, but each
test's queries should filter by its own run_id.
"""

import pytest

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder


@pytest.fixture(scope="module")
def landscape_db() -> LandscapeDB:
    """Create a module-scoped in-memory landscape database.

    Module scope avoids repeated schema creation (15+ tables, indexes)
    which takes ~5-10ms per instantiation. With hundreds of tests,
    this saves 20-30 seconds of test runtime.

    Tests should use unique run_ids to isolate their data.
    """
    return LandscapeDB.in_memory()


@pytest.fixture
def recorder(landscape_db: LandscapeDB) -> LandscapeRecorder:
    """Create a LandscapeRecorder with the shared test database.

    Function-scoped because the recorder is a lightweight wrapper.
    Uses the module-scoped landscape_db for actual storage.
    """
    return LandscapeRecorder(landscape_db)
