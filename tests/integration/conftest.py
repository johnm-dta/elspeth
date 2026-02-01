"""Shared fixtures for integration tests.

Fixture Scoping Strategy
========================
- landscape_db: Module-scoped for performance (schema creation is expensive)
- recorder: Function-scoped (lightweight wrapper, uses shared db)

Integration tests should use unique run_ids to avoid data pollution.
The shared database contains data from all tests in the module, but each
test's queries should filter by its own run_id.
"""

from pathlib import Path
from typing import Any

import pytest

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder


@pytest.fixture(scope="module")
def landscape_db() -> LandscapeDB:
    """Create a module-scoped in-memory landscape database.

    Module scope avoids repeated schema creation (15+ tables, indexes)
    which takes ~5-10ms per instantiation. With many integration tests,
    this saves significant test runtime.

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


# Resume/Checkpoint test fixtures (shared by resume test files)


@pytest.fixture
def resume_test_env(tmp_path: Path) -> dict[str, Any]:
    """Set up complete test environment for resume/checkpoint tests.

    Provides:
    - db: LandscapeDB (file-based for resume tests)
    - payload_store: FilesystemPayloadStore
    - checkpoint_manager: CheckpointManager
    - recovery_manager: RecoveryManager
    - checkpoint_config: RuntimeCheckpointConfig
    - recorder: LandscapeRecorder
    - tmp_path: Path
    """
    from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
    from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
    from elspeth.core.config import CheckpointSettings
    from elspeth.core.payload_store import FilesystemPayloadStore

    db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
    payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    checkpoint_mgr = CheckpointManager(db)
    recovery_mgr = RecoveryManager(db, checkpoint_mgr)
    checkpoint_settings = CheckpointSettings(frequency="every_row")
    checkpoint_config = RuntimeCheckpointConfig.from_settings(checkpoint_settings)
    recorder = LandscapeRecorder(db)

    return {
        "db": db,
        "payload_store": payload_store,
        "checkpoint_manager": checkpoint_mgr,
        "recovery_manager": recovery_mgr,
        "checkpoint_config": checkpoint_config,
        "recorder": recorder,
        "tmp_path": tmp_path,
    }
