"""Shared fixtures for checkpoint tests."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    runs_table,
    tokens_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore


@pytest.fixture
def db(tmp_path: Path) -> LandscapeDB:
    """Create test database."""
    return LandscapeDB(f"sqlite:///{tmp_path}/test.db")


@pytest.fixture
def payload_store(tmp_path: Path) -> FilesystemPayloadStore:
    """Create test payload store."""
    return FilesystemPayloadStore(tmp_path / "payloads")


@pytest.fixture
def checkpoint_manager(db: LandscapeDB) -> CheckpointManager:
    """Create checkpoint manager."""
    return CheckpointManager(db)


@pytest.fixture
def recovery_manager(
    db: LandscapeDB, checkpoint_manager: CheckpointManager
) -> RecoveryManager:
    """Create recovery manager."""
    return RecoveryManager(db, checkpoint_manager)


@pytest.fixture
def run_with_checkpoint_and_payloads(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    payload_store: FilesystemPayloadStore,
) -> str:
    """Create a failed run with checkpoint and payload data.

    Creates 5 rows (0-4), checkpoint at row 2, so rows 3-4 are unprocessed.
    All rows have payload data stored.
    """
    run_id = "test-run-resume"
    now = datetime.now(UTC)

    with db.engine.connect() as conn:
        # Create run (failed status)
        conn.execute(
            runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                config_hash="test",
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status="failed",
            )
        )

        # Create source node
        conn.execute(
            nodes_table.insert().values(
                node_id="source-node",
                run_id=run_id,
                plugin_name="csv",
                node_type="source",
                plugin_version="1.0",
                determinism="io_read",
                config_hash="x",
                config_json="{}",
                registered_at=now,
            )
        )

        # Create rows with payload data
        for i in range(5):
            row_id = f"row-{i:03d}"
            row_data = {"id": i, "value": f"data-{i}"}

            # Store payload - CORRECT SIGNATURE: store(content) returns hash
            payload_bytes = json.dumps(row_data).encode("utf-8")
            payload_ref = payload_store.store(payload_bytes)

            conn.execute(
                rows_table.insert().values(
                    row_id=row_id,
                    run_id=run_id,
                    source_node_id="source-node",
                    row_index=i,
                    source_data_hash=f"hash{i}",
                    source_data_ref=payload_ref,  # Reference to stored payload
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id=f"tok-{i:03d}",
                    row_id=row_id,
                    created_at=now,
                )
            )

        conn.commit()

    # Create checkpoint at row 2 (rows 3-4 are unprocessed)
    checkpoint_manager.create_checkpoint(
        run_id=run_id,
        token_id="tok-002",
        node_id="source-node",
        sequence_number=2,
    )

    return run_id
