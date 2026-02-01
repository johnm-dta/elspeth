# tests/integration/test_schema_not_null_constraints.py
"""Integration tests for Bug #7: Schema allows NULL on audit fields.

These tests verify that audit-critical fields in the database schema
enforce NOT NULL constraints, preventing silent data corruption.
"""

from pathlib import Path

import pytest
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError

from elspeth.contracts import Determinism, NodeType, RunStatus
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import checkpoints_table


class TestSchemaNotNullConstraints:
    """Integration tests for schema NOT NULL constraints."""

    @pytest.fixture
    def test_db(self, tmp_path: Path) -> LandscapeDB:
        """Create a test database with full schema."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        # Schema is created automatically by LandscapeDB
        return db

    def test_checkpoint_upstream_topology_hash_not_null(
        self,
        test_db: LandscapeDB,
    ) -> None:
        """Verify upstream_topology_hash field does not allow NULL.

        Scenario:
        1. Attempt to insert checkpoint with NULL upstream_topology_hash
        2. Verify: IntegrityError raised (NOT NULL constraint)

        This is Bug #7 fix: audit-critical topology hash must never be NULL.
        """
        from datetime import UTC, datetime

        checkpoint_data = {
            "checkpoint_id": "test_checkpoint_1",
            "run_id": "test_run",
            "token_id": "test_token",
            "node_id": "test_node",
            "sequence_number": 1,
            "upstream_topology_hash": None,  # ← Intentionally NULL
            "checkpoint_node_config_hash": "valid_hash",
            "created_at": datetime.now(UTC),
        }

        # Attempt to insert with NULL hash should fail
        with pytest.raises(IntegrityError) as exc_info, test_db.engine.begin() as conn:
            conn.execute(insert(checkpoints_table).values(**checkpoint_data))

        # Verify error message mentions NOT NULL constraint
        error_msg = str(exc_info.value).lower()
        assert "not null" in error_msg or "null constraint" in error_msg
        assert "upstream_topology_hash" in error_msg

    def test_checkpoint_node_config_hash_not_null(
        self,
        test_db: LandscapeDB,
    ) -> None:
        """Verify checkpoint_node_config_hash field does not allow NULL.

        Scenario:
        1. Attempt to insert checkpoint with NULL checkpoint_node_config_hash
        2. Verify: IntegrityError raised (NOT NULL constraint)

        This is Bug #7 fix: audit-critical config hash must never be NULL.
        """
        from datetime import UTC, datetime

        checkpoint_data = {
            "checkpoint_id": "test_checkpoint_2",
            "run_id": "test_run",
            "token_id": "test_token",
            "node_id": "test_node",
            "sequence_number": 2,
            "upstream_topology_hash": "valid_hash",
            "checkpoint_node_config_hash": None,  # ← Intentionally NULL
            "created_at": datetime.now(UTC),
        }

        # Attempt to insert with NULL hash should fail
        with pytest.raises(IntegrityError) as exc_info, test_db.engine.begin() as conn:
            conn.execute(insert(checkpoints_table).values(**checkpoint_data))

        # Verify error message mentions NOT NULL constraint
        error_msg = str(exc_info.value).lower()
        assert "not null" in error_msg or "null constraint" in error_msg
        assert "checkpoint_node_config_hash" in error_msg

    def test_checkpoint_with_valid_hashes_succeeds(
        self,
        test_db: LandscapeDB,
    ) -> None:
        """Verify checkpoint with valid (non-NULL) hashes succeeds.

        Scenario:
        1. Create required parent records (run, node, token)
        2. Insert checkpoint with valid topology and config hashes
        3. Verify: Insert succeeds
        4. Verify: Can retrieve checkpoint from database

        This confirms the NOT NULL constraints don't break valid inserts.
        """
        from datetime import UTC, datetime

        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table, tokens_table

        now = datetime.now(UTC)

        # Create parent records to satisfy foreign keys
        with test_db.engine.begin() as conn:
            # Create run
            conn.execute(
                insert(runs_table).values(
                    run_id="test_run",
                    started_at=now,
                    config_hash="test_hash",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.RUNNING,
                )
            )

            # Create node
            conn.execute(
                insert(nodes_table).values(
                    node_id="test_node",
                    run_id="test_run",
                    plugin_name="test_plugin",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test_hash",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create row
            conn.execute(
                insert(rows_table).values(
                    row_id="test_row",
                    run_id="test_run",
                    source_node_id="test_node",
                    row_index=0,
                    source_data_hash="test_hash",
                    created_at=now,
                )
            )

            # Create token
            conn.execute(
                insert(tokens_table).values(
                    token_id="test_token",
                    row_id="test_row",
                    created_at=now,
                )
            )

            # Now insert checkpoint with valid hashes
            conn.execute(
                insert(checkpoints_table).values(
                    checkpoint_id="test_checkpoint_3",
                    run_id="test_run",
                    token_id="test_token",
                    node_id="test_node",
                    sequence_number=3,
                    upstream_topology_hash="abc123def456",  # Valid hash
                    checkpoint_node_config_hash="xyz789uvw012",  # Valid hash
                    created_at=now,
                )
            )

        # Verify checkpoint was inserted
        with test_db.engine.connect() as conn:
            result = conn.execute(select(checkpoints_table).where(checkpoints_table.c.checkpoint_id == "test_checkpoint_3")).fetchone()

        assert result is not None
        assert result.upstream_topology_hash == "abc123def456"
        assert result.checkpoint_node_config_hash == "xyz789uvw012"

    def test_both_hashes_null_fails(
        self,
        test_db: LandscapeDB,
    ) -> None:
        """Verify both topology hashes cannot be NULL simultaneously.

        Scenario:
        1. Attempt to insert checkpoint with both hashes NULL
        2. Verify: IntegrityError raised

        This is the most severe violation of audit integrity.
        """
        from datetime import UTC, datetime

        checkpoint_data = {
            "checkpoint_id": "test_checkpoint_4",
            "run_id": "test_run",
            "token_id": "test_token",
            "node_id": "test_node",
            "sequence_number": 4,
            "upstream_topology_hash": None,  # ← NULL
            "checkpoint_node_config_hash": None,  # ← NULL
            "created_at": datetime.now(UTC),
        }

        # Attempt to insert with both hashes NULL should fail
        with pytest.raises(IntegrityError) as exc_info, test_db.engine.begin() as conn:
            conn.execute(insert(checkpoints_table).values(**checkpoint_data))

        # Verify error mentions NOT NULL constraint
        error_msg = str(exc_info.value).lower()
        assert "not null" in error_msg or "null constraint" in error_msg
