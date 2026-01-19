"""Tests for RecoveryManager row data retrieval."""

import json
from datetime import UTC, datetime

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


class TestRecoveryManagerRowData:
    """Tests for get_unprocessed_row_data()."""

    def test_get_unprocessed_row_data_returns_row_dicts(
        self,
        recovery_manager: RecoveryManager,
        payload_store: FilesystemPayloadStore,
        run_with_checkpoint_and_payloads: str,
    ) -> None:
        """get_unprocessed_row_data returns actual row data, not just IDs."""
        run_id = run_with_checkpoint_and_payloads

        # Get row data for unprocessed rows
        row_data_list = recovery_manager.get_unprocessed_row_data(
            run_id=run_id,
            payload_store=payload_store,
        )

        # Should return list of (row_index, row_data) tuples
        assert len(row_data_list) == 2  # rows 3 and 4
        assert all(isinstance(item, tuple) for item in row_data_list)
        assert all(len(item) == 2 for item in row_data_list)

        # Verify row indices are correct and in order
        indices = [item[0] for item in row_data_list]
        assert indices == [3, 4]

        # Verify row data is correct
        for row_index, row_data in row_data_list:
            assert isinstance(row_data, dict)
            assert row_data["id"] == row_index
            assert row_data["value"] == f"data-{row_index}"

    def test_get_unprocessed_row_data_empty_when_all_processed(
        self,
        recovery_manager: RecoveryManager,
        payload_store: FilesystemPayloadStore,
        db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """Returns empty list when all rows are processed."""
        # Create run where checkpoint is at last row
        run_id = "test-all-processed"
        now = datetime.now(UTC)

        with db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="x",
                    settings_json="{}",
                    canonical_version="v1",
                    status="failed",
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node",
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

            # Single row
            payload_ref = payload_store.store(json.dumps({"id": 0}).encode())
            conn.execute(
                rows_table.insert().values(
                    row_id="row-0",
                    run_id=run_id,
                    source_node_id="node",
                    row_index=0,
                    source_data_hash="h",
                    source_data_ref=payload_ref,
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-0",
                    row_id="row-0",
                    created_at=now,
                )
            )
            conn.commit()

        # Checkpoint at last row
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-0",
            node_id="node",
            sequence_number=0,
        )

        result = recovery_manager.get_unprocessed_row_data(run_id, payload_store)
        assert result == []

    def test_get_unprocessed_row_data_raises_on_missing_payload(
        self,
        recovery_manager: RecoveryManager,
        payload_store: FilesystemPayloadStore,
        db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """Raises ValueError when payload cannot be retrieved."""
        run_id = "test-missing-payload"
        now = datetime.now(UTC)

        with db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="x",
                    settings_json="{}",
                    canonical_version="v1",
                    status="failed",
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node",
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

            # Row with invalid payload ref (not stored)
            conn.execute(
                rows_table.insert().values(
                    row_id="row-0",
                    run_id=run_id,
                    source_node_id="node",
                    row_index=0,
                    source_data_hash="h",
                    source_data_ref="nonexistent_hash",  # Not in payload store
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-0",
                    row_id="row-0",
                    created_at=now,
                )
            )

            # Second row (will be unprocessed)
            conn.execute(
                rows_table.insert().values(
                    row_id="row-1",
                    run_id=run_id,
                    source_node_id="node",
                    row_index=1,
                    source_data_hash="h",
                    source_data_ref="also_nonexistent",
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-1",
                    row_id="row-1",
                    created_at=now,
                )
            )
            conn.commit()

        # Checkpoint at row 0
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-0",
            node_id="node",
            sequence_number=0,
        )

        with pytest.raises(ValueError, match="payload has been purged"):
            recovery_manager.get_unprocessed_row_data(run_id, payload_store)
