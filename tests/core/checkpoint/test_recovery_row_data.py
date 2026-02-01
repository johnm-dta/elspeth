"""Tests for RecoveryManager row data retrieval."""

import json
from datetime import UTC, datetime

import pytest

from elspeth.contracts import Determinism, NodeType, RowOutcome, RunStatus
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    runs_table,
    token_outcomes_table,
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

        # Create mock schema (required after Bug #4 fix)
        from elspeth.plugins.schema_factory import _create_dynamic_schema

        mock_schema = _create_dynamic_schema("MockSchema")

        # Get row data for unprocessed rows
        row_data_list = recovery_manager.get_unprocessed_row_data(
            run_id=run_id,
            payload_store=payload_store,
            source_schema_class=mock_schema,
        )

        # Should return list of (row_id, row_index, row_data) tuples
        assert len(row_data_list) == 2  # rows 3 and 4
        assert all(isinstance(item, tuple) for item in row_data_list)
        assert all(len(item) == 3 for item in row_data_list)

        # Verify row indices are correct and in order
        indices = [item[1] for item in row_data_list]
        assert indices == [3, 4]

        # Verify row_ids are strings
        row_ids = [item[0] for item in row_data_list]
        assert all(isinstance(r, str) for r in row_ids)

        # Verify row data is correct
        for _row_id, row_index, row_data in row_data_list:
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
                    status=RunStatus.FAILED,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node",
                    run_id=run_id,
                    plugin_name="csv",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.IO_READ,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create sink node (needed for token outcomes)
            conn.execute(
                nodes_table.insert().values(
                    node_id="sink",
                    run_id=run_id,
                    plugin_name="csv_sink",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
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

            # Record terminal outcome for the row (all rows processed)
            # P1-2026-01-22 fix: get_unprocessed_rows uses token_outcomes, not row_index
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="outcome-0",
                    run_id=run_id,
                    token_id="tok-0",
                    outcome=RowOutcome.COMPLETED.value,
                    is_terminal=1,
                    recorded_at=now,
                    sink_name="sink",
                )
            )

            conn.commit()

        # Checkpoint at last row
        from tests.core.checkpoint.conftest import _create_test_graph

        graph = _create_test_graph(checkpoint_node="node")
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-0",
            node_id="node",
            sequence_number=0,
            graph=graph,
        )

        # Create mock schema (required after Bug #4 fix)
        from elspeth.plugins.schema_factory import _create_dynamic_schema

        mock_schema = _create_dynamic_schema("MockSchema")

        result = recovery_manager.get_unprocessed_row_data(run_id, payload_store, source_schema_class=mock_schema)
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
                    status=RunStatus.FAILED,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node",
                    run_id=run_id,
                    plugin_name="csv",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.IO_READ,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create sink node (needed for token outcomes)
            conn.execute(
                nodes_table.insert().values(
                    node_id="sink",
                    run_id=run_id,
                    plugin_name="csv_sink",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Row 0 (will be marked as processed via outcome)
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

            # Record terminal outcome for row-0 (processed)
            # P1-2026-01-22 fix: get_unprocessed_rows uses token_outcomes, not row_index
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="outcome-0",
                    run_id=run_id,
                    token_id="tok-0",
                    outcome=RowOutcome.COMPLETED.value,
                    is_terminal=1,
                    recorded_at=now,
                    sink_name="sink",
                )
            )

            # Row 1 (unprocessed - no outcome, will have missing payload)
            conn.execute(
                rows_table.insert().values(
                    row_id="row-1",
                    run_id=run_id,
                    source_node_id="node",
                    row_index=1,
                    source_data_hash="h",
                    source_data_ref="d" * 64,  # Valid hex format, but doesn't exist in store
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
        from tests.core.checkpoint.conftest import _create_test_graph

        graph = _create_test_graph(checkpoint_node="node")
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-0",
            node_id="node",
            sequence_number=0,
            graph=graph,
        )

        # Create mock schema (required after Bug #4 fix)
        from elspeth.plugins.schema_factory import _create_dynamic_schema

        mock_schema = _create_dynamic_schema("MockSchema")

        with pytest.raises(ValueError, match="payload has been purged"):
            recovery_manager.get_unprocessed_row_data(run_id, payload_store, source_schema_class=mock_schema)
