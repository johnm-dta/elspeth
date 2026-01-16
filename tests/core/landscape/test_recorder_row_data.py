# tests/core/landscape/test_recorder_row_data.py
"""Tests for LandscapeRecorder.get_row_data() with explicit states."""

import json
from pathlib import Path

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.landscape.row_data import RowDataResult, RowDataState
from elspeth.core.payload_store import FilesystemPayloadStore


class TestGetRowDataExplicitStates:
    """Tests for get_row_data() returning RowDataResult."""

    def test_row_not_found(self, tmp_path: Path) -> None:
        """Returns ROW_NOT_FOUND when row doesn't exist."""
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        result = recorder.get_row_data("nonexistent-row")

        assert isinstance(result, RowDataResult)
        assert result.state == RowDataState.ROW_NOT_FOUND
        assert result.data is None

    def test_never_stored(self, tmp_path: Path) -> None:
        """Returns NEVER_STORED when source_data_ref is None."""
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        # Create run and row without payload_ref
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},
            # No payload_ref - source_data_ref will be None
        )

        result = recorder.get_row_data(row.row_id)

        assert result.state == RowDataState.NEVER_STORED
        assert result.data is None

    def test_store_not_configured(self, tmp_path: Path) -> None:
        """Returns STORE_NOT_CONFIGURED when payload_store is None."""
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")

        # Create recorder WITH payload store to store the row
        recorder_with_store = LandscapeRecorder(db, payload_store=payload_store)
        run = recorder_with_store.begin_run(config={}, canonical_version="v1")
        source = recorder_with_store.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        # Store payload and create row with ref
        test_data = {"field": "value"}
        payload_ref = payload_store.store(json.dumps(test_data).encode())
        row = recorder_with_store.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=test_data,
            payload_ref=payload_ref,
        )

        # Create new recorder WITHOUT payload store, using same db
        recorder_no_store = LandscapeRecorder(db, payload_store=None)

        result = recorder_no_store.get_row_data(row.row_id)

        assert result.state == RowDataState.STORE_NOT_CONFIGURED
        assert result.data is None

    def test_purged(self, tmp_path: Path) -> None:
        """Returns PURGED when payload_store raises KeyError."""
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        # Store payload and create row with ref
        test_data = {"field": "value"}
        payload_ref = payload_store.store(json.dumps(test_data).encode())
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=test_data,
            payload_ref=payload_ref,
        )

        # Delete the payload (simulating retention policy purge)
        payload_store.delete(payload_ref)

        result = recorder.get_row_data(row.row_id)

        assert result.state == RowDataState.PURGED
        assert result.data is None

    def test_available(self, tmp_path: Path) -> None:
        """Returns AVAILABLE with data when payload exists."""
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        # Store payload and create row with ref
        test_data = {"field": "value", "number": 42}
        payload_ref = payload_store.store(json.dumps(test_data).encode())
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=test_data,
            payload_ref=payload_ref,
        )

        result = recorder.get_row_data(row.row_id)

        assert result.state == RowDataState.AVAILABLE
        assert result.data == test_data
