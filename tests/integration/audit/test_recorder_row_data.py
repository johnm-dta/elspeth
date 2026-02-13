# tests/core/landscape/test_recorder_row_data.py
"""Tests for LandscapeRecorder.get_row_data() with explicit states."""

import json
from pathlib import Path

from elspeth.contracts.enums import NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.landscape.row_data import RowDataResult, RowDataState
from elspeth.core.payload_store import FilesystemPayloadStore

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


class TestGetRowDataExplicitStates:
    """Tests for get_row_data() returning RowDataResult."""

    def test_row_not_found(self, tmp_path: Path, payload_store) -> None:
        """Returns ROW_NOT_FOUND when row doesn't exist."""
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        result = recorder.get_row_data("nonexistent-row")

        assert isinstance(result, RowDataResult)
        assert result.state == RowDataState.ROW_NOT_FOUND
        assert result.data is None

    def test_never_stored(self, tmp_path: Path) -> None:
        """Returns NEVER_STORED when payload_store is not configured.

        When LandscapeRecorder is created without a payload_store, rows are
        created without payload storage (source_data_ref is None).
        """
        db = LandscapeDB.in_memory()
        # No payload_store configured - payloads won't be stored
        recorder = LandscapeRecorder(db, payload_store=None)

        # Create run and row - payload will not be stored
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
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

    def test_store_not_configured(self, tmp_path: Path, payload_store) -> None:
        """Returns STORE_NOT_CONFIGURED when payload_store is None."""
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")

        # Create recorder WITH payload store to store the row
        recorder_with_store = LandscapeRecorder(db, payload_store=payload_store)
        run = recorder_with_store.begin_run(config={}, canonical_version="v1")
        source = recorder_with_store.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
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

    def test_purged(self, tmp_path: Path, payload_store) -> None:
        """Returns PURGED when payload_store raises KeyError."""
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
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

    def test_available(self, tmp_path: Path, payload_store) -> None:
        """Returns AVAILABLE with data when payload exists."""
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
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


class TestGetRowDataTier1Corruption:
    """Tier 1 corruption tests: get_row_data must propagate integrity failures.

    Per the Three-Tier Trust Model, corrupted audit data (Tier 1) must crash
    immediately - no silent recovery, no coercion, no defaults.
    """

    def test_integrity_error_propagates(self, tmp_path: Path, payload_store) -> None:
        """IntegrityError from corrupted payload must propagate (not be swallowed).

        When payload bytes have been tampered with (hash mismatch),
        FilesystemPayloadStore.retrieve raises IntegrityError.
        This must propagate to the caller - Tier 1 data corruption
        is a crash-worthy event.
        """
        from elspeth.contracts.payload_store import IntegrityError

        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Store valid payload and create row
        test_data = {"field": "value"}
        payload_ref = payload_store.store(json.dumps(test_data).encode())
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=test_data,
            payload_ref=payload_ref,
        )

        # Corrupt the payload file by tampering with its contents
        # FilesystemPayloadStore uses hash[:2]/hash as path structure
        payload_path = tmp_path / "payloads" / payload_ref[:2] / payload_ref
        payload_path.write_bytes(b"corrupted data that won't match hash")

        # get_row_data must raise IntegrityError, not silently return garbage
        import pytest

        with pytest.raises(IntegrityError):
            recorder.get_row_data(row.row_id)

    def test_invalid_json_propagates(self, tmp_path: Path, payload_store) -> None:
        """Invalid JSON in payload must propagate JSONDecodeError.

        When payload bytes are valid (hash matches) but contain non-JSON,
        the JSON parse failure must propagate - this indicates
        Tier 1 corruption or a bug in our code.
        """
        import pytest

        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Store non-JSON bytes (but with valid hash)
        non_json_data = b"this is not valid JSON {"
        payload_ref = payload_store.store(non_json_data)
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"placeholder": "ignored"},  # This won't be retrieved
            payload_ref=payload_ref,
        )

        # get_row_data must raise JSONDecodeError, not return None or garbage
        with pytest.raises(json.JSONDecodeError):
            recorder.get_row_data(row.row_id)

    def test_non_object_json_raises_audit_integrity_error(self, tmp_path: Path, payload_store) -> None:
        """JSON payloads must decode to objects for AVAILABLE row data."""
        import pytest

        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        payload_ref = payload_store.store(json.dumps([1, 2, 3]).encode())
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"placeholder": "ignored"},
            payload_ref=payload_ref,
        )

        with pytest.raises(AuditIntegrityError, match="expected JSON object"):
            recorder.get_row_data(row.row_id)
