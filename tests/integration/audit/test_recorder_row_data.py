# tests/core/landscape/test_recorder_row_data.py
"""Tests for RecorderFactory query.get_row_data() with explicit states."""

import json
from pathlib import Path

from elspeth.contracts.enums import NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
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
        factory = RecorderFactory(db, payload_store=payload_store)

        result = factory.query.get_row_data("nonexistent-row")

        assert isinstance(result, RowDataResult)
        assert result.state == RowDataState.ROW_NOT_FOUND
        assert result.data is None

    def test_never_stored(self, tmp_path: Path) -> None:
        """Returns NEVER_STORED when payload_store is not configured.

        When RecorderFactory is created without a payload_store, rows are
        created without payload storage (source_data_ref is None).
        """
        db = LandscapeDB.in_memory()
        # No payload_store configured - payloads won't be stored
        factory = RecorderFactory(db, payload_store=None)

        # Create run and row - payload will not be stored
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},
            # No payload_store configured - source_data_ref will be None
        )

        result = factory.query.get_row_data(row.row_id)

        assert result.state == RowDataState.NEVER_STORED
        assert result.data is None

    def test_store_not_configured(self, tmp_path: Path, payload_store) -> None:
        """Returns STORE_NOT_CONFIGURED when payload_store is None."""
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")

        # Create factory WITH payload store to store the row
        factory_with_store = RecorderFactory(db, payload_store=payload_store)
        run = factory_with_store.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory_with_store.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # create_row auto-stores payload via configured payload_store
        test_data = {"field": "value"}
        row = factory_with_store.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=test_data,
        )

        # Create new factory WITHOUT payload store, using same db
        factory_no_store = RecorderFactory(db, payload_store=None)

        result = factory_no_store.query.get_row_data(row.row_id)

        assert result.state == RowDataState.STORE_NOT_CONFIGURED
        assert result.data is None

    def test_purged(self, tmp_path: Path, payload_store) -> None:
        """Returns PURGED when payload has been deleted (PayloadNotFoundError)."""
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # create_row auto-stores payload via configured payload_store
        test_data = {"field": "value"}
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=test_data,
        )

        # Delete the payload (simulating retention policy purge)
        payload_store.delete(row.source_data_ref)

        result = factory.query.get_row_data(row.row_id)

        assert result.state == RowDataState.PURGED
        assert result.data is None

    def test_available(self, tmp_path: Path, payload_store) -> None:
        """Returns AVAILABLE with data when payload exists."""
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # create_row auto-stores payload via configured payload_store
        test_data = {"field": "value", "number": 42}
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=test_data,
        )

        result = factory.query.get_row_data(row.row_id)

        assert result.state == RowDataState.AVAILABLE
        assert result.data == test_data


class TestGetRowDataTier1Corruption:
    """Tier 1 corruption tests: get_row_data must propagate integrity failures.

    Per the Three-Tier Trust Model, corrupted audit data (Tier 1) must crash
    immediately - no silent recovery, no coercion, no defaults.
    """

    def test_integrity_error_propagates(self, tmp_path: Path, payload_store) -> None:
        """IntegrityError from corrupted payload must propagate as AuditIntegrityError.

        When payload bytes have been tampered with (hash mismatch),
        FilesystemPayloadStore.retrieve raises IntegrityError.
        QueryRepository wraps this as AuditIntegrityError with row context —
        Tier 1 data corruption is a crash-worthy event.
        """
        from elspeth.contracts.errors import AuditIntegrityError

        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # create_row auto-stores payload via configured payload_store
        test_data = {"field": "value"}
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=test_data,
        )

        # Corrupt the payload file by tampering with its contents
        # FilesystemPayloadStore uses hash[:2]/hash as path structure
        payload_ref = row.source_data_ref
        assert payload_ref is not None
        payload_path = tmp_path / "payloads" / payload_ref[:2] / payload_ref
        payload_path.write_bytes(b"corrupted data that won't match hash")

        # get_row_data must raise AuditIntegrityError with row context
        import pytest

        with pytest.raises(AuditIntegrityError, match="Payload integrity check failed"):
            factory.query.get_row_data(row.row_id)

    def test_invalid_json_propagates(self, tmp_path: Path, payload_store) -> None:
        """Invalid JSON in payload must propagate as AuditIntegrityError.

        When payload bytes are valid (hash matches) but contain non-JSON,
        QueryRepository wraps the JSONDecodeError as AuditIntegrityError
        with row context — Tier 1 corruption is a crash-worthy event.
        """
        import pytest

        from elspeth.contracts.errors import AuditIntegrityError

        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # create_row auto-stores valid canonical JSON via payload_store
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"placeholder": "ignored"},
        )

        # Store non-JSON bytes separately (hash-valid but not JSON)
        bad_ref = payload_store.store(b"this is not valid JSON {")

        # Point the row's source_data_ref to the bad payload
        from elspeth.core.landscape.schema import rows_table

        with db.engine.connect() as conn:
            conn.execute(rows_table.update().where(rows_table.c.row_id == row.row_id).values(source_data_ref=bad_ref))
            conn.commit()

        # get_row_data must raise AuditIntegrityError with row context
        with pytest.raises(AuditIntegrityError, match="Corrupt payload"):
            factory.query.get_row_data(row.row_id)

    def test_non_object_json_raises_audit_integrity_error(self, tmp_path: Path, payload_store) -> None:
        """JSON payloads must decode to objects for AVAILABLE row data."""
        import pytest

        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # create_row auto-stores valid canonical JSON via payload_store
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"placeholder": "ignored"},
        )

        # Store non-object JSON separately (hash-valid but not a JSON object)
        bad_ref = payload_store.store(json.dumps([1, 2, 3]).encode())

        # Point the row's source_data_ref to the bad payload
        from elspeth.core.landscape.schema import rows_table

        with db.engine.connect() as conn:
            conn.execute(rows_table.update().where(rows_table.c.row_id == row.row_id).values(source_data_ref=bad_ref))
            conn.commit()

        with pytest.raises(AuditIntegrityError, match="expected JSON object"):
            factory.query.get_row_data(row.row_id)
