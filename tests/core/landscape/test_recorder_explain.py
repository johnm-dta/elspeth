"""Tests for LandscapeRecorder explain functionality and graceful degradation."""

from __future__ import annotations

from pathlib import Path

from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestExplainGracefulDegradation:
    """Tests for explain_row() when payloads are unavailable."""

    def test_explain_with_missing_row_payload(self, tmp_path: Path, payload_store) -> None:
        """explain_row() succeeds even when row payload is purged."""
        import json

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.payload_store import FilesystemPayloadStore

        # Set up with payload store
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

        # Store row data in payload store
        row_data = {"name": "test", "value": 42}
        payload_ref = payload_store.store(json.dumps(row_data).encode())

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=row_data,
            payload_ref=payload_ref,
        )

        # Purge the payload (simulate retention policy)
        payload_store.delete(payload_ref)

        # explain_row should still work
        lineage = recorder.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.source_data_hash is not None  # Hash preserved
        assert lineage.source_data is None  # Payload unavailable
        assert lineage.payload_available is False

    def test_explain_reports_payload_status(self, tmp_path: Path, payload_store) -> None:
        """explain_row() explicitly reports payload availability."""
        import json

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.payload_store import FilesystemPayloadStore

        # Set up with payload store
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

        # Store row data in payload store
        row_data = {"name": "test"}
        payload_ref = payload_store.store(json.dumps(row_data).encode())

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=row_data,
            payload_ref=payload_ref,
        )

        # Purge the payload
        payload_store.delete(payload_ref)

        # Check payload_available attribute
        lineage = recorder.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.payload_available is False

    def test_explain_with_available_payload(self, tmp_path: Path, payload_store) -> None:
        """explain_row() returns payload when available."""
        import json

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.payload_store import FilesystemPayloadStore

        # Set up with payload store
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

        # Store row data in payload store
        row_data = {"name": "test", "value": 123}
        payload_ref = payload_store.store(json.dumps(row_data).encode())

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=row_data,
            payload_ref=payload_ref,
        )

        # Payload NOT purged
        lineage = recorder.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.source_data is not None  # Payload available
        assert lineage.source_data == row_data
        assert lineage.payload_available is True

    def test_explain_row_not_found(self) -> None:
        """explain_row() returns None when row doesn't exist."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")

        lineage = recorder.explain_row(
            run_id=run.run_id,
            row_id="nonexistent",
        )

        assert lineage is None

    def test_explain_row_without_payload_store(self) -> None:
        """explain_row() works when no payload store is configured."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)  # No payload store

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
        )

        lineage = recorder.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.source_data_hash is not None
        assert lineage.source_data is None  # No payload store
        assert lineage.payload_available is False

    def test_explain_row_with_no_payload_ref(self, tmp_path: Path) -> None:
        """explain_row() handles rows when no payload_store is configured.

        When LandscapeRecorder is created without a payload_store, rows are
        created without payload storage (payload_ref is None).
        """
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        # No payload_store configured - payloads won't be stored
        recorder = LandscapeRecorder(db, payload_store=None)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create row without payload_ref
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},
            # No payload_ref provided
        )

        lineage = recorder.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.source_data_hash is not None
        assert lineage.source_data is None  # No payload_ref
        assert lineage.payload_available is False

    def test_explain_row_with_corrupted_payload(self, tmp_path: Path, payload_store) -> None:
        """explain_row() handles corrupted payload (invalid JSON) gracefully."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.payload_store import FilesystemPayloadStore

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

        # Store corrupted (non-JSON) data directly to payload store
        corrupted_data = b"this is not valid json {{{{"
        payload_ref = payload_store.store(corrupted_data)

        # Create row with the corrupted payload ref
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},  # Valid data for hash
            payload_ref=payload_ref,
        )

        # explain_row should handle JSONDecodeError gracefully
        lineage = recorder.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.source_data_hash is not None  # Hash preserved
        assert lineage.source_data is None  # Corrupted payload not returned
        assert lineage.payload_available is False  # Reports as unavailable

    def test_explain_row_rejects_run_id_mismatch(self, tmp_path: Path, payload_store) -> None:
        """explain_row() returns None when row belongs to different run."""
        import json

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        # Create two runs
        run1 = recorder.begin_run(config={}, canonical_version="v1")
        run2 = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run1.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create row in run1
        row_data = {"name": "test"}
        payload_ref = payload_store.store(json.dumps(row_data).encode())
        row = recorder.create_row(
            run_id=run1.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=row_data,
            payload_ref=payload_ref,
        )

        # Try to explain using run2's ID - should return None
        lineage = recorder.explain_row(
            run_id=run2.run_id,  # Wrong run!
            row_id=row.row_id,
        )

        assert lineage is None

        # Same row with correct run_id should work
        lineage_correct = recorder.explain_row(
            run_id=run1.run_id,
            row_id=row.row_id,
        )

        assert lineage_correct is not None
        assert lineage_correct.row_id == row.row_id
