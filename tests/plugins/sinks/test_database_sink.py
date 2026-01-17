"""Tests for database sink plugin."""

import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import MetaData, Table, create_engine, select

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SinkProtocol

# Dynamic schema config for tests - DataPluginConfig now requires schema
DYNAMIC_SCHEMA = {"fields": "dynamic"}


class TestDatabaseSink:
    """Tests for DatabaseSink plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    @pytest.fixture
    def db_url(self, tmp_path: Path) -> str:
        """Create a SQLite database URL."""
        return f"sqlite:///{tmp_path / 'test.db'}"

    def test_implements_protocol(self) -> None:
        """DatabaseSink implements SinkProtocol."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink(
            {"url": "sqlite:///:memory:", "table": "test", "schema": DYNAMIC_SCHEMA}
        )
        assert isinstance(sink, SinkProtocol)

    def test_has_required_attributes(self) -> None:
        """DatabaseSink has name and input_schema."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        assert DatabaseSink.name == "database"
        # input_schema is now set per-instance based on config
        sink = DatabaseSink(
            {"url": "sqlite:///:memory:", "table": "test", "schema": DYNAMIC_SCHEMA}
        )
        assert hasattr(sink, "input_schema")

    def test_write_creates_table(self, db_url: str, ctx: PluginContext) -> None:
        """write() creates table and inserts rows."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink(
            {"url": db_url, "table": "output", "schema": DYNAMIC_SCHEMA}
        )

        sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.write([{"id": 2, "name": "bob"}], ctx)
        sink.close()

        # Verify data was written
        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table("output", metadata, autoload_with=engine)

        with engine.connect() as conn:
            rows = list(conn.execute(select(table)))

        assert len(rows) == 2
        # SQLite returns tuples; check by position or use dict access
        assert rows[0][1] == "alice"  # name column

    def test_batch_insert(self, db_url: str, ctx: PluginContext) -> None:
        """Multiple batches can be written to the same table."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink(
            {"url": db_url, "table": "output", "schema": DYNAMIC_SCHEMA}
        )

        # Write rows in multiple batches (batching now handled by caller)
        sink.write([{"id": 0, "value": "val0"}, {"id": 1, "value": "val1"}], ctx)
        sink.write([{"id": 2, "value": "val2"}, {"id": 3, "value": "val3"}], ctx)
        sink.write([{"id": 4, "value": "val4"}], ctx)
        sink.close()

        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table("output", metadata, autoload_with=engine)

        with engine.connect() as conn:
            rows = list(conn.execute(select(table)))

        assert len(rows) == 5

    def test_close_is_idempotent(self, db_url: str, ctx: PluginContext) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink(
            {"url": db_url, "table": "output", "schema": DYNAMIC_SCHEMA}
        )

        sink.write([{"id": 1}], ctx)
        sink.close()
        sink.close()  # Should not raise

    def test_memory_database(self, ctx: PluginContext) -> None:
        """Works with in-memory SQLite."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink(
            {"url": "sqlite:///:memory:", "table": "test", "schema": DYNAMIC_SCHEMA}
        )

        sink.write([{"col": "value"}], ctx)
        # Can't verify in-memory after close, but should not raise
        sink.close()

    def test_batch_write_returns_artifact_descriptor(
        self, db_url: str, ctx: PluginContext
    ) -> None:
        """write() returns ArtifactDescriptor with content hash."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink(
            {"url": db_url, "table": "output", "schema": DYNAMIC_SCHEMA}
        )

        artifact = sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.artifact_type == "database"
        assert artifact.content_hash  # Non-empty
        assert artifact.size_bytes > 0

    def test_batch_write_content_hash_is_payload_hash(
        self, db_url: str, ctx: PluginContext
    ) -> None:
        """content_hash is SHA-256 of canonical JSON payload BEFORE insert."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        rows = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        sink = DatabaseSink(
            {"url": db_url, "table": "output", "schema": DYNAMIC_SCHEMA}
        )

        artifact = sink.write(rows, ctx)
        sink.close()

        # Hash should be of the canonical JSON payload
        # Note: We use sorted keys for canonical form
        payload_json = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        expected_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

        assert artifact.content_hash == expected_hash

    def test_batch_write_metadata_has_row_count(
        self, db_url: str, ctx: PluginContext
    ) -> None:
        """ArtifactDescriptor metadata includes row_count."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink(
            {"url": db_url, "table": "output", "schema": DYNAMIC_SCHEMA}
        )

        artifact = sink.write([{"id": 1}, {"id": 2}, {"id": 3}], ctx)
        sink.close()

        assert artifact.metadata is not None
        assert artifact.metadata["row_count"] == 3

    def test_batch_write_empty_list(self, db_url: str, ctx: PluginContext) -> None:
        """Batch write with empty list returns descriptor with zero size."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink(
            {"url": db_url, "table": "output", "schema": DYNAMIC_SCHEMA}
        )

        artifact = sink.write([], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.size_bytes == 0
        # Empty payload hash
        empty_json = json.dumps([], sort_keys=True, separators=(",", ":"))
        assert artifact.content_hash == hashlib.sha256(empty_json.encode()).hexdigest()

    def test_has_plugin_version(self) -> None:
        """DatabaseSink has plugin_version attribute."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink(
            {"url": "sqlite:///:memory:", "table": "test", "schema": DYNAMIC_SCHEMA}
        )
        assert sink.plugin_version == "1.0.0"

    def test_has_determinism(self) -> None:
        """DatabaseSink has determinism attribute."""
        from elspeth.contracts import Determinism
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink(
            {"url": "sqlite:///:memory:", "table": "test", "schema": DYNAMIC_SCHEMA}
        )
        assert sink.determinism == Determinism.IO_WRITE
