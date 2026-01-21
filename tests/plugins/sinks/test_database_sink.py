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

        sink = DatabaseSink({"url": "sqlite:///:memory:", "table": "test", "schema": DYNAMIC_SCHEMA})
        assert isinstance(sink, SinkProtocol)

    def test_has_required_attributes(self) -> None:
        """DatabaseSink has name and input_schema."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        assert DatabaseSink.name == "database"
        # input_schema is now set per-instance based on config
        sink = DatabaseSink({"url": "sqlite:///:memory:", "table": "test", "schema": DYNAMIC_SCHEMA})
        assert hasattr(sink, "input_schema")

    def test_write_creates_table(self, db_url: str, ctx: PluginContext) -> None:
        """write() creates table and inserts rows."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output", "schema": DYNAMIC_SCHEMA})

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

        sink = DatabaseSink({"url": db_url, "table": "output", "schema": DYNAMIC_SCHEMA})

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

        sink = DatabaseSink({"url": db_url, "table": "output", "schema": DYNAMIC_SCHEMA})

        sink.write([{"id": 1}], ctx)
        sink.close()
        sink.close()  # Should not raise

    def test_memory_database(self, ctx: PluginContext) -> None:
        """Works with in-memory SQLite."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": "sqlite:///:memory:", "table": "test", "schema": DYNAMIC_SCHEMA})

        sink.write([{"col": "value"}], ctx)
        # Can't verify in-memory after close, but should not raise
        sink.close()

    def test_batch_write_returns_artifact_descriptor(self, db_url: str, ctx: PluginContext) -> None:
        """write() returns ArtifactDescriptor with content hash."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output", "schema": DYNAMIC_SCHEMA})

        artifact = sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.artifact_type == "database"
        assert artifact.content_hash  # Non-empty
        assert artifact.size_bytes > 0

    def test_batch_write_content_hash_is_payload_hash(self, db_url: str, ctx: PluginContext) -> None:
        """content_hash is SHA-256 of canonical JSON payload BEFORE insert."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        rows = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        sink = DatabaseSink({"url": db_url, "table": "output", "schema": DYNAMIC_SCHEMA})

        artifact = sink.write(rows, ctx)
        sink.close()

        # Hash should be of the canonical JSON payload
        # Note: We use sorted keys for canonical form
        payload_json = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        expected_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

        assert artifact.content_hash == expected_hash

    def test_batch_write_metadata_has_row_count(self, db_url: str, ctx: PluginContext) -> None:
        """ArtifactDescriptor metadata includes row_count."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output", "schema": DYNAMIC_SCHEMA})

        artifact = sink.write([{"id": 1}, {"id": 2}, {"id": 3}], ctx)
        sink.close()

        assert artifact.metadata is not None
        assert artifact.metadata["row_count"] == 3

    def test_batch_write_empty_list(self, db_url: str, ctx: PluginContext) -> None:
        """Batch write with empty list returns descriptor with zero size."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output", "schema": DYNAMIC_SCHEMA})

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

        sink = DatabaseSink({"url": "sqlite:///:memory:", "table": "test", "schema": DYNAMIC_SCHEMA})
        assert sink.plugin_version == "1.0.0"

    def test_has_determinism(self) -> None:
        """DatabaseSink has determinism attribute."""
        from elspeth.contracts import Determinism
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": "sqlite:///:memory:", "table": "test", "schema": DYNAMIC_SCHEMA})
        assert sink.determinism == Determinism.IO_WRITE

    def test_explicit_schema_creates_all_columns_including_optional(self, db_url: str, ctx: PluginContext) -> None:
        """Table should include all schema fields, not just first row keys.

        Bug: P1-2026-01-19-databasesink-schema-inferred-from-first-row
        When schema is explicit, table columns should come from schema config,
        not from first row keys. This ensures optional fields are present.
        """
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        # Explicit schema with optional field 'score'
        explicit_schema = {
            "mode": "free",
            "fields": ["id: int", "score: float?"],
        }

        sink = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": explicit_schema,
            }
        )

        # First batch WITHOUT optional field
        sink.write([{"id": 1}], ctx)

        # Second batch WITH optional field - should NOT fail
        # Bug: This fails because table was created without 'score' column
        sink.write([{"id": 2, "score": 1.5}], ctx)

        sink.close()

        # Verify both rows written correctly
        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table("output", metadata, autoload_with=engine)

        # Table should have 'score' column even though first row didn't have it
        assert "score" in [c.name for c in table.columns]

        with engine.connect() as conn:
            rows = list(conn.execute(select(table)))

        assert len(rows) == 2

    def test_explicit_schema_maps_types_correctly(self, db_url: str, ctx: PluginContext) -> None:
        """Schema field types should map to appropriate SQLAlchemy types."""
        from sqlalchemy import Boolean, Float, Integer, String

        from elspeth.plugins.sinks.database_sink import DatabaseSink

        # Explicit schema with all supported types
        explicit_schema = {
            "mode": "strict",
            "fields": ["id: int", "name: str", "score: float", "active: bool"],
        }

        sink = DatabaseSink(
            {
                "url": db_url,
                "table": "typed_output",
                "schema": explicit_schema,
            }
        )

        sink.write([{"id": 1, "name": "test", "score": 1.5, "active": True}], ctx)
        sink.close()

        # Verify column types
        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table("typed_output", metadata, autoload_with=engine)

        columns_by_name = {c.name: c for c in table.columns}
        assert isinstance(columns_by_name["id"].type, Integer)
        assert isinstance(columns_by_name["name"].type, (String, type(String())))
        assert isinstance(columns_by_name["score"].type, (Float, type(Float())))
        # SQLite stores booleans as integers, so check for either
        assert isinstance(columns_by_name["active"].type, (Boolean, Integer))

    def test_dynamic_schema_still_infers_from_row(self, db_url: str, ctx: PluginContext) -> None:
        """Dynamic schema should continue to infer columns from first row."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink(
            {
                "url": db_url,
                "table": "dynamic_output",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        # First row defines columns
        sink.write([{"a": 1, "b": 2}], ctx)
        sink.close()

        # Verify columns match first row (not schema)
        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table("dynamic_output", metadata, autoload_with=engine)

        column_names = [c.name for c in table.columns]
        assert sorted(column_names) == ["a", "b"]


class TestDatabaseSinkIfExistsReplace:
    """Regression tests for if_exists='replace' behavior.

    Bug: P2-2026-01-19-databasesink-if-exists-replace-ignored
    The if_exists config option was stored but never used. Replace mode
    should drop the existing table on first write, following pandas
    to_sql semantics.
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    @pytest.fixture
    def db_url(self, tmp_path: Path) -> str:
        """Create a SQLite database URL."""
        return f"sqlite:///{tmp_path / 'test.db'}"

    def _get_row_count(self, db_url: str, table_name: str) -> int:
        """Helper to count rows in a table."""
        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine)
        with engine.connect() as conn:
            rows = list(conn.execute(select(table)))
        engine.dispose()
        return len(rows)

    def test_if_exists_replace_drops_existing_table(self, db_url: str, ctx: PluginContext) -> None:
        """if_exists='replace' drops existing table on first write.

        When a new sink instance with if_exists='replace' writes to an
        existing table, the old data should be dropped first.
        """
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        # First sink with append (default) - creates table with initial data
        sink1 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": DYNAMIC_SCHEMA,
                "if_exists": "append",
            }
        )
        sink1.write([{"id": 1}, {"id": 2}], ctx)
        sink1.close()

        assert self._get_row_count(db_url, "output") == 2

        # Second sink with replace - should drop table and start fresh
        sink2 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": DYNAMIC_SCHEMA,
                "if_exists": "replace",
            }
        )
        sink2.write([{"id": 3}], ctx)
        sink2.close()

        # Only the new row should exist (old rows dropped)
        assert self._get_row_count(db_url, "output") == 1

    def test_if_exists_replace_subsequent_writes_append(self, db_url: str, ctx: PluginContext) -> None:
        """After initial drop, subsequent writes within same instance append.

        The replace behavior (drop table) only happens on first write
        of the sink instance. Additional writes to the same instance
        should append.
        """
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": DYNAMIC_SCHEMA,
                "if_exists": "replace",
            }
        )

        # First write - would drop if table existed
        sink.write([{"id": 1}], ctx)
        # Second write - should append, not drop again
        sink.write([{"id": 2}], ctx)
        sink.close()

        # Both rows should exist (second write appended)
        assert self._get_row_count(db_url, "output") == 2

    def test_if_exists_append_accumulates(self, db_url: str, ctx: PluginContext) -> None:
        """if_exists='append' (default) accumulates data across sink instances."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        # First sink writes initial data
        sink1 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": DYNAMIC_SCHEMA,
                "if_exists": "append",  # Explicit, but also the default
            }
        )
        sink1.write([{"id": 1}], ctx)
        sink1.close()

        # Second sink appends more data
        sink2 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": DYNAMIC_SCHEMA,
                "if_exists": "append",
            }
        )
        sink2.write([{"id": 2}], ctx)
        sink2.close()

        # Both rows should exist
        assert self._get_row_count(db_url, "output") == 2

    def test_if_exists_replace_works_when_table_does_not_exist(self, db_url: str, ctx: PluginContext) -> None:
        """if_exists='replace' works correctly when table doesn't exist yet.

        The first write should succeed even though there's nothing to drop.
        """
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink(
            {
                "url": db_url,
                "table": "new_table",
                "schema": DYNAMIC_SCHEMA,
                "if_exists": "replace",
            }
        )

        # Should not raise - creates table since it doesn't exist
        sink.write([{"id": 1}], ctx)
        sink.close()

        assert self._get_row_count(db_url, "new_table") == 1
