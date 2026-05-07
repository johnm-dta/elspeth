"""Tests for database sink plugin."""

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import MetaData, Table, create_engine, select

from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.sinks.database_sink import DatabaseSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_operation_context

# Strict schema config for tests - DataPluginConfig now requires schema
# DatabaseSink requires fixed-column structure, so we use strict mode
# Tests that need specific fields define their own schema
STRICT_SCHEMA = {"mode": "fixed", "fields": ["id: int", "name: str"]}


class TestDatabaseSink:
    """Tests for DatabaseSink plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a plugin context with real landscape and operation records."""
        return make_operation_context(
            node_id="sink",
            plugin_name="database_sink",
            node_type="SINK",
            operation_type="sink_write",
        )

    @pytest.fixture
    def db_url(self, tmp_path: Path) -> str:
        """Create a SQLite database URL."""
        return f"sqlite:///{tmp_path / 'test.db'}"

    def test_write_creates_table(self, db_url: str, ctx: PluginContext) -> None:
        """write() creates table and inserts rows."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA}))

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

        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA}))

        # Write rows in multiple batches (batching now handled by caller)
        sink.write([{"id": 0, "name": "val0"}, {"id": 1, "name": "val1"}], ctx)
        sink.write([{"id": 2, "name": "val2"}, {"id": 3, "name": "val3"}], ctx)
        sink.write([{"id": 4, "name": "val4"}], ctx)
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

        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA}))

        sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.close()
        sink.close()  # Should not raise

    def test_memory_database(self, ctx: PluginContext) -> None:
        """Works with in-memory SQLite."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = inject_write_failure(DatabaseSink({"url": "sqlite:///:memory:", "table": "test", "schema": STRICT_SCHEMA}))

        sink.write([{"id": 1, "name": "alice"}], ctx)
        # Can't verify in-memory after close, but should not raise
        sink.close()

    def test_batch_write_returns_artifact_descriptor(self, db_url: str, ctx: PluginContext) -> None:
        """write() returns ArtifactDescriptor with content hash."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA}))

        artifact = sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.close()

        assert isinstance(artifact.artifact, ArtifactDescriptor)
        assert artifact.artifact.artifact_type == "database"
        assert artifact.artifact.content_hash  # Non-empty
        assert artifact.artifact.size_bytes > 0

    def test_batch_write_content_hash_is_payload_hash(self, db_url: str, ctx: PluginContext) -> None:
        """content_hash is SHA-256 of canonical JSON payload BEFORE insert."""
        from elspeth.core.canonical import stable_hash
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        rows = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA}))

        artifact = sink.write(rows, ctx)
        sink.close()

        # Hash should be of the canonical JSON payload (RFC 8785)
        expected_hash = stable_hash(rows)

        assert artifact.artifact.content_hash == expected_hash

    def test_batch_write_metadata_has_row_count(self, db_url: str, ctx: PluginContext) -> None:
        """ArtifactDescriptor metadata includes row_count."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA}))

        artifact = sink.write([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": "c"}], ctx)
        sink.close()

        assert artifact.artifact.metadata is not None
        assert artifact.artifact.metadata["row_count"] == 3

    def test_batch_write_empty_list(self, db_url: str, ctx: PluginContext) -> None:
        """Batch write with empty list returns descriptor with canonical size."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.core.canonical import canonical_json, stable_hash
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA}))

        artifact = sink.write([], ctx)
        sink.close()

        assert isinstance(artifact.artifact, ArtifactDescriptor)
        assert artifact.artifact.size_bytes == len(canonical_json([]).encode("utf-8"))
        # Empty payload hash (canonical JSON of empty list)
        assert artifact.artifact.content_hash == stable_hash([])

    def test_has_plugin_version(self) -> None:
        """DatabaseSink has plugin_version attribute."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = inject_write_failure(DatabaseSink({"url": "sqlite:///:memory:", "table": "test", "schema": STRICT_SCHEMA}))
        assert sink.plugin_version == "1.0.0"

    def test_has_determinism(self) -> None:
        """DatabaseSink has determinism attribute."""
        from elspeth.contracts import Determinism
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = inject_write_failure(DatabaseSink({"url": "sqlite:///:memory:", "table": "test", "schema": STRICT_SCHEMA}))
        assert sink.determinism == Determinism.IO_WRITE

    def test_explicit_schema_creates_all_columns_including_optional(self, db_url: str, ctx: PluginContext) -> None:
        """Table should include all schema fields, not just first row keys.

        Bug: P1-2026-01-19-databasesink-schema-inferred-from-first-row
        When schema is explicit, table columns should come from schema config,
        not from first row keys. This ensures optional fields are present.
        """
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        # Strict schema with optional field 'score'
        explicit_schema = {
            "mode": "fixed",
            "fields": ["id: int", "score: float?"],
        }

        sink = inject_write_failure(
            DatabaseSink(
                {
                    "url": db_url,
                    "table": "output",
                    "schema": explicit_schema,
                }
            )
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
            "mode": "fixed",
            "fields": ["id: int", "name: str", "score: float", "active: bool"],
        }

        sink = inject_write_failure(
            DatabaseSink(
                {
                    "url": db_url,
                    "table": "typed_output",
                    "schema": explicit_schema,
                }
            )
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


class TestDatabaseSinkIfExistsReplace:
    """Regression tests for if_exists='replace' behavior.

    Bug: P2-2026-01-19-databasesink-if-exists-replace-ignored
    The if_exists config option was stored but never used. Replace mode
    should drop the existing table on first write, following pandas
    to_sql semantics.
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a plugin context with real landscape and operation records."""
        return make_operation_context(
            node_id="sink",
            plugin_name="database_sink",
            node_type="SINK",
            operation_type="sink_write",
        )

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
        sink1 = inject_write_failure(
            DatabaseSink(
                {
                    "url": db_url,
                    "table": "output",
                    "schema": STRICT_SCHEMA,
                    "if_exists": "append",
                }
            )
        )
        sink1.write([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}], ctx)
        sink1.close()

        assert self._get_row_count(db_url, "output") == 2

        # Second sink with replace - should drop table and start fresh
        sink2 = inject_write_failure(
            DatabaseSink(
                {
                    "url": db_url,
                    "table": "output",
                    "schema": STRICT_SCHEMA,
                    "if_exists": "replace",
                }
            )
        )
        sink2.write([{"id": 3, "name": "c"}], ctx)
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

        sink = inject_write_failure(
            DatabaseSink(
                {
                    "url": db_url,
                    "table": "output",
                    "schema": STRICT_SCHEMA,
                    "if_exists": "replace",
                }
            )
        )

        # First write - would drop if table existed
        sink.write([{"id": 1, "name": "a"}], ctx)
        # Second write - should append, not drop again
        sink.write([{"id": 2, "name": "b"}], ctx)
        sink.close()

        # Both rows should exist (second write appended)
        assert self._get_row_count(db_url, "output") == 2

    def test_if_exists_append_accumulates(self, db_url: str, ctx: PluginContext) -> None:
        """if_exists='append' (default) accumulates data across sink instances."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        # First sink writes initial data
        sink1 = inject_write_failure(
            DatabaseSink(
                {
                    "url": db_url,
                    "table": "output",
                    "schema": STRICT_SCHEMA,
                    "if_exists": "append",  # Explicit, but also the default
                }
            )
        )
        sink1.write([{"id": 1, "name": "a"}], ctx)
        sink1.close()

        # Second sink appends more data
        sink2 = inject_write_failure(
            DatabaseSink(
                {
                    "url": db_url,
                    "table": "output",
                    "schema": STRICT_SCHEMA,
                    "if_exists": "append",
                }
            )
        )
        sink2.write([{"id": 2, "name": "b"}], ctx)
        sink2.close()

        # Both rows should exist
        assert self._get_row_count(db_url, "output") == 2

    def test_if_exists_replace_works_when_table_does_not_exist(self, db_url: str, ctx: PluginContext) -> None:
        """if_exists='replace' works correctly when table doesn't exist yet.

        The first write should succeed even though there's nothing to drop.
        """
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = inject_write_failure(
            DatabaseSink(
                {
                    "url": db_url,
                    "table": "new_table",
                    "schema": STRICT_SCHEMA,
                    "if_exists": "replace",
                }
            )
        )

        # Should not raise - creates table since it doesn't exist
        sink.write([{"id": 1, "name": "a"}], ctx)
        sink.close()

        assert self._get_row_count(db_url, "new_table") == 1


class TestDatabaseSinkSecretHandling:
    """Tests for DatabaseSink secret sanitization behavior.

    These tests verify that DatabaseSink honors the ELSPETH_ALLOW_RAW_SECRETS
    environment variable consistently with other parts of the codebase.
    """

    def test_url_with_password_honors_dev_mode_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DatabaseSink honors ELSPETH_ALLOW_RAW_SECRETS in dev environments.

        When ELSPETH_ALLOW_RAW_SECRETS=true is set but ELSPETH_FINGERPRINT_KEY
        is not set, the sink should initialize successfully by sanitizing the
        URL without requiring a fingerprint.
        """
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        # Simulate dev environment: no fingerprint key, but allow raw secrets
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_ALLOW_RAW_SECRETS", "true")

        # Should not raise - dev mode allows sanitization without fingerprint
        sink = inject_write_failure(
            DatabaseSink(
                {
                    "url": "postgresql://user:secret@localhost/db",
                    "table": "test",
                    "schema": STRICT_SCHEMA,
                }
            )
        )

        # Verify URL was sanitized (password removed)
        assert "secret" not in sink._sanitized_url.sanitized_url
        # No fingerprint in dev mode
        assert sink._sanitized_url.fingerprint is None

    def test_url_with_password_fails_without_key_in_production_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DatabaseSink raises error when password present but no key in production.

        When ELSPETH_ALLOW_RAW_SECRETS is not set (production mode) and no
        fingerprint key is available, initialization should fail.
        """
        from elspeth.core.config import SecretFingerprintError
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

        # Should raise in production mode
        with pytest.raises(SecretFingerprintError, match="ELSPETH_FINGERPRINT_KEY"):
            DatabaseSink(
                {
                    "url": "postgresql://user:secret@localhost/db",
                    "table": "test",
                    "schema": STRICT_SCHEMA,
                }
            )

    def test_url_without_password_works_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DatabaseSink works without key when URL has no password."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)

        # Should work - no password, no key needed
        sink = inject_write_failure(
            DatabaseSink(
                {
                    "url": "postgresql://user@localhost/db",
                    "table": "test",
                    "schema": STRICT_SCHEMA,
                }
            )
        )

        assert sink._sanitized_url.fingerprint is None


class TestDatabaseSinkCanonicalHashing:
    """Tests for canonical JSON hashing in DatabaseSink.

    Bug: P1-2026-01-21-databasesink-noncanonical-hash
    DatabaseSink uses json.dumps instead of canonical_json, causing:
    - Different hashes for unicode (RFC 8785 vs json.dumps escaping)
    - Crashes with numpy/pandas types
    - Invalid JSON output with NaN/Infinity (silently)

    The contract (docs/contracts/plugin-protocol.md:685) requires:
    "SHA-256 of canonical JSON payload BEFORE insert"
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a plugin context with real landscape and operation records."""
        return make_operation_context(
            node_id="sink",
            plugin_name="database_sink",
            node_type="SINK",
            operation_type="sink_write",
        )

    @pytest.fixture
    def db_url(self, tmp_path: Path) -> str:
        """Create a SQLite database URL."""
        return f"sqlite:///{tmp_path / 'test.db'}"

    def test_content_hash_uses_canonical_json(self, db_url: str, ctx: PluginContext) -> None:
        """content_hash must use canonical_json, not json.dumps.

        This test uses unicode that produces different output between
        json.dumps (escapes to \\uXXXX) and canonical_json (literal UTF-8).
        """
        from elspeth.core.canonical import stable_hash
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        # Unicode that json.dumps escapes but RFC 8785 keeps literal
        rows = [{"emoji": "😀", "text": "café"}]
        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": {"mode": "observed"}}))

        artifact = sink.write(rows, ctx)
        sink.close()

        # Hash MUST match stable_hash (canonical JSON)
        expected_hash = stable_hash(rows)
        assert artifact.artifact.content_hash == expected_hash, (
            f"DatabaseSink must use canonical JSON hashing. Got {artifact.artifact.content_hash}, expected {expected_hash}"
        )

    def test_content_hash_rejects_nan(self, db_url: str, ctx: PluginContext) -> None:
        """content_hash computation must reject NaN values.

        NaN is not valid JSON per RFC 8785. The current buggy implementation
        silently produces invalid JSON like [{"val":NaN}].
        """
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        rows = [{"value": float("nan")}]
        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": {"mode": "observed"}}))

        # canonical_json raises ValueError for NaN
        with pytest.raises(ValueError, match="non-finite float"):
            sink.write(rows, ctx)

        sink.close()

    def test_content_hash_rejects_infinity(self, db_url: str, ctx: PluginContext) -> None:
        """content_hash computation must reject Infinity values.

        Infinity is not valid JSON per RFC 8785.
        """
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        rows = [{"value": float("inf")}]
        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": {"mode": "observed"}}))

        # canonical_json raises ValueError for Infinity
        with pytest.raises(ValueError, match="non-finite float"):
            sink.write(rows, ctx)

        sink.close()

    def test_content_hash_handles_numpy_types(self, db_url: str, ctx: PluginContext) -> None:
        """content_hash must handle numpy types without crashing.

        The current buggy implementation raises TypeError for numpy.int64.
        canonical_json normalizes these to Python primitives.
        """
        import numpy as np

        from elspeth.core.canonical import stable_hash
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        rows = [{"id": np.int64(42), "score": np.float64(3.14)}]
        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": {"mode": "observed"}}))

        # Should not raise - canonical_json normalizes numpy types
        artifact = sink.write(rows, ctx)
        sink.close()

        # Hash must match stable_hash
        expected_hash = stable_hash(rows)
        assert artifact.artifact.content_hash == expected_hash

    def test_payload_size_uses_canonical_bytes(self, db_url: str, ctx: PluginContext) -> None:
        """payload_size must be byte length of canonical JSON, not json.dumps.

        Unicode characters have different byte lengths depending on escaping:
        - json.dumps: "😀" -> "\\ud83d\\ude00" (12 bytes)
        - canonical:  "😀" -> "😀" (4 bytes UTF-8)
        """
        from elspeth.core.canonical import canonical_json
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        rows = [{"emoji": "😀"}]
        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": {"mode": "observed"}}))

        artifact = sink.write(rows, ctx)
        sink.close()

        # Size must match canonical JSON byte length
        expected_size = len(canonical_json(rows).encode("utf-8"))
        assert artifact.artifact.size_bytes == expected_size, (
            f"payload_size must use canonical JSON bytes. Got {artifact.artifact.size_bytes}, expected {expected_size}"
        )


class TestDatabaseSinkSchemaValidation:
    """Tests for DatabaseSink schema modes using infer-and-lock pattern.

    DatabaseSink supports all schema modes:
    - strict: columns from config, extras rejected at insert time
    - free: declared columns + extras from first row, then locked
    - dynamic: columns from first row, then locked

    Table schema is created on first write; subsequent rows must match.
    """

    @pytest.fixture
    def db_url(self, tmp_path: Path) -> str:
        """Create a SQLite database URL."""
        return f"sqlite:///{tmp_path / 'test.db'}"

    def test_accepts_strict_mode_schema(self, db_url: str) -> None:
        """DatabaseSink accepts strict mode - columns from config."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        strict_schema = {"mode": "fixed", "fields": ["id: int", "name: str"]}

        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": strict_schema}))
        assert sink is not None

    def test_accepts_free_mode_schema(self, db_url: str) -> None:
        """DatabaseSink accepts free mode - declared + first-row extras, then locked."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        free_schema = {"mode": "flexible", "fields": ["id: int"]}

        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": free_schema}))
        assert sink is not None

    def test_accepts_dynamic_schema(self, db_url: str) -> None:
        """DatabaseSink accepts dynamic mode - columns from first row, then locked."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        dynamic_schema = {"mode": "observed"}

        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": dynamic_schema}))
        assert sink is not None

    def test_flexible_mode_includes_extras_from_first_row(self, db_url: str) -> None:
        """Flexible mode includes declared fields + extras from first row.

        This is the documented behavior: flexible mode should accept at least
        the declared fields, plus any extras present in the first row.
        """
        from sqlalchemy import MetaData, Table, create_engine, inspect, select

        from elspeth.plugins.sinks.database_sink import DatabaseSink

        ctx = make_operation_context(
            node_id="sink",
            plugin_name="database_sink",
            node_type="SINK",
            operation_type="sink_write",
        )
        # Schema declares only 'id', but first row has 'id', 'name', 'extra'
        flexible_schema = {"mode": "flexible", "fields": ["id: int"]}
        sink = inject_write_failure(
            DatabaseSink(
                {
                    "url": db_url,
                    "table": "flexible_test",
                    "schema": flexible_schema,
                    "if_exists": "replace",
                }
            )
        )

        # First write has declared field + two extras
        sink.write([{"id": 1, "name": "alice", "extra": "value"}], ctx)
        sink.close()

        # Verify all columns are in table
        engine = create_engine(db_url)
        inspector = inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("flexible_test")}

        # All three columns should be present
        assert "id" in columns, "Declared field 'id' should be a column"
        assert "name" in columns, "Extra field 'name' from first row should be a column"
        assert "extra" in columns, "Extra field 'extra' from first row should be a column"

        # Verify data was written correctly
        metadata = MetaData()
        table = Table("flexible_test", metadata, autoload_with=engine)
        with engine.connect() as conn:
            result = conn.execute(select(table)).fetchall()

        assert len(result) == 1
        row_dict = dict(result[0]._mapping)
        assert row_dict["id"] == 1
        assert row_dict["name"] == "alice"
        assert row_dict["extra"] == "value"

        engine.dispose()

    def test_flexible_mode_declared_columns_have_proper_types(self, db_url: str) -> None:
        """Flexible mode should use declared types for schema fields, String for extras."""
        from sqlalchemy import create_engine, inspect

        from elspeth.plugins.sinks.database_sink import DatabaseSink

        ctx = make_operation_context(
            node_id="sink",
            plugin_name="database_sink",
            node_type="SINK",
            operation_type="sink_write",
        )
        # Schema declares 'id' as int, 'score' as float
        flexible_schema = {"mode": "flexible", "fields": ["id: int", "score: float"]}
        sink = inject_write_failure(
            DatabaseSink(
                {
                    "url": db_url,
                    "table": "flexible_types_test",
                    "schema": flexible_schema,
                    "if_exists": "replace",
                }
            )
        )

        # First row has declared fields + an extra string field
        sink.write([{"id": 1, "score": 3.14, "extra": "text"}], ctx)
        sink.close()

        # Verify column types
        engine = create_engine(db_url)
        inspector = inspect(engine)
        columns = {col["name"]: col["type"] for col in inspector.get_columns("flexible_types_test")}

        # Declared fields should have proper types
        assert "INTEGER" in str(columns["id"]).upper(), f"'id' should be INTEGER, got {columns['id']}"
        # SQLite uses REAL/FLOAT, others may use FLOAT/DOUBLE
        assert any(t in str(columns["score"]).upper() for t in ["FLOAT", "REAL"]), f"'score' should be FLOAT/REAL, got {columns['score']}"
        # Extra field should be String (VARCHAR/TEXT)
        assert any(t in str(columns["extra"]).upper() for t in ["VARCHAR", "TEXT", "STRING"]), (
            f"'extra' should be VARCHAR/TEXT/STRING, got {columns['extra']}"
        )

        engine.dispose()


class TestDatabaseSinkFalseSuccess:
    """Engine/table None at INSERT time must crash, not fabricate SUCCESS.

    _ensure_table() guarantees both are set. If somehow they're None at
    INSERT time, that's a bug — the old code silently skipped the INSERT
    but still recorded SUCCESS in the audit trail.
    """

    def test_extra_fields_at_write_time_rejected(self, tmp_path: Path) -> None:
        """write() raises ValueError when row has fields not in table schema.

        After table creation, any row with extra fields (not in table columns)
        must be rejected to prevent silent data loss from SQLAlchemy dropping
        unknown keys.
        """
        db_path = tmp_path / "test.db"
        sink = inject_write_failure(
            DatabaseSink(
                {
                    "url": f"sqlite:///{db_path}",
                    "table": "test_table",
                    "schema": STRICT_SCHEMA,
                }
            )
        )
        ctx = make_operation_context(
            node_id="sink-0",
            plugin_name="database",
            node_type="SINK",
            operation_type="sink_write",
        )

        # First write creates the table with columns [id, name]
        sink.write([{"id": 1, "name": "alice"}], ctx)

        # Second write has an extra field not in the table schema
        with pytest.raises(ValueError, match="extra_field"):
            sink.write([{"id": 2, "name": "bob", "extra_field": "bad"}], ctx)

        sink.close()

    def test_insert_failure_records_error_and_reraises(self, tmp_path: Path) -> None:
        """When INSERT fails, ctx.record_call is invoked with ERROR status
        and the original exception is re-raised.
        """
        from unittest.mock import patch

        from elspeth.contracts import CallStatus

        db_path = tmp_path / "test.db"
        sink = inject_write_failure(
            DatabaseSink(
                {
                    "url": f"sqlite:///{db_path}",
                    "table": "test_table",
                    "schema": STRICT_SCHEMA,
                }
            )
        )
        ctx = make_operation_context(
            node_id="sink-0",
            plugin_name="database",
            node_type="SINK",
            operation_type="sink_write",
        )

        # First write to create the table
        sink.write([{"id": 1, "name": "alice"}], ctx)

        # Patch the engine's begin() to raise on the next INSERT
        def failing_begin():
            raise RuntimeError("simulated INSERT failure")

        with patch.object(sink._engine, "begin", side_effect=failing_begin):
            # Track record_call invocations
            original_record_call = ctx.record_call

            recorded_statuses: list[CallStatus] = []

            def tracking_record_call(**kwargs: object) -> None:
                recorded_statuses.append(kwargs.get("status"))  # type: ignore[arg-type]
                original_record_call(**kwargs)  # type: ignore[arg-type]

            with (
                patch.object(ctx, "record_call", side_effect=tracking_record_call),
                pytest.raises(RuntimeError, match="simulated INSERT failure"),
            ):
                sink.write([{"id": 2, "name": "bob"}], ctx)

        # Verify record_call was invoked with ERROR status
        assert CallStatus.ERROR in recorded_statuses

        sink.close()

    def test_serialize_any_typed_fields_observed_mode(self, tmp_path: Path) -> None:
        """In observed-mode schemas, dict values are serialized to JSON strings.

        When schema mode is 'observed', _serialize_any_typed_fields checks ALL
        fields for dict/list values and serializes them to JSON strings for
        storage in SQL TEXT columns.
        """
        import json

        db_path = tmp_path / "test.db"
        sink = inject_write_failure(
            DatabaseSink(
                {
                    "url": f"sqlite:///{db_path}",
                    "table": "test_table",
                    "schema": {"mode": "observed"},
                }
            )
        )
        ctx = make_operation_context(
            node_id="sink-0",
            plugin_name="database",
            node_type="SINK",
            operation_type="sink_write",
        )

        # Write a row with a dict value — should be serialized to JSON string
        original_dict = {"key": "value", "nested": [1, 2, 3]}
        sink.write([{"id": 1, "data": original_dict, "plain": "text"}], ctx)
        sink.close()

        # Verify data was stored as JSON string, not Python dict
        from sqlalchemy import MetaData, Table, create_engine, select

        engine = create_engine(f"sqlite:///{db_path}")
        metadata = MetaData()
        table = Table("test_table", metadata, autoload_with=engine)

        with engine.connect() as conn:
            rows = list(conn.execute(select(table)))

        row_dict = dict(rows[0]._mapping)
        # The dict value should have been serialized to a JSON string
        assert isinstance(row_dict["data"], str)
        assert json.loads(row_dict["data"]) == original_dict
        # Plain string value should be unchanged
        assert row_dict["plain"] == "text"

        engine.dispose()

    def test_none_engine_at_insert_raises_invariant_error(self, tmp_path: Path) -> None:
        """If _ensure_table fails to set engine/table, assertion catches it.

        This tests the defense-in-depth assertion. We mock _ensure_table to
        be a no-op, simulating a code path where the invariant is broken.
        Previously, the code would silently skip INSERT and record SUCCESS.
        """
        from unittest.mock import patch

        db_path = tmp_path / "test.db"
        sink = inject_write_failure(
            DatabaseSink(
                {
                    "url": f"sqlite:///{db_path}",
                    "table": "test_table",
                    "schema": STRICT_SCHEMA,
                }
            )
        )
        ctx = make_operation_context(
            node_id="sink-0",
            plugin_name="database",
            node_type="SINK",
            operation_type="sink_write",
        )

        # Mock _ensure_table to be a no-op, leaving _engine and _table as None
        with patch.object(sink, "_ensure_table"), pytest.raises(AssertionError, match=r"engine.*None.*invariant"):
            sink.write([{"id": 1, "name": "should_fail"}], ctx)


class TestDDLAuditRecording:
    """Tests that DDL operations (CREATE TABLE, DROP TABLE) are recorded in the audit trail.

    The database sink instruments DDL via ctx.record_call with CallType.SQL.
    INSERT-level record_call is tested elsewhere; these tests cover the DDL paths.
    """

    def test_create_table_records_audit_call(self, tmp_path: Path) -> None:
        """First write triggers CREATE TABLE and records it via record_call."""
        from unittest.mock import patch

        from elspeth.contracts import CallStatus, CallType

        db_path = tmp_path / "test.db"
        sink = inject_write_failure(
            DatabaseSink(
                {
                    "url": f"sqlite:///{db_path}",
                    "table": "audit_test",
                    "schema": STRICT_SCHEMA,
                }
            )
        )
        ctx = make_operation_context(
            node_id="sink-0",
            plugin_name="database",
            node_type="SINK",
            operation_type="sink_write",
        )

        original_record_call = ctx.record_call
        recorded_calls: list[dict[str, Any]] = []

        def tracking_record_call(**kwargs: object) -> None:
            recorded_calls.append(dict(kwargs))
            original_record_call(**kwargs)  # type: ignore[arg-type]

        with patch.object(ctx, "record_call", side_effect=tracking_record_call):
            sink.write([{"id": 1, "name": "alice"}], ctx)

        sink.close()

        # Find the CREATE_TABLE call
        ddl_calls = [c for c in recorded_calls if c.get("request_data", {}).get("operation") == "CREATE_TABLE"]
        assert len(ddl_calls) == 1, f"Expected 1 CREATE_TABLE call, got {len(ddl_calls)}"

        create_call = ddl_calls[0]
        assert create_call["call_type"] == CallType.SQL
        assert create_call["status"] == CallStatus.SUCCESS
        assert create_call["request_data"]["table"] == "audit_test"
        assert create_call["response_data"]["table_created"] == "audit_test"
        assert create_call["provider"] == "sqlalchemy"

    def test_replace_mode_records_drop_and_create(self, tmp_path: Path) -> None:
        """With if_exists='replace', both DROP TABLE and CREATE TABLE are audited."""
        from unittest.mock import patch

        from elspeth.contracts import CallStatus, CallType

        db_path = tmp_path / "test.db"

        # Create initial table
        sink1 = inject_write_failure(
            DatabaseSink(
                {
                    "url": f"sqlite:///{db_path}",
                    "table": "replace_test",
                    "schema": STRICT_SCHEMA,
                }
            )
        )
        ctx1 = make_operation_context(
            node_id="sink-0",
            plugin_name="database",
            node_type="SINK",
            operation_type="sink_write",
        )
        sink1.write([{"id": 1, "name": "alice"}], ctx1)
        sink1.close()

        # Now open with replace mode — should DROP then CREATE
        sink2 = inject_write_failure(
            DatabaseSink(
                {
                    "url": f"sqlite:///{db_path}",
                    "table": "replace_test",
                    "if_exists": "replace",
                    "schema": STRICT_SCHEMA,
                }
            )
        )
        ctx2 = make_operation_context(
            node_id="sink-0",
            plugin_name="database",
            node_type="SINK",
            operation_type="sink_write",
        )

        original_record_call = ctx2.record_call
        recorded_calls: list[dict[str, Any]] = []

        def tracking_record_call(**kwargs: object) -> None:
            recorded_calls.append(dict(kwargs))
            original_record_call(**kwargs)  # type: ignore[arg-type]

        with patch.object(ctx2, "record_call", side_effect=tracking_record_call):
            sink2.write([{"id": 2, "name": "bob"}], ctx2)

        sink2.close()

        operations = [c.get("request_data", {}).get("operation") for c in recorded_calls]

        # Should have DROP_TABLE then CREATE_TABLE (then INSERT)
        assert "DROP_TABLE" in operations, f"Expected DROP_TABLE in {operations}"
        assert "CREATE_TABLE" in operations, f"Expected CREATE_TABLE in {operations}"

        # Verify DROP came before CREATE
        drop_idx = operations.index("DROP_TABLE")
        create_idx = operations.index("CREATE_TABLE")
        assert drop_idx < create_idx, "DROP_TABLE should precede CREATE_TABLE"

        # Verify DROP details
        drop_call = recorded_calls[drop_idx]
        assert drop_call["call_type"] == CallType.SQL
        assert drop_call["status"] == CallStatus.SUCCESS
        assert drop_call["request_data"]["table"] == "replace_test"
        assert drop_call["response_data"]["table_dropped"] == "replace_test"
