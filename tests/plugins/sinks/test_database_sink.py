"""Tests for database sink plugin.

NOTE: Protocol compliance tests (test_implements_protocol, test_has_required_attributes)
are in conftest.py as parametrized tests covering all sink plugins.
"""

from pathlib import Path

import pytest
from sqlalchemy import MetaData, Table, create_engine, select

from elspeth.plugins.context import PluginContext

# Strict schema config for tests - DataPluginConfig now requires schema
# DatabaseSink requires fixed-column structure, so we use strict mode
# Tests that need specific fields define their own schema
STRICT_SCHEMA = {"mode": "strict", "fields": ["id: int", "name: str"]}


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

    def test_write_creates_table(self, db_url: str, ctx: PluginContext) -> None:
        """write() creates table and inserts rows."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA})

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

        sink = DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA})

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

        sink = DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA})

        sink.write([{"id": 1}], ctx)
        sink.close()
        sink.close()  # Should not raise

    def test_memory_database(self, ctx: PluginContext) -> None:
        """Works with in-memory SQLite."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": "sqlite:///:memory:", "table": "test", "schema": STRICT_SCHEMA})

        sink.write([{"col": "value"}], ctx)
        # Can't verify in-memory after close, but should not raise
        sink.close()

    def test_batch_write_returns_artifact_descriptor(self, db_url: str, ctx: PluginContext) -> None:
        """write() returns ArtifactDescriptor with content hash."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA})

        artifact = sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.artifact_type == "database"
        assert artifact.content_hash  # Non-empty
        assert artifact.size_bytes > 0

    def test_batch_write_content_hash_is_payload_hash(self, db_url: str, ctx: PluginContext) -> None:
        """content_hash is SHA-256 of canonical JSON payload BEFORE insert."""
        from elspeth.core.canonical import stable_hash
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        rows = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        sink = DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA})

        artifact = sink.write(rows, ctx)
        sink.close()

        # Hash should be of the canonical JSON payload (RFC 8785)
        expected_hash = stable_hash(rows)

        assert artifact.content_hash == expected_hash

    def test_batch_write_metadata_has_row_count(self, db_url: str, ctx: PluginContext) -> None:
        """ArtifactDescriptor metadata includes row_count."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA})

        artifact = sink.write([{"id": 1}, {"id": 2}, {"id": 3}], ctx)
        sink.close()

        assert artifact.metadata is not None
        assert artifact.metadata["row_count"] == 3

    def test_batch_write_empty_list(self, db_url: str, ctx: PluginContext) -> None:
        """Batch write with empty list returns descriptor with canonical size."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.core.canonical import canonical_json, stable_hash
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA})

        artifact = sink.write([], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.size_bytes == len(canonical_json([]).encode("utf-8"))
        # Empty payload hash (canonical JSON of empty list)
        assert artifact.content_hash == stable_hash([])

    def test_has_plugin_version(self) -> None:
        """DatabaseSink has plugin_version attribute."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": "sqlite:///:memory:", "table": "test", "schema": STRICT_SCHEMA})
        assert sink.plugin_version == "1.0.0"

    def test_has_determinism(self) -> None:
        """DatabaseSink has determinism attribute."""
        from elspeth.contracts import Determinism
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": "sqlite:///:memory:", "table": "test", "schema": STRICT_SCHEMA})
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
            "mode": "strict",
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
                "schema": STRICT_SCHEMA,
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
                "schema": STRICT_SCHEMA,
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
                "schema": STRICT_SCHEMA,
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
                "schema": STRICT_SCHEMA,
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
                "schema": STRICT_SCHEMA,
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
                "schema": STRICT_SCHEMA,
                "if_exists": "replace",
            }
        )

        # Should not raise - creates table since it doesn't exist
        sink.write([{"id": 1}], ctx)
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
        monkeypatch.delenv("ELSPETH_KEYVAULT_URL", raising=False)
        monkeypatch.setenv("ELSPETH_ALLOW_RAW_SECRETS", "true")

        # Should not raise - dev mode allows sanitization without fingerprint
        sink = DatabaseSink(
            {
                "url": "postgresql://user:secret@localhost/db",
                "table": "test",
                "schema": STRICT_SCHEMA,
            }
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
        monkeypatch.delenv("ELSPETH_KEYVAULT_URL", raising=False)
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
        monkeypatch.delenv("ELSPETH_KEYVAULT_URL", raising=False)

        # Should work - no password, no key needed
        sink = DatabaseSink(
            {
                "url": "postgresql://user@localhost/db",
                "table": "test",
                "schema": STRICT_SCHEMA,
            }
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
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

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
        sink = DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA})

        artifact = sink.write(rows, ctx)
        sink.close()

        # Hash MUST match stable_hash (canonical JSON)
        expected_hash = stable_hash(rows)
        assert artifact.content_hash == expected_hash, (
            f"DatabaseSink must use canonical JSON hashing. Got {artifact.content_hash}, expected {expected_hash}"
        )

    def test_content_hash_rejects_nan(self, db_url: str, ctx: PluginContext) -> None:
        """content_hash computation must reject NaN values.

        NaN is not valid JSON per RFC 8785. The current buggy implementation
        silently produces invalid JSON like [{"val":NaN}].
        """
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        rows = [{"value": float("nan")}]
        sink = DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA})

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
        sink = DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA})

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
        sink = DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA})

        # Should not raise - canonical_json normalizes numpy types
        artifact = sink.write(rows, ctx)
        sink.close()

        # Hash must match stable_hash
        expected_hash = stable_hash(rows)
        assert artifact.content_hash == expected_hash

    def test_payload_size_uses_canonical_bytes(self, db_url: str, ctx: PluginContext) -> None:
        """payload_size must be byte length of canonical JSON, not json.dumps.

        Unicode characters have different byte lengths depending on escaping:
        - json.dumps: "😀" -> "\\ud83d\\ude00" (12 bytes)
        - canonical:  "😀" -> "😀" (4 bytes UTF-8)
        """
        from elspeth.core.canonical import canonical_json
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        rows = [{"emoji": "😀"}]
        sink = DatabaseSink({"url": db_url, "table": "output", "schema": STRICT_SCHEMA})

        artifact = sink.write(rows, ctx)
        sink.close()

        # Size must match canonical JSON byte length
        expected_size = len(canonical_json(rows).encode("utf-8"))
        assert artifact.size_bytes == expected_size, (
            f"payload_size must use canonical JSON bytes. Got {artifact.size_bytes}, expected {expected_size}"
        )


class TestDatabaseSinkSchemaValidation:
    """Tests for DatabaseSink schema compatibility validation.

    DatabaseSink requires fixed-column structure. Schemas that allow extra fields
    (free mode, dynamic mode) are incompatible because:
    - Table columns are fixed at table creation
    - Extra fields would either be silently dropped (audit violation) or cause errors
    """

    @pytest.fixture
    def db_url(self, tmp_path: Path) -> str:
        """Create a SQLite database URL."""
        return f"sqlite:///{tmp_path / 'test.db'}"

    def test_rejects_free_mode_schema(self, db_url: str) -> None:
        """DatabaseSink should reject free mode schemas at initialization.

        Free mode allows extra fields, but database requires fixed columns.
        This would cause silent data loss or runtime errors.
        """
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        free_schema = {"mode": "free", "fields": ["id: int"]}

        with pytest.raises(ValueError, match="allows_extra_fields"):
            DatabaseSink({"url": db_url, "table": "output", "schema": free_schema})

    def test_rejects_dynamic_schema(self, db_url: str) -> None:
        """DatabaseSink should reject dynamic schemas at initialization.

        Dynamic schemas allow any fields, but database requires fixed columns.
        This would cause silent data loss or runtime errors.
        """
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        dynamic_schema = {"fields": "dynamic"}

        with pytest.raises(ValueError, match="allows_extra_fields"):
            DatabaseSink({"url": db_url, "table": "output", "schema": dynamic_schema})

    def test_accepts_strict_mode_schema(self, db_url: str) -> None:
        """DatabaseSink should accept strict mode schemas.

        Strict mode has fixed fields - compatible with database structure.
        """
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        strict_schema = {"mode": "strict", "fields": ["id: int", "name: str"]}

        # Should not raise
        sink = DatabaseSink({"url": db_url, "table": "output", "schema": strict_schema})
        assert sink is not None
