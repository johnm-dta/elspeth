"""Tests for database sink plugin."""

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import MetaData, Table, create_engine, select

from elspeth.contracts import Determinism
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.sink_effects import ResolvedSinkEffectMode, SinkEffectExecutionPurpose, SinkEffectInputKind
from elspeth.engine.orchestrator.preflight import SinkEffectCapabilityError, validate_sink_effect_capability
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.sinks.database_sink import DatabaseSink, DatabaseSinkConfig
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_operation_context

# Strict schema config for tests - DataPluginConfig now requires schema
# DatabaseSink requires fixed-column structure, so we use strict mode
# Tests that need specific fields define their own schema
STRICT_SCHEMA = {"mode": "fixed", "fields": ["id: int", "name: str"]}
EFFECT_LEDGER = {
    "table": "_elspeth_sink_effects",
    "schema_version": 1,
    "permissions": ["select", "insert"],
}


class TestDatabaseSinkEffectCapability:
    def test_raw_mode_resolution_rejects_missing_ledger_without_construction(self, tmp_path: Path) -> None:
        with pytest.raises(SinkEffectCapabilityError, match="target-side effect ledger"):
            DatabaseSink._resolve_sink_effect_mode(
                {"url": f"sqlite:///{tmp_path / 'target.db'}", "table": "output", "schema": STRICT_SCHEMA},
                purpose=SinkEffectExecutionPurpose.FRESH,
            )

    def test_raw_mode_resolution_accepts_declared_append_contract(self, tmp_path: Path) -> None:
        mode = DatabaseSink._resolve_sink_effect_mode(
            {
                "url": f"sqlite:///{tmp_path / 'target.db'}",
                "table": "output",
                "schema": STRICT_SCHEMA,
                "effect_ledger": EFFECT_LEDGER,
            },
            purpose=SinkEffectExecutionPurpose.FRESH,
        )

        assert mode == ResolvedSinkEffectMode("append")

    def test_effect_capability_requires_operator_declared_target_ledger(self, tmp_path: Path) -> None:
        target_path = tmp_path / "target.db"
        sink = DatabaseSink({"url": f"sqlite:///{target_path}", "table": "output", "schema": STRICT_SCHEMA})

        with pytest.raises(SinkEffectCapabilityError, match="target-side effect ledger"):
            validate_sink_effect_capability(sink, "append", SinkEffectInputKind.PIPELINE_MEMBERS)
        assert not target_path.exists()

    def test_effect_capability_rejects_replace_before_target_io(self, tmp_path: Path) -> None:
        target_path = tmp_path / "target.db"
        sink = DatabaseSink(
            {
                "url": f"sqlite:///{target_path}",
                "table": "output",
                "schema": STRICT_SCHEMA,
                "if_exists": "replace",
                "effect_ledger": EFFECT_LEDGER,
            }
        )

        with pytest.raises(SinkEffectCapabilityError, match=r"replace.*not supported"):
            validate_sink_effect_capability(sink, "replace", SinkEffectInputKind.PIPELINE_MEMBERS)
        assert not target_path.exists()

    def test_effect_capability_rejects_unsupported_dialect_before_target_io(self) -> None:
        sink = DatabaseSink(
            {
                "url": "mysql+pymysql://localhost/example",
                "table": "output",
                "schema": STRICT_SCHEMA,
                "effect_ledger": EFFECT_LEDGER,
            }
        )

        with pytest.raises(SinkEffectCapabilityError, match=r"dialect.*mysql"):
            validate_sink_effect_capability(sink, "append", SinkEffectInputKind.PIPELINE_MEMBERS)

    def test_config_schema_advertises_target_ledger_contract(self) -> None:
        schema = DatabaseSink.get_config_schema()

        ledger = schema["$defs"]["DatabaseEffectLedgerConfig"]
        assert schema["properties"]["effect_ledger"]["anyOf"][0]["$ref"].endswith("DatabaseEffectLedgerConfig")
        assert ledger["required"] == ["table", "permissions"]
        assert ledger["properties"]["schema_version"]["const"] == 1


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

    @pytest.mark.parametrize("table_name", ["<OPERATOR_REQUIRED>", "operator required", "operator_required"])
    def test_config_rejects_placeholder_table_name(self, table_name: str) -> None:
        with pytest.raises(PluginConfigError, match="placeholder"):
            DatabaseSinkConfig.from_dict(
                {
                    "url": "sqlite:///:memory:",
                    "table": table_name,
                    "schema": STRICT_SCHEMA,
                },
                plugin_name="database",
            )

    @pytest.mark.parametrize("table_name", ["todo", "unknown", "unset", "required"])
    def test_config_accepts_plain_placeholder_words_as_table_names(self, table_name: str) -> None:
        cfg = DatabaseSinkConfig.from_dict(
            {
                "url": "sqlite:///:memory:",
                "table": table_name,
                "schema": STRICT_SCHEMA,
            },
            plugin_name="database",
        )
        assert cfg.table == table_name

    def test_has_plugin_version(self) -> None:
        """DatabaseSink has plugin_version attribute."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = inject_write_failure(DatabaseSink({"url": "sqlite:///:memory:", "table": "test", "schema": STRICT_SCHEMA}))
        assert sink.plugin_version == "1.0.0"

    def test_has_determinism(self) -> None:
        """DatabaseSink has determinism attribute."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = inject_write_failure(DatabaseSink({"url": "sqlite:///:memory:", "table": "test", "schema": STRICT_SCHEMA}))
        assert sink.determinism == Determinism.IO_WRITE


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
                    "url": "postgresql://user:secret@localhost/db",  # secret-scan: allow-this-line
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
                    "url": "postgresql://user:secret@localhost/db",  # secret-scan: allow-this-line
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


class TestPerRowConstraintDiversion:
    """Per-row diversion for per-row-attributable constraint failures.

    A bulk INSERT cannot say WHICH row violated a UNIQUE/NOT-NULL/CHECK/FK
    constraint. The sink attempts the batch inside a SAVEPOINT; on
    IntegrityError it rolls the savepoint back and re-executes the batch
    row-by-row (each in its own SAVEPOINT) so the offending row(s) can be
    diverted via _divert_row while the good rows commit. The whole sequence
    runs inside one outer engine.begin() transaction: nothing is durable
    until the outer commit, so a connection/operational failure mid-fallback
    rolls back EVERYTHING and is re-raised — never leaving committed-but-
    unrecorded rows.

    Connection/operational/programming errors are NOT per-row attributable
    and still raise (batch integrity).
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_operation_context(
            node_id="sink-0",
            plugin_name="database",
            node_type="SINK",
            operation_type="sink_write",
        )

    def _read_rows(self, db_url: str, table: str) -> list[dict[str, Any]]:
        engine = create_engine(db_url)
        metadata = MetaData()
        tbl = Table(table, metadata, autoload_with=engine)
        with engine.connect() as conn:
            result = [dict(r._mapping) for r in conn.execute(select(tbl))]
        engine.dispose()
        return result
