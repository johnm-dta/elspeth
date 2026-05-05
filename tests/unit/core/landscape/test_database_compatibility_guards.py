"""Regression tests for LandscapeDB schema and journal compatibility guardrails."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, text

import elspeth.core.landscape.database as database_module
from elspeth.core.landscape.database import LandscapeDB, SchemaCompatibilityError
from elspeth.core.landscape.schema import SQLITE_SCHEMA_EPOCH, metadata


def _make_instance(url: str) -> LandscapeDB:
    """Create a LandscapeDB instance without running constructor side effects."""
    instance = LandscapeDB.__new__(LandscapeDB)
    instance.connection_string = url
    instance._passphrase = None
    instance._journal = None
    instance._engine = create_engine(url, echo=False)
    instance._require_existing_schema = False
    return instance


class TestSyncSchemaEpochDirectionalGuard:
    """Coverage for _sync_sqlite_schema_epoch directional guard."""

    def test_sync_rejects_future_epoch(self, tmp_path: Path) -> None:
        """_sync_sqlite_schema_epoch must refuse to downgrade a future epoch.

        Regression: The method used != comparison, so a database from a
        newer ELSPETH version (epoch 2 when code expects 1) would be
        silently overwritten — destroying the newer epoch stamp.
        """
        db_path = tmp_path / "future_epoch.db"
        engine = create_engine(f"sqlite:///{db_path}")
        metadata.create_all(engine)
        with engine.begin() as conn:
            conn.exec_driver_sql(f"PRAGMA user_version = {SQLITE_SCHEMA_EPOCH + 1}")
        engine.dispose()

        instance = _make_instance(f"sqlite:///{db_path}")

        with pytest.raises(SchemaCompatibilityError, match=r"newer.*epoch"):
            instance._sync_sqlite_schema_epoch()

        # Epoch must NOT have been downgraded
        with instance.engine.connect() as conn:
            epoch = conn.exec_driver_sql("PRAGMA user_version").scalar_one()
        assert epoch == SQLITE_SCHEMA_EPOCH + 1
        instance.close()

    def test_sync_upgrades_epoch_zero(self, tmp_path: Path) -> None:
        """_sync_sqlite_schema_epoch upgrades unstamped databases (epoch 0)."""
        db_path = tmp_path / "unstamped.db"
        engine = create_engine(f"sqlite:///{db_path}")
        metadata.create_all(engine)
        engine.dispose()

        instance = _make_instance(f"sqlite:///{db_path}")
        instance._sync_sqlite_schema_epoch()

        with instance.engine.connect() as conn:
            epoch = conn.exec_driver_sql("PRAGMA user_version").scalar_one()
        assert epoch == SQLITE_SCHEMA_EPOCH
        instance.close()

    def test_sync_noops_on_current_epoch(self, tmp_path: Path) -> None:
        """_sync_sqlite_schema_epoch is a no-op when epoch already matches."""
        db_path = tmp_path / "current_epoch.db"
        engine = create_engine(f"sqlite:///{db_path}")
        metadata.create_all(engine)
        with engine.begin() as conn:
            conn.exec_driver_sql(f"PRAGMA user_version = {SQLITE_SCHEMA_EPOCH}")
        engine.dispose()

        instance = _make_instance(f"sqlite:///{db_path}")
        instance._sync_sqlite_schema_epoch()  # Should not raise

        with instance.engine.connect() as conn:
            epoch = conn.exec_driver_sql("PRAGMA user_version").scalar_one()
        assert epoch == SQLITE_SCHEMA_EPOCH
        instance.close()


class TestSchemaCompatibilityGuards:
    """Coverage for fail-fast schema compatibility checks."""

    def test_validate_schema_rejects_incompatible_schema_epoch(self, tmp_path: Path) -> None:
        """Stamped SQLite schema epochs provide an explicit future migration seam."""
        db_path = tmp_path / "wrong_epoch.db"
        engine = create_engine(f"sqlite:///{db_path}")
        metadata.create_all(engine)
        with engine.begin() as conn:
            conn.exec_driver_sql(f"PRAGMA user_version = {SQLITE_SCHEMA_EPOCH + 1}")
        engine.dispose()

        instance = _make_instance(f"sqlite:///{db_path}")

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            instance._validate_schema()

        msg = str(exc_info.value)
        assert "schema epoch is incompatible" in msg
        assert f"Database epoch: {SQLITE_SCHEMA_EPOCH + 1}" in msg
        assert f"Current epoch: {SQLITE_SCHEMA_EPOCH}" in msg
        instance.close()

    def test_validate_schema_rejects_adr018_token_outcomes_shape(self, tmp_path: Path) -> None:
        """ADR-018 token_outcomes stores must fail with ADR-019 remediation text."""
        db_path = tmp_path / "adr018_audit.db"
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA user_version = 6")
            conn.execute(
                text(
                    """
                    CREATE TABLE token_outcomes (
                        outcome_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        token_id TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        is_terminal INTEGER NOT NULL,
                        recorded_at TEXT NOT NULL
                    )
                    """
                )
            )
        engine.dispose()

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            LandscapeDB(f"sqlite:///{db_path}")

        msg = str(exc_info.value)
        assert "Landscape database schema is outdated." in msg
        assert "schema epoch is incompatible" in msg
        assert "token_outcomes.completed" in msg
        assert "token_outcomes.path" in msg
        assert "token_outcomes.is_terminal" in msg
        assert "docs/operator/migrations/adr-019.md" in msg

    def test_validate_schema_rejects_token_outcomes_outcome_not_nullable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ADR-019 requires nullable outcome for non-terminal BUFFERED rows."""
        db_path = tmp_path / "outcome_not_nullable.db"
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as conn:
            conn.exec_driver_sql(f"PRAGMA user_version = {SQLITE_SCHEMA_EPOCH}")
            conn.execute(
                text(
                    """
                    CREATE TABLE token_outcomes (
                        outcome_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        token_id TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        path TEXT NOT NULL,
                        completed INTEGER NOT NULL,
                        recorded_at TEXT NOT NULL
                    )
                    """
                )
            )
        engine.dispose()

        instance = _make_instance(f"sqlite:///{db_path}")
        monkeypatch.setattr(database_module, "metadata", SimpleNamespace(tables={"token_outcomes": object()}))
        monkeypatch.setattr(database_module, "_REQUIRED_COLUMNS", (("token_outcomes", "completed"), ("token_outcomes", "path")))
        monkeypatch.setattr(database_module, "_REQUIRED_FOREIGN_KEYS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_COMPOSITE_FOREIGN_KEYS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_CHECK_CONSTRAINTS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_INDEXES", ())

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            instance._validate_schema()

        msg = str(exc_info.value)
        assert "token_outcomes.outcome nullable" in msg
        assert "docs/operator/migrations/adr-019.md" in msg
        instance.close()

    def test_validate_schema_rejects_partial_manual_schema_with_stale_is_terminal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Adding completed/path manually is not enough if stale is_terminal remains."""
        db_path = tmp_path / "stale_is_terminal.db"
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as conn:
            conn.exec_driver_sql(f"PRAGMA user_version = {SQLITE_SCHEMA_EPOCH}")
            conn.execute(
                text(
                    """
                    CREATE TABLE token_outcomes (
                        outcome_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        token_id TEXT NOT NULL,
                        outcome TEXT,
                        path TEXT NOT NULL,
                        completed INTEGER NOT NULL,
                        is_terminal INTEGER NOT NULL,
                        recorded_at TEXT NOT NULL
                    )
                    """
                )
            )
        engine.dispose()

        instance = _make_instance(f"sqlite:///{db_path}")
        monkeypatch.setattr(database_module, "metadata", SimpleNamespace(tables={"token_outcomes": object()}))
        monkeypatch.setattr(database_module, "_REQUIRED_COLUMNS", (("token_outcomes", "completed"), ("token_outcomes", "path")))
        monkeypatch.setattr(database_module, "_REQUIRED_FOREIGN_KEYS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_COMPOSITE_FOREIGN_KEYS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_CHECK_CONSTRAINTS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_INDEXES", ())

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            instance._validate_schema()

        msg = str(exc_info.value)
        assert "token_outcomes.is_terminal" in msg
        assert "docs/operator/migrations/adr-019.md" in msg
        instance.close()

    def test_validate_schema_rejects_stale_terminal_index_predicate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SQLite partial unique indexes must predicate on completed, not is_terminal."""
        db_path = tmp_path / "stale_index_predicate.db"
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as conn:
            conn.exec_driver_sql(f"PRAGMA user_version = {SQLITE_SCHEMA_EPOCH}")
            conn.execute(
                text(
                    """
                    CREATE TABLE token_outcomes (
                        outcome_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        token_id TEXT NOT NULL,
                        outcome TEXT,
                        path TEXT NOT NULL,
                        completed INTEGER NOT NULL,
                        is_terminal INTEGER NOT NULL,
                        recorded_at TEXT NOT NULL
                    )
                    """
                )
            )
            conn.execute(text("CREATE UNIQUE INDEX ix_token_outcomes_terminal_unique ON token_outcomes(token_id) WHERE is_terminal = 1"))
        engine.dispose()

        instance = _make_instance(f"sqlite:///{db_path}")
        monkeypatch.setattr(database_module, "metadata", SimpleNamespace(tables={"token_outcomes": object()}))
        monkeypatch.setattr(database_module, "_REQUIRED_COLUMNS", (("token_outcomes", "completed"), ("token_outcomes", "path")))
        monkeypatch.setattr(database_module, "_REQUIRED_FOREIGN_KEYS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_COMPOSITE_FOREIGN_KEYS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_CHECK_CONSTRAINTS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_INDEXES", ())

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            instance._validate_schema()

        msg = str(exc_info.value)
        assert "stale terminal index predicate" in msg
        assert "is_terminal" in msg
        assert "docs/operator/migrations/adr-019.md" in msg
        instance.close()

    def test_validate_schema_rejects_adr018_postgres_shape(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-SQLite existing Landscape schemas get the same ADR-019 guard."""
        import sqlalchemy

        instance = _make_instance(f"sqlite:///{tmp_path / 'postgres_shape.db'}")
        instance.connection_string = "postgresql://user@host/db"

        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["runs", "token_outcomes"]
        mock_inspector.get_columns.side_effect = lambda table_name: {
            "runs": [{"name": "run_id"}],
            "token_outcomes": [
                {"name": "outcome_id", "nullable": False},
                {"name": "run_id", "nullable": False},
                {"name": "token_id", "nullable": False},
                {"name": "outcome", "nullable": False},
                {"name": "is_terminal", "nullable": False},
                {"name": "recorded_at", "nullable": False},
            ],
        }[table_name]
        mock_inspector.get_foreign_keys.return_value = []
        mock_inspector.get_check_constraints.return_value = []
        mock_inspector.get_indexes.return_value = []

        monkeypatch.setattr(sqlalchemy, "inspect", lambda engine: mock_inspector)
        monkeypatch.setattr(
            database_module,
            "metadata",
            SimpleNamespace(tables={"runs": object(), "token_outcomes": object()}),
        )
        monkeypatch.setattr(database_module, "_REQUIRED_COLUMNS", (("token_outcomes", "completed"), ("token_outcomes", "path")))
        monkeypatch.setattr(database_module, "_REQUIRED_FOREIGN_KEYS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_COMPOSITE_FOREIGN_KEYS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_CHECK_CONSTRAINTS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_INDEXES", ())

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            instance._validate_schema()

        msg = str(exc_info.value)
        assert "token_outcomes.completed" in msg
        assert "token_outcomes.path" in msg
        assert "token_outcomes.is_terminal" in msg
        assert "token_outcomes.outcome nullable" in msg
        assert "docs/operator/migrations/adr-019.md" in msg
        instance.close()

    def test_validate_schema_reports_missing_tables_with_actionable_guidance(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Existing partial Landscape DBs must fail with clear remediation text."""
        db_path = tmp_path / "partial_landscape.db"
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as conn:
            # "runs" is a known Landscape table, so this DB is recognized as partial.
            conn.execute(text("CREATE TABLE runs (run_id TEXT PRIMARY KEY)"))
        engine.dispose()

        instance = _make_instance(f"sqlite:///{db_path}")
        monkeypatch.setattr(database_module, "_REQUIRED_COLUMNS", [])
        monkeypatch.setattr(database_module, "_REQUIRED_FOREIGN_KEYS", [])

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            instance._validate_schema()

        msg = str(exc_info.value)
        assert "Landscape database schema is outdated." in msg
        assert "Missing tables:" in msg
        assert "To fix this, either:" in msg
        assert "Delete the database file and let ELSPETH recreate it" in msg
        assert "elspeth landscape migrate" in msg
        assert f"Database: sqlite:///{db_path}" in msg
        instance.close()

    def test_validate_schema_reports_missing_required_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing required columns must be listed deterministically in the error."""
        db_path = tmp_path / "missing_columns.db"
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE tokens (token_id TEXT PRIMARY KEY)"))
        engine.dispose()

        instance = _make_instance(f"sqlite:///{db_path}")
        monkeypatch.setattr(database_module, "metadata", SimpleNamespace(tables={"tokens": object()}))
        monkeypatch.setattr(database_module, "_REQUIRED_COLUMNS", [("tokens", "expand_group_id")])
        monkeypatch.setattr(database_module, "_REQUIRED_FOREIGN_KEYS", [])

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            instance._validate_schema()

        msg = str(exc_info.value)
        assert "Missing columns: tokens.expand_group_id" in msg
        assert "To fix this, either:" in msg
        instance.close()

    def test_validate_schema_reports_missing_required_foreign_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing required foreign keys must fail fast with table/column context."""
        db_path = tmp_path / "missing_fk.db"
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE transform_errors (token_id TEXT, transform_id TEXT)"))
        engine.dispose()

        instance = _make_instance(f"sqlite:///{db_path}")
        monkeypatch.setattr(database_module, "metadata", SimpleNamespace(tables={"transform_errors": object()}))
        monkeypatch.setattr(database_module, "_REQUIRED_COLUMNS", [])
        monkeypatch.setattr(database_module, "_REQUIRED_FOREIGN_KEYS", [("transform_errors", "token_id", "tokens")])

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            instance._validate_schema()

        msg = str(exc_info.value)
        assert "Missing foreign keys:" in msg
        assert "transform_errors.token_id" in msg
        assert "tokens" in msg
        assert "To fix this, either:" in msg
        instance.close()

    def test_validate_schema_rejects_stale_single_column_foreign_keys_for_run_scoped_error_tables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run-scoped error tables must require exact composite FK shapes."""
        import sqlalchemy

        instance = _make_instance(f"sqlite:///{tmp_path / 'stale_fk_shapes.db'}")

        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["transform_errors", "tokens", "nodes"]
        mock_inspector.get_columns.side_effect = lambda table_name: [
            {"name": column_name}
            for column_name in {
                "transform_errors": ("run_id", "token_id", "transform_id"),
                "tokens": ("token_id", "run_id"),
                "nodes": ("node_id", "run_id"),
            }[table_name]
        ]
        mock_inspector.get_foreign_keys.return_value = [
            {
                "constrained_columns": ["token_id"],
                "referred_table": "tokens",
                "referred_columns": ["token_id"],
            },
            {
                "constrained_columns": ["transform_id"],
                "referred_table": "nodes",
                "referred_columns": ["node_id"],
            },
        ]

        monkeypatch.setattr(sqlalchemy, "inspect", lambda engine: mock_inspector)
        monkeypatch.setattr(
            database_module,
            "metadata",
            SimpleNamespace(
                tables={
                    "transform_errors": object(),
                    "tokens": object(),
                    "nodes": object(),
                }
            ),
        )
        monkeypatch.setattr(database_module, "_REQUIRED_COLUMNS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_CHECK_CONSTRAINTS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_INDEXES", ())

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            instance._validate_schema()

        msg = str(exc_info.value)
        assert "Missing composite foreign keys:" in msg
        assert "transform_errors(token_id, run_id) → tokens(token_id, run_id)" in msg
        assert "transform_errors(transform_id, run_id) → nodes(node_id, run_id)" in msg
        instance.close()

    def test_validate_schema_rejects_missing_runtime_required_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Runtime write paths must not pass compatibility checks without their physical columns."""
        db_path = tmp_path / "missing_runtime_required_columns.db"
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE runs ("
                    "run_id TEXT PRIMARY KEY, "
                    "source_field_resolution_json TEXT, "
                    "schema_contract_json TEXT, "
                    "schema_contract_hash TEXT, "
                    "runtime_val_manifest_json TEXT)"
                )
            )
            conn.execute(text("CREATE TABLE checkpoints (checkpoint_id TEXT PRIMARY KEY, coalesce_state_json TEXT)"))
        engine.dispose()

        instance = _make_instance(f"sqlite:///{db_path}")
        monkeypatch.setattr(
            database_module,
            "metadata",
            SimpleNamespace(tables={"runs": object(), "checkpoints": object()}),
        )
        monkeypatch.setattr(database_module, "_REQUIRED_FOREIGN_KEYS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_COMPOSITE_FOREIGN_KEYS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_CHECK_CONSTRAINTS", ())
        monkeypatch.setattr(database_module, "_REQUIRED_INDEXES", ())

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            instance._validate_schema()

        msg = str(exc_info.value)
        assert "Missing columns:" in msg
        assert "runs.source_schema_json" in msg
        assert "checkpoints.format_version" in msg
        instance.close()

    def test_missing_check_constraint_raises_compatibility_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing required check constraints must be listed in the error."""
        db_path = tmp_path / "missing_check.db"
        engine = create_engine(f"sqlite:///{db_path}")
        metadata.create_all(engine)
        engine.dispose()

        instance = _make_instance(f"sqlite:///{db_path}")
        monkeypatch.setattr(
            database_module,
            "_REQUIRED_CHECK_CONSTRAINTS",
            (("runs", "ck_nonexistent_constraint"),),
        )

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            instance._validate_schema()

        msg = str(exc_info.value)
        assert "Missing check constraints:" in msg
        assert "runs.ck_nonexistent_constraint" in msg
        assert "To fix this, either:" in msg
        instance.close()

    def test_missing_index_raises_compatibility_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing required indexes must be listed in the error."""
        db_path = tmp_path / "missing_index.db"
        engine = create_engine(f"sqlite:///{db_path}")
        metadata.create_all(engine)
        engine.dispose()

        instance = _make_instance(f"sqlite:///{db_path}")
        monkeypatch.setattr(
            database_module,
            "_REQUIRED_INDEXES",
            (("runs", "ix_nonexistent_index"),),
        )

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            instance._validate_schema()

        msg = str(exc_info.value)
        assert "Missing indexes:" in msg
        assert "runs.ix_nonexistent_index" in msg
        assert "To fix this, either:" in msg
        instance.close()

    def test_non_sqlite_require_existing_schema_rejects_empty_database(self, tmp_path: Path) -> None:
        """Non-SQLite path must reject empty databases when _require_existing_schema is set."""
        db_path = tmp_path / "empty_non_sqlite.db"
        engine = create_engine(f"sqlite:///{db_path}")
        # Do NOT call metadata.create_all — database has no Landscape tables
        engine.dispose()

        instance = _make_instance(f"sqlite:///{db_path}")
        # Override connection_string to trigger the non-SQLite path
        instance.connection_string = "postgresql://user@host/db"
        instance._require_existing_schema = True

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            instance._validate_schema()

        msg = str(exc_info.value)
        assert "does not contain any Landscape tables" in msg
        instance.close()

    def test_validate_schema_translates_sqlcipher_error_on_get_table_names(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """SQLCipher passphrase errors during get_table_names() must produce SchemaCompatibilityError.

        Regression: The OperationalError guard only covered inspect(engine),
        but SQLAlchemy's inspect is lazy — the actual DB read happens on
        get_table_names(), where the same "file is not a database" error fires.
        """
        import sqlalchemy
        from sqlalchemy.exc import OperationalError

        instance = _make_instance(f"sqlite:///{tmp_path / 'encrypted.db'}")

        mock_inspector = Mock()
        mock_inspector.get_table_names.side_effect = OperationalError(
            "SELECT name FROM sqlite_master",
            {},
            Exception("file is not a database"),
        )

        monkeypatch.setattr(sqlalchemy, "inspect", lambda engine: mock_inspector)

        with pytest.raises(SchemaCompatibilityError, match="encrypted or passphrase is incorrect"):
            instance._validate_schema()

        instance.close()


class TestJournalPathGuards:
    """Coverage for from_url journal path derivation failure modes."""

    def test_from_url_stamps_schema_epoch_for_compatible_sqlite_db(self, tmp_path: Path) -> None:
        """Compatible SQLite databases should be stamped for future migrations."""
        db_path = tmp_path / "epoch_stamp.db"

        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        db.close()

        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            epoch = conn.exec_driver_sql("PRAGMA user_version").scalar_one()
        engine.dispose()

        assert epoch == SQLITE_SCHEMA_EPOCH

    def test_from_url_create_tables_false_does_not_stamp_schema_epoch(self, tmp_path: Path) -> None:
        """Read-only opens must not mutate compatible legacy SQLite databases."""
        db_path = tmp_path / "readonly_epoch.db"
        engine = create_engine(f"sqlite:///{db_path}")
        metadata.create_all(engine)
        engine.dispose()

        db = LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=False)
        db.close()

        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            epoch = conn.exec_driver_sql("PRAGMA user_version").scalar_one()
        engine.dispose()

        assert epoch == 0

    def test_from_url_dump_to_jsonl_requires_explicit_path_for_non_sqlite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-SQLite URLs must provide dump_to_jsonl_path explicitly."""
        mock_create_engine = Mock(return_value=Mock())
        monkeypatch.setattr(database_module, "create_engine", mock_create_engine)

        with pytest.raises(ValueError, match="dump_to_jsonl requires dump_to_jsonl_path for non-SQLite databases"):
            LandscapeDB.from_url("postgresql://user@host/db", dump_to_jsonl=True)

        mock_create_engine.assert_called_once_with("postgresql://user@host/db", echo=False)

    def test_from_url_dump_to_jsonl_rejects_in_memory_sqlite_without_path(self) -> None:
        """In-memory SQLite has no file path, so automatic journal derivation must fail."""
        with pytest.raises(ValueError, match="dump_to_jsonl requires a file-backed SQLite database"):
            LandscapeDB.from_url("sqlite:///:memory:", dump_to_jsonl=True)
