# tests/core/landscape/test_database.py
"""Tests for Landscape database connection management."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.core.landscape.database import LandscapeDB


class TestDatabaseConnection:
    """Database connection and initialization."""

    def test_connect_creates_tables(self, tmp_path: Path) -> None:
        from sqlalchemy import inspect

        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "landscape.db"
        db = LandscapeDB(f"sqlite:///{db_path}")

        # Tables should be created
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()

        assert "runs" in tables
        assert "nodes" in tables

    def test_sqlite_wal_mode(self, tmp_path: Path) -> None:
        from sqlalchemy import text

        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "landscape.db"
        db = LandscapeDB(f"sqlite:///{db_path}")

        with db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode"))
            mode = result.scalar()
            assert mode == "wal"

    def test_sqlite_foreign_keys_enabled(self, tmp_path: Path) -> None:
        """Verify constructor enables foreign key enforcement."""
        from sqlalchemy import text

        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "fk_test.db"
        db = LandscapeDB(f"sqlite:///{db_path}")

        with db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA foreign_keys"))
            fk_enabled = result.scalar()
            assert fk_enabled == 1, f"Expected foreign_keys=1, got {fk_enabled}"

    def test_context_manager(self, tmp_path: Path) -> None:
        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "landscape.db"

        with LandscapeDB(f"sqlite:///{db_path}") as db:
            assert db.engine is not None


class TestPhase3ADBMethods:
    """Tests for methods added in Phase 3A."""

    def test_in_memory_factory(self) -> None:
        from sqlalchemy import inspect

        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB.in_memory()
        assert db.engine is not None
        inspector = inspect(db.engine)
        assert "runs" in inspector.get_table_names()

    def test_connection_context_manager(self, landscape_db: "LandscapeDB") -> None:
        from sqlalchemy import text

        with landscape_db.connection() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_from_url_factory(self, tmp_path: Path) -> None:
        from sqlalchemy import inspect

        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "test.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        assert db_path.exists()
        inspector = inspect(db.engine)
        assert "runs" in inspector.get_table_names()

    def test_from_url_skip_table_creation(self, tmp_path: Path) -> None:
        """Test that create_tables=False doesn't create tables."""
        from sqlalchemy import create_engine, inspect

        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "empty.db"
        # First create an empty database file (no tables)
        empty_engine = create_engine(f"sqlite:///{db_path}")
        empty_engine.dispose()

        # Connect with create_tables=False - should NOT create tables
        db = LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=False)
        inspector = inspect(db.engine)
        assert "runs" not in inspector.get_table_names()  # No tables!

    def test_in_memory_enables_foreign_keys(self) -> None:
        """Verify in_memory() enables foreign key enforcement."""
        from sqlalchemy import text

        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB.in_memory()
        with db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA foreign_keys"))
            fk_enabled = result.scalar()
            assert fk_enabled == 1, f"Expected foreign_keys=1, got {fk_enabled}"

    def test_in_memory_enables_wal_mode(self) -> None:
        """Verify in_memory() sets WAL journal mode.

        Note: In-memory databases report 'memory' for journal_mode,
        which is acceptable since they don't persist to disk.
        """
        from sqlalchemy import text

        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB.in_memory()
        with db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode"))
            mode = result.scalar()
            # In-memory DBs report 'memory' not 'wal', which is fine
            assert mode in ("wal", "memory"), f"Expected wal or memory, got {mode}"

    def test_from_url_enables_foreign_keys(self, tmp_path: Path) -> None:
        """Verify from_url() enables foreign key enforcement for SQLite."""
        from sqlalchemy import text

        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "fk_test.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        with db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA foreign_keys"))
            fk_enabled = result.scalar()
            assert fk_enabled == 1, f"Expected foreign_keys=1, got {fk_enabled}"

    def test_from_url_enables_wal_mode(self, tmp_path: Path) -> None:
        """Verify from_url() sets WAL journal mode for SQLite."""
        from sqlalchemy import text

        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "wal_test.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        with db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode"))
            mode = result.scalar()
            assert mode == "wal", f"Expected wal, got {mode}"


class TestSchemaCompatibility:
    """Tests for schema version checking."""

    def test_fresh_database_passes_validation(self, tmp_path: Path) -> None:
        """A new database should pass schema validation."""
        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "test.db"
        # Should not raise
        db = LandscapeDB(f"sqlite:///{db_path}")
        db.close()

    def test_in_memory_database_passes_validation(self) -> None:
        """In-memory database should pass schema validation."""
        from elspeth.core.landscape.database import LandscapeDB

        # Should not raise
        db = LandscapeDB.in_memory()
        db.close()

    def test_old_schema_missing_column_fails_validation(self, tmp_path: Path) -> None:
        """Database missing required columns should fail with clear error."""
        import pytest
        from sqlalchemy import create_engine, text

        from elspeth.core.landscape.database import (
            LandscapeDB,
            SchemaCompatibilityError,
        )

        db_path = tmp_path / "old_schema.db"

        # Create a database with old schema (tokens without expand_group_id)
        old_engine = create_engine(f"sqlite:///{db_path}")
        with old_engine.begin() as conn:
            # Create minimal old schema
            conn.execute(
                text("""
                CREATE TABLE runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TIMESTAMP NOT NULL
                )
            """)
            )
            conn.execute(
                text("""
                CREATE TABLE rows (
                    row_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL
                )
            """)
            )
            conn.execute(
                text("""
                CREATE TABLE tokens (
                    token_id TEXT PRIMARY KEY,
                    row_id TEXT NOT NULL,
                    fork_group_id TEXT,
                    join_group_id TEXT,
                    branch_name TEXT,
                    step_in_pipeline INTEGER,
                    created_at TIMESTAMP NOT NULL
                )
            """)
            )
            # Note: expand_group_id is intentionally missing
        old_engine.dispose()

        # Now try to open with current LandscapeDB
        with pytest.raises(SchemaCompatibilityError) as exc_info:
            LandscapeDB(f"sqlite:///{db_path}")

        error_msg = str(exc_info.value)
        assert "tokens.expand_group_id" in error_msg
        assert "schema is outdated" in error_msg.lower()
        assert "delete the database" in error_msg.lower() or "migrate" in error_msg.lower()

    def test_from_url_with_old_schema_fails(self, tmp_path: Path) -> None:
        """Verify from_url() catches old schema before create_all()."""
        import pytest
        from sqlalchemy import create_engine, text

        from elspeth.core.landscape.database import (
            LandscapeDB,
            SchemaCompatibilityError,
        )

        db_path = tmp_path / "old_schema.db"

        # Create old schema
        old_engine = create_engine(f"sqlite:///{db_path}")
        with old_engine.begin() as conn:
            conn.execute(
                text("""
                CREATE TABLE tokens (
                    token_id TEXT PRIMARY KEY,
                    row_id TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            """)
            )
        old_engine.dispose()

        # from_url should fail even with create_tables=True
        with pytest.raises(SchemaCompatibilityError) as exc_info:
            LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=True)

        assert "expand_group_id" in str(exc_info.value)

    def test_error_message_includes_remediation(self, tmp_path: Path) -> None:
        """Error message should tell user how to fix the problem."""
        import pytest
        from sqlalchemy import create_engine, text

        from elspeth.core.landscape.database import (
            LandscapeDB,
            SchemaCompatibilityError,
        )

        db_path = tmp_path / "old_schema.db"

        # Create old schema
        old_engine = create_engine(f"sqlite:///{db_path}")
        with old_engine.begin() as conn:
            conn.execute(text("CREATE TABLE tokens (token_id TEXT PRIMARY KEY)"))
        old_engine.dispose()

        with pytest.raises(SchemaCompatibilityError) as exc_info:
            LandscapeDB(f"sqlite:///{db_path}")

        error_msg = str(exc_info.value)
        # Should include actionable remediation
        assert "Delete" in error_msg or "delete" in error_msg
        assert str(db_path) in error_msg or "sqlite" in error_msg

    def test_missing_foreign_key_fails_validation(self, tmp_path: Path) -> None:
        """P1: Database with table but missing FK constraint should fail validation.

        Schema compatibility tests previously only covered missing columns.
        FK enforcement is Tier 1 audit integrity - missing FKs would allow
        orphan error records, violating the Data Manifesto.
        """
        import pytest
        from sqlalchemy import create_engine, text

        from elspeth.core.landscape.database import (
            LandscapeDB,
            SchemaCompatibilityError,
        )

        db_path = tmp_path / "missing_fk.db"

        # Create schema with tables but intentionally missing FK constraints
        old_engine = create_engine(f"sqlite:///{db_path}")
        with old_engine.begin() as conn:
            # Create runs table (needed for FK)
            conn.execute(
                text("""
                CREATE TABLE runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TIMESTAMP NOT NULL
                )
            """)
            )
            # Create nodes table (needed for FK)
            conn.execute(
                text("""
                CREATE TABLE nodes (
                    node_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL
                )
            """)
            )
            # Create rows table
            conn.execute(
                text("""
                CREATE TABLE rows (
                    row_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL
                )
            """)
            )
            # Create tokens table with required columns
            conn.execute(
                text("""
                CREATE TABLE tokens (
                    token_id TEXT PRIMARY KEY,
                    row_id TEXT NOT NULL,
                    fork_group_id TEXT,
                    join_group_id TEXT,
                    expand_group_id TEXT,
                    branch_name TEXT,
                    step_in_pipeline INTEGER,
                    created_at TIMESTAMP NOT NULL
                )
            """)
            )
            # Create validation_errors WITHOUT FK constraint on node_id
            conn.execute(
                text("""
                CREATE TABLE validation_errors (
                    error_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    node_id TEXT,
                    row_hash TEXT NOT NULL,
                    error TEXT NOT NULL,
                    schema_mode TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            """)
            )
            # Note: node_id has no FOREIGN KEY constraint - this is the defect
        old_engine.dispose()

        # P1: Schema validation should catch missing FK
        with pytest.raises(SchemaCompatibilityError) as exc_info:
            LandscapeDB(f"sqlite:///{db_path}")

        error_msg = str(exc_info.value)
        assert "foreign key" in error_msg.lower() or "Missing foreign keys" in error_msg


class TestLandscapeJournal:
    """Tests for JSONL change journal."""

    def test_journal_records_committed_writes(self, tmp_path: Path) -> None:
        import json

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db_path = tmp_path / "audit.db"
        journal_path = tmp_path / "audit.journal.jsonl"

        db = LandscapeDB.from_url(
            f"sqlite:///{db_path}",
            dump_to_jsonl=True,
            dump_to_jsonl_path=str(journal_path),
        )
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(
            config={"example": "journal"},
            canonical_version="sha256-rfc8785-v1",
        )

        assert journal_path.exists()
        records = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]

        assert records
        assert any("INSERT INTO RUNS" in record["statement"].upper() for record in records)
        assert any(run.run_id in json.dumps(record["parameters"]) for record in records)

    def test_journal_skips_rolled_back_writes(self, tmp_path: Path) -> None:
        import pytest

        from elspeth.contracts import RunStatus
        from elspeth.core.landscape._helpers import generate_id, now
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import runs_table

        db_path = tmp_path / "audit.db"
        journal_path = tmp_path / "audit.journal.jsonl"

        db = LandscapeDB.from_url(
            f"sqlite:///{db_path}",
            dump_to_jsonl=True,
            dump_to_jsonl_path=str(journal_path),
        )

        with pytest.raises(RuntimeError), db.connection() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=generate_id(),
                    started_at=now(),
                    config_hash="0" * 64,
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.RUNNING.value,
                )
            )
            raise RuntimeError("force rollback")

        assert not journal_path.exists() or journal_path.read_text().strip() == ""
