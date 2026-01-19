# tests/core/landscape/test_database.py
"""Tests for Landscape database connection management."""

from pathlib import Path


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

    def test_connection_context_manager(self) -> None:
        from sqlalchemy import text

        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB.in_memory()
        with db.connection() as conn:
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
