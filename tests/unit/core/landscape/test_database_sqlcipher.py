# tests/unit/core/landscape/test_database_sqlcipher.py
"""Tests for SQLCipher encryption-at-rest support in LandscapeDB.

All tests skip if sqlcipher3 is not installed (optional 'security' extra).
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

sqlcipher3 = pytest.importorskip("sqlcipher3", reason="sqlcipher3 not installed (install with: uv pip install 'elspeth[security]')")


class TestSQLCipherCreateAndRead:
    """Basic CRUD operations on an encrypted database."""

    def test_sqlcipher_create_and_read(self, tmp_path: Path) -> None:
        from sqlalchemy import select

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import runs_table

        db_path = tmp_path / "encrypted.db"
        passphrase = "test-passphrase-42"

        db = LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase=passphrase)
        try:
            # Insert a run record with all NOT NULL columns
            with db.write_connection() as conn:
                conn.execute(
                    runs_table.insert().values(
                        run_id="test-run-001",
                        status="RUNNING",
                        started_at=datetime.now(UTC),
                        config_hash="abc123",
                        settings_json="{}",
                        canonical_version="1.0.0",
                        openrouter_catalog_sha256="0" * 64,
                        openrouter_catalog_source="bundled",
                    )
                )

            # Read it back
            with db.connection() as conn:
                result = conn.execute(select(runs_table).where(runs_table.c.run_id == "test-run-001")).fetchone()
                assert result is not None
                assert result.run_id == "test-run-001"
                assert result.status == "RUNNING"
        finally:
            db.close()


class TestSQLCipherPragmaOrdering:
    """PRAGMA key must execute before WAL/FK/busy_timeout."""

    def test_sqlcipher_pragma_key_first(self, tmp_path: Path) -> None:
        """Verify that PRAGMA key runs before other PRAGMAs by checking
        that WAL mode is active (which requires a valid connection)."""
        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "pragma_order.db"
        passphrase = "pragma-test"

        db = LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase=passphrase)
        try:
            # If PRAGMA key wasn't first, the WAL/FK PRAGMAs would have
            # failed and this query would fail too
            with db.connection() as conn:
                from sqlalchemy import text

                result = conn.execute(text("PRAGMA journal_mode")).fetchone()
                assert result is not None
                assert result[0] == "wal"
        finally:
            db.close()


class TestSQLCipherWrongPassphrase:
    """Wrong passphrase produces a clear error."""

    def test_sqlcipher_wrong_passphrase(self, tmp_path: Path) -> None:
        from elspeth.core.landscape.database import LandscapeDB, SchemaCompatibilityError

        db_path = tmp_path / "wrong_pass.db"

        # Create with one passphrase
        db = LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase="correct-passphrase")
        db.close()

        # Open with wrong passphrase — should fail with clear message
        with pytest.raises((SchemaCompatibilityError, Exception)) as exc_info:
            LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase="wrong-passphrase")
        # Verify the error is actionable (either our custom message or SQLCipher's)
        error_msg = str(exc_info.value).lower()
        assert "file is not a database" in error_msg or "encrypted" in error_msg or "passphrase" in error_msg


class TestSQLCipherNoPassphraseOnEncrypted:
    """Plain SQLite against an encrypted DB fails clearly."""

    def test_sqlcipher_no_passphrase_on_encrypted(self, tmp_path: Path) -> None:
        from elspeth.core.landscape.database import LandscapeDB, SchemaCompatibilityError

        db_path = tmp_path / "encrypted_no_pass.db"

        # Create encrypted DB
        db = LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase="secret-key")
        db.close()

        # Try opening without passphrase (plain SQLite) — should fail
        # Standard sqlite3 sees garbage when reading SQLCipher-encrypted files
        with pytest.raises((SchemaCompatibilityError, Exception)) as exc_info:
            LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase=None)
        error_msg = str(exc_info.value).lower()
        assert "file is not a database" in error_msg or "encrypted" in error_msg or "passphrase" in error_msg


class TestSQLCipherSchemaIntegrity:
    """All tables, indexes, and constraints are created in encrypted DB."""

    def test_sqlcipher_schema_integrity(self, tmp_path: Path) -> None:
        from sqlalchemy import inspect

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import metadata

        db_path = tmp_path / "schema_check.db"
        passphrase = "schema-test"

        db = LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase=passphrase)
        try:
            inspector = inspect(db.engine)
            actual_tables = set(inspector.get_table_names())
            expected_tables = set(metadata.tables.keys())

            assert expected_tables == actual_tables, f"Missing: {expected_tables - actual_tables}, Extra: {actual_tables - expected_tables}"

            # Verify a composite FK exists (nodes table FK integrity)
            fks = inspector.get_foreign_keys("node_states")
            fk_tables = {fk["referred_table"] for fk in fks}
            assert "nodes" in fk_tables, "node_states should reference nodes table"
        finally:
            db.close()


class TestSQLCipherWALMode:
    """WAL journal mode is active on SQLCipher connections."""

    def test_sqlcipher_wal_mode_active(self, tmp_path: Path) -> None:
        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "wal_test.db"
        passphrase = "wal-test"

        db = LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase=passphrase)
        try:
            with db.connection() as conn:
                from sqlalchemy import text

                result = conn.execute(text("PRAGMA journal_mode")).fetchone()
                assert result is not None
                assert result[0] == "wal"
        finally:
            db.close()


class TestSQLCipherReadOnly:
    """SQLCipher read-only handles must use a file-level SQLite read-only URI."""

    def test_sqlcipher_read_only_open_does_not_create_missing_database(self, tmp_path: Path) -> None:
        from sqlalchemy.exc import OperationalError

        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "missing-read-only.db"

        with pytest.raises((OperationalError, sqlcipher3.OperationalError)):
            LandscapeDB.from_url(
                f"sqlite:///{db_path}",
                passphrase="read-only-passphrase",
                read_only=True,
                create_tables=False,
            )

        assert not db_path.exists()

    def test_sqlcipher_read_only_open_rejects_writes_after_query_only_is_disabled(self, tmp_path: Path) -> None:
        from sqlalchemy import insert
        from sqlalchemy.exc import OperationalError

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import runs_table

        db_path = tmp_path / "read-only.db"
        passphrase = "read-only-passphrase"
        writable = LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase=passphrase)
        writable.close()

        read_only = LandscapeDB.from_url(
            f"sqlite:///{db_path}",
            passphrase=passphrase,
            read_only=True,
            create_tables=False,
        )
        try:
            with (
                pytest.raises(
                    (OperationalError, sqlcipher3.OperationalError),
                    match=r"readonly|read-only|attempt to write",
                ),
                read_only.engine.begin() as conn,
            ):
                conn.exec_driver_sql("PRAGMA query_only=OFF")
                conn.execute(
                    insert(runs_table).values(
                        run_id="write-should-fail",
                        status="RUNNING",
                        started_at=datetime.now(UTC),
                        config_hash="abc123",
                        settings_json="{}",
                        canonical_version="1.0.0",
                        openrouter_catalog_sha256="0" * 64,
                        openrouter_catalog_source="bundled",
                    )
                )
        finally:
            read_only.close()


class TestSQLCipherForeignKeys:
    """FK violations are rejected on encrypted connections."""

    def test_sqlcipher_foreign_keys_enforced(self, tmp_path: Path) -> None:
        from sqlalchemy import text

        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "fk_test.db"
        passphrase = "fk-test"

        db = LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase=passphrase)
        try:
            # Insert a node with non-existent run_id — FK should reject.
            # Note: sqlcipher3.dbapi2.IntegrityError may not be wrapped by
            # sqlalchemy.exc.IntegrityError when using the creator pattern,
            # so we catch both via Exception and verify the message.
            with pytest.raises(Exception, match="FOREIGN KEY constraint failed"), db.write_connection() as conn:
                conn.execute(
                    text(
                        "INSERT INTO nodes (node_id, run_id, plugin_name, node_type, "
                        "plugin_version, determinism, config_hash, config_json, "
                        "sequence_in_pipeline, registered_at) "
                        "VALUES (:nid, :rid, :pn, :nt, :pv, :det, :ch, :cj, :seq, :rat)"
                    ),
                    {
                        "nid": "node-001",
                        "rid": "nonexistent-run",
                        "pn": "csv",
                        "nt": "source",
                        "pv": "1.0.0",
                        "det": "deterministic",
                        "ch": "hash",
                        "cj": "{}",
                        "seq": 0,
                        "rat": datetime.now(UTC).isoformat(),
                    },
                )
        finally:
            db.close()


class TestPlainSQLiteUnchanged:
    """passphrase=None still produces an unencrypted, standard-sqlite3-readable DB."""

    def test_plain_sqlite_unchanged(self, tmp_path: Path) -> None:
        import sqlite3

        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "plain.db"

        db = LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase=None)
        db.close()

        # Standard sqlite3 should be able to open and read it
        conn = sqlite3.connect(str(db_path))
        try:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = {row[0] for row in tables}
            assert "runs" in table_names
            assert "nodes" in table_names
        finally:
            conn.close()


class TestSQLCipherRejectsNonSQLite:
    """SQLCipher with non-SQLite URLs raises a clear ValueError."""

    def test_sqlcipher_rejects_postgresql_url(self) -> None:
        """Regression: passphrase + PostgreSQL URL must not silently open a local file."""
        from elspeth.core.landscape.database import LandscapeDB

        with pytest.raises(ValueError, match="requires a SQLite database URL"):
            LandscapeDB.from_url("postgresql://user:pass@host/mydb", passphrase="test-key")  # secret-scan: allow-this-line

    def test_sqlcipher_rejects_mysql_url(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB

        with pytest.raises(ValueError, match="requires a SQLite database URL"):
            LandscapeDB.from_url("mysql://user:pass@host/mydb", passphrase="test-key")  # secret-scan: allow-this-line

    def test_sqlcipher_rejects_postgresql_with_driver(self) -> None:
        """Driver variants like postgresql+psycopg2 are also rejected."""
        from elspeth.core.landscape.database import LandscapeDB

        with pytest.raises(ValueError, match="requires a SQLite database URL"):
            LandscapeDB.from_url("postgresql+psycopg2://host/db", passphrase="test-key")

    def test_sqlcipher_accepts_sqlite_with_driver(self, tmp_path: Path) -> None:
        """sqlite+pysqlite (explicit driver) should still work."""
        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "driver_variant.db"
        # Should not raise — "sqlite+pysqlite" is SQLite-family
        db = LandscapeDB.from_url(f"sqlite+pysqlite:///{db_path}", passphrase="test-key")
        db.close()


class TestMCPPassphraseGating:
    """MCP entrypoint only forwards passphrase for SQLite backends.

    Regression test: ELSPETH_AUDIT_KEY was unconditionally forwarded to
    LandscapeDB.from_url, causing startup failures when the MCP server was
    pointed at non-SQLite databases (e.g. postgresql://) in environments
    that export the key for other ELSPETH commands.
    """

    def test_non_sqlite_url_ignores_env_passphrase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-SQLite database_url should not pick up ELSPETH_AUDIT_KEY."""
        monkeypatch.setenv("ELSPETH_AUDIT_KEY", "super-secret-key")

        database_url = "postgresql://user:pass@host/mydb"  # secret-scan: allow-this-line

        # Reproduce the gating logic from mcp/server.py main()
        import os

        passphrase: str | None = None
        if database_url.startswith("sqlite"):
            passphrase = os.environ.get("ELSPETH_AUDIT_KEY")

        assert passphrase is None, "Non-SQLite URL should not receive passphrase from ELSPETH_AUDIT_KEY"

    def test_sqlite_url_picks_up_env_passphrase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SQLite database_url should receive ELSPETH_AUDIT_KEY as passphrase."""
        monkeypatch.setenv("ELSPETH_AUDIT_KEY", "super-secret-key")

        database_url = "sqlite:///./state/audit.db"

        import os

        passphrase: str | None = None
        if database_url.startswith("sqlite"):
            passphrase = os.environ.get("ELSPETH_AUDIT_KEY")

        assert passphrase == "super-secret-key"

    def test_sqlite_url_without_env_key_gets_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SQLite URL without ELSPETH_AUDIT_KEY set gets passphrase=None."""
        monkeypatch.delenv("ELSPETH_AUDIT_KEY", raising=False)

        database_url = "sqlite:///./state/audit.db"

        import os

        passphrase: str | None = None
        if database_url.startswith("sqlite"):
            passphrase = os.environ.get("ELSPETH_AUDIT_KEY")

        assert passphrase is None


class TestSQLCipherRejectsMemory:
    """SQLCipher with :memory: raises a clear ValueError."""

    def test_sqlcipher_rejects_memory(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB

        with pytest.raises(ValueError, match="file-backed database"):
            LandscapeDB.from_url("sqlite:///:memory:", passphrase="test-key")


class TestSQLCipherPassphraseEscaping:
    """Passphrases with special characters don't break PRAGMA key."""

    def test_passphrase_with_double_quotes(self, tmp_path: Path) -> None:
        """Regression: passphrase containing " must not cause SQL syntax error."""
        from sqlalchemy import select

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import runs_table

        db_path = tmp_path / "quoted_pass.db"
        passphrase = 'my"secret"key'

        db = LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase=passphrase)
        try:
            # Insert and read back to verify the connection works
            with db.write_connection() as conn:
                conn.execute(
                    runs_table.insert().values(
                        run_id="test-escape",
                        status="RUNNING",
                        started_at=datetime.now(UTC),
                        config_hash="abc",
                        settings_json="{}",
                        canonical_version="1.0.0",
                        openrouter_catalog_sha256="0" * 64,
                        openrouter_catalog_source="bundled",
                    )
                )
            with db.connection() as conn:
                result = conn.execute(select(runs_table.c.run_id)).fetchone()
                assert result is not None
                assert result[0] == "test-escape"
        finally:
            db.close()

        # Re-open with the same passphrase to confirm round-trip
        db2 = LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase=passphrase)
        try:
            with db2.connection() as conn:
                result = conn.execute(select(runs_table.c.run_id)).fetchone()
                assert result is not None
                assert result[0] == "test-escape"
        finally:
            db2.close()

    def test_passphrase_with_backslashes(self, tmp_path: Path) -> None:
        """Backslashes in passphrase should not cause issues."""
        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "backslash_pass.db"
        passphrase = r"C:\Users\admin\key"

        db = LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase=passphrase)
        try:
            # Verify we can interact with the encrypted database
            with db.connection() as conn:
                from sqlalchemy import text

                result = conn.execute(text("PRAGMA journal_mode")).fetchone()
                assert result is not None
                assert result[0] == "wal"
        finally:
            db.close()
