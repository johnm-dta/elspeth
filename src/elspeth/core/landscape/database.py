# src/elspeth/core/landscape/database.py
"""Database connection management for Landscape.

Handles SQLite (development) and PostgreSQL (production) backends
with appropriate settings for each.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Self

from sqlalchemy import Connection, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

from elspeth.core.landscape.journal import LandscapeJournal
from elspeth.core.landscape.schema import metadata


class SchemaCompatibilityError(Exception):
    """Raised when the Landscape database schema is incompatible with current code."""

    pass


# Required columns that have been added since initial schema.
# Used by _validate_schema() to detect outdated SQLite databases.
_REQUIRED_COLUMNS: list[tuple[str, str]] = [
    ("tokens", "expand_group_id"),
    # Added for composite FK to nodes (node_id, run_id) - enables run-isolated queries
    ("node_states", "run_id"),
    # Field resolution audit trail - captures original→final header mapping
    ("runs", "source_field_resolution_json"),
    # Fork/expand branch contract - enables recovery validation
    ("token_outcomes", "expected_branches_json"),
    # Transform success reason audit trail - captures why transform succeeded
    ("node_states", "success_reason_json"),
    # Operation call linkage - enables source/sink call tracking
    ("calls", "operation_id"),
]

# Required foreign keys for audit integrity (Tier 1 trust).
# Format: (table_name, column_name, referenced_table)
# Bug fix: P2-2026-01-19-error-tables-missing-foreign-keys
_REQUIRED_FOREIGN_KEYS: list[tuple[str, str, str]] = [
    ("validation_errors", "node_id", "nodes"),
    ("transform_errors", "token_id", "tokens"),
    ("transform_errors", "transform_id", "nodes"),
]


class LandscapeDB:
    """Landscape database connection manager."""

    def __init__(
        self,
        connection_string: str,
        *,
        dump_to_jsonl: bool = False,
        dump_to_jsonl_path: str | None = None,
        dump_to_jsonl_fail_on_error: bool = False,
        dump_to_jsonl_include_payloads: bool = False,
        dump_to_jsonl_payload_base_path: str | None = None,
    ) -> None:
        """Initialize database connection.

        Args:
            connection_string: SQLAlchemy connection string
                e.g., "sqlite:///./state/audit.db"
                      "postgresql://user:pass@host/dbname"
            dump_to_jsonl: Enable JSONL change journal for emergency backups
            dump_to_jsonl_path: Optional override path for JSONL journal
            dump_to_jsonl_fail_on_error: Fail if journal write fails
            dump_to_jsonl_include_payloads: Inline payloads in journal records
            dump_to_jsonl_payload_base_path: Payload store base path for inlining
        """
        self.connection_string = connection_string
        self._engine: Engine | None = None
        self._journal: LandscapeJournal | None = None
        if dump_to_jsonl:
            journal_path = dump_to_jsonl_path or self._derive_journal_path(connection_string)
            self._journal = LandscapeJournal(
                journal_path,
                fail_on_error=dump_to_jsonl_fail_on_error,
                include_payloads=dump_to_jsonl_include_payloads,
                payload_base_path=dump_to_jsonl_payload_base_path,
            )
        self._setup_engine()
        self._validate_schema()  # Check BEFORE create_tables
        self._create_tables()

    def _setup_engine(self) -> None:
        """Create and configure the database engine."""
        self._engine = create_engine(
            self.connection_string,
            echo=False,  # Set True for SQL debugging
        )

        # SQLite-specific configuration
        if self.connection_string.startswith("sqlite"):
            LandscapeDB._configure_sqlite(self._engine)
        if self._journal is not None:
            self._journal.attach(self._engine)

    @staticmethod
    def _configure_sqlite(engine: Engine) -> None:
        """Configure SQLite engine for reliability.

        Registers a connection event hook that sets:
        - PRAGMA journal_mode=WAL (better concurrency)
        - PRAGMA foreign_keys=ON (referential integrity)

        Args:
            engine: SQLAlchemy Engine to configure
        """

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection: object, connection_record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            # Enable foreign key enforcement
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    def _create_tables(self) -> None:
        """Create all tables if they don't exist."""
        metadata.create_all(self.engine)

    def _validate_schema(self) -> None:
        """Validate that existing database has all required columns and foreign keys.

        Only validates SQLite databases. PostgreSQL deployments are expected
        to use Alembic migrations which handle schema evolution properly.
        This check catches developers using stale local audit.db files.

        Raises:
            SchemaCompatibilityError: If database is missing required columns or FKs
        """
        if not self.connection_string.startswith("sqlite"):
            return

        from sqlalchemy import inspect

        inspector = inspect(self.engine)

        missing_columns: list[tuple[str, str]] = []

        for table_name, column_name in _REQUIRED_COLUMNS:
            # Check if table exists
            if table_name not in inspector.get_table_names():
                # Table will be created by create_all, skip
                continue

            # Check if column exists
            columns = {c["name"] for c in inspector.get_columns(table_name)}
            if column_name not in columns:
                missing_columns.append((table_name, column_name))

        # Check for required foreign keys (Tier 1 audit integrity)
        missing_fks: list[tuple[str, str, str]] = []

        for table_name, column_name, referenced_table in _REQUIRED_FOREIGN_KEYS:
            # Check if table exists
            if table_name not in inspector.get_table_names():
                # Table will be created by create_all, skip
                continue

            # Check if FK exists AND targets the correct referenced table
            # SQLAlchemy inspector API guarantees constrained_columns and referred_table keys
            fks = inspector.get_foreign_keys(table_name)
            has_correct_fk = any(column_name in fk["constrained_columns"] and fk["referred_table"] == referenced_table for fk in fks)

            if not has_correct_fk:
                missing_fks.append((table_name, column_name, referenced_table))

        # Raise errors for missing columns or FKs
        if missing_columns or missing_fks:
            error_parts = []

            if missing_columns:
                missing_str = ", ".join(f"{t}.{c}" for t, c in missing_columns)
                error_parts.append(f"Missing columns: {missing_str}")

            if missing_fks:
                missing_fk_str = ", ".join(f"{t}.{c} → {ref}" for t, c, ref in missing_fks)
                error_parts.append(f"Missing foreign keys: {missing_fk_str}")

            raise SchemaCompatibilityError(
                "Landscape database schema is outdated.\n\n" + "\n".join(error_parts) + "\n\n"
                f"To fix this, either:\n"
                f"  1. Delete the database file and let ELSPETH recreate it, or\n"
                f"  2. Run: elspeth landscape migrate (when available)\n\n"
                f"Database: {self.connection_string}"
            )

    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine."""
        if self._engine is None:
            raise RuntimeError("Database not initialized")
        return self._engine

    def close(self) -> None:
        """Close database connection."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    @classmethod
    def in_memory(cls) -> Self:
        """Create an in-memory SQLite database for testing.

        Tables are created automatically.

        Returns:
            LandscapeDB instance with in-memory SQLite
        """
        engine = create_engine("sqlite:///:memory:", echo=False)
        cls._configure_sqlite(engine)
        metadata.create_all(engine)
        instance = cls.__new__(cls)
        instance.connection_string = "sqlite:///:memory:"
        instance._engine = engine
        instance._journal = None
        return instance

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        create_tables: bool = True,
        dump_to_jsonl: bool = False,
        dump_to_jsonl_path: str | None = None,
        dump_to_jsonl_fail_on_error: bool = False,
        dump_to_jsonl_include_payloads: bool = False,
        dump_to_jsonl_payload_base_path: str | None = None,
    ) -> Self:
        """Create database from connection URL.

        Args:
            url: SQLAlchemy connection URL
            create_tables: Whether to create tables if they don't exist.
                           Set to False when connecting to an existing database.
            dump_to_jsonl: Enable JSONL change journal for emergency backups
            dump_to_jsonl_path: Optional override path for JSONL journal
            dump_to_jsonl_fail_on_error: Fail if journal write fails
            dump_to_jsonl_include_payloads: Inline payloads in journal records
            dump_to_jsonl_payload_base_path: Payload store base path for inlining

        Returns:
            LandscapeDB instance
        """
        engine = create_engine(url, echo=False)
        # SQLite-specific configuration
        if url.startswith("sqlite"):
            cls._configure_sqlite(engine)

        # Create instance first - _validate_schema needs self.connection_string and self.engine
        instance = cls.__new__(cls)
        instance.connection_string = url
        instance._engine = engine
        instance._journal = None
        if dump_to_jsonl:
            journal_path = dump_to_jsonl_path or cls._derive_journal_path(url)
            instance._journal = LandscapeJournal(
                journal_path,
                fail_on_error=dump_to_jsonl_fail_on_error,
                include_payloads=dump_to_jsonl_include_payloads,
                payload_base_path=dump_to_jsonl_payload_base_path,
            )
            instance._journal.attach(engine)

        # Validate BEFORE create_all - catches old schema with missing columns
        # before we try to use it. For fresh DBs, validation passes (no tables yet).
        instance._validate_schema()

        if create_tables:
            metadata.create_all(engine)
        return instance

    @staticmethod
    def _derive_journal_path(connection_string: str) -> str:
        """Derive a default JSONL journal path from the connection string."""
        url = make_url(connection_string)
        if not url.drivername.startswith("sqlite"):
            raise ValueError("dump_to_jsonl requires dump_to_jsonl_path for non-SQLite databases")
        database = url.database
        if database is None or database == ":memory:":
            raise ValueError("dump_to_jsonl requires a file-backed SQLite database")
        return str(Path(database).with_suffix(".journal.jsonl"))

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        """Get a database connection with automatic transaction handling.

        Uses engine.begin() for proper transaction semantics:
        - Auto-commits on successful block exit
        - Auto-rolls back on exception

        Usage:
            with db.connection() as conn:
                conn.execute(runs_table.insert().values(...))
            # Committed automatically if no exception raised
        """
        with self.engine.begin() as conn:
            yield conn
