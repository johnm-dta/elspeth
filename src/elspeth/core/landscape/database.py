# src/elspeth/core/landscape/database.py
"""Database connection management for Landscape.

Handles SQLite (development) and PostgreSQL (production) backends
with appropriate settings for each.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Self

from sqlalchemy import Connection, create_engine, event
from sqlalchemy.engine import Engine

from elspeth.core.landscape.schema import metadata


class SchemaCompatibilityError(Exception):
    """Raised when the Landscape database schema is incompatible with current code."""

    pass


# Required columns that have been added since initial schema.
# Used by _validate_schema() to detect outdated SQLite databases.
_REQUIRED_COLUMNS: list[tuple[str, str]] = [
    ("tokens", "expand_group_id"),
]


class LandscapeDB:
    """Landscape database connection manager."""

    def __init__(self, connection_string: str) -> None:
        """Initialize database connection.

        Args:
            connection_string: SQLAlchemy connection string
                e.g., "sqlite:///./runs/landscape.db"
                      "postgresql://user:pass@host/dbname"
        """
        self.connection_string = connection_string
        self._engine: Engine | None = None
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
        """Validate that existing database has all required columns.

        Only validates SQLite databases. PostgreSQL deployments are expected
        to use Alembic migrations which handle schema evolution properly.
        This check catches developers using stale local audit.db files.

        Raises:
            SchemaCompatibilityError: If database is missing required columns
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

        if missing_columns:
            missing_str = ", ".join(f"{t}.{c}" for t, c in missing_columns)
            raise SchemaCompatibilityError(
                f"Landscape database schema is outdated. "
                f"Missing columns: {missing_str}\n\n"
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
        return instance

    @classmethod
    def from_url(cls, url: str, *, create_tables: bool = True) -> Self:
        """Create database from connection URL.

        Args:
            url: SQLAlchemy connection URL
            create_tables: Whether to create tables if they don't exist.
                           Set to False when connecting to an existing database.

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

        # Validate BEFORE create_all - catches old schema with missing columns
        # before we try to use it. For fresh DBs, validation passes (no tables yet).
        instance._validate_schema()

        if create_tables:
            metadata.create_all(engine)
        return instance

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
