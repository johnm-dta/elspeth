# src/elspeth/core/landscape/database.py
"""Database connection management for Landscape.

Handles SQLite (development) and PostgreSQL (production) backends
with appropriate settings for each.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self

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
    # Phase 5: Schema contract audit trail - captures contracts in effect for run
    ("runs", "schema_contract_json"),
    ("runs", "schema_contract_hash"),
    # Phase 5: Plugin contract audit trail - captures input/output contracts per node
    ("nodes", "input_contract_json"),
    ("nodes", "output_contract_json"),
    # Operation I/O hashes - survive payload purge for integrity verification
    ("operations", "input_data_hash"),
    ("operations", "output_data_hash"),
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
        passphrase: str | None = None,
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
            passphrase: SQLCipher encryption passphrase. When provided, the
                database is opened with AES-256 encryption via sqlcipher3.
                The passphrase is never stored in the URL or audit trail.
            dump_to_jsonl: Enable JSONL change journal for emergency backups
            dump_to_jsonl_path: Optional override path for JSONL journal
            dump_to_jsonl_fail_on_error: Fail if journal write fails
            dump_to_jsonl_include_payloads: Inline payloads in journal records
            dump_to_jsonl_payload_base_path: Payload store base path for inlining
        """
        self.connection_string = connection_string
        self._passphrase = passphrase
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
        if self._passphrase is not None:
            self._engine = self._create_sqlcipher_engine(self.connection_string, self._passphrase)
            LandscapeDB._configure_sqlite(self._engine)
        else:
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
        - PRAGMA busy_timeout=5000 (contention tolerance)

        For SQLCipher engines, these PRAGMAs execute AFTER the creator callback
        returns (where PRAGMA key is issued), preserving the required ordering.

        Args:
            engine: SQLAlchemy Engine to configure
        """

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection: object, connection_record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]  # SQLAlchemy event passes DBAPI connection (has .cursor()) typed as object
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            # Enable foreign key enforcement
            cursor.execute("PRAGMA foreign_keys=ON")
            # Set busy timeout to avoid immediate SQLITE_BUSY errors under contention
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    @staticmethod
    def _create_sqlcipher_engine(url: str, passphrase: str) -> Engine:
        """Create a SQLAlchemy engine backed by SQLCipher (AES-256 encryption).

        Uses the creator callback pattern to keep the passphrase out of the
        connection URL entirely (prevents leaks in logs, tracebacks, repr()).

        PRAGMA key MUST be the first statement on a new SQLCipher connection.
        The creator issues it before returning, so SQLAlchemy's "connect" event
        (used by _configure_sqlite for WAL/FK/busy_timeout) fires afterwards.

        Args:
            url: SQLAlchemy SQLite URL (e.g., "sqlite:///./state/audit.db")
            passphrase: Encryption passphrase for PRAGMA key

        Returns:
            Configured SQLAlchemy Engine

        Raises:
            ImportError: If sqlcipher3 is not installed
            ValueError: If URL points to :memory: (SQLCipher requires a file)
        """
        try:
            import sqlcipher3  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "sqlcipher3 is required for encrypted audit databases. "
                "Install it with: uv pip install 'elspeth[security]'\n"
                "Note: requires libsqlcipher-dev system package."
            ) from None

        parsed = make_url(url)

        # SQLCipher only works with SQLite — reject other backends early
        # to prevent silently opening a local file when the URL points elsewhere
        # (e.g., postgresql://host/db with ELSPETH_AUDIT_KEY set in env).
        driver = parsed.drivername.split("+")[0]  # "sqlite+aiosqlite" → "sqlite"
        if driver != "sqlite":
            raise ValueError(
                f"SQLCipher encryption requires a SQLite database URL, "
                f"got driver '{parsed.drivername}'. "
                f"Either remove the passphrase/encryption_key_env or change "
                f"the URL to sqlite:///path/to/audit.db"
            )

        db_path = parsed.database
        if db_path is None or db_path == ":memory:":
            raise ValueError("SQLCipher requires a file-backed database (cannot encrypt :memory:)")

        # Resolve relative paths the same way SQLite does
        resolved_path = str(Path(db_path).resolve())

        # Forward URL query params as connect kwargs (parity with non-encrypted
        # path, where create_engine extracts them automatically).  Coerce known
        # sqlite3.connect() params from their string URL representation.
        #
        # SQLite URI-style params (mode, cache, immutable, vfs) are NOT valid
        # connect() kwargs — they must be embedded in a file: URI when uri=True.
        _CONNECT_KWARGS = {"check_same_thread", "uri", "timeout", "detect_types", "cached_statements", "isolation_level", "factory"}
        # Match SQLAlchemy's pysqlite default: allow cross-thread usage so the
        # connection pool can hand connections to worker threads.  URL params
        # parsed below can still override this explicitly.
        connect_kwargs: dict[str, Any] = {"check_same_thread": False}
        uri_params: dict[str, str] = {}

        for key, raw_value in parsed.query.items():
            value = raw_value if isinstance(raw_value, str) else raw_value[0]
            if key in _CONNECT_KWARGS:
                if key in ("check_same_thread", "uri"):
                    connect_kwargs[key] = value.lower() in ("true", "1", "yes")
                elif key == "timeout":
                    connect_kwargs[key] = float(value)
                elif key in ("detect_types", "cached_statements"):
                    connect_kwargs[key] = int(value)
                else:
                    connect_kwargs[key] = value
            else:
                # URI-style param (mode, cache, immutable, vfs, etc.)
                uri_params[key] = value

        # When URI params are present, build a file: URI and enable uri=True
        # so that SQLite interprets them via the URI interface.
        if uri_params:
            from urllib.parse import quote, urlencode

            file_uri = f"file:{quote(resolved_path)}?{urlencode(uri_params)}"
            connect_kwargs["uri"] = True
        else:
            file_uri = None

        def _creator() -> object:
            db = file_uri if file_uri is not None else resolved_path
            conn = sqlcipher3.connect(db, **connect_kwargs)
            # PRAGMA key MUST be the first statement — SQLCipher contract.
            # Escape double quotes in the passphrase (SQLite literal syntax:
            # a literal " inside a double-quoted string is written as "").
            # PRAGMA doesn't support parameter binding, so escaping is required.
            escaped = passphrase.replace('"', '""')
            conn.execute(f'PRAGMA key = "{escaped}"')
            return conn

        return create_engine("sqlite:///", creator=_creator, echo=False)

    def _create_tables(self) -> None:
        """Create all tables if they don't exist."""
        metadata.create_all(self.engine)

    def _validate_schema(self) -> None:
        """Validate that existing database has all required columns and foreign keys.

        Only validates SQLite databases. PostgreSQL deployments are expected
        to use Alembic migrations which handle schema evolution properly.
        This check catches developers using stale local audit.db files.

        Raises:
            SchemaCompatibilityError: If database is missing required columns or FKs,
                or if an encrypted database is opened without the correct passphrase.
        """
        if not self.connection_string.startswith("sqlite"):
            return

        from sqlalchemy import inspect
        from sqlalchemy.exc import OperationalError

        try:
            inspector = inspect(self.engine)
            existing_tables = set(inspector.get_table_names())
        except OperationalError as e:
            error_msg = str(e)
            if "file is not a database" in error_msg or "file is encrypted" in error_msg:
                raise SchemaCompatibilityError(
                    "Cannot open Landscape database — file is encrypted or passphrase is incorrect.\n\n"
                    "If this is an encrypted (SQLCipher) database, ensure:\n"
                    "  1. The correct passphrase is set in the configured environment variable\n"
                    "     (landscape.encryption_key_env in settings.yaml, default: ELSPETH_AUDIT_KEY)\n"
                    "  2. backend: sqlcipher is set in settings.yaml\n\n"
                    f"Database: {self.connection_string}"
                ) from e
            raise
        expected_tables = set(metadata.tables.keys())
        present_landscape_tables = existing_tables & expected_tables

        # If this looks like an existing Landscape database, all known tables must exist.
        # For brand-new DB files (no Landscape tables yet), creation happens in create_all().
        missing_tables = sorted(expected_tables - existing_tables) if present_landscape_tables else []

        missing_columns: list[tuple[str, str]] = []

        for table_name, column_name in _REQUIRED_COLUMNS:
            # Check if table exists
            if table_name not in existing_tables:
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
            if table_name not in existing_tables:
                # Table will be created by create_all, skip
                continue

            # Check if FK exists AND targets the correct referenced table
            # SQLAlchemy inspector API guarantees constrained_columns and referred_table keys
            fks = inspector.get_foreign_keys(table_name)
            has_correct_fk = any(column_name in fk["constrained_columns"] and fk["referred_table"] == referenced_table for fk in fks)

            if not has_correct_fk:
                missing_fks.append((table_name, column_name, referenced_table))

        # Raise errors for missing columns or FKs
        if missing_tables or missing_columns or missing_fks:
            error_parts = []

            if missing_tables:
                missing_tables_str = ", ".join(missing_tables)
                error_parts.append(f"Missing tables: {missing_tables_str}")

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
        passphrase: str | None = None,
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
            passphrase: SQLCipher encryption passphrase. When provided, the
                database is opened with AES-256 encryption via sqlcipher3.
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
        if passphrase is not None:
            engine = cls._create_sqlcipher_engine(url, passphrase)
            cls._configure_sqlite(engine)
        else:
            engine = create_engine(url, echo=False)
            # SQLite-specific configuration
            if url.startswith("sqlite"):
                cls._configure_sqlite(engine)

        # Create instance first - _validate_schema needs self.connection_string and self.engine
        instance = cls.__new__(cls)
        instance.connection_string = url
        instance._passphrase = passphrase
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
