"""Database connection management for Landscape.

Handles SQLite (development) and PostgreSQL (production) backends
with appropriate settings for each.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self

from sqlalchemy import Connection, create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import make_url

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.journal import LandscapeJournal
from elspeth.core.landscape.schema import SQLITE_SCHEMA_EPOCH, metadata

# Canonical SQLite PRAGMA invariants for the Landscape audit DB.
#
# Tier-1 doctrine: the audit DB is the legal record. If SQLite doesn't accept
# any of these PRAGMAs, the crash-safety / concurrency guarantees the audit
# subsystem depends on are unmet and we MUST refuse to open the database.
#
# - journal_mode=WAL — file-backed DBs MUST run in WAL mode (better
#   concurrency, fsync-on-checkpoint instead of fsync-per-write).  SQLite
#   silently downgrades to 'memory' for ``:memory:`` databases (WAL has no
#   meaning without an on-disk journal file); the probe accepts that.
# - synchronous=NORMAL — paired with WAL, this is the canonical
#   write-heavy crash-safe shape: durable across application crashes and
#   sufficiently durable across OS crashes when fsync semantics are honoured
#   by the filesystem.  FULL is slower without meaningful added durability
#   under WAL.  See https://www.sqlite.org/pragma.html#pragma_synchronous
# - foreign_keys=ON — referential integrity is mandatory for the audit
#   schema (run_id / token_id / node_id FKs must enforce).
# - busy_timeout=5000 ms — tolerate brief contention between recorders
#   without surfacing SQLITE_BUSY to the caller.
_SQLITE_PRAGMA_INVARIANTS_FILE: tuple[tuple[str, str], ...] = (
    ("journal_mode", "wal"),
    ("synchronous", "1"),
    ("foreign_keys", "1"),
    ("busy_timeout", "5000"),
)
_SQLITE_PRAGMA_INVARIANTS_MEMORY: tuple[tuple[str, str], ...] = (
    # ``:memory:`` databases have no on-disk journal file; SQLite reports
    # ``journal_mode=memory`` regardless of what we requested.  That is
    # the only sanctioned deviation from the file-backed contract.
    ("journal_mode", "memory"),
    ("synchronous", "1"),
    ("foreign_keys", "1"),
    ("busy_timeout", "5000"),
)


class SchemaCompatibilityError(Exception):
    """Raised when the Landscape database schema is incompatible with current code."""

    pass


ADR019_MIGRATION_GUIDE = "docs/operator/migrations/adr-019.md"


# Required columns that have been added since initial schema.
# Used by _validate_schema() to detect outdated SQLite databases.
_REQUIRED_COLUMNS: tuple[tuple[str, str], ...] = (
    # Web auth audit trail - records login, token, and auth failure events.
    ("auth_events", "event_id"),
    ("auth_events", "occurred_at"),
    ("auth_events", "event_type"),
    ("auth_events", "outcome"),
    ("auth_events", "provider"),
    ("auth_events", "metadata_json"),
    # Web run attribution - records the authenticated user that initiated a run.
    ("run_attributions", "run_id"),
    ("run_attributions", "recorded_at"),
    ("run_attributions", "initiated_by_user_id"),
    ("run_attributions", "auth_provider_type"),
    ("tokens", "expand_group_id"),
    # Added for run ownership — prevents cross-run contamination of token-linked records
    ("tokens", "run_id"),
    # Added for composite FK to nodes (node_id, run_id) - enables run-isolated queries
    ("node_states", "run_id"),
    # Source schema persists typed resume metadata; runtime always writes this on new runs
    ("runs", "source_schema_json"),
    # Field resolution audit trail - captures original→final header mapping
    ("runs", "source_field_resolution_json"),
    # Fork/expand branch contract - enables recovery validation
    ("token_outcomes", "expected_branches_json"),
    # Transform success reason audit trail - captures why transform succeeded
    ("node_states", "success_reason_json"),
    # Operation call linkage - enables source/sink call tracking
    ("calls", "operation_id"),
    # Phase 5: Plugin contract audit trail - captures input/output contracts per node
    ("nodes", "input_contract_json"),
    ("nodes", "output_contract_json"),
    # Operation I/O hashes - survive payload purge for integrity verification
    ("operations", "input_data_hash"),
    ("operations", "output_data_hash"),
    # Coalesce state for checkpoint recovery - serialized pending merge state
    ("checkpoints", "coalesce_state_json"),
    # Checkpoint compatibility gate - runtime always stamps the checkpoint format version
    ("checkpoints", "format_version"),
    # Mechanical change detection — hash of plugin source file at registration
    ("nodes", "source_file_hash"),
    # ADR-010 §Decision 3 M3: runtime VAL manifest — set of
    # declaration contracts + Tier-1 error classes registered at bootstrap,
    # serialized as canonical JSON on the runs row.
    ("runs", "runtime_val_manifest_json"),
    # Quarantine lineage exactness - links validation errors to persisted rows
    ("validation_errors", "row_id"),
    # Batch membership run ownership - enables composite FK enforcement to both
    # batches and tokens so cross-run batch contamination fails at the database.
    ("batch_members", "run_id"),
    # Retry lineage exactness - retry_batch() must deduplicate per failed batch.
    ("batches", "retry_of_batch_id"),
    # ADR-019 two-axis terminal model: old is_terminal DBs must fail fast.
    ("token_outcomes", "completed"),
    ("token_outcomes", "path"),
    # Phase 5b interpretation-review audit anchor — runtime LLM calls must
    # carry the resolved prompt hash used to join back to session DB events.
    ("calls", "resolved_prompt_template_hash"),
    # Phase 4 tutorial audit-story projection fields.
    ("runs", "llm_call_count"),
    ("runs", "seeded_from_cache"),
    ("runs", "cache_key"),
    # Preflight results are written after run creation; partial stale tables
    # otherwise pass schema validation and fail later on insert.
    ("preflight_results", "result_id"),
    ("preflight_results", "run_id"),
    ("preflight_results", "result_type"),
    ("preflight_results", "name"),
    ("preflight_results", "result_json"),
    ("preflight_results", "created_at"),
    ("run_sources", "run_id"),
    ("run_sources", "source_node_id"),
    ("run_sources", "source_name"),
    ("run_sources", "lifecycle_state"),
    ("rows", "source_row_index"),
    ("rows", "ingest_sequence"),
    ("token_work_items", "work_item_id"),
    ("token_work_items", "run_id"),
    ("token_work_items", "token_id"),
    ("token_work_items", "row_id"),
    ("token_work_items", "node_id"),
    ("token_work_items", "step_index"),
    ("token_work_items", "ingest_sequence"),
    ("token_work_items", "status"),
    ("token_work_items", "queue_key"),
    ("token_work_items", "barrier_key"),
    ("token_work_items", "available_at"),
    ("token_work_items", "row_payload_json"),
    ("token_work_items", "on_success_sink"),
    ("token_work_items", "pending_sink_name"),
    ("token_work_items", "pending_outcome"),
    ("token_work_items", "pending_path"),
    ("token_work_items", "pending_error_hash"),
    ("token_work_items", "pending_error_message"),
    ("token_work_items", "branch_name"),
    ("token_work_items", "fork_group_id"),
    ("token_work_items", "join_group_id"),
    ("token_work_items", "expand_group_id"),
    ("token_work_items", "coalesce_node_id"),
    ("token_work_items", "coalesce_name"),
    ("token_work_items", "attempt"),
    ("token_work_items", "lease_owner"),
    ("token_work_items", "lease_expires_at"),
    ("token_work_items", "created_at"),
    ("token_work_items", "updated_at"),
)

# Required foreign keys for audit integrity (Tier 1 trust).
# Format: (table_name, column_name, referenced_table)
# Use this only for exact single-column contracts. Run-scoped contracts belong in
# _REQUIRED_COMPOSITE_FOREIGN_KEYS so stale single-column FKs cannot satisfy them.
_REQUIRED_FOREIGN_KEYS: tuple[tuple[str, str, str], ...] = (
    ("validation_errors", "row_id", "rows"),
    ("preflight_results", "run_id", "runs"),
)

# Required composite foreign keys for run-scoped audit integrity.
# Format: (table_name, constrained_columns, referenced_table, referenced_columns)
_REQUIRED_COMPOSITE_FOREIGN_KEYS: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    ("token_outcomes", ("token_id", "run_id"), "tokens", ("token_id", "run_id")),
    ("token_outcomes", ("batch_id", "run_id"), "batches", ("batch_id", "run_id")),
    ("node_states", ("token_id", "run_id"), "tokens", ("token_id", "run_id")),
    ("node_states", ("node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("validation_errors", ("node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("transform_errors", ("token_id", "run_id"), "tokens", ("token_id", "run_id")),
    ("transform_errors", ("transform_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("artifacts", ("produced_by_state_id", "run_id"), "node_states", ("state_id", "run_id")),
    ("artifacts", ("sink_node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("run_sources", ("source_node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("batches", ("aggregation_node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("batches", ("aggregation_state_id", "run_id"), "node_states", ("state_id", "run_id")),
    ("batches", ("retry_of_batch_id", "run_id"), "batches", ("batch_id", "run_id")),
    ("batch_members", ("batch_id", "run_id"), "batches", ("batch_id", "run_id")),
    ("batch_members", ("token_id", "run_id"), "tokens", ("token_id", "run_id")),
    ("token_work_items", ("token_id", "run_id"), "tokens", ("token_id", "run_id")),
    ("token_work_items", ("row_id", "run_id"), "rows", ("row_id", "run_id")),
    ("token_work_items", ("node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("token_work_items", ("coalesce_node_id", "run_id"), "nodes", ("node_id", "run_id")),
)

# Required check constraints for audit integrity.
# Format: (table_name, constraint_name)
_REQUIRED_CHECK_CONSTRAINTS: tuple[tuple[str, str], ...] = (
    ("auth_events", "ck_auth_events_event_type"),
    ("auth_events", "ck_auth_events_outcome"),
    ("auth_events", "ck_auth_events_provider"),
    ("run_attributions", "ck_run_attributions_auth_provider_type"),
    ("run_sources", "ck_run_sources_lifecycle_state"),
    ("calls", "calls_has_parent"),
    ("preflight_results", "ck_preflight_result_type"),
)

# Required indexes (including partial unique indexes) for audit integrity.
# Format: (table_name, index_name)
_REQUIRED_INDEXES: tuple[tuple[str, str], ...] = (
    ("auth_events", "ix_auth_events_occurred_at"),
    ("auth_events", "ix_auth_events_type_outcome"),
    ("auth_events", "ix_auth_events_user"),
    ("run_attributions", "ix_run_attributions_user"),
    ("calls", "ix_calls_state_call_index_unique"),
    ("calls", "ix_calls_operation_call_index_unique"),
    ("calls", "ix_calls_resolved_prompt_template_hash"),
    ("checkpoints", "ix_checkpoints_run_sequence_unique"),
    ("preflight_results", "ix_preflight_results_run"),
    ("token_outcomes", "ix_token_outcomes_terminal_unique"),
    ("token_work_items", "ix_token_work_items_ready"),
    ("token_work_items", "ix_token_work_items_lease"),
    ("token_work_items", "uq_token_work_items_terminal_identity"),
    ("validation_errors", "ix_validation_errors_run_row"),
)

_ADDITIVE_INDEX_NAMES: frozenset[str] = frozenset({"ix_tokens_run_id"})
_ADDITIVE_TABLE_NAMES: frozenset[str] = frozenset({"auth_events", "run_attributions"})


def _collect_missing_required_columns(inspector: Inspector) -> list[tuple[str, str]]:
    """Return required columns missing from existing tables."""
    existing_tables = set(inspector.get_table_names())
    missing: list[tuple[str, str]] = []
    for table_name, column_name in _REQUIRED_COLUMNS:
        if table_name not in existing_tables:
            continue
        existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
        if column_name not in existing_columns:
            missing.append((table_name, column_name))
    return missing


def _collect_token_outcomes_shape_errors(
    inspector: Inspector,
    *,
    engine: Engine | None = None,
    inspect_sqlite_indexes: bool = False,
) -> list[str]:
    """Return ADR-019 shape errors for existing token_outcomes tables."""
    existing_tables = set(inspector.get_table_names())
    if "token_outcomes" not in existing_tables:
        return []

    columns = {column["name"]: column for column in inspector.get_columns("token_outcomes")}
    errors: list[str] = []

    if "is_terminal" in columns:
        errors.append("token_outcomes.is_terminal is stale; ADR-019 uses completed")
    if "outcome" in columns and columns["outcome"]["nullable"] is False:
        errors.append("token_outcomes.outcome nullable shape is stale; ADR-019 requires nullable outcome for BUFFERED rows")

    if inspect_sqlite_indexes and engine is not None:
        with engine.connect() as conn:
            index_sql = conn.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'ix_token_outcomes_terminal_unique'"
            ).scalar_one_or_none()
        if index_sql is not None and "is_terminal" in str(index_sql).lower():
            errors.append("token_outcomes stale terminal index predicate references is_terminal; ADR-019 uses completed")

    return errors


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
                      "postgresql://user@host/dbname"
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
        self._require_existing_schema = False
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
        self._create_additive_indexes()
        self._sync_sqlite_schema_epoch()

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
        # Tier-1: probe the SQLite PRAGMAs we just configured — if any
        # didn't take effect, the audit DB does not meet the durability /
        # concurrency contract and we MUST refuse to open it.  Skipped for
        # non-SQLite backends (PostgreSQL has no equivalent surface here).
        if self._engine is not None and self.connection_string.startswith("sqlite"):
            LandscapeDB._verify_sqlite_pragmas(self._engine, self.connection_string)

    @staticmethod
    def _configure_sqlite(engine: Engine) -> None:
        """Configure SQLite engine for reliability.

        Registers a connection event hook that sets:
        - PRAGMA journal_mode=WAL (better concurrency)
        - PRAGMA synchronous=NORMAL (canonical WAL crash-safety shape)
        - PRAGMA foreign_keys=ON (referential integrity)
        - PRAGMA busy_timeout=5000 (contention tolerance)

        For SQLCipher engines, these PRAGMAs execute AFTER the creator callback
        returns (where PRAGMA key is issued), preserving the required ordering.

        The values set here MUST stay aligned with the invariants enforced by
        :meth:`_verify_sqlite_pragmas`; the probe is the audit-integrity gate
        that fails the database open if SQLite ever refuses to honour them.

        Args:
            engine: SQLAlchemy Engine to configure
        """

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection: object, connection_record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]  # SQLAlchemy event passes DBAPI connection (has .cursor()) typed as object
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            # synchronous=NORMAL is the canonical pairing with WAL: durable
            # across app crashes, sufficiently durable across OS crashes,
            # without the per-write fsync overhead of FULL.
            cursor.execute("PRAGMA synchronous=NORMAL")
            # Enable foreign key enforcement
            cursor.execute("PRAGMA foreign_keys=ON")
            # Set busy timeout to avoid immediate SQLITE_BUSY errors under contention
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    @staticmethod
    def _verify_sqlite_pragmas(engine: Engine, connection_string: str) -> None:
        """Probe SQLite to confirm the configured PRAGMAs actually took effect.

        Opens a connection (which triggers the ``connect`` event hook that
        applies the PRAGMAs), then reads each PRAGMA back and asserts the
        value matches the invariant.  Any mismatch raises
        :class:`AuditIntegrityError` — Tier-1 doctrine: the audit DB cannot
        proceed with weaker durability/concurrency guarantees than the audit
        subsystem was designed against.

        SQLite silently downgrades ``journal_mode`` to ``memory`` for
        ``:memory:`` databases (WAL requires an on-disk journal file); the
        probe selects the file-backed or memory-backed invariant set from
        the connection string.

        Args:
            engine: SQLAlchemy Engine configured by :meth:`_configure_sqlite`.
            connection_string: Original connection string, used to decide
                whether ``journal_mode=memory`` is acceptable.

        Raises:
            AuditIntegrityError: If any PRAGMA reports a value other than
                what :meth:`_configure_sqlite` requested.  The message names
                the PRAGMA, the expected value, and the observed value so
                operators can diagnose without needing to re-instrument.
        """
        # Resolve invariants once — :memory: is a single sanctioned deviation.
        url = make_url(connection_string)
        is_memory = url.database is None or url.database == ":memory:"
        invariants = _SQLITE_PRAGMA_INVARIANTS_MEMORY if is_memory else _SQLITE_PRAGMA_INVARIANTS_FILE

        with engine.connect() as conn:
            observed: dict[str, str] = {}
            for pragma, _expected in invariants:
                # PRAGMA names are static (literal table above), not user
                # input — f-string interpolation is safe here.
                result = conn.exec_driver_sql(f"PRAGMA {pragma}").scalar_one_or_none()
                observed[pragma] = "" if result is None else str(result).lower()

        mismatches: list[str] = []
        for pragma, expected in invariants:
            actual = observed[pragma]
            if actual != expected.lower():
                mismatches.append(f"PRAGMA {pragma}: expected {expected!r}, observed {actual!r}")

        if mismatches:
            # Audit DB cannot proceed with degraded guarantees.  Tier-1:
            # raise immediately, do not attempt remediation.  The message
            # avoids the connection string (path may be sensitive) but
            # names every offending PRAGMA so the operator can act.
            raise AuditIntegrityError(
                "Landscape SQLite PRAGMA invariants violated at engine startup; audit-integrity guarantees unmet. " + "; ".join(mismatches)
            )

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
            import sqlcipher3
        except ImportError as exc:
            raise ImportError(
                "sqlcipher3 is required for encrypted audit databases. "
                "Install it with: uv pip install 'elspeth[security]'\n"
                "Note: requires libsqlcipher-dev system package."
            ) from exc

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

    def _create_additive_indexes(self) -> None:
        """Create non-gating performance indexes for existing schemas."""
        for table in metadata.tables.values():
            for index in table.indexes:
                if index.name in _ADDITIVE_INDEX_NAMES:
                    index.create(self.engine, checkfirst=True)

    def _get_sqlite_schema_epoch(self) -> int:
        """Return SQLite schema epoch from PRAGMA user_version.

        Uses SQLite's built-in schema version slot as a lightweight marker for
        intentional pre-1.0 schema breaks. This is not a migration system; it
        simply gives future migration code a stable entry point.
        """
        if not self.connection_string.startswith("sqlite"):
            return 0

        with self.engine.connect() as conn:
            return int(conn.exec_driver_sql("PRAGMA user_version").scalar_one())

    def _set_sqlite_schema_epoch(self, epoch: int) -> None:
        """Persist the SQLite schema epoch in PRAGMA user_version."""
        if not self.connection_string.startswith("sqlite"):
            return

        with self.engine.begin() as conn:
            conn.exec_driver_sql(f"PRAGMA user_version = {int(epoch)}")

    def _sync_sqlite_schema_epoch(self) -> None:
        """Stamp compatible SQLite databases with the current schema epoch.

        New databases get the epoch immediately after create_all(). Existing
        compatible databases without an epoch are upgraded in place to the
        current stamp, which preserves a future migration path without requiring
        a full migration framework today. Call this only from schema-managing
        paths; read-only/inspection opens must not mutate the database.

        Raises:
            SchemaCompatibilityError: If the database has a newer epoch than
                this code version expects (prevents silent downgrades).
        """
        if not self.connection_string.startswith("sqlite"):
            return

        current_epoch = self._get_sqlite_schema_epoch()
        if current_epoch > SQLITE_SCHEMA_EPOCH:
            raise SchemaCompatibilityError(
                "Cannot sync schema epoch: database has a newer epoch than this "
                "ELSPETH version supports.\n\n"
                f"Database epoch: {current_epoch}\n"
                f"Current epoch: {SQLITE_SCHEMA_EPOCH}\n\n"
                "This database was created or stamped by a newer ELSPETH version. "
                "Upgrade ELSPETH to open this database.\n\n"
                f"Database: {self.connection_string}"
            )
        if current_epoch < SQLITE_SCHEMA_EPOCH:
            self._set_sqlite_schema_epoch(SQLITE_SCHEMA_EPOCH)

    def _validate_schema(self) -> None:
        """Validate that existing database has the required schema.

        For non-SQLite backends, validates table existence when
        _require_existing_schema is set (inspection callers like MCP/CLI).
        For SQLite, performs full validation of columns, foreign keys,
        check constraints, and indexes to catch stale local audit.db files.

        Raises:
            SchemaCompatibilityError: If database is missing required schema
                elements, or if an encrypted database is opened without the
                correct passphrase.
        """
        from sqlalchemy import inspect
        from sqlalchemy.exc import OperationalError

        try:
            inspector = inspect(self.engine)
            existing_tables = set(inspector.get_table_names())
        except OperationalError as e:
            error_msg = str(e)
            if self.connection_string.startswith("sqlite") and ("file is not a database" in error_msg or "file is encrypted" in error_msg):
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
        schema_epoch = self._get_sqlite_schema_epoch() if self.connection_string.startswith("sqlite") else 0

        # If this looks like an existing Landscape database, all known tables must exist.
        # For brand-new DB files (no Landscape tables yet), creation happens in create_all().
        #
        # If _require_existing_schema is set (create_tables=False callers like MCP/CLI),
        # we require at least some Landscape tables to be present. An empty/non-Landscape
        # DB with create_tables=False would fail later with raw SQL errors — fail fast instead.
        if self._require_existing_schema and not present_landscape_tables:
            raise SchemaCompatibilityError(
                "Database does not contain any Landscape tables.\n\n"
                "This does not appear to be an ELSPETH audit database. "
                "Verify the database path is correct.\n\n"
                f"Database: {self.connection_string}"
            )
        missing_tables = sorted((expected_tables - existing_tables) - _ADDITIVE_TABLE_NAMES) if present_landscape_tables else []

        missing_columns = _collect_missing_required_columns(inspector)
        token_outcomes_shape_errors = _collect_token_outcomes_shape_errors(
            inspector,
            engine=self.engine,
            inspect_sqlite_indexes=self.connection_string.startswith("sqlite"),
        )

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

        # Check for required composite foreign keys (Tier 1 audit integrity)
        missing_composite_fks: list[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = []

        for table_name, constrained_columns, referenced_table, referenced_columns in _REQUIRED_COMPOSITE_FOREIGN_KEYS:
            if table_name not in existing_tables:
                continue

            fks = inspector.get_foreign_keys(table_name)
            has_correct_fk = any(
                tuple(fk["constrained_columns"]) == constrained_columns
                and fk["referred_table"] == referenced_table
                and tuple(fk["referred_columns"]) == referenced_columns
                for fk in fks
            )

            if not has_correct_fk:
                missing_composite_fks.append((table_name, constrained_columns, referenced_table, referenced_columns))

        # Check for required check constraints (Tier 1 audit integrity)
        missing_checks: list[tuple[str, str]] = []

        for table_name, constraint_name in _REQUIRED_CHECK_CONSTRAINTS:
            if table_name not in existing_tables:
                continue

            checks = inspector.get_check_constraints(table_name)
            has_constraint = any(c["name"] == constraint_name for c in checks)

            if not has_constraint:
                missing_checks.append((table_name, constraint_name))

        # Check for required indexes (Tier 1 audit integrity)
        missing_indexes: list[tuple[str, str]] = []

        for table_name, index_name in _REQUIRED_INDEXES:
            if table_name not in existing_tables:
                continue

            indexes = inspector.get_indexes(table_name)
            has_index = any(idx["name"] == index_name for idx in indexes)

            if not has_index:
                missing_indexes.append((table_name, index_name))

        epoch_incompatible = present_landscape_tables and schema_epoch not in (0, SQLITE_SCHEMA_EPOCH)

        # Raise errors for missing columns, FKs, check constraints, indexes, or stale ADR-019 shapes.
        if (
            missing_tables
            or missing_columns
            or token_outcomes_shape_errors
            or missing_fks
            or missing_composite_fks
            or missing_checks
            or missing_indexes
            or epoch_incompatible
        ):
            error_parts = []

            if epoch_incompatible:
                error_parts.append(f"schema epoch is incompatible:\nDatabase epoch: {schema_epoch}\nCurrent epoch: {SQLITE_SCHEMA_EPOCH}")

            if missing_tables:
                missing_tables_str = ", ".join(missing_tables)
                error_parts.append(f"Missing tables: {missing_tables_str}")

            if missing_columns:
                missing_str = ", ".join(f"{t}.{c}" for t, c in missing_columns)
                error_parts.append(f"Missing columns: {missing_str}")

            if token_outcomes_shape_errors:
                error_parts.append("ADR-019 stale token_outcomes shape: " + "; ".join(token_outcomes_shape_errors))

            if missing_fks:
                missing_fk_str = ", ".join(f"{t}.{c} → {ref}" for t, c, ref in missing_fks)
                error_parts.append(f"Missing foreign keys: {missing_fk_str}")

            if missing_composite_fks:
                missing_composite_fk_str = ", ".join(
                    f"{table}({', '.join(columns)}) → {ref_table}({', '.join(ref_columns)})"
                    for table, columns, ref_table, ref_columns in missing_composite_fks
                )
                error_parts.append(f"Missing composite foreign keys: {missing_composite_fk_str}")

            if missing_checks:
                missing_checks_str = ", ".join(f"{t}.{name}" for t, name in missing_checks)
                error_parts.append(f"Missing check constraints: {missing_checks_str}")

            if missing_indexes:
                missing_indexes_str = ", ".join(f"{t}.{name}" for t, name in missing_indexes)
                error_parts.append(f"Missing indexes: {missing_indexes_str}")

            if (
                ("token_outcomes", "completed") in missing_columns
                or ("token_outcomes", "path") in missing_columns
                or token_outcomes_shape_errors
            ):
                error_parts.append(
                    "ADR-019 changed token_outcomes from the old single-axis outcome/is_terminal to "
                    "(TerminalOutcome, TerminalPath, completed). See "
                    f"{ADR019_MIGRATION_GUIDE} and replace the stale audit.db "
                    "before starting this ELSPETH version."
                )

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
    def _from_parts(
        cls,
        connection_string: str,
        engine: Engine,
        *,
        passphrase: str | None = None,
        journal: LandscapeJournal | None = None,
        require_existing_schema: bool = False,
    ) -> Self:
        """Construct instance from pre-created components.

        Single place that sets all instance attributes — eliminates drift risk
        from parallel ``cls.__new__()`` paths that manually assign fields.
        """
        instance = cls.__new__(cls)
        instance.connection_string = connection_string
        instance._passphrase = passphrase
        instance._engine = engine
        instance._journal = journal
        instance._require_existing_schema = require_existing_schema
        return instance

    @classmethod
    def in_memory(cls) -> Self:
        """Create an in-memory SQLite database for testing.

        Tables are created automatically.

        Returns:
            LandscapeDB instance with in-memory SQLite
        """
        engine = create_engine("sqlite:///:memory:", echo=False)
        cls._configure_sqlite(engine)
        cls._verify_sqlite_pragmas(engine, "sqlite:///:memory:")
        metadata.create_all(engine)
        instance = cls._from_parts("sqlite:///:memory:", engine)
        instance._sync_sqlite_schema_epoch()
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
            # Tier-1 PRAGMA probe — see _verify_sqlite_pragmas docstring.
            cls._verify_sqlite_pragmas(engine, url)
        else:
            engine = create_engine(url, echo=False)
            # SQLite-specific configuration
            if url.startswith("sqlite"):
                cls._configure_sqlite(engine)
                cls._verify_sqlite_pragmas(engine, url)

        journal: LandscapeJournal | None = None
        if dump_to_jsonl:
            journal_path = dump_to_jsonl_path or cls._derive_journal_path(url)
            journal = LandscapeJournal(
                journal_path,
                fail_on_error=dump_to_jsonl_fail_on_error,
                include_payloads=dump_to_jsonl_include_payloads,
                payload_base_path=dump_to_jsonl_payload_base_path,
            )
            journal.attach(engine)

        instance = cls._from_parts(
            url,
            engine,
            passphrase=passphrase,
            journal=journal,
            require_existing_schema=not create_tables,
        )

        # Validate BEFORE create_all - catches old schema with missing columns
        # before we try to use it. For fresh DBs, validation passes (no tables yet).
        instance._validate_schema()

        if create_tables:
            metadata.create_all(engine)
            instance._create_additive_indexes()
            instance._sync_sqlite_schema_epoch()
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

    @contextmanager
    def read_only_connection(self) -> Iterator[Connection]:
        """Get a database connection that rejects all write operations.

        Defense-in-depth for untrusted SQL execution (e.g., MCP query tool).
        For SQLite, sets PRAGMA query_only = ON at the connection level and
        resets it to OFF in the finally block so the pooled DBAPI connection
        is returned in a writable state. For PostgreSQL, marks the current
        transaction READ ONLY. Unsupported backends fail closed instead of
        yielding a writable transaction.
        """
        dialect_name = self.engine.dialect.name
        with self.engine.begin() as conn:
            if dialect_name == "sqlite":
                conn.execute(text("PRAGMA query_only = ON"))
            elif dialect_name == "postgresql":
                conn.execute(text("SET TRANSACTION READ ONLY"))
            else:
                raise RuntimeError(f"read_only_connection is unsupported for backend '{dialect_name}'")
            try:
                yield conn
            finally:
                if dialect_name == "sqlite":
                    conn.execute(text("PRAGMA query_only = OFF"))
