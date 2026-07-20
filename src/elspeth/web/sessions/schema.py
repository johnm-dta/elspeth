"""Current-schema bootstrap for the web session database.

ELSPETH is pre-release, so the web session database has no migration
pathway. Fresh databases are created directly from the current
SQLAlchemy metadata. Existing databases must already match that current
schema; stale local/runtime files should be deleted and recreated.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, NoReturn

from sqlalchemy import Connection, Engine, inspect, text
from sqlalchemy.engine.reflection import Inspector

from elspeth.core.schema_identity import (
    SCHEMA_IDENTITY_TABLE_NAME,
    insert_schema_identity,
    read_schema_identities,
    schema_identity_mismatch,
)
from elspeth.core.schema_shape import collect_metadata_shape_issues
from elspeth.web.sessions.models import (
    SESSION_DB_APPLICATION_ID,
    SESSION_SCHEMA_EPOCH,
    metadata,
    schema_identity_table,
)

_SQLITE_INTERNAL_TABLES: frozenset[str] = frozenset({"sqlite_sequence"})
_SESSION_METADATA_CREATE_LOCK = Lock()

# Required audit triggers. Both supported database dialects install these
# stable trigger names to enforce invariants
# that cannot be expressed as table CHECK constraints:
#
# * ``trg_interpretation_events_immutable_resolved`` — resolved
#   ``interpretation_events`` rows cannot be retroactively edited.
# * ``trg_interpretation_events_no_delete_resolved`` — resolved
#   ``interpretation_events`` rows cannot be deleted directly, while
#   unresolved PENDING rows remain deletable for orphan recovery and
#   whole-session archival can still cascade session-scoped rows.
# * ``trg_composer_completion_events_no_update`` — Phase 6A: every
#   completion-gesture row (mark_ready_for_review, export_yaml) is a
#   permanent audit fact. Unconditional UPDATE ABORT, no PENDING-row
#   carve-out (completion events have no recovery path).
# * ``trg_composer_completion_events_no_delete`` — same posture for DELETE.
#   Both triggers ship from day 1 — completion events have no recovery
#   path, unlike the ``interpretation_events`` PENDING-row carve-out.
# * ``trg_chat_messages_immutable_content`` — ``chat_messages.content`` is
#   append-only once written.
# * ``trg_chat_messages_no_delete`` — ``chat_messages`` rows cannot be
#   deleted directly because they anchor the ``blobs.created_from_message_id``
#   lineage walk. Whole-session archival remains the bounded lifecycle
#   purge path and is implemented by deleting the owning ``sessions`` row,
#   allowing schema-owned cascades to remove session-scoped children.
# * ``trg_guided_operations_terminal_immutable`` — a reservation may renew or
#   settle while in progress, but a completed/failed replay result cannot be
#   changed after the terminal transition.
# * ``trg_guided_operation_events_no_update`` / ``no_delete`` — lease,
#   takeover, and settlement evidence is append-only; only the owning-session
#   lifecycle cascade may remove it.
# * ``trg_guided_operation_admission_blocks_no_update`` / ``no_delete`` — a
#   negative admission decision is append-only; only the owning-session
#   lifecycle cascade may remove it.
# * the admission coexistence triggers reject either insertion order, plus an
#   operation identity update, so a block and operation can never share one
#   session-scoped operation id.
#
# The validator catches the case where schema bootstrap succeeded but the
# trigger DDL failed silently (e.g., if the DDL event listener was removed
# or reordered). Live validation against each dialect's system catalogue
# means a dropped or disabled trigger is caught at startup, not at the first
# attempt to mutate the protected rows.
_REQUIRED_AUDIT_TRIGGERS: frozenset[str] = frozenset(
    {
        "trg_interpretation_events_immutable_resolved",
        "trg_interpretation_events_no_delete_resolved",
        "trg_composer_completion_events_no_update",
        "trg_composer_completion_events_no_delete",
        "trg_chat_messages_immutable_content",
        "trg_chat_messages_no_delete",
        "trg_guided_operations_terminal_immutable",
        "trg_guided_operation_events_no_update",
        "trg_guided_operation_events_no_delete",
        "trg_guided_operation_admission_blocks_no_update",
        "trg_guided_operation_admission_blocks_no_delete",
        "trg_guided_operation_admission_blocks_reject_existing_operation",
        "trg_guided_operations_reject_admission_block_insert",
        "trg_guided_operations_reject_admission_block_update",
    }
)


class SessionSchemaError(RuntimeError):
    """Raised when a session database does not match the current schema epoch.

    Actionable instruction is always the same: delete the session DB file and
    restart. Pre-release ELSPETH has no migration pathway for session DBs.
    """


def _create_session_tables(bind: Engine | Connection, *, checkfirst: bool = True) -> None:
    """Create session tables without leaking SQLAlchemy's dialect-local state.

    PostgreSQL breaks the session metadata's foreign-key cycles with ALTER
    statements and temporarily annotates those constraints via the private
    ``_create_rule`` attribute. SQLAlchemy leaves those annotations on the
    shared metadata object after ``create_all``; a later SQLite bootstrap in
    the same process would then omit the affected inline foreign keys.

    Snapshotting and restoring the exact prior values keeps the module-level
    metadata reusable across database dialects without weakening either
    schema.
    """
    with _SESSION_METADATA_CREATE_LOCK:
        missing = object()
        create_rules: list[tuple[Any, object]] = []
        for table in metadata.tables.values():
            for constraint in table.constraints:
                create_rules.append((constraint, getattr(constraint, "_create_rule", missing)))

        try:
            metadata.create_all(bind=bind, checkfirst=checkfirst)
        finally:
            for constraint, create_rule in create_rules:
                if create_rule is missing:
                    if hasattr(constraint, "_create_rule"):
                        del constraint._create_rule
                else:
                    constraint._create_rule = create_rule


def initialize_session_schema(engine: Engine) -> None:
    """Create or validate the current web session database schema.

    Empty databases are initialized from ``sessions.models.metadata``.
    Non-empty databases are validated in place and are never altered.
    This keeps the pre-release policy mechanical: delete stale runtime
    DB files instead of carrying migration code or repair branches.
    """

    inspector = inspect(engine)
    existing_tables = _user_tables(inspector)
    if not existing_tables:
        # Fresh DB: build the tables, THEN stamp the schema-version
        # sentinels. Stamping before ``create_all`` would let a partial
        # failure (e.g. SQL syntax error in a model) leave a DB on disk
        # that claims the current epoch but has no tables. Stamping after
        # ``create_all`` succeeds binds the sentinel to "schema actually
        # present at this epoch."
        _create_session_tables(engine)
        _stamp_schema_sentinels(engine)
        _validate_current_schema(engine)
        return

    # Existing DB: enforce the schema-version guard BEFORE the deeper
    # column/FK/constraint validation. A stale DB with the wrong
    # ``user_version`` is the high-frequency operator failure mode
    # (forgot to delete the DB across a schema bump); surfacing that
    # specific cause with the actionable "Delete the session DB file
    # and restart" message is materially more useful than the column-
    # mismatch error the validator would otherwise produce downstream.
    _assert_schema_sentinels(engine)
    _validate_current_schema(engine)


def probe_current_schema(bind: Engine | Connection) -> bool:
    """Return whether ``bind`` already carries the current session schema.

    This is a read-only probe: it neither creates objects nor stamps schema
    sentinels.  A supplied ``Connection`` is used directly, so callers holding
    the only connection in a bounded pool cannot deadlock on a second checkout.
    """

    supplied_connection = bind if isinstance(bind, Connection) else None
    supplied_connection_was_idle = supplied_connection is not None and not supplied_connection.in_transaction()
    try:
        _assert_schema_sentinels(bind)
        _validate_current_schema(bind)
    except SessionSchemaError:
        return False
    finally:
        if supplied_connection_was_idle and supplied_connection is not None and supplied_connection.in_transaction():
            supplied_connection.rollback()
    return True


def _stamp_schema_sentinels(bind: Engine | Connection) -> None:
    """Write the SQLite PRAGMAs and cross-dialect identity row after creation.

    The identity insert is intentionally not an upsert: this function is for a
    freshly-created schema, and any pre-existing row is evidence that creation
    ordering or target selection is wrong.
    """
    if isinstance(bind, Connection):
        _stamp_schema_sentinels_on_connection(bind)
    else:
        with bind.begin() as connection:
            _stamp_schema_sentinels_on_connection(connection)


def _stamp_schema_sentinels_on_connection(connection: Connection) -> None:
    if SCHEMA_IDENTITY_TABLE_NAME not in _user_tables(inspect(connection)):
        raise SessionSchemaError("Session database initialization did not produce the current schema.")
    if connection.dialect.name == "sqlite":
        connection.execute(text(f"PRAGMA application_id = {SESSION_DB_APPLICATION_ID}"))
        connection.execute(text(f"PRAGMA user_version = {SESSION_SCHEMA_EPOCH}"))
    insert_schema_identity(connection, schema_identity_table, store_kind="session", schema_epoch=SESSION_SCHEMA_EPOCH)


def _assert_schema_sentinels(bind: Engine | Connection) -> None:
    """Crash with an actionable message if the session DB schema-version
    sentinels do not match the values this build expects.

    Mechanical enforcement of the operator-delete-DB policy: without
    this guard, a stale DB silently fails in obscure SQLAlchemy errors
    the first time a new code path touches a column/table that does not
    exist. With it, the operator gets a precise instruction to delete
    the DB file and restart.

    SQLite keeps its header-level ``application_id`` / ``user_version`` proof.
    Both SQLite and PostgreSQL additionally require exactly one
    ``elspeth_schema_identity`` row with the expected application, store kind,
    and epoch. This catches semantic-only schema bumps on PostgreSQL, where no
    PRAGMA equivalent exists.

    - Wrong SQLite ``application_id`` (non-zero, non-ELSP): the configured DB
      file belongs to some other application entirely. We refuse to
      touch it.
    - Wrong SQLite ``user_version`` (non-zero, not the current epoch): the
      file is ours but predates this release's schema. The operator
      must delete it (pre-release ELSPETH has no migration pathway).
    - Missing, malformed, or mismatched identity rows fail closed on either
      dialect.

    The zero values are accepted in both cases because a brand-new
    SQLite file starts with ``application_id=0`` and ``user_version=0``;
    that state is indistinguishable from "empty DB about to be
    initialised" and is handled by the fresh-DB branch upstream.
    """
    if isinstance(bind, Connection):
        _assert_schema_sentinels_on_connection(bind)
    else:
        with bind.connect() as connection:
            _assert_schema_sentinels_on_connection(connection)


def _assert_schema_sentinels_on_connection(connection: Connection) -> None:
    inspector = inspect(connection)
    tables = _user_tables(inspector)

    if connection.dialect.name == "sqlite":
        app_id = connection.execute(text("PRAGMA application_id")).scalar_one()
        user_ver = connection.execute(text("PRAGMA user_version")).scalar_one()
        if app_id not in {0, SESSION_DB_APPLICATION_ID}:
            raise SessionSchemaError(
                f"Session DB has unexpected application_id={app_id:#010x}. "
                f"Expected {SESSION_DB_APPLICATION_ID:#010x} (ELSP) or 0 (new database). "
                f"This SQLite file does not belong to ELSPETH. "
                f"Delete the session DB file and restart."
            )
        if user_ver not in {0, SESSION_SCHEMA_EPOCH}:
            raise SessionSchemaError(
                f"Session DB schema version {user_ver} does not match "
                f"SESSION_SCHEMA_EPOCH={SESSION_SCHEMA_EPOCH}. Pre-release ELSPETH "
                f"does not migrate session databases. "
                f"Delete the session DB file and restart."
            )

    if SCHEMA_IDENTITY_TABLE_NAME not in tables:
        if tables:
            raise SessionSchemaError(
                f"Session DB is missing {SCHEMA_IDENTITY_TABLE_NAME} for "
                f"SESSION_SCHEMA_EPOCH={SESSION_SCHEMA_EPOCH}. Pre-release ELSPETH "
                "does not migrate session databases. Delete the session DB file and restart."
            )
        return

    # Validate the live identity-table shape BEFORE selecting from it:
    # ``read_schema_identities`` selects the declared model columns, so a
    # missing or renamed column would otherwise leak a raw SQLAlchemy
    # OperationalError instead of the actionable delete-and-restart error
    # (elspeth-5cf1ca2852). Column presence is all the read requires; type
    # drift is classified by ``read_schema_identities`` itself and full-shape
    # drift by the downstream schema validator.
    live_identity_columns = {column["name"] for column in inspector.get_columns(SCHEMA_IDENTITY_TABLE_NAME)}
    missing_identity_columns = {column.name for column in schema_identity_table.columns} - live_identity_columns
    if missing_identity_columns:
        raise SessionSchemaError(
            f"Session DB {SCHEMA_IDENTITY_TABLE_NAME} table is missing column(s) "
            f"{', '.join(sorted(missing_identity_columns))} for "
            f"SESSION_SCHEMA_EPOCH={SESSION_SCHEMA_EPOCH}. Pre-release ELSPETH "
            "does not migrate session databases. Delete the session DB file and restart."
        )

    rows = read_schema_identities(connection, schema_identity_table)
    mismatch = schema_identity_mismatch(rows, store_kind="session", schema_epoch=SESSION_SCHEMA_EPOCH)
    if mismatch is not None:
        raise SessionSchemaError(
            f"Session DB schema identity mismatch ({mismatch}) for "
            f"SESSION_SCHEMA_EPOCH={SESSION_SCHEMA_EPOCH}. Pre-release ELSPETH "
            "does not migrate session databases. Delete the session DB file and restart."
        )


def _user_tables(inspector: Inspector) -> frozenset[str]:
    return frozenset(name for name in inspector.get_table_names() if name not in _SQLITE_INTERNAL_TABLES)


def _validate_current_schema(bind: Engine | Connection) -> None:
    # Static partial-index symmetry check fires before any inspector-driven
    # validation. Catches the elspeth-obs-2ef48619d5 drift class at app-
    # start time (or import time when called from a hot-reload loop), which
    # is strictly better than waiting for a per-test fresh-DB creation.
    # This remains a distinct model-layer guard even though the shared
    # collector now also compares each live index predicate: a one-sided
    # sqlite_where/postgresql_where declaration cannot be discovered by
    # inspecting only the current runtime dialect.
    _validate_partial_index_dialect_symmetry()

    inspector = inspect(bind)
    expected_tables = frozenset(metadata.tables)
    actual_tables = _user_tables(inspector)
    if actual_tables != expected_tables:
        _schema_error(
            "table set mismatch",
            expected=sorted(expected_tables),
            actual=sorted(actual_tables),
        )

    issues = collect_metadata_shape_issues(
        inspector,
        metadata,
        dialect=bind.dialect,
        present_tables=actual_tables,
    )
    if issues:
        first = issues[0]
        _schema_error(first.subject, expected=first.expected, actual=first.actual)

    _validate_required_triggers(bind)


def _validate_required_triggers(bind: Engine | Connection) -> None:
    """Confirm the required audit triggers are present in the live DB.

    Catches the case where ``metadata.create_all`` succeeded but a trigger
    listener was bypassed or removed, OR where a trigger on an existing DB
    was dropped (manually or by a faulty migration). Without this check, a
    silently-missing trigger means the audit invariants it enforces are not
    actually being enforced — a Tier-1 audit integrity failure that would
    otherwise only surface the next time someone tried to mutate a
    protected row (which may be never on a quiescent DB).
    """
    if bind.dialect.name == "sqlite":
        query = text("SELECT name FROM sqlite_master WHERE type='trigger'")
    elif bind.dialect.name == "postgresql":
        query = text(
            """
            SELECT trigger.tgname
            FROM pg_catalog.pg_trigger AS trigger
            JOIN pg_catalog.pg_class AS relation ON relation.oid = trigger.tgrelid
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE NOT trigger.tgisinternal
              AND trigger.tgenabled IN ('O', 'A')
              AND namespace.nspname = current_schema()
              AND (
                (relation.relname = 'interpretation_events' AND trigger.tgname IN (
                  'trg_interpretation_events_immutable_resolved',
                  'trg_interpretation_events_no_delete_resolved'
                ))
                OR (relation.relname = 'composer_completion_events' AND trigger.tgname IN (
                  'trg_composer_completion_events_no_update',
                  'trg_composer_completion_events_no_delete'
                ))
                OR (relation.relname = 'chat_messages' AND trigger.tgname IN (
                  'trg_chat_messages_immutable_content',
                  'trg_chat_messages_no_delete'
                ))
                OR (relation.relname = 'guided_operation_events' AND trigger.tgname IN (
                  'trg_guided_operation_events_no_update',
                  'trg_guided_operation_events_no_delete'
                ))
                OR (relation.relname = 'guided_operation_admission_blocks' AND trigger.tgname IN (
                  'trg_guided_operation_admission_blocks_no_update',
                  'trg_guided_operation_admission_blocks_no_delete',
                  'trg_guided_operation_admission_blocks_reject_existing_operation'
                ))
                OR (relation.relname = 'guided_operations' AND trigger.tgname IN (
                  'trg_guided_operations_terminal_immutable',
                  'trg_guided_operations_reject_admission_block_insert',
                  'trg_guided_operations_reject_admission_block_update'
                ))
              )
            """
        )
    else:
        _schema_error(
            "audit trigger validation unsupported dialect",
            expected=["postgresql", "sqlite"],
            actual=[bind.dialect.name],
        )

    if isinstance(bind, Connection):
        present = {str(row[0]) for row in bind.execute(query)}
    else:
        with bind.connect() as connection:
            present = {str(row[0]) for row in connection.execute(query)}
    missing = _REQUIRED_AUDIT_TRIGGERS - present
    if missing:
        _schema_error(
            "missing audit trigger(s)",
            expected=sorted(_REQUIRED_AUDIT_TRIGGERS),
            actual=sorted(present),
        )


def _validate_partial_index_dialect_symmetry() -> None:
    """Enforce the partial-index dialect-symmetry contract at the model layer.

    For every ``Index`` whose ``dialect_options`` contain a ``where`` predicate
    under any dialect, BOTH ``sqlite_where`` AND ``postgresql_where`` MUST be
    set, AND they MUST compile to the same SQL text under their respective
    dialects.

    Closes elspeth-obs-2ef48619d5: the pre-fix validator only compared
    index NAMES, so an index declared with ``sqlite_where=`` only (or with
    a different ``postgresql_where=`` predicate) would pass schema
    validation while silently enforcing different invariants across
    dialects. Concrete prior incident: ``uq_runs_one_active_per_session``
    originally had only ``sqlite_where=``, which on Postgres silently
    became a non-partial unique index — over-restricting "at most one
    ACTIVE run per session" to "at most one run per session ever."

    This static model check complements the shared runtime shape collector:
    each runtime can validate its live predicate, while only this check can
    prove both dialect declarations remain paired before either DDL is emitted.
    """
    from sqlalchemy.dialects import postgresql, sqlite

    sqlite_dialect = sqlite.dialect()
    postgresql_dialect = postgresql.dialect()  # type: ignore[no-untyped-call]

    for table_name, table in metadata.tables.items():
        for index in table.indexes:
            if index.name is None:
                continue
            sqlite_where = _dialect_where(index, "sqlite")
            postgres_where = _dialect_where(index, "postgresql")

            if sqlite_where is None and postgres_where is None:
                continue

            # One-sided declaration is the original drift class.
            if sqlite_where is None or postgres_where is None:
                _schema_error(
                    f"{table_name}.{index.name} partial-index dialect asymmetry",
                    expected="both sqlite_where= and postgresql_where= set",
                    actual=f"sqlite_where={'set' if sqlite_where is not None else 'unset'}, "
                    f"postgresql_where={'set' if postgres_where is not None else 'unset'}",
                )

            # Both set — compile under their respective dialects and
            # compare the literal-bound SQL text. Mismatched predicates
            # are the second drift class: silently divergent invariants
            # under the same name.
            sqlite_text = str(sqlite_where.compile(dialect=sqlite_dialect, compile_kwargs={"literal_binds": True}))
            postgres_text = str(postgres_where.compile(dialect=postgresql_dialect, compile_kwargs={"literal_binds": True}))
            if sqlite_text != postgres_text:
                _schema_error(
                    f"{table_name}.{index.name} partial-index WHERE clause text diverges between dialects",
                    expected=sqlite_text,
                    actual=postgres_text,
                )


def _dialect_where(index: Any, dialect_key: str) -> Any:
    """Return the ``where`` predicate for ``dialect_key`` on ``index``, or
    ``None`` when the index declared no kwargs for that dialect or the
    dialect declared no ``where`` predicate.

    Direct dictionary access rather than ``.get()`` to satisfy the
    Tier-1 defensive-programming gate (``enforce_tier_model``). The
    DialectKWArgs container's ``__contains__`` semantics are the
    canonical absence test; ``in`` is the offensive equivalent of
    ``.get(..., None)`` here.
    """
    if dialect_key not in index.dialect_options:
        return None
    options = index.dialect_options[dialect_key]
    if "where" not in options:
        return None
    return options["where"]


def _schema_error(detail: str, *, expected: object | None = None, actual: object | None = None) -> NoReturn:
    message = (
        f"Session database schema does not match SESSION_SCHEMA_EPOCH={SESSION_SCHEMA_EPOCH}. "
        "Delete the old session database and restart; pre-release ELSPETH "
        f"does not migrate web session databases. Detail: {detail}."
    )
    if expected is not None:
        message = f"{message} Expected: {expected!r}."
    if actual is not None:
        message = f"{message} Found: {actual!r}."
    raise SessionSchemaError(message)
