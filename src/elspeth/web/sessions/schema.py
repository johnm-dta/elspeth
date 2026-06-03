"""Current-schema bootstrap for the web session database.

ELSPETH is pre-release, so the web session database has no migration
pathway. Fresh databases are created directly from the current
SQLAlchemy metadata. Existing databases must already match that current
schema; stale local/runtime files should be deleted and recreated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn, cast

from sqlalchemy import Engine, inspect, text
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.schema import CheckConstraint, ForeignKeyConstraint, Table, UniqueConstraint

from elspeth.web.sessions.models import (
    SESSION_DB_APPLICATION_ID,
    SESSION_SCHEMA_EPOCH,
    metadata,
)

_SQLITE_INTERNAL_TABLES: frozenset[str] = frozenset({"sqlite_sequence"})

# Required SQLite triggers. These triggers enforce audit invariants
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
#
# The validator catches the case where schema bootstrap succeeded but the
# trigger DDL failed silently (e.g., if the DDL event listener was removed
# or reordered). Live validation against ``sqlite_master`` so a dropped
# trigger on an existing DB is caught at startup, not at the first attempt
# to mutate the protected rows.
_REQUIRED_SQLITE_TRIGGERS: frozenset[str] = frozenset(
    {
        "trg_interpretation_events_immutable_resolved",
        "trg_interpretation_events_no_delete_resolved",
        "trg_composer_completion_events_no_update",
        "trg_composer_completion_events_no_delete",
        "trg_chat_messages_immutable_content",
        "trg_chat_messages_no_delete",
    }
)


class SessionSchemaError(RuntimeError):
    """Raised when a session database does not match the current schema epoch.

    Actionable instruction is always the same: delete the session DB file and
    restart. Pre-release ELSPETH has no migration pathway for session DBs.
    """


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
        metadata.create_all(engine)
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


def _stamp_schema_sentinels(engine: Engine) -> None:
    """Write SESSION_DB_APPLICATION_ID and SESSION_SCHEMA_EPOCH onto a
    freshly-created session DB.

    Both PRAGMAs are persistent attributes of the SQLite file (stored in
    the header), not per-connection settings, so they only need to be
    written once at create time. The startup validator reads them back
    on every subsequent open.
    """
    if engine.dialect.name != "sqlite":
        return
    with engine.connect() as conn:
        conn.execute(text(f"PRAGMA application_id = {SESSION_DB_APPLICATION_ID}"))
        conn.execute(text(f"PRAGMA user_version = {SESSION_SCHEMA_EPOCH}"))
        conn.commit()


def _assert_schema_sentinels(engine: Engine) -> None:
    """Crash with an actionable message if the session DB schema-version
    sentinels do not match the values this build expects.

    Mechanical enforcement of the operator-delete-DB policy: without
    this guard, a stale DB silently fails in obscure SQLAlchemy errors
    the first time a new code path touches a column/table that does not
    exist. With it, the operator gets a precise instruction to delete
    the DB file and restart.

    Two failure modes are distinguished:

    - Wrong ``application_id`` (non-zero, non-ELSP): the configured DB
      file belongs to some other application entirely. We refuse to
      touch it.
    - Wrong ``user_version`` (non-zero, not the current epoch): the
      file is ours but predates this release's schema. The operator
      must delete it (pre-release ELSPETH has no migration pathway).

    The zero values are accepted in both cases because a brand-new
    SQLite file starts with ``application_id=0`` and ``user_version=0``;
    that state is indistinguishable from "empty DB about to be
    initialised" and is handled by the fresh-DB branch upstream.
    """
    if engine.dialect.name != "sqlite":
        return
    with engine.connect() as conn:
        app_id = conn.execute(text("PRAGMA application_id")).scalar_one()
        user_ver = conn.execute(text("PRAGMA user_version")).scalar_one()
    if app_id != 0 and app_id != SESSION_DB_APPLICATION_ID:
        raise SessionSchemaError(
            f"Session DB has unexpected application_id={app_id:#010x}. "
            f"Expected {SESSION_DB_APPLICATION_ID:#010x} (ELSP) or 0 (new database). "
            f"This SQLite file does not belong to ELSPETH. "
            f"Delete the session DB file and restart."
        )
    if user_ver != 0 and user_ver != SESSION_SCHEMA_EPOCH:
        raise SessionSchemaError(
            f"Session DB schema version {user_ver} does not match "
            f"SESSION_SCHEMA_EPOCH={SESSION_SCHEMA_EPOCH}. Pre-release ELSPETH "
            f"does not migrate session databases. "
            f"Delete the session DB file and restart."
        )


def _user_tables(inspector: Inspector) -> frozenset[str]:
    return frozenset(name for name in inspector.get_table_names() if name not in _SQLITE_INTERNAL_TABLES)


def _validate_current_schema(engine: Engine) -> None:
    # Static partial-index symmetry check fires before any inspector-driven
    # validation. Catches the elspeth-obs-2ef48619d5 drift class at app-
    # start time (or import time when called from a hot-reload loop), which
    # is strictly better than waiting for a per-test fresh-DB creation.
    # Runtime cross-dialect text comparison was considered and rejected:
    # the inspector's reported WHERE text diverges from the model's
    # compiled SQL (different key prefixes, qualified vs unqualified
    # column refs, TextClause vs BinaryExpression). The static guard
    # below is the load-bearing defense.
    _validate_partial_index_dialect_symmetry()

    inspector = inspect(engine)
    expected_tables = frozenset(metadata.tables)
    actual_tables = _user_tables(inspector)
    if actual_tables != expected_tables:
        _schema_error(
            "table set mismatch",
            expected=sorted(expected_tables),
            actual=sorted(actual_tables),
        )

    for table_name, table in metadata.tables.items():
        _validate_columns(inspector, table_name, table)
        _validate_foreign_keys(inspector, table_name, table)
        _validate_named_checks(inspector, table_name, table)
        _validate_named_unique_constraints(inspector, table_name, table)
        _validate_named_indexes(inspector, table_name, table)

    _validate_required_triggers(engine)


def _validate_required_triggers(engine: Engine) -> None:
    """Confirm the required SQLite triggers are present in the live DB.

    Catches the case where ``metadata.create_all`` succeeded but a trigger
    listener was bypassed or removed, OR where a trigger on an existing DB
    was dropped (manually or by a faulty migration). Without this check, a
    silently-missing trigger means the audit invariants it enforces are not
    actually being enforced — a Tier-1 audit integrity failure that would
    otherwise only surface the next time someone tried to mutate a
    protected row (which may be never on a quiescent DB).
    """
    if engine.dialect.name != "sqlite":
        return
    with engine.connect() as conn:
        present = {str(row[0]) for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='trigger'"))}
    missing = _REQUIRED_SQLITE_TRIGGERS - present
    if missing:
        _schema_error(
            "missing SQLite trigger(s)",
            expected=sorted(_REQUIRED_SQLITE_TRIGGERS),
            actual=sorted(present),
        )


def _validate_columns(inspector: Inspector, table_name: str, table: Table) -> None:
    inspected_columns = inspector.get_columns(table_name)
    primary_key_columns = {str(column_name) for column_name in inspector.get_pk_constraint(table_name)["constrained_columns"]}
    expected_names = tuple(column.name for column in table.columns)
    actual_names = tuple(str(column["name"]) for column in inspected_columns)
    if actual_names != expected_names:
        _schema_error(
            f"{table_name} column mismatch",
            expected=list(expected_names),
            actual=list(actual_names),
        )

    columns_by_name = {str(column["name"]): column for column in inspected_columns}
    for column in table.columns:
        actual_column = columns_by_name[column.name]
        actual_primary_key = column.name in primary_key_columns
        if actual_primary_key != bool(column.primary_key):
            _schema_error(
                f"{table_name}.{column.name} primary-key mismatch",
                expected=bool(column.primary_key),
                actual=actual_primary_key,
            )

        if not column.primary_key:
            actual_nullable = bool(actual_column["nullable"])
            if actual_nullable != bool(column.nullable):
                _schema_error(
                    f"{table_name}.{column.name} nullable mismatch",
                    expected=bool(column.nullable),
                    actual=actual_nullable,
                )


def _validate_foreign_keys(inspector: Inspector, table_name: str, table: Table) -> None:
    expected = {_expected_foreign_key_shape(constraint) for constraint in table.foreign_key_constraints}
    actual = {_actual_foreign_key_shape(fk) for fk in inspector.get_foreign_keys(table_name)}
    if actual != expected:
        _schema_error(
            f"{table_name} foreign-key mismatch",
            expected=sorted(expected),
            actual=sorted(actual),
        )


def _expected_foreign_key_shape(
    constraint: ForeignKeyConstraint,
) -> tuple[tuple[str, ...], str, tuple[str, ...], str | None]:
    elements = tuple(constraint.elements)
    if not elements:
        _schema_error(f"{constraint.name or '<unnamed>'} has no foreign-key elements")

    referred_table = elements[0].column.table.name
    constrained_columns = tuple(element.parent.name for element in elements)
    referred_columns = tuple(element.column.name for element in elements)
    ondelete = elements[0].ondelete
    return constrained_columns, referred_table, referred_columns, ondelete.lower() if ondelete is not None else None


def _actual_foreign_key_shape(fk: Mapping[str, Any]) -> tuple[tuple[str, ...], str, tuple[str, ...], str | None]:
    raw_options = fk["options"] if "options" in fk else {}
    options = cast("Mapping[str, Any]", raw_options)
    raw_ondelete = options["ondelete"] if "ondelete" in options else None
    ondelete = str(raw_ondelete).lower() if raw_ondelete is not None else None
    return (
        tuple(str(column) for column in fk["constrained_columns"]),
        str(fk["referred_table"]),
        tuple(str(column) for column in fk["referred_columns"]),
        ondelete,
    )


def _validate_named_checks(inspector: Inspector, table_name: str, table: Table) -> None:
    expected = {
        str(constraint.name) for constraint in table.constraints if type(constraint) is CheckConstraint and constraint.name is not None
    }
    actual = {str(constraint["name"]) for constraint in inspector.get_check_constraints(table_name) if constraint["name"] is not None}
    if actual != expected:
        _schema_error(
            f"{table_name} CHECK constraint mismatch",
            expected=sorted(expected),
            actual=sorted(actual),
        )


def _validate_named_unique_constraints(inspector: Inspector, table_name: str, table: Table) -> None:
    """Validate the table's UNIQUE-constraint surface.

    The expected set unions two model-side shapes: ``UniqueConstraint``
    declarations on ``table.constraints`` AND ``Index(name=..., unique=True)``
    declarations on ``table.indexes``. Both are semantically unique
    constraints; SQLAlchemy's choice of declaration shape is purely
    convenience (an ``Index(unique=True)`` accepts ``sqlite_where=`` /
    ``postgresql_where=`` for partial-index predicates while a
    ``UniqueConstraint`` does not). Per elspeth-obs-3ac0c829c5, the
    pre-fix validator only iterated ``table.constraints`` for
    ``UniqueConstraint`` and would silently leave a future
    ``Index(name=..., unique=True)`` unvalidated.

    Symmetric on the actual side: ``inspector.get_unique_constraints``
    surfaces only "true" unique constraints on SQLite (not Index-backed
    uniques), but Postgres surfaces unique-backed indexes in BOTH
    ``get_unique_constraints`` and ``get_indexes``. Unioning
    ``get_indexes`` entries where ``index['unique']`` is set covers the
    SQLite case; ``_validate_named_indexes`` strips the same names from
    its check to avoid double-counting.
    """
    expected = {
        str(constraint.name) for constraint in table.constraints if type(constraint) is UniqueConstraint and constraint.name is not None
    } | {str(index.name) for index in table.indexes if index.unique and index.name is not None}
    actual = {str(constraint["name"]) for constraint in inspector.get_unique_constraints(table_name) if constraint["name"] is not None} | {
        str(index["name"]) for index in inspector.get_indexes(table_name) if index["unique"] and index["name"] is not None
    }
    if actual != expected:
        _schema_error(
            f"{table_name} UNIQUE constraint mismatch",
            expected=sorted(expected),
            actual=sorted(actual),
        )


def _validate_named_indexes(inspector: Inspector, table_name: str, table: Table) -> None:
    # Names already validated as UNIQUE constraints (via UniqueConstraint
    # declaration OR Index(unique=True)) are excluded from the index check
    # so we don't double-count. _validate_named_unique_constraints unions
    # both shapes; _validate_named_indexes is the residual non-unique
    # index validator.
    expected_unique = {
        str(constraint.name) for constraint in table.constraints if type(constraint) is UniqueConstraint and constraint.name is not None
    } | {str(index.name) for index in table.indexes if index.unique and index.name is not None}
    actual_unique = {
        str(constraint["name"]) for constraint in inspector.get_unique_constraints(table_name) if constraint["name"] is not None
    } | {str(index["name"]) for index in inspector.get_indexes(table_name) if index["unique"] and index["name"] is not None}
    expected = {str(index.name) for index in table.indexes if index.name is not None and not index.unique} - expected_unique
    actual = {
        str(index["name"])
        for index in inspector.get_indexes(table_name)
        if index["name"] is not None and not index["unique"] and str(index["name"]) not in actual_unique
    }
    if actual != expected:
        _schema_error(
            f"{table_name} index mismatch",
            expected=sorted(expected),
            actual=sorted(actual),
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

    Static (model-import-time) check rather than runtime inspector
    comparison: the inspector's reported WHERE text diverges from the
    model's compiled SQL across dialects (different key prefixes,
    qualified vs unqualified column refs, TextClause vs BinaryExpression),
    so a runtime cross-dialect text comparison would be brittle without
    proportionate signal. The drift class this guards against is a
    model-side mistake catchable at import time, where a structured
    crash is materially more informative than a per-test fresh-DB
    failure.
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
