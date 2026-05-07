"""Current-schema bootstrap for the web session database.

ELSPETH is pre-release, so the web session database has no migration
pathway. Fresh databases are created directly from the current
SQLAlchemy metadata. Existing databases must already match that current
schema; stale local/runtime files should be deleted and recreated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn, cast

from sqlalchemy import Engine, inspect
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.schema import CheckConstraint, ForeignKeyConstraint, Table, UniqueConstraint

from elspeth.web.sessions.models import metadata

_SQLITE_INTERNAL_TABLES: frozenset[str] = frozenset({"sqlite_sequence"})


class SessionSchemaError(RuntimeError):
    """Raised when a session database does not match the current V0 schema."""


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
        metadata.create_all(engine)
        _validate_current_schema(engine)
        return

    _validate_current_schema(engine)


def _user_tables(inspector: Inspector) -> frozenset[str]:
    return frozenset(name for name in inspector.get_table_names() if name not in _SQLITE_INTERNAL_TABLES)


def _validate_current_schema(engine: Engine) -> None:
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
    expected = {
        str(constraint.name) for constraint in table.constraints if type(constraint) is UniqueConstraint and constraint.name is not None
    }
    actual = {str(constraint["name"]) for constraint in inspector.get_unique_constraints(table_name) if constraint["name"] is not None}
    if actual != expected:
        _schema_error(
            f"{table_name} UNIQUE constraint mismatch",
            expected=sorted(expected),
            actual=sorted(actual),
        )


def _validate_named_indexes(inspector: Inspector, table_name: str, table: Table) -> None:
    expected = {str(index.name) for index in table.indexes if index.name is not None}
    actual = {str(index["name"]) for index in inspector.get_indexes(table_name) if index["name"] is not None}
    if actual != expected:
        _schema_error(
            f"{table_name} index mismatch",
            expected=sorted(expected),
            actual=sorted(actual),
        )


def _schema_error(detail: str, *, expected: object | None = None, actual: object | None = None) -> NoReturn:
    message = (
        "Session database schema does not match the current V0 schema. "
        "Delete the old session database and restart; pre-release ELSPETH "
        f"does not migrate web session databases. Detail: {detail}."
    )
    if expected is not None:
        message = f"{message} Expected: {expected!r}."
    if actual is not None:
        message = f"{message} Found: {actual!r}."
    raise SessionSchemaError(message)
