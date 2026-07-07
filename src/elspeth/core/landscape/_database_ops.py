"""Database operation helpers to reduce recorder boilerplate.

Consolidates read-only and write connection management for simple statements.
"""

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Any, Protocol

from sqlalchemy import Executable
from sqlalchemy.engine import Connection, Row
from sqlalchemy.exc import SQLAlchemyError

from elspeth.core.landscape.errors import LandscapeRecordError, LandscapeRecordNotFoundError


def _safe_database_error_message(
    *,
    operation: str,
    action: str,
    exc: SQLAlchemyError,
    context: str = "",
) -> str:
    detail = f" ({context})" if context else ""
    return f"{operation} failed{detail} — database rejected audit {action}: {type(exc).__name__}"


class DatabaseOpsConnectionProvider(Protocol):
    """Connection surface required by database operation helpers."""

    def read_only_connection(self) -> AbstractContextManager[Connection]:
        raise NotImplementedError

    def write_connection(self) -> AbstractContextManager[Connection]:
        raise NotImplementedError


class ReadOnlyDatabaseOps:
    """Helper for read-only database operations.

    Uses the database's read-only connection path so query helpers cannot
    mutate the audit store, even if a caller passes a write-capable statement.
    """

    def __init__(self, db: DatabaseOpsConnectionProvider) -> None:
        self._db = db

    def execute_fetchone(self, query: Executable) -> Row[Any] | None:
        """Execute a single-row query.

        Returns the row when exactly one matches, ``None`` when no rows match,
        and raises ``LandscapeRecordError`` when multiple rows match.
        """
        try:
            with self._db.read_only_connection() as conn:
                result = conn.execute(query)
                rows = result.fetchmany(2)
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(_safe_database_error_message(operation="execute_fetchone", action="query", exc=exc)) from exc

        if len(rows) > 1:
            raise LandscapeRecordError("execute_fetchone matched multiple rows — single-row audit query is ambiguous")
        if not rows:
            return None
        return rows[0]

    def execute_fetchall(self, query: Executable) -> list[Row[Any]]:
        """Execute a read-only query and return all rows."""
        try:
            with self._db.read_only_connection() as conn:
                result = conn.execute(query)
                return list(result.fetchall())
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(_safe_database_error_message(operation="execute_fetchall", action="query", exc=exc)) from exc

    def execute_fetchall_many(self, queries: Sequence[Executable]) -> list[list[Row[Any]]]:
        """Execute read-only queries through one read snapshot."""
        try:
            with self._db.read_only_connection() as conn:
                if conn.dialect.name == "sqlite" and getattr(self._db, "is_read_only", False):
                    # Read-only engines keep stock pysqlite autocommit for ordinary
                    # inspectors; this multi-query boundary needs one stable snapshot.
                    conn.exec_driver_sql("BEGIN")
                return [list(conn.execute(query).fetchall()) for query in queries]
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(_safe_database_error_message(operation="execute_fetchall_many", action="query", exc=exc)) from exc


class DatabaseOps(ReadOnlyDatabaseOps):
    """Helper for common database operations.

    Reduces boilerplate in recorder methods by centralizing
    connection management.
    """

    def execute_insert(self, stmt: Executable, *, context: str = "") -> None:
        """Execute insert statement.

        Args:
            stmt: SQLAlchemy insert statement
            context: Optional context string for error messages (e.g., table/operation name)

        Raises:
            LandscapeRecordError: If the write fails or zero rows are affected.
        """
        detail = f" ({context})" if context else ""
        try:
            with self._db.write_connection() as conn:
                result = conn.execute(stmt)
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                _safe_database_error_message(operation="execute_insert", action="write", exc=exc, context=context)
            ) from exc
        if result.rowcount == 0:
            raise LandscapeRecordError(
                f"execute_insert: zero rows affected{detail} — audit write failed (missing parent row or constraint violation)"
            )

    def execute_update(self, stmt: Executable, *, context: str = "") -> None:
        """Execute update statement.

        Args:
            stmt: SQLAlchemy update statement
            context: Optional context string for error messages (e.g., table/operation name)

        Raises:
            LandscapeRecordError: If the write fails or zero rows are affected.
        """
        detail = f" ({context})" if context else ""
        try:
            with self._db.write_connection() as conn:
                result = conn.execute(stmt)
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                _safe_database_error_message(operation="execute_update", action="update", exc=exc, context=context)
            ) from exc
        if result.rowcount == 0:
            raise LandscapeRecordNotFoundError(
                f"execute_update: zero rows affected{detail} — target row does not exist (audit data corruption)"
            )
