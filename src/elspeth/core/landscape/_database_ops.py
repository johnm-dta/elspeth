"""Database operation helpers to reduce boilerplate in recorder.

Consolidates the repeated `with self._db.connection() as conn:` pattern.
"""

from typing import TYPE_CHECKING, Any

from sqlalchemy import Executable
from sqlalchemy.engine import Row
from sqlalchemy.exc import SQLAlchemyError

from elspeth.core.landscape.errors import LandscapeRecordError

if TYPE_CHECKING:
    from elspeth.core.landscape.database import LandscapeDB


class ReadOnlyDatabaseOps:
    """Helper for read-only database operations.

    Uses the database's read-only connection path so query helpers cannot
    mutate the audit store, even if a caller passes a write-capable statement.
    """

    def __init__(self, db: "LandscapeDB") -> None:
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
            raise LandscapeRecordError(f"execute_fetchone failed — database rejected audit query: {type(exc).__name__}: {exc}") from exc

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
            raise LandscapeRecordError(f"execute_fetchall failed — database rejected audit query: {type(exc).__name__}: {exc}") from exc


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
            with self._db.connection() as conn:
                result = conn.execute(stmt)
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"execute_insert failed{detail} — database rejected audit write: {type(exc).__name__}: {exc}"
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
            with self._db.connection() as conn:
                result = conn.execute(stmt)
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"execute_update failed{detail} — database rejected audit update: {type(exc).__name__}: {exc}"
            ) from exc
        if result.rowcount == 0:
            raise LandscapeRecordError(f"execute_update: zero rows affected{detail} — target row does not exist (audit data corruption)")
