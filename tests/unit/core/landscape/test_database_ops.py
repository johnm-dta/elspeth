"""Tests for DatabaseOps — the Tier-1 write guard.

Every audit write must succeed.  execute_insert and execute_update raise
AuditIntegrityError when rowcount == 0, which is the enforcement mechanism
for that invariant.  These tests verify all four methods against a real
in-memory SQLite database; no mocks.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import event

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape._database_ops import DatabaseOps, ReadOnlyDatabaseOps
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordNotFoundError
from elspeth.core.landscape.schema import auth_events_table


@pytest.fixture
def ldb() -> LandscapeDB:
    """In-memory SQLite database with a simple test table."""
    db = LandscapeDB.in_memory()
    metadata = sa.MetaData()
    sa.Table(
        "test_rows",
        metadata,
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("value", sa.String),
    )
    metadata.create_all(db.engine)
    return db


@pytest.fixture
def test_table(ldb: LandscapeDB) -> sa.Table:
    """Return the test_rows Table object (reflected so it carries its columns)."""
    meta = sa.MetaData()
    meta.reflect(bind=ldb.engine, only=["test_rows"])
    return meta.tables["test_rows"]


@pytest.fixture
def ops(ldb: LandscapeDB) -> DatabaseOps:
    """DatabaseOps wired to the in-memory database."""
    return DatabaseOps(ldb)


class TestExecuteFetchone:
    """execute_fetchone returns a row when one exists, None when not found."""

    def test_returns_row_when_found(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        ops.execute_insert(test_table.insert().values(id="r1", value="hello"))
        row = ops.execute_fetchone(test_table.select().where(test_table.c.id == "r1"))
        assert row is not None
        assert row.id == "r1"
        assert row.value == "hello"

    def test_returns_none_when_not_found(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        row = ops.execute_fetchone(test_table.select().where(test_table.c.id == "missing"))
        assert row is None

    def test_raises_on_multiple_rows(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        ops.execute_insert(test_table.insert().values(id="a", value="first"))
        ops.execute_insert(test_table.insert().values(id="b", value="second"))
        with pytest.raises(AuditIntegrityError, match=r"multiple rows|ambiguous"):
            ops.execute_fetchone(test_table.select().order_by(test_table.c.id))


class TestExecuteFetchall:
    """execute_fetchall returns all matching rows, empty list when none."""

    def test_returns_all_rows(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        ops.execute_insert(test_table.insert().values(id="x", value="one"))
        ops.execute_insert(test_table.insert().values(id="y", value="two"))
        rows = ops.execute_fetchall(test_table.select().order_by(test_table.c.id))
        assert len(rows) == 2
        assert rows[0].id == "x"
        assert rows[1].id == "y"

    def test_returns_empty_list_when_no_rows(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        rows = ops.execute_fetchall(test_table.select())
        assert rows == []

    def test_returns_list_type(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        rows = ops.execute_fetchall(test_table.select())
        assert isinstance(rows, list)

    def test_rejects_write_statement_even_with_returning(self, tmp_path: Path) -> None:
        """Fetch helpers must execute on a read-only connection."""
        db = LandscapeDB.from_url(f"sqlite:///{tmp_path / 'test.db'}")
        metadata = sa.MetaData()
        table = sa.Table(
            "test_rows",
            metadata,
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("value", sa.String),
        )
        metadata.create_all(db.engine)
        ops = DatabaseOps(db)
        ops.execute_insert(table.insert().values(id="a", value="before"))

        with pytest.raises(AuditIntegrityError, match=r"read.?only|execute_fetchall failed|database rejected"):
            ops.execute_fetchall(table.update().where(table.c.id == "a").values(value="after").returning(table.c.id, table.c.value))

        row = ops.execute_fetchone(table.select().where(table.c.id == "a"))
        assert row is not None
        assert row.value == "before"
        db.close()

    def test_fetchall_many_read_only_handle_uses_one_sqlite_snapshot(self, tmp_path: Path) -> None:
        """Multi-query reads on file-backed read-only handles must be snapshot-consistent."""
        db_path = tmp_path / "test.db"
        writable = LandscapeDB.from_url(f"sqlite:///{db_path}")
        table = auth_events_table
        now = datetime.now(UTC)
        with writable.write_connection() as conn:
            conn.execute(
                table.insert(),
                [
                    {
                        "event_id": "first",
                        "occurred_at": now,
                        "event_type": "login",
                        "outcome": "success",
                        "provider": "local",
                        "username": "first",
                        "metadata_json": "{}",
                    },
                    {
                        "event_id": "second",
                        "occurred_at": now,
                        "event_type": "login",
                        "outcome": "success",
                        "provider": "local",
                        "username": "before",
                        "metadata_json": "{}",
                    },
                ],
            )

        read_only = LandscapeDB.from_url(f"sqlite:///{db_path}", read_only=True, create_tables=False)
        mutation_count = 0

        @event.listens_for(read_only.engine, "after_cursor_execute")
        def _mutate_after_first_select(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
            nonlocal mutation_count
            if mutation_count or "auth_events" not in statement:
                return
            mutation_count += 1
            with writable.write_connection() as write_conn:
                write_conn.execute(table.update().where(table.c.event_id == "second").values(username="after"))

        try:
            rows_by_query = ReadOnlyDatabaseOps(read_only).execute_fetchall_many(
                [
                    table.select().where(table.c.event_id == "first"),
                    table.select().where(table.c.event_id == "second"),
                ]
            )
        finally:
            event.remove(read_only.engine, "after_cursor_execute", _mutate_after_first_select)
            read_only.close()

        assert mutation_count == 1
        assert rows_by_query[0][0].username == "first"
        assert rows_by_query[1][0].username == "before"
        with writable.read_only_connection() as conn:
            assert conn.execute(table.select().where(table.c.event_id == "second")).one().username == "after"
        writable.close()

    def test_fetchall_many_requires_read_only_flag_on_provider(self, ldb: LandscapeDB, test_table: sa.Table) -> None:
        """Snapshot setup must not default a malformed provider to writable semantics."""

        class _MissingReadOnlyFlagProvider:
            def read_only_connection(self):  # type: ignore[no-untyped-def]
                return ldb.read_only_connection()

            def write_connection(self):  # type: ignore[no-untyped-def]
                return ldb.write_connection()

        with pytest.raises(AttributeError, match="is_read_only"):
            ReadOnlyDatabaseOps(_MissingReadOnlyFlagProvider()).execute_fetchall_many([test_table.select()])


class TestDatabaseOpsErrorScrubbing:
    """DatabaseOps errors must not echo SQLAlchemy statement text or bound values."""

    SENTINEL = "SECRET_BOUND_VALUE_SHOULD_NOT_LEAK"

    def _assert_safe_message(self, exc: BaseException, *, operation: str, context: str | None = None) -> None:
        message = str(exc)
        assert operation in message
        assert "OperationalError" in message
        if context is not None:
            assert context in message
        assert self.SENTINEL not in message
        assert "parameters" not in message
        assert "missing_table" not in message

    def test_fetchone_error_message_omits_sqlalchemy_statement_and_bound_values(self, ops: DatabaseOps) -> None:
        with pytest.raises(AuditIntegrityError) as exc_info:
            ops.execute_fetchone(sa.text("SELECT * FROM missing_table WHERE value = :value").bindparams(value=self.SENTINEL))

        self._assert_safe_message(exc_info.value, operation="execute_fetchone")

    def test_fetchall_error_message_omits_sqlalchemy_statement_and_bound_values(self, ops: DatabaseOps) -> None:
        with pytest.raises(AuditIntegrityError) as exc_info:
            ops.execute_fetchall(sa.text("SELECT * FROM missing_table WHERE value = :value").bindparams(value=self.SENTINEL))

        self._assert_safe_message(exc_info.value, operation="execute_fetchall")

    def test_insert_error_message_omits_sqlalchemy_statement_and_bound_values(self, ops: DatabaseOps) -> None:
        with pytest.raises(AuditIntegrityError) as exc_info:
            ops.execute_insert(
                sa.text("INSERT INTO missing_table (value) VALUES (:value)").bindparams(value=self.SENTINEL),
                context="insert sensitive audit row",
            )

        self._assert_safe_message(exc_info.value, operation="execute_insert", context="insert sensitive audit row")

    def test_update_error_message_omits_sqlalchemy_statement_and_bound_values(self, ops: DatabaseOps) -> None:
        with pytest.raises(AuditIntegrityError) as exc_info:
            ops.execute_update(
                sa.text("UPDATE missing_table SET value = :value").bindparams(value=self.SENTINEL),
                context="update sensitive audit row",
            )

        self._assert_safe_message(exc_info.value, operation="execute_update", context="update sensitive audit row")


class TestExecuteInsert:
    """execute_insert succeeds normally; raises AuditIntegrityError on rowcount==0."""

    def test_insert_succeeds(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        # Should not raise
        ops.execute_insert(test_table.insert().values(id="new", value="data"))
        row = ops.execute_fetchone(test_table.select().where(test_table.c.id == "new"))
        assert row is not None
        assert row.value == "data"

    def test_insert_zero_rows_raises(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        """INSERT OR IGNORE on a duplicate PK executes without SQL error but returns rowcount=0."""
        ops.execute_insert(test_table.insert().values(id="row1", value="first"))
        with pytest.raises(AuditIntegrityError, match="zero rows affected"):
            ops.execute_insert(
                sa.text("INSERT OR IGNORE INTO test_rows (id, value) VALUES ('row1', 'dup')"),
            )

    def test_insert_error_message_contains_operation_name(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        ops.execute_insert(test_table.insert().values(id="row2", value="original"))
        with pytest.raises(AuditIntegrityError) as exc_info:
            ops.execute_insert(
                sa.text("INSERT OR IGNORE INTO test_rows (id, value) VALUES ('row2', 'dup')"),
            )
        assert "execute_insert" in str(exc_info.value)

    def test_insert_context_in_error(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        ops.execute_insert(test_table.insert().values(id="row3", value="original"))
        with pytest.raises(AuditIntegrityError, match="insert context label"):
            ops.execute_insert(
                sa.text("INSERT OR IGNORE INTO test_rows (id, value) VALUES ('row3', 'dup')"),
                context="insert context label",
            )

    def test_insert_no_context_omits_parens(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        ops.execute_insert(test_table.insert().values(id="row4", value="original"))
        with pytest.raises(AuditIntegrityError) as exc_info:
            ops.execute_insert(
                sa.text("INSERT OR IGNORE INTO test_rows (id, value) VALUES ('row4', 'dup')"),
            )
        # Without context, the error message should not contain extra parentheses
        message = str(exc_info.value)
        assert "( )" not in message
        assert "()" not in message


class TestExecuteUpdate:
    """execute_update succeeds normally; raises AuditIntegrityError on rowcount==0."""

    def test_update_succeeds(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        ops.execute_insert(test_table.insert().values(id="upd1", value="before"))
        ops.execute_update(test_table.update().where(test_table.c.id == "upd1").values(value="after"))
        row = ops.execute_fetchone(test_table.select().where(test_table.c.id == "upd1"))
        assert row is not None
        assert row.value == "after"

    def test_update_nonexistent_raises(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        with pytest.raises(LandscapeRecordNotFoundError, match="zero rows affected"):
            ops.execute_update(
                test_table.update().where(test_table.c.id == "nonexistent").values(value="new"),
            )

    def test_update_error_message_contains_operation_name(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        with pytest.raises(AuditIntegrityError) as exc_info:
            ops.execute_update(
                test_table.update().where(test_table.c.id == "ghost").values(value="x"),
            )
        assert "execute_update" in str(exc_info.value)

    def test_update_context_in_error(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        with pytest.raises(AuditIntegrityError, match="my context"):
            ops.execute_update(
                test_table.update().where(test_table.c.id == "nope").values(value="x"),
                context="my context",
            )

    def test_update_no_context_omits_parens(self, ops: DatabaseOps, test_table: sa.Table) -> None:
        with pytest.raises(AuditIntegrityError) as exc_info:
            ops.execute_update(
                test_table.update().where(test_table.c.id == "ghost2").values(value="x"),
            )
        message = str(exc_info.value)
        assert "( )" not in message
        assert "()" not in message
