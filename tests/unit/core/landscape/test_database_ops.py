"""Tests for DatabaseOps — the Tier-1 write guard.

Every audit write must succeed.  execute_insert and execute_update raise
AuditIntegrityError when rowcount == 0, which is the enforcement mechanism
for that invariant.  These tests verify all four methods against a real
in-memory SQLite database; no mocks.
"""

from pathlib import Path

import pytest
import sqlalchemy as sa

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.database import LandscapeDB


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
    with db.connection() as conn:
        metadata.create_all(conn.engine)
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
        with db.connection() as conn:
            metadata.create_all(conn.engine)
        ops = DatabaseOps(db)
        ops.execute_insert(table.insert().values(id="a", value="before"))

        with pytest.raises(AuditIntegrityError, match=r"read.?only|execute_fetchall failed|database rejected"):
            ops.execute_fetchall(table.update().where(table.c.id == "a").values(value="after").returning(table.c.id, table.c.value))

        row = ops.execute_fetchone(table.select().where(table.c.id == "a"))
        assert row is not None
        assert row.value == "before"
        db.close()


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
        with pytest.raises(AuditIntegrityError, match="zero rows affected"):
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
