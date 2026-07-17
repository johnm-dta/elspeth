"""Tests for the portable SQLite insert-builder contract rule."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from elspeth_lints.core.protocols import Finding
from elspeth_lints.rules.contract_invariants.portable_sqlite_insert.rule import find_portable_sqlite_insert_findings

REPO_ROOT = Path(__file__).resolve().parents[3]
PORTABLE_DISPATCH_MODULES = (
    "src/elspeth/core/landscape/scheduler/branch_losses.py",
    "src/elspeth/web/preferences/service.py",
    "src/elspeth/web/secrets/user_store.py",
    "src/elspeth/web/sessions/service.py",
)


def test_sqlite_only_insert_import_is_rejected() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        def write(conn, table, values):
            return conn.execute(sqlite_insert(table).values(**values))
        """
    )

    assert len(findings) == 1
    assert findings[0].rule_id == "contract_invariants.portable_sqlite_insert"
    assert findings[0].line == 2


def test_paired_imports_and_both_dialect_branches_are_accepted() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects.postgresql import insert as postgresql_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        def write(conn, table, values):
            dialect = conn.dialect.name
            if dialect == "sqlite":
                stmt = sqlite_insert(table).values(**values)
            elif dialect == "postgresql":
                stmt = postgresql_insert(table).values(**values)
            else:
                raise NotImplementedError(dialect)
            return conn.execute(stmt)
        """
    )

    assert findings == []


def test_postgresql_import_without_both_dialect_branches_is_rejected() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects.postgresql import insert as postgresql_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        def write(conn, table, values):
            if conn.dialect.name == "sqlite":
                return sqlite_insert(table).values(**values)
            return postgresql_insert(table).values(**values)
        """
    )

    assert len(findings) == 1


def test_both_dialect_branches_without_postgresql_insert_import_are_rejected() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        def write(conn, table, values):
            dialect = conn.dialect.name
            if dialect == "sqlite":
                return sqlite_insert(table).values(**values)
            if dialect == "postgresql":
                raise NotImplementedError("missing PostgreSQL insert builder")
            raise NotImplementedError(dialect)
        """
    )

    assert len(findings) == 1


def test_postgresql_branch_invoking_sqlite_builder_is_rejected() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects.postgresql import insert as postgresql_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        def write(conn, table, values):
            dialect = conn.dialect.name
            if dialect == "sqlite":
                stmt = sqlite_insert(table).values(**values)
            elif dialect == "postgresql":
                stmt = sqlite_insert(table).values(**values)
            else:
                raise NotImplementedError(dialect)
            return conn.execute(stmt)
        """
    )

    assert len(findings) == 1
    assert findings[0].rule_id == "contract_invariants.portable_sqlite_insert"
    assert findings[0].line == 10
    assert "postgresql" in findings[0].message
    assert "sqlite_insert" in findings[0].message


def test_sqlite_branch_invoking_postgresql_builder_is_rejected() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects.postgresql import insert as postgresql_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        def write(conn, table, values):
            dialect = conn.dialect.name
            if dialect == "sqlite":
                stmt = postgresql_insert(table).values(**values)
            elif dialect == "postgresql":
                stmt = postgresql_insert(table).values(**values)
            else:
                raise NotImplementedError(dialect)
            return conn.execute(stmt)
        """
    )

    assert len(findings) == 1
    assert findings[0].line == 8
    assert "sqlite" in findings[0].message
    assert "postgresql_insert" in findings[0].message


def test_misbound_branches_report_each_wrong_call() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects.postgresql import insert as postgresql_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        def write(conn, table, values):
            dialect = conn.dialect.name
            if dialect == "sqlite":
                stmt = postgresql_insert(table).values(**values)
            elif dialect == "postgresql":
                stmt = sqlite_insert(table).values(**values)
            else:
                raise NotImplementedError(dialect)
            return conn.execute(stmt)
        """
    )

    assert len(findings) == 2
    assert [finding.line for finding in findings] == [8, 10]


def test_branch_local_builder_imports_bound_to_their_branches_are_accepted() -> None:
    findings = _analyze(
        """
        def resolve(engine, table, values):
            dialect = engine.dialect.name
            if dialect == "sqlite":
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                def build(table, values):
                    return sqlite_insert(table).values(**values)

                return build
            if dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as postgresql_insert

                def build(table, values):
                    return postgresql_insert(table).values(**values)

                return build
            raise NotImplementedError(dialect)
        """
    )

    assert findings == []


def test_unrelated_sqlite_import_is_accepted() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects.sqlite import JSON

        payload_type = JSON
        """
    )

    assert findings == []


def test_dialects_module_import_insert_call_is_rejected() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects import sqlite

        def write(conn, table, values):
            return conn.execute(sqlite.insert(table).values(**values))
        """
    )

    assert len(findings) == 1
    assert findings[0].rule_id == "contract_invariants.portable_sqlite_insert"
    assert findings[0].line == 5


def test_aliased_dialects_module_import_insert_call_is_rejected() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects import sqlite as sq

        def write(conn, table, values):
            return conn.execute(sq.insert(table).values(**values))
        """
    )

    assert len(findings) == 1
    assert findings[0].line == 5


def test_dotted_module_import_insert_call_is_rejected() -> None:
    findings = _analyze(
        """
        import sqlalchemy.dialects.sqlite

        def write(conn, table, values):
            return conn.execute(sqlalchemy.dialects.sqlite.insert(table).values(**values))
        """
    )

    assert len(findings) == 1
    assert findings[0].line == 5


def test_aliased_dotted_module_import_insert_call_is_rejected() -> None:
    findings = _analyze(
        """
        import sqlalchemy.dialects.sqlite as sq

        def write(conn, table, values):
            return conn.execute(sq.insert(table).values(**values))
        """
    )

    assert len(findings) == 1
    assert findings[0].line == 5


def test_module_attribute_dispatch_pairing_is_accepted() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects import postgresql, sqlite

        def write(conn, table, values):
            dialect = conn.dialect.name
            if dialect == "sqlite":
                stmt = sqlite.insert(table).values(**values)
            elif dialect == "postgresql":
                stmt = postgresql.insert(table).values(**values)
            else:
                raise NotImplementedError(dialect)
            return conn.execute(stmt)
        """
    )

    assert findings == []


def test_symbol_and_module_attribute_pairing_is_accepted() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects import sqlite
        from sqlalchemy.dialects.postgresql import insert as postgresql_insert

        def write(conn, table, values):
            dialect = conn.dialect.name
            if dialect == "sqlite":
                stmt = sqlite.insert(table).values(**values)
            elif dialect == "postgresql":
                stmt = postgresql_insert(table).values(**values)
            else:
                raise NotImplementedError(dialect)
            return conn.execute(stmt)
        """
    )

    assert findings == []


def test_module_attribute_builder_without_postgresql_call_is_rejected() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects import postgresql, sqlite

        def write(conn, table, values):
            dialect = conn.dialect.name
            if dialect == "sqlite":
                stmt = sqlite.insert(table).values(**values)
            elif dialect == "postgresql":
                stmt = sqlite.insert(table).values(**values)
            else:
                raise NotImplementedError(dialect)
            return conn.execute(stmt)
        """
    )

    assert len(findings) == 2
    assert [finding.line for finding in findings] == [7, 9]
    assert all("no complete PostgreSQL counterpart" in finding.message for finding in findings)


def test_swapped_module_attribute_builders_are_rejected() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects import postgresql, sqlite

        def write(conn, table, values):
            dialect = conn.dialect.name
            if dialect == "sqlite":
                stmt = postgresql.insert(table).values(**values)
            elif dialect == "postgresql":
                stmt = sqlite.insert(table).values(**values)
            else:
                raise NotImplementedError(dialect)
            return conn.execute(stmt)
        """
    )

    assert len(findings) == 2
    assert [finding.line for finding in findings] == [7, 9]
    assert "postgresql.insert" in findings[0].message
    assert "sqlite.insert" in findings[1].message


def test_module_import_without_insert_call_is_accepted() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects import postgresql, sqlite

        sqlite_dialect = sqlite.dialect()
        postgresql_dialect = postgresql.dialect()
        """
    )

    assert findings == []


def test_unrelated_module_alias_insert_call_is_accepted() -> None:
    findings = _analyze(
        """
        from mypkg.dialects import sqlite

        def write(items, value):
            return sqlite.insert(items, value)
        """
    )

    assert findings == []


def test_unbound_attribute_insert_call_is_accepted() -> None:
    findings = _analyze(
        """
        def write(items, value):
            items.insert(0, value)
        """
    )

    assert findings == []


@pytest.mark.parametrize("relative_path", PORTABLE_DISPATCH_MODULES)
def test_current_repository_dispatch_modules_are_accepted(relative_path: str) -> None:
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    findings = find_portable_sqlite_insert_findings(ast.parse(source), relative_path)

    assert findings == []


def _analyze(source: str) -> list[Finding]:
    dedented = textwrap.dedent(source)
    return find_portable_sqlite_insert_findings(ast.parse(dedented), "src/elspeth/example.py")
