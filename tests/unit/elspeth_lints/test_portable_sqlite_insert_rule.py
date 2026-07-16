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


def test_unrelated_sqlite_import_is_accepted() -> None:
    findings = _analyze(
        """
        from sqlalchemy.dialects.sqlite import JSON

        payload_type = JSON
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
