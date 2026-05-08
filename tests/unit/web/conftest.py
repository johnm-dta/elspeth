"""Shared fixtures and helpers for Phase 1 web unit tests.

Hoisted from ``tests/unit/web/sessions/conftest.py`` to the parent
``tests/unit/web/`` package so both the sessions suite
(``tests/unit/web/sessions/test_*.py``) and the composer suite see the
same fixtures. pytest auto-loads parent-directory conftests for any
test under that directory tree, so no per-subdirectory shim is needed.

Provides:
- ``engine`` — an in-memory SQLite engine with FK enforcement applied
  via ``create_session_engine``'s connect-event listener (so EVERY
  pool checkout enforces FKs, not just the first), backed by
  ``StaticPool`` so worker threads dispatched via ``_run_sync`` see
  the same in-memory database. Schema is bootstrapped via
  ``initialize_session_schema`` (the same path production uses).
- ``_make_session`` — non-fixture helper that inserts a row into
  ``sessions_table`` with every NOT NULL column populated. Test code
  imports it explicitly via the absolute path
  ``from tests.unit.web.conftest import _make_session`` (matches the
  codebase convention for cross-package shared helpers — see
  ``tests/fixtures/`` and ``tests/helpers/`` import sites).

Why these live in conftest rather than each test file inlining them:
the four NOT NULL columns on ``sessions_table`` (``user_id``,
``auth_provider_type``, ``title``, ``updated_at``) and the
``StaticPool`` requirement are easy to forget. Centralising both
makes the fixture banned-pattern violations (plain ``create_engine``;
minimum-columns inserts) literally absent from the test code.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import Connection, insert
from sqlalchemy.pool import StaticPool

from elspeth.web.sessions import models
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema


@pytest.fixture
def engine():
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return eng


def _make_session(
    conn: Connection,
    *,
    session_id: str,
    user_id: str = "test_user",
    auth_provider_type: str = "local",
    title: str = "test session",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> None:
    """Insert a session row with every NOT NULL column populated.

    The defaults are sufficient for tests that do not care about
    user/auth-provider fields. Tests that exercise auth-scoped
    behaviour should pass explicit ``user_id`` and
    ``auth_provider_type`` values.
    """
    now = created_at or datetime.now(UTC)
    conn.execute(
        insert(models.sessions_table).values(
            id=session_id,
            user_id=user_id,
            auth_provider_type=auth_provider_type,
            title=title,
            created_at=now,
            updated_at=updated_at or now,
        )
    )
