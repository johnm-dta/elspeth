"""Shared helpers for Phase 1 web integration tests.

Duplicates ``_make_session`` from
``tests/unit/web/conftest.py``; if either copy changes the
other must be updated to match. Engine fixtures are NOT shared because
integration tests own their engine fixture and call ``_make_session``
against whatever connection they have.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Connection, insert

from elspeth.web.sessions import models


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
    """Insert a session row with every NOT NULL column populated."""
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
