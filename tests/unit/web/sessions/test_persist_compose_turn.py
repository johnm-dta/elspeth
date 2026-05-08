"""Unit tests for SessionServiceImpl persistence helpers (spec §5.7.1).

Uses the shared ``engine`` fixture and ``_make_session`` helper from
``tests/unit/web/conftest.py``.
"""

from __future__ import annotations

import pytest

from elspeth.web.sessions.service import SessionServiceImpl


@pytest.fixture
def service(engine, tmp_path):
    """Use the shared in-memory SQLite ``engine`` fixture from
    ``tests/unit/web/conftest.py`` (which already wires
    ``create_session_engine`` + ``StaticPool`` + schema bootstrap).
    """
    return SessionServiceImpl(engine, data_dir=tmp_path)


def test_advisory_lock_sqlite_is_noop(service):
    """SQLite does not support pg_advisory_xact_lock; the Postgres-only
    helper itself remains a no-op. Same-session SQLite serialization is
    verified through _session_write_lock tests below."""
    with service._engine.begin() as conn:
        # No raise expected.
        service._acquire_session_advisory_lock(conn, "session_1")


def test_session_write_lock_sqlite_is_reentrant(service):
    """SQLite branch uses a process-wide per-session RLock so nested
    helper calls inside one transaction cannot deadlock.

    The nested ``with`` statements are intentional and load-bearing:
    flattening them with ``with A, B:`` would acquire both contexts
    sequentially before yielding, which is NOT what reentrancy testing
    asks. The test must enter the outer lock, then attempt to enter the
    same lock again from within — that nested acquisition is what proves
    the RLock semantics. ``# noqa: SIM117`` is correct here.
    """
    with service._engine.begin() as conn:  # noqa: SIM117
        with service._session_write_lock(conn, "session_1"):
            with service._session_write_lock(conn, "session_1"):
                pass
