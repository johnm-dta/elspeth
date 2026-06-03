"""Tests for session DB engine PRAGMA discipline and schema-epoch guard.

Phase 5b Task 1.5 (filigree elspeth-6815a49a7d). Mechanically enforces
the operator-delete-DB policy: WAL mode, busy_timeout, FK enforcement,
and a schema-version sentinel that produces an actionable crash when an
operator forgets to delete a stale session DB across a schema-bumping
release.

The five tests below cover:

1. WAL journal_mode actually takes effect on a fresh file-backed DB.
2. busy_timeout is set to 5000ms (matches Landscape PRAGMA discipline).
3. PRAGMA foreign_keys=ON (existing Tier 1 invariant preserved).
4. After ``initialize_session_schema``, PRAGMA application_id and
   user_version carry the project's SESSION_DB_APPLICATION_ID and
   SESSION_SCHEMA_EPOCH sentinels.
5. A stale DB (correct application_id but mismatched user_version) is
   rejected with the actionable "Delete the session DB file and restart"
   message rather than failing later with a cryptic SQLAlchemy error.

In-memory DBs are deliberately avoided for the WAL test: SQLite reports
journal_mode='memory' for ``:memory:`` databases and never enters WAL,
so a meaningful WAL assertion requires a file-backed DB.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    SESSION_DB_APPLICATION_ID,
    SESSION_SCHEMA_EPOCH,
)
from elspeth.web.sessions.schema import (
    SessionSchemaError,
    initialize_session_schema,
)


@pytest.fixture
def file_db_url(tmp_path: Path) -> str:
    """File-backed sqlite URL. WAL only works on real files."""
    return f"sqlite:///{tmp_path / 'session_test.db'}"


def test_create_session_engine_enters_wal_mode_on_file_db(file_db_url: str) -> None:
    """A fresh on-disk session DB MUST report journal_mode=wal.

    Mirrors the Landscape DB's WAL discipline at
    ``core/landscape/database.py:_configure_sqlite``. Production session
    workload involves concurrent reads (route handlers) alongside writes
    (compose loop, audit persistence), which is exactly WAL's strength.
    """
    engine = create_session_engine(file_db_url)
    with engine.connect() as conn:
        journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar_one()
    assert journal_mode == "wal", f"session engine did not enter WAL mode (got {journal_mode!r}). Phase 5b requires WAL for the session DB."


def test_create_session_engine_sets_busy_timeout(file_db_url: str) -> None:
    """PRAGMA busy_timeout MUST be 5000ms (matches Landscape discipline)."""
    engine = create_session_engine(file_db_url)
    with engine.connect() as conn:
        busy_timeout = conn.execute(text("PRAGMA busy_timeout")).scalar_one()
    assert busy_timeout == 5000


def test_create_session_engine_enables_foreign_keys(file_db_url: str) -> None:
    """PRAGMA foreign_keys=ON is the pre-existing Tier 1 invariant; this
    test pins it across the PRAGMA listener extension so a future edit
    cannot quietly drop FK enforcement while adding WAL/busy_timeout.
    """
    engine = create_session_engine(file_db_url)
    with engine.connect() as conn:
        foreign_keys = conn.execute(text("PRAGMA foreign_keys")).scalar_one()
    assert foreign_keys == 1


def test_create_session_engine_reapplies_sqlite_pragmas_after_reconnect(
    file_db_url: str,
) -> None:
    """PRAGMA discipline MUST be per DBAPI connection, not one-shot setup.

    ``foreign_keys``, ``busy_timeout``, and ``synchronous`` are connection-
    scoped in SQLite. Disposing the pool forces the next checkout to create
    a new DBAPI connection, so this pins the connect-listener guarantee.
    """
    engine = create_session_engine(file_db_url)
    with engine.connect() as conn:
        first = {
            "foreign_keys": conn.execute(text("PRAGMA foreign_keys")).scalar_one(),
            "journal_mode": conn.execute(text("PRAGMA journal_mode")).scalar_one(),
            "busy_timeout": conn.execute(text("PRAGMA busy_timeout")).scalar_one(),
            "synchronous": conn.execute(text("PRAGMA synchronous")).scalar_one(),
        }

    engine.dispose()

    with engine.connect() as conn:
        second = {
            "foreign_keys": conn.execute(text("PRAGMA foreign_keys")).scalar_one(),
            "journal_mode": conn.execute(text("PRAGMA journal_mode")).scalar_one(),
            "busy_timeout": conn.execute(text("PRAGMA busy_timeout")).scalar_one(),
            "synchronous": conn.execute(text("PRAGMA synchronous")).scalar_one(),
        }

    assert first == {
        "foreign_keys": 1,
        "journal_mode": "wal",
        "busy_timeout": 5000,
        "synchronous": 1,
    }
    assert second == first


def test_initialize_session_schema_stamps_application_id_and_user_version(
    file_db_url: str,
) -> None:
    """A freshly-initialized session DB MUST carry both schema sentinels.

    ``application_id`` lets the operator (and forensics tooling) confirm a
    given SQLite file is in fact an ELSPETH session DB rather than some
    other SQLite file that happens to live at the configured path.
    ``user_version`` tracks SESSION_SCHEMA_EPOCH; the startup guard reads
    it to detect stale DBs that survive across a schema bump.
    """
    engine = create_session_engine(file_db_url)
    initialize_session_schema(engine)
    with engine.connect() as conn:
        app_id = conn.execute(text("PRAGMA application_id")).scalar_one()
        user_ver = conn.execute(text("PRAGMA user_version")).scalar_one()
    assert app_id == SESSION_DB_APPLICATION_ID
    assert user_ver == SESSION_SCHEMA_EPOCH


def test_initialize_session_schema_rejects_stale_user_version(
    file_db_url: str,
) -> None:
    """A pre-existing DB with wrong user_version MUST crash on startup
    with the actionable delete-the-DB message.

    Simulates an operator who upgraded ELSPETH across a SESSION_SCHEMA_EPOCH
    bump without deleting their staging session DB. Without this guard
    they would see obscure SQLAlchemy errors (missing columns, missing
    tables) the first time a Phase 5b code path touched the DB. With it
    they see a precise instruction to delete the DB and restart.
    """
    # Build a "stale" DB: full current schema BUT user_version pinned at
    # an earlier epoch. The application_id is correct (this is genuinely
    # one of our DBs, not a foreign file) — the mismatch is purely on the
    # schema-version axis.
    engine = create_session_engine(file_db_url)
    initialize_session_schema(engine)
    with engine.connect() as conn:
        conn.execute(text(f"PRAGMA user_version = {SESSION_SCHEMA_EPOCH - 1}"))
        conn.commit()

    with pytest.raises(SessionSchemaError, match="Delete the session DB file and restart"):
        initialize_session_schema(engine)


def test_initialize_session_schema_rejects_foreign_application_id(
    file_db_url: str,
) -> None:
    """A SQLite file with a non-zero, non-ELSP application_id MUST be
    rejected. Catches the failure mode where someone configures the
    session DB URL to point at a SQLite file produced by some other
    application — we refuse to overwrite it.
    """
    engine = create_session_engine(file_db_url)
    # Stamp a foreign application_id BEFORE any schema work, simulating
    # a pre-existing non-ELSPETH SQLite file at the configured path.
    with engine.connect() as conn:
        conn.execute(text("PRAGMA application_id = 305419896"))  # 0x12345678
        conn.commit()
        # Force the DB file to materialise something so the validator
        # sees the application_id we just stamped on a subsequent open.
        conn.execute(text("CREATE TABLE _foreign_marker (id INTEGER)"))
        conn.commit()

    with pytest.raises(SessionSchemaError, match="Delete the session DB file and restart"):
        initialize_session_schema(engine)
