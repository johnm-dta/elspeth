"""Shared same-session locking for every session/blob writer."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import Engine, event
from sqlalchemy.engine import Connection

from elspeth.contracts.advisory_locks import ELSPETH_SESSIONS_LOCK_CLASSID
from elspeth.contracts.errors import AuditIntegrityError

_SQLITE_SESSION_LOCKS_GUARD = threading.RLock()
_SQLITE_SESSION_LOCKS: dict[tuple[tuple[str, ...], str], threading.RLock] = {}


def database_lock_identity(engine: Engine) -> tuple[str, ...]:
    """Return a credential-free identity shared by engines for one database."""
    url = engine.url
    backend = url.get_backend_name()
    if backend == "sqlite":
        database = url.database
        if database is None or database in {"", ":memory:"}:
            return (backend, "memory", str(id(engine.pool)))
        return (backend, "file", str(Path(database).expanduser().resolve()))
    return (
        backend,
        url.host or "",
        str(url.port) if url.port is not None else "",
        url.database or "",
    )


def sqlite_session_mutex(engine: Engine, session_id: str) -> threading.RLock:
    """Return the process-wide SQLite mutex for one DB/session pair."""
    key = (database_lock_identity(engine), session_id)
    with _SQLITE_SESSION_LOCKS_GUARD:
        if key not in _SQLITE_SESSION_LOCKS:
            _SQLITE_SESSION_LOCKS[key] = threading.RLock()
        return _SQLITE_SESSION_LOCKS[key]


def acquire_session_advisory_xact_lock(conn: Connection, session_id: str) -> None:
    """Acquire the shared PostgreSQL transaction-scoped session lock."""
    conn.exec_driver_sql(
        "SELECT pg_catalog.pg_advisory_xact_lock(%s, pg_catalog.hashtext(%s))",
        (ELSPETH_SESSIONS_LOCK_CLASSID, session_id),
    )


@contextlib.contextmanager
def transaction_session_lock(conn: Connection, engine: Engine, session_id: str) -> Iterator[None]:
    """Hold the shared same-session lock through the current transaction."""
    dialect = engine.dialect.name
    if dialect == "sqlite":
        with sqlite_transaction_session_lock(conn, engine, session_id):
            yield
        return
    if dialect == "postgresql":
        acquire_session_advisory_xact_lock(conn, session_id)
        yield
        return
    raise NotImplementedError(f"transaction_session_lock not implemented for dialect {dialect}")


@contextlib.contextmanager
def sqlite_transaction_session_lock(conn: Connection, engine: Engine, session_id: str) -> Iterator[None]:
    """Hold the shared SQLite mutex until the transaction terminates."""
    lock = sqlite_session_mutex(engine, session_id)
    lock.acquire()
    release_state = {"released": False}

    def _release(_conn: Connection) -> None:
        if release_state["released"]:
            return
        release_state["released"] = True
        lock.release()

    def _remove_listener(identifier: Literal["commit", "rollback"], fn: Any) -> None:
        if event.contains(conn, identifier, fn):
            event.remove(conn, identifier, fn)

    def _release_on_commit(_conn: Connection) -> None:
        _release(_conn)
        _remove_listener("rollback", _release_on_rollback)

    def _release_on_rollback(_conn: Connection) -> None:
        _release(_conn)
        _remove_listener("commit", _release_on_commit)

    if conn.in_transaction():
        event.listen(conn, "commit", _release_on_commit, once=True)
        event.listen(conn, "rollback", _release_on_rollback, once=True)
    try:
        yield
    finally:
        if not conn.in_transaction():
            _release(conn)


@contextlib.contextmanager
def postgres_session_advisory_lock(conn: Connection, session_id: str) -> Iterator[None]:
    """Hold a PostgreSQL session-level lock across multiple transactions."""
    conn.exec_driver_sql(
        "SELECT pg_catalog.pg_advisory_lock(%s, pg_catalog.hashtext(%s))",
        (ELSPETH_SESSIONS_LOCK_CLASSID, session_id),
    )
    conn.commit()
    try:
        yield
    finally:
        if conn.in_transaction():
            conn.rollback()
        unlocked = conn.exec_driver_sql(
            "SELECT pg_catalog.pg_advisory_unlock(%s, pg_catalog.hashtext(%s))",
            (ELSPETH_SESSIONS_LOCK_CLASSID, session_id),
        ).scalar_one()
        conn.commit()
        if unlocked is not True:
            raise AuditIntegrityError("PostgreSQL session advisory lock was not held during release")
