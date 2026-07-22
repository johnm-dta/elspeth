"""Shared same-session locking for every session/blob writer."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import stat
import threading
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import Engine, event
from sqlalchemy.engine import Connection

from elspeth.contracts.advisory_locks import ELSPETH_SESSIONS_LOCK_CLASSID
from elspeth.contracts.errors import AuditIntegrityError

_fcntl: Any
if os.name == "posix" and importlib.util.find_spec("fcntl") is not None:
    import fcntl as _fcntl
else:
    _fcntl = None

_SQLITE_SESSION_LOCKS_GUARD = threading.RLock()
_SQLITE_SESSION_LOCKS: dict[tuple[tuple[str, ...], str], threading.RLock] = {}


class _SQLiteFlockLeaseState(threading.local):
    """Per-thread depth for re-entrant file-lock leases."""

    def __init__(self) -> None:
        self.leases: dict[tuple[tuple[str, ...], str], tuple[int, int]] = {}


_SQLITE_FLOCK_LEASE_STATE = _SQLiteFlockLeaseState()


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


def _sqlite_session_lock_path(engine: Engine, session_id: str) -> Path | None:
    """Return a stable sidecar path, or None for process-local memory DBs."""
    identity = database_lock_identity(engine)
    if identity[1] == "memory":
        return None
    database_path = Path(identity[2])
    database_parent = database_path.parent.resolve()
    lock_dir = database_parent / f".{database_path.name}.session-locks"
    lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if lock_dir.resolve().parent != database_parent:
        raise AuditIntegrityError("SQLite session lock directory escaped the database directory")
    session_digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return lock_dir / f"{session_digest}.lock"


@contextlib.contextmanager
def sqlite_process_session_lock(engine: Engine, session_id: str) -> Iterator[None]:
    """Exclude same-session SQLite writers across threads and OS processes.

    File-backed databases use a stable, never-unlinked sidecar inode. The
    process RLock is always acquired first; ``flock`` is crash-released by the
    kernel. In-memory databases remain process-local because they cannot be
    shared across processes.
    """
    with sqlite_session_mutex(engine, session_id):
        lease_key = (database_lock_identity(engine), session_id)
        lock_path = _sqlite_session_lock_path(engine, session_id)
        if lock_path is None:
            yield
            return
        if _fcntl is None:
            raise AuditIntegrityError("File-backed SQLite requires POSIX flock support for cross-process session safety")
        leases = _SQLITE_FLOCK_LEASE_STATE.leases
        if lease_key in leases:
            descriptor, depth = leases[lease_key]
            leases[lease_key] = (descriptor, depth + 1)
            try:
                yield
            finally:
                leases[lease_key] = (descriptor, depth)
            return
        flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise AuditIntegrityError("SQLite session lock sidecar is not a regular file")
            try:
                _fcntl.flock(descriptor, _fcntl.LOCK_EX)
            except OSError as exc:
                raise AuditIntegrityError("Unable to acquire SQLite process-shared session lock") from exc
            leases[lease_key] = (descriptor, 1)
            try:
                yield
            finally:
                del leases[lease_key]
                _fcntl.flock(descriptor, _fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextlib.contextmanager
def process_session_lock(engine: Engine, session_id: str) -> Iterator[None]:
    """Acquire process-spanning exclusion before a session DB transaction."""
    dialect = engine.dialect.name
    if dialect == "sqlite":
        with sqlite_process_session_lock(engine, session_id):
            yield
        return
    if dialect == "postgresql":
        # PostgreSQL exclusion is transaction-scoped/session-scoped in the
        # caller because its advisory lock lives on a database connection.
        yield
        return
    raise NotImplementedError(f"process_session_lock not implemented for dialect {dialect}")


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
def locked_session_transaction(engine: Engine, session_id: str) -> Iterator[Connection]:
    """Acquire process exclusion before opening a same-session transaction."""
    with process_session_lock(engine, session_id), engine.begin() as conn, transaction_session_lock(conn, engine, session_id):
        yield conn


@contextlib.contextmanager
def sqlite_transaction_session_lock(conn: Connection, engine: Engine, session_id: str) -> Iterator[None]:
    """Hold the shared SQLite mutex until the transaction terminates."""
    lock_stack = ExitStack()
    lock_stack.enter_context(sqlite_process_session_lock(engine, session_id))
    release_state = {"released": False}

    def _release(_conn: Connection) -> None:
        if release_state["released"]:
            return
        release_state["released"] = True
        lock_stack.close()

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
