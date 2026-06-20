"""Session database engine factory.

Centralizes session engine construction so PRAGMA invariants — chiefly
``foreign_keys=ON``, WAL journal mode, and ``busy_timeout=5000`` —
cannot be bypassed by accident. Every caller that needs a session engine
MUST use ``create_session_engine()``. Bare ``sqlalchemy.create_engine``
calls that target the sessions DB are forbidden and caught by CI lint.

The sessions database is Tier 1 ("our data"); silent FK non-enforcement
or absent WAL mode is a Tier 1 integrity / availability failure, not a
warning, so the factory also asserts both PRAGMAs took effect on first
connect and refuses to return an engine that does not meet production
guarantees on SQLite.

PRAGMA discipline mirrors ``core/landscape/database.py:_configure_sqlite``:
both Tier 1 databases (Landscape and session) use the same WAL + FK +
busy_timeout + synchronous=NORMAL settings.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import MethodType
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import Connection
from sqlalchemy.engine.interfaces import DBAPIConnection

_SESSION_WRITE_INTENT_OPTION = "elspeth_session_write_intent"


def create_session_engine(url: str, **kwargs: Any) -> Engine:
    """Build an engine for the sessions DB with PRAGMA discipline wired in.

    For SQLite engines, registers a ``connect`` event listener that runs
    the production PRAGMA block on every new DBAPI connection, then opens
    a probe connection to assert the PRAGMAs actually took effect:

    - ``foreign_keys=ON`` — Tier 1 referential integrity.
    - ``journal_mode=WAL`` — concurrent-read tolerance under writer load.
    - ``synchronous=NORMAL`` — WAL-paired durability/perf tradeoff matching
      the Landscape DB; safe with WAL because the WAL itself is fsynced.
    - ``busy_timeout=5000`` — survive transient contention without
      raising ``SQLITE_BUSY`` to the application layer.

    If FK enforcement or WAL mode is not active on a file-backed DB,
    raises ``RuntimeError`` rather than returning a Tier 1 engine that
    silently weakens its own guarantees.

    The WAL probe is gated to file-backed DBs because SQLite's
    ``:memory:`` databases cannot enter WAL — they report
    ``journal_mode='memory'``. The connect listener still issues the
    PRAGMA (SQLite silently ignores it), so the production behaviour is
    identical; only the assertion is skipped for in-memory test DBs.

    Non-SQLite dialects are returned unmodified; their concurrency
    settings and FK enforcement are the database's responsibility, not
    ours.

    Parameters
    ----------
    url:
        SQLAlchemy URL for the sessions database.
    **kwargs:
        Forwarded to ``sqlalchemy.create_engine`` (e.g. ``poolclass``,
        ``connect_args``).
    """
    engine = create_engine(url, **kwargs)

    if engine.dialect.name != "sqlite":
        return engine

    @event.listens_for(engine, "connect")
    def _configure_sqlite_session(
        dbapi_conn: DBAPIConnection,
        _record: object,  # SQLAlchemy internal _ConnectionRecord; unused
    ) -> None:
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()
        # Take manual control of transaction starts so SQLAlchemy's begin
        # event below is the only BEGIN source. Without this pysqlite can emit
        # its own deferred BEGIN before the first DML statement, preserving the
        # read-then-write lock-upgrade race this engine is meant to close.
        dbapi_conn.isolation_level = None

    @event.listens_for(engine, "begin")
    def _begin_immediate(conn: Connection) -> None:
        if conn.dialect.name != "sqlite":
            return
        if conn.get_execution_options().get(_SESSION_WRITE_INTENT_OPTION, False):
            conn.exec_driver_sql("BEGIN IMMEDIATE")
        else:
            conn.exec_driver_sql("BEGIN")

    @contextmanager
    def _begin_session_write(self: Engine) -> Iterator[Connection]:
        # SQLAlchemy's begin event does not distinguish explicit
        # engine.begin() from autobegin on a plain engine.connect().execute().
        # Mark only the explicit session write transaction as IMMEDIATE so
        # bare read connections keep the lock-free deferred BEGIN path.
        with self.connect() as conn:
            write_conn = conn.execution_options(**{_SESSION_WRITE_INTENT_OPTION: True})
            with write_conn.begin():
                yield write_conn

    engine.begin = MethodType(_begin_session_write, engine)  # type: ignore[method-assign]

    # Startup probe. On QueuePool this connection is a fresh checkout
    # whose ``connect`` listener just ran, so reading the PRAGMAs here
    # genuinely validates that the listener took effect for newly
    # pooled connections. On StaticPool (the test configuration) the
    # same single connection is reused for every checkout, so this
    # probe is tautologically true — but it is still the canonical
    # failure site if the listener is ever deleted, reordered, or
    # shadowed by a subclass, and the cost is three trivial queries at
    # process start. Removing it "because tests don't need it" would
    # silently weaken production's Tier 1 guarantee, so do not.
    with engine.connect() as conn:
        foreign_keys = conn.execute(text("PRAGMA foreign_keys")).scalar_one()
        journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar_one()
    if foreign_keys != 1:
        raise RuntimeError(
            f"Session engine {engine.url!r} rejected PRAGMA foreign_keys=ON "
            f"(got {foreign_keys!r}). Refusing to start — Tier 1 integrity requires "
            f"foreign-key enforcement on SQLite."
        )
    # WAL mode is meaningless on ``:memory:`` databases — SQLite reports
    # ``journal_mode='memory'`` and there is no on-disk file to journal
    # against. Skip the WAL assertion for in-memory DBs so the
    # production listener can be unit-tested with the cheap ``:memory:``
    # fixture without producing a spurious "WAL refused" crash. File-
    # backed DBs (the production deployment shape) still trip the
    # assertion.
    db_path = engine.url.database
    is_memory_db = db_path is None or db_path == ":memory:"
    if not is_memory_db and journal_mode != "wal":
        raise RuntimeError(
            f"Session engine {engine.url!r} did not enter WAL mode "
            f"(got {journal_mode!r}). Refusing to start — production requires WAL "
            f"for concurrent read/write tolerance under the Phase 5b session DB "
            f"workload."
        )

    return engine
