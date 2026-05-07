"""Session database engine factory.

Centralizes session engine construction so invariants — chiefly
``PRAGMA foreign_keys=ON`` on SQLite — cannot be bypassed by accident.
Every caller that needs a session engine MUST use
``create_session_engine()``. Bare ``sqlalchemy.create_engine`` calls that
target the sessions DB are forbidden and caught by CI lint.

The sessions database is Tier 1 ("our data"); silent FK non-enforcement
is a Tier 1 integrity failure, not a warning, so the factory also
asserts the PRAGMA took effect on first connect and refuses to return
an engine that does not enforce foreign keys on SQLite.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine.interfaces import DBAPIConnection


def create_session_engine(url: str, **kwargs: Any) -> Engine:
    """Build an engine for the sessions DB with FK enforcement wired in.

    For SQLite engines, registers a ``connect`` event listener that runs
    ``PRAGMA foreign_keys=ON`` for every new DBAPI connection, then opens
    a probe connection to assert the PRAGMA actually took effect. If
    enforcement is not active, raises ``RuntimeError`` rather than
    returning a Tier 1 engine that silently accepts FK violations.

    Non-SQLite dialects are returned unmodified; their FK enforcement is
    always on and is the database's responsibility, not ours.

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
    def _enable_sqlite_foreign_keys(
        dbapi_conn: DBAPIConnection,
        _record: object,  # SQLAlchemy internal _ConnectionRecord; unused
    ) -> None:
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    # Startup probe. On QueuePool this connection is a fresh checkout
    # whose ``connect`` listener just ran, so reading PRAGMA here
    # genuinely validates that the listener took effect for newly
    # pooled connections. On StaticPool (the test configuration) the
    # same single connection is reused for every checkout, so this
    # probe is tautologically true — but it is still the canonical
    # failure site if the listener is ever deleted, reordered, or
    # shadowed by a subclass, and the cost is one trivial query at
    # process start. Removing it "because tests don't need it" would
    # silently weaken production's Tier 1 guarantee, so do not.
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA foreign_keys")).scalar_one()
    if result != 1:
        raise RuntimeError(
            f"Session engine {engine.url!r} rejected PRAGMA foreign_keys=ON "
            f"(got {result!r}). Refusing to start — Tier 1 integrity requires "
            f"foreign-key enforcement on SQLite."
        )

    return engine
