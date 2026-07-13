"""Fail-closed schema probes and serialized database initialization."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypedDict

import structlog
from sqlalchemy import Connection, Engine, inspect, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import OperationalError

from elspeth.contracts.advisory_locks import ELSPETH_SCHEMA_INIT_LOCK_CLASSID
from elspeth.core.landscape.database import (
    LandscapeSchemaShape,
    SchemaCompatibilityError,
    create_additive_indexes,
    probe_schema_shape,
)
from elspeth.core.landscape.schema import metadata as landscape_metadata
from elspeth.web.sessions.models import metadata as session_metadata
from elspeth.web.sessions.schema import (
    SessionSchemaError,
    _assert_schema_sentinels,
    _stamp_schema_sentinels,
    probe_current_schema,
)

_slog = structlog.get_logger(__name__)
_LOCK_TARGET = "elspeth_schema_init"
_LOCK_TIMEOUT = "5s"
_TARGET_ERROR = "PostgreSQL database target cannot be proven safe from static URL configuration."
_SCHEMA_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*\Z")
_SEARCH_PATH = re.compile(r"-c(?:\s+)?search_path=([^\s]+)\Z")


class SchemaState(Enum):
    MISSING = "missing"
    PARTIAL = "partial"
    CURRENT = "current"
    STALE = "stale"


class SchemaInitBusyError(RuntimeError):
    """Raised when another initializer holds the PostgreSQL schema lock."""


class SchemaLockCleanupError(RuntimeError):
    """Raised when advisory-lock cleanup cannot be proven."""


class DatabaseTargetConflictError(ValueError):
    """Raised when two PostgreSQL logical targets are not provably distinct."""


@dataclass(frozen=True, slots=True)
class PostgresLogicalTarget:
    host: str
    port: int
    database: str
    explicit_schema: str | None


AWS_ECS_POOL_KWARGS: Mapping[str, object] = MappingProxyType({"pool_size": 5, "max_overflow": 5, "pool_pre_ping": True})


class PostgresEngineKwargs(TypedDict, total=False):
    pool_size: int
    max_overflow: int
    pool_pre_ping: bool


def postgres_engine_kwargs(url: str | URL) -> PostgresEngineKwargs:
    parsed = make_url(url)
    if parsed.drivername.split("+", 1)[0] != "postgresql":
        return {}
    return {"pool_size": 5, "max_overflow": 5, "pool_pre_ping": True}


def _target_error() -> DatabaseTargetConflictError:
    return DatabaseTargetConflictError(_TARGET_ERROR)


def postgres_logical_target_key(url: str | URL) -> PostgresLogicalTarget:
    """Parse only statically provable PostgreSQL logical-target attributes."""
    try:
        parsed = make_url(url)
    except (TypeError, ValueError):
        raise _target_error() from None
    if parsed.drivername.split("+", 1)[0] != "postgresql" or not parsed.host or not parsed.database:
        raise _target_error()

    options_value = parsed.query.get("options")
    explicit_schema: str | None = None
    if options_value is not None:
        if not isinstance(options_value, str):
            raise _target_error()
        match = _SEARCH_PATH.fullmatch(options_value.strip())
        if match is None:
            raise _target_error()
        candidate = match.group(1)
        if not _SCHEMA_NAME.fullmatch(candidate) or candidate.startswith("$"):
            raise _target_error()
        explicit_schema = candidate.lower()

    return PostgresLogicalTarget(
        host=parsed.host.lower(),
        port=parsed.port or 5432,
        database=parsed.database,
        explicit_schema=explicit_schema,
    )


def require_distinct_postgres_targets(session_url: str | URL, landscape_url: str | URL) -> None:
    session = postgres_logical_target_key(session_url)
    landscape = postgres_logical_target_key(landscape_url)
    if (session.host, session.port, session.database) != (landscape.host, landscape.port, landscape.database):
        return
    if session.explicit_schema is None or landscape.explicit_schema is None:
        raise _target_error()
    if session.explicit_schema == landscape.explicit_schema:
        raise _target_error()


def probe_session_schema(bind: Engine | Connection) -> SchemaState:
    inspector = inspect(bind)
    existing = set(inspector.get_table_names()) - {"sqlite_sequence"}
    if not existing:
        try:
            _assert_schema_sentinels(bind)
        except SessionSchemaError:
            return SchemaState.STALE
        return SchemaState.MISSING
    expected = set(session_metadata.tables)
    if not existing & expected or existing != expected:
        return SchemaState.STALE
    return SchemaState.CURRENT if probe_current_schema(bind) else SchemaState.STALE


def probe_landscape_schema(bind: Engine | Connection) -> SchemaState:
    return {
        LandscapeSchemaShape.EMPTY: SchemaState.MISSING,
        LandscapeSchemaShape.INCOMPLETE: SchemaState.PARTIAL,
        LandscapeSchemaShape.MATCHES: SchemaState.CURRENT,
        LandscapeSchemaShape.FOREIGN: SchemaState.STALE,
        LandscapeSchemaShape.DIVERGENT: SchemaState.STALE,
    }[probe_schema_shape(bind)]


def _sqlstate(exc: OperationalError) -> str | None:
    original = exc.orig
    value = getattr(original, "sqlstate", None)
    if value is None:
        value = getattr(original, "pgcode", None)
    return value if isinstance(value, str) else None


def _invalidate(conn: Connection) -> None:
    try:
        if conn.in_transaction():
            conn.rollback()
    finally:
        conn.invalidate()


def _run_locked(
    engine: Engine,
    *,
    target: str,
    body: Callable[[Connection], None],
    verify: Callable[[Connection], None],
) -> None:
    """Run schema DDL and verification on one connection under one lock."""
    with engine.connect() as conn:
        postgres = conn.dialect.name == "postgresql"
        acquired = False
        earlier: BaseException | None = None
        try:
            if postgres:
                try:
                    conn.execute(
                        text("SELECT set_config('lock_timeout', :timeout, true)"),
                        {"timeout": _LOCK_TIMEOUT},
                    )
                    conn.execute(
                        text("SELECT pg_advisory_lock(:classid, hashtext(:target))"),
                        {"classid": ELSPETH_SCHEMA_INIT_LOCK_CLASSID, "target": target},
                    )
                    acquired = True
                except OperationalError as exc:
                    _invalidate(conn)
                    if _sqlstate(exc) == "55P03":
                        raise SchemaInitBusyError("Another schema initialization is in progress; retry shortly.") from exc
                    raise
                except BaseException:
                    _invalidate(conn)
                    raise

            body(conn)
            if conn.in_transaction():
                conn.commit()
            verify(conn)
        except BaseException as exc:
            earlier = exc
            raise
        finally:
            if conn.in_transaction():
                conn.rollback()
            if postgres and acquired:
                cleanup_error: BaseException | None = None
                try:
                    unlocked = conn.execute(
                        text("SELECT pg_advisory_unlock(:classid, hashtext(:target))"),
                        {"classid": ELSPETH_SCHEMA_INIT_LOCK_CLASSID, "target": target},
                    ).scalar_one()
                    if unlocked is not True:
                        cleanup_error = RuntimeError("unlock was not confirmed")
                except BaseException as exc:
                    cleanup_error = exc
                finally:
                    if conn.in_transaction():
                        conn.rollback()
                if cleanup_error is not None:
                    _invalidate(conn)
                    if earlier is None:
                        raise SchemaLockCleanupError(
                            "Schema initialization may have completed but lock cleanup was not verified; investigate and rerun."
                        ) from cleanup_error
                    _slog.error(
                        "schema_lock_cleanup_unverified",
                        original_exc_class=type(earlier).__name__,
                        cleanup_exc_class=type(cleanup_error).__name__,
                    )


def init_session_schema(engine: Engine) -> None:
    def body(conn: Connection) -> None:
        state = probe_session_schema(conn)
        if state is SchemaState.CURRENT:
            return
        if state is SchemaState.MISSING:
            session_metadata.create_all(bind=conn, checkfirst=True)
            _stamp_schema_sentinels(conn)
            return
        raise SessionSchemaError("Session database schema is stale or partial; delete the old session database and restart.")

    def verify(conn: Connection) -> None:
        if probe_session_schema(conn) is not SchemaState.CURRENT:
            raise SessionSchemaError("Session database initialization did not produce the current schema.")

    _run_locked(engine, target=_LOCK_TARGET, body=body, verify=verify)


def init_landscape_schema(engine: Engine) -> None:
    def body(conn: Connection) -> None:
        state = probe_landscape_schema(conn)
        if state is SchemaState.CURRENT:
            return
        if state in (SchemaState.MISSING, SchemaState.PARTIAL):
            landscape_metadata.create_all(bind=conn, checkfirst=True)
            create_additive_indexes(conn)
            return
        raise SchemaCompatibilityError("Landscape database schema is stale or foreign; delete/recreate it or run the supported migration.")

    def verify(conn: Connection) -> None:
        if probe_landscape_schema(conn) is not SchemaState.CURRENT:
            raise SchemaCompatibilityError("Landscape database initialization did not produce the current schema.")

    _run_locked(engine, target=_LOCK_TARGET, body=body, verify=verify)
