#!/usr/bin/env python3
"""Migrate the exact AWS ECS database cohort from release 0.7.0 to 0.7.1.

This is deliberately a versioned one-shot command, not a general migration
framework. It accepts only the known pre-state, the command's one recoverable
cross-database partial, or the fully-current state. Every other shape fails
before DDL.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO, NoReturn

from sqlalchemy import Connection, Engine, MetaData, create_engine, inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError
from sqlalchemy.schema import CreateTable

from elspeth.contracts.advisory_locks import ELSPETH_SCHEMA_INIT_LOCK_CLASSID
from elspeth.core.landscape.schema import SQLITE_SCHEMA_EPOCH
from elspeth.core.landscape.schema import metadata as landscape_metadata
from elspeth.core.landscape.schema import schema_identity_table as landscape_schema_identity_table
from elspeth.core.schema_identity import SCHEMA_IDENTITY_TABLE_NAME, insert_schema_identity
from elspeth.core.schema_shape import SchemaShapeIssue, collect_metadata_shape_issues
from elspeth.web.schema_probe import (
    DatabaseTargetConflictError,
    SchemaState,
    postgres_engine_kwargs,
    postgres_logical_target_key,
    probe_landscape_schema,
    probe_session_schema,
    require_distinct_postgres_targets,
)
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import POSTGRESQL_AUDIT_DDL_COHORT, SESSION_SCHEMA_EPOCH
from elspeth.web.sessions.models import metadata as session_metadata
from elspeth.web.sessions.models import schema_identity_table as session_schema_identity_table

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SESSION_OWNER_URL_ENV = "ELSPETH_RELEASE_0_7_1_SESSION_SCHEMA_OWNER_URL"
LANDSCAPE_OWNER_URL_ENV = "ELSPETH_RELEASE_0_7_1_LANDSCAPE_SCHEMA_OWNER_URL"

_LOCK_TARGET = "elspeth_schema_init"
_LOCK_TIMEOUT = "5s"
_WIDEN_HASH_SQL = "ALTER TABLE run_sources ALTER COLUMN schema_contract_hash TYPE VARCHAR(32)"


class SessionState(StrEnum):
    NOT_CHECKED = "not_checked"
    RELEASE_0_7_0 = "release_0_7_0"
    CURRENT = "current"
    INVALID = "invalid"


class LandscapeState(StrEnum):
    NOT_CHECKED = "not_checked"
    RELEASE_0_7_0_WIDTH_16 = "release_0_7_0_width_16"
    RELEASE_0_7_0_WIDTH_32 = "release_0_7_0_width_32"
    CURRENT = "current"
    INVALID = "invalid"


class MigrationPlan(StrEnum):
    APPLY_BOTH = "apply_both"
    RESUME_LANDSCAPE = "resume_landscape"
    ALREADY_APPLIED = "already_applied"


class ResultStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ResultCode(StrEnum):
    MIGRATION_APPLIED = "MIGRATION_APPLIED"
    ALREADY_APPLIED = "ALREADY_APPLIED"
    APPLY_REQUIRED = "APPLY_REQUIRED"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    ENVIRONMENT_REQUIRED = "ENVIRONMENT_REQUIRED"
    TARGET_CONFLICT = "TARGET_CONFLICT"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    LOCK_BUSY = "LOCK_BUSY"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    DATABASE_ERROR = "DATABASE_ERROR"
    POST_VERIFY_FAILED = "POST_VERIFY_FAILED"
    LOCK_CLEANUP_UNVERIFIED = "LOCK_CLEANUP_UNVERIFIED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True, slots=True)
class MigrationSummary:
    status: ResultStatus
    code: ResultCode
    already_applied: bool
    session_state: SessionState
    landscape_state: LandscapeState

    def to_dict(self) -> dict[str, bool | str]:
        """Return the sole public, closed, redacted result shape."""
        return {
            "already_applied": self.already_applied,
            "code": self.code.value,
            "landscape_state": self.landscape_state.value,
            "session_state": self.session_state.value,
            "status": self.status.value,
        }


class MigrationFailure(RuntimeError):
    """A closed-code failure that never carries database or row material."""

    def __init__(
        self,
        code: ResultCode,
        *,
        session_state: SessionState = SessionState.NOT_CHECKED,
        landscape_state: LandscapeState = LandscapeState.NOT_CHECKED,
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.session_state = session_state
        self.landscape_state = landscape_state


def release_0_7_0_session_table_names() -> frozenset[str]:
    """Return the exact recognized pre-identity session table set."""
    return frozenset(session_metadata.tables) - {SCHEMA_IDENTITY_TABLE_NAME}


def release_0_7_0_landscape_table_names() -> frozenset[str]:
    """Return the exact recognized pre-identity Landscape table set."""
    return frozenset(landscape_metadata.tables) - {SCHEMA_IDENTITY_TABLE_NAME}


def release_0_7_1_ddl_statements(*, hash_width: int) -> tuple[str, ...]:
    """Expose the bounded DDL vocabulary for static safety tests."""
    if hash_width not in {16, 32}:
        raise ValueError("hash_width must be 16 or 32")
    dialect = postgresql.dialect()  # type: ignore[no-untyped-call]
    statements = (
        str(CreateTable(session_schema_identity_table).compile(dialect=dialect)),
        str(CreateTable(landscape_schema_identity_table).compile(dialect=dialect)),
    )
    if hash_width == 16:
        return (*statements, _WIDEN_HASH_SQL)
    return statements


def select_migration_plan(session_state: SessionState, landscape_state: LandscapeState) -> MigrationPlan:
    """Select the only safe action for an exact two-database state pair."""
    if session_state is SessionState.RELEASE_0_7_0 and landscape_state in {
        LandscapeState.RELEASE_0_7_0_WIDTH_16,
        LandscapeState.RELEASE_0_7_0_WIDTH_32,
    }:
        return MigrationPlan.APPLY_BOTH
    if session_state is SessionState.CURRENT and landscape_state in {
        LandscapeState.RELEASE_0_7_0_WIDTH_16,
        LandscapeState.RELEASE_0_7_0_WIDTH_32,
    }:
        return MigrationPlan.RESUME_LANDSCAPE
    if session_state is SessionState.CURRENT and landscape_state is LandscapeState.CURRENT:
        return MigrationPlan.ALREADY_APPLIED
    raise MigrationFailure(
        ResultCode.PRECONDITION_FAILED,
        session_state=session_state,
        landscape_state=landscape_state,
    )


def _pre_identity_shape_issues(connection: Connection, expected: MetaData) -> tuple[SchemaShapeIssue, ...] | None:
    inspector = inspect(connection)
    actual_tables = set(inspector.get_table_names())
    expected_tables = set(expected.tables) - {SCHEMA_IDENTITY_TABLE_NAME}
    if actual_tables != expected_tables:
        return None
    return collect_metadata_shape_issues(
        inspector,
        expected,
        dialect=connection.dialect,
        present_tables=actual_tables,
    )


def _function_body(function_sql: str) -> str:
    marker = "AS $$"
    if marker not in function_sql:
        raise AssertionError("PostgreSQL audit function has no AS $$ body")
    return function_sql.split(marker, 1)[1].rsplit("$$", 1)[0].strip()


def _target_function_rows(connection: Connection) -> tuple[tuple[str, str, str, str, str], ...]:
    target_names = {entry.function_name for entry in POSTGRESQL_AUDIT_DDL_COHORT}
    rows = connection.execute(
        text(
            """
            SELECT procedure.proname,
                   pg_catalog.pg_get_function_identity_arguments(procedure.oid),
                   pg_catalog.format_type(procedure.prorettype, NULL),
                   language.lanname,
                   procedure.prosrc
            FROM pg_catalog.pg_proc AS procedure
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = procedure.pronamespace
            JOIN pg_catalog.pg_language AS language ON language.oid = procedure.prolang
            WHERE namespace.nspname = current_schema()
            ORDER BY procedure.proname,
                     pg_catalog.pg_get_function_identity_arguments(procedure.oid)
            """
        )
    )
    return tuple(
        (str(name), str(arguments), str(result), str(language), str(body).strip())
        for name, arguments, result, language, body in rows
        if str(name) in target_names
    )


def _expected_function_rows() -> tuple[tuple[str, str, str, str, str], ...]:
    return tuple(
        sorted(
            (
                entry.function_name,
                "",
                "trigger",
                "plpgsql",
                _function_body(entry.function_sql),
            )
            for entry in POSTGRESQL_AUDIT_DDL_COHORT
        )
    )


def _trigger_rows(connection: Connection) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (str(name), str(table_name), str(function_name), str(enabled))
        for name, table_name, function_name, enabled in connection.execute(
            text(
                """
                SELECT trigger.tgname, relation.relname, procedure.proname,
                       trigger.tgenabled::text
                FROM pg_catalog.pg_trigger AS trigger
                JOIN pg_catalog.pg_class AS relation ON relation.oid = trigger.tgrelid
                JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                JOIN pg_catalog.pg_proc AS procedure ON procedure.oid = trigger.tgfoid
                WHERE NOT trigger.tgisinternal
                  AND namespace.nspname = current_schema()
                ORDER BY trigger.tgname, relation.relname
                """
            )
        )
    )


def _target_triggers_match(connection: Connection) -> bool:
    actual = _trigger_rows(connection)
    expected = {(entry.trigger_name, entry.table.name, entry.function_name) for entry in POSTGRESQL_AUDIT_DDL_COHORT}
    return (
        len(actual) == len(expected)
        and {(name, table_name, function_name) for name, table_name, function_name, _enabled in actual} == expected
        and all(enabled in {"O", "A"} for _name, _table, _function, enabled in actual)
    )


def classify_session_state(connection: Connection) -> SessionState:
    """Classify only the supported release-0.7.0 or exact target shape."""
    if connection.dialect.name != "postgresql":
        return SessionState.INVALID
    functions = _target_function_rows(connection)
    triggers = _trigger_rows(connection)
    if probe_session_schema(connection) is SchemaState.CURRENT:
        if functions == _expected_function_rows() and _target_triggers_match(connection):
            return SessionState.CURRENT
        return SessionState.INVALID
    if functions or triggers:
        return SessionState.INVALID
    issues = _pre_identity_shape_issues(connection, session_metadata)
    if issues == ():
        return SessionState.RELEASE_0_7_0
    return SessionState.INVALID


def _landscape_hash_width(connection: Connection) -> int | None:
    value = connection.execute(
        text(
            """
            SELECT character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'run_sources'
              AND column_name = 'schema_contract_hash'
            """
        )
    ).scalar_one_or_none()
    return value if type(value) is int else None


def classify_landscape_state(connection: Connection) -> LandscapeState:
    """Classify only epoch 23 at width 16/32 or the exact epoch-24 target."""
    if connection.dialect.name != "postgresql":
        return LandscapeState.INVALID
    if probe_landscape_schema(connection) is SchemaState.CURRENT:
        return LandscapeState.CURRENT
    width = _landscape_hash_width(connection)
    if width not in {16, 32}:
        return LandscapeState.INVALID
    issues = _pre_identity_shape_issues(connection, landscape_metadata)
    if issues is None:
        return LandscapeState.INVALID
    if width == 16:
        if len(issues) != 1:
            return LandscapeState.INVALID
        issue = issues[0]
        if (
            issue.subject != "run_sources.schema_contract_hash type mismatch"
            or issue.expected != "VARCHAR(32)"
            or issue.actual != "VARCHAR(16)"
        ):
            return LandscapeState.INVALID
        return LandscapeState.RELEASE_0_7_0_WIDTH_16
    return LandscapeState.RELEASE_0_7_0_WIDTH_32 if issues == () else LandscapeState.INVALID


def _sqlstate(exc: DBAPIError) -> str | None:
    original = exc.orig
    value = getattr(original, "sqlstate", None)
    if value is None:
        value = getattr(original, "pgcode", None)
    return value if isinstance(value, str) else None


def _invalidate_uncertain(connection: Connection) -> None:
    with suppress(BaseException):
        connection.invalidate()


def _acquire_lock(connection: Connection) -> None:
    try:
        connection.execute(
            text("SELECT pg_catalog.set_config('lock_timeout', :timeout, true)"),
            {"timeout": _LOCK_TIMEOUT},
        )
        connection.execute(
            text("SELECT pg_catalog.pg_advisory_lock(:classid, pg_catalog.hashtext(:target))"),
            {"classid": ELSPETH_SCHEMA_INIT_LOCK_CLASSID, "target": _LOCK_TARGET},
        )
        connection.commit()
    except OperationalError as exc:
        _invalidate_uncertain(connection)
        code = ResultCode.LOCK_BUSY if _sqlstate(exc) == "55P03" else ResultCode.DATABASE_ERROR
        raise MigrationFailure(code) from None
    except BaseException:
        _invalidate_uncertain(connection)
        raise


def _release_lock(connection: Connection) -> None:
    if connection.in_transaction():
        connection.rollback()
    unlocked = connection.execute(
        text("SELECT pg_catalog.pg_advisory_unlock(:classid, pg_catalog.hashtext(:target))"),
        {"classid": ELSPETH_SCHEMA_INIT_LOCK_CLASSID, "target": _LOCK_TARGET},
    ).scalar_one()
    if unlocked is not True:
        raise RuntimeError("advisory unlock was not confirmed")
    if connection.in_transaction():
        connection.rollback()


@contextmanager
def _both_schema_locks(session: Connection, landscape: Connection) -> Iterator[None]:
    acquired: list[Connection] = []
    earlier: BaseException | None = None
    try:
        _acquire_lock(session)
        acquired.append(session)
        _acquire_lock(landscape)
        acquired.append(landscape)
        yield
    except BaseException as exc:
        earlier = exc
        raise
    finally:
        cleanup_failed = False
        for connection in reversed(acquired):
            try:
                _release_lock(connection)
            except BaseException:
                cleanup_failed = True
                _invalidate_uncertain(connection)
        if cleanup_failed and earlier is None:
            raise MigrationFailure(ResultCode.LOCK_CLEANUP_UNVERIFIED)


def _apply_session(connection: Connection) -> None:
    with connection.begin():
        session_schema_identity_table.create(bind=connection, checkfirst=False)
        insert_schema_identity(
            connection,
            session_schema_identity_table,
            store_kind="session",
            schema_epoch=SESSION_SCHEMA_EPOCH,
        )
        for entry in POSTGRESQL_AUDIT_DDL_COHORT:
            connection.execute(text(entry.function_sql))
            connection.execute(text(entry.trigger_sql))


def _apply_landscape(connection: Connection, state: LandscapeState) -> None:
    with connection.begin():
        if state is LandscapeState.RELEASE_0_7_0_WIDTH_16:
            connection.execute(text(_WIDEN_HASH_SQL))
        landscape_schema_identity_table.create(bind=connection, checkfirst=False)
        insert_schema_identity(
            connection,
            landscape_schema_identity_table,
            store_kind="landscape",
            schema_epoch=SQLITE_SCHEMA_EPOCH,
        )


def _raise_database_failure(exc: SQLAlchemyError) -> NoReturn:
    if isinstance(exc, DBAPIError) and _sqlstate(exc) == "42501":
        raise MigrationFailure(ResultCode.PERMISSION_DENIED) from None
    raise MigrationFailure(ResultCode.DATABASE_ERROR) from None


def _run_locked_migration(session: Connection, landscape: Connection) -> MigrationSummary:
    with _both_schema_locks(session, landscape):
        session_state = classify_session_state(session)
        landscape_state = classify_landscape_state(landscape)
        if session.in_transaction():
            session.rollback()
        if landscape.in_transaction():
            landscape.rollback()
        plan = select_migration_plan(session_state, landscape_state)

        if plan is MigrationPlan.ALREADY_APPLIED:
            return MigrationSummary(
                status=ResultStatus.SUCCEEDED,
                code=ResultCode.ALREADY_APPLIED,
                already_applied=True,
                session_state=SessionState.CURRENT,
                landscape_state=LandscapeState.CURRENT,
            )

        if plan is MigrationPlan.APPLY_BOTH:
            _apply_session(session)
        _apply_landscape(landscape, landscape_state)

        verified_session = classify_session_state(session)
        verified_landscape = classify_landscape_state(landscape)
        if verified_session is not SessionState.CURRENT or verified_landscape is not LandscapeState.CURRENT:
            raise MigrationFailure(
                ResultCode.POST_VERIFY_FAILED,
                session_state=verified_session,
                landscape_state=verified_landscape,
            )
        if session.in_transaction():
            session.rollback()
        if landscape.in_transaction():
            landscape.rollback()
        return MigrationSummary(
            status=ResultStatus.SUCCEEDED,
            code=ResultCode.MIGRATION_APPLIED,
            already_applied=False,
            session_state=verified_session,
            landscape_state=verified_landscape,
        )


def run_migration(session_url: str, landscape_url: str) -> MigrationSummary:
    """Run the bounded migration while holding both database-scoped locks."""
    try:
        postgres_logical_target_key(session_url)
        postgres_logical_target_key(landscape_url)
        require_distinct_postgres_targets(session_url, landscape_url)
    except DatabaseTargetConflictError:
        raise MigrationFailure(ResultCode.TARGET_CONFLICT) from None

    session_engine: Engine | None = None
    landscape_engine: Engine | None = None
    try:
        session_engine = create_session_engine(session_url, **postgres_engine_kwargs(session_url))
        landscape_engine = create_engine(landscape_url, **postgres_engine_kwargs(landscape_url))
        with session_engine.connect() as session, landscape_engine.connect() as landscape:
            return _run_locked_migration(session, landscape)
    except MigrationFailure:
        raise
    except SQLAlchemyError as exc:
        _raise_database_failure(exc)
    finally:
        if session_engine is not None:
            session_engine.dispose()
        if landscape_engine is not None:
            landscape_engine.dispose()


def _failure_summary(failure: MigrationFailure) -> MigrationSummary:
    return MigrationSummary(
        status=ResultStatus.FAILED,
        code=failure.code,
        already_applied=False,
        session_state=failure.session_state,
        landscape_state=failure.landscape_state,
    )


def _write_summary(stdout: IO[str], summary: MigrationSummary) -> None:
    stdout.write(json.dumps(summary.to_dict(), sort_keys=True, separators=(",", ":")))
    stdout.write("\n")


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    stdout: IO[str] | None = None,
) -> int:
    """Execute with an exact flag and environment-only owner credentials."""
    args = tuple(sys.argv[1:] if argv is None else argv)
    environment = os.environ if environ is None else environ
    output = sys.stdout if stdout is None else stdout

    if args != ("--apply",):
        code = ResultCode.APPLY_REQUIRED if not args else ResultCode.INVALID_ARGUMENTS
        _write_summary(output, _failure_summary(MigrationFailure(code)))
        return 2

    session_url = environment.get(SESSION_OWNER_URL_ENV)
    landscape_url = environment.get(LANDSCAPE_OWNER_URL_ENV)
    if not session_url or not landscape_url:
        _write_summary(output, _failure_summary(MigrationFailure(ResultCode.ENVIRONMENT_REQUIRED)))
        return 2

    try:
        summary = run_migration(session_url, landscape_url)
    except MigrationFailure as failure:
        _write_summary(output, _failure_summary(failure))
        return 1
    except BaseException:
        _write_summary(output, _failure_summary(MigrationFailure(ResultCode.INTERNAL_ERROR)))
        return 1
    _write_summary(output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
