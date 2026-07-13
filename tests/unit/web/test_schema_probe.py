"""Fail-closed database target, schema-state, and initialization tests."""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError
from structlog.testing import capture_logs

from elspeth.core.landscape.database import SchemaCompatibilityError
from elspeth.core.landscape.schema import metadata as landscape_metadata
from elspeth.web import schema_probe as schema_probe_module
from elspeth.web.schema_probe import (
    AWS_ECS_POOL_KWARGS,
    DatabaseTargetConflictError,
    SchemaLockCleanupError,
    SchemaState,
    _run_locked,
    init_landscape_schema,
    init_session_schema,
    postgres_engine_kwargs,
    postgres_logical_target_key,
    probe_landscape_schema,
    probe_session_schema,
    require_distinct_postgres_targets,
)
from elspeth.web.sessions.models import metadata as session_metadata
from elspeth.web.sessions.schema import SessionSchemaError

_SENTINEL = "opaque-sentinel-secret SELECT raw_secret FROM vault"


class _ScalarResult:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar_one(self) -> object:
        return self._value


class _FakeConnection:
    def __init__(
        self,
        *,
        acquisition_error: BaseException | None = None,
        unlock_error: BaseException | None = None,
        unlock_value: object = True,
        rollback_fail_at: int | None = None,
    ) -> None:
        self.dialect = SimpleNamespace(name="postgresql")
        self.acquisition_error = acquisition_error
        self.unlock_error = unlock_error
        self.unlock_value = unlock_value
        self.rollback_fail_at = rollback_fail_at
        self.transaction_active = False
        self.rollback_calls = 0
        self.invalidated = False

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, statement: object, _params: object = None) -> _ScalarResult:
        sql = str(statement)
        if "pg_advisory_lock" in sql and "unlock" not in sql and self.acquisition_error is not None:
            raise self.acquisition_error
        if "pg_advisory_unlock" in sql:
            if self.unlock_error is not None:
                raise self.unlock_error
            self.transaction_active = True
            return _ScalarResult(self.unlock_value)
        return _ScalarResult(None)

    def in_transaction(self) -> bool:
        return self.transaction_active

    def commit(self) -> None:
        self.transaction_active = False

    def rollback(self) -> None:
        self.rollback_calls += 1
        if self.rollback_calls == self.rollback_fail_at:
            raise RuntimeError(_SENTINEL)
        self.transaction_active = False

    def invalidate(self) -> None:
        self.invalidated = True


class _FakeEngine:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection

    def connect(self) -> _FakeConnection:
        return self.connection


def _assert_redacted(value: object) -> None:
    rendered = repr(value)
    assert "sentinel-secret" not in rendered
    assert "raw_secret" not in rendered
    assert "vault" not in rendered


def _run_fake(
    connection: _FakeConnection,
    *,
    body: Callable[[Any], None] | None = None,
    verify: Callable[[Any], None] | None = None,
) -> None:
    _run_locked(
        _FakeEngine(connection),  # type: ignore[arg-type]
        target="elspeth_schema_init",
        body=body or (lambda _conn: None),
        verify=verify or (lambda _conn: None),
    )


def test_pool_kwargs_are_postgres_only_and_fresh() -> None:
    first = postgres_engine_kwargs("postgresql+psycopg://db.example/audit")
    second = postgres_engine_kwargs("postgresql://db.example/audit")
    assert first == {"pool_size": 5, "max_overflow": 5, "pool_pre_ping": True}
    assert first == second
    assert first is not second
    first["pool_size"] = 1
    assert second["pool_size"] == 5
    assert postgres_engine_kwargs("sqlite:///audit.db") == {}
    assert isinstance(AWS_ECS_POOL_KWARGS, MappingProxyType)


@pytest.mark.parametrize("driver", ["postgresql", "postgresql+psycopg", "postgresql+psycopg2"])
def test_logical_target_normalizes_postgres_driver_host_and_port(driver: str) -> None:
    target = postgres_logical_target_key(f"{driver}://user:ignored@DB.EXAMPLE/audit")
    assert target.host == "db.example"
    assert target.port == 5432
    assert target.database == "audit"
    assert target.explicit_schema is None


@pytest.mark.parametrize(
    ("options", "schema"),
    [
        ("-csearch_path=Foo", "foo"),
        ("-c search_path=foo_2", "foo_2"),
    ],
)
def test_logical_target_parses_single_explicit_schema(options: str, schema: str) -> None:
    target = postgres_logical_target_key(f"postgresql+psycopg://host/audit?options={options}")
    assert target.explicit_schema == schema


@pytest.mark.parametrize(
    "url",
    [
        "sqlite:///audit.db",
        "postgresql+psycopg:///audit",
        "postgresql+psycopg://host/",
        "postgresql+psycopg://host/audit?options=-csearch_path=foo,public",
        "postgresql+psycopg://host/audit?options=-csearch_path=%22Foo%22",
        "postgresql+psycopg://host/audit?options=-csearch_path=$user",
        "postgresql+psycopg://host/audit?options=-csearch_path=foo%20-csearch_path=bar",
    ],
)
def test_unprovable_target_is_rejected_with_static_message(url: str) -> None:
    with pytest.raises(DatabaseTargetConflictError) as exc_info:
        postgres_logical_target_key(url)
    assert str(exc_info.value) == "PostgreSQL database target cannot be proven safe from static URL configuration."
    assert "audit" not in str(exc_info.value)


def test_distinct_servers_pass_without_schema_options() -> None:
    require_distinct_postgres_targets("postgresql://one/audit", "postgresql://two/audit")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("postgresql://host/audit", "postgresql://host/audit"),
        ("postgresql://host/audit", "postgresql://host/audit?options=-csearch_path=public"),
        (
            "postgresql://host/audit?options=-csearch_path=Foo",
            "postgresql://host/audit?options=-csearch_path=foo",
        ),
    ],
)
def test_same_database_unproven_or_equal_schema_fails(left: str, right: str) -> None:
    with pytest.raises(DatabaseTargetConflictError):
        require_distinct_postgres_targets(left, right)


def test_same_database_distinct_explicit_schemas_pass() -> None:
    require_distinct_postgres_targets(
        "postgresql://host/audit?options=-csearch_path=sessions",
        "postgresql://host/audit?options=-csearch_path=landscape",
    )


def test_empty_sqlite_targets_are_missing() -> None:
    engine = create_engine("sqlite:///:memory:")
    assert probe_session_schema(engine) is SchemaState.MISSING
    assert probe_landscape_schema(engine) is SchemaState.MISSING
    engine.dispose()


def test_session_foreign_partial_and_current_states() -> None:
    foreign = create_engine("sqlite:///:memory:")
    with foreign.begin() as conn:
        conn.execute(text("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)"))
    assert probe_session_schema(foreign) is SchemaState.STALE

    partial = create_engine("sqlite:///:memory:")
    next(iter(session_metadata.tables.values())).create(partial)
    assert probe_session_schema(partial) is SchemaState.STALE

    current = create_engine("sqlite:///:memory:")
    init_session_schema(current)
    assert probe_session_schema(current) is SchemaState.CURRENT
    foreign.dispose()
    partial.dispose()
    current.dispose()


def test_landscape_additive_gap_is_partial_and_initializer_repairs_it() -> None:
    engine = create_engine("sqlite:///:memory:")
    landscape_metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP INDEX ix_tokens_run_id")
    assert probe_landscape_schema(engine) is SchemaState.PARTIAL
    init_landscape_schema(engine)
    assert probe_landscape_schema(engine) is SchemaState.CURRENT
    engine.dispose()


def test_initializers_refuse_stale_nonempty_targets_without_mutation() -> None:
    session = create_engine("sqlite:///:memory:")
    landscape = create_engine("sqlite:///:memory:")
    for engine in (session, landscape):
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)"))
    with pytest.raises(SessionSchemaError):
        init_session_schema(session)
    with pytest.raises(SchemaCompatibilityError):
        init_landscape_schema(landscape)
    assert inspect(session).get_table_names() == ["unrelated"]
    assert inspect(landscape).get_table_names() == ["unrelated"]
    session.dispose()
    landscape.dispose()


def test_session_tableless_foreign_sentinels_are_stale() -> None:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA application_id = 123456")
        conn.exec_driver_sql("PRAGMA user_version = 654321")
    assert probe_session_schema(engine) is SchemaState.STALE
    engine.dispose()


def test_session_initializer_verifies_noop_create_all(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    monkeypatch.setattr(session_metadata, "create_all", lambda **_kwargs: None)
    with pytest.raises(SessionSchemaError, match="did not produce the current schema"):
        init_session_schema(engine)
    engine.dispose()


def test_session_initializer_is_noop_when_schema_is_current(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    init_session_schema(engine)

    def unexpected_create_all(**_kwargs: object) -> None:
        pytest.fail("create_all must not run for a current schema")

    monkeypatch.setattr(session_metadata, "create_all", unexpected_create_all)
    init_session_schema(engine)
    engine.dispose()


def test_locked_body_and_verify_share_the_same_connection() -> None:
    engine = create_engine("sqlite:///:memory:")
    seen: list[int] = []
    _run_locked(
        engine,
        target="test",
        body=lambda conn: seen.append(id(conn)),
        verify=lambda conn: seen.append(id(conn)),
    )
    assert len(seen) == 2
    assert seen[0] == seen[1]
    engine.dispose()


def _earlier_body(conn: Any) -> None:
    conn.transaction_active = True
    raise KeyboardInterrupt(_SENTINEL)


@pytest.mark.parametrize("with_earlier", [False, True])
def test_pre_unlock_rollback_failure_is_redacted_and_preserves_earlier(with_earlier: bool) -> None:
    conn = _FakeConnection(rollback_fail_at=1)
    body: Callable[[Any], None]
    if with_earlier:
        body = _earlier_body
        expected: type[BaseException] = KeyboardInterrupt
    else:

        def body(_candidate: Any) -> None:
            return None

        expected = SchemaLockCleanupError
    with capture_logs() as logs, pytest.raises(expected) as exc_info:
        _run_fake(conn, body=body, verify=lambda candidate: setattr(candidate, "transaction_active", True))
    assert conn.invalidated
    if not with_earlier:
        _assert_redacted(exc_info.value)
    _assert_redacted(logs)


@pytest.mark.parametrize("with_earlier", [False, True])
def test_post_unlock_rollback_failure_is_redacted_and_preserves_earlier(with_earlier: bool) -> None:
    conn = _FakeConnection(rollback_fail_at=1)

    def body(_conn: Any) -> None:
        if with_earlier:
            raise KeyboardInterrupt(_SENTINEL)

    expected: type[BaseException] = KeyboardInterrupt if with_earlier else SchemaLockCleanupError
    with capture_logs() as logs, pytest.raises(expected) as exc_info:
        _run_fake(conn, body=body)
    assert conn.invalidated
    if not with_earlier:
        _assert_redacted(exc_info.value)
    _assert_redacted(logs)


@pytest.mark.parametrize("mode", ["false", "exception", "interrupt"])
@pytest.mark.parametrize("with_earlier", [False, True])
def test_unlock_failure_invalidates_redacts_and_preserves_earlier(mode: str, with_earlier: bool) -> None:
    kwargs: dict[str, object] = {}
    if mode == "false":
        kwargs["unlock_value"] = False
    elif mode == "exception":
        kwargs["unlock_error"] = RuntimeError(_SENTINEL)
    else:
        kwargs["unlock_error"] = KeyboardInterrupt(_SENTINEL)
    conn = _FakeConnection(**kwargs)

    def body(_conn: Any) -> None:
        if with_earlier:
            raise KeyboardInterrupt(_SENTINEL)

    expected: type[BaseException] = KeyboardInterrupt if with_earlier else SchemaLockCleanupError
    with capture_logs() as logs, pytest.raises(expected) as exc_info:
        _run_fake(conn, body=body)
    assert conn.invalidated
    if not with_earlier:
        _assert_redacted(exc_info.value)
    _assert_redacted(logs)


class _OriginalDatabaseError(RuntimeError):
    sqlstate = "08006"


def test_non_busy_lock_operational_error_is_static_redacted_and_invalidates() -> None:
    raw = OperationalError(
        "SELECT raw_secret FROM vault",
        {"password": "sentinel-secret"},
        _OriginalDatabaseError(_SENTINEL),
    )
    conn = _FakeConnection(acquisition_error=raw)
    with capture_logs() as logs, pytest.raises(RuntimeError) as exc_info:
        _run_fake(conn)
    assert type(exc_info.value) is schema_probe_module.SchemaInitError
    assert exc_info.value.__cause__ is raw
    assert conn.invalidated
    _assert_redacted(exc_info.value)
    _assert_redacted(logs)
