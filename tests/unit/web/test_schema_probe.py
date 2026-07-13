"""Fail-closed database target, schema-state, and initialization tests."""

from __future__ import annotations

from types import MappingProxyType

import pytest
from sqlalchemy import create_engine, inspect, text

from elspeth.core.landscape.database import SchemaCompatibilityError
from elspeth.core.landscape.schema import metadata as landscape_metadata
from elspeth.web.schema_probe import (
    AWS_ECS_POOL_KWARGS,
    DatabaseTargetConflictError,
    SchemaState,
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
