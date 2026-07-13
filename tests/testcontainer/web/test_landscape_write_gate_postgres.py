"""PostgreSQL proof that request-time Landscape writers require no DDL."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi import Request
from psycopg import sql
from pydantic import SecretBytes
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ProgrammingError
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from elspeth.core.landscape.database import SchemaCompatibilityError
from elspeth.web.auth.audit import AuthAuditRecorder
from elspeth.web.config import WebSettings
from elspeth.web.deployment_contract import validate_aws_ecs_settings
from elspeth.web.landscape_access import open_landscape_db
from elspeth.web.schema_probe import SchemaState, init_landscape_schema, probe_landscape_schema

pytestmark = pytest.mark.testcontainer

_SAFE_IDENTIFIER = re.compile(r"[a-z0-9_]+\Z")


def _identifier(prefix: str) -> str:
    value = f"{prefix}_{uuid.uuid4().hex}"
    assert _SAFE_IDENTIFIER.fullmatch(value)
    return value


def _render_url(
    base_url: str | URL,
    *,
    database: str,
    username: str | None = None,
    password: str | None = None,
) -> str:
    parsed = make_url(base_url).set(database=database)
    if username is not None:
        parsed = parsed.set(username=username, password=password)
    return parsed.render_as_string(hide_password=False)


def _connect(url: str) -> psycopg.Connection[Any]:
    parsed = make_url(url)
    assert parsed.host is not None
    assert parsed.username is not None
    assert parsed.database is not None
    return psycopg.connect(
        host=parsed.host,
        port=parsed.port or 5432,
        dbname=parsed.database,
        user=parsed.username,
        password=parsed.password,
        autocommit=True,
    )


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


@dataclass(frozen=True, slots=True)
class _RuntimeDatabase:
    session_runtime_url: str
    landscape_runtime_url: str
    owner_engine: Engine
    runtime_engine: Engine


@pytest.fixture
def runtime_database(postgres_url: str) -> Iterator[_RuntimeDatabase]:
    session_database = _identifier("session")
    landscape_database = _identifier("landscape")
    runtime_role = _identifier("runtime")
    runtime_password = f"runtime-{uuid.uuid4().hex}"
    owner_url = _render_url(postgres_url, database=landscape_database)
    session_runtime_url = _render_url(
        postgres_url,
        database=session_database,
        username=runtime_role,
        password=runtime_password,
    )
    landscape_runtime_url = _render_url(
        postgres_url,
        database=landscape_database,
        username=runtime_role,
        password=runtime_password,
    )

    with _connect(postgres_url) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(session_database)))
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(landscape_database)))
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(runtime_role),
                sql.Literal(runtime_password),
            )
        )
        for database in (session_database, landscape_database):
            admin.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                    sql.Identifier(database),
                    sql.Identifier(runtime_role),
                )
            )

    owner_engine = create_engine(owner_url)
    init_landscape_schema(owner_engine)
    with _connect(owner_url) as owner:
        owner.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        owner.execute(sql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(sql.Identifier(runtime_role)))
        owner.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(runtime_role)))
        owner.execute(sql.SQL("GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO {}").format(sql.Identifier(runtime_role)))
        owner.execute(sql.SQL("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {}").format(sql.Identifier(runtime_role)))
    runtime_engine = create_engine(landscape_runtime_url)
    database = _RuntimeDatabase(
        session_runtime_url=session_runtime_url,
        landscape_runtime_url=landscape_runtime_url,
        owner_engine=owner_engine,
        runtime_engine=runtime_engine,
    )
    try:
        yield database
    finally:
        runtime_engine.dispose()
        owner_engine.dispose()
        with _connect(postgres_url) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(session_database)))
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(landscape_database)))
            admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(runtime_role)))


def _settings(tmp_path: Path, database: _RuntimeDatabase) -> WebSettings:
    data_dir = tmp_path / "data"
    payload_dir = tmp_path / "payloads"
    for directory in (data_dir, data_dir / "blobs", payload_dir):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
    settings = WebSettings(
        deployment_target="aws-ecs",
        operator_telemetry="aws-otlp",
        operator_telemetry_environment="test",
        operator_telemetry_release="git-test",
        operator_telemetry_ecs_cluster="elspeth-test",
        operator_telemetry_ecs_service="elspeth-web",
        operator_telemetry_task_definition_family="elspeth-web-task",
        operator_telemetry_task_definition_revision="1",
        host="0.0.0.0",
        session_db_url=database.session_runtime_url,
        landscape_url=database.landscape_runtime_url,
        landscape_passphrase=None,
        data_dir=data_dir,
        payload_store_path=payload_dir,
        secret_key="production-shaped-test-secret-material-40",
        shareable_link_signing_key=SecretBytes(bytes(range(32))),
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        composer_boot_probe_enabled=False,
    )
    assert all(check.ok for check in validate_aws_ecs_settings(settings))
    return settings


def _request(request_id: str) -> Request:
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/auth/me",
            "headers": [(b"user-agent", b"bounded-test-agent")],
            "client": ("127.0.0.1", 12345),
        }
    )
    request.state.request_id = request_id
    return request


def _record_bounded_event(settings: WebSettings, *, request_id: str) -> None:
    AuthAuditRecorder.from_settings(settings).record_auth_failure(
        _request(request_id),
        provider="oidc",
        failure_category="invalid_token",
        failure_stage="authenticate",
        user_id=None,
        username=None,
        exception_class="AuthenticationError",
    )


def _catalog_identity(engine: Engine) -> tuple[tuple[object, ...], ...]:
    statements = (
        """
        SELECT 'table', c.relname, c.relkind::text
        FROM pg_catalog.pg_class AS c
        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p')
        """,
        """
        SELECT 'index', i.tablename, i.indexname, i.indexdef
        FROM pg_catalog.pg_indexes AS i
        WHERE i.schemaname = 'public'
        """,
        """
        SELECT 'constraint', c.relname, con.conname, con.contype::text,
               pg_catalog.pg_get_constraintdef(con.oid, true)
        FROM pg_catalog.pg_constraint AS con
        JOIN pg_catalog.pg_class AS c ON c.oid = con.conrelid
        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
        """,
    )
    rows: list[tuple[object, ...]] = []
    with engine.connect() as conn:
        for statement in statements:
            rows.extend(tuple(row) for row in conn.execute(text(statement)))
    return tuple(sorted(rows, key=repr))


def _assert_ddl_denied(engine: Engine) -> None:
    with pytest.raises(ProgrammingError) as exc_info, engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE ddl_must_be_denied (id integer primary key)")
    assert isinstance(exc_info.value.orig, psycopg.errors.InsufficientPrivilege)
    assert exc_info.value.orig.sqlstate == "42501"


def test_aws_ecs_factory_and_auth_writer_succeed_without_ddl(
    tmp_path: Path,
    runtime_database: _RuntimeDatabase,
) -> None:
    _assert_ddl_denied(runtime_database.runtime_engine)
    settings = _settings(tmp_path, runtime_database)
    before = _catalog_identity(runtime_database.owner_engine)

    with open_landscape_db(settings):
        pass
    _record_bounded_event(settings, request_id="current-schema-event")

    with runtime_database.runtime_engine.connect() as conn:
        event_count = conn.execute(
            text("SELECT count(*) FROM auth_events WHERE request_id = :request_id"),
            {"request_id": "current-schema-event"},
        ).scalar_one()
    assert event_count == 1
    assert _catalog_identity(runtime_database.owner_engine) == before


def test_request_open_does_not_repair_missing_additive_index(
    tmp_path: Path,
    runtime_database: _RuntimeDatabase,
) -> None:
    original = _catalog_identity(runtime_database.owner_engine)
    with runtime_database.owner_engine.begin() as conn:
        conn.exec_driver_sql("DROP INDEX ix_tokens_run_id")
    assert probe_landscape_schema(runtime_database.owner_engine) is SchemaState.PARTIAL
    before_request = _catalog_identity(runtime_database.owner_engine)
    assert before_request != original
    settings = _settings(tmp_path, runtime_database)

    with open_landscape_db(settings):
        pass
    _record_bounded_event(settings, request_id="missing-index-event")

    assert "ix_tokens_run_id" not in {index["name"] for index in inspect(runtime_database.owner_engine).get_indexes("tokens")}
    assert _catalog_identity(runtime_database.owner_engine) == before_request


def test_request_open_rejects_missing_additive_table_without_repair(
    tmp_path: Path,
    runtime_database: _RuntimeDatabase,
) -> None:
    with runtime_database.owner_engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE run_attributions")
    assert probe_landscape_schema(runtime_database.owner_engine) is SchemaState.PARTIAL
    before_request = _catalog_identity(runtime_database.owner_engine)
    settings = _settings(tmp_path, runtime_database)

    with pytest.raises(SchemaCompatibilityError):
        open_landscape_db(settings)
    with pytest.raises(SchemaCompatibilityError):
        _record_bounded_event(settings, request_id="missing-table-event")

    assert "run_attributions" not in inspect(runtime_database.owner_engine).get_table_names()
    assert _catalog_identity(runtime_database.owner_engine) == before_request
