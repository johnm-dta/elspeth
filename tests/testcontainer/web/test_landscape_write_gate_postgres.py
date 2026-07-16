"""PostgreSQL proof that request-time Landscape writers require no DDL."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi import Request
from psycopg import sql
from pydantic import SecretBytes
from sqlalchemy import Engine, create_engine, insert, inspect, select, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ProgrammingError
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from elspeth.contracts import NodeType
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import SchemaCompatibilityError
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table, scheduler_events_table, token_work_items_table, tokens_table
from elspeth.web.auth.audit import AuthAuditRecorder
from elspeth.web.config import WebSettings
from elspeth.web.deployment_contract import validate_aws_ecs_settings
from elspeth.web.execution.accounting import load_run_accounting_from_db
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


def test_postgres_scheduler_enqueue_and_accounting_projection_are_dialect_safe(
    tmp_path: Path,
    runtime_database: _RuntimeDatabase,
) -> None:
    """A real psycopg enqueue succeeds even when INSERT rowcount is unknown."""
    settings = _settings(tmp_path, runtime_database)
    now = datetime.now(UTC)
    payload = TokenSchedulerRepository.serialize_row_payload(
        PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
    )

    with open_landscape_db(settings) as landscape:
        with landscape.engine.begin() as conn:
            conn.execute(
                insert(runs_table).values(
                    run_id="scheduler-postgres-run",
                    started_at=now,
                    config_hash="config",
                    settings_json="{}",
                    canonical_version="v1",
                    status="running",
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )
            conn.execute(
                insert(nodes_table),
                [
                    {
                        "run_id": "scheduler-postgres-run",
                        "node_id": "source",
                        "plugin_name": "csv",
                        "node_type": NodeType.SOURCE.value,
                        "plugin_version": "1.0",
                        "determinism": "deterministic",
                        "config_hash": "config",
                        "config_json": "{}",
                        "registered_at": now,
                    },
                    {
                        "run_id": "scheduler-postgres-run",
                        "node_id": "transform",
                        "plugin_name": "llm",
                        "node_type": NodeType.TRANSFORM.value,
                        "plugin_version": "1.0",
                        "determinism": "nondeterministic",
                        "config_hash": "config",
                        "config_json": "{}",
                        "registered_at": now,
                    },
                ],
            )
            conn.execute(
                insert(rows_table).values(
                    row_id="row-1",
                    run_id="scheduler-postgres-run",
                    source_node_id="source",
                    row_index=0,
                    source_row_index=0,
                    ingest_sequence=0,
                    source_data_hash="hash-row-1",
                    created_at=now,
                )
            )
            conn.execute(
                insert(tokens_table).values(
                    token_id="token-1",
                    row_id="row-1",
                    run_id="scheduler-postgres-run",
                    created_at=now,
                )
            )

        scheduler = TokenSchedulerRepository(landscape.engine)
        item = scheduler.enqueue_ready(
            run_id="scheduler-postgres-run",
            token_id="token-1",
            row_id="row-1",
            node_id="transform",
            step_index=1,
            ingest_sequence=0,
            available_at=now,
            row_payload_json=payload,
        )
        duplicate = scheduler.enqueue_ready(
            run_id="scheduler-postgres-run",
            token_id="token-1",
            row_id="row-1",
            node_id="transform",
            step_index=1,
            ingest_sequence=0,
            available_at=now,
            row_payload_json=payload,
        )
        assert duplicate.work_item_id == item.work_item_id
        accounting = load_run_accounting_from_db(landscape, landscape_run_id="scheduler-postgres-run")
        assert accounting.source.rows_processed == 1
        assert accounting.sources["source"].rows_processed == 1

        with landscape.engine.connect() as conn:
            assert (
                conn.execute(
                    select(token_work_items_table.c.work_item_id).where(token_work_items_table.c.run_id == "scheduler-postgres-run")
                ).scalar_one()
                == item.work_item_id
            )
            assert (
                conn.execute(
                    select(scheduler_events_table.c.work_item_id).where(scheduler_events_table.c.run_id == "scheduler-postgres-run")
                ).scalar_one()
                == item.work_item_id
            )


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
