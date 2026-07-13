"""PostgreSQL proof that AWS ECS web startup is validate-only and DDL-free."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql
from pydantic import SecretBytes
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ProgrammingError
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from elspeth.web.app import create_app
from elspeth.web.aws_ecs_startup import AwsEcsSchemaNotReadyError, AwsEcsStartupContractError
from elspeth.web.config import WebSettings
from elspeth.web.schema_probe import (
    SchemaState,
    init_landscape_schema,
    init_session_schema,
    probe_landscape_schema,
    probe_session_schema,
)

pytestmark = pytest.mark.testcontainer

_SAFE_IDENTIFIER = re.compile(r"[a-z0-9_]+\Z")


def _identifier(prefix: str) -> str:
    value = f"{prefix}_{uuid.uuid4().hex}"
    assert _SAFE_IDENTIFIER.fullmatch(value)
    return value


def _render_url(base_url: str | URL, *, database: str, username: str | None = None, password: str | None = None) -> str:
    parsed = make_url(base_url).set(database=database)
    if username is not None:
        parsed = parsed.set(username=username, password=password)
    return parsed.render_as_string(hide_password=False)


def _psycopg_connect(url: str) -> psycopg.Connection[Any]:
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


@dataclass
class _RuntimeDatabases:
    postgres_url: str
    session_database: str
    landscape_database: str
    runtime_role: str
    runtime_password: str
    role_created: bool = False

    @property
    def session_owner_url(self) -> str:
        return _render_url(self.postgres_url, database=self.session_database)

    @property
    def landscape_owner_url(self) -> str:
        return _render_url(self.postgres_url, database=self.landscape_database)

    @property
    def session_runtime_url(self) -> str:
        assert self.role_created
        return _render_url(
            self.postgres_url,
            database=self.session_database,
            username=self.runtime_role,
            password=self.runtime_password,
        )

    @property
    def landscape_runtime_url(self) -> str:
        assert self.role_created
        return _render_url(
            self.postgres_url,
            database=self.landscape_database,
            username=self.runtime_role,
            password=self.runtime_password,
        )

    def provision_runtime_role(self) -> None:
        assert self.role_created is False
        with _psycopg_connect(self.postgres_url) as admin:
            admin.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(self.runtime_role),
                    sql.Literal(self.runtime_password),
                )
            )
            for database in (self.session_database, self.landscape_database):
                admin.execute(
                    sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                        sql.Identifier(database),
                        sql.Identifier(self.runtime_role),
                    )
                )

        for owner_url in (self.session_owner_url, self.landscape_owner_url):
            with _psycopg_connect(owner_url) as owner:
                owner.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
                owner.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(self.runtime_role)))
                owner.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(sql.Identifier(self.runtime_role)))
        self.role_created = True


@pytest.fixture
def runtime_databases(postgres_url: str) -> Iterator[_RuntimeDatabases]:
    databases = _RuntimeDatabases(
        postgres_url=postgres_url,
        session_database=_identifier("session"),
        landscape_database=_identifier("landscape"),
        runtime_role=_identifier("runtime"),
        runtime_password=f"runtime-{uuid.uuid4().hex}",
    )
    assert databases.session_database != databases.landscape_database
    with _psycopg_connect(postgres_url) as admin:
        for database in (databases.session_database, databases.landscape_database):
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))

    try:
        yield databases
    finally:
        with _psycopg_connect(postgres_url) as admin:
            for database in (databases.session_database, databases.landscape_database):
                admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))
            if databases.role_created:
                admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(databases.runtime_role)))


def _settings(tmp_path: Path, *, session_url: str, landscape_url: str) -> WebSettings:
    data_dir = tmp_path / "data"
    payload_dir = tmp_path / "payload"
    for directory in (data_dir, data_dir / "blobs", payload_dir):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
    return WebSettings(
        deployment_target="aws-ecs",
        host="0.0.0.0",
        data_dir=data_dir,
        payload_store_path=payload_dir,
        session_db_url=session_url,
        landscape_url=landscape_url,
        secret_key="s" * 40,
        shareable_link_signing_key=SecretBytes(bytes(range(32))),
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        composer_boot_probe_enabled=False,
    )


def _assert_ddl_denied(url: str) -> None:
    engine = create_engine(url)
    try:
        with pytest.raises(ProgrammingError) as exc_info, engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE ddl_must_be_denied (id integer primary key)")
        assert isinstance(exc_info.value.orig, psycopg.errors.InsufficientPrivilege)
        assert exc_info.value.orig.sqlstate == "42501"
    finally:
        engine.dispose()


def _assert_runtime_ddl_denied(databases: _RuntimeDatabases) -> None:
    _assert_ddl_denied(databases.session_runtime_url)
    _assert_ddl_denied(databases.landscape_runtime_url)


def _catalog_identity(engine: Engine) -> tuple[tuple[object, ...], ...]:
    statements = (
        """
        SELECT 'table', c.relname, c.relkind::text
        FROM pg_catalog.pg_class AS c
        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p')
        ORDER BY c.relname
        """,
        """
        SELECT 'column', c.relname, a.attname,
               pg_catalog.format_type(a.atttypid, a.atttypmod),
               a.attnotnull::text, a.attidentity::text, a.attgenerated::text,
               COALESCE(pg_catalog.pg_get_expr(d.adbin, d.adrelid), '')
        FROM pg_catalog.pg_attribute AS a
        JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid
        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
        LEFT JOIN pg_catalog.pg_attrdef AS d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
        WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p')
          AND a.attnum > 0 AND NOT a.attisdropped
        ORDER BY c.relname, a.attnum
        """,
        """
        SELECT 'index', tablename, indexname, indexdef
        FROM pg_catalog.pg_indexes
        WHERE schemaname = 'public'
        ORDER BY tablename, indexname
        """,
        """
        SELECT 'constraint', c.relname, con.conname, con.contype::text,
               pg_catalog.pg_get_constraintdef(con.oid, true)
        FROM pg_catalog.pg_constraint AS con
        JOIN pg_catalog.pg_class AS c ON c.oid = con.conrelid
        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
        ORDER BY c.relname, con.conname
        """,
    )
    rows: list[tuple[object, ...]] = []
    with engine.connect() as conn:
        for statement in statements:
            rows.extend(tuple(row) for row in conn.execute(text(statement)))
    return tuple(rows)


def _assert_redacted(exc: BaseException, databases: _RuntimeDatabases) -> None:
    rendered = repr(exc)
    assert databases.runtime_password not in rendered
    assert databases.session_runtime_url not in rendered
    assert databases.landscape_runtime_url not in rendered
    assert "CREATE TABLE" not in rendered


def test_aws_ecs_startup_fails_closed_on_missing_session_schema(
    tmp_path: Path,
    runtime_databases: _RuntimeDatabases,
) -> None:
    runtime_databases.provision_runtime_role()
    _assert_runtime_ddl_denied(runtime_databases)
    session_owner = create_engine(runtime_databases.session_owner_url)
    landscape_owner = create_engine(runtime_databases.landscape_owner_url)
    try:
        before_session = _catalog_identity(session_owner)
        before_landscape = _catalog_identity(landscape_owner)
        settings = _settings(
            tmp_path,
            session_url=runtime_databases.session_runtime_url,
            landscape_url=runtime_databases.landscape_runtime_url,
        )

        with pytest.raises(AwsEcsSchemaNotReadyError, match="session_schema") as exc_info:
            create_app(settings)

        _assert_redacted(exc_info.value, runtime_databases)
        assert _catalog_identity(session_owner) == before_session == ()
        assert _catalog_identity(landscape_owner) == before_landscape == ()
    finally:
        session_owner.dispose()
        landscape_owner.dispose()


def test_aws_ecs_startup_fails_closed_when_only_landscape_schema_missing(
    tmp_path: Path,
    runtime_databases: _RuntimeDatabases,
) -> None:
    session_owner = create_engine(runtime_databases.session_owner_url)
    landscape_owner = create_engine(runtime_databases.landscape_owner_url)
    try:
        init_session_schema(session_owner)
        runtime_databases.provision_runtime_role()
        _assert_runtime_ddl_denied(runtime_databases)
        before_session = _catalog_identity(session_owner)
        before_landscape = _catalog_identity(landscape_owner)
        settings = _settings(
            tmp_path,
            session_url=runtime_databases.session_runtime_url,
            landscape_url=runtime_databases.landscape_runtime_url,
        )

        with pytest.raises(AwsEcsSchemaNotReadyError, match="landscape_schema") as exc_info:
            create_app(settings)

        _assert_redacted(exc_info.value, runtime_databases)
        assert _catalog_identity(session_owner) == before_session
        assert _catalog_identity(landscape_owner) == before_landscape == ()
    finally:
        session_owner.dispose()
        landscape_owner.dispose()


def test_aws_ecs_startup_succeeds_with_current_schema_under_ddl_denied_roles(
    tmp_path: Path,
    runtime_databases: _RuntimeDatabases,
) -> None:
    session_owner = create_engine(runtime_databases.session_owner_url)
    landscape_owner = create_engine(runtime_databases.landscape_owner_url)
    app = None
    try:
        init_session_schema(session_owner)
        init_landscape_schema(landscape_owner)
        runtime_databases.provision_runtime_role()
        _assert_runtime_ddl_denied(runtime_databases)
        before_session = _catalog_identity(session_owner)
        before_landscape = _catalog_identity(landscape_owner)
        settings = _settings(
            tmp_path,
            session_url=runtime_databases.session_runtime_url,
            landscape_url=runtime_databases.landscape_runtime_url,
        )

        app = create_app(settings)

        assert _catalog_identity(session_owner) == before_session
        assert _catalog_identity(landscape_owner) == before_landscape
        session_runtime = create_engine(runtime_databases.session_runtime_url)
        landscape_runtime = create_engine(runtime_databases.landscape_runtime_url)
        try:
            assert probe_session_schema(session_runtime) is SchemaState.CURRENT
            assert probe_landscape_schema(landscape_runtime) is SchemaState.CURRENT
        finally:
            session_runtime.dispose()
            landscape_runtime.dispose()
    finally:
        if app is not None:
            app.state.session_engine.dispose()
        session_owner.dispose()
        landscape_owner.dispose()


@pytest.mark.parametrize("target_shape", ["same", "ambiguous"])
def test_aws_ecs_startup_rejects_same_or_ambiguous_target_before_connecting(
    tmp_path: Path,
    runtime_databases: _RuntimeDatabases,
    target_shape: str,
) -> None:
    runtime_databases.provision_runtime_role()
    _assert_runtime_ddl_denied(runtime_databases)
    session_url = runtime_databases.session_runtime_url
    if target_shape == "same":
        landscape_url = session_url
    else:
        landscape_url = (
            make_url(session_url).update_query_dict({"options": "-csearch_path=sessions,public"}).render_as_string(hide_password=False)
        )
    settings = _settings(tmp_path, session_url=session_url, landscape_url=landscape_url)

    with pytest.raises(AwsEcsStartupContractError, match="database targets") as exc_info:
        create_app(settings)

    _assert_redacted(exc_info.value, runtime_databases)
    assert "DBAPI" not in str(exc_info.value)
    owner = create_engine(runtime_databases.session_owner_url)
    try:
        assert inspect(owner).get_table_names() == []
    finally:
        owner.dispose()
