"""PostgreSQL 16 proofs for bounded, redacted AWS ECS readiness."""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql
from pydantic import SecretBytes
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL, make_url
from structlog.testing import capture_logs
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

import elspeth.web.readiness as readiness_module
from elspeth.web.app import create_app
from elspeth.web.config import WebSettings
from elspeth.web.readiness import READINESS_CHECK_NAMES
from elspeth.web.schema_probe import (
    SchemaState,
    init_landscape_schema,
    init_session_schema,
    probe_landscape_schema,
    probe_session_schema,
)
from elspeth.web.sessions.engine import create_session_engine

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

    def set_login(self, *, enabled: bool) -> None:
        with _psycopg_connect(self.postgres_url) as admin:
            state = sql.SQL("LOGIN") if enabled else sql.SQL("NOLOGIN")
            admin.execute(sql.SQL("ALTER ROLE {} {}").format(sql.Identifier(self.runtime_role), state))
            if not enabled:
                admin.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename = %s AND pid <> pg_backend_pid()",
                    (self.runtime_role,),
                )


@pytest.fixture
def runtime_databases(postgres_url: str) -> Iterator[_RuntimeDatabases]:
    databases = _RuntimeDatabases(
        postgres_url=postgres_url,
        session_database=_identifier("ready_session"),
        landscape_database=_identifier("ready_landscape"),
        runtime_role=_identifier("ready_runtime"),
        runtime_password=f"runtime-{uuid.uuid4().hex}",
    )
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


def _settings(tmp_path: Path, databases: _RuntimeDatabases) -> WebSettings:
    data_dir = tmp_path / "data"
    payload_dir = tmp_path / "payload"
    for directory in (data_dir, data_dir / "blobs", payload_dir):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
    return WebSettings(
        deployment_target="aws-ecs",
        operator_telemetry="aws-otlp",
        operator_telemetry_environment="test",
        operator_telemetry_release="git-test",
        operator_telemetry_ecs_cluster="elspeth-test",
        operator_telemetry_ecs_service="elspeth-web",
        operator_telemetry_task_definition_family="elspeth-web-task",
        operator_telemetry_task_definition_revision="1",
        host="0.0.0.0",
        data_dir=data_dir,
        payload_store_path=payload_dir,
        session_db_url=databases.session_runtime_url,
        landscape_url=databases.landscape_runtime_url,
        secret_key="s" * 40,
        shareable_link_signing_key=SecretBytes(bytes(range(32))),
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        composer_boot_probe_enabled=False,
    )


def _initialized_app(tmp_path: Path, databases: _RuntimeDatabases) -> tuple[object, Engine, Engine]:
    session_owner = create_session_engine(databases.session_owner_url)
    landscape_owner = create_engine(databases.landscape_owner_url)
    init_session_schema(session_owner)
    init_landscape_schema(landscape_owner)
    databases.provision_runtime_role()
    return create_app(_settings(tmp_path, databases)), session_owner, landscape_owner


def _dispose_app(app: object) -> None:
    state = app.state  # type: ignore[attr-defined]
    state.readiness_probe_runner.close()
    state.session_engine.dispose()


def test_ready_returns_200_for_current_postgres(tmp_path: Path, runtime_databases: _RuntimeDatabases) -> None:
    app, session_owner, landscape_owner = _initialized_app(tmp_path, runtime_databases)
    try:
        response = TestClient(app).get("/api/ready")
        assert response.status_code == 200
        payload = response.json()
        assert payload["ready"] is True
        assert [check["name"] for check in payload["checks"]] == list(READINESS_CHECK_NAMES)
        assert len({check["name"] for check in payload["checks"]}) == 8
        assert all(check["ok"] for check in payload["checks"])
        assert probe_session_schema(session_owner) is SchemaState.CURRENT
        assert probe_landscape_schema(landscape_owner) is SchemaState.CURRENT
    finally:
        _dispose_app(app)
        session_owner.dispose()
        landscape_owner.dispose()


def test_ready_returns_bounded_redacted_503_after_connect_revocation(
    tmp_path: Path,
    runtime_databases: _RuntimeDatabases,
) -> None:
    app, session_owner, landscape_owner = _initialized_app(tmp_path, runtime_databases)
    runtime_databases.set_login(enabled=False)
    try:
        with pytest.raises(psycopg.OperationalError):
            _psycopg_connect(runtime_databases.session_runtime_url)

        started = time.monotonic()
        with capture_logs() as logs:
            response = TestClient(app).get("/api/ready")
        elapsed = time.monotonic() - started

        assert response.status_code == 503
        assert elapsed < 5.0
        by_name = {check["name"]: check for check in response.json()["checks"]}
        for kind in ("session", "landscape"):
            assert by_name[f"{kind}_db"]["ok"] is False
            assert by_name[f"{kind}_schema"]["ok"] is False
            assert by_name[f"{kind}_schema"]["detail"] == "not checked: connectivity probe failed"
        rendered = repr((response.text, logs))
        assert runtime_databases.runtime_role not in rendered
        assert runtime_databases.runtime_password not in rendered
        assert runtime_databases.session_runtime_url not in rendered
        assert runtime_databases.landscape_runtime_url not in rendered
        assert "SELECT 1" not in rendered
        assert "psycopg" not in rendered.lower()
    finally:
        runtime_databases.set_login(enabled=True)
        _dispose_app(app)
        session_owner.dispose()
        landscape_owner.dispose()


def test_repeated_failed_refreshes_do_not_grow_probe_work(
    tmp_path: Path,
    runtime_databases: _RuntimeDatabases,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, session_owner, landscape_owner = _initialized_app(tmp_path, runtime_databases)
    original = readiness_module._check_session_database
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def held(*args: object) -> tuple[readiness_module.ReadinessCheck, ...]:
        nonlocal calls
        calls += 1
        entered.set()
        release.wait()
        return original(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(readiness_module, "_check_session_database", held)
    client = TestClient(app)
    try:
        first = client.get("/api/ready")
        assert entered.is_set()
        assert first.status_code == 503
        assert calls == 1
        assert "session" in app.state.readiness_probe_runner._futures

        for _ in range(10):
            app.state.readiness_cache._completed_at = float("-inf")
            response = client.get("/api/ready")
            assert response.status_code == 503
            assert calls == 1
            assert len(app.state.readiness_probe_runner._futures) <= 5
            assert "session" in app.state.readiness_probe_runner._futures

        release.set()
        deadline = time.monotonic() + 5.0
        while "session" in app.state.readiness_probe_runner._futures and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "session" not in app.state.readiness_probe_runner._futures
        app.state.readiness_cache._completed_at = float("-inf")
        recovered = client.get("/api/ready")
        assert recovered.status_code == 200
        assert calls == 2
    finally:
        release.set()
        _dispose_app(app)
        session_owner.dispose()
        landscape_owner.dispose()
