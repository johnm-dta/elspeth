"""Real PostgreSQL CLI proofs for ``elspeth doctor aws-ecs``."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]
from typer.testing import CliRunner

from elspeth.cli import app
from elspeth.web.schema_probe import SchemaState, probe_landscape_schema, probe_session_schema

pytestmark = pytest.mark.testcontainer

_SAFE_IDENTIFIER = re.compile(r"[a-z0-9_]+\Z")
_PROCESS_TIMEOUT_SECONDS = 120.0
_PROCESS_STOP_TIMEOUT_SECONDS = 5.0


def _identifier(prefix: str) -> str:
    value = f"{prefix}_{uuid.uuid4().hex}"
    assert _SAFE_IDENTIFIER.fullmatch(value)
    return value


def _render_url(base_url: str | URL, *, database: str) -> str:
    return make_url(base_url).set(database=database).render_as_string(hide_password=False)


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


@dataclass(frozen=True, slots=True)
class _DatabasePair:
    session_url: str
    landscape_url: str


@pytest.fixture
def database_pair(postgres_url: str) -> Iterator[_DatabasePair]:
    session_database = _identifier("doctor_session")
    landscape_database = _identifier("doctor_landscape")
    assert session_database != landscape_database

    admin = create_engine(postgres_url)
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.exec_driver_sql(f'CREATE DATABASE "{session_database}"')
        connection.exec_driver_sql(f'CREATE DATABASE "{landscape_database}"')

    try:
        yield _DatabasePair(
            session_url=_render_url(postgres_url, database=session_database),
            landscape_url=_render_url(postgres_url, database=landscape_database),
        )
    finally:
        with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            connection.exec_driver_sql(f'DROP DATABASE "{session_database}" WITH (FORCE)')
            connection.exec_driver_sql(f'DROP DATABASE "{landscape_database}" WITH (FORCE)')
        admin.dispose()


@pytest.fixture(autouse=True)
def _clear_inherited_web_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("ELSPETH_WEB__"):
            monkeypatch.delenv(key, raising=False)


def _doctor_environment(tmp_path: Path, databases: _DatabasePair) -> dict[str, str]:
    data_dir = tmp_path / "data"
    payload_dir = tmp_path / "payloads"
    for directory in (data_dir, data_dir / "blobs", payload_dir):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)

    environment = {key: value for key, value in os.environ.items() if not key.startswith("ELSPETH_WEB__")}
    environment.update(
        {
            "ELSPETH_WEB__DEPLOYMENT_TARGET": "aws-ecs",
            "ELSPETH_WEB__SESSION_DB_URL": databases.session_url,
            "ELSPETH_WEB__LANDSCAPE_URL": databases.landscape_url,
            "ELSPETH_WEB__DATA_DIR": str(data_dir),
            "ELSPETH_WEB__PAYLOAD_STORE_PATH": str(payload_dir),
            "ELSPETH_WEB__HOST": "0.0.0.0",
            "ELSPETH_WEB__SECRET_KEY": "doctor-aws-ecs-secret-0123456789-abcdefghijklmnopqrstuvwxyz",
            "ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY": base64.b64encode(bytes(range(32))).decode("ascii"),
            "ELSPETH_WEB__OPERATOR_TELEMETRY": "aws-otlp",
            "ELSPETH_WEB__OPERATOR_TELEMETRY_SERVICE_NAME": "elspeth-web",
            "ELSPETH_WEB__OPERATOR_TELEMETRY_ENVIRONMENT": "test",
            "ELSPETH_WEB__OPERATOR_TELEMETRY_RELEASE": "git-test",
            "ELSPETH_WEB__OPERATOR_TELEMETRY_ECS_CLUSTER": "elspeth-test",
            "ELSPETH_WEB__OPERATOR_TELEMETRY_ECS_SERVICE": "elspeth-web",
            "ELSPETH_WEB__OPERATOR_TELEMETRY_TASK_DEFINITION_FAMILY": "elspeth-web-task",
            "ELSPETH_WEB__OPERATOR_TELEMETRY_TASK_DEFINITION_REVISION": "1",
            "ELSPETH_WEB__OPERATOR_TELEMETRY_EXPORT_INTERVAL_SECONDS": "60",
            "ELSPETH_WEB__OPERATOR_PIPELINE_TELEMETRY_GRANULARITY": "lifecycle",
            "ELSPETH_WEB__COMPOSER_MAX_COMPOSITION_TURNS": "15",
            "ELSPETH_WEB__COMPOSER_MAX_DISCOVERY_TURNS": "10",
            "ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS": "85.0",
            "ELSPETH_WEB__COMPOSER_RATE_LIMIT_PER_MINUTE": "10",
            "ELSPETH_WEB__COMPOSER_BOOT_PROBE_ENABLED": "false",
        }
    )
    return environment


def _assert_all_green_report(stdout: str) -> list[dict[str, Any]]:
    payload = json.loads(stdout)
    assert isinstance(payload, list)
    assert payload
    assert all(isinstance(item, dict) for item in payload)
    report: list[dict[str, Any]] = payload
    names = [item.get("name") for item in report]
    assert all(isinstance(name, str) for name in names)
    assert len(names) == len(set(names))
    assert all(item.get("ok") is True for item in report)
    return report


def _assert_schemas_current(databases: _DatabasePair) -> None:
    session_engine = create_engine(databases.session_url)
    landscape_engine = create_engine(databases.landscape_url)
    try:
        assert probe_session_schema(session_engine) is SchemaState.CURRENT
        assert probe_landscape_schema(landscape_engine) is SchemaState.CURRENT
    finally:
        session_engine.dispose()
        landscape_engine.dispose()


def _assert_private_database_values_absent(
    output: str,
    databases: _DatabasePair,
) -> None:
    for url in (databases.session_url, databases.landscape_url):
        parsed = make_url(url)
        private_values = (url, parsed.username, parsed.password, parsed.host, parsed.database)
        for value in private_values:
            if value:
                assert value not in output


def test_doctor_init_schema_cli_succeeds_against_fresh_postgres(
    tmp_path: Path,
    database_pair: _DatabasePair,
) -> None:
    environment = _doctor_environment(tmp_path, database_pair)

    result = CliRunner().invoke(
        app,
        ["--no-dotenv", "doctor", "aws-ecs", "--init-schema", "--json"],
        env=environment,
    )

    assert result.exit_code == 0, result.output
    _assert_all_green_report(result.stdout)
    _assert_private_database_values_absent(result.stdout + result.stderr, database_pair)
    _assert_schemas_current(database_pair)


def _stop_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)


def test_concurrent_doctor_init_schema_cli_runs_are_safe(
    tmp_path: Path,
    database_pair: _DatabasePair,
) -> None:
    environment = _doctor_environment(tmp_path, database_pair)
    command = [
        sys.executable,
        "-m",
        "elspeth.cli",
        "--no-dotenv",
        "doctor",
        "aws-ecs",
        "--init-schema",
        "--json",
    ]
    processes = [
        subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    completed: list[tuple[int, str, str]] = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=_PROCESS_TIMEOUT_SECONDS)
            completed.append((process.returncode, stdout, stderr))

        for returncode, stdout, stderr in completed:
            assert returncode == 0, stderr or stdout
            _assert_all_green_report(stdout)
            _assert_private_database_values_absent(stdout + stderr, database_pair)
        _assert_schemas_current(database_pair)
    finally:
        _stop_processes(processes)
