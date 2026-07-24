"""Fail-closed, validate-only startup gates for the AWS ECS web target."""

from __future__ import annotations

import stat
import time
from collections.abc import Callable
from pathlib import Path

import structlog
from sqlalchemy import Connection, Engine, create_engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from elspeth.core.landscape.database import SchemaCompatibilityError
from elspeth.web.config import WebSettings
from elspeth.web.deployment_contract import validate_aws_ecs_settings
from elspeth.web.paths import managed_blob_directory
from elspeth.web.schema_probe import (
    DatabaseTargetConflictError,
    SchemaState,
    postgres_engine_kwargs,
    probe_landscape_schema,
    probe_session_schema,
    require_distinct_postgres_targets,
)
from elspeth.web.sessions.schema import SessionSchemaError

_slog = structlog.get_logger(__name__)

_CONNECT_TIMEOUT_SECONDS = 10
_CONNECT_RETRY_BUDGET_SECONDS = 45
_INITIAL_RETRY_BACKOFF_SECONDS = 1.0
_MAX_RETRY_BACKOFF_SECONDS = 8.0
_DOCTOR_GUIDANCE = "Run 'elspeth doctor aws-ecs' for full diagnostics."


class AwsEcsStartupContractError(RuntimeError):
    """Raised when the static AWS ECS startup contract is not satisfied."""


class AwsEcsSchemaNotReadyError(RuntimeError):
    """Raised when a required database cannot prove its current schema."""


def _contract_error(detail: str) -> AwsEcsStartupContractError:
    return AwsEcsStartupContractError(f"{detail} {_DOCTOR_GUIDANCE}")


def _schema_error(label: str) -> AwsEcsSchemaNotReadyError:
    return AwsEcsSchemaNotReadyError(f"AWS ECS {label} is not ready and startup repair is disabled. {_DOCTOR_GUIDANCE}")


def enforce_aws_ecs_contract(settings: WebSettings) -> None:
    """Enforce Plan 01 settings and Plan 02 logical-target separation."""
    checks = validate_aws_ecs_settings(settings)
    failed_names = [check.name for check in checks if not check.ok]
    if failed_names:
        raise _contract_error(
            "AWS ECS deployment settings failed checks: "
            f"{', '.join(failed_names)}. Set the named ELSPETH_WEB environment variables to valid production values."
        )

    session_url = settings.session_db_url
    landscape_url = settings.landscape_url
    assert session_url is not None
    assert landscape_url is not None
    try:
        require_distinct_postgres_targets(session_url, landscape_url)
    except DatabaseTargetConflictError:
        raise _contract_error(
            "AWS ECS session and Landscape database targets must be statically provable as distinct database targets."
        ) from None


def _require_existing_directory(path: Path, *, label: str, env_var: str) -> None:
    try:
        directory_stat = path.lstat()
    except (OSError, RuntimeError, ValueError):
        raise _contract_error(f"AWS ECS runtime directory {label} ({env_var}) is missing or invalid.") from None
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise _contract_error(f"AWS ECS runtime directory {label} ({env_var}) is missing or invalid.")


def _require_safe_payload_directory(path: Path) -> None:
    label = "payload_store"
    env_var = "ELSPETH_WEB__PAYLOAD_STORE_PATH"
    try:
        directory_stat = path.lstat()
        if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
            raise ValueError("invalid directory shape")
        if directory_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError("unsafe directory mode")
        path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise _contract_error(f"AWS ECS runtime directory {label} ({env_var}) is missing or invalid.") from None


def require_runtime_directories_mounted(settings: WebSettings) -> None:
    """Require every AWS ECS runtime directory without creating anything."""
    _require_existing_directory(settings.data_dir, label="data_dir", env_var="ELSPETH_WEB__DATA_DIR")

    raw_payload_path = settings.payload_store_path
    if raw_payload_path is None:
        raise _contract_error("AWS ECS runtime directory payload_store (ELSPETH_WEB__PAYLOAD_STORE_PATH) is missing or invalid.")
    _require_safe_payload_directory(raw_payload_path.expanduser())

    try:
        blob_root = managed_blob_directory(str(settings.data_dir))
    except (OSError, RuntimeError, ValueError):
        raise _contract_error("AWS ECS runtime directory blob (derived from ELSPETH_WEB__DATA_DIR) is missing or invalid.") from None
    _require_existing_directory(blob_root, label="blob", env_var="ELSPETH_WEB__DATA_DIR")


def _probe_with_connection_budget(
    engine: Engine,
    probe: Callable[[Engine | Connection], SchemaState],
    *,
    label: str,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> SchemaState:
    """Probe on one connection per attempt within the connection-failure budget."""
    started = monotonic()
    deadline = started + _CONNECT_RETRY_BUDGET_SECONDS
    attempt = 0
    backoff = _INITIAL_RETRY_BACKOFF_SECONDS

    while True:
        attempt += 1
        try:
            with engine.connect() as conn:
                return probe(conn)
        except OperationalError as exc:
            now = monotonic()
            _slog.warning(
                "aws_ecs_schema_probe_retry",
                label=label,
                attempt=attempt,
                elapsed_seconds=max(0.0, now - started),
                exc_class=type(exc).__name__,
            )
            remaining = deadline - now
            if remaining <= 0 or remaining < _CONNECT_TIMEOUT_SECONDS:
                raise _schema_error(label) from None

            sleep_for = min(backoff, remaining - _CONNECT_TIMEOUT_SECONDS)
            sleep(sleep_for)
            remaining = deadline - monotonic()
            if remaining <= 0 or remaining < _CONNECT_TIMEOUT_SECONDS:
                raise _schema_error(label) from None
            backoff = min(backoff * 2, _MAX_RETRY_BACKOFF_SECONDS)
        except (SQLAlchemyError, SessionSchemaError, SchemaCompatibilityError):
            raise _schema_error(label) from None


def validate_only_schema_or_raise(settings: WebSettings, session_engine: Engine) -> None:
    """Require current session and Landscape schemas without running DDL."""
    session_state = _probe_with_connection_budget(
        session_engine,
        probe_session_schema,
        label="session_schema",
    )
    if session_state is not SchemaState.CURRENT:
        raise _schema_error("session_schema")

    raw_landscape_url = settings.landscape_url
    assert raw_landscape_url is not None
    try:
        landscape_engine = create_engine(
            raw_landscape_url,
            connect_args={"connect_timeout": _CONNECT_TIMEOUT_SECONDS},
            **postgres_engine_kwargs(raw_landscape_url),
        )
    except SQLAlchemyError:
        raise _schema_error("landscape_schema") from None

    try:
        landscape_state = _probe_with_connection_budget(
            landscape_engine,
            probe_landscape_schema,
            label="landscape_schema",
        )
        if landscape_state is not SchemaState.CURRENT:
            raise _schema_error("landscape_schema")
    finally:
        landscape_engine.dispose()
