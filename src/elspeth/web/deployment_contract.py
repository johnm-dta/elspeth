"""Pure AWS ECS deployment contract validator. No I/O, no network, no
filesystem. ContractCheck.detail is pre-redacted -- never a URL, path,
or secret value."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.engine.url import make_url

from elspeth.web.config import (
    WebSettings,
    is_default_secret_key_placeholder,
    is_undersized_secret_key,
    is_uniform_byte_key,
)

DEPLOYMENT_TARGET_AWS_ECS = "aws-ecs"
_POSTGRES_DRIVER = "postgresql"
# This IS the container-serving check, not a bind-all bug; exact-match only --
# IPv6 dual-stack ("::") is out of scope for this milestone.
_CONTAINER_SERVING_HOST = "0.0.0.0"


@dataclass(frozen=True)
class ContractCheck:
    name: str
    ok: bool
    detail: str


def _check_postgres_url(name: str, env_var: str, url: str | None) -> ContractCheck:
    if url is None:
        return ContractCheck(name, False, f"{env_var} is required in aws-ecs deployment mode")
    driver_parts = make_url(url).drivername.split("+")
    is_postgres = driver_parts == [_POSTGRES_DRIVER] or (
        len(driver_parts) == 2 and driver_parts[0] == _POSTGRES_DRIVER and bool(driver_parts[1])
    )
    if not is_postgres:
        return ContractCheck(
            name,
            False,
            f"{env_var} must use a PostgreSQL SQLAlchemy scheme; no fallback scheme is permitted in aws-ecs mode",
        )
    return ContractCheck(name, True, f"{env_var} uses a PostgreSQL scheme")


def _check_required_path(name: str, env_var: str, value: Path | None, *, explicitly_set: bool) -> ContractCheck:
    # `explicitly_set` is `field_name in settings.model_fields_set` at the
    # call site. `payload_store_path` truly defaults to None, so a bare
    # None-check would already catch its omission -- but `data_dir` defaults
    # to `Path("data")` (never None, never blank), so without this flag an
    # omitted ELSPETH_WEB__DATA_DIR would silently report ok=True. Checking
    # explicit-construction presence closes that gap for both fields
    # uniformly.
    if not explicitly_set or value is None or not str(value).strip():
        return ContractCheck(
            name,
            False,
            f"{env_var} is required and must not be blank in aws-ecs deployment mode",
        )
    return ContractCheck(name, True, f"{env_var} is set")


def validate_aws_ecs_settings(settings: WebSettings) -> list[ContractCheck]:
    """Run every strict AWS ECS contract check, including deployment target."""
    target_ok = settings.deployment_target == DEPLOYMENT_TARGET_AWS_ECS
    checks = [
        ContractCheck(
            "deployment_target",
            target_ok,
            "deployment_target is aws-ecs" if target_ok else "ELSPETH_WEB__DEPLOYMENT_TARGET must be aws-ecs",
        ),
        _check_postgres_url("session_db_url", "ELSPETH_WEB__SESSION_DB_URL", settings.session_db_url),
        _check_postgres_url("landscape_url", "ELSPETH_WEB__LANDSCAPE_URL", settings.landscape_url),
        _check_required_path(
            "data_dir",
            "ELSPETH_WEB__DATA_DIR",
            settings.data_dir,
            explicitly_set="data_dir" in settings.model_fields_set,
        ),
        _check_required_path(
            "payload_store_path",
            "ELSPETH_WEB__PAYLOAD_STORE_PATH",
            settings.payload_store_path,
            explicitly_set="payload_store_path" in settings.model_fields_set,
        ),
        ContractCheck(
            "operator_telemetry",
            settings.operator_telemetry == "aws-otlp",
            (
                "operator telemetry uses the task-local OTLP collector"
                if settings.operator_telemetry == "aws-otlp"
                else "ELSPETH_WEB__OPERATOR_TELEMETRY must be aws-otlp in aws-ecs deployment mode"
            ),
        ),
        ContractCheck(
            "operator_telemetry_environment",
            settings.operator_telemetry_environment is not None,
            (
                "operator telemetry environment is set"
                if settings.operator_telemetry_environment is not None
                else "ELSPETH_WEB__OPERATOR_TELEMETRY_ENVIRONMENT is required in aws-ecs deployment mode"
            ),
        ),
    ]
    host_ok = settings.host == _CONTAINER_SERVING_HOST
    checks.append(
        ContractCheck(
            "host",
            host_ok,
            (
                "host is suitable for container serving"
                if host_ok
                else f"ELSPETH_WEB__HOST must be {_CONTAINER_SERVING_HOST} for container/ALB reachability"
            ),
        )
    )
    # Outside pytest-local-host construction, WebSettings' own boot guards
    # reject weak secret_key/signing_key values during model construction.
    # Therefore doctor can emit these named checks only after settings load;
    # invalid non-local values surface through Plan 03's generic settings_load
    # ValidationError diagnostic. The explicit checks still cover successfully
    # constructed settings, including the pytest-local-host bypass fixtures.
    secret_ok = not (is_default_secret_key_placeholder(settings.secret_key) or is_undersized_secret_key(settings.secret_key))
    checks.append(
        ContractCheck(
            "secret_key",
            secret_ok,
            (
                "secret_key is production-shaped"
                if secret_ok
                else "ELSPETH_WEB__SECRET_KEY must not be the default placeholder and must meet the length floor"
            ),
        )
    )
    key_ok = not is_uniform_byte_key(settings.shareable_link_signing_key.get_secret_value())
    checks.append(
        ContractCheck(
            "shareable_link_signing_key",
            key_ok,
            (
                "shareable_link_signing_key is production-shaped"
                if key_ok
                else "ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY is a known-weak uniform-byte placeholder"
            ),
        )
    )
    return checks
