"""Tests for the pure AWS ECS deployment contract validator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from elspeth.web.config import WebSettings
from elspeth.web.deployment_contract import ContractCheck, validate_aws_ecs_settings


def _base_kwargs() -> dict[str, Any]:
    return {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "secret_key": "a" * 40,
        "shareable_link_signing_key": bytes(range(32)),
    }


def _checks(**overrides: Any) -> dict[str, ContractCheck]:
    settings = WebSettings(**(_base_kwargs() | overrides))
    return {check.name: check for check in validate_aws_ecs_settings(settings)}


def test_default_deployment_target_fails() -> None:
    assert _checks()["deployment_target"].ok is False


def test_aws_ecs_deployment_target_passes() -> None:
    assert _checks(deployment_target="aws-ecs")["deployment_target"].ok is True


def test_missing_session_db_url_fails() -> None:
    assert _checks()["session_db_url"].ok is False


def test_missing_landscape_url_fails() -> None:
    assert _checks()["landscape_url"].ok is False


def test_sqlite_session_db_url_rejected() -> None:
    assert _checks(session_db_url="sqlite:///x.db")["session_db_url"].ok is False


def test_sqlite_landscape_url_rejected() -> None:
    assert _checks(landscape_url="sqlite:///x.db")["landscape_url"].ok is False


def test_postgresql_psycopg_driver_accepted() -> None:
    assert _checks(session_db_url="postgresql+psycopg://u:p@host/db")["session_db_url"].ok is True


@pytest.mark.parametrize(
    ("url", "operator_fragment"),
    [
        ("postgresql+://host/db", "postgresql+"),
        ("postgresql+psycopg+extra://host/db", "psycopg+extra"),
    ],
)
def test_malformed_postgresql_driver_rejected_and_redacted(url: str, operator_fragment: str) -> None:
    check = _checks(session_db_url=url)["session_db_url"]

    assert check.ok is False
    assert operator_fragment not in check.detail


def test_unknown_driver_and_credentials_are_redacted() -> None:
    check = _checks(session_db_url="x_secret_123://user:hunter2@host/db")["session_db_url"]

    assert check.ok is False
    assert "x_secret_123" not in check.detail
    assert "hunter2" not in check.detail


def test_missing_payload_store_path_fails() -> None:
    assert _checks()["payload_store_path"].ok is False


def test_missing_data_dir_fails() -> None:
    assert _checks()["data_dir"].ok is False


def test_non_container_host_fails() -> None:
    assert _checks()["host"].ok is False


def test_container_host_passes() -> None:
    assert _checks(host="0.0.0.0")["host"].ok is True


def test_placeholder_secret_key_fails() -> None:
    assert _checks(secret_key="change-me-in-production", host="127.0.0.1")["secret_key"].ok is False


def test_undersized_secret_key_fails() -> None:
    assert _checks(secret_key="short", host="127.0.0.1")["secret_key"].ok is False


def test_uniform_byte_signing_key_fails() -> None:
    check = _checks(shareable_link_signing_key=b"\x00" * 32, host="127.0.0.1")

    assert check["shareable_link_signing_key"].ok is False


def test_check_names_are_exact_ordered_and_unique() -> None:
    names = [check.name for check in validate_aws_ecs_settings(WebSettings(**_base_kwargs()))]

    assert names == [
        "deployment_target",
        "session_db_url",
        "landscape_url",
        "data_dir",
        "payload_store_path",
        "host",
        "secret_key",
        "shareable_link_signing_key",
    ]
    assert len(names) == len(set(names))


def test_all_checks_pass_for_fully_valid_ecs_settings(tmp_path: Path) -> None:
    settings = WebSettings(
        **(
            _base_kwargs()
            | {
                "deployment_target": "aws-ecs",
                "host": "0.0.0.0",
                "session_db_url": "postgresql://u:p@host/session",
                "landscape_url": "postgresql://u:p@host/landscape",
                "data_dir": tmp_path / "data",
                "payload_store_path": tmp_path / "payloads",
            }
        )
    )

    checks = validate_aws_ecs_settings(settings)

    assert all(check.ok for check in checks)
    assert all(check.detail for check in checks)
