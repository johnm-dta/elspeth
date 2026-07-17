"""Deployment-policy and fail-closed tests for web authentication audit writes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import create_autospec

import jwt
import pytest
from fastapi import Request
from structlog.testing import capture_logs

from elspeth.core.landscape.auth_audit_repository import AuthAuditRepository
from elspeth.core.landscape.database import SchemaCompatibilityError
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.web.auth import audit as audit_module
from elspeth.web.auth.audit import AuthAuditRecorder
from elspeth.web.deployment_contract import DEPLOYMENT_TARGET_AWS_ECS


def _settings(deployment_target: str) -> Any:
    return SimpleNamespace(
        deployment_target=deployment_target,
        landscape_passphrase=None,
        get_landscape_url=lambda: "sqlite:///auth-audit.db",
    )


def test_from_settings_disables_create_tables_for_aws_ecs() -> None:
    recorder = AuthAuditRecorder.from_settings(_settings(DEPLOYMENT_TARGET_AWS_ECS))

    assert recorder.create_tables is False


def test_from_settings_keeps_create_tables_for_default() -> None:
    recorder = AuthAuditRecorder.from_settings(_settings("default"))

    assert recorder.create_tables is True


def test_from_settings_rejects_unknown_deployment_target() -> None:
    with pytest.raises(ValueError, match="unsupported deployment_target"):
        AuthAuditRecorder.from_settings(_settings("future-target"))


def test_direct_construction_requires_create_tables_policy() -> None:
    with pytest.raises(TypeError):
        AuthAuditRecorder(
            landscape_url="sqlite:///auth-audit.db",
            landscape_passphrase=None,
        )


def _request() -> Request:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/auth/login",
            "headers": [(b"user-agent", b"bounded-agent")],
            "client": ("127.0.0.1", 12345),
        }
    )
    request.state.request_id = "request-id"
    return request


def _writer_kwargs(method_name: str) -> dict[str, object]:
    common: dict[str, object] = {"provider": "local"}
    if method_name == "record_login_success_and_token_issued":
        token = jwt.encode({"iat": 1, "exp": 2}, "bounded-test-key-that-is-at-least-32-bytes", algorithm="HS256")
        return {
            **common,
            "user_id": "user-1",
            "username": "alice",
            "access_token": token,
        }
    if method_name == "record_login_success":
        return {**common, "user_id": "user-1", "username": "alice"}
    if method_name == "record_login_failure":
        return {**common, "username": "alice", "failure_category": "invalid_credentials"}
    if method_name == "record_token_issued":
        token = jwt.encode({"iat": 1, "exp": 2}, "bounded-test-key-that-is-at-least-32-bytes", algorithm="HS256")
        return {
            **common,
            "user_id": "user-1",
            "username": "alice",
            "access_token": token,
            "issuance_path": "login",
        }
    if method_name == "record_auth_failure":
        return {
            **common,
            "failure_category": "invalid_token",
            "failure_stage": "authenticate",
            "user_id": None,
            "username": None,
            "exception_class": "AuthenticationError",
        }
    raise AssertionError(f"unknown writer {method_name}")


@pytest.mark.parametrize(
    ("method_name", "repository_method"),
    [
        ("record_login_success_and_token_issued", "record_login_success_and_token_issued"),
        ("record_login_success", "record_login_outcome"),
        ("record_token_issued", "record_token_issued"),
        ("record_auth_failure", "record_auth_failure"),
        ("record_login_failure", "record_login_outcome"),
    ],
)
def test_every_writer_forwards_required_create_tables_policy(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    repository_method: str,
) -> None:
    open_calls: list[tuple[str, dict[str, object]]] = []
    db_sentinel = object()

    class _DBContext:
        def __enter__(self) -> object:
            return db_sentinel

        def __exit__(self, *args: object) -> None:
            return None

    class _FakeLandscapeDB:
        @classmethod
        def from_url(cls, url: str, **kwargs: object) -> _DBContext:
            open_calls.append((url, kwargs))
            return _DBContext()

    auth_repository = create_autospec(AuthAuditRepository, instance=True)
    factory = SimpleNamespace(auth_audit=auth_repository)
    monkeypatch.setattr(audit_module, "LandscapeDB", _FakeLandscapeDB)
    monkeypatch.setattr(
        audit_module,
        "RecorderFactory",
        create_autospec(audit_module.RecorderFactory, return_value=factory),
    )
    recorder = AuthAuditRecorder(
        landscape_url="sqlite:///auth-audit.db",
        landscape_passphrase=None,
        create_tables=False,
    )

    getattr(recorder, method_name)(_request(), **_writer_kwargs(method_name))

    assert open_calls == [
        (
            "sqlite:///auth-audit.db",
            {"passphrase": None, "create_tables": False},
        )
    ]
    getattr(auth_repository, repository_method).assert_called_once()


_OPERATION_NAMES = {
    "record_login_success_and_token_issued": "login_success_and_token_issued",
    "record_login_success": "login_success",
    "record_token_issued": "token_issued",
    "record_auth_failure": "auth_failure",
    "record_login_failure": "login_failure",
}


@pytest.mark.parametrize(
    ("method_name", "repository_method"),
    [
        ("record_login_success_and_token_issued", "record_login_success_and_token_issued"),
        ("record_login_success", "record_login_outcome"),
        ("record_token_issued", "record_token_issued"),
        ("record_auth_failure", "record_auth_failure"),
        ("record_login_failure", "record_login_outcome"),
    ],
)
@pytest.mark.parametrize("failure_location", ["open", "repository"])
def test_every_writer_propagates_and_redacts_expected_database_failures(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    repository_method: str,
    failure_location: str,
) -> None:
    failure: Exception
    if failure_location == "open":
        failure = SchemaCompatibilityError("RAW_SQL_MARKER CREDENTIAL_MARKER")
    else:
        failure = LandscapeRecordError("RAW_SQL_MARKER CREDENTIAL_MARKER")
    db_sentinel = object()

    class _DBContext:
        def __enter__(self) -> object:
            return db_sentinel

        def __exit__(self, *args: object) -> None:
            return None

    class _FakeLandscapeDB:
        @classmethod
        def from_url(cls, url: str, **kwargs: object) -> _DBContext:
            del url, kwargs
            if failure_location == "open":
                raise failure
            return _DBContext()

    auth_repository = create_autospec(AuthAuditRepository, instance=True)
    if failure_location == "repository":
        getattr(auth_repository, repository_method).side_effect = failure
    monkeypatch.setattr(audit_module, "LandscapeDB", _FakeLandscapeDB)
    monkeypatch.setattr(
        audit_module,
        "RecorderFactory",
        create_autospec(
            audit_module.RecorderFactory,
            return_value=SimpleNamespace(auth_audit=auth_repository),
        ),
    )
    recorder = AuthAuditRecorder(
        landscape_url="sqlite:///SENSITIVE_URL_MARKER.db",
        landscape_passphrase="PASSPHRASE_MARKER",
        create_tables=False,
    )

    with capture_logs() as logs, pytest.raises(type(failure)) as exc_info:
        getattr(recorder, method_name)(_request(), **_writer_kwargs(method_name))

    assert exc_info.value is failure
    assert len(logs) == 1
    log = logs[0]
    assert log["event"] == "auth_audit_write_failed"
    assert getattr(log["operation"], "value", log["operation"]) == _OPERATION_NAMES[method_name]
    assert log["exception_class"] == type(failure).__name__
    assert set(log) == {"event", "operation", "exception_class", "log_level"}
    rendered = repr(logs)
    for sentinel in (
        "RAW_SQL_MARKER",
        "CREDENTIAL_MARKER",
        "SENSITIVE_URL_MARKER",
        "PASSPHRASE_MARKER",
        "bounded-agent",
        "/api/auth/login",
    ):
        assert sentinel not in rendered
