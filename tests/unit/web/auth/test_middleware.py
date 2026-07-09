"""Tests for the get_current_user FastAPI auth dependency."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi import FastAPI, HTTPException, Request

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import AuthenticationError, AuthProviderUnavailable, UserIdentity, UserProfile
from elspeth.web.config import WebSettings
from elspeth.web.middleware.rate_limit import ComposerRateLimiter


@dataclass
class _FakeAuthProvider:
    identity: UserIdentity = field(
        default_factory=lambda: UserIdentity(
            user_id="alice",
            username="alice",
        )
    )
    error: AuthenticationError | None = None
    authenticated_tokens: list[str] = field(default_factory=list)

    async def authenticate(self, token: str) -> UserIdentity:
        self.authenticated_tokens.append(token)
        if self.error is not None:
            raise self.error
        return self.identity

    async def get_user_info(self, token: str) -> UserProfile:
        raise NotImplementedError


@dataclass
class _FakeCounter:
    additions: list[tuple[int, dict[str, str]]] = field(default_factory=list)

    def add(self, amount: int, attributes: dict[str, str]) -> None:
        self.additions.append((amount, attributes))


class _NoopAuthAuditRecorder:
    def record_auth_failure(self, *args, **kwargs) -> None:
        return None


class _CountingAuthAuditRecorder:
    def __init__(self) -> None:
        self.failures: list[dict[str, object]] = []

    def record_auth_failure(self, *args, **kwargs) -> None:
        self.failures.append(kwargs)


def _make_app(
    auth_provider,
    *,
    recorder=None,
    auth_rate_limit: int = 100,
) -> FastAPI:
    app = FastAPI()
    app.state.auth_provider = auth_provider
    app.state.settings = WebSettings(
        auth_provider="local",
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    app.state.auth_audit_recorder = recorder if recorder is not None else _NoopAuthAuditRecorder()
    app.state.auth_rate_limiter = ComposerRateLimiter(limit=auth_rate_limit)
    return app


def _make_request(
    auth_provider,
    authorization: str | None = None,
    *,
    app: FastAPI | None = None,
    recorder=None,
    auth_rate_limit: int = 100,
) -> Request:
    """Create a Starlette request carrying the auth provider under app.state."""
    if app is None:
        app = _make_app(auth_provider, recorder=recorder, auth_rate_limit=auth_rate_limit)
    headers: list[tuple[bytes, bytes]] = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode("latin-1")))
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/protected",
            "headers": headers,
            "app": app,
            "client": ("127.0.0.1", 12345),
        }
    )
    request.state.request_id = "test-request"
    return request


class TestGetCurrentUser:
    """Tests for the auth middleware dependency."""

    pytestmark = pytest.mark.asyncio

    async def test_valid_bearer_token(self) -> None:
        provider = _FakeAuthProvider()
        request = _make_request(provider, "Bearer valid-token-here")

        user = await get_current_user(request)

        assert user.user_id == "alice"
        assert request.state.auth_token == "valid-token-here"
        assert provider.authenticated_tokens == ["valid-token-here"]

    async def test_missing_authorization_header(self) -> None:
        provider = _FakeAuthProvider()
        request = _make_request(provider)

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Missing or invalid Authorization header"

    async def test_missing_authorization_header_rate_limits_audit_writes(self) -> None:
        provider = _FakeAuthProvider()
        recorder = _CountingAuthAuditRecorder()
        app = _make_app(provider, recorder=recorder, auth_rate_limit=1)

        first_request = _make_request(provider, app=app)
        with pytest.raises(HTTPException) as first_exc_info:
            await get_current_user(first_request)

        second_request = _make_request(provider, app=app)
        with pytest.raises(HTTPException) as second_exc_info:
            await get_current_user(second_request)

        assert first_exc_info.value.status_code == 401
        assert second_exc_info.value.status_code == 429
        assert len(recorder.failures) == 1
        assert recorder.failures[0]["failure_category"] == "missing_authorization_header"

    async def test_rate_limited_audit_write_emits_suppression_telemetry(self, monkeypatch) -> None:
        """Over-limit auth failures are not silently dropped: the skipped audit
        write increments a telemetry counter, attributed by failure category."""
        import elspeth.web.auth.middleware as middleware_module

        counter = _FakeCounter()
        monkeypatch.setattr(middleware_module, "_AUTH_FAILURE_AUDIT_SUPPRESSED_TOTAL", counter)

        provider = _FakeAuthProvider()
        recorder = _CountingAuthAuditRecorder()
        app = _make_app(provider, recorder=recorder, auth_rate_limit=1)

        with pytest.raises(HTTPException):  # first attempt: 401, records the row
            await get_current_user(_make_request(provider, app=app))
        with pytest.raises(HTTPException) as over_limit:
            await get_current_user(_make_request(provider, app=app))

        assert over_limit.value.status_code == 429
        assert len(recorder.failures) == 1  # over-limit write was capped
        assert counter.additions == [(1, {"failure_category": "missing_authorization_header"})]

    async def test_non_bearer_scheme(self) -> None:
        provider = _FakeAuthProvider()
        request = _make_request(provider, "Basic dXNlcjpwYXNz")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request)

        assert exc_info.value.status_code == 401

    async def test_invalid_authorization_header_rate_limits_audit_writes(self) -> None:
        provider = _FakeAuthProvider()
        recorder = _CountingAuthAuditRecorder()
        app = _make_app(provider, recorder=recorder, auth_rate_limit=1)

        first_request = _make_request(provider, "Basic dXNlcjpwYXNz", app=app)
        with pytest.raises(HTTPException) as first_exc_info:
            await get_current_user(first_request)

        second_request = _make_request(provider, "Basic dXNlcjpwYXNz", app=app)
        with pytest.raises(HTTPException) as second_exc_info:
            await get_current_user(second_request)

        assert first_exc_info.value.status_code == 401
        assert second_exc_info.value.status_code == 429
        assert len(recorder.failures) == 1
        assert recorder.failures[0]["failure_category"] == "invalid_authorization_header"

    async def test_bearer_with_no_token(self) -> None:
        provider = _FakeAuthProvider()
        request = _make_request(provider, "Bearer")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request)

        assert exc_info.value.status_code == 401

    async def test_invalid_token_returns_401_with_detail(self) -> None:
        provider = _FakeAuthProvider(error=AuthenticationError("Token expired"))
        request = _make_request(provider, "Bearer expired-token")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Token expired"

    async def test_authentication_error_rate_limits_audit_writes(self) -> None:
        provider = _FakeAuthProvider(error=AuthenticationError("Token expired"))
        recorder = _CountingAuthAuditRecorder()
        app = _make_app(provider, recorder=recorder, auth_rate_limit=1)

        first_request = _make_request(provider, "Bearer expired-token", app=app)
        with pytest.raises(HTTPException) as first_exc_info:
            await get_current_user(first_request)

        second_request = _make_request(provider, "Bearer expired-token", app=app)
        with pytest.raises(HTTPException) as second_exc_info:
            await get_current_user(second_request)

        assert first_exc_info.value.status_code == 401
        assert second_exc_info.value.status_code == 429
        assert len(recorder.failures) == 1
        assert recorder.failures[0]["failure_category"] == "authentication_error"

    async def test_valid_bearer_token_does_not_consume_auth_failure_limiter(self) -> None:
        provider = _FakeAuthProvider()
        recorder = _CountingAuthAuditRecorder()
        app = _make_app(provider, recorder=recorder, auth_rate_limit=1)

        first_user = await get_current_user(_make_request(provider, "Bearer valid-token-one", app=app))
        second_user = await get_current_user(_make_request(provider, "Bearer valid-token-two", app=app))

        assert first_user.user_id == "alice"
        assert second_user.user_id == "alice"
        assert provider.authenticated_tokens == ["valid-token-one", "valid-token-two"]
        assert recorder.failures == []

    async def test_provider_unavailable_returns_503_with_detail(self) -> None:
        provider = _FakeAuthProvider(error=AuthProviderUnavailable("JWKS unavailable: ConnectError"))
        request = _make_request(provider, "Bearer maybe-valid-token")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request)

        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "JWKS unavailable: ConnectError"

    async def test_bearer_with_whitespace_only_token(self) -> None:
        provider = _FakeAuthProvider()
        request = _make_request(provider, "Bearer   ")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request)

        assert exc_info.value.status_code == 401
