"""Tests for the get_current_user FastAPI auth dependency."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException, Request

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import AuthenticationError, UserIdentity


def _make_request(auth_provider, authorization: str | None = None) -> Request:
    """Create a Starlette request carrying the auth provider under app.state."""
    app = FastAPI()
    app.state.auth_provider = auth_provider
    headers: list[tuple[bytes, bytes]] = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode("latin-1")))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/protected",
            "headers": headers,
            "app": app,
        }
    )


class TestGetCurrentUser:
    """Tests for the auth middleware dependency."""

    pytestmark = pytest.mark.asyncio

    async def test_valid_bearer_token(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.authenticate.return_value = UserIdentity(
            user_id="alice",
            username="alice",
        )
        request = _make_request(mock_provider, "Bearer valid-token-here")

        user = await get_current_user(request)

        assert user.user_id == "alice"
        assert request.state.auth_token == "valid-token-here"
        mock_provider.authenticate.assert_awaited_once_with("valid-token-here")

    async def test_missing_authorization_header(self) -> None:
        mock_provider = AsyncMock()
        request = _make_request(mock_provider)

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Missing or invalid Authorization header"

    async def test_non_bearer_scheme(self) -> None:
        mock_provider = AsyncMock()
        request = _make_request(mock_provider, "Basic dXNlcjpwYXNz")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request)

        assert exc_info.value.status_code == 401

    async def test_bearer_with_no_token(self) -> None:
        mock_provider = AsyncMock()
        request = _make_request(mock_provider, "Bearer")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request)

        assert exc_info.value.status_code == 401

    async def test_invalid_token_returns_401_with_detail(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.authenticate.side_effect = AuthenticationError("Token expired")
        request = _make_request(mock_provider, "Bearer expired-token")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Token expired"

    async def test_bearer_with_whitespace_only_token(self) -> None:
        mock_provider = AsyncMock()
        request = _make_request(mock_provider, "Bearer   ")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request)

        assert exc_info.value.status_code == 401
