"""Tests for auth API routes -- /api/auth/login, /api/auth/register, /api/auth/token, /api/auth/me."""

from __future__ import annotations

from unittest.mock import AsyncMock

import jwt as pyjwt
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from elspeth.web.auth.local import LocalAuthProvider
from elspeth.web.auth.models import AuthenticationError, AuthProviderUnavailable, UserIdentity
from elspeth.web.auth.routes import RegisterRequest, create_auth_router
from elspeth.web.config import WebSettings

_OIDC_FIELDS = {
    "oidc_issuer": "https://issuer.example.com",
    "oidc_audience": "test-audience",
    "oidc_client_id": "test-client-id",
}
_ENTRA_FIELDS = {**_OIDC_FIELDS, "entra_tenant_id": "test-tenant-id"}


def _create_test_app(provider, auth_provider_type: str = "local", **settings_overrides) -> FastAPI:
    """Create a FastAPI app with auth routes for testing."""
    from elspeth.web.middleware.rate_limit import ComposerRateLimiter

    app = FastAPI()
    app.state.auth_provider = provider
    app.state.settings = WebSettings(
        auth_provider=auth_provider_type,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        **settings_overrides,
    )
    app.state.oidc_authorization_endpoint = None
    # Auth rate limiter — generous limit for tests that aren't testing rate limiting
    app.state.auth_rate_limiter = ComposerRateLimiter(limit=100)
    router = create_auth_router()
    app.include_router(router)
    return app


def _client_for(app: FastAPI) -> AsyncClient:
    """Create an async ASGI client without Starlette's sync TestClient portal."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _assert_token_response_uncacheable(response: Response) -> None:
    """Token responses carry bearer credentials and must not be cacheable."""
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Pragma"] == "no-cache"


@pytest.mark.asyncio
class TestLoginEndpoint:
    """Tests for POST /api/auth/login."""

    async def test_login_valid_credentials(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        provider.create_user("alice", "password123", display_name="Alice")
        app = _create_test_app(provider)

        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/login",
                json={"username": "alice", "password": "password123"},
            )
        assert response.status_code == 200
        _assert_token_response_uncacheable(response)
        body = response.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        # Verify it's a valid JWT (three segments)
        assert len(body["access_token"].split(".")) == 3

    async def test_login_invalid_credentials(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        provider.create_user("alice", "password123", display_name="Alice")
        app = _create_test_app(provider)

        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/login",
                json={"username": "alice", "password": "wrong"},
            )
        assert response.status_code == 401

    async def test_login_not_available_for_oidc(self, tmp_path) -> None:
        provider = AsyncMock()
        app = _create_test_app(provider, auth_provider_type="oidc", **_OIDC_FIELDS)

        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/login",
                json={"username": "alice", "password": "pw"},
            )
        assert response.status_code == 404

    async def test_login_not_available_for_entra(self) -> None:
        provider = AsyncMock()
        app = _create_test_app(provider, auth_provider_type="entra", **_ENTRA_FIELDS)
        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/login",
                json={"username": "alice", "password": "pw"},
            )
        assert response.status_code == 404


@pytest.mark.asyncio
class TestRegisterEndpoint:
    """Tests for POST /api/auth/register."""

    async def test_register_open_mode_creates_user_and_returns_token(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        app = _create_test_app(provider, registration_mode="open")

        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/register",
                json={"username": "bob", "password": "pw123", "display_name": "Bob"},
            )
        assert response.status_code == 200
        _assert_token_response_uncacheable(response)
        body = response.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert len(body["access_token"].split(".")) == 3

    async def test_register_open_mode_with_email(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        app = _create_test_app(provider, registration_mode="open")

        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/register",
                json={
                    "username": "bob",
                    "password": "pw123",
                    "display_name": "Bob",
                    "email": "bob@example.com",
                },
            )
        assert response.status_code == 200

    async def test_register_closed_mode_returns_404(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        app = _create_test_app(provider, registration_mode="closed")

        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/register",
                json={"username": "bob", "password": "pw123", "display_name": "Bob"},
            )
        assert response.status_code == 404

    async def test_register_email_verified_mode_returns_501(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        app = _create_test_app(provider, registration_mode="email_verified")

        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/register",
                json={"username": "bob", "password": "pw123", "display_name": "Bob"},
            )
        assert response.status_code == 501
        assert "Email verification" in response.json()["detail"]

    async def test_register_non_local_provider_returns_404(self) -> None:
        provider = AsyncMock()
        app = _create_test_app(
            provider,
            auth_provider_type="oidc",
            registration_mode="open",
            **_OIDC_FIELDS,
        )

        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/register",
                json={"username": "bob", "password": "pw123", "display_name": "Bob"},
            )
        assert response.status_code == 404

    async def test_register_duplicate_username_returns_409(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        provider.create_user("bob", "existing", display_name="Bob")
        app = _create_test_app(provider, registration_mode="open")

        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/register",
                json={"username": "bob", "password": "pw123", "display_name": "Bob"},
            )
        assert response.status_code == 409

    async def test_register_blank_username_returns_422(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        app = _create_test_app(provider, registration_mode="open")

        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/register",
                json={"username": "  ", "password": "pw123", "display_name": "Bob"},
            )
        assert response.status_code == 422

    async def test_register_blank_password_returns_422(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        app = _create_test_app(provider, registration_mode="open")

        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/register",
                json={"username": "bob", "password": "", "display_name": "Bob"},
            )
        assert response.status_code == 422


@pytest.mark.asyncio
class TestTokenRefreshEndpoint:
    """Tests for POST /api/auth/token."""

    async def test_token_refresh_returns_new_token(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        provider.create_user("alice", "pw", display_name="Alice")
        app = _create_test_app(provider)

        async with _client_for(app) as client:
            # Login first
            login_resp = await client.post(
                "/api/auth/login",
                json={"username": "alice", "password": "pw"},
            )
            old_token = login_resp.json()["access_token"]

            # Refresh
            refresh_resp = await client.post(
                "/api/auth/token",
                headers={"Authorization": f"Bearer {old_token}"},
            )
        assert refresh_resp.status_code == 200
        _assert_token_response_uncacheable(refresh_resp)
        new_body = refresh_resp.json()
        assert "access_token" in new_body
        assert new_body["token_type"] == "bearer"

    async def test_token_refresh_unparseable_claims_rejected(self, tmp_path) -> None:
        """Refresh must fail if pre-verification claim decode failed.

        When the middleware can't decode claims (auth_claims=None), the refresh
        endpoint cannot enforce chain lifetime. It must reject with 401 rather
        than silently skipping the chain age check.
        """
        from unittest.mock import patch

        import jwt as pyjwt

        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        provider.create_user("alice", "pw", display_name="Alice")
        app = _create_test_app(provider)

        async with _client_for(app) as client:
            # Login to get a valid token
            login_resp = await client.post(
                "/api/auth/login",
                json={"username": "alice", "password": "pw"},
            )
            valid_token = login_resp.json()["access_token"]

            # Patch jwt.decode to fail on unverified decode but let authenticate()
            # succeed (it uses the provider's own decode path, not the middleware's).
            original_decode = pyjwt.decode

            def selective_decode(token, *args, **kwargs):
                opts = kwargs.get("options", {})
                if not opts.get("verify_signature", True):
                    raise pyjwt.PyJWTError("simulated decode failure")
                return original_decode(token, *args, **kwargs)

            with patch("jwt.decode", side_effect=selective_decode):
                response = await client.post(
                    "/api/auth/token",
                    headers={"Authorization": f"Bearer {valid_token}"},
                )
        assert response.status_code == 401
        assert "claims could not be parsed" in response.json()["detail"]

    async def test_token_refresh_missing_iat_rejected(self, tmp_path) -> None:
        """Refresh must reject valid local tokens whose chain origin is absent."""
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        provider.create_user("alice", "pw", display_name="Alice")
        app = _create_test_app(provider)
        token_without_iat = pyjwt.encode(
            {
                "sub": "alice",
                "username": "alice",
                "exp": 9_999_999_999,
            },
            "test-key",
            algorithm="HS256",
        )

        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/token",
                headers={"Authorization": f"Bearer {token_without_iat}"},
            )

        assert response.status_code == 401
        assert "iat" in response.json()["detail"]
        assert "re-authenticate" in response.json()["detail"]

    async def test_token_refresh_invalid_token(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        app = _create_test_app(provider)

        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/token",
                headers={"Authorization": "Bearer garbage"},
            )
        assert response.status_code == 401


@pytest.mark.asyncio
class TestMeEndpoint:
    """Tests for GET /api/auth/me."""

    async def test_me_returns_profile(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        provider.create_user(
            "alice",
            "pw",
            display_name="Alice Smith",
            email="alice@example.com",
        )
        app = _create_test_app(provider)

        async with _client_for(app) as client:
            login_resp = await client.post(
                "/api/auth/login",
                json={"username": "alice", "password": "pw"},
            )
            token = login_resp.json()["access_token"]

            me_resp = await client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert me_resp.status_code == 200
        body = me_resp.json()
        assert body["user_id"] == "alice"
        assert body["display_name"] == "Alice Smith"
        assert body["email"] == "alice@example.com"
        assert body["groups"] == []

    async def test_me_unauthenticated(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        app = _create_test_app(provider)

        async with _client_for(app) as client:
            response = await client.get("/api/auth/me")
        assert response.status_code == 401


@pytest.mark.asyncio
class TestAuthConfigEndpoint:
    """Tests for GET /api/auth/config (S9/D5)."""

    async def test_local_provider_returns_null_oidc_fields(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        app = _create_test_app(provider, auth_provider_type="local")

        async with _client_for(app) as client:
            response = await client.get("/api/auth/config")
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "local"
        assert body["registration_mode"] == "open"
        assert body["oidc_issuer"] is None
        assert body["oidc_client_id"] is None

    async def test_oidc_provider_returns_issuer_and_client_id(self) -> None:
        provider = AsyncMock()
        app = _create_test_app(
            provider,
            auth_provider_type="oidc",
            oidc_issuer="https://login.example.com",
            oidc_audience="test-audience",
            oidc_client_id="my-client-id",
        )

        async with _client_for(app) as client:
            response = await client.get("/api/auth/config")
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "oidc"
        assert body["oidc_issuer"] == "https://login.example.com"
        assert body["oidc_client_id"] == "my-client-id"

    async def test_config_endpoint_is_unauthenticated(self) -> None:
        """GET /api/auth/config must not require a Bearer token."""
        provider = AsyncMock()
        app = _create_test_app(provider, auth_provider_type="local")

        # No Authorization header -- should still return 200
        async with _client_for(app) as client:
            response = await client.get("/api/auth/config")
        assert response.status_code == 200


@pytest.mark.asyncio
class TestTokenRefreshNonLocal:
    """Token refresh must be unavailable for non-local providers."""

    async def test_token_refresh_not_available_for_oidc(self) -> None:
        provider = AsyncMock()
        provider.authenticate.return_value = UserIdentity(user_id="alice", username="alice")
        app = _create_test_app(provider, auth_provider_type="oidc", **_OIDC_FIELDS)
        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/token",
                headers={"Authorization": "Bearer some-token"},
            )
        assert response.status_code == 404

    @pytest.mark.parametrize(
        ("provider_type", "settings_fields"),
        [
            ("oidc", _OIDC_FIELDS),
            ("entra", _ENTRA_FIELDS),
        ],
    )
    async def test_token_refresh_non_local_without_auth_returns_404_before_authentication(
        self,
        provider_type: str,
        settings_fields: dict[str, str],
    ) -> None:
        provider = AsyncMock()
        app = _create_test_app(provider, auth_provider_type=provider_type, **settings_fields)

        async with _client_for(app) as client:
            response = await client.post("/api/auth/token")

        assert response.status_code == 404
        provider.authenticate.assert_not_awaited()

    @pytest.mark.parametrize(
        ("provider_type", "settings_fields"),
        [
            ("oidc", _OIDC_FIELDS),
            ("entra", _ENTRA_FIELDS),
        ],
    )
    async def test_token_refresh_non_local_invalid_auth_returns_404_before_authentication(
        self,
        provider_type: str,
        settings_fields: dict[str, str],
    ) -> None:
        provider = AsyncMock()
        provider.authenticate.side_effect = AuthenticationError("bad token")
        app = _create_test_app(provider, auth_provider_type=provider_type, **settings_fields)

        async with _client_for(app) as client:
            response = await client.post(
                "/api/auth/token",
                headers={"Authorization": "Bearer garbage"},
            )

        assert response.status_code == 404
        provider.authenticate.assert_not_awaited()


@pytest.mark.asyncio
class TestMeErrorPath:
    """Tests for /me when get_user_info raises."""

    async def test_me_get_user_info_failure_returns_401(self) -> None:
        """If get_user_info raises, /me returns 401 with the detail."""
        mock_provider = AsyncMock()
        mock_provider.authenticate.return_value = UserIdentity(user_id="alice", username="alice")
        mock_provider.get_user_info.side_effect = AuthenticationError("Profile lookup failed")
        app = _create_test_app(mock_provider, auth_provider_type="oidc", **_OIDC_FIELDS)
        async with _client_for(app) as client:
            response = await client.get(
                "/api/auth/me",
                headers={"Authorization": "Bearer valid-token"},
            )
        assert response.status_code == 401
        assert response.json()["detail"] == "Profile lookup failed"

    async def test_me_get_user_info_unavailable_returns_503(self) -> None:
        """Provider availability failures during profile lookup return 503."""
        mock_provider = AsyncMock()
        mock_provider.authenticate.return_value = UserIdentity(user_id="alice", username="alice")
        mock_provider.get_user_info.side_effect = AuthProviderUnavailable("JWKS unavailable: ConnectError")
        app = _create_test_app(mock_provider, auth_provider_type="oidc", **_OIDC_FIELDS)
        async with _client_for(app) as client:
            response = await client.get(
                "/api/auth/me",
                headers={"Authorization": "Bearer valid-token"},
            )

        assert response.status_code == 503
        assert response.json()["detail"] == "JWKS unavailable: ConnectError"


class TestRegisterRequestValidation:
    """Registration must reject invisible-only fields to stay aligned with UserIdentity."""

    def test_rejects_zero_width_space_username(self) -> None:
        with pytest.raises(ValueError, match="visible character"):
            RegisterRequest(username="\u200b", password="password123", display_name="Test")

    def test_rejects_bom_only_username(self) -> None:
        with pytest.raises(ValueError, match="visible character"):
            RegisterRequest(username="\ufeff", password="password123", display_name="Test")

    def test_rejects_invisible_display_name(self) -> None:
        with pytest.raises(ValueError, match="visible character"):
            RegisterRequest(username="alice", password="password123", display_name="\u200b")

    def test_rejects_invisible_password(self) -> None:
        with pytest.raises(ValueError, match="visible character"):
            RegisterRequest(username="alice", password="\u200b", display_name="Test")

    def test_accepts_normal_registration(self) -> None:
        req = RegisterRequest(username="alice", password="password123", display_name="Alice")
        assert req.username == "alice"


@pytest.mark.asyncio
class TestAuthRateLimiting:
    """Tests that auth endpoints are rate-limited by client IP."""

    async def test_login_returns_429_when_rate_limited(self, tmp_path) -> None:
        """Login returns 429 after exceeding per-IP rate limit."""
        from elspeth.web.middleware.rate_limit import ComposerRateLimiter

        provider = LocalAuthProvider(db_path=tmp_path / "auth.db", secret_key="test-key")
        provider.create_user("alice", "password123", display_name="Alice")
        app = _create_test_app(provider)
        # Override with a very low limit
        app.state.auth_rate_limiter = ComposerRateLimiter(limit=2)

        async with _client_for(app) as client:
            # First two requests should succeed
            for _ in range(2):
                resp = await client.post("/api/auth/login", json={"username": "alice", "password": "password123"})
                assert resp.status_code == 200

            # Third request should be rate-limited
            resp = await client.post("/api/auth/login", json={"username": "alice", "password": "password123"})
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    async def test_register_returns_429_when_rate_limited(self, tmp_path) -> None:
        """Register returns 429 after exceeding per-IP rate limit."""
        from elspeth.web.middleware.rate_limit import ComposerRateLimiter

        provider = LocalAuthProvider(db_path=tmp_path / "auth.db", secret_key="test-key")
        app = _create_test_app(provider)
        app.state.auth_rate_limiter = ComposerRateLimiter(limit=1)

        async with _client_for(app) as client:
            # First registration should succeed
            resp = await client.post(
                "/api/auth/register",
                json={"username": "user1", "password": "password123", "display_name": "User 1"},
            )
            assert resp.status_code == 200

            # Second registration should be rate-limited (regardless of different username)
            resp = await client.post(
                "/api/auth/register",
                json={"username": "user2", "password": "password123", "display_name": "User 2"},
            )
        assert resp.status_code == 429

    async def test_auth_rate_limit_independent_from_composer(self, tmp_path) -> None:
        """Auth and composer rate limiters are separate instances."""
        from elspeth.web.middleware.rate_limit import ComposerRateLimiter

        provider = LocalAuthProvider(db_path=tmp_path / "auth.db", secret_key="test-key")
        provider.create_user("alice", "password123", display_name="Alice")
        app = _create_test_app(provider)
        # Auth limiter at 1, composer limiter stays at default (100)
        app.state.auth_rate_limiter = ComposerRateLimiter(limit=1)

        async with _client_for(app) as client:
            # Exhaust auth limit
            resp = await client.post("/api/auth/login", json={"username": "alice", "password": "password123"})
            assert resp.status_code == 200

            # Second should be rate-limited
            resp = await client.post("/api/auth/login", json={"username": "alice", "password": "password123"})
        assert resp.status_code == 429
