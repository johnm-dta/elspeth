# Web UX Task-Plan 2B: Auth Middleware & Routes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Implement auth middleware (FastAPI dependency) and auth REST endpoints (login, token, me, config)
**Parent Plan:** `plans/2026-03-28-web-ux-sub2-auth-sessions.md`
**Spec:** `specs/2026-03-28-web-ux-sub2-auth-sessions-design.md`
**Depends On:** Task-Plan 2A (Auth Protocol & Providers)
**Blocks:** Task-Plan 2E (Session API & Wiring)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/elspeth/web/auth/local.py` | Add `refresh()` method for token re-issue |
| Create | `src/elspeth/web/auth/middleware.py` | get_current_user FastAPI dependency (stashes token on request.state) |
| Create | `src/elspeth/web/auth/routes.py` | /api/auth/login, /api/auth/token, /api/auth/config, /api/auth/me |
| Create | `tests/unit/web/auth/test_middleware.py` | Auth middleware tests |
| Create | `tests/unit/web/auth/test_routes.py` | Auth route tests including /api/auth/config |

---

### Task 2.5: Auth Middleware

**Files:**
- Create: `src/elspeth/web/auth/middleware.py`
- Create: `tests/unit/web/auth/test_middleware.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/web/auth/test_middleware.py
"""Tests for the get_current_user FastAPI auth dependency."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Depends
from starlette.testclient import TestClient

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import AuthenticationError, UserIdentity


def _create_test_app(auth_provider) -> FastAPI:
    """Create a minimal FastAPI app with auth middleware for testing."""
    app = FastAPI()
    app.state.auth_provider = auth_provider

    @app.get("/protected")
    async def protected(user: UserIdentity = Depends(get_current_user)):
        return {"user_id": user.user_id, "username": user.username}

    return app


class TestGetCurrentUser:
    """Tests for the auth middleware dependency."""

    def test_valid_bearer_token(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.authenticate.return_value = UserIdentity(
            user_id="alice", username="alice",
        )
        app = _create_test_app(mock_provider)
        client = TestClient(app)

        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer valid-token-here"},
        )
        assert response.status_code == 200
        assert response.json()["user_id"] == "alice"
        mock_provider.authenticate.assert_called_once_with("valid-token-here")

    def test_missing_authorization_header(self) -> None:
        mock_provider = AsyncMock()
        app = _create_test_app(mock_provider)
        client = TestClient(app)

        response = client.get("/protected")
        assert response.status_code == 401
        assert response.json()["detail"] == "Missing or invalid Authorization header"

    def test_non_bearer_scheme(self) -> None:
        mock_provider = AsyncMock()
        app = _create_test_app(mock_provider)
        client = TestClient(app)

        response = client.get(
            "/protected",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert response.status_code == 401

    def test_bearer_with_no_token(self) -> None:
        mock_provider = AsyncMock()
        app = _create_test_app(mock_provider)
        client = TestClient(app)

        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer"},
        )
        assert response.status_code == 401

    def test_invalid_token_returns_401_with_detail(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.authenticate.side_effect = AuthenticationError("Token expired")
        app = _create_test_app(mock_provider)
        client = TestClient(app)

        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer expired-token"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Token expired"
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/test_middleware.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.auth.middleware'`

- [ ] **Step 3: Implement auth middleware**

```python
# src/elspeth/web/auth/middleware.py
"""FastAPI auth dependency -- extracts UserIdentity from Bearer tokens.

This is a FastAPI dependency function, not ASGI middleware. All protected
routes declare it via Depends(get_current_user).
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from elspeth.web.auth.models import AuthenticationError, UserIdentity


async def get_current_user(request: Request) -> UserIdentity:
    """Extract and validate a Bearer token from the request.

    Retrieves the auth_provider from request.app.state and calls
    authenticate(token). Converts AuthenticationError to HTTP 401.

    Stashes the raw token on request.state.auth_token so downstream
    route handlers (e.g. /me) can reuse it without re-parsing the
    Authorization header.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )

    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )

    token = parts[1].strip()
    request.state.auth_token = token
    auth_provider = request.app.state.auth_provider

    try:
        return await auth_provider.authenticate(token)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=exc.detail) from exc
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/test_middleware.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/auth/middleware.py tests/unit/web/auth/test_middleware.py
git commit -m "feat(web/auth): implement get_current_user auth middleware dependency"
```

---

### Task 2.6: Auth Routes

**Files:**
- Modify: `src/elspeth/web/auth/local.py` (add `refresh()` method)
- Create: `src/elspeth/web/auth/routes.py`
- Create: `tests/unit/web/auth/test_routes.py`

- [ ] **Step 1: Add `refresh()` method to LocalAuthProvider**

The token refresh route needs to issue a new JWT for an already-authenticated user without requiring their password. Add a public `refresh()` method to `LocalAuthProvider` so the route doesn't need to access private `_secret_key` or `_token_expiry_hours` attributes.

Add this method to `src/elspeth/web/auth/local.py` after the `login()` method:

```python
    def refresh(self, user_id: str, username: str) -> str:
        """Issue a new JWT for an already-authenticated user.

        Called by the token refresh route. Does NOT re-verify
        credentials — the caller (get_current_user middleware)
        has already validated the existing token.
        """
        payload = {
            "sub": user_id,
            "username": username,
            "exp": int(time.time()) + self._token_expiry_hours * 3600,
        }
        token: str = jwt.encode(payload, self._secret_key, algorithm="HS256")
        return token
```

- [ ] **Step 2: Write tests**

```python
# tests/unit/web/auth/test_routes.py
"""Tests for auth API routes -- /api/auth/login, /api/auth/token, /api/auth/me."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from elspeth.web.auth.local import LocalAuthProvider
from elspeth.web.auth.models import AuthenticationError, UserProfile
from elspeth.web.auth.routes import create_auth_router


from elspeth.web.config import WebSettings


def _create_test_app(
    provider, auth_provider_type: str = "local", **settings_overrides
) -> FastAPI:
    """Create a FastAPI app with auth routes for testing."""
    app = FastAPI()
    app.state.auth_provider = provider
    app.state.settings = WebSettings(auth_provider=auth_provider_type, **settings_overrides)
    router = create_auth_router()
    app.include_router(router)
    return app


class TestLoginEndpoint:
    """Tests for POST /api/auth/login."""

    def test_login_valid_credentials(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        provider.create_user("alice", "password123", display_name="Alice")
        app = _create_test_app(provider)
        client = TestClient(app)

        response = client.post(
            "/api/auth/login",
            json={"username": "alice", "password": "password123"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        # Verify it's a valid JWT (three segments)
        assert len(body["access_token"].split(".")) == 3

    def test_login_invalid_credentials(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        provider.create_user("alice", "password123", display_name="Alice")
        app = _create_test_app(provider)
        client = TestClient(app)

        response = client.post(
            "/api/auth/login",
            json={"username": "alice", "password": "wrong"},
        )
        assert response.status_code == 401

    def test_login_not_available_for_oidc(self, tmp_path) -> None:
        provider = AsyncMock()
        app = _create_test_app(provider, auth_provider_type="oidc")
        client = TestClient(app)

        response = client.post(
            "/api/auth/login",
            json={"username": "alice", "password": "pw"},
        )
        assert response.status_code == 404


class TestTokenRefreshEndpoint:
    """Tests for POST /api/auth/token."""

    def test_token_refresh_returns_new_token(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        provider.create_user("alice", "pw", display_name="Alice")
        app = _create_test_app(provider)
        client = TestClient(app)

        # Login first
        login_resp = client.post(
            "/api/auth/login",
            json={"username": "alice", "password": "pw"},
        )
        old_token = login_resp.json()["access_token"]

        # Refresh
        refresh_resp = client.post(
            "/api/auth/token",
            headers={"Authorization": f"Bearer {old_token}"},
        )
        assert refresh_resp.status_code == 200
        new_body = refresh_resp.json()
        assert "access_token" in new_body
        assert new_body["token_type"] == "bearer"

    def test_token_refresh_invalid_token(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        app = _create_test_app(provider)
        client = TestClient(app)

        response = client.post(
            "/api/auth/token",
            headers={"Authorization": "Bearer garbage"},
        )
        assert response.status_code == 401


class TestMeEndpoint:
    """Tests for GET /api/auth/me."""

    def test_me_returns_profile(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        provider.create_user(
            "alice", "pw",
            display_name="Alice Smith",
            email="alice@example.com",
        )
        app = _create_test_app(provider)
        client = TestClient(app)

        login_resp = client.post(
            "/api/auth/login",
            json={"username": "alice", "password": "pw"},
        )
        token = login_resp.json()["access_token"]

        me_resp = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert me_resp.status_code == 200
        body = me_resp.json()
        assert body["user_id"] == "alice"
        assert body["display_name"] == "Alice Smith"
        assert body["email"] == "alice@example.com"
        assert body["groups"] == []

    def test_me_unauthenticated(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        app = _create_test_app(provider)
        client = TestClient(app)

        response = client.get("/api/auth/me")
        assert response.status_code == 401


class TestAuthConfigEndpoint:
    """Tests for GET /api/auth/config (S9/D5)."""

    def test_local_provider_returns_null_oidc_fields(self, tmp_path) -> None:
        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
        )
        app = _create_test_app(provider, auth_provider_type="local")
        client = TestClient(app)

        response = client.get("/api/auth/config")
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "local"
        assert body["oidc_issuer"] is None
        assert body["oidc_client_id"] is None

    def test_oidc_provider_returns_issuer_and_client_id(self) -> None:
        provider = AsyncMock()
        app = _create_test_app(
            provider,
            auth_provider_type="oidc",
            oidc_issuer="https://login.example.com",
            oidc_client_id="my-client-id",
        )
        client = TestClient(app)

        response = client.get("/api/auth/config")
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "oidc"
        assert body["oidc_issuer"] == "https://login.example.com"
        assert body["oidc_client_id"] == "my-client-id"

    def test_config_endpoint_is_unauthenticated(self) -> None:
        """GET /api/auth/config must not require a Bearer token."""
        provider = AsyncMock()
        app = _create_test_app(provider, auth_provider_type="local")
        client = TestClient(app)

        # No Authorization header -- should still return 200
        response = client.get("/api/auth/config")
        assert response.status_code == 200
```

- [ ] **Step 3: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/test_routes.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.auth.routes'`

- [ ] **Step 4: Implement auth routes**

```python
# src/elspeth/web/auth/routes.py
"""Auth API routes -- /api/auth/login, /api/auth/token, /api/auth/config, /api/auth/me.

POST /login is only available when auth_provider is "local".
POST /token re-issues a JWT from a valid existing token (local only).
GET /config returns auth configuration for frontend discovery (unauthenticated).
GET /me returns the full UserProfile for any auth provider.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import AuthenticationError, UserIdentity


class LoginRequest(BaseModel):
    """Request body for POST /api/auth/login."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """Response for login and token refresh."""

    access_token: str
    token_type: str = "bearer"


class UserProfileResponse(BaseModel):
    """Response for GET /api/auth/me."""

    user_id: str
    username: str
    display_name: str
    email: str | None = None
    groups: list[str] = []


class AuthConfigResponse(BaseModel):
    """Response for GET /api/auth/config."""

    provider: str
    oidc_issuer: str | None = None
    oidc_client_id: str | None = None


def create_auth_router() -> APIRouter:
    """Create the auth router with /api/auth prefix."""
    router = APIRouter(prefix="/api/auth", tags=["auth"])

    @router.post("/login", response_model=TokenResponse)
    async def login(body: LoginRequest, request: Request) -> TokenResponse:
        """Authenticate with username/password (local auth only).

        login() is synchronous (bcrypt is intentionally slow ~200ms),
        so it is offloaded to a thread to avoid blocking the event loop.
        """
        settings = request.app.state.settings
        if settings.auth_provider != "local":
            raise HTTPException(status_code=404, detail="Not found")

        provider = request.app.state.auth_provider
        try:
            token = await asyncio.to_thread(
                provider.login, body.username, body.password,
            )
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=exc.detail) from exc

        return TokenResponse(access_token=token)

    @router.post("/token", response_model=TokenResponse)
    async def refresh_token(
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> TokenResponse:
        """Re-issue a JWT from a valid existing token (local auth only).

        Uses the provider's public refresh() method rather than
        reaching into private attributes.
        """
        settings = request.app.state.settings
        if settings.auth_provider != "local":
            raise HTTPException(status_code=404, detail="Not found")

        provider = request.app.state.auth_provider
        new_token = provider.refresh(user.user_id, user.username)
        return TokenResponse(access_token=new_token)

    @router.get("/config", response_model=AuthConfigResponse)
    async def auth_config(request: Request) -> AuthConfigResponse:
        """Return auth configuration for frontend discovery.

        This endpoint is unauthenticated -- the frontend needs it
        before any login flow.
        """
        settings = request.app.state.settings
        return AuthConfigResponse(
            provider=settings.auth_provider,
            oidc_issuer=settings.oidc_issuer,
            oidc_client_id=settings.oidc_client_id,
        )

    @router.get("/me", response_model=UserProfileResponse)
    async def me(
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> UserProfileResponse:
        """Return the full profile of the authenticated user.

        The raw token was stashed on request.state.auth_token by
        get_current_user, so we don't re-parse the Authorization header.
        """
        token = request.state.auth_token
        provider = request.app.state.auth_provider

        try:
            profile = await provider.get_user_info(token)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=exc.detail) from exc

        return UserProfileResponse(
            user_id=profile.user_id,
            username=profile.username,
            display_name=profile.display_name,
            email=profile.email,
            groups=list(profile.groups),
        )

    return router
```

- [ ] **Step 5: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/test_routes.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 6: Run all auth tests**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/ -v
```

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/auth/local.py src/elspeth/web/auth/routes.py \
    tests/unit/web/auth/test_routes.py
git commit -m "feat(web/auth): implement auth routes -- login, token refresh, user profile, config"
```

---

## Self-Review Checklist

After completing both tasks, verify:

```bash
# Middleware tests
.venv/bin/python -m pytest tests/unit/web/auth/test_middleware.py -v

# Route tests
.venv/bin/python -m pytest tests/unit/web/auth/test_routes.py -v

# All auth tests together
.venv/bin/python -m pytest tests/unit/web/auth/ -v

# Type checking
.venv/bin/python -m mypy src/elspeth/web/auth/middleware.py src/elspeth/web/auth/routes.py

# Linting
.venv/bin/python -m ruff check src/elspeth/web/auth/middleware.py src/elspeth/web/auth/routes.py
```

**Expected results:**

- [ ] All 15 tests pass (5 middleware + 10 routes)
- [ ] `get_current_user` extracts Bearer token, stashes it on `request.state.auth_token`, delegates to `auth_provider.authenticate()`
- [ ] Missing/malformed Authorization header returns 401, not 500
- [ ] `AuthenticationError` from provider is converted to HTTP 401 with `exc.detail`
- [ ] `POST /api/auth/login` returns 404 when `auth_provider != "local"` (not 405 or 500)
- [ ] `POST /api/auth/login` offloads sync `provider.login()` via `asyncio.to_thread` (bcrypt is ~200ms)
- [ ] `POST /api/auth/token` uses `provider.refresh()` — no access to private `_secret_key` or `_token_expiry_hours`
- [ ] `GET /api/auth/config` accesses `settings.oidc_issuer` and `settings.oidc_client_id` directly — no `getattr()` with defaults
- [ ] `GET /api/auth/config` is unauthenticated -- no `Depends(get_current_user)`
- [ ] `GET /api/auth/me` uses `request.state.auth_token` from middleware — no duplicate header parsing
- [ ] `GET /api/auth/me` returns full `UserProfileResponse` including groups
- [ ] Tests use `WebSettings(...)` — no `type()` hacks for fake settings
- [ ] No imports from `elspeth.web.sessions` (Task-Plan 2B has no session dependencies)
- [ ] Exception chains preserved with `from exc` on all `raise HTTPException`
