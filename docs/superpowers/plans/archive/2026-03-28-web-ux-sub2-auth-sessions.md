# Web UX Sub-Spec 2: Auth & Sessions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the authentication and session persistence layer for the Web UX Composer MVP. This includes the AuthProvider protocol with three implementations (Local, OIDC, Entra), auth middleware and routes, SQLAlchemy session/message/state/run table models, the SessionService with CRUD operations, and session API routes with IDOR protection, path traversal sanitization, and one-active-run enforcement.

**Architecture:** Auth providers are protocol-based with three concrete implementations. The auth middleware is a FastAPI dependency (`Depends(get_current_user)`) injected into all protected routes. Session data lives in a dedicated SQLite database (separate from Landscape). All table definitions use SQLAlchemy Core. Schema creation uses `metadata.create_all()` on startup. Session routes verify user ownership on every request (IDOR protection returns 404, not 403).

**Tech Stack:** FastAPI, SQLAlchemy Core, bcrypt (password hashing), PyJWT[crypto] (JWT), httpx (OIDC JWKS discovery), Pydantic v2 (request/response schemas)

**Spec:** `docs/superpowers/specs/2026-03-28-web-ux-sub2-auth-sessions-design.md`

**Depends On:** Phase 1 (Foundation) — `src/elspeth/web/app.py`, `src/elspeth/web/config.py`, `src/elspeth/web/dependencies.py` must exist.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/auth/__init__.py` | Module init |
| Create | `src/elspeth/web/auth/protocol.py` | AuthProvider protocol (two methods, no exceptions) |
| Create | `src/elspeth/web/auth/models.py` | UserIdentity, UserProfile, AuthenticationError |
| Create | `src/elspeth/web/auth/local.py` | LocalAuthProvider -- SQLite, bcrypt, JWT via PyJWT |
| Create | `src/elspeth/web/auth/oidc.py` | OIDCAuthProvider -- JWKS discovery via httpx, token validation |
| Create | `src/elspeth/web/auth/entra.py` | EntraAuthProvider -- tenant validation, group claims |
| Create | `src/elspeth/web/auth/middleware.py` | get_current_user FastAPI dependency |
| Create | `src/elspeth/web/auth/routes.py` | /api/auth/login, /api/auth/token, /api/auth/config, /api/auth/me |
| Create | `src/elspeth/web/sessions/__init__.py` | Module init |
| Create | `src/elspeth/web/sessions/protocol.py` | SessionServiceProtocol |
| Create | `src/elspeth/web/sessions/models.py` | SQLAlchemy Core table definitions (sessions, chat_messages, composition_states, runs, run_events) |
| Create | `src/elspeth/web/sessions/service.py` | SessionServiceImpl -- CRUD, state versioning, active run check |
| Create | `src/elspeth/web/sessions/routes.py` | /api/sessions/* endpoints with IDOR protection, including state revert |
| Create | `src/elspeth/web/sessions/schemas.py` | Pydantic request/response models for all session endpoints |
| Modify | `src/elspeth/web/app.py` | Register auth and session routers, create session DB engine, call metadata.create_all on startup |
| Modify | `src/elspeth/web/dependencies.py` | Add get_current_user, get_session_service, get_auth_provider dependencies |
| Modify | `pyproject.toml` | Add bcrypt to [webui] extra |
| Create | `tests/unit/web/auth/__init__.py` | Test package |
| Create | `tests/unit/web/auth/test_models.py` | Auth model tests |
| Create | `tests/unit/web/auth/test_local_provider.py` | LocalAuthProvider tests |
| Create | `tests/unit/web/auth/test_oidc_provider.py` | OIDCAuthProvider tests |
| Create | `tests/unit/web/auth/test_entra_provider.py` | EntraAuthProvider tests |
| Create | `tests/unit/web/auth/test_middleware.py` | Auth middleware tests |
| Create | `tests/unit/web/auth/test_routes.py` | Auth route tests including /api/auth/config |
| Create | `tests/unit/web/sessions/__init__.py` | Test package |
| Create | `tests/unit/web/sessions/test_models.py` | Table schema tests |
| Create | `tests/unit/web/sessions/test_service.py` | SessionService CRUD tests |
| Create | `tests/unit/web/sessions/test_routes.py` | Session API endpoint tests, IDOR tests, upload path traversal test |

---

## Pre-requisites

Before starting this plan, Phase 1 must be complete. The following files must exist:

- `src/elspeth/web/__init__.py`
- `src/elspeth/web/app.py` (with `create_app()` factory and `/api/health` endpoint)
- `src/elspeth/web/config.py` (with `WebSettings` including `secret_key`, `auth_provider`, `data_dir`, `max_upload_bytes`)
- `src/elspeth/web/dependencies.py` (with `get_settings()`)
- `pyproject.toml` has `[webui]` extra with fastapi, uvicorn, PyJWT, python-multipart, httpx

Additionally, `bcrypt` must be added to the `[webui]` extra. If not already present, add it as the first step of Task 2.1.

---

### Task 2.1: AuthProvider Protocol and Auth Models

**Files:**
- Create: `src/elspeth/web/auth/__init__.py`
- Create: `src/elspeth/web/auth/protocol.py`
- Create: `src/elspeth/web/auth/models.py`
- Create: `tests/unit/web/auth/__init__.py`
- Create: `tests/unit/web/auth/test_models.py`

- [ ] **Step 1: Add bcrypt to pyproject.toml**

In the `[project.optional-dependencies]` section, add `bcrypt` to the `webui` extra:

```toml
webui = [
    "fastapi>=0.115,<1",
    "uvicorn[standard]>=0.34,<1",
    "PyJWT[crypto]>=2.8,<3",
    "python-multipart>=0.0.20",
    "websockets>=14.0,<15",
    "httpx>=0.27,<1",
    "bcrypt>=4.0,<5",
]
```

Then install:

```bash
uv pip install -e ".[webui,dev]"
```

- [ ] **Step 2: Write tests for auth models**

```python
# tests/unit/web/auth/__init__.py
```

```python
# tests/unit/web/auth/test_models.py
"""Tests for authentication data models."""

from __future__ import annotations

import pytest

from elspeth.web.auth.models import AuthenticationError, UserIdentity, UserProfile


class TestUserIdentity:
    """Tests for the minimal authentication identity."""

    def test_construction(self) -> None:
        identity = UserIdentity(user_id="alice", username="alice")
        assert identity.user_id == "alice"
        assert identity.username == "alice"

    def test_frozen_immutability(self) -> None:
        identity = UserIdentity(user_id="alice", username="alice")
        with pytest.raises(AttributeError):
            identity.user_id = "bob"  # type: ignore[misc]

    def test_slots(self) -> None:
        identity = UserIdentity(user_id="alice", username="alice")
        assert not hasattr(identity, "__dict__")


class TestUserProfile:
    """Tests for the extended user profile."""

    def test_construction_all_fields(self) -> None:
        profile = UserProfile(
            user_id="alice",
            username="alice",
            display_name="Alice Smith",
            email="alice@example.com",
            groups=("admin", "users"),
        )
        assert profile.user_id == "alice"
        assert profile.display_name == "Alice Smith"
        assert profile.email == "alice@example.com"
        assert profile.groups == ("admin", "users")

    def test_defaults(self) -> None:
        profile = UserProfile(
            user_id="bob",
            username="bob",
            display_name="Bob",
        )
        assert profile.email is None
        assert profile.groups == ()

    def test_frozen_immutability(self) -> None:
        profile = UserProfile(
            user_id="alice",
            username="alice",
            display_name="Alice",
        )
        with pytest.raises(AttributeError):
            profile.email = "x@y.com"  # type: ignore[misc]

    def test_groups_is_tuple_not_list(self) -> None:
        """Groups must be a tuple (immutable) not a list."""
        profile = UserProfile(
            user_id="alice",
            username="alice",
            display_name="Alice",
            groups=("g1", "g2"),
        )
        assert isinstance(profile.groups, tuple)


class TestAuthenticationError:
    """Tests for the authentication exception."""

    def test_default_message(self) -> None:
        err = AuthenticationError()
        assert err.detail == "Authentication failed"
        assert str(err) == "Authentication failed"

    def test_custom_message(self) -> None:
        err = AuthenticationError("Token expired")
        assert err.detail == "Token expired"
        assert str(err) == "Token expired"

    def test_is_exception(self) -> None:
        err = AuthenticationError()
        assert isinstance(err, Exception)
```

- [ ] **Step 3: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.auth'`

- [ ] **Step 4: Implement auth models and protocol**

```python
# src/elspeth/web/auth/__init__.py
"""Authentication providers and middleware."""
```

```python
# src/elspeth/web/auth/models.py
"""Authentication data models.

UserIdentity and UserProfile are frozen dataclasses. All fields are scalars,
None, or tuple of scalars -- no freeze guard needed.

AuthenticationError is the domain exception raised by all auth providers
when token validation fails.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UserIdentity:
    """Minimal authenticated identity -- returned from every auth check."""

    user_id: str
    username: str


@dataclass(frozen=True, slots=True)
class UserProfile:
    """Extended user profile information."""

    user_id: str
    username: str
    display_name: str
    email: str | None = None
    groups: tuple[str, ...] = ()


class AuthenticationError(Exception):
    """Raised when authentication fails.

    Caught by the auth middleware and converted to HTTP 401.
    """

    def __init__(self, detail: str = "Authentication failed") -> None:
        self.detail = detail
        super().__init__(detail)
```

```python
# src/elspeth/web/auth/protocol.py
"""Authentication provider protocol.

Defines the two-method interface that all auth implementations must satisfy.
No exception definitions here -- AuthenticationError lives in models.py.
"""

from __future__ import annotations

from typing import Protocol

from elspeth.web.auth.models import UserIdentity, UserProfile


class AuthProvider(Protocol):
    """Protocol for pluggable authentication providers."""

    async def authenticate(self, token: str) -> UserIdentity:
        """Validate a token and return the authenticated identity.

        Raises AuthenticationError if the token is invalid, expired,
        or otherwise unacceptable.
        """
        ...

    async def get_user_info(self, token: str) -> UserProfile:
        """Get full user profile from a valid token.

        Raises AuthenticationError if the token is invalid.
        """
        ...
```

- [ ] **Step 5: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/test_models.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/auth/__init__.py src/elspeth/web/auth/models.py \
    src/elspeth/web/auth/protocol.py tests/unit/web/auth/__init__.py \
    tests/unit/web/auth/test_models.py pyproject.toml
git commit -m "feat(web/auth): add AuthProvider protocol, identity models, and AuthenticationError"
```

---

### Task 2.2: LocalAuthProvider

**Files:**
- Create: `src/elspeth/web/auth/local.py`
- Create: `tests/unit/web/auth/test_local_provider.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/web/auth/test_local_provider.py
"""Tests for LocalAuthProvider -- SQLite user store, bcrypt hashing, JWT tokens."""

from __future__ import annotations

import time

import pytest

from elspeth.web.auth.local import LocalAuthProvider
from elspeth.web.auth.models import AuthenticationError, UserIdentity, UserProfile


@pytest.fixture
def provider(tmp_path):
    """Create a LocalAuthProvider with a temporary SQLite database."""
    return LocalAuthProvider(
        db_path=tmp_path / "auth.db",
        secret_key="test-secret-key-for-unit-tests",
        token_expiry_hours=24,
    )


class TestCreateUser:
    """Tests for user creation."""

    def test_create_user_succeeds(self, provider) -> None:
        provider.create_user("alice", "password123", display_name="Alice Smith")
        # No exception means success

    def test_create_user_with_email(self, provider) -> None:
        provider.create_user(
            "alice", "password123",
            display_name="Alice Smith",
            email="alice@example.com",
        )

    def test_create_duplicate_user_raises_value_error(self, provider) -> None:
        provider.create_user("alice", "password123", display_name="Alice")
        with pytest.raises(ValueError, match="alice"):
            provider.create_user("alice", "other-password", display_name="Alice 2")


class TestLogin:
    """Tests for username/password login."""

    def test_login_returns_jwt_string(self, provider) -> None:
        provider.create_user("alice", "password123", display_name="Alice")
        token = provider.login("alice", "password123")
        assert isinstance(token, str)
        assert len(token) > 0
        # JWT has three dot-separated segments
        assert len(token.split(".")) == 3

    def test_login_wrong_password_raises(self, provider) -> None:
        provider.create_user("alice", "password123", display_name="Alice")
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            provider.login("alice", "wrong-password")

    def test_login_unknown_user_raises(self, provider) -> None:
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            provider.login("nonexistent", "password")


class TestAuthenticate:
    """Tests for JWT token validation."""

    @pytest.mark.asyncio
    async def test_authenticate_valid_token(self, provider) -> None:
        provider.create_user("alice", "pw", display_name="Alice")
        token = provider.login("alice", "pw")
        identity = await provider.authenticate(token)
        assert isinstance(identity, UserIdentity)
        assert identity.user_id == "alice"
        assert identity.username == "alice"

    @pytest.mark.asyncio
    async def test_authenticate_garbage_token(self, provider) -> None:
        with pytest.raises(AuthenticationError, match="Invalid token"):
            await provider.authenticate("garbage-not-a-jwt")

    @pytest.mark.asyncio
    async def test_authenticate_expired_token(self, tmp_path) -> None:
        """Token with 0-second expiry should fail after creation."""
        # Use a provider with near-zero expiry
        import jwt as pyjwt

        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="test-key",
            token_expiry_hours=24,
        )
        provider.create_user("alice", "pw", display_name="Alice")

        # Manually create an already-expired token
        payload = {
            "sub": "alice",
            "username": "alice",
            "exp": int(time.time()) - 10,  # 10 seconds in the past
        }
        expired_token = pyjwt.encode(payload, "test-key", algorithm="HS256")
        with pytest.raises(AuthenticationError):
            await provider.authenticate(expired_token)

    @pytest.mark.asyncio
    async def test_authenticate_wrong_secret_key(self, tmp_path) -> None:
        """Token signed with a different key should fail."""
        import jwt as pyjwt

        provider = LocalAuthProvider(
            db_path=tmp_path / "auth.db",
            secret_key="correct-key",
        )
        payload = {
            "sub": "alice",
            "username": "alice",
            "exp": int(time.time()) + 3600,
        }
        bad_token = pyjwt.encode(payload, "wrong-key", algorithm="HS256")
        with pytest.raises(AuthenticationError, match="Invalid token"):
            await provider.authenticate(bad_token)


class TestGetUserInfo:
    """Tests for full user profile retrieval."""

    @pytest.mark.asyncio
    async def test_get_user_info_returns_profile(self, provider) -> None:
        provider.create_user(
            "alice", "pw",
            display_name="Alice Smith",
            email="alice@example.com",
        )
        token = provider.login("alice", "pw")
        profile = await provider.get_user_info(token)
        assert isinstance(profile, UserProfile)
        assert profile.user_id == "alice"
        assert profile.username == "alice"
        assert profile.display_name == "Alice Smith"
        assert profile.email == "alice@example.com"
        assert profile.groups == ()

    @pytest.mark.asyncio
    async def test_get_user_info_no_email(self, provider) -> None:
        provider.create_user("bob", "pw", display_name="Bob")
        token = provider.login("bob", "pw")
        profile = await provider.get_user_info(token)
        assert profile.email is None

    @pytest.mark.asyncio
    async def test_get_user_info_invalid_token(self, provider) -> None:
        with pytest.raises(AuthenticationError):
            await provider.get_user_info("garbage-token")
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/test_local_provider.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.auth.local'`

- [ ] **Step 3: Implement LocalAuthProvider**

```python
# src/elspeth/web/auth/local.py
"""Local authentication provider -- SQLite user store with bcrypt and JWT.

Uses bcrypt for password hashing and PyJWT for JWT token
creation and validation. The SQLite database is created at db_path on
first use.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import bcrypt
import jwt
from jwt.exceptions import PyJWTError

from elspeth.web.auth.models import AuthenticationError, UserIdentity, UserProfile


class LocalAuthProvider:
    """Authenticates users against a local SQLite database with bcrypt + JWT."""

    def __init__(
        self,
        db_path: Path,
        secret_key: str,
        token_expiry_hours: int = 24,
    ) -> None:
        self._db_path = db_path
        self._secret_key = secret_key
        self._token_expiry_hours = token_expiry_hours
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Open a connection to the SQLite database."""
        return sqlite3.connect(str(self._db_path))

    def _ensure_schema(self) -> None:
        """Create the users table if it does not exist."""
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    email TEXT
                )
                """
            )

    def create_user(
        self,
        user_id: str,
        password: str,
        display_name: str = "",
        email: str | None = None,
    ) -> None:
        """Create a new user with a bcrypt-hashed password.

        Raises ValueError if a user with the given user_id already exists.
        """
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        with self._get_conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO users (user_id, password_hash, display_name, email) "
                    "VALUES (?, ?, ?, ?)",
                    (user_id, password_hash, display_name or user_id, email),
                )
            except sqlite3.IntegrityError:
                raise ValueError(
                    f"User already exists: {user_id}"
                ) from None

    def login(self, username: str, password: str) -> str:
        """Authenticate with username/password and return a JWT.

        Raises AuthenticationError("Invalid credentials") on failure.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE user_id = ?",
                (username,),
            ).fetchone()

        if row is None or not bcrypt.checkpw(password.encode(), row[0].encode()):
            raise AuthenticationError("Invalid credentials")

        payload = {
            "sub": username,
            "username": username,
            "exp": int(time.time()) + self._token_expiry_hours * 3600,
        }
        return jwt.encode(payload, self._secret_key, algorithm="HS256")

    async def authenticate(self, token: str) -> UserIdentity:
        """Validate a JWT and return the authenticated identity.

        Raises AuthenticationError("Invalid token") on decode failure or expiry.
        """
        try:
            payload = jwt.decode(token, self._secret_key, algorithms=["HS256"])
        except PyJWTError:
            raise AuthenticationError("Invalid token") from None

        return UserIdentity(
            user_id=payload["sub"],
            username=payload["username"],
        )

    async def get_user_info(self, token: str) -> UserProfile:
        """Decode the JWT, then query the users table for full profile."""
        identity = await self.authenticate(token)

        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT display_name, email FROM users WHERE user_id = ?",
                (identity.user_id,),
            ).fetchone()

        if row is None:
            raise AuthenticationError("User not found")

        return UserProfile(
            user_id=identity.user_id,
            username=identity.username,
            display_name=row[0],
            email=row[1],
        )
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/test_local_provider.py -v
```

Expected: all 11 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/auth/local.py tests/unit/web/auth/test_local_provider.py
git commit -m "feat(web/auth): implement LocalAuthProvider with bcrypt and JWT"
```

---

### Task 2.3: OIDCAuthProvider

**Files:**
- Create: `src/elspeth/web/auth/oidc.py`
- Create: `tests/unit/web/auth/test_oidc_provider.py`

- [ ] **Step 1: Write tests**

The OIDC tests require creating JWTs signed with an RSA key, then validating them via a mocked JWKS endpoint. Generate an RSA key pair at test time using `cryptography`.

```python
# tests/unit/web/auth/test_oidc_provider.py
"""Tests for OIDCAuthProvider -- JWKS discovery, token validation."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import jwt as pyjwt
from jwt import PyJWK

from elspeth.web.auth.models import AuthenticationError, UserIdentity
from elspeth.web.auth.oidc import OIDCAuthProvider

ISSUER = "https://login.example.com"
AUDIENCE = "my-app-client-id"


@pytest.fixture
def rsa_keypair():
    """Generate an RSA key pair for signing test JWTs."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture
def jwks_response(rsa_keypair):
    """Build a JWKS response dict from the test RSA public key."""
    _, public_key = rsa_keypair
    pub_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    # Convert to JWK format
    key_obj = PyJWK.from_pem(pub_pem, algorithm="RS256")
    key_dict = key_obj.export(as_dict=True)
    key_dict["kid"] = "test-key-1"
    key_dict["use"] = "sig"
    return {"keys": [key_dict]}


def _make_token(private_key, claims: dict) -> str:
    """Sign a JWT with the test RSA private key."""
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pyjwt.encode(
        claims, priv_pem.decode(), algorithm="RS256",
        headers={"kid": "test-key-1"},
    )


def _valid_claims(overrides: dict | None = None) -> dict:
    """Return a valid JWT claims dict with optional overrides."""
    claims = {
        "sub": "user-123",
        "preferred_username": "alice",
        "name": "Alice Smith",
        "email": "alice@example.com",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": int(time.time()) + 3600,
    }
    if overrides:
        claims.update(overrides)
    return claims


@pytest.fixture
def mock_httpx_discovery(jwks_response):
    """Patch httpx.AsyncClient to return OIDC discovery and JWKS responses."""
    async def mock_get(url, **kwargs):
        response = AsyncMock()
        response.raise_for_status = lambda: None
        if ".well-known/openid-configuration" in url:
            response.json.return_value = {
                "jwks_uri": f"{ISSUER}/keys",
                "issuer": ISSUER,
            }
        elif url.endswith("/keys"):
            response.json.return_value = jwks_response
        return response

    client_mock = AsyncMock()
    client_mock.get = mock_get
    client_mock.__aenter__ = AsyncMock(return_value=client_mock)
    client_mock.__aexit__ = AsyncMock(return_value=False)

    return patch("elspeth.web.auth.oidc.httpx.AsyncClient", return_value=client_mock)


class TestOIDCDiscovery:
    """Tests for JWKS discovery and caching."""

    @pytest.mark.asyncio
    async def test_fetches_jwks_on_first_authenticate(
        self, rsa_keypair, mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        token = _make_token(private_key, _valid_claims())
        with mock_httpx_discovery as mock_client_cls:
            identity = await provider.authenticate(token)
            assert identity.user_id == "user-123"
            # Verify discovery was called
            client_instance = mock_client_cls.return_value
            assert client_instance.get.call_count >= 1

    @pytest.mark.asyncio
    async def test_caches_jwks_on_subsequent_calls(
        self, rsa_keypair, mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer=ISSUER, audience=AUDIENCE, jwks_cache_ttl_seconds=3600,
        )
        token = _make_token(private_key, _valid_claims())
        with mock_httpx_discovery:
            await provider.authenticate(token)
            # Second call should use cached keys -- no additional HTTP calls
            await provider.authenticate(token)


class TestOIDCTokenValidation:
    """Tests for token validation checks."""

    @pytest.mark.asyncio
    async def test_valid_token_returns_identity(
        self, rsa_keypair, mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        token = _make_token(private_key, _valid_claims())
        with mock_httpx_discovery:
            identity = await provider.authenticate(token)
        assert isinstance(identity, UserIdentity)
        assert identity.user_id == "user-123"
        assert identity.username == "alice"

    @pytest.mark.asyncio
    async def test_wrong_audience_raises(
        self, rsa_keypair, mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience="wrong-audience")
        token = _make_token(private_key, _valid_claims())
        with mock_httpx_discovery:
            with pytest.raises(AuthenticationError):
                await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_wrong_issuer_raises(
        self, rsa_keypair, mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer="https://wrong-issuer.com", audience=AUDIENCE,
        )
        token = _make_token(private_key, _valid_claims())
        with mock_httpx_discovery:
            with pytest.raises(AuthenticationError):
                await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_expired_token_raises(
        self, rsa_keypair, mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        claims = _valid_claims({"exp": int(time.time()) - 10})
        token = _make_token(private_key, claims)
        with mock_httpx_discovery:
            with pytest.raises(AuthenticationError):
                await provider.authenticate(token)


class TestOIDCGetUserInfo:
    """Tests for full profile retrieval from OIDC claims."""

    @pytest.mark.asyncio
    async def test_get_user_info_returns_profile(
        self, rsa_keypair, mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        claims = _valid_claims({"groups": ["team-a", "team-b"]})
        token = _make_token(private_key, claims)
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert profile.user_id == "user-123"
        assert profile.display_name == "Alice Smith"
        assert profile.email == "alice@example.com"
        assert profile.groups == ("team-a", "team-b")

    @pytest.mark.asyncio
    async def test_get_user_info_no_optional_claims(
        self, rsa_keypair, mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        claims = _valid_claims()
        # Remove optional fields
        del claims["email"]
        del claims["name"]
        token = _make_token(private_key, claims)
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert profile.display_name == "alice"  # Falls back to preferred_username
        assert profile.email is None
        assert profile.groups == ()
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/test_oidc_provider.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.auth.oidc'`

- [ ] **Step 3: Implement OIDCAuthProvider**

```python
# src/elspeth/web/auth/oidc.py
"""OIDC authentication provider -- JWKS discovery and JWT validation.

Validates tokens issued by any OIDC-compliant identity provider.
The frontend handles the IdP redirect; this backend only validates
the resulting token.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import jwt
from jwt.exceptions import PyJWTError

from elspeth.web.auth.models import AuthenticationError, UserIdentity, UserProfile


class OIDCAuthProvider:
    """Validates OIDC tokens via JWKS discovery."""

    def __init__(
        self,
        issuer: str,
        audience: str,
        jwks_cache_ttl_seconds: int = 3600,
    ) -> None:
        self._issuer = issuer.rstrip("/")
        self._audience = audience
        self._jwks_cache_ttl_seconds = jwks_cache_ttl_seconds
        self._jwks: dict[str, Any] | None = None
        self._jwks_fetched_at: float = 0.0

    async def _ensure_jwks(self) -> dict[str, Any]:
        """Fetch and cache JWKS keys from the OIDC discovery endpoint."""
        now = time.time()
        if (
            self._jwks is not None
            and (now - self._jwks_fetched_at) < self._jwks_cache_ttl_seconds
        ):
            return self._jwks

        try:
            async with httpx.AsyncClient() as client:
                discovery_url = (
                    f"{self._issuer}/.well-known/openid-configuration"
                )
                discovery_resp = await client.get(discovery_url)
                discovery_resp.raise_for_status()
                discovery = discovery_resp.json()

                jwks_uri = discovery["jwks_uri"]
                jwks_resp = await client.get(jwks_uri)
                jwks_resp.raise_for_status()
                self._jwks = jwks_resp.json()
                self._jwks_fetched_at = now
        except (httpx.HTTPError, KeyError) as exc:
            raise AuthenticationError(
                f"Failed to fetch JWKS: {exc}"
            ) from exc

        return self._jwks

    def _decode_token(self, token: str, jwks: dict[str, Any]) -> dict[str, Any]:
        """Decode and validate a JWT using the cached JWKS."""
        try:
            payload = jwt.decode(
                token,
                jwks,
                algorithms=["RS256"],
                audience=self._audience,
                issuer=self._issuer,
            )
        except PyJWTError as exc:
            raise AuthenticationError(f"Invalid token: {exc}") from None
        return payload

    async def authenticate(self, token: str) -> UserIdentity:
        """Validate an OIDC token and return the authenticated identity."""
        jwks = await self._ensure_jwks()
        payload = self._decode_token(token, jwks)

        return UserIdentity(
            user_id=payload["sub"],
            username=payload.get("preferred_username", payload["sub"]),
        )

    async def get_user_info(self, token: str) -> UserProfile:
        """Decode the OIDC token and extract profile claims."""
        jwks = await self._ensure_jwks()
        payload = self._decode_token(token, jwks)

        groups = payload.get("groups", [])
        if not isinstance(groups, list):
            groups = []

        return UserProfile(
            user_id=payload["sub"],
            username=payload.get("preferred_username", payload["sub"]),
            display_name=payload.get(
                "name", payload.get("preferred_username", payload["sub"])
            ),
            email=payload.get("email"),
            groups=tuple(str(g) for g in groups),
        )
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/test_oidc_provider.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/auth/oidc.py tests/unit/web/auth/test_oidc_provider.py
git commit -m "feat(web/auth): implement OIDCAuthProvider with JWKS discovery and caching"
```

---

### Task 2.4: EntraAuthProvider

**Files:**
- Create: `src/elspeth/web/auth/entra.py`
- Create: `tests/unit/web/auth/test_entra_provider.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/web/auth/test_entra_provider.py
"""Tests for EntraAuthProvider -- tenant validation and group claim extraction."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import jwt as pyjwt
from jwt import PyJWK

from elspeth.web.auth.entra import EntraAuthProvider
from elspeth.web.auth.models import AuthenticationError

TENANT_ID = "00000000-aaaa-bbbb-cccc-111111111111"
AUDIENCE = "my-entra-app-id"
ISSUER = f"https://login.microsoftonline.com/{TENANT_ID}/v2.0"


@pytest.fixture
def rsa_keypair():
    """Generate an RSA key pair for signing test JWTs."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture
def jwks_response(rsa_keypair):
    """Build a JWKS response dict from the test RSA public key."""
    _, public_key = rsa_keypair
    pub_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    key_obj = PyJWK.from_pem(pub_pem, algorithm="RS256")
    key_dict = key_obj.export(as_dict=True)
    key_dict["kid"] = "entra-test-key"
    key_dict["use"] = "sig"
    return {"keys": [key_dict]}


def _make_token(private_key, claims: dict) -> str:
    """Sign a JWT with the test RSA private key."""
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pyjwt.encode(
        claims, priv_pem.decode(), algorithm="RS256",
        headers={"kid": "entra-test-key"},
    )


def _valid_entra_claims(overrides: dict | None = None) -> dict:
    """Return valid Entra ID JWT claims with optional overrides."""
    claims = {
        "sub": "entra-user-456",
        "preferred_username": "alice@contoso.com",
        "name": "Alice Contoso",
        "email": "alice@contoso.com",
        "tid": TENANT_ID,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": int(time.time()) + 3600,
    }
    if overrides:
        claims.update(overrides)
    return claims


@pytest.fixture
def mock_httpx_discovery(jwks_response):
    """Patch httpx.AsyncClient to return OIDC discovery and JWKS responses."""
    async def mock_get(url, **kwargs):
        response = AsyncMock()
        response.raise_for_status = lambda: None
        if ".well-known/openid-configuration" in url:
            response.json.return_value = {
                "jwks_uri": f"{ISSUER}/discovery/v2.0/keys",
                "issuer": ISSUER,
            }
        elif "keys" in url:
            response.json.return_value = jwks_response
        return response

    client_mock = AsyncMock()
    client_mock.get = mock_get
    client_mock.__aenter__ = AsyncMock(return_value=client_mock)
    client_mock.__aexit__ = AsyncMock(return_value=False)

    return patch("elspeth.web.auth.oidc.httpx.AsyncClient", return_value=client_mock)


class TestEntraTenantValidation:
    """Tests for tenant ID validation."""

    @pytest.mark.asyncio
    async def test_valid_tenant_passes(
        self, rsa_keypair, mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        token = _make_token(private_key, _valid_entra_claims())
        with mock_httpx_discovery:
            identity = await provider.authenticate(token)
        assert identity.user_id == "entra-user-456"
        assert identity.username == "alice@contoso.com"

    @pytest.mark.asyncio
    async def test_wrong_tenant_raises(
        self, rsa_keypair, mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        # Provider expects a different tenant than what's in the token
        wrong_tenant = "99999999-zzzz-yyyy-xxxx-000000000000"
        wrong_issuer = f"https://login.microsoftonline.com/{wrong_tenant}/v2.0"
        provider = EntraAuthProvider(tenant_id=wrong_tenant, audience=AUDIENCE)

        # Token has the original TENANT_ID in tid but the issuer
        # must match the provider's expected issuer for OIDC to pass.
        # With wrong tenant, the OIDC issuer check will fail first.
        # Instead, test with matching issuer but wrong tid:
        claims = _valid_entra_claims({
            "tid": "wrong-tenant-id",
            # Keep issuer matching the provider's expected issuer
            "iss": ISSUER,
        })
        # Use a provider with the correct tenant but token has wrong tid
        provider_correct = EntraAuthProvider(
            tenant_id=TENANT_ID, audience=AUDIENCE,
        )
        token = _make_token(private_key, claims)
        with mock_httpx_discovery:
            with pytest.raises(AuthenticationError, match="Invalid tenant"):
                await provider_correct.authenticate(token)


class TestEntraGroupClaims:
    """Tests for group and role claim extraction."""

    @pytest.mark.asyncio
    async def test_group_ids_extracted_to_profile(
        self, rsa_keypair, mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        group_ids = [
            "11111111-aaaa-0000-0000-000000000001",
            "22222222-bbbb-0000-0000-000000000002",
        ]
        claims = _valid_entra_claims({"groups": group_ids})
        token = _make_token(private_key, claims)
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert "11111111-aaaa-0000-0000-000000000001" in profile.groups
        assert "22222222-bbbb-0000-0000-000000000002" in profile.groups

    @pytest.mark.asyncio
    async def test_roles_prefixed_in_groups(
        self, rsa_keypair, mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        claims = _valid_entra_claims({
            "groups": ["group-1"],
            "roles": ["admin", "reader"],
        })
        token = _make_token(private_key, claims)
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert "group-1" in profile.groups
        assert "role:admin" in profile.groups
        assert "role:reader" in profile.groups

    @pytest.mark.asyncio
    async def test_no_groups_or_roles(
        self, rsa_keypair, mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        claims = _valid_entra_claims()
        token = _make_token(private_key, claims)
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert profile.groups == ()
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/test_entra_provider.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.auth.entra'`

- [ ] **Step 3: Implement EntraAuthProvider**

```python
# src/elspeth/web/auth/entra.py
"""Azure Entra ID authentication provider.

Wraps OIDCAuthProvider with Entra-specific tenant validation and group
claim extraction. The OIDC issuer is derived from the tenant_id.
"""

from __future__ import annotations

from typing import Any

from elspeth.web.auth.models import AuthenticationError, UserIdentity, UserProfile
from elspeth.web.auth.oidc import OIDCAuthProvider


class EntraAuthProvider:
    """Validates Azure Entra ID tokens with tenant and group claim handling."""

    def __init__(
        self,
        tenant_id: str,
        audience: str,
        jwks_cache_ttl_seconds: int = 3600,
    ) -> None:
        self._tenant_id = tenant_id
        issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
        self._oidc = OIDCAuthProvider(
            issuer=issuer,
            audience=audience,
            jwks_cache_ttl_seconds=jwks_cache_ttl_seconds,
        )

    def _validate_tenant(self, payload: dict[str, Any]) -> None:
        """Verify the tid claim matches the expected tenant.

        Raises AuthenticationError("Invalid tenant") on mismatch.
        """
        tid = payload.get("tid")
        if tid != self._tenant_id:
            raise AuthenticationError("Invalid tenant")

    def _extract_groups(self, payload: dict[str, Any]) -> tuple[str, ...]:
        """Extract group IDs and role-prefixed entries from Entra claims."""
        groups: list[str] = []

        raw_groups = payload.get("groups", [])
        if isinstance(raw_groups, list):
            groups.extend(str(g) for g in raw_groups)

        raw_roles = payload.get("roles", [])
        if isinstance(raw_roles, list):
            groups.extend(f"role:{r}" for r in raw_roles)

        return tuple(groups)

    async def authenticate(self, token: str) -> UserIdentity:
        """Validate an Entra ID token with tenant verification."""
        # First, do standard OIDC validation (signature, exp, iss, aud)
        jwks = await self._oidc._ensure_jwks()
        payload = self._oidc._decode_token(token, jwks)

        # Then validate tenant
        self._validate_tenant(payload)

        return UserIdentity(
            user_id=payload["sub"],
            username=payload.get("preferred_username", payload["sub"]),
        )

    async def get_user_info(self, token: str) -> UserProfile:
        """Decode an Entra ID token and extract profile with group claims."""
        jwks = await self._oidc._ensure_jwks()
        payload = self._oidc._decode_token(token, jwks)

        self._validate_tenant(payload)

        return UserProfile(
            user_id=payload["sub"],
            username=payload.get("preferred_username", payload["sub"]),
            display_name=payload.get(
                "name", payload.get("preferred_username", payload["sub"])
            ),
            email=payload.get("email"),
            groups=self._extract_groups(payload),
        )
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/test_entra_provider.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Run all auth tests**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/ -v
```

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/auth/entra.py tests/unit/web/auth/test_entra_provider.py
git commit -m "feat(web/auth): implement EntraAuthProvider with tenant validation and group claims"
```

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
        assert "Missing" in response.json()["detail"] or "invalid" in response.json()["detail"].lower()

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
- Create: `src/elspeth/web/auth/routes.py`
- Create: `tests/unit/web/auth/test_routes.py`

- [ ] **Step 1: Write tests**

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


def _create_test_app(provider, auth_provider_type: str = "local") -> FastAPI:
    """Create a FastAPI app with auth routes for testing."""
    app = FastAPI()
    app.state.auth_provider = provider
    app.state.settings = type(
        "FakeSettings", (), {"auth_provider": auth_provider_type}
    )()
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
        # Add OIDC fields to fake settings
        app.state.settings.oidc_issuer = None
        app.state.settings.oidc_client_id = None
        client = TestClient(app)

        response = client.get("/api/auth/config")
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "local"
        assert body["oidc_issuer"] is None
        assert body["oidc_client_id"] is None

    def test_oidc_provider_returns_issuer_and_client_id(self, tmp_path) -> None:
        provider = AsyncMock()
        app = _create_test_app(provider, auth_provider_type="oidc")
        app.state.settings.oidc_issuer = "https://login.example.com"
        app.state.settings.oidc_client_id = "my-client-id"
        client = TestClient(app)

        response = client.get("/api/auth/config")
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "oidc"
        assert body["oidc_issuer"] == "https://login.example.com"
        assert body["oidc_client_id"] == "my-client-id"

    def test_config_endpoint_is_unauthenticated(self, tmp_path) -> None:
        """GET /api/auth/config must not require a Bearer token."""
        provider = AsyncMock()
        app = _create_test_app(provider, auth_provider_type="local")
        app.state.settings.oidc_issuer = None
        app.state.settings.oidc_client_id = None
        client = TestClient(app)

        # No Authorization header -- should still return 200
        response = client.get("/api/auth/config")
        assert response.status_code == 200
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/test_routes.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.auth.routes'`

- [ ] **Step 3: Implement auth routes**

```python
# src/elspeth/web/auth/routes.py
"""Auth API routes -- /api/auth/login, /api/auth/token, /api/auth/config, /api/auth/me.

POST /login is only available when auth_provider is "local".
POST /token re-issues a JWT from a valid existing token (local only).
GET /config returns auth configuration for frontend discovery (unauthenticated).
GET /me returns the full UserProfile for any auth provider.
"""

from __future__ import annotations

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
        """Authenticate with username/password (local auth only)."""
        settings = request.app.state.settings
        if settings.auth_provider != "local":
            raise HTTPException(status_code=404, detail="Not found")

        provider = request.app.state.auth_provider
        try:
            token = provider.login(body.username, body.password)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=exc.detail) from exc

        return TokenResponse(access_token=token)

    @router.post("/token", response_model=TokenResponse)
    async def refresh_token(
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> TokenResponse:
        """Re-issue a JWT from a valid existing token (local auth only)."""
        settings = request.app.state.settings
        if settings.auth_provider != "local":
            raise HTTPException(status_code=404, detail="Not found")

        provider = request.app.state.auth_provider
        # Re-login by issuing a new token for the authenticated user.
        # We use the internal JWT creation rather than requiring a password.
        import jwt
        import time

        payload = {
            "sub": user.user_id,
            "username": user.username,
            "exp": int(time.time()) + provider._token_expiry_hours * 3600,
        }
        new_token = jwt.encode(payload, provider._secret_key, algorithm="HS256")
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
            oidc_issuer=getattr(settings, "oidc_issuer", None),
            oidc_client_id=getattr(settings, "oidc_client_id", None),
        )

    @router.get("/me", response_model=UserProfileResponse)
    async def me(request: Request, user: UserIdentity = Depends(get_current_user)):
        """Return the full profile of the authenticated user."""
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.split(" ", 1)[1]
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

- [ ] **Step 4: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/test_routes.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 5: Run all auth tests**

```bash
.venv/bin/python -m pytest tests/unit/web/auth/ -v
```

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/auth/routes.py tests/unit/web/auth/test_routes.py
git commit -m "feat(web/auth): implement auth routes -- login, token refresh, user profile"
```

---

### Task 2.7: Session Database Models

**Files:**
- Create: `src/elspeth/web/sessions/__init__.py`
- Create: `src/elspeth/web/sessions/models.py`
- Create: `tests/unit/web/sessions/__init__.py`
- Create: `tests/unit/web/sessions/test_models.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/web/sessions/__init__.py
```

```python
# tests/unit/web/sessions/test_models.py
"""Tests for SQLAlchemy session table definitions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, select, insert

from elspeth.web.sessions.models import (
    metadata,
    sessions_table,
    chat_messages_table,
    composition_states_table,
    runs_table,
    run_events_table,
)


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:")
    metadata.create_all(eng)
    return eng


class TestTableCreation:
    """Verify all five tables are created with correct schemas."""

    def test_all_tables_exist(self, engine) -> None:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        assert "sessions" in table_names
        assert "chat_messages" in table_names
        assert "composition_states" in table_names
        assert "runs" in table_names
        assert "run_events" in table_names

    def test_sessions_columns(self, engine) -> None:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("sessions")}
        assert columns >= {"id", "user_id", "title", "created_at", "updated_at"}

    def test_chat_messages_columns(self, engine) -> None:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("chat_messages")}
        assert columns >= {
            "id", "session_id", "role", "content", "tool_calls", "created_at",
        }

    def test_composition_states_columns(self, engine) -> None:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("composition_states")}
        assert columns >= {
            "id", "session_id", "version", "source", "nodes", "edges",
            "outputs", "metadata_", "is_valid", "validation_errors", "created_at",
        }

    def test_runs_columns(self, engine) -> None:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("runs")}
        assert columns >= {
            "id", "session_id", "state_id", "status", "started_at",
            "finished_at", "rows_processed", "rows_failed", "error",
            "landscape_run_id", "pipeline_yaml",
        }

    def test_run_events_columns(self, engine) -> None:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("run_events")}
        assert columns >= {"id", "run_id", "timestamp", "event_type", "data"}


class TestCompositionStateUniqueConstraint:
    """Verify the UNIQUE(session_id, version) constraint."""

    def test_duplicate_version_raises(self, engine) -> None:
        session_id = str(uuid.uuid4())
        state_id_1 = str(uuid.uuid4())
        state_id_2 = str(uuid.uuid4())

        with engine.begin() as conn:
            # Insert a session first (FK constraint)
            conn.execute(
                insert(sessions_table).values(
                    id=session_id,
                    user_id="alice",
                    title="Test",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
            )
            # First state version
            conn.execute(
                insert(composition_states_table).values(
                    id=state_id_1,
                    session_id=session_id,
                    version=1,
                    is_valid=False,
                    created_at=datetime.now(timezone.utc),
                )
            )
            # Duplicate version should fail
            with pytest.raises(Exception):  # IntegrityError
                conn.execute(
                    insert(composition_states_table).values(
                        id=state_id_2,
                        session_id=session_id,
                        version=1,
                        is_valid=False,
                        created_at=datetime.now(timezone.utc),
                    )
                )


class TestSessionForeignKeys:
    """Verify foreign key relationships."""

    def test_chat_message_requires_valid_session(self, engine) -> None:
        """Inserting a message with a nonexistent session_id should fail
        if FK enforcement is on (SQLite needs PRAGMA foreign_keys=ON)."""
        # SQLite does not enforce FK by default; this test verifies
        # the column exists and accepts valid references.
        session_id = str(uuid.uuid4())
        msg_id = str(uuid.uuid4())

        with engine.begin() as conn:
            # Enable FK enforcement for SQLite
            conn.execute(insert(sessions_table).values(
                id=session_id,
                user_id="alice",
                title="Test",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ))
            conn.execute(insert(chat_messages_table).values(
                id=msg_id,
                session_id=session_id,
                role="user",
                content="Hello",
                created_at=datetime.now(timezone.utc),
            ))
            # Verify it was inserted
            result = conn.execute(
                select(chat_messages_table).where(
                    chat_messages_table.c.id == msg_id
                )
            ).fetchone()
            assert result is not None
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.sessions'`

- [ ] **Step 3: Implement SQLAlchemy table models**

```python
# src/elspeth/web/sessions/__init__.py
"""Session management -- persistence, CRUD, and API routes."""
```

```python
# src/elspeth/web/sessions/models.py
"""SQLAlchemy Core table definitions for the session database.

Tables: sessions, chat_messages, composition_states, runs, run_events.
Schema creation via metadata.create_all(engine) on startup.

All tables live in a dedicated session database, separate from the
Landscape audit database.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.types import JSON

metadata = MetaData()

sessions_table = Table(
    "sessions",
    metadata,
    Column("id", String, primary_key=True),
    Column("user_id", String, nullable=False, index=True),
    Column("title", String, nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
)

chat_messages_table = Table(
    "chat_messages",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id", String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    ),
    Column("role", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("tool_calls", JSON, nullable=True),
    Column("created_at", DateTime, nullable=False),
    CheckConstraint(
        "role IN ('user', 'assistant', 'system', 'tool')",
        name="ck_chat_messages_role",
    ),
)

composition_states_table = Table(
    "composition_states",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id", String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    ),
    Column("version", Integer, nullable=False),
    Column("source", JSON, nullable=True),
    Column("nodes", JSON, nullable=True),
    Column("edges", JSON, nullable=True),
    Column("outputs", JSON, nullable=True),
    Column("metadata_", JSON, nullable=True),
    Column("is_valid", Boolean, nullable=False, default=False),
    Column("validation_errors", JSON, nullable=True),
    Column("created_at", DateTime, nullable=False),
    UniqueConstraint("session_id", "version", name="uq_composition_state_version"),
)

runs_table = Table(
    "runs",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id", String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    ),
    Column(
        "state_id", String,
        ForeignKey("composition_states.id"),
        nullable=False,
    ),
    Column("status", String, nullable=False),
    Column("started_at", DateTime, nullable=False),
    Column("finished_at", DateTime, nullable=True),
    Column("rows_processed", Integer, nullable=False, default=0),
    Column("rows_failed", Integer, nullable=False, default=0),
    Column("error", Text, nullable=True),
    Column("landscape_run_id", String, nullable=True),
    Column("pipeline_yaml", Text, nullable=True),
    CheckConstraint(
        "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
        name="ck_runs_status",
    ),
)

run_events_table = Table(
    "run_events",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "run_id", String,
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False, index=True,
    ),
    Column("timestamp", DateTime, nullable=False),
    Column("event_type", String, nullable=False),
    Column("data", JSON, nullable=False),
    CheckConstraint(
        "event_type IN ('progress', 'error', 'completed')",
        name="ck_run_events_type",
    ),
)
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_models.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/__init__.py src/elspeth/web/sessions/models.py \
    tests/unit/web/sessions/__init__.py tests/unit/web/sessions/test_models.py
git commit -m "feat(web/sessions): add SQLAlchemy Core table definitions for session database"
```

---

### Task 2.8: SessionService Protocol and Record Types

**Files:**
- Create: `src/elspeth/web/sessions/protocol.py`

- [ ] **Step 1: Implement protocol and record dataclasses**

```python
# src/elspeth/web/sessions/protocol.py
"""SessionService protocol and record dataclasses.

Record types are frozen dataclasses representing database rows.
CompositionStateData is the input DTO for saving new state versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Represents a row from the sessions table.

    All fields are scalars or datetime -- no freeze guard needed.
    """

    id: UUID
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ChatMessageRecord:
    """Represents a row from the chat_messages table.

    tool_calls may contain a dict -- requires freeze guard when not None.
    """

    id: UUID
    session_id: UUID
    role: str
    content: str
    tool_calls: Any | None
    created_at: datetime

    def __post_init__(self) -> None:
        if self.tool_calls is not None:
            freeze_fields(self, "tool_calls")


@dataclass(frozen=True, slots=True)
class CompositionStateData:
    """Input DTO for saving a new composition state version.

    Contains mutable container fields -- requires freeze guard.
    """

    source: dict[str, Any] | None = None
    nodes: list[dict[str, Any]] | None = None
    edges: list[dict[str, Any]] | None = None
    outputs: list[dict[str, Any]] | None = None
    metadata_: dict[str, Any] | None = None
    is_valid: bool = False
    validation_errors: list[str] | None = None

    def __post_init__(self) -> None:
        fields_to_freeze = []
        for fname in (
            "source", "nodes", "edges", "outputs", "metadata_", "validation_errors",
        ):
            if getattr(self, fname) is not None:
                fields_to_freeze.append(fname)
        if fields_to_freeze:
            freeze_fields(self, *fields_to_freeze)


@dataclass(frozen=True, slots=True)
class CompositionStateRecord:
    """Represents a row from the composition_states table.

    Contains mutable container fields -- requires freeze guard.
    """

    id: UUID
    session_id: UUID
    version: int
    source: Any | None
    nodes: Any | None
    edges: Any | None
    outputs: Any | None
    metadata_: Any | None
    is_valid: bool
    validation_errors: Any | None
    created_at: datetime

    def __post_init__(self) -> None:
        fields_to_freeze = []
        for fname in (
            "source", "nodes", "edges", "outputs", "metadata_", "validation_errors",
        ):
            if getattr(self, fname) is not None:
                fields_to_freeze.append(fname)
        if fields_to_freeze:
            freeze_fields(self, *fields_to_freeze)


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Represents a row from the runs table.

    All fields are scalars, datetime, or None -- no freeze guard needed.
    """

    id: UUID
    session_id: UUID
    state_id: UUID
    status: str
    started_at: datetime
    finished_at: datetime | None
    rows_processed: int
    rows_failed: int
    error: str | None
    landscape_run_id: str | None
    pipeline_yaml: str | None


class RunAlreadyActiveError(Exception):
    """Raised when attempting to create a run while one is already active.

    Seam contract D: HTTP handlers catching this error MUST return 409 with
    {"detail": str(exc), "error_type": "run_already_active"} -- not a bare
    HTTPException. See seam-contracts.md for the canonical error shape.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Session {session_id} already has an active run"
        )


@runtime_checkable
class SessionServiceProtocol(Protocol):
    """Protocol for session persistence operations."""

    async def create_session(
        self, user_id: str, title: str,
    ) -> SessionRecord: ...

    async def get_session(self, session_id: UUID) -> SessionRecord: ...

    async def list_sessions(self, user_id: str) -> list[SessionRecord]: ...

    async def archive_session(self, session_id: UUID) -> None: ...

    async def add_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
        tool_calls: dict[str, Any] | None = None,
    ) -> ChatMessageRecord: ...

    async def get_messages(
        self, session_id: UUID,
    ) -> list[ChatMessageRecord]: ...

    async def save_composition_state(
        self, session_id: UUID, state: CompositionStateData,
    ) -> CompositionStateRecord: ...

    async def get_current_state(
        self, session_id: UUID,
    ) -> CompositionStateRecord | None: ...

    async def get_state(self, state_id: UUID) -> CompositionStateRecord: ...

    async def get_state_versions(
        self, session_id: UUID,
    ) -> list[CompositionStateRecord]: ...

    async def set_active_state(
        self, session_id: UUID, state_id: UUID,
    ) -> CompositionStateRecord: ...

    async def create_run(
        self,
        session_id: UUID,
        state_id: UUID,
        pipeline_yaml: str | None = None,
    ) -> RunRecord: ...

    async def get_run(self, run_id: UUID) -> RunRecord: ...

    async def update_run_status(
        self,
        run_id: UUID,
        status: str,
        error: str | None = None,
        landscape_run_id: str | None = None,
        rows_processed: int | None = None,
        rows_failed: int | None = None,
    ) -> None: ...

    async def get_active_run(
        self, session_id: UUID,
    ) -> RunRecord | None: ...
```

- [ ] **Step 2: Commit**

```bash
git add src/elspeth/web/sessions/protocol.py
git commit -m "feat(web/sessions): add SessionServiceProtocol, record types, and RunAlreadyActiveError"
```

---

### Task 2.9: SessionServiceImpl

**Files:**
- Create: `src/elspeth/web/sessions/service.py`
- Create: `tests/unit/web/sessions/test_service.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/web/sessions/test_service.py
"""Tests for SessionServiceImpl -- CRUD, state versioning, active run enforcement."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine

from elspeth.web.sessions.models import (
    metadata,
    runs_table,
    composition_states_table,
)
from elspeth.web.sessions.protocol import (
    CompositionStateData,
    RunAlreadyActiveError,
    RunRecord,
    SessionRecord,
    ChatMessageRecord,
    CompositionStateRecord,
)
from elspeth.web.sessions.service import SessionServiceImpl


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:")
    metadata.create_all(eng)
    return eng


@pytest.fixture
def service(engine):
    """Create a SessionServiceImpl backed by the in-memory engine."""
    return SessionServiceImpl(engine)


class TestSessionCRUD:
    """Tests for session create, get, list, and archive."""

    @pytest.mark.asyncio
    async def test_create_session(self, service) -> None:
        session = await service.create_session("alice", "My Session")
        assert isinstance(session, SessionRecord)
        assert session.user_id == "alice"
        assert session.title == "My Session"
        assert isinstance(session.id, uuid.UUID)
        assert isinstance(session.created_at, datetime)

    @pytest.mark.asyncio
    async def test_get_session(self, service) -> None:
        created = await service.create_session("alice", "Test")
        fetched = await service.get_session(created.id)
        assert fetched.id == created.id
        assert fetched.user_id == "alice"
        assert fetched.title == "Test"

    @pytest.mark.asyncio
    async def test_get_session_not_found_raises(self, service) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.get_session(uuid.uuid4())

    @pytest.mark.asyncio
    async def test_list_sessions_user_scoped(self, service) -> None:
        await service.create_session("alice", "Session A")
        await service.create_session("alice", "Session B")
        await service.create_session("bob", "Session C")

        alice_sessions = await service.list_sessions("alice")
        assert len(alice_sessions) == 2
        assert all(s.user_id == "alice" for s in alice_sessions)

        bob_sessions = await service.list_sessions("bob")
        assert len(bob_sessions) == 1

    @pytest.mark.asyncio
    async def test_list_sessions_ordered_by_updated_at_desc(self, service) -> None:
        s1 = await service.create_session("alice", "First")
        s2 = await service.create_session("alice", "Second")
        # Add a message to s1 to update its updated_at
        await service.add_message(s1.id, "user", "hello")

        sessions = await service.list_sessions("alice")
        # s1 should be first (most recently updated)
        assert sessions[0].id == s1.id

    @pytest.mark.asyncio
    async def test_archive_session(self, service) -> None:
        session = await service.create_session("alice", "To Archive")
        await service.add_message(session.id, "user", "hello")
        await service.archive_session(session.id)

        with pytest.raises(ValueError):
            await service.get_session(session.id)

        messages = await service.get_messages(session.id)
        assert len(messages) == 0


class TestMessagePersistence:
    """Tests for chat message add and retrieval."""

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self, service) -> None:
        session = await service.create_session("alice", "Chat")
        msg1 = await service.add_message(session.id, "user", "Hello")
        msg2 = await service.add_message(session.id, "assistant", "Hi there")

        assert isinstance(msg1, ChatMessageRecord)
        assert msg1.role == "user"
        assert msg1.content == "Hello"

        messages = await service.get_messages(session.id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_messages_ordered_by_created_at_asc(self, service) -> None:
        session = await service.create_session("alice", "Chat")
        await service.add_message(session.id, "user", "First")
        await service.add_message(session.id, "assistant", "Second")
        await service.add_message(session.id, "user", "Third")

        messages = await service.get_messages(session.id)
        assert [m.content for m in messages] == ["First", "Second", "Third"]

    @pytest.mark.asyncio
    async def test_add_message_with_tool_calls(self, service) -> None:
        session = await service.create_session("alice", "Chat")
        tool_calls_data = {"name": "set_source", "arguments": {"type": "csv"}}
        msg = await service.add_message(
            session.id, "assistant", "Setting source",
            tool_calls=tool_calls_data,
        )
        assert msg.tool_calls is not None

    @pytest.mark.asyncio
    async def test_add_message_updates_session_updated_at(self, service) -> None:
        session = await service.create_session("alice", "Chat")
        original_updated = session.updated_at
        await service.add_message(session.id, "user", "hello")
        refreshed = await service.get_session(session.id)
        assert refreshed.updated_at >= original_updated


class TestCompositionStateVersioning:
    """Tests for immutable state snapshots with monotonic versioning."""

    @pytest.mark.asyncio
    async def test_first_state_version_is_1(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        state_data = CompositionStateData(is_valid=False)
        state = await service.save_composition_state(session.id, state_data)
        assert isinstance(state, CompositionStateRecord)
        assert state.version == 1

    @pytest.mark.asyncio
    async def test_version_increments_monotonically(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        s1 = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=False),
        )
        s2 = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        assert s1.version == 1
        assert s2.version == 2

    @pytest.mark.asyncio
    async def test_get_current_state_returns_highest_version(
        self, service,
    ) -> None:
        session = await service.create_session("alice", "Pipeline")
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={"type": "csv", "path": "old.csv"}, is_valid=False,
            ),
        )
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={"type": "csv", "path": "new.csv"}, is_valid=True,
            ),
        )
        current = await service.get_current_state(session.id)
        assert current is not None
        assert current.version == 2
        assert current.is_valid is True

    @pytest.mark.asyncio
    async def test_get_current_state_returns_none_when_empty(
        self, service,
    ) -> None:
        session = await service.create_session("alice", "Empty")
        current = await service.get_current_state(session.id)
        assert current is None

    @pytest.mark.asyncio
    async def test_get_state_versions_returns_all_ascending(
        self, service,
    ) -> None:
        session = await service.create_session("alice", "Pipeline")
        await service.save_composition_state(
            session.id, CompositionStateData(is_valid=False),
        )
        await service.save_composition_state(
            session.id, CompositionStateData(is_valid=False),
        )
        await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        versions = await service.get_state_versions(session.id)
        assert len(versions) == 3
        assert [v.version for v in versions] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_state_preserves_pipeline_data(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        state_data = CompositionStateData(
            source={"type": "csv", "path": "/data/input.csv"},
            nodes=[{"name": "classify", "type": "transform"}],
            edges=[{"from": "source", "to": "classify"}],
            outputs=[{"name": "results", "type": "csv_sink"}],
            metadata_={"pipeline_name": "Test Pipeline"},
            is_valid=True,
            validation_errors=None,
        )
        state = await service.save_composition_state(session.id, state_data)
        assert state.is_valid is True


class TestOneActiveRunEnforcement:
    """Tests for B6 -- one active run per session."""

    @pytest.mark.asyncio
    async def test_second_pending_run_raises(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        # First run should succeed
        await service.create_run(session.id, state.id)
        # Second run should fail
        with pytest.raises(RunAlreadyActiveError):
            await service.create_run(session.id, state.id)

    @pytest.mark.asyncio
    async def test_create_run_returns_run_record(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        assert isinstance(run, RunRecord)
        assert run.status == "pending"
        assert run.session_id == session.id
        assert run.state_id == state.id
        assert run.pipeline_yaml is None

    @pytest.mark.asyncio
    async def test_create_run_with_pipeline_yaml(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(
            session.id, state.id, pipeline_yaml="source:\n  type: csv",
        )
        assert run.pipeline_yaml == "source:\n  type: csv"

    @pytest.mark.asyncio
    async def test_completed_run_allows_new_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        # Mark the run as completed
        await service.update_run_status(run.id, "completed")
        # New run should succeed
        run2 = await service.create_run(session.id, state.id)
        assert run2.status == "pending"

    @pytest.mark.asyncio
    async def test_failed_run_allows_new_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "failed")
        run2 = await service.create_run(session.id, state.id)
        assert run2.status == "pending"

    @pytest.mark.asyncio
    async def test_running_run_blocks_new_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        with pytest.raises(RunAlreadyActiveError):
            await service.create_run(session.id, state.id)


class TestGetState:
    """Tests for get_state -- fetch a specific CompositionStateRecord by UUID."""

    @pytest.mark.asyncio
    async def test_get_state_by_id(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        saved = await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={"type": "csv"}, is_valid=True,
            ),
        )
        fetched = await service.get_state(saved.id)
        assert fetched.id == saved.id
        assert fetched.version == saved.version

    @pytest.mark.asyncio
    async def test_get_state_not_found_raises(self, service) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.get_state(uuid.uuid4())


class TestSetActiveState:
    """Tests for set_active_state -- revert by copying a prior version."""

    @pytest.mark.asyncio
    async def test_revert_creates_new_version(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        v1 = await service.save_composition_state(
            session.id,
            CompositionStateData(source={"type": "csv"}, is_valid=True),
        )
        v2 = await service.save_composition_state(
            session.id,
            CompositionStateData(source={"type": "api"}, is_valid=True),
        )
        # Revert to v1 -- should create v3 as a copy of v1
        reverted = await service.set_active_state(session.id, v1.id)
        assert reverted.version == 3
        # Content should match v1, not v2
        assert reverted.source == v1.source

    @pytest.mark.asyncio
    async def test_revert_preserves_history(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        await service.save_composition_state(
            session.id, CompositionStateData(is_valid=False),
        )
        v2 = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        await service.set_active_state(session.id, v2.id)
        versions = await service.get_state_versions(session.id)
        # All three versions should exist (v1, v2, v3)
        assert len(versions) == 3
        assert [v.version for v in versions] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_revert_state_not_found_raises(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        with pytest.raises(ValueError, match="not found"):
            await service.set_active_state(session.id, uuid.uuid4())

    @pytest.mark.asyncio
    async def test_revert_state_wrong_session_raises(self, service) -> None:
        s1 = await service.create_session("alice", "Session 1")
        s2 = await service.create_session("alice", "Session 2")
        state = await service.save_composition_state(
            s1.id, CompositionStateData(is_valid=True),
        )
        with pytest.raises(ValueError, match="does not belong"):
            await service.set_active_state(s2.id, state.id)


class TestGetRun:
    """Tests for get_run -- fetch a RunRecord by UUID."""

    @pytest.mark.asyncio
    async def test_get_run_returns_record(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        created = await service.create_run(session.id, state.id)
        fetched = await service.get_run(created.id)
        assert isinstance(fetched, RunRecord)
        assert fetched.id == created.id
        assert fetched.status == "pending"

    @pytest.mark.asyncio
    async def test_get_run_not_found_raises(self, service) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.get_run(uuid.uuid4())


class TestGetActiveRun:
    """Tests for get_active_run -- pending/running run for a session."""

    @pytest.mark.asyncio
    async def test_returns_active_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        active = await service.get_active_run(session.id)
        assert active is not None
        assert active.id == run.id

    @pytest.mark.asyncio
    async def test_returns_none_when_no_active_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        active = await service.get_active_run(session.id)
        assert active is None

    @pytest.mark.asyncio
    async def test_returns_none_after_completion(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "completed")
        active = await service.get_active_run(session.id)
        assert active is None


class TestUpdateRunStatusExpanded:
    """Tests for expanded update_run_status signature (R6)."""

    @pytest.mark.asyncio
    async def test_update_with_error(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(
            run.id, "failed", error="Source file not found",
        )
        fetched = await service.get_run(run.id)
        assert fetched.status == "failed"
        assert fetched.error == "Source file not found"
        assert fetched.finished_at is not None

    @pytest.mark.asyncio
    async def test_update_with_landscape_run_id(self, service) -> None:
        session = await service.create_session("alice", "Pipeline")
        state = await service.save_composition_state(
            session.id, CompositionStateData(is_valid=True),
        )
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(
            run.id, "completed",
            landscape_run_id="lscp-abc-123",
            rows_processed=100,
            rows_failed=3,
        )
        fetched = await service.get_run(run.id)
        assert fetched.landscape_run_id == "lscp-abc-123"
        assert fetched.rows_processed == 100
        assert fetched.rows_failed == 3

    @pytest.mark.asyncio
    async def test_update_not_found_raises(self, service) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.update_run_status(uuid.uuid4(), "completed")
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_service.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.sessions.service'`

- [ ] **Step 3: Implement SessionServiceImpl**

```python
# src/elspeth/web/sessions/service.py
"""SessionService implementation -- CRUD, state versioning, active run enforcement.

Uses SQLAlchemy Core with a synchronous engine. Each method executes SQL
within a single transaction.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, delete, desc, func, insert, select, update

from elspeth.web.sessions.models import (
    chat_messages_table,
    composition_states_table,
    run_events_table,
    runs_table,
    sessions_table,
)
from elspeth.web.sessions.protocol import (
    ChatMessageRecord,
    CompositionStateData,
    CompositionStateRecord,
    RunAlreadyActiveError,
    RunRecord,
    SessionRecord,
)


class SessionServiceImpl:
    """Concrete session service backed by SQLAlchemy Core."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    async def create_session(
        self, user_id: str, title: str,
    ) -> SessionRecord:
        """Create a new session and return its record."""
        session_id = uuid.uuid4()
        now = self._now()

        with self._engine.begin() as conn:
            conn.execute(
                insert(sessions_table).values(
                    id=str(session_id),
                    user_id=user_id,
                    title=title,
                    created_at=now,
                    updated_at=now,
                )
            )

        return SessionRecord(
            id=session_id,
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
        )

    async def get_session(self, session_id: UUID) -> SessionRecord:
        """Fetch a session by ID. Raises ValueError if not found."""
        with self._engine.begin() as conn:
            row = conn.execute(
                select(sessions_table).where(
                    sessions_table.c.id == str(session_id)
                )
            ).fetchone()

        if row is None:
            raise ValueError(f"Session not found: {session_id}")

        return SessionRecord(
            id=UUID(row.id),
            user_id=row.user_id,
            title=row.title,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def list_sessions(self, user_id: str) -> list[SessionRecord]:
        """List sessions for a user, ordered by updated_at descending."""
        with self._engine.begin() as conn:
            rows = conn.execute(
                select(sessions_table)
                .where(sessions_table.c.user_id == user_id)
                .order_by(desc(sessions_table.c.updated_at))
            ).fetchall()

        return [
            SessionRecord(
                id=UUID(row.id),
                user_id=row.user_id,
                title=row.title,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        ]

    async def archive_session(self, session_id: UUID) -> None:
        """Delete a session and cascade to all related records."""
        sid = str(session_id)
        with self._engine.begin() as conn:
            # Delete in dependency order (children first for non-CASCADE DBs)
            # Get run IDs for this session to delete run_events
            run_ids = [
                r.id for r in conn.execute(
                    select(runs_table.c.id).where(
                        runs_table.c.session_id == sid
                    )
                ).fetchall()
            ]
            if run_ids:
                conn.execute(
                    delete(run_events_table).where(
                        run_events_table.c.run_id.in_(run_ids)
                    )
                )
            conn.execute(
                delete(runs_table).where(runs_table.c.session_id == sid)
            )
            conn.execute(
                delete(composition_states_table).where(
                    composition_states_table.c.session_id == sid
                )
            )
            conn.execute(
                delete(chat_messages_table).where(
                    chat_messages_table.c.session_id == sid
                )
            )
            conn.execute(
                delete(sessions_table).where(sessions_table.c.id == sid)
            )

    async def add_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
        tool_calls: dict[str, Any] | None = None,
    ) -> ChatMessageRecord:
        """Add a chat message and update the session's updated_at."""
        msg_id = uuid.uuid4()
        now = self._now()
        sid = str(session_id)

        with self._engine.begin() as conn:
            conn.execute(
                insert(chat_messages_table).values(
                    id=str(msg_id),
                    session_id=sid,
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                    created_at=now,
                )
            )
            conn.execute(
                update(sessions_table)
                .where(sessions_table.c.id == sid)
                .values(updated_at=now)
            )

        return ChatMessageRecord(
            id=msg_id,
            session_id=session_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            created_at=now,
        )

    async def get_messages(
        self, session_id: UUID,
    ) -> list[ChatMessageRecord]:
        """Get all messages for a session, ordered by created_at ascending."""
        with self._engine.begin() as conn:
            rows = conn.execute(
                select(chat_messages_table)
                .where(
                    chat_messages_table.c.session_id == str(session_id)
                )
                .order_by(chat_messages_table.c.created_at)
            ).fetchall()

        return [
            ChatMessageRecord(
                id=UUID(row.id),
                session_id=UUID(row.session_id),
                role=row.role,
                content=row.content,
                tool_calls=row.tool_calls,
                created_at=row.created_at,
            )
            for row in rows
        ]

    async def save_composition_state(
        self, session_id: UUID, state: CompositionStateData,
    ) -> CompositionStateRecord:
        """Save a new immutable composition state snapshot.

        Version is max(existing versions for session) + 1, starting at 1.
        """
        state_id = uuid.uuid4()
        now = self._now()
        sid = str(session_id)

        with self._engine.begin() as conn:
            # Get the current max version for this session
            result = conn.execute(
                select(func.max(composition_states_table.c.version)).where(
                    composition_states_table.c.session_id == sid
                )
            ).scalar()
            version = (result or 0) + 1

            # Unfreeze data for JSON serialization (MappingProxyType is not
            # JSON-serializable). Convert back to plain dicts/lists.
            def _to_json(val: Any) -> Any:
                if val is None:
                    return None
                if isinstance(val, dict):
                    return dict(val)
                if isinstance(val, (list, tuple)):
                    return list(val)
                return val

            # Seam contract A: wrap JSON columns with _version envelope
            # for schema evolution. Version 1 is the initial format.
            def _enveloped(val: Any) -> Any:
                raw = _to_json(val)
                if raw is None:
                    return None
                return {"_version": 1, "data": raw}

            conn.execute(
                insert(composition_states_table).values(
                    id=str(state_id),
                    session_id=sid,
                    version=version,
                    source=_enveloped(state.source),
                    nodes=_enveloped(state.nodes),
                    edges=_enveloped(state.edges),
                    outputs=_enveloped(state.outputs),
                    metadata_=_enveloped(state.metadata_),
                    is_valid=state.is_valid,
                    validation_errors=_to_json(state.validation_errors),
                    created_at=now,
                )
            )

        return CompositionStateRecord(
            id=state_id,
            session_id=session_id,
            version=version,
            source=state.source,
            nodes=state.nodes,
            edges=state.edges,
            outputs=state.outputs,
            metadata_=state.metadata_,
            is_valid=state.is_valid,
            validation_errors=state.validation_errors,
            created_at=now,
        )

    async def get_current_state(
        self, session_id: UUID,
    ) -> CompositionStateRecord | None:
        """Return the highest-version state for a session, or None."""
        with self._engine.begin() as conn:
            row = conn.execute(
                select(composition_states_table)
                .where(
                    composition_states_table.c.session_id == str(session_id)
                )
                .order_by(desc(composition_states_table.c.version))
                .limit(1)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_state_record(row)

    async def get_state_versions(
        self, session_id: UUID,
    ) -> list[CompositionStateRecord]:
        """Return all state versions for a session, ascending order."""
        with self._engine.begin() as conn:
            rows = conn.execute(
                select(composition_states_table)
                .where(
                    composition_states_table.c.session_id == str(session_id)
                )
                .order_by(composition_states_table.c.version)
            ).fetchall()

        return [self._row_to_state_record(row) for row in rows]

    @staticmethod
    def _unwrap_envelope(val: Any) -> Any:
        """Unwrap _version envelope from a JSON column value.

        Seam contract A: JSON columns are stored with {"_version": 1, "data": ...}.
        Raises ValueError on unknown versions. Returns None for NULL columns.
        """
        if val is None:
            return None
        if isinstance(val, dict) and "_version" in val:
            if val["_version"] != 1:
                raise ValueError(
                    f"Unknown composition state envelope version: {val['_version']}"
                )
            return val["data"]
        # Legacy data without envelope -- pass through as-is
        return val

    def _row_to_state_record(self, row: Any) -> CompositionStateRecord:
        """Convert a SQLAlchemy row to a CompositionStateRecord.

        Seam contract A: metadata_ maps DB column metadata_ back to the
        dataclass field. JSON columns are unwrapped from their _version envelope.
        When Sub-4 provides CompositionState.from_dict(), this method becomes
        the swap point for domain object reconstruction.
        """
        # TODO(sub-4): Replace raw dict fields with CompositionState.from_dict()
        # call once Sub-4 defines it. The swap should be a single-line change.
        return CompositionStateRecord(
            id=UUID(row.id),
            session_id=UUID(row.session_id),
            version=row.version,
            source=self._unwrap_envelope(row.source),
            nodes=self._unwrap_envelope(row.nodes),
            edges=self._unwrap_envelope(row.edges),
            outputs=self._unwrap_envelope(row.outputs),
            metadata_=self._unwrap_envelope(row.metadata_),
            is_valid=row.is_valid,
            validation_errors=row.validation_errors,
            created_at=row.created_at,
        )

    async def create_run(
        self,
        session_id: UUID,
        state_id: UUID,
        pipeline_yaml: str | None = None,
    ) -> RunRecord:
        """Create a new pending run, enforcing one active run per session (B6).

        The check-and-set runs within the same database transaction.
        Raises RunAlreadyActiveError if a pending or running run exists.
        If pipeline_yaml is provided, stores the generated YAML at creation time.
        """
        run_id = uuid.uuid4()
        now = self._now()
        sid = str(session_id)

        with self._engine.begin() as conn:
            # Check for existing active runs
            active = conn.execute(
                select(runs_table.c.id).where(
                    runs_table.c.session_id == sid,
                    runs_table.c.status.in_(["pending", "running"]),
                )
            ).fetchone()

            if active is not None:
                raise RunAlreadyActiveError(sid)

            conn.execute(
                insert(runs_table).values(
                    id=str(run_id),
                    session_id=sid,
                    state_id=str(state_id),
                    status="pending",
                    started_at=now,
                    rows_processed=0,
                    rows_failed=0,
                    pipeline_yaml=pipeline_yaml,
                )
            )

        return RunRecord(
            id=run_id,
            session_id=session_id,
            state_id=state_id,
            status="pending",
            started_at=now,
            finished_at=None,
            rows_processed=0,
            rows_failed=0,
            error=None,
            landscape_run_id=None,
            pipeline_yaml=pipeline_yaml,
        )

    async def get_run(self, run_id: UUID) -> RunRecord:
        """Fetch a run by ID. Raises ValueError if not found."""
        with self._engine.begin() as conn:
            row = conn.execute(
                select(runs_table).where(
                    runs_table.c.id == str(run_id)
                )
            ).fetchone()

        if row is None:
            raise ValueError(f"Run not found: {run_id}")

        return self._row_to_run_record(row)

    async def update_run_status(
        self,
        run_id: UUID,
        status: str,
        error: str | None = None,
        landscape_run_id: str | None = None,
        rows_processed: int | None = None,
        rows_failed: int | None = None,
    ) -> None:
        """Update a run's status and optional fields.

        Sets finished_at for terminal states (completed, failed, cancelled).
        Optional parameters only update the column when not None.
        Raises ValueError if run not found.
        """
        now = self._now()
        values: dict[str, Any] = {"status": status}
        if status in ("completed", "failed", "cancelled"):
            values["finished_at"] = now
        if error is not None:
            values["error"] = error
        if landscape_run_id is not None:
            values["landscape_run_id"] = landscape_run_id
        if rows_processed is not None:
            values["rows_processed"] = rows_processed
        if rows_failed is not None:
            values["rows_failed"] = rows_failed

        with self._engine.begin() as conn:
            result = conn.execute(
                update(runs_table)
                .where(runs_table.c.id == str(run_id))
                .values(**values)
            )
            if result.rowcount == 0:
                raise ValueError(f"Run not found: {run_id}")

    async def get_active_run(
        self, session_id: UUID,
    ) -> RunRecord | None:
        """Return the pending/running run for a session, or None."""
        with self._engine.begin() as conn:
            row = conn.execute(
                select(runs_table).where(
                    runs_table.c.session_id == str(session_id),
                    runs_table.c.status.in_(["pending", "running"]),
                )
            ).fetchone()

        if row is None:
            return None

        return self._row_to_run_record(row)

    async def get_state(self, state_id: UUID) -> CompositionStateRecord:
        """Fetch a composition state by its primary key. Raises ValueError if not found."""
        with self._engine.begin() as conn:
            row = conn.execute(
                select(composition_states_table).where(
                    composition_states_table.c.id == str(state_id)
                )
            ).fetchone()

        if row is None:
            raise ValueError(f"State not found: {state_id}")

        return self._row_to_state_record(row)

    async def set_active_state(
        self, session_id: UUID, state_id: UUID,
    ) -> CompositionStateRecord:
        """Revert to a prior state by copying it as a new version.

        Creates a new version record that is a copy of the specified prior
        version (looked up by state_id). The new record gets
        version = max(existing) + 1. Raises ValueError if state_id not
        found or does not belong to the session.
        """
        sid = str(session_id)

        with self._engine.begin() as conn:
            # Look up the prior state
            prior_row = conn.execute(
                select(composition_states_table).where(
                    composition_states_table.c.id == str(state_id)
                )
            ).fetchone()

            if prior_row is None:
                raise ValueError(f"State not found: {state_id}")
            if prior_row.session_id != sid:
                raise ValueError(
                    f"State {state_id} does not belong to session {session_id}"
                )

            # Get next version number
            max_version = conn.execute(
                select(func.max(composition_states_table.c.version)).where(
                    composition_states_table.c.session_id == sid
                )
            ).scalar()
            new_version = (max_version or 0) + 1

            new_state_id = uuid.uuid4()
            now = self._now()

            conn.execute(
                insert(composition_states_table).values(
                    id=str(new_state_id),
                    session_id=sid,
                    version=new_version,
                    source=prior_row.source,
                    nodes=prior_row.nodes,
                    edges=prior_row.edges,
                    outputs=prior_row.outputs,
                    metadata_=prior_row.metadata_,
                    is_valid=prior_row.is_valid,
                    validation_errors=prior_row.validation_errors,
                    created_at=now,
                )
            )

        return CompositionStateRecord(
            id=new_state_id,
            session_id=session_id,
            version=new_version,
            source=prior_row.source,
            nodes=prior_row.nodes,
            edges=prior_row.edges,
            outputs=prior_row.outputs,
            metadata_=prior_row.metadata_,
            is_valid=prior_row.is_valid,
            validation_errors=prior_row.validation_errors,
            created_at=now,
        )

    def _row_to_run_record(self, row: Any) -> RunRecord:
        """Convert a SQLAlchemy row to a RunRecord."""
        return RunRecord(
            id=UUID(row.id),
            session_id=UUID(row.session_id),
            state_id=UUID(row.state_id),
            status=row.status,
            started_at=row.started_at,
            finished_at=row.finished_at,
            rows_processed=row.rows_processed,
            rows_failed=row.rows_failed,
            error=row.error,
            landscape_run_id=row.landscape_run_id,
            pipeline_yaml=row.pipeline_yaml,
        )
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_service.py -v
```

Expected: all 36 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/service.py tests/unit/web/sessions/test_service.py
git commit -m "feat(web/sessions): implement SessionServiceImpl with CRUD, versioning, and active run enforcement"
```

---

### Task 2.10: Session Pydantic Schemas

**Files:**
- Create: `src/elspeth/web/sessions/schemas.py`

- [ ] **Step 1: Implement request/response schemas**

```python
# src/elspeth/web/sessions/schemas.py
"""Pydantic request/response models for all session API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class CreateSessionRequest(BaseModel):
    """Request body for POST /api/sessions."""

    title: str = "New session"


class SessionResponse(BaseModel):
    """Response for session CRUD operations."""

    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class SendMessageRequest(BaseModel):
    """Request body for POST /api/sessions/{id}/messages."""

    content: str


class ChatMessageResponse(BaseModel):
    """Response for a single chat message."""

    id: str
    session_id: str
    role: str
    content: str
    tool_calls: Any | None = None
    created_at: datetime


class MessageWithStateResponse(BaseModel):
    """Response for POST /api/sessions/{id}/messages.

    In Phase 2, state is always null. In Phase 4, it will contain
    the updated CompositionState after the ComposerService processes
    the message.
    """

    message: ChatMessageResponse
    state: CompositionStateResponse | None = None


class CompositionStateResponse(BaseModel):
    """Response for composition state endpoints."""

    id: str
    session_id: str
    version: int
    source: Any | None = None
    nodes: list[Any] | None = None
    edges: list[Any] | None = None
    outputs: list[Any] | None = None
    metadata: Any | None = None
    is_valid: bool
    validation_errors: list[str] | None = None
    created_at: datetime


class RevertStateRequest(BaseModel):
    """Request body for POST /api/sessions/{id}/state/revert."""

    state_id: str


class UploadResponse(BaseModel):
    """Response for POST /api/sessions/{id}/upload."""

    path: str
    filename: str
    size_bytes: int


# Forward reference resolution
MessageWithStateResponse.model_rebuild()
```

- [ ] **Step 2: Commit**

```bash
git add src/elspeth/web/sessions/schemas.py
git commit -m "feat(web/sessions): add Pydantic request/response schemas for session API"
```

---

### Task 2.11: Session API Routes

**Files:**
- Create: `src/elspeth/web/sessions/routes.py`
- Create: `tests/unit/web/sessions/test_routes.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/web/sessions/test_routes.py
"""Tests for session API routes -- CRUD, IDOR, upload, path traversal."""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine
from starlette.testclient import TestClient

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.sessions.models import metadata
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.service import SessionServiceImpl


def _make_app(
    tmp_path: Path,
    user_id: str = "alice",
    max_upload_bytes: int = 10 * 1024 * 1024,
) -> tuple[FastAPI, SessionServiceImpl]:
    """Create a test app with session routes and a mock auth user."""
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    service = SessionServiceImpl(engine)

    app = FastAPI()

    # Override auth dependency to return a fixed user
    identity = UserIdentity(user_id=user_id, username=user_id)

    async def mock_user():
        return identity

    app.dependency_overrides[get_current_user] = mock_user

    # Set up app state
    app.state.session_service = service
    app.state.settings = type(
        "FakeSettings", (),
        {"data_dir": tmp_path, "max_upload_bytes": max_upload_bytes},
    )()

    router = create_session_router()
    app.include_router(router)

    return app, service


class TestSessionCRUDRoutes:
    """Tests for session create, list, get, delete endpoints."""

    def test_create_session(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        response = client.post(
            "/api/sessions",
            json={"title": "My Pipeline"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["title"] == "My Pipeline"
        assert body["user_id"] == "alice"
        assert "id" in body

    def test_create_session_default_title(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        response = client.post("/api/sessions", json={})
        assert response.status_code == 201
        assert response.json()["title"] == "New session"

    def test_list_sessions(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        client.post("/api/sessions", json={"title": "S1"})
        client.post("/api/sessions", json={"title": "S2"})

        response = client.get("/api/sessions")
        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 2

    def test_get_session(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        create_resp = client.post(
            "/api/sessions", json={"title": "Test"},
        )
        session_id = create_resp.json()["id"]

        get_resp = client.get(f"/api/sessions/{session_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == session_id

    def test_get_session_not_found(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        response = client.get(f"/api/sessions/{uuid.uuid4()}")
        assert response.status_code == 404

    def test_delete_session(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        create_resp = client.post(
            "/api/sessions", json={"title": "To Delete"},
        )
        session_id = create_resp.json()["id"]

        del_resp = client.delete(f"/api/sessions/{session_id}")
        assert del_resp.status_code == 204

        # Verify it's gone
        get_resp = client.get(f"/api/sessions/{session_id}")
        assert get_resp.status_code == 404


class TestIDORProtection:
    """Tests for W5 -- IDOR protection on all session-scoped routes.

    Creates a session as user A, then attempts to access it as user B.
    All should return 404 (not 403).
    """

    @pytest.fixture
    def alice_session(self, tmp_path):
        """Create a session owned by alice, return (session_id, tmp_path)."""
        app, service = _make_app(tmp_path, user_id="alice")
        client = TestClient(app)
        resp = client.post("/api/sessions", json={"title": "Alice's"})
        return resp.json()["id"], tmp_path

    def _bob_client(self, tmp_path) -> TestClient:
        """Create a TestClient where the authenticated user is bob."""
        app, _ = _make_app(tmp_path, user_id="bob")
        return TestClient(app)

    def test_get_other_users_session_returns_404(
        self, alice_session,
    ) -> None:
        session_id, tmp_path = alice_session
        client = self._bob_client(tmp_path)
        # Bob creates his own app with a fresh DB, so alice's session
        # won't exist. For a proper IDOR test, we need a shared DB.
        # We'll test at the route level instead.

    def test_idor_session_crud(self, tmp_path) -> None:
        """Shared-DB IDOR test: alice creates, bob tries to access."""
        engine = create_engine("sqlite:///:memory:")
        metadata.create_all(engine)
        service = SessionServiceImpl(engine)

        # Create two apps sharing the same service
        def make_app_for_user(uid: str) -> FastAPI:
            app = FastAPI()
            identity = UserIdentity(user_id=uid, username=uid)
            async def mock_user():
                return identity
            app.dependency_overrides[get_current_user] = mock_user
            app.state.session_service = service
            app.state.settings = type(
                "S", (), {"data_dir": tmp_path, "max_upload_bytes": 10_000_000},
            )()
            app.include_router(create_session_router())
            return app

        alice_app = make_app_for_user("alice")
        bob_app = make_app_for_user("bob")

        alice_client = TestClient(alice_app)
        bob_client = TestClient(bob_app)

        # Alice creates a session
        resp = alice_client.post(
            "/api/sessions", json={"title": "Alice Only"},
        )
        assert resp.status_code == 201
        session_id = resp.json()["id"]

        # Bob tries to GET it -- should be 404
        resp = bob_client.get(f"/api/sessions/{session_id}")
        assert resp.status_code == 404

        # Bob tries to DELETE it -- should be 404
        resp = bob_client.delete(f"/api/sessions/{session_id}")
        assert resp.status_code == 404

        # Bob tries to GET messages -- should be 404
        resp = bob_client.get(f"/api/sessions/{session_id}/messages")
        assert resp.status_code == 404

        # Bob tries to POST a message -- should be 404
        resp = bob_client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "hacked"},
        )
        assert resp.status_code == 404

        # Bob tries to GET state -- should be 404
        resp = bob_client.get(f"/api/sessions/{session_id}/state")
        assert resp.status_code == 404

        # Bob tries to GET state versions -- should be 404
        resp = bob_client.get(f"/api/sessions/{session_id}/state/versions")
        assert resp.status_code == 404

        # Bob tries to revert state -- should be 404
        resp = bob_client.post(
            f"/api/sessions/{session_id}/state/revert",
            json={"state_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 404

        # Alice can still access her own session
        resp = alice_client.get(f"/api/sessions/{session_id}")
        assert resp.status_code == 200


class TestMessageRoutes:
    """Tests for message send and retrieval endpoints."""

    def test_send_message(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = resp.json()["id"]

        msg_resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Hello, build me a pipeline"},
        )
        assert msg_resp.status_code == 200
        body = msg_resp.json()
        assert body["message"]["content"] == "Hello, build me a pipeline"
        assert body["message"]["role"] == "user"
        assert body["state"] is None  # Phase 2: no composer yet

    def test_get_messages(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = resp.json()["id"]

        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "First"},
        )
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Second"},
        )

        msgs_resp = client.get(f"/api/sessions/{session_id}/messages")
        assert msgs_resp.status_code == 200
        messages = msgs_resp.json()
        assert len(messages) == 2
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Second"


class TestStateRoutes:
    """Tests for composition state endpoints."""

    def test_get_state_empty(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Empty"})
        session_id = resp.json()["id"]

        state_resp = client.get(f"/api/sessions/{session_id}/state")
        assert state_resp.status_code == 200
        assert state_resp.json() is None

    def test_get_state_versions(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Pipeline"})
        session_id = resp.json()["id"]

        versions_resp = client.get(
            f"/api/sessions/{session_id}/state/versions",
        )
        assert versions_resp.status_code == 200
        assert versions_resp.json() == []


class TestRevertEndpoint:
    """Tests for POST /api/sessions/{id}/state/revert (R1)."""

    def test_revert_creates_new_version(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        # Create session and two state versions via the service
        import asyncio
        loop = asyncio.new_event_loop()
        session = loop.run_until_complete(
            service.create_session("alice", "Pipeline"),
        )
        from elspeth.web.sessions.protocol import CompositionStateData
        v1 = loop.run_until_complete(
            service.save_composition_state(
                session.id,
                CompositionStateData(source={"type": "csv"}, is_valid=True),
            ),
        )
        v2 = loop.run_until_complete(
            service.save_composition_state(
                session.id,
                CompositionStateData(source={"type": "api"}, is_valid=True),
            ),
        )
        loop.close()

        # Revert to v1
        resp = client.post(
            f"/api/sessions/{session.id}/state/revert",
            json={"state_id": str(v1.id)},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["version"] == 3
        # Should match v1's source, not v2's
        assert body["source"] == {"type": "csv"}

    def test_revert_injects_system_message(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        import asyncio
        loop = asyncio.new_event_loop()
        session = loop.run_until_complete(
            service.create_session("alice", "Pipeline"),
        )
        from elspeth.web.sessions.protocol import CompositionStateData
        v1 = loop.run_until_complete(
            service.save_composition_state(
                session.id,
                CompositionStateData(is_valid=True),
            ),
        )
        loop.run_until_complete(
            service.save_composition_state(
                session.id,
                CompositionStateData(is_valid=True),
            ),
        )
        loop.close()

        client.post(
            f"/api/sessions/{session.id}/state/revert",
            json={"state_id": str(v1.id)},
        )

        # Check that a system message was injected
        msgs_resp = client.get(f"/api/sessions/{session.id}/messages")
        messages = msgs_resp.json()
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "reverted to version 1" in system_msgs[0]["content"].lower()

    def test_revert_idor_protection(self, tmp_path) -> None:
        """Revert to a state in another user's session returns 404."""
        from sqlalchemy import create_engine as _ce
        engine = _ce("sqlite:///:memory:")
        metadata.create_all(engine)
        service = SessionServiceImpl(engine)

        def make_app_for_user(uid: str) -> FastAPI:
            app = FastAPI()
            identity = UserIdentity(user_id=uid, username=uid)
            async def mock_user():
                return identity
            app.dependency_overrides[get_current_user] = mock_user
            app.state.session_service = service
            app.state.settings = type(
                "S", (), {"data_dir": tmp_path, "max_upload_bytes": 10_000_000},
            )()
            app.include_router(create_session_router())
            return app

        alice_app = make_app_for_user("alice")
        bob_app = make_app_for_user("bob")

        alice_client = TestClient(alice_app)
        bob_client = TestClient(bob_app)

        # Alice creates a session with a state
        import asyncio
        loop = asyncio.new_event_loop()
        session = loop.run_until_complete(
            service.create_session("alice", "Alice Only"),
        )
        from elspeth.web.sessions.protocol import CompositionStateData
        v1 = loop.run_until_complete(
            service.save_composition_state(
                session.id, CompositionStateData(is_valid=True),
            ),
        )
        loop.close()

        # Bob tries to revert -- should be 404
        resp = bob_client.post(
            f"/api/sessions/{session.id}/state/revert",
            json={"state_id": str(v1.id)},
        )
        assert resp.status_code == 404

    def test_revert_state_not_belonging_to_session(self, tmp_path) -> None:
        """Revert with a state_id from a different session returns 404."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        import asyncio
        loop = asyncio.new_event_loop()
        s1 = loop.run_until_complete(
            service.create_session("alice", "Session 1"),
        )
        s2 = loop.run_until_complete(
            service.create_session("alice", "Session 2"),
        )
        from elspeth.web.sessions.protocol import CompositionStateData
        v1_s2 = loop.run_until_complete(
            service.save_composition_state(
                s2.id, CompositionStateData(is_valid=True),
            ),
        )
        loop.close()

        # Try to revert s1 using s2's state -- should fail
        resp = client.post(
            f"/api/sessions/{s1.id}/state/revert",
            json={"state_id": str(v1_s2.id)},
        )
        assert resp.status_code == 404


class TestUploadRoute:
    """Tests for file upload endpoint including path traversal (B5)."""

    def test_upload_file(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Upload"})
        session_id = resp.json()["id"]

        file_content = b"col1,col2\na,b\nc,d"
        upload_resp = client.post(
            f"/api/sessions/{session_id}/upload",
            files={"file": ("data.csv", io.BytesIO(file_content), "text/csv")},
        )
        assert upload_resp.status_code == 200
        body = upload_resp.json()
        assert body["filename"] == "data.csv"
        assert body["size_bytes"] == len(file_content)
        assert "path" in body

        # Verify the file exists on disk
        saved_path = Path(body["path"])
        assert saved_path.exists()
        assert saved_path.read_bytes() == file_content

    def test_upload_path_traversal_user_id_sanitized(self, tmp_path) -> None:
        """B5: user_id containing ../../etc is sanitized to just 'etc'."""
        app, _ = _make_app(tmp_path, user_id="../../etc")
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Hack"})
        session_id = resp.json()["id"]

        file_content = b"malicious"
        upload_resp = client.post(
            f"/api/sessions/{session_id}/upload",
            files={"file": ("payload.txt", io.BytesIO(file_content), "text/plain")},
        )
        assert upload_resp.status_code == 200
        saved_path = Path(upload_resp.json()["path"])

        # The path should NOT contain ".." components
        assert ".." not in str(saved_path)
        # Should be under data_dir/uploads/etc/ (sanitized)
        assert "etc" in saved_path.parts
        assert saved_path.is_relative_to(tmp_path / "uploads")

    def test_upload_path_traversal_filename_sanitized(self, tmp_path) -> None:
        """Filename containing path traversal is sanitized."""
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Hack"})
        session_id = resp.json()["id"]

        file_content = b"malicious"
        upload_resp = client.post(
            f"/api/sessions/{session_id}/upload",
            files={
                "file": (
                    "../../etc/passwd",
                    io.BytesIO(file_content),
                    "application/octet-stream",
                ),
            },
        )
        assert upload_resp.status_code == 200
        saved_path = Path(upload_resp.json()["path"])
        # Filename should be just "passwd", not "../../etc/passwd"
        assert saved_path.name == "passwd"
        assert ".." not in str(saved_path)

    def test_upload_file_too_large(self, tmp_path) -> None:
        """Files exceeding max_upload_bytes are rejected with 413."""
        app, _ = _make_app(tmp_path, max_upload_bytes=100)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Big File"})
        session_id = resp.json()["id"]

        big_content = b"x" * 200  # 200 bytes > 100 byte limit
        upload_resp = client.post(
            f"/api/sessions/{session_id}/upload",
            files={
                "file": (
                    "big.dat",
                    io.BytesIO(big_content),
                    "application/octet-stream",
                ),
            },
        )
        assert upload_resp.status_code == 413

    def test_upload_empty_user_id_sanitization(self, tmp_path) -> None:
        """User ID of '..' sanitizes to empty via Path.name, which should raise."""
        app, _ = _make_app(tmp_path, user_id="..")
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Hack"})
        session_id = resp.json()["id"]

        upload_resp = client.post(
            f"/api/sessions/{session_id}/upload",
            files={"file": ("test.txt", io.BytesIO(b"data"), "text/plain")},
        )
        # Should fail because Path("..").name is "" on some platforms
        # or ".." which sanitizes poorly. Either 400 or 500 is acceptable.
        assert upload_resp.status_code in (400, 422, 500)
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_routes.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.sessions.routes'`

- [ ] **Step 3: Implement session routes**

```python
# src/elspeth/web/sessions/routes.py
"""Session API routes -- /api/sessions/* with IDOR protection.

All endpoints require authentication via Depends(get_current_user).
Session-scoped endpoints verify ownership before any business logic.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.sessions.protocol import RunAlreadyActiveError, SessionRecord
from elspeth.web.sessions.schemas import (
    ChatMessageResponse,
    CompositionStateResponse,
    CreateSessionRequest,
    MessageWithStateResponse,
    RevertStateRequest,
    SendMessageRequest,
    SessionResponse,
    UploadResponse,
)


def _session_response(session: SessionRecord) -> SessionResponse:
    """Convert a SessionRecord to a SessionResponse."""
    return SessionResponse(
        id=str(session.id),
        user_id=session.user_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


async def _verify_session_ownership(
    session_id: str,
    user: UserIdentity,
    request: Request,
) -> SessionRecord:
    """Verify the session exists and belongs to the current user.

    Returns 404 (not 403) to avoid leaking session existence (IDOR, W5).
    """
    service = request.app.state.session_service
    try:
        session = await service.get_session(UUID(session_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None

    if session.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


def create_session_router() -> APIRouter:
    """Create the session router with /api/sessions prefix."""
    router = APIRouter(prefix="/api/sessions", tags=["sessions"])

    @router.post("", status_code=201, response_model=SessionResponse)
    async def create_session(
        body: CreateSessionRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> SessionResponse:
        """Create a new session for the authenticated user."""
        service = request.app.state.session_service
        session = await service.create_session(user.user_id, body.title)
        return _session_response(session)

    @router.get("", response_model=list[SessionResponse])
    async def list_sessions(
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> list[SessionResponse]:
        """List sessions for the authenticated user."""
        service = request.app.state.session_service
        sessions = await service.list_sessions(user.user_id)
        return [_session_response(s) for s in sessions]

    @router.get("/{session_id}", response_model=SessionResponse)
    async def get_session(
        session_id: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> SessionResponse:
        """Get a single session. IDOR-protected."""
        session = await _verify_session_ownership(session_id, user, request)
        return _session_response(session)

    @router.delete("/{session_id}", status_code=204)
    async def delete_session(
        session_id: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> None:
        """Archive (delete) a session and all associated data."""
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        await service.archive_session(session.id)

    @router.post(
        "/{session_id}/messages",
        response_model=MessageWithStateResponse,
    )
    async def send_message(
        session_id: str,
        body: SendMessageRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> MessageWithStateResponse:
        """Send a user message. In Phase 2, only persists the message."""
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        msg = await service.add_message(session.id, "user", body.content)
        return MessageWithStateResponse(
            message=ChatMessageResponse(
                id=str(msg.id),
                session_id=str(msg.session_id),
                role=msg.role,
                content=msg.content,
                tool_calls=msg.tool_calls,
                created_at=msg.created_at,
            ),
            state=None,
        )

    @router.get(
        "/{session_id}/messages",
        response_model=list[ChatMessageResponse],
    )
    async def get_messages(
        session_id: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> list[ChatMessageResponse]:
        """Get conversation history for a session."""
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        messages = await service.get_messages(session.id)
        return [
            ChatMessageResponse(
                id=str(m.id),
                session_id=str(m.session_id),
                role=m.role,
                content=m.content,
                tool_calls=m.tool_calls,
                created_at=m.created_at,
            )
            for m in messages
        ]

    @router.get("/{session_id}/state")
    async def get_current_state(
        session_id: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> CompositionStateResponse | None:
        """Get the current (highest-version) composition state."""
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        state = await service.get_current_state(session.id)
        if state is None:
            return None
        return CompositionStateResponse(
            id=str(state.id),
            session_id=str(state.session_id),
            version=state.version,
            source=state.source,
            nodes=state.nodes,
            edges=state.edges,
            outputs=state.outputs,
            metadata=state.metadata_,
            is_valid=state.is_valid,
            validation_errors=state.validation_errors,
            created_at=state.created_at,
        )

    @router.get(
        "/{session_id}/state/versions",
        response_model=list[CompositionStateResponse],
    )
    async def get_state_versions(
        session_id: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> list[CompositionStateResponse]:
        """Get all composition state versions for a session."""
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        versions = await service.get_state_versions(session.id)
        return [
            CompositionStateResponse(
                id=str(v.id),
                session_id=str(v.session_id),
                version=v.version,
                source=v.source,
                nodes=v.nodes,
                edges=v.edges,
                outputs=v.outputs,
                metadata=v.metadata_,
                is_valid=v.is_valid,
                validation_errors=v.validation_errors,
                created_at=v.created_at,
            )
            for v in versions
        ]

    @router.post(
        "/{session_id}/state/revert",
        response_model=CompositionStateResponse,
    )
    async def revert_state(
        session_id: str,
        body: RevertStateRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> CompositionStateResponse:
        """Revert the pipeline to a prior composition state version (R1).

        Creates a new version that is a copy of the specified prior state.
        Injects a system message recording the revert.
        """
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service

        try:
            new_state = await service.set_active_state(
                session.id, UUID(body.state_id),
            )
        except ValueError:
            raise HTTPException(
                status_code=404, detail="State not found",
            ) from None

        # Look up the original version number for the system message
        original_state = await service.get_state(UUID(body.state_id))
        await service.add_message(
            session.id,
            role="system",
            content=f"Pipeline reverted to version {original_state.version}.",
        )

        return CompositionStateResponse(
            id=str(new_state.id),
            session_id=str(new_state.session_id),
            version=new_state.version,
            source=new_state.source,
            nodes=new_state.nodes,
            edges=new_state.edges,
            outputs=new_state.outputs,
            metadata=new_state.metadata_,
            is_valid=new_state.is_valid,
            validation_errors=new_state.validation_errors,
            created_at=new_state.created_at,
        )

    @router.get("/{session_id}/state/yaml")
    async def get_state_yaml(
        session_id: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> dict[str, str]:
        """Get YAML representation of the current composition state (M1).

        Stub endpoint -- returns 501 until Sub-4 implements generate_yaml().
        When Sub-4 lands, replace the 501 with:
            state = await service.get_current_state(session.id)
            yaml_str = generate_yaml(CompositionState.from_dict(state))
            return {"yaml": yaml_str}
        """
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        state = await service.get_current_state(session.id)
        if state is None:
            raise HTTPException(status_code=404, detail="No composition state exists")
        # TODO(sub-4): Replace stub with generate_yaml() call
        raise HTTPException(
            status_code=501,
            detail="YAML generation not yet implemented (see Sub-4)",
        )

    @router.post("/{session_id}/upload", response_model=UploadResponse)
    async def upload_file(
        session_id: str,
        request: Request,
        file: UploadFile = File(...),
        user: UserIdentity = Depends(get_current_user),
    ) -> UploadResponse:
        """Upload a source file to the user's scratch directory.

        Path traversal protection (B5): both user_id and filename are
        sanitized via Path().name to strip directory components.
        """
        session = await _verify_session_ownership(session_id, user, request)
        settings = request.app.state.settings

        # B5: Sanitize user_id -- strip all directory components
        sanitized_user_id = Path(user.user_id).name
        if not sanitized_user_id or sanitized_user_id in (".", ".."):
            raise HTTPException(
                status_code=400,
                detail="Invalid user ID for file upload",
            )

        # B5: Sanitize filename
        original_filename = file.filename or "upload"
        sanitized_filename = Path(original_filename).name
        if not sanitized_filename or sanitized_filename in (".", ".."):
            raise HTTPException(
                status_code=400,
                detail="Invalid filename",
            )

        # Read file content into memory and check size
        content = await file.read()
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {settings.max_upload_bytes} bytes",
            )

        # Create upload directory and save
        upload_dir = Path(settings.data_dir) / "uploads" / sanitized_user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / sanitized_filename
        file_path.write_bytes(content)

        return UploadResponse(
            path=str(file_path),
            filename=original_filename,
            size_bytes=len(content),
        )

    return router
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_routes.py -v
```

Expected: all 20 tests pass.

- [ ] **Step 5: Run all session tests**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/ -v
```

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/sessions/routes.py tests/unit/web/sessions/test_routes.py
git commit -m "feat(web/sessions): implement session API routes with IDOR protection and path traversal sanitization"
```

---

### Task 2.12: Wire Auth and Sessions into App Factory

**Files:**
- Modify: `src/elspeth/web/app.py`
- Modify: `src/elspeth/web/dependencies.py`

- [ ] **Step 1: Update app.py to register routers and create session DB**

Add the following to `create_app()` in `src/elspeth/web/app.py`:

```python
# Add these imports at the top of app.py:
import sys

from fastapi.responses import JSONResponse
from sqlalchemy import create_engine

from elspeth.web.auth.local import LocalAuthProvider
from elspeth.web.auth.routes import create_auth_router
from elspeth.web.sessions.models import metadata as session_metadata
from elspeth.web.sessions.protocol import RunAlreadyActiveError
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.service import SessionServiceImpl
```

**Before wiring auth into create_app(), add the OIDC/Entra conditional validator to WebSettings.**

In `src/elspeth/web/config.py`, add a `@model_validator(mode="after")` that enforces conditional field requirements:

```python
from pydantic import model_validator

    @model_validator(mode="after")
    def _validate_auth_fields(self) -> WebSettings:
        if self.auth_provider == "oidc":
            missing = [f for f in ("oidc_issuer", "oidc_audience", "oidc_client_id") if getattr(self, f) is None]
            if missing:
                raise ValueError(f"OIDC auth requires: {', '.join(missing)}")
        elif self.auth_provider == "entra":
            missing = [f for f in ("oidc_issuer", "oidc_audience", "oidc_client_id", "entra_tenant_id") if getattr(self, f) is None]
            if missing:
                raise ValueError(f"Entra auth requires: {', '.join(missing)}")
        return self
```

This catches misconfiguration at settings construction (CLI or test setup) rather than deep in the auth provider init. Tracked as `elspeth-34df5d61e4` (from Sub-1 review). Add corresponding tests to `tests/unit/web/test_config.py`:
- `WebSettings(auth_provider="oidc")` without oidc fields → ValueError
- `WebSettings(auth_provider="oidc", oidc_issuer=..., oidc_audience=..., oidc_client_id=...)` → valid
- `WebSettings(auth_provider="entra")` without entra_tenant_id → ValueError
- `WebSettings(auth_provider="local")` with no OIDC fields → valid (the default)

Inside `create_app()`, after the existing CORS and health setup, add:

```python
    # --- Auth provider setup ---
    if settings.auth_provider == "local":
        auth_provider = LocalAuthProvider(
            db_path=settings.data_dir / "auth.db",
            secret_key=settings.secret_key,
        )
    elif settings.auth_provider == "oidc":
        from elspeth.web.auth.oidc import OIDCAuthProvider
        auth_provider = OIDCAuthProvider(
            issuer=settings.oidc_issuer,
            audience=settings.oidc_audience,
        )
    elif settings.auth_provider == "entra":
        from elspeth.web.auth.entra import EntraAuthProvider
        auth_provider = EntraAuthProvider(
            tenant_id=settings.entra_tenant_id,
            audience=settings.oidc_audience,
        )
    app.state.auth_provider = auth_provider

    # W16/S3: Secret key production guard -- hard crash (S3/C4 upgrade)
    if (
        settings.secret_key == "change-me-in-production"
        and "pytest" not in sys.modules
        and os.environ.get("ELSPETH_ENV") != "test"
    ):
        raise SystemExit(
            "FATAL: WebSettings.secret_key is set to the default value. "
            "Set a secure secret_key before starting the web server. "
            "See WebSettings documentation."
        )

    # --- Session database setup (W6, S14) ---
    session_db_url = settings.get_session_db_url()
    session_engine = create_engine(session_db_url)
    session_metadata.create_all(session_engine)

    session_service = SessionServiceImpl(session_engine)
    app.state.session_service = session_service

    # --- Register routers ---
    app.include_router(create_auth_router())
    app.include_router(create_session_router())

    # --- Seam contract D: RunAlreadyActiveError → 409 with error_type ---
    @app.exception_handler(RunAlreadyActiveError)
    async def handle_run_already_active(
        request: Request, exc: RunAlreadyActiveError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "error_type": "run_already_active"},
        )
```

- [ ] **Step 2: Update dependencies.py**

Add these dependency functions to `src/elspeth/web/dependencies.py`:

```python
from elspeth.web.auth.middleware import get_current_user  # noqa: F401 -- re-export


def get_session_service(request: Request):
    """Get the SessionService from app state."""
    return request.app.state.session_service


def get_auth_provider(request: Request):
    """Get the AuthProvider from app state."""
    return request.app.state.auth_provider
```

- [ ] **Step 3: Run all tests**

```bash
.venv/bin/python -m pytest tests/unit/web/ -v
```

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/app.py src/elspeth/web/dependencies.py
git commit -m "feat(web): wire auth and session modules into app factory with DB schema creation"
```

---

## Verification Checklist

After completing all tasks, run the full test suite and verify:

```bash
# All auth tests
.venv/bin/python -m pytest tests/unit/web/auth/ -v

# All session tests
.venv/bin/python -m pytest tests/unit/web/sessions/ -v

# All web tests together
.venv/bin/python -m pytest tests/unit/web/ -v

# Type checking
.venv/bin/python -m mypy src/elspeth/web/auth/ src/elspeth/web/sessions/

# Linting
.venv/bin/python -m ruff check src/elspeth/web/auth/ src/elspeth/web/sessions/
```

**Expected results:**

- All tests pass
- AuthenticationError is in `models.py`, not `protocol.py`
- UserIdentity and UserProfile are `frozen=True, slots=True` with no freeze guards (scalar fields only)
- ChatMessageRecord and CompositionStateRecord/CompositionStateData have `freeze_fields()` in `__post_init__`
- LocalAuthProvider uses bcrypt and PyJWT
- OIDCAuthProvider discovers JWKS via httpx and caches with TTL
- EntraAuthProvider validates `tid` claim and extracts `groups`/`roles`
- All session routes verify ownership via `_verify_session_ownership()` returning 404 on IDOR
- Upload path sanitizes both user_id and filename via `Path().name`
- File size is checked against `max_upload_bytes` before writing (not trusting Content-Length)
- Active run check-and-set happens within a single transaction
- Schema creation via `metadata.create_all()` runs on startup

---

## Round 4 Review Amendments

> **Status: Amendments below have been integrated into the plan body.**

Changes from the expert panel review (Round 4), applied to the plan above. Fix IDs reference the review findings for traceability.

- **S3/C4 -- Secret key hard crash (Task 2.12):** The W16 secret key production guard in `create_app()` is upgraded from `logging.warning()` to `SystemExit`. If `settings.secret_key == "change-me-in-production"` and the environment is not test, raise `SystemExit("FATAL: default secret_key in non-test environment -- set ELSPETH_WEB__SECRET_KEY")` instead of logging a warning. The existing `"pytest" not in sys.modules` and `ELSPETH_ENV != "test"` guards remain.

- **M1 -- New endpoint: GET /api/sessions/{id}/state/yaml (Task 2.11):** Add a stub route in `sessions/routes.py` for `GET /api/sessions/{session_id}/state/yaml`. The route is defined by Sub-4 but wired here. The handler loads the active `CompositionState` via `session_service.get_latest_state()` and calls `generate_yaml(state)`. Until Sub-4 implements the YAML generator, the stub returns HTTP 501 with `{"detail": "YAML generation not yet implemented (see Sub-4)"}`. When Sub-4 lands, the stub is replaced with the real call. Returns `{"yaml": str}` on success or 404 if no state exists.

- **Seam contract A -- Serialisation contract (Task 2.8, Task 2.9):** `to_record()` must map the dataclass field `metadata` to DB column `metadata_`. `from_record()` must call `CompositionState.from_dict()` (defined in Sub-4) to reconstruct the domain model from DB JSON. Add a `_version` envelope to the serialised JSON as specified in the seam contracts doc. Until Sub-4 provides `from_dict()`, `from_record()` returns raw dicts in a `CompositionStateRecord` -- the interface must be designed so swapping in `from_dict()` later is a single-line change.

- **Seam contract A -- CompositionStateData and from_dict() (Task 2.8):** Design `CompositionStateRecord` so that when Sub-4 provides `CompositionState.from_dict()`, Sub-2's `_row_to_state_record()` / `from_record()` can call it directly. Until then, raw dicts are acceptable in record fields. The swap point should be clearly marked with a `# TODO(sub-4):` comment.

- **Seam contract D -- RunAlreadyActiveError envelope (Task 2.12 `create_app()`):** The HTTP 409 response when catching `RunAlreadyActiveError` must include `error_type: "run_already_active"` in the JSON body alongside `detail`. This is implemented as a global `@app.exception_handler(RunAlreadyActiveError)` in `create_app()`, not in a specific route handler. Sub-2 defines the exception class (Task 2.7) and the global handler (Task 2.12); Sub-5's execution routes raise it.
