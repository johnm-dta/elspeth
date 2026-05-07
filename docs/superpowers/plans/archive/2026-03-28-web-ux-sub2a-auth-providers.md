# Web UX Task-Plan 2A: Auth Protocol & Providers

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Implement the AuthProvider protocol and three provider implementations (Local, OIDC, Entra)
**Parent Plan:** `plans/2026-03-28-web-ux-sub2-auth-sessions.md`
**Spec:** `specs/2026-03-28-web-ux-sub2-auth-sessions-design.md`
**Depends On:** Sub-Plan 1 (Foundation) — completed
**Blocks:** Task-Plan 2B (Auth Middleware & Routes)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `pyproject.toml` | Add bcrypt to [webui] extra |
| Create | `src/elspeth/web/auth/__init__.py` | Module init |
| Create | `src/elspeth/web/auth/protocol.py` | AuthProvider protocol (two methods, no exceptions) |
| Create | `src/elspeth/web/auth/models.py` | UserIdentity, UserProfile, AuthenticationError |
| Create | `src/elspeth/web/auth/local.py` | LocalAuthProvider -- SQLite, bcrypt, JWT via PyJWT |
| Create | `src/elspeth/web/auth/oidc.py` | OIDCAuthProvider -- JWKS discovery via httpx, token validation |
| Create | `src/elspeth/web/auth/entra.py` | EntraAuthProvider -- tenant validation, group claims |
| Create | `tests/unit/web/auth/__init__.py` | Test package |
| Create | `tests/unit/web/auth/conftest.py` | Shared RSA keypair, JWKS response, and token signing fixtures |
| Create | `tests/unit/web/auth/test_models.py` | Auth model tests |
| Create | `tests/unit/web/auth/test_local_provider.py` | LocalAuthProvider tests |
| Create | `tests/unit/web/auth/test_oidc_provider.py` | OIDCAuthProvider tests |
| Create | `tests/unit/web/auth/test_entra_provider.py` | EntraAuthProvider tests |

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

    def test_create_user_empty_display_name_raises(self, provider) -> None:
        with pytest.raises(ValueError, match="display_name must not be empty"):
            provider.create_user("alice", "password123", display_name="")


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

import asyncio
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
        display_name: str,
        email: str | None = None,
    ) -> None:
        """Create a new user with a bcrypt-hashed password.

        Raises ValueError if a user with the given user_id already exists
        or if display_name is empty.
        """
        if not display_name:
            raise ValueError("display_name must not be empty")
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        with self._get_conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO users (user_id, password_hash, display_name, email) "
                    "VALUES (?, ?, ?, ?)",
                    (user_id, password_hash, display_name, email),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"User already exists: {user_id}"
                ) from exc

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
        except PyJWTError as exc:
            raise AuthenticationError("Invalid token") from exc

        return UserIdentity(
            user_id=payload["sub"],
            username=payload["username"],
        )

    def _query_user(self, user_id: str) -> tuple[str, str | None] | None:
        """Synchronous DB lookup — called via asyncio.to_thread."""
        with self._get_conn() as conn:
            return conn.execute(
                "SELECT display_name, email FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()

    async def get_user_info(self, token: str) -> UserProfile:
        """Decode the JWT, then query the users table for full profile.

        The DB query is offloaded to a thread to avoid blocking the
        event loop — sqlite3 is synchronous.
        """
        identity = await self.authenticate(token)

        row = await asyncio.to_thread(self._query_user, identity.user_id)

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

Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/auth/local.py tests/unit/web/auth/test_local_provider.py
git commit -m "feat(web/auth): implement LocalAuthProvider with bcrypt and JWT"
```

---

### Task 2.3: OIDCAuthProvider

**Files:**
- Create: `tests/unit/web/auth/conftest.py`
- Create: `src/elspeth/web/auth/oidc.py`
- Create: `tests/unit/web/auth/test_oidc_provider.py`

- [ ] **Step 1: Create shared test conftest and write tests**

The OIDC and Entra tests both need RSA key generation, JWKS response building, and token signing. Extract these into a shared conftest so Task 2.4 can reuse them.

```python
# tests/unit/web/auth/conftest.py
"""Shared fixtures for auth provider tests.

Provides RSA keypair generation, JWKS response building, and JWT
signing for both OIDC and Entra test modules.
"""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import jwt as pyjwt
from jwt import PyJWK


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


def make_rs256_token(private_key, claims: dict) -> str:
    """Sign a JWT with an RSA private key (RS256, kid=test-key-1).

    Not a fixture — a plain helper function imported explicitly by
    test modules that need it.
    """
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pyjwt.encode(
        claims, priv_pem.decode(), algorithm="RS256",
        headers={"kid": "test-key-1"},
    )
```

```python
# tests/unit/web/auth/test_oidc_provider.py
"""Tests for OIDCAuthProvider -- JWKS discovery, token validation."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from elspeth.web.auth.models import AuthenticationError, UserIdentity
from elspeth.web.auth.oidc import OIDCAuthProvider
from tests.unit.web.auth.conftest import make_rs256_token


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
        token = make_rs256_token(private_key, _valid_claims())
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
        token = make_rs256_token(private_key, _valid_claims())
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
        token = make_rs256_token(private_key, _valid_claims())
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
        token = make_rs256_token(private_key, _valid_claims())
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
        token = make_rs256_token(private_key, _valid_claims())
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
        token = make_rs256_token(private_key, claims)
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
        token = make_rs256_token(private_key, claims)
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
        token = make_rs256_token(private_key, claims)
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
            raise AuthenticationError(f"Invalid token: {exc}") from exc
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
git add src/elspeth/web/auth/oidc.py tests/unit/web/auth/conftest.py \
    tests/unit/web/auth/test_oidc_provider.py
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

from elspeth.web.auth.entra import EntraAuthProvider
from elspeth.web.auth.models import AuthenticationError
from tests.unit.web.auth.conftest import make_rs256_token

TENANT_ID = "00000000-aaaa-bbbb-cccc-111111111111"
AUDIENCE = "my-entra-app-id"
ISSUER = f"https://login.microsoftonline.com/{TENANT_ID}/v2.0"


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
        token = make_rs256_token(private_key, _valid_entra_claims())
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
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery:
            with pytest.raises(AuthenticationError, match="Invalid tenant"):
                await provider_correct.authenticate(token)

    @pytest.mark.asyncio
    async def test_missing_tid_claim_raises(
        self, rsa_keypair, mock_httpx_discovery,
    ) -> None:
        """Token without a tid claim should fail with a specific message."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        claims = _valid_entra_claims()
        del claims["tid"]
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery:
            with pytest.raises(AuthenticationError, match="Missing tenant claim"):
                await provider.authenticate(token)


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
        token = make_rs256_token(private_key, claims)
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
        token = make_rs256_token(private_key, claims)
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
        token = make_rs256_token(private_key, claims)
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

Inherits from OIDCAuthProvider, adding Entra-specific tenant validation
and group/role claim extraction. The OIDC issuer is derived from the
tenant_id.
"""

from __future__ import annotations

from typing import Any

from elspeth.web.auth.models import AuthenticationError, UserIdentity, UserProfile
from elspeth.web.auth.oidc import OIDCAuthProvider


class EntraAuthProvider(OIDCAuthProvider):
    """Validates Azure Entra ID tokens with tenant and group claim handling.

    Extends OIDCAuthProvider with:
    - Tenant ID verification (``tid`` claim must match expected tenant)
    - Group claim extraction (``groups`` + ``role:``-prefixed ``roles``)
    """

    def __init__(
        self,
        tenant_id: str,
        audience: str,
        jwks_cache_ttl_seconds: int = 3600,
    ) -> None:
        self._tenant_id = tenant_id
        issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
        super().__init__(
            issuer=issuer,
            audience=audience,
            jwks_cache_ttl_seconds=jwks_cache_ttl_seconds,
        )

    def _validate_tenant(self, payload: dict[str, Any]) -> None:
        """Verify the tid claim matches the expected tenant.

        Raises AuthenticationError if ``tid`` is missing or mismatched.
        The ``tid`` claim is required in Entra ID tokens — absence
        indicates a non-Entra token or a configuration error.
        """
        try:
            tid = payload["tid"]
        except KeyError as exc:
            raise AuthenticationError(
                "Missing tenant claim (tid) — token may not be from Entra ID"
            ) from exc
        if tid != self._tenant_id:
            raise AuthenticationError("Invalid tenant")

    def _extract_groups(self, payload: dict[str, Any]) -> tuple[str, ...]:
        """Extract group IDs and role-prefixed entries from Entra claims.

        ``groups`` and ``roles`` are optional Entra claims (Tier 3 data
        from the IdP) — ``.get()`` with empty-list default is correct
        here because absence means "no groups/roles assigned."
        """
        groups: list[str] = []

        raw_groups = payload.get("groups", [])
        if isinstance(raw_groups, list):
            groups.extend(str(g) for g in raw_groups)

        raw_roles = payload.get("roles", [])
        if isinstance(raw_roles, list):
            groups.extend(f"role:{r}" for r in raw_roles)

        return tuple(groups)

    async def authenticate(self, token: str) -> UserIdentity:
        """Validate an Entra ID token with tenant verification.

        Performs standard OIDC validation (signature, expiry, issuer,
        audience) via the inherited _decode_token, then checks the
        tenant claim.
        """
        jwks = await self._ensure_jwks()
        payload = self._decode_token(token, jwks)

        self._validate_tenant(payload)

        return UserIdentity(
            user_id=payload["sub"],
            username=payload.get("preferred_username", payload["sub"]),
        )

    async def get_user_info(self, token: str) -> UserProfile:
        """Decode an Entra ID token and extract profile with group claims."""
        jwks = await self._ensure_jwks()
        payload = self._decode_token(token, jwks)

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

Expected: all 6 tests pass.

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

## Self-Review Checklist

Before marking Task-Plan 2A complete, verify:

- [ ] **pyproject.toml** has `bcrypt` in the `[webui]` extra
- [ ] **AuthProvider protocol** (`protocol.py`) defines exactly two async methods: `authenticate` and `get_user_info`
- [ ] **Auth models** (`models.py`) -- `UserIdentity` and `UserProfile` are `frozen=True, slots=True`; `AuthenticationError` has a `detail` attribute
- [ ] **LocalAuthProvider** (`local.py`) -- creates SQLite schema on init, bcrypt-hashes passwords, issues HS256 JWTs, validates token expiry
- [ ] **OIDCAuthProvider** (`oidc.py`) -- fetches JWKS via discovery, caches with TTL, validates RS256 tokens against issuer and audience
- [ ] **EntraAuthProvider** (`entra.py`) -- inherits from OIDCAuthProvider, validates `tid` claim, extracts `groups` and `role:`-prefixed `roles`
- [ ] **All test files exist** and pass: `test_models.py` (9 tests), `test_local_provider.py` (12 tests), `test_oidc_provider.py` (8 tests), `test_entra_provider.py` (6 tests)
- [ ] **No freeze guards needed** -- all dataclass fields are scalars, `None`, or `tuple[str, ...]`
- [ ] **No defensive `.get()` on our own types** -- `.get()` only used on external JWT payloads (Tier 3 data)
- [ ] **Full test suite still passes**: `.venv/bin/python -m pytest tests/unit/web/auth/ -v`
