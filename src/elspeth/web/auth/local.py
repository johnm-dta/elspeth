"""Local authentication provider -- SQLite user store with bcrypt and JWT.

Uses bcrypt for password hashing and PyJWT for JWT token creation
and validation. The SQLite database is created at db_path on first use.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import bcrypt
import jwt
from jwt.exceptions import PyJWTError

from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth.models import AuthenticationError, UserIdentity, UserProfile


class LocalAuthProvider:
    """Authenticates users against a local SQLite database with bcrypt + JWT."""

    # Pre-computed dummy hash for constant-time comparison on failed lookups.
    # Eagerly initialized to avoid a data race on first concurrent access.
    # The ~200ms bcrypt cost is paid once at class load time.
    _dummy_hash: bytes = bcrypt.hashpw(b"dummy", bcrypt.gensalt())

    @classmethod
    def _get_dummy_hash(cls) -> bytes:
        return cls._dummy_hash

    def __init__(
        self,
        db_path: Path,
        secret_key: str,
        token_expiry_hours: int = 24,
        max_refresh_chain_hours: int = 168,
    ) -> None:
        self._db_path = db_path
        self._secret_key = secret_key
        self._token_expiry_hours = token_expiry_hours
        self._max_refresh_chain_hours = max_refresh_chain_hours
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
                    "INSERT INTO users (user_id, password_hash, display_name, email) VALUES (?, ?, ?, ?)",
                    (user_id, password_hash, display_name, email),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"User already exists: {user_id}") from exc

    async def login(self, username: str, password: str) -> str:
        """Authenticate with username/password and return a JWT.

        Raises AuthenticationError("Invalid credentials") on failure.
        Uses constant-time comparison to prevent username enumeration
        via timing side-channel.

        Blocking bcrypt/sqlite work is offloaded to a bounded worker.
        """
        return await run_sync_in_worker(self._login_sync, username, password)

    def _login_sync(self, username: str, password: str) -> str:
        """Synchronous login — called via run_sync_in_worker."""
        # Early rejection for empty credentials. This exits before the
        # bcrypt path, so it is faster than a real login attempt. This is
        # acceptable: empty credentials are syntactically invalid (not a
        # guessable input), so the timing difference does not enable
        # credential enumeration.
        if not username or not password:
            raise AuthenticationError("Invalid credentials")

        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE user_id = ?",
                (username,),
            ).fetchone()

        if row is None:
            # Constant-time: hash against dummy to prevent timing oracle
            bcrypt.checkpw(password.encode(), self._get_dummy_hash())
            raise AuthenticationError("Invalid credentials")

        if not bcrypt.checkpw(password.encode(), row[0].encode()):
            raise AuthenticationError("Invalid credentials")

        now = int(time.time())
        payload = {
            "sub": username,
            "username": username,
            "iat": now,
            "exp": now + self._token_expiry_hours * 3600,
        }
        token: str = jwt.encode(payload, self._secret_key, algorithm="HS256")
        return token

    async def refresh(self, user_id: str, username: str, *, original_iat: int | None = None) -> str:
        """Issue a new JWT for an already-authenticated user.

        Verifies the user still exists in the database — a deleted
        user must not be able to obtain fresh tokens via refresh.

        Called by the token refresh route. Does NOT re-verify
        credentials — the caller (get_current_user middleware)
        has already validated the existing token.

        Blocking sqlite work is offloaded to a bounded worker.
        """
        return await run_sync_in_worker(self._refresh_sync, user_id, username, original_iat)

    def _refresh_sync(self, user_id: str, username: str, original_iat: int | None = None) -> str:
        """Synchronous refresh — called via run_sync_in_worker."""
        now = int(time.time())

        # Max refresh chain: reject if the original token was issued too
        # long ago.  This bounds how long a stolen token can be refreshed
        # indefinitely without re-authentication.  Without a session DB
        # (Sub-2c/2d), this is the only revocation-like mechanism.
        if original_iat is not None:
            chain_age_hours = (now - original_iat) / 3600
            if chain_age_hours > self._max_refresh_chain_hours:
                raise AuthenticationError("Token refresh chain expired — please re-authenticate")

        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            raise AuthenticationError("User not found")

        # Carry forward the original iat so the chain age accumulates.
        # New logins get a fresh iat; refreshes preserve the original.
        token_iat = original_iat if original_iat is not None else now
        payload = {
            "sub": user_id,
            "username": username,
            "iat": token_iat,
            "exp": now + self._token_expiry_hours * 3600,
        }
        token: str = jwt.encode(payload, self._secret_key, algorithm="HS256")
        return token

    async def authenticate(self, token: str) -> UserIdentity:
        """Validate a JWT and return the authenticated identity.

        Raises AuthenticationError("Invalid token") on decode failure, expiry,
        or if the user has been deleted since the token was issued.
        """
        try:
            payload = jwt.decode(token, self._secret_key, algorithms=["HS256"])
        except PyJWTError as exc:
            raise AuthenticationError("Invalid token") from exc

        user_id = payload["sub"]

        # Verify user still exists — deleted users must not retain access
        exists = await run_sync_in_worker(self._user_exists, user_id)
        if not exists:
            raise AuthenticationError("Invalid token")

        return UserIdentity(
            user_id=user_id,
            username=payload["username"],
        )

    def _user_exists(self, user_id: str) -> bool:
        """Check if a user still exists in auth.db (sync, called via to_thread)."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return row is not None

    def _query_user(self, user_id: str) -> tuple[str, str | None] | None:
        """Synchronous DB lookup — called via run_sync_in_worker."""
        with self._get_conn() as conn:
            row: tuple[str, str | None] | None = conn.execute(
                "SELECT display_name, email FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return row

    async def get_user_info(self, token: str) -> UserProfile:
        """Decode the JWT, then query the users table for full profile.

        The DB query is offloaded to a thread to avoid blocking the
        event loop — sqlite3 is synchronous.
        """
        identity = await self.authenticate(token)

        row = await run_sync_in_worker(self._query_user, identity.user_id)

        if row is None:
            raise AuthenticationError("User not found")

        return UserProfile(
            user_id=identity.user_id,
            username=identity.username,
            display_name=row[0],
            email=row[1],
        )
