"""Local authentication provider -- SQLite user store with bcrypt and JWT.

Uses bcrypt for password hashing and PyJWT for JWT token creation
and validation. The SQLite database is created at db_path on first use.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import bcrypt
import jwt
from jwt.exceptions import PyJWTError

from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth.models import AuthenticationError, UserIdentity, UserProfile
from elspeth.web.validation import has_visible_content

_EMAIL_VERIFICATION_TOKEN_BYTES = 32
_EMAIL_VERIFICATION_TOKEN_TTL_SECONDS = 24 * 60 * 60


def _required_visible_string_claim(payload: dict[str, object], claim_name: str) -> str:
    """Extract a required local-JWT claim as a visible string."""
    try:
        value = payload[claim_name]
    except KeyError as exc:
        raise AuthenticationError("Invalid token") from exc
    if not isinstance(value, str) or not has_visible_content(value):
        raise AuthenticationError("Invalid token")
    return value


def _verification_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


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

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a transaction-scoped SQLite connection and always close it."""
        conn = self._get_conn()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        """Create the users table if it does not exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    email TEXT,
                    email_verified INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
            if "email_verified" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 1")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS email_verification_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    used_at INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
                """
            )

    def create_user(
        self,
        user_id: str,
        password: str,
        display_name: str,
        email: str | None = None,
        *,
        email_verified: bool = True,
    ) -> None:
        """Create a new user with a bcrypt-hashed password.

        Raises ValueError if a user with the given user_id already exists
        or if display_name is empty.
        """
        if not display_name:
            raise ValueError("display_name must not be empty")
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO users (user_id, password_hash, display_name, email, email_verified) VALUES (?, ?, ?, ?, ?)",
                    (
                        user_id,
                        password_hash,
                        display_name,
                        email,
                        1 if email_verified else 0,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"User already exists: {user_id}") from exc

    def delete_user(self, user_id: str) -> bool:
        """Delete a local auth user and any pending verification tokens."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM email_verification_tokens WHERE user_id = ?",
                (user_id,),
            )
            cursor = conn.execute(
                "DELETE FROM users WHERE user_id = ?",
                (user_id,),
            )
            return cursor.rowcount > 0

    def create_email_verification_token(
        self,
        user_id: str,
        *,
        ttl_seconds: int = _EMAIL_VERIFICATION_TOKEN_TTL_SECONDS,
    ) -> str:
        """Create a one-use verification token for an unverified local user."""
        now = int(time.time())
        token = secrets.token_urlsafe(_EMAIL_VERIFICATION_TOKEN_BYTES)
        token_hash = _verification_token_hash(token)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT email_verified FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"User not found: {user_id}")
            if row[0]:
                raise ValueError(f"User already verified: {user_id}")
            conn.execute(
                "DELETE FROM email_verification_tokens WHERE user_id = ? AND used_at IS NULL",
                (user_id,),
            )
            conn.execute(
                """
                INSERT INTO email_verification_tokens
                    (token_hash, user_id, created_at, expires_at, used_at)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (token_hash, user_id, now, now + ttl_seconds),
            )
        return token

    def verify_email_token(self, token: str) -> UserIdentity:
        """Consume a verification token and activate the corresponding user."""
        token_hash = _verification_token_hash(token)
        now = int(time.time())
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_id, expires_at, used_at
                FROM email_verification_tokens
                WHERE token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if row is None:
                raise AuthenticationError("Invalid email verification token")
            user_id, expires_at, used_at = row
            if used_at is not None:
                raise AuthenticationError("Email verification token already used")
            if expires_at < now:
                raise AuthenticationError("Email verification token expired")
            user_row = conn.execute(
                "SELECT 1 FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if user_row is None:
                raise AuthenticationError("User not found")
            conn.execute(
                "UPDATE users SET email_verified = 1 WHERE user_id = ?",
                (user_id,),
            )
            conn.execute(
                "UPDATE email_verification_tokens SET used_at = ? WHERE token_hash = ?",
                (now, token_hash),
            )
        return UserIdentity(user_id=user_id, username=user_id)

    def restore_email_verification_token(self, token: str, user_id: str) -> bool:
        """Compensate a just-completed verification when required audit fails.

        The token and user activation are restored together, guarded by the
        exact token/user relationship and the consumed marker observed in the
        same transaction. A successful return makes the original request safe
        to retry.
        """
        token_hash = _verification_token_hash(token)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT tokens.used_at, users.email_verified
                FROM email_verification_tokens AS tokens
                JOIN users ON users.user_id = tokens.user_id
                WHERE tokens.token_hash = ? AND tokens.user_id = ?
                """,
                (token_hash, user_id),
            ).fetchone()
            if row is None or row[0] is None or not row[1]:
                return False
            used_at = row[0]
            user_result = conn.execute(
                "UPDATE users SET email_verified = 0 WHERE user_id = ? AND email_verified = 1",
                (user_id,),
            )
            token_result = conn.execute(
                """
                UPDATE email_verification_tokens
                SET used_at = NULL
                WHERE token_hash = ? AND user_id = ? AND used_at = ?
                """,
                (token_hash, user_id, used_at),
            )
            if user_result.rowcount != 1 or token_result.rowcount != 1:
                raise RuntimeError("Email verification compensation lost its state precondition")
        return True

    def issue_token_for_user(self, user_id: str, username: str) -> str:
        """Issue a local JWT for an already-authorized user."""
        return self._issue_token(user_id, username)

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

        with self._connect() as conn:
            row = conn.execute(
                "SELECT password_hash, email_verified FROM users WHERE user_id = ?",
                (username,),
            ).fetchone()

        if row is None:
            # Constant-time: hash against dummy to prevent timing oracle
            bcrypt.checkpw(password.encode(), self._get_dummy_hash())
            raise AuthenticationError("Invalid credentials")

        if not bcrypt.checkpw(password.encode(), row[0].encode()):
            raise AuthenticationError("Invalid credentials")

        if not row[1]:
            raise AuthenticationError("Email verification required")

        return self._issue_token(username, username)

    def _issue_token(self, user_id: str, username: str, *, issued_at: int | None = None) -> str:
        now = int(time.time())
        iat = now if issued_at is None else issued_at
        payload = {
            "sub": user_id,
            "username": username,
            "iat": iat,
            "exp": now + self._token_expiry_hours * 3600,
        }
        token: str = jwt.encode(payload, self._secret_key, algorithm="HS256")
        return token

    async def refresh(self, user_id: str, username: str, *, original_iat: int) -> str:
        """Issue a new JWT for an already-authenticated user.

        Verifies the user still exists in the database — a deleted
        user must not be able to obtain fresh tokens via refresh.

        Called by the token refresh route. Does NOT re-verify
        credentials — the caller (get_current_user middleware)
        has already validated the existing token.

        Blocking sqlite work is offloaded to a bounded worker.
        """
        return await run_sync_in_worker(self._refresh_sync, user_id, username, original_iat)

    def _refresh_sync(self, user_id: str, username: str, original_iat: int | None) -> str:
        """Synchronous refresh — called via run_sync_in_worker."""
        now = int(time.time())

        # Max refresh chain: reject if the original token was issued too
        # long ago.  This bounds how long a stolen token can be refreshed
        # indefinitely without re-authentication.  Without a session DB
        # (Sub-2c/2d), this is the only revocation-like mechanism.
        if original_iat is None:
            raise AuthenticationError("Token missing iat — please re-authenticate")
        chain_age_hours = (now - original_iat) / 3600
        if chain_age_hours > self._max_refresh_chain_hours:
            raise AuthenticationError("Token refresh chain expired — please re-authenticate")

        with self._connect() as conn:
            row = conn.execute(
                "SELECT email_verified FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            raise AuthenticationError("User not found")
        if not row[0]:
            raise AuthenticationError("Email verification required")

        # Carry forward the original iat so the chain age accumulates.
        # New logins get a fresh iat; refreshes preserve the original.
        return self._issue_token(user_id, username, issued_at=original_iat)

    async def authenticate(self, token: str) -> UserIdentity:
        """Validate a JWT and return the authenticated identity.

        Raises AuthenticationError("Invalid token") on decode failure, expiry,
        or if the user has been deleted since the token was issued.
        """
        try:
            payload = jwt.decode(token, self._secret_key, algorithms=["HS256"])
        except PyJWTError as exc:
            raise AuthenticationError("Invalid token") from exc

        user_id = _required_visible_string_claim(payload, "sub")
        username = _required_visible_string_claim(payload, "username")

        # Verify user still exists — deleted users must not retain access
        exists = await run_sync_in_worker(self._user_exists, user_id)
        if not exists:
            raise AuthenticationError("Invalid token")

        return UserIdentity(
            user_id=user_id,
            username=username,
        )

    def _user_exists(self, user_id: str) -> bool:
        """Check if a verified user still exists in auth.db."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE user_id = ? AND email_verified = 1",
                (user_id,),
            ).fetchone()
            return row is not None

    def _query_user(self, user_id: str) -> tuple[str, str | None] | None:
        """Synchronous DB lookup — called via run_sync_in_worker."""
        with self._connect() as conn:
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
