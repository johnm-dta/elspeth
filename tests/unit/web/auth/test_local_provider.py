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
            "alice",
            "password123",
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

    @pytest.mark.asyncio
    async def test_login_returns_jwt_string(self, provider) -> None:
        provider.create_user("alice", "password123", display_name="Alice")
        token = await provider.login("alice", "password123")
        assert isinstance(token, str)
        assert len(token) > 0
        # JWT has three dot-separated segments
        assert len(token.split(".")) == 3

    @pytest.mark.asyncio
    async def test_login_wrong_password_raises(self, provider) -> None:
        provider.create_user("alice", "password123", display_name="Alice")
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            await provider.login("alice", "wrong-password")

    @pytest.mark.asyncio
    async def test_login_unknown_user_raises(self, provider) -> None:
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            await provider.login("nonexistent", "password")


class TestAuthenticate:
    """Tests for JWT token validation."""

    @pytest.mark.asyncio
    async def test_authenticate_valid_token(self, provider) -> None:
        provider.create_user("alice", "pw", display_name="Alice")
        token = await provider.login("alice", "pw")
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
    async def test_authenticate_deleted_user_rejected(self, provider) -> None:
        """A deleted user's JWT must be rejected by authenticate()."""
        import sqlite3

        provider.create_user("alice", "pw", display_name="Alice")
        token = await provider.login("alice", "pw")

        # Delete the user behind the provider's back
        with sqlite3.connect(str(provider._db_path)) as conn:
            conn.execute("DELETE FROM users WHERE user_id = ?", ("alice",))

        with pytest.raises(AuthenticationError, match="Invalid token"):
            await provider.authenticate(token)

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
            "alice",
            "pw",
            display_name="Alice Smith",
            email="alice@example.com",
        )
        token = await provider.login("alice", "pw")
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
        token = await provider.login("bob", "pw")
        profile = await provider.get_user_info(token)
        assert profile.email is None

    @pytest.mark.asyncio
    async def test_get_user_info_invalid_token(self, provider) -> None:
        with pytest.raises(AuthenticationError):
            await provider.get_user_info("garbage-token")

    @pytest.mark.asyncio
    async def test_get_user_info_deleted_user(self, provider) -> None:
        """User deleted between login (token issued) and get_user_info call."""
        provider.create_user("alice", "pw", display_name="Alice")
        token = await provider.login("alice", "pw")

        # Access _db_path directly — no public API to delete users by design
        import sqlite3

        with sqlite3.connect(str(provider._db_path)) as conn:
            conn.execute("DELETE FROM users WHERE user_id = ?", ("alice",))

        with pytest.raises(AuthenticationError, match="Invalid token"):
            await provider.get_user_info(token)


class TestLoginEdgeCases:
    """Edge-case tests for login input validation."""

    @pytest.mark.asyncio
    async def test_login_empty_username_raises(self, provider) -> None:
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            await provider.login("", "some-password")

    @pytest.mark.asyncio
    async def test_login_empty_password_raises(self, provider) -> None:
        provider.create_user("alice", "pw", display_name="Alice")
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            await provider.login("alice", "")


class TestProtocolConformance:
    """Verify LocalAuthProvider satisfies the AuthProvider protocol."""

    def test_local_satisfies_auth_provider(self, provider) -> None:
        from elspeth.web.auth.protocol import AuthProvider

        assert isinstance(provider, AuthProvider)


class TestTimingDefense:
    """Verify constant-time behavior for unknown users."""

    @pytest.mark.asyncio
    async def test_login_unknown_user_still_hashes(self, provider) -> None:
        """Verify constant-time behavior: bcrypt.checkpw is called even for unknown users."""
        import unittest.mock as mock

        with mock.patch("elspeth.web.auth.local.bcrypt.checkpw", return_value=False) as mock_checkpw:
            with pytest.raises(AuthenticationError, match="Invalid credentials"):
                await provider.login("nonexistent", "password")
            # bcrypt.checkpw must be called even for nonexistent users (timing defense)
            mock_checkpw.assert_called_once()


class TestRefresh:
    """Tests for the token refresh method."""

    @pytest.mark.asyncio
    async def test_refresh_deleted_user_raises(self, provider) -> None:
        """A deleted user cannot obtain fresh tokens via refresh."""
        import sqlite3

        provider.create_user("alice", "pw", display_name="Alice")
        # Access _db_path directly — no public API to delete users by design
        with sqlite3.connect(str(provider._db_path)) as conn:
            conn.execute("DELETE FROM users WHERE user_id = ?", ("alice",))
        with pytest.raises(AuthenticationError, match="User not found"):
            await provider.refresh("alice", "alice")

    @pytest.mark.asyncio
    async def test_refresh_valid_user_returns_jwt(self, provider) -> None:
        provider.create_user("alice", "pw", display_name="Alice")
        token = await provider.refresh("alice", "alice")
        assert isinstance(token, str)
        assert len(token.split(".")) == 3

    @pytest.mark.asyncio
    async def test_refresh_with_iat_within_limit_succeeds(self, provider) -> None:
        """Refresh with original_iat within max_refresh_chain_hours succeeds."""
        provider.create_user("alice", "pw", display_name="Alice")
        recent_iat = int(time.time()) - 3600  # 1 hour ago
        token = await provider.refresh("alice", "alice", original_iat=recent_iat)
        assert isinstance(token, str)
        assert len(token.split(".")) == 3

    @pytest.mark.asyncio
    async def test_refresh_with_expired_chain_raises(self, provider) -> None:
        """Refresh with original_iat older than max_refresh_chain_hours raises."""
        provider.create_user("alice", "pw", display_name="Alice")
        # Default max_refresh_chain_hours=168 (7 days). Set iat to 8 days ago.
        old_iat = int(time.time()) - (8 * 24 * 3600)
        with pytest.raises(AuthenticationError, match="Token refresh chain expired"):
            await provider.refresh("alice", "alice", original_iat=old_iat)

    @pytest.mark.asyncio
    async def test_refresh_carries_original_iat_forward(self, provider) -> None:
        """Refreshed token preserves the original iat, not a fresh one."""
        import jwt

        provider.create_user("alice", "pw", display_name="Alice")
        original_iat = int(time.time()) - 7200  # 2 hours ago
        token = await provider.refresh("alice", "alice", original_iat=original_iat)
        claims = jwt.decode(token, "test-secret-key-for-unit-tests", algorithms=["HS256"])
        assert claims["iat"] == original_iat

    @pytest.mark.asyncio
    async def test_refresh_without_iat_gets_fresh_timestamp(self, provider) -> None:
        """Refresh with no original_iat (legacy token) gets a current iat."""
        import jwt

        provider.create_user("alice", "pw", display_name="Alice")
        before = int(time.time())
        token = await provider.refresh("alice", "alice", original_iat=None)
        claims = jwt.decode(token, "test-secret-key-for-unit-tests", algorithms=["HS256"])
        assert claims["iat"] >= before
