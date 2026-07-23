"""Tests for LocalAuthProvider -- SQLite user store, bcrypt hashing, JWT tokens."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from typing import Any

import jwt as pyjwt
import pytest

from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth import local as auth_local
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


def _signed_local_token(provider: LocalAuthProvider, claims: dict[str, Any]) -> str:
    """Create a signed local JWT for boundary-shape tests."""
    return pyjwt.encode(claims, provider._secret_key, algorithm="HS256")


def _delete_user(provider: LocalAuthProvider, user_id: str) -> None:
    """Delete a test user without leaking sqlite3's transaction-only context manager."""
    with closing(sqlite3.connect(str(provider._db_path))) as conn, conn:
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))


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

    @pytest.mark.asyncio
    async def test_unverified_user_cannot_login_until_email_token_is_verified(self, provider) -> None:
        provider.create_user(
            "alice",
            "password123",
            display_name="Alice",
            email="alice@example.com",
            email_verified=False,
        )
        token = provider.create_email_verification_token("alice")

        with pytest.raises(AuthenticationError, match="Email verification required"):
            await provider.login("alice", "password123")

        verified_token = provider.verify_email_and_issue_token(
            token,
            record_token_issued=lambda _identity, _access_token: None,
        )
        assert len(verified_token.split(".")) == 3
        login_token = await provider.login("alice", "password123")
        assert len(login_token.split(".")) == 3

        with pytest.raises(AuthenticationError, match="already used"):
            provider.verify_email_and_issue_token(
                token,
                record_token_issued=lambda _identity, _access_token: None,
            )

    @pytest.mark.asyncio
    async def test_delete_user_removes_account_and_invalidates_tokens(self, provider) -> None:
        provider.create_user("alice", "password123", display_name="Alice")
        token = await provider.login("alice", "password123")

        assert provider.delete_user("alice") is True
        assert provider.delete_user("alice") is False

        with pytest.raises(AuthenticationError, match="Invalid token"):
            await provider.authenticate(token)

    def test_open_registration_is_invisible_until_required_audit_commits(self, provider) -> None:
        audit_entered = threading.Event()
        release_audit = threading.Event()

        def fail_required_audit(_token: str) -> None:
            audit_entered.set()
            assert release_audit.wait(timeout=2)
            raise OSError("Landscape unavailable")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                provider.register_open_user_with_audit,
                "alice",
                "password123",
                "Alice",
                None,
                record_token_issued=fail_required_audit,
            )
            assert audit_entered.wait(timeout=2)
            with pytest.raises(AuthenticationError, match="Invalid credentials"):
                provider._login_sync("alice", "password123")
            release_audit.set()
            with pytest.raises(OSError, match="Landscape unavailable"):
                future.result(timeout=2)

        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            provider._login_sync("alice", "password123")

    @pytest.mark.asyncio
    async def test_cancelled_open_registration_finishes_audit_and_state_together(self, provider) -> None:
        audit_entered = threading.Event()
        release_audit = threading.Event()
        audit_finished = threading.Event()

        def record_required_audit(_token: str) -> None:
            audit_entered.set()
            assert release_audit.wait(timeout=2)
            audit_finished.set()

        task = asyncio.create_task(
            run_sync_in_worker(
                provider.register_open_user_with_audit,
                "alice",
                "password123",
                "Alice",
                None,
                record_token_issued=record_required_audit,
            )
        )
        assert await asyncio.to_thread(audit_entered.wait, 2)
        task.cancel()
        release_audit.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await asyncio.to_thread(audit_finished.wait, 2)

        token = await provider.login("alice", "password123")
        assert len(token.split(".")) == 3

    def test_email_verification_token_has_exactly_one_concurrent_consumer(self, provider) -> None:
        provider.create_user(
            "alice",
            "password123",
            display_name="Alice",
            email="alice@example.com",
            email_verified=False,
        )
        token = provider.create_email_verification_token("alice")

        def consume() -> str:
            return provider.verify_email_and_issue_token(
                token,
                record_token_issued=lambda _identity, _access_token: None,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(consume) for _ in range(2)]
        outcomes: list[str] = []
        failures: list[BaseException] = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except BaseException as exc:
                failures.append(exc)

        assert len(outcomes) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], AuthenticationError)

    def test_verification_audit_failure_restores_bounded_retry_lifetime(self, provider, monkeypatch: pytest.MonkeyPatch) -> None:
        now = [1_000]
        monkeypatch.setattr(auth_local.time, "time", lambda: now[0])
        provider.create_user(
            "alice",
            "password123",
            display_name="Alice",
            email="alice@example.com",
            email_verified=False,
        )
        token = provider.create_email_verification_token("alice", ttl_seconds=1)

        def fail_required_audit(_identity: UserIdentity, _access_token: str) -> None:
            now[0] = 1_001
            raise OSError("Landscape unavailable")

        with pytest.raises(OSError, match="Landscape unavailable"):
            provider.verify_email_and_issue_token(token, record_token_issued=fail_required_audit)

        now[0] = 1_002
        access_token = provider.verify_email_and_issue_token(
            token,
            record_token_issued=lambda _identity, _access_token: None,
        )
        assert len(access_token.split(".")) == 3

    def test_email_registration_outbox_recovers_after_publish_failure_and_restart(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = LocalAuthProvider(db_path=tmp_path / "auth.db", secret_key="test-key")
        outbox_path = tmp_path / "email-verifications.jsonl"
        real_append = auth_local._append_email_verification_record

        def fail_publish(*args, **kwargs) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(auth_local, "_append_email_verification_record", fail_publish)
        with pytest.raises(OSError, match="disk full"):
            provider.register_email_verified_user(
                "alice",
                "password123",
                "Alice",
                "alice@example.com",
                verification_origin="https://composer.example.test",
                outbox_path=outbox_path,
            )

        with pytest.raises(AuthenticationError, match="Email verification required"):
            provider._login_sync("alice", "password123")

        monkeypatch.setattr(auth_local, "_append_email_verification_record", real_append)
        restarted = LocalAuthProvider(db_path=tmp_path / "auth.db", secret_key="test-key")
        restarted.publish_pending_email_verifications(outbox_path)

        records = [json.loads(line) for line in outbox_path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 1
        assert records[0]["delivery_id"]
        assert records[0]["user_id"] == "alice"

    def test_email_outbox_partial_append_is_truncated_before_retry(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        outbox_path = tmp_path / "email-verifications.jsonl"
        record = {
            "delivery_id": "delivery-1",
            "user_id": "alice",
            "email": "alice@example.com",
            "token": "verification-token",
            "verification_url": "https://composer.example.test/?verify_token=verification-token",
        }
        real_write = auth_local.os.write

        def short_write(fd: int, payload: bytes) -> int:
            return real_write(fd, payload[: len(payload) // 2])

        monkeypatch.setattr(auth_local.os, "write", short_write)
        with pytest.raises(OSError, match="incomplete"):
            auth_local._append_email_verification_record(outbox_path, record)
        assert outbox_path.read_bytes() == b""

        monkeypatch.setattr(auth_local.os, "write", real_write)
        auth_local._append_email_verification_record(outbox_path, record)
        assert [json.loads(line) for line in outbox_path.read_text().splitlines()] == [record]

    def test_email_outbox_repairs_partial_crash_tail_before_republication(self, tmp_path) -> None:
        outbox_path = tmp_path / "email-verifications.jsonl"
        outbox_path.write_bytes(b'{"delivery_id":"crashed"')
        record = {
            "delivery_id": "delivery-1",
            "user_id": "alice",
            "email": "alice@example.com",
            "token": "verification-token",
            "verification_url": "https://composer.example.test/?verify_token=verification-token",
        }

        auth_local._append_email_verification_record(outbox_path, record)

        assert [json.loads(line) for line in outbox_path.read_text().splitlines()] == [record]

    def test_retry_after_expiry_rotates_pending_registration_delivery(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        now = [1_000]
        monkeypatch.setattr(auth_local.time, "time", lambda: now[0])
        provider = LocalAuthProvider(db_path=tmp_path / "auth.db", secret_key="test-key")
        outbox_path = tmp_path / "email-verifications.jsonl"
        kwargs = {
            "verification_origin": "https://composer.example.test",
            "outbox_path": outbox_path,
        }
        provider.register_email_verified_user(
            "alice",
            "password123",
            "Alice",
            "alice@example.com",
            **kwargs,
        )
        first = json.loads(outbox_path.read_text().splitlines()[0])

        now[0] += auth_local._EMAIL_VERIFICATION_TOKEN_TTL_SECONDS + 1
        provider.register_email_verified_user(
            "alice",
            "password123",
            "Alice",
            "alice@example.com",
            **kwargs,
        )
        records = [json.loads(line) for line in outbox_path.read_text().splitlines()]

        assert len(records) == 2
        assert records[1]["delivery_id"] != first["delivery_id"]
        assert records[1]["token"] != first["token"]

    def test_publish_retry_deduplicates_append_before_ack_crash(self, tmp_path) -> None:
        provider = LocalAuthProvider(db_path=tmp_path / "auth.db", secret_key="test-key")
        outbox_path = tmp_path / "email-verifications.jsonl"
        provider.register_email_verified_user(
            "alice",
            "password123",
            "Alice",
            "alice@example.com",
            verification_origin="https://composer.example.test",
            outbox_path=outbox_path,
        )
        with provider._connect() as conn:
            conn.execute("UPDATE email_verification_outbox SET published_at = NULL")

        provider.publish_pending_email_verifications(outbox_path)

        records = [json.loads(line) for line in outbox_path.read_text().splitlines()]
        assert len(records) == 1

    def test_startup_reclaims_pending_registration_after_retention_window(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = [1_000]
        monkeypatch.setattr(auth_local.time, "time", lambda: now[0])
        provider = LocalAuthProvider(db_path=tmp_path / "auth.db", secret_key="test-key")
        provider.register_email_verified_user(
            "alice",
            "password123",
            "Alice",
            "alice@example.com",
            verification_origin="https://composer.example.test",
            outbox_path=tmp_path / "email-verifications.jsonl",
        )

        now[0] += auth_local._EMAIL_VERIFICATION_TOKEN_TTL_SECONDS + auth_local._PENDING_REGISTRATION_RETENTION_SECONDS + 1
        restarted = LocalAuthProvider(db_path=tmp_path / "auth.db", secret_key="test-key")
        restarted.create_user("alice", "replacement-password", display_name="Replacement")

        token = restarted._login_sync("alice", "replacement-password")
        assert len(token.split(".")) == 3


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
        provider.create_user("alice", "pw", display_name="Alice")
        token = await provider.login("alice", "pw")

        # Delete the user behind the provider's back
        _delete_user(provider, "alice")

        with pytest.raises(AuthenticationError, match="Invalid token"):
            await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_authenticate_wrong_secret_key(self, tmp_path) -> None:
        """Token signed with a different key should fail."""
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

    @pytest.mark.asyncio
    async def test_authenticate_missing_username_claim_raises_authentication_error(self, provider) -> None:
        """Signed local tokens without username must not escape as KeyError."""
        provider.create_user("alice", "pw", display_name="Alice")
        token = _signed_local_token(
            provider,
            {
                "sub": "alice",
                "exp": int(time.time()) + 3600,
            },
        )

        with pytest.raises(AuthenticationError, match="Invalid token"):
            await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_authenticate_missing_sub_claim_raises_authentication_error(self, provider) -> None:
        """Signed local tokens without sub must not escape as KeyError."""
        token = _signed_local_token(
            provider,
            {
                "username": "alice",
                "exp": int(time.time()) + 3600,
            },
        )

        with pytest.raises(AuthenticationError, match="Invalid token"):
            await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_authenticate_non_string_username_claim_raises_authentication_error(self, provider) -> None:
        """Signed local tokens with non-string username must not reach UserIdentity."""
        provider.create_user("alice", "pw", display_name="Alice")
        token = _signed_local_token(
            provider,
            {
                "sub": "alice",
                "username": {"name": "alice"},
                "exp": int(time.time()) + 3600,
            },
        )

        with pytest.raises(AuthenticationError, match="Invalid token"):
            await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_authenticate_non_string_sub_claim_raises_authentication_error(self, provider) -> None:
        """Signed local tokens with non-string sub must not reach the user lookup."""
        token = _signed_local_token(
            provider,
            {
                "sub": {"id": "alice"},
                "username": "alice",
                "exp": int(time.time()) + 3600,
            },
        )

        with pytest.raises(AuthenticationError, match="Invalid token"):
            await provider.authenticate(token)


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
        _delete_user(provider, "alice")

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
        provider.create_user("alice", "pw", display_name="Alice")
        # Access _db_path directly — no public API to delete users by design
        _delete_user(provider, "alice")
        with pytest.raises(AuthenticationError, match="User not found"):
            await provider.refresh("alice", "alice", original_iat=int(time.time()))

    @pytest.mark.asyncio
    async def test_refresh_valid_user_returns_jwt(self, provider) -> None:
        provider.create_user("alice", "pw", display_name="Alice")
        token = await provider.refresh("alice", "alice", original_iat=int(time.time()))
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
    async def test_refresh_without_iat_raises(self, provider) -> None:
        """Refresh without original_iat must not start a fresh chain."""
        provider.create_user("alice", "pw", display_name="Alice")
        with pytest.raises(AuthenticationError, match="Token missing iat"):
            await provider.refresh("alice", "alice", original_iat=None)
