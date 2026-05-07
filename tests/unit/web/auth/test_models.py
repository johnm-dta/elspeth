"""Tests for authentication data models."""

from __future__ import annotations

import pytest

from elspeth.web.auth.models import AuthenticationError, UserIdentity, UserProfile


class TestUserIdentity:
    """Tests for the minimal authentication identity."""

    def test_frozen_immutability(self) -> None:
        identity = UserIdentity(user_id="alice", username="alice")
        with pytest.raises(AttributeError):
            identity.user_id = "bob"  # type: ignore[misc]

    def test_accepts_valid_identity(self) -> None:
        identity = UserIdentity(user_id="sub-123", username="alice")
        assert identity.user_id == "sub-123"
        assert identity.username == "alice"

    def test_rejects_empty_user_id(self) -> None:
        with pytest.raises(AuthenticationError, match="user_id"):
            UserIdentity(user_id="", username="alice")

    def test_rejects_whitespace_only_user_id(self) -> None:
        with pytest.raises(AuthenticationError, match="user_id"):
            UserIdentity(user_id="   ", username="alice")

    def test_rejects_zero_width_space_user_id(self) -> None:
        with pytest.raises(AuthenticationError, match="user_id"):
            UserIdentity(user_id="\u200b", username="alice")

    def test_rejects_bom_only_user_id(self) -> None:
        with pytest.raises(AuthenticationError, match="user_id"):
            UserIdentity(user_id="\ufeff", username="alice")

    def test_rejects_empty_username(self) -> None:
        with pytest.raises(AuthenticationError, match="username"):
            UserIdentity(user_id="sub-123", username="")

    def test_rejects_whitespace_only_username(self) -> None:
        with pytest.raises(AuthenticationError, match="username"):
            UserIdentity(user_id="sub-123", username="   ")


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

    def test_rejects_empty_user_id(self) -> None:
        with pytest.raises(AuthenticationError, match="user_id"):
            UserProfile(user_id="", username="alice")

    def test_rejects_whitespace_only_user_id(self) -> None:
        with pytest.raises(AuthenticationError, match="user_id"):
            UserProfile(user_id="   ", username="alice")

    def test_rejects_zero_width_space_user_id(self) -> None:
        with pytest.raises(AuthenticationError, match="user_id"):
            UserProfile(user_id="\u200b", username="alice")

    def test_rejects_bom_only_user_id(self) -> None:
        with pytest.raises(AuthenticationError, match="user_id"):
            UserProfile(user_id="\ufeff", username="alice")

    def test_rejects_empty_username(self) -> None:
        with pytest.raises(AuthenticationError, match="username"):
            UserProfile(user_id="sub-123", username="")

    def test_rejects_whitespace_only_username(self) -> None:
        with pytest.raises(AuthenticationError, match="username"):
            UserProfile(user_id="sub-123", username="   ")

    def test_rejects_zero_width_space_username(self) -> None:
        with pytest.raises(AuthenticationError, match="username"):
            UserProfile(user_id="sub-123", username="\u200b")

    def test_rejects_bom_only_username(self) -> None:
        with pytest.raises(AuthenticationError, match="username"):
            UserProfile(user_id="sub-123", username="\ufeff")

    def test_coerces_invisible_display_name_to_none(self) -> None:
        profile = UserProfile(user_id="sub-123", username="alice", display_name="\u200b")
        assert profile.display_name is None

    def test_coerces_bom_only_display_name_to_none(self) -> None:
        profile = UserProfile(user_id="sub-123", username="alice", display_name="\ufeff")
        assert profile.display_name is None

    def test_preserves_visible_display_name(self) -> None:
        profile = UserProfile(user_id="sub-123", username="alice", display_name="Alice")
        assert profile.display_name == "Alice"

    def test_coerces_invisible_email_to_none(self) -> None:
        profile = UserProfile(user_id="sub-123", username="alice", email="\u200b")
        assert profile.email is None

    def test_preserves_visible_email(self) -> None:
        profile = UserProfile(user_id="sub-123", username="alice", email="alice@example.com")
        assert profile.email == "alice@example.com"

    def test_accepts_none_optional_fields(self) -> None:
        profile = UserProfile(user_id="sub-123", username="alice")
        assert profile.display_name is None
        assert profile.email is None
        assert profile.groups == ()


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
