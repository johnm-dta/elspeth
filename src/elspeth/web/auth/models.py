"""Authentication data models.

UserIdentity and UserProfile are frozen dataclasses. All fields are scalars,
None, or tuple of scalars -- no freeze guard needed.

AuthenticationError is the domain exception raised by all auth providers
when token validation fails.
"""

from __future__ import annotations

from dataclasses import dataclass

from elspeth.web.validation import has_visible_content


@dataclass(frozen=True, slots=True)
class UserIdentity:
    """Minimal authenticated identity -- returned from every auth check."""

    user_id: str
    username: str

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str) or not has_visible_content(self.user_id):
            raise AuthenticationError("user_id must be a non-blank string with visible content")
        if not isinstance(self.username, str) or not has_visible_content(self.username):
            raise AuthenticationError("username must be a non-blank string with visible content")


@dataclass(frozen=True, slots=True)
class UserProfile:
    """Extended user profile information."""

    user_id: str
    username: str
    display_name: str | None = None
    email: str | None = None
    groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str) or not has_visible_content(self.user_id):
            raise AuthenticationError("user_id must be a non-blank string with visible content")
        if not isinstance(self.username, str) or not has_visible_content(self.username):
            raise AuthenticationError("username must be a non-blank string with visible content")
        # Coerce invisible-only display_name to None rather than raising —
        # display_name is cosmetic IdP metadata, not a security-critical
        # identity field.  Denying auth for a bad display name would be
        # disproportionate.
        if self.display_name is not None and not has_visible_content(self.display_name):
            object.__setattr__(self, "display_name", None)
        if self.email is not None and not has_visible_content(self.email):
            object.__setattr__(self, "email", None)


class AuthenticationError(Exception):
    """Raised when authentication fails.

    Caught by the auth middleware and converted to HTTP 401.
    """

    def __init__(self, detail: str = "Authentication failed") -> None:
        self.detail = detail
        super().__init__(detail)
