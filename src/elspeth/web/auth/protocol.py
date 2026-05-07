"""Authentication provider protocols.

AuthProvider: the two-method interface that all auth implementations satisfy.
CredentialAuthProvider: extends AuthProvider with login() and refresh() for
providers that support username/password authentication (e.g., LocalAuthProvider).

No exception definitions here -- AuthenticationError lives in models.py.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from elspeth.web.auth.models import UserIdentity, UserProfile


@runtime_checkable
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


@runtime_checkable
class CredentialAuthProvider(AuthProvider, Protocol):
    """AuthProvider that also supports username/password login and token refresh.

    Used by local (and future LDAP) auth providers. Routes check
    settings.auth_provider to determine if these methods are available,
    then narrow the type to CredentialAuthProvider for method access.
    """

    async def login(self, username: str, password: str) -> str:
        """Authenticate with credentials and return a JWT.

        Raises AuthenticationError on invalid credentials.
        """
        ...

    async def refresh(self, user_id: str, username: str, *, original_iat: int | None = None) -> str:
        """Issue a new JWT for an already-authenticated user.

        Args:
            original_iat: The ``iat`` claim from the token being refreshed.
                Carried forward to enforce a maximum refresh chain lifetime.

        Raises AuthenticationError if the user no longer exists or the
        refresh chain has expired.
        """
        ...
