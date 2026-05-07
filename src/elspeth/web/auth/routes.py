"""Auth API routes -- /api/auth/login, /api/auth/register, /api/auth/token, /api/auth/config, /api/auth/me.

POST /login is only available when auth_provider is "local".
POST /register is only available when auth_provider is "local" and registration is open.
POST /token re-issues a JWT from a valid existing token (local only).
GET /config returns auth configuration for frontend discovery (unauthenticated).
GET /me returns the full UserProfile for any auth provider.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, field_validator

from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth.local import LocalAuthProvider
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import AuthenticationError, UserIdentity
from elspeth.web.auth.protocol import AuthProvider, CredentialAuthProvider
from elspeth.web.config import WebSettings
from elspeth.web.middleware.rate_limit import check_auth_rate_limit
from elspeth.web.validation import has_visible_content


class _StrictResponse(BaseModel):
    """Tier 1 base for auth responses — no coercion, no extras.

    Auth responses carry identity material (user_id, groups, tokens) that
    the audit trail and authorization boundary both rely on.  Strict
    construction means a buggy provider adapter cannot smuggle unexpected
    claims into ``/me`` or ``/config``, and cannot silently add a
    ``refresh_token`` field to the bearer-only ``TokenResponse``.
    """

    model_config = ConfigDict(strict=True, extra="forbid")


class LoginRequest(BaseModel):
    """Request body for POST /api/auth/login."""

    username: str
    password: str


class RegisterRequest(BaseModel):
    """Request body for POST /api/auth/register."""

    username: str
    password: str
    display_name: str
    email: str | None = None

    @field_validator("username", "password", "display_name")
    @classmethod
    def _must_not_be_blank(cls, v: str) -> str:
        if not has_visible_content(v):
            raise ValueError("must contain at least one visible character")
        return v


class TokenResponse(_StrictResponse):
    """Response for login and token refresh."""

    access_token: str
    token_type: str = "bearer"


class UserProfileResponse(_StrictResponse):
    """Response for GET /api/auth/me."""

    user_id: str
    username: str
    display_name: str | None = None
    email: str | None = None
    groups: list[str] = []


class AuthConfigResponse(_StrictResponse):
    """Response for GET /api/auth/config."""

    provider: str
    registration_mode: str
    oidc_issuer: str | None = None
    oidc_client_id: str | None = None
    authorization_endpoint: str | None = None


def create_auth_router() -> APIRouter:
    """Create the auth router with /api/auth prefix."""
    router = APIRouter(prefix="/api/auth", tags=["auth"])

    @router.post("/login", response_model=TokenResponse)
    async def login(
        body: LoginRequest,
        request: Request,
        _rate_limit: None = Depends(check_auth_rate_limit),
    ) -> TokenResponse:
        """Authenticate with username/password (local auth only).

        login() is synchronous (bcrypt is intentionally slow ~200ms),
        so it is offloaded to a thread to avoid blocking the event loop.

        No CSRF token required — this is a bearer-token-only API (no
        cookies). If cookie-based sessions are added later, CSRF
        protection must be revisited.
        """
        settings: WebSettings = request.app.state.settings
        if settings.auth_provider != "local":
            raise HTTPException(status_code=404, detail="Not found")

        provider: CredentialAuthProvider = request.app.state.auth_provider
        try:
            token = await provider.login(body.username, body.password)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=exc.detail) from exc

        return TokenResponse(access_token=token)

    @router.post("/register", response_model=TokenResponse)
    async def register(
        body: RegisterRequest,
        request: Request,
        _rate_limit: None = Depends(check_auth_rate_limit),
    ) -> TokenResponse:
        """Register a new user account (local auth, open registration only).

        Creates the user with bcrypt-hashed password (offloaded to a thread),
        then auto-logs in and returns a JWT so the caller can proceed
        immediately without a separate login round-trip.
        """
        settings: WebSettings = request.app.state.settings
        if settings.auth_provider != "local":
            raise HTTPException(status_code=404, detail="Not found")

        if settings.registration_mode == "closed":
            raise HTTPException(status_code=404, detail="Not found")

        if settings.registration_mode == "email_verified":
            raise HTTPException(
                status_code=501,
                detail="Email verification not yet available",
            )

        provider: LocalAuthProvider = request.app.state.auth_provider
        try:
            await run_sync_in_worker(
                provider.create_user,
                body.username,
                body.password,
                body.display_name,
                body.email,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        token = await provider.login(body.username, body.password)
        return TokenResponse(access_token=token)

    @router.post("/token", response_model=TokenResponse)
    async def refresh_token(
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> TokenResponse:
        """Re-issue a JWT from a valid existing token (local auth only).

        Passes the original ``iat`` claim through so the provider can
        enforce a maximum refresh chain lifetime.
        """
        settings: WebSettings = request.app.state.settings
        if settings.auth_provider != "local":
            raise HTTPException(status_code=404, detail="Not found")

        # Extract iat from claims parsed by the auth middleware.
        # The middleware decodes claims without signature verification for
        # downstream use, then verifies the signature via authenticate().
        # If authenticate() fails, this route handler never executes.
        #
        # If claims are None (decode failed despite valid signature), refuse
        # the refresh — we cannot enforce chain lifetime without iat.
        claims = request.state.auth_claims
        if claims is None:
            raise HTTPException(status_code=401, detail="Token claims could not be parsed — re-authenticate")
        original_iat: int | None = claims.get("iat")

        provider: CredentialAuthProvider = request.app.state.auth_provider
        try:
            new_token = await provider.refresh(user.user_id, user.username, original_iat=original_iat)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=exc.detail) from exc
        return TokenResponse(access_token=new_token)

    @router.get("/config", response_model=AuthConfigResponse)
    async def auth_config(request: Request) -> AuthConfigResponse:
        """Return auth configuration for frontend discovery.

        This endpoint is unauthenticated -- the frontend needs it
        before any login flow.
        """
        settings: WebSettings = request.app.state.settings
        return AuthConfigResponse(
            provider=settings.auth_provider,
            registration_mode=settings.registration_mode,
            oidc_issuer=settings.oidc_issuer,
            oidc_client_id=settings.oidc_client_id,
            authorization_endpoint=request.app.state.oidc_authorization_endpoint,
        )

    @router.get("/me", response_model=UserProfileResponse)
    async def me(
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> UserProfileResponse:
        """Return the full profile of the authenticated user.

        The raw token was stashed on request.state.auth_token by
        get_current_user, so we don't re-parse the Authorization header.
        """
        # Note: authenticate() already decoded the JWT in get_current_user.
        # get_user_info() decodes it again to extract profile claims.
        # This is intentional — the middleware returns UserIdentity (minimal),
        # while /me needs the full UserProfile with groups/email/display_name.
        token: str = request.state.auth_token
        auth_provider: AuthProvider = request.app.state.auth_provider

        try:
            profile = await auth_provider.get_user_info(token)
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
