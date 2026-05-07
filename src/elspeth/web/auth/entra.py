"""Azure Entra ID authentication provider.

Uses JWKSTokenValidator via composition, adding Entra-specific tenant
validation and group/role claim extraction. The OIDC issuer is derived
from the tenant_id.
"""

from __future__ import annotations

from typing import Any

from elspeth.web.auth.models import AuthenticationError, UserIdentity, UserProfile
from elspeth.web.auth.oidc import JWKSTokenValidator


class EntraAuthProvider:
    """Validates Azure Entra ID tokens with tenant and group claim handling.

    Composes JWKSTokenValidator for JWKS discovery and JWT decode, adding:
    - Tenant ID verification (``tid`` claim must match expected tenant)
    - Group claim extraction (``groups`` + ``role:``-prefixed ``roles``)
    """

    def __init__(
        self,
        tenant_id: str,
        audience: str,
        jwks_cache_ttl_seconds: int = 3600,
        jwks_failure_retry_seconds: int = 300,
    ) -> None:
        self._tenant_id = tenant_id
        issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
        self._validator = JWKSTokenValidator(
            issuer=issuer,
            audience=audience,
            jwks_cache_ttl_seconds=jwks_cache_ttl_seconds,
            jwks_failure_retry_seconds=jwks_failure_retry_seconds,
        )

    def _validate_tenant(self, payload: dict[str, Any]) -> None:
        """Verify the tid claim matches the expected tenant.

        Raises AuthenticationError if ``tid`` is missing or mismatched.
        The ``tid`` claim is required in Entra ID tokens -- absence
        indicates a non-Entra token or a configuration error.
        """
        try:
            tid = payload["tid"]
        except KeyError as exc:
            raise AuthenticationError("Missing tenant claim (tid) — token may not be from Entra ID") from exc
        if tid != self._tenant_id:
            raise AuthenticationError(f"Invalid tenant: received tid={tid!r}")

    def _extract_groups(self, payload: dict[str, Any]) -> tuple[str, ...]:
        """Extract group IDs and role-prefixed entries from Entra claims.

        ``groups`` and ``roles`` are optional Entra claims (Tier 3 data
        from the IdP). Absence means "no groups/roles assigned";
        non-list values indicate IdP misconfiguration.
        """
        groups: list[str] = []

        raw_groups = payload.get("groups")
        if raw_groups is None:
            pass  # Absent claim -- no groups assigned
        elif isinstance(raw_groups, list):
            # Coerce group IDs to str — IdPs may send integers (e.g. Entra
            # group object IDs). This is intentional Tier 3 coercion.
            groups.extend(str(g) for g in raw_groups)
        else:
            raise AuthenticationError(
                f"Unexpected type for 'groups' claim: {type(raw_groups).__name__} (expected list) — check IdP token configuration"
            )

        raw_roles = payload.get("roles")
        if raw_roles is None:
            pass  # Absent claim -- no roles assigned
        elif isinstance(raw_roles, list):
            # Coerce group IDs to str — IdPs may send integers (e.g. Entra
            # group object IDs). This is intentional Tier 3 coercion.
            groups.extend(f"role:{r}" for r in raw_roles)
        else:
            raise AuthenticationError(
                f"Unexpected type for 'roles' claim: {type(raw_roles).__name__} (expected list) — check IdP token configuration"
            )

        return tuple(groups)

    async def authenticate(self, token: str) -> UserIdentity:
        """Validate an Entra ID token with tenant verification.

        Performs standard OIDC validation (signature, expiry, issuer,
        audience) via JWKSTokenValidator, then checks the tenant claim.
        """
        jwks = await self._validator.ensure_jwks()
        payload = self._validator.decode_token(token, jwks)

        self._validate_tenant(payload)

        try:
            sub = payload["sub"]
        except KeyError as exc:
            raise AuthenticationError("Missing required 'sub' claim in token") from exc

        return UserIdentity(
            user_id=sub,
            # preferred_username is optional — fall back to sub if absent,
            # null, or empty.
            username=payload.get("preferred_username") or sub,
        )

    async def get_user_info(self, token: str) -> UserProfile:
        """Decode an Entra ID token and extract profile with group claims."""
        jwks = await self._validator.ensure_jwks()
        payload = self._validator.decode_token(token, jwks)

        self._validate_tenant(payload)

        try:
            sub = payload["sub"]
        except KeyError as exc:
            raise AuthenticationError("Missing required 'sub' claim in token") from exc

        return UserProfile(
            user_id=sub,
            username=payload.get("preferred_username") or sub,
            display_name=payload.get("name") or payload.get("preferred_username"),
            email=payload.get("email"),
            groups=self._extract_groups(payload),
        )
