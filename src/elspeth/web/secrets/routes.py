"""REST API routes for secret reference management.

All endpoints require authentication.  SECURITY INVARIANT: no route in
this module may ever return a plaintext secret value in any response body.

Endpoints
---------
GET    /api/secrets              -- list visible secret refs (metadata only)
POST   /api/secrets              -- create/update a user-scoped secret (write-only)
DELETE /api/secrets/{name}       -- delete a user-scoped secret
POST   /api/secrets/{name}/validate -- check whether a secret ref is resolvable
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.config import WebSettings
from elspeth.web.secrets.schemas import (
    CreateSecretRequest,
    CreateSecretResponse,
    SecretInventoryResponse,
    ValidateSecretResponse,
)
from elspeth.web.secrets.service import WebSecretService
from elspeth.web.validation import validate_secret_name


def create_secrets_router() -> APIRouter:
    """Create the secret management router."""
    router = APIRouter(
        prefix="/api/secrets",
        tags=["secrets"],
    )

    def _get_service(request: Request) -> WebSecretService:
        service: WebSecretService = request.app.state.secret_service
        return service

    def _get_settings(request: Request) -> WebSettings:
        settings: WebSettings = request.app.state.settings
        return settings

    def _validated_path_secret_name(name: str) -> str:
        try:
            return validate_secret_name(name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid secret name") from exc

    @router.get("", response_model=list[SecretInventoryResponse])
    async def list_secrets(
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> list[SecretInventoryResponse]:
        """List all visible secret references (user + server scopes).

        SECURITY: returns metadata only -- no values.
        """
        service = _get_service(request)
        settings = _get_settings(request)
        items = await run_sync_in_worker(service.list_refs, user.user_id, auth_provider_type=settings.auth_provider)
        return [
            SecretInventoryResponse(
                name=item.name,
                scope=item.scope,
                available=item.available,
                source_kind=item.source_kind,
                reason=item.reason,
            )
            for item in items
        ]

    @router.post("", status_code=201, response_model=CreateSecretResponse)
    async def create_secret(
        body: CreateSecretRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> CreateSecretResponse:
        """Create or update a user-scoped secret.

        SECURITY: response is write-only acknowledgement -- NEVER includes
        the value.

        TOCTOU: the service returns a :class:`CreateSecretResult` whose
        eager fingerprint was computed at write-time
        (see ``UserSecretStore.set_secret``). A successful return means
        the row was BOTH persisted AND immediately resolvable — no
        second ``has_ref`` probe is needed or performed, closing the
        race window where a concurrent DELETE or an env-var clear
        between set and probe could invalidate a row that was in fact
        correctly written. If the deployment is misconfigured (e.g., missing
        ``ELSPETH_FINGERPRINT_KEY``), ``set_user_secret`` raises
        ``FingerprintKeyMissingError`` which the app-level handler
        translates to 503.
        """
        service = _get_service(request)
        settings = _get_settings(request)
        result = await run_sync_in_worker(
            service.set_user_secret,
            user.user_id,
            body.name,
            body.value,
            auth_provider_type=settings.auth_provider,
        )
        return CreateSecretResponse(
            name=result.name,
            scope=result.scope,
        )

    @router.delete("/{name}", status_code=204)
    async def delete_secret(
        name: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> None:
        """Delete a user-scoped secret.

        Returns 204 on success, 404 if the secret does not exist for this user.
        """
        name = _validated_path_secret_name(name)
        service = _get_service(request)
        settings = _get_settings(request)
        deleted = await run_sync_in_worker(
            service.delete_user_secret,
            user.user_id,
            name,
            auth_provider_type=settings.auth_provider,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Secret {name!r} not found")

    @router.post("/{name}/validate", response_model=ValidateSecretResponse)
    async def validate_secret(
        name: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> ValidateSecretResponse:
        """Check whether a named secret reference is resolvable.

        SECURITY: does NOT return the value -- only resolvability status.

        Typed-error surface: uses
        :meth:`WebSecretService.check_user_ref_resolvable` rather than
        ``has_ref`` so deployment / server-state issues produce
        actionable HTTP responses:

        * ``FingerprintKeyMissingError`` → 503 (``fingerprint_key_missing``)
        * ``SecretDecryptionError``       → 409 (``secret_decryption_failed``)
        * resolvable                      → 200 ``available=True``
        * absent                          → 200 ``available=False``

        Both typed exceptions propagate past this handler to the
        app-level exception handlers in ``web/app.py``; only the
        success / absence cases are represented as a ``ValidateSecretResponse``.
        """
        name = _validated_path_secret_name(name)
        service = _get_service(request)
        settings = _get_settings(request)
        available = await run_sync_in_worker(
            service.check_user_ref_resolvable,
            user.user_id,
            name,
            auth_provider_type=settings.auth_provider,
        )
        return ValidateSecretResponse(name=name, available=available)

    return router
