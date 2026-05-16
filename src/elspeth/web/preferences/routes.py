"""FastAPI router for composer-preferences endpoints.

Two endpoints under ``/api/composer-preferences``:
  - ``GET`` — returns the authenticated user's preferences (guided
    default for users with no row).
  - ``PATCH`` — partial update; missing fields are preserved. Empty
    payload is a no-op success.

Both endpoints require auth (the standard ``get_current_user``
dependency). Cross-user isolation is via ``user.user_id`` scoping at the
service layer.
"""

from fastapi import APIRouter, Depends, Request

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.preferences.models import (
    ComposerPreferences,
    UpdateComposerPreferencesRequest,
)
from elspeth.web.preferences.service import PreferencesService


def create_preferences_router() -> APIRouter:
    router = APIRouter(prefix="/api/composer-preferences", tags=["preferences"])

    @router.get("", response_model=ComposerPreferences)
    async def get_preferences(
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> ComposerPreferences:
        service: PreferencesService = request.app.state.preferences_service
        return await service.get_composer_preferences(user.user_id)

    @router.patch("", response_model=ComposerPreferences)
    async def update_preferences(
        body: UpdateComposerPreferencesRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> ComposerPreferences:
        service: PreferencesService = request.app.state.preferences_service
        return await service.update_composer_preferences(user.user_id, body)

    return router
