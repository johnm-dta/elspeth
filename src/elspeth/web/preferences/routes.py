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
from elspeth.web.composer.telemetry_phase8 import (
    SessionsTelemetry,
    record_mode_opted_in,
    record_mode_opted_out,
)
from elspeth.web.middleware.rate_limit import ComposerRateLimiter, get_rate_limiter
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
        rate_limiter: ComposerRateLimiter = Depends(get_rate_limiter),  # noqa: B008
    ) -> ComposerPreferences:
        # Panel C1: per-user rate limit. Sibling write routes
        # (sessions/routes.py:3716, 4435) all check the limiter; this
        # PATCH was previously the only authenticated write endpoint
        # without one. Read GET is intentionally unguarded — idempotent,
        # safe to spam, no write amplification.
        await rate_limiter.check(user.user_id)
        service: PreferencesService = request.app.state.preferences_service
        # B2 (load-bearing): the service returns ``(prior, current)`` for
        # code-shape symmetry with the per-session reshape. Per B2.b the
        # account-level Phase 8 telemetry emit consumes only
        # ``current.default_mode`` (post-state-only counters), so we
        # discard ``prior`` here. The seam stays open for future
        # promotion of account-level preferences into a Landscape emit;
        # see the "Operational signal only" module-level comment in
        # ``preferences/service.py`` for the future-promotion criterion.
        transition = await service.update_composer_preferences(user.user_id, body)

        # Phase 8 Task 2 — account-level mode opt-out / opt-in emit.
        #
        # The emit fires ONLY when the PATCH body actually included
        # ``default_mode`` (``body.default_mode is not None``), matching
        # the ``mode_changed`` field-presence semantic the service uses
        # for its own composer-preferences PATCH counter. This is a
        # **set-rate**, not a transition-rate (B3-r3 semantic caveat):
        # a PATCH that sets ``default_mode=freeform`` on a session
        # whose prior was already ``freeform`` still fires the
        # ``record_mode_opted_out`` emit because the user re-asserted
        # the value. Inferring "changed" from
        # ``transition.prior.default_mode != transition.current.default_mode``
        # would silently convert the set-rate to a transition-rate and
        # break §"Account-level scope narrowing (B2.b — load-bearing)".
        #
        # Post-state-only per B2.b: helpers are kwarg-free and read the
        # post-state from ``transition.current.default_mode`` rather
        # than carrying ``from_mode`` attributes. The route does NOT
        # accompany this emit with an audit event — the account-level
        # surface is an operational signal; the future-promotion seam
        # is documented in ``preferences/service.py``.
        if body.default_mode is not None:
            telemetry: SessionsTelemetry = request.app.state.sessions_telemetry
            if transition.current.default_mode == "freeform":
                record_mode_opted_out(telemetry)
            elif transition.current.default_mode == "guided":
                record_mode_opted_in(telemetry)

        return transition.current

    return router
