"""FastAPI router for the audit-readiness endpoints.

GET /api/sessions/{sid}/audit-readiness         → AuditReadinessSnapshot
GET /api/sessions/{sid}/audit-readiness/explain → AuditReadinessExplain

Both routes are GET-only, return ``Cache-Control: no-store`` (the panel
must reflect the current composition_version on every render), require
authentication, and translate the service's ``LookupError`` /
``record is None`` outcomes into HTTP 404.

Layer: L3 (application). Imports only L3 peers (sessions, execution
schemas, composer state, auth).
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from elspeth.web.audit_readiness.explain import build_narrative
from elspeth.web.audit_readiness.models import (
    AuditReadinessExplain,
    AuditReadinessSnapshot,
)
from elspeth.web.audit_readiness.service import (
    CompositionStateNotFoundError,
    ReadinessService,
    build_boot_plugin_policy_readiness,
)
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.telemetry_phase8 import record_audit_fetch_failure
from elspeth.web.config import WebSettings
from elspeth.web.middleware.rate_limit import ComposerRateLimiter, get_rate_limiter
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.ownership import verify_session_ownership
from elspeth.web.sessions.protocol import SessionServiceProtocol
from elspeth.web.sessions.telemetry import _SessionsTelemetry

_NO_STORE = "no-store"


def create_audit_readiness_router() -> APIRouter:
    """Create the audit-readiness router (snapshot + explain endpoints)."""
    router = APIRouter(tags=["audit-readiness"])

    @router.get(
        "/api/sessions/{session_id}/audit-readiness",
        response_model=AuditReadinessSnapshot,
    )
    async def snapshot(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        rate_limiter: ComposerRateLimiter = Depends(get_rate_limiter),  # noqa: B008
    ) -> JSONResponse:
        """Return the six-row audit-readiness snapshot for ``session_id``."""
        await rate_limiter.check(user.user_id)
        await verify_session_ownership(session_id, user, request)
        service: ReadinessService = request.app.state.readiness_service
        # Phase 8 Sub-task 7f (B3 cohort b2). A 404 from the "no
        # composition state" branch is a not-found signal, not a
        # fetch failure, so it MUST NOT emit
        # composer.audit.fetch_failure_total. Any other exception is
        # a read-path-health signal: emit telemetry, then re-raise
        # so the exception still propagates to FastAPI's default
        # error handling (we deliberately do not swallow into a 200).
        # Telemetry-only signal under CLAUDE.md non-decision read
        # superset exception — no companion audit event is required.
        try:
            result = await service.compute_snapshot(
                session_id=session_id,
                user_id=user.user_id,
            )
        except CompositionStateNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except Exception:
            telemetry: _SessionsTelemetry = request.app.state.sessions_telemetry
            record_audit_fetch_failure(telemetry)
            raise
        return JSONResponse(
            content=result.model_dump(mode="json"),
            headers={"Cache-Control": _NO_STORE},
        )

    @router.get(
        "/api/sessions/{session_id}/audit-readiness/explain",
        response_model=AuditReadinessExplain,
    )
    async def explain(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        rate_limiter: ComposerRateLimiter = Depends(get_rate_limiter),  # noqa: B008
    ) -> JSONResponse:
        """Return the narrative prose form for the Explain detail view.

        Accepted double-read of the composition state record (the
        snapshot route already calls ``get_current_state`` through
        ``ReadinessService``; here we re-read it to build the
        narrative). Tune if profiling shows the cost; the simpler
        contract is worth the second read until then.
        """
        await rate_limiter.check(user.user_id)
        await verify_session_ownership(session_id, user, request)
        session_service: SessionServiceProtocol = request.app.state.session_service
        settings: WebSettings = request.app.state.settings
        # Phase 8 Sub-task 7f. Emit on any read-path failure other
        # than the explicit "no composition state" 404 (which is
        # not-found, not fetch-failed). The 404 is raised after the
        # except block so it remains the documented not-found path.
        try:
            record = await session_service.get_current_state(session_id)
        except Exception:
            telemetry: _SessionsTelemetry = request.app.state.sessions_telemetry
            record_audit_fetch_failure(telemetry)
            raise
        if record is None:
            raise HTTPException(
                status_code=404,
                detail="No composition state for this session",
            )
        try:
            state = state_from_record(record)
            result = AuditReadinessExplain(
                session_id=str(session_id),
                composition_version=state.version,
                narrative=build_narrative(
                    state,
                    retention_days=settings.payload_store_retention_days,
                    plugin_policy_readiness=build_boot_plugin_policy_readiness(
                        policy=request.app.state.web_plugin_policy,
                        settings=request.app.state.runtime_web_plugin_config,
                        catalog=request.app.state.catalog_service,
                    ),
                ),
            )
        except Exception:
            # Annotated in the first except above; mypy carries the type
            # across branches even though that branch ended in ``raise``.
            # Re-annotating here would trip mypy's no-redef rule. The
            # runtime binding always happens on this assignment — the
            # first except never reaches this point because it re-raises.
            telemetry = request.app.state.sessions_telemetry
            record_audit_fetch_failure(telemetry)
            raise
        return JSONResponse(
            content=result.model_dump(mode="json"),
            headers={"Cache-Control": _NO_STORE},
        )

    return router
