from __future__ import annotations

from ._helpers import (
    SESSION_TERMINAL_RUN_STATUS_VALUES,
    UUID,
    APIRouter,
    AuditStoryIntegrityError,
    AuditStoryService,
    Depends,
    HTTPException,
    LandscapeDB,
    Request,
    RunAuditStoryResponse,
    RunResponse,
    SessionServiceProtocol,
    UserIdentity,
    _run_accounting_integrity_http,
    _validate_run_status_accounting_for_list,
    _verify_session_ownership,
    get_current_user,
    load_run_accounting_for_settings,
    run_sync_in_worker,
)


def register_run_routes(router: APIRouter) -> None:

    @router.get(
        "/{session_id}/runs",
        response_model=list[RunResponse],
    )
    async def list_session_runs(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> list[RunResponse]:
        """List all runs for a session, newest first."""
        session = await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        runs = await service.list_runs_for_session(session.id)
        from elspeth.web.execution.discard_summary import load_discard_summaries_for_settings

        terminal_landscape_run_ids = tuple(
            run.landscape_run_id for run in runs if run.status in SESSION_TERMINAL_RUN_STATUS_VALUES and run.landscape_run_id is not None
        )
        accounting_by_run_id = {}
        if terminal_landscape_run_ids:
            try:
                accounting_by_run_id = await run_sync_in_worker(
                    load_run_accounting_for_settings,
                    request.app.state.settings,
                    terminal_landscape_run_ids,
                )
            except ValueError as exc:
                raise _run_accounting_integrity_http(
                    "Session run accounting projection failed.",
                    landscape_run_ids=terminal_landscape_run_ids,
                    error=str(exc),
                ) from exc
        discard_summaries = {}
        if terminal_landscape_run_ids:
            discard_summaries = await run_sync_in_worker(
                load_discard_summaries_for_settings,
                request.app.state.settings,
                terminal_landscape_run_ids,
            )

        # Resolve composition_version from each run's state_id.
        # A missing state is Tier 1 data corruption — crash, don't hide.
        # Scope the read to the current session: the current-schema
        # composite FK prevents cross-session state refs at the schema
        # layer. ``get_state_in_session`` raises
        # ``AuditIntegrityError`` on session mismatch, surfacing Tier 1
        # corruption rather than silently returning the wrong state's
        # version number in another session's listing.
        responses: list[RunResponse] = []
        for run in runs:
            state = await service.get_state_in_session(run.state_id, session.id)
            version = state.version
            discard_summary = None
            if run.landscape_run_id is not None and run.landscape_run_id in discard_summaries:
                discard_summary = discard_summaries[run.landscape_run_id]
            accounting = None
            if run.landscape_run_id is not None and run.landscape_run_id in accounting_by_run_id:
                accounting = accounting_by_run_id[run.landscape_run_id]
            _validate_run_status_accounting_for_list(run, accounting)
            responses.append(
                RunResponse(
                    id=str(run.id),
                    session_id=str(run.session_id),
                    status=run.status,
                    accounting=accounting,
                    error=run.error,
                    started_at=run.started_at,
                    finished_at=run.finished_at,
                    composition_version=version,
                    discard_summary=discard_summary,
                )
            )
        return responses

    @router.get(
        "/{session_id}/runs/{run_id}/audit-story",
        response_model=RunAuditStoryResponse,
    )
    async def get_run_audit_story(
        session_id: UUID,
        run_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> RunAuditStoryResponse:
        """Return the Landscape-backed audit story for a run."""
        session = await _verify_session_ownership(session_id, user, request)
        service: SessionServiceProtocol = request.app.state.session_service
        try:
            run = await service.get_run(run_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        if run.session_id != session.id:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.landscape_run_id is None:
            # Sessions-DB run row exists but was never linked to a Landscape
            # run — same Tier-1 audit-DB invariant violation as the failure
            # modes inside ``AuditStoryService``. Promoting to the named type
            # routes both code paths through the single
            # ``AuditStoryIntegrityError`` handler in ``app.py`` so the
            # response carries the ``error_type`` discriminator that
            # incident-response code switches on.
            raise AuditStoryIntegrityError(f"Run {run_id} has no Landscape run id; audit story cannot be reconstructed")
        landscape_run_id = run.landscape_run_id

        settings = request.app.state.settings

        def _load_story() -> RunAuditStoryResponse:
            db = LandscapeDB(
                connection_string=settings.get_landscape_url(),
                passphrase=settings.landscape_passphrase,
            )
            return AuditStoryService(db).get_run_audit_story(
                landscape_run_id,
                public_run_id=str(run.id),
                session_id=str(session.id),
            )

        # Let ``AuditStoryIntegrityError`` propagate unflattened — the FastAPI
        # exception handler in ``app.py`` matches on the named type and
        # returns a structured 500. Catching it here and re-raising as bare
        # ``RuntimeError`` would route the failure to FastAPI's default
        # handler and erase the discriminator.
        return await run_sync_in_worker(_load_story)
