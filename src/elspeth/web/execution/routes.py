"""REST endpoints and WebSocket for pipeline execution.

POST /api/sessions/{session_id}/validate — dry-run validation
POST /api/sessions/{session_id}/execute — start background run
GET  /api/runs/{run_id}                 — run status
POST /api/runs/{run_id}/cancel          — cancel run
GET  /api/runs/{run_id}/results         — run results (terminal only)
WS   /ws/runs/{run_id}                  — live progress stream

All endpoints require authentication. Session-scoped endpoints verify
session ownership. Run-scoped endpoints verify run ownership via the
run's parent session.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import UUID

import pydantic
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import AuthenticationError, UserIdentity
from elspeth.web.auth.protocol import AuthProvider
from elspeth.web.blobs.protocol import BlobNotFoundError
from elspeth.web.composer.protocol import ComposerService, ComposerServiceError
from elspeth.web.config import WebSettings
from elspeth.web.execution.diagnostics import load_run_diagnostics_for_settings
from elspeth.web.execution.errors import BlobSourcePathMismatchError, SemanticContractViolationError
from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.protocol import ExecutionService, StateAccessError
from elspeth.web.execution.schemas import (
    RUN_STATUS_NON_TERMINAL_VALUES,
    RUN_STATUS_TERMINAL_VALUES,
    CancelledData,
    CompletedData,
    FailedData,
    RunDiagnosticsEvaluationResponse,
    RunDiagnosticsResponse,
    RunDiagnosticsWorkingView,
    RunEvent,
    RunResultsResponse,
    RunStatusResponse,
    ValidationResult,
)
from elspeth.web.sessions.protocol import SessionServiceProtocol, TerminalSessionRunStatus

slog = structlog.get_logger()


# ── Dependency providers (using app.state, matching existing pattern) ──


async def _get_execution_service(request: Request) -> ExecutionService:
    return cast(ExecutionService, request.app.state.execution_service)


async def _get_session_service(request: Request) -> SessionServiceProtocol:
    return cast(SessionServiceProtocol, request.app.state.session_service)


# ── Ownership verification helpers ────────────────────────────────────


async def _verify_session_ownership(session_id: UUID, user: UserIdentity, request: Request) -> None:
    """Verify the session exists and belongs to the current user.

    Returns 404 (not 403) to avoid leaking session existence (IDOR).
    Matches the pattern in sessions/routes.py.
    """
    session_service: SessionServiceProtocol = request.app.state.session_service
    settings: WebSettings = request.app.state.settings
    try:
        session = await session_service.get_session(session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None

    if session.user_id != user.user_id or session.auth_provider_type != settings.auth_provider:
        raise HTTPException(status_code=404, detail="Session not found")


async def _verify_run_ownership(run_id: UUID, user: UserIdentity, request: Request) -> None:
    """Verify the run exists and belongs to the current user's session.

    Looks up the run's parent session and checks ownership.
    Returns 404 (not 403) to avoid leaking run existence (IDOR).
    """
    session_service: SessionServiceProtocol = request.app.state.session_service
    settings: WebSettings = request.app.state.settings
    try:
        run = await session_service.get_run(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Run not found") from None

    try:
        session = await session_service.get_session(run.session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Run not found") from None

    if session.user_id != user.user_id or session.auth_provider_type != settings.auth_provider:
        raise HTTPException(status_code=404, detail="Run not found")


def _run_not_found_http() -> HTTPException:
    """Canonical IDOR-safe not-found response for run-scoped routes."""
    return HTTPException(status_code=404, detail="Run not found")


def _build_terminal_run_event(current: RunStatusResponse) -> RunEvent:
    """Synthesize a terminal RunEvent from authoritative run status.

    ``current`` comes from our session database and is therefore Tier 1.
    Impossible terminal states must raise rather than degrade into
    partial client-visible payloads.

    Phase 2.2 (elspeth-0de989c56d): the operator-completion subset
    (``completed``, ``completed_with_failures``, ``empty``) all map to the
    SSE ``event_type="completed"`` envelope; the operator-completion status
    travels in the ``CompletedData.status`` discriminator so the frontend
    can render the widened taxonomy without re-deriving from row counts.
    """
    if current.status in ("completed", "completed_with_failures", "empty"):
        if current.landscape_run_id is None:
            raise RuntimeError(f"Completed run {current.run_id} has no landscape_run_id — Tier 1 anomaly (audit trail incomplete)")
        # Cast is sound: the membership check above narrows to the
        # operator-completion subset that CompletedData.status accepts.
        completed_status = cast(
            Literal["completed", "completed_with_failures", "empty"],
            current.status,
        )
        try:
            payload: CompletedData | FailedData | CancelledData = CompletedData(
                status=completed_status,
                rows_processed=current.rows_processed,
                rows_succeeded=current.rows_succeeded,
                rows_failed=current.rows_failed,
                rows_routed_success=current.rows_routed_success,
                rows_routed_failure=current.rows_routed_failure,
                rows_quarantined=current.rows_quarantined,
                landscape_run_id=current.landscape_run_id,
            )
        except pydantic.ValidationError as exc:
            raise RuntimeError(
                f"Completed run {current.run_id} failed CompletedData validation — Tier 1 anomaly (audit trail inconsistent): {exc}"
            ) from exc
        event_type: Literal["progress", "error", "completed", "cancelled", "failed"] = "completed"
    elif current.status == "failed":
        if current.error is None:
            raise RuntimeError(f"Failed run {current.run_id} has no error message — Tier 1 anomaly (error column NULL on terminal failure)")
        payload = FailedData(
            detail=current.error,
            node_id=None,
        )
        event_type = "failed"
    elif current.status == "cancelled":
        payload = CancelledData(
            rows_processed=current.rows_processed,
            rows_succeeded=current.rows_succeeded,
            rows_failed=current.rows_failed,
            rows_quarantined=current.rows_quarantined,
            rows_routed_success=current.rows_routed_success,
            rows_routed_failure=current.rows_routed_failure,
        )
        event_type = "cancelled"
    else:
        raise RuntimeError(f"_build_terminal_run_event called for non-terminal status {current.status!r}")

    timestamp = current.finished_at or current.started_at
    if timestamp is None:
        raise RuntimeError(f"Terminal run {current.run_id} has no timestamps — Tier 1 anomaly (both finished_at and started_at are NULL)")
    return RunEvent(
        run_id=current.run_id,
        timestamp=timestamp,
        event_type=event_type,
        data=payload,
    )


def _counted(label: str, count: int) -> str:
    """Return a small English count phrase."""
    if count == 1:
        return f"1 {label}"
    return f"{count} {label}s"


def _summarize_counts(prefix: str, counts: dict[str, int]) -> str | None:
    """Render snapshot counts without implying hidden progress."""
    if not counts:
        return None
    details = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    return f"{prefix} include {details}."


def _diagnostics_evidence(diagnostics: RunDiagnosticsResponse) -> list[str]:
    """Build plain-English evidence from the visible diagnostics snapshot."""
    evidence: list[str] = []
    token_count = diagnostics.summary.token_count
    if token_count > 0:
        evidence.append(f"{_counted('token', token_count)} {'is' if token_count == 1 else 'are'} visible in the runtime trace.")
        if diagnostics.summary.preview_truncated:
            evidence.append(f"The preview is limited to the first {_counted('token', diagnostics.summary.preview_limit)}.")

    state_summary = _summarize_counts("Node states", diagnostics.summary.state_counts)
    if state_summary is not None:
        evidence.append(state_summary)

    operation_summary = _summarize_counts("Operation records", diagnostics.summary.operation_counts)
    if operation_summary is not None:
        evidence.append(operation_summary)

    for artifact in diagnostics.artifacts[:3]:
        evidence.append(f"Saved output is visible at {artifact.path_or_uri}.")
    if len(diagnostics.artifacts) > 3:
        additional_artifacts = len(diagnostics.artifacts) - 3
        evidence.append(
            f"{_counted('additional saved output', additional_artifacts)} {'is' if additional_artifacts == 1 else 'are'} visible."
        )

    if diagnostics.summary.latest_activity_at is not None:
        evidence.append(f"Latest recorded activity is {diagnostics.summary.latest_activity_at.isoformat()}.")

    if not evidence:
        evidence.append("No tokens, operations, or saved outputs are visible yet.")
    return evidence


def _fallback_diagnostics_working_view(
    explanation: str,
    diagnostics: RunDiagnosticsResponse,
) -> RunDiagnosticsWorkingView:
    """Synthesize a working view when the LLM returns plain text."""
    has_runtime_records = bool(
        diagnostics.summary.token_count or diagnostics.summary.state_counts or diagnostics.summary.operation_counts or diagnostics.artifacts
    )
    if diagnostics.artifacts:
        headline = "The run has produced saved output"
    elif has_runtime_records:
        headline = "Runtime records are updating"
    else:
        headline = "No runtime records are visible yet"

    if explanation.strip():
        meaning = explanation.strip()
    elif has_runtime_records:
        meaning = "The run has visible runtime records, so the server is doing work beyond showing the spinner."
    else:
        meaning = "The run may still be setting up; no bounded runtime records are visible in Landscape yet."

    next_steps: list[str] = []
    if diagnostics.artifacts:
        next_steps.append("Check the saved output path when the run completes.")
    if diagnostics.run_status in RUN_STATUS_NON_TERMINAL_VALUES:
        next_steps.append("Refresh diagnostics if the visible evidence does not change soon.")

    return RunDiagnosticsWorkingView(
        headline=headline,
        evidence=_diagnostics_evidence(diagnostics),
        meaning=meaning,
        next_steps=next_steps,
    )


def _strip_json_code_fence(text: str) -> str:
    """Accept fenced JSON defensively while the prompt still forbids it."""
    lines = text.strip().splitlines()
    if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text.strip()


def _parse_run_diagnostics_working_view(
    explanation: str,
    diagnostics: RunDiagnosticsResponse,
) -> tuple[str, RunDiagnosticsWorkingView]:
    """Parse the composer JSON response, falling back to visible evidence."""
    stripped = explanation.strip()
    try:
        working_view = RunDiagnosticsWorkingView.model_validate_json(_strip_json_code_fence(stripped))
    except pydantic.ValidationError:
        return stripped, _fallback_diagnostics_working_view(stripped, diagnostics)
    return working_view.meaning, working_view


# ── Router ─────────────────────────────────────────────────────────────


def create_execution_router() -> APIRouter:
    """Create the execution router with REST + WebSocket endpoints."""
    router = APIRouter(tags=["execution"])

    # ── Session-scoped endpoints (validate, execute) ──────────────────

    @router.post(
        "/api/sessions/{session_id}/validate",
        response_model=ValidationResult,
    )
    async def validate_session_pipeline(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        service: ExecutionService = Depends(_get_execution_service),  # noqa: B008
    ) -> ValidationResult:
        """Dry-run validation using real engine code paths."""
        await _verify_session_ownership(session_id, user, request)
        result = await service.validate(session_id, user_id=user.user_id)
        return result

    @router.post(
        "/api/sessions/{session_id}/execute",
        status_code=202,
    )
    async def execute_pipeline(
        session_id: UUID,
        request: Request,
        state_id: UUID | None = None,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        service: ExecutionService = Depends(_get_execution_service),  # noqa: B008
    ) -> dict[str, str]:
        """Start a background pipeline run. Returns run_id immediately.

        RunAlreadyActiveError propagates to the app-level exception handler
        (Seam Contract D) which returns the canonical 409 envelope:
        {"detail": str(exc), "error_type": "run_already_active"}.
        """
        await _verify_session_ownership(session_id, user, request)
        try:
            run_id = await service.execute(session_id, state_id, user_id=user.user_id)
        except StateAccessError:
            # IDOR contract: the "state does not exist" and
            # "state belongs to another session" branches in the
            # service MUST surface here as byte-identical 404
            # responses.  Distinguishable ``detail`` strings would
            # let an authenticated attacker probe arbitrary UUIDs
            # against their own /execute and learn which ones exist
            # in OTHER users' sessions — the same oracle commit
            # e73a921a closed on ``send_message``.  If a future
            # refactor needs diagnostic precision, route it through
            # server-side audit/telemetry, never through the HTTP
            # response body.
            raise HTTPException(status_code=404, detail="State not found") from None
        except BlobNotFoundError:
            # IDOR contract (mirrors StateAccessError above): the
            # nonexistent-blob and cross-session-blob branches MUST
            # surface here as byte-identical 404 responses.  Before
            # this handler existed, nonexistent-blob propagated as a
            # 500 while cross-session-blob returned a 404 — the IDOR
            # status itself was a side channel.
            raise HTTPException(status_code=404, detail="Blob not found") from None
        except BlobSourcePathMismatchError as exc:
            # Tier 1 audit-integrity violation: composer-stored source
            # path diverges from the canonical blob storage_path.  The
            # exception carries both paths for operator triage; we log
            # them server-side but redact them from the HTTP response
            # because the path discloses internal storage layout to any
            # caller (including the LLM agent driving the composer in
            # an MCP context).  See elspeth-07089fbaa3.
            # Per CLAUDE.md logging policy, slog is permitted for
            # audit-system failures.  Tier 1 corruption of
            # composition_states.source.options.path qualifies: the
            # audit row exists but its content is structurally invalid,
            # so neither audit (Landscape — not yet open for this run)
            # nor operational telemetry can capture the divergence.
            # slog is the only channel the operator can use to
            # correlate the redacted HTTP body with the actual paths.
            slog.error(
                "blob_source_path_mismatch",
                blob_id=exc.blob_id,
                session_id=exc.session_id,
                stored_path=exc.stored_path,
                canonical_path=exc.canonical_path,
                issue="elspeth-07089fbaa3",
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "kind": "blob_source_path_mismatch",
                    "issue": "elspeth-07089fbaa3",
                    "message": (
                        "Composer-stored blob source path is not "
                        "structurally valid for the bound blob.  This "
                        "indicates a bug in composer persistence; the "
                        "operator must investigate the captured "
                        "composition state."
                    ),
                },
            ) from exc
        except SemanticContractViolationError as exc:
            # Structured 422 with the same payload shape /validate
            # surfaces. Status 422 (Unprocessable Entity) — the
            # request was syntactically valid but the composition
            # fails plugin-declared semantic contracts. The
            # bare-ValueError branch below maps to 404 because most
            # other ValueErrors at this site are state-not-found
            # cases that echo the caller's own input; semantic
            # violations are NOT state-not-found and need their own
            # status. SemanticContractViolationError IS a
            # ValueError, so this handler MUST sit above the bare
            # ``except ValueError`` (the catch-order discipline hook
            # enforces that).
            raise HTTPException(
                status_code=422,
                detail={
                    "kind": "semantic_contract_violation",
                    "errors": [
                        {
                            "component": entry.component,
                            "message": entry.message,
                            "severity": entry.severity,
                        }
                        for entry in exc.entries
                    ],
                    "semantic_contracts": [
                        {
                            "from_id": contract.from_id,
                            "to_id": contract.to_id,
                            "consumer_plugin": contract.consumer_plugin,
                            "producer_plugin": contract.producer_plugin,
                            "producer_field": contract.producer_field,
                            "consumer_field": contract.consumer_field,
                            "outcome": contract.outcome.value,
                            "requirement_code": contract.requirement.requirement_code,
                        }
                        for contract in exc.contracts
                    ],
                },
            ) from exc
        except ValueError as exc:
            # Remaining ValueError sources are non-IDOR: the user's
            # OWN session having no composition state (when state_id
            # is None), source/sink path-allowlist rejections that
            # echo the caller's own input, and UUID parse errors on
            # malformed blob_ref strings.  These are not cross-user
            # oracles, so the diagnostic body is kept.
            raise HTTPException(status_code=404, detail=str(exc)) from None
        return {"run_id": str(run_id)}

    # ── Run-scoped endpoints (status, cancel, results) ────────────────

    @router.get(
        "/api/runs/{run_id}",
        response_model=RunStatusResponse,
    )
    async def get_run_status(
        run_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        service: ExecutionService = Depends(_get_execution_service),  # noqa: B008
    ) -> RunStatusResponse:
        """Return current run status."""
        await _verify_run_ownership(run_id, user, request)
        try:
            status = await service.get_status(run_id)
        except ValueError:
            raise _run_not_found_http() from None
        if status.status in RUN_STATUS_TERMINAL_VALUES and status.landscape_run_id is not None and status.discard_summary is None:
            from elspeth.web.execution.discard_summary import load_discard_summaries_for_settings

            discard_summaries = await run_sync_in_worker(
                load_discard_summaries_for_settings,
                request.app.state.settings,
                (status.landscape_run_id,),
            )
            if status.landscape_run_id in discard_summaries:
                status = status.model_copy(update={"discard_summary": discard_summaries[status.landscape_run_id]})
        return status

    @router.get(
        "/api/runs/{run_id}/diagnostics",
        response_model=RunDiagnosticsResponse,
    )
    async def get_run_diagnostics(
        run_id: UUID,
        request: Request,
        limit: int = Query(50, ge=1, le=100),
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        service: ExecutionService = Depends(_get_execution_service),  # noqa: B008
    ) -> RunDiagnosticsResponse:
        """Return a bounded Landscape diagnostics snapshot for a run."""
        await _verify_run_ownership(run_id, user, request)
        try:
            status = await service.get_status(run_id)
        except ValueError:
            raise _run_not_found_http() from None

        landscape_run_id = status.landscape_run_id or status.run_id
        return await run_sync_in_worker(
            load_run_diagnostics_for_settings,
            request.app.state.settings,
            run_id=status.run_id,
            landscape_run_id=landscape_run_id,
            run_status=status.status,
            limit=limit,
        )

    @router.post(
        "/api/runs/{run_id}/diagnostics/evaluate",
        response_model=RunDiagnosticsEvaluationResponse,
    )
    async def evaluate_run_diagnostics(
        run_id: UUID,
        request: Request,
        limit: int = Query(50, ge=1, le=100),
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        service: ExecutionService = Depends(_get_execution_service),  # noqa: B008
    ) -> RunDiagnosticsEvaluationResponse:
        """Ask the configured LLM to explain the current diagnostics snapshot."""
        await _verify_run_ownership(run_id, user, request)
        try:
            status = await service.get_status(run_id)
        except ValueError:
            raise _run_not_found_http() from None

        landscape_run_id = status.landscape_run_id or status.run_id
        diagnostics = await run_sync_in_worker(
            load_run_diagnostics_for_settings,
            request.app.state.settings,
            run_id=status.run_id,
            landscape_run_id=landscape_run_id,
            run_status=status.status,
            limit=limit,
        )

        composer: ComposerService = request.app.state.composer_service
        try:
            explanation = await composer.explain_run_diagnostics(diagnostics.model_dump(mode="json"))
        except ComposerServiceError as exc:
            raise HTTPException(
                status_code=502,
                detail={"error_type": "run_diagnostics_explanation_failed", "detail": str(exc)},
            ) from exc

        explanation, working_view = _parse_run_diagnostics_working_view(explanation, diagnostics)
        return RunDiagnosticsEvaluationResponse(
            run_id=status.run_id,
            generated_at=datetime.now(UTC),
            explanation=explanation,
            working_view=working_view,
        )

    @router.post("/api/runs/{run_id}/cancel")
    async def cancel_run(
        run_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        service: ExecutionService = Depends(_get_execution_service),  # noqa: B008
    ) -> dict[str, str]:
        """Cancel a run. Idempotent on terminal runs."""
        await _verify_run_ownership(run_id, user, request)
        try:
            await service.cancel(run_id)
            status = await service.get_status(run_id)
        except ValueError:
            raise _run_not_found_http() from None
        return {"status": status.status}

    @router.get(
        "/api/runs/{run_id}/results",
        response_model=RunResultsResponse,
    )
    async def get_run_results(
        run_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        service: ExecutionService = Depends(_get_execution_service),  # noqa: B008
    ) -> RunResultsResponse:
        """Return final run results. 409 if run is not terminal."""
        await _verify_run_ownership(run_id, user, request)
        try:
            status = await service.get_status(run_id)
        except ValueError:
            raise _run_not_found_http() from None
        if status.status in RUN_STATUS_NON_TERMINAL_VALUES:
            raise HTTPException(
                status_code=409,
                detail=f"Run is still {status.status}",
            )
        if status.landscape_run_id is not None and status.discard_summary is None:
            from elspeth.web.execution.discard_summary import load_discard_summaries_for_settings

            discard_summaries = await run_sync_in_worker(
                load_discard_summaries_for_settings,
                request.app.state.settings,
                (status.landscape_run_id,),
            )
            if status.landscape_run_id in discard_summaries:
                status = status.model_copy(update={"discard_summary": discard_summaries[status.landscape_run_id]})
        # mypy can't narrow a Literal through frozenset membership — the
        # cast is safe because RUN_STATUS_NON_TERMINAL_VALUES is the exact
        # complement of RunResultsResponse's Literal values, enforced by a
        # module-load assertion in schemas.py.
        # Phase 2.2 (elspeth-0de989c56d): cast to the canonical 5-value
        # TerminalSessionRunStatus, not a hardcoded 3-value tuple — the
        # latter would mislead readers into thinking the API still uses the
        # narrow taxonomy after the widening.
        terminal_status = cast(TerminalSessionRunStatus, status.status)
        return RunResultsResponse(
            run_id=status.run_id,
            status=terminal_status,
            rows_processed=status.rows_processed,
            rows_succeeded=status.rows_succeeded,
            rows_failed=status.rows_failed,
            rows_routed_success=status.rows_routed_success,
            rows_routed_failure=status.rows_routed_failure,
            rows_quarantined=status.rows_quarantined,
            landscape_run_id=status.landscape_run_id,
            error=status.error,
            discard_summary=status.discard_summary,
        )

    # ── WebSocket Endpoint ─────────────────────────────────────────────

    @router.websocket("/ws/runs/{run_id}")
    async def websocket_run_progress(
        websocket: WebSocket,
        run_id: str,
        token: str | None = None,
    ) -> None:
        """Stream RunEvent JSON payloads for a specific run.

        AC #12: Authentication via ?token=<jwt> query parameter.
        Close code 4001 on auth failure — client MUST NOT auto-reconnect
        on 4001 (token must be refreshed or user must re-authenticate).
        """
        broadcaster: ProgressBroadcaster = websocket.app.state.broadcaster
        auth_provider: AuthProvider = websocket.app.state.auth_provider
        service: ExecutionService = websocket.app.state.execution_service

        # Auth: validate JWT from query parameter
        if token is None:
            await websocket.close(code=4001, reason="Missing authentication token")
            return
        try:
            user = await auth_provider.authenticate(token)
        except AuthenticationError:
            await websocket.close(code=4001, reason="Invalid authentication token")
            return

        await websocket.accept()

        # IDOR protection: verify authenticated user owns this run's session
        try:
            run_ownership = await service.verify_run_ownership(user, run_id)
            if not run_ownership:
                await websocket.close(code=4004, reason="Run not found")
                return
        except ValueError:
            await websocket.close(code=4004, reason="Run not found")
            return

        # Subscribe BEFORE checking terminal status to close the race
        # window where a run finishes between get_status() and subscribe().
        # If subscribed first, any terminal event broadcast during the check
        # lands in the queue and won't be lost.
        queue = broadcaster.subscribe(run_id)
        try:
            # Seed: if the run already reached a terminal state before the
            # client connected (short runs, page refresh), send the terminal
            # status immediately and close.
            try:
                current = await service.get_status(UUID(run_id))
            except ValueError:
                await websocket.close(code=4004, reason="Run not found")
                return
            if current.status in RUN_STATUS_TERMINAL_VALUES:
                event = _build_terminal_run_event(current)
                await websocket.send_json(event.model_dump(mode="json"))
                await websocket.close(code=1000)
                return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=60.0)
                except TimeoutError:
                    # Idle timeout — a terminal broadcast may have been missed.
                    # Re-check authoritative run status instead of sending an
                    # ad-hoc payload outside the RunEvent contract.
                    try:
                        current = await service.get_status(UUID(run_id))
                    except ValueError:
                        await websocket.close(code=4004, reason="Run not found")
                        break
                    if current.status in RUN_STATUS_TERMINAL_VALUES:
                        terminal_event = _build_terminal_run_event(current)
                        await websocket.send_json(terminal_event.model_dump(mode="json"))
                        await websocket.close(code=1000)
                        break
                    continue
                await websocket.send_json(event.model_dump(mode="json"))
                # "error" events are non-terminal (per-row exceptions).
                # "completed", "cancelled", and "failed" are terminal.
                if event.event_type in ("completed", "cancelled", "failed"):
                    await websocket.close(code=1000)
                    break
        except WebSocketDisconnect:
            pass  # Client disconnected — fall through to finally
        except (ConnectionError, OSError) as exc:
            slog.error(
                "websocket_handler_error",
                run_id=run_id,
                error=str(exc),
            )
            try:
                await websocket.close(code=1011, reason="Internal server error")
            except (WebSocketDisconnect, ConnectionError, OSError) as close_err:
                slog.error("websocket_close_failed", run_id=run_id, error=str(close_err))
        finally:
            broadcaster.unsubscribe(run_id, queue)

    return router
