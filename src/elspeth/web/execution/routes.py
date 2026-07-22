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
import hashlib
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote
from uuid import UUID

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from starlette.types import Receive, Scope, Send

from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.protocol import BlobNotFoundError
from elspeth.web.composer.protocol import ComposerService, ComposerServiceError
from elspeth.web.composer.service import _BadRequestLLMError
from elspeth.web.config import WebSettings
from elspeth.web.execution.accounting import load_run_accounting_for_settings
from elspeth.web.execution.diagnostics import llm_safe_diagnostics_snapshot, load_run_diagnostics_for_settings
from elspeth.web.execution.errors import (
    BlobSourcePathMismatchError,
    ExecuteRequestValidationError,
    PipelineValidationError,
    RunSessionIntegrityError,
    SemanticContractViolationError,
    UnresolvedInterpretationPlaceholderError,
)
from elspeth.web.execution.fanout_guard import FANOUT_GUARD_ERROR_TYPE, ExecutionFanoutGuardRequired
from elspeth.web.execution.outputs import (
    RunOutputsAuditUnavailableError,
    filesystem_path_candidates,
    load_run_outputs_for_settings,
)
from elspeth.web.execution.preview import DEFAULT_ARTIFACT_PREVIEW_BYTE_CAP, build_artifact_preview_from_head
from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.protocol import ExecutionService, StateAccessError
from elspeth.web.execution.schemas import (
    RUN_STATUS_NON_TERMINAL_VALUES,
    RUN_STATUS_TERMINAL_VALUES,
    CancelledData,
    CompletedData,
    ExecuteRequest,
    FailedData,
    RunDiagnosticsEvaluationResponse,
    RunDiagnosticsResponse,
    RunDiagnosticsWorkingView,
    RunEvent,
    RunEventType,
    RunOutputArtifact,
    RunOutputArtifactPreview,
    RunOutputsResponse,
    RunResultsResponse,
    RunStatusResponse,
    ValidationResult,
    WebSocketTicketResponse,
)
from elspeth.web.execution.websocket_ticket import WebSocketTicketStore
from elspeth.web.paths import allowed_sink_directories
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.ownership import verify_session_ownership
from elspeth.web.sessions.protocol import (
    RunEventRecord,
    RunRecord,
    SessionServiceProtocol,
    TerminalSessionRunStatus,
)
from elspeth.web.sessions.routes._helpers import _litellm_error_detail

slog = structlog.get_logger()
_ARTIFACT_SNAPSHOT_CHUNK_SIZE = 1024 * 1024


# ── Dependency providers (using app.state, matching existing pattern) ──


async def _get_execution_service(request: Request) -> ExecutionService:
    return cast(ExecutionService, request.app.state.execution_service)


async def _get_session_service(request: Request) -> SessionServiceProtocol:
    return cast(SessionServiceProtocol, request.app.state.session_service)


def _get_websocket_ticket_store(app: Any) -> WebSocketTicketStore:
    return cast(WebSocketTicketStore, app.state.websocket_ticket_store)


@dataclass(frozen=True)
class _ArtifactFileSnapshot:
    path: Path
    size_bytes: int
    content_hash: str


@dataclass(frozen=True)
class _ArtifactPreviewHeadSnapshot:
    total_size_bytes: int
    content_hash: str
    head_bytes: bytes


@dataclass(frozen=True)
class _ByteRange:
    start: int
    end_inclusive: int

    @property
    def length(self) -> int:
        return self.end_inclusive - self.start + 1


def _artifact_content_drift_http(
    artifact: RunOutputArtifact,
    *,
    actual_size_bytes: int,
) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "error_type": "artifact_content_drift",
            "path_or_uri": artifact.path_or_uri,
            "expected_size_bytes": artifact.size_bytes,
            "actual_size_bytes": actual_size_bytes,
            "expected_content_hash": artifact.content_hash,
        },
    )


def _reject_artifact_content_drift(
    artifact: RunOutputArtifact,
    *,
    actual_size_bytes: int,
    actual_content_hash: str,
) -> None:
    if actual_size_bytes != artifact.size_bytes or actual_content_hash != artifact.content_hash:
        raise _artifact_content_drift_http(
            artifact,
            actual_size_bytes=actual_size_bytes,
        )


def _artifact_purged_or_moved_http(artifact: RunOutputArtifact) -> HTTPException:
    return HTTPException(
        status_code=410,
        detail={
            "error_type": "artifact_purged_or_moved",
            "path_or_uri": artifact.path_or_uri,
        },
    )


def _artifact_error_type(exc: HTTPException) -> str | None:
    if not isinstance(exc.detail, Mapping):
        return None
    error_type = exc.detail.get("error_type")
    return error_type if isinstance(error_type, str) else None


def _resolved_allowed_artifact_paths(
    artifact: RunOutputArtifact,
    *,
    data_dir: str | Path,
    session_id: str | None,
    object_store_error_type: str,
) -> tuple[Path, ...]:
    fs_paths = filesystem_path_candidates(artifact.path_or_uri)
    if fs_paths is None:
        raise HTTPException(
            status_code=415,
            detail={
                "error_type": object_store_error_type,
                "path_or_uri": artifact.path_or_uri,
            },
        )

    allowed = allowed_sink_directories(str(data_dir), session_id=session_id)
    resolved_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for fs_path in fs_paths:
        try:
            resolved = fs_path.resolve()
        except OSError:
            continue
        if resolved in seen_paths:
            continue
        if any(resolved.is_relative_to(base) for base in allowed):
            resolved_paths.append(resolved)
            seen_paths.add(resolved)

    if not resolved_paths:
        raise HTTPException(
            status_code=403,
            detail={
                "error_type": "output_path_outside_allowlist",
                "path_or_uri": artifact.path_or_uri,
            },
        )
    return tuple(resolved_paths)


def _unlink_path(path: Path) -> None:
    path.unlink(missing_ok=True)


def _download_content_disposition(filename: str) -> str:
    quoted = quote(filename, safe="")
    if quoted != filename:
        return f"attachment; filename*=utf-8''{quoted}"
    return f'attachment; filename="{filename}"'


def _range_not_satisfiable_http(size_bytes: int) -> HTTPException:
    return HTTPException(
        status_code=416,
        detail={"error_type": "range_not_satisfiable"},
        headers={"Content-Range": f"bytes */{size_bytes}"},
    )


def _parse_single_range(range_header: str | None, *, size_bytes: int) -> _ByteRange | None:
    if range_header is None:
        return None
    if not range_header.startswith("bytes=") or "," in range_header:
        raise _range_not_satisfiable_http(size_bytes)

    range_spec = range_header.removeprefix("bytes=")
    start_raw, separator, end_raw = range_spec.partition("-")
    if separator == "":
        raise _range_not_satisfiable_http(size_bytes)

    try:
        if start_raw == "":
            suffix_length = int(end_raw)
            if suffix_length <= 0:
                raise ValueError
            start = max(size_bytes - suffix_length, 0)
            end_inclusive = size_bytes - 1
        else:
            start = int(start_raw)
            end_inclusive = size_bytes - 1 if end_raw == "" else int(end_raw)
    except ValueError:
        raise _range_not_satisfiable_http(size_bytes) from None

    if size_bytes <= 0 or start < 0 or end_inclusive < start or start >= size_bytes:
        raise _range_not_satisfiable_http(size_bytes)

    return _ByteRange(
        start=start,
        end_inclusive=min(end_inclusive, size_bytes - 1),
    )


def _stream_temp_snapshot(path: Path, *, byte_range: _ByteRange | None = None) -> Iterator[bytes]:
    try:
        with path.open("rb") as source:
            remaining: int | None
            if byte_range is None:
                remaining = None
            else:
                source.seek(byte_range.start)
                remaining = byte_range.length

            while remaining is None or remaining > 0:
                read_size = _ARTIFACT_SNAPSHOT_CHUNK_SIZE if remaining is None else min(_ARTIFACT_SNAPSHOT_CHUNK_SIZE, remaining)
                chunk = source.read(read_size)
                if not chunk:
                    break
                if remaining is not None:
                    remaining -= len(chunk)
                yield chunk
    finally:
        _unlink_path(path)


class _TempSnapshotStreamingResponse(StreamingResponse):
    def __init__(
        self,
        snapshot_path: Path,
        *,
        headers: Mapping[str, str],
        status_code: int = 200,
        byte_range: _ByteRange | None = None,
    ) -> None:
        self._snapshot_path = snapshot_path
        super().__init__(
            _stream_temp_snapshot(snapshot_path, byte_range=byte_range),
            status_code=status_code,
            media_type="application/octet-stream",
            headers=headers,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            _unlink_path(self._snapshot_path)


def _copy_artifact_to_temp_snapshot(path: Path, snapshot_dir: Path) -> _ArtifactFileSnapshot:
    digest = hashlib.sha256()
    size_bytes = 0
    temp_path: Path | None = None
    snapshot_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        with (
            tempfile.NamedTemporaryFile(
                "wb",
                prefix="elspeth-run-output-",
                suffix=".snapshot",
                delete=False,
                dir=snapshot_dir,
            ) as temp_file,
            path.open("rb") as source,
        ):
            temp_path = Path(temp_file.name)
            while chunk := source.read(_ARTIFACT_SNAPSHOT_CHUNK_SIZE):
                size_bytes += len(chunk)
                digest.update(chunk)
                temp_file.write(chunk)
    except Exception:
        if temp_path is not None:
            _unlink_path(temp_path)
        raise

    assert temp_path is not None
    return _ArtifactFileSnapshot(
        path=temp_path,
        size_bytes=size_bytes,
        content_hash=digest.hexdigest(),
    )


async def _verified_artifact_file_snapshot(
    resolved: Path,
    artifact: RunOutputArtifact,
    *,
    snapshot_dir: Path,
) -> _ArtifactFileSnapshot:
    try:
        snapshot = await run_sync_in_worker(_copy_artifact_to_temp_snapshot, resolved, snapshot_dir)
    except FileNotFoundError:
        raise _artifact_purged_or_moved_http(artifact) from None

    try:
        _reject_artifact_content_drift(
            artifact,
            actual_size_bytes=snapshot.size_bytes,
            actual_content_hash=snapshot.content_hash,
        )
    except HTTPException:
        _unlink_path(snapshot.path)
        raise

    return snapshot


def _read_artifact_preview_head(path: Path, *, byte_cap: int) -> _ArtifactPreviewHeadSnapshot:
    digest = hashlib.sha256()
    total_size_bytes = 0
    head = bytearray()
    with path.open("rb") as source:
        while chunk := source.read(_ARTIFACT_SNAPSHOT_CHUNK_SIZE):
            total_size_bytes += len(chunk)
            digest.update(chunk)
            remaining_head_bytes = byte_cap - len(head)
            if remaining_head_bytes > 0:
                head.extend(chunk[:remaining_head_bytes])

    return _ArtifactPreviewHeadSnapshot(
        total_size_bytes=total_size_bytes,
        content_hash=digest.hexdigest(),
        head_bytes=bytes(head),
    )


async def _verified_artifact_preview_head(
    resolved: Path,
    artifact: RunOutputArtifact,
    *,
    byte_cap: int = DEFAULT_ARTIFACT_PREVIEW_BYTE_CAP,
) -> _ArtifactPreviewHeadSnapshot:
    try:
        snapshot = await run_sync_in_worker(_read_artifact_preview_head, resolved, byte_cap=byte_cap)
    except FileNotFoundError:
        raise _artifact_purged_or_moved_http(artifact) from None

    _reject_artifact_content_drift(
        artifact,
        actual_size_bytes=snapshot.total_size_bytes,
        actual_content_hash=snapshot.content_hash,
    )
    return snapshot


async def _verified_artifact_file_snapshot_from_candidates(
    artifact: RunOutputArtifact,
    *,
    data_dir: str | Path,
    session_id: str | None,
    snapshot_dir: Path,
) -> tuple[Path, _ArtifactFileSnapshot]:
    candidates = _resolved_allowed_artifact_paths(
        artifact,
        data_dir=data_dir,
        session_id=session_id,
        object_store_error_type="object_store_artifact_not_streamable",
    )
    purged_error: HTTPException | None = None
    drift_error: HTTPException | None = None
    for resolved in candidates:
        try:
            snapshot = await _verified_artifact_file_snapshot(
                resolved,
                artifact,
                snapshot_dir=snapshot_dir,
            )
        except HTTPException as exc:
            error_type = _artifact_error_type(exc)
            if error_type == "artifact_purged_or_moved":
                purged_error = exc
                continue
            if error_type == "artifact_content_drift":
                drift_error = exc
                continue
            raise
        return resolved, snapshot

    if drift_error is not None:
        raise drift_error
    if purged_error is not None:
        raise purged_error
    raise _artifact_purged_or_moved_http(artifact)


async def _verified_artifact_preview_head_from_candidates(
    artifact: RunOutputArtifact,
    *,
    data_dir: str | Path,
    session_id: str | None,
) -> tuple[Path, _ArtifactPreviewHeadSnapshot]:
    candidates = _resolved_allowed_artifact_paths(
        artifact,
        data_dir=data_dir,
        session_id=session_id,
        object_store_error_type="object_store_artifact_not_previewable",
    )
    purged_error: HTTPException | None = None
    drift_error: HTTPException | None = None
    for resolved in candidates:
        try:
            snapshot = await _verified_artifact_preview_head(resolved, artifact)
        except HTTPException as exc:
            error_type = _artifact_error_type(exc)
            if error_type == "artifact_purged_or_moved":
                purged_error = exc
                continue
            if error_type == "artifact_content_drift":
                drift_error = exc
                continue
            raise
        return resolved, snapshot

    if drift_error is not None:
        raise drift_error
    if purged_error is not None:
        raise purged_error
    raise _artifact_purged_or_moved_http(artifact)


# ── Ownership verification helpers ────────────────────────────────────
#
# Session-ownership verification lives in ``web/sessions/ownership.py`` as
# ``verify_session_ownership`` so ``execution/routes.py`` and
# ``audit_readiness/routes.py`` share a single IDOR-safe implementation.
# Run-ownership verification remains here — only execution/ runs care.


async def _verify_run_ownership(run_id: UUID, user: UserIdentity, request: Request) -> RunRecord:
    """Verify the run exists and belongs to the current user's session.

    Looks up the run's parent session and checks ownership.
    Returns 404 (not 403) to avoid leaking run existence (IDOR).
    Returns the verified run record so callers can scope further checks
    (e.g. the per-session sink allowlist, elspeth-bdc17cfdb1) to the
    run's owning session without a second lookup.
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

    if session.archived_at is not None or session.user_id != user.user_id or session.auth_provider_type != settings.auth_provider:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


def _run_not_found_http() -> HTTPException:
    """Canonical IDOR-safe not-found response for run-scoped routes."""
    return HTTPException(status_code=404, detail="Run not found")


class _RunStatusNotFoundError(Exception):
    """Run disappeared between ownership verification and status projection."""


class _RunStatusIntegrityError(Exception):
    """Run status accounting projection failed internal integrity validation."""


@dataclass(frozen=True, slots=True)
class _LoadedRunStatus:
    """Run status projection paired with the exact session-row snapshot used to build it."""

    response: RunStatusResponse
    record: RunRecord


def _run_integrity_http(exc: ValidationError | _RunStatusIntegrityError) -> HTTPException:
    detail: dict[str, Any] = {
        "code": "run_integrity_error",
        # "detail" (not "message"): parseResponse (frontend/src/api/client.ts)
        # reads nestedDetail.detail as the human-readable string, with no
        # "message" fallback; keeps this site in shape-lockstep with
        # _run_accounting_integrity_http in sessions/routes/_helpers.py.
        "detail": "Run status failed internal accounting validation.",
    }
    if isinstance(exc, ValidationError):
        detail["validation_errors"] = exc.errors(include_url=False, include_context=False, include_input=False)
    else:
        detail["error"] = str(exc)
    return HTTPException(status_code=500, detail=detail)


async def _load_run_status_snapshot_with_accounting(
    run_id: UUID,
    *,
    app: Any,
    service: ExecutionService,
) -> _LoadedRunStatus:
    """Load run status with Landscape-derived accounting when a run has audit data."""
    session_service: SessionServiceProtocol = app.state.session_service
    try:
        run_record = await session_service.get_run(run_id)
    except ValueError as exc:
        raise _RunStatusNotFoundError from exc

    accounting = None
    if run_record.landscape_run_id and run_record.status in RUN_STATUS_TERMINAL_VALUES:
        try:
            accounting_by_run_id = await run_sync_in_worker(
                load_run_accounting_for_settings,
                app.state.settings,
                (run_record.landscape_run_id,),
            )
        except ValueError as exc:
            raise _RunStatusIntegrityError(str(exc)) from exc
        if run_record.landscape_run_id in accounting_by_run_id:
            accounting = accounting_by_run_id[run_record.landscape_run_id]

    try:
        status = await service.get_status(run_id, accounting=accounting, run_record=run_record)
    except ValidationError:
        raise
    except ValueError as exc:
        raise _RunStatusNotFoundError from exc
    return _LoadedRunStatus(response=status, record=run_record)


async def _load_run_status_with_accounting(
    run_id: UUID,
    *,
    app: Any,
    service: ExecutionService,
) -> RunStatusResponse:
    """Load run status with accounting, returning only the public status response."""
    loaded = await _load_run_status_snapshot_with_accounting(run_id, app=app, service=service)
    return loaded.response


def _build_terminal_run_event(current: RunStatusResponse, *, cancelled_run_record: RunRecord | None = None) -> RunEvent:
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
    completion_status = current.status
    if completion_status == "completed" or completion_status == "completed_with_failures" or completion_status == "empty":
        if current.landscape_run_id is None:
            raise RuntimeError(f"Completed run {current.run_id} has no landscape_run_id — Tier 1 anomaly (audit trail incomplete)")
        if current.accounting is None:
            raise RuntimeError(f"Completed run {current.run_id} has no accounting — Tier 1 anomaly (audit trail incomplete)")
        try:
            payload: CompletedData | FailedData | CancelledData = CompletedData(
                status=completion_status,
                accounting=current.accounting,
                landscape_run_id=current.landscape_run_id,
            )
        except ValidationError as exc:
            raise RuntimeError(
                f"Completed run {current.run_id} failed CompletedData validation — Tier 1 anomaly (audit trail inconsistent): {exc}"
            ) from exc
        event_type: RunEventType = "completed"
    elif current.status == "failed":
        if current.error is None:
            raise RuntimeError(f"Failed run {current.run_id} has no error message — Tier 1 anomaly (error column NULL on terminal failure)")
        payload = FailedData(
            detail=current.error,
            node_id=None,
        )
        event_type = "failed"
    elif current.status == "cancelled":
        if current.accounting is not None:
            payload = CancelledData(
                source_rows_processed=current.accounting.source.rows_processed,
                tokens_succeeded=current.accounting.tokens.succeeded,
                tokens_failed=current.accounting.tokens.failed,
                tokens_quarantined=current.accounting.routing.quarantined,
                tokens_routed_success=current.accounting.routing.routed_success,
                tokens_routed_failure=current.accounting.routing.routed_failure,
            )
        else:
            if cancelled_run_record is None:
                raise RuntimeError(
                    f"Cancelled run {current.run_id} has no accounting and no RunRecord counters — "
                    f"Tier 1 anomaly (cancellation replay cannot be reconstructed)"
                )
            if str(cancelled_run_record.id) != current.run_id:
                raise RuntimeError(
                    f"Cancelled replay RunRecord mismatch: status run {current.run_id} received counters for run {cancelled_run_record.id}"
                )
            if cancelled_run_record.status != "cancelled":
                raise RuntimeError(f"Cancelled replay RunRecord status mismatch for run {current.run_id}: {cancelled_run_record.status!r}")
            payload = CancelledData(
                source_rows_processed=cancelled_run_record.rows_processed,
                tokens_succeeded=cancelled_run_record.rows_succeeded,
                tokens_failed=cancelled_run_record.rows_failed,
                tokens_quarantined=cancelled_run_record.rows_quarantined,
                tokens_routed_success=cancelled_run_record.rows_routed_success,
                tokens_routed_failure=cancelled_run_record.rows_routed_failure,
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


def _run_event_from_record(record: RunEventRecord) -> RunEvent:
    return RunEvent.model_validate(
        {
            "run_id": str(record.run_id),
            "timestamp": record.timestamp,
            "event_type": record.event_type,
            "data": record.data,
        }
    ).with_event_sequence(record.sequence)


def _counted(label: str, count: int) -> str:
    """Return a small English count phrase."""
    if count == 1:
        return f"1 {label}"
    return f"{count} {label}s"


def _summarize_counts(prefix: str, counts: Mapping[Any, int]) -> str | None:
    """Render snapshot counts without implying hidden progress."""
    if not counts:
        return None
    details = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    return f"{prefix} include {details}."


def _diagnostics_evidence(diagnostics: RunDiagnosticsResponse) -> list[str]:
    """Build plain-English evidence from the visible diagnostics snapshot."""
    evidence: list[str] = []
    if diagnostics.cancel_requested:
        evidence.append("Cancellation has been requested; active work is draining toward a terminal cancelled status.")
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
    elif diagnostics.cancel_requested:
        headline = "Cancellation requested"
    elif has_runtime_records:
        headline = "Runtime records are updating"
    else:
        headline = "No runtime records are visible yet"

    if explanation.strip():
        meaning = explanation.strip()
    elif diagnostics.cancel_requested:
        meaning = "The server has received the cancel request and is waiting for active work to stop."
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
    except ValidationError:
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
        state_id: UUID | None = None,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        service: ExecutionService = Depends(_get_execution_service),  # noqa: B008
        session_service: SessionServiceProtocol = Depends(_get_session_service),  # noqa: B008
    ) -> ValidationResult:
        """Dry-run validation using real engine code paths."""
        await verify_session_ownership(session_id, user, request)
        if state_id is None:
            result = await service.validate(session_id, user_id=user.user_id)
            return result
        try:
            state_record = await session_service.get_state(state_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="State not found") from exc
        if state_record.session_id != session_id:
            raise HTTPException(status_code=404, detail="State not found")
        result = await service.validate_state(
            state_from_record(state_record),
            user_id=user.user_id,
            session_id=session_id,
        )
        return result

    @router.post(
        "/api/sessions/{session_id}/execute",
        status_code=202,
    )
    async def execute_pipeline(
        session_id: UUID,
        request: Request,
        state_id: UUID | None = None,
        execute_request: ExecuteRequest | None = Body(default=None),  # noqa: B008
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        service: ExecutionService = Depends(_get_execution_service),  # noqa: B008
    ) -> dict[str, str]:
        """Start a background pipeline run. Returns run_id immediately.

        RunAlreadyActiveError propagates to the app-level exception handler
        (Seam Contract D) which returns the canonical 409 envelope:
        {"detail": str(exc), "error_type": "run_already_active"}.
        """
        await verify_session_ownership(session_id, user, request)
        settings: WebSettings = request.app.state.settings
        fanout_ack_token = execute_request.fanout_ack_token if execute_request is not None else None
        try:
            run_id = await service.execute(
                session_id,
                state_id,
                user_id=user.user_id,
                auth_provider_type=settings.auth_provider,
                fanout_ack_token=fanout_ack_token,
            )
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
            # an MCP context).
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
            )
            public_detail = (
                "Composer-stored blob source path is not "
                "structurally valid for the bound blob.  This "
                "indicates a bug in composer persistence; the "
                "operator must investigate the captured "
                "composition state."
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error_type": "blob_source_path_mismatch",
                    "detail": public_detail,
                    "kind": "blob_source_path_mismatch",
                    "message": public_detail,
                },
            ) from exc
        except ExecutionFanoutGuardRequired as exc:
            raise HTTPException(
                status_code=428,
                detail={
                    "error_type": FANOUT_GUARD_ERROR_TYPE,
                    "detail": str(exc),
                    "fanout_guard": exc.guard.to_dict(),
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
                    "error_type": "semantic_contract_violation",
                    "detail": str(exc),
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
        except UnresolvedInterpretationPlaceholderError as exc:
            # F-17 / F-21 (Phase 5b Task 5 follow-on). Structured 422 with
            # the structured interpretation-review sites so the frontend
            # banner can list every unresolved site without parsing the
            # message string. The legacy ``placeholders`` field is preserved
            # for transform/vague-term callers during the contract migration.
            # 422 mirrors the SemanticContractViolationError precedent
            # above — the request was syntactically valid but the
            # composition state is not yet executable until the operator
            # resolves the surfaced placeholders. Placement: above the
            # ``except ExecuteRequestValidationError`` (which maps to 400)
            # and the bare ``except ValueError`` (which maps to 404) so
            # this 422 path is reached for the specific exception type.
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "interpretation_placeholder_unresolved",
                    "detail": str(exc),
                    "kind": "interpretation_placeholder_unresolved",
                    "message": str(exc),
                    "placeholders": [{"node_id": node_id, "term": term} for node_id, term in exc.placeholders],
                    "interpretation_sites": [
                        {
                            "component_id": site.component_id,
                            "component_type": site.component_type,
                            "kind": site.kind.value,
                            "user_term": site.user_term,
                        }
                        for site in exc.sites
                    ],
                },
            ) from exc
        except PipelineValidationError as exc:
            # Fail-closed pre-run validation: the composed pipeline failed the
            # dry-run validate_pipeline BEFORE any run was created. 422 mirrors the
            # SemanticContractViolationError precedent (syntactically valid request,
            # non-executable composition). MUST sit above ``except ValueError`` (404)
            # and ``except ExecuteRequestValidationError`` (400) — PipelineValidationError
            # subclasses ValueError. The structured ``errors`` mirror the /validate
            # ValidationResult.errors shape so the frontend banner needs no string parsing.
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "pipeline_validation_failure",
                    "detail": str(exc),
                    "kind": "pipeline_validation_failure",
                    "errors": [
                        {
                            "component_id": err.component_id,
                            "component_type": err.component_type,
                            "message": err.message,
                            "suggestion": err.suggestion,
                            "error_code": err.error_code,
                        }
                        for err in exc.errors
                    ],
                },
            ) from exc
        except ExecuteRequestValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except ValueError as exc:
            # Remaining ValueError sources are non-IDOR: the user's
            # OWN session having no composition state (when state_id
            # is None). Caller-authored request validation failures
            # (path allowlist, malformed blob_ref) raise
            # ExecuteRequestValidationError above and return 400.
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
            status = await _load_run_status_with_accounting(run_id, app=request.app, service=service)
        except _RunStatusNotFoundError:
            raise _run_not_found_http() from None
        except (ValidationError, _RunStatusIntegrityError) as exc:
            raise _run_integrity_http(exc) from exc
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
            status = await _load_run_status_with_accounting(run_id, app=request.app, service=service)
        except _RunStatusNotFoundError:
            raise _run_not_found_http() from None
        except (ValidationError, _RunStatusIntegrityError) as exc:
            raise _run_integrity_http(exc) from exc

        landscape_run_id = status.landscape_run_id or status.run_id
        return await run_sync_in_worker(
            load_run_diagnostics_for_settings,
            request.app.state.settings,
            run_id=status.run_id,
            landscape_run_id=landscape_run_id,
            run_status=status.status,
            cancel_requested=status.cancel_requested,
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
            status = await _load_run_status_with_accounting(run_id, app=request.app, service=service)
        except _RunStatusNotFoundError:
            raise _run_not_found_http() from None
        except (ValidationError, _RunStatusIntegrityError) as exc:
            raise _run_integrity_http(exc) from exc

        landscape_run_id = status.landscape_run_id or status.run_id
        diagnostics = await run_sync_in_worker(
            load_run_diagnostics_for_settings,
            request.app.state.settings,
            run_id=status.run_id,
            landscape_run_id=landscape_run_id,
            run_status=status.status,
            cancel_requested=status.cancel_requested,
            limit=limit,
        )

        composer: ComposerService = request.app.state.composer_service
        settings: WebSettings = request.app.state.settings
        try:
            explanation = await composer.explain_run_diagnostics(llm_safe_diagnostics_snapshot(diagnostics))
        except _BadRequestLLMError as exc:
            # Provider rejected the request (400-class). Carrier exposes
            # `provider_detail` / `provider_status_code` precisely because
            # `str(exc)` is redacted to the class-name wrap. Delegate to
            # `_litellm_error_detail` — the same helper sessions routes use
            # — so the staging-debug surface is symmetric across endpoints.
            raise HTTPException(
                status_code=502,
                detail=_litellm_error_detail(
                    "run_diagnostics_explanation_failed",
                    exc,
                    expose_provider_error=settings.composer_expose_provider_errors,
                ),
            ) from exc
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
    ) -> dict[str, str | bool]:
        """Cancel a run. Idempotent on terminal runs."""
        await _verify_run_ownership(run_id, user, request)
        try:
            await service.cancel(run_id)
            status = await _load_run_status_with_accounting(run_id, app=request.app, service=service)
        except _RunStatusNotFoundError:
            raise _run_not_found_http() from None
        except (ValidationError, _RunStatusIntegrityError) as exc:
            raise _run_integrity_http(exc) from exc
        return {"status": status.status, "cancel_requested": status.cancel_requested}

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
            status = await _load_run_status_with_accounting(run_id, app=request.app, service=service)
        except _RunStatusNotFoundError:
            raise _run_not_found_http() from None
        except (ValidationError, _RunStatusIntegrityError) as exc:
            raise _run_integrity_http(exc) from exc
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
            accounting=status.accounting,
            landscape_run_id=status.landscape_run_id,
            error=status.error,
            discard_summary=status.discard_summary,
        )

    # ── WebSocket Endpoint ─────────────────────────────────────────────

    @router.websocket("/ws/runs/{run_id}")
    async def websocket_run_progress(
        websocket: WebSocket,
        run_id: str,
        ticket: str | None = None,
        token: str | None = None,
    ) -> None:
        """Stream RunEvent JSON payloads for a specific run.

        Authentication uses a short-lived, single-use ?ticket=<opaque> query
        parameter minted by POST /api/runs/{run_id}/ws-ticket. Session JWTs
        are never accepted in the WebSocket URL.
        Close code 4001 on auth failure — client MUST NOT auto-reconnect
        on 4001 (ticket must be refreshed or user must re-authenticate).
        """
        broadcaster: ProgressBroadcaster = websocket.app.state.broadcaster
        service: ExecutionService = websocket.app.state.execution_service

        if token is not None:
            await websocket.close(code=4001, reason="Use a WebSocket ticket, not a session token")
            return
        if ticket is None:
            await websocket.close(code=4001, reason="Missing WebSocket ticket")
            return
        user = _get_websocket_ticket_store(websocket.app).consume(ticket=ticket, run_id=run_id)
        if user is None:
            await websocket.close(code=4001, reason="Invalid or expired WebSocket ticket")
            return

        await websocket.accept()

        # IDOR protection: verify authenticated user owns this run's session
        try:
            run_ownership = await service.verify_run_ownership(user, run_id)
            if not run_ownership:
                await websocket.close(code=4004, reason="Run not found")
                return
        except RunSessionIntegrityError as integrity_exc:
            # Tier-1 sessions-DB corruption: an existing run references a
            # session row that does not exist. This is NOT a Tier-3 not-found
            # case — surfacing it as 4004 would launder internal referential
            # corruption into a benign client response. Landscape carries the
            # run audit, not this sessions-DB invariant breach, so slog is the
            # operator channel (CLAUDE.md logging policy: audit-system
            # failure). Close 1011 (internal error), mirroring the seed-snapshot
            # integrity handling below.
            slog.error(
                "websocket_run_ownership_session_integrity_error",
                run_id=run_id,
                session_id=integrity_exc.session_id,
                error=str(integrity_exc),
            )
            await websocket.close(code=1011, reason="Run ownership check failed internal integrity validation")
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
                current_snapshot = await _load_run_status_snapshot_with_accounting(UUID(run_id), app=websocket.app, service=service)
            except _RunStatusNotFoundError:
                await websocket.close(code=4004, reason="Run not found")
                return
            except (ValidationError, _RunStatusIntegrityError) as integrity_exc:
                # Tier-1 accounting projection failed integrity validation on
                # the seed snapshot. The close frame is the routed outcome for
                # the external client, but the operator needs the detail to
                # diagnose the divergence — Landscape carries the run audit,
                # not this projection failure, so slog is the only channel
                # (CLAUDE.md logging policy: audit-system failure).
                slog.error(
                    "websocket_run_status_integrity_error",
                    run_id=run_id,
                    phase="seed",
                    error=str(integrity_exc),
                )
                await websocket.close(code=1011, reason="Run status failed internal accounting validation")
                return
            current = current_snapshot.response
            max_replayed_sequence = 0
            replayed_terminal = False
            for persisted in await websocket.app.state.session_service.list_run_events(UUID(run_id)):
                replay_event = _run_event_from_record(persisted)
                await websocket.send_json(replay_event.model_dump(mode="json"))
                max_replayed_sequence = max(max_replayed_sequence, persisted.sequence)
                if replay_event.event_type in ("completed", "cancelled", "failed"):
                    replayed_terminal = True
            if replayed_terminal:
                await websocket.close(code=1000)
                return
            if current.status in RUN_STATUS_TERMINAL_VALUES:
                event = _build_terminal_run_event(current, cancelled_run_record=current_snapshot.record)
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
                        current_snapshot = await _load_run_status_snapshot_with_accounting(UUID(run_id), app=websocket.app, service=service)
                    except _RunStatusNotFoundError:
                        await websocket.close(code=4004, reason="Run not found")
                        break
                    except (ValidationError, _RunStatusIntegrityError) as integrity_exc:
                        # Same Tier-1 accounting integrity failure as the seed
                        # path, on the idle-timeout recheck. Record the detail
                        # before signalling internal-error close (see seed
                        # handler above for the logging-channel rationale).
                        slog.error(
                            "websocket_run_status_integrity_error",
                            run_id=run_id,
                            phase="idle_recheck",
                            error=str(integrity_exc),
                        )
                        await websocket.close(code=1011, reason="Run status failed internal accounting validation")
                        break
                    current = current_snapshot.response
                    if current.status in RUN_STATUS_TERMINAL_VALUES:
                        terminal_event = _build_terminal_run_event(current, cancelled_run_record=current_snapshot.record)
                        await websocket.send_json(terminal_event.model_dump(mode="json"))
                        await websocket.close(code=1000)
                        break
                    continue
                if event.event_sequence is not None and event.event_sequence <= max_replayed_sequence:
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

    # NOTE on placement: the run-outputs endpoints sit AFTER
    # websocket_run_progress in this file rather than next to
    # get_run_diagnostics (their conceptual sibling). Reason: the
    # tier-model allowlist (config/cicd/enforce_tier_model/web.yaml)
    # uses AST-path-based fingerprints which include the function's
    # body-level index. Inserting siblings BEFORE websocket_run_progress
    # shifts that index and invalidates the existing allowlist entries.
    # Appending here keeps existing fingerprints stable.

    @router.get(
        "/api/runs/{run_id}/outputs",
        response_model=RunOutputsResponse,
    )
    async def get_run_outputs(
        run_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        service: ExecutionService = Depends(_get_execution_service),  # noqa: B008
    ) -> RunOutputsResponse:
        """Return the FULL manifest of sink-write artefacts for a run.

        Distinct from ``GET /api/runs/{run_id}/diagnostics``, whose
        ``artifacts`` field is capped at 20 for operator-UI pacing. This
        endpoint is the audit-evidence retrieval surface — every artefact
        the run wrote, with ``content_hash`` and ``exists_now``.
        """
        run = await _verify_run_ownership(run_id, user, request)
        try:
            status = await _load_run_status_with_accounting(run_id, app=request.app, service=service)
        except _RunStatusNotFoundError:
            raise _run_not_found_http() from None
        except (ValidationError, _RunStatusIntegrityError) as exc:
            raise _run_integrity_http(exc) from exc

        landscape_run_id = status.landscape_run_id or status.run_id
        try:
            return await run_sync_in_worker(
                load_run_outputs_for_settings,
                request.app.state.settings,
                run_id=status.run_id,
                landscape_run_id=landscape_run_id,
                session_id=str(run.session_id),
            )
        except RunOutputsAuditUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_type": "run_outputs_audit_unavailable",
                    "landscape_run_id": exc.landscape_run_id,
                    "audit_location": exc.audit_location,
                },
            ) from exc

    @router.get("/api/runs/{run_id}/outputs/{artifact_id}/content")
    async def get_run_output_content(
        run_id: UUID,
        artifact_id: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        service: ExecutionService = Depends(_get_execution_service),  # noqa: B008
    ) -> Any:
        """Stream the bytes of one artefact written by a run.

        Path-allowlist guard: refuses any artefact whose ``path_or_uri``
        resolves outside the canonical sink set for the run's owning
        session — ``data_dir/outputs`` plus that session's own
        ``data_dir/blobs/<session>/`` subtree (elspeth-bdc17cfdb1). This
        is defence-in-depth — the path was already allowlisted at write
        time, but the audit row is read-mutable in principle and the
        read-side guard MUST NOT trust it.

        Returns:
        * 200 with file bytes when path is in-allowlist and exists.
        * 403 when path is outside allowlist.
        * 404 when artefact is not in the run's manifest.
        * 410 when path was in-allowlist but file no longer exists.
        """
        run = await _verify_run_ownership(run_id, user, request)
        try:
            status = await _load_run_status_with_accounting(run_id, app=request.app, service=service)
        except _RunStatusNotFoundError:
            raise _run_not_found_http() from None
        except (ValidationError, _RunStatusIntegrityError) as exc:
            raise _run_integrity_http(exc) from exc

        landscape_run_id = status.landscape_run_id or status.run_id
        try:
            manifest = await run_sync_in_worker(
                load_run_outputs_for_settings,
                request.app.state.settings,
                run_id=status.run_id,
                landscape_run_id=landscape_run_id,
                session_id=str(run.session_id),
            )
        except RunOutputsAuditUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_type": "run_outputs_audit_unavailable",
                    "landscape_run_id": exc.landscape_run_id,
                    "audit_location": exc.audit_location,
                },
            ) from exc
        artifact = next(
            (a for a in manifest.artifacts if a.artifact_id == artifact_id),
            None,
        )
        if artifact is None:
            raise HTTPException(
                status_code=404,
                detail={"error_type": "artifact_not_found", "artifact_id": artifact_id},
            )

        data_dir = request.app.state.settings.data_dir

        # Audit-evidence integrity: the artifact row and the on-disk file are
        # read-mutable in principle, so an in-allowlist file can be overwritten
        # after the run. Verify the current bytes match the audit-recorded size
        # AND content_hash before streaming; otherwise we would serve
        # non-audited content under the artifact's identity, breaking the
        # audit-evidence retrieval contract (elspeth-50189c547c). content_hash is
        # the whole-file SHA-256 for every file-streamable sink, so a whole-file
        # comparison is correct — and catches same-size byte substitution, which
        # a size check alone would miss.
        resolved, snapshot = await _verified_artifact_file_snapshot_from_candidates(
            artifact,
            data_dir=data_dir,
            session_id=str(run.session_id),
            snapshot_dir=Path(request.app.state.settings.data_dir) / ".run-output-snapshots",
        )
        try:
            byte_range = _parse_single_range(request.headers.get("range"), size_bytes=snapshot.size_bytes)
        except HTTPException:
            _unlink_path(snapshot.path)
            raise

        response_headers = {
            "Accept-Ranges": "bytes",
            "Content-Disposition": _download_content_disposition(resolved.name),
            "Content-Length": str(snapshot.size_bytes if byte_range is None else byte_range.length),
        }
        status_code = 200
        if byte_range is not None:
            status_code = 206
            response_headers["Content-Range"] = f"bytes {byte_range.start}-{byte_range.end_inclusive}/{snapshot.size_bytes}"

        return _TempSnapshotStreamingResponse(
            snapshot.path,
            headers=response_headers,
            status_code=status_code,
            byte_range=byte_range,
        )

    @router.get(
        "/api/runs/{run_id}/outputs/{artifact_id}/preview",
        response_model=RunOutputArtifactPreview,
    )
    async def get_run_output_preview(
        run_id: UUID,
        artifact_id: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        service: ExecutionService = Depends(_get_execution_service),  # noqa: B008
    ) -> RunOutputArtifactPreview:
        """Return a bounded head-of-file preview of one sink-write artefact.

        Companion to ``/content``: where ``/content`` streams the full
        file, ``/preview`` reads at most 256 KiB or 100 rows so the
        operator UI can render an inline preview without a full
        download. Same path-allowlist guard, same ownership check —
        the only behavioural difference is bounded read.

        Returns:
        * 200 with ``RunOutputArtifactPreview`` on success.
        * 403 when path is outside allowlist.
        * 404 when artefact is not in the run's manifest.
        * 410 when path was in-allowlist but file no longer exists
          (frontend treats this as the "no longer available on disk"
          state, mirroring the manifest's ``exists_now=False``).
        * 415 when the artefact is non-file (object-store URI).
        """
        run = await _verify_run_ownership(run_id, user, request)
        try:
            status = await _load_run_status_with_accounting(run_id, app=request.app, service=service)
        except _RunStatusNotFoundError:
            raise _run_not_found_http() from None
        except (ValidationError, _RunStatusIntegrityError) as exc:
            raise _run_integrity_http(exc) from exc

        landscape_run_id = status.landscape_run_id or status.run_id
        try:
            manifest = await run_sync_in_worker(
                load_run_outputs_for_settings,
                request.app.state.settings,
                run_id=status.run_id,
                landscape_run_id=landscape_run_id,
                session_id=str(run.session_id),
            )
        except RunOutputsAuditUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_type": "run_outputs_audit_unavailable",
                    "landscape_run_id": exc.landscape_run_id,
                    "audit_location": exc.audit_location,
                },
            ) from exc
        artifact = next(
            (a for a in manifest.artifacts if a.artifact_id == artifact_id),
            None,
        )
        if artifact is None:
            raise HTTPException(
                status_code=404,
                detail={"error_type": "artifact_not_found", "artifact_id": artifact_id},
            )

        data_dir = request.app.state.settings.data_dir

        resolved, snapshot = await _verified_artifact_preview_head_from_candidates(
            artifact,
            data_dir=data_dir,
            session_id=str(run.session_id),
        )

        return build_artifact_preview_from_head(
            resolved,
            artifact_id=artifact_id,
            total_size_bytes=snapshot.total_size_bytes,
            head_bytes=snapshot.head_bytes,
        )

    @router.post(
        "/api/runs/{run_id}/ws-ticket",
        response_model=WebSocketTicketResponse,
    )
    async def create_run_websocket_ticket(
        run_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> WebSocketTicketResponse:
        """Issue a short-lived one-use credential for the progress WebSocket."""
        await _verify_run_ownership(run_id, user, request)
        ticket = _get_websocket_ticket_store(request.app).issue(run_id=run_id, user=user)
        return WebSocketTicketResponse(ticket=ticket.ticket, expires_at=ticket.expires_at)

    return router
