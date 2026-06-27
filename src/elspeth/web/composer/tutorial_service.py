"""Tutorial run orchestration for ``POST /api/tutorial/run``."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from sqlalchemy import func, select, update

from elspeth.contracts import CallType
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    artifacts_table,
    calls_table,
    node_states_table,
    operations_table,
    rows_table,
    runs_table,
    validation_errors_table,
)
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.tutorial_models import TutorialOrphanCleanupResponse, TutorialRunOutput, TutorialRunResponse
from elspeth.web.config import WebSettings
from elspeth.web.execution.outputs import filesystem_path_candidates
from elspeth.web.execution.protocol import ExecutionService
from elspeth.web.paths import allowed_sink_directories
from elspeth.web.preferences.service import PreferencesService
from elspeth.web.sessions.ownership import verify_session_ownership
from elspeth.web.sessions.protocol import (
    OPERATOR_COMPLETION_RUN_STATUS_VALUES,
    SESSION_TERMINAL_RUN_STATUS_VALUES,
    RunRecord,
    SessionServiceProtocol,
)

_TUTORIAL_RUN_TIMEOUT_SECONDS = 120.0
_TUTORIAL_RUN_POLL_SECONDS = 0.25
_TUTORIAL_SESSION_TITLE_PREFIX = "hello-world ("
_ABANDONED_SESSION_TITLE_PREFIX = "abandoned-"


class TutorialRunIntegrityError(RuntimeError):
    """Raised when the tutorial run cannot be projected from real audit data."""


@dataclass(frozen=True, slots=True)
class _LiveTutorialProjection:
    output: TutorialRunOutput
    llm_call_count: int


@dataclass(frozen=True, slots=True)
class _LiveTutorialRun:
    response: TutorialRunResponse
    run_record: RunRecord
    projection: _LiveTutorialProjection


@dataclass(frozen=True, slots=True)
class _VerifiedArtifactBytes:
    path: Path
    content: bytes


async def run_tutorial_pipeline(
    *,
    request: Request,
    user: UserIdentity,
    session_id: str,
) -> TutorialRunResponse:
    """Run the first-run tutorial pipeline LIVE for the current user.

    The tutorial run is a real execution through the normal
    ``ExecutionService``: it scrapes the hosted sample pages and summarises
    them with the configured LLM exactly as any composed pipeline would, and
    projects results only from real Landscape/artifact rows after the run
    reaches a terminal operator-completion status.

    There is deliberately NO cached/replayed fast path. A user watches the
    pipeline assemble over a second or two; returning instant pre-baked rows
    immediately afterwards is the seam that read as a fake run, and it bypassed
    the very page-hosting the live scrape depends on. Every tutorial run is now
    the same backend path a real composed pipeline takes (tutorial backend
    parity), so what the learner sees is what the system actually does.
    """
    from uuid import UUID

    session_uuid = UUID(session_id)
    await verify_session_ownership(session_uuid, user, request)

    settings: WebSettings = request.app.state.settings
    session_service: SessionServiceProtocol = request.app.state.session_service

    live_run = await _run_live_tutorial(
        request=request,
        user=user,
        session_id=session_uuid,
        settings=settings,
        session_service=session_service,
    )
    return live_run.response


async def _run_live_tutorial(
    *,
    request: Request,
    user: UserIdentity,
    session_id: Any,
    settings: WebSettings,
    session_service: SessionServiceProtocol,
) -> _LiveTutorialRun:
    execution_service: ExecutionService = request.app.state.execution_service
    run_id = await execution_service.execute(
        session_id,
        user_id=user.user_id,
        auth_provider_type=settings.auth_provider,
    )
    run_record = await _wait_for_terminal_run(session_service, run_id)
    if run_record.status not in OPERATOR_COMPLETION_RUN_STATUS_VALUES:
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "tutorial_live_run_failed",
                "status": run_record.status,
                "detail": run_record.error or "The tutorial run did not complete successfully.",
            },
        )
    if run_record.landscape_run_id is None:
        raise TutorialRunIntegrityError(f"Completed tutorial run {run_id} has no Landscape run id")

    projection = await run_sync_in_worker(
        _project_live_tutorial_output,
        settings,
        run_id=str(run_id),
        landscape_run_id=run_record.landscape_run_id,
    )
    response = TutorialRunResponse(
        run_id=str(run_id),
        output=projection.output,
        seeded_from_cache=False,
        cache_key=None,
    )
    return _LiveTutorialRun(response=response, run_record=run_record, projection=projection)


async def _wait_for_terminal_run(session_service: SessionServiceProtocol, run_id: Any) -> RunRecord:
    deadline = time.monotonic() + _TUTORIAL_RUN_TIMEOUT_SECONDS
    while True:
        run_record = await session_service.get_run(run_id)
        if run_record.status in SESSION_TERMINAL_RUN_STATUS_VALUES:
            return run_record
        if time.monotonic() >= deadline:
            raise HTTPException(
                status_code=504,
                detail={"error_type": "tutorial_run_timeout", "detail": "The tutorial run did not finish before the request timeout."},
            )
        await asyncio.sleep(_TUTORIAL_RUN_POLL_SECONDS)


def _project_live_tutorial_output(settings: WebSettings, *, run_id: str, landscape_run_id: str) -> _LiveTutorialProjection:
    # Despite the read-shaped name this is a WRITER surface: it stamps
    # ``llm_call_count`` / ``seeded_from_cache`` / ``cache_key`` onto the run
    # row (Tier-1 contract assertion below) in the same transaction as its
    # projection SELECTs.  ``write_connection()`` declares the write intent
    # so the transaction begins ``BEGIN IMMEDIATE`` (ADR-030 §D5) — a
    # read-then-write shape on a DEFERRED BEGIN is exactly the
    # SQLITE_BUSY_SNAPSHOT hazard the write-intent discipline closes.
    with (
        LandscapeDB.from_url(
            settings.get_landscape_url(),
            passphrase=settings.landscape_passphrase,
        ) as db,
        db.write_connection() as conn,
    ):
        llm_call_count = _count_calls_for_run(conn, landscape_run_id)
        discarded_row_count = _count_discarded_rows(conn, landscape_run_id)
        conn.execute(
            update(runs_table)
            .where(runs_table.c.run_id == landscape_run_id)
            # Tier-1 contract assertion: live runs are non-cache-replay identity.
            .values(llm_call_count=llm_call_count, seeded_from_cache=False, cache_key=None)
        )
        source_hashes = tuple(
            row.source_data_hash
            for row in conn.execute(
                select(rows_table.c.source_data_hash)
                .where(rows_table.c.run_id == landscape_run_id)
                .distinct()
                .order_by(rows_table.c.source_data_hash.asc())
            )
        )
        source_data_hash = _coalesce_run_source_hashes(source_hashes, run_id=run_id)
        artifact_rows = tuple(
            conn.execute(
                select(
                    artifacts_table.c.artifact_id,
                    artifacts_table.c.artifact_type,
                    artifacts_table.c.path_or_uri,
                    artifacts_table.c.content_hash,
                    artifacts_table.c.size_bytes,
                    artifacts_table.c.created_at,
                )
                .where(artifacts_table.c.run_id == landscape_run_id)
                .order_by(artifacts_table.c.created_at.desc(), artifacts_table.c.artifact_id.asc())
            )
        )
    rows = _rows_from_artifacts(
        artifact_rows,
        data_dir=settings.data_dir,
        run_id=run_id,
    )
    return _LiveTutorialProjection(
        output=TutorialRunOutput(
            rows=tuple(rows),
            source_data_hash=source_data_hash,
            discarded_row_count=discarded_row_count,
        ),
        llm_call_count=llm_call_count,
    )


def _coalesce_run_source_hashes(source_hashes: Sequence[str], *, run_id: str) -> str:
    if not source_hashes:
        raise TutorialRunIntegrityError(f"Tutorial live run {run_id} has no source_data_hash rows")
    if len(source_hashes) == 1:
        return source_hashes[0]
    return stable_hash({"source_data_hashes": list(source_hashes)})


def _count_calls_for_run(conn: Any, landscape_run_id: str) -> int:
    state_call_count = conn.execute(
        select(func.count())
        .select_from(calls_table.join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id))
        .where(node_states_table.c.run_id == landscape_run_id)
        .where(calls_table.c.call_type == CallType.LLM.value)
    ).scalar_one()
    operation_call_count = conn.execute(
        select(func.count())
        .select_from(calls_table.join(operations_table, calls_table.c.operation_id == operations_table.c.operation_id))
        .where(operations_table.c.run_id == landscape_run_id)
        .where(calls_table.c.call_type == CallType.LLM.value)
    ).scalar_one()
    return int(state_call_count) + int(operation_call_count)


def _count_discarded_rows(conn: Any, landscape_run_id: str) -> int:
    """Count rows the source DISCARDED for this run.

    A discarded row is a Landscape validation_errors entry whose ``destination`` is
    the sentinel ``"discard"`` (as opposed to a sink name, which is a quarantine with
    a visible destination). These rows are recorded for audit but never reach the
    output, so the tutorial UX must surface their count to avoid silently presenting
    only the survivors.
    """
    discarded = conn.execute(
        select(func.count())
        .select_from(validation_errors_table)
        .where(
            validation_errors_table.c.run_id == landscape_run_id,
            validation_errors_table.c.destination == "discard",
        )
    ).scalar_one()
    return int(discarded)


_ROW_FORMAT_SUFFIXES = frozenset({".csv", ".tsv", ".jsonl", ".ndjson", ".json"})


def _has_stored_row_format_suffix(path_or_uri: str) -> bool:
    fs_paths = filesystem_path_candidates(path_or_uri)
    if fs_paths is None:
        return False
    return fs_paths[-1].suffix.lower() in _ROW_FORMAT_SUFFIXES


def _read_audited_artifact_bytes(
    artifact: Any,
    *,
    allowed: Sequence[Path],
    run_id: str,
) -> _VerifiedArtifactBytes | None:
    fs_paths = filesystem_path_candidates(artifact.path_or_uri)
    if fs_paths is None:
        return None
    if not _has_stored_row_format_suffix(artifact.path_or_uri):
        return None

    saw_allowed = False
    saw_existing = False
    for fs_path in fs_paths:
        resolved = fs_path.resolve()
        if not any(resolved.is_relative_to(base) for base in allowed):
            continue
        saw_allowed = True
        try:
            content = resolved.read_bytes()
        except FileNotFoundError:
            continue
        saw_existing = True
        content_hash = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)
        if size_bytes == artifact.size_bytes and content_hash == artifact.content_hash:
            return _VerifiedArtifactBytes(path=resolved, content=content)

    if not saw_allowed:
        raise TutorialRunIntegrityError(f"Tutorial run {run_id} artifact {artifact.artifact_id!r} is outside the sink allowlist")
    if not saw_existing:
        raise TutorialRunIntegrityError(f"Tutorial run {run_id} artifact {artifact.artifact_id!r} is missing from disk")
    raise TutorialRunIntegrityError(
        f"Tutorial run {run_id} artifact {artifact.artifact_id!r} does not match audited content_hash/size_bytes"
    )


def _rows_from_artifacts(artifact_rows: Sequence[Any], *, data_dir: Path, run_id: str) -> list[dict[str, Any]]:
    """Project the rows produced by a tutorial run from its file artifacts.

    Three distinct Tier-1 failure modes are surfaced separately rather than
    collapsed into one "empty result" message:

    1. **Corrupt row-format artifact** — ``_parse_rows_file`` raises
       ``TutorialRunIntegrityError`` directly. Caller never sees the
       ambiguous empty list.
    2. **No row-bearing artifact** — every file artifact has a suffix outside
       ``_ROW_FORMAT_SUFFIXES`` (e.g. only ``.txt`` or ``.parquet`` files
       were emitted). Distinct error names the recognised formats so the
       operator can fix the sink configuration or extend the parser.
    3. **All row-bearing artifacts yielded zero rows** — distinct error makes
       it clear the parse succeeded but the pipeline produced no output rows.
    """
    allowed = allowed_sink_directories(str(data_dir))
    saw_row_format = False
    for artifact in artifact_rows:
        if artifact.artifact_type != "file":
            continue
        verified = _read_audited_artifact_bytes(artifact, allowed=allowed, run_id=run_id)
        if verified is None:
            continue
        rows = _parse_rows_content(verified.path, verified.content)
        if rows is None:
            # Non-row-bearing artifact (auxiliary debug file, parquet we
            # don't read, etc.). Skip cleanly — distinct from "this row
            # artifact parsed empty".
            continue
        saw_row_format = True
        if rows:
            return rows
    if saw_row_format:
        raise TutorialRunIntegrityError(
            f"Tutorial run {run_id} row-bearing artifacts yielded zero rows after parsing — the pipeline succeeded but produced no output"
        )
    raise TutorialRunIntegrityError(
        f"Tutorial run {run_id}: no row-bearing artifact found (recognised formats: {sorted(_ROW_FORMAT_SUFFIXES)})"
    )


def _parse_rows_file(path: Path) -> list[dict[str, Any]] | None:
    return _parse_rows_content(path, path.read_bytes())


def _parse_rows_content(path: Path, content: bytes) -> list[dict[str, Any]] | None:
    """Parse a file artifact as a row sequence.

    Returns:
        ``list[dict[str, Any]]`` — rows successfully parsed from a recognised
            row format (the list may be empty: a legitimate "header-only
            CSV" or "empty JSON list" yields zero rows).
        ``None`` — the file's suffix is not in ``_ROW_FORMAT_SUFFIXES``. The
            caller should skip this artifact and try the next; distinct
            from "this row artifact yielded zero rows".

    Raises:
        TutorialRunIntegrityError: the file IS a row-format artifact but its
            contents are structurally corrupt (non-object JSONL row, bare
            scalar/null at the top level of a JSON document, JSON object
            without a ``rows: object[]`` field, etc.). Corruption in a
            Tier-1 audit artifact must crash — silently coalescing to an
            empty list would shadow the corruption behind the misleading
            ``"no row-bearing artifact"`` projection message.
    """
    suffix = path.suffix.lower()
    if suffix not in _ROW_FORMAT_SUFFIXES:
        return None
    if suffix == ".csv":
        return [dict(row) for row in csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))]
    if suffix == ".tsv":
        return [dict(row) for row in csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""), delimiter="\t")]
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        for line in content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if type(value) is not dict:
                raise TutorialRunIntegrityError(f"Tutorial artifact {path} contains a non-object JSONL row")
            rows.append(dict(value))
        return rows
    # suffix == ".json"
    value = json.loads(content.decode("utf-8"))
    if type(value) is list:
        if not all(type(item) is dict for item in value):
            raise TutorialRunIntegrityError(f"Tutorial artifact {path} JSON list contains a non-object row")
        return [dict(item) for item in value]
    if type(value) is dict:
        rows_value = value["rows"] if "rows" in value else None
        if type(rows_value) is not list or not all(type(item) is dict for item in rows_value):
            raise TutorialRunIntegrityError(f"Tutorial artifact {path} JSON object must contain rows: object[]")
        return [dict(item) for item in rows_value]
    raise TutorialRunIntegrityError(
        f"Tutorial artifact {path} JSON top-level must be a list of objects or an object with a "
        f"'rows: object[]' field; got {type(value).__name__}"
    )


async def cleanup_tutorial_orphans(
    *,
    request: Request,
    user: UserIdentity,
) -> TutorialOrphanCleanupResponse:
    """Soft-delete abandoned tutorial sessions by renaming them.

    The frontend calls this on tutorial entry. The response preserves the
    historical ``deleted_count`` contract, but the operation is intentionally a
    rename so the user's audit history remains available.
    """
    preferences_service: PreferencesService = request.app.state.preferences_service
    settings: WebSettings = request.app.state.settings
    session_service: SessionServiceProtocol = request.app.state.session_service
    prefs = await preferences_service.get_composer_preferences(user.user_id)
    if prefs.tutorial_completed_at is not None:
        return TutorialOrphanCleanupResponse(deleted_count=0)

    deleted_count = 0
    offset = 0
    limit = 200
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    while True:
        sessions = await session_service.list_sessions(
            user.user_id,
            settings.auth_provider,
            limit=limit,
            offset=offset,
        )
        if not sessions:
            break
        for session in sessions:
            if session.title.startswith(_TUTORIAL_SESSION_TITLE_PREFIX) and not session.title.startswith(_ABANDONED_SESSION_TITLE_PREFIX):
                await session_service.update_session_title(
                    session.id,
                    f"{_ABANDONED_SESSION_TITLE_PREFIX}{session.title}-{timestamp}",
                )
                deleted_count += 1
        if len(sessions) < limit:
            break
        offset += limit
    return TutorialOrphanCleanupResponse(deleted_count=deleted_count)
