"""Tutorial run orchestration for ``POST /api/tutorial/run``."""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from fastapi import HTTPException, Request
from sqlalchemy import func, select, update

from elspeth.contracts.freeze import deep_thaw
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    artifacts_table,
    calls_table,
    node_states_table,
    operations_table,
    rows_table,
    runs_table,
)
from elspeth.core.landscape.write_repository import LandscapeWriteRepository
from elspeth.plugins.infrastructure.discovery import discover_all_plugins
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.tutorial_models import TutorialOrphanCleanupResponse, TutorialRunOutput, TutorialRunResponse
from elspeth.web.config import WebSettings
from elspeth.web.execution.outputs import path_or_uri_to_filesystem_path
from elspeth.web.execution.protocol import ExecutionService
from elspeth.web.paths import allowed_sink_directories
from elspeth.web.preferences.service import PreferencesService
from elspeth.web.preferences.tutorial_cache import (
    CANONICAL_SEED_PROMPT,
    TutorialCache,
    TutorialCacheEntry,
    tutorial_cache_key,
)
from elspeth.web.sessions.ownership import verify_session_ownership
from elspeth.web.sessions.protocol import (
    OPERATOR_COMPLETION_RUN_STATUS_VALUES,
    SESSION_TERMINAL_RUN_STATUS_VALUES,
    CompositionStateData,
    RunRecord,
    SessionServiceProtocol,
)
from elspeth.web.validation import INTERPRETATION_PLACEHOLDER_RE

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


async def run_tutorial_pipeline(
    *,
    request: Request,
    user: UserIdentity,
    session_id: str,
    prompt: str,
) -> TutorialRunResponse:
    """Run or replay the first-run tutorial pipeline for the current user.

    Cache replay writes a fresh current-session ``runs`` row plus a fresh
    current-session Landscape run. Cache miss delegates to the normal
    ``ExecutionService`` and only projects results from real Landscape/artifact
    rows after the run reaches a terminal operator-completion status.
    """
    from uuid import UUID

    session_uuid = UUID(session_id)
    await verify_session_ownership(session_uuid, user, request)

    session_service: SessionServiceProtocol = request.app.state.session_service
    preferences_service: PreferencesService = request.app.state.preferences_service
    settings: WebSettings = request.app.state.settings
    cache: TutorialCache = request.app.state.tutorial_cache

    prefs = await preferences_service.get_composer_preferences(user.user_id)
    effective_prompt = prompt.strip() or CANONICAL_SEED_PROMPT
    is_canonical_prompt = effective_prompt == CANONICAL_SEED_PROMPT
    model_id = _tutorial_model_id(settings)

    bypass_reason: str | None = None
    if prefs.tutorial_completed_at is not None:
        bypass_reason = "completed"
    elif prefs.default_mode == "freeform":
        bypass_reason = "freeform"

    if bypass_reason is None and is_canonical_prompt:
        cache_entry = cache.lookup(CANONICAL_SEED_PROMPT, model_id)
        if cache_entry is not None:
            return await _replay_cache_entry(
                session_service=session_service,
                settings=settings,
                session_id=session_uuid,
                cache_entry=cache_entry,
                cache_key=tutorial_cache_key(CANONICAL_SEED_PROMPT, model_id),
            )

    await _normalise_current_tutorial_state_for_execution(
        session_service=session_service,
        session_id=session_uuid,
    )
    live_run = await _run_live_tutorial(
        request=request,
        user=user,
        session_id=session_uuid,
        settings=settings,
        session_service=session_service,
    )
    if bypass_reason is None and is_canonical_prompt:
        await _store_successful_live_projection(
            cache=cache,
            model_id=model_id,
            run_record=live_run.run_record,
            output=live_run.response.output,
            llm_call_count=live_run.projection.llm_call_count,
        )
    return live_run.response


async def _normalise_current_tutorial_state_for_execution(
    *,
    session_service: SessionServiceProtocol,
    session_id: Any,
) -> None:
    current_state = await session_service.get_current_state(session_id)
    if current_state is None:
        return

    normalised_nodes, changed = _normalise_bare_required_field_templates(current_state.nodes)
    if not changed:
        return

    composer_meta = dict(current_state.composer_meta or {})
    composer_meta["tutorial_runtime_normalized"] = True
    composer_meta["tutorial_normalization"] = "bare_required_field_templates"
    composer_meta["tutorial_normalized_from_state_id"] = str(current_state.id)
    await session_service.save_composition_state(
        session_id,
        CompositionStateData(
            source=current_state.source,
            nodes=normalised_nodes,
            edges=current_state.edges,
            outputs=current_state.outputs,
            metadata_=current_state.metadata_,
            is_valid=current_state.is_valid,
            validation_errors=current_state.validation_errors,
            composer_meta=composer_meta,
        ),
        provenance="convergence_persist",
    )


def _normalise_bare_required_field_templates(
    nodes: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]] | None, bool]:
    if nodes is None:
        return None, False

    changed = False
    normalised_nodes: list[dict[str, Any]] = []
    for node in nodes:
        node_copy = cast(dict[str, Any], deep_thaw(dict(node)))
        options_obj = node_copy["options"] if "options" in node_copy else None
        plugin_name = node_copy["plugin"] if "plugin" in node_copy else None
        if plugin_name != "llm" or type(options_obj) is not dict:
            normalised_nodes.append(node_copy)
            continue
        options = cast(dict[str, Any], options_obj)

        prompt_template = options["prompt_template"] if "prompt_template" in options else None
        required_input_fields = options["required_input_fields"] if "required_input_fields" in options else None
        if type(prompt_template) is not str or not _is_string_sequence(required_input_fields):
            normalised_nodes.append(node_copy)
            continue

        fields = cast(Sequence[str], required_input_fields)
        normalised_template = _normalise_tutorial_prompt_template(
            prompt_template,
            required_input_fields=fields,
        )

        if normalised_template != prompt_template:
            options["prompt_template"] = normalised_template
            hash_value = options["resolved_prompt_template_hash"] if "resolved_prompt_template_hash" in options else None
            if type(hash_value) is str:
                options["resolved_prompt_template_hash"] = stable_hash(normalised_template)
            changed = True
        normalised_nodes.append(node_copy)

    return normalised_nodes, changed


def _is_string_sequence(value: object) -> bool:
    return type(value) in {list, tuple} and all(type(item) is str for item in cast(Sequence[object], value))


def _normalise_tutorial_prompt_template(
    prompt_template: str,
    *,
    required_input_fields: Sequence[str],
) -> str:
    normalised_template = prompt_template
    for field_name in required_input_fields:
        if field_name.isidentifier():
            normalised_template = re.sub(
                rf"{{{{\s*{re.escape(field_name)}\s*}}}}",
                f"{{{{ row.{field_name} }}}}",
                normalised_template,
            )

    return INTERPRETATION_PLACEHOLDER_RE.sub(
        lambda match: match.group(1).strip(),
        normalised_template,
    )


async def _replay_cache_entry(
    *,
    session_service: SessionServiceProtocol,
    settings: WebSettings,
    session_id: Any,
    cache_entry: TutorialCacheEntry,
    cache_key: str,
) -> TutorialRunResponse:
    current_state = await session_service.get_current_state(session_id)
    if current_state is None:
        raise HTTPException(
            status_code=409,
            detail={"error_type": "tutorial_state_missing", "detail": "The tutorial session has no composition state to attach a run to."},
        )

    run_record = await session_service.create_run(
        session_id=session_id,
        state_id=current_state.id,
        pipeline_yaml=cache_entry.pipeline_yaml,
    )
    plugin_versions = _plugin_versions_from_pipeline_yaml(cache_entry.pipeline_yaml)

    def _write_landscape_run() -> str:
        with LandscapeDB.from_url(
            settings.get_landscape_url(),
            passphrase=settings.landscape_passphrase,
        ) as db:
            return LandscapeWriteRepository(db).record_synthesised_run(
                pipeline_yaml=cache_entry.pipeline_yaml,
                rows=cache_entry.rows,
                source_data_hash=cache_entry.source_data_hash,
                llm_call_count=0,
                plugin_versions=plugin_versions,
                started_at=run_record.started_at,
                metadata={
                    "seeded_from_cache": True,
                    "cache_key": cache_key,
                    "cache_seeding_llm_call_count": cache_entry.llm_call_count,
                },
            )

    landscape_run_id = await run_sync_in_worker(_write_landscape_run)
    await session_service.update_run_status(
        run_record.id,
        status="running",
        landscape_run_id=landscape_run_id,
    )
    await session_service.update_run_status(
        run_record.id,
        status="completed",
        rows_processed=len(cache_entry.rows),
        rows_succeeded=len(cache_entry.rows),
        rows_failed=0,
        rows_routed_success=0,
        rows_routed_failure=0,
        rows_quarantined=0,
    )
    return TutorialRunResponse(
        run_id=str(run_record.id),
        output=TutorialRunOutput(
            rows=[dict(row) for row in cache_entry.rows],
            source_data_hash=cache_entry.source_data_hash,
        ),
        seeded_from_cache=True,
        cache_key=cache_key,
    )


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
    with (
        LandscapeDB.from_url(
            settings.get_landscape_url(),
            passphrase=settings.landscape_passphrase,
        ) as db,
        db.connection() as conn,
    ):
        llm_call_count = _count_calls_for_run(conn, landscape_run_id)
        conn.execute(
            update(runs_table)
            .where(runs_table.c.run_id == landscape_run_id)
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
        output=TutorialRunOutput(rows=rows, source_data_hash=source_data_hash),
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
    ).scalar_one()
    operation_call_count = conn.execute(
        select(func.count())
        .select_from(calls_table.join(operations_table, calls_table.c.operation_id == operations_table.c.operation_id))
        .where(operations_table.c.run_id == landscape_run_id)
    ).scalar_one()
    return int(state_call_count) + int(operation_call_count)


def _rows_from_artifacts(artifact_rows: Sequence[Any], *, data_dir: Path, run_id: str) -> list[dict[str, Any]]:
    allowed = allowed_sink_directories(str(data_dir))
    for artifact in artifact_rows:
        if artifact.artifact_type != "file":
            continue
        fs_path = path_or_uri_to_filesystem_path(artifact.path_or_uri)
        if fs_path is None:
            continue
        resolved = fs_path.resolve()
        if not any(resolved.is_relative_to(base) for base in allowed):
            raise TutorialRunIntegrityError(f"Tutorial run {run_id} artifact {artifact.artifact_id!r} is outside the sink allowlist")
        if not resolved.exists():
            raise TutorialRunIntegrityError(f"Tutorial run {run_id} artifact {artifact.artifact_id!r} is missing from disk")
        rows = _parse_rows_file(resolved)
        if rows:
            return rows
    raise TutorialRunIntegrityError(f"Tutorial run {run_id} has no readable file artifact with rows")


def _parse_rows_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix == ".tsv":
        with path.open(newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f, delimiter="\t")]
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                value = json.loads(line)
                if type(value) is not dict:
                    raise TutorialRunIntegrityError(f"Tutorial artifact {path} contains a non-object JSONL row")
                rows.append(dict(value))
        return rows
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if type(value) is list:
            if not all(type(item) is dict for item in value):
                raise TutorialRunIntegrityError(f"Tutorial artifact {path} JSON list contains a non-object row")
            return [dict(item) for item in value]
        if type(value) is dict:
            rows_value = value["rows"] if "rows" in value else None
            if type(rows_value) is not list or not all(type(item) is dict for item in rows_value):
                raise TutorialRunIntegrityError(f"Tutorial artifact {path} JSON object must contain rows: object[]")
            return [dict(item) for item in rows_value]
    return []


async def _store_successful_live_projection(
    *,
    cache: TutorialCache,
    model_id: str,
    run_record: RunRecord,
    output: TutorialRunOutput,
    llm_call_count: int,
) -> None:
    if not _all_rows_succeeded(run_record):
        return
    if run_record.pipeline_yaml is None:
        raise TutorialRunIntegrityError(f"Tutorial run {run_record.id} completed without stored pipeline YAML")
    cache.store(
        TutorialCacheEntry(
            canonical_prompt=CANONICAL_SEED_PROMPT,
            model_id=model_id,
            cached_at=datetime.now(UTC),
            rows=output.rows,
            source_data_hash=output.source_data_hash,
            llm_call_count=llm_call_count,
            pipeline_yaml=run_record.pipeline_yaml,
        )
    )


def _all_rows_succeeded(run_record: RunRecord) -> bool:
    return (
        run_record.status == "completed"
        and run_record.rows_processed > 0
        and run_record.rows_succeeded == run_record.rows_processed
        and run_record.rows_failed == 0
        and run_record.rows_routed_success == 0
        and run_record.rows_routed_failure == 0
        and run_record.rows_quarantined == 0
    )


def _plugin_versions_from_pipeline_yaml(pipeline_yaml: str) -> dict[str, str]:
    parsed = yaml.safe_load(pipeline_yaml)
    if type(parsed) is not dict:
        raise TutorialRunIntegrityError("Cached tutorial pipeline YAML must parse to an object")
    plugin_names = _plugin_names_from_pipeline_dict(parsed)
    if not plugin_names:
        raise TutorialRunIntegrityError("Cached tutorial pipeline YAML contains no plugin nodes")
    versions = _discovered_plugin_versions()
    missing = [name for name in plugin_names if name not in versions]
    if missing:
        raise TutorialRunIntegrityError(f"Cached tutorial pipeline YAML references unknown plugins: {missing!r}")
    return {name: versions[name] for name in plugin_names}


def _plugin_names_from_pipeline_dict(doc: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    source = doc["source"] if "source" in doc else None
    if type(source) is dict and "plugin" in source:
        names.append(_require_plugin_name(source["plugin"]))
    transforms = doc["transforms"] if "transforms" in doc else {}
    if type(transforms) is dict:
        for transform in transforms.values():
            if type(transform) is dict and "plugin" in transform:
                names.append(_require_plugin_name(transform["plugin"]))
    sinks = doc["sinks"] if "sinks" in doc else {}
    if type(sinks) is dict:
        for sink in sinks.values():
            if type(sink) is dict and "plugin" in sink:
                names.append(_require_plugin_name(sink["plugin"]))
    return tuple(dict.fromkeys(names))


def _require_plugin_name(value: object) -> str:
    if type(value) is not str or not value:
        raise TutorialRunIntegrityError(f"Pipeline plugin name must be a non-empty string, got {value!r}")
    return value


def _discovered_plugin_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    discovered = discover_all_plugins()
    for plugin_classes in discovered.values():
        for plugin_class in plugin_classes:
            plugin = cast(Any, plugin_class)
            name = plugin.name
            version = plugin.plugin_version
            if type(name) is not str or type(version) is not str:
                raise TutorialRunIntegrityError(f"Plugin class {plugin_class!r} has invalid name/version metadata")
            versions[name] = version
    return versions


def _tutorial_model_id(settings: WebSettings) -> str:
    return settings.composer_model


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
