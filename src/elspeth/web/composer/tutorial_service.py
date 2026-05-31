"""Tutorial run orchestration for ``POST /api/tutorial/run``."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, cast

import yaml
from fastapi import HTTPException, Request
from sqlalchemy import func, select, update

from elspeth.contracts import NodeType
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
    validation_errors_table,
)
from elspeth.core.landscape.write_repository import LandscapeWriteRepository, SynthesisedNodeSpec
from elspeth.plugins.infrastructure.discovery import discover_all_plugins
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.skills import load_deployment_skill, load_skill_with_hash
from elspeth.web.composer.tutorial_models import TutorialOrphanCleanupResponse, TutorialRunOutput, TutorialRunResponse
from elspeth.web.composer.tutorial_telemetry import _CacheSkipReason, record_tutorial_cache_skipped, record_tutorial_runtime_normalization
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
    model_id = tutorial_model_id(settings)

    bypass_reason: str | None = None
    if prefs.tutorial_completed_at is not None:
        bypass_reason = "completed"
    elif prefs.default_mode == "freeform":
        bypass_reason = "freeform"

    if bypass_reason is None and is_canonical_prompt:
        cache_entry = cache.lookup(CANONICAL_SEED_PROMPT, model_id)
        if cache_entry is not None:
            # Verify the user's current composition state has the same plugin
            # topology as the cached tutorial pipeline. Without this gate, a
            # client posting the canonical prompt against an unrelated or
            # edited session would attach cached pipeline_yaml + rows to a
            # state_id pointing at a structurally different pipeline — a
            # Tier-1 audit lie. Mismatch falls through to live compose so the
            # synthesised audit faithfully describes whatever runs.
            current_state = await session_service.get_current_state(session_uuid)
            if current_state is not None and _state_matches_cached_topology(current_state, cache_entry.pipeline_yaml):
                return await _replay_cache_entry(
                    session_service=session_service,
                    settings=settings,
                    session_id=session_uuid,
                    current_state=current_state,
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
        provenance="tutorial_normalization",
    )
    record_tutorial_runtime_normalization("bare_required_field_templates")


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
    current_state: Any,
    cache_entry: TutorialCacheEntry,
    cache_key: str,
) -> TutorialRunResponse:
    """Project a cache hit into a synthesised Landscape run.

    The caller MUST have already verified
    ``_state_matches_cached_topology(current_state, cache_entry.pipeline_yaml)``
    is True. This function attaches the cached ``pipeline_yaml`` and rows to
    ``current_state.id``; that attachment is only audit-honest when the
    state and cache describe the same plugin topology.
    """
    run_record = await session_service.create_run(
        session_id=session_id,
        state_id=current_state.id,
        pipeline_yaml=cache_entry.pipeline_yaml,
    )
    node_specs = _node_specs_from_pipeline_yaml(cache_entry.pipeline_yaml)

    # L3→L3 import: web/composer reads the OpenRouter catalog snapshot id
    # at synthesis time so the cache-replay run row carries the same
    # audit-anchor as any live-executed run. Without this the replay row
    # would have NULL columns and the audit trail would be incomplete.
    from elspeth.plugins.transforms.llm.model_catalog import read_openrouter_catalog_snapshot_id

    catalog_sha, catalog_source = read_openrouter_catalog_snapshot_id()

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
                node_specs=node_specs,
                started_at=run_record.started_at,
                metadata={
                    "seeded_from_cache": True,
                    "cache_key": cache_key,
                    "cache_seeding_llm_call_count": cache_entry.llm_call_count,
                },
                openrouter_catalog_sha256=catalog_sha,
                openrouter_catalog_source=catalog_source,
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
            # ``TutorialRunOutput.model_config`` is strict; list-to-tuple
            # coercion is disabled, so build the tuple explicitly.
            rows=tuple(dict(row) for row in cache_entry.rows),
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
    ).scalar_one()
    operation_call_count = conn.execute(
        select(func.count())
        .select_from(calls_table.join(operations_table, calls_table.c.operation_id == operations_table.c.operation_id))
        .where(operations_table.c.run_id == landscape_run_id)
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
        fs_path = path_or_uri_to_filesystem_path(artifact.path_or_uri)
        if fs_path is None:
            continue
        resolved = fs_path.resolve()
        if not any(resolved.is_relative_to(base) for base in allowed):
            raise TutorialRunIntegrityError(f"Tutorial run {run_id} artifact {artifact.artifact_id!r} is outside the sink allowlist")
        if not resolved.exists():
            raise TutorialRunIntegrityError(f"Tutorial run {run_id} artifact {artifact.artifact_id!r} is missing from disk")
        rows = _parse_rows_file(resolved)
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
    # suffix == ".json"
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
    raise TutorialRunIntegrityError(
        f"Tutorial artifact {path} JSON top-level must be a list of objects or an object with a "
        f"'rows: object[]' field; got {type(value).__name__}"
    )


async def _store_successful_live_projection(
    *,
    cache: TutorialCache,
    model_id: str,
    run_record: RunRecord,
    output: TutorialRunOutput,
    llm_call_count: int,
) -> None:
    skip_reason = _cache_seed_skip_reason(run_record)
    if skip_reason is not None:
        # I6: instrument the silent-skip path. Before this counter, a
        # persistent tutorial degradation (e.g. every live run produces
        # one quarantined row) would leave the cache un-seeded forever
        # and every billed live run would discard its cache-seed value
        # — LLM-billing surge before anyone noticed. Operators can now
        # alert on a non-trivial floor of ``composer.tutorial.cache_skipped_total``
        # broken down by ``skip_reason``.
        record_tutorial_cache_skipped(skip_reason)
        return
    if run_record.pipeline_yaml is None:
        raise TutorialRunIntegrityError(f"Tutorial run {run_record.id} completed without stored pipeline YAML")
    cache.store(
        TutorialCacheEntry(
            canonical_prompt=CANONICAL_SEED_PROMPT,
            model_id=model_id,
            cached_at=datetime.now(UTC),
            # TutorialRunOutput.rows is a tuple (frozen-model immutability);
            # TutorialCacheEntry.rows is typed list[dict] for JSON round-trip.
            # Explicit conversion at the boundary keeps each type honest.
            rows=list(output.rows),
            source_data_hash=output.source_data_hash,
            llm_call_count=llm_call_count,
            pipeline_yaml=run_record.pipeline_yaml,
        )
    )


def _cache_seed_skip_reason(run_record: RunRecord) -> _CacheSkipReason | None:
    """Classify why the tutorial cache cannot be seeded from this run.

    Returns ``None`` when all rows succeeded; otherwise the most-specific
    closed-list reason. Conditional ordering matters: quarantined and
    routed_failure rows are subsets of rows_failed (Tier-1 constraints in
    ``RunRecord._validate_counters``); routed_success rows are a subset
    of rows_succeeded. Checking the specific subsets first ensures the
    emitted ``skip_reason`` is actionable on the operator dashboard
    rather than collapsing every classification into the catch-all
    ``rows_failed``.

    Status is the outermost gate because a non-completed status implies
    nothing about the counters' correctness.
    """
    if run_record.status != "completed":
        return "status_not_completed"
    if run_record.rows_processed == 0:
        return "zero_rows_processed"
    if run_record.rows_quarantined > 0:
        return "rows_quarantined"
    if run_record.rows_routed_failure > 0:
        return "rows_routed_failure"
    if run_record.rows_failed > 0:
        return "rows_failed"
    if run_record.rows_routed_success > 0:
        return "rows_routed_success"
    if run_record.rows_succeeded != run_record.rows_processed:
        return "rows_partial_success"
    return None


def _node_specs_from_pipeline_yaml(pipeline_yaml: str) -> tuple[SynthesisedNodeSpec, ...]:
    """Build the ordered Tier-1 audit topology for a cached tutorial pipeline.

    One ``SynthesisedNodeSpec`` per YAML occurrence — plugin reuse (e.g. two
    ``llm`` transforms, csv source plus csv sink) must produce one node row
    per occurrence, with ``node_type`` taken from the YAML's source / transforms
    / sinks key, not derived from list position.
    """
    parsed = yaml.safe_load(pipeline_yaml)
    if type(parsed) is not dict:
        raise TutorialRunIntegrityError("Cached tutorial pipeline YAML must parse to an object")
    roles = _plugin_nodes_from_pipeline_dict(parsed)
    if not roles:
        raise TutorialRunIntegrityError("Cached tutorial pipeline YAML contains no plugin nodes")
    versions = _discovered_plugin_versions()
    missing = sorted({name for _, name in roles if name not in versions})
    if missing:
        raise TutorialRunIntegrityError(f"Cached tutorial pipeline YAML references unknown plugins: {missing!r}")
    return tuple(SynthesisedNodeSpec(node_type=node_type, plugin_name=name, plugin_version=versions[name]) for node_type, name in roles)


def _plugin_nodes_from_pipeline_dict(doc: Mapping[str, Any]) -> tuple[tuple[NodeType, str], ...]:
    """Return ``(node_type, plugin_name)`` pairs in YAML order, preserving duplicates.

    Tier-1 audit topology — the ``nodes`` table — must reflect one row per
    YAML node occurrence. Deduplicating by plugin name (the prior shape) made
    plugin reuse invisible and shifted role assignment onto list-index
    inference. Duplicates are preserved here; the role is carried explicitly.

    Schema accepted (the production composer YAML shape emitted by
    ``yaml_generator.generate_pipeline_dict`` and consumed by
    ``core.config`` at ``elspeth/core/config.py:1798``):

    - ``source``: single dict with a ``plugin`` key.
    - ``transforms``: **list[dict]** of plugin entries (each with ``plugin``).
    - ``aggregations``: **list[dict]** of plugin entries (each with ``plugin``).
    - ``sinks``: dict keyed by sink name, values containing ``plugin``.

    ``gates`` and ``coalesce`` are pipeline routing primitives without plugin
    identity (yaml_generator.py:110, 159) — they cannot be faithfully
    encoded in a (NodeType, plugin_name) topology row and the tutorial
    canonical pipeline never generates them. If the cached YAML contains
    either, raise rather than emit a half-truth audit.
    """
    roles: list[tuple[NodeType, str]] = []
    source = doc["source"] if "source" in doc else None
    if type(source) is dict and "plugin" in source:
        roles.append((NodeType.SOURCE, _require_plugin_name(source["plugin"])))
    _collect_list_form_plugins(doc, "transforms", NodeType.TRANSFORM, roles)
    _collect_list_form_plugins(doc, "aggregations", NodeType.AGGREGATION, roles)
    for routing_section in ("gates", "coalesce"):
        if routing_section in doc and doc[routing_section]:
            raise TutorialRunIntegrityError(
                f"Cached tutorial pipeline YAML contains {routing_section!r} — "
                "routing primitives without plugin identity cannot be encoded "
                "into the synthesised Tier-1 audit topology; the tutorial canonical "
                "pipeline never generates them"
            )
    sinks = doc["sinks"] if "sinks" in doc else {}
    if type(sinks) is dict:
        for sink in sinks.values():
            if type(sink) is dict and "plugin" in sink:
                roles.append((NodeType.SINK, _require_plugin_name(sink["plugin"])))
    return tuple(roles)


def _plugin_nodes_from_composition_state(record: Any) -> tuple[tuple[NodeType, str], ...]:
    """Extract the plugin-node topology from a ``CompositionStateRecord``.

    Mirrors the ordering of ``yaml_generator.generate_pipeline_dict``:
    source first, then every ``transform``-typed node in ``record.nodes``
    list order, then every ``aggregation``-typed node in list order, then
    ``record.outputs`` (sinks) in list order. The sequence is the same one
    ``_plugin_nodes_from_pipeline_dict`` extracts from the rendered YAML,
    so the two are directly comparable for cache-replay state matching.

    Gate and coalesce nodes are routing primitives without plugin identity
    (consistent with the YAML parser policy) and raise rather than emit a
    half-truth topology row.

    ``record`` is typed as ``Any`` to avoid a hard L3 backward import on
    ``CompositionStateRecord`` from sessions/protocol; the attribute
    contract is enforced by ``__post_init__`` on the dataclass.
    """
    # ``CompositionStateRecord`` is a frozen dataclass whose source/nodes/
    # outputs fields are typed ``Mapping[str, Any] | None`` (or
    # ``Sequence[Mapping[str, Any]] | None``) and deep-frozen by
    # ``__post_init__``. Trust the contract — no defensive isinstance at a
    # Tier-1 internal read boundary. If the record was constructed with a
    # non-Mapping where a Mapping was promised, that is a contract violation
    # the dataclass constructor should have rejected; letting a missing-key
    # KeyError or non-subscriptable TypeError propagate here surfaces the
    # writer bug rather than masking it.
    roles: list[tuple[NodeType, str]] = []
    source = record.source
    if source is not None and "plugin" in source:
        roles.append((NodeType.SOURCE, _require_plugin_name(source["plugin"])))
    nodes = record.nodes if record.nodes is not None else ()
    transform_entries: list[Mapping[str, Any]] = []
    aggregation_entries: list[Mapping[str, Any]] = []
    for node in nodes:
        node_type = node["node_type"] if "node_type" in node else None
        if node_type == "transform":
            transform_entries.append(node)
        elif node_type == "aggregation":
            aggregation_entries.append(node)
        elif node_type in ("gate", "coalesce"):
            raise TutorialRunIntegrityError(
                f"Composition state contains {node_type!r} node — routing primitives "
                "without plugin identity cannot participate in synthesised Tier-1 "
                "audit topology comparisons"
            )
    for entry in transform_entries:
        if "plugin" in entry:
            roles.append((NodeType.TRANSFORM, _require_plugin_name(entry["plugin"])))
    for entry in aggregation_entries:
        if "plugin" in entry:
            roles.append((NodeType.AGGREGATION, _require_plugin_name(entry["plugin"])))
    outputs = record.outputs if record.outputs is not None else ()
    for output in outputs:
        if "plugin" in output:
            roles.append((NodeType.SINK, _require_plugin_name(output["plugin"])))
    return tuple(roles)


def _state_matches_cached_topology(record: Any, cached_pipeline_yaml: str) -> bool:
    """Return True when the state's plugin topology equals the cache's.

    Tier-1 audit invariant: cache replay attaches cached pipeline_yaml and
    rows to the user's current ``state_id``. If the state and cached
    pipeline describe different plugin topologies, that attachment is an
    audit lie. This check gates the replay path; mismatches fall through to
    live compose so the audit faithfully reflects what was actually run.

    Comparison is by ``(NodeType, plugin_name)`` sequence — option values,
    node names, and YAML formatting do not matter. The LLM composer is
    non-deterministic in those surface details but the canonical pipeline
    topology should be stable for the cache hit to be safe.
    """
    cached_doc = yaml.safe_load(cached_pipeline_yaml)
    if type(cached_doc) is not dict:
        raise TutorialRunIntegrityError("Cached tutorial pipeline YAML must parse to an object for topology comparison")
    cached_topology = _plugin_nodes_from_pipeline_dict(cached_doc)
    state_topology = _plugin_nodes_from_composition_state(record)
    return state_topology == cached_topology


def _collect_list_form_plugins(
    doc: Mapping[str, Any],
    section: str,
    node_type: NodeType,
    roles: list[tuple[NodeType, str]],
) -> None:
    """Append ``(node_type, plugin_name)`` for each entry in a list-form section.

    Production composer YAML emits ``transforms`` and ``aggregations`` as
    ``list[dict]`` (yaml_generator.py:86, 126). The engine config loader at
    ``core/config.py:1798, 1811`` requires list-form. Per No Legacy Code
    Policy, dict-form is rejected — it is not a production-reachable
    contract.
    """
    if section not in doc:
        return
    entries = doc[section]
    if not entries:
        return
    if type(entries) is not list:
        raise TutorialRunIntegrityError(
            f"Cached tutorial pipeline YAML {section!r} must be a list of plugin entries, got {type(entries).__name__}"
        )
    for entry in entries:
        if type(entry) is dict and "plugin" in entry:
            roles.append((node_type, _require_plugin_name(entry["plugin"])))


def _require_plugin_name(value: object) -> str:
    if type(value) is not str or not value:
        raise TutorialRunIntegrityError(f"Pipeline plugin name must be a non-empty string, got {value!r}")
    return value


@cache  # Process-scoped: plugin discovery is idempotent within a process.
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


def tutorial_model_id(settings: WebSettings) -> str:
    """Build the compound tutorial-cache identifier.

    The cache key (``SHA-256(canonical_prompt + ":" + model_id)``) invalidates
    on the operator-controlled inputs that determine the composer's choice of
    pipeline shape and transform model. Three such inputs are folded into the
    returned identifier; any one change forces a fresh live composition.

    Covered (automatic invalidation):

    1. ``settings.composer_model`` — the LLM that authors the pipeline YAML.
    2. Core composer skill markdown (``pipeline_composer.md``) shipped with
       the package — biases the composer's plugin selection and the
       transform model named inside the generated pipeline YAML.
    3. Optional deployment skill overlay at
       ``{data_dir}/skills/pipeline_composer.md`` — operator-supplied
       guidance that further shapes composer behaviour.

    Out of scope (operator clears ``{data_dir}/tutorial_cache/`` manually
    when one of these changes — same "operator deletes the artifact"
    pattern as elsewhere in the project):

    - LLM non-determinism. Same composer_model + same skill may produce a
      different ``pipeline_yaml`` on re-compose; the cache freezes whichever
      pipeline was authored first. The Tier-1 audit replay remains internally
      consistent because the cached ``pipeline_yaml`` (with its embedded
      transform model) is what the synthesised run records.
    - Plugin pack defaults (``packs/llm/defaults.yaml``) and profile YAMLs.
      These can bias the composer's transform-model choice without changing
      the three keyed inputs. Cache replay remains attribution-correct (the
      cached YAML's embedded model is what's recorded) but the canonical
      experience may be older than the operator expects until the cache is
      cleared.
    """
    _, core_skill_hash = load_skill_with_hash("pipeline_composer")
    deployment_overlay = load_deployment_skill("pipeline_composer", settings.data_dir)
    deployment_hash = hashlib.sha256(deployment_overlay.encode("utf-8")).hexdigest()
    return f"composer={settings.composer_model}|skill={core_skill_hash}|deployment_skill={deployment_hash}"


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
