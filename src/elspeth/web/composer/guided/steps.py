"""Step handlers for guided mode commit logic.

Each handler takes a resolved Step result (decided plugin + options +
observed columns) and writes it to CompositionState via the existing
freeform tools.py handlers. Handlers do NOT decide what to commit —
that's the route handler's job. They translate already-resolved data
into ToolResult calls.

Handlers thread CatalogService and data_dir from the route handler;
they never touch HTTP, session storage, or audit emission directly.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from sqlalchemy import Engine

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import BLOB_REF_PATH_PREFIX, GuidedStep
from elspeth.web.composer.guided.state_machine import (
    ChainProposal,
    GuidedSession,
    SinkOutputResolved,
    SinkResolved,
    SourceResolved,
    TerminalKind,
    TerminalState,
)
from elspeth.web.composer.source_inspection import observed_columns_from_path
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools import (
    ToolContext,
    ToolResult,
    _execute_set_output,
    _execute_set_pipeline,
    _execute_set_source,
    _sync_get_blob_by_id,
    _sync_get_blob_by_storage_path,
)
from elspeth.web.composer.yaml_generator import generate_public_yaml
from elspeth.web.paths import allowed_source_directories, resolve_data_path


@dataclass(frozen=True, slots=True)
class StepHandlerResult:
    """Result returned by each step commit handler.

    Attributes:
        state: The CompositionState after the handler ran. Unchanged if
            the underlying tool handler reports failure.
        session: The GuidedSession after the handler ran. Unchanged if
            the underlying tool handler reports failure.
        tool_result: The raw ToolResult from the underlying tool handler.
            Callers inspect tool_result.success to decide whether to
            advance the session step or re-emit the inspect turn.
    """

    state: CompositionState
    session: GuidedSession
    tool_result: ToolResult


def _resolve_blob_ref_path(
    resolved: SourceResolved,
    *,
    session_engine: Engine | None,
    session_id: str | None,
) -> SourceResolved:
    """Re-resolve a masked ``blob:<ref>`` source path to the real storage_path.

    ``build_step_1_schema_form_turn_from_resolved`` renders a blob-backed
    source's ``path`` knob as ``blob:<blob_ref>`` so the absolute storage_path
    (deploy dir + OS username) never reaches the wire. On commit we must restore
    the real path or the pipeline cannot read the blob. The lookup is
    authoritative (by-id DB query, session-scoped), mirroring the storage_path
    enrichment below.

    A non-sentinel path — an operator-typed path, or a first chat-resolve commit
    that already carries the real path — is returned unchanged.
    """
    path_value = resolved.options.get("path")
    if not (isinstance(path_value, str) and path_value.startswith(BLOB_REF_PATH_PREFIX)):
        return resolved
    blob_ref = path_value[len(BLOB_REF_PATH_PREFIX) :]
    if session_engine is None or session_id is None:
        raise InvariantError(
            "handle_step_1_source: a blob:<ref> source path requires session_engine and session_id to resolve, but they were not provided."
        )
    blob = _sync_get_blob_by_id(session_engine, blob_ref, session_id)
    if blob is None:
        raise InvariantError(f"handle_step_1_source: blob:<ref> path references blob {blob_ref!r}, which no longer exists in this session.")
    restored = {**dict(resolved.options), "path": blob["storage_path"]}
    return dataclasses.replace(resolved, options=restored)


def handle_step_1_source(
    *,
    state: CompositionState,
    session: GuidedSession,
    resolved: SourceResolved,
    catalog: CatalogService,
    data_dir: str | None = None,
    session_engine: Engine | None = None,
    session_id: str | None = None,
) -> StepHandlerResult:
    """Commit *resolved* as the pipeline source via _execute_set_source.

    Constructs the args dict the freeform handler expects and delegates
    entirely to _execute_set_source. The handler does not validate or
    coerce — validation is the tool handler's responsibility.

    Returns a StepHandlerResult with:
    - state, session unchanged if the underlying handler reports failure
      (the route layer turns that into a re-emit of the inspect turn).
    - state advanced, session.step_1_result=resolved on success.

    The session step pointer is NOT advanced here — that is the
    dispatcher's (route handler's) responsibility in Task 3.4.

    blob_ref enrichment
    ~~~~~~~~~~~~~~~~~~~
    If the source options contain a ``path`` key and ``session_engine`` /
    ``session_id`` are provided, we look up the blob by storage_path.  When
    a blob row is found the blob UUID is injected as ``blob_ref`` into the
    stored ``step_1_result.options`` (the ``SourceResolved`` snapshot).

    This records the blob UUID on ``source.options["blob_ref"]`` even when the
    operator supplied the path via the guided SchemaForm rather than the
    ``set_source_from_blob`` tool. The lookup is authoritative (DB query) rather
    than path-parsing, so it cannot be fooled by paths that coincidentally look
    like blob paths but aren't registered blobs.

    If no matching blob is found (path-only source, not blob-backed), the
    enrichment is silently skipped, which is the correct behavior for non-blob
    sources.
    """
    # Restore the real storage_path from a masked blob:<ref> path (emitted to keep
    # the absolute path off the wire) before committing, so the pipeline can read
    # the blob. No-op for non-sentinel paths.
    resolved = _resolve_blob_ref_path(resolved, session_engine=session_engine, session_id=session_id)
    source_path = resolved.options.get("path")
    blob: Mapping[str, Any] | None = None
    if source_path is not None and session_engine is not None and session_id is not None:
        blob = _sync_get_blob_by_storage_path(session_engine, str(source_path), session_id)

    observed_columns = resolved.observed_columns
    schema_columns: tuple[str, ...] = ()
    if not observed_columns and blob is not None:
        observed_columns = _observed_columns_from_blob(blob)
        schema_columns = observed_columns
    if not observed_columns:
        observed_columns = _observed_columns_from_allowed_path(resolved.options, data_dir)
        schema_columns = observed_columns

    commit_options = _source_options_with_guaranteed_fields(resolved.options, schema_columns)
    commit_resolved = dataclasses.replace(resolved, options=commit_options, observed_columns=observed_columns)

    args = {
        "plugin": commit_resolved.plugin,
        "options": dict(commit_resolved.options),
        "on_success": "main",
        # The composer (chat resolution) owns this routing choice; commit what it
        # picked. Manual / schema_form-submission resolved values carry the
        # "discard" default, so this remains "discard" for those paths.
        "on_validation_failure": commit_resolved.on_validation_failure,
    }

    tool_result = _execute_set_source(args, state, ToolContext(catalog=catalog, data_dir=data_dir))

    if not tool_result.success:
        return StepHandlerResult(
            state=state,
            session=session,
            tool_result=tool_result,
        )

    # Attempt to enrich step_1_result with blob_ref when the source path
    # points to an uploaded blob.  This is the authoritative lookup —
    # we query by storage_path rather than parsing the filename.
    enriched_resolved = commit_resolved
    if blob is not None:
        # Inject blob_ref into the SourceResolved snapshot as an authoritative
        # overlay. The original options are Tier-3 (user-submitted SchemaForm
        # values); we add blob_ref from our own DB (Tier 1 source), so no .get()
        # or coercion needed here.
        enriched_options: dict[str, Any] = {**dict(commit_resolved.options), "blob_ref": blob["id"]}
        enriched_resolved = dataclasses.replace(commit_resolved, options=enriched_options)

    return StepHandlerResult(
        state=tool_result.updated_state,
        session=dataclasses.replace(session, step_1_result=enriched_resolved),
        tool_result=tool_result,
    )


def _observed_columns_from_allowed_path(options: Mapping[str, Any], data_dir: str | None) -> tuple[str, ...]:
    """Derive observed columns from a committed local source path, or ``()``.

    Validation errors remain owned by ``_execute_set_source``. This enrichment
    mirrors the source path allowlist before reading anything, then degrades
    unreadable files to no enrichment.
    """
    if data_dir is None:
        return ()
    raw_path = options.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return ()
    try:
        resolved_path = resolve_data_path(raw_path, data_dir)
    except (OSError, RuntimeError, ValueError):
        return ()
    if not any(resolved_path.is_relative_to(directory) for directory in allowed_source_directories(data_dir)):
        return ()
    return observed_columns_from_path(
        path=resolved_path,
        filename=resolved_path.name,
        mime_type="",
    )


def _source_options_with_guaranteed_fields(options: Mapping[str, Any], observed_columns: tuple[str, ...]) -> dict[str, object]:
    """Publish observed source fields into the committed schema contract."""
    enriched: dict[str, object] = dict(options)
    if not observed_columns:
        return enriched
    raw_schema = enriched.get("schema")
    if not isinstance(raw_schema, Mapping):
        return enriched
    schema: dict[str, object] = dict(cast(Mapping[str, object], raw_schema))
    if not schema.get("guaranteed_fields"):
        schema["guaranteed_fields"] = list(observed_columns)
    enriched["schema"] = schema
    return enriched


def _observed_columns_from_blob(blob: Mapping[str, Any]) -> tuple[str, ...]:
    """Derive observed column names from a registered blob's content, or ``()``.

    Reads the blob file at its ``storage_path`` and runs the bounded source
    inspector. A missing/unreadable file degrades to ``()`` (the caller then
    keeps whatever columns it had) rather than failing the commit — column
    backfill is best-effort enrichment, never a gate on committing the source.
    """
    storage_path = blob["storage_path"]
    filename = blob["filename"]
    mime_type = blob["mime_type"]
    if type(storage_path) is not str or not storage_path:
        raise InvariantError("handle_step_1_source: blob.storage_path must be a non-empty string")
    if type(filename) is not str:
        raise InvariantError("handle_step_1_source: blob.filename must be a string")
    if type(mime_type) is not str:
        raise InvariantError("handle_step_1_source: blob.mime_type must be a string")
    # observed_columns_from_path reads only the bounded prefix the inspector
    # actually scans (the whole-file read here could allocate hundreds of MB for
    # a large blob just to recover a header) and degrades a missing/unreadable
    # file to () itself — so no separate existence pre-check is needed.
    return observed_columns_from_path(
        path=Path(storage_path),
        filename=filename,
        mime_type=mime_type,
    )


def _sink_options_with_step_2_schema_contract(output: SinkOutputResolved) -> dict[str, Any]:
    """Merge Step-2 field selection into the sink's schema config."""
    options = dict(output.options)
    schema_key = "schema" if "schema" in options else "schema_config" if "schema_config" in options else "schema"
    if schema_key in options:
        raw_schema = options[schema_key]
        if type(raw_schema) not in (dict, MappingProxyType):
            raise InvariantError("handle_step_2_sink: sink schema options must be dict-shaped when present")
        schema = dict(cast(Mapping[str, Any], raw_schema))
    else:
        schema = {}
    schema["mode"] = output.schema_mode
    schema["required_fields"] = list(output.required_fields)
    if "schema" in options:
        del options["schema"]
    if "schema_config" in options:
        del options["schema_config"]
    options["schema"] = schema
    return options


def handle_step_2_sink(
    *,
    state: CompositionState,
    session: GuidedSession,
    resolved: SinkResolved,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> StepHandlerResult:
    """Commit each resolved sink output via _execute_set_output.

    Constructs the args dict the freeform handler expects, translating the
    guided field-selection decision into ``options.schema`` before delegating
    to _execute_set_output. The handler does not validate or coerce —
    validation is the tool handler's responsibility.

    MVP constraint: all outputs use sink_name="main" to match the
    source's on_success="main" wiring set in handle_step_1_source.
    Multi-output pipelines will collide on "main" on the second iteration
    — that is the correct behaviour; multi-output support is deferred to
    the Phase 4 chain solver.

    Returns a StepHandlerResult with:
    - state, session unchanged (entry state) if ANY underlying handler
      reports failure — partial commits are NOT retained.
    - state advanced, session.step_2_result=resolved on success of all outputs.

    Raises:
        InvariantError: if resolved.outputs is empty. The dispatcher upstream
            guarantees at least one output.
    """
    if not resolved.outputs:
        raise InvariantError("step 2 sink resolved with no outputs — handler refuses empty list")

    entry_state = state
    current_state = state
    last_result: ToolResult | None = None

    for output in resolved.outputs:
        args = {
            "plugin": output.plugin,
            "sink_name": "main",
            "options": _sink_options_with_step_2_schema_contract(output),
            "on_write_failure": "discard",
        }
        tool_result = _execute_set_output(args, current_state, ToolContext(catalog=catalog, data_dir=data_dir))
        if not tool_result.success:
            return StepHandlerResult(
                state=entry_state,
                session=session,
                tool_result=tool_result,
            )
        current_state = tool_result.updated_state
        last_result = tool_result

    if last_result is None:
        # The empty-outputs check at the top of this function raises before the
        # loop; therefore reaching this point with last_result=None means the
        # loop body never ran despite resolved.outputs being non-empty — a bug
        # in iteration logic that would otherwise silently feed None to the
        # StepHandlerResult dataclass constructor.  Use InvariantError (not a
        # bare assert) so python -O does not strip the gate.
        raise InvariantError(
            "step 2 sink handler loop completed with last_result=None despite "
            "non-empty resolved.outputs — bug in the empty-outputs guard above"
        )

    return StepHandlerResult(
        state=current_state,
        session=dataclasses.replace(session, step_2_result=resolved),
        tool_result=last_result,
    )


def handle_step_3_chain_accept(
    *,
    state: CompositionState,
    session: GuidedSession,
    proposal: ChainProposal,
    catalog: CatalogService,
    data_dir: str | None = None,
    session_engine: Engine | None = None,
    session_id: str | None = None,
) -> StepHandlerResult:
    """Commit *proposal* atomically via _execute_set_pipeline and redirect to wire.

    Reconstructs the full pipeline spec from the existing single guided source
    + state.outputs and the new transforms from the proposal.
    When the proposal has transforms, Source.on_success is rewired to
    "chain_in" so the chain sits between source and sinks; the last transform
    produces "main" so outputs (which were committed against the source's
    original "main" label in Step 2) remain reachable. An empty proposal is a
    valid pass-through: source continues to produce "main" directly.

    Wiring for N transforms:
        source.on_success="chain_in"
        idx=0:   input="chain_in", on_success="chain_0" (or "main" if N==1)
        idx=k:   input=f"chain_{k-1}", on_success=f"chain_{k}"   (0<k<N-1)
        idx=N-1: input=f"chain_{N-2}", on_success="main"          (N>1)
        outputs: unchanged — still consume "main"

    On _execute_set_pipeline success the session moves to STEP_4_WIRE and
    step_3_proposal is recorded; the wire confirm handler owns the COMPLETED
    terminal stamp. On failure the state and session are unchanged and the
    route layer re-emits the propose_chain turn with the validation errors.

    Args:
        state: Current composition state. MUST have a committed source and
            at least one output (dispatcher invariant — Steps 1 and 2 must
            have completed before Step 3).
        session: Current guided session.
        proposal: Chain proposal with zero or more transform steps.
        catalog: Plugin catalogue service.
        data_dir: Optional data directory for blob path validation.
        session_engine: Optional session DB engine (forwarded to
            _execute_set_pipeline for inline-blob and source-blob support).
        session_id: Optional session ID (forwarded with session_engine).

    Raises:
        InvariantError: if state has no source or no outputs (handler invariant
            violation — dispatcher guarantees Step 1 and Step 2 have committed
            before reaching Step 3).
    """
    if len(state.sources) != 1:
        raise InvariantError(f"step 3 requires exactly one committed source, got {len(state.sources)}; dispatcher bug")
    if not state.outputs:
        raise InvariantError("step 3 reached without committed outputs; dispatcher bug")
    source_name, source = next(iter(state.sources.items()))

    n = len(proposal.steps)
    node_args: list[dict[str, Any]] = []
    for idx, step in enumerate(proposal.steps):
        input_label = "chain_in" if idx == 0 else f"chain_{idx - 1}"
        on_success_label = "main" if idx == n - 1 else f"chain_{idx}"
        options = dict(step["options"])
        node_args.append(
            {
                "id": f"guided_xform_{idx}",
                "node_type": "transform",
                "plugin": step["plugin"],
                "input": input_label,
                "on_success": on_success_label,
                "options": options,
            }
        )

    arguments: dict[str, Any] = {
        "sources": {
            source_name: {
                "plugin": source.plugin,
                "on_success": "chain_in" if node_args else "main",
                "options": dict(source.options),
                "on_validation_failure": source.on_validation_failure,
            }
        },
        "nodes": node_args,
        "edges": [],
        "outputs": [
            {
                "sink_name": o.name,
                "plugin": o.plugin,
                "options": dict(o.options),
                "on_write_failure": o.on_write_failure,
            }
            for o in state.outputs
        ],
        "metadata": {},
    }

    tool_result = _execute_set_pipeline(
        arguments,
        state,
        ToolContext(
            catalog=catalog,
            data_dir=data_dir,
            session_engine=session_engine,
            session_id=session_id,
        ),
    )

    if not tool_result.success:
        return StepHandlerResult(
            state=state,
            session=session,
            tool_result=tool_result,
        )

    new_session = dataclasses.replace(
        session,
        step=GuidedStep.STEP_4_WIRE,
        step_3_proposal=proposal,
        terminal=None,
    )

    return StepHandlerResult(
        state=tool_result.updated_state,
        session=new_session,
        tool_result=tool_result,
    )


def handle_step_4_wire_confirm(
    *,
    state: CompositionState,
    session: GuidedSession,
) -> StepHandlerResult:
    """Confirm wiring and stamp COMPLETED only when validation is clean.

    This is the terminal-stamp gate for guided mode. Recipe apply and chain
    accept already commit the pipeline and move the session to STEP_4_WIRE; this
    handler re-runs validation and either stamps the completed terminal with
    rendered YAML or leaves the session open for another wire-stage turn.
    """
    validation = state.validate()
    tool_result = ToolResult(
        success=validation.is_valid,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data=None,
    )
    if not validation.is_valid:
        return StepHandlerResult(state=state, session=session, tool_result=tool_result)

    yaml_text = generate_public_yaml(state)
    terminal = TerminalState(
        kind=TerminalKind.COMPLETED,
        reason=None,
        pipeline_yaml=yaml_text,
    )
    new_session = dataclasses.replace(session, terminal=terminal)
    return StepHandlerResult(state=state, session=new_session, tool_result=tool_result)
