"""Step handlers for guided mode commit logic.

Each handler takes a resolved Step result (decided plugin + options +
observed columns) and writes it to CompositionState via the existing
freeform tools.py handlers. Handlers do NOT decide what to commit —
that's the route handler's job. They translate already-resolved data
into ToolResult calls.

Handlers thread a request policy catalog, snapshot, and data_dir from the route handler;
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

from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.profile import TUTORIAL_PROFILE
from elspeth.web.composer.guided.protocol import BLOB_REF_PATH_PREFIX, GuidedStep
from elspeth.web.composer.guided.state_machine import (
    ChainProposal,
    GuidedSession,
    SinkOutputResolved,
    SinkResolved,
    SourceResolved,
)
from elspeth.web.composer.source_inspection import observed_columns_from_path
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools import (
    ToolContext,
    ToolResult,
    _execute_set_output,
    _execute_set_pipeline,
    _execute_set_source,
    _failure_result,
    _sync_get_blob_by_id,
    _sync_get_blob_by_storage_path,
)
from elspeth.web.interpretation_state import AUTHORING_METADATA_OPTION_KEYS
from elspeth.web.paths import allowed_source_directories, resolve_data_path
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId

_PROFILE_AUTHORING_METADATA_OPTION_KEYS = AUTHORING_METADATA_OPTION_KEYS | {"resolved_prompt_template_hash"}


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


def _normalize_profiled_options(
    catalog: PolicyCatalogView,
    plugin_id: PluginId,
    options: Mapping[str, Any],
    *,
    profile: object,
) -> dict[str, Any]:
    """Keep only public options plus audit metadata for an operator-profiled plugin.

    The guided solver is untrusted and can emit legacy executable binding
    fields even though the policy catalog exposes only an opaque profile
    alias. Those provider/model/credential values are operator-owned: discard
    them at the commit seam and let the profile resolver supply the executable
    binding later. Unknown non-public fields are discarded for the same
    reason, while interpretation metadata remains attached to the authored
    node for its review gates.
    """
    public_schema = catalog.get_schema(plugin_id.kind, plugin_id.name).json_schema
    properties = public_schema.get("properties")
    if not isinstance(properties, Mapping) or "profile" not in properties:
        raise InvariantError(f"selected operator profile has no public profile schema for plugin {plugin_id}")
    allowed = set(properties) | _PROFILE_AUTHORING_METADATA_OPTION_KEYS
    normalized = {name: value for name, value in options.items() if name in allowed}
    normalized["profile"] = profile
    return normalized


@trust_boundary(
    tier=3,
    source="committed SourceResolved carrying a web-authored path option (possibly a blob:<ref> sentinel)",
    source_param="resolved",
    suppresses=("R1", "R5"),
    invariant=(
        "non-sentinel or absent paths pass through unchanged; a blob:<ref> sentinel that cannot "
        "be resolved (no session engine/id, or the blob no longer exists) raises InvariantError"
    ),
    test_ref="tests/unit/web/composer/guided/test_prefill_from_resolved.py::test_resolve_blob_ref_path_raises_on_unresolvable_blob_ref",
    test_fingerprint="8c5df0aeb31e64a24ee183b6a97722f0c91b3d76a0ed4ad36282aed5c535353f",
)
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


@trust_boundary(
    tier=3,
    source="committed SourceResolved carrying web-authored options (untrusted path value)",
    source_param="resolved",
    suppresses=("R1",),
    invariant=(
        "an absent path option means no blob enrichment (commit proceeds unchanged); "
        "malformed blob:<ref> sentinel paths are rejected upstream by _resolve_blob_ref_path, "
        "which carries its own boundary; this function's own reads never raise"
    ),
    non_raising=True,
)
def handle_step_1_source(
    *,
    state: CompositionState,
    session: GuidedSession,
    resolved: SourceResolved,
    catalog: PolicyCatalogView,
    plugin_snapshot: PluginAvailabilitySnapshot,
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

    tool_result = _execute_set_source(
        args,
        state,
        ToolContext(catalog=catalog, plugin_snapshot=plugin_snapshot, data_dir=data_dir),
    )

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


@trust_boundary(
    tier=3,
    source="web-authored guided source options (untrusted path value)",
    source_param="options",
    suppresses=("R1", "R5"),
    invariant=("returns () for any missing, mistyped, unresolvable, or non-allowlisted path; enrichment degrades, never raises"),
    non_raising=True,
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


@trust_boundary(
    tier=3,
    source="web-authored guided source options (untrusted schema mapping)",
    source_param="options",
    suppresses=("R1", "R5"),
    invariant=(
        "returns the options unchanged when the schema is absent or not a mapping; "
        "only a well-formed schema without guaranteed_fields is enriched; never raises"
    ),
    non_raising=True,
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
    catalog: PolicyCatalogView,
    plugin_snapshot: PluginAvailabilitySnapshot,
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
        tool_result = _execute_set_output(
            args,
            current_state,
            ToolContext(catalog=catalog, plugin_snapshot=plugin_snapshot, data_dir=data_dir),
        )
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


_ROW_PRESERVING_ASSIGNMENT_PLUGINS: frozenset[str] = frozenset({"value_transform", "passthrough", "truncate"})
"""Transforms that emit every input row: assignment/pass-through only.

A rationale claiming these plugins filter, keep, drop, or route rows is a
false capability claim (2026-07-10 web eval, elspeth-c1d78dac70: the solver
proposed ``value_transform`` writing a ``_keep`` boolean and told the user
False rows would error-route out — all rows reached the sink plus a leaked
helper column). ``type_coerce`` and ``keyword_filter`` are deliberately
absent: both genuinely error-route rows (failed conversions / matched
patterns), so a routing claim in their rationale can be true.
"""

_ROW_FILTER_CLAIM_PHRASES: tuple[str, ...] = (
    "keep only",
    "filter",
    "error-rout",
    "error rout",
    "route out",
    "routes out",
    "routed out",
)
"""Substrings that assert row filtering regardless of surrounding context."""

_ROW_FILTER_VERB_STEMS: tuple[str, ...] = ("drop", "remov", "discard", "exclud", "skip", "reject", "block")
"""Verb stems that only read as row filtering when they sit near "row"."""

_ROW_VERB_WINDOW_CHARS: int = 24
"""How far (chars) before a "row" occurrence a verb stem counts as adjacent.

Wide enough for "drops every row" / "excluding the rows", narrow enough that
"Remove the currency symbol from the price field of each row" (a field edit)
stays clean.
"""


def _row_filter_claim_marker(rationale: str) -> str | None:
    """Return the matched row-filtering claim marker in *rationale*, or None.

    Pure substring scan (no regex) so this module adds no imports: the
    tier-model allowlist binds a signed entry in this file to its module-level
    ast_path, and a new import statement would shift every top-level index.
    """
    text = rationale.lower()
    for phrase in _ROW_FILTER_CLAIM_PHRASES:
        if phrase in text:
            return phrase
    # Ambiguous verbs legitimately describe FIELD edits ("remove the currency
    # symbol"); they only assert row filtering next to "row"/"rows". Scan the
    # window preceding each "row" occurrence for a verb stem.
    pos = text.find("row")
    while pos != -1:
        window = text[max(0, pos - _ROW_VERB_WINDOW_CHARS) : pos]
        for stem in _ROW_FILTER_VERB_STEMS:
            if stem in window:
                return f"{stem} … row"
        pos = text.find("row", pos + 1)
    return None


def _row_filter_claim_error(step_index: int, plugin: str, marker: str) -> str:
    """Rejection text for a false row-filter claim on an assignment-only step.

    Travels verbatim into ``validation.errors`` (via ``_failure_result``), so
    the accept-failure repair loop feeds exactly this coaching back to the
    solver as ``repair_context``.
    """
    return (
        f"Step {step_index + 1} ('{plugin}'): the rationale claims row filtering ('{marker}'), "
        f"but {plugin} is assignment-only — every row passes through, and an expression that "
        "evaluates to False does not drop or error-route the row; it just stores False and "
        "leaks the helper column into the output. Conditional row filtering is a gate node, "
        "and guided chains cannot include gates. Re-propose without emulating a filter: keep "
        "the honestly-buildable steps, and state plainly in `why` that the conditional row "
        "filter is not included and must be added after the guided build completes (the "
        "composer chat can add a gate). Only keyword_filter genuinely blocks rows, and only "
        "by regex match on string fields."
    )


def handle_step_3_chain_accept(
    *,
    state: CompositionState,
    session: GuidedSession,
    proposal: ChainProposal,
    catalog: PolicyCatalogView,
    plugin_snapshot: PluginAvailabilitySnapshot,
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

    # Reject false row-filter claims BEFORE committing: an assignment-only
    # transform ships every row, so a chain sold to the user as a filter
    # silently doesn't filter. The failure rides the normal accept-failure
    # path — the route reads validation.errors and re-solves with this text
    # as repair_context. ``str(step["rationale"])`` mirrors the emitter
    # (emitters.py build_step_3_propose_chain_turn), which has already
    # direct-accessed these keys to render the proposal the user accepted.
    for lint_idx, lint_step in enumerate(proposal.steps):
        lint_plugin = str(lint_step["plugin"])
        if lint_plugin not in _ROW_PRESERVING_ASSIGNMENT_PLUGINS:
            continue
        marker = _row_filter_claim_marker(str(lint_step["rationale"]))
        if marker is not None:
            return StepHandlerResult(
                state=state,
                session=session,
                tool_result=_failure_result(state, _row_filter_claim_error(lint_idx, lint_plugin, marker)),
            )

    n = len(proposal.steps)
    node_args: list[dict[str, Any]] = []
    selected_profiles = dict(plugin_snapshot.selected_profile_aliases)
    for idx, step in enumerate(proposal.steps):
        input_label = "chain_in" if idx == 0 else f"chain_{idx - 1}"
        on_success_label = "main" if idx == n - 1 else f"chain_{idx}"
        options = dict(step["options"])
        plugin_id = PluginId("transform", str(step["plugin"]))
        selected_profile = selected_profiles.get(plugin_id)
        if selected_profile is not None:
            # Provider bindings are operator-owned. The guided solver may omit
            # the opaque alias entirely; and the tutorial must always use the
            # specifically configured tutorial profile rather than an
            # arbitrary usable alternative. Live guided sessions retain an
            # explicit allowed choice but receive the operator-selected default
            # when the model did not choose one.
            profile = selected_profile if session.profile == TUTORIAL_PROFILE or "profile" not in options else options["profile"]
            options = _normalize_profiled_options(catalog, plugin_id, options, profile=profile)
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
            plugin_snapshot=plugin_snapshot,
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
