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
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.recipe_match import RecipeMatch
from elspeth.web.composer.guided.state_machine import (
    ChainProposal,
    GuidedSession,
    SinkResolved,
    SourceResolved,
    TerminalKind,
    TerminalState,
)
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools import (
    ToolContext,
    ToolResult,
    _execute_apply_pipeline_recipe,
    _execute_set_output,
    _execute_set_pipeline,
    _execute_set_source,
    _sync_get_blob_by_storage_path,
)
from elspeth.web.composer.yaml_generator import generate_yaml


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

    This lets the recipe slot resolvers in ``recipe_match.py`` read
    ``source.options["blob_ref"]`` even when the operator supplied the path
    via the guided SchemaForm rather than the ``set_source_from_blob`` tool.
    The lookup is authoritative (DB query) rather than path-parsing, so it
    cannot be fooled by paths that coincidentally look like blob paths but
    aren't registered blobs.

    If no matching blob is found (path-only source, not blob-backed), the
    enrichment is silently skipped — recipe matching will not populate
    ``source_blob_id`` and the recipe offer is omitted, which is the correct
    behavior for non-blob sources.
    """
    args = {
        "plugin": resolved.plugin,
        "options": dict(resolved.options),
        "on_success": "main",
        "on_validation_failure": "discard",
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
    enriched_resolved = resolved
    source_path = resolved.options.get("path")
    if source_path is not None and session_engine is not None and session_id is not None:
        blob = _sync_get_blob_by_storage_path(session_engine, str(source_path), session_id)
        if blob is not None:
            # Inject blob_ref into the SourceResolved snapshot so that
            # _classify_slot_resolver (recipe_match.py) can read it.
            # The original options are Tier-3 (user-submitted SchemaForm
            # values); we add blob_ref as an authoritative overlay from our
            # own DB (Tier 1 source), so no .get() or coercion needed here.
            enriched_options: dict[str, Any] = {**dict(resolved.options), "blob_ref": blob["id"]}
            enriched_resolved = dataclasses.replace(resolved, options=enriched_options)

    return StepHandlerResult(
        state=tool_result.updated_state,
        session=dataclasses.replace(session, step_1_result=enriched_resolved),
        tool_result=tool_result,
    )


def handle_step_2_sink(
    *,
    state: CompositionState,
    session: GuidedSession,
    resolved: SinkResolved,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> StepHandlerResult:
    """Commit each resolved sink output via _execute_set_output.

    Constructs the args dict the freeform handler expects and delegates
    entirely to _execute_set_output. The handler does not validate or
    coerce — validation is the tool handler's responsibility.

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
            "options": dict(output.options),
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


def handle_step_2_5_recipe_apply(
    *,
    state: CompositionState,
    session: GuidedSession,
    match: RecipeMatch,
    catalog: CatalogService,
    data_dir: str | None = None,
    session_engine: Engine | None = None,
    session_id: str | None = None,
) -> StepHandlerResult:
    """Apply the matched recipe and mark the session COMPLETED.

    On success the returned state is the recipe-applied pipeline,
    session.terminal is TerminalState(COMPLETED, reason=None,
    pipeline_yaml=<rendered>), and tool_result is the canonical
    ToolResult from _execute_apply_pipeline_recipe.

    On failure the state and session are unchanged, tool_result
    reflects the failure; the route layer will re-emit the recipe-
    offer turn with the validation errors attached.
    """
    arguments: dict[str, object] = {
        "recipe_name": match.recipe_name,
        "slots": dict(match.slots),
    }

    tool_result = _execute_apply_pipeline_recipe(
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

    yaml_text = generate_yaml(tool_result.updated_state)
    terminal = TerminalState(
        kind=TerminalKind.COMPLETED,
        reason=None,
        pipeline_yaml=yaml_text,
    )
    new_session = dataclasses.replace(session, terminal=terminal)

    return StepHandlerResult(
        state=tool_result.updated_state,
        session=new_session,
        tool_result=tool_result,
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
    """Commit *proposal* atomically via _execute_set_pipeline and terminate the session.

    Reconstructs the full pipeline spec from the existing state.source +
    state.outputs and the new transforms from the proposal. Source.on_success
    is rewired to "chain_in" so the chain sits between source and sinks; the
    last transform produces "main" so outputs (which were committed against
    the source's original "main" label in Step 2) remain reachable.

    Wiring for N transforms:
        source.on_success="chain_in"
        idx=0:   input="chain_in", on_success="chain_0" (or "main" if N==1)
        idx=k:   input=f"chain_{k-1}", on_success=f"chain_{k}"   (0<k<N-1)
        idx=N-1: input=f"chain_{N-2}", on_success="main"          (N>1)
        outputs: unchanged — still consume "main"

    On _execute_set_pipeline success the session terminal is COMPLETED with
    rendered YAML and step_3_proposal is recorded. On failure the state and
    session are unchanged and the route layer re-emits the propose_chain turn
    with the validation errors.

    Args:
        state: Current composition state. MUST have a committed source and
            at least one output (dispatcher invariant — Steps 1 and 2 must
            have completed before Step 3).
        session: Current guided session.
        proposal: Chain proposal with one or more transform steps.
        catalog: Plugin catalogue service.
        data_dir: Optional data directory for blob path validation.
        session_engine: Optional session DB engine (forwarded to
            _execute_set_pipeline for inline-blob and source-blob support).
        session_id: Optional session ID (forwarded with session_engine).

    Raises:
        InvariantError: if the proposal has zero steps, or if state has no source
            or no outputs (handler invariant violation — dispatcher guarantees
            Step 1 and Step 2 have committed before reaching Step 3).
    """
    if not proposal.steps:
        raise InvariantError("step 3 proposal had zero steps; refusing empty commit")
    if state.source is None:
        raise InvariantError("step 3 reached without a committed source; dispatcher bug")
    if not state.outputs:
        raise InvariantError("step 3 reached without committed outputs; dispatcher bug")

    n = len(proposal.steps)
    node_args: list[dict[str, Any]] = []
    for idx, step in enumerate(proposal.steps):
        input_label = "chain_in" if idx == 0 else f"chain_{idx - 1}"
        on_success_label = "main" if idx == n - 1 else f"chain_{idx}"
        node_args.append(
            {
                "id": f"guided_xform_{idx}",
                "node_type": "transform",
                "plugin": step["plugin"],
                "input": input_label,
                "on_success": on_success_label,
                "options": dict(step["options"]),
            }
        )

    arguments: dict[str, Any] = {
        "source": {
            "plugin": state.source.plugin,
            "on_success": "chain_in",  # rewired; was "main" pre-Step-3
            "options": dict(state.source.options),
            "on_validation_failure": state.source.on_validation_failure,
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

    yaml_text = generate_yaml(tool_result.updated_state)
    terminal = TerminalState(
        kind=TerminalKind.COMPLETED,
        reason=None,
        pipeline_yaml=yaml_text,
    )
    new_session = dataclasses.replace(
        session,
        step_3_proposal=proposal,
        terminal=terminal,
    )

    return StepHandlerResult(
        state=tool_result.updated_state,
        session=new_session,
        tool_result=tool_result,
    )
