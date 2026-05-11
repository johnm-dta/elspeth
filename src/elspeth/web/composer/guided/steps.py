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

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.guided.state_machine import GuidedSession, SinkResolved, SourceResolved
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools import ToolResult, _execute_set_output, _execute_set_source


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
    """
    args = {
        "plugin": resolved.plugin,
        "options": dict(resolved.options),
        "on_success": "main",
        "on_validation_failure": "discard",
    }

    tool_result = _execute_set_source(args, state, catalog, data_dir)

    if not tool_result.success:
        return StepHandlerResult(
            state=state,
            session=session,
            tool_result=tool_result,
        )

    return StepHandlerResult(
        state=tool_result.updated_state,
        session=dataclasses.replace(session, step_1_result=resolved),
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
        ValueError: if resolved.outputs is empty. The dispatcher upstream
            guarantees at least one output.
    """
    if not resolved.outputs:
        raise ValueError("step 2 sink resolved with no outputs — handler refuses empty list")

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
        tool_result = _execute_set_output(args, current_state, catalog, data_dir)
        if not tool_result.success:
            return StepHandlerResult(
                state=entry_state,
                session=session,
                tool_result=tool_result,
            )
        current_state = tool_result.updated_state
        last_result = tool_result

    assert last_result is not None  # guaranteed: loop ran ≥1 time (empty check above)

    return StepHandlerResult(
        state=current_state,
        session=dataclasses.replace(session, step_2_result=resolved),
        tool_result=last_result,
    )
