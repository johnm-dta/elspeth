"""Guided-mode state-machine data: GuidedSession, TerminalState, TurnRecord.

See docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md §5.

Trust tier: Tier 1 (audit). Coercion forbidden — every field crashes on
malformed input. The freeze_fields contract applies because these structures
are persisted and re-read across the audit trail.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from elspeth.contracts.freeze import freeze_fields
from elspeth.web.composer.guided.protocol import GuidedStep, Turn, TurnResponse, TurnType


class TerminalKind(StrEnum):
    COMPLETED = "completed"
    EXITED_TO_FREEFORM = "exited_to_freeform"


class TerminalReason(StrEnum):
    USER_PRESSED_EXIT = "user_pressed_exit"
    PROTOCOL_VIOLATION = "protocol_violation"
    SOLVER_EXHAUSTED = "solver_exhausted"


@dataclass(frozen=True, slots=True)
class TerminalState:
    """Outcome of a guided session.

    `reason` is None when `kind == COMPLETED`; required when
    `kind == EXITED_TO_FREEFORM`. `pipeline_yaml` is set only on COMPLETED.
    Callers must construct consistently — invariants enforced by step_advance().
    """

    kind: TerminalKind
    reason: TerminalReason | None
    pipeline_yaml: str | None


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One emitted turn + its (optional) user response, recorded for audit."""

    step: GuidedStep
    turn_type: TurnType
    payload_hash: str
    response_hash: str | None
    emitter: str  # "server" | "llm"


@dataclass(frozen=True, slots=True)
class SourceResolved:
    """Source plugin state after Step 1.

    Frozen, audit-tier. Stored in GuidedSession.step_1_result.
    """

    plugin: str
    options: Mapping[str, Any]
    observed_columns: Sequence[str]
    sample_rows: Sequence[Mapping[str, Any]]

    def __post_init__(self) -> None:
        freeze_fields(self, "options", "observed_columns", "sample_rows")


@dataclass(frozen=True, slots=True)
class SinkOutputResolved:
    """A single sink output after Step 2.

    Frozen, audit-tier. Part of SinkResolved.outputs sequence.
    """

    plugin: str
    options: Mapping[str, Any]
    required_fields: Sequence[str]
    schema_mode: str  # "fixed" | "flexible" | "observed"

    def __post_init__(self) -> None:
        freeze_fields(self, "options", "required_fields")


@dataclass(frozen=True, slots=True)
class SinkResolved:
    """Sink configuration after Step 2.

    Frozen, audit-tier. Stored in GuidedSession.step_2_result.
    """

    outputs: Sequence[SinkOutputResolved]

    def __post_init__(self) -> None:
        freeze_fields(self, "outputs")


@dataclass(frozen=True, slots=True)
class ChainProposal:
    """A transform chain proposed by Step 3 LLM.

    Frozen, audit-tier. Stored in GuidedSession.step_3_proposal.
    """

    steps: Sequence[Mapping[str, Any]]  # each step: {plugin, options, rationale}
    why: str

    def __post_init__(self) -> None:
        freeze_fields(self, "steps")


@dataclass(frozen=True, slots=True)
class GuidedSession:
    """The guided-mode session state.

    Persisted in CompositionState.guided_session. `terminal` becomes non-None
    when the wizard ends; subsequent freeform turns honour progressive
    disclosure (see §8.2 of the spec).
    """

    step: GuidedStep
    history: tuple[TurnRecord, ...]
    step_1_result: SourceResolved | None
    step_2_result: SinkResolved | None
    step_3_proposal: ChainProposal | None
    terminal: TerminalState | None

    @classmethod
    def initial(cls) -> GuidedSession:
        return cls(
            step=GuidedStep.STEP_1_SOURCE,
            history=(),
            step_1_result=None,
            step_2_result=None,
            step_3_proposal=None,
            terminal=None,
        )


# ---------------------------------------------------------------------------
# GuidedAuditDirective — L3-internal coordination type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GuidedAuditDirective:
    """Pure-function directive: "fire this guided audit event."

    ``step_advance()`` is pure (no uuid, no clock, no recorder), so it
    cannot construct ``ComposerToolInvocation`` records directly — those
    need a tool_call_id, timestamps, version snapshot, and operator
    actor that only the route handler has. Instead, step_advance returns
    a list of directives; the route handler (Phase 3) maps each
    directive's ``tool_name`` to the corresponding ``emit_*`` helper in
    ``composer/guided/audit.py`` and calls it with the live recorder,
    composition version, and actor.

    Per Errata C4: no new audit primitive at L0. ``GuidedAuditDirective``
    is L3-internal coordination only. The on-the-wire record is still
    ``ComposerToolInvocation``.

    Allowed ``tool_name`` values (closed list):
    - ``guided_turn_emitted``
    - ``guided_turn_answered``
    - ``guided_step_advanced``
    - ``guided_dropped_to_freeform``

    ``arguments`` is a payload dict; the Phase 3 route handler will
    translate it into the matching ``emit_*`` keyword arguments.
    """

    tool_name: str  # one of: guided_turn_emitted, guided_turn_answered,
    #                          guided_step_advanced, guided_dropped_to_freeform
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        freeze_fields(self, "arguments")


# ---------------------------------------------------------------------------
# step_advance — pure function, no I/O, no clock, no uuid
# ---------------------------------------------------------------------------

_StepAdvanceResult = tuple[GuidedSession, Turn | None, TerminalState | None, list[GuidedAuditDirective]]


def step_advance(
    session: GuidedSession,
    response: TurnResponse,
    *,
    current_turn_type: TurnType,
) -> _StepAdvanceResult:
    """Apply *response* to *session*. Pure function (no I/O, no clock, no uuid).

    Returns ``(new_session, next_turn_or_None, terminal_or_None, directives)``.
    The caller (route handler) emits each directive via the matching
    ``emit_*`` helper in ``composer.guided.audit``.

    Per spec §5.3:
    - A ``control_signal`` of ``"exit_to_freeform"`` terminates the wizard with
      ``TerminalKind.EXITED_TO_FREEFORM / TerminalReason.USER_PRESSED_EXIT``
      and produces a ``guided_dropped_to_freeform`` directive.
    - Otherwise, the current ``session.step`` selects the branch handler.
    """
    directives: list[GuidedAuditDirective] = []

    if response["control_signal"] == "exit_to_freeform":
        directives.append(
            GuidedAuditDirective(
                tool_name="guided_dropped_to_freeform",
                arguments={
                    "prev_step": session.step.value,
                    "drop_reason": TerminalReason.USER_PRESSED_EXIT.value,
                    "validation_result": None,
                },
            )
        )
        terminal = TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.USER_PRESSED_EXIT,
            pipeline_yaml=None,
        )
        return (replace(session, terminal=terminal), None, terminal, directives)

    if session.step is GuidedStep.STEP_1_SOURCE:
        return _advance_step_1(session, response, current_turn_type)
    if session.step is GuidedStep.STEP_2_SINK:
        return _advance_step_2(session, response, current_turn_type)
    if session.step is GuidedStep.STEP_2_5_RECIPE_MATCH:
        return _advance_step_2_5(session, response, current_turn_type)
    if session.step is GuidedStep.STEP_3_TRANSFORMS:
        return _advance_step_3(session, response, current_turn_type)
    raise AssertionError(f"unhandled step: {session.step}")


def _advance_step_1(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
) -> _StepAdvanceResult:
    """Handle a Step 1 (source) response.

    Only ``INSPECT_AND_CONFIRM`` causes a step transition; all other Step 1
    turn types (``SINGLE_SELECT`` for plugin selection, ``SCHEMA_FORM`` for
    options) are intra-step turns that do not advance the wizard. Those
    branches emit the next intra-step turn and are out of scope here.
    """
    if turn_type is not TurnType.INSPECT_AND_CONFIRM:
        # Intra-step turns (plugin select, options form) — do not advance.
        # Tasks 2.2+ wire these when the full intra-step flow is added.
        return (session, None, None, [])

    edited = response["edited_values"]
    if edited is None:
        raise ValueError("inspect_and_confirm response must carry edited_values; got None")

    source = SourceResolved(
        plugin=str(edited["plugin"]),
        options=dict(edited["options"]),
        observed_columns=tuple(edited["observed_columns"]),
        sample_rows=tuple(dict(r) for r in edited["sample_rows"]),
    )
    directives: list[GuidedAuditDirective] = [
        GuidedAuditDirective(
            tool_name="guided_step_advanced",
            arguments={
                "prev_step": GuidedStep.STEP_1_SOURCE.value,
                "next_step": GuidedStep.STEP_2_SINK.value,
                "reason": "user_advanced",
            },
        ),
    ]
    new_sess = replace(
        session,
        step=GuidedStep.STEP_2_SINK,
        step_1_result=source,
    )
    return (new_sess, None, None, directives)


def _advance_step_2(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
) -> _StepAdvanceResult:
    """Handle a Step 2 (sink) response.

    Only ``MULTI_SELECT_WITH_CUSTOM`` causes a step transition; all other Step 2
    turn types are intra-step turns that do not advance the wizard.

    The ``edited_values["outputs"]`` list is Tier-3 external data: each field
    is coerced at this boundary (``str``, ``dict``, ``tuple``, ``str``) before
    being stored in the audit-tier ``SinkOutputResolved`` dataclass.
    """
    if turn_type is not TurnType.MULTI_SELECT_WITH_CUSTOM:
        return (session, None, None, [])

    edited = response["edited_values"]
    if edited is None:
        raise ValueError("multi_select_with_custom response must carry edited_values")
    raw_outputs = edited["outputs"]
    outputs = tuple(
        SinkOutputResolved(
            plugin=str(o["plugin"]),
            options=dict(o["options"]),
            required_fields=tuple(o["required_fields"]),
            schema_mode=str(o["schema_mode"]),
        )
        for o in raw_outputs
    )
    sink = SinkResolved(outputs=outputs)
    directives = [
        GuidedAuditDirective(
            tool_name="guided_step_advanced",
            arguments={
                "prev_step": GuidedStep.STEP_2_SINK.value,
                "next_step": GuidedStep.STEP_2_5_RECIPE_MATCH.value,
                "reason": "user_advanced",
            },
        ),
    ]
    new_sess = replace(
        session,
        step=GuidedStep.STEP_2_5_RECIPE_MATCH,
        step_2_result=sink,
    )
    return (new_sess, None, None, directives)


def _advance_step_2_5(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
) -> _StepAdvanceResult:
    """Handle a Step 2.5 (recipe match) response.

    Two valid paths:
    - chosen == ["accept"]: the session stays at STEP_2_5 with no directive.
      The endpoint handler (Task 3.3 / Errata C2) detects response["chosen"] ==
      ["accept"] and invokes ``_execute_apply_pipeline_recipe`` to commit the
      recipe and produce a COMPLETED terminal. step_advance is pure and does not
      run apply_recipe; emitting emit_turn_answered is the handler's
      responsibility.
    - chosen == ["build_manually"]: advance to STEP_3_TRANSFORMS with a
      ``guided_step_advanced`` directive.

    Any other chosen value is a protocol violation — raises ValueError.
    Non-RECIPE_OFFER turn types are intra-step turns; no advance.
    """
    if turn_type is not TurnType.RECIPE_OFFER:
        return (session, None, None, [])

    chosen = response["chosen"] or []
    if list(chosen) == ["accept"]:
        # Endpoint handler reads response["chosen"] == ["accept"] and runs
        # apply_recipe (Errata C2). step_advance leaves the session at
        # STEP_2_5 unchanged; the handler advances to terminal=COMPLETED after
        # committing. No directive here — the handler emits emit_turn_answered.
        return (session, None, None, [])
    if list(chosen) == ["build_manually"]:
        directives = [
            GuidedAuditDirective(
                tool_name="guided_step_advanced",
                arguments={
                    "prev_step": GuidedStep.STEP_2_5_RECIPE_MATCH.value,
                    "next_step": GuidedStep.STEP_3_TRANSFORMS.value,
                    "reason": "user_advanced",
                },
            ),
        ]
        new_sess = replace(session, step=GuidedStep.STEP_3_TRANSFORMS)
        return (new_sess, None, None, directives)
    raise ValueError(f"unexpected chosen for recipe_offer: {chosen!r}")


def _advance_step_3(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
) -> _StepAdvanceResult:
    raise NotImplementedError("step 3 advance — implemented in Task 2.5")
