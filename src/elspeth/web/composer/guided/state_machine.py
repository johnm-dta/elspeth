"""Guided-mode state-machine data: GuidedSession, TerminalState, TurnRecord.

See docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md §5.

Trust tier: Tier 1 (audit). Coercion forbidden — every field crashes on
malformed input. The freeze_fields contract applies because these structures
are persisted and re-read across the audit trail.

Serialisation:
  Each type exposes ``to_dict()`` → plain JSON-serialisable dict and a
  corresponding ``from_dict(d)`` classmethod.  ``from_dict`` is Tier 1
  strict: it uses direct key access (never ``.get()``), constructs enums
  directly (ValueError on unknown value), and chains exceptions via
  ``from exc``.  The round-trip invariant holds for all types:
      obj == type.from_dict(obj.to_dict())
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import ChatRole, ChatTurn, ControlSignal, GuidedStep, Turn, TurnResponse, TurnType
from elspeth.web.composer.guided.resolved import (
    SinkOutputResolved as SinkOutputResolved,
)
from elspeth.web.composer.guided.resolved import (
    SinkResolved as SinkResolved,
)
from elspeth.web.composer.guided.resolved import (
    SourceResolved as SourceResolved,
)
from elspeth.web.composer.source_inspection import SourceInspectionFacts, facts_from_dict, facts_to_dict

# Pre-v5 persisted sessions are intentionally incompatible with v5: the
# operator must delete the guided sessions DB before deploying this change.
GUIDED_SESSION_SCHEMA_VERSION = 5

if TYPE_CHECKING:
    # Imported for type annotations only — avoids a circular dependency.
    # recipe_match.py imports state_machine.py (SourceResolved, SinkResolved);
    # a runtime import of RecipeMatch here would create a cycle.
    # RecipeMatch is a frozen dataclass; no freeze_fields needed on GuidedSession
    # for this field — frozen dataclass instances are already immutable.
    from elspeth.web.composer.guided.recipe_match import RecipeMatch


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-serialisable dict."""
        return {
            "kind": self.kind.value,
            "reason": self.reason.value if self.reason is not None else None,
            "pipeline_yaml": self.pipeline_yaml,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TerminalState:
        """Reconstruct from a plain dict.  Tier 1 strict — crashes on bad data."""
        try:
            return cls(
                kind=TerminalKind(d["kind"]),
                reason=TerminalReason(d["reason"]) if d["reason"] is not None else None,
                pipeline_yaml=d["pipeline_yaml"],
            )
        except (KeyError, ValueError) as exc:
            raise InvariantError(f"TerminalState.from_dict: malformed record {d!r}") from exc


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One emitted turn + its (optional) user response, recorded for audit."""

    step: GuidedStep
    turn_type: TurnType
    payload_hash: str
    response_hash: str | None
    emitter: str  # "server" | "llm"
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-serialisable dict."""
        return {
            "step": self.step.value,
            "turn_type": self.turn_type.value,
            "payload_hash": self.payload_hash,
            "response_hash": self.response_hash,
            "emitter": self.emitter,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TurnRecord:
        """Reconstruct from a plain dict.  Tier 1 strict — crashes on bad data."""
        try:
            return cls(
                step=GuidedStep(d["step"]),
                turn_type=TurnType(d["turn_type"]),
                payload_hash=d["payload_hash"],
                response_hash=d["response_hash"],
                emitter=d["emitter"],
                summary=d["summary"],
            )
        except (KeyError, ValueError) as exc:
            raise InvariantError(f"TurnRecord.from_dict: malformed record {d!r}") from exc


@dataclass(frozen=True, slots=True)
class SourceIntent:
    """Source plugin name, options, and observed inspection facts captured during Step 1.

    Persisted in GuidedSession.step_1_source_intent as a mid-Step-1 staging
    field.  The INSPECT_AND_CONFIRM emit site writes it; _advance_step_1 reads it
    when processing the INSPECT_AND_CONFIRM response to construct SourceResolved.

    It is cleared (set to None) as part of the same atomic replace() that
    consumes it, so it cannot be misread by a later step.

    Frozen, audit-tier: same freeze contract as SourceResolved and SinkIntent.
    options is a Mapping because the source schema can contain arbitrary types;
    observed_columns and sample_rows are Sequences for the same reason;
    freeze_fields enforces deep immutability on all container fields.
    """

    plugin: str
    options: Mapping[str, Any]
    observed_columns: Sequence[str]
    sample_rows: Sequence[Mapping[str, Any]]

    def __post_init__(self) -> None:
        freeze_fields(self, "options", "observed_columns", "sample_rows")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-serialisable dict."""
        return {
            "plugin": self.plugin,
            "options": deep_thaw(self.options),
            "observed_columns": list(deep_thaw(self.observed_columns)),
            "sample_rows": [dict(deep_thaw(r)) for r in self.sample_rows],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceIntent:
        """Reconstruct from a plain dict.  Tier 1 strict — crashes on bad data."""
        try:
            return cls(
                plugin=d["plugin"],
                options=d["options"],
                observed_columns=tuple(d["observed_columns"]),
                sample_rows=tuple(dict(r) for r in d["sample_rows"]),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvariantError(f"SourceIntent.from_dict: malformed record {d!r}") from exc


@dataclass(frozen=True, slots=True)
class SinkIntent:
    """Sink plugin name and options captured during the Step-2 SCHEMA_FORM turn.

    Persisted in GuidedSession.step_2_sink_intent as a mid-Step-2 staging
    field.  The SCHEMA_FORM dispatcher writes it; _advance_step_2 reads it
    when processing the subsequent MULTI_SELECT_WITH_CUSTOM response to
    construct the full SinkOutputResolved entry.

    It is cleared (set to None) as part of the same atomic replace() that
    consumes it, so it cannot be misread by a later step.

    Frozen, audit-tier: same freeze contract as SourceResolved and
    SinkOutputResolved.  options is a Mapping because the sink schema can
    contain arbitrary types; freeze_fields enforces deep immutability.
    """

    plugin: str
    options: Mapping[str, Any]

    def __post_init__(self) -> None:
        freeze_fields(self, "options")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-serialisable dict."""
        return {
            "plugin": self.plugin,
            "options": deep_thaw(self.options),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SinkIntent:
        """Reconstruct from a plain dict.  Tier 1 strict — crashes on bad data."""
        try:
            return cls(
                plugin=d["plugin"],
                options=d["options"],
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvariantError(f"SinkIntent.from_dict: malformed record {d!r}") from exc


@dataclass(frozen=True, slots=True)
class ChainProposal:
    """A transform chain proposed by Step 3 LLM.

    Frozen, audit-tier. Stored in GuidedSession.step_3_proposal.
    """

    steps: Sequence[Mapping[str, Any]]  # each step: {plugin, options, rationale}
    why: str

    def __post_init__(self) -> None:
        freeze_fields(self, "steps")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-serialisable dict."""
        return {
            "steps": [dict(deep_thaw(s)) for s in self.steps],
            "why": self.why,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChainProposal:
        """Reconstruct from a plain dict.  Tier 1 strict — crashes on bad data."""
        try:
            return cls(
                steps=tuple(dict(s) for s in d["steps"]),
                why=d["why"],
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvariantError(f"ChainProposal.from_dict: malformed record {d!r}") from exc


@dataclass(frozen=True, slots=True)
class GuidedSession:
    """The guided-mode session state.

    Persisted in CompositionState.guided_session. `terminal` becomes non-None
    when the wizard ends; subsequent freeform turns honour progressive
    disclosure (see §8.2 of the spec).

    Serialisation: use ``to_dict()`` / ``from_dict()`` for persistence via
    ``composer_meta["guided_session"]`` (see Task 3.5a implementation notes).
    The frozen dataclass equality check is the round-trip invariant test.

    ``step_1_source_intent`` is a mid-Step-1 staging field.  The INSPECT_AND_CONFIRM
    emit site writes the chosen source plugin + options + observed inspection facts
    into it before emitting the INSPECT_AND_CONFIRM turn.  ``_advance_step_1`` reads
    it to reconstruct the full SourceResolved and clears it in the same atomic
    replace(); it is always None after Step 1 completes.

    ``step_1_chosen_plugin`` is a mid-Step-1 staging field. The Step-1
    SINGLE_SELECT dispatcher writes the selected source plugin name before
    emitting the SCHEMA_FORM turn. GET /guided uses it with
    ``step_1_inspection_facts`` to rebuild the same prefilled schema form after
    refresh. It is cleared when the SCHEMA_FORM response commits Step 1.

    ``step_2_sink_intent`` is a mid-Step-2 staging field.  The SCHEMA_FORM
    dispatcher writes the chosen sink plugin + options into it before emitting
    the MULTI_SELECT_WITH_CUSTOM turn.  ``_advance_step_2`` reads it to
    reconstruct the full SinkOutputResolved and clears it in the same atomic
    replace(); it is always None after Step 2 completes.

    ``step_2_5_recipe_offer`` is a mid-Step-2.5 staging field.  The Step 2.5
    dispatcher writes the emitted ``RecipeMatch`` into it immediately before
    emitting the RECIPE_OFFER turn.  The recipe-accept branch in the dispatcher
    reads it to verify that the client-supplied ``recipe_name`` matches the
    recipe that was actually offered — binding the acceptance to the server-emitted
    offer and preventing a crafted client from accepting a different recipe.  The
    field is cleared (set to None) in the same atomic replace() that consumes it
    (the terminal=COMPLETED path).  It is always None when the session is not at
    STEP_2_5_RECIPE_MATCH.

    ``step_2_chosen_plugin`` is a mid-Step-2 staging field.  The Step-2
    SINGLE_SELECT dispatcher writes the chosen sink plugin name into it
    immediately before emitting the SCHEMA_FORM turn.  The SCHEMA_FORM
    dispatcher reads it when rebuilding the SCHEMA_FORM on GET /guided —
    the chosen plugin is needed to retrieve the correct schema from the
    catalog.  It is cleared (set to None) in the same atomic replace()
    that sets ``step_2_sink_intent`` (i.e. when the SCHEMA_FORM response
    arrives and the MULTI_SELECT_WITH_CUSTOM turn is about to be emitted);
    it cannot be non-None at the same time as ``step_2_sink_intent``.
    It is always None outside the SINGLE_SELECT→SCHEMA_FORM intra-step
    window at STEP_2_SINK.
    """

    step: GuidedStep
    history: tuple[TurnRecord, ...]
    step_1_result: SourceResolved | None
    step_2_result: SinkResolved | None
    step_3_proposal: ChainProposal | None
    step_1_inspection_facts: SourceInspectionFacts | None = None
    step_1_chosen_plugin: str | None = None
    terminal: TerminalState | None = None
    transition_consumed: bool = False
    step_1_source_intent: SourceIntent | None = None
    step_2_sink_intent: SinkIntent | None = None
    step_2_5_recipe_offer: RecipeMatch | None = None
    step_2_chosen_plugin: str | None = None
    step_3_edit_index: int | None = None
    # Phase A slice 5 — per-step chat history persistence.
    # `chat_history` is a tuple of frozen ChatTurn dataclasses containing
    # scalars and enums only. The tuple plus frozen element type is already
    # deeply immutable, so GuidedSession does not need a freeze_fields guard
    # for this field.
    # `chat_turn_seq` is monotonic per session across all chat turns
    # (user + assistant share the counter); incremented on every append.
    chat_history: tuple[ChatTurn, ...] = ()
    chat_turn_seq: int = 0

    def __post_init__(self) -> None:
        if self.step_1_inspection_facts is not None and type(self.step_1_inspection_facts) is not SourceInspectionFacts:
            raise TypeError(
                f"step_1_inspection_facts must be SourceInspectionFacts or None, got {type(self.step_1_inspection_facts).__name__}"
            )
        if self.step_1_chosen_plugin is not None and type(self.step_1_chosen_plugin) is not str:
            raise TypeError(f"step_1_chosen_plugin must be str or None, got {type(self.step_1_chosen_plugin).__name__}")

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-serialisable dict.

        All nested optional types serialise their presence — ``None`` round-
        trips as ``None`` (never fabricated).

        ``chat_history`` entries are frozen dataclasses; their ``role`` and
        ``step`` members are ``StrEnum`` instances, which serialise to their
        string values via the explicit ``.value`` accessors below so JSON
        output never carries enum reprs.
        """
        return {
            "schema_version": GUIDED_SESSION_SCHEMA_VERSION,
            "step": self.step.value,
            "history": [r.to_dict() for r in self.history],
            "step_1_result": self.step_1_result.to_dict() if self.step_1_result is not None else None,
            "step_2_result": self.step_2_result.to_dict() if self.step_2_result is not None else None,
            "step_3_proposal": self.step_3_proposal.to_dict() if self.step_3_proposal is not None else None,
            "step_1_inspection_facts": facts_to_dict(self.step_1_inspection_facts) if self.step_1_inspection_facts is not None else None,
            "step_1_chosen_plugin": self.step_1_chosen_plugin,
            "terminal": self.terminal.to_dict() if self.terminal is not None else None,
            "transition_consumed": self.transition_consumed,
            "step_1_source_intent": self.step_1_source_intent.to_dict() if self.step_1_source_intent is not None else None,
            "step_2_sink_intent": self.step_2_sink_intent.to_dict() if self.step_2_sink_intent is not None else None,
            "step_2_5_recipe_offer": self.step_2_5_recipe_offer.to_dict() if self.step_2_5_recipe_offer is not None else None,
            "step_2_chosen_plugin": self.step_2_chosen_plugin,
            "step_3_edit_index": self.step_3_edit_index,
            "chat_history": [
                {
                    "role": t.role.value,
                    "content": t.content,
                    "seq": t.seq,
                    "step": t.step.value,
                    "ts_iso": t.ts_iso,
                }
                for t in self.chat_history
            ],
            "chat_turn_seq": self.chat_turn_seq,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GuidedSession:
        """Reconstruct from a plain dict.  Tier 1 strict — crashes on bad data.

        Used when restoring state from ``composer_meta["guided_session"]``.
        KeyError, ValueError, and TypeError all indicate Tier 1 corruption.
        """
        try:
            schema_version = int(d["schema_version"])
            if schema_version != GUIDED_SESSION_SCHEMA_VERSION:
                raise InvariantError(f"GuidedSession.from_dict: unsupported schema_version {schema_version}")
            history = tuple(TurnRecord.from_dict(r) for r in d["history"])
            step_1_raw = d["step_1_result"]
            step_2_raw = d["step_2_result"]
            step_3_raw = d["step_3_proposal"]
            inspection_facts_raw = d["step_1_inspection_facts"]
            step_1_chosen_plugin_raw = d["step_1_chosen_plugin"]
            terminal_raw = d["terminal"]
            source_intent_raw = d["step_1_source_intent"]
            sink_intent_raw = d["step_2_sink_intent"]
            recipe_offer_raw = d["step_2_5_recipe_offer"]
            step_2_chosen_plugin_raw = d["step_2_chosen_plugin"]
            step_3_edit_index_raw = d["step_3_edit_index"]
            # Deferred import to avoid a circular dependency at module level.
            # recipe_match.py imports from state_machine.py; importing RecipeMatch
            # at module level here would create a cycle.
            from elspeth.web.composer.guided.recipe_match import RecipeMatch as _RecipeMatch

            # Phase A slice 5 chat-history fields.  Tier-1 strict: every entry
            # must declare role / content / seq / step / ts_iso.  Per CLAUDE.md
            # "Our data crash on any anomaly" — no coercion of missing keys
            # to defaults.  An empty list (default for sessions created before
            # slice 5 landed in production) is valid; the entries themselves
            # must be well-formed.
            chat_history_raw = d["chat_history"]
            chat_turn_seq_raw = d["chat_turn_seq"]
            chat_history: tuple[ChatTurn, ...] = tuple(
                ChatTurn(
                    role=ChatRole(entry["role"]),
                    content=entry["content"],
                    seq=entry["seq"],
                    step=GuidedStep(entry["step"]),
                    ts_iso=entry["ts_iso"],
                )
                for entry in chat_history_raw
            )
            return cls(
                step=GuidedStep(d["step"]),
                history=history,
                step_1_result=SourceResolved.from_dict(step_1_raw) if step_1_raw is not None else None,
                step_2_result=SinkResolved.from_dict(step_2_raw) if step_2_raw is not None else None,
                step_3_proposal=ChainProposal.from_dict(step_3_raw) if step_3_raw is not None else None,
                step_1_inspection_facts=facts_from_dict(inspection_facts_raw) if inspection_facts_raw is not None else None,
                step_1_chosen_plugin=str(step_1_chosen_plugin_raw) if step_1_chosen_plugin_raw is not None else None,
                terminal=TerminalState.from_dict(terminal_raw) if terminal_raw is not None else None,
                transition_consumed=d["transition_consumed"],
                step_1_source_intent=SourceIntent.from_dict(source_intent_raw) if source_intent_raw is not None else None,
                step_2_sink_intent=SinkIntent.from_dict(sink_intent_raw) if sink_intent_raw is not None else None,
                step_2_5_recipe_offer=_RecipeMatch.from_dict(recipe_offer_raw) if recipe_offer_raw is not None else None,
                step_2_chosen_plugin=str(step_2_chosen_plugin_raw) if step_2_chosen_plugin_raw is not None else None,
                step_3_edit_index=int(step_3_edit_index_raw) if step_3_edit_index_raw is not None else None,
                chat_history=chat_history,
                chat_turn_seq=int(chat_turn_seq_raw),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvariantError(f"GuidedSession.from_dict: malformed record {d!r}") from exc


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
    - A ``control_signal`` of ``ControlSignal.EXIT_TO_FREEFORM`` terminates the wizard with
      ``TerminalKind.EXITED_TO_FREEFORM / TerminalReason.USER_PRESSED_EXIT``
      and produces a ``guided_dropped_to_freeform`` directive.
    - Otherwise, the current ``session.step`` selects the branch handler.
    """
    directives: list[GuidedAuditDirective] = []

    if response["control_signal"] is ControlSignal.EXIT_TO_FREEFORM:
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
    raise InvariantError(f"unhandled step: {session.step}")


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

    The inspection state (plugin, options, sample_rows) is held in
    ``session.step_1_source_intent`` — set by the INSPECT_AND_CONFIRM emit site
    before the turn is returned to the client.  The wire response carries only
    ``edited_values = {"columns": list[str]}``, since that is the only field
    the widget can authoritatively edit.  plugin/options/sample_rows are recovered
    from intent; columns come from the response.

    All values from the response are Tier-3 external data and are coerced
    (str, tuple) before being stored in the audit-tier ``SourceResolved``.
    """
    if turn_type is not TurnType.INSPECT_AND_CONFIRM:
        # Intra-step turns (plugin select, options form) — do not advance.
        # Tasks 2.2+ wire these when the full intra-step flow is added.
        return (session, None, None, [])

    intent = session.step_1_source_intent
    if intent is None:
        raise InvariantError(
            "_advance_step_1: step_1_source_intent is None when INSPECT_AND_CONFIRM "
            "was received — the INSPECT_AND_CONFIRM emit site must set it before "
            "emitting the turn. This is a state-machine invariant violation."
        )

    edited = response["edited_values"]
    if edited is None:
        raise ValueError("inspect_and_confirm response must carry edited_values; got None")

    # columns is the only field the widget edits; it is Tier-3: coerce to tuple[str, ...].
    columns_raw = edited["columns"]  # KeyError propagates as protocol violation
    columns = tuple(str(c) for c in columns_raw)

    source = SourceResolved(
        plugin=intent.plugin,
        options=dict(intent.options),
        observed_columns=columns,
        sample_rows=tuple(dict(r) for r in intent.sample_rows),
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
        step_1_source_intent=None,  # consumed; clear to prevent misread by later steps
        step_1_chosen_plugin=None,
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

    The MULTI_SELECT_WITH_CUSTOM response carries:
    - ``chosen``: the list of required field names the user selected
    - ``custom_inputs``: any additional field names the user typed in

    The sink plugin name and options were persisted in
    ``session.step_2_sink_intent`` by the preceding SCHEMA_FORM dispatcher.
    ``_advance_step_2`` reads them here, combines with ``chosen`` +
    ``custom_inputs`` to construct a single ``SinkOutputResolved``, and
    clears ``step_2_sink_intent`` in the same atomic replace() so it cannot
    be misread by a later step.

    All values from the response are Tier-3 external data and are coerced
    (str, tuple) before being stored in the audit-tier ``SinkOutputResolved``.
    """
    if turn_type is not TurnType.MULTI_SELECT_WITH_CUSTOM:
        return (session, None, None, [])

    intent = session.step_2_sink_intent
    if intent is None:
        raise InvariantError(
            "_advance_step_2: step_2_sink_intent is None when MULTI_SELECT_WITH_CUSTOM "
            "was received — the SCHEMA_FORM dispatcher must set it before emitting "
            "the MULTI_SELECT_WITH_CUSTOM turn. This is a state-machine invariant "
            "violation."
        )

    # chosen and custom_inputs are Tier-3: coerce to tuple[str, ...].
    chosen_raw = response["chosen"] or []
    custom_inputs_raw = response["custom_inputs"] or []
    required_fields = tuple(str(f) for f in chosen_raw) + tuple(str(f) for f in custom_inputs_raw)

    output = SinkOutputResolved(
        plugin=intent.plugin,
        options=dict(deep_thaw(intent.options)),
        required_fields=required_fields,
        schema_mode="observed",
    )
    sink = SinkResolved(outputs=(output,))
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
        step_2_sink_intent=None,  # consumed; clear to prevent misread by later steps
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
    """Handle a Step 3 (transform chain) response.

    Acceptance/rejection of a chain proposal is interpreted by the endpoint
    handler (Task 4.4), which runs preview_pipeline and commits via tools.py.
    step_advance is pure and does not mutate state on accept; the handler does.

    Legal turn types at Step 3:
    - PROPOSE_CHAIN: The LLM has proposed a chain. Accept/reject is decided
      by the endpoint handler after running preview_pipeline. step_advance
      passes through unchanged.
    - SINGLE_SELECT: A clarifying question was answered — no step change.
      The handler interprets the response and either re-emits propose_chain
      or asks another question.
    - SCHEMA_FORM: The operator edited one proposed transform's options.
      The handler patches the staged proposal and re-emits propose_chain.

    Any other turn type is a server-side invariant violation: Step 3 only ever
    emits PROPOSE_CHAIN or SINGLE_SELECT turns, so a different turn type in
    ``current_turn_type`` means the emitter stamped an invalid type on the
    history record — raises InvariantError (server bug, not client fault).
    """
    if turn_type is TurnType.PROPOSE_CHAIN:
        return (session, None, None, [])
    if turn_type is TurnType.SINGLE_SELECT:
        # Clarifying question answered — no step change. The handler interprets
        # the response and either re-emits propose_chain or asks another question.
        return (session, None, None, [])
    if turn_type is TurnType.SCHEMA_FORM:
        return (session, None, None, [])
    raise InvariantError(
        f"_advance_step_3: unexpected turn_type {turn_type!r} — Step 3 only "
        "emits PROPOSE_CHAIN, SINGLE_SELECT, and SCHEMA_FORM turns; any other type in the "
        "history record indicates a server-side emitter bug."
    )


# ---------------------------------------------------------------------------
# Terminal-failure helpers — standalone endpoint helpers for spec §5.4
# ---------------------------------------------------------------------------


def mark_solver_exhausted(
    session: GuidedSession,
    *,
    validation_result: Mapping[str, Any] | None,
) -> tuple[GuidedSession, TerminalState, list[GuidedAuditDirective]]:
    """Endpoint helper: stamp the session as solver-exhausted and emit a directive.

    Called by the Step 3 endpoint handler after repair attempt + advisor
    consultation both fail (spec §5.4). Pure function; the route handler
    fans the directive out to emit_dropped_to_freeform.

    Returns ``(new_session, terminal, directives)`` where ``directives`` is
    a ``list[GuidedAuditDirective]`` carrying the ``guided_dropped_to_freeform``
    event (Errata C4). The route handler maps each directive to the matching
    ``emit_*`` helper in ``composer/guided/audit.py``.
    """
    directives: list[GuidedAuditDirective] = [
        GuidedAuditDirective(
            tool_name="guided_dropped_to_freeform",
            arguments={
                "prev_step": session.step.value,
                "drop_reason": TerminalReason.SOLVER_EXHAUSTED.value,
                "validation_result": (dict(validation_result) if validation_result is not None else None),
            },
        ),
    ]
    terminal = TerminalState(
        kind=TerminalKind.EXITED_TO_FREEFORM,
        reason=TerminalReason.SOLVER_EXHAUSTED,
        pipeline_yaml=None,
    )
    new_sess = replace(session, terminal=terminal)
    return (new_sess, terminal, directives)


def mark_protocol_violation(
    session: GuidedSession,
) -> tuple[GuidedSession, TerminalState, list[GuidedAuditDirective]]:
    """Endpoint helper: stamp the session as protocol-violated and emit a directive.

    Called by the route handler after the LLM emits an illegal turn type
    twice in a row (spec §5.4). ``validation_result`` is ``None`` — the
    violation is at the turn-type level, not the schema level.

    Returns ``(new_session, terminal, directives)`` where ``directives`` is
    a ``list[GuidedAuditDirective]`` carrying the ``guided_dropped_to_freeform``
    event (Errata C4). The route handler maps each directive to the matching
    ``emit_*`` helper in ``composer/guided/audit.py``.
    """
    directives: list[GuidedAuditDirective] = [
        GuidedAuditDirective(
            tool_name="guided_dropped_to_freeform",
            arguments={
                "prev_step": session.step.value,
                "drop_reason": TerminalReason.PROTOCOL_VIOLATION.value,
                "validation_result": None,
            },
        ),
    ]
    terminal = TerminalState(
        kind=TerminalKind.EXITED_TO_FREEFORM,
        reason=TerminalReason.PROTOCOL_VIOLATION,
        pipeline_yaml=None,
    )
    new_sess = replace(session, terminal=terminal)
    return (new_sess, terminal, directives)
