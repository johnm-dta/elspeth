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
from typing import Any, cast

from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.profile import EMPTY_PROFILE, WorkflowProfile
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

# Pre-v7 persisted sessions are intentionally incompatible with v7: the
# operator must delete the guided sessions DB before deploying this change.
# (v6->v7 dropped the vestigial ``entry_seed`` key from the nested
# WorkflowProfile sub-shape; bumped in lockstep with SESSION_SCHEMA_EPOCH.)
GUIDED_SESSION_SCHEMA_VERSION = 7


def _require_guided_int(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise InvariantError(f"GuidedSession.from_dict: {field_name} must be int")
    return value


def _require_guided_non_negative_int(value: Any, field_name: str) -> int:
    parsed = _require_guided_int(value, field_name)
    if parsed < 0:
        raise InvariantError(f"GuidedSession.from_dict: {field_name} must be a non-negative int")
    return parsed


def _require_guided_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise InvariantError(f"GuidedSession.from_dict: {field_name} must be bool")
    return value


def _require_guided_optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise InvariantError(f"GuidedSession.from_dict: {field_name} must be str or None")
    return value


def _require_guided_sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise InvariantError(f"GuidedSession.from_dict: {field_name} must be a sequence")
    return cast(Sequence[Any], value)


def _chat_turn_from_guided_dict(entry: Any) -> ChatTurn:
    if not isinstance(entry, Mapping):
        raise InvariantError("GuidedSession.from_dict: chat_history entries must be mappings")
    role_raw = entry["role"]
    content_raw = entry["content"]
    seq_raw = entry["seq"]
    step_raw = entry["step"]
    ts_iso_raw = entry["ts_iso"]
    if type(role_raw) is not str:
        raise InvariantError("GuidedSession.from_dict: chat_history.role must be str")
    if type(step_raw) is not str:
        raise InvariantError("GuidedSession.from_dict: chat_history.step must be str")
    if type(content_raw) is not str:
        raise InvariantError("GuidedSession.from_dict: chat_history.content must be str")
    if type(ts_iso_raw) is not str:
        raise InvariantError("GuidedSession.from_dict: chat_history.ts_iso must be str")
    # assistant_message_kind / synthetic_failure_reason (fp-review C-2
    # persisted-history closure): genuinely OPTIONAL, unlike every field
    # above — a turn persisted before this field existed has no key at all.
    # ``.get()`` (not direct indexing) is the correct read here precisely
    # because absence is a valid, non-fabricated state, not missing required
    # data; ChatTurn's own __post_init__ enforces the closed value sets and
    # the cross-field/role invariants (surfacing as ValueError, caught by the
    # broad except below like every other malformed-record case).
    assistant_message_kind_raw = entry.get("assistant_message_kind")
    if assistant_message_kind_raw is not None and type(assistant_message_kind_raw) is not str:
        raise InvariantError("GuidedSession.from_dict: chat_history.assistant_message_kind must be str or None")
    synthetic_failure_reason_raw = entry.get("synthetic_failure_reason")
    if synthetic_failure_reason_raw is not None and type(synthetic_failure_reason_raw) is not str:
        raise InvariantError("GuidedSession.from_dict: chat_history.synthetic_failure_reason must be str or None")
    return ChatTurn(
        role=ChatRole(role_raw),
        content=content_raw,
        seq=_require_guided_non_negative_int(seq_raw, "chat_history.seq"),
        step=GuidedStep(step_raw),
        ts_iso=ts_iso_raw,
        assistant_message_kind=cast(Any, assistant_message_kind_raw),
        synthetic_failure_reason=cast(Any, synthetic_failure_reason_raw),
    )


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
    profile: WorkflowProfile = EMPTY_PROFILE
    advisor_checkpoint_passes_used: int = 0
    advisor_signoff_escape_offered: bool = False
    step_1_inspection_facts: SourceInspectionFacts | None = None
    step_1_chosen_plugin: str | None = None
    terminal: TerminalState | None = None
    transition_consumed: bool = False
    step_1_source_intent: SourceIntent | None = None
    step_2_sink_intent: SinkIntent | None = None
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
        if type(self.profile) is not WorkflowProfile:
            raise TypeError(f"profile must be WorkflowProfile, got {type(self.profile).__name__}")
        if type(self.advisor_checkpoint_passes_used) is not int or self.advisor_checkpoint_passes_used < 0:
            raise TypeError("advisor_checkpoint_passes_used must be a non-negative int")
        if type(self.advisor_signoff_escape_offered) is not bool:
            raise TypeError(f"advisor_signoff_escape_offered must be bool, got {type(self.advisor_signoff_escape_offered).__name__}")
        if self.step_1_inspection_facts is not None and type(self.step_1_inspection_facts) is not SourceInspectionFacts:
            raise TypeError(
                f"step_1_inspection_facts must be SourceInspectionFacts or None, got {type(self.step_1_inspection_facts).__name__}"
            )
        if self.step_1_chosen_plugin is not None and type(self.step_1_chosen_plugin) is not str:
            raise TypeError(f"step_1_chosen_plugin must be str or None, got {type(self.step_1_chosen_plugin).__name__}")

    @classmethod
    def initial(cls, profile: WorkflowProfile = EMPTY_PROFILE) -> GuidedSession:
        return cls(
            step=GuidedStep.STEP_1_SOURCE,
            history=(),
            step_1_result=None,
            step_2_result=None,
            step_3_proposal=None,
            profile=profile,
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
            "profile": self.profile.to_dict(),
            "advisor_checkpoint_passes_used": self.advisor_checkpoint_passes_used,
            "advisor_signoff_escape_offered": self.advisor_signoff_escape_offered,
            "step_1_inspection_facts": facts_to_dict(self.step_1_inspection_facts) if self.step_1_inspection_facts is not None else None,
            "step_1_chosen_plugin": self.step_1_chosen_plugin,
            "terminal": self.terminal.to_dict() if self.terminal is not None else None,
            "transition_consumed": self.transition_consumed,
            "step_1_source_intent": self.step_1_source_intent.to_dict() if self.step_1_source_intent is not None else None,
            "step_2_sink_intent": self.step_2_sink_intent.to_dict() if self.step_2_sink_intent is not None else None,
            "step_2_chosen_plugin": self.step_2_chosen_plugin,
            "step_3_edit_index": self.step_3_edit_index,
            "chat_history": [
                {
                    "role": t.role.value,
                    "content": t.content,
                    "seq": t.seq,
                    "step": t.step.value,
                    "ts_iso": t.ts_iso,
                    "assistant_message_kind": t.assistant_message_kind,
                    "synthetic_failure_reason": t.synthetic_failure_reason,
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
            schema_version = _require_guided_int(d["schema_version"], "schema_version")
            if schema_version != GUIDED_SESSION_SCHEMA_VERSION:
                raise InvariantError(f"GuidedSession.from_dict: unsupported schema_version {schema_version}")
            history = tuple(TurnRecord.from_dict(r) for r in d["history"])
            step_1_raw = d["step_1_result"]
            step_2_raw = d["step_2_result"]
            step_3_raw = d["step_3_proposal"]
            profile_raw = d["profile"]
            advisor_checkpoint_passes_used_raw = d["advisor_checkpoint_passes_used"]
            advisor_signoff_escape_offered_raw = d["advisor_signoff_escape_offered"]
            try:
                profile = WorkflowProfile.from_dict(profile_raw)
            except InvariantError as exc:
                raise InvariantError("GuidedSession.from_dict: malformed profile") from exc
            if type(advisor_checkpoint_passes_used_raw) is not int or advisor_checkpoint_passes_used_raw < 0:
                raise InvariantError("GuidedSession.from_dict: advisor_checkpoint_passes_used must be a non-negative int")
            if type(advisor_signoff_escape_offered_raw) is not bool:
                raise InvariantError("GuidedSession.from_dict: advisor_signoff_escape_offered must be bool")
            inspection_facts_raw = d["step_1_inspection_facts"]
            step_1_chosen_plugin_raw = d["step_1_chosen_plugin"]
            terminal_raw = d["terminal"]
            source_intent_raw = d["step_1_source_intent"]
            sink_intent_raw = d["step_2_sink_intent"]
            step_2_chosen_plugin_raw = d["step_2_chosen_plugin"]
            step_3_edit_index_raw = d["step_3_edit_index"]
            transition_consumed = _require_guided_bool(d["transition_consumed"], "transition_consumed")
            step_1_chosen_plugin = _require_guided_optional_str(step_1_chosen_plugin_raw, "step_1_chosen_plugin")
            step_2_chosen_plugin = _require_guided_optional_str(step_2_chosen_plugin_raw, "step_2_chosen_plugin")
            step_3_edit_index = (
                _require_guided_non_negative_int(step_3_edit_index_raw, "step_3_edit_index") if step_3_edit_index_raw is not None else None
            )
            # Phase A slice 5 chat-history fields.  Tier-1 strict: every entry
            # must declare role / content / seq / step / ts_iso.  Per CLAUDE.md
            # "Our data crash on any anomaly" — no coercion of missing keys
            # to defaults.  An empty list (default for sessions created before
            # slice 5 landed in production) is valid; the entries themselves
            # must be well-formed.
            chat_history_raw = _require_guided_sequence(d["chat_history"], "chat_history")
            chat_turn_seq = _require_guided_non_negative_int(d["chat_turn_seq"], "chat_turn_seq")
            chat_history: tuple[ChatTurn, ...] = tuple(_chat_turn_from_guided_dict(entry) for entry in chat_history_raw)
            return cls(
                step=GuidedStep(d["step"]),
                history=history,
                step_1_result=SourceResolved.from_dict(step_1_raw) if step_1_raw is not None else None,
                step_2_result=SinkResolved.from_dict(step_2_raw) if step_2_raw is not None else None,
                step_3_proposal=ChainProposal.from_dict(step_3_raw) if step_3_raw is not None else None,
                profile=profile,
                advisor_checkpoint_passes_used=advisor_checkpoint_passes_used_raw,
                advisor_signoff_escape_offered=advisor_signoff_escape_offered_raw,
                step_1_inspection_facts=facts_from_dict(inspection_facts_raw) if inspection_facts_raw is not None else None,
                step_1_chosen_plugin=step_1_chosen_plugin,
                terminal=TerminalState.from_dict(terminal_raw) if terminal_raw is not None else None,
                transition_consumed=transition_consumed,
                step_1_source_intent=SourceIntent.from_dict(source_intent_raw) if source_intent_raw is not None else None,
                step_2_sink_intent=SinkIntent.from_dict(sink_intent_raw) if sink_intent_raw is not None else None,
                step_2_chosen_plugin=step_2_chosen_plugin,
                step_3_edit_index=step_3_edit_index,
                chat_history=chat_history,
                chat_turn_seq=chat_turn_seq,
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
    if session.step is GuidedStep.STEP_3_TRANSFORMS:
        return _advance_step_3(session, response, current_turn_type)
    if session.step is GuidedStep.STEP_4_WIRE:
        return _advance_step_4(session, response, current_turn_type)
    raise InvariantError(f"unhandled step: {session.step}")


def _advance_step_1(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
) -> _StepAdvanceResult:
    """Handle a Step 1 (source) response. Pure self-loop for every Step 1 turn type.

    Step 1 advancement (the INSPECT_AND_CONFIRM -> STEP_2_SINK transition) is
    owned entirely by the dispatcher/handler path
    (``_dispatch_guided_respond``'s STEP_1_SOURCE -> STEP_2_SINK
    INSPECT_AND_CONFIRM branch in ``sessions/routes/_helpers.py``) — mirroring
    ``_advance_step_2`` (elspeth-948eb9c0b8 C-3(b)) and how Step 3/Step 4
    already work: the resolve (``step_1_source_intent`` +
    ``edited_values["columns"]`` -> ``SourceResolved``) and the source commit
    via ``handle_step_1_source`` both happen in the dispatcher, and
    ``step``/``step_1_result`` are only ever set after the commit is known to
    have succeeded.

    Previously this function unconditionally pre-set ``step_1_result`` and
    advanced ``step`` to STEP_2_SINK for INSPECT_AND_CONFIRM *before* the
    source was ever committed via ``handle_step_1_source`` — the same
    eager-pre-set shape that caused the Step 2 divergence — and even coerced
    a malformed (non-list) ``columns`` payload silently (iterating a scalar
    string's characters) before the dispatcher's type guard ever ran. This
    turn type has no live production emitter today (``_build_get_guided_turn``
    always passes ``blob_inspection=None``; only the integration test suite's
    ``_seed_inspect_and_confirm_history`` reaches it), so the defect was
    latent, not observed — fixed here for the same reason Step 2 was: the
    commit-then-advance discipline must hold for every step this state
    machine owns, reachable today or not.

    All other Step 1 turn types (``SINGLE_SELECT``, ``SCHEMA_FORM``) were
    already pure self-loops here.
    """
    return (session, None, None, [])


def _advance_step_2(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
) -> _StepAdvanceResult:
    """Handle a Step 2 (sink) response. Pure self-loop for every Step 2 turn type.

    Step 2 advancement (the MULTI_SELECT_WITH_CUSTOM -> STEP_3_TRANSFORMS
    transition) is owned entirely by the dispatcher/handler path
    (``_dispatch_guided_respond``'s STEP_2_SINK intra-step MULTI_SELECT_WITH_CUSTOM
    branch in ``sessions/routes/_helpers.py``) — mirroring how Step 3/Step 4
    already work (``_advance_step_3``/``_advance_step_4`` are pure self-loops;
    ``handle_step_3_chain_accept`` sets the step pointer itself, atomically
    with the state mutation).

    Previously this function unconditionally pre-set ``step_2_result`` and
    advanced ``step`` to STEP_3_TRANSFORMS for MULTI_SELECT_WITH_CUSTOM
    *before* the sink was ever committed via ``handle_step_2_sink`` — a
    downstream commit failure then left ``guided_session.step_2_result``
    (and ``step``) advanced while ``composition_state.outputs`` stayed
    unchanged, a persisted state-integrity divergence (elspeth-948eb9c0b8
    C-3(b)). Resolving the sink from ``step_2_sink_intent`` + the response,
    validating it, and setting ``step``/``step_2_result`` now all happen in
    the dispatcher, gated on ``handle_step_2_sink`` reporting success — so a
    failure leaves this pure function's return value (and therefore the
    persisted session) untouched.
    """
    return (session, None, None, [])


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


def _advance_step_4(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
) -> _StepAdvanceResult:
    """Handle Step 4 (wire skeleton) responses.

    Step 4 advancement is owned by the dispatcher/handler path in later work.
    The state-machine branch is intentionally a pure self-loop for the
    CONFIRM_WIRING turn and must not stamp terminal state.
    """
    if turn_type is TurnType.CONFIRM_WIRING:
        return (session, None, None, [])
    raise InvariantError(
        f"_advance_step_4: unexpected turn_type {turn_type!r} for {GuidedStep.STEP_4_WIRE.name}; Step 4 only emits CONFIRM_WIRING turns."
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
