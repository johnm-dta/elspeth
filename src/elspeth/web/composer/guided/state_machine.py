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
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Literal, TypedDict, TypeVar, cast
from uuid import UUID

from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.profile import EMPTY_PROFILE, WorkflowProfile
from elspeth.web.composer.guided.protocol import ChatRole, ChatTurn, ControlSignal, GuidedStep, Turn, TurnResponse, TurnType
from elspeth.web.composer.guided.resolved import (
    GUIDED_JSON_MAX_ITEMS,
    GuidedJsonBudget,
    freeze_guided_json_mapping,
    freeze_guided_str_sequence,
)
from elspeth.web.composer.guided.resolved import (
    SinkOutputResolved as SinkOutputResolved,
)
from elspeth.web.composer.guided.resolved import (
    SinkResolved as SinkResolved,
)
from elspeth.web.composer.guided.resolved import (
    SourceResolved as SourceResolved,
)
from elspeth.web.composer.pipeline_proposal import AbsentBase, PresentBase, ProposalBase, reviewed_anchor_hash
from elspeth.web.composer.source_inspection import SourceInspectionFacts, facts_from_dict, facts_to_dict

# Schema 8 is a pre-release hard cut. There is no schema-7 decoder or
# converter: session epoch 29 owns the corresponding store recreation.
GUIDED_SESSION_SCHEMA_VERSION = 8
GUIDED_MAX_COMPONENTS_PER_KIND = 256
GUIDED_MAX_DEFERRED_INTENTS = 256
GUIDED_MAX_CONSTRAINTS_PER_INTENT = 64
GUIDED_MAX_TOTAL_CONSTRAINTS = 4_096
GUIDED_MAX_HISTORY_RECORDS = 4_096
GUIDED_MAX_CHAT_TURNS = 4_096
GUIDED_MAX_HISTORY_SUMMARY_CHARS = 1_048_576
GUIDED_MAX_CHAT_HISTORY_CHARS = 4_194_304
GUIDED_MAX_REDACTED_SUMMARY_CHARS = 4_096
GUIDED_MAX_CHAT_CONTENT_CHARS = 65_536
GUIDED_MAX_TURN_SUMMARY_CHARS = 4_096

_SHA256_HEX = frozenset("0123456789abcdef")
_ComponentT = TypeVar("_ComponentT")
_GUIDED_SESSION_KEYS = frozenset(
    {
        "schema_version",
        "step",
        "history",
        "profile",
        "advisor_checkpoint_passes_used",
        "advisor_signoff_escape_offered",
        "terminal",
        "transition_consumed",
        "chat_history",
        "chat_turn_seq",
        "source_order",
        "reviewed_sources",
        "pending_source_intents",
        "output_order",
        "reviewed_outputs",
        "pending_output_intents",
        "deferred_intents",
        "active_proposal",
        "active_edit_target",
        "root_intent_message_id",
    }
)


def _require_exact_dict(value: object, expected: frozenset[str], owner: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise InvariantError(f"{owner}: record must be an exact dict")
    record = value
    unexpected = set(record) - expected
    if unexpected:
        raise InvariantError(f"{owner}: unexpected keys {sorted(unexpected)!r}")
    missing = expected - set(record)
    if missing:
        raise InvariantError(f"{owner}: missing keys {sorted(missing)!r}")
    return record


def _require_nonempty_str(value: object, field_name: str) -> str:
    if type(value) is not str or value == "":
        raise InvariantError(f"{field_name} must be a non-empty exact str")
    return value


def _is_exact_str(value: object) -> bool:
    return type(value) is str


def _require_optional_nonempty_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_str(value, field_name)


def _require_hash(value: object, field_name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise InvariantError(f"{field_name} must be exactly 64 lowercase hexadecimal characters")
    return value


def _canonical_uuid_text(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise InvariantError(f"{field_name} must be a canonical lowercase UUID string")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise InvariantError(f"{field_name} must be a canonical lowercase UUID string") from exc
    if str(parsed) != value:
        raise InvariantError(f"{field_name} must be a canonical lowercase UUID string")
    return value


def _require_uuid(value: object, field_name: str) -> UUID:
    if type(value) is not UUID:
        raise InvariantError(f"{field_name} must be a UUID")
    return value


def _uuid_from_text(value: object, field_name: str) -> UUID:
    return UUID(_canonical_uuid_text(value, field_name))


def _require_exact_str_tuple(value: object, field_name: str, *, uuid_items: bool = False) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise InvariantError(f"{field_name} must be a tuple[str, ...]")
    result: list[str] = []
    for index, item in enumerate(value):
        parsed = _canonical_uuid_text(item, f"{field_name}[{index}]") if uuid_items else _require_nonempty_str(item, field_name)
        result.append(parsed)
    return tuple(result)


def _str_tuple_from_list(value: object, field_name: str, *, uuid_items: bool = False) -> tuple[str, ...]:
    if type(value) is not list:
        raise InvariantError(f"{field_name} must be a list[str]")
    result: list[str] = []
    for index, item in enumerate(value):
        parsed = _canonical_uuid_text(item, f"{field_name}[{index}]") if uuid_items else _require_nonempty_str(item, field_name)
        result.append(parsed)
    return tuple(result)


def _require_str_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise InvariantError(f"{field_name} must be an exact dict")
    for key in value:
        if type(key) is not str:
            raise InvariantError(f"{field_name} keys must be exact str values")
    return value


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


def _require_guided_sequence(value: Any, field_name: str) -> list[Any]:
    if type(value) is not list:
        raise InvariantError(f"GuidedSession.from_dict: {field_name} must be a list")
    return value


def _chat_turn_from_guided_dict(entry: Any) -> ChatTurn:
    entry = _require_exact_dict(
        entry,
        frozenset(
            {
                "role",
                "content",
                "seq",
                "step",
                "ts_iso",
                "assistant_message_kind",
                "synthetic_failure_reason",
            }
        ),
        "GuidedSession.from_dict: chat_history entry",
    )
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
    assistant_message_kind_raw = entry["assistant_message_kind"]
    if assistant_message_kind_raw is not None and type(assistant_message_kind_raw) is not str:
        raise InvariantError("GuidedSession.from_dict: chat_history.assistant_message_kind must be str or None")
    synthetic_failure_reason_raw = entry["synthetic_failure_reason"]
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

    def __post_init__(self) -> None:
        if type(self.kind) is not TerminalKind:
            raise TypeError("TerminalState.kind must be TerminalKind")
        if self.kind is TerminalKind.COMPLETED:
            if self.reason is not None:
                raise ValueError("TerminalState completed terminal must not carry a reason")
            if type(self.pipeline_yaml) is not str or self.pipeline_yaml == "":
                raise ValueError("TerminalState completed terminal requires non-empty pipeline_yaml")
            return
        if type(self.reason) is not TerminalReason:
            raise ValueError("TerminalState exited terminal requires a reason")
        if self.pipeline_yaml is not None:
            raise ValueError("TerminalState exited terminal must not carry pipeline_yaml")

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
            d = dict(_require_exact_dict(d, frozenset({"kind", "reason", "pipeline_yaml"}), "TerminalState.from_dict"))
            return cls(
                kind=TerminalKind(d["kind"]),
                reason=TerminalReason(d["reason"]) if d["reason"] is not None else None,
                pipeline_yaml=d["pipeline_yaml"],
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvariantError(f"TerminalState.from_dict: malformed record {d!r}") from exc


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One emitted turn + its (optional) user response, recorded for audit."""

    step: GuidedStep
    turn_type: TurnType
    payload_hash: str
    response_hash: str | None
    emitter: Literal["server", "llm"]
    summary: str | None = None

    def __post_init__(self) -> None:
        if type(self.step) is not GuidedStep:
            raise TypeError("TurnRecord.step must be GuidedStep")
        if type(self.turn_type) is not TurnType:
            raise TypeError("TurnRecord.turn_type must be TurnType")
        _require_hash(self.payload_hash, "TurnRecord.payload_hash")
        if self.response_hash is not None:
            _require_hash(self.response_hash, "TurnRecord.response_hash")
        if not _is_exact_str(self.emitter) or self.emitter not in {"server", "llm"}:
            raise InvariantError("TurnRecord.emitter must be 'server' or 'llm'")
        if self.summary is not None:
            if not _is_exact_str(self.summary):
                raise TypeError("TurnRecord.summary must be an exact str or None")
            if len(self.summary) > GUIDED_MAX_TURN_SUMMARY_CHARS:
                raise InvariantError(f"TurnRecord.summary exceeds {GUIDED_MAX_TURN_SUMMARY_CHARS} characters")

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
            d = dict(
                _require_exact_dict(
                    d,
                    frozenset({"step", "turn_type", "payload_hash", "response_hash", "emitter", "summary"}),
                    "TurnRecord.from_dict",
                )
            )
            return cls(
                step=GuidedStep(d["step"]),
                turn_type=TurnType(d["turn_type"]),
                payload_hash=d["payload_hash"],
                response_hash=d["response_hash"],
                emitter=d["emitter"],
                summary=d["summary"],
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvariantError(f"TurnRecord.from_dict: malformed record {d!r}") from exc


@dataclass(frozen=True, slots=True)
class SourceIntent:
    """One pending source workflow, keyed externally by a stable UUID."""

    name: str
    phase: Literal["plugin_selection", "plugin_options", "inspection_review"]
    plugin: str | None
    options: Mapping[str, Any] | None
    inspection_facts: SourceInspectionFacts | None
    observed_columns: Sequence[str]
    sample_rows: Sequence[Mapping[str, Any]]

    def __post_init__(self) -> None:
        _require_nonempty_str(self.name, "SourceIntent.name")
        if self.phase not in {"plugin_selection", "plugin_options", "inspection_review"}:
            raise ValueError("SourceIntent.phase is not in the closed phase vocabulary")
        if self.plugin is not None:
            _require_nonempty_str(self.plugin, "SourceIntent.plugin")
        if self.inspection_facts is not None and type(self.inspection_facts) is not SourceInspectionFacts:
            raise TypeError("SourceIntent.inspection_facts must be SourceInspectionFacts or None")
        sample_rows_value = cast(object, self.sample_rows)
        if not isinstance(sample_rows_value, Sequence) or isinstance(sample_rows_value, (str, bytes, bytearray)):
            raise TypeError("SourceIntent.sample_rows must be a sequence[mapping]")
        if len(sample_rows_value) > GUIDED_JSON_MAX_ITEMS:
            raise InvariantError(f"SourceIntent.sample_rows exceeds the {GUIDED_JSON_MAX_ITEMS}-item limit")
        if any(not isinstance(row, Mapping) for row in self.sample_rows):
            raise TypeError("SourceIntent.sample_rows must contain mappings")
        if self.phase == "plugin_selection":
            if (
                self.plugin is not None
                or self.options is not None
                or self.inspection_facts is not None
                or self.observed_columns
                or self.sample_rows
            ):
                raise ValueError("SourceIntent plugin_selection phase cannot carry later-phase values")
        elif self.phase == "plugin_options":
            if (
                self.plugin is None
                or self.options is not None
                or self.inspection_facts is not None
                or self.observed_columns
                or self.sample_rows
            ):
                raise ValueError("SourceIntent plugin_options phase requires only a selected plugin")
        elif self.plugin is None or self.options is None or self.inspection_facts is None:
            raise ValueError("SourceIntent inspection_review phase requires plugin, options, and inspection_facts")
        json_budget = GuidedJsonBudget()
        if self.options is not None:
            object.__setattr__(
                self,
                "options",
                freeze_guided_json_mapping(self.options, "SourceIntent.options", budget=json_budget),
            )
        object.__setattr__(
            self,
            "observed_columns",
            freeze_guided_str_sequence(self.observed_columns, "SourceIntent.observed_columns", budget=json_budget),
        )
        object.__setattr__(
            self,
            "sample_rows",
            tuple(
                freeze_guided_json_mapping(row, f"SourceIntent.sample_rows[{index}]", budget=json_budget)
                for index, row in enumerate(self.sample_rows)
            ),
        )
        if self.inspection_facts is not None:
            frozen_facts = freeze_guided_json_mapping(
                facts_to_dict(self.inspection_facts),
                "SourceIntent.inspection_facts",
            )
            object.__setattr__(self, "inspection_facts", facts_from_dict(cast(Mapping[str, Any], deep_thaw(frozen_facts))))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-serialisable dict."""
        return {
            "name": self.name,
            "phase": self.phase,
            "plugin": self.plugin,
            "options": deep_thaw(self.options) if self.options is not None else None,
            "inspection_facts": facts_to_dict(self.inspection_facts) if self.inspection_facts is not None else None,
            "observed_columns": list(deep_thaw(self.observed_columns)),
            "sample_rows": [dict(deep_thaw(r)) for r in self.sample_rows],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceIntent:
        """Reconstruct from a plain dict.  Tier 1 strict — crashes on bad data."""
        try:
            d = dict(
                _require_exact_dict(
                    d,
                    frozenset({"name", "phase", "plugin", "options", "inspection_facts", "observed_columns", "sample_rows"}),
                    "SourceIntent.from_dict",
                )
            )
            phase_raw = d["phase"]
            if phase_raw not in {"plugin_selection", "plugin_options", "inspection_review"}:
                raise InvariantError("SourceIntent.phase is not in the closed phase vocabulary")
            options_raw = d["options"]
            if options_raw is not None and type(options_raw) is not dict:
                raise InvariantError("SourceIntent.options must be an exact dict or None")
            inspection_raw = d["inspection_facts"]
            if inspection_raw is not None:
                _require_exact_dict(
                    inspection_raw,
                    frozenset(
                        {
                            "source_kind",
                            "redacted_identity",
                            "byte_range_inspected",
                            "sample_row_count",
                            "observed_headers",
                            "inferred_types",
                            "url_candidates",
                            "warnings",
                        }
                    ),
                    "SourceIntent.inspection_facts",
                )
            observed = _str_tuple_from_list(d["observed_columns"], "SourceIntent.observed_columns")
            samples_raw = d["sample_rows"]
            if type(samples_raw) is not list or any(type(row) is not dict for row in samples_raw):
                raise InvariantError("SourceIntent.sample_rows must be a list[dict]")
            return cls(
                name=_require_nonempty_str(d["name"], "SourceIntent.name"),
                phase=cast(Any, phase_raw),
                plugin=_require_optional_nonempty_str(d["plugin"], "SourceIntent.plugin"),
                options=options_raw,
                inspection_facts=facts_from_dict(inspection_raw) if inspection_raw is not None else None,
                observed_columns=observed,
                sample_rows=tuple(samples_raw),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvariantError(f"SourceIntent.from_dict: malformed record {d!r}") from exc


@dataclass(frozen=True, slots=True)
class SinkIntent:
    """One pending output workflow, keyed externally by a stable UUID."""

    name: str
    phase: Literal["plugin_selection", "plugin_options", "field_review"]
    plugin: str | None
    options: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.name, "SinkIntent.name")
        if self.phase not in {"plugin_selection", "plugin_options", "field_review"}:
            raise ValueError("SinkIntent.phase is not in the closed phase vocabulary")
        if self.plugin is not None:
            _require_nonempty_str(self.plugin, "SinkIntent.plugin")
        if self.phase == "plugin_selection":
            if self.plugin is not None or self.options is not None:
                raise ValueError("SinkIntent plugin_selection phase cannot carry later-phase values")
        elif self.phase == "plugin_options":
            if self.plugin is None or self.options is not None:
                raise ValueError("SinkIntent plugin_options phase requires only a selected plugin")
        elif self.plugin is None or self.options is None:
            raise ValueError("SinkIntent field_review phase requires plugin and options")
        if self.options is not None:
            object.__setattr__(self, "options", freeze_guided_json_mapping(self.options, "SinkIntent.options"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-serialisable dict."""
        return {
            "name": self.name,
            "phase": self.phase,
            "plugin": self.plugin,
            "options": deep_thaw(self.options) if self.options is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SinkIntent:
        """Reconstruct from a plain dict.  Tier 1 strict — crashes on bad data."""
        try:
            d = dict(_require_exact_dict(d, frozenset({"name", "phase", "plugin", "options"}), "SinkIntent.from_dict"))
            phase_raw = d["phase"]
            if phase_raw not in {"plugin_selection", "plugin_options", "field_review"}:
                raise InvariantError("SinkIntent.phase is not in the closed phase vocabulary")
            options_raw = d["options"]
            if options_raw is not None and type(options_raw) is not dict:
                raise InvariantError("SinkIntent.options must be an exact dict or None")
            return cls(
                name=_require_nonempty_str(d["name"], "SinkIntent.name"),
                phase=cast(Any, phase_raw),
                plugin=_require_optional_nonempty_str(d["plugin"], "SinkIntent.plugin"),
                options=options_raw,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvariantError(f"SinkIntent.from_dict: malformed record {d!r}") from exc


@dataclass(frozen=True, slots=True)
class ChainProposal:
    """Temporary non-persisted construction shim for the cutover cohort.

    Schema 8 has no field, encoder, or decoder for this type. Tasks 3/4 replace
    its remaining in-process callers and Task 6 deletes the shim with the old
    protocol arm. It must never regain a persistence path.
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


class GuidedProposalRefData(TypedDict):
    proposal_id: str
    draft_hash: str
    base: dict[str, str]
    reviewed_anchor_hash: str
    covered_deferred_intent_ids: list[str]
    creation_event_schema: str
    supersedes_proposal_id: str | None
    supersedes_draft_hash: str | None


@dataclass(frozen=True, slots=True)
class GuidedProposalRef:
    """Verified safe reference to one private durable pipeline proposal."""

    proposal_id: UUID
    draft_hash: str
    base: ProposalBase
    reviewed_anchor_hash: str
    covered_deferred_intent_ids: tuple[str, ...]
    creation_event_schema: Literal["pipeline_proposal_created.v1"]
    supersedes_proposal_id: UUID | None = None
    supersedes_draft_hash: str | None = None

    def __post_init__(self) -> None:
        _require_uuid(self.proposal_id, "GuidedProposalRef.proposal_id")
        _require_hash(self.draft_hash, "GuidedProposalRef.draft_hash")
        if type(self.base) not in {AbsentBase, PresentBase}:
            raise InvariantError("GuidedProposalRef.base must be AbsentBase or PresentBase")
        _require_hash(self.reviewed_anchor_hash, "GuidedProposalRef.reviewed_anchor_hash")
        covered = _require_exact_str_tuple(
            self.covered_deferred_intent_ids,
            "GuidedProposalRef.covered_deferred_intent_ids",
            uuid_items=True,
        )
        if len(set(covered)) != len(covered):
            raise InvariantError("GuidedProposalRef.covered_deferred_intent_ids must be unique")
        if self.creation_event_schema != "pipeline_proposal_created.v1":
            raise InvariantError("GuidedProposalRef.creation_event_schema is unsupported")
        if (self.supersedes_proposal_id is None) != (self.supersedes_draft_hash is None):
            raise InvariantError("GuidedProposalRef supersedes fields must be paired")
        if self.supersedes_proposal_id is not None:
            _require_uuid(self.supersedes_proposal_id, "GuidedProposalRef.supersedes_proposal_id")
            _require_hash(self.supersedes_draft_hash, "GuidedProposalRef.supersedes_draft_hash")
            if self.supersedes_proposal_id == self.proposal_id:
                raise InvariantError("GuidedProposalRef cannot supersede itself")

    def to_dict(self) -> GuidedProposalRefData:
        if type(self.base) is AbsentBase:
            base: dict[str, str] = {"kind": "absent"}
        elif type(self.base) is PresentBase:
            base = {
                "kind": "present",
                "state_id": str(self.base.state_id),
                "composition_content_hash": self.base.composition_content_hash,
            }
        else:  # pragma: no cover - guarded by __post_init__
            raise InvariantError("GuidedProposalRef.base has an unsupported type")
        return {
            "proposal_id": str(self.proposal_id),
            "draft_hash": self.draft_hash,
            "base": base,
            "reviewed_anchor_hash": self.reviewed_anchor_hash,
            "covered_deferred_intent_ids": list(self.covered_deferred_intent_ids),
            "creation_event_schema": self.creation_event_schema,
            "supersedes_proposal_id": str(self.supersedes_proposal_id) if self.supersedes_proposal_id is not None else None,
            "supersedes_draft_hash": self.supersedes_draft_hash,
        }

    @classmethod
    def from_dict(cls, value: object) -> GuidedProposalRef:
        record = _require_exact_dict(
            value,
            frozenset(
                {
                    "proposal_id",
                    "draft_hash",
                    "base",
                    "reviewed_anchor_hash",
                    "covered_deferred_intent_ids",
                    "creation_event_schema",
                    "supersedes_proposal_id",
                    "supersedes_draft_hash",
                }
            ),
            "GuidedProposalRef.from_dict",
        )
        base_raw = _require_str_mapping(record["base"], "GuidedProposalRef.base")
        kind = base_raw.get("kind")
        if kind == "absent":
            _require_exact_dict(base_raw, frozenset({"kind"}), "GuidedProposalRef.base")
            base: ProposalBase = AbsentBase()
        elif kind == "present":
            _require_exact_dict(
                base_raw,
                frozenset({"kind", "state_id", "composition_content_hash"}),
                "GuidedProposalRef.base",
            )
            base = PresentBase(
                state_id=_uuid_from_text(base_raw["state_id"], "GuidedProposalRef.base.state_id"),
                composition_content_hash=_require_hash(
                    base_raw["composition_content_hash"],
                    "GuidedProposalRef.base.composition_content_hash",
                ),
            )
        else:
            raise InvariantError("GuidedProposalRef.base.kind is unsupported")
        supersedes_raw = record["supersedes_proposal_id"]
        creation_schema = record["creation_event_schema"]
        if creation_schema != "pipeline_proposal_created.v1":
            raise InvariantError("GuidedProposalRef.creation_event_schema is unsupported")
        return cls(
            proposal_id=_uuid_from_text(record["proposal_id"], "GuidedProposalRef.proposal_id"),
            draft_hash=_require_hash(record["draft_hash"], "GuidedProposalRef.draft_hash"),
            base=base,
            reviewed_anchor_hash=_require_hash(record["reviewed_anchor_hash"], "GuidedProposalRef.reviewed_anchor_hash"),
            covered_deferred_intent_ids=_str_tuple_from_list(
                record["covered_deferred_intent_ids"],
                "GuidedProposalRef.covered_deferred_intent_ids",
                uuid_items=True,
            ),
            creation_event_schema=cast(Any, creation_schema),
            supersedes_proposal_id=(
                _uuid_from_text(supersedes_raw, "GuidedProposalRef.supersedes_proposal_id") if supersedes_raw is not None else None
            ),
            supersedes_draft_hash=(
                _require_hash(record["supersedes_draft_hash"], "GuidedProposalRef.supersedes_draft_hash")
                if record["supersedes_draft_hash"] is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ComponentTarget:
    kind: Literal["source", "node", "edge", "output"]
    stable_id: str

    def __post_init__(self) -> None:
        if self.kind not in {"source", "node", "edge", "output"}:
            raise InvariantError("ComponentTarget.kind is unsupported")
        _canonical_uuid_text(self.stable_id, "ComponentTarget.stable_id")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "stable_id": self.stable_id}

    @classmethod
    def from_dict(cls, value: object) -> ComponentTarget:
        record = _require_exact_dict(value, frozenset({"kind", "stable_id"}), "ComponentTarget.from_dict")
        kind = record["kind"]
        if kind not in {"source", "node", "edge", "output"}:
            raise InvariantError("ComponentTarget.kind is unsupported")
        return cls(kind=cast(Any, kind), stable_id=_canonical_uuid_text(record["stable_id"], "ComponentTarget.stable_id"))


@dataclass(frozen=True, slots=True)
class StableSubject:
    kind: Literal["stable"]
    component_kind: Literal["source", "node", "edge", "output"]
    stable_id: str

    def __post_init__(self) -> None:
        if self.kind != "stable":
            raise InvariantError("StableSubject.kind must be 'stable'")
        if self.component_kind not in {"source", "node", "edge", "output"}:
            raise InvariantError("StableSubject.component_kind is unsupported")
        _canonical_uuid_text(self.stable_id, "StableSubject.stable_id")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "component_kind": self.component_kind, "stable_id": self.stable_id}


@dataclass(frozen=True, slots=True)
class PluginSubject:
    kind: Literal["plugin"]
    subject_id: str
    plugin_kind: Literal["source", "transform", "sink"]
    plugin_name: str

    def __post_init__(self) -> None:
        if self.kind != "plugin":
            raise InvariantError("PluginSubject.kind must be 'plugin'")
        _canonical_uuid_text(self.subject_id, "PluginSubject.subject_id")
        if self.plugin_kind not in {"source", "transform", "sink"}:
            raise InvariantError("PluginSubject.plugin_kind is unsupported")
        _require_nonempty_str(self.plugin_name, "PluginSubject.plugin_name")

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "subject_id": self.subject_id,
            "plugin_kind": self.plugin_kind,
            "plugin_name": self.plugin_name,
        }


type DeferredSubject = StableSubject | PluginSubject


class SubjectPresenceConstraintData(TypedDict):
    kind: str
    subject: dict[str, str]
    present: bool


def _subject_from_dict(value: object) -> DeferredSubject:
    if type(value) is not dict:
        raise InvariantError("Deferred subject must be an exact dict")
    kind = value.get("kind")
    if kind == "stable":
        record = _require_exact_dict(value, frozenset({"kind", "component_kind", "stable_id"}), "StableSubject.from_dict")
        component_kind = record["component_kind"]
        if component_kind not in {"source", "node", "edge", "output"}:
            raise InvariantError("StableSubject.component_kind is unsupported")
        return StableSubject(
            kind="stable",
            component_kind=cast(Any, component_kind),
            stable_id=_canonical_uuid_text(record["stable_id"], "StableSubject.stable_id"),
        )
    if kind == "plugin":
        record = _require_exact_dict(
            value,
            frozenset({"kind", "subject_id", "plugin_kind", "plugin_name"}),
            "PluginSubject.from_dict",
        )
        plugin_kind = record["plugin_kind"]
        if plugin_kind not in {"source", "transform", "sink"}:
            raise InvariantError("PluginSubject.plugin_kind is unsupported")
        return PluginSubject(
            kind="plugin",
            subject_id=_canonical_uuid_text(record["subject_id"], "PluginSubject.subject_id"),
            plugin_kind=cast(Any, plugin_kind),
            plugin_name=_require_nonempty_str(record["plugin_name"], "PluginSubject.plugin_name"),
        )
    raise InvariantError("Deferred subject kind is unsupported")


@dataclass(frozen=True, slots=True)
class SubjectPresenceConstraint:
    kind: Literal["subject_presence"]
    subject: DeferredSubject
    present: bool

    def __post_init__(self) -> None:
        if self.kind != "subject_presence" or type(self.subject) not in {StableSubject, PluginSubject} or type(self.present) is not bool:
            raise InvariantError("SubjectPresenceConstraint is malformed")

    def to_dict(self) -> SubjectPresenceConstraintData:
        return {"kind": self.kind, "subject": self.subject.to_dict(), "present": self.present}


type JsonScalar = str | int | float | bool | None


class OptionValueConstraintData(TypedDict):
    kind: str
    subject: dict[str, str]
    option_path: list[str]
    operator: str
    value: JsonScalar


def _require_json_scalar(value: object, field_name: str) -> JsonScalar:
    if type(value) not in {str, int, float, bool, type(None)}:
        raise InvariantError(f"{field_name} must be a strict JSON scalar")
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise InvariantError(f"{field_name} must be in the canonical JSON number domain") from exc
    if type(value) is str and len(value) > 65_536:
        raise InvariantError(f"{field_name} JSON string exceeds 65536 characters")
    return cast(JsonScalar, value)


def _subject_component_kind(subject: DeferredSubject) -> Literal["source", "node", "edge", "output"]:
    if type(subject) is StableSubject:
        return subject.component_kind
    if type(subject) is PluginSubject:
        return cast(
            Literal["source", "node", "edge", "output"],
            {"source": "source", "transform": "node", "sink": "output"}[subject.plugin_kind],
        )
    raise InvariantError("Deferred subject is malformed")


@dataclass(frozen=True, slots=True)
class OptionValueConstraint:
    kind: Literal["option_value"]
    subject: DeferredSubject
    option_path: tuple[str, ...]
    operator: Literal["equals", "not_equals"]
    value: JsonScalar

    def __post_init__(self) -> None:
        if self.kind != "option_value" or type(self.subject) not in {StableSubject, PluginSubject}:
            raise InvariantError("OptionValueConstraint is malformed")
        if type(self.option_path) is not tuple or not 1 <= len(self.option_path) <= 16:
            raise InvariantError("OptionValueConstraint.option_path must contain 1 to 16 segments")
        for segment in self.option_path:
            if type(segment) is not str or not segment or len(segment) > 128:
                raise InvariantError("OptionValueConstraint.option_path segments must be 1 to 128 characters")
        if self.operator not in {"equals", "not_equals"}:
            raise InvariantError("OptionValueConstraint.operator is unsupported")
        if _subject_component_kind(self.subject) == "edge":
            raise InvariantError("OptionValueConstraint subject cannot be an edge")
        _require_json_scalar(self.value, "OptionValueConstraint.value")

    def to_dict(self) -> OptionValueConstraintData:
        return {
            "kind": self.kind,
            "subject": self.subject.to_dict(),
            "option_path": list(self.option_path),
            "operator": self.operator,
            "value": self.value,
        }


class ComponentCountConstraintData(TypedDict):
    kind: str
    component_kind: str
    plugin_kind: str | None
    plugin_name: str | None
    operator: str
    count: int


@dataclass(frozen=True, slots=True)
class ComponentCountConstraint:
    kind: Literal["component_count"]
    component_kind: Literal["source", "node", "edge", "output"]
    plugin_kind: Literal["source", "transform", "sink"] | None
    plugin_name: str | None
    operator: Literal["equals", "at_least", "at_most"]
    count: int

    def __post_init__(self) -> None:
        if self.kind != "component_count" or self.component_kind not in {"source", "node", "edge", "output"}:
            raise InvariantError("ComponentCountConstraint component kind is unsupported")
        if (self.plugin_kind is None) != (self.plugin_name is None):
            raise InvariantError("ComponentCountConstraint plugin_kind/plugin_name must be paired")
        if self.plugin_kind is not None and self.plugin_kind not in {"source", "transform", "sink"}:
            raise InvariantError("ComponentCountConstraint.plugin_kind is unsupported")
        expected_plugin_kind = {"source": "source", "node": "transform", "output": "sink"}
        if self.component_kind == "edge" and self.plugin_kind is not None:
            raise InvariantError("ComponentCountConstraint edge counts cannot carry plugin_kind/plugin_name")
        if self.component_kind != "edge" and self.plugin_kind is not None and self.plugin_kind != expected_plugin_kind[self.component_kind]:
            raise InvariantError("ComponentCountConstraint.plugin_kind is incompatible with component_kind")
        if self.plugin_name is not None:
            _require_nonempty_str(self.plugin_name, "ComponentCountConstraint.plugin_name")
        if self.operator not in {"equals", "at_least", "at_most"}:
            raise InvariantError("ComponentCountConstraint.operator is unsupported")
        if type(self.count) is not int or self.count < 0:
            raise InvariantError("ComponentCountConstraint.count must be a non-negative exact int")

    def to_dict(self) -> ComponentCountConstraintData:
        return {
            "kind": self.kind,
            "component_kind": self.component_kind,
            "plugin_kind": self.plugin_kind,
            "plugin_name": self.plugin_name,
            "operator": self.operator,
            "count": self.count,
        }


class EdgeRouteConstraintData(TypedDict):
    kind: str
    from_subject: dict[str, str]
    edge_type: str
    to_subject: dict[str, str]
    present: bool


@dataclass(frozen=True, slots=True)
class EdgeRouteConstraint:
    kind: Literal["edge_route"]
    from_subject: DeferredSubject
    edge_type: Literal["on_success", "on_error", "route_true", "route_false", "fork"]
    to_subject: DeferredSubject
    present: bool

    def __post_init__(self) -> None:
        if self.kind != "edge_route":
            raise InvariantError("EdgeRouteConstraint.kind must be 'edge_route'")
        if type(self.from_subject) not in {StableSubject, PluginSubject} or type(self.to_subject) not in {StableSubject, PluginSubject}:
            raise InvariantError("EdgeRouteConstraint subjects are malformed")
        if self.edge_type not in {"on_success", "on_error", "route_true", "route_false", "fork"}:
            raise InvariantError("EdgeRouteConstraint.edge_type is unsupported")
        if _subject_component_kind(self.from_subject) not in {"source", "node"}:
            raise InvariantError("EdgeRouteConstraint.from_subject must identify a source or node")
        if _subject_component_kind(self.to_subject) not in {"node", "output"}:
            raise InvariantError("EdgeRouteConstraint.to_subject must identify a node or output")
        if type(self.present) is not bool:
            raise InvariantError("EdgeRouteConstraint.present must be an exact bool")

    def to_dict(self) -> EdgeRouteConstraintData:
        return {
            "kind": self.kind,
            "from_subject": self.from_subject.to_dict(),
            "edge_type": self.edge_type,
            "to_subject": self.to_subject.to_dict(),
            "present": self.present,
        }


type FailureTarget = Literal["discard"] | StableSubject | PluginSubject


class FailureRouteConstraintData(TypedDict):
    kind: str
    subject: dict[str, str]
    failure_kind: str
    operator: str
    target: str | dict[str, str]


@dataclass(frozen=True, slots=True)
class FailureRouteConstraint:
    kind: Literal["failure_route"]
    subject: DeferredSubject
    failure_kind: Literal["source_validation", "node_error", "output_write"]
    operator: Literal["equals", "not_equals"]
    target: FailureTarget

    def __post_init__(self) -> None:
        if self.kind != "failure_route" or type(self.subject) not in {StableSubject, PluginSubject}:
            raise InvariantError("FailureRouteConstraint is malformed")
        if self.failure_kind not in {"source_validation", "node_error", "output_write"}:
            raise InvariantError("FailureRouteConstraint.failure_kind is unsupported")
        expected_subject_kind = {
            "source_validation": "source",
            "node_error": "node",
            "output_write": "output",
        }[self.failure_kind]
        if _subject_component_kind(self.subject) != expected_subject_kind:
            raise InvariantError("FailureRouteConstraint.failure_kind is incompatible with subject")
        if self.operator not in {"equals", "not_equals"}:
            raise InvariantError("FailureRouteConstraint.operator is unsupported")
        if self.target != "discard" and type(self.target) not in {StableSubject, PluginSubject}:
            raise InvariantError("FailureRouteConstraint.target must be 'discard' or a closed subject")
        if self.target != "discard" and _subject_component_kind(self.target) != "output":
            raise InvariantError("FailureRouteConstraint.target subject must identify an output")

    def to_dict(self) -> FailureRouteConstraintData:
        return {
            "kind": self.kind,
            "subject": self.subject.to_dict(),
            "failure_kind": self.failure_kind,
            "operator": self.operator,
            "target": self.target if self.target == "discard" else self.target.to_dict(),
        }


type DeferredConstraint = (
    SubjectPresenceConstraint | OptionValueConstraint | ComponentCountConstraint | EdgeRouteConstraint | FailureRouteConstraint
)
type DeferredConstraintData = (
    SubjectPresenceConstraintData
    | OptionValueConstraintData
    | ComponentCountConstraintData
    | EdgeRouteConstraintData
    | FailureRouteConstraintData
)


def _constraint_from_dict(value: object) -> DeferredConstraint:
    if type(value) is not dict:
        raise InvariantError("Deferred constraint must be an exact dict")
    kind = value.get("kind")
    if kind == "subject_presence":
        record = _require_exact_dict(value, frozenset({"kind", "subject", "present"}), "SubjectPresenceConstraint.from_dict")
        if type(record["present"]) is not bool:
            raise InvariantError("SubjectPresenceConstraint.present must be an exact bool")
        return SubjectPresenceConstraint(kind="subject_presence", subject=_subject_from_dict(record["subject"]), present=record["present"])
    if kind == "option_value":
        record = _require_exact_dict(
            value,
            frozenset({"kind", "subject", "option_path", "operator", "value"}),
            "OptionValueConstraint.from_dict",
        )
        operator = record["operator"]
        if operator not in {"equals", "not_equals"}:
            raise InvariantError("OptionValueConstraint.operator is unsupported")
        return OptionValueConstraint(
            kind="option_value",
            subject=_subject_from_dict(record["subject"]),
            option_path=_str_tuple_from_list(record["option_path"], "OptionValueConstraint.option_path"),
            operator=cast(Any, operator),
            value=_require_json_scalar(record["value"], "OptionValueConstraint.value"),
        )
    if kind == "component_count":
        record = _require_exact_dict(
            value,
            frozenset({"kind", "component_kind", "plugin_kind", "plugin_name", "operator", "count"}),
            "ComponentCountConstraint.from_dict",
        )
        return ComponentCountConstraint(
            kind="component_count",
            component_kind=cast(Any, record["component_kind"]),
            plugin_kind=cast(Any, record["plugin_kind"]),
            plugin_name=_require_optional_nonempty_str(record["plugin_name"], "ComponentCountConstraint.plugin_name"),
            operator=cast(Any, record["operator"]),
            count=record["count"],
        )
    if kind == "edge_route":
        record = _require_exact_dict(
            value,
            frozenset({"kind", "from_subject", "edge_type", "to_subject", "present"}),
            "EdgeRouteConstraint.from_dict",
        )
        return EdgeRouteConstraint(
            kind="edge_route",
            from_subject=_subject_from_dict(record["from_subject"]),
            edge_type=cast(Any, record["edge_type"]),
            to_subject=_subject_from_dict(record["to_subject"]),
            present=record["present"],
        )
    if kind == "failure_route":
        record = _require_exact_dict(
            value,
            frozenset({"kind", "subject", "failure_kind", "operator", "target"}),
            "FailureRouteConstraint.from_dict",
        )
        target_raw = record["target"]
        target: FailureTarget
        if target_raw == "discard":
            target = "discard"
        else:
            target = _subject_from_dict(target_raw)
        return FailureRouteConstraint(
            kind="failure_route",
            subject=_subject_from_dict(record["subject"]),
            failure_kind=cast(Any, record["failure_kind"]),
            operator=cast(Any, record["operator"]),
            target=target,
        )
    raise InvariantError("Deferred constraint kind is unsupported")


_STAGE_ORDINAL = {"source": 0, "output": 1, "topology": 2, "wire_review": 3}


class DeferredStageIntentData(TypedDict):
    intent_id: str
    receiving_stage: str
    target_stage: str
    catalog_kind: str | None
    catalog_name: str | None
    redacted_summary: str
    summary_hash: str
    originating_message_id: str
    message_content_hash: str
    constraints: list[DeferredConstraintData]


@dataclass(frozen=True, slots=True)
class DeferredStageIntent:
    intent_id: str
    receiving_stage: Literal["source", "output", "topology", "wire_review"]
    target_stage: Literal["source", "output", "topology", "wire_review"]
    catalog_kind: Literal["source", "transform", "sink"] | None
    catalog_name: str | None
    redacted_summary: str
    summary_hash: str
    originating_message_id: str
    message_content_hash: str
    constraints: tuple[DeferredConstraint, ...]

    def __post_init__(self) -> None:
        _canonical_uuid_text(self.intent_id, "DeferredStageIntent.intent_id")
        if self.receiving_stage not in _STAGE_ORDINAL or self.target_stage not in _STAGE_ORDINAL:
            raise InvariantError("DeferredStageIntent stage is unsupported")
        if _STAGE_ORDINAL[self.receiving_stage] >= _STAGE_ORDINAL[self.target_stage]:
            raise InvariantError("DeferredStageIntent must move forward to a later stage")
        if (self.catalog_kind is None) != (self.catalog_name is None):
            raise InvariantError("DeferredStageIntent catalog fields must be paired")
        if self.catalog_kind is not None and self.catalog_kind not in {"source", "transform", "sink"}:
            raise InvariantError("DeferredStageIntent.catalog_kind is unsupported")
        if self.catalog_name is not None:
            _require_nonempty_str(self.catalog_name, "DeferredStageIntent.catalog_name")
        _require_nonempty_str(self.redacted_summary, "DeferredStageIntent.redacted_summary")
        if len(self.redacted_summary) > GUIDED_MAX_REDACTED_SUMMARY_CHARS:
            raise InvariantError(f"DeferredStageIntent.redacted_summary exceeds {GUIDED_MAX_REDACTED_SUMMARY_CHARS} characters")
        expected_summary_hash = stable_hash({"schema": "guided.deferred-summary.v1", "summary": self.redacted_summary})
        if self.summary_hash != expected_summary_hash:
            raise InvariantError("DeferredStageIntent.summary_hash mismatch")
        _canonical_uuid_text(self.originating_message_id, "DeferredStageIntent.originating_message_id")
        _require_hash(self.message_content_hash, "DeferredStageIntent.message_content_hash")
        if type(self.constraints) is not tuple:
            raise InvariantError("DeferredStageIntent.constraints must be a tuple")
        if len(self.constraints) > GUIDED_MAX_CONSTRAINTS_PER_INTENT:
            raise InvariantError(f"DeferredStageIntent.constraints exceeds the {GUIDED_MAX_CONSTRAINTS_PER_INTENT}-constraint limit")
        allowed = {
            SubjectPresenceConstraint,
            OptionValueConstraint,
            ComponentCountConstraint,
            EdgeRouteConstraint,
            FailureRouteConstraint,
        }
        if any(type(constraint) not in allowed for constraint in self.constraints):
            raise InvariantError("DeferredStageIntent.constraints contains an unsupported constraint")
        freeze_fields(self, "constraints")

    @classmethod
    def create(
        cls,
        *,
        intent_id: str,
        receiving_stage: Literal["source", "output", "topology", "wire_review"],
        target_stage: Literal["source", "output", "topology", "wire_review"],
        catalog_kind: Literal["source", "transform", "sink"] | None,
        catalog_name: str | None,
        redacted_summary: str,
        originating_message_id: str,
        message_content_hash: str,
        constraints: tuple[DeferredConstraint, ...],
    ) -> DeferredStageIntent:
        return cls(
            intent_id=intent_id,
            receiving_stage=receiving_stage,
            target_stage=target_stage,
            catalog_kind=catalog_kind,
            catalog_name=catalog_name,
            redacted_summary=redacted_summary,
            summary_hash=stable_hash({"schema": "guided.deferred-summary.v1", "summary": redacted_summary}),
            originating_message_id=originating_message_id,
            message_content_hash=message_content_hash,
            constraints=constraints,
        )

    def to_dict(self) -> DeferredStageIntentData:
        return {
            "intent_id": self.intent_id,
            "receiving_stage": self.receiving_stage,
            "target_stage": self.target_stage,
            "catalog_kind": self.catalog_kind,
            "catalog_name": self.catalog_name,
            "redacted_summary": self.redacted_summary,
            "summary_hash": self.summary_hash,
            "originating_message_id": self.originating_message_id,
            "message_content_hash": self.message_content_hash,
            "constraints": [constraint.to_dict() for constraint in self.constraints],
        }

    @classmethod
    def from_dict(cls, value: object) -> DeferredStageIntent:
        record = _require_exact_dict(
            value,
            frozenset(
                {
                    "intent_id",
                    "receiving_stage",
                    "target_stage",
                    "catalog_kind",
                    "catalog_name",
                    "redacted_summary",
                    "summary_hash",
                    "originating_message_id",
                    "message_content_hash",
                    "constraints",
                }
            ),
            "DeferredStageIntent.from_dict",
        )
        constraints_raw = record["constraints"]
        if type(constraints_raw) is not list:
            raise InvariantError("DeferredStageIntent.constraints must be a list")
        if len(constraints_raw) > GUIDED_MAX_CONSTRAINTS_PER_INTENT:
            raise InvariantError(f"DeferredStageIntent.constraints exceeds the {GUIDED_MAX_CONSTRAINTS_PER_INTENT}-constraint limit")
        return cls(
            intent_id=_canonical_uuid_text(record["intent_id"], "DeferredStageIntent.intent_id"),
            receiving_stage=cast(Any, record["receiving_stage"]),
            target_stage=cast(Any, record["target_stage"]),
            catalog_kind=cast(Any, record["catalog_kind"]),
            catalog_name=_require_optional_nonempty_str(record["catalog_name"], "DeferredStageIntent.catalog_name"),
            redacted_summary=_require_nonempty_str(record["redacted_summary"], "DeferredStageIntent.redacted_summary"),
            summary_hash=_require_hash(record["summary_hash"], "DeferredStageIntent.summary_hash"),
            originating_message_id=_canonical_uuid_text(record["originating_message_id"], "DeferredStageIntent.originating_message_id"),
            message_content_hash=_require_hash(record["message_content_hash"], "DeferredStageIntent.message_content_hash"),
            constraints=tuple(_constraint_from_dict(constraint) for constraint in constraints_raw),
        )


def guided_reviewed_anchor_hash(
    *,
    source_order: tuple[str, ...],
    reviewed_sources: Mapping[str, SourceResolved],
    output_order: tuple[str, ...],
    reviewed_outputs: Mapping[str, SinkOutputResolved],
) -> str:
    """Hash the exact ordered reviewed facts bound by a guided proposal."""

    reviewed_sources_snapshot = dict(reviewed_sources.items())
    reviewed_outputs_snapshot = dict(reviewed_outputs.items())
    validated_source_order = _require_exact_str_tuple(source_order, "guided_reviewed_anchor_hash.source_order", uuid_items=True)
    validated_output_order = _require_exact_str_tuple(output_order, "guided_reviewed_anchor_hash.output_order", uuid_items=True)
    if len(set(validated_source_order)) != len(validated_source_order):
        raise InvariantError("guided_reviewed_anchor_hash.source_order must not contain duplicates")
    if len(set(validated_output_order)) != len(validated_output_order):
        raise InvariantError("guided_reviewed_anchor_hash.output_order must not contain duplicates")
    if set(validated_source_order) != set(reviewed_sources_snapshot):
        raise InvariantError("guided_reviewed_anchor_hash.source_order must exactly match reviewed_sources")
    if set(validated_output_order) != set(reviewed_outputs_snapshot):
        raise InvariantError("guided_reviewed_anchor_hash.output_order must exactly match reviewed_outputs")
    if any(type(source) is not SourceResolved for source in reviewed_sources_snapshot.values()):
        raise InvariantError("guided_reviewed_anchor_hash.reviewed_sources values must be SourceResolved")
    if any(type(output) is not SinkOutputResolved for output in reviewed_outputs_snapshot.values()):
        raise InvariantError("guided_reviewed_anchor_hash.reviewed_outputs values must be SinkOutputResolved")

    facts = {
        "source_order": list(validated_source_order),
        "reviewed_sources": {stable_id: reviewed_sources_snapshot[stable_id].to_dict() for stable_id in validated_source_order},
        "output_order": list(validated_output_order),
        "reviewed_outputs": {stable_id: reviewed_outputs_snapshot[stable_id].to_dict() for stable_id in validated_output_order},
    }
    return reviewed_anchor_hash(facts)


@dataclass(frozen=True, slots=True)
class GuidedSession:
    """The only persisted guided checkpoint shape: closed schema version 8."""

    step: GuidedStep
    history: tuple[TurnRecord, ...] = ()
    profile: WorkflowProfile = EMPTY_PROFILE
    advisor_checkpoint_passes_used: int = 0
    advisor_signoff_escape_offered: bool = False
    terminal: TerminalState | None = None
    transition_consumed: bool = False
    chat_history: tuple[ChatTurn, ...] = ()
    chat_turn_seq: int = 0
    source_order: tuple[str, ...] = ()
    reviewed_sources: Mapping[str, SourceResolved] = field(default_factory=dict)
    pending_source_intents: Mapping[str, SourceIntent] = field(default_factory=dict)
    output_order: tuple[str, ...] = ()
    reviewed_outputs: Mapping[str, SinkOutputResolved] = field(default_factory=dict)
    pending_output_intents: Mapping[str, SinkIntent] = field(default_factory=dict)
    deferred_intents: tuple[DeferredStageIntent, ...] = ()
    active_proposal: GuidedProposalRef | None = None
    active_edit_target: ComponentTarget | None = None
    root_intent_message_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.step) is not GuidedStep:
            raise TypeError(f"step must be GuidedStep, got {type(self.step).__name__}")
        if type(self.history) is not tuple:
            raise TypeError("history must be tuple[TurnRecord, ...]")
        if len(self.history) > GUIDED_MAX_HISTORY_RECORDS:
            raise InvariantError(f"GuidedSession.history exceeds the {GUIDED_MAX_HISTORY_RECORDS}-record limit")
        if any(type(record) is not TurnRecord for record in self.history):
            raise TypeError("history must be tuple[TurnRecord, ...]")
        if sum(len(record.summary) for record in self.history if record.summary is not None) > GUIDED_MAX_HISTORY_SUMMARY_CHARS:
            raise InvariantError(f"GuidedSession.history aggregate summaries exceed {GUIDED_MAX_HISTORY_SUMMARY_CHARS} characters")
        if type(self.profile) is not WorkflowProfile:
            raise TypeError(f"profile must be WorkflowProfile, got {type(self.profile).__name__}")
        if type(self.advisor_checkpoint_passes_used) is not int or self.advisor_checkpoint_passes_used < 0:
            raise TypeError("advisor_checkpoint_passes_used must be a non-negative int")
        if type(self.advisor_signoff_escape_offered) is not bool:
            raise TypeError(f"advisor_signoff_escape_offered must be bool, got {type(self.advisor_signoff_escape_offered).__name__}")
        if self.terminal is not None and type(self.terminal) is not TerminalState:
            raise TypeError("terminal must be TerminalState or None")
        if type(self.transition_consumed) is not bool:
            raise TypeError("transition_consumed must be bool")
        if type(self.chat_history) is not tuple:
            raise TypeError("chat_history must be tuple[ChatTurn, ...]")
        if len(self.chat_history) > GUIDED_MAX_CHAT_TURNS:
            raise InvariantError(f"GuidedSession.chat_history exceeds the {GUIDED_MAX_CHAT_TURNS}-turn limit")
        if any(type(turn) is not ChatTurn for turn in self.chat_history):
            raise TypeError("chat_history must be tuple[ChatTurn, ...]")
        if type(self.chat_turn_seq) is not int or self.chat_turn_seq < 0:
            raise TypeError("chat_turn_seq must be a non-negative exact int")
        previous_chat_seq = -1
        for turn in self.chat_history:
            if turn.seq <= previous_chat_seq:
                raise InvariantError("GuidedSession chat_history seq values must be unique and strictly increasing")
            if len(turn.content) > GUIDED_MAX_CHAT_CONTENT_CHARS:
                raise InvariantError(f"GuidedSession chat content exceeds {GUIDED_MAX_CHAT_CONTENT_CHARS} characters")
            previous_chat_seq = turn.seq
        if sum(len(turn.content) for turn in self.chat_history) > GUIDED_MAX_CHAT_HISTORY_CHARS:
            raise InvariantError(f"GuidedSession.chat_history aggregate content exceeds {GUIDED_MAX_CHAT_HISTORY_CHARS} characters")
        expected_chat_turn_seq = self.chat_history[-1].seq + 1 if self.chat_history else 0
        if self.chat_turn_seq != expected_chat_turn_seq:
            raise InvariantError("GuidedSession.chat_turn_seq must be the exact next unused persisted chat seq")
        if self.root_intent_message_id is not None:
            _canonical_uuid_text(self.root_intent_message_id, "GuidedSession.root_intent_message_id")

        for mapping, field_name in (
            (self.reviewed_sources, "reviewed_sources"),
            (self.pending_source_intents, "pending_source_intents"),
            (self.reviewed_outputs, "reviewed_outputs"),
            (self.pending_output_intents, "pending_output_intents"),
        ):
            if not isinstance(mapping, Mapping):
                raise TypeError(f"GuidedSession.{field_name} must be a mapping")
        if len(self.reviewed_sources) + len(self.pending_source_intents) > GUIDED_MAX_COMPONENTS_PER_KIND:
            raise InvariantError(f"GuidedSession source components exceed the {GUIDED_MAX_COMPONENTS_PER_KIND}-component limit")
        if len(self.reviewed_outputs) + len(self.pending_output_intents) > GUIDED_MAX_COMPONENTS_PER_KIND:
            raise InvariantError(f"GuidedSession output components exceed the {GUIDED_MAX_COMPONENTS_PER_KIND}-component limit")

        if type(self.source_order) is not tuple:
            raise TypeError("GuidedSession.source_order must be an exact tuple")
        if type(self.output_order) is not tuple:
            raise TypeError("GuidedSession.output_order must be an exact tuple")
        if len(self.source_order) > GUIDED_MAX_COMPONENTS_PER_KIND:
            raise InvariantError(f"GuidedSession source components exceed the {GUIDED_MAX_COMPONENTS_PER_KIND}-component limit")
        if len(self.output_order) > GUIDED_MAX_COMPONENTS_PER_KIND:
            raise InvariantError(f"GuidedSession output components exceed the {GUIDED_MAX_COMPONENTS_PER_KIND}-component limit")
        source_order = _require_exact_str_tuple(self.source_order, "GuidedSession.source_order", uuid_items=True)
        output_order = _require_exact_str_tuple(self.output_order, "GuidedSession.output_order", uuid_items=True)
        if len(set(source_order)) != len(source_order):
            raise InvariantError("GuidedSession.source_order must not contain duplicates")
        if len(set(output_order)) != len(output_order):
            raise InvariantError("GuidedSession.output_order must not contain duplicates")

        reviewed_sources = self._validated_component_mapping(self.reviewed_sources, SourceResolved, "reviewed_sources")
        pending_sources = self._validated_component_mapping(self.pending_source_intents, SourceIntent, "pending_source_intents")
        reviewed_outputs = self._validated_component_mapping(self.reviewed_outputs, SinkOutputResolved, "reviewed_outputs")
        pending_outputs = self._validated_component_mapping(self.pending_output_intents, SinkIntent, "pending_output_intents")
        # Validate and freeze the same detached snapshots. A caller-supplied
        # Mapping may be a read-only view over mutable storage; retaining it
        # after validation would create a check/use race at this Tier-1 seam.
        object.__setattr__(self, "reviewed_sources", reviewed_sources)
        object.__setattr__(self, "pending_source_intents", pending_sources)
        object.__setattr__(self, "reviewed_outputs", reviewed_outputs)
        object.__setattr__(self, "pending_output_intents", pending_outputs)
        if set(reviewed_sources) & set(pending_sources):
            raise InvariantError("GuidedSession reviewed and pending source keysets must be disjoint")
        if set(reviewed_outputs) & set(pending_outputs):
            raise InvariantError("GuidedSession reviewed and pending output keysets must be disjoint")
        if set(source_order) != set(reviewed_sources) | set(pending_sources):
            raise InvariantError("GuidedSession.source_order must be an exact permutation of source keys")
        if set(output_order) != set(reviewed_outputs) | set(pending_outputs):
            raise InvariantError("GuidedSession.output_order must be an exact permutation of output keys")
        if (set(reviewed_sources) | set(pending_sources)) & (set(reviewed_outputs) | set(pending_outputs)):
            raise InvariantError("GuidedSession stable component IDs must be globally unique")
        source_names = [source.name for source in reviewed_sources.values()] + [intent.name for intent in pending_sources.values()]
        output_names = [output.name for output in reviewed_outputs.values()] + [intent.name for intent in pending_outputs.values()]
        if len(set(source_names)) != len(source_names):
            raise InvariantError("GuidedSession source names must be unique")
        if len(set(output_names)) != len(output_names):
            raise InvariantError("GuidedSession output names must be unique")

        if type(self.deferred_intents) is not tuple or any(type(intent) is not DeferredStageIntent for intent in self.deferred_intents):
            raise TypeError("deferred_intents must be tuple[DeferredStageIntent, ...]")
        if len(self.deferred_intents) > GUIDED_MAX_DEFERRED_INTENTS:
            raise InvariantError(f"GuidedSession deferred_intents exceed the {GUIDED_MAX_DEFERRED_INTENTS}-intent limit")
        if sum(len(intent.constraints) for intent in self.deferred_intents) > GUIDED_MAX_TOTAL_CONSTRAINTS:
            raise InvariantError(f"GuidedSession deferred constraints exceed the {GUIDED_MAX_TOTAL_CONSTRAINTS}-constraint limit")
        deferred_ids = [intent.intent_id for intent in self.deferred_intents]
        if len(set(deferred_ids)) != len(deferred_ids):
            raise InvariantError("GuidedSession deferred intent IDs must be unique")

        if self.active_proposal is not None:
            if type(self.active_proposal) is not GuidedProposalRef:
                raise TypeError("active_proposal must be GuidedProposalRef or None")
            if pending_sources or pending_outputs:
                raise InvariantError("GuidedSession active_proposal cannot coexist with pending source/output intents")
            expected_anchor = guided_reviewed_anchor_hash(
                source_order=source_order,
                reviewed_sources=reviewed_sources,
                output_order=output_order,
                reviewed_outputs=reviewed_outputs,
            )
            if self.active_proposal.reviewed_anchor_hash != expected_anchor:
                raise InvariantError("GuidedSession active_proposal reviewed_anchor_hash mismatch")
            positions = {intent_id: index for index, intent_id in enumerate(deferred_ids)}
            previous = -1
            for intent_id in self.active_proposal.covered_deferred_intent_ids:
                if intent_id not in positions or positions[intent_id] <= previous:
                    raise InvariantError("GuidedSession active_proposal covered_deferred_intent_ids must be an ordered subsequence")
                if not self.deferred_intents[positions[intent_id]].constraints:
                    raise InvariantError("GuidedSession active_proposal cannot cover a deferred intent with empty constraints")
                previous = positions[intent_id]

        if self.active_edit_target is not None:
            if type(self.active_edit_target) is not ComponentTarget:
                raise TypeError("active_edit_target must be ComponentTarget or None")
            target = self.active_edit_target
            if target.kind == "source" and target.stable_id not in reviewed_sources:
                raise InvariantError("GuidedSession active_edit_target source does not resolve")
            if target.kind == "output" and target.stable_id not in reviewed_outputs:
                raise InvariantError("GuidedSession active_edit_target output does not resolve")
            if target.kind in {"node", "edge"} and self.active_proposal is None:
                raise InvariantError("GuidedSession node/edge active_edit_target requires active_proposal")
        if self.terminal is not None and (self.active_proposal is not None or self.active_edit_target is not None):
            raise InvariantError("GuidedSession terminal state must clear active_proposal and active_edit_target")

        freeze_fields(
            self,
            "history",
            "chat_history",
            "source_order",
            "reviewed_sources",
            "pending_source_intents",
            "output_order",
            "reviewed_outputs",
            "pending_output_intents",
            "deferred_intents",
        )

    @staticmethod
    def _validated_component_mapping(
        value: object,
        item_type: type[_ComponentT],
        field_name: str,
    ) -> dict[str, _ComponentT]:
        if not isinstance(value, Mapping):
            raise TypeError(f"GuidedSession.{field_name} must be a mapping")
        result: dict[str, _ComponentT] = {}
        for stable_id, item in value.items():
            canonical = _canonical_uuid_text(stable_id, f"GuidedSession.{field_name} key")
            if type(item) is not item_type:
                raise TypeError(f"GuidedSession.{field_name} values must be {item_type.__name__}")
            result[canonical] = item
        return result

    @classmethod
    def initial(cls, profile: WorkflowProfile = EMPTY_PROFILE) -> GuidedSession:
        return cls(
            step=GuidedStep.STEP_1_SOURCE,
            profile=profile,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the exact schema-8 keyset to a JSON-safe dictionary."""
        return {
            "schema_version": GUIDED_SESSION_SCHEMA_VERSION,
            "step": self.step.value,
            "history": [r.to_dict() for r in self.history],
            "profile": self.profile.to_dict(),
            "advisor_checkpoint_passes_used": self.advisor_checkpoint_passes_used,
            "advisor_signoff_escape_offered": self.advisor_signoff_escape_offered,
            "terminal": self.terminal.to_dict() if self.terminal is not None else None,
            "transition_consumed": self.transition_consumed,
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
            "source_order": list(self.source_order),
            "reviewed_sources": {stable_id: source.to_dict() for stable_id, source in self.reviewed_sources.items()},
            "pending_source_intents": {stable_id: intent.to_dict() for stable_id, intent in self.pending_source_intents.items()},
            "output_order": list(self.output_order),
            "reviewed_outputs": {stable_id: output.to_dict() for stable_id, output in self.reviewed_outputs.items()},
            "pending_output_intents": {stable_id: intent.to_dict() for stable_id, intent in self.pending_output_intents.items()},
            "deferred_intents": [intent.to_dict() for intent in self.deferred_intents],
            "active_proposal": self.active_proposal.to_dict() if self.active_proposal is not None else None,
            "active_edit_target": self.active_edit_target.to_dict() if self.active_edit_target is not None else None,
            "root_intent_message_id": self.root_intent_message_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GuidedSession:
        """Restore and revalidate one exact schema-8 checkpoint."""
        try:
            if type(d) is not dict:
                raise InvariantError("GuidedSession.from_dict: record must be an exact dict")
            schema_version = _require_guided_int(d["schema_version"], "schema_version")
            if schema_version != GUIDED_SESSION_SCHEMA_VERSION:
                raise InvariantError(f"GuidedSession.from_dict: unsupported schema_version {schema_version}")
            d = dict(_require_exact_dict(d, _GUIDED_SESSION_KEYS, "GuidedSession.from_dict"))
            history_raw = _require_guided_sequence(d["history"], "history")
            if len(history_raw) > GUIDED_MAX_HISTORY_RECORDS:
                raise InvariantError(f"GuidedSession.from_dict: history exceeds the {GUIDED_MAX_HISTORY_RECORDS}-record limit")
            history_summary_chars = 0
            for index, record in enumerate(history_raw):
                if type(record) is not dict or "summary" not in record:
                    raise InvariantError(f"GuidedSession.from_dict: history[{index}] must expose summary in an exact dict")
                summary_raw = record["summary"]
                if summary_raw is not None and type(summary_raw) is not str:
                    raise InvariantError(f"GuidedSession.from_dict: history[{index}].summary must be an exact str or None")
                if summary_raw is not None:
                    history_summary_chars += len(summary_raw)
                    if history_summary_chars > GUIDED_MAX_HISTORY_SUMMARY_CHARS:
                        raise InvariantError(
                            f"GuidedSession.from_dict: history aggregate summaries exceed {GUIDED_MAX_HISTORY_SUMMARY_CHARS} characters"
                        )
            history = tuple(TurnRecord.from_dict(record) for record in history_raw)
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
            terminal_raw = d["terminal"]
            transition_consumed = _require_guided_bool(d["transition_consumed"], "transition_consumed")
            chat_history_raw = _require_guided_sequence(d["chat_history"], "chat_history")
            if len(chat_history_raw) > GUIDED_MAX_CHAT_TURNS:
                raise InvariantError(f"GuidedSession.from_dict: chat_history exceeds the {GUIDED_MAX_CHAT_TURNS}-turn limit")
            chat_history_chars = 0
            for index, turn in enumerate(chat_history_raw):
                if type(turn) is not dict or "content" not in turn:
                    raise InvariantError(f"GuidedSession.from_dict: chat_history[{index}] must expose content in an exact dict")
                content_raw = turn["content"]
                if type(content_raw) is not str:
                    raise InvariantError(f"GuidedSession.from_dict: chat_history[{index}].content must be an exact str")
                chat_history_chars += len(content_raw)
                if chat_history_chars > GUIDED_MAX_CHAT_HISTORY_CHARS:
                    raise InvariantError(
                        f"GuidedSession.from_dict: chat_history aggregate content exceeds {GUIDED_MAX_CHAT_HISTORY_CHARS} characters"
                    )
            chat_turn_seq = _require_guided_non_negative_int(d["chat_turn_seq"], "chat_turn_seq")
            source_order_raw = d["source_order"]
            output_order_raw = d["output_order"]
            if type(source_order_raw) is not list:
                raise InvariantError("GuidedSession.from_dict: source_order must be an exact list")
            if type(output_order_raw) is not list:
                raise InvariantError("GuidedSession.from_dict: output_order must be an exact list")
            if len(source_order_raw) > GUIDED_MAX_COMPONENTS_PER_KIND:
                raise InvariantError(
                    f"GuidedSession.from_dict: source components exceed the {GUIDED_MAX_COMPONENTS_PER_KIND}-component limit"
                )
            if len(output_order_raw) > GUIDED_MAX_COMPONENTS_PER_KIND:
                raise InvariantError(
                    f"GuidedSession.from_dict: output components exceed the {GUIDED_MAX_COMPONENTS_PER_KIND}-component limit"
                )
            source_order = _str_tuple_from_list(source_order_raw, "GuidedSession.source_order", uuid_items=True)
            output_order = _str_tuple_from_list(output_order_raw, "GuidedSession.output_order", uuid_items=True)
            reviewed_sources_raw = _require_str_mapping(d["reviewed_sources"], "GuidedSession.reviewed_sources")
            pending_sources_raw = _require_str_mapping(d["pending_source_intents"], "GuidedSession.pending_source_intents")
            reviewed_outputs_raw = _require_str_mapping(d["reviewed_outputs"], "GuidedSession.reviewed_outputs")
            pending_outputs_raw = _require_str_mapping(d["pending_output_intents"], "GuidedSession.pending_output_intents")
            if len(reviewed_sources_raw) + len(pending_sources_raw) > GUIDED_MAX_COMPONENTS_PER_KIND:
                raise InvariantError(
                    f"GuidedSession.from_dict: source components exceed the {GUIDED_MAX_COMPONENTS_PER_KIND}-component limit"
                )
            if len(reviewed_outputs_raw) + len(pending_outputs_raw) > GUIDED_MAX_COMPONENTS_PER_KIND:
                raise InvariantError(
                    f"GuidedSession.from_dict: output components exceed the {GUIDED_MAX_COMPONENTS_PER_KIND}-component limit"
                )
            deferred_raw = _require_guided_sequence(d["deferred_intents"], "deferred_intents")
            if len(deferred_raw) > GUIDED_MAX_DEFERRED_INTENTS:
                raise InvariantError(f"GuidedSession.from_dict: deferred_intents exceed the {GUIDED_MAX_DEFERRED_INTENTS}-intent limit")
            deferred_constraint_count = 0
            for index, intent in enumerate(deferred_raw):
                if type(intent) is not dict or "constraints" not in intent:
                    raise InvariantError(f"GuidedSession.from_dict: deferred_intents[{index}] must expose constraints in an exact dict")
                constraints_raw = intent["constraints"]
                if type(constraints_raw) is not list:
                    raise InvariantError(f"GuidedSession.from_dict: deferred_intents[{index}].constraints must be a list")
                if len(constraints_raw) > GUIDED_MAX_CONSTRAINTS_PER_INTENT:
                    raise InvariantError(
                        "GuidedSession.from_dict: deferred_intents constraint list exceeds "
                        f"the {GUIDED_MAX_CONSTRAINTS_PER_INTENT}-constraint limit"
                    )
                deferred_constraint_count += len(constraints_raw)
                if deferred_constraint_count > GUIDED_MAX_TOTAL_CONSTRAINTS:
                    raise InvariantError(
                        f"GuidedSession.from_dict: deferred constraints exceed the {GUIDED_MAX_TOTAL_CONSTRAINTS}-constraint limit"
                    )
            chat_history: tuple[ChatTurn, ...] = tuple(_chat_turn_from_guided_dict(entry) for entry in chat_history_raw)
            active_proposal_raw = d["active_proposal"]
            active_edit_target_raw = d["active_edit_target"]
            return cls(
                step=GuidedStep(d["step"]),
                history=history,
                profile=profile,
                advisor_checkpoint_passes_used=advisor_checkpoint_passes_used_raw,
                advisor_signoff_escape_offered=advisor_signoff_escape_offered_raw,
                terminal=TerminalState.from_dict(terminal_raw) if terminal_raw is not None else None,
                transition_consumed=transition_consumed,
                chat_history=chat_history,
                chat_turn_seq=chat_turn_seq,
                source_order=source_order,
                reviewed_sources={
                    _canonical_uuid_text(stable_id, "GuidedSession.reviewed_sources key"): SourceResolved.from_dict(source)
                    for stable_id, source in reviewed_sources_raw.items()
                },
                pending_source_intents={
                    _canonical_uuid_text(stable_id, "GuidedSession.pending_source_intents key"): SourceIntent.from_dict(intent)
                    for stable_id, intent in pending_sources_raw.items()
                },
                output_order=output_order,
                reviewed_outputs={
                    _canonical_uuid_text(stable_id, "GuidedSession.reviewed_outputs key"): SinkOutputResolved.from_dict(output)
                    for stable_id, output in reviewed_outputs_raw.items()
                },
                pending_output_intents={
                    _canonical_uuid_text(stable_id, "GuidedSession.pending_output_intents key"): SinkIntent.from_dict(intent)
                    for stable_id, intent in pending_outputs_raw.items()
                },
                deferred_intents=tuple(DeferredStageIntent.from_dict(intent) for intent in deferred_raw),
                active_proposal=GuidedProposalRef.from_dict(active_proposal_raw) if active_proposal_raw is not None else None,
                active_edit_target=ComponentTarget.from_dict(active_edit_target_raw) if active_edit_target_raw is not None else None,
                root_intent_message_id=(
                    _canonical_uuid_text(d["root_intent_message_id"], "GuidedSession.root_intent_message_id")
                    if d["root_intent_message_id"] is not None
                    else None
                ),
            )
        except InvariantError:
            raise
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
    """Leave source-stage mutation to the lock-owning dispatcher."""
    return (session, None, None, [])


def _advance_step_2(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
) -> _StepAdvanceResult:
    """Leave output-stage mutation to the lock-owning dispatcher."""
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
