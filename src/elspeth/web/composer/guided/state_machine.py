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
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypedDict, TypeVar, cast
from uuid import UUID

from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.core.canonical import stable_hash
from elspeth.web.composer.guided import stage_subjects
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.profile import EMPTY_PROFILE, WorkflowProfile
from elspeth.web.composer.guided.protocol import GUIDED_MAX_COMPONENTS_PER_KIND, ChatRole, ChatTurn, GuidedStep, TurnType
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

# Schema 9 is a pre-release hard cut. There is no older-schema decoder or
# converter: session epoch 32 owns the current store recreation boundary.
GUIDED_SESSION_SCHEMA_VERSION = 9
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


@dataclass(frozen=True, slots=True)
class TerminalState:
    """Outcome of a guided session.

    `reason` is None when `kind == COMPLETED`; required when
    `kind == EXITED_TO_FREEFORM`. `pipeline_yaml` is set only on COMPLETED.
    Construction invariants are enforced by ``__post_init__``.
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
            if self.plugin is None or self.options is not None or self.observed_columns or self.sample_rows:
                raise ValueError(
                    "SourceIntent plugin_options phase requires a selected plugin, optional inspection_facts, and no resolved values"
                )
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


_STAGE_ORDINAL: dict[stage_subjects.StageName, int] = {"source": 0, "output": 1, "topology": 2, "wire_review": 3}


class DeferredStageIntentData(TypedDict):
    intent_id: str
    receiving_stage: stage_subjects.StageName
    target_stage: stage_subjects.StageName
    catalog_kind: str | None
    catalog_name: str | None
    redacted_summary: str
    summary_hash: str
    originating_message_id: str
    message_content_hash: str
    constraints: list[stage_subjects.DeferredConstraintData]


@dataclass(frozen=True, slots=True)
class DeferredStageIntent:
    intent_id: str
    receiving_stage: stage_subjects.StageName
    target_stage: stage_subjects.StageName
    catalog_kind: Literal["source", "transform", "sink"] | None
    catalog_name: str | None
    redacted_summary: str
    summary_hash: str
    originating_message_id: str
    message_content_hash: str
    constraints: tuple[stage_subjects.DeferredConstraint, ...]

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
            stage_subjects.SubjectPresenceConstraint,
            stage_subjects.OptionValueConstraint,
            stage_subjects.ComponentCountConstraint,
            stage_subjects.EdgeRouteConstraint,
            stage_subjects.FailureRouteConstraint,
        }
        if any(type(constraint) not in allowed for constraint in self.constraints):
            raise InvariantError("DeferredStageIntent.constraints contains an unsupported constraint")
        freeze_fields(self, "constraints")

    @classmethod
    def create(
        cls,
        *,
        intent_id: str,
        receiving_stage: stage_subjects.StageName,
        target_stage: stage_subjects.StageName,
        catalog_kind: Literal["source", "transform", "sink"] | None,
        catalog_name: str | None,
        redacted_summary: str,
        originating_message_id: str,
        message_content_hash: str,
        constraints: tuple[stage_subjects.DeferredConstraint, ...],
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
            receiving_stage=stage_subjects.stage_name_from_value(record["receiving_stage"], "DeferredStageIntent stage"),
            target_stage=stage_subjects.stage_name_from_value(record["target_stage"], "DeferredStageIntent stage"),
            catalog_kind=cast(Any, record["catalog_kind"]),
            catalog_name=_require_optional_nonempty_str(record["catalog_name"], "DeferredStageIntent.catalog_name"),
            redacted_summary=_require_nonempty_str(record["redacted_summary"], "DeferredStageIntent.redacted_summary"),
            summary_hash=_require_hash(record["summary_hash"], "DeferredStageIntent.summary_hash"),
            originating_message_id=_canonical_uuid_text(record["originating_message_id"], "DeferredStageIntent.originating_message_id"),
            message_content_hash=_require_hash(record["message_content_hash"], "DeferredStageIntent.message_content_hash"),
            constraints=tuple(stage_subjects.constraint_from_dict(constraint) for constraint in constraints_raw),
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
        if len(set(self.reviewed_sources) | set(self.pending_source_intents)) > GUIDED_MAX_COMPONENTS_PER_KIND:
            raise InvariantError(f"GuidedSession source components exceed the {GUIDED_MAX_COMPONENTS_PER_KIND}-component limit")
        if len(set(self.reviewed_outputs) | set(self.pending_output_intents)) > GUIDED_MAX_COMPONENTS_PER_KIND:
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
        source_overlap = set(reviewed_sources) & set(pending_sources)
        output_overlap = set(reviewed_outputs) & set(pending_outputs)
        active_target = self.active_edit_target if type(self.active_edit_target) is ComponentTarget else None
        if active_target is not None and active_target.kind == "source":
            if pending_outputs or (pending_sources and set(pending_sources) != {active_target.stable_id}):
                raise InvariantError("GuidedSession active source edit may only coexist with its own inspection_review intent")
            if pending_sources and pending_sources[active_target.stable_id].phase != "inspection_review":
                raise InvariantError("GuidedSession active source edit may only coexist with its own inspection_review intent")
        if active_target is not None and active_target.kind == "output":
            if pending_sources or (pending_outputs and set(pending_outputs) != {active_target.stable_id}):
                raise InvariantError("GuidedSession active output edit may only coexist with its own field_review intent")
            if pending_outputs and pending_outputs[active_target.stable_id].phase != "field_review":
                raise InvariantError("GuidedSession active output edit may only coexist with its own field_review intent")
        allowed_source_overlap = {active_target.stable_id} if active_target is not None and active_target.kind == "source" else set()
        allowed_output_overlap = {active_target.stable_id} if active_target is not None and active_target.kind == "output" else set()
        if source_overlap - allowed_source_overlap:
            raise InvariantError(
                "GuidedSession reviewed and pending source keysets must be disjoint except for the active source edit target"
            )
        if output_overlap - allowed_output_overlap:
            raise InvariantError(
                "GuidedSession reviewed and pending output keysets must be disjoint except for the active output edit target"
            )
        for stable_id in source_overlap:
            source_intent = pending_sources[stable_id]
            reviewed_source = reviewed_sources[stable_id]
            if (
                source_intent.phase != "inspection_review"
                or source_intent.name != reviewed_source.name
                or source_intent.plugin != reviewed_source.plugin
            ):
                raise InvariantError("GuidedSession active source edit overlap has inconsistent review custody")
        for stable_id in output_overlap:
            output_intent = pending_outputs[stable_id]
            reviewed_output = reviewed_outputs[stable_id]
            if (
                output_intent.phase != "field_review"
                or output_intent.name != reviewed_output.name
                or output_intent.plugin != reviewed_output.plugin
            ):
                raise InvariantError("GuidedSession active output edit overlap has inconsistent review custody")
        if set(source_order) != set(reviewed_sources) | set(pending_sources):
            raise InvariantError("GuidedSession.source_order must be an exact permutation of source keys")
        if set(output_order) != set(reviewed_outputs) | set(pending_outputs):
            raise InvariantError("GuidedSession.output_order must be an exact permutation of output keys")
        if (set(reviewed_sources) | set(pending_sources)) & (set(reviewed_outputs) | set(pending_outputs)):
            raise InvariantError("GuidedSession stable component IDs must be globally unique")
        source_names = [source.name for source in reviewed_sources.values()] + [
            intent.name for stable_id, intent in pending_sources.items() if stable_id not in source_overlap
        ]
        output_names = [output.name for output in reviewed_outputs.values()] + [
            intent.name for stable_id, intent in pending_outputs.items() if stable_id not in output_overlap
        ]
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
            unanswered = tuple(index for index, turn in enumerate(self.history) if turn.response_hash is None)
            if unanswered != (len(self.history) - 1,):
                raise InvariantError("GuidedSession active_proposal requires one sole unanswered trailing turn")
            trailing = self.history[-1]
            legal_pending_shape = (
                self.step is GuidedStep.STEP_3_TRANSFORMS
                and trailing.step is GuidedStep.STEP_3_TRANSFORMS
                and trailing.turn_type is TurnType.PROPOSE_PIPELINE
            ) or (
                self.step is GuidedStep.STEP_4_WIRE
                and trailing.step is GuidedStep.STEP_4_WIRE
                and trailing.turn_type is TurnType.CONFIRM_WIRING
            )
            if not legal_pending_shape:
                raise InvariantError("GuidedSession active_proposal is not bound to the current proposal/wire turn")
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
        """Serialize the exact schema-9 keyset to a JSON-safe dictionary."""
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
        """Restore and revalidate one exact schema-9 checkpoint."""
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
