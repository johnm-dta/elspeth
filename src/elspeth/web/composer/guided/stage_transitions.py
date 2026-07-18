"""Pure schema-8 RESPOND transitions for guided source and sink stages.

This module owns only immutable checkpoint movement.  Routes remain
responsible for catalog/policy lookup, blob/path resolution, audit events,
history settlement, HTTP errors, and compare-and-swap persistence.  The
transition result deliberately carries a closed *preparation* record rather
than a wire ``Turn`` so this layer cannot grow a second renderer or topology
model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, cast
from uuid import UUID

from elspeth.contracts.freeze import deep_thaw
from elspeth.core.canonical import stable_hash
from elspeth.web.catalog.knob_schema import KnobSchema, validate_knob_schema
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import BLOB_REF_PATH_PREFIX, ControlSignal, GuidedStep, TurnType
from elspeth.web.composer.guided.resolved import (
    SinkOutputResolved,
    SourceResolved,
    freeze_guided_json_mapping,
    freeze_guided_str_sequence,
)
from elspeth.web.composer.guided.state_machine import GuidedSession, SinkIntent, SourceIntent
from elspeth.web.composer.source_inspection import SourceInspectionFacts, facts_from_dict, facts_to_dict

GUIDED_TURN_TOKEN_SCHEMA: Final = "guided.turn-token.v1"
_HASH_CHARS: Final = frozenset("0123456789abcdef")
_PATH_OPTION_NAMES: Final = frozenset({"path", "file"})
_SOURCE_KIND_PLUGIN: Final = {"csv": "csv", "json": "json", "jsonl": "json", "text": "text"}
_KNOB_KINDS: Final = frozenset(
    {
        "text",
        "number-int",
        "number-float",
        "checkbox",
        "enum",
        "string-list",
        "blob-ref",
        "json-object",
        "json-array",
        "json-value",
    }
)


def _require_nonempty_exact_str(value: object, field_name: str) -> str:
    if type(value) is not str or value == "":
        raise TypeError(f"{field_name} must be a non-empty exact str")
    return value


def _require_hash(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be an exact str")
    if len(value) != 64 or any(char not in _HASH_CHARS for char in value):
        raise ValueError(f"{field_name} must be a canonical lowercase sha256 hex digest")
    return value


def _canonical_uuid(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a canonical UUID string") from exc
    if str(parsed) != value:
        raise ValueError(f"{field_name} must be a canonical UUID string")
    return value


def guided_turn_token(
    *,
    schema: str,
    history_index: int,
    step: GuidedStep,
    turn_type: TurnType,
    payload_hash: str,
) -> str:
    """Bind an unanswered turn occurrence, including repeated-payload ABA.

    The token is deterministic rather than secret.  It gives the later route
    and CAS cohort one canonical identity for "this payload at this occurrence"
    without changing the current HTTP DTO in this task.
    """

    if type(schema) is not str:
        raise TypeError("guided turn token schema must be an exact str")
    if schema != GUIDED_TURN_TOKEN_SCHEMA:
        raise ValueError(f"unsupported guided turn token schema {schema!r}")
    if type(history_index) is not int:
        raise TypeError("guided turn token history_index must be an exact int")
    if history_index < 0:
        raise ValueError("guided turn token history_index must be non-negative")
    if type(step) is not GuidedStep:
        raise TypeError("guided turn token step must be GuidedStep")
    if type(turn_type) is not TurnType:
        raise TypeError("guided turn token turn_type must be TurnType")
    canonical_payload_hash = _require_hash(payload_hash, "guided turn token payload_hash")
    return stable_hash(
        {
            "schema": schema,
            "history_index": history_index,
            "step": step.value,
            "turn_type": turn_type.value,
            "payload_hash": canonical_payload_hash,
        }
    )


@dataclass(frozen=True, slots=True)
class AnsweredTurn:
    """Index of the persisted unanswered turn being consumed.

    Step, type, payload hash, and deterministic occurrence id are derived from
    the current persisted :class:`TurnRecord`; callers cannot restate them.
    """

    history_index: int

    def __post_init__(self) -> None:
        if type(self.history_index) is not int:
            raise TypeError("AnsweredTurn.history_index must be an exact int")
        if self.history_index < 0:
            raise ValueError("AnsweredTurn.history_index must be non-negative")


@dataclass(frozen=True, slots=True)
class PluginSelectionResponse:
    """Closed response to source or sink ``single_select``."""

    chosen: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "chosen", freeze_guided_str_sequence(self.chosen, "PluginSelectionResponse.chosen"))


@dataclass(frozen=True, slots=True)
class SchemaFormResponse:
    """Closed client-authored portion of a plugin schema-form response."""

    plugin: str
    options: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_nonempty_exact_str(self.plugin, "SchemaFormResponse.plugin")
        object.__setattr__(self, "options", freeze_guided_json_mapping(self.options, "SchemaFormResponse.options"))


@dataclass(frozen=True, slots=True)
class SchemaFormAuthority:
    """Server-held schema and custody fields paired with a form response.

    ``server_options`` may include hidden custody fields such as a resolved blob
    binding.  A client may echo such a field only byte-for-byte; it cannot alter
    it.  The server value always wins the merge.
    """

    knobs: Mapping[str, Any]
    model_validated_options: Mapping[str, Any]
    server_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "knobs", freeze_guided_json_mapping(self.knobs, "SchemaFormAuthority.knobs"))
        object.__setattr__(
            self,
            "model_validated_options",
            freeze_guided_json_mapping(
                self.model_validated_options,
                "SchemaFormAuthority.model_validated_options",
            ),
        )
        object.__setattr__(
            self,
            "server_options",
            freeze_guided_json_mapping(self.server_options, "SchemaFormAuthority.server_options"),
        )


@dataclass(frozen=True, slots=True)
class InspectionResponse:
    """The only client-authored part of inspection confirmation."""

    columns: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", freeze_guided_str_sequence(self.columns, "InspectionResponse.columns"))


@dataclass(frozen=True, slots=True)
class FieldSelectionResponse:
    """Closed response to sink ``multi_select_with_custom``."""

    chosen: Sequence[str]
    custom_inputs: Sequence[str]
    control_signal: ControlSignal | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "chosen", freeze_guided_str_sequence(self.chosen, "FieldSelectionResponse.chosen"))
        object.__setattr__(
            self,
            "custom_inputs",
            freeze_guided_str_sequence(self.custom_inputs, "FieldSelectionResponse.custom_inputs"),
        )
        if self.control_signal is not None and type(self.control_signal) is not ControlSignal:
            raise TypeError("FieldSelectionResponse.control_signal must be ControlSignal or None")


@dataclass(frozen=True, slots=True)
class SourceSchemaFormPreparation:
    stable_id: str
    plugin: str
    inspection_facts: SourceInspectionFacts | None

    def __post_init__(self) -> None:
        _canonical_uuid(self.stable_id, "SourceSchemaFormPreparation.stable_id")
        _require_nonempty_exact_str(self.plugin, "SourceSchemaFormPreparation.plugin")
        if self.inspection_facts is not None:
            object.__setattr__(self, "inspection_facts", _validated_inspection_facts(self.inspection_facts))


@dataclass(frozen=True, slots=True)
class SourceInspectionPreparation:
    stable_id: str
    inspection_facts: SourceInspectionFacts

    def __post_init__(self) -> None:
        _canonical_uuid(self.stable_id, "SourceInspectionPreparation.stable_id")
        object.__setattr__(self, "inspection_facts", _validated_inspection_facts(self.inspection_facts))


@dataclass(frozen=True, slots=True)
class SinkPluginSelectionPreparation:
    """Prepare the catalog-backed initial Step-2 sink selection turn."""


@dataclass(frozen=True, slots=True)
class SinkSchemaFormPreparation:
    stable_id: str
    plugin: str

    def __post_init__(self) -> None:
        _canonical_uuid(self.stable_id, "SinkSchemaFormPreparation.stable_id")
        _require_nonempty_exact_str(self.plugin, "SinkSchemaFormPreparation.plugin")


@dataclass(frozen=True, slots=True)
class SinkFieldReviewPreparation:
    stable_id: str
    candidate_fields: Sequence[str]

    def __post_init__(self) -> None:
        _canonical_uuid(self.stable_id, "SinkFieldReviewPreparation.stable_id")
        fields = freeze_guided_str_sequence(self.candidate_fields, "SinkFieldReviewPreparation.candidate_fields")
        if len(set(fields)) != len(fields):
            raise InvariantError("SinkFieldReviewPreparation.candidate_fields must be unique")
        object.__setattr__(self, "candidate_fields", fields)


type NextTurnPreparation = (
    SourceSchemaFormPreparation
    | SourceInspectionPreparation
    | SinkPluginSelectionPreparation
    | SinkSchemaFormPreparation
    | SinkFieldReviewPreparation
)


@dataclass(frozen=True, slots=True)
class StageTransitionResult:
    """One replacement checkpoint plus the semantic next-turn instruction."""

    session: GuidedSession
    next_turn: NextTurnPreparation | None

    def __post_init__(self) -> None:
        if type(self.session) is not GuidedSession:
            raise TypeError("StageTransitionResult.session must be GuidedSession")
        allowed = (
            SourceSchemaFormPreparation,
            SourceInspectionPreparation,
            SinkPluginSelectionPreparation,
            SinkSchemaFormPreparation,
            SinkFieldReviewPreparation,
        )
        if self.next_turn is not None and type(self.next_turn) not in allowed:
            raise TypeError("StageTransitionResult.next_turn is not a closed preparation type")


def _require_active_turn(
    session: GuidedSession,
    turn: AnsweredTurn,
    *,
    expected_step: GuidedStep,
    expected_turn_type: TurnType,
) -> None:
    if session.step is not expected_step:
        raise InvariantError(f"stage transition requires session step {expected_step.value}, got {session.step.value}")
    if session.terminal is not None:
        raise InvariantError("stage transition cannot run for a terminal guided session")
    if session.transition_consumed:
        raise InvariantError("stage transition cannot run after the guided transition was consumed")
    if turn.history_index != len(session.history) - 1:
        raise ValueError("answered turn is stale; history_index is not the current unanswered occurrence")
    record = session.history[turn.history_index]
    if record.response_hash is not None:
        raise ValueError("answered turn occurrence was already answered")
    if record.step is not expected_step:
        raise ValueError(f"persisted answered turn step must be {expected_step.value}")
    if record.turn_type is not expected_turn_type:
        raise ValueError(f"persisted answered turn type must be {expected_turn_type.value}")


def _validated_inspection_facts(facts: SourceInspectionFacts) -> SourceInspectionFacts:
    if type(facts) is not SourceInspectionFacts:
        raise TypeError("inspection_facts must be SourceInspectionFacts")
    snapshot = facts_from_dict(facts_to_dict(facts))
    headers = snapshot.observed_headers
    inferred = snapshot.inferred_types
    if (headers is None) != (inferred is None):
        raise InvariantError("inspection facts headers and inferred types must be present or absent together")
    if headers is not None and inferred is not None and set(headers) != set(inferred):
        raise InvariantError("inspection facts inferred type keys must exactly match observed headers")
    return snapshot


def _require_inspection_plugin_match(plugin: str, facts: SourceInspectionFacts) -> None:
    expected_plugin = _SOURCE_KIND_PLUGIN[facts.source_kind] if facts.source_kind in _SOURCE_KIND_PLUGIN else None
    if expected_plugin is not None and plugin != expected_plugin:
        raise ValueError(f"selected source plugin {plugin!r} does not match inspection facts for {facts.source_kind!r} content")


def _inspection_blob_id(facts: SourceInspectionFacts) -> str | None:
    if "blob_id" not in facts.redacted_identity:
        return None
    return _canonical_uuid(facts.redacted_identity["blob_id"], "inspection facts blob_id")


def _option_blob_ids(options: Mapping[str, Any]) -> tuple[str, ...]:
    ids: list[str] = []
    for option_name, option_value in options.items():
        if (option_name == "blob_ref" or option_name == "blob_id" or option_name.endswith("_blob_id")) and option_value is not None:
            ids.append(_canonical_uuid(option_value, f"server blob custody option {option_name!r}"))
        if option_name in _PATH_OPTION_NAMES and type(option_value) is str and option_value.startswith(BLOB_REF_PATH_PREFIX):
            ids.append(
                _canonical_uuid(
                    option_value.removeprefix(BLOB_REF_PATH_PREFIX),
                    f"server blob custody path {option_name!r}",
                )
            )
    return tuple(ids)


def _require_inspection_custody_match(
    facts: SourceInspectionFacts,
    *,
    submitted_and_server_options: Mapping[str, Any],
    server_options: Mapping[str, Any],
) -> None:
    inspected_blob_id = _inspection_blob_id(facts)
    if inspected_blob_id is None:
        return
    custody_ids = (*_option_blob_ids(submitted_and_server_options), *_option_blob_ids(server_options))
    if not custody_ids or any(blob_id != inspected_blob_id for blob_id in custody_ids):
        raise ValueError("source form blob custody does not match the blob named by inspection facts")


def _validated_permitted_plugins(permitted_plugins: Sequence[str]) -> tuple[str, ...]:
    plugins = freeze_guided_str_sequence(permitted_plugins, "permitted_plugins")
    if not plugins:
        raise InvariantError("permitted_plugins must not be empty")
    if any(plugin == "" for plugin in plugins):
        raise InvariantError("permitted_plugins must contain non-empty names")
    if len(set(plugins)) != len(plugins):
        raise InvariantError("permitted_plugins must be unique")
    return plugins


def _selected_plugin(response: PluginSelectionResponse, permitted_plugins: Sequence[str]) -> str:
    if type(response) is not PluginSelectionResponse:
        raise TypeError("response must be PluginSelectionResponse")
    if len(response.chosen) != 1:
        raise ValueError("single_select response must choose exactly one plugin")
    plugin = response.chosen[0]
    if plugin == "":
        raise ValueError("single_select plugin must be non-empty")
    if plugin not in _validated_permitted_plugins(permitted_plugins):
        raise ValueError(f"plugin {plugin!r} is not in the server-emitted permitted set")
    return plugin


def _next_component_name(base: str, existing_names: Sequence[str]) -> str:
    names = frozenset(existing_names)
    if base not in names:
        return base
    suffix = 2
    while f"{base}_{suffix}" in names:
        suffix += 1
    return f"{base}_{suffix}"


def _admit_new_stable_id(session: GuidedSession, candidate: UUID | None) -> str:
    if candidate is None:
        raise InvariantError("a preallocated UUID is required for a new guided component")
    if type(candidate) is not UUID:
        raise InvariantError("preallocated stable id must be an exact UUID")
    stable_id = str(candidate)
    all_ids = (
        set(session.reviewed_sources)
        | set(session.pending_source_intents)
        | set(session.reviewed_outputs)
        | set(session.pending_output_intents)
    )
    if stable_id in all_ids:
        raise InvariantError("preallocated stable id already names a guided component")
    return stable_id


def _source_selection_target(
    session: GuidedSession,
    *,
    target_id: str | None,
    new_stable_id: UUID | None,
) -> tuple[str, str, bool]:
    if target_id is None and not session.pending_source_intents:
        stable_id = _admit_new_stable_id(session, new_stable_id)
        names = [source.name for source in session.reviewed_sources.values()]
        return stable_id, _next_component_name("source", names), True
    if new_stable_id is not None:
        raise ValueError("a preallocated source id must not be supplied when reusing a pending source")
    if target_id is None:
        if len(session.pending_source_intents) != 1:
            raise ValueError("source selection target is required when more than one source intent is pending")
        stable_id = next(iter(session.pending_source_intents))
    else:
        stable_id = _canonical_uuid(target_id, "source selection target_id")
    if stable_id not in session.pending_source_intents:
        raise ValueError("source selection target does not name a pending source intent")
    intent = session.pending_source_intents[stable_id]
    if intent.phase != "plugin_selection":
        raise ValueError("source selection target is not in plugin_selection phase")
    if sum(item.phase == "plugin_selection" for item in session.pending_source_intents.values()) != 1:
        raise ValueError("source selection target is ambiguous for the current turn occurrence")
    return stable_id, intent.name, False


def _sink_selection_target(
    session: GuidedSession,
    *,
    target_id: str | None,
    new_stable_id: UUID | None,
) -> tuple[str, str, bool]:
    if target_id is None and not session.pending_output_intents:
        stable_id = _admit_new_stable_id(session, new_stable_id)
        names = [output.name for output in session.reviewed_outputs.values()]
        return stable_id, _next_component_name("output", names), True
    if new_stable_id is not None:
        raise ValueError("a preallocated output id must not be supplied when reusing a pending output")
    if target_id is None:
        if len(session.pending_output_intents) != 1:
            raise ValueError("sink selection target is required when more than one output intent is pending")
        stable_id = next(iter(session.pending_output_intents))
    else:
        stable_id = _canonical_uuid(target_id, "sink selection target_id")
    if stable_id not in session.pending_output_intents:
        raise ValueError("sink selection target does not name a pending output intent")
    intent = session.pending_output_intents[stable_id]
    if intent.phase != "plugin_selection":
        raise ValueError("sink selection target is not in plugin_selection phase")
    if sum(item.phase == "plugin_selection" for item in session.pending_output_intents.values()) != 1:
        raise ValueError("sink selection target is ambiguous for the current turn occurrence")
    return stable_id, intent.name, False


def _require_source_intent(session: GuidedSession, target_id: str, phase: str) -> tuple[str, SourceIntent]:
    stable_id = _canonical_uuid(target_id, "source target_id")
    if stable_id not in session.pending_source_intents:
        raise ValueError("source target does not name a pending source intent")
    intent = session.pending_source_intents[stable_id]
    if intent.phase != phase:
        raise ValueError(f"source target must be in {phase} phase")
    if sum(item.phase == phase for item in session.pending_source_intents.values()) != 1:
        raise ValueError(f"source target is ambiguous for the current {phase} turn occurrence")
    return stable_id, intent


def _require_sink_intent(session: GuidedSession, target_id: str, phase: str) -> tuple[str, SinkIntent]:
    stable_id = _canonical_uuid(target_id, "sink target_id")
    if stable_id not in session.pending_output_intents:
        raise ValueError("sink target does not name a pending output intent")
    intent = session.pending_output_intents[stable_id]
    if intent.phase != phase:
        raise ValueError(f"sink target must be in {phase} phase")
    if sum(item.phase == phase for item in session.pending_output_intents.values()) != 1:
        raise ValueError(f"sink target is ambiguous for the current {phase} turn occurrence")
    return stable_id, intent


def _knob_fields(authority: SchemaFormAuthority, *, plugin_kind: str, plugin_name: str) -> tuple[Mapping[str, Any], ...]:
    raw_knobs = deep_thaw(authority.knobs)
    if type(raw_knobs) is not dict or set(raw_knobs) != {"fields"}:
        raise InvariantError("server-held knob schema must contain exactly the fields key")
    raw_fields = raw_knobs["fields"]
    if type(raw_fields) is not list:
        raise InvariantError("server-held knob schema fields must be a list")
    for field_index, raw_field in enumerate(raw_fields):
        if type(raw_field) is not dict:
            raise InvariantError(f"server-held knob schema field {field_index} must be an exact dict")
        if "name" not in raw_field or "kind" not in raw_field or "required" not in raw_field or "nullable" not in raw_field:
            raise InvariantError(f"server-held knob schema field {field_index} must carry name, kind, required, and nullable")
        _require_nonempty_exact_str(raw_field["name"], f"knob field {field_index}.name")
        kind = _require_nonempty_exact_str(raw_field["kind"], f"knob field {field_index}.kind")
        if kind not in _KNOB_KINDS:
            raise InvariantError(f"server-held knob schema field {field_index} has unsupported kind {kind!r}")
        if type(raw_field["required"]) is not bool or type(raw_field["nullable"]) is not bool:
            raise InvariantError(f"server-held knob schema field {field_index} required/nullable must be exact bools")
    try:
        validate_knob_schema(cast(KnobSchema, raw_knobs), plugin_kind=plugin_kind, plugin_name=plugin_name)
    except (KeyError, TypeError, ValueError) as exc:
        raise InvariantError(f"server-held knob schema is malformed for {plugin_kind}/{plugin_name}") from exc
    return tuple(cast(Mapping[str, Any], item) for item in raw_fields)


def _validate_blob_ref(field_name: str, value: object) -> None:
    if value is None:
        return
    if type(value) is not str:
        raise ValueError(f"blob-ref option {field_name!r} must be a canonical UUID string")
    _canonical_uuid(value, f"blob-ref option {field_name!r}")


def _validate_path(field_name: str, value: object) -> None:
    if value is None:
        return
    if type(value) is not str or value == "" or "\x00" in value:
        raise ValueError(f"path option {field_name!r} must be a non-empty string without NUL")
    if value.startswith(BLOB_REF_PATH_PREFIX):
        _canonical_uuid(value.removeprefix(BLOB_REF_PATH_PREFIX), f"path option {field_name!r} blob reference")


def _field_is_visible(field: Mapping[str, Any], submitted: Mapping[str, Any]) -> bool:
    if "visible_when" not in field:
        return True
    predicate = field["visible_when"]
    if type(predicate) is not dict or set(predicate) != {"field", "equals"}:
        raise InvariantError(f"server-held knob predicate for {field['name']!r} is malformed")
    discriminator = predicate["field"]
    if type(discriminator) is not str:
        raise InvariantError(f"server-held knob predicate for {field['name']!r} has a non-string field")
    return discriminator in submitted and submitted[discriminator] == predicate["equals"]


def _validate_knob_value(field: Mapping[str, Any], field_name: str, value: object) -> None:
    if value is None:
        if not field["nullable"]:
            raise ValueError(f"option {field_name!r} is not nullable")
        return
    kind = field["kind"]
    valid = False
    if kind == "text":
        valid = type(value) is str
    elif kind == "number-int":
        valid = type(value) is int
    elif kind == "number-float":
        valid = type(value) in {int, float}
    elif kind == "checkbox":
        valid = type(value) is bool
    elif kind == "enum":
        if "enum" not in field or type(field["enum"]) is not list or any(type(item) is not str for item in field["enum"]):
            raise InvariantError(f"server-held enum knob {field_name!r} has a malformed enum vocabulary")
        valid = type(value) is str and value in field["enum"]
    elif kind == "string-list":
        if type(value) is list:
            item_kind = field["item_kind"] if "item_kind" in field else "text"
            if item_kind == "text":
                valid = all(type(item) is str for item in value)
            elif item_kind == "number-int":
                valid = all(type(item) is int for item in value)
            elif item_kind == "number-float":
                valid = all(type(item) in {int, float} for item in value)
            else:
                raise InvariantError(f"server-held string-list knob {field_name!r} has unsupported item_kind")
    elif kind == "blob-ref":
        _validate_blob_ref(field_name, value)
        valid = True
    elif kind == "json-object":
        valid = type(value) is dict
    elif kind == "json-array":
        valid = type(value) is list
    elif kind == "json-value":
        valid = True
    else:
        raise InvariantError(f"server-held knob {field_name!r} has unsupported kind {kind!r}")
    if not valid:
        raise ValueError(f"option {field_name!r} must satisfy knob kind {kind!r}")


def _validate_visible_option(
    option_name: str,
    option_value: object,
    *,
    submitted: Mapping[str, Any],
    fields: Sequence[Mapping[str, Any]],
) -> None:
    candidates = [field for field in fields if field["name"] == option_name]
    if not candidates and option_name == "schema_config":
        candidates = [field for field in fields if field["name"] == "schema"]
    if not candidates:
        raise ValueError(f"option {option_name!r} is not exposed by the server-held plugin schema")

    visible = [candidate for candidate in candidates if _field_is_visible(candidate, submitted)]
    if not visible:
        raise ValueError(f"option {option_name!r} is hidden under the submitted form state")
    first_visible = visible[0]
    if any(candidate["kind"] != first_visible["kind"] or candidate["nullable"] != first_visible["nullable"] for candidate in visible[1:]):
        raise InvariantError(f"simultaneously visible knob definitions for {option_name!r} disagree on type")
    _validate_knob_value(first_visible, option_name, option_value)
    if option_name in _PATH_OPTION_NAMES:
        _validate_path(option_name, option_value)


def _validate_required_options(submitted: Mapping[str, Any], fields: Sequence[Mapping[str, Any]]) -> None:
    names = tuple(dict.fromkeys(cast(str, field["name"]) for field in fields))
    for field_name in names:
        candidates = [field for field in fields if field["name"] == field_name and _field_is_visible(field, submitted)]
        if not candidates:
            continue
        required_without_default = any(field["required"] and "default" not in field for field in candidates)
        alias_present = field_name == "schema" and "schema_config" in submitted
        if required_without_default and field_name not in submitted and not alias_present:
            raise ValueError(f"required option {field_name!r} is missing")


def _validated_merged_options(
    response: SchemaFormResponse,
    authority: SchemaFormAuthority,
    *,
    plugin_kind: str,
    plugin_name: str,
    structural_defaults: Mapping[str, str],
) -> tuple[Mapping[str, Any], Mapping[str, str]]:
    if type(response) is not SchemaFormResponse:
        raise TypeError("response must be SchemaFormResponse")
    if type(authority) is not SchemaFormAuthority:
        raise TypeError("authority must be SchemaFormAuthority")
    fields = _knob_fields(authority, plugin_kind=plugin_kind, plugin_name=plugin_name)
    submitted = cast(dict[str, Any], deep_thaw(response.options))
    server_options = cast(dict[str, Any], deep_thaw(authority.server_options))
    for option_name, option_value in submitted.items():
        _validate_visible_option(option_name, option_value, submitted=submitted, fields=fields)
    for option_name, server_value in server_options.items():
        if option_name in submitted and stable_hash(submitted[option_name]) != stable_hash(server_value):
            raise ValueError(f"client altered server-held option {option_name!r}")
        if option_name in _PATH_OPTION_NAMES:
            _validate_path(option_name, server_value)
        if option_name == "blob_ref" or option_name.endswith("_blob_id"):
            _validate_blob_ref(option_name, server_value)
    merged = dict(submitted)
    merged.update(server_options)
    _validate_required_options(merged, fields)
    structural: dict[str, str] = {}
    for option_name, default_value in structural_defaults.items():
        if option_name in merged:
            structural[option_name] = _require_nonempty_exact_str(merged[option_name], f"structural policy {option_name!r}")
            del merged[option_name]
        else:
            structural[option_name] = default_value
    model_validated = cast(dict[str, Any], deep_thaw(authority.model_validated_options))
    for policy_name, policy_value in structural.items():
        if policy_name not in model_validated:
            continue
        if model_validated[policy_name] != policy_value:
            raise InvariantError(f"server model validation changed structural policy {policy_name!r}")
        del model_validated[policy_name]
    for option_name, option_value in merged.items():
        if option_name not in model_validated or stable_hash(model_validated[option_name]) != stable_hash(option_value):
            raise InvariantError(f"server model validation did not preserve submitted {plugin_kind} option {option_name!r}")
    return (
        freeze_guided_json_mapping(model_validated, f"{plugin_kind} resolved options"),
        cast(Mapping[str, str], freeze_guided_json_mapping(structural, f"{plugin_kind} structural policies")),
    )


def _candidate_fields(session: GuidedSession) -> tuple[str, ...]:
    candidates: list[str] = []
    seen: set[str] = set()
    for stable_id in session.source_order:
        if stable_id not in session.reviewed_sources:
            raise InvariantError("sink field review requires every ordered source to be reviewed")
        for field_name in session.reviewed_sources[stable_id].observed_columns:
            if field_name not in seen:
                seen.add(field_name)
                candidates.append(field_name)
    return tuple(candidates)


def _sink_schema_mode(options: Mapping[str, Any]) -> str:
    thawed = cast(dict[str, Any], deep_thaw(options))
    if "schema" not in thawed:
        return "observed"
    schema = thawed["schema"]
    if type(schema) is not dict:
        raise InvariantError("model-validated sink schema option must be an exact object")
    if "mode" not in schema:
        return "observed"
    mode = schema["mode"]
    if type(mode) is not str or mode not in {"fixed", "flexible", "observed"}:
        raise ValueError("sink schema mode must be fixed, flexible, or observed")
    return mode


def _pending_options_with_structural(options: Mapping[str, Any], structural: Mapping[str, str]) -> Mapping[str, Any]:
    pending = cast(dict[str, Any], deep_thaw(options))
    for policy_name, policy_value in structural.items():
        if policy_name in pending:
            raise InvariantError(f"plugin options unexpectedly contain structural policy {policy_name!r}")
        pending[policy_name] = policy_value
    return freeze_guided_json_mapping(pending, "pending options with structural policies")


def _split_pending_structural(options: Mapping[str, Any], policy_name: str) -> tuple[Mapping[str, Any], str]:
    plugin_options = cast(dict[str, Any], deep_thaw(options))
    if policy_name not in plugin_options:
        raise InvariantError(f"pending intent is missing structural policy {policy_name!r}")
    policy = _require_nonempty_exact_str(plugin_options[policy_name], f"pending structural policy {policy_name!r}")
    del plugin_options[policy_name]
    return freeze_guided_json_mapping(plugin_options, "pending plugin options"), policy


def _require_no_other_pending(mapping: Mapping[str, object], target_id: str, component_kind: str) -> None:
    remaining = set(mapping) - {target_id}
    if remaining:
        raise InvariantError(f"cannot advance {component_kind} stage while other pending intents remain")


def transition_source_plugin_selection(
    session: GuidedSession,
    *,
    turn: AnsweredTurn,
    response: PluginSelectionResponse,
    permitted_plugins: Sequence[str],
    inspection_facts: SourceInspectionFacts | None,
    new_stable_id: UUID | None = None,
    target_id: str | None = None,
) -> StageTransitionResult:
    """Move one source from plugin selection to plugin options."""

    _require_active_turn(
        session,
        turn,
        expected_step=GuidedStep.STEP_1_SOURCE,
        expected_turn_type=TurnType.SINGLE_SELECT,
    )
    plugin = _selected_plugin(response, permitted_plugins)
    facts = _validated_inspection_facts(inspection_facts) if inspection_facts is not None else None
    if facts is not None:
        _require_inspection_plugin_match(plugin, facts)
    stable_id, name, created = _source_selection_target(
        session,
        target_id=target_id,
        new_stable_id=new_stable_id,
    )
    pending = dict(session.pending_source_intents)
    pending[stable_id] = SourceIntent(
        name=name,
        phase="plugin_options",
        plugin=plugin,
        options=None,
        inspection_facts=facts,
        observed_columns=(),
        sample_rows=(),
    )
    source_order = (*session.source_order, stable_id) if created else session.source_order
    replacement = replace(session, source_order=source_order, pending_source_intents=pending)
    return StageTransitionResult(
        session=replacement,
        next_turn=SourceSchemaFormPreparation(stable_id=stable_id, plugin=plugin, inspection_facts=facts),
    )


def transition_source_schema_form(
    session: GuidedSession,
    *,
    target_id: str,
    turn: AnsweredTurn,
    response: SchemaFormResponse,
    authority: SchemaFormAuthority,
) -> StageTransitionResult:
    """Validate source options and either review inspection or finish Step 1."""

    _require_active_turn(
        session,
        turn,
        expected_step=GuidedStep.STEP_1_SOURCE,
        expected_turn_type=TurnType.SCHEMA_FORM,
    )
    stable_id, intent = _require_source_intent(session, target_id, "plugin_options")
    if response.plugin != intent.plugin:
        raise ValueError("source schema-form plugin echo does not match the server-held source intent")
    if intent.plugin is None:
        raise InvariantError("plugin_options source intent is missing its server-held plugin")
    options, structural = _validated_merged_options(
        response,
        authority,
        plugin_kind="source",
        plugin_name=intent.plugin,
        structural_defaults={"on_validation_failure": "discard"},
    )

    if intent.inspection_facts is not None:
        facts = _validated_inspection_facts(intent.inspection_facts)
        _require_inspection_plugin_match(intent.plugin, facts)
        _require_inspection_custody_match(
            facts,
            submitted_and_server_options=options,
            server_options=authority.server_options,
        )
        pending = dict(session.pending_source_intents)
        pending[stable_id] = SourceIntent(
            name=intent.name,
            phase="inspection_review",
            plugin=intent.plugin,
            options=_pending_options_with_structural(options, structural),
            inspection_facts=facts,
            observed_columns=facts.observed_headers or (),
            sample_rows=(),
        )
        replacement = replace(session, pending_source_intents=pending)
        return StageTransitionResult(
            session=replacement,
            next_turn=SourceInspectionPreparation(stable_id=stable_id, inspection_facts=facts),
        )

    _require_no_other_pending(session.pending_source_intents, stable_id, "source")
    reviewed = dict(session.reviewed_sources)
    reviewed[stable_id] = SourceResolved(
        name=intent.name,
        plugin=intent.plugin,
        options=options,
        observed_columns=(),
        sample_rows=(),
        on_validation_failure=structural["on_validation_failure"],
    )
    pending = dict(session.pending_source_intents)
    del pending[stable_id]
    replacement = replace(
        session,
        step=GuidedStep.STEP_2_SINK,
        reviewed_sources=reviewed,
        pending_source_intents=pending,
    )
    return StageTransitionResult(session=replacement, next_turn=SinkPluginSelectionPreparation())


def transition_source_inspection_review(
    session: GuidedSession,
    *,
    target_id: str,
    turn: AnsweredTurn,
    response: InspectionResponse,
) -> StageTransitionResult:
    """Resolve one inspection-review source using only edited column names."""

    _require_active_turn(
        session,
        turn,
        expected_step=GuidedStep.STEP_1_SOURCE,
        expected_turn_type=TurnType.INSPECT_AND_CONFIRM,
    )
    if type(response) is not InspectionResponse:
        raise TypeError("response must be InspectionResponse")
    columns = tuple(response.columns)
    if not columns:
        raise ValueError("inspection confirmation requires at least one column")
    if any(column == "" for column in columns):
        raise ValueError("inspection columns must be non-empty")
    if len(set(columns)) != len(columns):
        raise ValueError("inspection columns must be unique")
    stable_id, intent = _require_source_intent(session, target_id, "inspection_review")
    if intent.plugin is None or intent.options is None or intent.inspection_facts is None:
        raise InvariantError("inspection-review source intent is missing server-held resolution facts")
    facts = _validated_inspection_facts(intent.inspection_facts)
    _require_inspection_plugin_match(intent.plugin, facts)
    _require_inspection_custody_match(
        facts,
        submitted_and_server_options=intent.options,
        server_options={},
    )
    _require_no_other_pending(session.pending_source_intents, stable_id, "source")
    reviewed = dict(session.reviewed_sources)
    plugin_options, on_validation_failure = _split_pending_structural(intent.options, "on_validation_failure")
    reviewed[stable_id] = SourceResolved(
        name=intent.name,
        plugin=intent.plugin,
        options=plugin_options,
        observed_columns=columns,
        sample_rows=(),
        on_validation_failure=on_validation_failure,
    )
    pending = dict(session.pending_source_intents)
    del pending[stable_id]
    replacement = replace(
        session,
        step=GuidedStep.STEP_2_SINK,
        reviewed_sources=reviewed,
        pending_source_intents=pending,
    )
    return StageTransitionResult(session=replacement, next_turn=SinkPluginSelectionPreparation())


def transition_sink_plugin_selection(
    session: GuidedSession,
    *,
    turn: AnsweredTurn,
    response: PluginSelectionResponse,
    permitted_plugins: Sequence[str],
    new_stable_id: UUID | None = None,
    target_id: str | None = None,
) -> StageTransitionResult:
    """Move one output from plugin selection to plugin options."""

    _require_active_turn(
        session,
        turn,
        expected_step=GuidedStep.STEP_2_SINK,
        expected_turn_type=TurnType.SINGLE_SELECT,
    )
    if not session.reviewed_sources:
        raise InvariantError("sink selection requires at least one reviewed source")
    _candidate_fields(session)
    plugin = _selected_plugin(response, permitted_plugins)
    stable_id, name, created = _sink_selection_target(
        session,
        target_id=target_id,
        new_stable_id=new_stable_id,
    )
    pending = dict(session.pending_output_intents)
    pending[stable_id] = SinkIntent(name=name, phase="plugin_options", plugin=plugin, options=None)
    output_order = (*session.output_order, stable_id) if created else session.output_order
    replacement = replace(session, output_order=output_order, pending_output_intents=pending)
    return StageTransitionResult(
        session=replacement,
        next_turn=SinkSchemaFormPreparation(stable_id=stable_id, plugin=plugin),
    )


def transition_sink_schema_form(
    session: GuidedSession,
    *,
    target_id: str,
    turn: AnsweredTurn,
    response: SchemaFormResponse,
    authority: SchemaFormAuthority,
) -> StageTransitionResult:
    """Hold validated sink options and prepare required-field review."""

    _require_active_turn(
        session,
        turn,
        expected_step=GuidedStep.STEP_2_SINK,
        expected_turn_type=TurnType.SCHEMA_FORM,
    )
    stable_id, intent = _require_sink_intent(session, target_id, "plugin_options")
    if response.plugin != intent.plugin:
        raise ValueError("sink schema-form plugin echo does not match the server-held output intent")
    if intent.plugin is None:
        raise InvariantError("plugin_options output intent is missing its server-held plugin")
    options, structural = _validated_merged_options(
        response,
        authority,
        plugin_kind="sink",
        plugin_name=intent.plugin,
        structural_defaults={"on_write_failure": "discard"},
    )
    pending = dict(session.pending_output_intents)
    pending[stable_id] = SinkIntent(
        name=intent.name,
        phase="field_review",
        plugin=intent.plugin,
        options=_pending_options_with_structural(options, structural),
    )
    candidates = _candidate_fields(session)
    replacement = replace(session, pending_output_intents=pending)
    return StageTransitionResult(
        session=replacement,
        next_turn=SinkFieldReviewPreparation(stable_id=stable_id, candidate_fields=candidates),
    )


def transition_sink_field_review(
    session: GuidedSession,
    *,
    target_id: str,
    turn: AnsweredTurn,
    response: FieldSelectionResponse,
) -> StageTransitionResult:
    """Resolve one output field contract and finish Step 2."""

    _require_active_turn(
        session,
        turn,
        expected_step=GuidedStep.STEP_2_SINK,
        expected_turn_type=TurnType.MULTI_SELECT_WITH_CUSTOM,
    )
    if type(response) is not FieldSelectionResponse:
        raise TypeError("response must be FieldSelectionResponse")
    stable_id, intent = _require_sink_intent(session, target_id, "field_review")
    if intent.plugin is None or intent.options is None:
        raise InvariantError("field-review output intent is missing server-held resolution facts")
    chosen = tuple(response.chosen)
    custom = tuple(response.custom_inputs)
    if any(field_name == "" for field_name in (*chosen, *custom)):
        raise ValueError("selected and custom field names must be non-empty")
    if len(set(chosen)) != len(chosen):
        raise ValueError("chosen fields must be unique")
    if len(set(custom)) != len(custom):
        raise ValueError("custom fields must be unique")
    candidates = _candidate_fields(session)
    unknown = set(chosen) - set(candidates)
    if unknown:
        raise ValueError(f"chosen fields are not a subset of the server-emitted candidates: {sorted(unknown)!r}")
    overlap = set(custom) & (set(candidates) | set(chosen))
    if overlap:
        raise ValueError(f"custom fields overlap server-emitted or chosen fields: {sorted(overlap)!r}")
    passthrough = response.control_signal is ControlSignal.PASSTHROUGH
    if response.control_signal not in (None, ControlSignal.PASSTHROUGH):
        raise ValueError("field review accepts only the passthrough control signal")
    if passthrough and (chosen or custom):
        raise ValueError("passthrough cannot be combined with selected or custom fields")
    if not passthrough and not chosen and not custom:
        raise ValueError("field review requires fields or explicit passthrough")

    _require_no_other_pending(session.pending_output_intents, stable_id, "output")
    reviewed = dict(session.reviewed_outputs)
    plugin_options, on_write_failure = _split_pending_structural(intent.options, "on_write_failure")
    reviewed[stable_id] = SinkOutputResolved(
        name=intent.name,
        plugin=intent.plugin,
        options=plugin_options,
        required_fields=() if passthrough else (*chosen, *custom),
        schema_mode=_sink_schema_mode(plugin_options),
        on_write_failure=on_write_failure,
    )
    pending = dict(session.pending_output_intents)
    del pending[stable_id]
    replacement = replace(
        session,
        step=GuidedStep.STEP_3_TRANSFORMS,
        reviewed_outputs=reviewed,
        pending_output_intents=pending,
    )
    return StageTransitionResult(session=replacement, next_turn=None)


__all__ = [
    "GUIDED_TURN_TOKEN_SCHEMA",
    "AnsweredTurn",
    "FieldSelectionResponse",
    "InspectionResponse",
    "NextTurnPreparation",
    "PluginSelectionResponse",
    "SchemaFormAuthority",
    "SchemaFormResponse",
    "SinkFieldReviewPreparation",
    "SinkPluginSelectionPreparation",
    "SinkSchemaFormPreparation",
    "SourceInspectionPreparation",
    "SourceSchemaFormPreparation",
    "StageTransitionResult",
    "guided_turn_token",
    "transition_sink_field_review",
    "transition_sink_plugin_selection",
    "transition_sink_schema_form",
    "transition_source_inspection_review",
    "transition_source_plugin_selection",
    "transition_source_schema_form",
]
