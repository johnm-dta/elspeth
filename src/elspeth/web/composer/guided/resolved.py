"""Reviewed guided source and output facts for the schema-8 checkpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, cast

from elspeth.contracts.freeze import FrozenJsonArray, deep_thaw, freeze_fields
from elspeth.core.canonical import canonical_json
from elspeth.web.composer.bounded_json import (
    JSON_MAX_DEPTH,
    JSON_MAX_ITEMS,
    JSON_MAX_STRING_CHARS,
    JSON_MAX_TOTAL_TEXT_CHARS,
    JSON_MAX_TOTAL_UTF8_BYTES,
)
from elspeth.web.composer.guided.errors import InvariantError

GUIDED_JSON_MAX_DEPTH: Final[int] = JSON_MAX_DEPTH
GUIDED_JSON_MAX_ITEMS: Final[int] = JSON_MAX_ITEMS
GUIDED_JSON_MAX_STRING_CHARS: Final[int] = JSON_MAX_STRING_CHARS
GUIDED_JSON_MAX_TOTAL_TEXT_CHARS: Final[int] = JSON_MAX_TOTAL_TEXT_CHARS
GUIDED_JSON_MAX_TOTAL_UTF8_BYTES: Final[int] = JSON_MAX_TOTAL_UTF8_BYTES


@dataclass(slots=True)
class GuidedJsonBudget:
    items: int = 0
    text_chars: int = 0
    utf8_bytes: int = 0

    def consume_text(self, value: str, field_name: str, path: str) -> None:
        self.text_chars += len(value)
        if self.text_chars > GUIDED_JSON_MAX_TOTAL_TEXT_CHARS:
            raise InvariantError(
                f"{field_name} exceeds the aggregate JSON text character limit of {GUIDED_JSON_MAX_TOTAL_TEXT_CHARS} at {path}"
            )
        try:
            encoded_length = len(value.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise InvariantError(f"{field_name} contains text that is not valid UTF-8 at {path}") from exc
        self.utf8_bytes += encoded_length
        if self.utf8_bytes > GUIDED_JSON_MAX_TOTAL_UTF8_BYTES:
            raise InvariantError(f"{field_name} exceeds the {GUIDED_JSON_MAX_TOTAL_UTF8_BYTES}-byte aggregate JSON UTF-8 limit at {path}")


def _validate_and_freeze_guided_json(
    value: object,
    field_name: str,
    *,
    path: str,
    depth: int,
    budget: GuidedJsonBudget,
    active_container_ids: set[int],
) -> Any:
    """Validate, bound, detach, and freeze one strict JSON value."""
    if depth > GUIDED_JSON_MAX_DEPTH:
        raise InvariantError(f"{field_name} exceeds the {GUIDED_JSON_MAX_DEPTH}-level JSON depth limit at {path}")

    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in active_container_ids:
            raise InvariantError(f"{field_name} contains a recursive JSON mapping at {path}")
        active_container_ids.add(container_id)
        try:
            frozen_children: dict[str, Any] = {}
            for key, child in value.items():
                if type(key) is not str:
                    raise InvariantError(f"{field_name} key at {path} must be an exact str")
                if len(key) > GUIDED_JSON_MAX_STRING_CHARS:
                    raise InvariantError(f"{field_name} key at {path} exceeds the JSON string limit")
                budget.consume_text(key, field_name, path)
                budget.items += 1
                if budget.items > GUIDED_JSON_MAX_ITEMS:
                    raise InvariantError(f"{field_name} exceeds the {GUIDED_JSON_MAX_ITEMS}-item JSON limit")
                frozen_children[key] = _validate_and_freeze_guided_json(
                    child,
                    field_name,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                    budget=budget,
                    active_container_ids=active_container_ids,
                )
        finally:
            active_container_ids.remove(container_id)
        return MappingProxyType(frozen_children)

    if type(value) in {list, FrozenJsonArray}:
        sequence = cast(Sequence[object], value)
        container_id = id(value)
        if container_id in active_container_ids:
            raise InvariantError(f"{field_name} contains a recursive JSON list at {path}")
        active_container_ids.add(container_id)
        try:
            frozen_items: list[Any] = []
            for index, child in enumerate(sequence):
                budget.items += 1
                if budget.items > GUIDED_JSON_MAX_ITEMS:
                    raise InvariantError(f"{field_name} exceeds the {GUIDED_JSON_MAX_ITEMS}-item JSON limit")
                frozen_items.append(
                    _validate_and_freeze_guided_json(
                        child,
                        field_name,
                        path=f"{path}[{index}]",
                        depth=depth + 1,
                        budget=budget,
                        active_container_ids=active_container_ids,
                    )
                )
        finally:
            active_container_ids.remove(container_id)
        return FrozenJsonArray(frozen_items)

    if value is None or type(value) in {bool, int, float, str}:
        if type(value) is str and len(value) > GUIDED_JSON_MAX_STRING_CHARS:
            raise InvariantError(f"{field_name} JSON string at {path} exceeds {GUIDED_JSON_MAX_STRING_CHARS} characters")
        if type(value) is str:
            budget.consume_text(value, field_name, path)
        try:
            canonical_json(value)
        except (TypeError, ValueError) as exc:
            raise InvariantError(f"{field_name} JSON number at {path} is outside the canonical JSON domain") from exc
        return value

    raise InvariantError(f"{field_name} value at {path} must be an exact JSON leaf, list, or mapping; got {type(value).__name__}")


def freeze_guided_json_mapping(value: object, field_name: str, *, budget: GuidedJsonBudget | None = None) -> Mapping[str, Any]:
    """Return a detached immutable strict-JSON object with repository bounds."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    frozen = _validate_and_freeze_guided_json(
        value,
        field_name,
        path="$",
        depth=0,
        budget=budget if budget is not None else GuidedJsonBudget(),
        active_container_ids=set(),
    )
    return cast(Mapping[str, Any], frozen)


def freeze_guided_str_sequence(
    value: object,
    field_name: str,
    *,
    budget: GuidedJsonBudget | None = None,
) -> tuple[str, ...]:
    """Validate and snapshot one bounded sequence of exact strings."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence[str]")
    if len(value) > GUIDED_JSON_MAX_ITEMS:
        raise InvariantError(f"{field_name} exceeds the {GUIDED_JSON_MAX_ITEMS}-item limit")
    text_budget = budget if budget is not None else GuidedJsonBudget()
    result: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not str:
            raise TypeError(f"{field_name}[{index}] must be an exact str")
        if len(item) > GUIDED_JSON_MAX_STRING_CHARS:
            raise InvariantError(f"{field_name}[{index}] exceeds {GUIDED_JSON_MAX_STRING_CHARS} characters")
        text_budget.consume_text(item, field_name, f"$[{index}]")
        result.append(item)
    return tuple(result)


def _require_exact_keys(value: object, expected: frozenset[str], owner: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise InvariantError(f"{owner}.from_dict: record must be an exact dict")
    record = value
    unexpected = set(record) - expected
    if unexpected:
        raise InvariantError(f"{owner}.from_dict: unexpected keys {sorted(unexpected)!r}")
    missing = expected - set(record)
    if missing:
        raise InvariantError(f"{owner}.from_dict: missing keys {sorted(missing)!r}")
    return record


def _require_nonempty_str(value: object, field_name: str) -> str:
    if type(value) is not str or value == "":
        raise InvariantError(f"{field_name} must be a non-empty exact str")
    return value


def _require_str_list(value: object, field_name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise InvariantError(f"{field_name} must be a list[str]")
    result: list[str] = []
    for item in value:
        if type(item) is not str:
            raise InvariantError(f"{field_name} must be a list[str]")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SourceResolved:
    """One reviewed source, named independently from its stable component id."""

    name: str
    plugin: str
    options: Mapping[str, Any]
    observed_columns: Sequence[str]
    sample_rows: Sequence[Mapping[str, Any]]
    on_validation_failure: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.name, "SourceResolved.name")
        _require_nonempty_str(self.plugin, "SourceResolved.plugin")
        _require_nonempty_str(self.on_validation_failure, "SourceResolved.on_validation_failure")
        sample_rows_value = cast(object, self.sample_rows)
        if not isinstance(sample_rows_value, Sequence) or isinstance(sample_rows_value, (str, bytes, bytearray)):
            raise TypeError("SourceResolved.sample_rows must be a sequence[mapping]")
        if len(sample_rows_value) > GUIDED_JSON_MAX_ITEMS:
            raise InvariantError(f"SourceResolved.sample_rows exceeds the {GUIDED_JSON_MAX_ITEMS}-item limit")
        if any(not isinstance(row, Mapping) for row in self.sample_rows):
            raise TypeError("SourceResolved.sample_rows must contain mappings")
        budget = GuidedJsonBudget()
        object.__setattr__(self, "options", freeze_guided_json_mapping(self.options, "SourceResolved.options", budget=budget))
        object.__setattr__(
            self,
            "sample_rows",
            tuple(
                freeze_guided_json_mapping(row, f"SourceResolved.sample_rows[{index}]", budget=budget)
                for index, row in enumerate(self.sample_rows)
            ),
        )
        object.__setattr__(
            self,
            "observed_columns",
            freeze_guided_str_sequence(self.observed_columns, "SourceResolved.observed_columns", budget=budget),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "plugin": self.plugin,
            "options": deep_thaw(self.options),
            "observed_columns": list(deep_thaw(self.observed_columns)),
            "sample_rows": [dict(deep_thaw(row)) for row in self.sample_rows],
            "on_validation_failure": self.on_validation_failure,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceResolved:
        record = _require_exact_keys(
            d,
            frozenset({"name", "plugin", "options", "observed_columns", "sample_rows", "on_validation_failure"}),
            "SourceResolved",
        )
        if type(record["options"]) is not dict:
            raise InvariantError("SourceResolved.options must be an exact dict")
        sample_rows_raw = record["sample_rows"]
        if type(sample_rows_raw) is not list or any(type(row) is not dict for row in sample_rows_raw):
            raise InvariantError("SourceResolved.sample_rows must be a list[dict]")
        try:
            return cls(
                name=_require_nonempty_str(record["name"], "SourceResolved.name"),
                plugin=_require_nonempty_str(record["plugin"], "SourceResolved.plugin"),
                options=record["options"],
                observed_columns=_require_str_list(record["observed_columns"], "SourceResolved.observed_columns"),
                sample_rows=tuple(sample_rows_raw),
                on_validation_failure=_require_nonempty_str(record["on_validation_failure"], "SourceResolved.on_validation_failure"),
            )
        except (TypeError, ValueError) as exc:
            raise InvariantError(f"SourceResolved.from_dict: malformed record {record!r}") from exc


@dataclass(frozen=True, slots=True)
class SinkOutputResolved:
    """One reviewed named output, separate from its stable component id."""

    name: str
    plugin: str
    options: Mapping[str, Any]
    required_fields: Sequence[str]
    schema_mode: str
    on_write_failure: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.name, "SinkOutputResolved.name")
        _require_nonempty_str(self.plugin, "SinkOutputResolved.plugin")
        if self.schema_mode not in {"fixed", "flexible", "observed"}:
            raise ValueError("SinkOutputResolved.schema_mode must be fixed, flexible, or observed")
        _require_nonempty_str(self.on_write_failure, "SinkOutputResolved.on_write_failure")
        budget = GuidedJsonBudget()
        object.__setattr__(self, "options", freeze_guided_json_mapping(self.options, "SinkOutputResolved.options", budget=budget))
        object.__setattr__(
            self,
            "required_fields",
            freeze_guided_str_sequence(self.required_fields, "SinkOutputResolved.required_fields", budget=budget),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "plugin": self.plugin,
            "options": deep_thaw(self.options),
            "required_fields": list(deep_thaw(self.required_fields)),
            "schema_mode": self.schema_mode,
            "on_write_failure": self.on_write_failure,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SinkOutputResolved:
        record = _require_exact_keys(
            d,
            frozenset({"name", "plugin", "options", "required_fields", "schema_mode", "on_write_failure"}),
            "SinkOutputResolved",
        )
        if type(record["options"]) is not dict:
            raise InvariantError("SinkOutputResolved.options must be an exact dict")
        try:
            return cls(
                name=_require_nonempty_str(record["name"], "SinkOutputResolved.name"),
                plugin=_require_nonempty_str(record["plugin"], "SinkOutputResolved.plugin"),
                options=record["options"],
                required_fields=_require_str_list(record["required_fields"], "SinkOutputResolved.required_fields"),
                schema_mode=_require_nonempty_str(record["schema_mode"], "SinkOutputResolved.schema_mode"),
                on_write_failure=_require_nonempty_str(record["on_write_failure"], "SinkOutputResolved.on_write_failure"),
            )
        except (TypeError, ValueError) as exc:
            raise InvariantError(f"SinkOutputResolved.from_dict: malformed record {record!r}") from exc


@dataclass(frozen=True, slots=True)
class SinkResolved:
    """Non-persisted grouping of reviewed named outputs."""

    outputs: Sequence[SinkOutputResolved]

    def __post_init__(self) -> None:
        freeze_fields(self, "outputs")

    def to_dict(self) -> dict[str, Any]:
        return {"outputs": [output.to_dict() for output in self.outputs]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SinkResolved:
        record = _require_exact_keys(d, frozenset({"outputs"}), "SinkResolved")
        outputs_raw = record["outputs"]
        if type(outputs_raw) is not list:
            raise InvariantError("SinkResolved.outputs must be a list")
        return cls(outputs=tuple(SinkOutputResolved.from_dict(output) for output in outputs_raw))
