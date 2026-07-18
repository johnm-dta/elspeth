"""Reviewed guided source and output facts for the schema-8 checkpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.web.composer.guided.errors import InvariantError


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

    plugin: str
    options: Mapping[str, Any]
    observed_columns: Sequence[str]
    sample_rows: Sequence[Mapping[str, Any]]
    on_validation_failure: str = "discard"
    # Temporary construction default for pre-cutover internal callers. Schema 8
    # always persists the key and its enclosing session enforces name uniqueness.
    name: str = "source"

    def __post_init__(self) -> None:
        _require_nonempty_str(self.name, "SourceResolved.name")
        _require_nonempty_str(self.plugin, "SourceResolved.plugin")
        _require_nonempty_str(self.on_validation_failure, "SourceResolved.on_validation_failure")
        if not isinstance(self.options, Mapping):
            raise TypeError("SourceResolved.options must be a mapping")
        if not isinstance(self.observed_columns, Sequence) or isinstance(self.observed_columns, (str, bytes)):
            raise TypeError("SourceResolved.observed_columns must be a sequence[str]")
        if any(type(column) is not str for column in self.observed_columns):
            raise TypeError("SourceResolved.observed_columns must contain exact str values")
        if not isinstance(self.sample_rows, Sequence):
            raise TypeError("SourceResolved.sample_rows must be a sequence[mapping]")
        if any(not isinstance(row, Mapping) for row in self.sample_rows):
            raise TypeError("SourceResolved.sample_rows must contain mappings")
        freeze_fields(self, "options", "observed_columns", "sample_rows")

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

    plugin: str
    options: Mapping[str, Any]
    required_fields: Sequence[str]
    schema_mode: str
    # Temporary construction defaults for pre-cutover internal callers. Schema
    # 8 persists both keys explicitly.
    name: str = "main"
    on_write_failure: str = "discard"

    def __post_init__(self) -> None:
        _require_nonempty_str(self.name, "SinkOutputResolved.name")
        _require_nonempty_str(self.plugin, "SinkOutputResolved.plugin")
        if self.schema_mode not in {"fixed", "flexible", "observed"}:
            raise ValueError("SinkOutputResolved.schema_mode must be fixed, flexible, or observed")
        _require_nonempty_str(self.on_write_failure, "SinkOutputResolved.on_write_failure")
        if not isinstance(self.options, Mapping):
            raise TypeError("SinkOutputResolved.options must be a mapping")
        if not isinstance(self.required_fields, Sequence) or isinstance(self.required_fields, (str, bytes)):
            raise TypeError("SinkOutputResolved.required_fields must be a sequence[str]")
        if any(type(field) is not str for field in self.required_fields):
            raise TypeError("SinkOutputResolved.required_fields must contain exact str values")
        freeze_fields(self, "options", "required_fields")

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
    """Temporary non-persisted compatibility carrier for pre-cutover callers."""

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
