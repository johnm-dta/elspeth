"""Closed subjects and constraints for guided deferred-stage intent.

This module is the canonical boundary for facts that may be retained until a
later guided stage. Persisted values are audit-tier: decoders accept exact
JSON shapes only and never coerce values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast
from uuid import UUID

from elspeth.core.canonical import canonical_json
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.schemas import PluginKind
from elspeth.web.composer.guided.errors import InvariantError

StageName = Literal["source", "output", "topology", "wire_review"]
type JsonScalar = str | int | float | bool | None

_PLUGIN_KINDS: tuple[PluginKind, ...] = ("source", "transform", "sink")
_PLUGIN_KIND_SET = frozenset(_PLUGIN_KINDS)
_COMPONENT_KINDS = frozenset({"source", "node", "edge", "output"})


def stage_name_from_value(value: object, field_name: str) -> StageName:
    """Validate an exact receiving/target stage value from persisted JSON."""

    if type(value) is not str or value not in {"source", "output", "topology", "wire_review"}:
        raise InvariantError(f"{field_name} is unsupported")
    return cast(StageName, value)


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


def _require_optional_nonempty_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_str(value, field_name)


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


def _str_tuple_from_list(value: object, field_name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise InvariantError(f"{field_name} must be a list[str]")
    return tuple(_require_nonempty_str(item, field_name) for item in value)


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


@dataclass(frozen=True, slots=True)
class StableSubject:
    kind: Literal["stable"]
    component_kind: Literal["source", "node", "edge", "output"]
    stable_id: str

    def __post_init__(self) -> None:
        if self.kind != "stable":
            raise InvariantError("StableSubject.kind must be 'stable'")
        if self.component_kind not in _COMPONENT_KINDS:
            raise InvariantError("StableSubject.component_kind is unsupported")
        _canonical_uuid_text(self.stable_id, "StableSubject.stable_id")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "component_kind": self.component_kind, "stable_id": self.stable_id}


@dataclass(frozen=True, slots=True)
class PluginSubject:
    kind: Literal["plugin"]
    subject_id: str
    plugin_kind: PluginKind
    plugin_name: str

    def __post_init__(self) -> None:
        if self.kind != "plugin":
            raise InvariantError("PluginSubject.kind must be 'plugin'")
        _canonical_uuid_text(self.subject_id, "PluginSubject.subject_id")
        if self.plugin_kind not in _PLUGIN_KIND_SET:
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


def subject_from_dict(value: object) -> DeferredSubject:
    """Decode one exact persisted deferred subject."""

    if type(value) is not dict:
        raise InvariantError("Deferred subject must be an exact dict")
    kind = value.get("kind")
    if kind == "stable":
        record = _require_exact_dict(value, frozenset({"kind", "component_kind", "stable_id"}), "StableSubject.from_dict")
        component_kind = record["component_kind"]
        if component_kind not in _COMPONENT_KINDS:
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
        if plugin_kind not in _PLUGIN_KIND_SET:
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


class OptionValueConstraintData(TypedDict):
    kind: str
    subject: dict[str, str]
    option_path: list[str]
    operator: str
    value: JsonScalar


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
    plugin_kind: PluginKind | None
    plugin_name: str | None
    operator: Literal["equals", "at_least", "at_most"]
    count: int

    def __post_init__(self) -> None:
        if self.kind != "component_count" or self.component_kind not in _COMPONENT_KINDS:
            raise InvariantError("ComponentCountConstraint component kind is unsupported")
        if (self.plugin_kind is None) != (self.plugin_name is None):
            raise InvariantError("ComponentCountConstraint plugin_kind/plugin_name must be paired")
        if self.plugin_kind is not None and self.plugin_kind not in _PLUGIN_KIND_SET:
            raise InvariantError("ComponentCountConstraint.plugin_kind is unsupported")
        expected_plugin_kind: dict[str, PluginKind] = {"source": "source", "node": "transform", "output": "sink"}
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


def constraint_from_dict(value: object) -> DeferredConstraint:
    """Decode one exact persisted deferred constraint."""

    if type(value) is not dict:
        raise InvariantError("Deferred constraint must be an exact dict")
    kind = value.get("kind")
    if kind == "subject_presence":
        record = _require_exact_dict(value, frozenset({"kind", "subject", "present"}), "SubjectPresenceConstraint.from_dict")
        if type(record["present"]) is not bool:
            raise InvariantError("SubjectPresenceConstraint.present must be an exact bool")
        return SubjectPresenceConstraint(kind="subject_presence", subject=subject_from_dict(record["subject"]), present=record["present"])
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
            subject=subject_from_dict(record["subject"]),
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
            from_subject=subject_from_dict(record["from_subject"]),
            edge_type=cast(Any, record["edge_type"]),
            to_subject=subject_from_dict(record["to_subject"]),
            present=record["present"],
        )
    if kind == "failure_route":
        record = _require_exact_dict(
            value,
            frozenset({"kind", "subject", "failure_kind", "operator", "target"}),
            "FailureRouteConstraint.from_dict",
        )
        target_raw = record["target"]
        target: FailureTarget = "discard" if target_raw == "discard" else subject_from_dict(target_raw)
        return FailureRouteConstraint(
            kind="failure_route",
            subject=subject_from_dict(record["subject"]),
            failure_kind=cast(Any, record["failure_kind"]),
            operator=cast(Any, record["operator"]),
            target=target,
        )
    raise InvariantError("Deferred constraint kind is unsupported")


@dataclass(frozen=True, slots=True)
class CatalogSubjectResolved:
    status: Literal["resolved"]
    plugin_kind: PluginKind
    plugin_name: str

    def __post_init__(self) -> None:
        if self.status != "resolved" or self.plugin_kind not in _PLUGIN_KIND_SET:
            raise InvariantError("CatalogSubjectResolved is malformed")
        _require_nonempty_str(self.plugin_name, "CatalogSubjectResolved.plugin_name")


@dataclass(frozen=True, slots=True)
class CatalogSubjectUnsupported:
    status: Literal["unsupported"]
    plugin_name: str
    expected_kind: PluginKind | None
    visible_kinds: tuple[PluginKind, ...]

    def __post_init__(self) -> None:
        if self.status != "unsupported":
            raise InvariantError("CatalogSubjectUnsupported.status is unsupported")
        _require_nonempty_str(self.plugin_name, "CatalogSubjectUnsupported.plugin_name")
        if self.expected_kind is not None and self.expected_kind not in _PLUGIN_KIND_SET:
            raise InvariantError("CatalogSubjectUnsupported.expected_kind is unsupported")
        if type(self.visible_kinds) is not tuple or self.visible_kinds != tuple(
            kind for kind in _PLUGIN_KINDS if kind in self.visible_kinds
        ):
            raise InvariantError("CatalogSubjectUnsupported.visible_kinds must be unique and canonical")


@dataclass(frozen=True, slots=True)
class CatalogSubjectClarification:
    status: Literal["clarification"]
    plugin_name: str
    plugin_kinds: tuple[PluginKind, ...]

    def __post_init__(self) -> None:
        if self.status != "clarification":
            raise InvariantError("CatalogSubjectClarification.status is unsupported")
        _require_nonempty_str(self.plugin_name, "CatalogSubjectClarification.plugin_name")
        canonical = tuple(kind for kind in _PLUGIN_KINDS if kind in self.plugin_kinds)
        if type(self.plugin_kinds) is not tuple or len(self.plugin_kinds) < 2 or self.plugin_kinds != canonical:
            raise InvariantError("CatalogSubjectClarification.plugin_kinds must contain multiple unique canonical kinds")


type CatalogSubjectResolution = CatalogSubjectResolved | CatalogSubjectUnsupported | CatalogSubjectClarification


def resolve_catalog_subject(
    catalog: PolicyCatalogView,
    *,
    plugin_name: str,
    expected_kind: PluginKind | None = None,
) -> CatalogSubjectResolution:
    """Resolve a name against one request-scoped, policy-filtered catalog."""

    name = _require_nonempty_str(plugin_name, "resolve_catalog_subject.plugin_name")
    visible_by_kind = {
        "source": catalog.list_sources(),
        "transform": catalog.list_transforms(),
        "sink": catalog.list_sinks(),
    }
    visible_kinds = tuple(kind for kind in _PLUGIN_KINDS if any(item.name == name for item in visible_by_kind[kind]))
    if expected_kind is not None:
        if expected_kind in visible_kinds:
            return CatalogSubjectResolved(status="resolved", plugin_kind=expected_kind, plugin_name=name)
        return CatalogSubjectUnsupported(
            status="unsupported",
            plugin_name=name,
            expected_kind=expected_kind,
            visible_kinds=visible_kinds,
        )
    if len(visible_kinds) == 1:
        return CatalogSubjectResolved(status="resolved", plugin_kind=visible_kinds[0], plugin_name=name)
    if len(visible_kinds) > 1:
        return CatalogSubjectClarification(status="clarification", plugin_name=name, plugin_kinds=visible_kinds)
    return CatalogSubjectUnsupported(status="unsupported", plugin_name=name, expected_kind=None, visible_kinds=())
