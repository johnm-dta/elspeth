"""Creation and validation boundary for guided wrong-stage intent.

The model may suggest only :class:`DeferredIntentAction`.  The server then
validates that suggestion against the request-scoped policy catalog and the
current guided stage before creating durable audit-tier state.  Raw user text
is deliberately accepted only by :func:`create_deferred_stage_intent`, where
it is reduced to a content hash; it is never stored in deferred metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

from elspeth.contracts.freeze import deep_thaw
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.schemas import PluginKind
from elspeth.web.composer.guided.connection_consumers import ConsumerIdentity, canonical_connection_consumers
from elspeth.web.composer.guided.errors import GuidedSolverResponseShapeError, InvariantError
from elspeth.web.composer.guided.stage_subjects import (
    CatalogSubjectClarification,
    CatalogSubjectResolved,
    CatalogSubjectUnsupported,
    ComponentCountConstraint,
    DeferredConstraint,
    EdgeRouteConstraint,
    FailureRouteConstraint,
    OptionValueConstraint,
    PluginSubject,
    StableSubject,
    StageName,
    SubjectPresenceConstraint,
    constraint_from_dict,
    resolve_catalog_subject,
    stage_name_from_value,
)
from elspeth.web.composer.guided.state_machine import (
    GUIDED_MAX_CONSTRAINTS_PER_INTENT,
    GUIDED_MAX_REDACTED_SUMMARY_CHARS,
    DeferredStageIntent,
    GuidedSession,
)
from elspeth.web.composer.state import CompositionState
from elspeth.web.plugin_policy.models import PluginId, PluginUnavailableReason

_STAGE_ORDINAL: dict[StageName, int] = {"source": 0, "output": 1, "topology": 2, "wire_review": 3}
_PLUGIN_STAGE: dict[PluginKind, StageName] = {"source": "source", "sink": "output", "transform": "topology"}
_COMPONENT_STAGE: dict[str, StageName] = {
    "source": "source",
    "output": "output",
    "node": "topology",
    "edge": "wire_review",
}
_ACTION_KEYS = frozenset({"target_stage", "catalog_kind", "catalog_name", "redacted_summary", "constraints"})
_ALLOWED_CONSTRAINT_TYPES = {
    SubjectPresenceConstraint,
    OptionValueConstraint,
    ComponentCountConstraint,
    EdgeRouteConstraint,
    FailureRouteConstraint,
}


class DeferredIntentActionShapeError(GuidedSolverResponseShapeError):
    """The model emitted a malformed future-stage action."""


class DeferredIntentClaimError(ValueError):
    """A planner terminal claimed deferred coverage it did not prove."""


def _require_nonempty_exact_str(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise InvariantError(f"{field_name} must be a non-empty exact str")
    return value


@dataclass(frozen=True, slots=True)
class DeferredIntentAction:
    """One model-suggested, non-authoritative future-stage instruction."""

    target_stage: StageName
    catalog_kind: PluginKind | None
    catalog_name: str | None
    redacted_summary: str
    constraints: tuple[DeferredConstraint, ...]

    def __post_init__(self) -> None:
        if self.target_stage not in _STAGE_ORDINAL:
            raise InvariantError("DeferredIntentAction.target_stage is unsupported")
        if (self.catalog_kind is None) != (self.catalog_name is None):
            raise InvariantError("DeferredIntentAction catalog fields must be paired")
        if self.catalog_kind is not None and self.catalog_kind not in _PLUGIN_STAGE:
            raise InvariantError("DeferredIntentAction.catalog_kind is unsupported")
        if self.catalog_name is not None:
            _require_nonempty_exact_str(self.catalog_name, "DeferredIntentAction.catalog_name")
        _require_nonempty_exact_str(self.redacted_summary, "DeferredIntentAction.redacted_summary")
        if len(self.redacted_summary) > GUIDED_MAX_REDACTED_SUMMARY_CHARS:
            raise InvariantError(f"DeferredIntentAction.redacted_summary exceeds {GUIDED_MAX_REDACTED_SUMMARY_CHARS} characters")
        if type(self.constraints) is not tuple:
            raise InvariantError("DeferredIntentAction.constraints must be an exact tuple")
        if not self.constraints:
            raise InvariantError("DeferredIntentAction requires at least one structural constraint")
        if len(self.constraints) > GUIDED_MAX_CONSTRAINTS_PER_INTENT:
            raise InvariantError(f"DeferredIntentAction.constraints exceeds the {GUIDED_MAX_CONSTRAINTS_PER_INTENT}-constraint limit")
        if any(type(constraint) not in _ALLOWED_CONSTRAINT_TYPES for constraint in self.constraints):
            raise InvariantError("DeferredIntentAction.constraints contains an unsupported constraint")


@dataclass(frozen=True, slots=True)
class DeferredIntentAccepted:
    action: DeferredIntentAction

    def __post_init__(self) -> None:
        if type(self.action) is not DeferredIntentAction:
            raise InvariantError("DeferredIntentAccepted.action must be exact")


@dataclass(frozen=True, slots=True)
class DeferredIntentClarification:
    plugin_name: str
    plugin_kinds: tuple[PluginKind, ...]

    def __post_init__(self) -> None:
        _require_nonempty_exact_str(self.plugin_name, "DeferredIntentClarification.plugin_name")
        canonical = tuple(kind for kind in ("source", "transform", "sink") if kind in self.plugin_kinds)
        if type(self.plugin_kinds) is not tuple or len(self.plugin_kinds) < 2 or self.plugin_kinds != canonical:
            raise InvariantError("DeferredIntentClarification.plugin_kinds must be multiple unique canonical kinds")


@dataclass(frozen=True, slots=True)
class DeferredIntentUnsupported:
    plugin_kind: PluginKind
    plugin_name: str
    reason: PluginUnavailableReason

    def __post_init__(self) -> None:
        if self.plugin_kind not in _PLUGIN_STAGE:
            raise InvariantError("DeferredIntentUnsupported.plugin_kind is unsupported")
        _require_nonempty_exact_str(self.plugin_name, "DeferredIntentUnsupported.plugin_name")
        if type(self.reason) is not PluginUnavailableReason:
            raise InvariantError("DeferredIntentUnsupported.reason must be exact")


DeferredIntentRejectionReason = Literal[
    "target_not_later",
    "wrong_responsible_stage",
    "catalog_kind_mismatch",
    "malformed_catalog_identity",
    "option_value_unproven",
]


@dataclass(frozen=True, slots=True)
class DeferredIntentRejected:
    reason: DeferredIntentRejectionReason

    def __post_init__(self) -> None:
        if self.reason not in {
            "target_not_later",
            "wrong_responsible_stage",
            "catalog_kind_mismatch",
            "malformed_catalog_identity",
            "option_value_unproven",
        }:
            raise InvariantError("DeferredIntentRejected.reason is unsupported")


type DeferredIntentValidation = DeferredIntentAccepted | DeferredIntentClarification | DeferredIntentUnsupported | DeferredIntentRejected


def deferred_intent_action_from_dict(value: object) -> DeferredIntentAction:
    """Decode one exact LLM tool argument object into the closed action."""

    try:
        if type(value) is not dict:
            raise InvariantError("DeferredIntentAction must be an exact dict")
        unexpected = set(value) - _ACTION_KEYS
        if unexpected:
            raise InvariantError(f"DeferredIntentAction has unexpected keys {sorted(unexpected)!r}")
        missing = _ACTION_KEYS - set(value)
        if missing:
            raise InvariantError(f"DeferredIntentAction has missing keys {sorted(missing)!r}")
        constraints_raw = value["constraints"]
        if type(constraints_raw) is not list:
            raise InvariantError("DeferredIntentAction.constraints must be a list")
        if len(constraints_raw) > GUIDED_MAX_CONSTRAINTS_PER_INTENT:
            raise InvariantError(f"DeferredIntentAction.constraints exceeds the {GUIDED_MAX_CONSTRAINTS_PER_INTENT}-constraint limit")
        catalog_kind = value["catalog_kind"]
        if catalog_kind is not None and catalog_kind not in _PLUGIN_STAGE:
            raise InvariantError("DeferredIntentAction.catalog_kind is unsupported")
        catalog_name = value["catalog_name"]
        if catalog_name is not None:
            catalog_name = _require_nonempty_exact_str(catalog_name, "DeferredIntentAction.catalog_name")
        return DeferredIntentAction(
            target_stage=stage_name_from_value(value["target_stage"], "DeferredIntentAction.target_stage"),
            catalog_kind=cast(PluginKind | None, catalog_kind),
            catalog_name=catalog_name,
            redacted_summary=_require_nonempty_exact_str(value["redacted_summary"], "DeferredIntentAction.redacted_summary"),
            constraints=tuple(constraint_from_dict(item) for item in constraints_raw),
        )
    except (InvariantError, KeyError, TypeError, ValueError) as exc:
        raise DeferredIntentActionShapeError(str(exc)) from exc


def _subject_stage(subject: StableSubject | PluginSubject) -> StageName:
    if type(subject) is StableSubject:
        return _COMPONENT_STAGE[subject.component_kind]
    if type(subject) is PluginSubject:
        return _PLUGIN_STAGE[subject.plugin_kind]
    raise InvariantError("DeferredIntentAction constraint subject is malformed")


def _constraint_stage(constraint: DeferredConstraint) -> StageName:
    if type(constraint) is SubjectPresenceConstraint:
        return _subject_stage(constraint.subject)
    if type(constraint) is OptionValueConstraint:
        return _subject_stage(constraint.subject)
    if type(constraint) is ComponentCountConstraint:
        return _COMPONENT_STAGE[constraint.component_kind]
    if type(constraint) is EdgeRouteConstraint:
        return "wire_review"
    if type(constraint) is FailureRouteConstraint:
        return "wire_review" if constraint.target != "discard" else _subject_stage(constraint.subject)
    raise InvariantError("DeferredIntentAction constraint is malformed")


def _plugin_identities(action: DeferredIntentAction) -> tuple[tuple[PluginKind, str], ...]:
    identities: list[tuple[PluginKind, str]] = []
    if action.catalog_kind is not None and action.catalog_name is not None:
        identities.append((action.catalog_kind, action.catalog_name))

    def add_subject(subject: StableSubject | PluginSubject) -> None:
        if type(subject) is PluginSubject:
            identities.append((subject.plugin_kind, subject.plugin_name))

    for constraint in action.constraints:
        if type(constraint) is SubjectPresenceConstraint or type(constraint) is OptionValueConstraint:
            add_subject(constraint.subject)
        elif type(constraint) is ComponentCountConstraint:
            if constraint.plugin_kind is not None and constraint.plugin_name is not None:
                identities.append((constraint.plugin_kind, constraint.plugin_name))
        elif type(constraint) is EdgeRouteConstraint:
            add_subject(constraint.from_subject)
            add_subject(constraint.to_subject)
        elif type(constraint) is FailureRouteConstraint:
            add_subject(constraint.subject)
            if constraint.target != "discard":
                add_subject(constraint.target)
    return tuple(dict.fromkeys(identities))


def _validate_catalog_identity(
    catalog: PolicyCatalogView,
    *,
    plugin_kind: PluginKind,
    plugin_name: str,
) -> DeferredIntentClarification | DeferredIntentUnsupported | DeferredIntentRejected | None:
    try:
        plugin_id = PluginId(plugin_kind, plugin_name)
    except ValueError:
        return DeferredIntentRejected(reason="malformed_catalog_identity")
    reason = catalog.unavailable_reason(plugin_id)
    if reason is not None and reason is not PluginUnavailableReason.NOT_INSTALLED:
        return DeferredIntentUnsupported(plugin_kind=plugin_kind, plugin_name=plugin_name, reason=reason)
    resolution = resolve_catalog_subject(catalog, plugin_name=plugin_name, expected_kind=plugin_kind)
    if type(resolution) is CatalogSubjectClarification:
        return DeferredIntentClarification(plugin_name=plugin_name, plugin_kinds=resolution.plugin_kinds)
    if type(resolution) is CatalogSubjectResolved:
        if resolution.plugin_kind != plugin_kind:
            return DeferredIntentRejected(reason="catalog_kind_mismatch")
        return None
    if type(resolution) is CatalogSubjectUnsupported and resolution.visible_kinds:
        return DeferredIntentRejected(reason="catalog_kind_mismatch")
    if reason is None:
        raise InvariantError("policy catalog returned unsupported for an available plugin identity")
    return DeferredIntentUnsupported(plugin_kind=plugin_kind, plugin_name=plugin_name, reason=reason)


def _stable_option_plugin_identity(subject: StableSubject, guided: GuidedSession) -> tuple[PluginKind, str] | None:
    if subject.component_kind == "source":
        reviewed_source = guided.reviewed_sources.get(subject.stable_id)
        return ("source", reviewed_source.plugin) if reviewed_source is not None else None
    if subject.component_kind == "output":
        reviewed_output = guided.reviewed_outputs.get(subject.stable_id)
        return ("sink", reviewed_output.plugin) if reviewed_output is not None else None
    return None


type _SchemaNode = dict[str, object] | bool
type _FiniteScalarDomain = tuple[object, ...]


class _ResolvedLookup(Protocol):
    @property
    def contents(self) -> object: ...

    @property
    def resolver(self) -> _SchemaResolver: ...


class _SchemaResolver(Protocol):
    def lookup(self, reference: str) -> _ResolvedLookup: ...

    def in_subresource(self, subresource: Resource[_SchemaNode]) -> _SchemaResolver: ...


@dataclass(frozen=True, slots=True)
class _ResolvedSchemaNode:
    schema: _SchemaNode
    resolver: _SchemaResolver


_SCHEMA_ANNOTATION_KEYS = frozenset(
    {
        "$comment",
        "default",
        "deprecated",
        "description",
        "examples",
        "readOnly",
        "title",
        "writeOnly",
    }
)
_MISSING_SCHEMA_KEY = object()


def _schema_resource(schema: _SchemaNode) -> Resource[_SchemaNode]:
    return Resource.from_contents(schema, default_specification=DRAFT202012)


def _root_schema_context(root: dict[str, object]) -> _ResolvedSchemaNode:
    resource = _schema_resource(root)
    return _ResolvedSchemaNode(schema=root, resolver=Registry[_SchemaNode]().resolver_with_root(resource))


def _resolve_schema_ref(context: _ResolvedSchemaNode, reference: object) -> _ResolvedSchemaNode:
    if type(reference) is not str:
        raise InvariantError("plugin option schema has a malformed $ref")
    try:
        resolved = context.resolver.lookup(reference)
    except (Unresolvable, ValueError) as exc:
        raise InvariantError("plugin option schema has a dangling or unsupported $ref") from exc
    target = resolved.contents
    if type(target) not in {dict, bool}:
        raise InvariantError("plugin option schema $ref does not target a schema")
    return _ResolvedSchemaNode(schema=cast(_SchemaNode, target), resolver=resolved.resolver)


def _preflight_schema_refs(root: _ResolvedSchemaNode) -> None:
    """Resolve every supported reference without consulting a proposed value.

    Draft 2020-12 has two reference applicators.  This retained-literal
    authority boundary supports ``$ref`` within the root resource registry and
    its discovered subresources.  It rejects ``$dynamicRef`` because proving
    dynamic-scope behavior is outside this boundary's deliberately closed
    authority model.
    """

    completed: set[int] = set()

    def walk(context: _ResolvedSchemaNode, *, active_nodes: frozenset[int]) -> None:
        node = context.schema
        if type(node) is bool:
            return
        node_identity = id(node)
        if node_identity in active_nodes:
            raise InvariantError("plugin option schema has a cyclic $ref")
        if node_identity in completed:
            return
        if "$dynamicRef" in node:
            raise InvariantError("plugin option schema uses unsupported $dynamicRef authority")
        descendants = active_nodes | {node_identity}
        if "$ref" in node:
            walk(_resolve_schema_ref(context, node["$ref"]), active_nodes=descendants)
        for subresource in _schema_resource(node).subresources():
            walk(
                _ResolvedSchemaNode(
                    schema=subresource.contents,
                    resolver=context.resolver.in_subresource(subresource),
                ),
                active_nodes=descendants,
            )
        completed.add(node_identity)

    walk(root, active_nodes=frozenset())


def _dereference_path_schema(context: _ResolvedSchemaNode) -> _ResolvedSchemaNode | None:
    current = context
    seen: set[int] = set()
    while type(current.schema) is dict and "$ref" in current.schema:
        semantic_siblings = set(current.schema) - _SCHEMA_ANNOTATION_KEYS - {"$ref"}
        if semantic_siblings:
            return None
        current = _resolve_schema_ref(current, current.schema["$ref"])
        identity = id(current.schema)
        if identity in seen:
            raise InvariantError("plugin option schema has a cyclic $ref")
        seen.add(identity)
    return current


def _subschema_context(context: _ResolvedSchemaNode, schema: object) -> _ResolvedSchemaNode:
    if type(schema) not in {dict, bool}:
        raise InvariantError("plugin option schema child does not contain a schema")
    subresource = _schema_resource(cast(_SchemaNode, schema))
    return _ResolvedSchemaNode(schema=subresource.contents, resolver=context.resolver.in_subresource(subresource))


def _option_schema_node(root: _ResolvedSchemaNode, option_path: tuple[str, ...]) -> _ResolvedSchemaNode | None:
    current = root
    for segment in option_path:
        resolved = _dereference_path_schema(current)
        if resolved is None:
            return None
        if type(resolved.schema) is bool:
            return None
        properties = resolved.schema.get("properties", _MISSING_SCHEMA_KEY)
        if properties is _MISSING_SCHEMA_KEY:
            return None
        if type(properties) is not dict:
            raise InvariantError("plugin option schema properties declaration is malformed")
        if segment not in properties:
            return None
        current = _subschema_context(resolved, properties[segment])
    return current


def _exact_json_scalar(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _is_exact_json_scalar(value: object) -> bool:
    if type(value) not in {str, int, float, bool, type(None)}:
        return False
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise InvariantError("plugin option schema contains a non-canonical JSON scalar") from exc
    return True


def _append_exact_scalar(domain: list[object], value: object) -> None:
    if any(_exact_json_scalar(existing, value) for existing in domain):
        return
    domain.append(value)


def _union_finite_domains(domains: tuple[_FiniteScalarDomain, ...]) -> _FiniteScalarDomain:
    union: list[object] = []
    for domain in domains:
        for value in domain:
            _append_exact_scalar(union, value)
    return tuple(union)


def _intersect_finite_domains(domains: tuple[_FiniteScalarDomain, ...]) -> _FiniteScalarDomain:
    return tuple(
        value for value in domains[0] if all(any(_exact_json_scalar(value, candidate) for candidate in domain) for domain in domains[1:])
    )


def _finite_type_domain(declared: object) -> _FiniteScalarDomain | None:
    if type(declared) is str:
        names = (declared,)
    elif type(declared) is list:
        names = tuple(declared)
    else:
        return None
    if any(name not in {"boolean", "null"} for name in names):
        return None
    domain: list[object] = []
    if "boolean" in names:
        domain.extend((False, True))
    if "null" in names:
        domain.append(None)
    return tuple(domain)


def _finite_scalar_domain(
    context: _ResolvedSchemaNode,
    *,
    active_refs: frozenset[int] = frozenset(),
) -> _FiniteScalarDomain | None:
    node = context.schema
    if type(node) is bool:
        return () if node is False else None
    candidates: list[_FiniteScalarDomain] = []

    if "$ref" in node:
        referenced = _resolve_schema_ref(context, node["$ref"])
        referenced_identity = id(referenced.schema)
        if referenced_identity in active_refs:
            raise InvariantError("plugin option schema has a cyclic $ref")
        referenced_domain = _finite_scalar_domain(referenced, active_refs=active_refs | {referenced_identity})
        if referenced_domain is not None:
            candidates.append(referenced_domain)

    for union_keyword in ("anyOf", "oneOf"):
        if union_keyword not in node:
            continue
        branches = node[union_keyword]
        if type(branches) is not list:  # pragma: no cover - Draft 2020-12 meta-validation owns this guard
            raise InvariantError(f"plugin option schema {union_keyword} declaration is malformed")
        branch_domains: list[_FiniteScalarDomain] = []
        for branch in branches:
            if type(branch) not in {dict, bool}:  # pragma: no cover - Draft 2020-12 meta-validation owns this guard
                raise InvariantError(f"plugin option schema {union_keyword} branch is malformed")
            branch_context = _subschema_context(context, branch)
            branch_domain = _finite_scalar_domain(branch_context, active_refs=active_refs)
            if branch_domain is None:
                return None
            branch_domains.append(branch_domain)
        candidates.append(_union_finite_domains(tuple(branch_domains)))

    if "allOf" in node:
        branches = node["allOf"]
        if type(branches) is not list:  # pragma: no cover - Draft 2020-12 meta-validation owns this guard
            raise InvariantError("plugin option schema allOf declaration is malformed")
        finite_branches: list[_FiniteScalarDomain] = []
        for branch in branches:
            if type(branch) not in {dict, bool}:  # pragma: no cover - Draft 2020-12 meta-validation owns this guard
                raise InvariantError("plugin option schema allOf branch is malformed")
            branch_context = _subschema_context(context, branch)
            branch_domain = _finite_scalar_domain(branch_context, active_refs=active_refs)
            if branch_domain is not None:
                finite_branches.append(branch_domain)
        if finite_branches:
            candidates.append(_intersect_finite_domains(tuple(finite_branches)))

    if "const" in node:
        constant = node["const"]
        if not _is_exact_json_scalar(constant):
            return None
        candidates.append((constant,))

    if "enum" in node:
        enum = node["enum"]
        if type(enum) is not list or not enum:
            raise InvariantError("plugin option schema enum declaration is malformed")
        enum_domain: list[object] = []
        for item in enum:
            if not _is_exact_json_scalar(item):
                return None
            _append_exact_scalar(enum_domain, item)
        candidates.append(tuple(enum_domain))

    type_domain = _finite_type_domain(node.get("type"))
    if type_domain is not None:
        candidates.append(type_domain)

    if not candidates:
        return None
    return _intersect_finite_domains(tuple(candidates))


def _schema_proves_closed_literal(
    schema: _ResolvedSchemaNode,
    value: object,
    *,
    validator: Draft202012Validator,
) -> bool:
    domain = _finite_scalar_domain(schema)
    if domain is None or not any(_exact_json_scalar(item, value) for item in domain):
        return False
    try:
        return next(validator.descend(value, schema.schema, resolver=schema.resolver), None) is None
    except (RecursionError, Unresolvable) as exc:
        raise InvariantError("plugin option schema could not resolve during Draft 2020-12 validation") from exc


def _validate_option_value_constraint(
    constraint: OptionValueConstraint,
    *,
    guided: GuidedSession,
    catalog: PolicyCatalogView,
) -> DeferredIntentUnsupported | DeferredIntentRejected | None:
    subject = constraint.subject
    if type(subject) is PluginSubject:
        identity: tuple[PluginKind, str] | None = (subject.plugin_kind, subject.plugin_name)
    elif type(subject) is StableSubject:
        identity = _stable_option_plugin_identity(subject, guided)
    else:  # pragma: no cover - OptionValueConstraint owns the closed subject type
        identity = None
    if identity is None:
        return DeferredIntentRejected(reason="option_value_unproven")
    plugin_kind, plugin_name = identity
    reason = catalog.unavailable_reason(PluginId(plugin_kind, plugin_name))
    if reason is not None:
        return DeferredIntentUnsupported(plugin_kind=plugin_kind, plugin_name=plugin_name, reason=reason)
    schema = catalog.get_schema(plugin_kind, plugin_name)
    if schema.plugin_type != plugin_kind or schema.name != plugin_name:
        raise InvariantError("policy catalog returned a mismatched plugin option schema identity")
    if type(schema.json_schema) is not dict:
        raise InvariantError("policy catalog returned a malformed plugin option schema root")
    root = cast(dict[str, object], schema.json_schema)
    try:
        Draft202012Validator.check_schema(root)
    except SchemaError as exc:
        raise InvariantError("policy catalog returned an invalid Draft 2020-12 plugin option schema") from exc
    root_context = _root_schema_context(root)
    _preflight_schema_refs(root_context)
    validator = Draft202012Validator(root)
    option_schema = _option_schema_node(root_context, constraint.option_path)
    if option_schema is None or not _schema_proves_closed_literal(
        option_schema,
        constraint.value,
        validator=validator,
    ):
        return DeferredIntentRejected(reason="option_value_unproven")
    return None


def validate_deferred_intent_action(
    action: DeferredIntentAction,
    *,
    receiving_stage: StageName,
    catalog: PolicyCatalogView,
    guided: GuidedSession,
) -> DeferredIntentValidation:
    """Validate a typed suggestion against live stage and policy authority."""

    if type(action) is not DeferredIntentAction:
        raise TypeError("action must be an exact DeferredIntentAction")
    if receiving_stage not in _STAGE_ORDINAL:
        raise InvariantError("receiving_stage is unsupported")
    if type(guided) is not GuidedSession:
        raise TypeError("guided must be an exact GuidedSession")
    if _STAGE_ORDINAL[action.target_stage] <= _STAGE_ORDINAL[receiving_stage]:
        return DeferredIntentRejected(reason="target_not_later")

    responsible_stages = [
        *([_PLUGIN_STAGE[action.catalog_kind]] if action.catalog_kind is not None else []),
        *(_constraint_stage(constraint) for constraint in action.constraints),
    ]
    if responsible_stages:
        responsible_stage = max(responsible_stages, key=_STAGE_ORDINAL.__getitem__)
        if action.target_stage != responsible_stage:
            return DeferredIntentRejected(reason="wrong_responsible_stage")

    for plugin_kind, plugin_name in _plugin_identities(action):
        invalid = _validate_catalog_identity(catalog, plugin_kind=plugin_kind, plugin_name=plugin_name)
        if invalid is not None:
            return invalid
    for constraint in action.constraints:
        if type(constraint) is OptionValueConstraint:
            invalid_option = _validate_option_value_constraint(constraint, guided=guided, catalog=catalog)
            if invalid_option is not None:
                return invalid_option
    return DeferredIntentAccepted(action=action)


type _ComponentKind = Literal["source", "node", "edge", "output"]


@dataclass(frozen=True, slots=True)
class _CandidateComponent:
    kind: _ComponentKind
    stable_id: str
    name: str
    plugin_kind: PluginKind | None = None
    plugin: str | None = None
    options: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _SubjectResolution:
    components: tuple[_CandidateComponent, ...]
    ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class _DeferredCoverageContext:
    candidate: CompositionState
    components: tuple[_CandidateComponent, ...]
    exact_components: Mapping[tuple[_ComponentKind, str], _CandidateComponent]
    consumers: Mapping[str, tuple[ConsumerIdentity, ...]]

    def resolve(self, subject: StableSubject | PluginSubject) -> _SubjectResolution:
        if type(subject) is StableSubject:
            exact = self.exact_components.get((subject.component_kind, subject.stable_id))
            components = (exact,) if exact is not None else ()
            return _SubjectResolution(components=components)
        plugin_subject = cast(PluginSubject, subject)
        component_kind = cast(
            Literal["source", "node", "output"],
            {"source": "source", "transform": "node", "sink": "output"}[plugin_subject.plugin_kind],
        )
        exact = self.exact_components.get((component_kind, plugin_subject.subject_id))
        matches = tuple(
            component
            for component in self.components
            if component.plugin_kind == plugin_subject.plugin_kind and component.plugin == plugin_subject.plugin_name
        )
        if exact is not None and exact.plugin_kind == plugin_subject.plugin_kind and exact.plugin == plugin_subject.plugin_name:
            return _SubjectResolution(components=(exact,))
        if len(matches) > 1:
            return _SubjectResolution(components=(), ambiguous=True)
        return _SubjectResolution(components=matches)

    @staticmethod
    def option_value(component: _CandidateComponent, path: tuple[str, ...]) -> tuple[bool, Any]:
        value: Any = component.options
        for segment in path:
            if not isinstance(value, Mapping) or segment not in value:
                return False, None
            value = value[segment]
        return True, value

    def route_targets(self, component: _CandidateComponent, edge_type: str) -> set[ConsumerIdentity]:
        connections: set[str]
        if component.kind == "source":
            source = self.candidate.sources[component.name]
            connections = {source.on_success} if edge_type == "on_success" and source.on_success is not None else set()
        elif component.kind != "node":
            connections = set()
        else:
            node = next(item for item in self.candidate.nodes if item.id == component.name)
            if edge_type == "on_success":
                connections = {node.on_success} if node.on_success is not None else set()
            elif edge_type == "on_error":
                connections = {node.on_error} if node.on_error is not None else set()
            elif edge_type in {"route_true", "route_false"}:
                key = "true" if edge_type == "route_true" else "false"
                value = dict(node.routes or {}).get(key)
                connections = {value} if value is not None and value != "fork" else set()
            else:
                connections = set(node.fork_to or ()) if edge_type == "fork" else set()
        return {destination for connection in connections for destination in self.consumers.get(connection, ())}

    def failure_target(self, component: _CandidateComponent, failure_kind: str) -> str | None:
        if failure_kind == "source_validation":
            return self.candidate.sources[component.name].on_validation_failure
        if failure_kind == "node_error":
            return next(item for item in self.candidate.nodes if item.id == component.name).on_error
        return next(item for item in self.candidate.outputs if item.name == component.name).on_write_failure

    def constraint_holds(self, constraint: DeferredConstraint) -> bool:
        if type(constraint) is SubjectPresenceConstraint:
            resolved = self.resolve(constraint.subject)
            return not resolved.ambiguous and bool(resolved.components) is constraint.present
        if type(constraint) is OptionValueConstraint:
            resolved = self.resolve(constraint.subject)
            if resolved.ambiguous or len(resolved.components) != 1:
                return False
            present, value = self.option_value(resolved.components[0], constraint.option_path)
            if not present:
                return False
            equals = _exact_json_scalar(value, constraint.value)
            return equals if constraint.operator == "equals" else not equals
        if type(constraint) is ComponentCountConstraint:
            count = sum(
                component.kind == constraint.component_kind
                and (
                    constraint.plugin_name is None
                    or (component.plugin_kind == constraint.plugin_kind and component.plugin == constraint.plugin_name)
                )
                for component in self.components
            )
            return {
                "equals": count == constraint.count,
                "at_least": count >= constraint.count,
                "at_most": count <= constraint.count,
            }[constraint.operator]
        if type(constraint) is EdgeRouteConstraint:
            origins = self.resolve(constraint.from_subject)
            destinations = self.resolve(constraint.to_subject)
            if origins.ambiguous or destinations.ambiguous or not origins.components or not destinations.components:
                return False
            targets = {(component.kind, component.stable_id) for component in destinations.components}
            present = any(self.route_targets(origin, constraint.edge_type) & targets for origin in origins.components)
            return present is constraint.present
        if type(constraint) is FailureRouteConstraint:
            subjects = self.resolve(constraint.subject)
            if subjects.ambiguous or len(subjects.components) != 1:
                return False
            if constraint.target == "discard":
                expected_targets = {"discard"}
            else:
                target_resolution = self.resolve(constraint.target)
                if target_resolution.ambiguous or len(target_resolution.components) != 1:
                    return False
                expected_targets = {component.name for component in target_resolution.components}
            actual = {self.failure_target(subject, constraint.failure_kind) for subject in subjects.components}
            equals = actual == expected_targets
            return equals if constraint.operator == "equals" else not equals
        raise InvariantError("deferred intent contains an unsupported constraint")


def _coverage_context(candidate: CompositionState, reviewed_guided: GuidedSession) -> _DeferredCoverageContext:
    source_ids = {source.name: stable_id for stable_id, source in reviewed_guided.reviewed_sources.items()}
    output_ids = {output.name: stable_id for stable_id, output in reviewed_guided.reviewed_outputs.items()}
    components: list[_CandidateComponent] = []
    for name, source in candidate.sources.items():
        components.append(
            _CandidateComponent(
                kind="source",
                stable_id=source_ids.get(name, name),
                name=name,
                plugin_kind="source",
                plugin=source.plugin,
                options=cast(Mapping[str, Any], deep_thaw(source.options)),
            )
        )
    for node in candidate.nodes:
        components.append(
            _CandidateComponent(
                kind="node",
                stable_id=node.id,
                name=node.id,
                plugin_kind="transform" if node.plugin is not None else None,
                plugin=node.plugin,
                options=cast(Mapping[str, Any], deep_thaw(node.options)),
            )
        )
    for edge in candidate.edges:
        components.append(_CandidateComponent(kind="edge", stable_id=edge.id, name=edge.id))
    for output in candidate.outputs:
        components.append(
            _CandidateComponent(
                kind="output",
                stable_id=output_ids.get(output.name, output.name),
                name=output.name,
                plugin_kind="sink",
                plugin=output.plugin,
                options=cast(Mapping[str, Any], deep_thaw(output.options)),
            )
        )
    component_tuple = tuple(components)
    exact_components = {(component.kind, component.stable_id): component for component in component_tuple}
    if len(exact_components) != len(component_tuple):
        raise InvariantError("guided candidate contains duplicate same-kind stable component identities")
    try:
        consumers = canonical_connection_consumers(
            candidate,
            node_identities={node.id: node.id for node in candidate.nodes},
            output_identities={output.name: output_ids.get(output.name, output.name) for output in candidate.outputs},
        )
    except ValueError as exc:
        raise InvariantError("guided candidate canonical consumer identities are malformed") from exc
    return _DeferredCoverageContext(
        candidate=candidate,
        components=component_tuple,
        exact_components=exact_components,
        consumers=consumers,
    )


def constraint_holds(candidate: CompositionState, reviewed_guided: GuidedSession, constraint: DeferredConstraint) -> bool:
    """Return whether one persisted structural predicate is true of a candidate."""

    if type(candidate) is not CompositionState or type(reviewed_guided) is not GuidedSession:
        raise TypeError("constraint_holds requires exact candidate and reviewed guided authority")
    return _coverage_context(candidate, reviewed_guided).constraint_holds(constraint)


def evaluate_deferred_intent_coverage(
    *,
    candidate: CompositionState,
    reviewed_guided: GuidedSession,
    claimed_intent_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Prove model claims and return only the verified reviewed-order subset."""

    if type(candidate) is not CompositionState or type(reviewed_guided) is not GuidedSession:
        raise TypeError("deferred coverage requires exact candidate and reviewed guided authority")
    if type(claimed_intent_ids) is not tuple or any(type(intent_id) is not str for intent_id in claimed_intent_ids):
        raise DeferredIntentClaimError("guided proposal claims must be an exact string tuple")
    if len(set(claimed_intent_ids)) != len(claimed_intent_ids):
        raise DeferredIntentClaimError("guided proposal contained a duplicate deferred intent claim")

    claimed = set(claimed_intent_ids)
    context = _coverage_context(candidate, reviewed_guided)
    verified: list[str] = []
    for intent in reviewed_guided.deferred_intents:
        if intent.intent_id not in claimed:
            continue
        if not intent.constraints or not all(context.constraint_holds(constraint) for constraint in intent.constraints):
            raise DeferredIntentClaimError("guided proposal claimed an unproven deferred intent")
        verified.append(intent.intent_id)
    if claimed != set(verified):
        raise DeferredIntentClaimError("guided proposal claimed an unknown deferred intent")
    return tuple(verified)


def create_deferred_stage_intent(
    action: DeferredIntentAction,
    *,
    receiving_stage: StageName,
    intent_id: str,
    originating_message_id: str,
    originating_message_content: str,
) -> DeferredStageIntent:
    """Create durable state from a server-validated action and private row.

    The model's ``redacted_summary`` is a classification hint, not durable
    text authority.  The stored summary is rendered from closed stage/catalog
    facts so even a model that echoes the user cannot copy raw prose out of the
    private message row.
    """

    if type(action) is not DeferredIntentAction:
        raise TypeError("action must be an exact DeferredIntentAction")
    _require_nonempty_exact_str(originating_message_content, "originating_message_content")
    subject = (
        f"{action.catalog_kind} plugin {action.catalog_name!r}"
        if action.catalog_kind is not None and action.catalog_name is not None
        else "structural requirement"
    )
    durable_summary = (
        f"Future {action.target_stage.replace('_', ' ')} instruction for {subject}; {len(action.constraints)} structural constraint(s)."
    )
    return DeferredStageIntent.create(
        intent_id=intent_id,
        receiving_stage=receiving_stage,
        target_stage=action.target_stage,
        catalog_kind=action.catalog_kind,
        catalog_name=action.catalog_name,
        redacted_summary=durable_summary,
        originating_message_id=originating_message_id,
        message_content_hash=stable_hash(originating_message_content),
        constraints=action.constraints,
    )
