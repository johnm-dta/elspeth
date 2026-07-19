"""Creation and validation boundary for guided wrong-stage intent.

The model may suggest only :class:`DeferredIntentAction`.  The server then
validates that suggestion against the request-scoped policy catalog and the
current guided stage before creating durable audit-tier state.  Raw user text
is deliberately accepted only by :func:`create_deferred_stage_intent`, where
it is reduced to a content hash; it is never stored in deferred metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.schemas import PluginKind
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
