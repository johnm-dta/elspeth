"""Closed creation boundary for guided wrong-stage intent."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from elspeth.core.canonical import stable_hash
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.schemas import PluginKind, PluginSchemaInfo, PluginSummary
from elspeth.web.composer.guided.deferred_intents import (
    DeferredIntentAccepted,
    DeferredIntentAction,
    DeferredIntentRejected,
    DeferredIntentUnsupported,
    create_deferred_stage_intent,
    deferred_intent_action_from_dict,
    validate_deferred_intent_action,
)
from elspeth.web.composer.guided.errors import GuidedSolverResponseShapeError, InvariantError
from elspeth.web.composer.guided.resolved import SinkOutputResolved
from elspeth.web.composer.guided.stage_subjects import (
    ComponentCountConstraint,
    OptionValueConstraint,
    PluginSubject,
    StableSubject,
)
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.models import PluginAvailability, PluginAvailabilitySnapshot, PluginId, PluginUnavailableReason

INTENT_ID = "11111111-1111-4111-8111-111111111111"
MESSAGE_ID = "22222222-2222-4222-8222-222222222222"


class _Catalog:
    def __init__(
        self,
        plugins: tuple[tuple[PluginKind, str], ...],
        schemas: dict[tuple[PluginKind, str], dict[str, object]] | None = None,
        schema_overrides: dict[tuple[PluginKind, str], PluginSchemaInfo] | None = None,
    ) -> None:
        self._plugins = plugins
        self._schemas = schemas or {}
        self._schema_overrides = schema_overrides or {}

    def _list(self, kind: PluginKind) -> list[PluginSummary]:
        return [
            PluginSummary(name=name, description=name, plugin_type=kind, config_fields=[])
            for plugin_kind, name in self._plugins
            if plugin_kind == kind
        ]

    def list_sources(self) -> list[PluginSummary]:
        return self._list("source")

    def list_transforms(self) -> list[PluginSummary]:
        return self._list("transform")

    def list_sinks(self) -> list[PluginSummary]:
        return self._list("sink")

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        overridden = self._schema_overrides.get((plugin_type, name))
        if overridden is not None:
            return overridden
        json_schema = self._schemas.get((plugin_type, name))
        if json_schema is None:
            raise AssertionError("deferred intent validation must inspect schemas only for option-value constraints")
        return PluginSchemaInfo(
            name=name,
            plugin_type=plugin_type,
            description=name,
            json_schema=json_schema,
            knob_schema={"fields": []},
        )

    def post_call_hints(
        self,
        *,
        plugin_type: PluginKind,
        plugin_name: str,
        tool_name: str,
        config_snapshot: dict[str, object],
    ) -> tuple[str, ...]:
        raise AssertionError("deferred intent validation must not dispatch plugins")


def _view(
    installed: tuple[tuple[PluginKind, str], ...],
    *,
    available: frozenset[PluginId] | None = None,
    schemas: dict[tuple[PluginKind, str], dict[str, object]] | None = None,
    schema_overrides: dict[tuple[PluginKind, str], PluginSchemaInfo] | None = None,
) -> PolicyCatalogView:
    permitted = frozenset(PluginId(kind, name) for kind, name in installed) if available is None else available
    unavailable = tuple(
        PluginAvailability(plugin_id=PluginId(kind, name), reason=PluginUnavailableReason.NOT_AUTHORIZED)
        for kind, name in installed
        if PluginId(kind, name) not in permitted
    )
    snapshot = PluginAvailabilitySnapshot.create(
        policy_hash="a" * 64,
        principal_scope="local:test",
        available=permitted,
        unavailable=unavailable,
        selected=(),
        usable_profile_aliases=(),
        selected_profile_aliases=(),
        binding_generation_fingerprint="b" * 64,
    )
    return PolicyCatalogView(_Catalog(installed, schemas, schema_overrides), snapshot, profiles=None)  # type: ignore[arg-type]


def _real_json_view() -> PolicyCatalogView:
    available = frozenset({PluginId("source", "json"), PluginId("sink", "json")})
    snapshot = PluginAvailabilitySnapshot.create(
        policy_hash="a" * 64,
        principal_scope="local:test",
        available=available,
        unavailable=(),
        selected=(),
        usable_profile_aliases=(),
        selected_profile_aliases=(),
        binding_generation_fingerprint="b" * 64,
    )
    return PolicyCatalogView(create_catalog_service(), snapshot, profiles=None)


def _real_text_view() -> PolicyCatalogView:
    available = frozenset({PluginId("source", "text"), PluginId("sink", "text")})
    snapshot = PluginAvailabilitySnapshot.create(
        policy_hash="a" * 64,
        principal_scope="local:test",
        available=available,
        unavailable=(),
        selected=(),
        usable_profile_aliases=(),
        selected_profile_aliases=(),
        binding_generation_fingerprint="b" * 64,
    )
    return PolicyCatalogView(create_catalog_service(), snapshot, profiles=None)


def _action(
    *,
    target_stage: str = "topology",
    catalog_kind: str | None = "transform",
    catalog_name: str | None = "llm",
) -> DeferredIntentAction:
    return DeferredIntentAction(
        target_stage=target_stage,  # type: ignore[arg-type]
        catalog_kind=catalog_kind,  # type: ignore[arg-type]
        catalog_name=catalog_name,
        redacted_summary="Use the named transform in the topology stage.",
        constraints=(
            ComponentCountConstraint(
                kind="component_count",
                component_kind="node",
                plugin_kind="transform",
                plugin_name="llm",
                operator="at_least",
                count=1,
            ),
        ),
    )


def _option_action(
    *,
    subject: PluginSubject | StableSubject,
    option_path: tuple[str, ...],
    value: object,
    target_stage: str = "topology",
    catalog_kind: str = "transform",
    catalog_name: str = "llm",
) -> DeferredIntentAction:
    return DeferredIntentAction(
        target_stage=target_stage,  # type: ignore[arg-type]
        catalog_kind=catalog_kind,  # type: ignore[arg-type]
        catalog_name=catalog_name,
        redacted_summary="Retain one closed catalog option value.",
        constraints=(
            OptionValueConstraint(
                kind="option_value",
                subject=subject,
                option_path=option_path,
                operator="equals",
                value=value,  # type: ignore[arg-type]
            ),
        ),
    )


_TRANSFORM_OPTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "mode": {"enum": ["safe", "strict"]},
        "fixed": {"const": "locked"},
        "nested": {"$ref": "#/$defs/NestedOptions"},
    },
    "$defs": {
        "NestedOptions": {
            "type": "object",
            "properties": {"flavor": {"enum": ["vanilla", "chocolate"]}},
        }
    },
}


def test_action_is_frozen_exact_and_decoder_rejects_open_or_coerced_shapes() -> None:
    action = _action()
    with pytest.raises(FrozenInstanceError):
        action.redacted_summary = "changed"  # type: ignore[misc]

    encoded = {
        "target_stage": "topology",
        "catalog_kind": "transform",
        "catalog_name": "llm",
        "redacted_summary": "Use the named transform in the topology stage.",
        "constraints": [constraint.to_dict() for constraint in action.constraints],
    }
    assert deferred_intent_action_from_dict(encoded) == action
    with pytest.raises(GuidedSolverResponseShapeError, match="unexpected keys"):
        deferred_intent_action_from_dict({**encoded, "raw_user_message": "secret"})
    with pytest.raises(GuidedSolverResponseShapeError, match="constraints must be a list"):
        deferred_intent_action_from_dict({**encoded, "constraints": tuple(encoded["constraints"])})
    with pytest.raises(InvariantError, match="catalog fields must be paired"):
        _action(catalog_kind=None, catalog_name="llm")
    with pytest.raises(InvariantError, match="at least one structural constraint"):
        DeferredIntentAction(
            target_stage="topology",
            catalog_kind="transform",
            catalog_name="llm",
            redacted_summary="Use the named transform in the topology stage.",
            constraints=(),
        )


@pytest.mark.parametrize(
    ("option_path", "value"),
    [(("mode",), "safe"), (("fixed",), "locked"), (("nested", "flavor"), "chocolate")],
)
def test_option_value_literal_is_accepted_only_when_public_schema_proves_closed_membership(
    option_path: tuple[str, ...],
    value: object,
) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=option_path,
        value=value,
    )
    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): _TRANSFORM_OPTION_SCHEMA}),
        guided=GuidedSession.initial(),
    )
    assert result == DeferredIntentAccepted(action=action)


@pytest.mark.parametrize(
    ("plugin_kind", "option_path", "value"),
    [
        ("source", ("format",), "json"),
        ("source", ("format",), None),
        ("sink", ("format",), "jsonl"),
        ("sink", ("format",), None),
        ("sink", ("collision_policy",), "auto_increment"),
        ("sink", ("collision_policy",), None),
    ],
)
def test_real_json_plugin_pydantic_nullable_enums_are_finite_closed_domains(
    plugin_kind: PluginKind,
    option_path: tuple[str, ...],
    value: object,
) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind=plugin_kind,
            plugin_name="json",
        ),
        option_path=option_path,
        value=value,
        target_stage="output",
        catalog_kind="sink",
        catalog_name="json",
    )

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_real_json_view(),
        guided=GuidedSession.initial(),
    )

    assert result == DeferredIntentAccepted(action=action)


@pytest.mark.parametrize(
    ("plugin_kind", "option_path", "value"),
    [
        ("source", ("format",), "csv"),
        ("sink", ("format",), "csv"),
        ("sink", ("collision_policy",), "overwrite"),
    ],
)
def test_real_json_plugin_pydantic_nullable_enums_reject_values_outside_the_domain(
    plugin_kind: PluginKind,
    option_path: tuple[str, ...],
    value: object,
) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind=plugin_kind,
            plugin_name="json",
        ),
        option_path=option_path,
        value=value,
        target_stage="output",
        catalog_kind="sink",
        catalog_name="json",
    )

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_real_json_view(),
        guided=GuidedSession.initial(),
    )

    assert result == DeferredIntentRejected(reason="option_value_unproven")


def test_real_text_source_pydantic_boolean_option_is_a_finite_closed_domain() -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="source",
            plugin_name="text",
        ),
        option_path=("strip_whitespace",),
        value=False,
        target_stage="output",
        catalog_kind="sink",
        catalog_name="text",
    )

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_real_text_view(),
        guided=GuidedSession.initial(),
    )

    assert result == DeferredIntentAccepted(action=action)


@pytest.mark.parametrize(
    ("declared_types", "value", "accepted"),
    [
        (["null"], None, True),
        (["boolean", "null"], False, True),
        (["boolean", "null"], True, True),
        (["boolean", "null"], None, True),
        (["boolean", "null"], 0, False),
        (["boolean", "string"], False, False),
    ],
)
def test_only_boolean_and_null_type_arrays_have_finite_exact_scalar_domains(
    declared_types: list[str],
    value: object,
    accepted: bool,
) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value=value,
    )
    schema = {"type": "object", "properties": {"mode": {"type": declared_types}}}

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
        guided=GuidedSession.initial(),
    )

    expected = DeferredIntentAccepted(action=action) if accepted else DeferredIntentRejected(reason="option_value_unproven")
    assert result == expected


def test_nested_local_ref_and_union_of_only_finite_branches_is_accepted() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"nested": {"$ref": "#/$defs/Nested"}},
        "$defs": {
            "Nested": {
                "type": "object",
                "properties": {
                    "mode": {
                        "oneOf": [
                            {"$ref": "#/$defs/Mode"},
                            {"type": "null"},
                        ]
                    }
                },
            },
            "Mode": {"enum": ["safe", "strict"], "type": "string"},
        },
    }
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("nested", "mode"),
        value=None,
    )

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
        guided=GuidedSession.initial(),
    )

    assert result == DeferredIntentAccepted(action=action)


@pytest.mark.parametrize(("schema_value", "constraint_value"), [(True, True), (1, 1)])
def test_finite_domain_uses_exact_type_and_value_identity(schema_value: object, constraint_value: object) -> None:
    schema = {"type": "object", "properties": {"mode": {"anyOf": [{"enum": [True]}, {"enum": [1]}]}}}
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value=constraint_value,
    )

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
        guided=GuidedSession.initial(),
    )

    assert result == DeferredIntentAccepted(action=action)
    assert type(schema_value) is type(constraint_value)


def test_finite_domain_does_not_coerce_bool_to_integer() -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value=True,
    )

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view(
            (("transform", "llm"),),
            schemas={("transform", "llm"): {"type": "object", "properties": {"mode": {"enum": [1]}}}},
        ),
        guided=GuidedSession.initial(),
    )

    assert result == DeferredIntentRejected(reason="option_value_unproven")


def test_union_with_one_free_form_branch_has_no_proven_finite_domain() -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value="safe",
    )
    schema = {
        "type": "object",
        "properties": {"mode": {"anyOf": [{"enum": ["safe"]}, {"type": "string"}]}},
    }

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
        guided=GuidedSession.initial(),
    )

    assert result == DeferredIntentRejected(reason="option_value_unproven")


@pytest.mark.parametrize(
    ("option_schema", "value", "accepted"),
    [
        ({"oneOf": [{"const": "x"}, {"enum": ["x", "y"]}]}, "x", False),
        ({"oneOf": [{"const": "x"}, {"enum": ["x", "y"]}]}, "y", True),
        ({"enum": ["x", "y"], "not": {"const": "x"}}, "x", False),
        ({"enum": ["x", "y"], "not": {"const": "x"}}, "y", True),
        ({"enum": [1, 2], "allOf": [{"minimum": 2}]}, 1, False),
        ({"enum": [1, 2], "allOf": [{"minimum": 2}]}, 2, True),
    ],
)
def test_finite_discovery_is_filtered_by_full_draft_2020_12_semantics(
    option_schema: dict[str, object],
    value: object,
    accepted: bool,
) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value=value,
    )
    schema = {"type": "object", "properties": {"mode": option_schema}}

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
        guided=GuidedSession.initial(),
    )

    expected = DeferredIntentAccepted(action=action) if accepted else DeferredIntentRejected(reason="option_value_unproven")
    assert result == expected


@pytest.mark.parametrize(("value", "accepted"), [("x", False), ("y", True)])
def test_all_of_can_supply_a_finite_candidate_domain_while_full_validation_applies_not(value: str, accepted: bool) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value=value,
    )
    schema = {
        "type": "object",
        "properties": {"mode": {"allOf": [{"enum": ["x", "y"]}, {"not": {"const": "x"}}]}},
    }

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
        guided=GuidedSession.initial(),
    )

    expected = DeferredIntentAccepted(action=action) if accepted else DeferredIntentRejected(reason="option_value_unproven")
    assert result == expected


@pytest.mark.parametrize(("value", "accepted"), [("safe", True), ("other", False)])
def test_finite_sibling_enum_bounds_an_otherwise_infinite_local_ref(value: str, accepted: bool) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value=value,
    )
    schema = {
        "$defs": {"String": {"type": "string"}},
        "type": "object",
        "properties": {"mode": {"$ref": "#/$defs/String", "enum": ["safe"]}},
    }

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
        guided=GuidedSession.initial(),
    )

    expected = DeferredIntentAccepted(action=action) if accepted else DeferredIntentRejected(reason="option_value_unproven")
    assert result == expected


def test_dangling_ref_in_restrictive_schema_is_integrity_failure_before_literal_membership() -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value="other",
    )
    schema = {
        "type": "object",
        "properties": {"mode": {"enum": ["safe"], "not": {"$ref": "#/$defs/Missing"}}},
    }

    with pytest.raises(InvariantError, match="schema"):
        validate_deferred_intent_action(
            action,
            receiving_stage="source",
            catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
            guided=GuidedSession.initial(),
        )


@pytest.mark.parametrize("value", ["safe", "other"])
def test_dynamic_ref_is_rejected_before_literal_membership(value: str) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8335-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value=value,
    )
    schema = {
        "type": "object",
        "properties": {"mode": {"enum": ["safe"], "not": {"$dynamicRef": "#missing"}}},
    }

    with pytest.raises(InvariantError, match="dynamicRef"):
        validate_deferred_intent_action(
            action,
            receiving_stage="source",
            catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
            guided=GuidedSession.initial(),
        )


@pytest.mark.parametrize(("value", "accepted"), [("inner-safe", True), ("root-safe", False)])
def test_nested_resource_ref_uses_its_own_base_for_domain_and_validation(value: str, accepted: bool) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8335-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("nested", "mode"),
        value=value,
    )
    schema = {
        "$defs": {"Mode": {"enum": ["root-safe"]}},
        "type": "object",
        "properties": {
            "nested": {
                "$id": "outer",
                "$defs": {"Mode": {"enum": ["inner-safe"]}},
                "type": "object",
                "properties": {"mode": {"$ref": "#/$defs/Mode"}},
            }
        },
    }

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
        guided=GuidedSession.initial(),
    )

    expected = DeferredIntentAccepted(action=action) if accepted else DeferredIntentRejected(reason="option_value_unproven")
    assert result == expected


def test_option_path_ref_keeps_nested_resource_base() -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8335-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("nested", "config", "mode"),
        value="inner-safe",
    )
    schema = {
        "$defs": {
            "Nested": {"type": "object", "properties": {"mode": {"enum": ["root-safe"]}}},
        },
        "type": "object",
        "properties": {
            "nested": {
                "$id": "outer",
                "$defs": {
                    "Nested": {"type": "object", "properties": {"mode": {"enum": ["inner-safe"]}}},
                },
                "type": "object",
                "properties": {"config": {"$ref": "#/$defs/Nested"}},
            }
        },
    }

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
        guided=GuidedSession.initial(),
    )

    assert result == DeferredIntentAccepted(action=action)


def test_dangling_ref_in_nested_resource_is_not_masked_by_root_definition() -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8335-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("nested", "mode"),
        value="root-safe",
    )
    schema = {
        "$defs": {"Mode": {"enum": ["root-safe"]}},
        "type": "object",
        "properties": {
            "nested": {
                "$id": "outer",
                "type": "object",
                "properties": {"mode": {"$ref": "#/$defs/Mode"}},
            }
        },
    }

    with pytest.raises(InvariantError, match="schema"):
        validate_deferred_intent_action(
            action,
            receiving_stage="source",
            catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
            guided=GuidedSession.initial(),
        )


@pytest.mark.parametrize(
    "option_schema",
    [
        {"$ref": "#/$defs/Missing"},
        {"$ref": 7},
        {"$ref": "external.json#/$defs/Mode"},
        {"anyOf": "not-a-branch-list"},
        {"oneOf": [{"enum": ["safe"]}, "not-a-schema"]},
        {"enum": "not-an-enum-list"},
        {"enum": []},
        {"enum": ["safe"], "not": None},
        None,
    ],
)
def test_malformed_ref_union_or_closed_domain_declaration_is_authority_corruption(option_schema: object) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value="safe",
    )
    schema = {"type": "object", "properties": {"mode": option_schema}}

    with pytest.raises(InvariantError, match="schema"):
        validate_deferred_intent_action(
            action,
            receiving_stage="source",
            catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
            guided=GuidedSession.initial(),
        )


@pytest.mark.parametrize("option_schema", [True, False])
def test_boolean_property_schemas_are_valid_but_do_not_authorize_a_retained_literal(option_schema: bool) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value="safe",
    )
    schema = {"type": "object", "properties": {"mode": option_schema}}

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
        guided=GuidedSession.initial(),
    )

    assert result == DeferredIntentRejected(reason="option_value_unproven")


@pytest.mark.parametrize("properties", [None, True, []])
def test_present_malformed_properties_declaration_is_authority_corruption(properties: object) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value="safe",
    )
    schema = {"type": "object", "properties": properties}

    with pytest.raises(InvariantError, match="schema"):
        validate_deferred_intent_action(
            action,
            receiving_stage="source",
            catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
            guided=GuidedSession.initial(),
        )


def test_cyclic_local_ref_is_authority_corruption() -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value="safe",
    )
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"mode": {"$ref": "#/$defs/A"}},
        "$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}},
    }

    with pytest.raises(InvariantError, match="schema"):
        validate_deferred_intent_action(
            action,
            receiving_stage="source",
            catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): schema}),
            guided=GuidedSession.initial(),
        )


@pytest.mark.parametrize(
    "schema_info",
    [
        PluginSchemaInfo(
            name="other",
            plugin_type="transform",
            description="wrong name",
            json_schema={"type": "object", "properties": {"mode": {"enum": ["safe"]}}},
            knob_schema={"fields": []},
        ),
        PluginSchemaInfo(
            name="llm",
            plugin_type="sink",
            description="wrong kind",
            json_schema={"type": "object", "properties": {"mode": {"enum": ["safe"]}}},
            knob_schema={"fields": []},
        ),
    ],
)
def test_catalog_schema_identity_mismatch_is_authority_corruption(schema_info: PluginSchemaInfo) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value="safe",
    )

    with pytest.raises(InvariantError, match="schema"):
        validate_deferred_intent_action(
            action,
            receiving_stage="source",
            catalog=_view(
                (("transform", "llm"),),
                schema_overrides={("transform", "llm"): schema_info},
            ),
            guided=GuidedSession.initial(),
        )


@pytest.mark.parametrize("root", [None, True, []])
def test_catalog_schema_root_must_be_an_object_schema(root: object) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value="safe",
    )
    schema_info = PluginSchemaInfo.model_construct(
        name="llm",
        plugin_type="transform",
        description="malformed root",
        json_schema=root,
        knob_schema={"fields": []},
    )

    with pytest.raises(InvariantError, match="schema root"):
        validate_deferred_intent_action(
            action,
            receiving_stage="source",
            catalog=_view(
                (("transform", "llm"),),
                schema_overrides={("transform", "llm"): schema_info},
            ),
            guided=GuidedSession.initial(),
        )


@pytest.mark.parametrize(
    ("option_path", "value"),
    [
        (("mode",), "send the full secret sentence to an arbitrary destination"),
        (("mode",), "unknown-mode"),
        (("missing",), "safe"),
        (("nested", "missing"), "vanilla"),
    ],
)
def test_option_value_free_form_wrong_enum_and_unresolved_paths_are_rejected(
    option_path: tuple[str, ...],
    value: object,
) -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="transform",
            plugin_name="llm",
        ),
        option_path=option_path,
        value=value,
    )
    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): _TRANSFORM_OPTION_SCHEMA}),
        guided=GuidedSession.initial(),
    )
    assert result == DeferredIntentRejected(reason="option_value_unproven")


def test_option_value_wrong_plugin_kind_is_rejected_before_schema_lookup() -> None:
    action = _option_action(
        subject=PluginSubject(
            kind="plugin",
            subject_id="33333333-3333-4333-8333-333333333333",
            plugin_kind="sink",
            plugin_name="llm",
        ),
        option_path=("mode",),
        value="safe",
    )
    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view((("transform", "llm"),), schemas={("transform", "llm"): _TRANSFORM_OPTION_SCHEMA}),
        guided=GuidedSession.initial(),
    )
    assert result == DeferredIntentRejected(reason="catalog_kind_mismatch")


def test_stable_option_subject_requires_reviewed_guided_plugin_authority_then_uses_same_schema_rule() -> None:
    stable_id = "44444444-4444-4444-8444-444444444444"
    reviewed = replace(
        GuidedSession.initial(),
        output_order=(stable_id,),
        reviewed_outputs={
            stable_id: SinkOutputResolved(
                name="main",
                plugin="json",
                options={},
                required_fields=(),
                schema_mode="observed",
                on_write_failure="discard",
            )
        },
    )
    action = _option_action(
        subject=StableSubject(kind="stable", component_kind="output", stable_id=stable_id),
        option_path=("mode",),
        value="safe",
        target_stage="output",
        catalog_kind="sink",
        catalog_name="json",
    )
    catalog = _view(
        (("sink", "json"),),
        schemas={("sink", "json"): {"type": "object", "properties": {"mode": {"enum": ["safe"]}}}},
    )

    accepted = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=catalog,
        guided=reviewed,
    )
    unresolved = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=catalog,
        guided=GuidedSession.initial(),
    )
    assert accepted == DeferredIntentAccepted(action=action)
    assert unresolved == DeferredIntentRejected(reason="option_value_unproven")


def test_live_unique_catalog_identity_is_accepted_only_at_its_responsible_later_stage() -> None:
    result = validate_deferred_intent_action(
        _action(),
        receiving_stage="source",
        catalog=_view((("transform", "llm"),)),
        guided=GuidedSession.initial(),
    )
    assert result == DeferredIntentAccepted(action=_action())

    wrong_target = validate_deferred_intent_action(
        _action(target_stage="output"),
        receiving_stage="source",
        catalog=_view((("transform", "llm"),)),
        guided=GuidedSession.initial(),
    )
    assert wrong_target == DeferredIntentRejected(reason="wrong_responsible_stage")


def test_kind_qualified_name_resolves_without_guessing_across_other_plugin_kinds() -> None:
    action = _action()
    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view((("source", "llm"), ("transform", "llm"), ("sink", "llm"))),
        guided=GuidedSession.initial(),
    )
    assert result == DeferredIntentAccepted(action=action)


def test_absent_and_policy_denied_catalog_subjects_remain_distinct() -> None:
    absent = validate_deferred_intent_action(
        _action(),
        receiving_stage="source",
        catalog=_view(()),
        guided=GuidedSession.initial(),
    )
    denied = validate_deferred_intent_action(
        _action(),
        receiving_stage="source",
        catalog=_view((("transform", "llm"),), available=frozenset()),
        guided=GuidedSession.initial(),
    )
    assert absent == DeferredIntentUnsupported(
        plugin_kind="transform",
        plugin_name="llm",
        reason=PluginUnavailableReason.NOT_INSTALLED,
    )
    assert denied == DeferredIntentUnsupported(
        plugin_kind="transform",
        plugin_name="llm",
        reason=PluginUnavailableReason.NOT_AUTHORIZED,
    )


def test_exact_policy_denial_wins_over_same_name_visible_in_another_plugin_kind() -> None:
    action = _action()

    result = validate_deferred_intent_action(
        action,
        receiving_stage="source",
        catalog=_view(
            (("source", "llm"), ("transform", "llm")),
            available=frozenset({PluginId("source", "llm")}),
        ),
        guided=GuidedSession.initial(),
    )

    assert result == DeferredIntentUnsupported(
        plugin_kind="transform",
        plugin_name="llm",
        reason=PluginUnavailableReason.NOT_AUTHORIZED,
    )


@pytest.mark.parametrize(
    ("receiving_stage", "target_stage"),
    [("source", "source"), ("output", "source"), ("topology", "output")],
)
def test_current_and_past_targets_are_rejected(receiving_stage: str, target_stage: str) -> None:
    result = validate_deferred_intent_action(
        _action(target_stage=target_stage),
        receiving_stage=receiving_stage,  # type: ignore[arg-type]
        catalog=_view((("transform", "llm"),)),
        guided=GuidedSession.initial(),
    )
    assert result == DeferredIntentRejected(reason="target_not_later")


def test_creation_binds_server_ids_and_exact_message_hash_without_raw_prose_in_metadata() -> None:
    action = _action()
    private_message = "Use llm on the private customer_notes field."
    intent = create_deferred_stage_intent(
        action,
        receiving_stage="source",
        intent_id=INTENT_ID,
        originating_message_id=MESSAGE_ID,
        originating_message_content=private_message,
    )

    assert intent.intent_id == INTENT_ID
    assert intent.originating_message_id == MESSAGE_ID
    assert intent.message_content_hash == stable_hash(private_message)
    assert intent.receiving_stage == "source"
    assert intent.target_stage == "topology"
    assert private_message not in repr(intent.to_dict())
    assert intent.to_dict()["redacted_summary"] == ("Future topology instruction for transform plugin 'llm'; 1 structural constraint(s).")


def test_model_summary_cannot_echo_private_message_into_durable_metadata() -> None:
    private_message = "Use the private customer_notes field with the llm transform."
    base = _action()
    echoing_action = DeferredIntentAction(
        target_stage=base.target_stage,
        catalog_kind=base.catalog_kind,
        catalog_name=base.catalog_name,
        redacted_summary=private_message,
        constraints=base.constraints,
    )

    intent = create_deferred_stage_intent(
        echoing_action,
        receiving_stage="source",
        intent_id=INTENT_ID,
        originating_message_id=MESSAGE_ID,
        originating_message_content=private_message,
    )

    assert private_message not in repr(intent.to_dict())
    assert intent.redacted_summary == "Future topology instruction for transform plugin 'llm'; 1 structural constraint(s)."
