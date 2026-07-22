"""Closed deferred-stage subject and constraint authority."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import get_args

import pytest

from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.schemas import PluginKind, PluginSchemaInfo, PluginSummary
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.stage_subjects import (
    CatalogSubjectClarification,
    CatalogSubjectResolved,
    CatalogSubjectUnsupported,
    ComponentCountConstraint,
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
    subject_from_dict,
)
from elspeth.web.plugin_policy.models import PluginAvailability, PluginAvailabilitySnapshot, PluginId, PluginUnavailableReason

SOURCE_ID = "11111111-1111-4111-8111-111111111111"
NODE_ID = "22222222-2222-4222-8222-222222222222"
OUTPUT_ID = "33333333-3333-4333-8333-333333333333"
SUBJECT_ID = "44444444-4444-4444-8444-444444444444"


class _Catalog:
    def __init__(self, plugins: tuple[tuple[PluginKind, str], ...]) -> None:
        self._plugins = plugins

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
        raise AssertionError("stage-subject resolution must use the policy-filtered list surface")

    def post_call_hints(
        self,
        *,
        plugin_type: PluginKind,
        plugin_name: str,
        tool_name: str,
        config_snapshot: dict[str, object],
    ) -> tuple[str, ...]:
        raise AssertionError("stage-subject resolution must not dispatch plugins")


def _view(
    installed: tuple[tuple[PluginKind, str], ...],
    *,
    available: frozenset[PluginId] | None = None,
) -> PolicyCatalogView:
    catalog = _Catalog(installed)
    permitted = frozenset(PluginId(kind, name) for kind, name in installed) if available is None else available
    denied = tuple(
        PluginAvailability(plugin_id=PluginId(kind, name), reason=PluginUnavailableReason.NOT_AUTHORIZED)
        for kind, name in installed
        if PluginId(kind, name) not in permitted
    )
    snapshot = PluginAvailabilitySnapshot.create(
        policy_hash="a" * 64,
        principal_scope="local:test",
        available=permitted,
        unavailable=denied,
        selected=(),
        usable_profile_aliases=(),
        selected_profile_aliases=(),
        binding_generation_fingerprint="b" * 64,
    )
    return PolicyCatalogView(catalog, snapshot, profiles=None)  # type: ignore[arg-type]


def test_stage_name_is_the_exact_closed_receiving_and_target_vocabulary() -> None:
    assert get_args(StageName) == ("source", "output", "topology", "wire_review")
    for stage in get_args(StageName):
        assert stage_name_from_value(stage, "DeferredStageIntent stage") == stage
    with pytest.raises(InvariantError, match=r"^DeferredStageIntent stage is unsupported$"):
        stage_name_from_value("deployment", "DeferredStageIntent stage")


def test_subject_and_constraint_fixtures_round_trip_without_schema_drift() -> None:
    stable = StableSubject(kind="stable", component_kind="source", stable_id=SOURCE_ID)
    plugin = PluginSubject(kind="plugin", subject_id=SUBJECT_ID, plugin_kind="transform", plugin_name="llm")
    subject_fixtures = (
        (stable, {"kind": "stable", "component_kind": "source", "stable_id": SOURCE_ID}),
        (
            plugin,
            {"kind": "plugin", "subject_id": SUBJECT_ID, "plugin_kind": "transform", "plugin_name": "llm"},
        ),
    )
    for subject, encoded in subject_fixtures:
        assert subject.to_dict() == encoded
        assert subject_from_dict(encoded) == subject

    constraints = (
        SubjectPresenceConstraint(kind="subject_presence", subject=plugin, present=True),
        OptionValueConstraint(kind="option_value", subject=stable, option_path=("dialect", "delimiter"), operator="equals", value=","),
        ComponentCountConstraint(
            kind="component_count",
            component_kind="node",
            plugin_kind="transform",
            plugin_name="llm",
            operator="at_least",
            count=2,
        ),
        EdgeRouteConstraint(
            kind="edge_route",
            from_subject=stable,
            edge_type="on_success",
            to_subject=PluginSubject(kind="plugin", subject_id=SUBJECT_ID, plugin_kind="sink", plugin_name="json"),
            present=True,
        ),
        FailureRouteConstraint(
            kind="failure_route",
            subject=StableSubject(kind="stable", component_kind="node", stable_id=NODE_ID),
            failure_kind="node_error",
            operator="equals",
            target=StableSubject(kind="stable", component_kind="output", stable_id=OUTPUT_ID),
        ),
    )
    expected = (
        {"kind": "subject_presence", "subject": plugin.to_dict(), "present": True},
        {
            "kind": "option_value",
            "subject": stable.to_dict(),
            "option_path": ["dialect", "delimiter"],
            "operator": "equals",
            "value": ",",
        },
        {
            "kind": "component_count",
            "component_kind": "node",
            "plugin_kind": "transform",
            "plugin_name": "llm",
            "operator": "at_least",
            "count": 2,
        },
        {
            "kind": "edge_route",
            "from_subject": stable.to_dict(),
            "edge_type": "on_success",
            "to_subject": constraints[3].to_subject.to_dict(),
            "present": True,
        },
        {
            "kind": "failure_route",
            "subject": constraints[4].subject.to_dict(),
            "failure_kind": "node_error",
            "operator": "equals",
            "target": constraints[4].target.to_dict(),  # type: ignore[union-attr]
        },
    )
    for constraint, encoded in zip(constraints, expected, strict=True):
        assert constraint.to_dict() == encoded
        assert constraint_from_dict(encoded) == constraint


@pytest.mark.parametrize(
    "value",
    [
        {"kind": "plugin", "subject_id": SUBJECT_ID, "plugin_kind": "queue", "plugin_name": "x"},
        {"kind": "stable", "component_kind": "source", "stable_id": SOURCE_ID, "extra": True},
        {"kind": "subject_presence", "subject": {"kind": "stable", "component_kind": "source", "stable_id": SOURCE_ID}, "present": 1},
        {
            "kind": "option_value",
            "subject": {"kind": "stable", "component_kind": "source", "stable_id": SOURCE_ID},
            "option_path": [],
            "operator": "equals",
            "value": 1,
        },
    ],
)
def test_decoders_keep_exact_types_closed_vocabularies_and_constraint_validation(value: object) -> None:
    decoder = (
        constraint_from_dict if type(value) is dict and value.get("kind") in {"subject_presence", "option_value"} else subject_from_dict
    )
    with pytest.raises(InvariantError):
        decoder(value)


def test_catalog_resolution_returns_unique_exact_kind_and_name() -> None:
    result = resolve_catalog_subject(_view((("source", "csv"), ("sink", "json"))), plugin_name="csv")
    assert result == CatalogSubjectResolved(status="resolved", plugin_kind="source", plugin_name="csv")


@pytest.mark.parametrize(
    "view",
    [
        _view((("source", "csv"),)),
        _view((("source", "private"),), available=frozenset()),
    ],
)
def test_absent_or_policy_denied_catalog_name_is_typed_unsupported(view: PolicyCatalogView) -> None:
    result = resolve_catalog_subject(view, plugin_name="private")
    assert result == CatalogSubjectUnsupported(status="unsupported", plugin_name="private", expected_kind=None, visible_kinds=())


def test_same_name_in_multiple_permitted_kinds_requires_deterministic_clarification() -> None:
    view = _view((("sink", "shared"), ("transform", "shared"), ("source", "shared")))
    assert resolve_catalog_subject(view, plugin_name="shared") == CatalogSubjectClarification(
        status="clarification",
        plugin_name="shared",
        plugin_kinds=("source", "transform", "sink"),
    )


def test_explicit_kind_resolves_ambiguity_but_exact_kind_mismatch_is_unsupported() -> None:
    view = _view((("source", "shared"), ("sink", "shared")))
    assert resolve_catalog_subject(view, plugin_name="shared", expected_kind="sink") == CatalogSubjectResolved(
        status="resolved", plugin_kind="sink", plugin_name="shared"
    )
    assert resolve_catalog_subject(view, plugin_name="shared", expected_kind="transform") == CatalogSubjectUnsupported(
        status="unsupported", plugin_name="shared", expected_kind="transform", visible_kinds=("source", "sink")
    )


def test_canonical_subject_definitions_exist_only_in_stage_subjects() -> None:
    guided_root = Path(__file__).parents[5] / "src/elspeth/web/composer/guided"
    canonical = {
        "StableSubject",
        "PluginSubject",
        "SubjectPresenceConstraint",
        "OptionValueConstraint",
        "ComponentCountConstraint",
        "EdgeRouteConstraint",
        "FailureRouteConstraint",
    }
    definitions: dict[str, list[str]] = {name: [] for name in canonical}
    for path in guided_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in canonical:
                definitions[node.name].append(path.name)
    assert definitions == {name: ["stage_subjects.py"] for name in canonical}

    state_machine = __import__("elspeth.web.composer.guided.state_machine", fromlist=["state_machine"])
    assert canonical.isdisjoint(vars(state_machine))
