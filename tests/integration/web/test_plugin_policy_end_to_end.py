"""Cross-surface acceptance for the universal web plugin policy."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
import sqlalchemy as sa
from fastapi import Request, Response
from sqlalchemy.pool import StaticPool

from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.plugin_capabilities import ControlMode, PluginCapability
from elspeth.core.secrets import resolve_secret_refs
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.service import content_hash as blob_content_hash
from elspeth.web.catalog import routes as catalog_routes
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginKind, PluginPolicyResponse, PluginSchemaInfo, PluginSummary
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
from elspeth.web.composer.guided.state_machine import ChainProposal, GuidedSession
from elspeth.web.composer.guided.steps import handle_step_3_chain_accept
from elspeth.web.composer.prompts import build_context_string
from elspeth.web.composer.recipes import get_recipe
from elspeth.web.composer.state import CompositionState, NodeSpec, OutputSpec, PipelineMetadata, SourceSpec
from elspeth.web.composer.tools._common import ToolContext
from elspeth.web.composer.tools.generation import _execute_get_plugin_assistance, _handle_get_plugin_schema
from elspeth.web.composer.tools.recipes import _execute_list_recipes
from elspeth.web.composer.tools.sessions import _execute_apply_pipeline_recipe
from elspeth.web.composer.tools.sources import _handle_list_sources
from elspeth.web.composer.tools.transforms import _handle_list_sinks, _handle_list_transforms, _handle_upsert_node
from elspeth.web.composer.yaml_generator import generate_public_yaml
from elspeth.web.composer.yaml_importer import composition_state_from_runtime_yaml
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.execution.preflight import require_settings_plugins_available, require_settings_sink_available
from elspeth.web.execution.service import _build_web_plugin_policy_evidence
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY
from elspeth.web.plugin_policy.availability import RequestPluginSnapshotFactory
from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
from elspeth.web.plugin_policy.models import (
    PluginAvailability,
    PluginAvailabilitySnapshot,
    PluginId,
    PluginUnavailableReason,
    WebPluginPolicy,
)
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, RuntimeWebPluginConfig
from elspeth.web.plugin_policy.validation import validate_plugin_policy
from elspeth.web.secrets.server_store import ServerSecretStore
from elspeth.web.secrets.service import ScopedSecretResolver, WebSecretService
from elspeth.web.secrets.user_store import UserSecretStore
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, sessions_table
from elspeth.web.sessions.routes._helpers import _dispatch_guided_respond
from elspeth.web.sessions.schema import initialize_session_schema
from tests.fixtures.stores import MockPayloadStore

_ROOT = Path(__file__).resolve().parents[3]
_MATRIX_FIXTURE = _ROOT / "src/elspeth/web/frontend/src/stores/__fixtures__/pluginPolicyMatrix.json"
_CORE_IDS = frozenset(
    {
        PluginId("source", "csv"),
        PluginId("source", "json"),
        PluginId("source", "text"),
        PluginId("transform", "field_mapper"),
        PluginId("transform", "llm"),
        PluginId("transform", "web_scrape"),
        PluginId("sink", "csv"),
        PluginId("sink", "json"),
        PluginId("sink", "text"),
    }
)
_AZURE_PROMPT = PluginId("transform", "azure_prompt_shield")
_AZURE_CONTENT = PluginId("transform", "azure_content_safety")
_AWS_PROMPT = PluginId("transform", "aws_bedrock_prompt_shield")
_AWS_CONTENT = PluginId("transform", "aws_bedrock_content_safety")


@dataclass(frozen=True)
class _MatrixCase:
    name: str
    extra_authorized: frozenset[PluginId]
    extra_available: frozenset[PluginId]
    selected_prompt: PluginId | None
    selected_content: PluginId | None


_AZURE_CONTROLS = frozenset({_AZURE_PROMPT, _AZURE_CONTENT})
_AWS_CONTROLS = frozenset({_AWS_PROMPT, _AWS_CONTENT})
_ALL_CONTROLS = _AZURE_CONTROLS | _AWS_CONTROLS


_MATRIX_CONTRACT = json.loads(_MATRIX_FIXTURE.read_text())
assert frozenset(map(PluginId.parse, _MATRIX_CONTRACT["core_plugin_ids"])) == _CORE_IDS


def _matrix_case(raw: dict[str, object]) -> _MatrixCase:
    selections = {entry["capability"]: entry["plugin_id"] for entry in raw["selections"]}  # type: ignore[index]
    return _MatrixCase(
        name=str(raw["name"]),
        extra_authorized=frozenset(map(PluginId.parse, raw["authorized_control_ids"])),  # type: ignore[arg-type]
        extra_available=frozenset(map(PluginId.parse, raw["available_control_ids"])),  # type: ignore[arg-type]
        selected_prompt=(None if selections["prompt_shield"] is None else PluginId.parse(str(selections["prompt_shield"]))),
        selected_content=(None if selections["content_safety"] is None else PluginId.parse(str(selections["content_safety"]))),
    )


_CASES = tuple(_matrix_case(raw) for raw in _MATRIX_CONTRACT["cases"])
_MATRIX_CASES_BY_NAME = {raw["name"]: raw for raw in _MATRIX_CONTRACT["cases"]}

_CONTROL_OPTIONS: dict[str, dict[str, object]] = {
    "azure_prompt_shield": {
        "endpoint": "https://example.com",
        "api_key": {"secret_ref": "AZURE_CONTENT_SAFETY_KEY"},
        "fields": ["text"],
        "schema": {"mode": "observed", "fields": None},
    },
    "azure_content_safety": {
        "endpoint": "https://example.com",
        "api_key": {"secret_ref": "AZURE_CONTENT_SAFETY_KEY"},
        "fields": ["text"],
        "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
        "schema": {"mode": "observed", "fields": None},
    },
    "aws_bedrock_prompt_shield": {
        "profile": "prompt-matrix",
        "fields": ["text"],
        "schema": {"mode": "observed", "fields": None},
    },
    "aws_bedrock_content_safety": {
        "profile": "content-matrix",
        "fields": ["text"],
        "source": "OUTPUT",
        "schema": {"mode": "observed", "fields": None},
    },
}


class _MatrixCatalog(CatalogService):
    def __init__(self) -> None:
        full = create_catalog_service()
        summaries = (
            {PluginId("source", item.name): item for item in full.list_sources()}
            | {PluginId("transform", item.name): item for item in full.list_transforms()}
            | {PluginId("sink", item.name): item for item in full.list_sinks()}
        )
        schemas = {plugin_id: full.get_schema(plugin_id.kind, plugin_id.name) for plugin_id in summaries}
        self._summaries = summaries
        self._schemas = schemas

    def _list(self, kind: PluginKind) -> list[PluginSummary]:
        return [summary for plugin_id, summary in self._summaries.items() if plugin_id.kind == kind]

    def list_sources(self) -> list[PluginSummary]:
        return self._list("source")

    def list_transforms(self) -> list[PluginSummary]:
        return self._list("transform")

    def list_sinks(self) -> list[PluginSummary]:
        return self._list("sink")

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        return self._schemas[PluginId(plugin_type, name)]

    def post_call_hints(
        self,
        *,
        plugin_type: PluginKind,
        plugin_name: str,
        tool_name: str,
        config_snapshot: object,
    ) -> tuple[str, ...]:
        del plugin_type, plugin_name, tool_name, config_snapshot
        return ()


def _policy(case: _MatrixCase) -> WebPluginPolicy:
    authorized = _CORE_IDS | case.extra_authorized

    def ordered(selected: PluginId | None, candidates: frozenset[PluginId]) -> tuple[PluginId, ...]:
        return (() if selected is None else (selected,)) + tuple(sorted(candidates - ({selected} if selected is not None else set())))

    return WebPluginPolicy.create(
        required=_CORE_IDS,
        configured_optional=case.extra_authorized,
        preferences=(
            (PluginCapability.PROMPT_SHIELD, ordered(case.selected_prompt, case.extra_authorized & {_AZURE_PROMPT, _AWS_PROMPT})),
            (PluginCapability.CONTENT_SAFETY, ordered(case.selected_content, case.extra_authorized & {_AZURE_CONTENT, _AWS_CONTENT})),
        ),
        control_modes=(
            (PluginCapability.PROMPT_SHIELD, ControlMode.RECOMMEND),
            (PluginCapability.CONTENT_SAFETY, ControlMode.RECOMMEND),
        ),
        plugin_code_identities=tuple((plugin_id, "1.0.0", "sha256:" + "a" * 16) for plugin_id in sorted(authorized)),
    )


def _snapshot(case: _MatrixCase, policy: WebPluginPolicy) -> PluginAvailabilitySnapshot:
    llm_id = PluginId("transform", "llm")
    usable_profiles: list[tuple[PluginId, tuple[str, ...]]] = [(llm_id, ("tutorial",))]
    selected_profiles: list[tuple[PluginId, str]] = [(llm_id, "tutorial")]
    if _AWS_PROMPT in case.extra_available:
        usable_profiles.append((_AWS_PROMPT, ("prompt-matrix",)))
        selected_profiles.append((_AWS_PROMPT, "prompt-matrix"))
    if _AWS_CONTENT in case.extra_available:
        usable_profiles.append((_AWS_CONTENT, ("content-matrix",)))
        selected_profiles.append((_AWS_CONTENT, "content-matrix"))
    return PluginAvailabilitySnapshot.create(
        policy_hash=policy.policy_hash,
        principal_scope=f"local:{case.name}",
        available=_CORE_IDS | case.extra_available,
        unavailable=tuple(
            PluginAvailability(plugin_id, PluginUnavailableReason.LOCAL_REQUIREMENT_MISSING)
            for plugin_id in sorted(case.extra_authorized - case.extra_available)
        ),
        selected=(
            (PluginCapability.LLM, llm_id),
            (PluginCapability.PROMPT_SHIELD, case.selected_prompt),
            (PluginCapability.CONTENT_SAFETY, case.selected_content),
        ),
        usable_profile_aliases=tuple(usable_profiles),
        selected_profile_aliases=tuple(selected_profiles),
        control_modes=(
            (PluginCapability.PROMPT_SHIELD, ControlMode.RECOMMEND),
            (PluginCapability.CONTENT_SAFETY, ControlMode.RECOMMEND),
        ),
        binding_generation_fingerprint="2" * 64,
    )


def _profiles(policy: WebPluginPolicy) -> OperatorProfileRegistry:
    settings = WebSettings(
        secret_key="0123456789abcdef0123456789abcdef",
        composer_max_composition_turns=4,
        composer_max_discovery_turns=4,
        composer_timeout_seconds=60,
        composer_rate_limit_per_minute=20,
        shareable_link_signing_key=b"0123456789abcdef0123456789abcdef",
        plugin_allowlist=[str(_AWS_PROMPT), str(_AWS_CONTENT)],
        llm_profiles={
            "tutorial": {
                "provider": "bedrock",
                "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
                "region_name": "ap-southeast-2",
            }
        },
        tutorial_llm_profile="tutorial",
        bedrock_guardrail_profiles=[
            {
                "alias": "prompt-matrix",
                "plugin": _AWS_PROMPT.name,
                "guardrail_identifier": "privatepromptguardrail",
                "guardrail_version": "7",
                "region": "ap-southeast-2",
            },
            {
                "alias": "content-matrix",
                "plugin": _AWS_CONTENT.name,
                "guardrail_identifier": "privatecontentguardrail",
                "guardrail_version": "11",
                "region": "ap-southeast-2",
            },
        ],
        bedrock_guardrail_default_profiles={
            _AWS_PROMPT.name: "prompt-matrix",
            _AWS_CONTENT.name: "content-matrix",
        },
    )
    return OperatorProfileRegistry(policy=policy, settings=RuntimeWebPluginConfig.from_settings(settings))


def _view(snapshot: PluginAvailabilitySnapshot, profiles: OperatorProfileRegistry) -> PolicyCatalogView:
    return PolicyCatalogView(_MatrixCatalog(), snapshot, profiles)


async def _catalog_http_surfaces(
    *,
    snapshot: PluginAvailabilitySnapshot,
    policy: WebPluginPolicy,
    profiles: OperatorProfileRegistry,
) -> tuple[frozenset[str], PluginPolicyResponse]:
    app = SimpleNamespace(
        state=SimpleNamespace(
            catalog_service=_MatrixCatalog(),
            plugin_snapshot_factory=lambda _user: snapshot,
            operator_profile_registry=profiles,
            web_plugin_policy=policy,
        )
    )
    request = Request({"type": "http", "method": "GET", "path": "/api/catalog", "headers": [], "app": app})
    response = Response()
    user = UserIdentity(user_id="matrix-user", username="matrix-user")
    sources, transforms, sinks, policy_response = await asyncio.gather(
        catalog_routes.list_sources(request, response, user),
        catalog_routes.list_transforms(request, response, user),
        catalog_routes.list_sinks(request, response, user),
        catalog_routes.get_policy(request, response, user),
    )
    assert response.headers["X-ELSPETH-Plugin-Snapshot"] == snapshot.snapshot_hash
    listed = frozenset(str(PluginId(item.plugin_type, item.name)) for item in (*sources, *transforms, *sinks))
    return listed, policy_response


def _control_probe_state(plugin_id: PluginId) -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id="control_probe",
                node_type="transform",
                plugin=plugin_id.name,
                input="rows",
                on_success="checked",
                on_error="discard",
                options=deepcopy(_CONTROL_OPTIONS[plugin_id.name]),
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _imported_control_probe_state(plugin_id: PluginId) -> CompositionState:
    return composition_state_from_runtime_yaml(generate_public_yaml(_control_probe_state(plugin_id)))


def _selected_contract(case: _MatrixCase) -> dict[str, str | None]:
    return {
        PluginCapability.LLM.value: "transform:llm",
        PluginCapability.PROMPT_SHIELD.value: None if case.selected_prompt is None else str(case.selected_prompt),
        PluginCapability.CONTENT_SAFETY.value: None if case.selected_content is None else str(case.selected_content),
    }


def _capability_contract(view: PolicyCatalogView) -> dict[str, tuple[str, ...]]:
    return {capability.value: tuple(map(str, plugin_ids)) for capability, plugin_ids in view.capability_groups().items()}


def _seed_recipe_blob(tmp_path: Path) -> tuple[sa.engine.Engine, str, str]:
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    session_id = str(uuid4())
    blob_id = str(uuid4())
    now = datetime.now(UTC)
    body = b'[{"url":"https://www.dta.gov.au"}]'
    storage_dir = tmp_path / "blobs" / session_id
    storage_dir.mkdir(parents=True)
    storage_path = storage_dir / f"{blob_id}_urls.json"
    storage_path.write_bytes(body)
    with engine.begin() as connection:
        connection.execute(
            sessions_table.insert().values(
                id=session_id,
                user_id="matrix-user",
                auth_provider_type="local",
                title="Policy matrix",
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            blobs_table.insert().values(
                id=blob_id,
                session_id=session_id,
                filename="urls.json",
                mime_type="application/json",
                size_bytes=len(body),
                content_hash=blob_content_hash(body),
                storage_path=str(storage_path),
                created_at=now,
                created_by="user",
                source_description=None,
                status="ready",
            )
        )
    return engine, session_id, blob_id


def _guided_base_state() -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="main",
            options={"path": "input.csv", "schema": {"mode": "observed", "guaranteed_fields": ["text"]}},
            on_validation_failure="discard",
        ),
        nodes=(),
        edges=(),
        outputs=(
            OutputSpec(
                name="output",
                plugin="json",
                options={"path": "output.jsonl", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


def _proposal(plugin_id: PluginId | None) -> ChainProposal:
    steps = (
        ()
        if plugin_id is None
        else (
            {
                "plugin": plugin_id.name,
                "options": deepcopy(_CONTROL_OPTIONS[plugin_id.name]),
                "rationale": "screen text without claiming row filtering",
            },
        )
    )
    return ChainProposal(steps=steps, why="five-configuration policy matrix")


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_policy_surface_parity_matrix(case: _MatrixCase, tmp_path: Path) -> None:
    policy = _policy(case)
    snapshot = _snapshot(case, policy)
    profiles = _profiles(policy)
    view = _view(snapshot, profiles)
    expected = frozenset(map(str, snapshot.available))
    expected_selected = _selected_contract(case)
    expected_capabilities = _capability_contract(view)
    empty_state = CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)
    engine, session_id, blob_id = _seed_recipe_blob(tmp_path)
    context = ToolContext(catalog=view, plugin_snapshot=snapshot, session_engine=engine, session_id=session_id)

    catalog_api, policy_wire = asyncio.run(_catalog_http_surfaces(snapshot=snapshot, policy=policy, profiles=profiles))
    guided_results = (
        _handle_list_sources({}, empty_state, context),
        _handle_list_transforms({}, empty_state, context),
        _handle_list_sinks({}, empty_state, context),
    )
    guided_discovery = frozenset(str(PluginId(item.plugin_type, item.name)) for result in guided_results for item in result.data)
    prompt = build_context_string(empty_state, view, plugin_snapshot=snapshot, schemas_loaded=frozenset())
    freeform_policy = json.loads(prompt.partition("\n")[2])["plugin_policy"]
    freeform_prompt = frozenset(freeform_policy["available_ids"])
    evidence = _build_web_plugin_policy_evidence(snapshot=snapshot, policy=policy)
    recipe_result = _execute_list_recipes({}, empty_state, context)
    fixture_case = _MATRIX_CASES_BY_NAME[case.name]
    fixture_projection = {
        "principal_scope": fixture_case["principal_scope"],
        "snapshot_fingerprint": fixture_case["snapshot_fingerprint"],
        "policy_hash": fixture_case["policy_hash"],
        "available_plugin_ids": frozenset(fixture_case["available_plugin_ids"]),
        "capability_groups": {row["capability"]: tuple(row["available_plugin_ids"]) for row in fixture_case["capability_groups"]},
        "selections": {row["capability"]: row["plugin_id"] for row in fixture_case["selections"]},
        "control_modes": {row["capability"]: row["mode"] for row in fixture_case["control_modes"]},
    }
    backend_projection = {
        "principal_scope": snapshot.principal_scope,
        "snapshot_fingerprint": snapshot.snapshot_hash,
        "policy_hash": policy.policy_hash,
        "available_plugin_ids": expected,
        "capability_groups": expected_capabilities,
        "selections": expected_selected,
        "control_modes": {capability.value: mode.value for capability, mode in snapshot.control_modes},
    }

    assert backend_projection == fixture_projection
    assert catalog_api == frozenset(policy_wire.available_plugin_ids) == guided_discovery == freeform_prompt == expected
    assert {row.capability.value: row.plugin_id for row in policy_wire.selections} == expected_selected
    assert freeform_policy["selected"] == expected_selected
    assert dict(evidence.selected_implementations) == expected_selected
    assert {row.capability.value: tuple(row.available_plugin_ids) for row in policy_wire.capability_groups} == expected_capabilities
    assert {name: tuple(plugin_ids) for name, plugin_ids in freeform_policy["capability_groups"].items()} == expected_capabilities
    guided_capabilities: dict[str, set[str]] = {}
    for result in guided_results:
        assert result.success is True
        for item in result.data:
            plugin_id = str(PluginId(item.plugin_type, item.name))
            for declaration in item.policy_capabilities:
                guided_capabilities.setdefault(declaration.capability.value, set()).add(plugin_id)
    assert {capability: tuple(sorted(plugin_ids)) for capability, plugin_ids in guided_capabilities.items()} == expected_capabilities
    assert frozenset(evidence.available_plugin_ids) == expected
    assert frozenset(evidence.authorized_plugin_ids) == frozenset(map(str, _CORE_IDS | case.extra_authorized))
    assert validate_plugin_policy(empty_state, snapshot=snapshot, profile_registry=profiles).findings == ()

    for plugin_id in _CORE_IDS | _ALL_CONTROLS:
        schema_result = _handle_get_plugin_schema(
            {"plugin_type": plugin_id.kind, "name": plugin_id.name},
            empty_state,
            context,
        )
        assistance_result = _execute_get_plugin_assistance(
            {"plugin_type": plugin_id.kind, "plugin_name": plugin_id.name},
            empty_state,
            context,
        )
        assert schema_result.success is (plugin_id in snapshot.available)
        assert assistance_result.success is (plugin_id in snapshot.available)
        if schema_result.success:
            assert schema_result.data.name == plugin_id.name
        else:
            expected_code = (
                PluginUnavailableReason.LOCAL_REQUIREMENT_MISSING.value
                if plugin_id in case.extra_authorized
                else PluginUnavailableReason.NOT_AUTHORIZED.value
            )
            assert schema_result.data["error_code"] == expected_code
            assert assistance_result.data["error_code"] == expected_code

    assert recipe_result.success is True
    assert "web-scrape-llm-rate-jsonl" in {recipe["name"] for recipe in recipe_result.data["recipes"]}
    for recipe_info in recipe_result.data["recipes"]:
        recipe = get_recipe(recipe_info["name"])
        assert recipe is not None
        assert recipe.required_plugins <= snapshot.available
        assert all(not alternatives.isdisjoint(snapshot.available) for alternatives in recipe.alternative_plugin_groups)

    recipe_fast_path = _execute_apply_pipeline_recipe(
        {
            "recipe_name": "web-scrape-llm-rate-jsonl",
            "slots": {
                "source_blob_id": blob_id,
                "source_plugin": "json",
                "profile": "tutorial",
                "abuse_contact": "matrix@example.invalid",
                "scraping_reason": "five-configuration policy matrix",
            },
        },
        empty_state,
        context,
    )
    assert recipe_fast_path.success is True
    assert [node.plugin for node in recipe_fast_path.updated_state.nodes] == ["web_scrape", "llm", "field_mapper"]

    for plugin_id in _ALL_CONTROLS:
        probe = _control_probe_state(plugin_id)
        validation = validate_plugin_policy(probe, snapshot=snapshot, profile_registry=profiles)
        imported_validation = validate_plugin_policy(
            _imported_control_probe_state(plugin_id),
            snapshot=snapshot,
            profile_registry=profiles,
        )
        direct_tool = _handle_upsert_node(
            {
                "id": "control_probe",
                "node_type": "transform",
                "plugin": plugin_id.name,
                "input": "rows",
                "on_success": "checked",
                "on_error": "discard",
                "options": deepcopy(_CONTROL_OPTIONS[plugin_id.name]),
            },
            empty_state,
            context,
        )
        guided_result = handle_step_3_chain_accept(
            state=_guided_base_state(),
            session=replace(GuidedSession.initial(), step=GuidedStep.STEP_3_TRANSFORMS),
            proposal=_proposal(plugin_id),
            catalog=view,
            plugin_snapshot=snapshot,
        )
        runtime_settings = SimpleNamespace(
            sources={},
            transforms=(SimpleNamespace(plugin=plugin_id.name),),
            aggregations=(),
            sinks={},
        )
        if plugin_id in snapshot.available:
            assert validation.findings_for("plugin_enablement") == ()
            assert validation.findings_for("operator_profile_options") == ()
            assert imported_validation.findings_for("plugin_enablement") == ()
            assert imported_validation.findings_for("operator_profile_options") == ()
            assert direct_tool.success is True
            assert guided_result.tool_result.success is True
            assert direct_tool.updated_state.nodes[0].options == probe.nodes[0].options
            if plugin_id in _AWS_CONTROLS:
                assert "profile" in probe.nodes[0].options
                assert "guardrail_identifier" not in probe.nodes[0].options
                executable_options = validation.executable_state.nodes[0].options
                assert executable_options["guardrail_identifier"].startswith("private")
                assert executable_options["guardrail_version"] in {"7", "11"}
                assert executable_options["region"] == "ap-southeast-2"
            require_settings_plugins_available(runtime_settings, snapshot)
        else:
            expected_code = (
                PluginUnavailableReason.LOCAL_REQUIREMENT_MISSING.value
                if plugin_id in case.extra_authorized
                else PluginUnavailableReason.NOT_AUTHORIZED.value
            )
            assert [finding.error_code for finding in validation.findings_for("plugin_enablement")] == [expected_code]
            assert [finding.error_code for finding in imported_validation.findings_for("plugin_enablement")] == [expected_code]
            assert direct_tool.success is False
            assert direct_tool.data["error_code"] == expected_code
            assert guided_result.tool_result.success is False
            assert guided_result.tool_result.data["error_code"] == expected_code
            with pytest.raises(ValueError, match="not available"):
                require_settings_plugins_available(runtime_settings, snapshot)

    dispatch_proposal = _proposal(case.selected_prompt)
    dispatch_session = replace(
        GuidedSession.initial(),
        step=GuidedStep.STEP_3_TRANSFORMS,
        step_3_proposal=dispatch_proposal,
    )
    dispatched_state, dispatched_session, next_turn = asyncio.run(
        _dispatch_guided_respond(
            state=_guided_base_state(),
            guided=dispatch_session,
            current_step=GuidedStep.STEP_3_TRANSFORMS,
            current_turn_type=TurnType.PROPOSE_CHAIN,
            turn_response={
                "chosen": ["accept"],
                "edited_values": None,
                "custom_inputs": None,
                "accepted_step_index": None,
                "edit_step_index": None,
                "control_signal": None,
            },
            catalog=view,
            plugin_snapshot=snapshot,
            recorder=BufferingRecorder(),
            user_id="matrix-user",
            data_dir=None,
            session_engine=None,
            session_id=session_id,
            blob_service=None,
            payload_store=MockPayloadStore(),
            model="matrix-model",
            temperature=None,
            seed=None,
        )
    )
    assert dispatched_session.step is GuidedStep.STEP_4_WIRE
    assert next_turn["type"] == TurnType.CONFIRM_WIRING.value
    assert [node.plugin for node in dispatched_state.nodes] == ([] if case.selected_prompt is None else [case.selected_prompt.name])

    delayed_sinks = {name: SimpleNamespace(plugin=name) for name in ("csv", "json", "text")}
    delayed_sinks["disabled"] = SimpleNamespace(plugin="azure_blob")
    delayed_settings = SimpleNamespace(sinks=delayed_sinks)
    for sink_name in ("csv", "json", "text"):
        require_settings_sink_available(delayed_settings, snapshot, sink_name)
    with pytest.raises(ValueError, match="not available"):
        require_settings_sink_available(delayed_settings, snapshot, "disabled")
    engine.dispose()


def test_policy_matrix_cases_are_five_distinct_authorization_and_availability_contracts() -> None:
    contracts = {
        (
            _policy(case).policy_hash,
            _snapshot(case, _policy(case)).snapshot_hash,
            tuple(sorted(map(str, case.extra_authorized))),
            tuple(sorted(map(str, case.extra_available))),
        )
        for case in _CASES
    }

    assert len(contracts) == 5


def test_every_web_tool_context_constructor_supplies_policy_context() -> None:
    missing: list[str] = []
    for path in (_ROOT / "src/elspeth/web").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ToolContext":
                keywords = {keyword.arg for keyword in node.keywords}
                if not {"catalog", "plugin_snapshot"} <= keywords:
                    missing.append(f"{path.relative_to(_ROOT)}:{node.lineno}")
    assert missing == []


def test_web_preflight_is_the_only_runtime_factory_caller() -> None:
    callers: list[str] = []
    for path in (_ROOT / "src/elspeth/web/execution").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        if any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "instantiate_plugins_from_config"
            for node in ast.walk(tree)
        ):
            callers.append(str(path.relative_to(_ROOT)))
    assert callers == ["src/elspeth/web/execution/preflight.py"]


def test_removed_provider_specific_availability_path_cannot_return() -> None:
    assert not (_ROOT / "src/elspeth/web/composer/tools/_shield_availability.py").exists()


def test_core_and_cli_layers_do_not_import_web_policy() -> None:
    offenders: list[str] = []
    for base in (_ROOT / "src/elspeth/core", _ROOT / "src/elspeth/cli"):
        if not base.exists():
            continue
        paths = base.rglob("*.py") if base.is_dir() else (base,)
        for path in paths:
            if "elspeth.web.plugin_policy" in path.read_text():
                offenders.append(str(path.relative_to(_ROOT)))
    assert offenders == []


def test_prompt_builder_has_no_environment_or_profile_binding_access() -> None:
    source = inspect.getsource(build_context_string)
    assert "os.environ" not in source
    assert "credential_ref" not in source
    assert "provider_options" not in source


def test_web_dispatch_policy_context_has_no_allow_all_defaults() -> None:
    parameters = inspect.signature(ToolContext).parameters
    assert parameters["catalog"].default is inspect.Parameter.empty
    assert parameters["plugin_snapshot"].default is inspect.Parameter.empty


def test_web_services_and_validation_require_explicit_policy_context() -> None:
    from elspeth.web.composer.service import ComposerServiceImpl
    from elspeth.web.execution.service import ExecutionServiceImpl
    from elspeth.web.execution.validation import validate_pipeline

    execution = inspect.signature(ExecutionServiceImpl).parameters
    composer = inspect.signature(ComposerServiceImpl).parameters
    validation = inspect.signature(validate_pipeline).parameters

    for parameters in (execution, composer):
        assert parameters["plugin_snapshot_factory"].default is inspect.Parameter.empty
        assert parameters["operator_profile_registry"].default is inspect.Parameter.empty
    assert validation["plugin_snapshot"].default is inspect.Parameter.empty
    assert validation["profile_registry"].default is inspect.Parameter.empty


def test_web_services_reject_explicit_none_policy_context() -> None:
    from elspeth.web.composer.service import ComposerServiceImpl
    from elspeth.web.execution.service import ExecutionServiceImpl

    unused_dependency: Any = object()

    def unexpected_snapshot_request(_user_id: str) -> PluginAvailabilitySnapshot:
        raise AssertionError("constructor must reject its missing policy dependency before requesting a snapshot")

    with pytest.raises(TypeError, match="plugin_snapshot_factory"):
        ComposerServiceImpl(
            catalog=unused_dependency,
            settings=unused_dependency,
            plugin_snapshot_factory=None,
            operator_profile_registry=unused_dependency,
        )
    with pytest.raises(TypeError, match="operator_profile_registry"):
        ComposerServiceImpl(
            catalog=unused_dependency,
            settings=unused_dependency,
            plugin_snapshot_factory=unexpected_snapshot_request,
            operator_profile_registry=None,
        )
    with pytest.raises(TypeError, match="plugin_snapshot_factory"):
        ExecutionServiceImpl(
            loop=unused_dependency,
            broadcaster=unused_dependency,
            settings=unused_dependency,
            session_service=unused_dependency,
            yaml_generator=unused_dependency,
            telemetry=unused_dependency,
            plugin_snapshot_factory=None,
            operator_profile_registry=unused_dependency,
            web_plugin_policy=unused_dependency,
        )
    with pytest.raises(TypeError, match="operator_profile_registry"):
        ExecutionServiceImpl(
            loop=unused_dependency,
            broadcaster=unused_dependency,
            settings=unused_dependency,
            session_service=unused_dependency,
            yaml_generator=unused_dependency,
            telemetry=unused_dependency,
            plugin_snapshot_factory=unexpected_snapshot_request,
            operator_profile_registry=None,
            web_plugin_policy=unused_dependency,
        )
    with pytest.raises(TypeError, match="web_plugin_policy"):
        ExecutionServiceImpl(
            loop=unused_dependency,
            broadcaster=unused_dependency,
            settings=unused_dependency,
            session_service=unused_dependency,
            yaml_generator=unused_dependency,
            telemetry=unused_dependency,
            plugin_snapshot_factory=unexpected_snapshot_request,
            operator_profile_registry=unused_dependency,
            web_plugin_policy=None,
        )


def test_web_tree_has_no_trained_operator_service_roots_or_calls() -> None:
    """HTTP composition roots must never opt into the named non-web mode."""
    offenders: list[str] = []
    request_root = _ROOT / "src/elspeth/web/sessions"
    paths = [*request_root.rglob("*.py"), _ROOT / "src/elspeth/web/app.py"]
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "for_trained_operator":
                offenders.append(f"{path.relative_to(_ROOT)}:{node.lineno}")
    assert offenders == []


def test_server_profile_scope_survives_lowering_and_same_name_user_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "profile-scope-test-fingerprint-key")
    monkeypatch.setenv("SHARED_LLM_KEY", "server-value")
    engine: sa.engine.Engine = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(engine)
    user_store = UserSecretStore(engine=engine, master_key="profile-scope-test-master-key")
    user_store.set_secret(
        "SHARED_LLM_KEY",
        value="user-value",
        user_id="alice",
        auth_provider_type="local",
    )
    server_store = ServerSecretStore(("SHARED_LLM_KEY",))
    service = WebSecretService(user_store=user_store, server_store=server_store)
    resolver = ScopedSecretResolver(service, "local")
    settings = WebSettings(
        composer_max_composition_turns=4,
        composer_max_discovery_turns=4,
        composer_timeout_seconds=60,
        composer_rate_limit_per_minute=20,
        shareable_link_signing_key=b"0123456789abcdef0123456789abcdef",
        llm_profiles={
            "server-profile": {
                "provider": "openrouter",
                "model": "openai/gpt-5-mini",
                "credential_scope": "server",
                "credential_ref": "SHARED_LLM_KEY",
            }
        },
    )
    runtime = RuntimeWebPluginConfig.from_settings(settings)
    policy = compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)
    profiles = OperatorProfileRegistry(policy=policy, settings=runtime)
    lowered = profiles.lower_options(
        PluginId("transform", "llm"),
        alias="server-profile",
        safe_options={"prompt_template": "{{ row }}", "schema": {"mode": "observed"}},
    )

    resolved, evidence = resolve_secret_refs(
        {"options": deep_thaw(lowered.executable_options)},
        resolver,
        "alice",
    )
    _server_value, server_ref = server_store.get_secret("SHARED_LLM_KEY")

    assert resolved["options"]["api_key"] == "server-value"
    assert evidence[0].scope == "server"
    assert evidence[0].fingerprint == server_ref.fingerprint


def test_validate_pipeline_resolves_server_profile_before_plugin_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from elspeth.web.composer import yaml_generator
    from elspeth.web.execution import validation as validation_module

    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "profile-validation-fingerprint-key")
    monkeypatch.setenv("SHARED_LLM_KEY", "server-value")
    engine: sa.engine.Engine = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(engine)
    user_store = UserSecretStore(engine=engine, master_key="profile-validation-master-key")
    user_store.set_secret(
        "SHARED_LLM_KEY",
        value="user-value",
        user_id="alice",
        auth_provider_type="local",
    )
    server_store = ServerSecretStore(("SHARED_LLM_KEY",))
    service = WebSecretService(user_store=user_store, server_store=server_store)
    delegate = ScopedSecretResolver(service, "local")
    settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=4,
        composer_max_discovery_turns=4,
        composer_timeout_seconds=60,
        composer_rate_limit_per_minute=20,
        shareable_link_signing_key=b"0123456789abcdef0123456789abcdef",
        llm_profiles={
            "server-profile": {
                "provider": "openrouter",
                "model": "openai/gpt-5-mini",
                "credential_scope": "server",
                "credential_ref": "SHARED_LLM_KEY",
            }
        },
    )
    runtime = RuntimeWebPluginConfig.from_settings(settings)
    policy = compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)
    profiles = OperatorProfileRegistry(policy=policy, settings=runtime)
    catalog = create_catalog_service()
    snapshot = RequestPluginSnapshotFactory(
        policy=policy,
        catalog=catalog,
        profiles=profiles,
        auth_provider="local",
        secret_service=service,
        server_store=server_store,
        user_store=user_store,
        generation_key=b"profile-validation-generation-key",
    ).for_user_id("alice")

    scoped_resolutions = []
    generic_resolution_names: list[str] = []

    class _RecordingResolver:
        def list_refs(self, user_id: str):
            return delegate.list_refs(user_id)

        def has_ref(self, user_id: str, name: str) -> bool:
            generic_resolution_names.append(name)
            return delegate.has_ref(user_id, name)

        def resolve(self, user_id: str, name: str):
            generic_resolution_names.append(name)
            return delegate.resolve(user_id, name)

        def resolve_scoped(self, user_id: str, name: str, scope: str):
            resolved = delegate.resolve_scoped(user_id, name, scope)
            if resolved is not None:
                scoped_resolutions.append(resolved)
            return resolved

    input_path = tmp_path / "blobs" / "input.csv"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("customer\nAlice\n")
    state = CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="rows",
            options={"path": str(input_path), "schema": {"mode": "fixed", "fields": ["customer: str"]}},
            on_validation_failure="discard",
        ),
        nodes=(
            NodeSpec(
                id="classifier",
                node_type="transform",
                plugin="llm",
                input="rows",
                on_success="labelled",
                on_error="discard",
                options={
                    "profile": "server-profile",
                    "prompt_template": "Classify {{ row['customer'] }}",
                    "response_field": "classification",
                    "schema": {"mode": "observed", "fields": None},
                    "required_input_fields": ["customer"],
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        {
                            "id": "prompt_template_review:classifier",
                            "kind": "llm_prompt_template",
                            "user_term": "llm_prompt_template:classifier",
                            "status": "resolved",
                            "draft": "Classify {{ row['customer'] }}",
                            "event_id": "evt-prompt",
                            "accepted_value": "Classify {{ row['customer'] }}",
                            "accepted_artifact_hash": None,
                            "resolved_prompt_template_hash": stable_hash("Classify {{ row['customer'] }}"),
                        },
                        {
                            "id": "model_choice_review:classifier",
                            "kind": "llm_model_choice",
                            "user_term": "llm_model_choice:classifier",
                            "status": "resolved",
                            "draft": "openai/gpt-5-mini",
                            "event_id": "evt-model",
                            "accepted_value": "openai/gpt-5-mini",
                            "accepted_artifact_hash": None,
                            "resolved_prompt_template_hash": stable_hash("openai/gpt-5-mini"),
                        },
                    ],
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(
                name="labelled",
                plugin="json",
                options={
                    "path": "outputs/classified.jsonl",
                    "format": "jsonl",
                    "schema": {"mode": "observed", "fields": None},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )
    original_constructor = validation_module.instantiate_runtime_plugins
    constructor_api_keys: list[object] = []

    def _capture_constructor(runtime_settings, *, plugin_snapshot):
        constructor_api_keys.append(runtime_settings.transforms[0].options["api_key"])
        assert scoped_resolutions
        return original_constructor(runtime_settings, plugin_snapshot=plugin_snapshot)

    monkeypatch.setattr(validation_module, "instantiate_runtime_plugins", _capture_constructor)

    result = validation_module.validate_pipeline(
        state,
        settings,
        yaml_generator,
        plugin_snapshot=snapshot,
        profile_registry=profiles,
        secret_service=_RecordingResolver(),
        user_id="alice",
        session_id="session-profile-validation",
    )
    _server_value, server_ref = server_store.get_secret("SHARED_LLM_KEY")

    assert result.is_valid, result.errors
    assert constructor_api_keys == ["server-value"]
    assert generic_resolution_names == []
    assert scoped_resolutions
    assert {item.scope for item in scoped_resolutions} == {"server"}
    assert {item.value for item in scoped_resolutions} == {"server-value"}
    assert {item.fingerprint for item in scoped_resolutions} == {server_ref.fingerprint}
