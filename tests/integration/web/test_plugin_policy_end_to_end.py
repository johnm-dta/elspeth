"""Cross-surface acceptance for the universal web plugin policy."""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa

from elspeth.contracts.plugin_capabilities import ControlMode, PluginCapability
from elspeth.core.secrets import resolve_secret_refs
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginKind, PluginSchemaInfo, PluginSummary
from elspeth.web.composer.prompts import build_context_string
from elspeth.web.composer.recipes import get_recipe, list_recipes
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools._common import ToolContext
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.execution.service import _build_web_plugin_policy_evidence
from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, RuntimeWebPluginConfig
from elspeth.web.plugin_policy.validation import validate_plugin_policy
from elspeth.web.secrets.server_store import ServerSecretStore
from elspeth.web.secrets.service import ScopedSecretResolver, WebSecretService
from elspeth.web.secrets.user_store import UserSecretStore
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema

_ROOT = Path(__file__).resolve().parents[3]
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
_AWS_PROMPT_FIXTURE = PluginId("transform", "aws_ready_prompt_shield_fixture")
_AWS_CONTENT_FIXTURE = PluginId("transform", "aws_ready_content_safety_fixture")


@dataclass(frozen=True)
class _MatrixCase:
    name: str
    extra_available: frozenset[PluginId]
    selected_prompt: PluginId | None
    selected_content: PluginId | None


_CASES = (
    _MatrixCase("core_only", frozenset(), None, None),
    _MatrixCase("azure_only", frozenset({_AZURE_PROMPT, _AZURE_CONTENT}), _AZURE_PROMPT, _AZURE_CONTENT),
    _MatrixCase(
        "aws_ready_fixture_only",
        frozenset({_AWS_PROMPT_FIXTURE, _AWS_CONTENT_FIXTURE}),
        _AWS_PROMPT_FIXTURE,
        _AWS_CONTENT_FIXTURE,
    ),
    _MatrixCase(
        "both_with_azure_preference",
        frozenset({_AZURE_PROMPT, _AZURE_CONTENT, _AWS_PROMPT_FIXTURE, _AWS_CONTENT_FIXTURE}),
        _AZURE_PROMPT,
        _AZURE_CONTENT,
    ),
    _MatrixCase("neither_control_implementation", frozenset(), None, None),
)


class _MatrixCatalog(CatalogService):
    def __init__(self) -> None:
        full = create_catalog_service()
        summaries = (
            {PluginId("source", item.name): item for item in full.list_sources()}
            | {PluginId("transform", item.name): item for item in full.list_transforms()}
            | {PluginId("sink", item.name): item for item in full.list_sinks()}
        )
        schemas = {plugin_id: full.get_schema(plugin_id.kind, plugin_id.name) for plugin_id in summaries}
        for fixture_id, template_id in (
            (_AWS_PROMPT_FIXTURE, _AZURE_PROMPT),
            (_AWS_CONTENT_FIXTURE, _AZURE_CONTENT),
        ):
            summaries[fixture_id] = summaries[template_id].model_copy(update={"name": fixture_id.name})
            schemas[fixture_id] = schemas[template_id].model_copy(update={"name": fixture_id.name})
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


def _snapshot(case: _MatrixCase) -> PluginAvailabilitySnapshot:
    llm_id = PluginId("transform", "llm")
    return PluginAvailabilitySnapshot.create(
        policy_hash="1" * 64,
        principal_scope=f"local:{case.name}",
        available=_CORE_IDS | case.extra_available,
        unavailable=(),
        selected=(
            (PluginCapability.LLM, llm_id),
            (PluginCapability.PROMPT_SHIELD, case.selected_prompt),
            (PluginCapability.CONTENT_SAFETY, case.selected_content),
        ),
        usable_profile_aliases=((llm_id, ("tutorial",)),),
        selected_profile_aliases=((llm_id, "tutorial"),),
        control_modes=(
            (PluginCapability.PROMPT_SHIELD, ControlMode.RECOMMEND),
            (PluginCapability.CONTENT_SAFETY, ControlMode.RECOMMEND),
        ),
        binding_generation_fingerprint="2" * 64,
    )


def _view(snapshot: PluginAvailabilitySnapshot) -> PolicyCatalogView:
    profiles = MagicMock(spec=OperatorProfileRegistry)
    profiles.public_schema.side_effect = lambda _plugin_id, full_schema, **_kwargs: full_schema
    return PolicyCatalogView(_MatrixCatalog(), snapshot, profiles)


def _visible_ids(view: PolicyCatalogView) -> frozenset[str]:
    return frozenset(
        {
            *(f"source:{item.name}" for item in view.list_sources()),
            *(f"transform:{item.name}" for item in view.list_transforms()),
            *(f"sink:{item.name}" for item in view.list_sinks()),
        }
    )


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_policy_surface_parity_matrix(case: _MatrixCase) -> None:
    snapshot = _snapshot(case)
    view = _view(snapshot)
    expected = frozenset(map(str, snapshot.available))
    empty_state = CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)

    catalog_api = _visible_ids(view)
    ui_catalog = frozenset(json.loads(json.dumps(sorted(catalog_api))))
    guided_discovery = _visible_ids(view)
    prompt = build_context_string(empty_state, view, plugin_snapshot=snapshot, schemas_loaded=frozenset())
    freeform_prompt = frozenset(json.loads(prompt.partition("\n")[2])["plugin_policy"]["available_ids"])
    evidence = _build_web_plugin_policy_evidence(snapshot=snapshot, policy=None)

    assert catalog_api == ui_catalog == guided_discovery == freeform_prompt == expected
    assert frozenset(evidence.available_plugin_ids) == expected
    assert (
        validate_plugin_policy(
            empty_state,
            snapshot=snapshot,
            profile_registry=MagicMock(spec=OperatorProfileRegistry),
        ).findings
        == ()
    )

    for plugin_id in snapshot.available:
        assert view.get_schema(plugin_id.kind, plugin_id.name).name == plugin_id.name

    for recipe_info in list_recipes(snapshot):
        recipe = get_recipe(recipe_info["name"])
        assert recipe is not None
        assert recipe.required_plugins <= snapshot.available
        assert all(not alternatives.isdisjoint(snapshot.available) for alternatives in recipe.alternative_plugin_groups)


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
        {"options": dict(lowered.executable_options)},
        resolver,
        "alice",
    )
    _server_value, server_ref = server_store.get_secret("SHARED_LLM_KEY")

    assert resolved["options"]["api_key"] == "server-value"
    assert evidence[0].scope == "server"
    assert evidence[0].fingerprint == server_ref.fingerprint
