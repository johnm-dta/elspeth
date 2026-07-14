"""End-to-end policy integration for Bedrock Guardrail transforms."""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from elspeth.contracts.plugin_capabilities import ControlMode, PluginCapability
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.prompts import build_context_string
from elspeth.web.composer.state import CompositionState, NodeSpec, OutputSpec, PipelineMetadata, SourceSpec
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.execution.service import _build_web_plugin_policy_evidence
from elspeth.web.plugin_policy.availability import build_plugin_snapshot
from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
from elspeth.web.plugin_policy.models import PluginId, PluginUnavailableReason
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, RuntimeWebPluginConfig
from elspeth.web.plugin_policy.validation import validate_plugin_policy

_ROOT = Path(__file__).resolve().parents[3]
_AWS_PROMPT = PluginId("transform", "aws_bedrock_prompt_shield")
_AWS_CONTENT = PluginId("transform", "aws_bedrock_content_safety")
_AZURE_PROMPT = PluginId("transform", "azure_prompt_shield")
_AZURE_CONTENT = PluginId("transform", "azure_content_safety")

_PRIVATE_MARKERS = (
    "privateguardrailmarker",
    "privatecontentmarker",
    "ap-east-2",
    '"73"',
    '"41"',
)


@dataclass(frozen=True)
class _AllSecretsInventory:
    generation: str = "request-generation"

    def has_server_ref(self, name: str) -> bool:
        del name
        return True

    def has_user_ref(self, principal: str, name: str) -> bool:
        del principal, name
        return True

    def has_ref(self, principal: str, name: str) -> bool:
        del principal, name
        return True

    def server_generation(self, name: str) -> str:
        del name
        return self.generation

    def user_generation(self, principal: str, name: str) -> str:
        del principal, name
        return self.generation


def _profiles() -> tuple[dict[str, str], ...]:
    return (
        {
            "alias": "prompt-default",
            "plugin": "aws_bedrock_prompt_shield",
            "guardrail_identifier": "privateguardrailmarker",
            "guardrail_version": "73",
            "region": "ap-east-2",
        },
        {
            "alias": "content-default",
            "plugin": "aws_bedrock_content_safety",
            "guardrail_identifier": "privatecontentmarker",
            "guardrail_version": "41",
            "region": "ap-east-2",
        },
    )


def _settings(**overrides: object) -> WebSettings:
    values: dict[str, object] = {
        "composer_max_composition_turns": 4,
        "composer_max_discovery_turns": 4,
        "composer_timeout_seconds": 60,
        "composer_rate_limit_per_minute": 20,
        "shareable_link_signing_key": b"0123456789abcdef0123456789abcdef",
    }
    values.update(overrides)
    return WebSettings.model_validate(values)


def _guardrail_settings(**overrides: object) -> WebSettings:
    values: dict[str, object] = {
        "plugin_allowlist": (str(_AWS_PROMPT), str(_AWS_CONTENT)),
        "bedrock_guardrail_profiles": _profiles(),
        "llm_profiles": {
            "llm-default": {
                "provider": "bedrock",
                "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
                "region_name": "us-east-1",
            }
        },
    }
    values.update(overrides)
    return _settings(**values)


def _policy_context(settings: WebSettings):
    runtime = RuntimeWebPluginConfig.from_settings(settings)
    policy = compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)
    profiles = OperatorProfileRegistry(policy=policy, settings=runtime)
    snapshot = build_plugin_snapshot(
        policy=policy,
        catalog=create_catalog_service(),
        profiles=profiles,
        principal_scope="local:alice",
        secret_inventory=_AllSecretsInventory(),
        generation_key=b"bedrock-policy-integration-generation-key",
    )
    return policy, profiles, snapshot


@pytest.mark.parametrize(
    ("settings", "expected_reason", "available"),
    [
        (_settings(bedrock_guardrail_profiles=(_profiles()[0],)), None, False),
        (
            _settings(plugin_allowlist=(str(_AWS_PROMPT),)),
            PluginUnavailableReason.PROFILE_UNAVAILABLE,
            False,
        ),
        (
            _settings(
                plugin_allowlist=(str(_AWS_PROMPT),),
                bedrock_guardrail_profiles=(_profiles()[0],),
            ),
            None,
            True,
        ),
    ],
    ids=("not-allowlisted", "allowlisted-without-profile", "one-profile"),
)
def test_allowlist_and_profile_availability_matrix(
    settings: WebSettings,
    expected_reason: PluginUnavailableReason | None,
    available: bool,
) -> None:
    _policy, _profiles_registry, snapshot = _policy_context(settings)

    assert (_AWS_PROMPT in snapshot.available) is available
    unavailable = {item.plugin_id: item.reason for item in snapshot.unavailable}
    if expected_reason is None:
        assert _AWS_PROMPT not in unavailable
    else:
        assert unavailable[_AWS_PROMPT] is expected_reason


def test_multiple_profiles_require_and_honor_an_explicit_default() -> None:
    profiles = (
        _profiles()[0],
        {**_profiles()[0], "alias": "prompt-secondary", "guardrail_version": "74"},
    )
    with pytest.raises(ValidationError, match="explicit plugin default"):
        _settings(
            plugin_allowlist=(str(_AWS_PROMPT),),
            bedrock_guardrail_profiles=profiles,
        )

    settings = _settings(
        plugin_allowlist=(str(_AWS_PROMPT),),
        bedrock_guardrail_profiles=profiles,
        bedrock_guardrail_default_profiles={str(_AWS_PROMPT).removeprefix("transform:"): "prompt-secondary"},
    )
    _policy, _profiles_registry, snapshot = _policy_context(settings)

    assert dict(snapshot.usable_profile_aliases)[_AWS_PROMPT] == ("prompt-secondary", "prompt-default")
    assert dict(snapshot.selected_profile_aliases)[_AWS_PROMPT] == "prompt-secondary"


@pytest.mark.parametrize(
    ("name", "settings", "expected_prompt", "expected_content"),
    [
        ("neither", _settings(), None, None),
        (
            "azure-only",
            _settings(plugin_allowlist=(str(_AZURE_PROMPT), str(_AZURE_CONTENT))),
            _AZURE_PROMPT,
            _AZURE_CONTENT,
        ),
        ("aws-only", _guardrail_settings(), _AWS_PROMPT, _AWS_CONTENT),
        (
            "both-with-aws-preference",
            _guardrail_settings(
                plugin_allowlist=(str(_AZURE_PROMPT), str(_AZURE_CONTENT), str(_AWS_PROMPT), str(_AWS_CONTENT)),
                plugin_preferences={
                    "prompt_shield": (str(_AWS_PROMPT), str(_AZURE_PROMPT)),
                    "content_safety": (str(_AWS_CONTENT), str(_AZURE_CONTENT)),
                },
            ),
            _AWS_PROMPT,
            _AWS_CONTENT,
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_implementation_selection_matrix(
    name: str,
    settings: WebSettings,
    expected_prompt: PluginId | None,
    expected_content: PluginId | None,
) -> None:
    del name
    _policy, _profiles_registry, snapshot = _policy_context(settings)
    selected = dict(snapshot.selected)

    assert selected[PluginCapability.PROMPT_SHIELD] == expected_prompt
    assert selected[PluginCapability.CONTENT_SAFETY] == expected_content


def _guarded_state() -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="prompt_raw",
            options={"path": "rows.csv", "schema": {"mode": "observed"}},
            on_validation_failure="discard",
        ),
        nodes=(
            NodeSpec(
                id="prompt_shield",
                node_type="transform",
                plugin=_AWS_PROMPT.name,
                input="prompt_raw",
                on_success="llm_in",
                on_error="discard",
                options={"profile": "prompt-default", "fields": ["prompt"], "schema": {"mode": "observed", "fields": None}},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="llm",
                node_type="transform",
                plugin="llm",
                input="llm_in",
                on_success="content_raw",
                on_error="discard",
                options={
                    "profile": "llm-default",
                    "prompt_template": "{{ row['prompt'] }}",
                    "schema": {"mode": "observed", "fields": None},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="content_safety",
                node_type="transform",
                plugin=_AWS_CONTENT.name,
                input="content_raw",
                on_success="output",
                on_error="discard",
                options={
                    "profile": "content-default",
                    "fields": ["response"],
                    "source": "OUTPUT",
                    "schema": {"mode": "observed", "fields": None},
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
                name="output",
                plugin="json",
                options={"path": "out.jsonl", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


@pytest.mark.parametrize("mode", [ControlMode.RECOMMEND, ControlMode.REQUIRED])
def test_web_runtime_lowering_and_control_modes_use_generic_policy(mode: ControlMode) -> None:
    settings = _guardrail_settings(
        plugin_control_modes={"prompt_shield": mode.value, "content_safety": mode.value},
    )
    _policy, profiles, snapshot = _policy_context(settings)
    state = _guarded_state()

    result = validate_plugin_policy(state, snapshot=snapshot, profile_registry=profiles)

    assert result.findings == ()
    assert dict(snapshot.control_modes)[PluginCapability.PROMPT_SHIELD] is mode
    assert dict(snapshot.control_modes)[PluginCapability.CONTENT_SAFETY] is mode
    assert state.nodes[0].options["profile"] == "prompt-default"
    assert "guardrail_identifier" not in state.nodes[0].options
    assert result.executable_state.nodes[0].options["guardrail_identifier"] == "privateguardrailmarker"
    assert result.executable_state.nodes[2].options["guardrail_identifier"] == "privatecontentmarker"
    for node in (result.executable_state.nodes[0], result.executable_state.nodes[2]):
        assert not {"access_key", "secret_key", "session_token", "endpoint_url"} & set(node.options)


def test_private_bindings_never_enter_authored_prompt_snapshot_or_policy_evidence() -> None:
    policy, profiles, snapshot = _policy_context(_guardrail_settings())
    state = _guarded_state()
    view = PolicyCatalogView(create_catalog_service(), snapshot, profiles)
    prompt = build_context_string(state, view, plugin_snapshot=snapshot, schemas_loaded=frozenset())
    evidence = _build_web_plugin_policy_evidence(snapshot=snapshot, policy=policy)
    surfaces = {
        "authored_state": json.dumps(state.to_dict(), sort_keys=True),
        "authored_nodes": json.dumps(state.to_dict()["nodes"], sort_keys=True),
        "prompt": prompt,
        "snapshot": repr(snapshot),
        "policy": repr(policy),
        "landscape_policy_evidence": json.dumps(asdict(evidence), sort_keys=True),
    }

    assert dict(evidence.selected_profile_aliases)[str(_AWS_PROMPT)] == "prompt-default"
    assert dict(evidence.selected_profile_aliases)[str(_AWS_CONTENT)] == "content-default"
    for surface, rendered in surfaces.items():
        for marker in _PRIVATE_MARKERS:
            assert marker not in rendered, surface


def test_generic_policy_modules_have_no_aws_specific_parallel_mechanism() -> None:
    forbidden_settings = {
        "aws_bedrock_prompt_shield_enabled",
        "aws_bedrock_content_safety_enabled",
        "aws_bedrock_prompt_shield_preference",
        "aws_bedrock_content_safety_preference",
    }
    web_config = (_ROOT / "src/elspeth/web/config.py").read_text()
    assert all(setting not in web_config for setting in forbidden_settings)

    generic_modules = (
        _ROOT / "src/elspeth/web/composer/prompts.py",
        _ROOT / "src/elspeth/web/plugin_policy/availability.py",
        _ROOT / "src/elspeth/web/plugin_policy/coverage.py",
    )
    for path in generic_modules:
        source = path.read_text()
        assert "elspeth.plugins.transforms.aws" not in source
        assert "startup_probe" not in source

    duplicate_types: list[str] = []
    for path in (_ROOT / "src/elspeth/web").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith("AWS") and ("Snapshot" in node.name or "Inventory" in node.name):
                duplicate_types.append(f"{path.relative_to(_ROOT)}:{node.lineno}:{node.name}")
    assert duplicate_types == []

    availability_source = inspect.getsource(build_plugin_snapshot)
    assert "aws_bedrock" not in availability_source
