from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
from elspeth.web.plugin_policy.models import PluginId
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, RuntimeWebPluginConfig


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


def test_openrouter_profile_requires_explicit_scoped_credential() -> None:
    with pytest.raises(ValidationError):
        _settings(llm_profiles={"tutorial": {"provider": "openrouter", "model": "openai/gpt-5-mini"}})


def test_bedrock_profile_is_keyless_and_uses_canonical_provider_registry() -> None:
    settings = _settings(
        llm_profiles={
            "tutorial": {
                "provider": "bedrock",
                "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
                "region_name": "ap-southeast-2",
            }
        },
        tutorial_llm_profile="tutorial",
    )
    runtime = RuntimeWebPluginConfig.from_settings(settings)

    profile = runtime.llm_profiles[0][1]
    assert profile.provider == "bedrock"
    assert profile.credential_scope is None
    assert profile.credential_ref is None
    assert "credential" not in repr(profile)


def test_runtime_conversion_is_frozen_and_canonical() -> None:
    settings = _settings(
        plugin_allowlist=("sink:database",),
        llm_profiles={
            "tutorial": {
                "provider": "openrouter",
                "model": "openai/gpt-5-mini",
                "credential_scope": "server",
                "credential_ref": "TOP_SECRET_MARKER",
            }
        },
    )
    runtime = RuntimeWebPluginConfig.from_settings(settings)

    assert runtime.plugin_allowlist == ("sink:database",)
    assert runtime.llm_profiles[0][0] == "tutorial"
    assert "TOP_SECRET_MARKER" not in repr(runtime)
    with pytest.raises(FrozenInstanceError):
        runtime.tutorial_llm_profile = "changed"  # type: ignore[misc]


def test_profile_reprs_hide_provider_and_provider_specific_settings() -> None:
    settings = _settings(
        llm_profiles={
            "private-binding": {
                "provider": "azure",
                "model": "PRIVATE_MODEL_MARKER",
                "credential_scope": "server",
                "credential_ref": "PRIVATE_CREDENTIAL_MARKER",
                "endpoint": "https://private-endpoint-marker.example.com",
                "deployment_name": "PRIVATE_DEPLOYMENT_MARKER",
                "api_version": "PRIVATE_API_VERSION_MARKER",
                "timeout_seconds": 47.25,
                "max_tokens": 12345,
            }
        }
    )

    settings_repr = repr(settings.llm_profiles["private-binding"])
    runtime_repr = repr(RuntimeWebPluginConfig.from_settings(settings).llm_profiles[0][1])

    for marker in (
        "azure",
        "PRIVATE_MODEL_MARKER",
        "PRIVATE_CREDENTIAL_MARKER",
        "private-endpoint-marker",
        "PRIVATE_DEPLOYMENT_MARKER",
        "PRIVATE_API_VERSION_MARKER",
        "47.25",
        "12345",
    ):
        assert marker not in settings_repr
        assert marker not in runtime_repr


def test_profile_aliases_are_opaque_canonical_identifiers() -> None:
    with pytest.raises(ValidationError):
        _settings(
            llm_profiles={
                "Not Valid": {
                    "provider": "bedrock",
                    "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
                }
            }
        )


def test_runtime_conversion_consumes_every_universal_setting_field() -> None:
    settings_fields = {
        "plugin_allowlist",
        "plugin_preferences",
        "plugin_control_modes",
        "llm_profiles",
        "tutorial_llm_profile",
    }
    runtime_fields = set(RuntimeWebPluginConfig.__dataclass_fields__)

    assert settings_fields == runtime_fields


def _profile_registry() -> OperatorProfileRegistry:
    runtime = RuntimeWebPluginConfig.from_settings(
        _settings(
            llm_profiles={
                "tutorial": {
                    "provider": "openrouter",
                    "model": "openai/gpt-5-mini",
                    "credential_scope": "server",
                    "credential_ref": "OPENROUTER_API_KEY",
                },
                "bedrock-task-role": {
                    "provider": "bedrock",
                    "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
                    "region_name": "ap-southeast-2",
                },
            }
        )
    )
    policy = compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)
    return OperatorProfileRegistry(policy=policy, settings=runtime)


def test_public_llm_schema_exposes_alias_not_private_provider_binding() -> None:
    full = create_catalog_service().get_schema("transform", "llm")
    public = _profile_registry().public_schema(
        PluginId("transform", "llm"),
        full,
        available_aliases=("tutorial",),
    )
    rendered = public.model_dump_json()

    assert '"profile"' in rendered
    assert '"tutorial"' in rendered
    for private_name in ("api_key", "base_url", "endpoint", "deployment_name", "region_name", '"provider"', '"model"'):
        assert private_name not in rendered


def test_profile_lowering_splits_executable_and_audit_safe_options() -> None:
    lowered = _profile_registry().lower_options(
        PluginId("transform", "llm"),
        alias="tutorial",
        safe_options={"prompt_template": "Summarise {{ row }}", "response_field": "summary"},
    )

    assert lowered.executable_options["provider"] == "openrouter"
    assert lowered.executable_options["model"] == "openai/gpt-5-mini"
    assert lowered.executable_options["api_key"] == {
        "secret_ref": "OPENROUTER_API_KEY",
        "secret_scope": "server",
    }
    assert lowered.audit_safe_options == {
        "profile": "tutorial",
        "prompt_template": "Summarise {{ row }}",
        "response_field": "summary",
    }
    assert "OPENROUTER_API_KEY" not in repr(lowered)


def test_bedrock_profile_lowering_is_keyless() -> None:
    lowered = _profile_registry().lower_options(
        PluginId("transform", "llm"),
        alias="bedrock-task-role",
        safe_options={"prompt_template": "{{ row }}"},
    )

    assert lowered.executable_options["provider"] == "bedrock"
    assert lowered.executable_options["region_name"] == "ap-southeast-2"
    assert "api_key" not in lowered.executable_options


def test_profile_lowering_rejects_raw_provider_options() -> None:
    with pytest.raises(ValueError, match="private_profile_option"):
        _profile_registry().lower_options(
            PluginId("transform", "llm"),
            alias="tutorial",
            safe_options={"provider": "bedrock", "prompt_template": "{{ row }}"},
        )
