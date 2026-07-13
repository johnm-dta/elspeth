from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from elspeth.web.config import WebSettings
from elspeth.web.plugin_policy.profiles import RuntimeWebPluginConfig


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
