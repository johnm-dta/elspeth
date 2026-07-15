from __future__ import annotations

import json
from pathlib import Path

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
from elspeth.web.plugin_policy.models import PluginId
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, RuntimeWebPluginConfig

_GOLDEN = Path("tests/golden/web/catalog/policy_view/transform__llm.json")


def test_profiled_llm_policy_schema_matches_golden() -> None:
    settings = WebSettings(
        composer_max_composition_turns=4,
        composer_max_discovery_turns=4,
        composer_timeout_seconds=60,
        composer_rate_limit_per_minute=20,
        secret_key="0123456789abcdef0123456789abcdef",
        shareable_link_signing_key=b"0123456789abcdef0123456789abcdef",
        llm_profiles={
            "tutorial": {
                "provider": "openrouter",
                "model": "openai/gpt-5-mini",
                "credential_scope": "server",
                "credential_ref": "OPENROUTER_API_KEY",
            }
        },
    )
    runtime = RuntimeWebPluginConfig.from_settings(settings)
    policy = compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)
    profiles = OperatorProfileRegistry(policy=policy, settings=runtime)
    public = profiles.public_schema(
        PluginId("transform", "llm"),
        create_catalog_service().get_schema("transform", "llm"),
        available_aliases=("tutorial",),
    )

    assert public.model_dump(mode="json") == json.loads(_GOLDEN.read_text())


def _bedrock_public_schema(plugin_name: str, alias: str) -> dict[str, object]:
    settings = WebSettings(
        composer_max_composition_turns=4,
        composer_max_discovery_turns=4,
        composer_timeout_seconds=60,
        composer_rate_limit_per_minute=20,
        secret_key="0123456789abcdef0123456789abcdef",
        shareable_link_signing_key=b"0123456789abcdef0123456789abcdef",
        plugin_allowlist=(f"transform:{plugin_name}",),
        bedrock_guardrail_profiles=(
            {
                "alias": alias,
                "plugin": plugin_name,
                "guardrail_identifier": "privateguardrail",
                "guardrail_version": "7",
                "region": "us-east-1",
            },
        ),
    )
    runtime = RuntimeWebPluginConfig.from_settings(settings)
    policy = compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)
    profiles = OperatorProfileRegistry(policy=policy, settings=runtime)
    public = profiles.public_schema(
        PluginId("transform", plugin_name),
        create_catalog_service().get_schema("transform", plugin_name),
        available_aliases=(alias,),
    )
    return public.model_dump(mode="json")


def test_profiled_bedrock_prompt_shield_policy_schema_matches_golden() -> None:
    actual = _bedrock_public_schema("aws_bedrock_prompt_shield", "prompt-default")
    golden = Path("tests/golden/web/catalog/policy_view/transform__aws_bedrock_prompt_shield.json")

    assert actual == json.loads(golden.read_text())


def test_profiled_bedrock_content_safety_policy_schema_matches_golden() -> None:
    actual = _bedrock_public_schema("aws_bedrock_content_safety", "content-default")
    golden = Path("tests/golden/web/catalog/policy_view/transform__aws_bedrock_content_safety.json")

    assert actual == json.loads(golden.read_text())
