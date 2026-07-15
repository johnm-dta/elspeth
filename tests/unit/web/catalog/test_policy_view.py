from __future__ import annotations

from dataclasses import dataclass

import pytest

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.availability import build_plugin_snapshot
from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, RuntimeWebPluginConfig


@dataclass
class _Inventory:
    def has_server_ref(self, name: str) -> bool:
        return False

    def has_user_ref(self, principal: str, name: str) -> bool:
        return False

    def has_ref(self, principal: str, name: str) -> bool:
        return False


@pytest.fixture
def view() -> PolicyCatalogView:
    settings = WebSettings(
        composer_max_composition_turns=4,
        composer_max_discovery_turns=4,
        composer_timeout_seconds=60,
        composer_rate_limit_per_minute=20,
        shareable_link_signing_key=b"0123456789abcdef0123456789abcdef",
        llm_profiles={
            "task-role": {
                "provider": "bedrock",
                "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            }
        },
    )
    runtime = RuntimeWebPluginConfig.from_settings(settings)
    policy = compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)
    profiles = OperatorProfileRegistry(policy=policy, settings=runtime)
    snapshot = build_plugin_snapshot(
        policy=policy,
        catalog=create_catalog_service(),
        profiles=profiles,
        principal_scope="local:alice",
        secret_inventory=_Inventory(),
        generation_key=b"policy-view-test-key",
    )
    return PolicyCatalogView(create_catalog_service(), snapshot, profiles)


def test_lists_only_snapshot_available_plugins(view: PolicyCatalogView) -> None:
    assert {item.name for item in view.list_sources()} == {"csv", "json", "text"}
    assert {item.name for item in view.list_sinks()} == {"csv", "json", "text"}
    assert {item.name for item in view.list_transforms()} == {"field_mapper", "llm", "web_scrape"}


def test_public_llm_schema_contains_only_usable_alias(view: PolicyCatalogView) -> None:
    rendered = view.get_schema("transform", "llm").model_dump_json()
    assert '"task-role"' in rendered
    assert '"api_key"' not in rendered


def test_hidden_schema_uses_sanitized_closed_error(view: PolicyCatalogView) -> None:
    with pytest.raises(ValueError) as exc_info:
        view.get_schema("transform", "azure_prompt_shield")

    assert str(exc_info.value) == "plugin_not_enabled"
