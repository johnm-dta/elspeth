from __future__ import annotations

from dataclasses import dataclass

import pytest

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.availability import build_plugin_snapshot
from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
from elspeth.web.plugin_policy.models import PluginId
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, RuntimeWebPluginConfig


@dataclass
class _Inventory:
    server: frozenset[str] = frozenset()
    users: dict[str, frozenset[str]] | None = None
    server_generations: dict[str, str] | None = None
    user_generations: dict[tuple[str, str], str] | None = None

    def has_server_ref(self, name: str) -> bool:
        return name in self.server

    def has_user_ref(self, principal: str, name: str) -> bool:
        return name in (self.users or {}).get(principal, frozenset())

    def has_ref(self, principal: str, name: str) -> bool:
        return self.has_user_ref(principal, name) or self.has_server_ref(name)

    def server_generation(self, name: str) -> str | None:
        if self.server_generations is not None:
            return self.server_generations.get(name)
        return "present" if self.has_server_ref(name) else None

    def user_generation(self, principal: str, name: str) -> str | None:
        if self.user_generations is not None:
            return self.user_generations.get((principal, name))
        return "present" if self.has_user_ref(principal, name) else None


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


def _build(settings: WebSettings, *, principal: str = "local:alice", inventory: _Inventory | None = None):
    runtime = RuntimeWebPluginConfig.from_settings(settings)
    manager = get_shared_plugin_manager()
    policy = compile_web_plugin_policy(registry=manager, settings=runtime)
    profiles = OperatorProfileRegistry(policy=policy, settings=runtime)
    return build_plugin_snapshot(
        policy=policy,
        catalog=create_catalog_service(),
        profiles=profiles,
        principal_scope=principal,
        secret_inventory=inventory or _Inventory(),
        generation_key=b"deterministic-test-generation-key",
    )


def test_operator_profiled_llm_is_unavailable_without_usable_alias() -> None:
    snapshot = _build(_settings())

    assert PluginId("transform", "llm") not in snapshot.available
    assert dict(snapshot.usable_profile_aliases)[PluginId("transform", "llm")] == ()


def test_bedrock_profile_is_locally_available_without_secret() -> None:
    snapshot = _build(
        _settings(
            llm_profiles={
                "task-role": {
                    "provider": "bedrock",
                    "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
                }
            }
        )
    )

    assert PluginId("transform", "llm") in snapshot.available
    assert dict(snapshot.selected_profile_aliases)[PluginId("transform", "llm")] == "task-role"


def test_configured_tutorial_profile_is_the_selected_usable_alias() -> None:
    snapshot = _build(
        _settings(
            llm_profiles={
                "alpha": {
                    "provider": "bedrock",
                    "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
                },
                "tutorial": {
                    "provider": "bedrock",
                    "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
                },
            },
            tutorial_llm_profile="tutorial",
        )
    )

    llm_id = PluginId("transform", "llm")
    assert dict(snapshot.usable_profile_aliases)[llm_id] == ("tutorial", "alpha")
    assert dict(snapshot.selected_profile_aliases)[llm_id] == "tutorial"


def test_user_secret_can_narrow_but_never_expand_policy() -> None:
    snapshot = _build(
        _settings(),
        inventory=_Inventory(users={"local:alice": frozenset({"AZURE_CONTENT_SAFETY_KEY"})}),
    )

    assert PluginId("transform", "azure_prompt_shield") not in snapshot.available


def test_profile_aliases_and_hash_are_principal_scoped() -> None:
    settings = _settings(
        llm_profiles={
            "personal": {
                "provider": "openrouter",
                "model": "openai/gpt-5-mini",
                "credential_scope": "user",
                "credential_ref": "OPENROUTER_API_KEY",
            }
        }
    )
    inventory = _Inventory(users={"local:alice": frozenset({"OPENROUTER_API_KEY"})})
    alice = _build(settings, principal="local:alice", inventory=inventory)
    bob = _build(settings, principal="local:bob", inventory=inventory)

    assert dict(alice.usable_profile_aliases)[PluginId("transform", "llm")] == ("personal",)
    assert dict(bob.usable_profile_aliases)[PluginId("transform", "llm")] == ()
    assert alice.snapshot_hash != bob.snapshot_hash
    assert "OPENROUTER_API_KEY" not in alice.binding_generation_fingerprint


@pytest.mark.parametrize("scope", ["user", "server"])
def test_in_place_profile_credential_rotation_changes_snapshot_identity(scope: str) -> None:
    settings = _settings(
        llm_profiles={
            "rotating": {
                "provider": "openrouter",
                "model": "openai/gpt-5-mini",
                "credential_scope": scope,
                "credential_ref": "OPENROUTER_API_KEY",
            }
        }
    )
    principal = "local:alice"
    availability = {"OPENROUTER_API_KEY"}
    if scope == "user":
        before_inventory = _Inventory(
            users={principal: frozenset(availability)},
            user_generations={(principal, "OPENROUTER_API_KEY"): "generation-one"},
        )
        after_inventory = _Inventory(
            users={principal: frozenset(availability)},
            user_generations={(principal, "OPENROUTER_API_KEY"): "generation-two"},
        )
    else:
        before_inventory = _Inventory(
            server=frozenset(availability),
            server_generations={"OPENROUTER_API_KEY": "generation-one"},
        )
        after_inventory = _Inventory(
            server=frozenset(availability),
            server_generations={"OPENROUTER_API_KEY": "generation-two"},
        )

    before = _build(settings, principal=principal, inventory=before_inventory)
    after = _build(settings, principal=principal, inventory=after_inventory)

    assert before.available == after.available
    assert before.usable_profile_aliases == after.usable_profile_aliases
    assert before.binding_generation_fingerprint != after.binding_generation_fingerprint
    assert before.snapshot_hash != after.snapshot_hash
    assert "generation-one" not in before.binding_generation_fingerprint
    assert "generation-two" not in after.binding_generation_fingerprint
