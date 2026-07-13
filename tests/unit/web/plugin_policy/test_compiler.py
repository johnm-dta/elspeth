from __future__ import annotations

from dataclasses import replace

import pytest

from elspeth.contracts.plugin_capabilities import CapabilityDeclaration, ControlRole, PluginCapability
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.config import WebSettings
from elspeth.web.plugin_policy.compiler import REQUIRED_WEB_PLUGIN_IDS, compile_web_plugin_policy
from elspeth.web.plugin_policy.models import PluginId
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


def test_default_policy_authorizes_exact_required_core() -> None:
    policy = compile_web_plugin_policy(
        registry=get_shared_plugin_manager(),
        settings=RuntimeWebPluginConfig.from_settings(_settings()),
    )

    assert policy.required == REQUIRED_WEB_PLUGIN_IDS
    assert policy.authorized == REQUIRED_WEB_PLUGIN_IDS
    assert PluginId("sink", "database") not in policy.authorized


def test_allowlist_is_set_like_for_hashing() -> None:
    runtime = RuntimeWebPluginConfig.from_settings(_settings(plugin_allowlist=("sink:database", "transform:azure_prompt_shield")))
    first = compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)
    second = compile_web_plugin_policy(
        registry=get_shared_plugin_manager(),
        settings=replace(runtime, plugin_allowlist=tuple(reversed(runtime.plugin_allowlist))),
    )

    assert first.policy_hash == second.policy_hash


def test_duplicate_allowlist_entry_is_rejected_without_echoing_value() -> None:
    marker = "sink:database"
    runtime = RuntimeWebPluginConfig.from_settings(_settings(plugin_allowlist=(marker, marker)))

    with pytest.raises(ValueError) as exc_info:
        compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)

    assert "duplicate_plugin_id" in str(exc_info.value)
    assert marker not in str(exc_info.value)


def test_uninstalled_allowlist_entry_is_rejected_by_closed_reason() -> None:
    runtime = RuntimeWebPluginConfig.from_settings(_settings(plugin_allowlist=("sink:not_installed",)))

    with pytest.raises(ValueError) as exc_info:
        compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)

    assert str(exc_info.value) == "web plugin policy invalid: plugin_not_installed"


def test_every_authorized_plugin_has_canonical_code_identity() -> None:
    policy = compile_web_plugin_policy(
        registry=get_shared_plugin_manager(),
        settings=RuntimeWebPluginConfig.from_settings(_settings()),
    )

    assert {plugin_id for plugin_id, _, _ in policy.plugin_code_identities} == policy.authorized
    assert all(version != "0.0.0" for _, version, _ in policy.plugin_code_identities)
    assert all(source_hash.startswith("sha256:") and len(source_hash) == 23 for _, _, source_hash in policy.plugin_code_identities)


class _FakeRegistry:
    def __init__(self, *, transforms: list[type]) -> None:
        manager = get_shared_plugin_manager()
        self._sources = manager.get_sources()
        self._transforms = [cls for cls in manager.get_transforms() if cls.name not in {item.name for item in transforms}] + transforms
        self._sinks = manager.get_sinks()

    def get_sources(self) -> list[type]:
        return self._sources

    def get_transforms(self) -> list[type]:
        return self._transforms

    def get_sinks(self) -> list[type]:
        return self._sinks


def _control(name: str, *, available: bool = True) -> type:
    return type(
        name.title().replace("_", ""),
        (),
        {
            "name": name,
            "plugin_version": "1.0.0",
            "source_file_hash": "sha256:0123456789abcdef",
            "policy_capabilities": frozenset(
                {
                    CapabilityDeclaration(
                        PluginCapability.PROMPT_SHIELD,
                        ControlRole.INPUT,
                        blocks_positive_detection=True,
                    )
                }
            ),
            "check_web_local_requirements": classmethod(lambda cls: available),
        },
    )


def test_preference_order_must_cover_every_authorized_implementation() -> None:
    first = _control("first_shield")
    second = _control("second_shield")
    runtime = RuntimeWebPluginConfig.from_settings(
        _settings(
            plugin_allowlist=("transform:first_shield", "transform:second_shield"),
            plugin_preferences={"prompt_shield": ("transform:first_shield",)},
        )
    )

    with pytest.raises(ValueError, match="incomplete_preference_order"):
        compile_web_plugin_policy(registry=_FakeRegistry(transforms=[first, second]), settings=runtime)


def test_explicit_authorization_checks_local_requirement_without_detail_leak() -> None:
    unavailable = _control("optional_shield", available=False)
    runtime = RuntimeWebPluginConfig.from_settings(_settings(plugin_allowlist=("transform:optional_shield",)))

    with pytest.raises(ValueError) as exc_info:
        compile_web_plugin_policy(registry=_FakeRegistry(transforms=[unavailable]), settings=runtime)

    assert str(exc_info.value) == "web plugin policy invalid: plugin_unavailable"
