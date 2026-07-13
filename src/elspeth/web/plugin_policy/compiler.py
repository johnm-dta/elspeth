"""Local-only compiler for universal web plugin authorization."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import NoReturn, Protocol, cast

from elspeth.contracts.plugin_capabilities import CapabilityDeclaration, ControlMode, PluginCapability
from elspeth.web.plugin_policy.models import PluginId, WebPluginPolicy
from elspeth.web.plugin_policy.profiles import RuntimeWebPluginConfig

_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?\Z")
_SOURCE_HASH = re.compile(r"sha256:[0-9a-f]{16}\Z")

REQUIRED_WEB_PLUGIN_IDS = frozenset(
    {
        PluginId("source", "csv"),
        PluginId("source", "json"),
        PluginId("source", "text"),
        PluginId("sink", "csv"),
        PluginId("sink", "json"),
        PluginId("sink", "text"),
        PluginId("transform", "field_mapper"),
        PluginId("transform", "llm"),
        PluginId("transform", "web_scrape"),
    }
)


class _PluginClass(Protocol):
    name: str
    plugin_version: str
    source_file_hash: str | None


class PluginRegistry(Protocol):
    def get_sources(self) -> Sequence[type[_PluginClass]]: ...
    def get_transforms(self) -> Sequence[type[_PluginClass]]: ...
    def get_sinks(self) -> Sequence[type[_PluginClass]]: ...


def _fail(reason: str) -> NoReturn:
    raise ValueError(f"web plugin policy invalid: {reason}")


def _registry_map(registry: PluginRegistry) -> dict[PluginId, type[_PluginClass]]:
    return {
        **{PluginId("source", cls.name): cls for cls in registry.get_sources()},
        **{PluginId("transform", cls.name): cls for cls in registry.get_transforms()},
        **{PluginId("sink", cls.name): cls for cls in registry.get_sinks()},
    }


def _parse_unique(raw_values: Iterable[str]) -> tuple[PluginId, ...]:
    parsed: list[PluginId] = []
    seen: set[PluginId] = set()
    for raw in raw_values:
        try:
            plugin_id = PluginId.parse(raw)
        except ValueError:
            _fail("invalid_plugin_id")
        if plugin_id in seen:
            _fail("duplicate_plugin_id")
        seen.add(plugin_id)
        parsed.append(plugin_id)
    return tuple(parsed)


def _validate_identity(plugin_cls: type[_PluginClass]) -> tuple[str, str]:
    version = getattr(plugin_cls, "plugin_version", None)
    source_hash = getattr(plugin_cls, "source_file_hash", None)
    if not isinstance(version, str) or version == "0.0.0" or _VERSION.fullmatch(version) is None:
        _fail("invalid_plugin_version")
    if not isinstance(source_hash, str) or _SOURCE_HASH.fullmatch(source_hash) is None:
        _fail("invalid_plugin_source_hash")
    return version, source_hash


def compile_web_plugin_policy(*, registry: PluginRegistry, settings: RuntimeWebPluginConfig) -> WebPluginPolicy:
    """Compile settings against the complete installed registry without I/O."""
    installed = _registry_map(registry)
    allowlist = _parse_unique(settings.plugin_allowlist)
    optional = frozenset(allowlist)
    authorized = REQUIRED_WEB_PLUGIN_IDS | optional
    if not authorized <= installed.keys():
        _fail("plugin_not_installed")

    identities = tuple((plugin_id, *_validate_identity(installed[plugin_id])) for plugin_id in sorted(authorized))
    implementations: dict[PluginCapability, set[PluginId]] = {capability: set() for capability in PluginCapability}
    for plugin_id in sorted(authorized):
        plugin_cls = installed[plugin_id]
        local_check = getattr(plugin_cls, "check_web_local_requirements", None)
        if local_check is not None and not local_check():
            _fail("plugin_unavailable")
        raw_declarations: object = getattr(plugin_cls, "policy_capabilities", frozenset())
        declarations = cast("frozenset[CapabilityDeclaration]", raw_declarations)
        if not isinstance(declarations, frozenset) or any(not isinstance(item, CapabilityDeclaration) for item in declarations):
            _fail("invalid_capability_declaration")
        for declaration in declarations:
            implementations[declaration.capability].add(plugin_id)

    preferences: list[tuple[PluginCapability, tuple[PluginId, ...]]] = []
    for capability, raw_order in settings.plugin_preferences:
        ordered = _parse_unique(raw_order)
        if any(plugin_id not in authorized for plugin_id in ordered):
            _fail("preference_not_authorized")
        if any(plugin_id not in implementations[capability] for plugin_id in ordered):
            _fail("capability_mismatch")
        if set(ordered) != implementations[capability]:
            _fail("incomplete_preference_order")
        preferences.append((capability, ordered))

    modes = settings.plugin_control_modes
    preference_caps = {capability for capability, _ in preferences}
    for capability, mode in modes:
        if mode is ControlMode.REQUIRED and not implementations[capability]:
            _fail("required_control_unconfigured")
        if len(implementations[capability]) > 1 and capability not in preference_caps:
            _fail("incomplete_preference_order")

    return WebPluginPolicy.create(
        required=REQUIRED_WEB_PLUGIN_IDS,
        configured_optional=optional,
        preferences=tuple(preferences),
        control_modes=modes,
        plugin_code_identities=identities,
    )
