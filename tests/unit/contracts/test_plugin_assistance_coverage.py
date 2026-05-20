"""Coverage assertion for builtin plugin discovery-time assistance.

Every builtin plugin that reaches the catalog/composer surface should publish
discovery-time hints through ``get_agent_assistance(issue_code=None)``. The
composer uses these hints as just-in-time guardrails; an empty tuple means the
LLM is left to infer plugin-specific contracts from schema names alone.

The hint content is advisory, but the presence, shape, and authoring discipline
are contract: every discovered plugin gets non-empty short hints and a summary.
"""

from __future__ import annotations

import pytest

from elspeth.plugins.infrastructure.discovery import discover_all_plugins


def _discovered_builtin_plugins() -> list[tuple[str, type]]:
    discovered = discover_all_plugins()
    return [(plugin_type.rstrip("s"), plugin_cls) for plugin_type, plugin_classes in discovered.items() for plugin_cls in plugin_classes]


@pytest.mark.parametrize(("plugin_type", "plugin_cls"), _discovered_builtin_plugins(), ids=lambda value: getattr(value, "name", str(value)))
def test_builtin_plugin_publishes_discovery_hints(plugin_type: str, plugin_cls: type) -> None:
    """Every discovered builtin plugin returns a non-empty composer_hints tuple."""
    assistance = plugin_cls.get_agent_assistance(issue_code=None)
    label = f"{plugin_type}/{plugin_cls.name}"
    assert assistance is not None, f"{label}.get_agent_assistance(issue_code=None) returned None"
    assert assistance.composer_hints, f"{label}.composer_hints is empty"
    assert isinstance(assistance.composer_hints, tuple)
    # Authoring discipline: each hint is a non-empty short imperative.
    for hint in assistance.composer_hints:
        assert isinstance(hint, str), f"{label}: composer_hint is not a string: {hint!r}"
        assert hint.strip(), f"{label}: composer_hint is blank"
        assert len(hint) <= 280, f"{label}: composer_hint exceeds 280 chars: {hint!r}"


@pytest.mark.parametrize(("plugin_type", "plugin_cls"), _discovered_builtin_plugins(), ids=lambda value: getattr(value, "name", str(value)))
def test_builtin_plugin_publishes_summary(plugin_type: str, plugin_cls: type) -> None:
    """Every discovered builtin plugin returns a non-empty summary."""
    assistance = plugin_cls.get_agent_assistance(issue_code=None)
    label = f"{plugin_type}/{plugin_cls.name}"
    assert assistance is not None
    assert assistance.summary, f"{label}.summary is empty"
    assert isinstance(assistance.summary, str)
