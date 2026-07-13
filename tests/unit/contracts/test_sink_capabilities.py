"""Tests for public sink capability policy contracts."""

from __future__ import annotations

from types import MappingProxyType

from elspeth.contracts.sink import (
    FAILSINK_ELIGIBLE_PLUGIN_TEXT,
    FAILSINK_ELIGIBLE_SINK_PLUGINS,
    FILE_SINK_PLUGIN_SLASH_TEXT,
    FILE_SINK_PLUGIN_TEXT,
    FILE_SINK_PLUGINS,
    FILE_SINK_REPAIR_EXTENSIONS,
    LOCAL_RECOVERY_SINK_PLUGINS,
    SINK_CAPABILITIES_BY_PLUGIN,
    SinkCapabilities,
)
from elspeth.plugins.infrastructure.manager import PluginManager


def test_sink_capability_sets_are_derived_from_registry() -> None:
    expected_file_sinks = frozenset(
        plugin_name for plugin_name, capabilities in SINK_CAPABILITIES_BY_PLUGIN.items() if capabilities.requires_path_option
    )
    expected_failsink_plugins = frozenset(
        plugin_name for plugin_name, capabilities in SINK_CAPABILITIES_BY_PLUGIN.items() if capabilities.eligible_as_failsink
    )
    expected_local_recovery_plugins = frozenset(
        plugin_name for plugin_name, capabilities in SINK_CAPABILITIES_BY_PLUGIN.items() if capabilities.local_recovery_file
    )

    assert expected_file_sinks == FILE_SINK_PLUGINS
    assert expected_failsink_plugins == FAILSINK_ELIGIBLE_SINK_PLUGINS
    assert expected_local_recovery_plugins == LOCAL_RECOVERY_SINK_PLUGINS


def test_sink_capability_policy_is_immutable() -> None:
    assert isinstance(SINK_CAPABILITIES_BY_PLUGIN, MappingProxyType)
    assert isinstance(FILE_SINK_REPAIR_EXTENSIONS, MappingProxyType)
    assert isinstance(SINK_CAPABILITIES_BY_PLUGIN["csv"], SinkCapabilities)


def test_sink_capability_plugins_are_registered_builtin_sinks() -> None:
    manager = PluginManager()
    manager.register_builtin_plugins()

    registered_sinks = {sink.name for sink in manager.get_sinks()}

    assert set(SINK_CAPABILITIES_BY_PLUGIN) <= registered_sinks


def test_sink_capability_message_text_is_shared() -> None:
    assert FAILSINK_ELIGIBLE_PLUGIN_TEXT == "csv or json"
    assert FILE_SINK_PLUGIN_TEXT == "csv, json, or text"
    assert FILE_SINK_PLUGIN_SLASH_TEXT == "csv/json/text"


def test_text_sink_is_file_sink_but_not_lossless_failure_sink() -> None:
    capability = SINK_CAPABILITIES_BY_PLUGIN["text"]
    assert capability.requires_path_option is True
    assert capability.default_file_extension == "txt"
    assert capability.eligible_as_failsink is False
    assert capability.local_recovery_file is False
