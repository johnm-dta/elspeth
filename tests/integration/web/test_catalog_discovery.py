"""Integration coverage for discovery-time ``composer_hints``.

Unlike the per-class unit assertion in
``tests/unit/contracts/test_plugin_assistance_coverage.py``, this test
drives the *catalog* surface — the same code path the web app and the
MCP tool dispatch exercise — with the real ``PluginManager``. It pins
the integration: an LLM calling ``list_sources`` over MCP sees every
builtin plugin's hints on the wire, not just via a plugin classmethod we
hope is reachable.
"""

from __future__ import annotations

from typing import cast

import pytest

from elspeth.plugins.infrastructure.discovery import discover_all_plugins
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.schemas import PluginKind, PluginSchemaInfo, PluginSummary
from elspeth.web.catalog.service import CatalogServiceImpl


def _discovered_builtin_plugin_names() -> tuple[tuple[str, str], ...]:
    discovered = discover_all_plugins()
    return tuple(
        (plugin_type.rstrip("s"), plugin_cls.name) for plugin_type, plugin_classes in discovered.items() for plugin_cls in plugin_classes
    )


HINTED_PLUGINS = _discovered_builtin_plugin_names()


@pytest.fixture(scope="module")
def catalog() -> CatalogServiceImpl:
    return CatalogServiceImpl(get_shared_plugin_manager())


def _summaries_for(catalog: CatalogServiceImpl, plugin_type: str) -> dict[str, PluginSummary]:
    if plugin_type == "source":
        listing = catalog.list_sources()
    elif plugin_type == "transform":
        listing = catalog.list_transforms()
    elif plugin_type == "sink":
        listing = catalog.list_sinks()
    else:
        raise AssertionError(f"unknown plugin_type: {plugin_type}")
    return {entry.name: entry for entry in listing}


@pytest.mark.parametrize(("plugin_type", "plugin_name"), HINTED_PLUGINS)
def test_list_response_carries_composer_hints_for_hinted_plugins(
    catalog: CatalogServiceImpl,
    plugin_type: str,
    plugin_name: str,
) -> None:
    """Discovery listing for a hinted plugin carries a non-empty composer_hints tuple."""
    summaries = _summaries_for(catalog, plugin_type)
    assert plugin_name in summaries, f"{plugin_type}/{plugin_name} not present in catalog listing"
    entry = summaries[plugin_name]
    assert isinstance(entry, PluginSummary)
    assert entry.composer_hints, (
        f"{plugin_type}/{plugin_name} listing entry has empty composer_hints; "
        "the plugin's get_agent_assistance(issue_code=None) override is not "
        "reaching the catalog summary."
    )
    assert isinstance(entry.composer_hints, tuple)
    for hint in entry.composer_hints:
        assert isinstance(hint, str)
        assert hint.strip()


@pytest.mark.parametrize(("plugin_type", "plugin_name"), HINTED_PLUGINS)
def test_get_schema_response_carries_composer_hints_for_hinted_plugins(
    catalog: CatalogServiceImpl,
    plugin_type: str,
    plugin_name: str,
) -> None:
    """The full-schema response (PluginSchemaInfo) carries the same hints as the listing."""
    schema = catalog.get_schema(plugin_type=cast(PluginKind, plugin_type), name=plugin_name)
    assert isinstance(schema, PluginSchemaInfo)
    assert schema.composer_hints, f"{plugin_type}/{plugin_name} get_schema response has empty composer_hints"
    # The hints surfaced from list_* and get_schema MUST be the same
    # tuple — they share the resolver (_discovery_composer_hints). Drift
    # here would mean an LLM that asked for the schema sees a different
    # coaching set than one that read the listing.
    summaries = _summaries_for(catalog, plugin_type)
    assert summaries[plugin_name].composer_hints == schema.composer_hints


def test_catalog_has_no_unhinted_builtin_plugins(catalog: CatalogServiceImpl) -> None:
    """Every discovered builtin plugin surfaces non-empty composer_hints in the catalog listing."""
    missing: list[str] = []
    for plugin_type in ("source", "transform", "sink"):
        for name, summary in _summaries_for(catalog, plugin_type).items():
            if not summary.composer_hints:
                missing.append(f"{plugin_type}/{name}")
    assert missing == []
