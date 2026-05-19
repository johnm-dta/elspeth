"""Integration coverage for discovery-time ``composer_hints``.

Done-bar item 2 for composer-jit-hints Phase 1 (see
``.claude/plans/composer-llm-drifting-hollerith.md``): ``list_sources``,
``list_transforms``, and ``list_sinks`` responses each include the
``composer_hints`` field on every entry where the plugin overrides the
assistance hook, and the field is the empty tuple where it doesn't.

Unlike the per-class unit assertion in
``tests/unit/contracts/test_plugin_assistance_coverage.py``, this test
drives the *catalog* surface — the same code path the web app and the
MCP tool dispatch exercise — with the real ``PluginManager``. It pins
the integration: an LLM calling ``list_sources`` over MCP sees the hints
on the wire, not just via a plugin classmethod we hope is reachable.
"""

from __future__ import annotations

from typing import cast

import pytest

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.schemas import PluginKind, PluginSchemaInfo, PluginSummary
from elspeth.web.catalog.service import CatalogServiceImpl

# Plugins that override ``get_agent_assistance`` with an
# ``issue_code=None`` branch returning composer_hints. Subset of the
# priority-17 set: the names exposed under each plugin family. Kept as
# a tuple of (plugin_type, plugin_name) for parametrize readability.
HINTED_PLUGINS: tuple[tuple[str, str], ...] = (
    # Sources
    ("source", "csv"),
    ("source", "json"),
    ("source", "dataverse"),
    # Transforms
    ("transform", "llm"),
    ("transform", "web_scrape"),
    ("transform", "line_explode"),
    ("transform", "json_explode"),
    ("transform", "field_mapper"),
    ("transform", "value_transform"),
    ("transform", "type_coerce"),
    ("transform", "truncate"),
    ("transform", "azure_content_safety"),
    ("transform", "rag_retrieval"),
    ("transform", "batch_distribution_profile"),
    # Sinks
    ("sink", "json"),
    ("sink", "csv"),
    ("sink", "database"),
)

# Plugin known to NOT override get_agent_assistance. The negative case
# pins the contract: empty tuple on the wire, not a missing field.
UNHINTED_PLUGIN: tuple[str, str] = ("source", "azure_blob")


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


def test_unhinted_plugin_carries_empty_composer_hints(catalog: CatalogServiceImpl) -> None:
    """A plugin without get_agent_assistance override surfaces empty composer_hints, not a missing field."""
    plugin_type, plugin_name = UNHINTED_PLUGIN
    summaries = _summaries_for(catalog, plugin_type)
    assert plugin_name in summaries, f"sanity: {plugin_type}/{plugin_name} not in catalog listing — pick a different unhinted plugin"
    entry = summaries[plugin_name]
    assert entry.composer_hints == (), (
        f"{plugin_type}/{plugin_name} listing carries hints but no override is defined; "
        "the resolver is inheriting hints from a base class — see "
        "CatalogServiceImpl._discovery_composer_hints."
    )

    schema = catalog.get_schema(plugin_type=cast(PluginKind, plugin_type), name=plugin_name)
    assert schema.composer_hints == ()
