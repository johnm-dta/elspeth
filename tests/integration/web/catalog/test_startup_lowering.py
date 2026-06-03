from __future__ import annotations

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.service import CatalogServiceImpl


def test_catalog_startup_lowering_covers_registered_plugins() -> None:
    plugin_manager = get_shared_plugin_manager()

    svc = CatalogServiceImpl(plugin_manager)

    expected_count = len(plugin_manager.get_sources()) + len(plugin_manager.get_transforms()) + len(plugin_manager.get_sinks())
    assert len(svc._schema_cache) == expected_count
    for info in svc._schema_cache.values():
        assert "fields" in info.knob_schema
        assert type(info.knob_schema["fields"]) is list
