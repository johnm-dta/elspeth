from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.service import CatalogServiceImpl

_SNAPSHOT_DIR = Path(__file__).parents[3] / "golden" / "web" / "catalog" / "knob_schema"


def _snapshot_name(plugin_kind: str, plugin_name: str) -> str:
    return f"{plugin_kind}__{plugin_name}.json"


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def test_live_catalog_knob_schema_matches_golden_snapshots() -> None:
    svc = CatalogServiceImpl(get_shared_plugin_manager())
    actual_files: set[str] = set()

    for plugin_kind, plugin_name in sorted(svc._schema_cache):
        info = svc._schema_cache[(plugin_kind, plugin_name)]
        filename = _snapshot_name(plugin_kind, plugin_name)
        actual_files.add(filename)
        expected_path = _SNAPSHOT_DIR / filename
        expected = expected_path.read_text(encoding="utf-8")
        actual = _stable_json(
            {
                "plugin_kind": plugin_kind,
                "plugin_name": plugin_name,
                "knob_schema": info.knob_schema,
            }
        )
        assert actual == expected

    expected_files = {path.name for path in _SNAPSHOT_DIR.glob("*.json")}
    assert actual_files == expected_files
