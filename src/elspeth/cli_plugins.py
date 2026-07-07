"""Typer-free helpers for plugin catalog CLI commands."""

from __future__ import annotations

import json
from typing import Any

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.schemas import PluginKind, PluginSummary
from elspeth.web.catalog.service import CatalogServiceImpl

VALID_PLUGIN_TYPES: tuple[PluginKind, ...] = ("source", "transform", "sink")


def parse_plugin_kind(value: str) -> PluginKind:
    """Validate a CLI plugin kind string."""
    if value not in VALID_PLUGIN_TYPES:
        valid = ", ".join(VALID_PLUGIN_TYPES)
        raise ValueError(f"Invalid plugin type {value!r}. Valid types: {valid}")
    return value


def build_catalog_service() -> CatalogServiceImpl:
    """Build the shared plugin catalog service."""
    return CatalogServiceImpl(get_shared_plugin_manager())


def list_plugins_payload(plugin_type: PluginKind | None) -> dict[str, list[dict[str, Any]]]:
    """Return JSON-ready plugin summaries grouped by plugin type."""
    service = build_catalog_service()
    kinds = (plugin_type,) if plugin_type is not None else VALID_PLUGIN_TYPES
    return {kind: [summary.model_dump(mode="json") for summary in _list_for_kind(service, kind)] for kind in kinds}


def inspect_plugin_payload(plugin_type: PluginKind, name: str) -> dict[str, Any]:
    """Return JSON-ready plugin schema detail plus summary config fields."""
    service = build_catalog_service()
    schema = service.get_schema(plugin_type, name).model_dump(mode="json")
    summary = next((item for item in _list_for_kind(service, plugin_type) if item.name == name), None)
    if summary is None:
        raise ValueError(f"Unknown {plugin_type} plugin: {name}")
    schema["config_fields"] = [field.model_dump(mode="json") for field in summary.config_fields]
    return schema


def format_plugins_list_text(payload: dict[str, list[dict[str, Any]]]) -> str:
    """Format grouped plugin summaries for humans."""
    lines: list[str] = []
    for plugin_type, plugins in payload.items():
        lines.append(f"\n{plugin_type.upper()}S:")
        if plugins:
            for plugin in plugins:
                lines.append(f"  {plugin['name']:20} - {plugin['description']}")
        else:
            lines.append("  (none available)")
    lines.append("")
    return "\n".join(lines)


def format_plugin_inspect_text(payload: dict[str, Any]) -> str:
    """Format plugin schema detail for humans."""
    lines = [
        f"{payload['name']} ({payload['plugin_type']})",
        "",
        str(payload["description"]),
        "",
        "Config Fields:",
    ]
    config_fields = payload["config_fields"]
    if isinstance(config_fields, list) and config_fields:
        for field in config_fields:
            required = "required" if field["required"] else "optional"
            lines.append(f"  {field['name']} ({field['type']}, {required})")
            if field.get("description"):
                lines.append(f"    {field['description']}")
    else:
        lines.append("  (none)")

    lines.extend(
        [
            "",
            "JSON Schema:",
            json.dumps(payload["json_schema"], indent=2, sort_keys=True),
            "",
            "Knob Schema:",
            json.dumps(payload["knob_schema"], indent=2, sort_keys=True),
            "",
        ]
    )
    return "\n".join(lines)


def _list_for_kind(service: CatalogServiceImpl, plugin_type: PluginKind) -> list[PluginSummary]:
    if plugin_type == "source":
        return service.list_sources()
    if plugin_type == "transform":
        return service.list_transforms()
    if plugin_type == "sink":
        return service.list_sinks()
    raise AssertionError(f"Unhandled plugin type: {plugin_type}")
