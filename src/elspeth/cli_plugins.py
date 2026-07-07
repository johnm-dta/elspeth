"""Typer-free helpers for plugin catalog CLI commands."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Required, TypedDict, cast

from elspeth.contracts import SinkProtocol, SourceProtocol, TransformProtocol
from elspeth.plugins.infrastructure.discovery import get_plugin_description
from elspeth.plugins.infrastructure.manager import PluginManager, get_shared_plugin_manager
from elspeth.web.catalog.schemas import PluginKind, PluginSummary
from elspeth.web.catalog.service import CatalogServiceImpl

VALID_PLUGIN_TYPES: tuple[PluginKind, ...] = ("source", "transform", "sink")
PluginClass = type[SourceProtocol] | type[TransformProtocol] | type[SinkProtocol]


class ConfigFieldPayload(TypedDict, total=False):
    """JSON-ready view of one plugin config field."""

    name: Required[str]
    type: Required[str]
    required: Required[bool]
    description: str | None
    default: object | None


class SecretRequirementPayload(TypedDict):
    """JSON-ready view of a plugin secret requirement."""

    field: str
    candidates: list[str]


class PluginSummaryPayload(TypedDict, total=False):
    """JSON-ready plugin summary used by ``plugins list``."""

    name: Required[str]
    description: Required[str]
    plugin_type: Required[PluginKind]
    config_fields: Required[list[ConfigFieldPayload]]
    usage_when_to_use: str | None
    usage_when_not_to_use: str | None
    example_use: str | None
    capability_tags: list[str]
    audit_characteristics: list[str]
    composer_hints: list[str]
    secret_requirements: list[SecretRequirementPayload]


class PluginInspectPayload(TypedDict, total=False):
    """JSON-ready plugin detail used by ``plugins inspect``."""

    name: Required[str]
    plugin_type: Required[PluginKind]
    description: Required[str]
    json_schema: Required[dict[str, object]]
    knob_schema: Required[dict[str, object]]
    config_fields: Required[list[ConfigFieldPayload]]
    composer_hints: list[str]
    secret_requirements: list[SecretRequirementPayload]


def parse_plugin_kind(value: str) -> PluginKind:
    """Validate a CLI plugin kind string."""
    if value not in VALID_PLUGIN_TYPES:
        valid = ", ".join(VALID_PLUGIN_TYPES)
        raise ValueError(f"Invalid plugin type {value!r}. Valid types: {valid}")
    return value


def build_catalog_service() -> CatalogServiceImpl:
    """Build the shared plugin catalog service."""
    return CatalogServiceImpl(get_shared_plugin_manager())


def list_plugins_payload(plugin_type: PluginKind | None) -> dict[PluginKind, list[PluginSummaryPayload]]:
    """Return JSON-ready plugin summaries grouped by plugin type."""
    service = build_catalog_service()
    kinds = (plugin_type,) if plugin_type is not None else VALID_PLUGIN_TYPES
    return {kind: [_summary_payload(summary) for summary in _list_for_kind(service, kind)] for kind in kinds}


def list_plugins_text_payload(plugin_type: PluginKind | None) -> dict[PluginKind, list[PluginSummaryPayload]]:
    """Return lightweight plugin summaries for human text listing."""
    manager = get_shared_plugin_manager()
    kinds = (plugin_type,) if plugin_type is not None else VALID_PLUGIN_TYPES
    return {kind: [_text_summary_payload(plugin_cls, kind) for plugin_cls in _classes_for_kind(manager, kind)] for kind in kinds}


def inspect_plugin_payload(plugin_type: PluginKind, name: str) -> PluginInspectPayload:
    """Return JSON-ready plugin schema detail plus summary config fields."""
    service = build_catalog_service()
    schema = cast(PluginInspectPayload, service.get_schema(plugin_type, name).model_dump(mode="json"))
    summary = next((item for item in _list_for_kind(service, plugin_type) if item.name == name), None)
    if summary is None:
        raise ValueError(f"Unknown {plugin_type} plugin: {name}")
    schema["config_fields"] = _config_field_payloads(summary)
    return schema


def format_plugins_list_text(payload: Mapping[PluginKind, Sequence[PluginSummaryPayload]]) -> str:
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


def format_plugin_inspect_text(payload: PluginInspectPayload) -> str:
    """Format plugin schema detail for humans."""
    lines = [
        f"{payload['name']} ({payload['plugin_type']})",
        "",
        str(payload["description"]),
        "",
        "Config Fields:",
    ]
    config_fields = payload["config_fields"]
    if config_fields:
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


def _summary_payload(summary: PluginSummary) -> PluginSummaryPayload:
    return cast(PluginSummaryPayload, summary.model_dump(mode="json"))


def _config_field_payloads(summary: PluginSummary) -> list[ConfigFieldPayload]:
    return [cast(ConfigFieldPayload, field.model_dump(mode="json")) for field in summary.config_fields]


def _text_summary_payload(plugin_cls: PluginClass, plugin_type: PluginKind) -> PluginSummaryPayload:
    return {
        "name": plugin_cls.name,
        "description": get_plugin_description(plugin_cls),
        "plugin_type": plugin_type,
        "config_fields": [],
    }


def _classes_for_kind(manager: PluginManager, plugin_type: PluginKind) -> list[PluginClass]:
    if plugin_type == "source":
        return list(cast(Sequence[PluginClass], manager.get_sources()))
    if plugin_type == "transform":
        return list(cast(Sequence[PluginClass], manager.get_transforms()))
    if plugin_type == "sink":
        return list(cast(Sequence[PluginClass], manager.get_sinks()))
    raise AssertionError(f"Unhandled plugin type: {plugin_type}")


def _list_for_kind(service: CatalogServiceImpl, plugin_type: PluginKind) -> list[PluginSummary]:
    if plugin_type == "source":
        return service.list_sources()
    if plugin_type == "transform":
        return service.list_transforms()
    if plugin_type == "sink":
        return service.list_sinks()
    raise AssertionError(f"Unhandled plugin type: {plugin_type}")
