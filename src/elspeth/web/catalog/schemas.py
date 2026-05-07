"""Catalog API response models.

All schemas in this module are Tier 1 responses describing system-owned
plugin metadata.  They inherit from ``_StrictResponse`` so that any
backend emission of a wrong type (or an extra field a frontend feature
flag would later quietly read) crashes at construction time instead of
reaching the UI.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

PluginKind = Literal["source", "transform", "sink"]


class _StrictResponse(BaseModel):
    """Tier 1 base for catalog responses — no coercion, no extras."""

    model_config = ConfigDict(strict=True, extra="forbid")


class ConfigFieldSummary(_StrictResponse):
    """Summary of a single field in a plugin's config model."""

    name: str
    type: str
    required: bool
    description: str | None = None
    default: Any | None = None


class PluginSummary(_StrictResponse):
    """Lightweight plugin info for catalog browsing."""

    name: str
    description: str
    plugin_type: PluginKind
    config_fields: list[ConfigFieldSummary]


class PluginSchemaInfo(_StrictResponse):
    """Full plugin schema detail for the composer.

    ``json_schema`` contains the raw output of ``ConfigModel.model_json_schema()``.
    It is ``{}`` (empty dict) when the plugin has no configuration model
    (e.g., the ``null`` source).
    """

    name: str
    plugin_type: PluginKind
    description: str
    json_schema: dict[str, Any]
