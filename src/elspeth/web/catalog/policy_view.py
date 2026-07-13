"""Catalog projection constrained by one frozen availability snapshot."""

from __future__ import annotations

from collections.abc import Mapping

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginKind, PluginSchemaInfo, PluginSummary
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry


class PolicyCatalogView:
    def __init__(
        self,
        full: CatalogService,
        snapshot: PluginAvailabilitySnapshot,
        profiles: OperatorProfileRegistry,
    ) -> None:
        self._full = full
        self.snapshot = snapshot
        self._profiles = profiles

    def _visible(self, kind: PluginKind, items: list[PluginSummary]) -> list[PluginSummary]:
        return [item for item in items if PluginId(kind, item.name) in self.snapshot.available]

    def list_sources(self) -> list[PluginSummary]:
        return self._visible("source", self._full.list_sources())

    def list_transforms(self) -> list[PluginSummary]:
        return self._visible("transform", self._full.list_transforms())

    def list_sinks(self) -> list[PluginSummary]:
        return self._visible("sink", self._full.list_sinks())

    def _require_available(self, plugin_id: PluginId) -> None:
        if plugin_id not in self.snapshot.available:
            raise ValueError("plugin_not_enabled")

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        plugin_id = PluginId(plugin_type, name)
        self._require_available(plugin_id)
        aliases = dict(self.snapshot.usable_profile_aliases).get(plugin_id, ())
        return self._profiles.public_schema(
            plugin_id,
            self._full.get_schema(plugin_type, name),
            available_aliases=aliases,
        )

    def post_call_hints(
        self,
        *,
        plugin_type: PluginKind,
        plugin_name: str,
        tool_name: str,
        config_snapshot: Mapping[str, object],
    ) -> tuple[str, ...]:
        plugin_id = PluginId(plugin_type, plugin_name)
        self._require_available(plugin_id)
        return self._full.post_call_hints(
            plugin_type=plugin_type,
            plugin_name=plugin_name,
            tool_name=tool_name,
            config_snapshot=config_snapshot,
        )
