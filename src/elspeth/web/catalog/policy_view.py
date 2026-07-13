"""Catalog projection constrained by one frozen availability snapshot."""

from __future__ import annotations

from collections.abc import Mapping

from elspeth.contracts.plugin_capabilities import PluginCapability
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginKind, PluginSchemaInfo, PluginSummary
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId, PluginUnavailableReason
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
        self._profiles: OperatorProfileRegistry | None = profiles

    @classmethod
    def for_trained_operator(
        cls,
        full: CatalogService,
        snapshot: PluginAvailabilitySnapshot,
    ) -> PolicyCatalogView:
        """Return the explicit full-catalog projection for the local MCP."""
        if snapshot.principal_scope != "local:trained-operator":
            raise ValueError("trained_operator_snapshot_required")
        view = cls.__new__(cls)
        view._full = full
        view.snapshot = snapshot
        view._profiles = None
        return view

    def _visible(self, kind: PluginKind, items: list[PluginSummary]) -> list[PluginSummary]:
        return [item for item in items if PluginId(kind, item.name) in self.snapshot.available]

    def list_sources(self) -> list[PluginSummary]:
        return self._visible("source", self._full.list_sources())

    def list_transforms(self) -> list[PluginSummary]:
        return self._visible("transform", self._full.list_transforms())

    def list_sinks(self) -> list[PluginSummary]:
        return self._visible("sink", self._full.list_sinks())

    def capability_groups(self) -> dict[PluginCapability, tuple[PluginId, ...]]:
        """Return safe visible plugin IDs grouped by declared capability."""
        groups: dict[PluginCapability, list[PluginId]] = {capability: [] for capability in PluginCapability}
        visible = (
            *((PluginId("source", item.name), item) for item in self.list_sources()),
            *((PluginId("transform", item.name), item) for item in self.list_transforms()),
            *((PluginId("sink", item.name), item) for item in self.list_sinks()),
        )
        for plugin_id, summary in visible:
            for declaration in summary.policy_capabilities:
                groups[declaration.capability].append(plugin_id)
        return {capability: tuple(sorted(plugin_ids)) for capability, plugin_ids in groups.items() if plugin_ids}

    def _require_available(self, plugin_id: PluginId) -> None:
        if plugin_id not in self.snapshot.available:
            raise ValueError("plugin_not_enabled")

    def unavailable_reason(self, plugin_id: PluginId) -> PluginUnavailableReason | None:
        """Return the closed policy reason for an identity, or ``None``."""
        if plugin_id in self.snapshot.available:
            return None
        unavailable = {item.plugin_id: item.reason for item in self.snapshot.unavailable}
        if plugin_id in unavailable:
            return unavailable[plugin_id]
        installed = {
            *(PluginId("source", item.name) for item in self._full.list_sources()),
            *(PluginId("transform", item.name) for item in self._full.list_transforms()),
            *(PluginId("sink", item.name) for item in self._full.list_sinks()),
        }
        if plugin_id not in installed:
            return PluginUnavailableReason.NOT_INSTALLED
        return PluginUnavailableReason.NOT_AUTHORIZED

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        plugin_id = PluginId(plugin_type, name)
        self._require_available(plugin_id)
        aliases = dict(self.snapshot.usable_profile_aliases).get(plugin_id, ())
        if self._profiles is None:
            return self._full.get_schema(plugin_type, name)
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
