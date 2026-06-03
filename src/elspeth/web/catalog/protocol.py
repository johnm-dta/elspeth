"""CatalogService protocol — internal service boundary for plugin catalog."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol, runtime_checkable

from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary

PluginKind = Literal["source", "transform", "sink"]


@runtime_checkable
class CatalogService(Protocol):
    """Read-only plugin catalog.

    Five methods, all synchronous (plugin discovery is CPU-bound).
    When the Catalog module is later extracted to a microservice,
    this protocol stays and the implementation becomes an HTTP client.
    """

    def list_sources(self) -> list[PluginSummary]: ...

    def list_transforms(self) -> list[PluginSummary]: ...

    def list_sinks(self) -> list[PluginSummary]: ...

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        """Get full JSON schema for a plugin's configuration.

        Raises:
            ValueError: If plugin_type is not a valid kind or name is
                not a registered plugin of that type.
        """
        ...

    def post_call_hints(
        self,
        *,
        plugin_type: PluginKind,
        plugin_name: str,
        tool_name: str,
        config_snapshot: Mapping[str, object],
    ) -> tuple[str, ...]:
        """Resolve postscript hints for a plugin after a successful mutation.

        Looks up the plugin class and dispatches to its
        ``get_post_call_hints`` classmethod. Returns the (possibly
        empty) tuple of hint strings. Advisory coaching only —
        identical audit-hash discipline to ``composer_hints`` on
        discovery DTOs.

        Raises:
            ValueError: If plugin_type is not a valid kind or
                plugin_name is not a registered plugin of that type.
        """
        ...
