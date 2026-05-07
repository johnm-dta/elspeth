"""CatalogService protocol — internal service boundary for plugin catalog."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary

PluginKind = Literal["source", "transform", "sink"]


@runtime_checkable
class CatalogService(Protocol):
    """Read-only plugin catalog.

    Four methods, all synchronous (plugin discovery is CPU-bound).
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
