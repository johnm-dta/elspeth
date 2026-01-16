"""Hook implementation for built-in source plugins."""

from typing import Any

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinSources:
    """Hook implementer for built-in source plugins."""

    @hookimpl
    def elspeth_get_source(self) -> list[type[Any]]:
        """Return built-in source plugin classes."""
        from elspeth.plugins.sources.csv_source import CSVSource
        from elspeth.plugins.sources.json_source import JSONSource

        return [CSVSource, JSONSource]


# Singleton instance for registration
builtin_sources = ElspethBuiltinSources()
