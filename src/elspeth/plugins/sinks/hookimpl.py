"""Hook implementation for built-in sink plugins."""

from typing import Any

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinSinks:
    """Hook implementer for built-in sink plugins."""

    @hookimpl
    def elspeth_get_sinks(self) -> list[type[Any]]:
        """Return built-in sink plugin classes."""
        from elspeth.plugins.sinks.csv_sink import CSVSink
        from elspeth.plugins.sinks.database_sink import DatabaseSink
        from elspeth.plugins.sinks.json_sink import JSONSink

        return [CSVSink, JSONSink, DatabaseSink]


# Singleton instance for registration
builtin_sinks = ElspethBuiltinSinks()
