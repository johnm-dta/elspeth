"""Hook implementation for built-in aggregation plugins."""

from typing import Any

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinAggregations:
    """Hook implementer for built-in aggregation plugins.

    Currently returns empty list - no built-in aggregations yet.
    This hookimpl ensures the hook is registered so external plugins
    can provide aggregations.
    """

    @hookimpl
    def elspeth_get_aggregations(self) -> list[type[Any]]:
        """Return built-in aggregation plugin classes.

        Returns:
            Empty list - no built-in aggregations yet.
        """
        return []


# Singleton instance for registration
builtin_aggregations = ElspethBuiltinAggregations()
