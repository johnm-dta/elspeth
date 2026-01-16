"""Hook implementation for built-in coalesce plugins."""

from typing import Any

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinCoalesces:
    """Hook implementer for built-in coalesce plugins.

    Currently returns empty list - no built-in coalesces yet.
    This hookimpl ensures the hook is registered so external plugins
    can provide coalesces.
    """

    @hookimpl
    def elspeth_get_coalesces(self) -> list[type[Any]]:
        """Return built-in coalesce plugin classes.

        Returns:
            Empty list - no built-in coalesces yet.
        """
        return []


# Singleton instance for registration
builtin_coalesces = ElspethBuiltinCoalesces()
