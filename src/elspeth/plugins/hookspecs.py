# src/elspeth/plugins/hookspecs.py
"""pluggy hook specifications for Elspeth plugins.

Plugins implement these hooks to register themselves with the framework.
The plugin manager calls these hooks during discovery.

Usage (implementing a plugin):
    from elspeth.plugins.hookspecs import hookimpl

    class MyPlugin:
        @hookimpl  # NOT @hookspec - that's for defining specs
        def elspeth_get_transforms(self):
            return [MyTransform]

Note: @hookspec defines the hook interface (done here).
      @hookimpl marks plugin implementations of those hooks.
"""

from typing import TYPE_CHECKING

import pluggy

if TYPE_CHECKING:
    from elspeth.plugins.protocols import (
        GateProtocol,
        SinkProtocol,
        SourceProtocol,
        TransformProtocol,
    )

# Project name for pluggy
PROJECT_NAME = "elspeth"

# Hook specification marker
hookspec = pluggy.HookspecMarker(PROJECT_NAME)

# Hook implementation marker (for plugins to use)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)


class ElspethSourceSpec:
    """Hook specifications for source plugins."""

    @hookspec
    def elspeth_get_source(self) -> list[type["SourceProtocol"]]:  # type: ignore[empty-body]
        """Return source plugin classes.

        Returns:
            List of Source plugin classes (not instances)
        """


class ElspethTransformSpec:
    """Hook specifications for transform plugins."""

    @hookspec
    def elspeth_get_transforms(self) -> list[type["TransformProtocol"]]:  # type: ignore[empty-body]
        """Return transform plugin classes.

        Returns:
            List of Transform plugin classes
        """

    @hookspec
    def elspeth_get_gates(self) -> list[type["GateProtocol"]]:  # type: ignore[empty-body]
        """Return gate plugin classes.

        Returns:
            List of Gate plugin classes
        """


class ElspethSinkSpec:
    """Hook specifications for sink plugins."""

    @hookspec
    def elspeth_get_sinks(self) -> list[type["SinkProtocol"]]:  # type: ignore[empty-body]
        """Return sink plugin classes.

        Returns:
            List of Sink plugin classes
        """
