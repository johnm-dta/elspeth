"""Result types for plugin operations.

Types are defined in elspeth.contracts.results.
This module re-exports them as part of the public plugin API.
"""

from elspeth.contracts import (
    RoutingAction,
    SourceRow,
    TerminalOutcome,
    TerminalPath,
    TransformResult,
)

__all__ = [
    "RoutingAction",
    "SourceRow",
    "TerminalOutcome",
    "TerminalPath",
    "TransformResult",
]
