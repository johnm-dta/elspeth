# src/elspeth/plugins/results.py
"""Result types for plugin operations.

These types define the contracts between plugins and the SDA engine.

IMPORTANT: Types are now defined in elspeth.contracts.results.
This module re-exports them as part of the public plugin API.
"""

from elspeth.contracts import (
    AcceptResult,
    GateResult,
    RoutingAction,
    RowOutcome,
    TransformResult,
)

# Re-export types as part of public plugin API
__all__ = [
    "AcceptResult",
    "GateResult",
    "RoutingAction",
    "RowOutcome",
    "TransformResult",
]
