# src/elspeth/plugins/results.py
"""Result types for plugin operations.

These types define the contracts between plugins and the SDA engine.
All fields needed for Phase 3 Landscape/OpenTelemetry integration are
included here, even if not used until Phase 3.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from elspeth.contracts import RoutingAction, RowOutcome

# Re-export types as part of public plugin API
__all__ = [
    "AcceptResult",
    "GateResult",
    "RoutingAction",
    "RowOutcome",
    "TransformResult",
]


@dataclass
class TransformResult:
    """Result from any transform operation.

    Note: Routing comes from GateResult, not TransformResult.
    TransformResult is for row transforms that either succeed or error.

    Includes all fields needed for Phase 3 Landscape audit.
    The engine populates audit fields; plugins set status/row/reason.

    Audit hashes are SHA-256 over RFC 8785 canonical JSON
    (computed by elspeth.core.canonical.stable_hash).
    """

    status: Literal["success", "error"]  # No "route" - use GateResult for routing
    row: dict[str, Any] | None
    reason: dict[str, Any] | None
    retryable: bool = False

    # === Phase 3 Audit Fields ===
    # Set by engine, not by plugins
    input_hash: str | None = field(default=None, repr=False)
    output_hash: str | None = field(default=None, repr=False)
    duration_ms: float | None = field(default=None, repr=False)

    @classmethod
    def success(cls, row: dict[str, Any]) -> "TransformResult":
        """Create a successful transform result."""
        return cls(status="success", row=row, reason=None)

    @classmethod
    def error(
        cls,
        reason: dict[str, Any],
        *,
        retryable: bool = False,
    ) -> "TransformResult":
        """Create an error result."""
        return cls(
            status="error",
            row=None,
            reason=reason,
            retryable=retryable,
        )


@dataclass
class GateResult:
    """Result from a gate transform.

    Gates evaluate rows and decide routing, possibly modifying the row.
    """

    row: dict[str, Any]
    action: RoutingAction

    # === Phase 3 Audit Fields ===
    input_hash: str | None = field(default=None, repr=False)
    output_hash: str | None = field(default=None, repr=False)
    duration_ms: float | None = field(default=None, repr=False)


@dataclass
class AcceptResult:
    """Result from aggregation accept().

    Indicates whether the row was accepted and if batch should trigger.
    """

    accepted: bool
    trigger: bool  # Should flush now?

    # === Phase 3 Audit Fields ===
    # batch_id is set by engine when creating/updating Landscape batch
    batch_id: str | None = field(default=None, repr=False)
