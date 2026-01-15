# src/elspeth/plugins/results.py
"""Result types for plugin operations.

These types define the contracts between plugins and the SDA engine.
All fields needed for Phase 3 Landscape/OpenTelemetry integration are
included here, even if not used until Phase 3.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

from elspeth.plugins.enums import RoutingKind, RoutingMode


class RowOutcome(Enum):
    """Terminal states for rows in the pipeline.

    DESIGN NOTE: Per architecture (00-overview.md:267-279), token terminal
    states are DERIVED from the combination of node_states, routing_events,
    and batch membership—not stored as a column. This enum is used at
    query/explain time to report final disposition, not at runtime.

    The engine does NOT set these directly. The Landscape query layer
    derives them when answering explain() queries.

    INVARIANT: Every row reaches exactly one terminal state.
    No silent drops.
    """

    COMPLETED = "completed"  # Reached output sink
    ROUTED = "routed"  # Sent to named sink by gate (move mode)
    FORKED = "forked"  # Split into child tokens (parent terminates)
    CONSUMED_IN_BATCH = "consumed_in_batch"  # Fed into aggregation
    COALESCED = "coalesced"  # Merged with other tokens
    QUARANTINED = "quarantined"  # Failed, stored for investigation
    FAILED = "failed"  # Failed, not recoverable


def _freeze_dict(d: dict[str, Any] | None) -> Mapping[str, Any]:
    """Create immutable view of dict with defensive deep copy.

    MappingProxyType only prevents mutation through the proxy.
    We deep copy to prevent mutation via retained references to
    the original dict or nested objects.
    """
    if d is None:
        return MappingProxyType({})
    # Deep copy to prevent mutation of original or nested dicts
    return MappingProxyType(copy.deepcopy(d))


@dataclass(frozen=True)
class RoutingAction:
    """What a gate decided to do with a row.

    Fully immutable: frozen dataclass with tuple destinations and
    MappingProxyType-wrapped reason.
    """

    kind: RoutingKind
    destinations: tuple[str, ...]  # Immutable sequence
    mode: RoutingMode
    reason: Mapping[str, Any]  # Immutable mapping (MappingProxyType)

    @classmethod
    def continue_(cls, reason: dict[str, Any] | None = None) -> "RoutingAction":
        """Row continues to next transform."""
        return cls(
            kind=RoutingKind.CONTINUE,
            destinations=(),
            mode=RoutingMode.MOVE,
            reason=_freeze_dict(reason),
        )

    @classmethod
    def route(
        cls,
        label: str,
        *,
        mode: RoutingMode = RoutingMode.MOVE,
        reason: dict[str, Any] | None = None,
    ) -> "RoutingAction":
        """Route row to a destination determined by route label.

        Gates return semantic route labels (e.g., "above", "below", "match").
        The executor resolves these labels via the plugin's `routes` config
        to determine the actual destination (sink name or "continue").

        Args:
            label: Route label that will be resolved via routes config
            mode: MOVE (default) or COPY
            reason: Audit trail information about why this route was chosen
        """
        return cls(
            kind=RoutingKind.ROUTE,
            destinations=(label,),
            mode=mode,
            reason=_freeze_dict(reason),
        )

    @classmethod
    def fork_to_paths(
        cls,
        paths: list[str],
        *,
        reason: dict[str, Any] | None = None,
    ) -> "RoutingAction":
        """Fork row to multiple parallel paths (copy mode)."""
        return cls(
            kind=RoutingKind.FORK_TO_PATHS,
            destinations=tuple(paths),
            mode=RoutingMode.COPY,
            reason=_freeze_dict(reason),
        )


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
