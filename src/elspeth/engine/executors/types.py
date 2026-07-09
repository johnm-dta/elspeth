"""Shared types for executor modules."""

from dataclasses import dataclass

from elspeth.contracts import GateResult, TokenInfo
from elspeth.contracts.enums import RoutingKind
from elspeth.contracts.types import NodeID


class MissingEdgeError(Exception):
    """Raised when routing refers to an unregistered edge.

    This is an audit integrity error - every routing decision must be
    traceable to a registered edge. Silent edge loss is unacceptable.
    """

    def __init__(self, node_id: NodeID, label: str) -> None:
        """Initialize with routing details.

        Args:
            node_id: Node that attempted routing
            label: Edge label that was not found
        """
        self.node_id = node_id
        self.label = label
        super().__init__(
            f"No edge registered from node {node_id} with label '{label}'. Audit trail would be incomplete - refusing to proceed."
        )


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """Result of gate execution with routing information.

    Contains the gate result plus information about how the token
    should be routed and any child tokens created.

    Invariant: sink_name and next_node_id are mutually exclusive.
    A gate routes to a sink OR jumps to a node, never both.
    """

    result: GateResult
    updated_token: TokenInfo
    child_tokens: tuple[TokenInfo, ...] = ()
    sink_name: str | None = None
    next_node_id: NodeID | None = None
    discarded: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "child_tokens", tuple(self.child_tokens))
        if self.sink_name is not None and self.next_node_id is not None:
            raise ValueError(
                f"GateOutcome invariant violation: sink_name={self.sink_name!r} and "
                f"next_node_id={self.next_node_id!r} are mutually exclusive. "
                f"A gate routes to a sink OR jumps to a node, not both."
            )

        # Validate routing fields against action kind.
        # A malformed GateOutcome can misroute a token or mark a fork as
        # successful without creating any child work.
        kind = self.result.action.kind
        if kind == RoutingKind.CONTINUE:
            if self.sink_name is not None:
                raise ValueError(f"GateOutcome invariant: CONTINUE action cannot have sink_name={self.sink_name!r}")
            if self.next_node_id is not None:
                raise ValueError(f"GateOutcome invariant: CONTINUE action cannot have next_node_id={self.next_node_id!r}")
            if self.child_tokens:
                raise ValueError(f"GateOutcome invariant: CONTINUE action cannot have child_tokens (got {len(self.child_tokens)})")
            if self.discarded:
                raise ValueError("GateOutcome invariant: CONTINUE action cannot have discarded=True")
        elif kind == RoutingKind.ROUTE:
            if self.child_tokens:
                raise ValueError(f"GateOutcome invariant: ROUTE action cannot have child_tokens (got {len(self.child_tokens)})")
            route_destinations = sum(
                1
                for present in (
                    self.sink_name is not None,
                    self.next_node_id is not None,
                    self.discarded,
                )
                if present
            )
            if route_destinations != 1:
                raise ValueError("GateOutcome invariant: ROUTE action must have exactly one of sink_name, next_node_id, or discarded=True")
        elif kind == RoutingKind.FORK_TO_PATHS:
            if not self.child_tokens:
                raise ValueError("GateOutcome invariant: FORK_TO_PATHS action must have non-empty child_tokens")
            if self.sink_name is not None:
                raise ValueError(f"GateOutcome invariant: FORK_TO_PATHS action cannot have sink_name={self.sink_name!r}")
            if self.next_node_id is not None:
                raise ValueError(f"GateOutcome invariant: FORK_TO_PATHS action cannot have next_node_id={self.next_node_id!r}")
            if self.discarded:
                raise ValueError("GateOutcome invariant: FORK_TO_PATHS action cannot have discarded=True")
            destination_count = len(self.result.action.destinations)
            if len(self.child_tokens) != destination_count:
                raise ValueError(
                    "GateOutcome invariant: FORK_TO_PATHS action destinations "
                    f"({destination_count}) must match child_tokens ({len(self.child_tokens)})"
                )
