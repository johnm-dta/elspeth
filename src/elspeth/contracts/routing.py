"""Flow control and edge definitions.

These types answer: "Where does data go next?"
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from elspeth.contracts.enums import RoutingKind, RoutingMode


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
    """A routing decision from a gate.

    Gates return this to indicate where tokens should go next.
    Use the factory methods to create instances.

    CRITICAL: The `mode` field determines move vs copy semantics:
    - MOVE: Token exits current path, goes to destination only
    - COPY: Token clones to destination AND continues on current path

    This field is REQUIRED per architecture. Without it, executors cannot
    correctly record routing events or determine token flow.

    Invariants (enforced by __post_init__):
    - CONTINUE must have empty destinations
    - FORK_TO_PATHS must use COPY mode
    - ROUTE must have exactly one destination
    """

    kind: RoutingKind
    destinations: tuple[str, ...]
    mode: RoutingMode
    reason: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Validate invariants between kind, mode, and destinations."""
        if self.kind == RoutingKind.CONTINUE and self.destinations:
            raise ValueError("CONTINUE must have empty destinations")

        if self.kind == RoutingKind.FORK_TO_PATHS and self.mode != RoutingMode.COPY:
            raise ValueError("FORK_TO_PATHS must use COPY mode")

        if self.kind == RoutingKind.ROUTE and len(self.destinations) != 1:
            raise ValueError("ROUTE must have exactly one destination")

    @classmethod
    def continue_(cls, *, reason: dict[str, Any] | None = None) -> "RoutingAction":
        """Continue to next node in pipeline."""
        return cls(
            kind=RoutingKind.CONTINUE,
            destinations=(),
            mode=RoutingMode.MOVE,  # Default for continue
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
        """Route to a specific labeled destination.

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
        """Fork token to multiple parallel paths (always copy mode).

        Raises:
            ValueError: If paths is empty or contains duplicates.
        """
        if not paths:
            raise ValueError("fork_to_paths requires at least one destination path")
        if len(paths) != len(set(paths)):
            duplicates = [p for p in paths if paths.count(p) > 1]
            raise ValueError(f"fork_to_paths requires unique path names (duplicates: {sorted(set(duplicates))})")
        return cls(
            kind=RoutingKind.FORK_TO_PATHS,
            destinations=tuple(paths),
            mode=RoutingMode.COPY,  # Fork always copies
            reason=_freeze_dict(reason),
        )


@dataclass(frozen=True)
class RoutingSpec:
    """Specification for a routing edge in the recorded audit trail.

    Strict contract - mode MUST be RoutingMode enum, not string.
    Conversion from DB strings happens in repository layer.
    """

    edge_id: str
    mode: RoutingMode


@dataclass(frozen=True)
class EdgeInfo:
    """Information about an edge in the execution graph.

    Replaces tuple[str, str, dict[str, Any]] for type safety.
    Strict contract - mode MUST be RoutingMode enum.
    """

    from_node: str
    to_node: str
    label: str
    mode: RoutingMode
