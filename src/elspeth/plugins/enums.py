# src/elspeth/plugins/enums.py
"""Enumerations for plugin and DAG concepts.

Using explicit enums prevents string typos and provides IDE support.
These are the canonical definitions - use them everywhere.
"""

from enum import Enum


class NodeType(str, Enum):
    """Types of nodes in the execution DAG.

    Using str as base allows direct JSON serialization and comparison.
    """

    SOURCE = "source"
    TRANSFORM = "transform"
    GATE = "gate"
    AGGREGATION = "aggregation"
    COALESCE = "coalesce"
    SINK = "sink"


class RoutingKind(str, Enum):
    """Kinds of routing decisions made by gates.

    CONTINUE: Row proceeds to next node in linear path
    ROUTE: Gate returns semantic label, executor resolves via routes config
    FORK_TO_PATHS: Row copies to multiple parallel paths
    """

    CONTINUE = "continue"
    ROUTE = "route"  # Label-based routing via config
    FORK_TO_PATHS = "fork_to_paths"


class RoutingMode(str, Enum):
    """How tokens are handled during routing.

    MOVE: Token transfers to destination (original disappears)
    COPY: Token clones to destination (original continues)
    """

    MOVE = "move"
    COPY = "copy"


class Determinism(str, Enum):
    """Plugin determinism classification for reproducibility.

    DETERMINISTIC: Same input always produces same output
    SEEDED: Reproducible with seed (e.g., random sampling)
    NONDETERMINISTIC: May vary (e.g., external API calls, LLMs)
    """

    DETERMINISTIC = "deterministic"
    SEEDED = "seeded"
    NONDETERMINISTIC = "nondeterministic"
