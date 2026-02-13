# src/elspeth/core/dag/models.py
"""Types, constants, and exceptions for DAG operations.

Leaf module — no intra-package imports (prevents import cycles).
Matches the orchestrator/types.py pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from elspeth.contracts.enums import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.types import NodeID
from elspeth.core.landscape.schema import NODE_ID_COLUMN_LENGTH

if TYPE_CHECKING:
    from types import MappingProxyType

    from elspeth.contracts import PluginSchema
    from elspeth.core.config import TransformSettings
    from elspeth.plugins.protocols import TransformProtocol


class GraphValidationError(ValueError):
    """Raised when graph validation fails."""

    pass


@dataclass(frozen=True, slots=True)
class GraphValidationWarning:
    """Non-fatal warning emitted during graph construction.

    Unlike GraphValidationError, warnings don't prevent graph construction.
    They alert the operator to configurations that are technically valid but
    likely to cause runtime surprises (e.g., rows silently lost).
    """

    code: str
    message: str
    node_ids: tuple[str, ...]


_NODE_ID_MAX_LENGTH = NODE_ID_COLUMN_LENGTH


# Config stored on graph nodes varies by node type:
# - Source/Transform/Sink: raw plugin config dict (arbitrary keys per plugin)
# - Gate: {schema, routes, condition, fork_to?}
# - Aggregation: {schema, trigger, output_mode, options}
# - Coalesce: {branches, policy, merge, timeout_seconds?, quorum_count?, select_branch?}
# Only "schema" is accessed cross-type. Other keys are opaque to the graph layer.
# dict[str, Any] is intentional: plugin configs are validated by each plugin's
# Pydantic model, not by the graph. The graph only hashes them for node IDs.
type NodeConfig = dict[str, Any]


@dataclass(frozen=True, slots=True)
class NodeInfo:
    """Information about a node in the execution graph.

    Frozen after construction — attribute reassignment is prevented to
    guarantee audit trail consistency. Schemas are locked at launch and
    never change during the run.

    Schema Contracts:
        input_schema_config and output_schema_config store the original
        SchemaConfig from plugin configuration. These contain contract
        declarations (guaranteed_fields, required_fields) used for DAG
        validation. The input_schema and output_schema are the generated
        Pydantic model types used for runtime validation.
    """

    node_id: NodeID
    node_type: NodeType
    plugin_name: str
    config: NodeConfig = field(default_factory=dict)
    # NOTE: config is typed as dict for construction compatibility, but is
    # frozen to MappingProxyType by build_execution_graph() in builder.py
    # after graph build.
    input_schema: type[PluginSchema] | None = None
    output_schema: type[PluginSchema] | None = None
    input_schema_config: SchemaConfig | None = None
    output_schema_config: SchemaConfig | None = None

    def __post_init__(self) -> None:
        if len(self.node_id) > _NODE_ID_MAX_LENGTH:
            msg = f"node_id exceeds {_NODE_ID_MAX_LENGTH} characters: '{self.node_id}' (length={len(self.node_id)})"
            raise GraphValidationError(msg)


@dataclass(frozen=True)
class _GateEntry:
    """Internal gate metadata for coalesce and routing wiring."""

    node_id: NodeID
    name: str
    fork_to: tuple[str, ...] | None
    routes: MappingProxyType[str, str]


@dataclass(frozen=True, slots=True)
class WiredTransform:
    """Pair a transform plugin instance with its wiring settings."""

    plugin: TransformProtocol
    settings: TransformSettings

    def __post_init__(self) -> None:
        """Ensure wiring metadata matches the instantiated plugin."""
        if self.plugin.name != self.settings.plugin:
            raise GraphValidationError(
                f"WiredTransform mismatch: settings.plugin='{self.settings.plugin}' but plugin instance name='{self.plugin.name}'."
            )


def _suggest_similar(name: str, candidates: list[str]) -> list[str]:
    """Suggest similar names for wiring validation errors."""
    import difflib

    return difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)
