"""Types, constants, and exceptions for DAG operations.

Leaf module — no intra-package imports (prevents import cycles).
Matches the orchestrator/types.py pattern.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from elspeth.contracts.data import CompatibilityResult
from elspeth.contracts.enums import NodeType
from elspeth.contracts.freeze import freeze_fields
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.types import NODE_ID_MAX_LENGTH, CoalesceName, NodeID

if TYPE_CHECKING:
    from elspeth.contracts import PluginSchema, TransformProtocol
    from elspeth.core.config import TransformSettings


class GraphValidationError(ValueError):
    """Raised when graph validation fails.

    Attributes:
        component_id: The node ID responsible for the error (e.g. ``'gate_1'``,
                      ``'transform_check'``), or None for structural errors
                      (cycles, wrong source count, etc.) where no single node
                      is at fault. Callers should read this instead of
                      regex-parsing str(exc).
        component_type: The node type (e.g. ``'gate'``, ``'transform'``,
                        ``'coalesce'``), or None for structural errors.
                        Set at the raise site where the type is known,
                        avoiding fragile string-splitting of component_id.
    """

    def __init__(
        self,
        message: str,
        *,
        component_id: str | None = None,
        component_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.component_id: str | None = component_id
        self.component_type: str | None = component_type


class EdgeContractError(GraphValidationError):
    """Raised when graph edge schema compatibility fails.

    Subclass of ``GraphValidationError`` carrying the structured
    ``CompatibilityResult`` plus producer/consumer identity, so downstream
    layers (composer runtime preflight error formatting) can build LLM-
    actionable suggestions without re-parsing the prose form.

    Why a subclass and not a separate exception:
      - All existing callers that catch ``GraphValidationError`` keep working
        without change (subclass IS-A parent).
      - The downstream catch site can opportunistically narrow with
        ``isinstance(exc, EdgeContractError)`` to access structured fields,
        falling back to the prose ``str(exc)`` when other graph validation
        paths raise the parent class.

    Attributes:
        from_node_id: Producer node ID (e.g. ``'web_scrape_a1b2c3'``).
        to_node_id: Consumer node ID (e.g. ``'line_explode_d4e5f6'``).
        producer_schema_name: Pydantic schema class name (e.g.
            ``'WebScrapeOutput'``) — useful for naming the contract surface
            to the LLM, but the per-field detail lives in
            ``compatibility_result``.
        consumer_schema_name: Pydantic schema class name (e.g.
            ``'LineExplodeInput'``).
        compatibility_result: The full structured incompatibility detail —
            ``missing_fields``, ``type_mismatches[(field, expected, actual)]``,
            ``extra_fields``, and ``constraint_mismatches``. The error-message
            formatter walks this to produce actionable per-field guidance.
    """

    def __init__(
        self,
        message: str,
        *,
        from_node_id: str,
        to_node_id: str,
        producer_schema_name: str,
        consumer_schema_name: str,
        compatibility_result: CompatibilityResult,
        component_type: str | None = None,
    ) -> None:
        super().__init__(
            message,
            component_id=to_node_id,
            component_type=component_type,
        )
        self.from_node_id: str = from_node_id
        self.to_node_id: str = to_node_id
        self.producer_schema_name: str = producer_schema_name
        self.consumer_schema_name: str = consumer_schema_name
        self.compatibility_result: CompatibilityResult = compatibility_result


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

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("GraphValidationWarning.code must not be empty")
        if not self.message:
            raise ValueError("GraphValidationWarning.message must not be empty")


@dataclass(frozen=True, slots=True)
class BranchInfo:
    """Properties of a fork branch that routes to a coalesce node.

    Consolidates branch routing facts that used to be carried by several
    builder-local structures: destination coalesce, producing gate, coalesce
    input connection, and whether the branch goes through a transform chain.

    The optional ``schema`` field stores the branch's producer schema,
    enabling runtime tracking of which fields would have been contributed
    by a branch that was lost (diverted to error sink). This supports
    audit trail queries like "what fields were expected from lost branch X?"
    """

    coalesce_name: CoalesceName
    gate_node_id: NodeID
    input_connection: str = ""
    uses_transform_chain: bool = False
    schema: SchemaConfig | None = None

    def __post_init__(self) -> None:
        if not self.coalesce_name:
            raise ValueError("BranchInfo.coalesce_name must not be empty")
        if not self.gate_node_id:
            raise ValueError("BranchInfo.gate_node_id must not be empty")


_NODE_ID_MAX_LENGTH = NODE_ID_MAX_LENGTH


# Config stored on graph nodes varies by node type:
# - Source/Transform/Sink: raw plugin config dict (arbitrary keys per plugin)
# - Gate: {routes, condition, fork_to?}
# - Aggregation: {input_schema, trigger, output_mode, options}
# - Coalesce: {branches, policy, merge, timeout_seconds?, quorum_count?, select_branch?}
# Schema data is accessed via output_schema_config on NodeInfo, not the raw config
# keys. Source/transform/sink nodes may retain either "schema" or "schema_config"
# in config for hashing/audit fidelity; runtime semantics read the parsed
# output_schema_config instead. Aggregations use "input_schema" on the wrapper
# config to prevent add_node() from auto-populating output_schema_config (see
# elspeth-c3a98c358c).
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
    input_schema: type[PluginSchema] | None = None
    output_schema: type[PluginSchema] | None = None
    input_schema_config: SchemaConfig | None = None
    output_schema_config: SchemaConfig | None = None
    # Populated only for SINK nodes by the builder from SinkProtocol.declared_required_fields.
    # Used for build-time validation that upstream coalesce output guarantees the
    # fields a sink requires. Empty frozenset for all non-sink nodes.
    declared_required_fields: frozenset[str] = field(default_factory=frozenset)

    # Pass-through contract flag (ADR-007). Populated only for TRANSFORM nodes
    # by the builder from TransformProtocol.passes_through_input. When True,
    # the validator walk propagates predecessor guarantees through this node
    # when computing effective guaranteed fields. Always False for non-TRANSFORM
    # nodes; non-False value on any non-TRANSFORM node raises GraphValidationError.
    passes_through_input: bool = False

    def __post_init__(self) -> None:
        component_type = self.node_type.name.lower()
        component_id = self.node_id or None
        if not self.node_id:
            raise GraphValidationError(
                "NodeInfo.node_id must not be empty",
                component_id=component_id,
                component_type=component_type,
            )
        if not self.plugin_name or not self.plugin_name.strip():
            raise GraphValidationError(
                f"NodeInfo.plugin_name must not be empty for node {self.node_id!r}",
                component_id=component_id,
                component_type=component_type,
            )
        if len(self.node_id) > _NODE_ID_MAX_LENGTH:
            msg = f"node_id exceeds {_NODE_ID_MAX_LENGTH} characters: '{self.node_id}' (length={len(self.node_id)})"
            raise GraphValidationError(msg, component_id=self.node_id, component_type=component_type)
        # Offensive programming: declared_required_fields is sink-specific.
        # Catch the misuse at construction time rather than letting stray
        # data sit on a non-sink node until a future validator widens its
        # scope and produces mysterious errors.
        if self.declared_required_fields and self.node_type != NodeType.SINK:
            raise GraphValidationError(
                f"NodeInfo.declared_required_fields is only meaningful for SINK nodes; "
                f"node {self.node_id!r} has type {self.node_type.name} "
                f"with declared_required_fields={sorted(self.declared_required_fields)!r}.",
                component_id=self.node_id,
                component_type=component_type,
            )
        # Offensive programming: passes_through_input is for nodes that execute
        # a TransformProtocol plugin — TRANSFORM and AGGREGATION. Aggregations
        # (including BatchReplicate wired under `aggregations:` in YAML) execute
        # transform-class plugins and share the propagation semantics per
        # ADR-007. Setting it on source/coalesce/sink/gate indicates a builder
        # bug or misrouted attribute assignment; surface at construction.
        if self.passes_through_input and self.node_type not in (NodeType.TRANSFORM, NodeType.AGGREGATION):
            raise GraphValidationError(
                f"NodeInfo.passes_through_input is only meaningful for TRANSFORM or "
                f"AGGREGATION nodes; node {self.node_id!r} has type {self.node_type.name}.",
                component_id=self.node_id,
                component_type=component_type,
            )
        # NOTE: config is NOT frozen here because the builder mutates
        # output_schema_config on pass-through nodes (gates, coalesce) via
        # object.__setattr__ during schema propagation. Deep freeze is
        # applied by build_execution_graph() after all mutations are complete.


@dataclass(frozen=True, slots=True)
class _GateEntry:
    """Internal gate metadata for coalesce and routing wiring."""

    node_id: NodeID
    name: str
    fork_to: tuple[str, ...] | None
    routes: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("_GateEntry.node_id must not be empty")
        if not self.name:
            raise ValueError("_GateEntry.name must not be empty")
        if self.fork_to is not None and len(self.fork_to) == 0:
            raise ValueError("_GateEntry.fork_to must not be empty tuple (use None for no fork)")
        if len(self.routes) == 0:
            raise ValueError("_GateEntry.routes must have at least one entry")
        freeze_fields(self, "routes")


@dataclass(frozen=True, slots=True)
class WiredTransform:
    """Pair a transform plugin instance with its wiring settings."""

    plugin: TransformProtocol
    settings: TransformSettings

    def __post_init__(self) -> None:
        """Ensure wiring metadata matches the instantiated plugin."""
        if self.plugin.name != self.settings.plugin:
            raise GraphValidationError(
                f"WiredTransform mismatch: settings.plugin='{self.settings.plugin}' but plugin instance name='{self.plugin.name}'.",
                component_id=self.settings.name,
                component_type="transform",
            )


def _suggest_similar(name: str, candidates: list[str]) -> list[str]:
    """Suggest similar names for wiring validation errors."""
    import difflib

    return difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)
