"""CompositionState and supporting data models for pipeline composition.

All dataclasses are frozen with slots. Container fields (options, routes,
fork_to, branches) are deep-frozen via freeze_fields() in __post_init__.
Mutation methods return new instances — they never modify the original.

Layer: L3 (application).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import Any, Literal, Self, TypedDict

from pydantic import ValidationError as PydanticValidationError

from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.contracts.guarantee_propagation import compose_propagation
from elspeth.contracts.plugin_protocols import TransformProtocol
from elspeth.contracts.plugin_semantics import SemanticEdgeContract
from elspeth.contracts.schema import (
    SchemaConfig,
    get_aggregation_contract_options,
    get_raw_node_required_fields,
    get_raw_producer_guaranteed_fields,
    get_raw_schema_config,
    get_raw_sink_required_fields,
    raw_options_have_schema,
)
from elspeth.contracts.sink import (
    FAILSINK_ELIGIBLE_PLUGIN_TEXT,
    FAILSINK_ELIGIBLE_SINK_PLUGINS,
    FILE_SINK_PLUGINS,
    LOCAL_RECOVERY_SINK_PLUGINS,
)
from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.contracts.wire_visible_identity import is_wire_visible_placeholder
from elspeth.core.config import (
    _MAX_NODE_NAME_LENGTH,
    _RESERVED_EDGE_LABELS,
    _VALID_NODE_NAME_RE,
    TriggerConfig,
    _validate_max_length,
    _validate_node_name_chars,
)
from elspeth.core.dag.coalesce_merge import merge_guaranteed_fields
from elspeth.web.composer._validation_probe import prepare_validation_probe_options
from elspeth.web.composer.guided.state_machine import GuidedSession

NodeType = Literal["transform", "gate", "aggregation", "coalesce", "queue"]
EdgeType = Literal["on_success", "on_error", "route_true", "route_false", "fork"]
CoalesceBranches = tuple[str, ...] | Mapping[str, str]

COMPOSER_NODE_TYPES: frozenset[str] = frozenset(("aggregation", "coalesce", "gate", "queue", "transform"))

_DECLARED_INPUT_FIELDS_OPTION = "required_input_fields"
_MISSING_DECLARED_INPUT_FIELDS = object()
_DISCARD_ROUTE_TARGET = "discard"
_FORK_ROUTE_TARGET = "fork"
# A queue's entire runtime surface is QueueSettings, whose only field is an
# optional operator-facing description (elspeth-a5b86149d4). Nothing else may
# ride in a queue node's options.
_QUEUE_OPTION_KEYS: frozenset[str] = frozenset({"description"})


def validate_composer_source_name(source_name: str) -> None:
    """Validate a composer source name against runtime settings constraints."""
    if not source_name or not source_name.strip():
        raise ValueError("source_name must be a non-empty string.")
    if source_name != source_name.lower():
        raise ValueError(f"Source name '{source_name}' must be lowercase. Suggested fix: '{source_name.lower()}'.")
    _validate_max_length(source_name, field_label="Source name", max_length=_MAX_NODE_NAME_LENGTH)
    _validate_node_name_chars(source_name, field_label="Source name")
    if source_name in _RESERVED_EDGE_LABELS:
        raise ValueError(f"Source name '{source_name}' is reserved. Reserved source/edge labels: {sorted(_RESERVED_EDGE_LABELS)}")
    if source_name.startswith("__"):
        raise ValueError(f"Source name '{source_name}' starts with '__', which is reserved for system edges")


def _composer_source_name_validation_message(source_name: str) -> str | None:
    """Return the runtime-equivalent source-name validation error, if any."""
    if not source_name or not source_name.strip():
        return "source_name must be a non-empty string."
    if source_name != source_name.lower():
        return f"Source name '{source_name}' must be lowercase. Suggested fix: '{source_name.lower()}'."
    if len(source_name) > _MAX_NODE_NAME_LENGTH:
        return f"Source name exceeds max length {_MAX_NODE_NAME_LENGTH} (got {len(source_name)})"
    if not _VALID_NODE_NAME_RE.match(source_name):
        return (
            f"Source name '{source_name}' contains invalid characters. "
            "Node names must start with a letter and contain only letters, digits, underscores, and hyphens."
        )
    if source_name in _RESERVED_EDGE_LABELS:
        return f"Source name '{source_name}' is reserved. Reserved source/edge labels: {sorted(_RESERVED_EDGE_LABELS)}"
    if source_name.startswith("__"):
        return f"Source name '{source_name}' starts with '__', which is reserved for system edges"
    return None


@dataclass(frozen=True, slots=True)
class PipelineMetadata:
    """Pipeline-level metadata.

    All fields are scalars or None. frozen=True is sufficient.
    """

    name: str = "Untitled Pipeline"
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Reconstruct from a plain dict (inverse of to_dict serialisation)."""
        return cls(
            name=d["name"],
            description=d["description"],
        )


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Pipeline source configuration.

    Attributes:
        plugin: Source plugin name (e.g. "csv", "json", "dataverse").
        on_success: Named connection point for the first downstream node.
        options: Plugin-specific configuration (path, schema, etc.).
        on_validation_failure: How to handle rows that fail schema validation.
    """

    plugin: str
    on_success: str
    options: Mapping[str, Any]
    on_validation_failure: str

    def __post_init__(self) -> None:
        freeze_fields(self, "options")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Reconstruct from a plain dict (inverse of to_dict serialisation)."""
        return cls(
            plugin=d["plugin"],
            on_success=d["on_success"],
            options=d["options"],
            on_validation_failure=d["on_validation_failure"],
        )


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Transform, gate, aggregation, or coalesce node.

    Attributes:
        id: Unique node identifier within the pipeline.
        node_type: One of "transform", "gate", "aggregation", "coalesce".
        plugin: Plugin name. None for gates and coalesces.
        input: Named connection point this node reads from.
        on_success: Named connection point for successful output. None for gates.
        on_error: Named connection point for error output. None if not diverted.
        options: Plugin-specific configuration.
        condition: Gate expression. None for non-gates.
        routes: Gate route mapping. None for non-gates.
        fork_to: Fork destinations for fork gates. None for non-fork nodes.
        branches: Branch inputs for coalesce nodes. None for non-coalesce nodes.
        policy: Coalesce policy. None for non-coalesce nodes.
        merge: Coalesce merge strategy. None for non-coalesce nodes.
        trigger: Aggregation batch trigger config. None for non-aggregation nodes.
        output_mode: Aggregation output mode ("passthrough" or "transform"). None for non-aggregation nodes.
        expected_output_count: Aggregation expected output count. None for non-aggregation nodes.
    """

    id: str
    node_type: NodeType
    plugin: str | None
    input: str
    on_success: str | None
    on_error: str | None
    options: Mapping[str, Any]
    condition: str | None
    routes: Mapping[str, str] | None
    fork_to: tuple[str, ...] | None
    branches: CoalesceBranches | None
    policy: str | None
    merge: str | None
    trigger: Mapping[str, Any] | None = None
    output_mode: str | None = None
    expected_output_count: int | None = None

    def __post_init__(self) -> None:
        # Mapping fields must be deep-frozen. Scalar, enum, and tuple fields
        # are already immutable and need no guard.
        freeze_fields(self, "options")
        if self.routes is not None:
            freeze_fields(self, "routes")
        if self.branches is not None:
            freeze_fields(self, "branches")
        if self.trigger is not None:
            freeze_fields(self, "trigger")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Reconstruct from a plain dict (inverse of to_dict serialisation).

        Optional fields (condition, routes, fork_to, branches, policy, merge,
        trigger, output_mode, expected_output_count) default to None when
        absent from the dict. fork_to is converted from list to tuple since
        to_dict() serialises tuples as lists. branches preserves mapping form
        for transformed coalesce branches and converts list form to tuple.
        """
        fork_to = d["fork_to"] if "fork_to" in d else None
        branches = d["branches"] if "branches" in d else None
        return cls(
            id=d["id"],
            node_type=d["node_type"],
            plugin=d["plugin"],
            input=d["input"],
            on_success=d["on_success"],
            on_error=d["on_error"],
            options=d["options"],
            condition=d["condition"] if "condition" in d else None,
            routes=d["routes"] if "routes" in d else None,
            fork_to=tuple(fork_to) if fork_to is not None else None,
            branches=dict(branches) if isinstance(branches, Mapping) else tuple(branches) if branches is not None else None,
            policy=d["policy"] if "policy" in d else None,
            merge=d["merge"] if "merge" in d else None,
            trigger=d["trigger"] if "trigger" in d else None,
            output_mode=d["output_mode"] if "output_mode" in d else None,
            expected_output_count=d["expected_output_count"] if "expected_output_count" in d else None,
        )


def _coalesce_branch_names(branches: CoalesceBranches | None) -> tuple[str, ...]:
    """Return branch identities declared by a coalesce node."""
    if branches is None:
        return ()
    if isinstance(branches, Mapping):
        return tuple(branches.keys())
    return branches


def _coalesce_branch_connections(branches: CoalesceBranches | None) -> tuple[str, ...]:
    """Return input connections consumed by a coalesce node."""
    if branches is None:
        return ()
    if isinstance(branches, Mapping):
        return tuple(branches.values())
    return branches


def _serialize_branches(branches: CoalesceBranches) -> list[str] | dict[str, str]:
    """Serialize coalesce branches preserving list-vs-mapping semantics."""
    if isinstance(branches, Mapping):
        return dict(deep_thaw(branches))
    return list(branches)


def queue_node_contract_error(node: NodeSpec) -> str | None:
    """Return the intrinsic (topology-free) contract violation for a queue node.

    A queue is a structural pass-through fan-in point (elspeth-a5b86149d4). Its
    canonical shape is ``id == input``, no plugin/routing/coalesce/aggregation
    fields, implicit output under its own id (``on_success is None``), and at
    most a string ``description`` option. This helper is the SINGLE source of
    truth for that shape so state validation, the mutation tools, and YAML
    generation all reject the same malformed queues identically. It performs no
    state/topology lookup — producer/consumer/namespace checks live in
    ``validate()``. Returns None for a non-queue node or a canonical queue.
    """
    if node.node_type != "queue":
        return None
    if node.input != node.id:
        return f"Queue '{node.id}' input must equal its id."
    forbidden = {
        "plugin": node.plugin,
        "on_success": node.on_success,
        "on_error": node.on_error,
        "condition": node.condition,
        "routes": node.routes,
        "fork_to": node.fork_to,
        "branches": node.branches,
        "policy": node.policy,
        "merge": node.merge,
        "trigger": node.trigger,
        "output_mode": node.output_mode,
        "expected_output_count": node.expected_output_count,
    }
    present = sorted(name for name, value in forbidden.items() if value is not None)
    if present:
        return f"Queue '{node.id}' does not accept field(s): {present}."
    unknown = sorted(set(node.options) - _QUEUE_OPTION_KEYS)
    if unknown:
        return f"Queue '{node.id}' contains unknown option(s): {unknown}."
    description = node.options.get("description")
    if description is not None and not isinstance(description, str):
        return f"Queue '{node.id}' options.description must be a string."
    return None


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    """Connection between two nodes.

    Attributes:
        id: Unique edge identifier.
        from_node: Source node ID (or "source" for the pipeline source).
        to_node: Destination node ID or sink name.
        edge_type: One of "on_success", "on_error", "route_true", "route_false", "fork".
        label: Display label (e.g. the route key for gate edges).
    """

    id: str
    from_node: str
    to_node: str
    edge_type: EdgeType
    label: str | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Reconstruct from a plain dict (inverse of to_dict serialisation)."""
        return cls(
            id=d["id"],
            from_node=d["from_node"],
            to_node=d["to_node"],
            edge_type=d["edge_type"],
            label=d["label"],
        )


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """Sink configuration.

    Attributes:
        name: Sink name (used as connection point in edges and routes).
        plugin: Sink plugin name (e.g. "csv", "json", "database").
        options: Plugin-specific configuration.
        on_write_failure: How to handle write failures ("discard" or a sink name).
    """

    name: str
    plugin: str
    options: Mapping[str, Any]
    on_write_failure: str

    def __post_init__(self) -> None:
        freeze_fields(self, "options")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Reconstruct from a plain dict (inverse of to_dict serialisation)."""
        return cls(
            name=d["name"],
            plugin=d["plugin"],
            options=d["options"],
            on_write_failure=d["on_write_failure"],
        )


Severity = Literal["high", "medium", "low"]


@dataclass(frozen=True, slots=True)
class ValidationEntry:
    """Structured validation message with component attribution.

    All fields are scalars. frozen=True is sufficient.
    """

    component: str
    message: str
    severity: Severity
    error_code: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Serialize to a plain dict for JSON responses."""
        result = {"component": self.component, "message": self.message, "severity": self.severity}
        if self.error_code is not None:
            result["error_code"] = self.error_code
        return result


EdgeContractDict = TypedDict(
    "EdgeContractDict",
    {
        "from": str,
        "to": str,
        "producer_guarantees": list[str],
        "consumer_requires": list[str],
        "missing_fields": list[str],
        "satisfied": bool,
    },
)


@dataclass(frozen=True, slots=True)
class EdgeContract:
    """Schema contract check result for a single producer->consumer edge."""

    from_id: str
    to_id: str
    producer_guarantees: tuple[str, ...]
    consumer_requires: tuple[str, ...]
    missing_fields: tuple[str, ...]
    satisfied: bool

    def to_dict(self) -> EdgeContractDict:
        """Serialize to a plain dict for JSON responses."""
        return {
            "from": self.from_id,
            "to": self.to_id,
            "producer_guarantees": list(self.producer_guarantees),
            "consumer_requires": list(self.consumer_requires),
            "missing_fields": list(self.missing_fields),
            "satisfied": self.satisfied,
        }


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Stage 1 validation result.

    errors block execution. warnings are advisory but actionable.
    suggestions are optional improvements. edge_contracts shows
    per-edge schema contract check results. semantic_contracts shows
    per-edge semantic contract check results (Phase 1: line_explode +
    web_scrape only). All are tuples for structured component
    attribution.
    """

    is_valid: bool
    errors: tuple[ValidationEntry, ...]
    warnings: tuple[ValidationEntry, ...] = ()
    suggestions: tuple[ValidationEntry, ...] = ()
    edge_contracts: tuple[EdgeContract, ...] = ()
    semantic_contracts: tuple[SemanticEdgeContract, ...] = ()


def _source_options_have_schema(options: Mapping[str, Any]) -> bool:
    """Return whether source options carry a schema under the current contract.

    Composer state can contain either the user-facing ``schema`` alias or the
    internal ``schema_config`` field name, because plugin config parsing allows
    population by either key. Read-only summaries and validation must use the
    same rule so they cannot drift.
    """
    return raw_options_have_schema(options)


def _known_batch_aware_transform_plugins() -> frozenset[str]:
    """Return transform names whose runtime config rejects declared inputs."""
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

    transforms = get_shared_plugin_manager().get_transforms()
    return frozenset(cls.name for cls in transforms if cls.is_batch_aware)


def _known_batch_aware_transform_plugins_requiring_aggregation() -> frozenset[str]:
    """Return batch-aware transform names that do not support row-mode dispatch."""
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

    transforms = get_shared_plugin_manager().get_transforms()
    return frozenset(cls.name for cls in transforms if cls.is_batch_aware and not cls.supports_row_mode_when_batch_aware)


def _declared_input_fields_option(options: Mapping[str, Any]) -> object:
    """Return the raw declared-input-field option, including wrapper-shaped aggregations."""
    if _DECLARED_INPUT_FIELDS_OPTION in options:
        return options[_DECLARED_INPUT_FIELDS_OPTION]

    if "options" in options:
        nested_options = options["options"]
        if isinstance(nested_options, Mapping) and _DECLARED_INPUT_FIELDS_OPTION in nested_options:
            return nested_options[_DECLARED_INPUT_FIELDS_OPTION]

    return _MISSING_DECLARED_INPUT_FIELDS


def _batch_aware_required_input_fields_error(
    node_id: str,
    plugin_name: str | None,
    options: Mapping[str, Any],
) -> str | None:
    """Reject ADR-013 declared input fields on batch-aware transform configs."""
    if plugin_name is None or plugin_name not in _known_batch_aware_transform_plugins():
        return None

    declared_input_fields = _declared_input_fields_option(options)
    if declared_input_fields is _MISSING_DECLARED_INPUT_FIELDS or declared_input_fields in (None, [], ()):
        return None

    return (
        f"Node '{node_id}' sets required_input_fields={declared_input_fields!r}, "
        f"but transform '{plugin_name}' is batch-aware. ADR-013 declared input "
        "fields only have a non-batch pre-emission dispatch site; remove "
        "required_input_fields and express batch input requirements with "
        "schema.required_fields."
    )


def _batch_aware_placement_error(
    node_id: str,
    node_type: str,
    plugin_name: str | None,
    output_mode: str | None,
) -> str | None:
    """Reject batch-only transforms from row-mode composer placement."""
    if plugin_name is None or plugin_name not in _known_batch_aware_transform_plugins_requiring_aggregation():
        return None

    if node_type == "transform":
        message = (
            f"Node '{node_id}' uses batch-aware transform '{plugin_name}' as node_type='transform'. "
            "Batch-aware transforms require the aggregation/batch path unless the plugin explicitly supports row mode. "
            "Configure this node as node_type='aggregation' with an aggregation trigger, or use a row-level transform instead."
        )
        if plugin_name == "batch_replicate":
            message += " For batch_replicate, set output_mode: transform so replicated rows create new downstream tokens."
        return message

    if plugin_name == "batch_replicate" and node_type == "aggregation" and output_mode != "transform":
        return (
            f"Node '{node_id}' uses batch_replicate, which deaggregates a batch into new rows. "
            "Configure it as an aggregation with output_mode: transform so replicated rows create new downstream tokens."
        )

    return None


def _batch_distribution_profile_contract_options(node: NodeSpec) -> Mapping[str, Any]:
    if node.node_type != "aggregation":
        return node.options
    contract_options, _owner = get_aggregation_contract_options(node.options, owner=f"node:{node.id}")
    return contract_options


def _batch_distribution_profile_value_field_message(
    *,
    value_field: str,
    field_type: str,
) -> str:
    return (
        f"batch_distribution_profile.value_field '{value_field}' is numeric-only, "
        f"but upstream declares type {field_type} "
        "(batch_distribution_profile.value_field.numeric). "
        "Categorical distributions, barrier counts, and theme frequency should use batch_top_k "
        "with field set to the categorical column and group_by as needed, not batch_distribution_profile."
    )


def _producer_declared_field_type(
    producer_id: str,
    plugin_name: str | None,
    options: Mapping[str, Any],
    *,
    node_by_id: Mapping[str, NodeSpec],
    field_name: str,
) -> str | None:
    """Return a declared schema field type for a producer, or None when unknown."""
    is_source_producer = producer_id == "source" or producer_id.startswith("source:")
    owner = producer_id if is_source_producer else f"node:{producer_id}"
    raw_schema = get_raw_schema_config(options, owner=owner)
    if raw_schema is not None and raw_schema.fields is not None:
        for field in raw_schema.fields:
            if field.name == field_name:
                return field.field_type
        return None

    if is_source_producer:
        return None

    if producer_id not in node_by_id:
        return None
    producer_node = node_by_id[producer_id]
    if producer_node.plugin is None:
        return None
    if producer_node.node_type not in {"transform", "aggregation"}:
        return None

    try:
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

        transform = get_shared_plugin_manager().create_transform(
            producer_node.plugin,
            prepare_validation_probe_options(producer_node.options),
        )
    except Exception as exc:
        if _is_static_contract_probe_exception(exc):
            return None
        raise

    output_schema = transform._output_schema_config
    if output_schema is None or output_schema.fields is None:
        return None
    for field in output_schema.fields:
        if field.name == field_name:
            return field.field_type
    return None


def _is_static_contract_probe_exception(exc: Exception) -> bool:
    """Return True for expected draft/config failures from static probes."""
    from elspeth.plugins.infrastructure.config_base import PluginConfigError
    from elspeth.plugins.infrastructure.manager import PluginNotFoundError
    from elspeth.plugins.infrastructure.templates import TemplateError
    from elspeth.plugins.infrastructure.validation import UnknownPluginTypeError

    if isinstance(exc, (PluginConfigError, PluginNotFoundError, TemplateError, UnknownPluginTypeError)):
        return True
    return type(exc) is ValueError and str(exc).startswith("Invalid configuration for transform ")


def _batch_distribution_profile_value_field_entries(
    sources: Mapping[str, SourceSpec],
    nodes: tuple[NodeSpec, ...],
) -> tuple[tuple[ValidationEntry, ...], tuple[ValidationEntry, ...]]:
    """Validate numeric-only batch_distribution_profile value_field contracts."""
    from elspeth.web.composer._producer_resolver import ProducerResolver

    errors: list[ValidationEntry] = []
    warnings: list[ValidationEntry] = []
    node_by_id = {node.id: node for node in nodes}
    resolver = ProducerResolver.build(
        source=None,
        sources=sources,
        nodes=nodes,
        sink_names=frozenset(),
    )
    numeric_types = {"int", "float"}

    for node in nodes:
        if node.plugin != "batch_distribution_profile":
            continue
        options = _batch_distribution_profile_contract_options(node)
        if "value_field" not in options:
            continue
        value_field = options["value_field"]
        if type(value_field) is not str or not value_field.strip():
            continue
        value_field = value_field.strip()

        producer = resolver.walk_to_real_producer(node.input)
        if producer is None:
            warnings.append(
                ValidationEntry(
                    f"node:{node.id}",
                    (
                        f"batch_distribution_profile.value_field '{value_field}' is numeric-only, "
                        "but the upstream producer is unresolved "
                        "(batch_distribution_profile.value_field.numeric). "
                        "If this field is categorical, use batch_top_k instead."
                    ),
                    "high",
                )
            )
            continue

        field_type = _producer_declared_field_type(
            producer.producer_id,
            producer.plugin_name,
            producer.options,
            node_by_id=node_by_id,
            field_name=value_field,
        )
        if field_type is None:
            warnings.append(
                ValidationEntry(
                    f"node:{node.id}",
                    (
                        f"batch_distribution_profile.value_field '{value_field}' is numeric-only, "
                        "but upstream schema is observed or does not declare the field type. "
                        "Inspect a data sample before execute "
                        "(batch_distribution_profile.value_field.numeric); "
                        "categorical distributions should use batch_top_k."
                    ),
                    "high",
                )
            )
            continue
        if field_type in numeric_types:
            continue
        errors.append(
            ValidationEntry(
                f"node:{node.id}",
                _batch_distribution_profile_value_field_message(
                    value_field=value_field,
                    field_type=field_type,
                ),
                "high",
            )
        )

    return tuple(errors), tuple(warnings)


def _runtime_connection_targets(
    sources: Mapping[str, SourceSpec],
    nodes: tuple[NodeSpec, ...],
) -> set[str]:
    """Collect runtime routing targets from connection fields.

    Stage 1 validity must follow the same routing model as generate_yaml()
    and DAG build: source/node connection fields define runtime topology, while
    non-sink UI edges are advisory/editor state.
    """
    targets: set[str] = set()
    for source in sources.values():
        targets.add(source.on_success)
    for node in nodes:
        if node.node_type == "coalesce" and node.on_success is None:
            targets.add(node.id)
        elif node.on_success is not None:
            targets.add(node.on_success)
        if node.on_error is not None and node.on_error != "discard":
            targets.add(node.on_error)
        if node.routes is not None:
            targets.update(target for target in node.routes.values() if target != _DISCARD_ROUTE_TARGET)
        if node.fork_to is not None:
            targets.update(node.fork_to)
    return targets


def _runtime_consumer_connections(nodes: tuple[NodeSpec, ...]) -> set[str]:
    """Return connection names runtime can resolve to processing nodes."""
    consumers = {node.input for node in nodes if node.node_type != "coalesce"}
    for node in nodes:
        if node.node_type == "coalesce" and node.branches is not None:
            consumers.update(_coalesce_branch_connections(node.branches))
    return consumers


def _validate_runtime_route_destinations(
    sources: Mapping[str, SourceSpec],
    nodes: tuple[NodeSpec, ...],
    outputs: tuple[OutputSpec, ...],
) -> tuple[ValidationEntry, ...]:
    """Mirror runtime DAG routing destination checks for terminal fields."""
    errors: list[ValidationEntry] = []
    output_names = {output.name for output in outputs}
    consumer_connections = _runtime_consumer_connections(nodes)
    _err = ValidationEntry

    for source_name, source in sources.items():
        target = source.on_success
        if target not in output_names and target not in consumer_connections:
            component = "source" if source_name == "source" else f"source:{source_name}"
            message = (
                f"Source on_success '{target}' is neither a sink nor a known connection."
                if source_name == "source"
                else f"Source '{source_name}' on_success '{target}' is neither a sink nor a known connection."
            )
            errors.append(
                _err(
                    component,
                    message,
                    "high",
                    "source_on_success_dangling",
                )
            )

    for node in nodes:
        if node.node_type == "transform":
            if node.on_success is not None and node.on_success not in output_names and node.on_success not in consumer_connections:
                errors.append(
                    _err(
                        f"node:{node.id}",
                        f"Transform '{node.id}' on_success '{node.on_success}' is neither a sink nor a known connection.",
                        "high",
                        "transform_on_success_dangling",
                    )
                )
            if node.on_error is not None and node.on_error != "discard" and node.on_error not in output_names:
                errors.append(
                    _err(
                        f"node:{node.id}",
                        f"Transform '{node.id}' on_error '{node.on_error}' references unknown sink.",
                        "high",
                        "transform_on_error_unknown_sink",
                    )
                )
            continue

        if node.node_type == "aggregation":
            if node.on_success is not None and node.on_success not in output_names and node.on_success not in consumer_connections:
                errors.append(
                    _err(
                        f"node:{node.id}",
                        f"Aggregation '{node.id}' on_success '{node.on_success}' is neither a sink nor a known connection.",
                        "high",
                        "aggregation_on_success_dangling",
                    )
                )
            continue

        if node.node_type == "coalesce":
            if node.on_success is None:
                continue
            if node.on_success in consumer_connections:
                errors.append(
                    _err(
                        f"node:{node.id}",
                        f"Coalesce '{node.id}' has on_success='{node.on_success}'. "
                        "Coalesce on_success must point to a sink when configured.",
                        "high",
                        "coalesce_on_success_must_be_sink",
                    )
                )
            elif node.on_success not in output_names:
                errors.append(
                    _err(
                        f"node:{node.id}",
                        f"Coalesce '{node.id}' on_success references unknown sink '{node.on_success}'.",
                        "high",
                        "coalesce_on_success_unknown_sink",
                    )
                )

    return tuple(errors)


def _validate_gate_expression(condition: str) -> str | None:
    """Validate a gate condition expression at composition time.

    Returns an error message if the expression is syntactically invalid or
    contains forbidden constructs, or None if valid.

    Uses a deferred import to keep the expression-parser dependency local to
    the validation path. The import is L3→L1, which is layer-legal.
    """
    from elspeth.core.expression_parser import (
        ExpressionParser,
        ExpressionSecurityError,
        ExpressionSyntaxError,
    )

    try:
        ExpressionParser(condition)
    except ExpressionSyntaxError as e:
        return f"Invalid gate condition syntax: {e}"
    except ExpressionSecurityError as e:
        return f"Forbidden construct in gate condition: {e}"
    return None


def _validate_gate_route_parity(condition: str, routes: Mapping[str, str] | None) -> str | None:
    """Validate gate route labels match the condition's static return type.

    Composition-time mirror of the runtime contract
    ``GateSettings.validate_boolean_routes`` (core/config.py): a boolean-typed
    condition (comparison, and/or/not, literal ``True``/``False``) must use route
    labels exactly ``{"true", "false"}``, and a provably-numeric condition can
    never produce a route label. Without this, ``CompositionState.validate()``
    would green-light a pipeline that runtime ``GateSettings`` construction later
    rejects.

    This is a deliberate second copy of the runtime predicate built on the same
    shared ``ExpressionParser`` substrate that ``_validate_gate_expression``
    already uses (durable unification is deferred to follow-up
    elspeth-f584eb820c). It must mirror ``validate_boolean_routes`` faithfully:
    same predicates, same ``boolean … elif non_routable`` precedence.

    Returns an error message when the route labels are inconsistent with the
    condition, or None when consistent (notably for string-returning conditions,
    which are routable by any label). The caller must only invoke this after the
    syntax/security check passed, so ``ExpressionParser(condition)`` does not
    re-raise here.
    """
    from elspeth.core.expression_parser import ExpressionParser

    parser = ExpressionParser(condition)
    if parser.is_boolean_expression():
        route_labels = set(routes or {})
        expected_labels = {"true", "false"}
        if route_labels != expected_labels:
            missing = expected_labels - route_labels
            extra = route_labels - expected_labels
            msg_parts = [f"Gate has a boolean condition ({condition!r}) but route labels don't match."]
            if extra:
                msg_parts.append(f"Found labels {sorted(extra)!r} but boolean expressions evaluate to True/False, not these values.")
            if missing:
                msg_parts.append(f"Missing required labels: {sorted(missing)!r}.")
            msg_parts.append('Use routes: {"true": <destination>, "false": <destination>}')
            return " ".join(msg_parts)
    elif parser.is_provably_non_routable():
        return (
            f"Gate condition ({condition!r}) statically returns a numeric value, "
            f"which can never be a route label. Gate conditions must evaluate to a boolean "
            f'(routes "true"/"false") or to a string route label.'
        )
    return None


# RFC 2606 / RFC 6761 reserved/special-use domain labels. Emails at these
# domains are not deliverable to anyone; values at these domains in
# `web_scrape.http.abuse_contact` are fabrications that ship as HTTP headers
# to scraped third parties — a Tier-1 audit-integrity defect — regardless of
# any prose rationale ("placeholder", "internal default") the composer LLM
# attached to them. Mechanical backstop for the skill-prompt rule in
# pipeline_composer.md (web_scrape.http section).
_RFC_RESERVED_DOMAIN_LABELS: tuple[str, ...] = (
    "example.com",
    "example.org",
    "example.net",
    "example",
    "test",
    "invalid",
    "localhost",
)


@trust_boundary(
    tier=3,
    source="NodeSpec carrying web-authored web_scrape options (untrusted abuse_contact value)",
    source_param="node",
    suppresses=("R1", "R5"),
    invariant=(
        "returns a high-severity ValidationEntry only for a well-formed abuse_contact at an "
        "RFC-reserved domain; absent, mistyped, or malformed values yield None (sibling "
        "plugin-schema rules report those) and never raise"
    ),
    non_raising=True,
)
def _validate_web_scrape_abuse_contact_not_reserved(node: NodeSpec) -> ValidationEntry | None:
    """Reject web_scrape.http.abuse_contact values at RFC-reserved domains.

    abuse_contact is wire-visible: it ships as an HTTP header on every
    outbound scrape request, and the receiving operator uses it to contact us.
    A reserved-domain address (`example.com`, `*.test`, etc.) is not
    deliverable and constitutes a fabricated identity on the wire.

    Returns None when the field is absent, has an unexpected type, lacks an
    `@`, or uses a real domain. Returns a high-severity ValidationEntry when
    the domain matches one of the RFC 2606/6761 reserved labels.
    """
    if node.plugin != "web_scrape":
        return None
    # node.options is statically Mapping[str, Any]; the value at "http" is
    # unstructured (Any) and may be absent or non-Mapping, so the inner
    # isinstance guard below remains.
    http = node.options.get("http")
    if not isinstance(http, Mapping):
        return None  # Plugin schema rule reports missing/malformed http block.
    abuse_contact = http.get("abuse_contact")
    if not isinstance(abuse_contact, str):
        return None
    if "@" not in abuse_contact:
        return None  # Malformed email — let the plugin schema rule report it.
    domain = abuse_contact.rsplit("@", 1)[1].strip().lower()
    for reserved in _RFC_RESERVED_DOMAIN_LABELS:
        if domain == reserved or domain.endswith("." + reserved):
            return ValidationEntry(
                component=f"node:{node.id}",
                message=(
                    f"web_scrape.http.abuse_contact has domain '{domain}' — RFC 2606/6761 reserves "
                    f"'{reserved}' for documentation/test use, so the value is not deliverable to "
                    "anyone and would ship as a fabricated identity in the HTTP header to the "
                    "scraped host. Set abuse_contact to an operator-supplied or "
                    "deployment-identity-sourced email (see the web_scrape.http rule in "
                    "pipeline_composer.md)."
                ),
                severity="high",
            )
    return None


@trust_boundary(
    tier=3,
    source="NodeSpec carrying web-authored web_scrape options (untrusted http identity fields)",
    source_param="node",
    suppresses=("R1", "R5"),
    invariant=(
        "emits a high-severity ValidationEntry per placeholder-valued wire-visible HTTP "
        "identity field; missing or mistyped options/http/field values are skipped "
        "(no entry) and never raised on"
    ),
    non_raising=True,
)
def _validate_web_scrape_http_identity_not_placeholder(node: NodeSpec) -> tuple[ValidationEntry, ...]:
    """Reject placeholder values in web_scrape's wire-visible HTTP identity fields."""
    if node.plugin != "web_scrape":
        return ()
    http = node.options.get("http")
    if not isinstance(http, Mapping):
        return ()

    errors: list[ValidationEntry] = []
    for field_name in ("abuse_contact", "scraping_reason"):
        value = http.get(field_name)
        if not isinstance(value, str):
            continue
        if not is_wire_visible_placeholder(value):
            continue
        errors.append(
            ValidationEntry(
                component=f"node:{node.id}",
                message=(
                    f"web_scrape.http.{field_name} is a placeholder value. This field ships as an HTTP "
                    "header to the scraped host, so it must be supplied by the operator or deployment "
                    "identity before the pipeline can be considered valid."
                ),
                severity="high",
            )
        )
    return tuple(errors)


def _validate_aggregation_trigger(node_id: str, trigger: Mapping[str, Any]) -> ValidationEntry | None:
    """Validate a composer-authored aggregation trigger at the Tier-3 boundary.

    ``node.trigger`` is composer/LLM/user-authored config read back from session
    state, so a malformed ``trigger`` is recoverable external input, not an
    invariant break. We run it through the same ``TriggerConfig`` parser the
    runtime settings load uses and convert a parse failure into an explicit
    blocking ``ValidationEntry`` — rejecting the bad trigger before runtime
    settings load rather than crashing the composer.
    """
    try:
        TriggerConfig.model_validate(deep_thaw(trigger))
    except PydanticValidationError as exc:
        detail = "; ".join(str(error["msg"]) for error in exc.errors())
        return ValidationEntry(
            component=f"node:{node_id}",
            message=f"Aggregation '{node_id}' trigger is invalid: {detail}",
            severity="high",
        )
    return None


def _locked_input_field_set(options: Mapping[str, Any], owner: str) -> frozenset[str] | None:
    """Return the consumer's accepted-input field set when its input is locked.

    Mirrors ``_create_explicit_schema`` (schema_factory.py): a generated input
    Pydantic model gets ``extra="forbid"`` only when ``schema.mode == "fixed"``.
    For ``mode: flexible`` (extras allowed) and ``mode: observed`` (no field
    constraints), returns None so the membership rule short-circuits.

    The accepted set IS the declared ``schema.fields`` — that is what the
    Pydantic model whitelists. Fields enumerated only via ``required_fields``
    or ``audit_fields`` do not appear in the input model and therefore are not
    part of the accepted set.
    """
    schema_config = get_raw_schema_config(options, owner=owner)
    if schema_config is None or schema_config.mode != "fixed" or schema_config.fields is None:
        return None
    return frozenset(field.name for field in schema_config.fields)


def _consumer_locked_input_set(node: NodeSpec) -> frozenset[str] | None:
    """Return the consumer node's accepted-input set when input is locked.

    Aggregation nodes carry their contract under either flat ``options`` or a
    nested ``options.options`` wrapper; resolve via ``get_aggregation_contract_options``
    so locked-input detection uses the same alias resolution as the rest of
    the contract pipeline. The augmented owner string the helper returns is
    discarded so the caller's existing error-message wording is preserved.
    """
    owner = f"node:{node.id}"
    if node.node_type == "aggregation":
        contract_options, _ = get_aggregation_contract_options(node.options, owner=owner)
        return _locked_input_field_set(contract_options, owner=owner)
    return _locked_input_field_set(node.options, owner=owner)


def _sink_locked_input_set(output: OutputSpec) -> frozenset[str] | None:
    """Return the sink's accepted-input set when its input contract is locked."""
    return _locked_input_field_set(output.options, owner=f"output:{output.name}")


@trust_boundary(
    tier=3,
    source="NodeSpecs carrying composer/LLM/user-authored options re-read from session state",
    source_param="nodes",
    suppresses=("R1",),
    invariant=(
        "optional node option flags (e.g. select_only) default at the read site; malformed "
        "node config surfaces as blocking ValidationEntry results, never a raise (genuine "
        "engine defects crash through via the config-probe re-raise guards)"
    ),
    non_raising=True,
)
def _check_schema_contracts(
    sources: Mapping[str, SourceSpec],
    nodes: tuple[NodeSpec, ...],
    outputs: tuple[OutputSpec, ...],
) -> tuple[
    tuple[ValidationEntry, ...],
    tuple[ValidationEntry, ...],
    tuple[EdgeContract, ...],
]:
    """Validate producer/consumer schema contracts across declarative routing."""
    from elspeth.web.composer._producer_resolver import ProducerEntry, ProducerResolver, is_source_producer_id, source_producer_id

    errors: list[ValidationEntry] = []
    contract_warnings: list[ValidationEntry] = []
    edge_contracts: list[EdgeContract] = []
    parse_failed_producers: set[str] = set()
    contract_probe_failed_producers: set[str] = set()
    sink_names = {output.name for output in outputs}
    sink_names_frozen = frozenset(sink_names)
    coalesce_branch_names = {
        branch_name
        for node in nodes
        if node.node_type == "coalesce" and node.branches is not None
        for branch_name in _coalesce_branch_names(node.branches)
    }
    internal_connection_names: set[str] = set()
    source_map = sources

    _err = ValidationEntry
    _warn = ValidationEntry

    if any(node.id == "source" for node in nodes):
        errors.append(
            _err(
                "pipeline",
                "Reserved node id 'source' cannot be used in composer state because contract walk-back uses it as the source sentinel.",
                "high",
            )
        )
        return tuple(errors), tuple(contract_warnings), ()

    # The resolver builds the connection -> producer map (with the
    # source-as-source-sentinel and same-node carve-out semantics), and
    # reports which connections have multiple distinct producers.
    resolver = ProducerResolver.build(
        source=None,
        sources=source_map,
        nodes=nodes,
        sink_names=sink_names_frozen,
    )
    node_by_id = {node.id: node for node in nodes}

    # Schema-specific bookkeeping: track per-connection producer
    # description (for richer duplicate-error messages) and the separate
    # direct-to-sink producers map (sink-targeted edges that the
    # resolver intentionally excludes from walk-back). Mirror the
    # resolver's registration order so first-seen descriptions match
    # the resolver's first-seen producer.
    producer_desc: dict[str, str] = {}
    duplicate_descs: dict[str, list[str]] = {}
    direct_sink_producers: dict[str, list[ProducerEntry]] = {}

    def _record_description(connection_name: str, description: str) -> None:
        if connection_name in producer_desc:
            if connection_name not in duplicate_descs:
                duplicate_descs[connection_name] = []
            duplicate_descs[connection_name].append(description)
            return
        producer_desc[connection_name] = description
        if connection_name not in sink_names:
            internal_connection_names.add(connection_name)

    def _record_direct_sink(
        sink_name: str,
        producer_id: str,
        plugin_name: str | None,
        options: Mapping[str, Any],
    ) -> None:
        if sink_name not in direct_sink_producers:
            direct_sink_producers[sink_name] = []
        direct_sink_producers[sink_name].append(ProducerEntry(producer_id=producer_id, plugin_name=plugin_name, options=options))

    for source_name, source in source_map.items():
        producer_id = source_producer_id(source_name)
        if source.on_success in sink_names:
            _record_direct_sink(
                source.on_success,
                producer_id,
                source.plugin,
                source.options,
            )
        else:
            source_desc = f"source '{source.plugin}'" if source_name == "source" else f"source '{source_name}' ({source.plugin})"
            _record_description(source.on_success, source_desc)

    for node in nodes:
        if node.node_type == "coalesce" and node.on_success is None:
            _record_description(node.id, f"coalesce '{node.id}'")
        elif node.on_success is not None:
            if node.on_success in sink_names:
                _record_direct_sink(node.on_success, node.id, node.plugin, node.options)
            else:
                _record_description(node.on_success, f"node '{node.id}' on_success")
        if node.on_error is not None and node.on_error != "discard":
            if node.on_error in sink_names:
                _record_direct_sink(node.on_error, node.id, node.plugin, node.options)
            else:
                _record_description(node.on_error, f"node '{node.id}' on_error")
        if node.routes is not None:
            for route_label, target in node.routes.items():
                if target == _DISCARD_ROUTE_TARGET:
                    continue
                if target == _FORK_ROUTE_TARGET:
                    # Reserved fork-mode keyword, not a connection. The
                    # resolver applies the same carve-out; description
                    # tracking must mirror it (see same-node carve-out
                    # below).
                    continue
                if target in sink_names:
                    _record_direct_sink(target, node.id, node.plugin, node.options)
                    continue
                # Same-node carve-out: a gate with multiple route labels
                # mapping to the same target is idempotent, not a
                # duplicate. The resolver applies the same carve-out, so
                # description tracking must mirror it to keep error
                # messages aligned.
                resolver_owner = resolver.find_producer_for(target)
                if resolver_owner is not None and resolver_owner.producer_id == node.id and target in producer_desc:
                    continue
                _record_description(target, f"gate '{node.id}' route '{route_label}'")
        if node.fork_to is not None:
            for branch_name in node.fork_to:
                if branch_name in sink_names:
                    # Fork branches that terminate at sinks behave like
                    # direct-to-sink edges from the gate: the runtime
                    # contract walks from the sink back through the gate
                    # to the gate's upstream producer (matched in
                    # _walk_producer_entry_to_real_producer's fork-vs-sink
                    # branch). Record both the direct-sink producer entry
                    # and the description so duplicate-error formatting
                    # remains identical to pre-resolver behaviour.
                    _record_direct_sink(branch_name, node.id, node.plugin, node.options)
                    continue
                _record_description(branch_name, f"gate '{node.id}' fork '{branch_name}'")

    # Surface duplicate-producer errors using the captured descriptions.
    for connection_name in sorted(resolver.duplicate_connections):
        first_desc = producer_desc[connection_name]
        # duplicate_descs may be missing if the duplicate was suppressed
        # by the same-node route carve-out; in that case the resolver
        # would not flag a duplicate either, so this branch is purely
        # defensive against future divergence.
        second_desc = duplicate_descs[connection_name][0]
        errors.append(
            _err(
                f"connection:{connection_name}",
                f"Duplicate producer for connection '{connection_name}': {first_desc} and {second_desc}.",
                "high",
            )
        )

    # Coalesce reads its branches (not its input); a queue reads its fan-in
    # predecessors but republishes under the same id, so counting it as a
    # consumer of its own connection would make the legal queue-then-consumer
    # pattern read as a duplicate consumer (elspeth-a5b86149d4). Both are
    # excluded from ordinary consumer accounting.
    consumer_claims: list[tuple[str, str, str]] = [
        (node.input, node.id, f"node '{node.id}'") for node in nodes if node.node_type not in ("coalesce", "queue")
    ]
    consumer_counts = Counter(connection_name for connection_name, _node_id, _desc in consumer_claims)
    duplicate_consumers = sorted(name for name, count in consumer_counts.items() if count > 1)
    for connection_name in duplicate_consumers:
        dup_entries = [(node_id, desc) for name, node_id, desc in consumer_claims if name == connection_name]
        first_node, first_desc = dup_entries[0]
        second_node, second_desc = dup_entries[1]
        errors.append(
            _err(
                f"connection:{connection_name}",
                f"Duplicate consumer for connection '{connection_name}': "
                f"{first_desc} ({first_node}) and {second_desc} ({second_node}). "
                "Use a gate for fan-out.",
                "high",
            )
        )

    internal_connection_names.update(connection_name for connection_name, _node_id, _desc in consumer_claims)
    # Runtime fork routing resolves coalesce branch names before sink names.
    # A branch identity that also names a sink would make composer preview treat
    # the branch as direct-to-sink while execution sends it to coalesce.
    internal_connection_names.update(coalesce_branch_names)
    overlap = sorted(internal_connection_names & sink_names)
    if overlap:
        errors.append(
            _err(
                "pipeline",
                f"Connection names overlap with sink names: {overlap}. Connection names and sink names must be disjoint.",
                "high",
            )
        )

    if errors:
        return tuple(errors), tuple(contract_warnings), ()

    def _walk_producer_entry_to_real_producer(
        producer: ProducerEntry,
        *,
        connection_name: str,
        warnings: list[ValidationEntry],
    ) -> ProducerEntry | None:
        """Schema-specific walk-back with coalesce/fork warning emission.

        Differs from ``ProducerResolver.walk_to_real_producer`` in two
        ways: it traverses fork gates and coalesce nodes only to emit
        skip-with-warning entries (the resolver returns None silently),
        and it stops at coalesce nodes because schema-contract
        propagation through coalesce branches is out of scope here.
        """
        visited_connections: set[str] = set()
        current_producer = producer
        while True:
            if is_source_producer_id(current_producer.producer_id):
                return current_producer

            producer_node = resolver.get_node(current_producer.producer_id)
            if producer_node is None:
                return None
            if producer_node.node_type == "coalesce":
                warnings.append(
                    _warn(
                        f"node:{producer_node.id}",
                        f"Contract check skipped because connection '{connection_name}' is produced by coalesce node '{producer_node.id}'; runtime validator will check this edge.",
                        "medium",
                    )
                )
                return None
            if producer_node.node_type == "queue":
                # A queue publishes an observed/unknown schema and never merges
                # its predecessors' guarantees (elspeth-a5b86149d4). Stop here
                # rather than picking one predecessor, so a downstream consumer's
                # required-field check abstains at the queue boundary instead of
                # comparing against a single arbitrary upstream.
                warnings.append(
                    _warn(
                        f"node:{producer_node.id}",
                        f"Contract check skipped because connection '{connection_name}' is produced by queue node '{producer_node.id}' with observed schema.",
                        "medium",
                    )
                )
                return None
            if producer_node.node_type != "gate":
                return current_producer
            if producer_node.fork_to is not None and connection_name not in sink_names:
                warnings.append(
                    _warn(
                        f"node:{producer_node.id}",
                        f"Contract check skipped because fork gate '{producer_node.id}' produces connection '{connection_name}'; branch-aware contract validation is out of scope for composer preview.",
                        "medium",
                    )
                )
                return None
            current_connection = producer_node.input
            if current_connection in visited_connections:
                warnings.append(
                    _warn(
                        f"connection:{connection_name}",
                        f"Contract check skipped for connection '{connection_name}' because producer walk-back encountered a routing loop.",
                        "medium",
                    )
                )
                return None
            visited_connections.add(current_connection)
            next_producer = resolver.find_producer_for(current_connection)
            if next_producer is None:
                return None
            current_producer = next_producer

    def _walk_to_real_producer(
        connection_name: str,
        *,
        warnings: list[ValidationEntry],
    ) -> ProducerEntry | None:
        producer = resolver.find_producer_for(connection_name)
        if producer is None:
            return None
        return _walk_producer_entry_to_real_producer(
            producer,
            connection_name=connection_name,
            warnings=warnings,
        )

    def _producer_owner(producer: ProducerEntry) -> str:
        return producer.producer_id if is_source_producer_id(producer.producer_id) else f"node:{producer.producer_id}"

    def _producer_label(producer: ProducerEntry) -> str:
        if producer.plugin_name is not None:
            return producer.plugin_name
        return node_by_id[producer.producer_id].node_type

    def _known_pass_through_plugins() -> frozenset[str]:
        """Lazily compute the set of pass-through plugin names from the live registry.

        Re-derived per call rather than cached at module-load — a plugin
        registered after composer module import (dynamic packs, test fixture
        ordering) was previously invisible to the fail-closed path. Cardinality
        is bounded by the annotated-transform set (short, known at startup).

        Reads ``cls.passes_through_input`` directly — no ``getattr`` defensive
        default. After the Phase A annotation, ``BaseTransform`` supplies the
        field for every transform class; a missing attribute IS a framework
        bug and must crash here loudly, not be silently coerced to ``False``.
        """
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

        transforms = get_shared_plugin_manager().get_transforms()
        return frozenset(cls.name for cls in transforms if cls.passes_through_input)

    def _is_config_probe_exception(exc: Exception) -> bool:
        """Return True only for expected draft/config failures from probe construction."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.infrastructure.manager import PluginNotFoundError
        from elspeth.plugins.infrastructure.templates import TemplateError
        from elspeth.plugins.infrastructure.validation import UnknownPluginTypeError

        if isinstance(exc, (PluginConfigError, PluginNotFoundError, TemplateError, UnknownPluginTypeError)):
            return True
        return type(exc) is ValueError and str(exc).startswith("Invalid configuration for transform ")

    def _probe_transform_construction(plugin: str, options: Mapping[str, Any]) -> TransformProtocol | None:
        """Construct ``plugin``'s transform, or None on an expected config failure.

        Genuine engine defects (non-config-probe exceptions) crash through —
        that re-raise is this helper's contract, keeping the enclosing
        non_raising boundary free of raises guarded by nodes-derived data.
        """
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

        try:
            return get_shared_plugin_manager().create_transform(
                plugin,
                prepare_validation_probe_options(options),
            )
        except Exception as exc:
            if not _is_config_probe_exception(exc):
                raise
            return None

    def _effective_producer_vote(producer: ProducerEntry) -> tuple[bool, frozenset[str]]:
        """Return (participates, guarantees) for preview propagation.

        Raw schema blocks are the baseline. For transform/aggregation nodes,
        prefer the plugin's computed output contract when construction succeeds;
        this keeps composer preview aligned with runtime for shape-changing
        producers like field_mapper/json_explode without turning incomplete
        draft configs into hard Stage 1 errors.

        Pass-through parity (ADR-007): for a transform whose plugin class is
        annotated ``passes_through_input=True``, the composer preview must
        mirror the runtime propagation — intersect the effective guarantees
        of upstream producers with the transform's own declared output. If
        the constructor probe fails for a *known* pass-through plugin, the
        composer fails closed (returns ``frozenset()``) so Stage 1 rejects
        the pipeline rather than silently serving a permissive preview that
        would diverge from runtime rejection.
        """
        raw_schema = get_raw_schema_config(
            producer.options,
            owner=_producer_owner(producer),
        )
        raw_guaranteed = get_raw_producer_guaranteed_fields(
            producer.plugin_name,
            producer.options,
            owner=_producer_owner(producer),
        )
        raw_participates = raw_schema is not None and raw_schema.participates_in_propagation
        if not raw_participates and raw_guaranteed:
            # Text-source heuristics can synthesize guarantees even when the
            # observed-mode schema itself abstains.
            raw_participates = True

        if is_source_producer_id(producer.producer_id):
            return raw_participates, raw_guaranteed

        producer_node = node_by_id[producer.producer_id]
        if producer_node.node_type not in {"transform", "aggregation"} or producer_node.plugin is None:
            return raw_participates, raw_guaranteed

        is_known_pass_through = producer_node.plugin in _known_pass_through_plugins()

        try:
            from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

            transform = get_shared_plugin_manager().create_transform(
                producer_node.plugin,
                prepare_validation_probe_options(producer_node.options),
            )
            is_pass_through_instance = transform.passes_through_input
            output_schema_config = transform._output_schema_config
        except Exception as exc:
            if not _is_config_probe_exception(exc):
                raise
            # Keep Stage 1 tolerant of partially configured draft nodes for
            # non-pass-through transforms — constructor-time errors must not
            # crash preview/export endpoints. For known pass-through plugins
            # we fail closed instead, because returning the raw (more permissive)
            # guarantees would let the composer accept pipelines the runtime
            # would reject.
            #
            # REDACTED: ``str(exc)`` from plugin constructors can carry
            # plugin option values (API URLs, file paths, DSN fragments,
            # occasionally secrets if an option is mis-typed into a connection
            # string), file system paths from credential-file readers, and
            # arbitrary library text routed from third-party validators. The
            # preview response surfaces these warnings directly to the
            # composer UI, where they render into an unauthenticated-
            # reachable error payload (preview is open to any logged-in
            # session owner, not just operators with secret-read grants).
            # Class name only — the triage signal ("something about this
            # plugin's config is wrong") is preserved; detailed diagnosis
            # belongs in server logs, not the UI warning list.
            if producer.producer_id not in contract_probe_failed_producers:
                contract_probe_failed_producers.add(producer.producer_id)
                if is_known_pass_through:
                    contract_warnings.append(
                        _warn(
                            f"node:{producer.producer_id}",
                            f"Computed contract probe for node '{producer.producer_id}' failed during preview "
                            f"({type(exc).__name__}); pipeline rejected "
                            f"(pass-through transform requires successful probe to mirror runtime propagation).",
                            "high",
                        )
                    )
                else:
                    contract_warnings.append(
                        _warn(
                            f"node:{producer.producer_id}",
                            f"Computed contract probe for node '{producer.producer_id}' failed during preview "
                            f"({type(exc).__name__}); falling back to raw schema declarations.",
                            "medium",
                        )
                    )
            if is_known_pass_through:
                return True, frozenset()
            return raw_participates, raw_guaranteed

        if is_pass_through_instance:
            base = output_schema_config.get_effective_guaranteed_fields() if output_schema_config is not None else frozenset()
            inherited_participates, inherited_fields = _connection_propagation_vote(producer_node.input)
            # ADR-009 §Clause 1: share the aggregation rule with graph.py.
            # Composer's producer-graph is single-upstream at this level
            # (coalesce absorbs fan-in via pre-computed output), so we pass a
            # one-element predecessor_guarantees list to compose_propagation.
            # An abstaining predecessor contributes None (skipped), not an
            # explicit empty set — same distinction the runtime walker makes.
            own_participates = output_schema_config.participates_in_propagation if output_schema_config is not None else raw_participates
            # Mirror walk_effective_guarantee_vote: a pass-through inherits
            # participation from its predecessors — own vote OR any upstream
            # vote. Dropping the inherited flag would let an own-abstaining
            # pass-through downstream of a participating producer read as
            # abstained here while the runtime marks it participated (and
            # validate_sink_required_fields rejects accordingly).
            return (
                own_participates or inherited_participates,
                compose_propagation(base, [inherited_fields if inherited_participates else None]),
            )

        if output_schema_config is None:
            return raw_participates, raw_guaranteed

        base = output_schema_config.get_effective_guaranteed_fields()
        return output_schema_config.participates_in_propagation, base

    def _effective_producer_guarantees(producer: ProducerEntry) -> frozenset[str]:
        """Return the producer guarantees Stage 1 should compare."""
        _participates, guarantees = _effective_producer_vote(producer)
        return guarantees

    def _connection_propagation_vote(connection_name: str) -> tuple[bool, frozenset[str]]:
        """Resolve a connection's propagation vote across structural nodes.

        Unlike ``_walk_to_real_producer()``, this helper is only used for
        pass-through inheritance and therefore follows structural fan-out/fan-in
        nodes instead of treating them as preview-stopping boundaries.

        Per ADR-009 §Clause 1, the aggregation rule (``compose_propagation``)
        and the participation predicate (``SchemaConfig.participates_in_propagation``)
        are shared with the runtime walker (``core/dag/guarantees.py``). The
        traversal logic remains separate — the composer walks a producer-graph
        (L3, connection-by-connection) while the runtime walks the DAG (L1,
        multi-predecessor). The two views legitimately differ; unifying
        traversal would pollute layers without eliminating duplication.
        """
        producer = resolver.find_producer_for(connection_name)
        if producer is None:
            return False, frozenset()

        if is_source_producer_id(producer.producer_id):
            return _effective_producer_vote(producer)

        producer_node = node_by_id[producer.producer_id]
        if producer_node.node_type == "gate":
            return _connection_propagation_vote(producer_node.input)

        if producer_node.node_type == "queue":
            # Observed/unknown schema: a queue never propagates or unions its
            # predecessors' pass-through guarantees, so a downstream consumer's
            # required_input_fields must not be resolved against one of them
            # (elspeth-a5b86149d4). Abstaining here keeps explicit required
            # fields on a valid consumer from being falsely rejected.
            return False, frozenset()

        if producer_node.node_type == "coalesce":
            if not producer_node.branches:
                return False, frozenset()

            branch_schemas: dict[str, SchemaConfig] = {}
            for branch_name, branch_connection in zip(
                _coalesce_branch_names(producer_node.branches),
                _coalesce_branch_connections(producer_node.branches),
                strict=True,
            ):
                branch_participates, branch_guarantees = _connection_propagation_vote(branch_connection)
                if not branch_participates:
                    continue
                branch_schemas[branch_name] = SchemaConfig(
                    mode="observed",
                    fields=None,
                    guaranteed_fields=tuple(sorted(branch_guarantees)),
                )

            if not branch_schemas:
                return False, frozenset()

            merged = merge_guaranteed_fields(
                branch_schemas,
                require_all=producer_node.policy == "require_all",
            )
            return True, frozenset(merged or ())

        return _effective_producer_vote(producer)

    def _format_fields(fields: frozenset[str]) -> str:
        return ", ".join(sorted(fields)) if fields else "(none)"

    def _producer_emit_set(producer: ProducerEntry) -> frozenset[str]:
        """Return the producer's *predicted emit* field set.

        This is distinct from ``_effective_producer_guarantees``: that one returns
        the producer's *declared* output set (``get_effective_guaranteed_fields``,
        which unions ``guaranteed_fields`` with declared-required ``fields``).
        Field-set membership rules need the *actual* emission — the set the
        runtime will see — which equals ``_output_schema_config.guaranteed_fields``
        for transforms whose plugins compute their own emit set
        (``field_mapper``, ``batch_stats``, etc.). When the two sets diverge,
        the transform itself is internally inconsistent (caught by the per-node
        self-consistency loop below); using the declared set for downstream
        Rule A/B checks would cascade Rule C breakage into spurious extras
        attribution at downstream consumers/sinks.

        For sources (no instance) and probe-failed transforms, falls back to
        the declared set — those are the cases where we don't have a separate
        emission inference, and the declared/raw set is the best signal.
        """
        if is_source_producer_id(producer.producer_id):
            return _effective_producer_guarantees(producer)

        if producer.producer_id not in node_by_id:
            # Sources have producer_id == "source" and are not members of the
            # locally-built node_by_id map; this is expected internal control
            # flow, not a missing-key anomaly, so fall back to the declared set.
            return _effective_producer_guarantees(producer)
        producer_node = node_by_id[producer.producer_id]
        if producer_node.plugin is None:
            return _effective_producer_guarantees(producer)
        if producer_node.node_type not in {"transform", "aggregation"}:
            return _effective_producer_guarantees(producer)

        try:
            from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

            transform = get_shared_plugin_manager().create_transform(
                producer_node.plugin,
                prepare_validation_probe_options(producer_node.options),
            )
        except Exception as exc:
            if not _is_config_probe_exception(exc):
                raise
            return _effective_producer_guarantees(producer)

        output_config = transform._output_schema_config
        if output_config is None or output_config.guaranteed_fields is None:
            # No computed emit set available — fall back to declared.
            return _effective_producer_guarantees(producer)
        return frozenset(output_config.guaranteed_fields)

    # Tier-3 contract-config parse boundary. node.options / output.options are
    # composer/LLM/user-authored config read back from session state, so a
    # malformed schema declaration is recoverable external input, not an
    # invariant break. These helpers convert the parser's ValueError into an
    # explicit (value, ValidationEntry) result the caller aggregates into the
    # validation report — the boundary surfaces the defect as a blocking entry
    # rather than crashing /validate on the first bad node.
    def _parse_node_required_fields(
        node: NodeSpec,
    ) -> tuple[frozenset[str] | None, ValidationEntry | None]:
        try:
            return (
                get_raw_node_required_fields(
                    node.options,
                    owner=f"node:{node.id}",
                    node_type=node.node_type,
                ),
                None,
            )
        except ValueError as exc:
            return None, _err(f"node:{node.id}", f"Invalid contract config: {exc}", "high")

    def _parse_consumer_locked_input(
        node: NodeSpec,
    ) -> tuple[frozenset[str] | None, ValidationEntry | None]:
        try:
            return _consumer_locked_input_set(node), None
        except ValueError as exc:
            return None, _err(f"node:{node.id}", f"Invalid contract config: {exc}", "high")

    def _parse_sink_required_fields(
        output: OutputSpec,
    ) -> tuple[frozenset[str] | None, ValidationEntry | None]:
        try:
            return (
                get_raw_sink_required_fields(output.options, owner=f"output:{output.name}"),
                None,
            )
        except ValueError as exc:
            return None, _err(f"output:{output.name}", f"Invalid contract config: {exc}", "high")

    def _parse_sink_locked_input(
        output: OutputSpec,
    ) -> tuple[frozenset[str] | None, ValidationEntry | None]:
        try:
            return _sink_locked_input_set(output), None
        except ValueError as exc:
            return None, _err(f"output:{output.name}", f"Invalid contract config: {exc}", "high")

    def _parse_producer_guarantees(
        producer: ProducerEntry,
    ) -> tuple[frozenset[str] | None, ValidationEntry | None]:
        try:
            return _effective_producer_guarantees(producer), None
        except ValueError as exc:
            return None, _err(_producer_owner(producer), f"Invalid contract config: {exc}", "high")

    def _parse_producer_vote(
        producer: ProducerEntry,
    ) -> tuple[tuple[bool, frozenset[str]] | None, ValidationEntry | None]:
        """Like ``_parse_producer_guarantees`` but preserves participation.

        The sink required-fields check needs to distinguish "abstained" from
        "participated and guarantees collapsed to empty" — the runtime twin
        (``validate_sink_required_fields``) skips abstaining producers and
        defers to per-row validation, and the composer must mirror that.
        """
        try:
            return _effective_producer_vote(producer), None
        except ValueError as exc:
            return None, _err(_producer_owner(producer), f"Invalid contract config: {exc}", "high")

    def _consumer_effective_required_set(node: NodeSpec) -> frozenset[str]:
        """Return the consumer's EFFECTIVE required-input fields.

        Unlike ``get_raw_node_required_fields`` (explicit ``required_fields``
        only — what runtime Phase-1 name-requirement checking consumes), this
        folds in a fixed/flexible schema's *implicitly* required declared
        fields via ``SchemaConfig.get_effective_required_fields``. Runtime
        marks those declared fields as required on the generated input Pydantic
        model, so Phase-2 type validation rejects a typed producer that does
        not guarantee one. This helper mirrors that, honouring the aggregation
        contract-options alias the rest of the contract pipeline uses.
        """
        contract_options = node.options
        contract_owner = f"node:{node.id}"
        if node.node_type == "aggregation":
            contract_options, contract_owner = get_aggregation_contract_options(node.options, owner=contract_owner)
        schema_config = get_raw_schema_config(contract_options, owner=contract_owner)
        if schema_config is None:
            return frozenset()
        return schema_config.get_effective_required_fields()

    def _parse_consumer_effective_required(
        node: NodeSpec,
    ) -> tuple[frozenset[str] | None, ValidationEntry | None]:
        try:
            return _consumer_effective_required_set(node), None
        except ValueError as exc:
            return None, _err(f"node:{node.id}", f"Invalid contract config: {exc}", "high")

    def _producer_is_typed_source(producer: ProducerEntry) -> bool:
        """Return whether the producer presents a TYPED (non-observed) schema.

        Mirrors the runtime Phase-2 bypass (``graph.py:1392-1403``): type
        validation fires only when the effective producer schema is a typed
        Pydantic model. A fixed/flexible *source* carries such a typed model;
        observed sources (including text-source auto-guarantee) and
        transform/gate/coalesce producers resolve to a dynamic (None) effective
        producer schema at runtime and are therefore skipped here. Gating on
        producer MODE — not guarantee-emptiness — avoids false rejection of the
        observed-source-with-auto-guaranteed-column case the runtime accepts.
        """
        if not is_source_producer_id(producer.producer_id):
            # transform/aggregation/gate/coalesce producers => dynamic effective
            # producer schema at runtime; the consumer's implicit requirement is
            # not statically enforced against them. Named sources mint
            # ``source:<name>`` producer ids, so match on the predicate, not the
            # literal "source" — else the parity check silently skips every
            # named typed source (elspeth-3332619032).
            return False
        schema_config = get_raw_schema_config(producer.options, owner=_producer_owner(producer))
        return schema_config is not None and not schema_config.is_observed

    for node in nodes:
        consumer_required, consumer_required_error = _parse_node_required_fields(node)
        if consumer_required_error is not None:
            errors.append(consumer_required_error)
            continue
        assert consumer_required is not None  # No error => resolved.

        consumer_locked_input, consumer_locked_error = _parse_consumer_locked_input(node)
        if consumer_locked_error is not None:
            errors.append(consumer_locked_error)
            continue

        consumer_effective_required, consumer_effective_error = _parse_consumer_effective_required(node)
        if consumer_effective_error is not None:
            errors.append(consumer_effective_error)
            continue
        assert consumer_effective_required is not None  # No error => resolved.

        # ``consumer_effective_required`` folds in a fixed/flexible consumer's
        # *implicitly* required declared fields (which the explicit-only
        # ``consumer_required`` misses), so a flexible consumer — whose input is
        # NOT locked (``consumer_locked_input is None``) and whose explicit
        # ``required_fields`` is empty — still reaches producer resolution.
        if not consumer_required and consumer_locked_input is None and not consumer_effective_required:
            continue

        actual_producer = _walk_to_real_producer(
            node.input,
            warnings=contract_warnings,
        )
        if actual_producer is None or actual_producer.producer_id in parse_failed_producers:
            continue

        producer_guaranteed, producer_error = _parse_producer_guarantees(actual_producer)
        if producer_error is not None:
            errors.append(producer_error)
            parse_failed_producers.add(actual_producer.producer_id)
            continue
        assert producer_guaranteed is not None  # No error => guarantees resolved.

        producer_is_typed_source = _producer_is_typed_source(actual_producer)
        contract_required = consumer_required
        if producer_is_typed_source:
            contract_required = consumer_required | consumer_effective_required

        if contract_required:
            contract_missing_fields = contract_required - producer_guaranteed
            edge_contracts.append(
                EdgeContract(
                    from_id=actual_producer.producer_id,
                    to_id=node.id,
                    producer_guarantees=tuple(sorted(producer_guaranteed)),
                    consumer_requires=tuple(sorted(contract_required)),
                    missing_fields=tuple(sorted(contract_missing_fields)),
                    satisfied=not contract_missing_fields,
                )
            )

        if consumer_required:
            missing_fields = consumer_required - producer_guaranteed
            if missing_fields:
                error_component = (
                    _producer_owner(actual_producer) if is_source_producer_id(actual_producer.producer_id) else f"node:{node.id}"
                )
                errors.append(
                    _err(
                        error_component,
                        f"Schema contract violation: '{actual_producer.producer_id}' -> '{node.id}'. "
                        f"Consumer ({node.plugin or node.node_type}) requires fields: [{_format_fields(consumer_required)}]. "
                        f"Producer ({_producer_label(actual_producer)}) guarantees: [{_format_fields(producer_guaranteed)}]. "
                        f"Missing fields: [{_format_fields(missing_fields)}].",
                        "high",
                    )
                )

        # Implicit-required parity: a fixed/flexible consumer schema implicitly
        # requires its declared (non-optional) fields. Runtime Phase-2 type
        # validation rejects a TYPED producer that does not guarantee one of
        # them; authoring's explicit-only ``consumer_required`` above misses
        # this. Mirror runtime by checking the consumer's *effective* required
        # set, but only against a typed (non-observed) producer — exactly the
        # producers runtime does NOT bypass (graph.py:1392-1403). Report only
        # the increment beyond the explicit set already handled above, so a
        # field is never double-reported.
        if producer_is_typed_source:
            implicit_missing = consumer_effective_required - consumer_required - producer_guaranteed
            if implicit_missing:
                errors.append(
                    _err(
                        f"node:{node.id}",
                        f"Schema contract violation: '{actual_producer.producer_id}' -> '{node.id}'. "
                        f"Consumer ({node.plugin or node.node_type}) requires fields: [{_format_fields(consumer_effective_required)}]. "
                        f"Producer ({_producer_label(actual_producer)}) guarantees: [{_format_fields(producer_guaranteed)}]. "
                        f"Missing fields: [{_format_fields(implicit_missing)}].",
                        "high",
                    )
                )

        # Rule A: producer emits a field that consumer's locked input forbids.
        # The runtime check is the auto-generated input Pydantic model with
        # ``extra="forbid"`` (schema_factory.py: triggered by ``mode: fixed``);
        # composer-time we mirror the same predicate against the producer's
        # *predicted emit set* (not its declared set — see _producer_emit_set).
        if consumer_locked_input is not None:
            # _effective_producer_guarantees(actual_producer) already succeeded
            # above (else we continued via parse_failed_producers), so this
            # producer's contract config parsed cleanly. _producer_emit_set walks
            # the same parse paths (create_transform / _effective_producer_guarantees)
            # on the same options, deterministically — any ValueError here would be
            # a non-determinism bug in our own code, not a fresh Tier-3 parse fault,
            # so it is left to crash rather than silently swallowed.
            producer_emit = _producer_emit_set(actual_producer)
            extras = producer_emit - consumer_locked_input
            if extras:
                # When consumer is itself a field_mapper, suggesting "insert a
                # field_mapper upstream" is degenerate — the operator is already
                # at one. The same applies to declared `fields` expansion: for
                # field_mapper, the input contract IS the declared output schema,
                # so widening fields means widening the schema declaration too.
                if node.plugin == "field_mapper":
                    fix_suggestion = (
                        f"Fix by adding {sorted(extras)!r} to the consumer's schema.fields, "
                        f"OR by setting schema.mode: flexible on the consumer, "
                        f"OR by adjusting upstream config so the extra field(s) are not emitted."
                    )
                else:
                    fix_suggestion = (
                        "Fix by relaxing the consumer schema (mode: flexible) or by inserting a "
                        "field_mapper with select_only: true to drop the extras before this consumer."
                    )
                errors.append(
                    _err(
                        f"node:{node.id}",
                        f"Schema contract violation: '{actual_producer.producer_id}' -> '{node.id}'. "
                        f"Consumer ({node.plugin or node.node_type}) input is locked (mode: fixed) and accepts: "
                        f"[{_format_fields(consumer_locked_input)}]. "
                        f"Producer ({_producer_label(actual_producer)}) will emit: "
                        f"[{_format_fields(producer_emit)}]. "
                        f"Extra fields rejected by consumer input contract: [{_format_fields(extras)}]. "
                        f"{fix_suggestion}",
                        "high",
                    )
                )

    for output in outputs:
        sink_required, sink_required_error = _parse_sink_required_fields(output)
        if sink_required_error is not None:
            errors.append(sink_required_error)
            continue

        sink_locked_input, sink_locked_error = _parse_sink_locked_input(output)
        if sink_locked_error is not None:
            errors.append(sink_locked_error)
            continue

        if not sink_required and sink_locked_input is None:
            continue

        if output.name in direct_sink_producers:
            sink_producers = tuple(direct_sink_producers[output.name])
        else:
            actual_producer = _walk_to_real_producer(
                output.name,
                warnings=contract_warnings,
            )
            sink_producers = () if actual_producer is None else (actual_producer,)

        seen_sink_contract_producers: set[str] = set()
        for sink_producer in sink_producers:
            actual_producer = _walk_producer_entry_to_real_producer(
                sink_producer,
                connection_name=output.name,
                warnings=contract_warnings,
            )
            if actual_producer is None:
                continue
            # Multiple direct routes from the same producer can converge on one
            # sink. edge_contracts has no route-label field, so emit one
            # producer->sink contract check per real upstream producer.
            if actual_producer.producer_id in seen_sink_contract_producers:
                continue
            seen_sink_contract_producers.add(actual_producer.producer_id)
            if actual_producer.producer_id in parse_failed_producers:
                continue

            producer_vote, producer_error = _parse_producer_vote(actual_producer)
            if producer_error is not None:
                errors.append(producer_error)
                parse_failed_producers.add(actual_producer.producer_id)
                continue
            assert producer_vote is not None  # No error => guarantees resolved.
            producer_participates, producer_guaranteed = producer_vote

            # ADR-007 parity: mirror the runtime abstention clause in
            # validate_sink_required_fields (core/dag/schema_validation.py).
            # An abstaining producer — no guarantees AND no participation, e.g.
            # a select_only field_mapper with an observed schema — makes no
            # static claim; the runtime builds the pipeline and enforces the
            # sink's required fields per-row. Emitting no EdgeContract renders
            # the edge as "not yet checked" rather than asserting a
            # satisfaction verdict the composer cannot support.
            if sink_required and (producer_guaranteed or producer_participates):
                missing_fields = sink_required - producer_guaranteed
                edge_contracts.append(
                    EdgeContract(
                        from_id=actual_producer.producer_id,
                        to_id=f"output:{output.name}",
                        producer_guarantees=tuple(sorted(producer_guaranteed)),
                        consumer_requires=tuple(sorted(sink_required)),
                        missing_fields=tuple(sorted(missing_fields)),
                        satisfied=not missing_fields,
                    )
                )
                if missing_fields:
                    errors.append(
                        _err(
                            f"output:{output.name}",
                            f"Schema contract violation: '{actual_producer.producer_id}' -> 'output:{output.name}'. "
                            f"Sink '{output.name}' requires fields: [{_format_fields(sink_required)}]. "
                            f"Producer ({_producer_label(actual_producer)}) guarantees: [{_format_fields(producer_guaranteed)}]. "
                            f"Missing fields: [{_format_fields(missing_fields)}].",
                            "high",
                        )
                    )

            # Rule B: producer emits a field that sink's locked input forbids.
            # Same predicate as Rule A but routed at the sink boundary; runtime
            # surface is the auto-generated sink Pydantic model with
            # ``extra="forbid"`` triggered by ``mode: fixed`` on the sink schema.
            if sink_locked_input is not None:
                # See Rule A above: _effective_producer_guarantees(actual_producer)
                # already succeeded for this producer in this iteration, so its
                # contract config parsed cleanly. _producer_emit_set walks the same
                # deterministic parse paths on the same options — a ValueError here
                # would be a non-determinism bug in our own code, not a fresh Tier-3
                # fault, so it is left to crash rather than silently swallowed.
                producer_emit = _producer_emit_set(actual_producer)
                extras = producer_emit - sink_locked_input
                if extras:
                    errors.append(
                        _err(
                            f"output:{output.name}",
                            f"Schema contract violation: '{actual_producer.producer_id}' -> 'output:{output.name}'. "
                            f"Sink '{output.name}' input is locked (mode: fixed) and accepts: "
                            f"[{_format_fields(sink_locked_input)}]. "
                            f"Producer ({_producer_label(actual_producer)}) will emit: "
                            f"[{_format_fields(producer_emit)}]. "
                            f"Extra fields rejected by sink input contract: [{_format_fields(extras)}]. "
                            f"Fix by relaxing the sink schema (mode: flexible) or by inserting a "
                            f"field_mapper with select_only: true to drop the extras before this sink.",
                            "high",
                        )
                    )

    # Rule C: per-transform self-consistency between declared output schema
    # and the *actual* predicted emit set, scoped to plugins whose emit set
    # can be computed deterministically from config alone. Currently:
    # ``field_mapper`` with ``select_only=True`` — the actual output is
    # exactly ``mapping.values()``, so any declared output field absent from
    # mapping targets cannot be emitted.
    #
    # Why this is plugin-scoped rather than generic: ``_output_schema_config.
    # guaranteed_fields`` has plugin-specific semantics. For field_mapper it
    # IS the actual emit set (computed by ``_build_field_mapper_output_schema_config``
    # from the mapping). For additive plugins like ``line_explode``/``web_scrape``
    # it is a *lower bound* on emission (only the fields the transform itself
    # adds — passes-through input fields are not enumerated), so a generic
    # ``get_effective_guaranteed_fields() - guaranteed_fields`` check would
    # mis-attribute every passthrough field as "missing". The runtime check
    # ``verify_schema_config_mode`` only sees the actually emitted row, so it
    # does not have this disambiguation problem; we earn that ambiguity-free
    # signal composer-time only by restricting to plugins where emit is fully
    # determined by config.
    #
    # As more reductive plugins land, extend the predicate below — do NOT
    # generalize by removing the plugin gate without first lifting an
    # ``actual_emit_set`` declaration onto each plugin class.
    for node in nodes:
        if node.node_type not in {"transform", "aggregation"} or node.plugin is None:
            continue
        if node.plugin != "field_mapper":
            continue
        if node.id in parse_failed_producers:
            continue
        # Read select_only directly from the raw options so the plugin gate
        # short-circuits before construction and stays free of access to
        # private plugin instance attributes from outside the plugin layer.
        # The semantics match ``FieldMapperConfig.select_only``: bool with
        # default False; any non-false-y option value triggers the reductive
        # emit semantics that make Rule C applicable.
        if not bool(node.options.get("select_only", False)):
            # Without select_only, field_mapper preserves input fields by
            # default and falls into the additive/loose-bound regime that we
            # cannot adjudicate without knowing the upstream emit set.
            continue
        transform = _probe_transform_construction(node.plugin, node.options)
        if transform is None:
            continue

        output_config = transform._output_schema_config
        if output_config is None:
            continue

        predicted_emit = frozenset(output_config.guaranteed_fields or ())
        declared_required = output_config.get_effective_guaranteed_fields()
        missing = declared_required - predicted_emit
        if not missing:
            continue
        errors.append(
            _err(
                f"node:{node.id}",
                f"Transform contract violation: node '{node.id}' ({node.plugin}) declares output fields "
                f"[{_format_fields(declared_required)}] (required) but with select_only: true the mapping will only emit "
                f"[{_format_fields(predicted_emit)}]. "
                f"Declared required output fields not produced by this transform: [{_format_fields(missing)}]. "
                f"Fix by removing the missing field(s) from the schema declaration, OR by extending "
                f"`mapping` so the transform actually emits them, OR by setting select_only: false.",
                "high",
            )
        )

    return tuple(errors), tuple(contract_warnings), tuple(edge_contracts)


@dataclass(frozen=True, slots=True, init=False)
class CompositionState:
    """Immutable, versioned snapshot of a pipeline under construction.

    Every edit produces a new instance with incremented version.
    All container fields are deep-frozen via freeze_fields().

    Attributes:
        sources: Named source roots keyed by stable composer/audit-visible name.
        nodes: Ordered tuple of transform, gate, aggregation, coalesce nodes.
        edges: Connections between nodes.
        outputs: Sink configurations.
        metadata: Pipeline name and description.
        version: Monotonically increasing per session, starting at 1.
        guided_session: Optional guided-mode session pointer. None for freeform
            sessions; set to GuidedSession.initial() at session-create time for
            guided sessions (spec §5.2).
    """

    nodes: tuple[NodeSpec, ...]
    edges: tuple[EdgeSpec, ...]
    outputs: tuple[OutputSpec, ...]
    metadata: PipelineMetadata
    version: int
    guided_session: GuidedSession | None = None
    sources: Mapping[str, SourceSpec] = field(default_factory=dict)

    def __init__(
        self,
        *,
        nodes: tuple[NodeSpec, ...],
        edges: tuple[EdgeSpec, ...],
        outputs: tuple[OutputSpec, ...],
        metadata: PipelineMetadata,
        version: int,
        guided_session: GuidedSession | None = None,
        sources: Mapping[str, SourceSpec] | None = None,
        source: SourceSpec | None = None,
    ) -> None:
        if version < 1:
            raise ValueError(f"CompositionState.version must be >= 1, got {version}")
        if source is not None and sources:
            raise ValueError("CompositionState accepts either source or sources, not both")
        source_map = {"source": source} if source is not None else dict(sources or {})
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "guided_session", guided_session)
        object.__setattr__(self, "sources", source_map)
        freeze_fields(self, "sources")

    # --- Mutation methods ---

    def with_source(self, source: SourceSpec) -> CompositionState:
        """Return new state with the default named source, version incremented."""
        return self.with_named_source("source", source)

    def without_source(self) -> CompositionState:
        """Return new state with all sources removed, version incremented."""
        return replace(self, sources={}, version=self.version + 1)

    def with_named_source(self, source_name: str, source: SourceSpec) -> CompositionState:
        """Add or replace a named source root. Version incremented."""
        validate_composer_source_name(source_name)
        sources = dict(self.sources)
        sources[source_name] = source
        return replace(self, sources=sources, version=self.version + 1)

    def without_named_source(self, source_name: str) -> CompositionState | None:
        """Remove one named source. Returns None if the source is not found."""
        if source_name not in self.sources:
            return None
        sources = dict(self.sources)
        del sources[source_name]
        return replace(self, sources=sources, version=self.version + 1)

    def with_node(self, node: NodeSpec) -> CompositionState:
        """Add or replace a node (matched by id). Version incremented."""
        existing_ids = [n.id for n in self.nodes]
        if node.id in existing_ids:
            # Replace at original position to preserve order
            idx = existing_ids.index(node.id)
            node_list = list(self.nodes)
            node_list[idx] = node
            nodes = tuple(node_list)
        else:
            # Append new node
            nodes = (*self.nodes, node)
        return replace(self, nodes=nodes, version=self.version + 1)

    def without_node(self, node_id: str) -> CompositionState | None:
        """Remove node by id. Returns None if node not found."""
        if not any(n.id == node_id for n in self.nodes):
            return None
        nodes = tuple(n for n in self.nodes if n.id != node_id)
        # Also remove edges referencing this node
        edges = tuple(e for e in self.edges if e.from_node != node_id and e.to_node != node_id)
        return replace(self, nodes=nodes, edges=edges, version=self.version + 1)

    def with_edge(self, edge: EdgeSpec) -> CompositionState:
        """Add or replace an edge (matched by id). Version incremented."""
        existing_ids = [e.id for e in self.edges]
        if edge.id in existing_ids:
            idx = existing_ids.index(edge.id)
            edge_list = list(self.edges)
            edge_list[idx] = edge
            edges = tuple(edge_list)
        else:
            edges = (*self.edges, edge)
        return replace(self, edges=edges, version=self.version + 1)

    def without_edge(self, edge_id: str) -> CompositionState | None:
        """Remove edge by id. Returns None if edge not found."""
        if not any(e.id == edge_id for e in self.edges):
            return None
        edges = tuple(e for e in self.edges if e.id != edge_id)
        return replace(self, edges=edges, version=self.version + 1)

    def with_output(self, output: OutputSpec) -> CompositionState:
        """Add or replace an output (matched by name). Version incremented."""
        existing_names = [o.name for o in self.outputs]
        if output.name in existing_names:
            idx = existing_names.index(output.name)
            output_list = list(self.outputs)
            output_list[idx] = output
            outputs = tuple(output_list)
        else:
            outputs = (*self.outputs, output)
        return replace(self, outputs=outputs, version=self.version + 1)

    def without_output(self, output_name: str) -> CompositionState | None:
        """Remove output by name. Returns None if output not found."""
        if not any(o.name == output_name for o in self.outputs):
            return None
        outputs = tuple(o for o in self.outputs if o.name != output_name)
        return replace(self, outputs=outputs, version=self.version + 1)

    def with_metadata(self, patch: dict[str, Any]) -> CompositionState:
        """Update metadata fields from partial dict. Version incremented."""
        current = self.metadata
        name = patch["name"] if "name" in patch else current.name
        description = patch["description"] if "description" in patch else current.description
        new_meta = PipelineMetadata(
            name=name,
            description=description,
        )
        return replace(self, metadata=new_meta, version=self.version + 1)

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Recursively unwrap frozen containers to plain Python types.

        Converts MappingProxyType -> dict, tuple -> list recursively.
        The result is suitable for yaml.dump() and JSON serialization.
        """

        result: dict[str, Any] = {
            "version": self.version,
            "metadata": {
                "name": self.metadata.name,
                "description": self.metadata.description,
            },
            "sources": {},
            "nodes": [],
            "edges": [],
            "outputs": [],
        }

        for source_name, source in self.sources.items():
            result["sources"][source_name] = {
                "plugin": source.plugin,
                "on_success": source.on_success,
                "options": deep_thaw(source.options),
                "on_validation_failure": source.on_validation_failure,
            }

        for node in self.nodes:
            node_dict: dict[str, Any] = {
                "id": node.id,
                "node_type": node.node_type,
                "plugin": node.plugin,
                "input": node.input,
                "on_success": node.on_success,
                "on_error": node.on_error,
                "options": deep_thaw(node.options),
            }
            if node.condition is not None:
                node_dict["condition"] = node.condition
            if node.routes is not None:
                node_dict["routes"] = deep_thaw(node.routes)
            if node.fork_to is not None:
                node_dict["fork_to"] = list(node.fork_to)
            if node.branches is not None:
                node_dict["branches"] = _serialize_branches(node.branches)
            if node.policy is not None:
                node_dict["policy"] = node.policy
            if node.merge is not None:
                node_dict["merge"] = node.merge
            if node.trigger is not None:
                node_dict["trigger"] = deep_thaw(node.trigger)
            if node.output_mode is not None:
                node_dict["output_mode"] = node.output_mode
            if node.expected_output_count is not None:
                node_dict["expected_output_count"] = node.expected_output_count
            result["nodes"].append(node_dict)

        for edge in self.edges:
            result["edges"].append(
                {
                    "id": edge.id,
                    "from_node": edge.from_node,
                    "to_node": edge.to_node,
                    "edge_type": edge.edge_type,
                    "label": edge.label,
                }
            )

        for output in self.outputs:
            result["outputs"].append(
                {
                    "name": output.name,
                    "plugin": output.plugin,
                    "options": deep_thaw(output.options),
                    "on_write_failure": output.on_write_failure,
                }
            )

        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Reconstruct from a plain dict (inverse of to_dict serialisation).

        Calls from_dict() on each nested Spec type. This is the only way
        to construct CompositionState from deserialised JSON (Spec AC #18).
        The round-trip invariant holds:
            state == CompositionState.from_dict(state.to_dict())
        """
        raw_sources = d["sources"] if "sources" in d and d["sources"] is not None else {}
        if not raw_sources and "source" in d and d["source"] is not None:
            raw_sources = {"source": d["source"]}
        sources = {name: SourceSpec.from_dict(source) for name, source in raw_sources.items()}
        return cls(
            sources=sources,
            nodes=tuple(NodeSpec.from_dict(n) for n in d["nodes"]),
            edges=tuple(EdgeSpec.from_dict(e) for e in d["edges"]),
            outputs=tuple(OutputSpec.from_dict(o) for o in d["outputs"]),
            metadata=PipelineMetadata.from_dict(d["metadata"]),
            version=d["version"],
        )

    # --- Validation ---

    def validate(self) -> ValidationSummary:
        """Run Stage 1 composition-time validation.

        Pure function of the current state — no DAG build or session mutation.
        Returns ValidationSummary with is_valid and human-readable errors.
        """
        errors: list[ValidationEntry] = []
        _err = ValidationEntry  # local alias for brevity

        # 1. Source exists
        if not self.sources:
            errors.append(_err("source", "No source configured.", "high"))
        for source_name in self.sources:
            source_name_error = _composer_source_name_validation_message(source_name)
            if source_name_error is not None:
                component = "source" if source_name == "source" else f"source:{source_name}"
                errors.append(_err(component, source_name_error, "high"))

        # 2. At least one output
        if not self.outputs:
            errors.append(_err("pipeline", "No sinks configured.", "high"))

        # 3. Edge references valid
        node_ids = {n.id for n in self.nodes}
        output_names = {o.name for o in self.outputs}
        valid_from = node_ids | set(self.sources) | {"source"}
        valid_to = node_ids | output_names
        for edge in self.edges:
            if edge.from_node not in valid_from:
                errors.append(_err(f"edge:{edge.id}", f"Edge '{edge.id}' references unknown node '{edge.from_node}' as from_node.", "high"))
            if edge.to_node not in valid_to:
                errors.append(_err(f"edge:{edge.id}", f"Edge '{edge.id}' references unknown node '{edge.to_node}' as to_node.", "high"))

        # 4. Node IDs unique
        seen_node_ids: set[str] = set()
        for node in self.nodes:
            if node.id in seen_node_ids:
                errors.append(_err(f"node:{node.id}", f"Duplicate node ID: '{node.id}'.", "high"))
            if node.id == "source" or node.id.startswith("source:"):
                errors.append(
                    _err(
                        f"node:{node.id}",
                        f"Reserved node id '{node.id}' cannot use the source producer namespace.",
                        "high",
                    )
                )
            seen_node_ids.add(node.id)

        # 5. Output names unique
        seen_output_names: set[str] = set()
        for output in self.outputs:
            if output.name in seen_output_names:
                errors.append(_err(f"output:{output.name}", f"Duplicate output name: '{output.name}'.", "high"))
            seen_output_names.add(output.name)

        # 6. Edge IDs unique
        seen_edge_ids: set[str] = set()
        for edge in self.edges:
            if edge.id in seen_edge_ids:
                errors.append(_err(f"edge:{edge.id}", f"Duplicate edge ID: '{edge.id}'.", "high"))
            seen_edge_ids.add(edge.id)

        # 7. Node type field consistency
        for node in self.nodes:
            if node.node_type not in COMPOSER_NODE_TYPES:
                expected = ", ".join(sorted(COMPOSER_NODE_TYPES))
                errors.append(
                    _err(
                        f"node:{node.id}",
                        f"Node '{node.id}' has unknown node_type '{node.node_type}'. Expected one of: {expected}.",
                        "high",
                        "unknown_node_type",
                    )
                )
                continue

            # Authored pipeline_decision reviews must use a registered decision
            # term: the resolve-side artifact-hash registry raises on unknown
            # terms, so a novel term mints an unresolvable review event and
            # wedges the session at the run gate.
            from elspeth.web.interpretation_state import REGISTERED_PIPELINE_DECISION_USER_TERMS

            authored_requirements = node.options.get("interpretation_requirements")
            if isinstance(authored_requirements, (list, tuple)):
                for requirement in authored_requirements:
                    if not isinstance(requirement, Mapping):
                        continue
                    if requirement.get("kind") != "pipeline_decision":
                        continue
                    term = requirement.get("user_term")
                    if not isinstance(term, str) or term.strip() not in REGISTERED_PIPELINE_DECISION_USER_TERMS:
                        errors.append(
                            _err(
                                f"node:{node.id}",
                                f"Node '{node.id}' declares a pipeline_decision review with unregistered "
                                f"user_term {term!r}. Registered decision kinds: "
                                f"{sorted(REGISTERED_PIPELINE_DECISION_USER_TERMS)}. Drop the requirement and "
                                "record the rationale in metadata.description, or use an "
                                "llm_prompt_template review for prompt-shaped decisions.",
                                "high",
                                "pipeline_decision_unregistered",
                            )
                        )

            batch_placement_error = _batch_aware_placement_error(node.id, node.node_type, node.plugin, node.output_mode)
            if batch_placement_error is not None:
                errors.append(_err(f"node:{node.id}", batch_placement_error, "high"))

            batch_required_error = _batch_aware_required_input_fields_error(node.id, node.plugin, node.options)
            if batch_required_error is not None:
                errors.append(_err(f"node:{node.id}", batch_required_error, "high"))

            abuse_contact_error = _validate_web_scrape_abuse_contact_not_reserved(node)
            if abuse_contact_error is not None:
                errors.append(abuse_contact_error)
            errors.extend(_validate_web_scrape_http_identity_not_placeholder(node))

            if node.node_type == "gate":
                if node.condition is None:
                    errors.append(
                        _err(
                            f"node:{node.id}", f"Gate '{node.id}' is missing required field 'condition'.", "high", "gate_missing_condition"
                        )
                    )
                else:
                    # Validate expression content — defense-in-depth catches
                    # malformed conditions from any entry path (including
                    # session deserialization).
                    expr_error = _validate_gate_expression(node.condition)
                    if expr_error is not None:
                        errors.append(_err(f"node:{node.id}", f"Gate '{node.id}': {expr_error}", "high"))
                    elif node.routes is not None:
                        # Route-label / condition-return-type parity — mirror of
                        # GateSettings.validate_boolean_routes so the composer does
                        # not green-light a pipeline runtime config later rejects.
                        parity_error = _validate_gate_route_parity(node.condition, node.routes)
                        if parity_error is not None:
                            errors.append(
                                _err(f"node:{node.id}", f"Gate '{node.id}': {parity_error}", "high", "gate_route_labels_mismatch")
                            )
                if node.routes is None:
                    errors.append(
                        _err(f"node:{node.id}", f"Gate '{node.id}' is missing required field 'routes'.", "high", "gate_missing_routes")
                    )
            elif node.node_type == "transform":
                # Negative constraints — transforms must not have gate fields
                if node.condition is not None:
                    errors.append(
                        _err(
                            f"node:{node.id}",
                            f"Transform '{node.id}' must not have 'condition' field.",
                            "high",
                            "transform_unexpected_condition",
                        )
                    )
                if node.routes is not None:
                    errors.append(
                        _err(
                            f"node:{node.id}", f"Transform '{node.id}' must not have 'routes' field.", "high", "transform_unexpected_routes"
                        )
                    )
                # Positive constraints — engine requires these as non-empty strings
                # (TransformSettings.plugin, .on_success, .on_error in config.py
                #  — field validators call .strip() and reject empty/blank)
                if not node.plugin:
                    errors.append(
                        _err(
                            f"node:{node.id}",
                            f"Transform '{node.id}' is missing required field 'plugin'.",
                            "high",
                            "transform_missing_plugin",
                        )
                    )
                if not node.on_success or not node.on_success.strip():
                    errors.append(
                        _err(
                            f"node:{node.id}",
                            f"Transform '{node.id}' is missing required field 'on_success'.",
                            "high",
                            "transform_missing_on_success",
                        )
                    )
                if not node.on_error or not node.on_error.strip():
                    errors.append(
                        _err(
                            f"node:{node.id}",
                            f"Transform '{node.id}' is missing required field 'on_error'.",
                            "high",
                            "transform_missing_on_error",
                        )
                    )
            elif node.node_type == "coalesce":
                if node.branches is None:
                    errors.append(
                        _err(
                            f"node:{node.id}",
                            f"Coalesce '{node.id}' is missing required field 'branches'.",
                            "high",
                            "coalesce_missing_branches",
                        )
                    )
                if node.policy is None:
                    errors.append(
                        _err(
                            f"node:{node.id}",
                            f"Coalesce '{node.id}' is missing required field 'policy'.",
                            "high",
                            "coalesce_missing_policy",
                        )
                    )
            elif node.node_type == "aggregation":
                if not node.plugin:
                    errors.append(_err(f"node:{node.id}", f"Aggregation '{node.id}' is missing required field 'plugin'.", "high"))
                # Engine requires on_error as non-empty string
                # (AggregationSettings.on_error in config.py)
                if not node.on_error or not node.on_error.strip():
                    errors.append(_err(f"node:{node.id}", f"Aggregation '{node.id}' is missing required field 'on_error'.", "high"))
                # Runtime treats a missing/empty trigger as end-of-source-only.
                # If early triggers are present, validate them through the same
                # TriggerConfig parser used by settings load.
                if node.trigger is not None:
                    trigger_error = _validate_aggregation_trigger(node.id, node.trigger)
                    if trigger_error is not None:
                        errors.append(trigger_error)
                # output_mode must be a valid OutputMode value when present
                if node.output_mode is not None and node.output_mode not in ("passthrough", "transform"):
                    errors.append(
                        _err(
                            f"node:{node.id}",
                            f"Aggregation '{node.id}' output_mode must be 'passthrough' or 'transform', got '{node.output_mode}'.",
                            "high",
                        )
                    )
            elif node.node_type == "queue":
                # Intrinsic (topology-free) queue shape: id == input, no
                # plugin/routing, description-only options (elspeth-a5b86149d4).
                # Producer/consumer/namespace checks run in the queue-structure
                # block after connection completeness.
                queue_error = queue_node_contract_error(node)
                if queue_error is not None:
                    errors.append(_err(f"node:{node.id}", queue_error, "high"))

        errors.extend(_validate_runtime_route_destinations(self.sources, self.nodes, self.outputs))

        # 8. Connection completeness
        runtime_connections = _runtime_connection_targets(self.sources, self.nodes)
        for node in self.nodes:
            if node.node_type == "coalesce":
                missing_branches = sorted(
                    branch for branch in _coalesce_branch_connections(node.branches) if branch not in runtime_connections
                )
                if missing_branches:
                    errors.append(
                        _err(
                            f"node:{node.id}",
                            f"Coalesce '{node.id}' branches {missing_branches} are not reachable from any runtime connection.",
                            "high",
                        )
                    )
                continue

            if node.input not in runtime_connections:
                errors.append(
                    _err(
                        f"node:{node.id}",
                        f"Node '{node.id}' input '{node.input}' is not reachable from any runtime connection "
                        "(source.on_success, node.on_success/on_error, routes, or fork_to).",
                        "high",
                    )
                )

        # Structural queue topology (elspeth-a5b86149d4). At-least-one-producer
        # is covered by the input-reachability check above (a queue's input is
        # its id, which its producers publish to); more-than-one ordinary
        # consumer is covered by the duplicate-consumer check. Here we require
        # exactly one downstream consumer (reject zero) and a name disjoint from
        # the source keys and the reserved source producer namespace, mirroring
        # the runtime's global source/queue name uniqueness. Sink-name disjoint-
        # ness rides the existing connection/sink overlap check via the queue's
        # consumer claim.
        sink_output_names = {output.name for output in self.outputs}
        for node in self.nodes:
            if node.node_type != "queue":
                continue
            ordinary_consumers = [n.id for n in self.nodes if n.node_type != "queue" and n.input == node.id]
            if not ordinary_consumers:
                errors.append(
                    _err(
                        f"node:{node.id}",
                        f"Queue '{node.id}' has no downstream consumer; a queue must feed exactly one ordinary node.",
                        "high",
                    )
                )
            if node.id in self.sources:
                errors.append(
                    _err(
                        f"node:{node.id}",
                        f"Queue '{node.id}' collides with a source of the same name; source and queue names must be globally unique.",
                        "high",
                    )
                )
            if node.id == "source" or node.id.startswith("source:"):
                # Mirrors _producer_resolver.is_source_producer_id — a queue may
                # not shadow the reserved source producer namespace.
                errors.append(
                    _err(
                        f"node:{node.id}",
                        f"Queue '{node.id}' uses the reserved source producer namespace ('source' / 'source:<name>').",
                        "high",
                    )
                )
            if node.id in sink_output_names:
                errors.append(
                    _err(
                        f"node:{node.id}",
                        f"Queue '{node.id}' collides with a sink of the same name; connection and sink names must be disjoint.",
                        "high",
                    )
                )

        # Generic semantic-contract check.
        from elspeth.web.composer._semantic_validator import validate_semantic_contracts

        semantic_errors, semantic_contracts = validate_semantic_contracts(self)
        errors.extend(semantic_errors)

        numeric_contract_errors, numeric_contract_warnings = _batch_distribution_profile_value_field_entries(self.sources, self.nodes)
        errors.extend(numeric_contract_errors)

        # --- Warnings (advisory, non-blocking) ---
        warnings: list[ValidationEntry] = []
        _warn = ValidationEntry
        warnings.extend(numeric_contract_warnings)
        from elspeth.web.interpretation_state import prompt_shield_recommendation_warning_pairs

        for component, message in prompt_shield_recommendation_warning_pairs(self):
            warnings.append(_warn(component, message, "medium"))

        # Build connection-field targets (wiring that doesn't require edges)
        connection_targets = _runtime_connection_targets(self.sources, self.nodes)

        # W1: Output has no runtime routing reference (on_success / on_error / routes)
        # Edges are UI-only — generate_yaml() uses only connection fields,
        # so an edge to a sink without a matching connection field is a
        # false positive for reachability.
        #
        # Also count implicit engine-level routes: on_validation_failure
        # and on_write_failure route data to outputs without explicit
        # connection fields.
        implicit_targets: set[str] = set()
        for source in self.sources.values():
            if source.on_validation_failure != "discard":
                implicit_targets.add(source.on_validation_failure)
        for output in self.outputs:
            if output.on_write_failure != "discard":
                implicit_targets.add(output.on_write_failure)
        for output in self.outputs:
            if output.name not in connection_targets and output.name not in implicit_targets:
                warnings.append(
                    _warn(
                        f"output:{output.name}",
                        f"Output '{output.name}' is not referenced by any on_success, on_error, or route — it will never receive data.",
                        "medium",
                    )
                )

        # W2: Source on_success target doesn't match any node input or output name
        node_inputs = {n.input for n in self.nodes if n.input is not None}
        for source_name, source in self.sources.items():
            source_on_success = source.on_success
            if source_on_success not in node_inputs and source_on_success not in output_names:
                component = "source" if source_name == "source" else f"source:{source_name}"
                message = (
                    f"Source on_success '{source_on_success}' does not match any node input or output — data may not flow."
                    if source_name == "source"
                    else f"Source '{source_name}' on_success '{source_on_success}' does not match any node input or output — data may not flow."
                )
                warnings.append(
                    _warn(
                        component,
                        message,
                        "medium",
                    )
                )

        # W3: Node has no outgoing edges and no connection-field targets
        edge_sources = {e.from_node for e in self.edges}
        for node in self.nodes:
            has_edge_out = node.id in edge_sources
            has_connection_out = (
                node.on_success is not None or node.on_error is not None or (node.routes is not None and len(node.routes) > 0)
            )
            if not has_edge_out and not has_connection_out:
                warnings.append(
                    _warn(
                        f"node:{node.id}",
                        f"Node '{node.id}' has no outgoing edges — its output is not connected to any downstream node or sink.",
                        "medium",
                    )
                )

        # W4: Sink plugin/filename extension mismatch
        _plugin_exts: dict[str, set[str]] = {
            "csv": {".csv"},
            "json": {".json", ".jsonl"},
            "jsonl": {".jsonl"},
        }
        for output in self.outputs:
            if "path" not in output.options:
                continue
            path_val = output.options["path"]
            if type(path_val) is not str:
                continue

            ext = PurePosixPath(path_val).suffix.lower()
            if output.plugin not in _plugin_exts:
                continue
            accepted = _plugin_exts[output.plugin]
            if ext and ext not in accepted:
                warnings.append(
                    _warn(
                        f"output:{output.name}",
                        f"Output '{output.name}' uses plugin '{output.plugin}' but filename extension suggests a different format.",
                        "low",
                    )
                )

        # W5: Transform/aggregation node has empty or incomplete options
        # These plugins require configuration to do anything useful.
        _plugins_requiring_config: dict[str, tuple[str, str]] = {
            "value_transform": ("operations", "no operations defined — nothing will be computed"),
            "type_coerce": ("conversions", "no conversions defined — no types will be changed"),
            "llm": ("prompt_template", "no prompt_template defined — nothing will be sent to the model"),
            "field_mapper": ("mapping", "no mapping defined — no fields will be renamed"),
            "truncate": ("fields", "no fields specified — nothing will be truncated"),
            "keyword_filter": ("keywords", "no keywords defined — all rows will pass through"),
            "web_scrape": ("url_field", "no url_field specified — cannot determine which field contains URLs"),
            "json_explode": ("field", "no field specified — cannot determine which field to explode"),
        }
        for node in self.nodes:
            if node.plugin in _plugins_requiring_config:
                required_key, reason = _plugins_requiring_config[node.plugin]
                if not node.options or required_key not in node.options:
                    warnings.append(
                        _warn(
                            f"node:{node.id}",
                            f"Transform '{node.id}' ({node.plugin}) appears incomplete: {reason}.",
                            "medium",
                        )
                    )
                # Also check for empty list/dict/tuple values (lists are frozen to tuples)
                elif node.options[required_key] in ([], (), {}, None, ""):
                    warnings.append(
                        _warn(
                            f"node:{node.id}",
                            f"Transform '{node.id}' ({node.plugin}) has empty '{required_key}': {reason}.",
                            "medium",
                        )
                    )

        # W6: File sink missing required path
        for output in self.outputs:
            if output.plugin in FILE_SINK_PLUGINS:
                if not output.options or "path" not in output.options:
                    warnings.append(
                        _warn(
                            f"output:{output.name}",
                            f"Output '{output.name}' ({output.plugin}) has no path configured — cannot write to file.",
                            "medium",
                        )
                    )
                elif not output.options["path"]:
                    warnings.append(
                        _warn(
                            f"output:{output.name}",
                            f"Output '{output.name}' ({output.plugin}) has empty path — cannot write to file.",
                            "medium",
                        )
                    )

        # W7: on_write_failure reference validation
        # Mirrors rules from engine/orchestrator/validation.py so LLMs get
        # early feedback instead of failing at pipeline build time.
        _failsink_eligible = FAILSINK_ELIGIBLE_SINK_PLUGINS
        output_name_set = {o.name for o in self.outputs}
        output_by_name = {o.name: o for o in self.outputs}
        for output in self.outputs:
            dest = output.on_write_failure
            if dest == "discard":
                continue
            # Rule 2: must reference an existing output
            if dest not in output_name_set:
                warnings.append(
                    _warn(
                        f"output:{output.name}",
                        f"Output '{output.name}' on_write_failure references '{dest}' which is not a configured output.",
                        "high",
                    )
                )
                continue  # Skip dependent checks
            # Rule 3: no self-reference
            if dest == output.name:
                warnings.append(
                    _warn(
                        f"output:{output.name}",
                        f"Output '{output.name}' on_write_failure references itself — a sink cannot be its own failsink.",
                        "high",
                    )
                )
                continue
            # Rule 4: target must use an eligible file plugin
            target = output_by_name[dest]
            if target.plugin not in _failsink_eligible:
                warnings.append(
                    _warn(
                        f"output:{output.name}",
                        f"Output '{output.name}' on_write_failure references '{dest}' (plugin='{target.plugin}'), but failsinks must use {FAILSINK_ELIGIBLE_PLUGIN_TEXT}.",
                        "medium",
                    )
                )
            # Rule 5: no chains — target must use 'discard'
            if target.on_write_failure != "discard":
                warnings.append(
                    _warn(
                        f"output:{output.name}",
                        f"Output '{output.name}' on_write_failure references '{dest}', but '{dest}' has on_write_failure='{target.on_write_failure}' — failsink targets must use 'discard' (no chains).",
                        "medium",
                    )
                )

        # W8: Source on_validation_failure reference validation
        # Mirrors rules from engine/orchestrator/validation.py so LLMs get
        # early feedback instead of failing at pipeline build time.
        for source_name, source in self.sources.items():
            vf_dest = source.on_validation_failure
            if vf_dest != "discard" and vf_dest not in output_name_set:
                component = "source" if source_name == "source" else f"source:{source_name}"
                if source_name == "source":
                    message = (
                        f"Source on_validation_failure references '{vf_dest}' which is not a configured output — "
                        "validation failures will cause a pipeline build error."
                    )
                else:
                    message = (
                        f"Source '{source_name}' on_validation_failure references '{vf_dest}' which is not a configured output — "
                        "validation failures will cause a pipeline build error."
                    )
                warnings.append(
                    _warn(
                        component,
                        message,
                        "high",
                    )
                )

        # --- Suggestions (optional improvements) ---
        suggestions: list[ValidationEntry] = []
        _sug = ValidationEntry

        # S1: No error routing
        has_gate = any(n.node_type == "gate" for n in self.nodes)
        has_error_routing = any(e.edge_type == "on_error" for e in self.edges) or any(n.on_error is not None for n in self.nodes)
        if not has_gate and not has_error_routing and self.nodes:
            suggestions.append(
                _sug("pipeline", "Consider adding error routing — rows that fail transforms currently have no explicit destination.", "low")
            )

        # S2: Single output to external sink — suggest a local fallback
        # Local file sinks don't benefit from a backup:
        # if the filesystem is failing, a second file will fail too.
        # External sinks (database, azure_blob, dataverse, http) benefit from a
        # local recovery file when the external system is unavailable.
        if len(self.outputs) == 1:
            output = self.outputs[0]
            if output.plugin not in LOCAL_RECOVERY_SINK_PLUGINS:
                suggestions.append(
                    _sug(
                        "pipeline",
                        f"Single external output ('{output.plugin}'). Consider adding a local file output for recovery if the external system is unavailable.",
                        "low",
                    )
                )

        # S3: Source has no schema under the current composer/plugin config contract
        for source_name, source in self.sources.items():
            has_schema = _source_options_have_schema(source.options)
            if not has_schema:
                component = "source" if source_name == "source" else f"source:{source_name}"
                message = (
                    "Source has no explicit schema. Downstream field references depend on runtime column names."
                    if source_name == "source"
                    else f"Source '{source_name}' has no explicit schema. Downstream field references depend on runtime column names."
                )
                suggestions.append(_sug(component, message, "low"))

        # 9. Schema contract validation
        contract_errors, contract_warnings, edge_contracts = _check_schema_contracts(self.sources, self.nodes, self.outputs)
        errors.extend(contract_errors)
        warnings.extend(contract_warnings)

        return ValidationSummary(
            is_valid=len(errors) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
            suggestions=tuple(suggestions),
            edge_contracts=edge_contracts,
            semantic_contracts=semantic_contracts,
        )
