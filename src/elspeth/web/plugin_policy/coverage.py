"""Queue-aware typed graph coverage for required web controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from jinja2 import TemplateSyntaxError

from elspeth.contracts.enums import Determinism
from elspeth.contracts.freeze import freeze_fields
from elspeth.contracts.plugin_capabilities import ControlRole, PluginCapability
from elspeth.core.templates import extract_jinja2_field_usage
from elspeth.plugins.infrastructure.manager import PluginNotFoundError, get_shared_plugin_manager
from elspeth.web.composer.state import CompositionState, NodeSpec

_NON_PRODUCED_ROUTE_TARGETS = frozenset({"discard", "fork", "stop"})


class _ControlEffectEvaluator(Protocol):
    @classmethod
    def is_effective_blocking_control(
        cls,
        *,
        capability: PluginCapability,
        role: ControlRole,
        options: Mapping[str, object],
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class OutputStreamGraph:
    """Canonical producer/consumer indexes for authored stream names."""

    producers_by_stream: Mapping[str, tuple[NodeSpec, ...]]
    queue_predecessors: Mapping[str, tuple[NodeSpec, ...]]
    consumers_by_stream: Mapping[str, tuple[NodeSpec, ...]]

    def __post_init__(self) -> None:
        freeze_fields(self, "producers_by_stream", "queue_predecessors", "consumers_by_stream")


@dataclass(frozen=True, slots=True)
class ControlCoverageFinding:
    component_id: str
    capability: PluginCapability
    role: ControlRole
    reason: Literal["input_not_dominated", "output_not_post_dominated"]


def build_output_stream_graph(nodes: Sequence[NodeSpec]) -> OutputStreamGraph:
    """Build insertion-order-independent queue-aware stream indexes."""
    queue_ids = {node.id for node in nodes if node.node_type == "queue"}
    ordinary: dict[str, dict[str, NodeSpec]] = {}
    queue_predecessors: dict[str, dict[str, NodeSpec]] = {queue_id: {} for queue_id in queue_ids}
    consumers: dict[str, dict[str, NodeSpec]] = {}

    def register(stream: str | None, producer: NodeSpec) -> None:
        if stream is None or stream == "" or stream in _NON_PRODUCED_ROUTE_TARGETS:
            return
        if stream in queue_ids and producer.id != stream:
            queue_predecessors[stream].setdefault(producer.id, producer)
            return
        ordinary.setdefault(stream, {}).setdefault(producer.id, producer)

    for node in nodes:
        for stream in _node_output_streams(node):
            register(stream, node)
        for stream in _node_input_streams(node):
            consumers.setdefault(stream, {}).setdefault(node.id, node)

    for node in nodes:
        if node.node_type == "queue":
            ordinary[node.id] = {node.id: node}

    return OutputStreamGraph(
        producers_by_stream={stream: tuple(entries[key] for key in sorted(entries)) for stream, entries in ordinary.items()},
        queue_predecessors={queue_id: tuple(entries[key] for key in sorted(entries)) for queue_id, entries in queue_predecessors.items()},
        consumers_by_stream={stream: tuple(entries[key] for key in sorted(entries)) for stream, entries in consumers.items()},
    )


def node_has_blocking_control(
    node: NodeSpec,
    capability: PluginCapability,
    role: ControlRole,
    *,
    protected_fields: frozenset[str] | None = None,
) -> bool:
    """Credit only registered typed blocking controls with effective config."""
    if node.plugin is None:
        return False
    try:
        plugin_cls = cast(
            "type[_ControlEffectEvaluator]",
            get_shared_plugin_manager().get_transform_by_name(node.plugin),
        )
    except PluginNotFoundError:
        return False
    effective = plugin_cls.is_effective_blocking_control(
        capability=capability,
        role=role,
        options=node.options,
    )
    if not effective:
        return False
    return protected_fields is None or _control_covers_fields(node, protected_fields)


def _control_covers_fields(node: NodeSpec, protected_fields: frozenset[str]) -> bool:
    """Return whether a control scans every field whose content it protects."""
    if not protected_fields:
        return False
    configured = node.options.get("fields")
    if configured == "all":
        return True
    if isinstance(configured, str):
        scanned_fields = frozenset({configured}) if configured.strip() else frozenset()
    elif isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        if any(not isinstance(field, str) or not field.strip() for field in configured):
            return False
        scanned_fields = frozenset(configured)
    else:
        return False
    return protected_fields.issubset(scanned_fields)


def _llm_input_fields(node: NodeSpec) -> frozenset[str]:
    """Return row fields interpolated into LLM prompts, or empty when unprovable."""
    prompt_fields = _template_input_fields(node.options.get("prompt_template"))
    if prompt_fields is None:
        return frozenset()
    fields = set(prompt_fields)

    queries = node.options.get("queries")
    if queries is None:
        return frozenset(fields)
    if isinstance(queries, Mapping):
        definitions = tuple(queries.values())
    elif isinstance(queries, Sequence) and not isinstance(queries, (str, bytes)):
        definitions = tuple(queries)
    else:
        return frozenset()

    for definition in definitions:
        if not isinstance(definition, Mapping):
            return frozenset()
        input_fields = definition.get("input_fields")
        if not isinstance(input_fields, Mapping):
            return frozenset()
        row_fields = tuple(input_fields.values())
        if any(not isinstance(field, str) or not field.strip() for field in row_fields):
            return frozenset()
        fields.update(cast("tuple[str, ...]", row_fields))
        template = definition.get("template")
        if template is not None:
            query_template_fields = _template_input_fields(template)
            if query_template_fields is None:
                return frozenset()
            fields.update(query_template_fields)
    return frozenset(fields)


def _template_input_fields(template: object) -> frozenset[str] | None:
    """Extract static row-field accesses; dynamic or malformed templates are unprovable."""
    if not isinstance(template, str):
        return None
    try:
        usage = extract_jinja2_field_usage(template)
    except TemplateSyntaxError:
        return None
    if usage.dynamic_accesses:
        return None
    return usage.fields


def _llm_output_fields(node: NodeSpec) -> frozenset[str]:
    """Return the raw model-response fields emitted by this LLM config."""
    response_field = node.options.get("response_field", "llm_response")
    if not isinstance(response_field, str) or not response_field.strip():
        return frozenset()
    queries = node.options.get("queries")
    if queries is None:
        return frozenset({response_field})

    query_names: list[str] = []
    if isinstance(queries, Mapping):
        query_names.extend(name for name in queries if isinstance(name, str) and name.strip())
        if len(query_names) != len(queries):
            return frozenset()
    elif isinstance(queries, Sequence) and not isinstance(queries, (str, bytes)):
        for query in queries:
            if not isinstance(query, Mapping):
                return frozenset()
            name = query.get("name")
            if not isinstance(name, str) or not name.strip():
                return frozenset()
            query_names.append(name)
    else:
        return frozenset()
    return frozenset(f"{name}_{response_field}" for name in query_names)


def node_has_capability(node: NodeSpec, capability: PluginCapability) -> bool:
    if node.plugin is None:
        return False
    try:
        plugin_cls = get_shared_plugin_manager().get_transform_by_name(node.plugin)
    except PluginNotFoundError:
        return False
    return any(declaration.capability is capability for declaration in plugin_cls.policy_capabilities)


def control_coverage_findings(
    state: CompositionState,
    capability: PluginCapability,
) -> tuple[ControlCoverageFinding, ...]:
    """Return one stable finding for each LLM lacking required coverage."""
    if capability not in (PluginCapability.PROMPT_SHIELD, PluginCapability.CONTENT_SAFETY):
        return ()
    graph = build_output_stream_graph(state.nodes)
    source_streams = frozenset(source.on_success for source in state.sources.values())
    sink_streams = frozenset(output.name for output in state.outputs)
    findings: list[ControlCoverageFinding] = []
    for node in state.nodes:
        if not node_has_capability(node, PluginCapability.LLM):
            continue
        if capability is PluginCapability.PROMPT_SHIELD:
            protected_fields = _llm_input_fields(node)
            covered = _stream_proves_input_control(
                node.input,
                graph,
                source_streams=source_streams,
                visited=frozenset(),
                protected_fields=protected_fields,
            )
            if not covered:
                findings.append(
                    ControlCoverageFinding(
                        component_id=node.id,
                        capability=capability,
                        role=ControlRole.INPUT,
                        reason="input_not_dominated",
                    )
                )
        else:
            protected_fields = _llm_output_fields(node)
            outputs = _node_output_streams(node)
            covered = bool(outputs) and all(
                _stream_proves_output_control(
                    stream,
                    graph,
                    sink_streams=sink_streams,
                    visited=frozenset({node.id}),
                    protected_fields=protected_fields,
                )
                for stream in outputs
            )
            if not covered:
                findings.append(
                    ControlCoverageFinding(
                        component_id=node.id,
                        capability=capability,
                        role=ControlRole.OUTPUT,
                        reason="output_not_post_dominated",
                    )
                )
    return tuple(findings)


def _node_output_streams(node: NodeSpec) -> tuple[str, ...]:
    streams: list[str] = []
    if node.on_success:
        streams.append(node.on_success)
    elif node.node_type == "coalesce":
        # Runtime publishes a non-terminal coalesce under its own name.
        streams.append(node.id)
    if node.on_error:
        streams.append(node.on_error)
    if node.routes:
        streams.extend(node.routes.values())
    if node.fork_to:
        streams.extend(node.fork_to)
    return tuple(dict.fromkeys(streams))


def _node_input_streams(node: NodeSpec) -> tuple[str, ...]:
    if node.node_type == "queue":
        return ()
    if node.node_type == "coalesce":
        if isinstance(node.branches, Mapping):
            return tuple(node.branches.values())
        return node.branches or ()
    return (node.input,) if node.input else ()


def _translate_protected_fields_through_mapper(
    node: NodeSpec,
    protected_fields: frozenset[str],
    *,
    direction: Literal["upstream", "downstream"],
) -> frozenset[str] | None:
    """Translate protected field names across an exact field-mapper node.

    ``None`` means the mapper configuration or resulting field lineage is
    unprovable, so required-control coverage must fail closed on that path.
    """
    if node.plugin != "field_mapper":
        return protected_fields

    configured_mapping = node.options.get("mapping", {})
    select_only = node.options.get("select_only", False)
    if not isinstance(configured_mapping, Mapping) or not isinstance(select_only, bool):
        return None

    mapping: dict[str, str] = {}
    for source, target in configured_mapping.items():
        if not isinstance(source, str) or not source.strip() or not isinstance(target, str) or not target.strip():
            return None
        mapping[source] = target

    targets = tuple(mapping.values())
    sources = frozenset(mapping)
    if len(frozenset(targets)) != len(targets):
        return None
    if any(source != target and target in sources for source, target in mapping.items()):
        return None

    translated: set[str] = set()
    if direction == "upstream":
        source_by_target = {target: source for source, target in mapping.items()}
        for field in protected_fields:
            if field in source_by_target:
                source = source_by_target[field]
                if "." in source:
                    # FieldMapper resolves dotted sources by nested traversal,
                    # while controls scan exact top-level row keys.
                    return None
                translated.add(source)
            elif field in sources:
                if "." in field and not select_only:
                    # Nested extraction does not remove a same-named literal
                    # top-level key from the passthrough row.
                    translated.add(field)
                else:
                    return None
            elif select_only:
                return None
            else:
                translated.add(field)
    else:
        target_fields = frozenset(targets)
        for field in protected_fields:
            if field in mapping:
                target = mapping[field]
                if "." in field:
                    translated.add(target)
                    if not select_only:
                        # Exact dotted keys are copied to the target but are
                        # not deleted from a passthrough row.
                        translated.add(field)
                else:
                    translated.add(target)
            elif field in target_fields or select_only:
                return None
            else:
                translated.add(field)
    return frozenset(translated)


def _stream_proves_input_control(
    stream: str | None,
    graph: OutputStreamGraph,
    *,
    source_streams: frozenset[str],
    visited: frozenset[str],
    protected_fields: frozenset[str],
) -> bool:
    if not isinstance(stream, str) or not stream:
        return False
    producers = graph.producers_by_stream.get(stream)
    if producers:
        return all(
            _producer_proves_input_control(
                producer,
                graph,
                source_streams=source_streams,
                visited=visited,
                protected_fields=protected_fields,
            )
            for producer in producers
        )
    # Source rows and missing/unknown producers are untrusted.
    return False


def _producer_proves_input_control(
    producer: NodeSpec,
    graph: OutputStreamGraph,
    *,
    source_streams: frozenset[str],
    visited: frozenset[str],
    protected_fields: frozenset[str],
) -> bool:
    if producer.id in visited:
        return False
    visited = visited | {producer.id}
    if node_has_blocking_control(
        producer,
        PluginCapability.PROMPT_SHIELD,
        ControlRole.INPUT,
        protected_fields=protected_fields,
    ):
        return True
    if producer.node_type == "queue":
        predecessors = graph.queue_predecessors.get(producer.id, ())
        return bool(predecessors) and all(
            _producer_proves_input_control(
                predecessor,
                graph,
                source_streams=source_streams,
                visited=visited,
                protected_fields=protected_fields,
            )
            for predecessor in predecessors
        )
    if producer.plugin is None:
        return False
    try:
        plugin_cls = get_shared_plugin_manager().get_transform_by_name(producer.plugin)
    except PluginNotFoundError:
        return False
    if plugin_cls.determinism is Determinism.EXTERNAL_CALL:
        return False
    translated_fields = _translate_protected_fields_through_mapper(
        producer,
        protected_fields,
        direction="upstream",
    )
    if translated_fields is None:
        return False
    return _stream_proves_input_control(
        producer.input,
        graph,
        source_streams=source_streams,
        visited=visited,
        protected_fields=translated_fields,
    )


def _stream_proves_output_control(
    stream: str,
    graph: OutputStreamGraph,
    *,
    sink_streams: frozenset[str],
    visited: frozenset[str],
    protected_fields: frozenset[str],
) -> bool:
    if stream in _NON_PRODUCED_ROUTE_TARGETS:
        return True
    if stream in sink_streams:
        return False
    consumers = graph.consumers_by_stream.get(stream)
    if not consumers:
        return False
    return all(
        _consumer_proves_output_control(
            consumer,
            graph,
            sink_streams=sink_streams,
            visited=visited,
            protected_fields=protected_fields,
        )
        for consumer in consumers
    )


def _consumer_proves_output_control(
    consumer: NodeSpec,
    graph: OutputStreamGraph,
    *,
    sink_streams: frozenset[str],
    visited: frozenset[str],
    protected_fields: frozenset[str],
) -> bool:
    if consumer.id in visited:
        return False
    visited = visited | {consumer.id}
    if node_has_blocking_control(
        consumer,
        PluginCapability.CONTENT_SAFETY,
        ControlRole.OUTPUT,
        protected_fields=protected_fields,
    ):
        return True
    translated_fields = _translate_protected_fields_through_mapper(
        consumer,
        protected_fields,
        direction="downstream",
    )
    if translated_fields is None:
        return False
    outputs = _node_output_streams(consumer)
    return bool(outputs) and all(
        _stream_proves_output_control(
            stream,
            graph,
            sink_streams=sink_streams,
            visited=visited,
            protected_fields=translated_fields,
        )
        for stream in outputs
    )
