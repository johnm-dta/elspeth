"""Queue-aware typed graph coverage for required web controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from elspeth.contracts.enums import Determinism
from elspeth.contracts.freeze import freeze_fields
from elspeth.contracts.plugin_capabilities import ControlRole, PluginCapability
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
    return plugin_cls.is_effective_blocking_control(
        capability=capability,
        role=role,
        options=node.options,
    )


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
            covered = _stream_proves_input_control(
                node.input,
                graph,
                source_streams=source_streams,
                visited=frozenset(),
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
            outputs = _node_output_streams(node)
            covered = bool(outputs) and all(
                _stream_proves_output_control(
                    stream,
                    graph,
                    sink_streams=sink_streams,
                    visited=frozenset({node.id}),
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


def _stream_proves_input_control(
    stream: str | None,
    graph: OutputStreamGraph,
    *,
    source_streams: frozenset[str],
    visited: frozenset[str],
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
) -> bool:
    if producer.id in visited:
        return False
    visited = visited | {producer.id}
    if node_has_blocking_control(producer, PluginCapability.PROMPT_SHIELD, ControlRole.INPUT):
        return True
    if producer.node_type == "queue":
        predecessors = graph.queue_predecessors.get(producer.id, ())
        return bool(predecessors) and all(
            _producer_proves_input_control(
                predecessor,
                graph,
                source_streams=source_streams,
                visited=visited,
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
    return _stream_proves_input_control(
        producer.input,
        graph,
        source_streams=source_streams,
        visited=visited,
    )


def _stream_proves_output_control(
    stream: str,
    graph: OutputStreamGraph,
    *,
    sink_streams: frozenset[str],
    visited: frozenset[str],
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
        )
        for consumer in consumers
    )


def _consumer_proves_output_control(
    consumer: NodeSpec,
    graph: OutputStreamGraph,
    *,
    sink_streams: frozenset[str],
    visited: frozenset[str],
) -> bool:
    if consumer.id in visited:
        return False
    visited = visited | {consumer.id}
    if node_has_blocking_control(consumer, PluginCapability.CONTENT_SAFETY, ControlRole.OUTPUT):
        return True
    outputs = _node_output_streams(consumer)
    return bool(outputs) and all(
        _stream_proves_output_control(
            stream,
            graph,
            sink_streams=sink_streams,
            visited=visited,
        )
        for stream in outputs
    )
