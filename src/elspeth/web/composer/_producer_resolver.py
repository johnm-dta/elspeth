"""Shared producer-map and walk-back primitive.

Both the schema-contract validator and the semantic-contract validator
need to: (1) build a map from connection name to producer node, (2) walk
back through structural gates to find the real producer of a connection.
This module provides the single implementation. Pass-through propagation
is intentionally NOT included — that remains schema-specific in
state.py because semantic validation does not propagate through
pass-through transforms in Phase 1.

Layer: L3 (web composer application code). Imports state types from
the same layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from elspeth.web.composer.state import NodeSpec, SourceSpec


@dataclass(frozen=True, slots=True)
class ProducerEntry:
    """A producer registered against one or more connection names.

    options is the producer's raw options Mapping — NOT deep-frozen here
    because state.py already deep-freezes node options in __post_init__.
    """

    producer_id: str
    plugin_name: str | None
    options: Mapping[str, Any]


class ProducerResolver:
    """Builds and queries the connection -> producer map for a composition.

    Construction is via ``build(...)`` rather than ``__init__`` so the
    primitive can compute and report duplicates as part of its result.
    Once built, ``find_producer_for`` and ``walk_to_real_producer`` are
    pure functions of the resolver state.

    NOT a frozen dataclass: holds derived dicts/sets that are
    construction-time fixed but expensive to deep-freeze. Treat as
    effectively immutable — do not mutate the public attributes.
    """

    __slots__ = (
        "_node_by_id",
        "_producer_map",
        "duplicate_connections",
    )

    def __init__(
        self,
        producer_map: dict[str, ProducerEntry],
        node_by_id: dict[str, NodeSpec],
        duplicate_connections: frozenset[str],
    ) -> None:
        self._producer_map = producer_map
        self._node_by_id = node_by_id
        self.duplicate_connections = duplicate_connections

    @classmethod
    def build(
        cls,
        *,
        source: SourceSpec | None,
        nodes: tuple[NodeSpec, ...],
        sink_names: frozenset[str],
    ) -> ProducerResolver:
        producer_map: dict[str, ProducerEntry] = {}
        duplicates: set[str] = set()
        node_by_id = {node.id: node for node in nodes}

        def register(connection_name: str | None, entry: ProducerEntry) -> None:
            if connection_name is None or connection_name == "discard":
                return
            if connection_name in sink_names:
                # Direct-to-sink edges aren't producers for downstream
                # walk-back; schema-contract code handles them separately.
                return
            if connection_name in producer_map:
                # Same node registering multiple times against the same
                # connection (e.g. a gate with two route labels both
                # mapping to the same target) is idempotent, not a
                # duplicate. Only record duplicates when distinct
                # producers contend for the connection.
                if producer_map[connection_name].producer_id == entry.producer_id:
                    return
                duplicates.add(connection_name)
                return
            producer_map[connection_name] = entry

        if source is not None:
            register(
                source.on_success,
                ProducerEntry(
                    producer_id="source",
                    plugin_name=source.plugin,
                    options=source.options,
                ),
            )

        for node in nodes:
            entry = ProducerEntry(
                producer_id=node.id,
                plugin_name=node.plugin,
                options=node.options,
            )
            if node.node_type == "coalesce" and node.on_success is None:
                register(node.id, entry)
            else:
                register(node.on_success, entry)
            register(node.on_error, entry)
            if node.routes is not None:
                for target in node.routes.values():
                    register(target, entry)
            if node.fork_to is not None:
                for target in node.fork_to:
                    register(target, entry)

        return cls(producer_map, node_by_id, frozenset(duplicates))

    def find_producer_for(self, connection_name: str) -> ProducerEntry | None:
        """Return the immediate producer for a connection, or None.

        Returns None for: unknown connection, duplicate (ambiguous)
        connection, or a connection produced only by a direct-to-sink edge.
        """
        if connection_name in self.duplicate_connections:
            return None
        if connection_name not in self._producer_map:
            return None
        return self._producer_map[connection_name]

    def walk_to_real_producer(self, connection_name: str) -> ProducerEntry | None:
        """Walk back through structural gates to the true producer.

        Returns None on: unknown connection, duplicate connection,
        routing loop, or any structural node that semantic walk-back
        does not traverse (currently: coalesce — its branch semantics
        are handled by callers that need them).

        Source producers (producer_id == "source") return immediately
        WITHOUT a node-table lookup. The source is registered in
        _producer_map but is intentionally absent from _node_by_id
        (it is not a NodeSpec). Any code path that called
        _node_by_id[producer.producer_id] for the source would raise
        KeyError — short-circuit here is load-bearing.
        """
        current = connection_name
        visited: set[str] = set()
        while True:
            if current in visited:
                return None
            visited.add(current)
            if current in self.duplicate_connections:
                return None
            if current not in self._producer_map:
                return None
            producer = self._producer_map[current]
            if producer.producer_id == "source":
                return producer
            producer_node = self._node_by_id[producer.producer_id]
            if producer_node.node_type == "gate":
                current = producer_node.input
                continue
            return producer

    def get_node(self, node_id: str) -> NodeSpec | None:
        """Return the registered NodeSpec for a producer id, or None.

        Returns None when the id is "source" (the source is intentionally
        not a NodeSpec) or when the id is unknown. Schema-contract code
        interpreting source-as-producer must short-circuit on None
        rather than indexing the underlying dict.
        """
        if node_id not in self._node_by_id:
            return None
        return self._node_by_id[node_id]
