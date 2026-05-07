"""Tests for the shared producer-map resolver primitive."""

from __future__ import annotations

from elspeth.web.composer._producer_resolver import ProducerResolver
from elspeth.web.composer.state import NodeSpec, SourceSpec


def _node(
    node_id: str,
    *,
    plugin: str | None,
    node_type: str = "transform",
    input: str = "",
    on_success: str | None = None,
    on_error: str | None = None,
    options: dict | None = None,
    routes: dict[str, str] | None = None,
    fork_to: tuple[str, ...] | None = None,
) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type=node_type,
        plugin=plugin,
        input=input,
        on_success=on_success,
        on_error=on_error,
        options=options or {},
        condition=None,
        routes=routes,
        fork_to=fork_to,
        branches=None,
        policy=None,
        merge=None,
    )


class TestProducerResolverBuild:
    def test_source_registers_as_producer_for_on_success(self):
        source = SourceSpec(plugin="csv", on_success="step1", options={}, on_validation_failure="discard")
        nodes = (_node("step1", plugin="t", input="step1", on_success="sink"),)
        resolver = ProducerResolver.build(source=source, nodes=nodes, sink_names=frozenset({"sink"}))

        producer = resolver.find_producer_for("step1")
        assert producer is not None
        assert producer.producer_id == "source"
        assert producer.plugin_name == "csv"

    def test_node_on_success_registers_producer(self):
        nodes = (_node("a", plugin="p1", input="src_out", on_success="b_in"), _node("b", plugin="p2", input="b_in", on_success="sink"))
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset({"sink"}))

        producer = resolver.find_producer_for("b_in")
        assert producer is not None
        assert producer.producer_id == "a"
        assert producer.plugin_name == "p1"

    def test_duplicate_producer_for_connection_is_recorded(self):
        nodes = (_node("a", plugin="p1", input="src", on_success="dup"), _node("b", plugin="p2", input="src", on_success="dup"))
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

        assert "dup" in resolver.duplicate_connections
        assert resolver.find_producer_for("dup") is None  # ambiguous

    def test_routes_register_producers(self):
        nodes = (_node("g", plugin="gate1", node_type="gate", input="src", routes={"yes": "yes_out", "no": "no_out"}),)
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

        for connection in ("yes_out", "no_out"):
            producer = resolver.find_producer_for(connection)
            assert producer is not None and producer.producer_id == "g"

    def test_fork_to_registers_producers(self):
        nodes = (_node("g", plugin="fork1", node_type="gate", input="src", fork_to=("a", "b")),)
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

        for branch in ("a", "b"):
            assert resolver.find_producer_for(branch) is not None

    def test_coalesce_without_on_success_publishes_under_own_id(self):
        nodes = (_node("c", plugin=None, node_type="coalesce", input="branches"),)
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

        producer = resolver.find_producer_for("c")
        assert producer is not None and producer.producer_id == "c"


class TestProducerResolverWalkBack:
    def test_walk_through_gate_returns_real_producer(self):
        nodes = (
            _node("scrape", plugin="web_scrape", input="src_out", on_success="gate_in"),
            _node("g", plugin="gate1", node_type="gate", input="gate_in", on_success="explode_in"),
            _node("explode", plugin="line_explode", input="explode_in", on_success="sink"),
        )
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset({"sink"}))

        producer = resolver.walk_to_real_producer("explode_in")
        assert producer is not None
        assert producer.producer_id == "scrape"
        assert producer.plugin_name == "web_scrape"

    def test_walk_returns_none_on_routing_loop(self):
        nodes = (
            _node("g1", plugin=None, node_type="gate", input="loop_b", on_success="loop_a"),
            _node("g2", plugin=None, node_type="gate", input="loop_a", on_success="loop_b"),
        )
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

        assert resolver.walk_to_real_producer("loop_a") is None

    def test_walk_returns_none_when_connection_is_duplicate(self):
        nodes = (_node("a", plugin="p1", input="src", on_success="dup"), _node("b", plugin="p2", input="src", on_success="dup"))
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

        assert resolver.walk_to_real_producer("dup") is None

    def test_walk_returns_none_when_connection_unknown(self):
        nodes: tuple[NodeSpec, ...] = ()
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

        assert resolver.walk_to_real_producer("nope") is None

    def test_walk_returns_source_producer_without_node_lookup(self):
        # Reviewer-found bug: walk_to_real_producer must NOT index
        # _node_by_id["source"]. Source producers must short-circuit
        # before any node-table lookup.
        source = SourceSpec(
            plugin="csv",
            on_success="step1",
            options={
                "path": "x.csv",
                "schema": {"mode": "fixed", "fields": ["url: str"]},
            },
            on_validation_failure="quarantine",
        )
        resolver = ProducerResolver.build(
            source=source,
            nodes=(),
            sink_names=frozenset(),
        )
        producer = resolver.walk_to_real_producer("step1")
        assert producer is not None
        assert producer.producer_id == "source"
        assert producer.plugin_name == "csv"

    def test_walk_through_gate_to_source(self):
        source = SourceSpec(
            plugin="csv",
            on_success="gate_in",
            options={
                "path": "x.csv",
                "schema": {"mode": "fixed", "fields": ["url: str"]},
            },
            on_validation_failure="quarantine",
        )
        nodes = (_node("g", plugin="gate1", node_type="gate", input="gate_in", on_success="explode_in"),)
        resolver = ProducerResolver.build(
            source=source,
            nodes=nodes,
            sink_names=frozenset(),
        )
        producer = resolver.walk_to_real_producer("explode_in")
        assert producer is not None
        assert producer.producer_id == "source"
