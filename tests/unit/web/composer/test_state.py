"""Tests for CompositionState and supporting data models."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from elspeth.plugins.infrastructure.templates import TemplateError
from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    EdgeType,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
    ValidationEntry,
    ValidationSummary,
)


class TestSourceSpec:
    def test_frozen(self) -> None:
        s = SourceSpec(plugin="csv", on_success="t1", options={}, on_validation_failure="discard")
        with pytest.raises(AttributeError):
            s.plugin = "json"  # type: ignore[misc]

    def test_options_deep_frozen(self) -> None:
        s = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"nested": {"key": "val"}},
            on_validation_failure="discard",
        )
        with pytest.raises(TypeError):
            s.options["new"] = "x"  # type: ignore[index]

    def test_options_nested_frozen(self) -> None:
        s = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"nested": {"key": "val"}},
            on_validation_failure="discard",
        )
        with pytest.raises(TypeError):
            s.options["nested"]["mutate"] = "x"

    def test_from_dict_round_trip(self) -> None:
        s = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"nested": {"key": "val"}},
            on_validation_failure="quarantine",
        )
        restored = SourceSpec.from_dict(
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"nested": {"key": "val"}},
                "on_validation_failure": "quarantine",
            }
        )
        assert restored == s


class TestCompositionStateNamedSources:
    def _source(self, plugin: str, on_success: str) -> SourceSpec:
        return SourceSpec(
            plugin=plugin,
            on_success=on_success,
            options={"schema": {"mode": "observed"}},
            on_validation_failure="discard",
        )

    def test_sources_mapping_preserves_named_source_order_without_singular_facade(self) -> None:
        state = CompositionState(
            source=None,
            sources={
                "customers": self._source("csv", "customer_rows"),
                "orders": self._source("json", "order_rows"),
            },
            nodes=(),
            edges=(),
            outputs=(OutputSpec(name="customer_rows", plugin="json", options={}, on_write_failure="discard"),),
            metadata=PipelineMetadata(),
            version=1,
        )

        assert tuple(state.sources) == ("customers", "orders")
        assert state.to_dict()["sources"]["orders"]["on_success"] == "order_rows"
        assert not hasattr(state, "source")

    def test_named_source_mutations_add_update_and_remove_one_source(self) -> None:
        state = CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)

        state = state.with_named_source("customers", self._source("csv", "customer_rows"))
        state = state.with_named_source("orders", self._source("json", "order_rows"))
        updated = state.with_named_source("customers", self._source("csv", "updated_customer_rows"))
        removed = updated.without_named_source("orders")

        assert tuple(updated.sources) == ("customers", "orders")
        assert updated.sources["customers"].on_success == "updated_customer_rows"
        assert tuple(removed.sources) == ("customers",)
        assert removed.sources["customers"].on_success == "updated_customer_rows"

    def test_from_dict_restores_sources_mapping(self) -> None:
        original = CompositionState(
            source=None,
            sources={"customers": self._source("csv", "customer_rows"), "orders": self._source("json", "order_rows")},
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=7,
        )

        restored = CompositionState.from_dict(original.to_dict())

        assert restored == original

    def test_validation_warnings_and_suggestions_cover_all_named_sources(self) -> None:
        """Named-source advisory checks must not stop at the compatibility source."""
        state = CompositionState(
            source=None,
            sources={
                "customers": self._source("csv", "customer_rows"),
                "orders": SourceSpec(
                    plugin="json",
                    on_success="order_rows",
                    options={"path": "/data/orders.json"},
                    on_validation_failure="missing_failures",
                ),
            },
            nodes=(),
            edges=(),
            outputs=(OutputSpec(name="customer_rows", plugin="json", options={}, on_write_failure="discard"),),
            metadata=PipelineMetadata(),
            version=1,
        )

        result = state.validate()

        assert any(w.component == "source:orders" and "on_validation_failure" in w.message for w in result.warnings)
        assert any(s.component == "source:orders" and "no explicit schema" in s.message for s in result.suggestions)

    def test_sources_mapping_is_the_only_domain_and_serialized_source_shape(self) -> None:
        """CompositionState must not expose a singular first-source facade."""
        state = CompositionState(
            sources={
                "customers": self._source("csv", "customer_rows"),
                "orders": self._source("json", "order_rows"),
            },
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

        serialized = state.to_dict()
        restored = CompositionState.from_dict(serialized)

        assert not hasattr(state, "source")
        assert "source" not in serialized
        assert tuple(restored.sources) == ("customers", "orders")


class TestNodeSpec:
    def _make_transform(self, **overrides: Any) -> NodeSpec:
        defaults: dict[str, Any] = {
            "id": "transform_1",
            "node_type": "transform",
            "plugin": "passthrough",
            "input": "source_out",
            "on_success": "sink_main",
            "on_error": None,
            "options": {"field": "name"},
            "condition": None,
            "routes": None,
            "fork_to": None,
            "branches": None,
            "policy": None,
            "merge": None,
        }
        defaults.update(overrides)
        return NodeSpec(**defaults)

    def _make_gate(self, **overrides: Any) -> NodeSpec:
        defaults: dict[str, Any] = {
            "id": "gate_1",
            "node_type": "gate",
            "plugin": None,
            "input": "source_out",
            "on_success": None,
            "on_error": None,
            "options": {},
            "condition": "row['score'] >= 0.5",
            "routes": {"high": "sink_good", "low": "sink_bad"},
            "fork_to": None,
            "branches": None,
            "policy": None,
            "merge": None,
        }
        defaults.update(overrides)
        return NodeSpec(**defaults)

    def test_frozen(self) -> None:
        n = self._make_transform()
        with pytest.raises(AttributeError):
            n.id = "new_id"  # type: ignore[misc]

    def test_options_deep_frozen(self) -> None:
        n = self._make_transform(options={"nested": {"k": "v"}})
        with pytest.raises(TypeError):
            n.options["new"] = 1  # type: ignore[index]

    def test_routes_deep_frozen(self) -> None:
        n = self._make_gate()
        with pytest.raises(TypeError):
            n.routes["extra"] = "val"  # type: ignore[index]

    def test_fork_to_is_tuple(self) -> None:
        n = self._make_gate(fork_to=("path_a", "path_b"))
        assert isinstance(n.fork_to, tuple)
        assert n.fork_to == ("path_a", "path_b")

    def test_branches_is_tuple(self) -> None:
        n = NodeSpec(
            id="coal_1",
            node_type="coalesce",
            plugin=None,
            input="join_point",
            on_success="sink_main",
            on_error=None,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=("path_a", "path_b"),
            policy="require_all",
            merge="nested",
        )
        assert isinstance(n.branches, tuple)

    def test_from_dict_with_optional_fields(self) -> None:
        """from_dict reconstructs optional fields; missing ones default to None."""
        d = {
            "id": "g1",
            "node_type": "gate",
            "plugin": None,
            "input": "in",
            "on_success": None,
            "on_error": None,
            "options": {},
            "condition": "row['x'] > 1",
            "routes": {"high": "s1"},
            "fork_to": ["path_a", "path_b"],
        }
        n = NodeSpec.from_dict(d)
        assert n.condition == "row['x'] > 1"
        assert n.fork_to == ("path_a", "path_b")
        assert n.branches is None
        assert n.policy is None
        assert n.merge is None

    def test_from_dict_converts_list_to_tuple(self) -> None:
        """to_dict() serialises tuples as lists; from_dict() must convert back."""
        d: dict[str, object] = {
            "id": "c1",
            "node_type": "coalesce",
            "plugin": None,
            "input": "join",
            "on_success": "out",
            "on_error": None,
            "options": {},
            "branches": ["a", "b"],
            "policy": "require_all",
            "merge": "nested",
        }
        n = NodeSpec.from_dict(d)
        assert isinstance(n.branches, tuple)
        assert n.branches == ("a", "b")


class TestEdgeSpec:
    def test_frozen(self) -> None:
        e = EdgeSpec(
            id="e1",
            from_node="source",
            to_node="t1",
            edge_type="on_success",
            label=None,
        )
        with pytest.raises(AttributeError):
            e.id = "e2"  # type: ignore[misc]

    def test_from_dict_round_trip(self) -> None:
        e = EdgeSpec(
            id="e1",
            from_node="source",
            to_node="t1",
            edge_type="on_success",
            label="main",
        )
        restored = EdgeSpec.from_dict(
            {
                "id": "e1",
                "from_node": "source",
                "to_node": "t1",
                "edge_type": "on_success",
                "label": "main",
            }
        )
        assert restored == e


class TestOutputSpec:
    def test_frozen(self) -> None:
        o = OutputSpec(name="out", plugin="csv", options={}, on_write_failure="discard")
        with pytest.raises(AttributeError):
            o.name = "new"  # type: ignore[misc]

    def test_options_deep_frozen(self) -> None:
        o = OutputSpec(
            name="out",
            plugin="csv",
            options={"nested": {"k": 1}},
            on_write_failure="discard",
        )
        with pytest.raises(TypeError):
            o.options["new"] = 2  # type: ignore[index]

    def test_from_dict_round_trip(self) -> None:
        o = OutputSpec(
            name="out",
            plugin="csv",
            options={"path": "/out.csv"},
            on_write_failure="quarantine",
        )
        restored = OutputSpec.from_dict(
            {
                "name": "out",
                "plugin": "csv",
                "options": {"path": "/out.csv"},
                "on_write_failure": "quarantine",
            }
        )
        assert restored == o


class TestPipelineMetadata:
    def test_frozen(self) -> None:
        m = PipelineMetadata()
        with pytest.raises(AttributeError):
            m.name = "new"  # type: ignore[misc]

    def test_from_dict_round_trip(self) -> None:
        m = PipelineMetadata(name="My Pipeline", description="Desc")
        restored = PipelineMetadata.from_dict(
            {
                "name": "My Pipeline",
                "description": "Desc",
            }
        )
        assert restored == m

    def test_from_dict_crashes_on_missing_fields(self) -> None:
        """Missing fields crash — this is Tier 1 data from to_dict()."""
        with pytest.raises(KeyError):
            PipelineMetadata.from_dict({})


class TestValidationSummary:
    def test_valid(self) -> None:
        v = ValidationSummary(is_valid=True, errors=())
        assert v.is_valid is True
        assert v.errors == ()

    def test_with_errors(self) -> None:
        v = ValidationSummary(is_valid=False, errors=(ValidationEntry("test", "No source configured.", "high"),))
        assert v.is_valid is False
        assert len(v.errors) == 1


class TestEdgeContract:
    def test_frozen(self) -> None:
        from elspeth.web.composer.state import EdgeContract

        ec = EdgeContract(
            from_id="source",
            to_id="add_world",
            producer_guarantees=("text",),
            consumer_requires=("text",),
            missing_fields=(),
            satisfied=True,
        )
        with pytest.raises(AttributeError):
            ec.satisfied = False  # type: ignore[misc]

    def test_to_dict_uses_from_key(self) -> None:
        """EdgeContract.to_dict() serializes from_id as 'from' (JSON key)."""
        from elspeth.web.composer.state import EdgeContract

        ec = EdgeContract(
            from_id="source",
            to_id="add_world",
            producer_guarantees=("text",),
            consumer_requires=("text",),
            missing_fields=(),
            satisfied=True,
        )
        d = ec.to_dict()
        assert d["from"] == "source"
        assert d["to"] == "add_world"
        assert d["producer_guarantees"] == ["text"]
        assert d["consumer_requires"] == ["text"]
        assert d["missing_fields"] == []
        assert d["satisfied"] is True

    def test_to_dict_empty_fields(self) -> None:
        from elspeth.web.composer.state import EdgeContract

        ec = EdgeContract(
            from_id="source",
            to_id="sink",
            producer_guarantees=(),
            consumer_requires=(),
            missing_fields=(),
            satisfied=True,
        )
        d = ec.to_dict()
        assert d["producer_guarantees"] == []
        assert d["consumer_requires"] == []
        assert d["missing_fields"] == []


class TestValidationSummaryEdgeContracts:
    def test_default_empty(self) -> None:
        vs = ValidationSummary(is_valid=True, errors=())
        assert vs.edge_contracts == ()

    def test_with_edge_contracts(self) -> None:
        from elspeth.web.composer.state import EdgeContract

        ec = EdgeContract(
            from_id="source",
            to_id="t1",
            producer_guarantees=("text",),
            consumer_requires=("text",),
            missing_fields=(),
            satisfied=True,
        )
        vs = ValidationSummary(is_valid=True, errors=(), edge_contracts=(ec,))
        assert len(vs.edge_contracts) == 1
        assert vs.edge_contracts[0].satisfied is True


class TestCompositionState:
    def _empty_state(self) -> CompositionState:
        return CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

    def _make_source(self) -> SourceSpec:
        return SourceSpec(
            plugin="csv",
            on_success="transform_1",
            options={"path": "/data/in.csv"},
            on_validation_failure="quarantine",
        )

    def _make_node(self, id: str = "transform_1") -> NodeSpec:
        return NodeSpec(
            id=id,
            node_type="transform",
            plugin="passthrough",
            input="source_out",
            on_success="sink_main",
            on_error=None,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )

    def _make_edge(self, id: str = "e1") -> EdgeSpec:
        return EdgeSpec(
            id=id,
            from_node="source",
            to_node="transform_1",
            edge_type="on_success",
            label=None,
        )

    def _make_output(self, name: str = "main_output") -> OutputSpec:
        return OutputSpec(
            name=name,
            plugin="csv",
            options={"path": "/out.csv"},
            on_write_failure="quarantine",
        )

    # --- Immutability ---

    def test_frozen(self) -> None:
        state = self._empty_state()
        with pytest.raises(AttributeError):
            state.version = 2  # type: ignore[misc]

    def test_nodes_tuple_frozen(self) -> None:
        """nodes is a tuple — cannot append."""
        state = self._empty_state()
        assert isinstance(state.nodes, tuple)

    def test_metadata_frozen(self) -> None:
        """metadata is a frozen dataclass — deep freeze via freeze_fields."""
        state = self._empty_state()
        with pytest.raises(AttributeError):
            state.metadata.name = "mutated"  # type: ignore[misc]

    # --- with_source ---

    def test_with_source_returns_new_instance(self) -> None:
        state = self._empty_state()
        src = self._make_source()
        new_state = state.with_source(src)
        assert new_state is not state
        assert new_state.sources["source"] is src
        assert state.sources == {}  # original unchanged

    def test_with_source_increments_version(self) -> None:
        state = self._empty_state()
        new_state = state.with_source(self._make_source())
        assert new_state.version == 2

    # --- with_node ---

    def test_with_node_adds(self) -> None:
        state = self._empty_state()
        node = self._make_node()
        new_state = state.with_node(node)
        assert len(new_state.nodes) == 1
        assert new_state.nodes[0].id == "transform_1"
        assert new_state.version == 2

    def test_with_node_replaces_existing(self) -> None:
        state = self._empty_state()
        node1 = self._make_node("t1")
        node2 = self._make_node("t1")  # same ID
        state2 = state.with_node(node1)
        state3 = state2.with_node(node2)
        assert len(state3.nodes) == 1
        assert state3.version == 3

    def test_with_node_preserves_order(self) -> None:
        state = self._empty_state()
        state = state.with_node(self._make_node("a"))
        state = state.with_node(self._make_node("b"))
        state = state.with_node(self._make_node("c"))
        assert [n.id for n in state.nodes] == ["a", "b", "c"]

    # --- without_node ---

    def test_without_node_removes(self) -> None:
        state = self._empty_state().with_node(self._make_node("t1"))
        new_state = state.without_node("t1")
        assert new_state is not None
        assert len(new_state.nodes) == 0
        assert new_state.version == 3

    def test_without_node_nonexistent_returns_none(self) -> None:
        state = self._empty_state()
        result = state.without_node("nonexistent")
        assert result is None

    # --- with_edge ---

    def test_with_edge_adds(self) -> None:
        state = self._empty_state()
        edge = self._make_edge()
        new_state = state.with_edge(edge)
        assert len(new_state.edges) == 1
        assert new_state.version == 2

    def test_with_edge_replaces_by_id(self) -> None:
        state = self._empty_state()
        e1 = EdgeSpec(id="e1", from_node="source", to_node="t1", edge_type="on_success", label=None)
        e1_updated = EdgeSpec(id="e1", from_node="source", to_node="t2", edge_type="on_success", label=None)
        state2 = state.with_edge(e1).with_edge(e1_updated)
        assert len(state2.edges) == 1
        assert state2.edges[0].to_node == "t2"

    def test_with_edge_preserves_order(self) -> None:
        """Updating an existing edge must preserve its position, not append."""
        state = self._empty_state()
        e1 = EdgeSpec(id="e1", from_node="source", to_node="t1", edge_type="on_success", label=None)
        e2 = EdgeSpec(id="e2", from_node="t1", to_node="t2", edge_type="on_success", label=None)
        e3 = EdgeSpec(id="e3", from_node="t2", to_node="sink", edge_type="on_success", label=None)
        state = state.with_edge(e1).with_edge(e2).with_edge(e3)
        assert [e.id for e in state.edges] == ["e1", "e2", "e3"]

        # Update e2 — should stay at index 1, not move to end
        e2_updated = EdgeSpec(id="e2", from_node="t1", to_node="t2_new", edge_type="on_success", label="updated")
        updated = state.with_edge(e2_updated)
        assert [e.id for e in updated.edges] == ["e1", "e2", "e3"]
        assert updated.edges[1].to_node == "t2_new"
        assert updated.edges[1].label == "updated"

    # --- without_edge ---

    def test_without_edge_removes(self) -> None:
        state = self._empty_state().with_edge(self._make_edge("e1"))
        new_state = state.without_edge("e1")
        assert new_state is not None
        assert len(new_state.edges) == 0

    def test_without_edge_nonexistent_returns_none(self) -> None:
        state = self._empty_state()
        result = state.without_edge("nonexistent")
        assert result is None

    # --- with_output ---

    def test_with_output_adds(self) -> None:
        state = self._empty_state()
        output = self._make_output()
        new_state = state.with_output(output)
        assert len(new_state.outputs) == 1
        assert new_state.version == 2

    def test_with_output_replaces_by_name(self) -> None:
        state = self._empty_state()
        o1 = self._make_output("out")
        o2 = OutputSpec(name="out", plugin="json", options={}, on_write_failure="discard")
        state2 = state.with_output(o1).with_output(o2)
        assert len(state2.outputs) == 1
        assert state2.outputs[0].plugin == "json"

    def test_with_output_preserves_order(self) -> None:
        """Updating an existing output must preserve its position, not append."""
        state = self._empty_state()
        o1 = self._make_output("alpha")
        o2 = self._make_output("beta")
        o3 = self._make_output("gamma")
        state = state.with_output(o1).with_output(o2).with_output(o3)
        assert [o.name for o in state.outputs] == ["alpha", "beta", "gamma"]

        # Update beta — should stay at index 1, not move to end
        o2_updated = OutputSpec(name="beta", plugin="json", options={"format": "lines"}, on_write_failure="discard")
        updated = state.with_output(o2_updated)
        assert [o.name for o in updated.outputs] == ["alpha", "beta", "gamma"]
        assert updated.outputs[1].plugin == "json"

    # --- without_output ---

    def test_without_output_removes(self) -> None:
        state = self._empty_state().with_output(self._make_output("out"))
        new_state = state.without_output("out")
        assert new_state is not None
        assert len(new_state.outputs) == 0

    def test_without_output_nonexistent_returns_none(self) -> None:
        result = self._empty_state().without_output("nope")
        assert result is None

    # --- with_metadata ---

    def test_with_metadata_partial_update(self) -> None:
        state = self._empty_state()
        new_state = state.with_metadata({"name": "My Pipeline"})
        assert new_state.metadata.name == "My Pipeline"
        assert new_state.metadata.description == ""  # unchanged
        assert new_state.version == 2

    def test_with_metadata_full_update(self) -> None:
        state = self._empty_state()
        new_state = state.with_metadata({"name": "P1", "description": "Desc"})
        assert new_state.metadata.name == "P1"
        assert new_state.metadata.description == "Desc"

    # --- to_dict ---

    def test_to_dict_unwraps_frozen_containers(self) -> None:
        """to_dict() converts MappingProxyType -> dict and tuple -> list."""
        state = self._empty_state()
        src = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"nested": {"k": "v"}},
            on_validation_failure="discard",
        )
        state = state.with_source(src)
        state = state.with_node(self._make_node("t1"))
        state = state.with_output(self._make_output("out"))

        d = state.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["nodes"], list)
        assert isinstance(d["sources"]["source"]["options"], dict)
        assert isinstance(d["sources"]["source"]["options"]["nested"], dict)
        assert isinstance(d["outputs"], list)

    def test_to_dict_roundtrip_yaml(self) -> None:
        """to_dict() output is yaml.dump()-safe (no MappingProxyType errors)."""
        import yaml

        state = self._empty_state()
        src = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"nested": {"deep": {"k": "v"}}},
            on_validation_failure="quarantine",
        )
        state = state.with_source(src)
        d = state.to_dict()
        yaml_str = yaml.dump(d, default_flow_style=False)
        assert "csv" in yaml_str

    def test_mutation_refreezes_containers(self) -> None:
        """Mutation methods must re-freeze since dataclasses.replace() skips __post_init__."""
        state = self._empty_state()
        src = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"nested": {"k": "v"}},
            on_validation_failure="discard",
        )
        new_state = state.with_source(src)
        assert isinstance(new_state.nodes, tuple)
        with pytest.raises(TypeError):
            new_state.sources["source"].options["new"] = "x"  # type: ignore[index]

    # --- from_dict round-trip ---

    def test_from_dict_round_trip_empty(self) -> None:
        """Empty state round-trips through to_dict/from_dict."""
        state = self._empty_state()
        restored = CompositionState.from_dict(state.to_dict())
        assert restored == state

    def test_from_dict_round_trip_fully_populated(self) -> None:
        """Fully populated state round-trips through to_dict/from_dict."""
        gate = NodeSpec(
            id="gate_1",
            node_type="gate",
            plugin=None,
            input="source_out",
            on_success=None,
            on_error=None,
            options={},
            condition="row['score'] >= 0.5",
            routes={"high": "sink_good", "low": "sink_bad"},
            fork_to=("path_a", "path_b"),
            branches=None,
            policy=None,
            merge=None,
        )
        coalesce = NodeSpec(
            id="coal_1",
            node_type="coalesce",
            plugin=None,
            input="join_point",
            on_success="main_output",
            on_error=None,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=("path_a", "path_b"),
            policy="require_all",
            merge="nested",
        )
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="transform_1",
                options={"path": "/data/in.csv", "nested": {"key": "val"}},
                on_validation_failure="quarantine",
            ),
            nodes=(self._make_node("transform_1"), gate, coalesce),
            edges=(
                self._make_edge("e1"),
                EdgeSpec(id="e2", from_node="gate_1", to_node="sink_good", edge_type="route_true", label="high"),
            ),
            outputs=(
                self._make_output("main_output"),
                OutputSpec(name="sink_good", plugin="json", options={"indent": 2}, on_write_failure="discard"),
            ),
            metadata=PipelineMetadata(name="Test Pipeline", description="A fully populated test state"),
            version=42,
        )
        restored = CompositionState.from_dict(state.to_dict())
        assert restored == state

    def test_from_dict_round_trip_none_optional_fields(self) -> None:
        """NodeSpec optional fields omitted by to_dict() reconstruct as None."""
        node = self._make_node("t1")
        state = self._empty_state().with_node(node)
        restored = CompositionState.from_dict(state.to_dict())
        restored_node = restored.nodes[0]
        assert restored_node.condition is None
        assert restored_node.routes is None
        assert restored_node.fork_to is None
        assert restored_node.branches is None
        assert restored_node.policy is None
        assert restored_node.merge is None

    def test_from_dict_containers_are_frozen(self) -> None:
        """from_dict() output has deep-frozen containers (not plain dicts)."""
        state = self._empty_state()
        src = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"nested": {"k": "v"}},
            on_validation_failure="discard",
        )
        state = state.with_source(src)
        restored = CompositionState.from_dict(state.to_dict())
        assert restored.sources["source"].options is not None
        with pytest.raises(TypeError):
            restored.sources["source"].options["new"] = "x"  # type: ignore[index]
        with pytest.raises(TypeError):
            restored.sources["source"].options["nested"]["mutate"] = "y"


class TestStage1Validation:
    def _empty_state(self) -> CompositionState:
        return CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

    def _make_source(self, on_success: str = "t1", on_validation_failure: str = "quarantine") -> SourceSpec:
        return SourceSpec(
            plugin="csv",
            on_success=on_success,
            options={},
            on_validation_failure=on_validation_failure,
        )

    def _make_transform(
        self,
        id: str,
        input: str,
        on_success: str,
        on_error: str = "discard",
    ) -> NodeSpec:
        return NodeSpec(
            id=id,
            node_type="transform",
            plugin="passthrough",
            input=input,
            on_success=on_success,
            on_error=on_error,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )

    def _make_output(self, name: str = "main") -> OutputSpec:
        return OutputSpec(name=name, plugin="csv", options={}, on_write_failure="discard")

    def _make_edge(
        self,
        id: str,
        from_node: str,
        to_node: str,
        edge_type: EdgeType = "on_success",
    ) -> EdgeSpec:
        return EdgeSpec(id=id, from_node=from_node, to_node=to_node, edge_type=edge_type, label=None)

    def _coalesce_route_state(self, *, on_success: str | None) -> CompositionState:
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="gate_in"))
        state = state.with_node(
            NodeSpec(
                id="fork_gate",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"true": "fork", "false": "fork"},
                fork_to=("path_a", "path_b"),
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_node(
            NodeSpec(
                id="merge_point",
                node_type="coalesce",
                plugin=None,
                input="join",
                on_success=on_success,
                on_error=None,
                options={},
                condition=None,
                routes=None,
                fork_to=None,
                branches=("path_a", "path_b"),
                policy="require_all",
                merge="nested",
            )
        )
        return state.with_output(self._make_output("main"))

    def test_empty_state_has_errors(self) -> None:
        result = self._empty_state().validate()
        assert not result.is_valid
        assert any(e.message == "No source configured." for e in result.errors)
        assert any(e.message == "No sinks configured." for e in result.errors)

    def test_minimal_valid_pipeline(self) -> None:
        """source -> transform -> sink, fully connected."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(self._make_output("main"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert result.is_valid, result.errors

    def test_connection_only_runtime_pipeline_is_valid_without_ui_edges(self) -> None:
        """Runtime connection fields, not UI edges, determine Stage 1 validity."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(self._make_output("main"))

        result = state.validate()

        assert result.is_valid, result.errors

    def test_connection_only_coalesce_pipeline_is_valid_without_ui_edges(self) -> None:
        """Coalesce terminal routes are valid when declared in runtime fields."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="gate_in"))
        state = state.with_node(
            NodeSpec(
                id="fork_gate",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"true": "fork", "false": "fork"},
                fork_to=("path_a", "path_b"),
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_node(
            NodeSpec(
                id="merge_point",
                node_type="coalesce",
                plugin=None,
                input="join",
                on_success="main",
                on_error=None,
                options={},
                condition=None,
                routes=None,
                fork_to=None,
                branches=("path_a", "path_b"),
                policy="require_all",
                merge="nested",
            )
        )
        state = state.with_output(self._make_output("main"))

        result = state.validate()

        assert result.is_valid, result.errors

    def test_gate_route_to_discard_is_valid_without_output_named_discard(self) -> None:
        """Gate routes may target virtual 'discard' without declaring a sink."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="gate_in"))
        state = state.with_node(
            NodeSpec(
                id="quality_gate",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="row['keep']",
                routes={"true": "main", "false": "discard"},
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(self._make_output("main"))

        result = state.validate()

        assert result.is_valid, result.errors

    def test_dangling_edge_from_node(self) -> None:
        state = self._empty_state()
        state = state.with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "nonexistent", "main"))
        result = state.validate()
        assert not result.is_valid
        assert any("nonexistent" in e.message and "from_node" in e.message for e in result.errors)

    def test_dangling_edge_to_node(self) -> None:
        state = self._empty_state()
        state = state.with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "nonexistent"))
        result = state.validate()
        assert not result.is_valid
        assert any("nonexistent" in e.message and "to_node" in e.message for e in result.errors)

    def test_duplicate_node_ids(self) -> None:
        """Two nodes with same id — caught by validation, not by with_node (which replaces)."""
        node = self._make_transform("dup", "in", "out")
        state = CompositionState(
            source=self._make_source(),
            nodes=(node, node),
            edges=(),
            outputs=(self._make_output(),),
            metadata=PipelineMetadata(),
            version=1,
        )
        result = state.validate()
        assert not result.is_valid
        assert any("Duplicate node ID" in e.message for e in result.errors)

    def test_duplicate_output_names(self) -> None:
        out = self._make_output("dup")
        state = CompositionState(
            source=self._make_source(),
            nodes=(),
            edges=(),
            outputs=(out, out),
            metadata=PipelineMetadata(),
            version=1,
        )
        result = state.validate()
        assert not result.is_valid
        assert any("Duplicate output name" in e.message for e in result.errors)

    def test_duplicate_edge_ids(self) -> None:
        edge = self._make_edge("dup", "source", "main")
        state = CompositionState(
            source=self._make_source(),
            nodes=(),
            edges=(edge, edge),
            outputs=(self._make_output(),),
            metadata=PipelineMetadata(),
            version=1,
        )
        result = state.validate()
        assert not result.is_valid
        assert any("Duplicate edge ID" in e.message for e in result.errors)

    def test_gate_missing_condition(self) -> None:
        gate = NodeSpec(
            id="g1",
            node_type="gate",
            plugin=None,
            input="in",
            on_success=None,
            on_error=None,
            options={},
            condition=None,
            routes={"high": "s1"},
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(gate)
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        result = state.validate()
        assert not result.is_valid
        assert any("condition" in e.message for e in result.errors)

    def test_gate_malformed_condition_syntax_error(self) -> None:
        """validate() catches condition with invalid Python syntax."""
        gate = NodeSpec(
            id="g1",
            node_type="gate",
            plugin=None,
            input="in",
            on_success=None,
            on_error=None,
            options={},
            condition="row['x'] >== 5",
            routes={"true": "s1", "false": "s2"},
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(gate)
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        result = state.validate()
        assert not result.is_valid
        assert any("Invalid gate condition syntax" in e.message for e in result.errors)

    def test_gate_injection_condition_security_error(self) -> None:
        """validate() catches injection attempts in conditions."""
        gate = NodeSpec(
            id="g1",
            node_type="gate",
            plugin=None,
            input="in",
            on_success=None,
            on_error=None,
            options={},
            condition="__import__('os').system('rm -rf /')",
            routes={"true": "s1", "false": "s2"},
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(gate)
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        result = state.validate()
        assert not result.is_valid
        assert any("Forbidden construct in gate condition" in e.message for e in result.errors)

    def test_gate_forbidden_function_call_condition(self) -> None:
        """validate() catches forbidden function calls (eval, exec, etc.)."""
        gate = NodeSpec(
            id="g1",
            node_type="gate",
            plugin=None,
            input="in",
            on_success=None,
            on_error=None,
            options={},
            condition="eval('1+1')",
            routes={"true": "s1", "false": "s2"},
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(gate)
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        result = state.validate()
        assert not result.is_valid
        assert any("Forbidden construct in gate condition" in e.message for e in result.errors)

    def test_gate_valid_condition_passes_validation(self) -> None:
        """validate() accepts well-formed gate conditions."""
        gate = NodeSpec(
            id="g1",
            node_type="gate",
            plugin=None,
            input="in",
            on_success=None,
            on_error=None,
            options={},
            condition="row['score'] >= 0.85",
            routes={"true": "s1", "false": "s2"},
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(gate)
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        result = state.validate()
        # Only structural errors may remain (connection completeness etc.),
        # but no expression-related errors
        expr_errors = [e for e in result.errors if "gate condition" in e.message.lower()]
        assert expr_errors == []

    def test_gate_lambda_condition_rejected(self) -> None:
        """validate() catches lambda expressions in conditions."""
        gate = NodeSpec(
            id="g1",
            node_type="gate",
            plugin=None,
            input="in",
            on_success=None,
            on_error=None,
            options={},
            condition="lambda: True",
            routes={"true": "s1", "false": "s2"},
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(gate)
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        result = state.validate()
        assert not result.is_valid
        assert any("Forbidden construct in gate condition" in e.message for e in result.errors)

    def test_gate_comprehension_condition_rejected(self) -> None:
        """validate() catches list comprehensions in conditions."""
        gate = NodeSpec(
            id="g1",
            node_type="gate",
            plugin=None,
            input="in",
            on_success=None,
            on_error=None,
            options={},
            condition="[x for x in range(10)]",
            routes={"true": "s1", "false": "s2"},
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(gate)
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        result = state.validate()
        assert not result.is_valid
        assert any("Forbidden construct in gate condition" in e.message for e in result.errors)

    def test_gate_missing_routes(self) -> None:
        gate = NodeSpec(
            id="g1",
            node_type="gate",
            plugin=None,
            input="in",
            on_success=None,
            on_error=None,
            options={},
            condition="row['x'] > 1",
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(gate)
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        result = state.validate()
        assert not result.is_valid
        assert any("routes" in e.message for e in result.errors)

    def test_transform_with_condition_is_error(self) -> None:
        node = NodeSpec(
            id="t1",
            node_type="transform",
            plugin="passthrough",
            input="in",
            on_success="out",
            on_error=None,
            options={},
            condition="row['x'] > 1",
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(node)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("condition" in e.message for e in result.errors)

    def test_coalesce_missing_branches(self) -> None:
        node = NodeSpec(
            id="c1",
            node_type="coalesce",
            plugin=None,
            input="join",
            on_success="out",
            on_error=None,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy="require_all",
            merge="nested",
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(node)
        state = state.with_edge(self._make_edge("e1", "source", "c1"))
        result = state.validate()
        assert not result.is_valid
        assert any("branches" in e.message for e in result.errors)

    def test_aggregation_missing_plugin(self) -> None:
        node = NodeSpec(
            id="a1",
            node_type="aggregation",
            plugin=None,
            input="in",
            on_success="out",
            on_error=None,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(node)
        state = state.with_edge(self._make_edge("e1", "source", "a1"))
        result = state.validate()
        assert not result.is_valid
        assert any("plugin" in e.message for e in result.errors)

    def test_unknown_node_type_is_invalid(self) -> None:
        """Stage 1 must reject node types outside the closed runtime set."""
        node = NodeSpec.from_dict(
            {
                "id": "mystery",
                "node_type": "bogus",
                "plugin": "passthrough",
                "input": "source_out",
                "on_success": "main",
                "on_error": "discard",
                "options": {},
                "condition": None,
                "routes": None,
                "fork_to": None,
                "branches": None,
                "policy": None,
                "merge": None,
            }
        )
        state = self._empty_state().with_source(self._make_source(on_success="source_out"))
        state = state.with_output(self._make_output())
        state = state.with_node(node)
        state = state.with_edge(self._make_edge("e1", "source", "mystery"))

        result = state.validate()

        assert not result.is_valid
        assert any("unknown node_type 'bogus'" in e.message for e in result.errors)

    def test_unreachable_node(self) -> None:
        """Node exists but no edge points to it and source.on_success doesn't match."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="other"))
        state = state.with_node(self._make_transform("t1", "somewhere", "main"))
        state = state.with_output(self._make_output())
        result = state.validate()
        assert not result.is_valid
        assert any("not reachable" in e.message for e in result.errors)

    def test_validate_after_from_dict_round_trip(self) -> None:
        """W-4A-2: validate() on reconstructed state matches original."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(self._make_output("main"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))

        restored = CompositionState.from_dict(state.to_dict())
        result = restored.validate()
        assert result.is_valid, result.errors

    def test_edge_only_pipeline_is_invalid_when_runtime_connections_do_not_match(self) -> None:
        """UI edges cannot rescue runtime wiring that generate_yaml() will not emit."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="wrong_connection"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(self._make_output("main"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))

        result = state.validate()

        assert not result.is_valid
        assert any("runtime connection" in e.message for e in result.errors)

    @pytest.mark.parametrize(
        ("case_name", "expected_message"),
        [
            (
                "source_on_success",
                "Source on_success 'dangling' is neither a sink nor a known connection",
            ),
            (
                "transform_on_success",
                "Transform 't1' on_success 'dangling' is neither a sink nor a known connection",
            ),
            (
                "aggregation_on_success",
                "Aggregation 'agg1' on_success 'dangling' is neither a sink nor a known connection",
            ),
            (
                "coalesce_unknown_sink",
                "Coalesce 'merge_point' on_success references unknown sink 'dangling'",
            ),
            (
                "coalesce_connection_target",
                "Coalesce 'merge_point' has on_success='next_step'. Coalesce on_success must point to a sink when configured.",
            ),
            (
                "transform_on_error",
                "Transform 't1' on_error 'missing_error_sink' references unknown sink",
            ),
        ],
    )
    def test_validate_rejects_runtime_unresolvable_route_destinations(
        self,
        case_name: str,
        expected_message: str,
    ) -> None:
        """Stage 1 rejects terminal routes that the runtime DAG builder rejects."""
        if case_name == "source_on_success":
            state = self._empty_state().with_source(self._make_source(on_success="dangling")).with_output(self._make_output("main"))
        elif case_name == "transform_on_success":
            state = self._empty_state()
            state = state.with_source(self._make_source(on_success="t1"))
            state = state.with_node(self._make_transform("t1", "t1", "dangling"))
            state = state.with_output(self._make_output("main"))
        elif case_name == "aggregation_on_success":
            state = self._empty_state()
            state = state.with_source(self._make_source(on_success="agg1"))
            state = state.with_node(
                NodeSpec(
                    id="agg1",
                    node_type="aggregation",
                    plugin="batch_counter",
                    input="agg1",
                    on_success="dangling",
                    on_error="discard",
                    options={},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                    trigger={"count": 1},
                )
            )
            state = state.with_output(self._make_output("main"))
        elif case_name == "coalesce_unknown_sink":
            state = self._coalesce_route_state(on_success="dangling")
        elif case_name == "coalesce_connection_target":
            state = self._coalesce_route_state(on_success="next_step")
            state = state.with_node(self._make_transform("after_merge", "next_step", "main"))
        elif case_name == "transform_on_error":
            state = self._empty_state()
            state = state.with_source(self._make_source(on_success="t1"))
            state = state.with_node(self._make_transform("t1", "t1", "main", on_error="missing_error_sink"))
            state = state.with_output(self._make_output("main"))
        else:
            raise AssertionError(f"Unhandled route validation case: {case_name}")

        result = state.validate()

        assert not result.is_valid, case_name
        assert any(expected_message in error.message for error in result.errors), (case_name, result.errors)

    # --- Warning rules (W1-W4) ---

    def test_validate_output_no_incoming_edge_warns(self) -> None:
        """W1: Output with no edge targeting it produces a warning."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(self._make_output("main"))
        state = state.with_output(self._make_output("orphan"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert result.is_valid
        assert any("orphan" in w.message and "never receive data" in w.message for w in result.warnings)

    def test_validate_source_on_success_mismatch_warns(self) -> None:
        """W2: Source on_success doesn't match any node input."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="nonexistent"))
        state = state.with_node(self._make_transform("t1", "other_input", "main"))
        state = state.with_output(self._make_output("main"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert any("nonexistent" in w.message and "does not match" in w.message for w in result.warnings)

    def test_validate_format_extension_mismatch_warns(self) -> None:
        """W4: Sink plugin/filename extension mismatch."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "results"))
        output = OutputSpec(
            name="results",
            plugin="csv",
            options={"path": "/output/data.json"},
            on_write_failure="discard",
        )
        state = state.with_output(output)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "results"))
        result = state.validate()
        assert result.is_valid
        assert any("extension suggests a different format" in w.message for w in result.warnings)

    def test_validate_transform_missing_required_options_warns(self) -> None:
        """W5: Transform that requires config has empty options."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        # value_transform requires 'operations' key
        incomplete_transform = NodeSpec(
            id="t1",
            node_type="transform",
            plugin="value_transform",
            input="t1",
            on_success="main",
            on_error="discard",
            options={},  # Empty - should trigger warning
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = state.with_node(incomplete_transform)
        output = OutputSpec(name="main", plugin="csv", options={"path": "out.csv"}, on_write_failure="discard")
        state = state.with_output(output)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert result.is_valid  # Still structurally valid
        assert any("value_transform" in w.message and "incomplete" in w.message for w in result.warnings)

    def test_validate_transform_empty_operations_warns(self) -> None:
        """W5: Transform has the required key but it's empty."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        # value_transform with empty operations list
        empty_ops_transform = NodeSpec(
            id="t1",
            node_type="transform",
            plugin="value_transform",
            input="t1",
            on_success="main",
            on_error="discard",
            options={"operations": []},  # Empty list - should trigger warning
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = state.with_node(empty_ops_transform)
        output = OutputSpec(name="main", plugin="csv", options={"path": "out.csv"}, on_write_failure="discard")
        state = state.with_output(output)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert result.is_valid
        assert any("value_transform" in w.message and "empty" in w.message for w in result.warnings)

    def test_validate_file_sink_missing_path_warns(self) -> None:
        """W6: File sink without path configured."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        # CSV sink with no path
        no_path_output = OutputSpec(name="main", plugin="csv", options={}, on_write_failure="discard")
        state = state.with_output(no_path_output)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert result.is_valid  # Structurally valid but won't run
        assert any("no path configured" in w.message for w in result.warnings)

    def test_validate_file_sink_empty_path_warns(self) -> None:
        """W6: File sink with empty string path."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        # JSON sink with empty path
        empty_path_output = OutputSpec(name="main", plugin="json", options={"path": ""}, on_write_failure="discard")
        state = state.with_output(empty_path_output)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert result.is_valid
        assert any("empty path" in w.message for w in result.warnings)

    def test_validate_non_file_sink_no_path_ok(self) -> None:
        """Non-file sinks (like database) don't require path."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        # Database sink - path is not a required option
        db_output = OutputSpec(
            name="main", plugin="database", options={"url": "sqlite:///:memory:", "table": "out"}, on_write_failure="discard"
        )
        state = state.with_output(db_output)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        # Should NOT warn about missing path for non-file sinks
        assert not any("no path configured" in w.message for w in result.warnings)

    # --- W7: on_write_failure reference validation ---

    def test_validate_on_write_failure_nonexistent_output_warns(self) -> None:
        """W7: on_write_failure references output that doesn't exist."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        bad_output = OutputSpec(name="main", plugin="csv", options={"path": "/out.csv"}, on_write_failure="nonexistent")
        state = state.with_output(bad_output)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert any("not a configured output" in w.message for w in result.warnings)

    def test_validate_on_write_failure_self_reference_warns(self) -> None:
        """W7: on_write_failure references itself."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        self_ref = OutputSpec(name="main", plugin="csv", options={"path": "/out.csv"}, on_write_failure="main")
        state = state.with_output(self_ref)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert any("references itself" in w.message for w in result.warnings)

    def test_validate_on_write_failure_ineligible_plugin_warns(self) -> None:
        """W7: failsink target uses non-file plugin (e.g. database)."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        main_out = OutputSpec(name="main", plugin="csv", options={"path": "/out.csv"}, on_write_failure="backup")
        backup_out = OutputSpec(
            name="backup", plugin="database", options={"url": "sqlite:///:memory:", "table": "t"}, on_write_failure="discard"
        )
        state = state.with_output(main_out)
        state = state.with_output(backup_out)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert any("must use csv or json" in w.message for w in result.warnings)

    def test_validate_on_write_failure_chain_warns(self) -> None:
        """W7: failsink target has its own non-discard on_write_failure (chain)."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        main_out = OutputSpec(name="main", plugin="csv", options={"path": "/out.csv"}, on_write_failure="errors")
        errors_out = OutputSpec(name="errors", plugin="csv", options={"path": "/errors.csv"}, on_write_failure="overflow")
        overflow_out = OutputSpec(name="overflow", plugin="csv", options={"path": "/overflow.csv"}, on_write_failure="discard")
        state = state.with_output(main_out)
        state = state.with_output(errors_out)
        state = state.with_output(overflow_out)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert any("no chains" in w.message for w in result.warnings)

    def test_validate_on_write_failure_valid_no_warning(self) -> None:
        """W7: Valid failsink reference produces no warning."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        main_out = OutputSpec(name="main", plugin="csv", options={"path": "/out.csv"}, on_write_failure="errors")
        errors_out = OutputSpec(name="errors", plugin="csv", options={"path": "/errors.csv"}, on_write_failure="discard")
        state = state.with_output(main_out)
        state = state.with_output(errors_out)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        # No on_write_failure warnings
        assert not any("on_write_failure" in w.message for w in result.warnings)

    def test_validate_on_write_failure_discard_no_warning(self) -> None:
        """W7: on_write_failure='discard' is always valid, no warning."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(OutputSpec(name="main", plugin="csv", options={"path": "/out.csv"}, on_write_failure="discard"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert not any("on_write_failure" in w.message for w in result.warnings)

    # --- W8: on_validation_failure reference validation ---

    def test_validate_on_validation_failure_nonexistent_output_warns(self) -> None:
        """W8: on_validation_failure references output that doesn't exist."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1", on_validation_failure="nonexistent_sink"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(OutputSpec(name="main", plugin="csv", options={"path": "/out.csv"}, on_write_failure="discard"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert any("not a configured output" in w.message for w in result.warnings)

    def test_validate_on_validation_failure_discard_no_warning(self) -> None:
        """W8: on_validation_failure='discard' is always valid, no warning."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1", on_validation_failure="discard"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(OutputSpec(name="main", plugin="csv", options={"path": "/out.csv"}, on_write_failure="discard"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert not any("on_validation_failure" in w.message for w in result.warnings)

    def test_validate_on_validation_failure_valid_output_no_warning(self) -> None:
        """W8: on_validation_failure references a valid output, no warning."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1", on_validation_failure="quarantine"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(OutputSpec(name="main", plugin="csv", options={"path": "/out.csv"}, on_write_failure="discard"))
        state = state.with_output(
            OutputSpec(name="quarantine", plugin="csv", options={"path": "/quarantine.csv"}, on_write_failure="discard")
        )
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert not any("on_validation_failure" in w.message for w in result.warnings)

    # --- Suggestion rules (S1-S3) ---

    def test_validate_no_error_routing_suggests(self) -> None:
        """S1: Transforms now require on_error (section 7), so a valid pipeline
        always has explicit error routing and S1 cannot fire.  Verify S1 is
        absent when on_error='discard' is set."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main", on_error="discard"))
        state = state.with_output(self._make_output("main"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert result.is_valid
        assert not any("error routing" in s.message for s in result.suggestions)

    def test_validate_single_output_suggests(self) -> None:
        """S2: Pipeline with single EXTERNAL output gets a backup suggestion.

        Local file sinks (csv, json) don't trigger this because if the
        filesystem fails, a backup file will fail too. External sinks
        (database, azure_blob) benefit from a local recovery file.
        """
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        # Use external sink (database) to trigger S2 suggestion
        external_output = OutputSpec(
            name="main",
            plugin="database",
            options={"url": "sqlite:///:memory:", "table": "output"},
            on_write_failure="discard",
        )
        state = state.with_output(external_output)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert any("local file output" in s.message for s in result.suggestions)

    def test_validate_single_file_output_no_suggestion(self) -> None:
        """S2: Pipeline with single LOCAL file output gets no backup suggestion.

        Local file sinks don't benefit from a backup file - if the filesystem
        is failing, the backup will fail too.
        """
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(self._make_output("main"))  # csv = local file sink
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        # Should NOT suggest backup for local file sinks
        assert not any("local file output" in s.message for s in result.suggestions)

    def test_validate_no_schema_config_suggests(self) -> None:
        """S3: Source without schema_config in options gets a suggestion."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(self._make_output("main"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert any("no explicit schema" in s.message for s in result.suggestions)

    def test_validate_schema_alias_suppresses_suggestion(self) -> None:
        """S3: Source with composer-facing ``schema`` alias must not trigger the suggestion.

        The composer/runtime boundary uses ``schema`` (user-facing) while
        plugin config parsing normalizes to ``schema_config``. The detection
        helper must accept either alias so correctly configured sources do
        not draw a false advisory through the LLM prompt.
        """
        source = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"schema": {"mode": "observed"}},
            on_validation_failure="quarantine",
        )
        state = self._empty_state()
        state = state.with_source(source)
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(self._make_output("main"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert not any("no explicit schema" in s.message for s in result.suggestions)

    def test_validate_schema_config_alias_suppresses_suggestion(self) -> None:
        """S3: Source with internal ``schema_config`` alias also suppresses the suggestion.

        Plugin config parsing may land internal shapes in composer state
        (e.g. after serialization round-trips). Both alias names must be
        recognized so internal and external shapes agree.
        """
        source = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"schema_config": {"mode": "observed"}},
            on_validation_failure="quarantine",
        )
        state = self._empty_state()
        state = state.with_source(source)
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(self._make_output("main"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert not any("no explicit schema" in s.message for s in result.suggestions)

    # --- Interaction tests ---

    def test_validate_warnings_dont_block(self) -> None:
        """Warnings don't affect is_valid."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(self._make_output("main"))
        state = state.with_output(self._make_output("orphan"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        assert result.is_valid is True
        assert len(result.warnings) > 0

    def test_validate_errors_and_warnings_coexist(self) -> None:
        """A state with both errors and warnings populates both."""
        state = self._empty_state()
        # No source = error, orphan output = warning
        state = state.with_output(self._make_output("orphan"))
        result = state.validate()
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("never receive data" in w.message for w in result.warnings)

    # --- Mandatory field enforcement (section 7 positive checks) ---

    def test_validate_transform_missing_plugin_errors(self) -> None:
        """Transform with plugin=None must fail validation."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        node = NodeSpec(
            id="t1",
            node_type="transform",
            plugin=None,
            input="t1",
            on_success="main",
            on_error="discard",
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = state.with_node(node)
        state = state.with_output(self._make_output("main"))
        result = state.validate()
        assert not result.is_valid
        assert any("plugin" in e.message.lower() and "t1" in e.message for e in result.errors)

    def test_validate_transform_missing_on_success_errors(self) -> None:
        """Transform with on_success=None must fail validation."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        node = NodeSpec(
            id="t1",
            node_type="transform",
            plugin="passthrough",
            input="t1",
            on_success=None,
            on_error="discard",
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = state.with_node(node)
        state = state.with_output(self._make_output("main"))
        result = state.validate()
        assert not result.is_valid
        assert any("on_success" in e.message and "t1" in e.message for e in result.errors)

    def test_validate_transform_blank_on_success_errors(self) -> None:
        """Transform with on_success='' must fail validation (engine rejects blank)."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        node = NodeSpec(
            id="t1",
            node_type="transform",
            plugin="passthrough",
            input="t1",
            on_success="",
            on_error="discard",
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = state.with_node(node)
        state = state.with_output(self._make_output("main"))
        result = state.validate()
        assert not result.is_valid
        assert any("on_success" in e.message and "t1" in e.message for e in result.errors)

    def test_validate_transform_blank_on_error_errors(self) -> None:
        """Transform with on_error='  ' must fail validation (engine rejects blank)."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        node = NodeSpec(
            id="t1",
            node_type="transform",
            plugin="passthrough",
            input="t1",
            on_success="main",
            on_error="  ",
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = state.with_node(node)
        state = state.with_output(self._make_output("main"))
        result = state.validate()
        assert not result.is_valid
        assert any("on_error" in e.message and "t1" in e.message for e in result.errors)

    def test_validate_transform_missing_on_error_errors(self) -> None:
        """Transform with on_error=None must fail validation."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        node = NodeSpec(
            id="t1",
            node_type="transform",
            plugin="passthrough",
            input="t1",
            on_success="main",
            on_error=None,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = state.with_node(node)
        state = state.with_output(self._make_output("main"))
        result = state.validate()
        assert not result.is_valid
        assert any("on_error" in e.message and "t1" in e.message for e in result.errors)

    def test_validate_aggregation_missing_trigger_is_end_of_source_only(self) -> None:
        """Aggregation with trigger=None means end-of-source-only flush."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="agg1"))
        node = NodeSpec(
            id="agg1",
            node_type="aggregation",
            plugin="batch_counter",
            input="agg1",
            on_success="main",
            on_error="discard",
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
            trigger=None,
        )
        state = state.with_node(node)
        state = state.with_output(self._make_output("main"))
        result = state.validate()
        assert result.is_valid

    def test_validate_aggregation_empty_trigger_is_end_of_source_only(self) -> None:
        """Aggregation with trigger={} means end-of-source-only flush."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="agg1"))
        node = NodeSpec(
            id="agg1",
            node_type="aggregation",
            plugin="batch_counter",
            input="agg1",
            on_success="main",
            on_error="discard",
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
            trigger={},
        )
        state = state.with_node(node)
        state = state.with_output(self._make_output("main"))
        result = state.validate()
        assert result.is_valid

    def test_validate_aggregation_end_of_source_condition_errors(self) -> None:
        """end_of_source must not be accepted in the boolean condition slot."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="agg1"))
        node = NodeSpec(
            id="agg1",
            node_type="aggregation",
            plugin="batch_counter",
            input="agg1",
            on_success="main",
            on_error="discard",
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
            trigger={"condition": "end_of_source"},
        )
        state = state.with_node(node)
        state = state.with_output(self._make_output("main"))
        result = state.validate()
        assert not result.is_valid
        assert any("end_of_source" in e.message and "agg1" in e.message for e in result.errors)

    def test_validate_aggregation_invalid_output_mode_errors(self) -> None:
        """Aggregation with invalid output_mode must fail validation."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="agg1"))
        node = NodeSpec(
            id="agg1",
            node_type="aggregation",
            plugin="batch_counter",
            input="agg1",
            on_success="main",
            on_error="discard",
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
            trigger={"count": 10},
            output_mode="invalid_mode",
        )
        state = state.with_node(node)
        state = state.with_output(self._make_output("main"))
        result = state.validate()
        assert not result.is_valid
        assert any("output_mode" in e.message and "agg1" in e.message for e in result.errors)

    def test_validate_aggregation_with_trigger_passes(self) -> None:
        """Aggregation with all required fields passes validation."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="agg1"))
        node = NodeSpec(
            id="agg1",
            node_type="aggregation",
            plugin="batch_counter",
            input="agg1",
            on_success="main",
            on_error="discard",
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
            trigger={"count": 100},
        )
        state = state.with_node(node)
        state = state.with_output(self._make_output("main"))
        state = state.with_edge(self._make_edge("e1", "source", "agg1"))
        state = state.with_edge(self._make_edge("e2", "agg1", "main"))
        result = state.validate()
        assert result.is_valid, result.errors

    def test_validate_aggregation_missing_on_error_errors(self) -> None:
        """Aggregation with on_error=None must fail validation."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="agg1"))
        node = NodeSpec(
            id="agg1",
            node_type="aggregation",
            plugin="batch_counter",
            input="agg1",
            on_success="main",
            on_error=None,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = state.with_node(node)
        state = state.with_output(self._make_output("main"))
        result = state.validate()
        assert not result.is_valid
        assert any("on_error" in e.message and "agg1" in e.message for e in result.errors)

    def test_validate_clean_pipeline_no_warnings(self) -> None:
        """Well-formed pipeline with gates, error routing, schema, and
        multiple outputs has empty warnings and suggestions."""
        state = self._empty_state()
        source = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"path": "/in.csv", "schema_config": {"fields": []}},
            on_validation_failure="quarantine",
        )
        state = state.with_source(source)
        state = state.with_node(self._make_transform("t1", "t1", "gate_in"))
        gate = NodeSpec(
            id="gate_1",
            node_type="gate",
            plugin=None,
            input="gate_in",
            on_success=None,
            on_error=None,
            options={},
            condition="row['score'] >= 0.5",
            routes={"true": "main", "false": "errors"},
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        state = state.with_node(gate)
        # Use properly configured outputs with paths (W6 semantic completeness)
        main_output = OutputSpec(name="main", plugin="csv", options={"path": "outputs/main.csv"}, on_write_failure="discard")
        errors_output = OutputSpec(name="errors", plugin="csv", options={"path": "outputs/errors.csv"}, on_write_failure="discard")
        quarantine_output = OutputSpec(
            name="quarantine", plugin="csv", options={"path": "outputs/quarantine.csv"}, on_write_failure="discard"
        )
        state = state.with_output(main_output)
        state = state.with_output(errors_output)
        state = state.with_output(quarantine_output)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "gate_1"))
        state = state.with_edge(self._make_edge("e3", "gate_1", "main"))
        state = state.with_edge(self._make_edge("e4", "gate_1", "errors"))
        result = state.validate()
        assert result.is_valid, result.errors
        assert result.warnings == ()
        assert result.suggestions == ()

    def _gate_pipeline(self, *, condition: str, routes: dict[str, str]) -> CompositionState:
        """Minimal source -> gate -> sink pipeline for route-parity checks."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="g1"))
        state = state.with_node(
            NodeSpec(
                id="g1",
                node_type="gate",
                plugin=None,
                input="g1",
                on_success=None,
                on_error=None,
                options={},
                condition=condition,
                routes=routes,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(self._make_output("main"))
        state = state.with_edge(self._make_edge("e0", "source", "g1"))
        return state

    def test_gate_boolean_condition_custom_labels_invalid(self) -> None:
        """Boolean gate condition with non-true/false labels is rejected (parity with GateSettings).

        Regression for elspeth-08e17b9253: composer validate() previously
        green-lit a shape runtime GateSettings.validate_boolean_routes rejects.
        """
        result = self._gate_pipeline(condition="row['x'] > 0", routes={"high": "main", "low": "main"}).validate()
        assert result.is_valid is False
        assert any("boolean condition" in e.message and e.severity == "high" for e in result.errors), [e.message for e in result.errors]

    def test_gate_numeric_condition_invalid(self) -> None:
        """Provably-numeric gate condition can never be a route label; rejected for any labels."""
        result = self._gate_pipeline(condition="row['x'] + 1", routes={"a": "main"}).validate()
        assert result.is_valid is False
        assert any("numeric value" in e.message and e.severity == "high" for e in result.errors), [e.message for e in result.errors]

    def test_gate_boolean_condition_true_false_labels_valid(self) -> None:
        """Boolean gate condition with exactly {true,false} labels stays valid."""
        result = self._gate_pipeline(condition="row['x'] > 0", routes={"true": "main", "false": "main"}).validate()
        assert result.is_valid is True, [e.message for e in result.errors]

    def test_gate_string_route_condition_custom_labels_valid(self) -> None:
        """POSITIVE CONTROL: a string-returning condition with custom labels is NOT over-rejected."""
        result = self._gate_pipeline(
            condition='"high" if row["x"] > 0 else "low"',
            routes={"high": "main", "low": "main"},
        ).validate()
        assert result.is_valid is True, [e.message for e in result.errors]


class TestWebScrapeAbuseContactValidation:
    """Mechanical backstop for skill-prompt rule in pipeline_composer.md.

    Rejects RFC 2606 / RFC 6761 reserved-domain emails in
    `web_scrape.http.abuse_contact`. Pairs with the prompt-level rule that
    forbids fabricating wire-visible identity values; without this validator
    the LLM has unlimited rationalisation room to ship `ops@example.com` and
    similar fabrications. See elspeth-457c8688ef and observation
    obs-69697091d9 for context.
    """

    def _state_with_web_scrape(
        self,
        abuse_contact: str | None,
        *,
        http_present: bool = True,
        plugin: str = "web_scrape",
        options_override: dict[str, Any] | None = None,
    ) -> CompositionState:
        """Build a minimal state with a single transform node carrying the
        given abuse_contact under options.http.

        Other validation rules will report unrelated errors (no source, no
        sinks, etc.); the tests assert only on the abuse_contact rule's
        message presence/absence.
        """
        if options_override is not None:
            options: dict[str, Any] = options_override
        elif http_present:
            http_block: dict[str, Any] = {"scraping_reason": "test", "allowed_hosts": "public_only"}
            if abuse_contact is not None:
                http_block["abuse_contact"] = abuse_contact
            options = {
                "schema": {"mode": "fixed", "fields": ["url: str"]},
                "url_field": "url",
                "content_field": "content",
                "fingerprint_field": "content_fingerprint",
                "format": "markdown",
                "http": http_block,
            }
        else:
            options = {
                "schema": {"mode": "fixed", "fields": ["url: str"]},
                "url_field": "url",
            }
        node = NodeSpec(
            id="fetch_pages",
            node_type="transform",
            plugin=plugin,
            input="url_rows",
            on_success="scraped_content",
            on_error="discard",
            options=options,
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )
        return CompositionState(
            source=None,
            nodes=(node,),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

    def _abuse_contact_error_messages(self, state: CompositionState) -> list[str]:
        return [e.message for e in state.validate().errors if "abuse_contact" in e.message]

    def _web_scrape_identity_error_messages(self, state: CompositionState) -> list[str]:
        return [e.message for e in state.validate().errors if "web_scrape.http." in e.message]

    @pytest.mark.parametrize(
        "address",
        [
            "ops@example.com",
            "compliance@example.com",
            "abuse@example.org",
            "ops@example.net",
            "user@something.test",
            "user@deep.something.test",
            "admin@something.invalid",
            "root@localhost",
            "user@host.localhost",
            "user@something.example",
        ],
    )
    def test_rejects_rfc_reserved_domains(self, address: str) -> None:
        """All RFC 2606/6761 reserved labels and their subdomains must be rejected."""
        state = self._state_with_web_scrape(address)
        messages = self._abuse_contact_error_messages(state)
        assert messages, f"Expected reject for {address!r}, got no abuse_contact error"
        msg = messages[0]
        assert "fabricated identity" in msg
        assert "abuse_contact" in msg

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("abuse_contact", "<OPERATOR_REQUIRED>"),
            ("abuse_contact", "operator required"),
            ("scraping_reason", "<OPERATOR_REQUIRED>"),
            ("scraping_reason", "operator required"),
        ],
    )
    def test_rejects_wire_visible_operator_required_placeholders(self, field_name: str, value: str) -> None:
        """The skill's sentinel values must be blocking validation errors, not persisted defaults."""
        http_block = {
            "abuse_contact": "ops@somecompany.gov.au",
            "scraping_reason": "User-authorised page colour lookup",
            "allowed_hosts": "public_only",
        }
        http_block[field_name] = value
        state = self._state_with_web_scrape(
            None,
            options_override={
                "schema": {"mode": "fixed", "fields": ["url: str"]},
                "url_field": "url",
                "content_field": "content",
                "fingerprint_field": "content_fingerprint",
                "format": "markdown",
                "http": http_block,
            },
        )

        messages = self._web_scrape_identity_error_messages(state)
        assert messages, f"Expected reject for web_scrape.http.{field_name}={value!r}"
        assert f"web_scrape.http.{field_name}" in messages[0]
        assert "placeholder" in messages[0]

    @pytest.mark.parametrize(
        "address",
        [
            "OPS@EXAMPLE.COM",
            "ops@Example.Com",
            "User@SOMETHING.TEST",
        ],
    )
    def test_case_insensitive_reject(self, address: str) -> None:
        """Domain matching must be case-insensitive — uppercase variants are still RFC-reserved."""
        state = self._state_with_web_scrape(address)
        assert self._abuse_contact_error_messages(state), f"Expected reject for {address!r}"

    @pytest.mark.parametrize(
        "address",
        [
            "abuse-contact-unset@elspeth.foundryside.dev",
            "ops@somecompany.gov.au",
            "abuse@example.foundryside.dev",  # 'example' as a label, not the reserved TLD
            "user@reallytest.example-mail.org",  # not endswith ".test" / ".example.org"
            "ops@notlocalhost.com",
            "ops@subdomain.example.io",  # 'example' inside string but not reserved
        ],
    )
    def test_accepts_real_domains(self, address: str) -> None:
        """Real, deliverable domains must pass even when they contain reserved
        labels as substrings (only label-boundary matches count)."""
        state = self._state_with_web_scrape(address)
        assert not self._abuse_contact_error_messages(state), f"Real domain {address!r} was incorrectly rejected"

    def test_skips_non_web_scrape_transform(self) -> None:
        """Rule is plugin-scoped — a passthrough or other transform with an
        accidental http.abuse_contact field is none of this rule's business."""
        state = self._state_with_web_scrape(
            "ops@example.com",
            plugin="passthrough",
        )
        assert not self._abuse_contact_error_messages(state), "Rule should not fire on non-web_scrape plugins"

    def test_skips_when_http_block_missing(self) -> None:
        """When http is absent entirely, the plugin-schema rule reports it; this
        rule must not double-report or fire on a node it cannot inspect."""
        state = self._state_with_web_scrape(None, http_present=False)
        assert not self._abuse_contact_error_messages(state)

    def test_skips_when_abuse_contact_missing(self) -> None:
        """abuse_contact absent inside http — plugin schema flags it; this
        rule remains silent."""
        state = self._state_with_web_scrape(None, http_present=True)
        assert not self._abuse_contact_error_messages(state)

    def test_skips_when_abuse_contact_wrong_type(self) -> None:
        """Non-string value (e.g. a secret_ref dict) — plugin schema handles
        type validation; this rule is value-shape-tolerant."""
        state = self._state_with_web_scrape(
            None,
            options_override={
                "schema": {"mode": "fixed", "fields": ["url: str"]},
                "url_field": "url",
                "content_field": "content",
                "fingerprint_field": "content_fingerprint",
                "format": "markdown",
                "http": {
                    "abuse_contact": {"secret_ref": "ABUSE_CONTACT"},
                    "scraping_reason": "test",
                    "allowed_hosts": "public_only",
                },
            },
        )
        assert not self._abuse_contact_error_messages(state)

    def test_skips_when_email_malformed(self) -> None:
        """No `@` character — let the plugin's email-format rule report it
        (this rule only cares about the domain part of a real-shaped email)."""
        state = self._state_with_web_scrape("not-an-email")
        assert not self._abuse_contact_error_messages(state)

    def test_error_severity_is_high(self) -> None:
        """The rule must produce a blocking (high-severity) error — Tier-1
        audit-integrity defects are not advisory."""
        state = self._state_with_web_scrape("ops@example.com")
        abuse_errors = [e for e in state.validate().errors if "abuse_contact" in e.message]
        assert abuse_errors
        assert abuse_errors[0].severity == "high"

    def test_error_names_the_node_and_field(self) -> None:
        """Message must identify the offending node id and field path so the
        operator (or composer LLM) can locate the violation."""
        state = self._state_with_web_scrape("ops@example.com")
        abuse_errors = [e for e in state.validate().errors if "abuse_contact" in e.message]
        assert abuse_errors
        assert abuse_errors[0].component == "node:fetch_pages"
        assert "web_scrape.http.abuse_contact" in abuse_errors[0].message

    def test_pipeline_with_reserved_address_is_invalid(self) -> None:
        """End-to-end: a fully-formed pipeline carrying a fabricated
        abuse_contact must fail validate() with is_valid=False, regardless of
        otherwise-valid structure."""
        # Single transform node with a reserved-domain address — even with
        # missing source/sinks, is_valid must be False *and* an
        # abuse_contact error must be among the reasons.
        state = self._state_with_web_scrape("ops@example.com")
        result = state.validate()
        assert not result.is_valid
        assert any("abuse_contact" in e.message for e in result.errors)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("abuse_contact", "<OPERATOR_REQUIRED>"),
            ("abuse_contact", "operator required"),
            ("scraping_reason", "<OPERATOR_REQUIRED>"),
            ("scraping_reason", "operator required"),
        ],
    )
    def test_rejects_wire_visible_identity_placeholders(self, field_name: str, value: str) -> None:
        """Composer validation must block placeholder values before preview/execution."""
        state = self._state_with_web_scrape("ops@somecompany.gov.au")
        http = dict(state.nodes[0].options["http"])
        http[field_name] = value
        options = dict(state.nodes[0].options)
        options["http"] = http
        node = replace(state.nodes[0], options=options)
        state = replace(state, nodes=(node,))

        messages = self._web_scrape_identity_error_messages(state)
        assert messages, f"Expected reject for {field_name}={value!r}, got no web_scrape identity error"
        assert field_name in messages[0]
        assert "placeholder" in messages[0]

    def test_accepts_real_wire_visible_identity_values(self) -> None:
        state = self._state_with_web_scrape("ops@somecompany.gov.au")
        messages = self._web_scrape_identity_error_messages(state)
        assert not messages


class TestSchemaContractValidation:
    """Tests for schema contract validation (pass 9) in CompositionState.validate()."""

    def _empty_state(self) -> CompositionState:
        return CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

    def _make_source(
        self,
        on_success: str = "t1",
        plugin: str = "csv",
        options: dict[str, Any] | None = None,
        on_validation_failure: str = "quarantine",
    ) -> SourceSpec:
        opts = dict(options or {})
        if plugin == "csv":
            opts = {"path": "/data/input.csv", **opts}
        elif plugin == "text":
            opts = {"path": "/data/input.txt", "column": "text", **opts}
        return SourceSpec(
            plugin=plugin,
            on_success=on_success,
            options=opts,
            on_validation_failure=on_validation_failure,
        )

    def _make_transform(
        self,
        id: str,
        input: str,
        on_success: str,
        plugin: str = "value_transform",
        options: dict[str, Any] | None = None,
        on_error: str = "discard",
    ) -> NodeSpec:
        opts = dict(options or {})
        if plugin == "value_transform":
            opts = {
                "schema": {"mode": "observed"},
                "operations": [{"target": "_placeholder", "expression": "row['text']"}],
                **opts,
            }
        return NodeSpec(
            id=id,
            node_type="transform",
            plugin=plugin,
            input=input,
            on_success=on_success,
            on_error=on_error,
            options=opts,
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )

    def _make_gate(
        self,
        id: str,
        input: str,
        routes: dict[str, str],
        condition: str = "True",
    ) -> NodeSpec:
        return NodeSpec(
            id=id,
            node_type="gate",
            plugin=None,
            input=input,
            on_success=None,
            on_error=None,
            options={},
            condition=condition,
            routes=routes,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )

    def _make_coalesce(
        self,
        id: str,
        input: str,
        on_success: str | None,
        branches: tuple[str, ...] | None = None,
    ) -> NodeSpec:
        return NodeSpec(
            id=id,
            node_type="coalesce",
            plugin=None,
            input=input,
            on_success=on_success,
            on_error=None,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=branches if branches is not None else (input,),
            policy="require_all",
            merge="nested",
        )

    def _make_output(self, name: str = "main") -> OutputSpec:
        return OutputSpec(
            name=name,
            plugin="csv",
            options={"path": f"outputs/{name}.csv", "schema": {"mode": "observed"}},
            on_write_failure="discard",
        )

    def _make_web_scrape_to_line_explode_state(
        self,
        *,
        scrape_options: dict[str, Any] | None = None,
        line_options: dict[str, Any] | None = None,
    ) -> CompositionState:
        scrape_opts = {
            "schema": {"mode": "flexible", "fields": ["url: str"]},
            "required_input_fields": ["url"],
            "url_field": "url",
            "content_field": "content",
            "fingerprint_field": "content_fingerprint",
            "format": "text",
            "fingerprint_mode": "content",
            "http": {
                "abuse_contact": "pipeline-tests@elspeth.foundryside.dev",
                "scraping_reason": "test scrape",
                "allowed_hosts": "public_only",
            },
        }
        scrape_opts.update(scrape_options or {})
        split_opts = {
            "schema": {
                "mode": "flexible",
                "fields": [
                    "url: str",
                    "content: str",
                    "content_fingerprint: str",
                ],
            },
            "required_input_fields": ["content"],
            "source_field": "content",
            "output_field": "line",
            "include_index": True,
            "index_field": "line_index",
        }
        split_opts.update(line_options or {})

        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="scrape_in",
                options={"schema": {"mode": "fixed", "fields": ["url: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "scrape_page",
                "scrape_in",
                "explode_in",
                plugin="web_scrape",
                options=scrape_opts,
            )
        )
        state = state.with_node(
            self._make_transform(
                "split_lines",
                "explode_in",
                "main",
                plugin="line_explode",
                options=split_opts,
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "scrape_page"))
        state = state.with_edge(self._make_edge("e2", "scrape_page", "split_lines"))
        state = state.with_edge(self._make_edge("e3", "split_lines", "main"))
        return state

    def test_line_explode_rejects_compact_web_scrape_text(self) -> None:
        """A text scrape with the default space separator is not line-framed.

        After Phase 6 the message no longer mentions ``text_separator`` or
        ``\\n`` — fix prose belongs in PluginAssistance, addressed by
        requirement_code. The state-level surface only has to surface the
        structured violation; the agent retrieves prose via
        ``get_plugin_assistance``.
        """
        state = self._make_web_scrape_to_line_explode_state()

        result = state.validate()

        assert not result.is_valid
        assert any(
            error.component == "node:split_lines"
            and "line_explode" in error.message
            and "line_explode.source_field.line_framed_text" in error.message
            for error in result.errors
        )

    def test_line_explode_accepts_newline_framed_web_scrape_text(self) -> None:
        state = self._make_web_scrape_to_line_explode_state(
            scrape_options={"text_separator": "\n"},
        )

        result = state.validate()

        assert result.is_valid, result.errors
        assert not any("line_explode.source_field.line_framed_text" in error.message for error in result.errors)

    def test_line_explode_accepts_markdown_web_scrape_content(self) -> None:
        state = self._make_web_scrape_to_line_explode_state(
            scrape_options={"format": "markdown"},
        )

        result = state.validate()

        assert result.is_valid, result.errors

    def _make_edge(
        self,
        id: str,
        from_node: str,
        to_node: str,
        edge_type: EdgeType = "on_success",
    ) -> EdgeSpec:
        return EdgeSpec(id=id, from_node=from_node, to_node=to_node, edge_type=edge_type, label=None)

    def test_fixed_schema_satisfies_requirement(self) -> None:
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert result.is_valid, result.errors
        assert not any("contract" in e.message.lower() for e in result.errors)

    def test_text_explicit_guaranteed_fields_satisfies_requirement(self) -> None:
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                plugin="text",
                options={
                    "column": "text",
                    "schema": {"mode": "observed", "guaranteed_fields": ["text"]},
                },
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert result.is_valid, result.errors

    def test_field_mapper_computed_output_contract_satisfies_sink_requirement(self) -> None:
        """Composer preview must honor field_mapper's computed output contract."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="mapper_in",
                plugin="text",
                options={"schema": {"mode": "observed"}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "map_body",
                "mapper_in",
                "main",
                plugin="field_mapper",
                options={
                    "schema": {"mode": "observed"},
                    "mapping": {"text": "body"},
                    "strict": True,
                },
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": "outputs/main.csv",
                    "schema": {"mode": "observed", "required_fields": ["body"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(self._make_edge("e1", "source", "map_body"))
        state = state.with_edge(self._make_edge("e2", "map_body", "main"))

        result = state.validate()

        assert result.is_valid, result.errors
        sink_contract = next(ec for ec in result.edge_contracts if ec.to_id == "output:main")
        assert sink_contract.from_id == "map_body"
        assert sink_contract.producer_guarantees == ("body",)
        assert sink_contract.consumer_requires == ("body",)
        assert sink_contract.satisfied is True

    def test_named_non_first_source_contract_violation_is_reported(self) -> None:
        """Schema validation must inspect every named source, not only the compatibility source."""
        state = CompositionState(
            source=None,
            sources={
                "customers": self._make_source(
                    on_success="customer_rows",
                    options={"schema": {"mode": "fixed", "fields": ["customer_id: str"]}},
                    on_validation_failure="discard",
                ),
                "orders": self._make_source(
                    on_success="order_rows",
                    plugin="json",
                    options={"schema": {"mode": "fixed", "fields": ["refund_id: str"]}},
                    on_validation_failure="discard",
                ),
            },
            nodes=(
                self._make_transform(
                    "validate_orders",
                    "order_rows",
                    "main",
                    options={"required_input_fields": ["order_id"]},
                ),
            ),
            edges=(),
            outputs=(self._make_output("main"), self._make_output("customer_rows")),
            metadata=PipelineMetadata(),
            version=1,
        )

        result = state.validate()

        assert not result.is_valid
        contract = next(edge for edge in result.edge_contracts if edge.to_id == "validate_orders")
        assert contract.from_id == "source:orders"
        assert contract.producer_guarantees == ("refund_id",)
        assert contract.consumer_requires == ("order_id",)
        assert contract.missing_fields == ("order_id",)
        assert any(
            error.component == "source:orders" and "'source:orders' -> 'validate_orders'" in error.message for error in result.errors
        )

    def test_contract_probe_constructor_exception_falls_back_instead_of_crashing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor-time probe failures must not escape Stage 1 validation."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="mapper_in",
                plugin="text",
                options={"schema": {"mode": "observed"}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "map_body",
                "mapper_in",
                "main",
                plugin="field_mapper",
                options={
                    "schema": {"mode": "observed"},
                    "mapping": {"text": "body"},
                },
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": "outputs/main.csv",
                    "schema": {"mode": "observed", "required_fields": ["body"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(self._make_edge("e1", "source", "map_body"))
        state = state.with_edge(self._make_edge("e2", "map_body", "main"))

        class _BrokenManager:
            def create_transform(self, plugin_name: str, options: dict[str, Any]) -> object:
                raise TemplateError("invalid template syntax")

            def get_transforms(self) -> list[type]:
                # ADR-007: composer now queries the plugin registry to compute
                # the known-pass-through set. For this mock, return an empty
                # list so the probe-failure path takes the non-pass-through
                # branch (medium warning, raw_guaranteed fallback) — matches
                # the v2 behavior the surrounding test pins.
                return []

        monkeypatch.setattr(
            "elspeth.plugins.infrastructure.manager.get_shared_plugin_manager",
            lambda: _BrokenManager(),
        )

        result = state.validate()

        assert not result.is_valid
        assert any("computed contract probe" in warning.message.lower() for warning in result.warnings)
        sink_contract = next(ec for ec in result.edge_contracts if ec.to_id == "output:main")
        assert sink_contract.producer_guarantees == ()
        assert sink_contract.consumer_requires == ("body",)
        assert sink_contract.satisfied is False

    def test_contract_probe_unexpected_constructor_exception_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Framework bugs in transform constructors must not be certified by raw fallback."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="mapper_in",
                plugin="text",
                options={"schema": {"mode": "observed"}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "map_body",
                "mapper_in",
                "main",
                plugin="field_mapper",
                options={
                    "schema": {"mode": "fixed", "fields": ["body: str"]},
                    "mapping": {"text": "body"},
                },
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": "outputs/main.csv",
                    "schema": {"mode": "fixed", "fields": ["body: str"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(self._make_edge("e1", "source", "map_body"))
        state = state.with_edge(self._make_edge("e2", "map_body", "main"))

        class _BrokenManager:
            def create_transform(self, plugin_name: str, options: dict[str, Any]) -> object:
                raise RuntimeError("framework bug inside transform __init__")

            def get_transforms(self) -> list[type]:
                return []

        monkeypatch.setattr(
            "elspeth.plugins.infrastructure.manager.get_shared_plugin_manager",
            lambda: _BrokenManager(),
        )

        with pytest.raises(RuntimeError, match="framework bug inside transform __init__"):
            state.validate()

    def test_rule_c_unexpected_constructor_exception_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rule C must apply the same probe-exception discipline as its siblings.

        Rule C (per-transform self-consistency for ``field_mapper`` with
        ``select_only: True``) constructs the transform to read its computed
        emit set. When that construction raises an unexpected exception (i.e.
        not in the closed set adjudicated by ``_is_config_probe_exception``),
        the discipline established by f3137ae8 — and already implemented by
        the producer-probe sites at ``state.py:884`` and ``state.py:1057`` and
        the semantic-validator helpers in ``_semantic_validator.py`` — is
        that the exception MUST propagate so the bug surfaces at composer-time
        rather than being silently deferred to ``/execute``. Per CLAUDE.md
        (plugin-as-system-code policy: a plugin method that raises is a bug
        we MUST know about), Rule C swallowing every exception with a bare
        ``except Exception: continue`` would conceal genuine framework bugs.

        ``_check_schema_contracts`` is invoked directly (rather than via
        ``state.validate()``) because the orchestration in ``validate()`` runs
        ``validate_semantic_contracts`` *before* the schema-contract pass, and
        ``_instantiate_consumer`` already implements the discipline — so a
        ``state.validate()``-level test would see the exception propagate from
        the earlier pass regardless of Rule C's behaviour, and silently miss
        the regression. Calling ``_check_schema_contracts`` in isolation pins
        Rule C as the discipline under test.

        Pipeline shape: source → field_mapper(select_only=True, declares an
        output field absent from the mapping) → sink. The field_mapper's
        upstream is the source sentinel (no producer-probe call), and the
        sink uses ``mode: observed`` with no required_fields (so neither
        sink-Rule-A nor sink-Rule-B reaches ``_producer_emit_set``). Rule C
        is therefore the only probe site that calls ``create_transform``
        for the broken plugin.
        """
        from elspeth.web.composer.state import _check_schema_contracts

        source = self._make_source(
            on_success="map_select",
            plugin="text",
            options={"schema": {"mode": "observed"}},
        )
        field_mapper_node = self._make_transform(
            "map_select",
            "map_select",
            "main",
            plugin="field_mapper",
            options={
                "schema": {
                    "mode": "fixed",
                    "fields": ["body: str", "batch_size: int"],
                    "required_fields": ["body", "batch_size"],
                },
                "mapping": {"text": "body"},
                "select_only": True,
                "strict": True,
            },
        )
        sink = OutputSpec(
            name="main",
            plugin="csv",
            options={
                "path": "outputs/main.csv",
                "schema": {"mode": "observed"},
            },
            on_write_failure="discard",
        )

        class _BrokenManager:
            def create_transform(self, plugin_name: str, options: dict[str, Any]) -> object:
                raise RuntimeError("framework bug inside field_mapper __init__")

            def get_transforms(self) -> list[type]:
                return []

        monkeypatch.setattr(
            "elspeth.plugins.infrastructure.manager.get_shared_plugin_manager",
            lambda: _BrokenManager(),
        )

        with pytest.raises(RuntimeError, match="framework bug inside field_mapper __init__"):
            _check_schema_contracts({"source": source}, (field_mapper_node,), (sink,))

    def test_contract_probe_redacts_exception_detail_from_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression (P2c): the constructor-time exception message is the
        plugin author's free-form text (plugin options, DSN fragments,
        filesystem paths, occasionally a mis-typed secret) and MUST NOT
        be surfaced to the preview response. The warning surfaced to the
        composer UI carries only ``type(exc).__name__`` — the class name
        is enough triage signal ("something about this plugin's config
        is wrong") without leaking the option values through a Stage 1
        preview endpoint.
        """
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="mapper_in",
                plugin="text",
                options={"schema": {"mode": "observed"}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "map_body",
                "mapper_in",
                "main",
                plugin="field_mapper",
                options={
                    "schema": {"mode": "observed"},
                    "mapping": {"text": "body"},
                },
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": "outputs/main.csv",
                    "schema": {"mode": "observed", "required_fields": ["body"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(self._make_edge("e1", "source", "map_body"))
        state = state.with_edge(self._make_edge("e2", "map_body", "main"))

        # A representative secret-bearing exception message: an API URL
        # with a bearer token fragment, a DSN, and a filesystem path.
        # Production constructors have raised all three shapes.
        leaked_substrings = (
            "Authorization: Bearer sk-SUPER-SECRET-TOKEN-123",
            "postgres://admin:hunter2@db.internal:5432/prod",  # secret-scan: allow-this-line
            "/home/appuser/.ssh/id_rsa",
        )

        class _LeakyManager:
            def create_transform(self, plugin_name: str, options: dict[str, Any]) -> object:
                raise TemplateError(f"plugin '{plugin_name}' failed to initialize: " + " | ".join(leaked_substrings))

            def get_transforms(self) -> list[type]:
                # ADR-007: composer now queries the plugin registry for the
                # known-pass-through set. Empty list keeps the probe-failure
                # path on the v2 (non-pass-through) branch — the redaction
                # test pins that path specifically.
                return []

        monkeypatch.setattr(
            "elspeth.plugins.infrastructure.manager.get_shared_plugin_manager",
            lambda: _LeakyManager(),
        )

        result = state.validate()

        # The warning still fires (triage signal preserved).
        probe_warnings = [w for w in result.warnings if "computed contract probe" in w.message.lower()]
        assert probe_warnings, "Probe-failure warning must still be emitted"
        # But none of the exception detail leaks into the message.
        for warning in probe_warnings:
            for leak in leaked_substrings:
                assert leak not in warning.message, (
                    f"Contract warning leaked plugin-option detail: {warning.message!r} "
                    f"contained {leak!r}. str(exc) must be replaced with type(exc).__name__."
                )
            # And the class name IS present (triage surface intact).
            assert "TemplateError" in warning.message

    def test_text_heuristic_rescues_original_bug_scenario(self) -> None:
        """Reported text-source scenario passes via the shared observed-text rule."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                plugin="text",
                options={"column": "text", "schema": {"mode": "observed"}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={
                    "required_input_fields": ["text"],
                    "operations": [
                        {
                            "target": "combined",
                            "expression": "row['text'] + ' world'",
                        }
                    ],
                },
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))

        result = state.validate()
        assert result.is_valid, result.errors
        edge_contract = next(ec for ec in result.edge_contracts if ec.to_id == "t1")
        assert edge_contract.satisfied is True
        assert edge_contract.producer_guarantees == ("text",)
        assert edge_contract.consumer_requires == ("text",)

    def test_no_required_input_fields_skips_check(self) -> None:
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "observed"}},
            )
        )
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert result.is_valid, result.errors

    def test_empty_required_input_fields_skips_to_schema_fallback(self) -> None:
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "observed"}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={"required_input_fields": []},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert result.is_valid, result.errors

    def test_source_direct_to_sink_records_contract(self) -> None:
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="main",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": "outputs/main.csv",
                    "schema": {"mode": "observed", "required_fields": ["text"]},
                },
                on_write_failure="discard",
            )
        )
        result = state.validate()
        assert result.is_valid, result.errors
        assert len(result.edge_contracts) == 1
        sink_contract = result.edge_contracts[0]
        assert sink_contract.from_id == "source"
        assert sink_contract.to_id == "output:main"
        assert sink_contract.satisfied is True

    def test_sink_required_fields_satisfied(self) -> None:
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="main",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": "outputs/main.csv",
                    "schema": {"mode": "observed", "required_fields": ["text"]},
                },
                on_write_failure="discard",
            )
        )
        result = state.validate()
        assert result.is_valid, result.errors
        sink_contract = next(ec for ec in result.edge_contracts if ec.to_id == "output:main")
        assert sink_contract.satisfied is True
        assert "text" in sink_contract.consumer_requires

    def test_consumer_schema_required_fields_satisfied(self) -> None:
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={"schema": {"mode": "observed", "required_fields": ["text"]}},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert result.is_valid, result.errors
        edge_contract = next(ec for ec in result.edge_contracts if ec.to_id == "t1")
        assert edge_contract.satisfied is True
        assert edge_contract.consumer_requires == ("text",)

    def test_observed_schema_no_guarantees_fails(self) -> None:
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "observed"}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("schema contract violation" in e.message.lower() for e in result.errors)
        assert any("text" in e.message for e in result.errors)

    def test_partial_match_fails(self) -> None:
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={"required_input_fields": ["text", "score"]},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("score" in e.message for e in result.errors)

    def test_optional_field_not_guaranteed(self) -> None:
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "fixed", "fields": ["text: str?"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("text" in e.message for e in result.errors)

    def test_no_schema_config_fails(self) -> None:
        state = self._empty_state()
        state = state.with_source(self._make_source(options={}))
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("schema contract violation" in e.message.lower() for e in result.errors)

    def test_malformed_schema_emits_error(self) -> None:
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "invalid_mode"}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("schema" in e.message.lower() for e in result.errors)

    def test_sink_required_fields_violation_fails(self) -> None:
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="main",
                options={"schema": {"mode": "fixed", "fields": ["line: str"]}},
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": "outputs/main.csv",
                    "schema": {"mode": "observed", "required_fields": ["text"]},
                },
                on_write_failure="discard",
            )
        )
        result = state.validate()
        assert not result.is_valid
        assert any("sink" in e.message.lower() and "text" in e.message.lower() for e in result.errors)
        sink_contract = next(ec for ec in result.edge_contracts if ec.to_id == "output:main")
        assert sink_contract.satisfied is False
        assert "text" in sink_contract.missing_fields

    def test_consumer_schema_required_fields_violation_fails(self) -> None:
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "fixed", "fields": ["line: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={
                    "required_input_fields": [],
                    "schema": {"mode": "observed", "required_fields": ["text"]},
                },
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("text" in e.message for e in result.errors)

    def test_malformed_consumer_schema_emits_error(self) -> None:
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={"schema": {"mode": "invalid_mode"}},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("schema" in e.message.lower() for e in result.errors)
        assert not any(ec.to_id == "t1" for ec in result.edge_contracts)

    def test_multiple_transforms_can_share_sink_target(self) -> None:
        """Shared sink targets stay outside the internal producer namespace.

        Uses ``mode: flexible`` for t1/t2 input contracts: this test exercises
        shared-sink-target wiring, not strict input schemas. With ``mode: fixed``
        the auto-injected ``_placeholder`` field that ``_make_transform`` adds
        via its default value_transform operation would be rejected by t2's
        locked input contract (Rule A) — surfacing as a real runtime violation
        but unrelated to the wiring property under test.
        """
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "t2",
                options={
                    "required_input_fields": ["text"],
                    "schema": {"mode": "flexible", "fields": ["text: str"]},
                },
                on_error="errors",
            )
        )
        state = state.with_node(
            self._make_transform(
                "t2",
                "t2",
                "main",
                options={
                    "required_input_fields": ["text"],
                    "schema": {"mode": "flexible", "fields": ["text: str"]},
                },
                on_error="errors",
            )
        )
        state = state.with_output(self._make_output("main"))
        state = state.with_output(
            OutputSpec(
                name="errors",
                plugin="csv",
                options={
                    "path": "outputs/errors.csv",
                    "schema": {"mode": "observed", "required_fields": ["text"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "t2"))

        result = state.validate()

        assert result.is_valid, result.errors
        assert not any("duplicate producer" in e.message.lower() for e in result.errors)
        error_sink_contracts = [ec for ec in result.edge_contracts if ec.to_id == "output:errors"]
        assert len(error_sink_contracts) == 2
        assert all(ec.satisfied for ec in error_sink_contracts)

    def test_same_gate_multiple_routes_to_same_sink_emit_one_contract(self) -> None:
        """Composer preview dedupes indistinguishable gate->sink contract rows."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="gate_in",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            NodeSpec(
                id="router",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"true": "errors", "false": "errors"},
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="errors",
                plugin="csv",
                options={
                    "path": "outputs/errors.csv",
                    "schema": {"mode": "observed", "required_fields": ["text"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(self._make_edge("e1", "source", "router"))

        result = state.validate()

        assert result.is_valid, result.errors
        error_sink_contracts = [ec for ec in result.edge_contracts if ec.to_id == "output:errors"]
        assert len(error_sink_contracts) == 1
        assert error_sink_contracts[0].from_id == "source"
        assert error_sink_contracts[0].satisfied is True

    def test_coalesce_placeholder_input_is_not_counted_as_consumer(self) -> None:
        """Coalesce.input is a composer placeholder, not a runtime consumer claim."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="gate_in",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            NodeSpec(
                id="fork_gate",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"true": "fork", "false": "fork"},
                fork_to=("branch_a", "branch_b"),
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_node(self._make_transform("ta", "branch_a", "a_out"))
        state = state.with_node(self._make_transform("tb", "branch_b", "b_out"))
        state = state.with_node(
            NodeSpec(
                id="merge",
                node_type="coalesce",
                plugin=None,
                input="branch_a",
                on_success="main",
                on_error=None,
                options={},
                condition=None,
                routes=None,
                fork_to=None,
                branches=("a_out", "b_out"),
                policy="require_all",
                merge="nested",
            )
        )
        state = state.with_output(self._make_output("main"))
        state = state.with_edge(self._make_edge("e1", "source", "fork_gate"))
        state = state.with_edge(self._make_edge("e2", "fork_gate", "ta"))
        state = state.with_edge(self._make_edge("e3", "fork_gate", "tb"))
        state = state.with_edge(self._make_edge("e4", "ta", "merge"))
        state = state.with_edge(self._make_edge("e5", "tb", "merge"))
        state = state.with_edge(self._make_edge("e6", "merge", "main"))

        result = state.validate()

        assert result.is_valid, result.errors
        assert not any("duplicate consumer" in e.message.lower() for e in result.errors)

    # --- Topology cases ---

    def test_gate_inherits_source_guarantees(self) -> None:
        """Gate route targets inherit source guarantees through walk-back."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="gate_in",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_gate(
                "g1",
                "gate_in",
                {"true": "main", "false": "errors"},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "main",
                "out",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output("out"))
        state = state.with_output(self._make_output("errors"))
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        state = state.with_edge(self._make_edge("e2", "g1", "t1"))

        result = state.validate()

        assert result.is_valid, result.errors
        t1_contract = next(ec for ec in result.edge_contracts if ec.to_id == "t1")
        assert t1_contract.from_id == "source"
        assert t1_contract.satisfied is True

    def test_route_gate_two_routes_inherit_guarantees(self) -> None:
        """Both route-gate paths inherit the same upstream guarantees."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="gate_in",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_gate(
                "g1",
                "gate_in",
                {"true": "path_a", "false": "path_b"},
            )
        )
        state = state.with_node(
            self._make_transform(
                "ta",
                "path_a",
                "out_a",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_node(
            self._make_transform(
                "tb",
                "path_b",
                "out_b",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output("out_a"))
        state = state.with_output(self._make_output("out_b"))
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        state = state.with_edge(self._make_edge("e2", "g1", "ta"))
        state = state.with_edge(self._make_edge("e3", "g1", "tb"))

        result = state.validate()

        assert result.is_valid, result.errors
        consumer_contracts = [ec for ec in result.edge_contracts if ec.to_id in {"ta", "tb"}]
        assert len(consumer_contracts) == 2
        assert {ec.from_id for ec in consumer_contracts} == {"source"}
        assert all(ec.satisfied for ec in consumer_contracts)

    def test_fork_gate_contract_check_skips_with_warning(self) -> None:
        """Fork-gate downstream contract checks stay unresolved with a warning."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="gate_in",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            NodeSpec(
                id="g1",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"true": "fork", "false": "fork"},
                fork_to=("path_a", "path_b"),
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_node(
            self._make_transform(
                "ta",
                "path_a",
                "out_a",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_node(
            self._make_transform(
                "tb",
                "path_b",
                "out_b",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output("out_a"))
        state = state.with_output(self._make_output("out_b"))
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        state = state.with_edge(self._make_edge("e2", "g1", "ta"))
        state = state.with_edge(self._make_edge("e3", "g1", "tb"))

        result = state.validate()

        assert result.is_valid, result.errors
        assert any("fork" in w.message.lower() and "contract" in w.message.lower() for w in result.warnings)
        assert not any(ec.to_id in {"ta", "tb"} for ec in result.edge_contracts)

    def test_fork_gate_direct_sink_contract_checked(self) -> None:
        """Fork branches that terminate at sinks stay statically checkable."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="gate_in",
                plugin="text",
                options={
                    "path": "/in.txt",
                    "column": "line",
                    "schema": {"mode": "observed"},
                },
            )
        )
        state = state.with_node(
            NodeSpec(
                id="g1",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={},
                fork_to=("main",),
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": "/out.csv",
                    "schema": {"mode": "fixed", "fields": ["text: str"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        state = state.with_edge(EdgeSpec(id="e2", from_node="g1", to_node="main", edge_type="fork", label="main"))

        result = state.validate()

        assert not result.is_valid
        assert not any("fork" in w.message.lower() and "contract" in w.message.lower() for w in result.warnings)
        sink_contract = next(ec for ec in result.edge_contracts if ec.to_id == "output:main")
        assert sink_contract.from_id == "source"
        assert sink_contract.consumer_requires == ("text",)
        assert sink_contract.satisfied is False

    def test_fork_branch_name_cannot_overlap_coalesce_branch_and_sink(self) -> None:
        """Composer must reject branch names that runtime routes to coalesce before sink."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="gate_in",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            NodeSpec(
                id="g1",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"true": "fork", "false": "fork"},
                fork_to=("main", "review"),
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_node(
            self._make_coalesce(
                "merge",
                "branches",
                "merged",
                branches=("main", "review"),
            )
        )
        state = state.with_output(self._make_output("main"))
        state = state.with_output(self._make_output("merged"))
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        state = state.with_edge(EdgeSpec(id="e2", from_node="g1", to_node="main", edge_type="fork", label="main"))
        state = state.with_edge(EdgeSpec(id="e3", from_node="g1", to_node="merge", edge_type="fork", label="review"))

        result = state.validate()

        assert not result.is_valid
        assert any(
            "Connection names overlap with sink names" in error.message and "main" in error.message
            for error in result.errors
        )

    def test_multi_hop_transform_no_schema_breaks_chain(self) -> None:
        """A schema-less transform breaks downstream guarantees across hops."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="source_to_ta",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "ta",
                "source_to_ta",
                "ta_out",
                plugin="passthrough",
            )
        )
        state = state.with_node(
            self._make_transform(
                "tb",
                "ta_out",
                "main",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "ta"))
        state = state.with_edge(self._make_edge("e2", "ta", "tb"))

        result = state.validate()

        assert not result.is_valid
        assert any("text" in e.message for e in result.errors)
        tb_contract = next(ec for ec in result.edge_contracts if ec.to_id == "tb")
        assert tb_contract.from_id == "ta"
        assert tb_contract.satisfied is False

    def test_transform_then_gate_walk_back_terminates(self) -> None:
        """Walk-back stops at the first non-gate producer in the chain."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="ta_in",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "ta",
                "ta_in",
                "gate_in",
                plugin="passthrough",
            )
        )
        state = state.with_node(
            self._make_gate(
                "g1",
                "gate_in",
                {"high": "tb_in", "low": "sink"},
            )
        )
        state = state.with_node(
            self._make_transform(
                "tb",
                "tb_in",
                "out",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output("out"))
        state = state.with_output(self._make_output("sink"))
        state = state.with_edge(self._make_edge("e1", "source", "ta"))
        state = state.with_edge(self._make_edge("e2", "ta", "g1"))
        state = state.with_edge(self._make_edge("e3", "g1", "tb"))

        result = state.validate()

        assert not result.is_valid
        assert any("text" in e.message for e in result.errors)
        tb_contract = next(ec for ec in result.edge_contracts if ec.to_id == "tb")
        assert tb_contract.from_id == "ta"
        assert tb_contract.satisfied is False

    def test_multi_sink_gate_routing(self) -> None:
        """Route gates emit one satisfied sink contract per direct sink target."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="gate_in",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_gate(
                "g1",
                "gate_in",
                {"true": "sink_a", "false": "sink_b"},
            )
        )
        state = state.with_output(
            OutputSpec(
                name="sink_a",
                plugin="csv",
                options={
                    "path": "outputs/sink_a.csv",
                    "schema": {"mode": "observed", "required_fields": ["text"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_output(
            OutputSpec(
                name="sink_b",
                plugin="csv",
                options={
                    "path": "outputs/sink_b.csv",
                    "schema": {"mode": "observed", "required_fields": ["text"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        state = state.with_edge(self._make_edge("e2", "g1", "sink_a"))
        state = state.with_edge(self._make_edge("e3", "g1", "sink_b"))

        result = state.validate()

        assert result.is_valid, result.errors
        sink_contracts = [ec for ec in result.edge_contracts if ec.to_id in {"output:sink_a", "output:sink_b"}]
        assert len(sink_contracts) == 2
        assert {ec.to_id for ec in sink_contracts} == {"output:sink_a", "output:sink_b"}
        assert {ec.from_id for ec in sink_contracts} == {"source"}
        assert all(ec.satisfied for ec in sink_contracts)

    def test_mixed_consumer_requirements_from_same_producer(self) -> None:
        """One upstream can satisfy one route and fail another."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="gate_in",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_gate(
                "g1",
                "gate_in",
                {"true": "path_a", "false": "path_b"},
            )
        )
        state = state.with_node(
            self._make_transform(
                "ta",
                "path_a",
                "out_a",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_node(
            self._make_transform(
                "tb",
                "path_b",
                "out_b",
                options={"required_input_fields": ["score"]},
            )
        )
        state = state.with_output(self._make_output("out_a"))
        state = state.with_output(self._make_output("out_b"))
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        state = state.with_edge(self._make_edge("e2", "g1", "ta"))
        state = state.with_edge(self._make_edge("e3", "g1", "tb"))

        result = state.validate()

        assert not result.is_valid
        assert any("score" in e.message for e in result.errors)
        ta_contract = next(ec for ec in result.edge_contracts if ec.to_id == "ta")
        assert ta_contract.satisfied is True
        tb_contract = next(ec for ec in result.edge_contracts if ec.to_id == "tb")
        assert tb_contract.satisfied is False
        assert "score" in tb_contract.missing_fields

    def test_aggregation_consumer_required_input_fields_fail(self) -> None:
        """Aggregation consumers honor required_input_fields contracts."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="agg1",
                options={"schema": {"mode": "fixed", "fields": ["line: str"]}},
            )
        )
        state = state.with_node(
            NodeSpec(
                id="agg1",
                node_type="aggregation",
                plugin="batch_stats",
                input="agg1",
                on_success="main",
                on_error=None,
                options={
                    "value_field": "value",
                    "required_input_fields": ["value"],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "agg1"))

        result = state.validate()

        assert not result.is_valid
        assert any("value" in e.message for e in result.errors)

    def test_aggregation_required_input_fields_rejected_even_when_upstream_satisfies_contract(self) -> None:
        """ADR-013 has no batch-aware pre-emission dispatch for required_input_fields."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="agg1",
                options={"schema": {"mode": "fixed", "fields": ["amount: float"]}},
            )
        )
        state = state.with_node(
            NodeSpec(
                id="agg1",
                node_type="aggregation",
                plugin="batch_stats",
                input="agg1",
                on_success="main",
                on_error="discard",
                options={
                    "value_field": "amount",
                    "required_input_fields": ["amount"],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(self._make_output(name="main"))
        state = state.with_edge(self._make_edge("e1", "source", "agg1"))

        result = state.validate()

        assert not result.is_valid
        messages = "\n".join(entry.message for entry in result.errors)
        assert "required_input_fields" in messages
        assert "batch-aware" in messages
        agg_contract = next(ec for ec in result.edge_contracts if ec.to_id == "agg1")
        assert agg_contract.satisfied is True

    def test_distribution_profile_unknown_value_type_warns_to_sample_or_use_top_k(self) -> None:
        """Observed upstream schema cannot prove value_field is numeric before execute."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="profile_in",
                options={
                    "schema": {
                        "mode": "observed",
                        "guaranteed_fields": ["community", "financial_barrier"],
                    }
                },
            )
        )
        state = state.with_node(
            NodeSpec(
                id="profile_barriers",
                node_type="aggregation",
                plugin="batch_distribution_profile",
                input="profile_in",
                on_success="main",
                on_error="discard",
                options={
                    "schema": {"mode": "observed"},
                    "value_field": "financial_barrier",
                    "group_by": "community",
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(self._make_output("main"))
        state = state.with_edge(self._make_edge("e1", "source", "profile_barriers"))
        state = state.with_edge(self._make_edge("e2", "profile_barriers", "main"))

        result = state.validate()

        assert result.is_valid, result.errors
        warnings = [entry for entry in result.warnings if entry.component == "node:profile_barriers"]
        assert any(
            warning.severity == "high"
            and "batch_distribution_profile.value_field.numeric" in warning.message
            and "batch_top_k" in warning.message
            for warning in warnings
        )

    def test_aggregation_nested_wrapper_required_input_fields_fail(self) -> None:
        """Aggregation wrapper-shaped options.required_input_fields is honored."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="agg1",
                options={"schema": {"mode": "fixed", "fields": ["line: str"]}},
            )
        )
        state = state.with_node(
            NodeSpec(
                id="agg1",
                node_type="aggregation",
                plugin="batch_stats",
                input="agg1",
                on_success="main",
                on_error=None,
                options={
                    "options": {
                        "value_field": "value",
                        "required_input_fields": ["value"],
                        "schema": {"mode": "observed"},
                    }
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "agg1"))

        result = state.validate()

        assert not result.is_valid
        assert any("value" in e.message for e in result.errors)
        agg_contract = next(ec for ec in result.edge_contracts if ec.to_id == "agg1")
        assert agg_contract.consumer_requires == ("value",)
        assert agg_contract.satisfied is False

    def test_aggregation_nested_wrapper_schema_required_fields_fail(self) -> None:
        """Aggregation wrapper-shaped options.schema.required_fields is honored."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="agg1",
                options={"schema": {"mode": "fixed", "fields": ["line: str"]}},
            )
        )
        state = state.with_node(
            NodeSpec(
                id="agg1",
                node_type="aggregation",
                plugin="batch_stats",
                input="agg1",
                on_success="main",
                on_error=None,
                options={
                    "options": {
                        "value_field": "value",
                        "schema": {"mode": "observed", "required_fields": ["value"]},
                    }
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "agg1"))

        result = state.validate()

        assert not result.is_valid
        assert any("value" in e.message for e in result.errors)
        agg_contract = next(ec for ec in result.edge_contracts if ec.to_id == "agg1")
        assert agg_contract.consumer_requires == ("value",)
        assert agg_contract.satisfied is False

    def test_aggregation_non_mapping_wrapper_options_surface_as_validation_error(self) -> None:
        """A non-Mapping ``options.options`` wrapper value surfaces as a high-severity
        ValidationEntry, not a silent fallback to the flat outer options.

        Pins the S-6 behavioral improvement: the inline duplication previously here
        silently fell through to the outer options when ``node.options["options"]``
        existed but was not a Mapping. The canonical helper
        ``get_aggregation_contract_options`` raises ValueError on that shape, and the
        ``_check_schema_contracts`` call site converts the error into a blocking
        ``ValidationEntry`` so misconfigured wrappers cannot bypass locked-input
        membership checks.
        """
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="agg1",
                options={"schema": {"mode": "fixed", "fields": ["line: str"]}},
            )
        )
        state = state.with_node(
            NodeSpec(
                id="agg1",
                node_type="aggregation",
                plugin="batch_stats",
                input="agg1",
                on_success="main",
                on_error=None,
                options={"options": "not-a-mapping"},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "agg1"))

        result = state.validate()

        assert not result.is_valid
        wrapper_errors = [e for e in result.errors if e.component == "node:agg1" and "Invalid contract config" in e.message]
        assert wrapper_errors, (
            "Expected a high-severity 'Invalid contract config' error on node:agg1 "
            f"for the non-Mapping wrapper, got: {[(e.component, e.message) for e in result.errors]}"
        )
        assert any(e.severity == "high" for e in wrapper_errors)

    def test_coalesce_producer_emits_skip_warning(self) -> None:
        """Coalesce producers stay unresolved until runtime validation."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="branch_a",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_coalesce(
                "after_merge",
                "branch_a",
                None,
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "after_merge",
                "main",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "after_merge"))
        state = state.with_edge(self._make_edge("e2", "after_merge", "t1"))

        result = state.validate()

        assert result.is_valid, result.errors
        assert any("coalesce node" in w.message.lower() and "runtime validator will check" in w.message.lower() for w in result.warnings)
        assert not any(ec.to_id == "t1" for ec in result.edge_contracts)

    # --- Guard tests ---

    def test_node_id_source_is_reserved(self) -> None:
        """A node cannot reuse the source sentinel id."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "source",
                "t1",
                "main",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "source"))

        result = state.validate()

        assert not result.is_valid
        assert any("reserved" in e.message.lower() for e in result.errors)

    def test_node_id_source_namespace_prefix_is_reserved(self) -> None:
        """Nodes cannot collide with named-source producer ids such as source:orders."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "source:orders",
                "t1",
                "main",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "source:orders"))

        result = state.validate()

        assert not result.is_valid
        assert any(error.component == "node:source:orders" and "source producer namespace" in error.message for error in result.errors)

    @pytest.mark.parametrize("source_name", ["Orders", "bad name", "continue", "__system", "x" * 39])
    def test_plural_source_names_follow_runtime_identifier_constraints(self, source_name: str) -> None:
        """Composer Stage 1 rejects names that runtime settings would reject later."""
        state = CompositionState(
            source=None,
            sources={source_name: self._make_source("main")},
            nodes=(),
            edges=(),
            outputs=(self._make_output("main"),),
            metadata=PipelineMetadata(),
            version=1,
        )

        result = state.validate()

        assert not result.is_valid
        assert any(error.component in {"source", f"source:{source_name}"} for error in result.errors)

    def test_bare_string_required_input_fields_emits_error(self) -> None:
        """Bare-string required_input_fields fails closed."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={"required_input_fields": "text"},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))

        result = state.validate()

        assert not result.is_valid
        assert any("bare string" in e.message.lower() for e in result.errors)

    def test_duplicate_producer_connection_emits_error_and_skips_contracts(self) -> None:
        """Duplicate producers fail closed instead of overwriting the namespace."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="gate_in",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_gate(
                "g1",
                "gate_in",
                {"a": "dup", "b": "path_b"},
            )
        )
        state = state.with_node(
            self._make_transform(
                "ta",
                "dup",
                "out_a",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_node(
            self._make_transform(
                "tb",
                "path_b",
                "dup",
            )
        )
        state = state.with_output(self._make_output("out_a"))
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        state = state.with_edge(self._make_edge("e2", "g1", "ta"))
        state = state.with_edge(self._make_edge("e3", "g1", "tb"))

        result = state.validate()

        assert not result.is_valid
        assert any("duplicate producer" in e.message.lower() for e in result.errors)
        assert result.edge_contracts == ()

    def test_duplicate_consumer_connection_emits_error_and_skips_contracts(self) -> None:
        """Duplicate consumers fail closed instead of fabricating edge checks."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="shared",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "ta",
                "shared",
                "out_a",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_node(
            self._make_transform(
                "tb",
                "shared",
                "out_b",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output("out_a"))
        state = state.with_output(self._make_output("out_b"))
        state = state.with_edge(self._make_edge("e1", "source", "ta"))
        state = state.with_edge(self._make_edge("e2", "source", "tb"))

        result = state.validate()

        assert not result.is_valid
        assert any("duplicate consumer" in e.message.lower() for e in result.errors)
        assert result.edge_contracts == ()

    def test_connection_name_overlaps_sink_name_emits_error_and_skips_contracts(self) -> None:
        """Connection/sink namespace overlap aborts contract telemetry."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="t1",
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
            )
        )
        state = state.with_node(
            self._make_transform(
                "t2",
                "main",
                "out",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output("main"))
        state = state.with_output(self._make_output("out"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "t2"))

        result = state.validate()

        assert not result.is_valid
        assert any("disjoint" in e.message.lower() or "overlap" in e.message.lower() for e in result.errors)
        assert result.edge_contracts == ()

    # --- Data integrity ---

    def test_edge_contracts_populated_correctly(self) -> None:
        """ValidationSummary.edge_contracts carries the expected edge data."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))

        result = state.validate()

        assert result.is_valid, result.errors
        assert len(result.edge_contracts) >= 1
        contract = next(ec for ec in result.edge_contracts if ec.to_id == "t1")
        assert contract.from_id == "source"
        assert contract.producer_guarantees == ("text",)
        assert contract.consumer_requires == ("text",)
        assert contract.missing_fields == ()
        assert contract.satisfied is True

    def test_edge_contract_to_dict_serialization(self) -> None:
        """A real emitted EdgeContract serializes with API-facing keys."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "t1",
                "t1",
                "main",
                options={"required_input_fields": ["text"]},
            )
        )
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))

        result = state.validate()

        contract = next(ec for ec in result.edge_contracts if ec.to_id == "t1")
        payload = contract.to_dict()
        assert payload["from"] == "source"
        assert payload["to"] == "t1"
        assert payload["producer_guarantees"] == ["text"]
        assert payload["consumer_requires"] == ["text"]
        assert payload["missing_fields"] == []
        assert payload["satisfied"] is True
        assert "from_id" not in payload
        assert "to_id" not in payload

    # --- Field-set membership tests (elspeth-3d25355784) ---
    #
    # The three S3 evaluation fixtures (msg{1,2,3}.json captured under
    # /tmp/elspeth_eval/2026-05-03/s3/) are the ground-truth reproducers for
    # the composer-time membership checks. Each surfaces a different rejection
    # shape that previously slipped past /validate (is_valid: true) and only
    # crashed at /execute with a structured engine error. The test cases below
    # mirror those YAMLs so the regression locks in the rejection at the
    # composer boundary.

    def _batch_stats_to_locked_sink_state(self) -> CompositionState:
        """v1 reproducer: ``batch_stats`` → locked-mode JSON sink."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="aggregate_by_tier",
                plugin="csv",
                options={
                    "schema": {
                        "mode": "fixed",
                        "fields": [
                            "ticket_id: str",
                            "subject: str",
                            "body: str",
                            "customer_tier: str",
                            "amount: float",
                        ],
                    },
                },
            )
        )
        state = state.with_node(
            NodeSpec(
                id="aggregate_by_tier",
                node_type="aggregation",
                plugin="batch_stats",
                input="aggregate_by_tier",
                on_success="results",
                on_error="discard",
                options={
                    "schema": {
                        "mode": "flexible",
                        "fields": ["customer_tier: str", "amount: float"],
                        "required_fields": ["customer_tier", "amount"],
                    },
                    "value_field": "amount",
                    "group_by": "customer_tier",
                    "compute_mean": False,
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
                output_mode="transform",
            )
        )
        state = state.with_output(
            OutputSpec(
                name="results",
                plugin="json",
                options={
                    "path": "outputs/ticket_totals_by_tier.json",
                    "schema": {
                        "mode": "fixed",
                        "fields": ["customer_tier: str", "count: int", "sum: float"],
                    },
                    "format": "json",
                    "indent": 2,
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                on_write_failure="discard",
            )
        )
        return state

    def test_v1_locked_sink_rejects_upstream_batch_size_extra(self) -> None:
        """Sink ``mode: fixed`` rejects ``batch_size`` emitted by upstream batch_stats.

        Reproduces /tmp/elspeth_eval/2026-05-03/s3/msg1.json. The composer
        previously accepted this YAML; the engine then crashed at sink_write
        with PluginContractViolation (``Extra inputs are not permitted:
        batch_size``). The new field-set membership check rejects this at
        /validate with a message that names ``batch_size`` — the same field
        the engine names — and points the operator at both fixes.
        """
        state = self._batch_stats_to_locked_sink_state()

        result = state.validate()

        assert not result.is_valid, "Composer must reject locked sink that forbids producer-emitted extras."
        sink_extra_errors = [
            e for e in result.errors if e.component == "output:results" and "batch_size" in e.message and "input is locked" in e.message
        ]
        assert sink_extra_errors, f"Expected sink locked-input rejection naming batch_size, got: {[e.message for e in result.errors]}"
        msg = sink_extra_errors[0].message
        assert "Extra fields rejected by sink input contract: [batch_size]" in msg
        assert "mode: flexible" in msg  # operator-actionable: relax sink schema
        assert "field_mapper" in msg and "select_only: true" in msg  # operator-actionable: drop extras upstream

    def test_v2_field_mapper_select_only_with_inconsistent_declared_output(self) -> None:
        """Rule C: field_mapper declares an output field its mapping won't emit.

        Reproduces /tmp/elspeth_eval/2026-05-03/s3/msg2.json. The composer
        previously accepted this YAML; the engine crashed at the schema
        config mode contract with SchemaConfigModeViolation (``missing
        required fields ['batch_size']``). The runtime check expects the
        emitted row to satisfy the declared output schema, but with
        ``select_only: true`` the actual emit is exactly ``mapping.values()``
        — which excludes ``batch_size``.
        """
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="aggregate_by_tier",
                plugin="csv",
                options={
                    "schema": {
                        "mode": "fixed",
                        "fields": [
                            "ticket_id: str",
                            "subject: str",
                            "body: str",
                            "customer_tier: str",
                            "amount: float",
                        ],
                    },
                },
            )
        )
        state = state.with_node(
            NodeSpec(
                id="aggregate_by_tier",
                node_type="aggregation",
                plugin="batch_stats",
                input="aggregate_by_tier",
                on_success="select_output_fields",
                on_error="discard",
                options={
                    "schema": {
                        "mode": "flexible",
                        "fields": ["customer_tier: str", "amount: float"],
                        "required_fields": ["customer_tier", "amount"],
                    },
                    "value_field": "amount",
                    "group_by": "customer_tier",
                    "compute_mean": False,
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
                output_mode="transform",
            )
        )
        state = state.with_node(
            self._make_transform(
                "select_output_fields",
                "select_output_fields",
                "results",
                plugin="field_mapper",
                options={
                    "schema": {
                        "mode": "flexible",
                        "fields": [
                            "batch_size: int",
                            "count: int",
                            "customer_tier: str",
                            "sum: float",
                        ],
                        "required_fields": ["customer_tier", "count", "sum"],
                    },
                    "required_input_fields": ["customer_tier", "count", "sum"],
                    "mapping": {
                        "customer_tier": "customer_tier",
                        "count": "count",
                        "sum": "sum",
                    },
                    "select_only": True,
                    "strict": True,
                },
            )
        )
        state = state.with_output(
            OutputSpec(
                name="results",
                plugin="json",
                options={
                    "path": "outputs/ticket_totals_by_tier.json",
                    "schema": {
                        "mode": "fixed",
                        "fields": ["customer_tier: str", "count: int", "sum: float"],
                    },
                    "format": "json",
                    "indent": 2,
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                on_write_failure="discard",
            )
        )

        result = state.validate()

        assert not result.is_valid, "Composer must reject field_mapper whose declared output won't be emitted."
        rule_c_errors = [
            e
            for e in result.errors
            if e.component == "node:select_output_fields" and "Transform contract violation" in e.message and "batch_size" in e.message
        ]
        assert rule_c_errors, f"Expected Rule C self-consistency rejection naming batch_size, got: {[e.message for e in result.errors]}"
        msg = rule_c_errors[0].message
        assert "select_only: true" in msg
        assert "Declared required output fields not produced by this transform: [batch_size]" in msg

    def test_v3_field_mapper_locked_input_rejects_upstream_batch_size_extra(self) -> None:
        """Rule A: locked-mode field_mapper input rejects upstream batch_stats extra.

        Reproduces /tmp/elspeth_eval/2026-05-03/s3/msg3.json. The composer
        previously accepted this YAML; the engine crashed at input validation
        with PluginContractViolation (``Extra inputs are not permitted:
        batch_size``). The field_mapper's input Pydantic model gets
        ``extra="forbid"`` because its ``schema.mode`` is fixed; upstream
        batch_stats emits ``batch_size`` which is not in the declared
        ``fields``.
        """
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="aggregate_by_tier",
                plugin="csv",
                options={
                    "schema": {
                        "mode": "fixed",
                        "fields": [
                            "ticket_id: str",
                            "subject: str",
                            "body: str",
                            "customer_tier: str",
                            "amount: float",
                        ],
                    },
                },
            )
        )
        state = state.with_node(
            NodeSpec(
                id="aggregate_by_tier",
                node_type="aggregation",
                plugin="batch_stats",
                input="aggregate_by_tier",
                on_success="select_output_fields",
                on_error="discard",
                options={
                    "schema": {
                        "mode": "flexible",
                        "fields": ["customer_tier: str", "amount: float"],
                        "required_fields": ["customer_tier", "amount"],
                    },
                    "value_field": "amount",
                    "group_by": "customer_tier",
                    "compute_mean": False,
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
                output_mode="transform",
            )
        )
        state = state.with_node(
            self._make_transform(
                "select_output_fields",
                "select_output_fields",
                "results",
                plugin="field_mapper",
                options={
                    "schema": {
                        "mode": "fixed",
                        "fields": ["customer_tier: str", "count: int", "sum: float"],
                        "required_fields": ["customer_tier", "count", "sum"],
                    },
                    "required_input_fields": ["customer_tier", "count", "sum"],
                    "mapping": {
                        "customer_tier": "customer_tier",
                        "count": "count",
                        "sum": "sum",
                    },
                    "select_only": True,
                    "strict": True,
                },
            )
        )
        state = state.with_output(
            OutputSpec(
                name="results",
                plugin="json",
                options={
                    "path": "outputs/ticket_totals_by_tier.json",
                    "schema": {
                        "mode": "fixed",
                        "fields": ["customer_tier: str", "count: int", "sum: float"],
                    },
                    "format": "json",
                    "indent": 2,
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                on_write_failure="discard",
            )
        )

        result = state.validate()

        assert not result.is_valid, "Composer must reject locked field_mapper input that forbids producer-emitted extras."
        consumer_extra_errors = [
            e
            for e in result.errors
            if e.component == "node:select_output_fields" and "input is locked" in e.message and "batch_size" in e.message
        ]
        assert consumer_extra_errors, (
            f"Expected consumer locked-input rejection naming batch_size, got: {[e.message for e in result.errors]}"
        )
        msg = consumer_extra_errors[0].message
        assert "Extra fields rejected by consumer input contract: [batch_size]" in msg
        assert "'aggregate_by_tier' -> 'select_output_fields'" in msg  # producer/consumer attribution
        assert "schema.mode: flexible" in msg  # operator-actionable: relax consumer schema
        assert "schema.fields" in msg  # operator-actionable: widen the field declaration
        assert "['batch_size']" in msg  # message names the specific field to add
        # When consumer IS field_mapper, the "insert a field_mapper" suggestion is degenerate.
        assert "insert a field_mapper" not in msg, "Rule A must not suggest inserting a field_mapper when the consumer is already one."

    def test_locked_input_check_does_not_fire_on_flexible_consumer(self) -> None:
        """Rule A negative: ``mode: flexible`` consumer accepts producer extras.

        Sanity guard against over-generalization: the same upstream
        (batch_stats with batch_size) feeding a ``mode: flexible`` consumer
        does not trigger the locked-input rejection. Only ``mode: fixed``
        produces ``extra="forbid"`` on the auto-generated input contract.
        """
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="aggregate_by_tier",
                plugin="csv",
                options={
                    "schema": {
                        "mode": "fixed",
                        "fields": ["customer_tier: str", "amount: float"],
                    },
                },
            )
        )
        state = state.with_node(
            NodeSpec(
                id="aggregate_by_tier",
                node_type="aggregation",
                plugin="batch_stats",
                input="aggregate_by_tier",
                on_success="select_output_fields",
                on_error="discard",
                options={
                    "schema": {"mode": "flexible", "fields": ["customer_tier: str", "amount: float"]},
                    "value_field": "amount",
                    "group_by": "customer_tier",
                    "compute_mean": False,
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
                output_mode="transform",
            )
        )
        state = state.with_node(
            self._make_transform(
                "select_output_fields",
                "select_output_fields",
                "main",
                plugin="field_mapper",
                options={
                    # Same shape as v3 except mode=flexible — extras allowed.
                    "schema": {
                        "mode": "flexible",
                        "fields": ["customer_tier: str", "count: int", "sum: float"],
                    },
                    "mapping": {
                        "customer_tier": "customer_tier",
                        "count": "count",
                        "sum": "sum",
                    },
                    "select_only": True,
                    "strict": True,
                },
            )
        )
        state = state.with_output(self._make_output("main"))

        result = state.validate()

        assert not any("input is locked" in e.message and "batch_size" in e.message for e in result.errors), (
            f"Flexible consumer must not trigger locked-input rejection, got errors: {[e.message for e in result.errors]}"
        )


class TestPassThroughComposerParity:
    """ADR-007 composer parity tests for known-pass-through plugins.

    The composer preview must mirror runtime propagation for pass-through
    plugins. Two behaviours are pinned:

    - Probe succeeds → the producer_guarantees on downstream edges include
      predecessor fields (not just the transform's own declared output).
    - Probe fails for a *known* pass-through plugin → fail-closed with
      high-severity warning, producer_guarantees=(), Stage 1 rejects the
      pipeline (mirroring runtime rejection).
    - Probe fails for a *non*-pass-through plugin → v2 behaviour preserved
      (medium-severity warning, return raw_guaranteed).
    """

    def _empty_state(self) -> CompositionState:
        return CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

    def _make_source(self, on_success: str, plugin: str = "csv", options: dict[str, Any] | None = None) -> SourceSpec:
        opts = dict(options or {})
        if plugin == "csv":
            opts = {"path": "/data/input.csv", **opts}
        return SourceSpec(
            plugin=plugin,
            on_success=on_success,
            options=opts,
            on_validation_failure="quarantine",
        )

    def _make_transform(
        self,
        id: str,
        input: str,
        on_success: str,
        plugin: str,
        options: dict[str, Any] | None = None,
        on_error: str = "discard",
    ) -> NodeSpec:
        return NodeSpec(
            id=id,
            node_type="transform",
            plugin=plugin,
            input=input,
            on_success=on_success,
            on_error=on_error,
            options=dict(options or {}),
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )

    def _make_edge(self, id: str, from_id: str, to_id: str) -> EdgeSpec:
        return EdgeSpec(id=id, from_node=from_id, to_node=to_id, edge_type="on_success", label=None)

    def test_preview_inherits_upstream_guarantees_when_pass_through_has_no_output_schema_config(self) -> None:
        """Successful passthrough probes still propagate inherited guarantees.

        The built-in passthrough plugin does not populate
        ``_output_schema_config``. Composer preview must therefore treat its
        own declaration set as empty and still apply ADR-007 propagation,
        mirroring ``ExecutionGraph.get_effective_guaranteed_fields()``.
        """
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="source",
                plugin="csv",
                options={
                    "schema": {
                        "mode": "fixed",
                        "fields": ["id: str", "body: str"],
                        "guaranteed_fields": ["id", "body"],
                    }
                },
            )
        )
        state = state.with_node(
            self._make_transform(
                "pt_node",
                "source",
                "main",
                plugin="passthrough",
                options={"schema": {"mode": "observed"}},
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": "outputs/main.csv",
                    "schema": {"mode": "observed", "required_fields": ["body"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(self._make_edge("e1", "source", "pt_node"))
        state = state.with_edge(self._make_edge("e2", "pt_node", "main"))

        result = state.validate()

        assert result.is_valid
        sink_contract = next(ec for ec in result.edge_contracts if ec.to_id == "output:main")
        assert set(sink_contract.producer_guarantees) == {"id", "body"}
        assert sink_contract.consumer_requires == ("body",)
        assert sink_contract.satisfied is True

    def test_preview_inherits_upstream_guarantees_through_fork_gate_into_pass_through(self) -> None:
        """Pass-through preview must follow fork branches back to their producer."""
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="gate_in",
                plugin="csv",
                options={
                    "schema": {
                        "mode": "fixed",
                        "fields": ["id: str", "body: str"],
                    }
                },
            )
        )
        state = state.with_node(
            NodeSpec(
                id="fork_gate",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"true": "fork", "false": "fork"},
                fork_to=("path_a", "overflow"),
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_node(
            self._make_transform(
                "pt_node",
                "path_a",
                "main",
                plugin="passthrough",
                options={"schema": {"mode": "observed"}},
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": "outputs/main.csv",
                    "schema": {"mode": "observed", "required_fields": ["body"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_output(
            OutputSpec(
                name="overflow",
                plugin="csv",
                options={
                    "path": "outputs/overflow.csv",
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(self._make_edge("e1", "source", "fork_gate"))
        state = state.with_edge(EdgeSpec(id="e2", from_node="fork_gate", to_node="pt_node", edge_type="fork", label="path_a"))
        state = state.with_edge(EdgeSpec(id="e3", from_node="fork_gate", to_node="overflow", edge_type="fork", label="overflow"))

        result = state.validate()

        assert result.is_valid, result.errors
        sink_contract = next(ec for ec in result.edge_contracts if ec.to_id == "output:main")
        assert set(sink_contract.producer_guarantees) == {"id", "body"}
        assert sink_contract.consumer_requires == ("body",)
        assert sink_contract.satisfied is True

    def test_preview_fails_closed_when_known_pass_through_constructor_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Probe failure on a known pass-through plugin → Stage 1 rejects pipeline.

        Composer preview must surface high-severity warning and return an
        empty producer_guarantees set, matching the runtime rejection that
        would occur if the transform were constructed at DAG build time.
        """
        state = self._empty_state()
        state = state.with_source(
            self._make_source(
                on_success="pt_node",
                plugin="csv",
                options={"schema": {"mode": "fixed", "fields": ["id: str", "body: str"], "guaranteed_fields": ["id", "body"]}},
            )
        )
        state = state.with_node(
            self._make_transform(
                "pt_node",
                "source",
                "main",
                plugin="passthrough",  # Known pass-through plugin
                options={"schema": {"mode": "observed"}},
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": "outputs/main.csv",
                    "schema": {"mode": "observed", "required_fields": ["body"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(self._make_edge("e1", "source", "pt_node"))
        state = state.with_edge(self._make_edge("e2", "pt_node", "main"))

        # Stub the plugin manager: get_transforms returns a minimal shim with
        # passthrough annotated True; create_transform raises for passthrough.
        class _StubPassThrough:
            name = "passthrough"
            passes_through_input = True
            is_batch_aware = False  # Required by _known_batch_aware_transform_plugins()

        class _StubPluginManager:
            def get_transforms(self) -> list[type]:
                return [_StubPassThrough]

            def create_transform(self, plugin_name: str, options: dict[str, Any]) -> object:
                raise TemplateError("intentional probe failure")

        monkeypatch.setattr(
            "elspeth.plugins.infrastructure.manager.get_shared_plugin_manager",
            lambda: _StubPluginManager(),
        )

        result = state.validate()

        # Stage 1 rejects because producer guarantees are empty and sink requires 'body'.
        assert not result.is_valid
        high_warnings = [w for w in result.warnings if w.severity == "high"]
        probe_high = [w for w in high_warnings if "computed contract probe" in w.message.lower() and "pass-through" in w.message.lower()]
        assert probe_high, f"Expected a high-severity probe warning mentioning pass-through; got warnings={result.warnings!r}"
        sink_contract = next(ec for ec in result.edge_contracts if ec.to_id == "output:main")
        assert sink_contract.producer_guarantees == ()
        assert sink_contract.satisfied is False


class TestCompositionStateValidateEmitsSemanticContracts:
    def test_compact_wardline_yields_semantic_error_in_validate(self):
        from tests.unit.web.composer.test_semantic_validator import _wardline_state

        state = _wardline_state(text_separator=" ", scrape_format="text")
        result = state.validate()

        assert result.is_valid is False
        # Wardline-shape with compact text: at least one error tagged with
        # node:explode reflecting the semantic contract violation.
        explode_errors = [e for e in result.errors if e.component == "node:explode"]
        assert any("Semantic contract" in e.message or "line_explode" in e.message for e in explode_errors)

        # And a SemanticEdgeContract record on the summary.
        assert len(result.semantic_contracts) == 1
        assert result.semantic_contracts[0].outcome.value == "conflict"

    def test_passing_wardline_yields_satisfied_contract(self):
        from tests.unit.web.composer.test_semantic_validator import _wardline_state

        state = _wardline_state(text_separator="\n", scrape_format="text")
        result = state.validate()
        # Other validation may pass or fail; what we assert is that
        # the semantic contract is SATISFIED.
        assert any(c.outcome.value == "satisfied" for c in result.semantic_contracts)
