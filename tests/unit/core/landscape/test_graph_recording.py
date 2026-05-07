from __future__ import annotations

import json
from typing import Literal

import pytest

from elspeth.contracts import Determinism, NodeType, RoutingMode
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from tests.fixtures.landscape import make_factory, make_landscape_db

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _setup(*, run_id: str = "run-1") -> tuple[LandscapeDB, RecorderFactory]:
    db = make_landscape_db()
    factory = make_factory(db)
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    return db, factory


def _make_contract(
    *,
    mode: Literal["FIXED", "FLEXIBLE", "OBSERVED"] = "OBSERVED",
    fields: tuple[FieldContract, ...] = (),
    locked: bool = False,
) -> SchemaContract:
    return SchemaContract(mode=mode, fields=fields, locked=locked)


def _make_field(
    name: str,
    python_type: type = str,
    *,
    required: bool = True,
    source: Literal["declared", "inferred"] = "inferred",
) -> FieldContract:
    return FieldContract(
        normalized_name=name,
        original_name=name,
        python_type=python_type,
        required=required,
        source=source,
    )


# ---------------------------------------------------------------------------
# register_node
# ---------------------------------------------------------------------------


class TestRegisterNode:
    """Tests for DataFlowRepository.register_node."""

    def test_creates_node_with_config_hash_and_json(self) -> None:
        _db, factory = _setup()
        config = {"key": "value", "nested": {"a": 1}}
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config=config,
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.config_hash is not None
        assert len(node.config_hash) > 0
        stored_config = json.loads(node.config_json)
        assert stored_config == config

    def test_generates_node_id_when_not_provided(self) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.node_id is not None
        assert len(node.node_id) > 0

    def test_uses_explicit_node_id(self) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="my-explicit-node",
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.node_id == "my-explicit-node"

    def test_stores_sequence_in_pipeline(self) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="field_mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            sequence=3,
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.sequence_in_pipeline == 3

    def test_sequence_none_by_default(self) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.sequence_in_pipeline is None

    def test_stores_determinism(self) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="llm_classifier",
            node_type=NodeType.TRANSFORM,
            plugin_version="2.0.0",
            config={},
            determinism=Determinism.EXTERNAL_CALL,
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.determinism == Determinism.EXTERNAL_CALL

    def test_determinism_defaults_to_deterministic(self) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="field_mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.determinism == Determinism.DETERMINISTIC

    @pytest.mark.parametrize(
        "determinism",
        [
            Determinism.DETERMINISTIC,
            Determinism.SEEDED,
            Determinism.NON_DETERMINISTIC,
            Determinism.EXTERNAL_CALL,
            Determinism.IO_READ,
            Determinism.IO_WRITE,
        ],
    )
    def test_all_determinism_variants(self, determinism: Determinism) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="test_plugin",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            determinism=determinism,
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.determinism == determinism

    def test_stores_node_type(self) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="threshold_gate",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={},
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.node_type == NodeType.GATE

    @pytest.mark.parametrize(
        "node_type",
        [
            NodeType.SOURCE,
            NodeType.TRANSFORM,
            NodeType.GATE,
            NodeType.AGGREGATION,
            NodeType.COALESCE,
            NodeType.SINK,
        ],
    )
    def test_all_node_type_variants(self, node_type: NodeType) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="test_plugin",
            node_type=node_type,
            plugin_version="1.0.0",
            config={},
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.node_type == node_type

    def test_stores_plugin_name_and_version(self) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="3.2.1",
            config={},
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.plugin_name == "csv"
        assert node.plugin_version == "3.2.1"

    def test_stores_run_id(self) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.run_id == "run-1"

    def test_schema_mode_from_schema_config(self) -> None:
        _db, factory = _setup()
        schema = SchemaConfig.from_dict({"mode": "observed"})
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=schema,
        )
        assert node.schema_mode == "observed"

    def test_stores_schema_hash_when_provided(self) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_hash="abc123hash",
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.schema_hash == "abc123hash"

    def test_schema_hash_none_by_default(self) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.schema_hash is None

    def test_registered_at_is_set(self) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.registered_at is not None

    def test_two_nodes_get_distinct_ids(self) -> None:
        _db, factory = _setup()
        node_a = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=_DYNAMIC_SCHEMA,
        )
        node_b = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="field_mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node_a.node_id != node_b.node_id

    def test_config_hash_changes_with_different_config(self) -> None:
        _db, factory = _setup()
        node_a = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={"path": "a.csv"},
            schema_config=_DYNAMIC_SCHEMA,
        )
        node_b = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={"path": "b.csv"},
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node_a.config_hash != node_b.config_hash

    def test_config_hash_stable_for_same_config(self) -> None:
        _db, factory = _setup()
        node_a = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={"path": "a.csv"},
            node_id="node-a",
            schema_config=_DYNAMIC_SCHEMA,
        )
        node_b = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={"path": "a.csv"},
            node_id="node-b",
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node_a.config_hash == node_b.config_hash

    def test_empty_config(self) -> None:
        _db, factory = _setup()
        node = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="null_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.config_hash is not None
        stored = json.loads(node.config_json)
        assert stored == {}


# ---------------------------------------------------------------------------
# register_edge
# ---------------------------------------------------------------------------


class TestRegisterEdge:
    """Tests for DataFlowRepository.register_edge."""

    def test_creates_edge(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm",
            schema_config=_DYNAMIC_SCHEMA,
        )
        edge = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="src",
            to_node_id="xfm",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        assert edge.from_node_id == "src"
        assert edge.to_node_id == "xfm"
        assert edge.run_id == "run-1"

    def test_generates_edge_id_when_not_provided(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            schema_config=_DYNAMIC_SCHEMA,
        )
        edge = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="src",
            to_node_id="sink",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        assert edge.edge_id is not None
        assert len(edge.edge_id) > 0

    def test_uses_explicit_edge_id(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            schema_config=_DYNAMIC_SCHEMA,
        )
        edge = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="src",
            to_node_id="sink",
            label="continue",
            mode=RoutingMode.MOVE,
            edge_id="my-edge-id",
        )
        assert edge.edge_id == "my-edge-id"

    def test_stores_label(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="gate",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={},
            node_id="gate",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            schema_config=_DYNAMIC_SCHEMA,
        )
        edge = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="gate",
            to_node_id="sink",
            label="high_risk",
            mode=RoutingMode.MOVE,
        )
        assert edge.label == "high_risk"

    def test_stores_default_mode(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="gate",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={},
            node_id="gate",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            schema_config=_DYNAMIC_SCHEMA,
        )
        edge = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="gate",
            to_node_id="sink",
            label="route_to_sink",
            mode=RoutingMode.COPY,
        )
        assert edge.default_mode == RoutingMode.COPY

    def test_created_at_is_set(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            schema_config=_DYNAMIC_SCHEMA,
        )
        edge = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="src",
            to_node_id="sink",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        assert edge.created_at is not None

    def test_two_edges_get_distinct_ids(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="xfm",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            schema_config=_DYNAMIC_SCHEMA,
        )
        edge_a = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="src",
            to_node_id="xfm",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        edge_b = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="xfm",
            to_node_id="sink",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        assert edge_a.edge_id != edge_b.edge_id

    @pytest.mark.parametrize(
        "mode",
        [RoutingMode.MOVE, RoutingMode.COPY, RoutingMode.DIVERT],
    )
    def test_all_routing_mode_variants(self, mode: RoutingMode) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            schema_config=_DYNAMIC_SCHEMA,
        )
        edge = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="src",
            to_node_id="sink",
            label="continue",
            mode=mode,
        )
        assert edge.default_mode == mode


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------


class TestGetNode:
    """Tests for DataFlowRepository.get_node with composite PK."""

    def test_roundtrip(self) -> None:
        _db, factory = _setup()
        original = factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={"path": "data.csv"},
            node_id="src-node",
            sequence=0,
            schema_config=_DYNAMIC_SCHEMA,
        )
        fetched = factory.data_flow.get_node("src-node", "run-1")
        assert fetched is not None
        assert fetched.node_id == original.node_id
        assert fetched.run_id == original.run_id
        assert fetched.plugin_name == "csv"
        assert fetched.node_type == NodeType.SOURCE
        assert fetched.plugin_version == "1.0.0"
        assert fetched.sequence_in_pipeline == 0

    def test_returns_none_for_unknown_node(self) -> None:
        _db, factory = _setup()
        result = factory.data_flow.get_node("nonexistent", "run-1")
        assert result is None

    def test_returns_none_for_unknown_run(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        result = factory.data_flow.get_node("src", "run-999")
        assert result is None

    def test_same_node_id_in_different_runs(self) -> None:
        db = make_landscape_db()
        factory = make_factory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-A")
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-B")

        factory.data_flow.register_node(
            run_id="run-A",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={"path": "a.csv"},
            node_id="shared-id",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-B",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="2.0.0",
            config={"path": "b.csv"},
            node_id="shared-id",
            schema_config=_DYNAMIC_SCHEMA,
        )

        fetched_a = factory.data_flow.get_node("shared-id", "run-A")
        fetched_b = factory.data_flow.get_node("shared-id", "run-B")

        assert fetched_a is not None
        assert fetched_b is not None
        assert fetched_a.plugin_version == "1.0.0"
        assert fetched_b.plugin_version == "2.0.0"
        assert fetched_a.run_id == "run-A"
        assert fetched_b.run_id == "run-B"

    def test_returns_none_for_both_unknown(self) -> None:
        _db, factory = _setup()
        result = factory.data_flow.get_node("nonexistent", "no-such-run")
        assert result is None


# ---------------------------------------------------------------------------
# get_nodes
# ---------------------------------------------------------------------------


class TestGetNodes:
    """Tests for DataFlowRepository.get_nodes ordering and completeness."""

    def test_returns_all_nodes_for_run(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            sequence=0,
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm",
            sequence=1,
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            sequence=2,
            schema_config=_DYNAMIC_SCHEMA,
        )
        nodes = factory.data_flow.get_nodes("run-1")
        assert len(nodes) == 3

    def test_ordered_by_sequence(self) -> None:
        _db, factory = _setup()
        # Register out of order
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            sequence=5,
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            sequence=0,
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm",
            sequence=2,
            schema_config=_DYNAMIC_SCHEMA,
        )
        nodes = factory.data_flow.get_nodes("run-1")
        sequences = [n.sequence_in_pipeline for n in nodes]
        assert sequences == [0, 2, 5]

    def test_null_sequence_sorted_last(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            sequence=0,
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="unsequenced",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="no-seq",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            sequence=1,
            schema_config=_DYNAMIC_SCHEMA,
        )
        nodes = factory.data_flow.get_nodes("run-1")
        assert nodes[0].sequence_in_pipeline == 0
        assert nodes[1].sequence_in_pipeline == 1
        assert nodes[2].sequence_in_pipeline is None

    def test_empty_for_unknown_run(self) -> None:
        _db, factory = _setup()
        nodes = factory.data_flow.get_nodes("no-such-run")
        assert nodes == []

    def test_does_not_return_nodes_from_other_runs(self) -> None:
        db = make_landscape_db()
        factory = make_factory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-A")
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-B")

        factory.data_flow.register_node(
            run_id="run-A",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src-a",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-B",
            plugin_name="json",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src-b",
            schema_config=_DYNAMIC_SCHEMA,
        )

        nodes_a = factory.data_flow.get_nodes("run-A")
        nodes_b = factory.data_flow.get_nodes("run-B")

        assert len(nodes_a) == 1
        assert nodes_a[0].node_id == "src-a"
        assert len(nodes_b) == 1
        assert nodes_b[0].node_id == "src-b"

    def test_multiple_nodes_preserves_attributes(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={"path": "data.csv"},
            node_id="src",
            sequence=0,
            determinism=Determinism.IO_READ,
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="gate",
            node_type=NodeType.GATE,
            plugin_version="2.0.0",
            config={"threshold": 0.5},
            node_id="gate",
            sequence=1,
            determinism=Determinism.DETERMINISTIC,
            schema_config=_DYNAMIC_SCHEMA,
        )
        nodes = factory.data_flow.get_nodes("run-1")
        assert len(nodes) == 2
        src_node = nodes[0]
        gate_node = nodes[1]
        assert src_node.plugin_name == "csv"
        assert src_node.node_type == NodeType.SOURCE
        assert src_node.determinism == Determinism.IO_READ
        assert gate_node.plugin_name == "gate"
        assert gate_node.node_type == NodeType.GATE
        assert gate_node.determinism == Determinism.DETERMINISTIC

    def test_null_sequence_nodes_deterministic_ordering(self) -> None:
        """Bug qfxc: multiple NULL-sequence nodes must have deterministic order."""
        _db, factory = _setup()
        # Register multiple nodes with NULL sequence — order must be stable
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="pluginC",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="node-c",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="pluginA",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="node-a",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="pluginB",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="node-b",
            schema_config=_DYNAMIC_SCHEMA,
        )

        # Call get_nodes() twice and verify identical ordering
        nodes_first = factory.data_flow.get_nodes("run-1")
        nodes_second = factory.data_flow.get_nodes("run-1")

        ids_first = [n.node_id for n in nodes_first]
        ids_second = [n.node_id for n in nodes_second]
        assert ids_first == ids_second

        # All nodes have NULL sequence — tiebreaker is (registered_at, node_id)
        for n in nodes_first:
            assert n.sequence_in_pipeline is None


# ---------------------------------------------------------------------------
# get_edges
# ---------------------------------------------------------------------------


class TestGetEdges:
    """Tests for DataFlowRepository.get_edges."""

    def test_returns_all_edges_for_run(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="xfm",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="src",
            to_node_id="xfm",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="xfm",
            to_node_id="sink",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        edges = factory.data_flow.get_edges("run-1")
        assert len(edges) == 2

    def test_empty_list_when_no_edges(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        edges = factory.data_flow.get_edges("run-1")
        assert edges == []

    def test_empty_for_unknown_run(self) -> None:
        _db, factory = _setup()
        edges = factory.data_flow.get_edges("no-such-run")
        assert edges == []

    def test_does_not_return_edges_from_other_runs(self) -> None:
        db = make_landscape_db()
        factory = make_factory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-A")
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-B")

        factory.data_flow.register_node(
            run_id="run-A",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-A",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_edge(
            run_id="run-A",
            from_node_id="src",
            to_node_id="sink",
            label="continue",
            mode=RoutingMode.MOVE,
        )

        edges_a = factory.data_flow.get_edges("run-A")
        edges_b = factory.data_flow.get_edges("run-B")
        assert len(edges_a) == 1
        assert edges_b == []

    def test_edges_ordered_by_creation(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="xfm1",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm1",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="xfm2",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            schema_config=_DYNAMIC_SCHEMA,
        )
        edge_1 = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="src",
            to_node_id="xfm1",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        edge_2 = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="xfm1",
            to_node_id="xfm2",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        edge_3 = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="xfm2",
            to_node_id="sink",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        edges = factory.data_flow.get_edges("run-1")
        edge_ids = [e.edge_id for e in edges]
        assert edge_ids == [edge_1.edge_id, edge_2.edge_id, edge_3.edge_id]


# ---------------------------------------------------------------------------
# get_edge
# ---------------------------------------------------------------------------


class TestGetEdge:
    """Tests for DataFlowRepository.get_edge -- Tier 1 audit integrity."""

    def test_roundtrip(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            schema_config=_DYNAMIC_SCHEMA,
        )
        original = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="src",
            to_node_id="sink",
            label="continue",
            mode=RoutingMode.MOVE,
            edge_id="edge-1",
        )
        fetched = factory.data_flow.get_edge("edge-1")
        assert fetched.edge_id == original.edge_id
        assert fetched.from_node_id == "src"
        assert fetched.to_node_id == "sink"
        assert fetched.label == "continue"
        assert fetched.default_mode == RoutingMode.MOVE
        assert fetched.run_id == "run-1"

    def test_raises_audit_integrity_error_for_unknown_edge(self) -> None:
        _db, factory = _setup()
        with pytest.raises(AuditIntegrityError, match="Audit integrity violation"):
            factory.data_flow.get_edge("nonexistent-edge")

    def test_raises_audit_integrity_error_message_includes_edge_id(self) -> None:
        _db, factory = _setup()
        with pytest.raises(AuditIntegrityError, match="nonexistent-xyz"):
            factory.data_flow.get_edge("nonexistent-xyz")


# ---------------------------------------------------------------------------
# get_edge_map
# ---------------------------------------------------------------------------


class TestGetEdgeMap:
    """Tests for DataFlowRepository.get_edge_map."""

    def test_returns_correct_mapping(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="gate",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={},
            node_id="gate",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink_a",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink_a",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink_b",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink_b",
            schema_config=_DYNAMIC_SCHEMA,
        )
        edge_a = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="gate",
            to_node_id="sink_a",
            label="high_risk",
            mode=RoutingMode.MOVE,
        )
        edge_b = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="gate",
            to_node_id="sink_b",
            label="low_risk",
            mode=RoutingMode.MOVE,
        )
        edge_map = factory.data_flow.get_edge_map("run-1")
        assert edge_map[("gate", "high_risk")] == edge_a.edge_id
        assert edge_map[("gate", "low_risk")] == edge_b.edge_id

    def test_raises_for_no_edges(self) -> None:
        """get_edge_map raises AuditIntegrityError when run has no edges registered."""
        _db, factory = _setup()
        with pytest.raises(AuditIntegrityError, match="no edges registered"):
            factory.data_flow.get_edge_map("run-1")

    def test_raises_for_unknown_run(self) -> None:
        """get_edge_map raises AuditIntegrityError for unknown run (no edges either)."""
        _db, factory = _setup()
        with pytest.raises(AuditIntegrityError, match="no edges registered"):
            factory.data_flow.get_edge_map("no-such-run")

    def test_multiple_source_nodes_with_different_labels(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="gate",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={},
            node_id="gate",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="xfm",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            node_id="sink",
            schema_config=_DYNAMIC_SCHEMA,
        )
        e1 = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="src",
            to_node_id="gate",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        e2 = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="gate",
            to_node_id="xfm",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        e3 = factory.data_flow.register_edge(
            run_id="run-1",
            from_node_id="gate",
            to_node_id="sink",
            label="escalate",
            mode=RoutingMode.MOVE,
        )
        edge_map = factory.data_flow.get_edge_map("run-1")
        assert len(edge_map) == 3
        assert edge_map[("src", "continue")] == e1.edge_id
        assert edge_map[("gate", "continue")] == e2.edge_id
        assert edge_map[("gate", "escalate")] == e3.edge_id


# ---------------------------------------------------------------------------
# get_node_contracts
# ---------------------------------------------------------------------------


class TestGetNodeContracts:
    """Tests for DataFlowRepository.get_node_contracts."""

    def test_returns_none_none_when_no_contracts_set(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        inp, out = factory.data_flow.get_node_contracts("run-1", "src")
        assert inp is None
        assert out is None

    def test_crashes_for_unknown_node_by_default(self) -> None:
        _db, factory = _setup()
        with pytest.raises(AuditIntegrityError, match="Node not found"):
            factory.data_flow.get_node_contracts("run-1", "nonexistent")

    def test_returns_none_none_for_unknown_node_when_allowed(self) -> None:
        _db, factory = _setup()
        inp, out = factory.data_flow.get_node_contracts("run-1", "nonexistent", allow_missing=True)
        assert inp is None
        assert out is None

    def test_crashes_for_unknown_run_by_default(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        with pytest.raises(AuditIntegrityError, match="Node not found"):
            factory.data_flow.get_node_contracts("run-999", "src")

    def test_returns_none_none_for_unknown_run_when_allowed(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        inp, out = factory.data_flow.get_node_contracts("run-999", "src", allow_missing=True)
        assert inp is None
        assert out is None

    def test_returns_input_contract_when_provided(self) -> None:
        _db, factory = _setup()
        input_contract = _make_contract(
            mode="FIXED",
            fields=(
                _make_field("customer_id", str, source="declared"),
                _make_field("amount", float, source="declared"),
            ),
            locked=True,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm",
            input_contract=input_contract,
            schema_config=_DYNAMIC_SCHEMA,
        )
        inp, out = factory.data_flow.get_node_contracts("run-1", "xfm")
        assert inp is not None
        assert out is None

    def test_returns_output_contract_when_provided(self) -> None:
        _db, factory = _setup()
        output_contract = _make_contract(
            mode="OBSERVED",
            fields=(_make_field("result", str),),
            locked=True,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm",
            output_contract=output_contract,
            schema_config=_DYNAMIC_SCHEMA,
        )
        inp, out = factory.data_flow.get_node_contracts("run-1", "xfm")
        assert inp is None
        assert out is not None

    def test_returns_both_contracts_when_provided(self) -> None:
        _db, factory = _setup()
        input_contract = _make_contract(
            mode="FIXED",
            fields=(_make_field("customer_id", str, source="declared"),),
            locked=True,
        )
        output_contract = _make_contract(
            mode="OBSERVED",
            fields=(_make_field("result", str),),
            locked=True,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm",
            input_contract=input_contract,
            output_contract=output_contract,
            schema_config=_DYNAMIC_SCHEMA,
        )
        inp, out = factory.data_flow.get_node_contracts("run-1", "xfm")
        assert inp is not None
        assert out is not None

    def test_roundtrip_preserves_contract_fields(self) -> None:
        _db, factory = _setup()
        input_contract = _make_contract(
            mode="FIXED",
            fields=(
                _make_field("id", int, required=True, source="declared"),
                _make_field("name", str, required=False, source="declared"),
            ),
            locked=True,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm",
            input_contract=input_contract,
            schema_config=_DYNAMIC_SCHEMA,
        )
        inp, _ = factory.data_flow.get_node_contracts("run-1", "xfm")
        assert inp is not None
        assert inp.mode == "FIXED"
        assert inp.locked is True
        assert len(inp.fields) == 2
        field_names = {fc.normalized_name for fc in inp.fields}
        assert field_names == {"id", "name"}


# ---------------------------------------------------------------------------
# update_node_output_contract
# ---------------------------------------------------------------------------


class TestUpdateNodeOutputContract:
    """Tests for DataFlowRepository.update_node_output_contract."""

    def test_sets_output_contract_on_node_without_one(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        _, out = factory.data_flow.get_node_contracts("run-1", "src")
        assert out is None

        new_contract = _make_contract(
            mode="OBSERVED",
            fields=(
                _make_field("id", int),
                _make_field("name", str),
            ),
            locked=True,
        )
        factory.data_flow.update_node_output_contract("run-1", "src", new_contract)

        _, out = factory.data_flow.get_node_contracts("run-1", "src")
        assert out is not None

    def test_overwrites_existing_output_contract(self) -> None:
        _db, factory = _setup()
        original_contract = _make_contract(
            mode="OBSERVED",
            fields=(_make_field("old_field", str),),
            locked=True,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm",
            output_contract=original_contract,
            schema_config=_DYNAMIC_SCHEMA,
        )
        _, out_before = factory.data_flow.get_node_contracts("run-1", "xfm")
        assert out_before is not None
        assert len(out_before.fields) == 1

        updated_contract = _make_contract(
            mode="OBSERVED",
            fields=(
                _make_field("new_field", int),
                _make_field("other", float),
            ),
            locked=True,
        )
        factory.data_flow.update_node_output_contract("run-1", "xfm", updated_contract)

        _, out_after = factory.data_flow.get_node_contracts("run-1", "xfm")
        assert out_after is not None
        assert len(out_after.fields) == 2

    def test_does_not_affect_input_contract(self) -> None:
        _db, factory = _setup()
        input_contract = _make_contract(
            mode="FIXED",
            fields=(_make_field("x", int, source="declared"),),
            locked=True,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            node_id="xfm",
            input_contract=input_contract,
            schema_config=_DYNAMIC_SCHEMA,
        )
        inp_before, _ = factory.data_flow.get_node_contracts("run-1", "xfm")
        assert inp_before is not None

        output_contract = _make_contract(
            mode="OBSERVED",
            fields=(_make_field("y", str),),
            locked=True,
        )
        factory.data_flow.update_node_output_contract("run-1", "xfm", output_contract)

        inp_after, out_after = factory.data_flow.get_node_contracts("run-1", "xfm")
        assert inp_after is not None
        assert out_after is not None
        # Input contract unchanged
        assert len(inp_after.fields) == 1
        assert inp_after.fields[0].normalized_name == "x"

    def test_roundtrip_preserves_updated_fields(self) -> None:
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            node_id="src",
            schema_config=_DYNAMIC_SCHEMA,
        )
        contract = _make_contract(
            mode="OBSERVED",
            fields=(
                _make_field("alpha", str),
                _make_field("beta", int),
                _make_field("gamma", float),
            ),
            locked=True,
        )
        factory.data_flow.update_node_output_contract("run-1", "src", contract)

        _, out = factory.data_flow.get_node_contracts("run-1", "src")
        assert out is not None
        assert out.mode == "OBSERVED"
        assert out.locked is True
        assert len(out.fields) == 3
        field_names = [fc.normalized_name for fc in out.fields]
        assert "alpha" in field_names
        assert "beta" in field_names
        assert "gamma" in field_names
