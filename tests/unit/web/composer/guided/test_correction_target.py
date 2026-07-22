"""Exact selected-component authority for guided wire corrections."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from uuid import uuid4

import pytest

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.composer.guided.planning import (
    require_guided_correction_target_changed,
    resolve_guided_correction_target,
)
from elspeth.web.composer.guided.state_machine import ComponentTarget
from elspeth.web.composer.state import CompositionState, EdgeSpec, NodeSpec, OutputSpec, PipelineMetadata, SourceSpec

SOURCE_A = "11111111-1111-4111-8111-111111111111"
SOURCE_B = "22222222-2222-4222-8222-222222222222"
NODE_A = "33333333-3333-4333-8333-333333333333"
NODE_B = "44444444-4444-4444-8444-444444444444"
OUTPUT_A = "55555555-5555-4555-8555-555555555555"
OUTPUT_B = "66666666-6666-4666-8666-666666666666"
EDGE_A = "77777777-7777-4777-8777-777777777777"
EDGE_B = "88888888-8888-4888-8888-888888888888"
SOURCE_FAILURE = "99999999-9999-4999-8999-999999999999"
OUTPUT_FAILURE = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


def _node(node_id: str, input_name: str, output_name: str) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type="transform",
        plugin="passthrough",
        input=input_name,
        on_success=output_name,
        on_error="discard",
        options={"schema": {"mode": "observed"}},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )


def _predecessor() -> CompositionState:
    return CompositionState(
        sources={
            "source_a": SourceSpec("csv", "rows_a", {"path": "a.csv"}, "discard"),
            "source_b": SourceSpec("csv", "rows_b", {"path": "b.csv"}, "discard"),
        },
        nodes=(_node("node_a", "rows_a", "result_a"), _node("node_b", "rows_b", "result_b")),
        edges=(
            EdgeSpec("edge_a", "source_a", "node_a", "on_success", None),
            EdgeSpec("edge_b", "source_b", "node_b", "on_success", None),
        ),
        outputs=(
            OutputSpec("result_a", "json", {"path": "a.jsonl"}, "discard"),
            OutputSpec("result_b", "json", {"path": "b.jsonl"}, "discard"),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


def _endpoint(kind: str, stable_id: str) -> dict[str, str]:
    return {"kind": kind, "stable_id": stable_id}


def _wire() -> dict[str, object]:
    return {
        "sources": [
            {"stable_id": SOURCE_A, "plugin": "csv", "row_cardinality": {"output": "zero_or_many"}},
            {"stable_id": SOURCE_B, "plugin": "csv", "row_cardinality": {"output": "zero_or_many"}},
        ],
        "nodes": [
            {"stable_id": NODE_A, "node_type": "transform", "plugin": "passthrough"},
            {"stable_id": NODE_B, "node_type": "transform", "plugin": "passthrough"},
        ],
        "outputs": [
            {"stable_id": OUTPUT_A, "plugin": "json", "required_fields": []},
            {"stable_id": OUTPUT_B, "plugin": "json", "required_fields": []},
        ],
        "connections": [
            {
                "stable_id": SOURCE_FAILURE,
                "from_endpoint": _endpoint("source", SOURCE_A),
                "to_endpoint": {"kind": "discard"},
                "flow": {"role": "source_validation_failure", "route": None},
                "schema_contract": {"satisfied": True, "missing_fields": []},
            },
            {
                "stable_id": EDGE_A,
                "from_endpoint": _endpoint("source", SOURCE_A),
                "to_endpoint": _endpoint("node", NODE_A),
                "flow": {"role": "on_success", "route": None},
                "schema_contract": {"satisfied": True, "missing_fields": []},
            },
            {
                "stable_id": EDGE_B,
                "from_endpoint": _endpoint("source", SOURCE_B),
                "to_endpoint": _endpoint("node", NODE_B),
                "flow": {"role": "on_success", "route": None},
                "schema_contract": {"satisfied": True, "missing_fields": []},
            },
            {
                "stable_id": OUTPUT_FAILURE,
                "from_endpoint": _endpoint("output", OUTPUT_A),
                "to_endpoint": {"kind": "discard"},
                "flow": {"role": "output_write_failure", "route": None},
                "schema_contract": {"satisfied": True, "missing_fields": []},
            },
        ],
    }


def _connection(wire: dict[str, object], stable_id: str) -> dict[str, object]:
    matches = [item for item in wire["connections"] if item["stable_id"] == stable_id]  # type: ignore[union-attr,index]
    assert len(matches) == 1
    return matches[0]


def _regenerate_public_stable_ids(wire: dict[str, object]) -> dict[str, object]:
    successor = deepcopy(wire)
    endpoint_ids: dict[str, str] = {}
    for collection in ("sources", "nodes", "outputs"):
        for component in successor[collection]:  # type: ignore[union-attr]
            previous = component["stable_id"]
            replacement = str(uuid4())
            endpoint_ids[previous] = replacement
            component["stable_id"] = replacement
    for connection in successor["connections"]:  # type: ignore[union-attr]
        connection["stable_id"] = str(uuid4())
        for endpoint_name in ("from_endpoint", "to_endpoint"):
            endpoint = connection[endpoint_name]
            if endpoint["kind"] != "discard":
                endpoint["stable_id"] = endpoint_ids[endpoint["stable_id"]]
    return successor


def test_same_kind_edit_must_change_the_exact_selected_component() -> None:
    wire = _wire()
    target = resolve_guided_correction_target(
        requested=ComponentTarget(kind="source", stable_id=SOURCE_A),
        wire_payload=wire,
        predecessor=_predecessor(),
    )
    wrong_successor = deepcopy(wire)
    wrong_successor["sources"][1]["plugin"] = "json"  # type: ignore[index]

    with pytest.raises(AuditIntegrityError, match="selected component"):
        require_guided_correction_target_changed(wrong_successor, target, _predecessor())

    reordered_wrong_successor = deepcopy(wire)
    reordered_wrong_successor["sources"][1]["plugin"] = "json"  # type: ignore[index]
    reordered_wrong_successor["sources"].reverse()  # type: ignore[union-attr]
    predecessor = _predecessor()
    reordered_state = replace(predecessor, sources=dict(reversed(predecessor.sources.items())))
    with pytest.raises(AuditIntegrityError, match="selected component"):
        require_guided_correction_target_changed(reordered_wrong_successor, target, reordered_state)

    exact_successor = deepcopy(wire)
    _connection(exact_successor, EDGE_A)["to_endpoint"] = _endpoint("node", NODE_B)
    require_guided_correction_target_changed(exact_successor, target, _predecessor())


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("to_endpoint", _endpoint("node", NODE_B)),
        ("flow", {"role": "route", "route": "accepted"}),
        ("schema_contract", {"satisfied": False, "missing_fields": ["text"]}),
    ),
)
def test_edge_edit_must_change_the_exact_selected_edge_semantics(field: str, replacement: object) -> None:
    wire = _wire()
    target = resolve_guided_correction_target(
        requested=ComponentTarget(kind="edge", stable_id=EDGE_A),
        wire_payload=wire,
        predecessor=_predecessor(),
    )
    assert target.planner_context() == {
        "kind": "edge",
        "stable_id": EDGE_A,
        "owner_kind": "source",
        "owner_key": "source_a",
        "target": _connection(wire, EDGE_A),
    }
    wrong_successor = deepcopy(wire)
    _connection(wrong_successor, EDGE_B)[field] = replacement

    with pytest.raises(AuditIntegrityError, match="selected component"):
        require_guided_correction_target_changed(wrong_successor, target, _predecessor())

    reordered_wrong_successor = deepcopy(wire)
    _connection(reordered_wrong_successor, EDGE_B)[field] = replacement
    reordered_wrong_successor = _regenerate_public_stable_ids(reordered_wrong_successor)
    reordered_wrong_successor["connections"].reverse()  # type: ignore[union-attr]
    with pytest.raises(AuditIntegrityError, match="selected component"):
        require_guided_correction_target_changed(reordered_wrong_successor, target, _predecessor())

    exact_successor = deepcopy(wire)
    _connection(exact_successor, EDGE_A)[field] = replacement
    require_guided_correction_target_changed(exact_successor, target, _predecessor())


def test_synthetic_public_edge_is_authoritative_without_a_private_edge_position() -> None:
    wire = _wire()
    target = resolve_guided_correction_target(
        requested=ComponentTarget(kind="edge", stable_id=OUTPUT_FAILURE),
        wire_payload=wire,
        predecessor=_predecessor(),
    )
    assert target.planner_context() == {
        "kind": "edge",
        "stable_id": OUTPUT_FAILURE,
        "owner_kind": "output",
        "owner_key": "result_a",
        "target": _connection(wire, OUTPUT_FAILURE),
    }

    wrong_successor = deepcopy(wire)
    _connection(wrong_successor, EDGE_A)["flow"] = {"role": "route", "route": "wrong"}
    wrong_successor = _regenerate_public_stable_ids(wrong_successor)
    wrong_successor["connections"].reverse()  # type: ignore[union-attr]
    with pytest.raises(AuditIntegrityError, match="selected component"):
        require_guided_correction_target_changed(wrong_successor, target, _predecessor())

    exact_successor = deepcopy(wire)
    _connection(exact_successor, OUTPUT_FAILURE)["to_endpoint"] = _endpoint("node", NODE_B)
    exact_successor["connections"].reverse()  # type: ignore[union-attr]
    require_guided_correction_target_changed(exact_successor, target, _predecessor())
