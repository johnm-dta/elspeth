"""Tests for __post_init__ validations on DAG model types.

Covers: GraphValidationWarning, BranchInfo, _GateEntry.
"""

from types import MappingProxyType

import pytest

from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.core.dag.models import BranchInfo, GraphValidationWarning, _GateEntry


class TestGraphValidationWarningPostInit:
    def test_rejects_empty_code(self) -> None:
        with pytest.raises(ValueError, match="code must not be empty"):
            GraphValidationWarning(code="", message="something", node_ids=())

    def test_rejects_empty_message(self) -> None:
        with pytest.raises(ValueError, match="message must not be empty"):
            GraphValidationWarning(code="W001", message="", node_ids=())

    def test_accepts_valid(self) -> None:
        w = GraphValidationWarning(code="W001", message="test", node_ids=("n1",))
        assert w.code == "W001"


class TestBranchInfoPostInit:
    def test_rejects_empty_coalesce_name(self) -> None:
        with pytest.raises(ValueError, match="coalesce_name must not be empty"):
            BranchInfo(coalesce_name=CoalesceName(""), gate_node_id=NodeID("g1"))

    def test_rejects_empty_gate_node_id(self) -> None:
        with pytest.raises(ValueError, match="gate_node_id must not be empty"):
            BranchInfo(coalesce_name=CoalesceName("merge1"), gate_node_id=NodeID(""))

    def test_accepts_valid(self) -> None:
        b = BranchInfo(coalesce_name=CoalesceName("merge1"), gate_node_id=NodeID("gate1"))
        assert b.coalesce_name == "merge1"


class TestGateEntryPostInit:
    def test_rejects_empty_node_id(self) -> None:
        with pytest.raises(ValueError, match="node_id must not be empty"):
            _GateEntry(node_id=NodeID(""), name="g1", fork_to=None, routes=MappingProxyType({"a": "b"}))

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            _GateEntry(node_id=NodeID("n1"), name="", fork_to=None, routes=MappingProxyType({"a": "b"}))

    def test_rejects_empty_fork_to_tuple(self) -> None:
        with pytest.raises(ValueError, match="fork_to must not be empty tuple"):
            _GateEntry(node_id=NodeID("n1"), name="g1", fork_to=(), routes=MappingProxyType({"a": "b"}))

    def test_rejects_empty_routes(self) -> None:
        with pytest.raises(ValueError, match="routes must have at least one entry"):
            _GateEntry(node_id=NodeID("n1"), name="g1", fork_to=None, routes=MappingProxyType({}))

    def test_accepts_valid(self) -> None:
        g = _GateEntry(node_id=NodeID("n1"), name="g1", fork_to=("a", "b"), routes=MappingProxyType({"x": "y"}))
        assert g.node_id == "n1"

    def test_accepts_none_fork_to(self) -> None:
        g = _GateEntry(node_id=NodeID("n1"), name="g1", fork_to=None, routes=MappingProxyType({"x": "y"}))
        assert g.fork_to is None
