"""Regression tests for finite MCP TypedDict string fields."""

from __future__ import annotations

from typing import Literal, Required, get_args, get_origin, get_type_hints

from elspeth.mcp import types
from elspeth.mcp.server import _TOOLS


def _literal_values(hint: object) -> set[str]:
    if get_origin(hint) is Required:
        hint = get_args(hint)[0]
    assert get_origin(hint) is Literal
    return set(get_args(hint))


def test_run_record_status_uses_current_run_status_literal() -> None:
    hints = get_type_hints(types.RunRecord)

    assert _literal_values(hints["status"]) == {
        "running",
        "completed",
        "completed_with_failures",
        "failed",
        "empty",
        "interrupted",
    }


def test_operation_record_uses_operation_literals() -> None:
    hints = get_type_hints(types.OperationRecord)

    assert _literal_values(hints["operation_type"]) == {
        "source_load",
        "sink_write",
        "runtime_preflight",
    }
    assert _literal_values(hints["status"]) == {"open", "completed", "failed", "pending"}


def test_operation_type_filter_schema_matches_operation_literals() -> None:
    """MCP schema must accept every operation type the runtime writes."""
    hints = get_type_hints(types.OperationRecord)
    operation_type_values = _literal_values(hints["operation_type"])
    schema_values = set(_TOOLS["list_operations"].schema_properties["operation_type"]["enum"])

    assert schema_values == operation_type_values


def test_node_state_and_dag_literals_match_current_contracts() -> None:
    node_state_hints = get_type_hints(types.NodeStateRecord)
    dag_node_hints = get_type_hints(types.DAGNode)
    dag_edge_hints = get_type_hints(types.DAGEdge)

    assert _literal_values(node_state_hints["status"]) == {"open", "pending", "completed", "failed"}
    assert _literal_values(dag_node_hints["node_type"]) == {
        "source",
        "transform",
        "gate",
        "aggregation",
        "coalesce",
        "sink",
    }
    assert _literal_values(dag_edge_hints["mode"]) == {"move", "copy", "divert"}
    assert _literal_values(dag_edge_hints["flow_type"]) == {"normal", "divert"}


def test_diagnostic_and_contract_literals_are_finite() -> None:
    diagnostic_hints = get_type_hints(types.DiagnosticProblem)
    run_contract_hints = get_type_hints(types.RunContractReport)
    field_explanation_hints = get_type_hints(types.FieldExplanation)
    contract_hints = get_type_hints(types.ContractViolationRecord)

    assert _literal_values(diagnostic_hints["severity"]) == {"CRITICAL", "WARNING", "INFO"}
    assert _literal_values(run_contract_hints["mode"]) == {"FIXED", "FLEXIBLE", "OBSERVED"}
    assert _literal_values(field_explanation_hints["contract_mode"]) == {"FIXED", "FLEXIBLE", "OBSERVED"}
    assert _literal_values(contract_hints["schema_mode"]) == {"fixed", "flexible", "observed", "parse"}
    assert _literal_values(contract_hints["violation_type"]) == {
        "type_mismatch",
        "missing_field",
        "extra_field",
    }
