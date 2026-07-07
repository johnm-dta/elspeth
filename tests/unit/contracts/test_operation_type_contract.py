"""Regression tests for the operation type contract vocabulary."""

from __future__ import annotations


def test_operation_type_contract_is_shared_across_boundaries() -> None:
    from elspeth.contracts import audit as audit_contracts
    from elspeth.core import operations as core_operations
    from elspeth.core.landscape.execution.operations import OperationRepository
    from elspeth.core.landscape.execution_repository import ExecutionRepository
    from elspeth.mcp import types as mcp_types
    from elspeth.web.execution import schemas as web_schemas

    assert hasattr(audit_contracts, "OperationType")
    assert hasattr(audit_contracts, "OPERATION_TYPE_VALUES")
    assert audit_contracts.OPERATION_TYPE_VALUES == ("source_load", "sink_write", "runtime_preflight")
    assert frozenset(audit_contracts.OPERATION_TYPE_VALUES) == audit_contracts.Operation._ALLOWED_OPERATION_TYPES

    assert core_operations.track_operation.__annotations__["operation_type"] == "OperationType"
    assert ExecutionRepository.begin_operation.__annotations__["operation_type"] == "OperationType"
    assert OperationRepository.begin_operation.__annotations__["operation_type"] == "OperationType"

    assert web_schemas.RunDiagnosticOperationType is audit_contracts.OperationType
    assert frozenset(audit_contracts.OPERATION_TYPE_VALUES) == web_schemas.RUN_DIAGNOSTIC_OPERATION_TYPE_VALUES

    assert mcp_types.OperationTypeValue is audit_contracts.OperationType
    assert mcp_types.OPERATION_TYPE_VALUES is audit_contracts.OPERATION_TYPE_VALUES
