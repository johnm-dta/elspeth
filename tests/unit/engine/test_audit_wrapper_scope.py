"""Regression tests for narrow recorder-failure wrapping on audit helpers."""

from __future__ import annotations

from typing import TypedDict
from unittest.mock import MagicMock, Mock

import pytest

from elspeth.contracts import TransformProtocol
from elspeth.contracts.declaration_contracts import (
    DeclarationContractViolation,
    _attach_contract_name_from_dispatcher,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.engine.executors import SinkExecutor, TransformExecutor
from elspeth.engine.spans import SpanFactory
from elspeth.testing import make_token_info


class _ViolationPayload(TypedDict):
    missing: list[str]


class _TestBoundaryViolation(DeclarationContractViolation):
    payload_schema = _ViolationPayload


def _make_violation(*, token_id: str, row_id: str) -> _TestBoundaryViolation:
    return _TestBoundaryViolation(
        plugin="test-plugin",
        node_id="node-1",
        run_id="run-1",
        row_id=row_id,
        token_id=token_id,
        payload={"missing": ["customer_id"]},
        message="boundary violation",
    )


def test_transform_terminal_contract_failure_keeps_to_audit_dict_bug_visible() -> None:
    """Transform helper must not relabel declaration-payload regressions as recorder failures."""
    execution = MagicMock()
    data_flow = MagicMock()
    executor = TransformExecutor(
        execution=execution,
        span_factory=SpanFactory(),
        step_resolver=lambda _node_id: 1,
        data_flow=data_flow,
    )
    transform = Mock(spec=TransformProtocol)
    transform.name = "test-transform"
    transform.node_id = "node-1"
    token = make_token_info(token_id="token-1", row_id="row-1")
    violation = _make_violation(token_id=token.token_id, row_id=token.row_id)

    with pytest.raises(RuntimeError, match="contract_name accessed before"):
        executor._record_terminal_contract_failure(
            transform=transform,
            token=token,
            run_id="run-1",
            violation=violation,
        )

    data_flow.record_token_outcome.assert_not_called()


def test_transform_terminal_contract_failure_wraps_typed_recorder_failures() -> None:
    """Transform helper still upgrades durable recorder failures to AuditIntegrityError."""
    execution = MagicMock()
    data_flow = MagicMock()
    data_flow.record_token_outcome.side_effect = LandscapeRecordError("audit DB down")
    executor = TransformExecutor(
        execution=execution,
        span_factory=SpanFactory(),
        step_resolver=lambda _node_id: 1,
        data_flow=data_flow,
    )
    transform = Mock(spec=TransformProtocol)
    transform.name = "test-transform"
    transform.node_id = "node-1"
    token = make_token_info(token_id="token-1", row_id="row-1")
    violation = _make_violation(token_id=token.token_id, row_id=token.row_id)
    _attach_contract_name_from_dispatcher(violation, "test_contract")

    with pytest.raises(AuditIntegrityError, match="Recorder failure: LandscapeRecordError: audit DB down"):
        executor._record_terminal_contract_failure(
            transform=transform,
            token=token,
            run_id="run-1",
            violation=violation,
        )


def test_sink_boundary_failure_outcomes_keep_non_recorder_bug_visible() -> None:
    """Sink helper must not relabel serializer/type bugs as recorder failures."""
    execution = MagicMock()
    data_flow = MagicMock()
    data_flow.record_token_outcome.side_effect = ValueError("serializer bug")
    executor = SinkExecutor(
        execution=execution,
        data_flow=data_flow,
        span_factory=SpanFactory(),
        run_id="run-1",
    )
    token = make_token_info(token_id="token-1", row_id="row-1")
    violation = _make_violation(token_id=token.token_id, row_id=token.row_id)
    _attach_contract_name_from_dispatcher(violation, "test_contract")

    with pytest.raises(ValueError, match="serializer bug"):
        executor._record_boundary_failure_outcomes(
            tokens=[token],
            sink_name="output",
            phase="boundary_check",
            violation=violation,
        )


def test_sink_boundary_failure_outcomes_wrap_typed_recorder_failures() -> None:
    """Sink helper still upgrades durable recorder failures to AuditIntegrityError."""
    execution = MagicMock()
    data_flow = MagicMock()
    data_flow.record_token_outcome.side_effect = LandscapeRecordError("audit DB down")
    executor = SinkExecutor(
        execution=execution,
        data_flow=data_flow,
        span_factory=SpanFactory(),
        run_id="run-1",
    )
    token = make_token_info(token_id="token-1", row_id="row-1")
    violation = _make_violation(token_id=token.token_id, row_id=token.row_id)
    _attach_contract_name_from_dispatcher(violation, "test_contract")

    with pytest.raises(AuditIntegrityError, match="Recorder failure: LandscapeRecordError: audit DB down"):
        executor._record_boundary_failure_outcomes(
            tokens=[token],
            sink_name="output",
            phase="boundary_check",
            violation=violation,
        )
