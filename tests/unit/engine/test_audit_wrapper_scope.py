"""Regression tests for narrow recorder-failure wrapping on audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import pytest

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


@dataclass(frozen=True)
class _TransformStub:
    name: str = "test-transform"
    node_id: str = "node-1"


class _RecordingDataFlow:
    def __init__(self, *, record_error: LandscapeRecordError | None = None) -> None:
        self._record_error = record_error
        self.token_outcomes: list[dict[str, object]] = []

    def record_token_outcome(self, **kwargs: object) -> None:
        if self._record_error is not None:
            raise self._record_error
        self.token_outcomes.append(kwargs)


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
    data_flow = _RecordingDataFlow()
    executor = TransformExecutor(
        execution=object(),
        span_factory=SpanFactory(),
        step_resolver=lambda _node_id: 1,
        data_flow=data_flow,
    )
    transform = _TransformStub()
    token = make_token_info(token_id="token-1", row_id="row-1")
    violation = _make_violation(token_id=token.token_id, row_id=token.row_id)

    with pytest.raises(RuntimeError, match="contract_name accessed before"):
        executor._record_terminal_contract_failure(
            transform=transform,
            token=token,
            run_id="run-1",
            violation=violation,
        )

    assert data_flow.token_outcomes == []


def test_transform_terminal_contract_failure_wraps_typed_recorder_failures() -> None:
    """Transform helper still upgrades durable recorder failures to AuditIntegrityError."""
    data_flow = _RecordingDataFlow(record_error=LandscapeRecordError("audit DB down"))
    executor = TransformExecutor(
        execution=object(),
        span_factory=SpanFactory(),
        step_resolver=lambda _node_id: 1,
        data_flow=data_flow,
    )
    transform = _TransformStub()
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
    data_flow = _RecordingDataFlow(record_error=ValueError("serializer bug"))
    executor = SinkExecutor(
        execution=object(),
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
    data_flow = _RecordingDataFlow(record_error=LandscapeRecordError("audit DB down"))
    executor = SinkExecutor(
        execution=object(),
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
