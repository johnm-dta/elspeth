"""NodeStateGuard.__exit__ populates context for any AuditEvidenceBase exception.

ADR-010 §Decision 1 widens the prior PluginContractViolation-only discriminator
so future violation classes (e.g., future Phase 2C checkpoint-integrity
violations that are NOT plugin-contract violations) still get structured
context in the audit trail — *iff* they explicitly inherit AuditEvidenceBase.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.contracts.audit_evidence import AuditEvidenceBase
from elspeth.contracts.errors import ExecutionError
from elspeth.engine.executors.state_guard import NodeStateGuard


class _NonPluginEvidence(AuditEvidenceBase, RuntimeError):
    def to_audit_dict(self) -> Mapping[str, Any]:
        return {"kind": "other", "detail": "widened-discriminator"}


def _make_execution() -> MagicMock:
    execution = MagicMock()
    execution.begin_node_state.return_value = MagicMock(state_id="s-1")
    return execution


def _make_guard(execution: MagicMock) -> NodeStateGuard:
    return NodeStateGuard(
        execution=execution,
        token_id="tok-1",
        node_id="node-1",
        run_id="run-1",
        step_index=0,
        input_data={},
        attempt=0,
    )


def test_non_plugin_audit_evidence_populates_context() -> None:
    execution = _make_execution()
    captured: dict[str, ExecutionError | None] = {"err": None}

    def _capture(*_, error: ExecutionError | None = None, **__) -> None:
        captured["err"] = error

    execution.complete_node_state.side_effect = _capture
    with pytest.raises(_NonPluginEvidence), _make_guard(execution):
        raise _NonPluginEvidence("widened")

    execution.complete_node_state.assert_called_once()
    err = captured["err"]
    assert err is not None and err.context is not None
    assert err.context["kind"] == "other"


def test_duck_typed_exception_does_NOT_populate_context() -> None:
    """Nominal check: a class exposing to_audit_dict but not inheriting
    AuditEvidenceBase must NOT reach the audit-evidence path."""

    class _Mimic(RuntimeError):
        def to_audit_dict(self) -> Mapping[str, Any]:
            return {"attacker": "payload"}

    execution = _make_execution()
    captured: dict[str, ExecutionError | None] = {"err": None}

    def _capture(*_, error: ExecutionError | None = None, **__) -> None:
        captured["err"] = error

    execution.complete_node_state.side_effect = _capture
    with pytest.raises(_Mimic), _make_guard(execution):
        raise _Mimic("mimic")

    execution.complete_node_state.assert_called_once()
    err = captured["err"]
    assert err is not None and err.context is None


def test_plain_runtime_error_leaves_context_none() -> None:
    execution = _make_execution()
    captured: dict[str, ExecutionError | None] = {"err": None}

    def _capture(*_, error: ExecutionError | None = None, **__) -> None:
        captured["err"] = error

    execution.complete_node_state.side_effect = _capture
    with pytest.raises(RuntimeError), _make_guard(execution):
        raise RuntimeError("plain")

    execution.complete_node_state.assert_called_once()
    err = captured["err"]
    assert err is not None and err.context is None
