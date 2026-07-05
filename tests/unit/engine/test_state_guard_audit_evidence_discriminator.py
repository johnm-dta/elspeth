"""NodeStateGuard.__exit__ populates context for any AuditEvidenceBase exception.

ADR-010 §Decision 1 widens the prior PluginContractViolation-only discriminator
so future violation classes (e.g., future Phase 2C checkpoint-integrity
violations that are NOT plugin-contract violations) still get structured
context in the audit trail — *iff* they explicitly inherit AuditEvidenceBase.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from elspeth.contracts import NodeStateStatus
from elspeth.contracts.audit_evidence import AuditEvidenceBase
from elspeth.contracts.errors import AuditIntegrityError, ExecutionError
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.engine.executors.state_guard import NodeStateGuard

REDACTED = "<redacted-secret>"


class _NonPluginEvidence(AuditEvidenceBase, RuntimeError):
    def to_audit_dict(self) -> Mapping[str, Any]:
        return {"kind": "other", "detail": "widened-discriminator"}


@dataclass(frozen=True)
class _CompletionCall:
    state_id: str
    status: NodeStateStatus
    duration_ms: float
    error: ExecutionError | None


class _ExecutionFake:
    def __init__(self) -> None:
        self.completion_calls: list[_CompletionCall] = []
        self.complete_error: BaseException | None = None

    def begin_node_state(self, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(state_id="s-1")

    def complete_node_state(
        self,
        *,
        state_id: str,
        status: NodeStateStatus,
        duration_ms: float,
        error: ExecutionError | None = None,
        **_kwargs: Any,
    ) -> None:
        if self.complete_error is not None:
            raise self.complete_error
        self.completion_calls.append(
            _CompletionCall(
                state_id=state_id,
                status=status,
                duration_ms=duration_ms,
                error=error,
            )
        )

    def assert_completed_once(self) -> _CompletionCall:
        assert len(self.completion_calls) == 1
        return self.completion_calls[0]


def _make_execution() -> _ExecutionFake:
    return _ExecutionFake()


def _make_guard(execution: _ExecutionFake) -> NodeStateGuard:
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
    with pytest.raises(_NonPluginEvidence), _make_guard(execution):
        raise _NonPluginEvidence("widened")

    err = execution.assert_completed_once().error
    assert err is not None and err.context is not None
    assert err.context["kind"] == "other"


def test_duck_typed_exception_does_NOT_populate_context() -> None:
    """Nominal check: a class exposing to_audit_dict but not inheriting
    AuditEvidenceBase must NOT reach the audit-evidence path."""

    class _Mimic(RuntimeError):
        def to_audit_dict(self) -> Mapping[str, Any]:
            return {"attacker": "payload"}

    execution = _make_execution()
    with pytest.raises(_Mimic), _make_guard(execution):
        raise _Mimic("mimic")

    err = execution.assert_completed_once().error
    assert err is not None and err.context is None


def test_plain_runtime_error_leaves_context_none() -> None:
    execution = _make_execution()
    with pytest.raises(RuntimeError), _make_guard(execution):
        raise RuntimeError("plain")

    err = execution.assert_completed_once().error
    assert err is not None and err.context is None


def test_exception_text_is_scrubbed_before_audit_integrity_error_messages() -> None:
    execution = _make_execution()
    execution.complete_error = LandscapeRecordError("database unavailable")

    with pytest.raises(AuditIntegrityError) as exc_info, _make_guard(execution):
        raise RuntimeError("provider leaked api_key=sk-1234567890abcdef1234567890abcdef")  # secret-scan: allow-this-line

    message = str(exc_info.value)
    assert REDACTED in message
    assert "sk-1234567890abcdef1234567890abcdef" not in message  # secret-scan: allow-this-line


def test_audit_evidence_context_is_scrubbed_before_recording() -> None:
    class _SecretEvidence(AuditEvidenceBase, RuntimeError):
        def to_audit_dict(self) -> Mapping[str, Any]:
            return {
                "kind": "secret-evidence",
                "api_key": "sk-1234567890abcdef1234567890abcdef",  # secret-scan: allow-this-line
                "nested": {
                    "message": "provider leaked api_key=sk-abcdef1234567890abcdef1234567890"  # secret-scan: allow-this-line
                },
            }

    execution = _make_execution()
    with pytest.raises(_SecretEvidence), _make_guard(execution):
        raise _SecretEvidence("secret context")

    err = execution.assert_completed_once().error
    assert err is not None and err.context is not None
    assert err.context["kind"] == "secret-evidence"
    assert err.context["api_key"] == REDACTED
    assert err.context["nested"]["message"] == REDACTED
