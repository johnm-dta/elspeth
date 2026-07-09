"""Tests for PluginContext.record_call() FrameworkBugError guard clauses.

These test the offensive programming guards that detect framework bugs:
- No landscape configured
- XOR violation: both state_id and operation_id set
- XOR violation: neither state_id nor operation_id set
- state_id set but node_state lookup returns None
- Token mismatch between ctx.token and authoritative node_state
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from elspeth.contracts import FrameworkBugError, NodeStateStatus
from elspeth.contracts.audit import Call, NodeStateCompleted
from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.enums import CallStatus, CallType
from elspeth.contracts.plugin_context import PluginContext
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from tests.fixtures.factories import make_token_info


def _completed_node_state(*, token_id: str) -> NodeStateCompleted:
    return NodeStateCompleted(
        state_id="state-1",
        token_id=token_id,
        node_id="transform-1",
        step_index=0,
        attempt=0,
        status=NodeStateStatus.COMPLETED,
        input_hash="input-hash",
        output_hash="output-hash",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        completed_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        duration_ms=1.0,
    )


class _StatePathAuditWriter:
    def __init__(self, *, node_state: NodeStateCompleted | None) -> None:
        self.node_state = node_state
        self.allocated_state_ids: list[str] = []
        self.recorded_calls: list[dict[str, Any]] = []
        self.node_state_lookups: list[str] = []

    def allocate_call_index(self, state_id: str) -> int:
        self.allocated_state_ids.append(state_id)
        return len(self.allocated_state_ids) - 1

    def record_call(
        self,
        state_id: str,
        call_index: int,
        call_type: CallType,
        status: CallStatus,
        request_data: RawCallPayload,
        response_data: RawCallPayload | None = None,
        error: RawCallPayload | None = None,
        latency_ms: float | None = None,
    ) -> Call:
        self.recorded_calls.append(
            {
                "state_id": state_id,
                "call_index": call_index,
                "call_type": call_type,
                "status": status,
                "request_data": request_data,
                "response_data": response_data,
                "error": error,
                "latency_ms": latency_ms,
            }
        )
        return Call(
            call_id=f"call-{call_index}",
            call_index=call_index,
            call_type=call_type,
            status=status,
            request_hash="request-hash",
            response_hash="response-hash" if response_data is not None else None,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            state_id=state_id,
            latency_ms=latency_ms,
        )

    def get_node_state(self, state_id: str) -> NodeStateCompleted | None:
        self.node_state_lookups.append(state_id)
        return self.node_state


class TestRecordCallNoLandscapeGuard:
    """record_call() must raise FrameworkBugError when landscape is None."""

    def test_raises_framework_bug_error_when_landscape_is_none(self) -> None:
        ctx = PluginContext(run_id="run-1", config={}, landscape=None, state_id="state-1")
        with pytest.raises(FrameworkBugError, match=r"record_call.*without landscape"):
            ctx.record_call(
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data={"prompt": "test"},
                latency_ms=100.0,
            )


class TestRecordCallXOREnforcement:
    """record_call() enforces exactly one of state_id or operation_id."""

    def test_raises_when_both_state_id_and_operation_id_set(self) -> None:
        """Both set = ambiguous parent for the call = framework bug."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=factory.plugin_audit_writer(),
            state_id="state-1",
            operation_id="op-1",
        )
        with pytest.raises(FrameworkBugError, match="BOTH state_id and operation_id"):
            ctx.record_call(
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data={"prompt": "test"},
                latency_ms=100.0,
            )

    def test_raises_when_neither_state_id_nor_operation_id_set(self) -> None:
        """Neither set = no parent for the call = framework bug."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=factory.plugin_audit_writer(),
            state_id=None,
            operation_id=None,
        )
        with pytest.raises(FrameworkBugError, match="without state_id or operation_id"):
            ctx.record_call(
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data={"prompt": "test"},
                latency_ms=100.0,
            )


class TestRecordCallNodeStateLookupGuard:
    """record_call() must raise when state_id doesn't resolve to a node_state."""

    def test_raises_when_get_node_state_returns_none(self) -> None:
        """state_id exists but no matching node_state in DB = framework bug."""
        landscape = _StatePathAuditWriter(node_state=None)

        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=landscape,
            state_id="state-orphan",
            token=make_token_info(token_id="token-1"),
        )
        with pytest.raises(FrameworkBugError, match=r"get_node_state.*returned None"):
            ctx.record_call(
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data={"prompt": "test"},
                latency_ms=100.0,
            )


class TestRecordCallTokenMismatchGuard:
    """record_call() must raise when ctx.token disagrees with authoritative node_state."""

    def test_raises_on_token_id_mismatch(self) -> None:
        """ctx.token.token_id != node_state.token_id = framework bug (ctx out of sync)."""
        # Authoritative node_state says token-AUTHORITATIVE
        node_state = _completed_node_state(token_id="token-AUTHORITATIVE")
        landscape = _StatePathAuditWriter(node_state=node_state)

        # But ctx.token says token-STALE
        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=landscape,
            state_id="state-1",
            token=make_token_info(token_id="token-STALE"),
        )
        with pytest.raises(FrameworkBugError, match="token mismatch"):
            ctx.record_call(
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data={"prompt": "test"},
                latency_ms=100.0,
            )

    def test_no_error_when_tokens_match(self) -> None:
        """When ctx.token.token_id matches node_state.token_id, no error."""
        node_state = _completed_node_state(token_id="token-row-1")
        landscape = _StatePathAuditWriter(node_state=node_state)

        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=landscape,
            state_id="state-1",
            token=make_token_info(token_id="token-row-1"),
        )
        # Should not raise — tokens are consistent
        ctx.record_call(
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "test"},
            latency_ms=100.0,
        )

    def test_no_error_when_ctx_token_is_none(self) -> None:
        """When ctx.token is None, skip the mismatch check (operation calls)."""
        node_state = _completed_node_state(token_id="token-1")
        landscape = _StatePathAuditWriter(node_state=node_state)

        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=landscape,
            state_id="state-1",
            token=None,
        )
        # Should not raise — no token to compare
        ctx.record_call(
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "test"},
            latency_ms=100.0,
        )
