"""Tests for PluginContext.record_call() FrameworkBugError guard clauses.

These test the offensive programming guards that detect framework bugs:
- No landscape configured
- XOR violation: both state_id and operation_id set
- XOR violation: neither state_id nor operation_id set
- state_id set but node_state lookup returns None
- Token mismatch between ctx.token and authoritative node_state
"""

from unittest.mock import Mock

import pytest

from elspeth.contracts import FrameworkBugError
from elspeth.contracts.enums import CallStatus, CallType
from elspeth.contracts.plugin_context import PluginContext
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from tests.fixtures.factories import make_token_info


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
        mock_landscape = Mock()
        mock_landscape.allocate_call_index = Mock(return_value=0)
        mock_landscape.record_call = Mock(return_value=Mock())
        mock_landscape.get_node_state = Mock(return_value=None)

        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=mock_landscape,
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
        mock_node_state = Mock()
        mock_node_state.token_id = "token-AUTHORITATIVE"

        mock_landscape = Mock()
        mock_landscape.allocate_call_index = Mock(return_value=0)
        mock_landscape.record_call = Mock(return_value=Mock())
        mock_landscape.get_node_state = Mock(return_value=mock_node_state)

        # But ctx.token says token-STALE
        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=mock_landscape,
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
        mock_node_state = Mock()
        mock_node_state.token_id = "token-row-1"

        mock_landscape = Mock()
        mock_landscape.allocate_call_index = Mock(return_value=0)
        mock_landscape.record_call = Mock(return_value=Mock())
        mock_landscape.get_node_state = Mock(return_value=mock_node_state)

        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=mock_landscape,
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
        mock_node_state = Mock()
        mock_node_state.token_id = "token-1"

        mock_landscape = Mock()
        mock_landscape.allocate_call_index = Mock(return_value=0)
        mock_landscape.record_call = Mock(return_value=Mock())
        mock_landscape.get_node_state = Mock(return_value=mock_node_state)

        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=mock_landscape,
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
