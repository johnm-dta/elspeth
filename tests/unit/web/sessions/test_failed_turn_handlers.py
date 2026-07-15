"""Failed-turn response fields for composer error helpers."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from elspeth.contracts.errors import FailedTurnMetadata
from elspeth.web.composer.protocol import ComposerConvergenceError, ComposerPluginCrashError, ComposerRuntimePreflightError
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry
from elspeth.web.sessions.routes import _handle_convergence_error, _handle_plugin_crash, _handle_runtime_preflight_failure

_CATALOG = create_catalog_service()
_PLUGIN_SNAPSHOT = PluginAvailabilitySnapshot.for_trained_operator(_CATALOG)
_PROFILE_REGISTRY = MagicMock(spec=OperatorProfileRegistry)


@pytest.mark.asyncio
async def test_handle_convergence_error_returns_failed_turn() -> None:
    failed_turn = FailedTurnMetadata(
        assistant_message_id="assistant_1",
        tool_calls_attempted=2,
        tool_responses_persisted=2,
    )
    body = await _handle_convergence_error(
        ComposerConvergenceError(3, failed_turn=failed_turn),
        service=cast(Any, object()),
        session_id=uuid4(),
        user_id="user_1",
        log_prefix="test",
        llm_composition_state_id=None,
        settings=object(),
        secret_service=None,
        plugin_snapshot=_PLUGIN_SNAPSHOT,
        profile_registry=_PROFILE_REGISTRY,
    )

    assert body["failed_turn"] == {
        "assistant_message_id": "assistant_1",
        "tool_calls_attempted": 2,
        "tool_responses_persisted": 2,
        "transcript_url": None,
    }


@pytest.mark.asyncio
async def test_handle_plugin_crash_returns_failed_turn() -> None:
    failed_turn = FailedTurnMetadata(
        assistant_message_id="assistant_2",
        tool_calls_attempted=3,
        tool_responses_persisted=2,
    )
    body = await _handle_plugin_crash(
        ComposerPluginCrashError(RuntimeError("boom"), failed_turn=failed_turn),
        service=cast(Any, object()),
        session_id=uuid4(),
        user_id="user_1",
        log_prefix="test",
        llm_composition_state_id=None,
        settings=object(),
        secret_service=None,
        plugin_snapshot=_PLUGIN_SNAPSHOT,
        profile_registry=_PROFILE_REGISTRY,
    )

    assert body["failed_turn"] == {
        "assistant_message_id": "assistant_2",
        "tool_calls_attempted": 3,
        "tool_responses_persisted": 2,
        "transcript_url": None,
    }


@pytest.mark.asyncio
async def test_handle_plugin_crash_counts_persisted_tool_responses() -> None:
    class _CountingService:
        async def count_tool_responses_for_assistant_async(
            self,
            *,
            session_id: str,
            assistant_message_id: str | None,
        ) -> int:
            assert assistant_message_id == "assistant_counted"
            assert session_id
            return 4

    failed_turn = FailedTurnMetadata(
        assistant_message_id="assistant_counted",
        tool_calls_attempted=4,
        tool_responses_persisted=None,
    )
    body = await _handle_plugin_crash(
        ComposerPluginCrashError(RuntimeError("boom"), failed_turn=failed_turn),
        service=cast(Any, _CountingService()),
        session_id=uuid4(),
        user_id="user_1",
        log_prefix="test",
        llm_composition_state_id=None,
        settings=object(),
        secret_service=None,
        plugin_snapshot=_PLUGIN_SNAPSHOT,
        profile_registry=_PROFILE_REGISTRY,
    )

    assert body["failed_turn"] == {
        "assistant_message_id": "assistant_counted",
        "tool_calls_attempted": 4,
        "tool_responses_persisted": 4,
        "transcript_url": None,
    }


@pytest.mark.asyncio
async def test_handle_runtime_preflight_failure_returns_failed_turn() -> None:
    failed_turn = FailedTurnMetadata(
        assistant_message_id=None,
        tool_calls_attempted=1,
        tool_responses_persisted=0,
    )
    body = await _handle_runtime_preflight_failure(
        ComposerRuntimePreflightError(
            original_exc=RuntimeError("preflight"),
            partial_state=None,
            failed_turn=failed_turn,
        ),
        service=cast(Any, object()),
        session_id=uuid4(),
        user_id="user_1",
        log_prefix="test",
        llm_composition_state_id=None,
        settings=object(),
        secret_service=None,
        plugin_snapshot=_PLUGIN_SNAPSHOT,
        profile_registry=_PROFILE_REGISTRY,
    )

    assert body["failed_turn"] == {
        "assistant_message_id": None,
        "tool_calls_attempted": 1,
        "tool_responses_persisted": 0,
        "transcript_url": None,
    }
