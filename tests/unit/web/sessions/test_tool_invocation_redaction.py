"""Regression coverage for legacy composer tool-invocation persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
from elspeth.core.canonical import canonical_json
from elspeth.web.sessions.protocol import SessionServiceProtocol
from elspeth.web.sessions.routes._helpers import _persist_tool_invocations


@dataclass
class _CapturedMessage:
    session_id: UUID
    role: str
    content: str
    kwargs: dict[str, Any]


@dataclass
class _CapturingSessionService:
    messages: list[_CapturedMessage] = field(default_factory=list)

    async def add_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
        **kwargs: Any,
    ) -> None:
        self.messages.append(_CapturedMessage(session_id=session_id, role=role, content=content, kwargs=kwargs))


def _hash_canonical(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@pytest.mark.asyncio
async def test_legacy_tool_invocation_persistence_redacts_advisor_payloads() -> None:
    """Legacy route drains must not mirror raw advisor arguments or guidance."""

    raw_problem = "RAW_PROBLEM: user pasted an internal exception and partial schema"
    raw_error = "RAW_ERROR: validator echoed the user's private column name"
    raw_action = "RAW_ACTION: tried set_pipeline with sensitive prose"
    raw_schema = "RAW_SCHEMA: internal schema excerpt"
    raw_guidance = "RAW_GUIDANCE: frontier model advice with sensitive details"
    arguments = {
        "trigger": "reactive_stuck",
        "problem_summary": raw_problem,
        "recent_errors": [raw_error],
        "attempted_actions": [raw_action],
        "schema_excerpt": raw_schema,
    }
    result = {
        "status": "SUCCESS",
        "guidance": raw_guidance,
        "model": "frontier-test",
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "cached_prompt_tokens": 0,
        "advisor_latency_ms": 42,
        "budget_used": 1,
        "budget_remaining": 2,
        "note": "metadata is safe",
    }
    arguments_canonical = canonical_json(arguments)
    result_canonical = canonical_json(result)
    invocation = ComposerToolInvocation(
        tool_call_id="call_advisor_1",
        tool_name="request_advisor_hint",
        arguments_canonical=arguments_canonical,
        arguments_hash=_hash_canonical(arguments_canonical),
        result_canonical=result_canonical,
        result_hash=_hash_canonical(result_canonical),
        status=ComposerToolStatus.SUCCESS,
        error_class=None,
        error_message=None,
        version_before=3,
        version_after=3,
        started_at=datetime(2026, 5, 24, tzinfo=UTC),
        finished_at=datetime(2026, 5, 24, tzinfo=UTC),
        latency_ms=12,
        actor="composer-web:user-test",
    )
    service = _CapturingSessionService()

    await _persist_tool_invocations(
        cast(SessionServiceProtocol, service),
        uuid4(),
        (invocation,),
        composition_state_id=None,
        parent_assistant_id=uuid4(),
        plugin_crash_pending=False,
    )

    assert len(service.messages) == 1
    message = service.messages[0]
    persisted_blob = json.dumps(
        {
            "content": json.loads(message.content),
            "tool_calls": message.kwargs["tool_calls"],
        },
        sort_keys=True,
    )
    assert raw_problem not in persisted_blob
    assert raw_error not in persisted_blob
    assert raw_action not in persisted_blob
    assert raw_schema not in persisted_blob
    assert raw_guidance not in persisted_blob
    assert "<advisor-problem-summary:" in persisted_blob
    assert "<advisor-recent-errors:1-entries>" in persisted_blob
    assert "<advisor-attempted-actions:1-entries>" in persisted_blob
    assert "<advisor-schema-excerpt:" in persisted_blob
    assert '"guidance": "<redacted>"' in persisted_blob


@pytest.mark.asyncio
async def test_arg_error_result_for_response_model_tool_persists_without_success_model_validation() -> None:
    """ARG_ERROR result payloads are failure envelopes, not success responses."""

    arguments = {"blob_id": str(uuid4())}
    result = {"error": "Tool 'get_blob_content' failed: blob is not ready"}
    arguments_canonical = canonical_json(arguments)
    result_canonical = canonical_json(result)
    invocation = ComposerToolInvocation(
        tool_call_id="call_blob_error_1",
        tool_name="get_blob_content",
        arguments_canonical=arguments_canonical,
        arguments_hash=_hash_canonical(arguments_canonical),
        result_canonical=result_canonical,
        result_hash=_hash_canonical(result_canonical),
        status=ComposerToolStatus.ARG_ERROR,
        error_class="ToolArgumentError",
        error_message="blob is not ready",
        version_before=3,
        version_after=None,
        started_at=datetime(2026, 5, 24, tzinfo=UTC),
        finished_at=datetime(2026, 5, 24, tzinfo=UTC),
        latency_ms=12,
        actor="composer-web:user-test",
    )
    service = _CapturingSessionService()

    await _persist_tool_invocations(
        cast(SessionServiceProtocol, service),
        uuid4(),
        (invocation,),
        composition_state_id=None,
        parent_assistant_id=None,
        plugin_crash_pending=False,
    )

    assert len(service.messages) == 1
    message = service.messages[0]
    assert json.loads(message.content) == result
    envelope = message.kwargs["tool_calls"][0]
    assert envelope["_kind"] == "audit"
    assert envelope["invocation"]["result_canonical"] == result_canonical
