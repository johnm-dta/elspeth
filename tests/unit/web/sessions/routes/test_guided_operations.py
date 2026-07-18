"""Route adapter tests for retry-safe composer operations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.sessions.protocol import (
    GuidedCompositionStateResult,
    GuidedOperationActive,
    GuidedOperationClaimed,
    GuidedOperationCompleted,
    GuidedOperationConflictError,
    GuidedOperationFailed,
    GuidedOperationFence,
)
from elspeth.web.sessions.routes.guided_operations import (
    GuidedOperationLease,
    guided_response_hash,
    reserve_or_replay_guided_operation,
)
from elspeth.web.sessions.schemas import ConvertGuidedRequest


class _Response(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    value: str


class _Service:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.reserve_calls = 0
        self.get_calls = 0

    async def reserve_guided_operation(self, **_kwargs):
        self.reserve_calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def get_guided_operation(self, **_kwargs):
        self.get_calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _request() -> ConvertGuidedRequest:
    return ConvertGuidedRequest(operation_id="00000000-0000-4000-8000-000000000001")


@pytest.mark.asyncio
async def test_new_operation_returns_live_fence() -> None:
    session_id = uuid4()
    fence = GuidedOperationFence(session_id=session_id, operation_id=_request().operation_id, lease_token="secret", attempt=1)
    service = _Service([GuidedOperationClaimed(fence=fence, lease_expires_at=datetime.now(UTC) + timedelta(minutes=1))])

    result = await reserve_or_replay_guided_operation(
        service=service,
        session_id=session_id,
        kind="guided_convert",
        request=_request(),
        replay=lambda _locator: _never(),
    )

    assert result == GuidedOperationLease(fence=fence)


@pytest.mark.asyncio
async def test_operation_id_reuse_conflict_is_static_409() -> None:
    session_id = uuid4()
    service = _Service([GuidedOperationConflictError(session_id=session_id, operation_id=_request().operation_id)])

    with pytest.raises(HTTPException) as caught:
        await reserve_or_replay_guided_operation(
            service=service,
            session_id=session_id,
            kind="guided_convert",
            request=_request(),
            replay=lambda _locator: _never(),
        )

    assert caught.value.status_code == 409
    assert caught.value.detail == "Operation id is already bound to a different request."


@pytest.mark.asyncio
async def test_completed_operation_reconstructs_exact_strict_response() -> None:
    session_id = uuid4()
    locator = GuidedCompositionStateResult(state_id=uuid4())
    response = _Response(value="same")
    service = _Service([GuidedOperationCompleted(result=locator, response_hash=guided_response_hash(response))])

    async def replay(actual):
        assert actual == locator
        return response

    result = await reserve_or_replay_guided_operation(
        service=service,
        session_id=session_id,
        kind="guided_convert",
        request=_request(),
        replay=replay,
    )

    assert result == response


@pytest.mark.asyncio
async def test_completed_operation_rejects_response_domain_hash_mismatch() -> None:
    session_id = uuid4()
    locator = GuidedCompositionStateResult(state_id=uuid4())
    service = _Service([GuidedOperationCompleted(result=locator, response_hash="0" * 64)])

    with pytest.raises(AuditIntegrityError, match="response hash"):
        await reserve_or_replay_guided_operation(
            service=service,
            session_id=session_id,
            kind="guided_convert",
            request=_request(),
            replay=lambda _locator: _response("changed"),
        )


@pytest.mark.asyncio
async def test_active_operation_polls_to_terminal_replay(monkeypatch) -> None:
    session_id = uuid4()
    locator = GuidedCompositionStateResult(state_id=uuid4())
    response = _Response(value="joined")
    service = _Service(
        [
            GuidedOperationActive(attempt=1, lease_expires_at=datetime.now(UTC) + timedelta(seconds=1)),
            GuidedOperationCompleted(result=locator, response_hash=guided_response_hash(response)),
        ]
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("elspeth.web.sessions.routes.guided_operations.asyncio.sleep", no_sleep)
    result = await reserve_or_replay_guided_operation(
        service=service,
        session_id=session_id,
        kind="guided_convert",
        request=_request(),
        replay=lambda _locator: _response("joined"),
    )

    assert result == response
    assert service.get_calls == 1


@pytest.mark.asyncio
async def test_failed_operation_maps_only_closed_safe_failure() -> None:
    service = _Service([GuidedOperationFailed(failure_code="provider_timeout")])

    with pytest.raises(HTTPException) as caught:
        await reserve_or_replay_guided_operation(
            service=service,
            session_id=uuid4(),
            kind="guided_convert",
            request=_request(),
            replay=lambda _locator: _never(),
        )

    assert caught.value.status_code == 504
    assert caught.value.detail == "The operation timed out. Retry with a new operation id."


async def _response(value: str) -> _Response:
    return _Response(value=value)


async def _never() -> _Response:
    raise AssertionError("replay callback must not run")
