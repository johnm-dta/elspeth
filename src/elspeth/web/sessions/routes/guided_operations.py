"""HTTP lifecycle adapter for retry-safe composer mutations.

This module deliberately sits outside ``composer/guided.py``.  Several legacy
guided handlers carry governance fingerprints whose AST locations are stable;
centralising the retry protocol here avoids inserting module-level definitions
above those handlers while the pre-release cutover replaces them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Never, overload
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import stable_hash
from elspeth.web.sessions.guided_operations import guided_operation_request_hash
from elspeth.web.sessions.protocol import (
    GuidedOperationActive,
    GuidedOperationClaimed,
    GuidedOperationCompleted,
    GuidedOperationConflictError,
    GuidedOperationFailed,
    GuidedOperationFence,
    GuidedOperationKind,
    GuidedOperationOutcome,
    GuidedOperationResult,
    GuidedOperationTakenOver,
    SessionServiceProtocol,
)

_ACTOR = "composer_route"
_LEASE_SECONDS = 300
_POLL_SECONDS = 0.05

_SAFE_FAILURES: dict[str, tuple[int, str]] = {
    "provider_unavailable": (503, "The provider is unavailable. Retry with a new operation id."),
    "provider_timeout": (504, "The operation timed out. Retry with a new operation id."),
    "invalid_provider_response": (502, "The provider returned an invalid response. Retry with a new operation id."),
    "stale_conflict": (409, "The guided state changed before settlement. Reload the authoritative state."),
    "integrity_error": (500, "The operation failed an integrity check."),
    "custody_error": (500, "The operation could not establish result custody."),
    "quota_exceeded": (413, "The operation exceeded the session storage quota."),
    "operation_failed": (500, "The operation failed."),
}


@dataclass(frozen=True, slots=True)
class GuidedOperationLease:
    """A route owns the only fence authorised to perform durable writes."""

    fence: GuidedOperationFence


@dataclass(frozen=True, slots=True)
class GuidedOperationExpired:
    """Replay-only lookup found an existing operation whose lease expired."""

    attempt: int


def guided_response_hash(response: BaseModel) -> str:
    """Hash the complete strict HTTP response domain used for replay."""

    config = type(response).model_config
    if config.get("strict") is not True or config.get("extra") != "forbid":
        raise AuditIntegrityError("Guided operation replay requires a strict, extra-forbid response DTO")
    # Re-validate the emitted representation strictly.  Constructed Pydantic
    # instances can bypass validation through ``model_construct``; a replay
    # hash must never bless such an object as Tier 1 response evidence.
    strict_response = type(response).model_validate(response.model_dump(mode="python"), strict=True)
    return stable_hash(strict_response.model_dump(mode="json"))


def raise_guided_operation_failure(outcome: GuidedOperationFailed) -> Never:
    """Raise the closed HTTP failure represented by a terminal operation."""

    safe = _SAFE_FAILURES.get(outcome.failure_code)
    if safe is None:
        raise AuditIntegrityError("Guided operation returned an unknown failure code")
    status_code, detail = safe
    raise HTTPException(
        status_code=status_code,
        detail={
            "error_type": "guided_operation_terminal_failure",
            "failure_code": outcome.failure_code,
            "detail": detail,
        },
    )


async def _replay_completed[ResponseT: BaseModel](
    outcome: GuidedOperationCompleted,
    replay: Callable[[GuidedOperationResult], Awaitable[ResponseT]],
) -> ResponseT:
    response = await replay(outcome.result)
    if guided_response_hash(response) != outcome.response_hash:
        raise AuditIntegrityError("Guided operation replay response hash does not match its stored response hash")
    return response


@overload
async def reserve_or_replay_guided_operation[ResponseT: BaseModel](
    *,
    service: SessionServiceProtocol,
    session_id: UUID,
    kind: GuidedOperationKind,
    request: BaseModel,
    replay: Callable[[GuidedOperationResult], Awaitable[ResponseT]],
    reserve_if_absent: Literal[False],
    takeover_expired: Literal[False],
) -> GuidedOperationLease | GuidedOperationExpired | ResponseT | None: ...


@overload
async def reserve_or_replay_guided_operation[ResponseT: BaseModel](
    *,
    service: SessionServiceProtocol,
    session_id: UUID,
    kind: GuidedOperationKind,
    request: BaseModel,
    replay: Callable[[GuidedOperationResult], Awaitable[ResponseT]],
    reserve_if_absent: bool = True,
    takeover_expired: Literal[True] = True,
) -> GuidedOperationLease | ResponseT | None: ...


@overload
async def reserve_or_replay_guided_operation[ResponseT: BaseModel](
    *,
    service: SessionServiceProtocol,
    session_id: UUID,
    kind: GuidedOperationKind,
    request: BaseModel,
    replay: Callable[[GuidedOperationResult], Awaitable[ResponseT]],
    reserve_if_absent: Literal[True] = True,
    takeover_expired: Literal[False],
) -> Never: ...


async def reserve_or_replay_guided_operation[ResponseT: BaseModel](
    *,
    service: SessionServiceProtocol,
    session_id: UUID,
    kind: GuidedOperationKind,
    request: BaseModel,
    replay: Callable[[GuidedOperationResult], Awaitable[ResponseT]],
    reserve_if_absent: bool = True,
    takeover_expired: bool = True,
) -> GuidedOperationLease | GuidedOperationExpired | ResponseT | None:
    """Claim one operation or synchronously join its immutable terminal result.

    Active requests are polled to a terminal result.  Once a lease expires the
    caller returns to the atomic reserve primitive, which either performs the
    sole takeover or observes the competing taker's active/terminal outcome.
    The HTTP surface never exposes an intermediate 202 response.

    ``takeover_expired=False`` makes an existing expired operation return a
    distinct marker so a route can finish live preflight before acquiring a
    new fence. This mode is valid only with ``reserve_if_absent=False``.
    Non-expired active operations are still joined outside route state locks.
    """

    if not takeover_expired and reserve_if_absent:
        raise AuditIntegrityError("Non-taking-over guided operation lookup must not reserve an absent operation")

    operation_id = request.model_dump(mode="python").get("operation_id")
    if not isinstance(operation_id, str):
        raise AuditIntegrityError("Strict guided operation request has a non-string operation_id")
    request_hash = guided_operation_request_hash(session_id=session_id, kind=kind, request=request)

    async def reserve() -> GuidedOperationOutcome:
        try:
            return await service.reserve_guided_operation(
                session_id=session_id,
                operation_id=operation_id,
                kind=kind,
                request_hash=request_hash,
                actor=_ACTOR,
                lease_seconds=_LEASE_SECONDS,
            )
        except GuidedOperationConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail="Operation id is already bound to a different request.",
            ) from exc

    if reserve_if_absent:
        outcome: GuidedOperationOutcome = await reserve()
        observed_by_get = False
    else:
        try:
            existing = await service.get_guided_operation(
                session_id=session_id,
                operation_id=operation_id,
                kind=kind,
                request_hash=request_hash,
            )
        except GuidedOperationConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail="Operation id is already bound to a different request.",
            ) from exc
        if existing is None:
            return None
        outcome = existing
        observed_by_get = True
    while True:
        if isinstance(outcome, (GuidedOperationClaimed, GuidedOperationTakenOver)):
            return GuidedOperationLease(fence=outcome.fence)
        if isinstance(outcome, GuidedOperationCompleted):
            return await _replay_completed(outcome, replay)
        if isinstance(outcome, GuidedOperationFailed):
            raise_guided_operation_failure(outcome)
        if not isinstance(outcome, GuidedOperationActive):
            raise AuditIntegrityError("Guided operation reserve returned an unknown outcome")

        if outcome.expired and observed_by_get:
            if not takeover_expired:
                return GuidedOperationExpired(attempt=outcome.attempt)
            # ``expired`` is computed against the database clock by the read
            # primitive. Never compare the persisted timestamp with the web
            # host clock: even modest skew can create a tight reserve loop.
            outcome = await reserve()
            observed_by_get = False
            continue
        await asyncio.sleep(_POLL_SECONDS)
        try:
            observed = await service.get_guided_operation(
                session_id=session_id,
                operation_id=operation_id,
                kind=kind,
                request_hash=request_hash,
            )
        except GuidedOperationConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail="Operation id is already bound to a different request.",
            ) from exc
        if observed is None:
            raise AuditIntegrityError("Guided operation disappeared while a caller was joining it")
        outcome = observed
        observed_by_get = True
