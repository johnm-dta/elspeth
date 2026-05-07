"""Runtime preflight in-flight coordination tests."""

from __future__ import annotations

import asyncio

import pytest

from elspeth.web.execution.runtime_preflight import (
    RuntimePreflightCoordinator,
    RuntimePreflightFailure,
    RuntimePreflightKey,
)
from elspeth.web.execution.schemas import ValidationResult


@pytest.mark.asyncio
async def test_coordinator_deduplicates_concurrent_same_session_state_settings() -> None:
    coordinator = RuntimePreflightCoordinator()
    key = RuntimePreflightKey(
        session_scope="session:abc123",
        state_version=7,
        settings_hash="settings-hash",
    )
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()
    expected = ValidationResult(is_valid=True, checks=[], errors=[])

    async def worker() -> ValidationResult:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return expected

    first_task = asyncio.create_task(coordinator.run(key, worker))
    await started.wait()
    second_task = asyncio.create_task(coordinator.run(key, worker))

    await asyncio.sleep(0)
    release.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert first is expected
    assert second is expected
    assert calls == 1


@pytest.mark.asyncio
async def test_coordinator_deduplicates_concurrent_failure_for_same_key() -> None:
    coordinator = RuntimePreflightCoordinator()
    key = RuntimePreflightKey(
        session_scope="session:abc123",
        state_version=7,
        settings_hash="settings-hash",
    )
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()
    original = RuntimeError("constructor failed")

    async def worker() -> ValidationResult:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        raise original

    first_task = asyncio.create_task(coordinator.run(key, worker))
    await started.wait()
    second_task = asyncio.create_task(coordinator.run(key, worker))

    await asyncio.sleep(0)
    release.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert isinstance(first, RuntimePreflightFailure)
    assert isinstance(second, RuntimePreflightFailure)
    assert first.original_exc is original
    assert second.original_exc is original
    assert calls == 1


@pytest.mark.asyncio
async def test_coordinator_evicts_inflight_entry_when_only_awaiter_is_cancelled() -> None:
    """Mid-flight cancellation must not leak the in-flight dict entry.

    The common composer path: a per-compose timeout fires, the awaiter is
    cancelled while the worker is still running, and no future caller arrives
    with the same key (because state_version rotates on every state mutation).
    Without an eviction-on-done callback, the dict entry would stay for the
    life of the process.
    """
    coordinator = RuntimePreflightCoordinator()
    key = RuntimePreflightKey(
        session_scope="session:abc123",
        state_version=11,
        settings_hash="settings-hash",
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def worker() -> ValidationResult:
        started.set()
        await release.wait()
        return ValidationResult(is_valid=True, checks=[], errors=[])

    awaiter = asyncio.create_task(coordinator.run(key, worker))
    await started.wait()
    assert key in coordinator._inflight

    awaiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await awaiter

    # Worker is still running; awaiter is cancelled. Release the worker and
    # let its done-callback fire on the event loop.
    release.set()
    for _ in range(10):
        await asyncio.sleep(0)
        if key not in coordinator._inflight:
            break

    assert key not in coordinator._inflight


@pytest.mark.asyncio
async def test_coordinator_does_not_share_different_session_scopes() -> None:
    coordinator = RuntimePreflightCoordinator()
    calls = 0

    async def worker() -> ValidationResult:
        nonlocal calls
        calls += 1
        return ValidationResult(is_valid=True, checks=[], errors=[])

    await asyncio.gather(
        coordinator.run(RuntimePreflightKey("session:http", 1, "settings"), worker),
        coordinator.run(RuntimePreflightKey("session:mcp", 1, "settings"), worker),
    )

    assert calls == 2
