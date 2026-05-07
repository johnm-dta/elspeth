"""Tests for run_sync_in_worker — async-over-sync executor bridge.

Pins the cancellation contract that triggered elspeth-e4949acbe1: when a
caller is cancelled (outer asyncio.wait_for timeout, or a CancelledError
from the request task itself), the shielded future continues running on
its worker thread. If that thread eventually raises, the asyncio Future
holds an unretrieved exception, and Python's GC emits a misleading
"Future exception was never retrieved" traceback through the asyncio
exception handler — operators saw this surface as a request-id
middleware traceback during composer load because the most recent
in-stack frame was the middleware's ``await call_next(request)``.

We intercept the asyncio loop's exception handler directly because
``redirect_stderr`` does not capture this output: asyncio routes the
warning through ``logger.getLogger("asyncio")`` whose default handler
holds a reference to the pre-redirect ``sys.stderr``.

The fix is a done-callback drain in ``finally`` so the exception is
considered retrieved. These tests pin that contract so a future refactor
of ``async_workers.py`` cannot reintroduce the journal noise.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from elspeth.web.async_workers import run_sync_in_worker


class _LoopExceptionRecorder:
    """Capture asyncio loop exception events by replacing the handler."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._previous: Any | None = None

    def install(self, loop: asyncio.AbstractEventLoop) -> None:
        self._previous = loop.get_exception_handler()
        loop.set_exception_handler(self._handle)

    def uninstall(self, loop: asyncio.AbstractEventLoop) -> None:
        loop.set_exception_handler(self._previous)

    def _handle(self, loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        self.events.append(dict(context))

    def messages(self) -> list[str]:
        return [str(event.get("message", "")) for event in self.events]

    def has_unretrieved_future_exception(self) -> bool:
        return any("Future exception was never retrieved" in msg for msg in self.messages())


def _slow_then_raise(seconds: float, message: str) -> str:
    """Worker that sleeps and then raises — models a sync op that fails late."""
    time.sleep(seconds)
    raise RuntimeError(message)


def _slow_success(seconds: float) -> str:
    time.sleep(seconds)
    return "ok"


def _run_in_isolated_loop(coro_factory) -> _LoopExceptionRecorder:
    """Run ``coro_factory()`` in a fresh event loop and return the recorded
    exception events.

    A fresh loop is used because asyncio's "Future exception was never
    retrieved" event fires from ``Future.__del__`` during loop GC/close,
    not during normal coroutine execution. With pytest-asyncio's shared
    per-test loop, the warning would fire AFTER the test function returns
    — too late for in-test asserts. By owning the loop ourselves, we can
    observe events emitted during ``loop.close()`` while our exception
    handler is still installed.
    """
    recorder = _LoopExceptionRecorder()
    loop = asyncio.new_event_loop()
    recorder.install(loop)
    try:
        loop.run_until_complete(coro_factory())
        # Drain any callbacks (including future-finalisation callbacks)
        # that were scheduled but not yet run.
        loop.run_until_complete(asyncio.sleep(0))
    finally:
        loop.close()
    return recorder


def test_normal_completion_returns_value() -> None:
    """Sanity: happy path returns the worker's value with no events."""

    async def scenario() -> None:
        result = await run_sync_in_worker(_slow_success, 0.05)
        assert result == "ok"

    recorder = _run_in_isolated_loop(scenario)
    assert recorder.events == []


def test_normal_exception_propagates() -> None:
    """Sanity: when sync work raises and the caller awaits, the exception
    propagates and is consumed normally (no loop exception event)."""

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="propagated cleanly"):
            await run_sync_in_worker(_slow_then_raise, 0.05, "propagated cleanly")

    recorder = _run_in_isolated_loop(scenario)
    assert not recorder.has_unretrieved_future_exception()


def test_outer_timeout_cancels_during_failing_sync_does_not_warn() -> None:
    """The elspeth-e4949acbe1 reproduction.

    Outer wait_for cancels the run_sync_in_worker await; the worker
    thread eventually raises; the shielded future is no longer awaited;
    asyncio MUST NOT emit a "Future exception was never retrieved" event
    via the loop's exception handler.

    Composer routes wrap ``run_sync_in_worker(_runtime_preflight, ...)``
    in ``asyncio.wait_for(timeout=composer_runtime_preflight_timeout_seconds)``.
    A timeout there + a real validation error in the underlying preflight
    is the production trigger that produced the misleading request_id
    middleware traceback in the journal.
    """

    async def scenario() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                run_sync_in_worker(_slow_then_raise, 1.0, "thread raised after cancel"),
                timeout=0.2,
            )
        # Let the worker thread finish so the future transitions to done.
        await asyncio.sleep(1.5)

    recorder = _run_in_isolated_loop(scenario)
    assert not recorder.has_unretrieved_future_exception(), (
        "run_sync_in_worker did not drain the shielded future's exception. "
        "asyncio fired its 'Future exception was never retrieved' handler "
        "with this context history:\n"
        + "\n".join(f"  - {msg}" for msg in recorder.messages())
        + "\n\nThis regression surfaces in the production journal as a "
        "traceback whose most recent in-stack frame is request_id.py:98 "
        "(the request-id middleware's await call_next), making operators "
        "believe the middleware crashed when it did not."
    )


def test_direct_cancel_during_failing_sync_does_not_warn() -> None:
    """Same contract as the outer-timeout test, but the cancel originates
    from a direct ``task.cancel()`` — modelling the new client-disconnect
    cancellation path added with elspeth-29e8bd8a1f. The drain MUST hold
    on this entry point too or the in-flight observability work will
    actively make the journal noise worse.
    """

    async def scenario() -> None:
        task = asyncio.create_task(run_sync_in_worker(_slow_then_raise, 1.0, "thread raised after task cancel"))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(1.5)

    recorder = _run_in_isolated_loop(scenario)
    assert not recorder.has_unretrieved_future_exception(), "run_sync_in_worker did not drain under direct task.cancel():\n" + "\n".join(
        f"  - {msg}" for msg in recorder.messages()
    )
