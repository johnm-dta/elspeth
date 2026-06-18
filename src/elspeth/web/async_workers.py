"""Async helpers for running bounded synchronous work off the event loop."""

from __future__ import annotations

import asyncio
import functools
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

_SHARED_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()


def _get_shared_executor() -> ThreadPoolExecutor:
    """Retrieve or construct the bounded process-wide ThreadPoolExecutor."""
    global _SHARED_EXECUTOR
    if _SHARED_EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _SHARED_EXECUTOR is None:
                # Sensible limit to prevent unbounded thread creation.
                # 16 concurrent workers is robust for typical Web IO tasks.
                _SHARED_EXECUTOR = ThreadPoolExecutor(
                    max_workers=16,
                    thread_name_prefix="async-worker",
                )
    return _SHARED_EXECUTOR


async def shutdown_async_workers() -> None:
    """Shut down the shared thread pool executor."""
    global _SHARED_EXECUTOR
    if _SHARED_EXECUTOR is not None:
        loop = asyncio.get_running_loop()
        executor = _SHARED_EXECUTOR
        _SHARED_EXECUTOR = None
        await loop.run_in_executor(None, executor.shutdown, True)


def _drain_future_exception[T](future: asyncio.Future[T]) -> None:
    """Mark a future's exception as retrieved so asyncio's GC handler stays quiet.

    Background — elspeth-e4949acbe1: when ``run_sync_in_worker``'s caller is
    cancelled (outer ``asyncio.wait_for`` timeout, or a ``CancelledError``
    raised on the request task), the future continues running on its worker
    thread because ``asyncio.wait`` never cancels the awaitables it is
    waiting on. If the underlying sync work eventually raises, the asyncio
    future holds an unretrieved exception.
    Python's GC then fires the loop's exception handler with the message
    ``"Future exception was never retrieved"``, whose default handler logs
    a traceback through ``logging.getLogger("asyncio")``. In production that
    traceback's most recent in-stack frame is ``web/middleware/request_id.py``
    line ``await call_next(request)`` — operators saw "the request-id
    middleware crashed" when in fact the middleware did exactly its job
    and a deeper worker thread completed-with-exception after its caller
    had already given up. Calling ``.exception()`` here marks the
    exception as retrieved and silences the misleading journal noise.

    The exception itself is intentionally discarded: the only reason we
    reach this drain path is that the caller's task has been cancelled
    (so ``run_sync_in_worker`` exited via its ``finally`` while the future
    was still running), so the abandoned work's outcome is no longer
    load-bearing for the cancelled request. If the worker raised because of a real
    infrastructure problem (DB down, disk full), other concurrent
    requests will surface it through their own paths — this drain does
    not hide such a failure mode, only its echo on a request that is
    already on its way out.
    """
    if future.cancelled():
        return
    # ``Future.exception()`` is the documented way to "retrieve" the
    # exception without re-raising it on the caller. The return value is
    # the exception object; we deliberately drop it.
    future.exception()


async def run_sync_in_worker[**P, T](func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run synchronous work in a process-wide shared worker pool without blocking the loop.

    The short wait loop keeps an explicit event-loop timer active while the
    worker runs.  This preserves the normal async-over-sync contract even in
    sandboxed runtimes where executor completion can fail to wake the selector
    promptly.
    """
    loop = asyncio.get_running_loop()
    executor = _get_shared_executor()
    future = loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))
    try:
        # ``asyncio.wait`` returns ``(done, pending)`` on each 0.1s tick rather
        # than raising on timeout, so there is no error-shaped sentinel to
        # catch — the empty ``done`` set is the explicit "not finished yet"
        # signal and we simply loop again. The short timeout keeps an explicit
        # event-loop timer active so the selector wakes promptly even in
        # sandboxed runtimes. Unlike ``wait_for(shield(...))``, ``asyncio.wait``
        # never cancels the future it is waiting on, so no ``shield`` wrapper is
        # needed to keep the worker alive across a caller cancellation.
        while True:
            done, _pending = await asyncio.wait({future}, timeout=0.1)
            if done:
                # ``future.result()`` retrieves the value or re-raises the
                # worker's exception on the awaited (non-cancelled) path.
                return future.result()
    finally:
        # If the await above was abandoned mid-flight (cancellation,
        # outer timeout), the future may still be running. Make sure any
        # exception it eventually raises is retrieved so the misleading
        # request_id-middleware traceback never reaches the journal — see
        # ``_drain_future_exception``'s docstring.
        if not future.done():
            future.add_done_callback(_drain_future_exception)
