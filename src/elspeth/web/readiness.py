"""Bounded, redacted dependency readiness probes for the web service."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

ReadinessProbeLabel = Literal["session", "landscape", "data_dir", "payload_store", "blob_dir"]


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    ready: bool
    checks: tuple[ReadinessCheck, ...]


_ProbeResult = tuple[ReadinessCheck, ...]
_SourceFuture = concurrent.futures.Future[_ProbeResult]


def _static_failures(check_names: tuple[str, ...], detail: str) -> _ProbeResult:
    return tuple(ReadinessCheck(name=name, ok=False, detail=detail) for name in check_names)


def _exception_failures(check_names: tuple[str, ...], exc: BaseException) -> _ProbeResult:
    exc_class = type(exc).__name__
    db_name = next((name for name in check_names if name.endswith("_db")), None)
    if db_name is None:
        return _static_failures(check_names, f"probe failed ({exc_class})")
    schema_name = f"{db_name.removesuffix('_db')}_schema"
    return tuple(
        ReadinessCheck(
            name=name,
            ok=False,
            detail=(
                f"probe failed ({exc_class})"
                if name == db_name
                else "not checked: connectivity probe failed"
                if name == schema_name
                else f"probe failed ({exc_class})"
            ),
        )
        for name in check_names
    )


def _drain_source(future: _SourceFuture) -> None:
    if future.cancelled():
        return
    with contextlib.suppress(BaseException):
        future.exception()


def _drain_wrapped(future: asyncio.Future[_ProbeResult]) -> None:
    if future.cancelled():
        return
    with contextlib.suppress(BaseException):
        future.exception()


class ReadinessProbeRunner:
    """Admit at most one unresolved worker future for each closed label."""

    def __init__(self) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=5,
            thread_name_prefix="readiness-worker",
        )
        self._lock = threading.Lock()
        self._futures: dict[ReadinessProbeLabel, _SourceFuture] = {}
        self._closed = False

    async def run(
        self,
        label: ReadinessProbeLabel,
        check_names: tuple[str, ...],
        fn: Callable[..., _ProbeResult],
        *args: object,
    ) -> _ProbeResult:
        with self._lock:
            if self._closed:
                return _static_failures(check_names, "probe runner closed")
            existing = self._futures.get(label)
            if existing is not None and not existing.done():
                return _static_failures(check_names, "probe already in flight")
            source = self._executor.submit(fn, *args)
            self._futures[label] = source

        def source_done(completed: _SourceFuture) -> None:
            with self._lock:
                if self._futures.get(label) is completed:
                    del self._futures[label]
            _drain_source(completed)

        # A Future that completed synchronously calls this callback inline.
        # Registration therefore must remain outside the registry lock.
        source.add_done_callback(source_done)

        loop = asyncio.get_running_loop()
        wrapped = asyncio.wrap_future(source, loop=loop)
        wrapped.add_done_callback(_drain_wrapped)
        deadline = loop.time() + 2.0
        try:
            while not wrapped.done():
                remaining = deadline - loop.time()
                if remaining <= 0:
                    wrapped.cancel()
                    source.cancel()
                    _drain_wrapped(wrapped)
                    return _static_failures(check_names, "probe timed out")
                await asyncio.wait({wrapped}, timeout=min(0.1, remaining))
            try:
                return wrapped.result()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                return _exception_failures(check_names, exc)
        except asyncio.CancelledError:
            wrapped.cancel()
            source.cancel()
            _drain_wrapped(wrapped)
            raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = tuple(self._futures.values())
        for future in futures:
            future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)


class ReadinessCache:
    """Cancellation-safe single-flight cache with a two-second success TTL."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._report: ReadinessReport | None = None
        self._completed_at: float | None = None
        self._task: asyncio.Task[ReadinessReport] | None = None
        self._task_completed_at: float | None = None

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    def _record_completion(self, task: asyncio.Task[ReadinessReport]) -> None:
        if self._task is task:
            self._task_completed_at = self._now()

    def _harvest_locked(self, task: asyncio.Task[ReadinessReport]) -> ReadinessReport | None:
        try:
            report = task.result()
        except BaseException:
            if self._task is task:
                self._task = None
                self._task_completed_at = None
            return None
        if self._task is task:
            self._report = report
            self._completed_at = self._task_completed_at if self._task_completed_at is not None else self._now()
            self._task = None
            self._task_completed_at = None
        return report

    async def get(self, compute: Callable[[], Awaitable[ReadinessReport]]) -> ReadinessReport:
        async with self._lock:
            if self._task is not None and self._task.done():
                self._harvest_locked(self._task)

            now = self._now()
            if self._report is not None and self._completed_at is not None and now - self._completed_at < 2.0:
                return self._report

            if self._task is None:

                async def invoke() -> ReadinessReport:
                    return await compute()

                self._task = asyncio.create_task(invoke())
                self._task_completed_at = None
                self._task.add_done_callback(self._record_completion)
            task = self._task

        try:
            report = await asyncio.shield(task)
        except BaseException:
            if task.done():
                async with self._lock:
                    if self._task is task:
                        self._harvest_locked(task)
            raise

        async with self._lock:
            if self._task is task:
                harvested = self._harvest_locked(task)
                if harvested is not None:
                    report = harvested
        return report
