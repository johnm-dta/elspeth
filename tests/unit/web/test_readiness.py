from __future__ import annotations

import asyncio
import gc
import threading
import time

import pytest

from elspeth.web.readiness import (
    ReadinessCache,
    ReadinessCheck,
    ReadinessProbeRunner,
    ReadinessReport,
)


def _ok(name: str) -> tuple[ReadinessCheck, ...]:
    return (ReadinessCheck(name=name, ok=True, detail="ok"),)


class TestReadinessProbeRunner:
    @pytest.mark.asyncio
    async def test_one_unresolved_submission_per_label_and_other_label_runs(self) -> None:
        runner = ReadinessProbeRunner()
        release = threading.Event()
        entered = threading.Event()
        calls = 0

        def blocked() -> tuple[ReadinessCheck, ...]:
            nonlocal calls
            calls += 1
            entered.set()
            release.wait()
            return _ok("data_dir")

        try:
            leading = asyncio.create_task(runner.run("data_dir", ("data_dir",), blocked))
            assert await asyncio.to_thread(entered.wait, 1.0)

            duplicates = await asyncio.gather(*(runner.run("data_dir", ("data_dir",), blocked) for _ in range(20)))
            assert calls == 1
            assert all(result == (ReadinessCheck("data_dir", False, "probe already in flight"),) for result in duplicates)

            other = await runner.run("blob_dir", ("blob_dir",), _ok, "blob_dir")
            assert other == _ok("blob_dir")
            assert len(runner._futures) == 1
        finally:
            release.set()
            if "leading" in locals():
                await leading
            runner.close()

    @pytest.mark.asyncio
    async def test_queued_cancellation_is_removed_by_source_callback(self) -> None:
        runner = ReadinessProbeRunner()
        releases = [threading.Event() for _ in range(5)]
        entered = [threading.Event() for _ in range(5)]

        def blocked(index: int, name: str) -> tuple[ReadinessCheck, ...]:
            entered[index].set()
            releases[index].wait()
            return _ok(name)

        labels = ("session", "landscape", "data_dir", "payload_store", "blob_dir")
        tasks: list[asyncio.Task[tuple[ReadinessCheck, ...]]] = []
        try:
            for index, label in enumerate(labels):
                tasks.append(asyncio.create_task(runner.run(label, (f"{label}_db",), blocked, index, f"{label}_db")))
            for event in entered:
                assert await asyncio.to_thread(event.wait, 1.0)
            assert len(runner._futures) == 5

            tasks[-1].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[-1]
            releases[-1].set()
            for _ in range(20):
                if "blob_dir" not in runner._futures:
                    break
                await asyncio.sleep(0)
            assert "blob_dir" not in runner._futures
        finally:
            for event in releases:
                event.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            runner.close()

    @pytest.mark.asyncio
    async def test_immediate_completion_and_concurrent_close_do_not_deadlock(self) -> None:
        for _ in range(50):
            runner = ReadinessProbeRunner()
            result = await asyncio.wait_for(
                runner.run("data_dir", ("data_dir",), _ok, "data_dir"),
                timeout=0.5,
            )
            assert result == _ok("data_dir")
            await asyncio.wait_for(asyncio.to_thread(runner.close), timeout=0.5)

        runner = ReadinessProbeRunner()
        release = threading.Event()
        entered = threading.Event()

        def blocked() -> tuple[ReadinessCheck, ...]:
            entered.set()
            release.wait()
            return _ok("data_dir")

        task = asyncio.create_task(runner.run("data_dir", ("data_dir",), blocked))
        try:
            assert await asyncio.to_thread(entered.wait, 1.0)
            await asyncio.wait_for(asyncio.to_thread(runner.close), timeout=0.5)
            closed = await runner.run("blob_dir", ("blob_dir",), _ok, "blob_dir")
            assert closed == (ReadinessCheck("blob_dir", False, "probe runner closed"),)
        finally:
            release.set()
            await task
            runner.close()

    @pytest.mark.asyncio
    async def test_real_loop_deadline_expires_at_two_seconds_while_heartbeat_runs(self) -> None:
        runner = ReadinessProbeRunner()
        release = threading.Event()
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.02)

        def blocked() -> tuple[ReadinessCheck, ...]:
            release.wait()
            return _ok("blob_dir")

        pulse = asyncio.create_task(heartbeat())
        try:
            started = time.monotonic()
            result = await runner.run("blob_dir", ("blob_dir",), blocked)
            elapsed = time.monotonic() - started
            assert result == (ReadinessCheck("blob_dir", False, "probe timed out"),)
            assert 1.85 <= elapsed <= 2.35
            assert ticks >= 50
        finally:
            release.set()
            pulse.cancel()
            await asyncio.gather(pulse, return_exceptions=True)
            runner.close()

    @pytest.mark.asyncio
    async def test_event_loop_stall_consumes_real_deadline_not_poll_iterations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = ReadinessProbeRunner()
        release = threading.Event()
        original_wait = asyncio.wait
        calls = 0

        async def stalling_wait(*args: object, **kwargs: object):
            nonlocal calls
            calls += 1
            if calls == 1:
                time.sleep(1.2)
            return await original_wait(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(asyncio, "wait", stalling_wait)

        def blocked() -> tuple[ReadinessCheck, ...]:
            release.wait()
            return _ok("data_dir")

        try:
            started = time.monotonic()
            result = await runner.run("data_dir", ("data_dir",), blocked)
            elapsed = time.monotonic() - started
            assert result[0].detail == "probe timed out"
            assert elapsed < 2.4
            assert calls < 15
        finally:
            release.set()
            runner.close()

    @pytest.mark.asyncio
    async def test_cancelled_caller_keeps_running_source_registered_then_readmits(self) -> None:
        runner = ReadinessProbeRunner()
        release = threading.Event()
        entered = threading.Event()

        def blocked() -> tuple[ReadinessCheck, ...]:
            entered.set()
            release.wait()
            return _ok("payload_store")

        task = asyncio.create_task(runner.run("payload_store", ("payload_store",), blocked))
        try:
            assert await asyncio.to_thread(entered.wait, 1.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert "payload_store" in runner._futures
            duplicate = await runner.run("payload_store", ("payload_store",), blocked)
            assert duplicate[0].detail == "probe already in flight"

            release.set()
            for _ in range(50):
                if "payload_store" not in runner._futures:
                    break
                await asyncio.sleep(0.01)
            assert "payload_store" not in runner._futures
            assert await runner.run("payload_store", ("payload_store",), _ok, "payload_store") == _ok("payload_store")
        finally:
            release.set()
            runner.close()

    @pytest.mark.asyncio
    async def test_abandoned_late_exception_is_drained_without_loop_leak(self) -> None:
        runner = ReadinessProbeRunner()
        release = threading.Event()
        entered = threading.Event()
        observed: list[dict[str, object]] = []
        loop = asyncio.get_running_loop()
        prior_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: observed.append(context))

        def failing() -> tuple[ReadinessCheck, ...]:
            entered.set()
            release.wait()
            raise RuntimeError("credential=late-secret /private/path")

        task = asyncio.create_task(runner.run("landscape", ("landscape_db", "landscape_schema"), failing))
        try:
            assert await asyncio.to_thread(entered.wait, 1.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            for _ in range(20):
                gc.collect()
                await asyncio.sleep(0.01)
            assert observed == []
        finally:
            release.set()
            runner.close()
            loop.set_exception_handler(prior_handler)

    @pytest.mark.asyncio
    async def test_exception_rendering_is_class_only_and_paired_for_database(self) -> None:
        runner = ReadinessProbeRunner()

        def fail() -> tuple[ReadinessCheck, ...]:
            raise ValueError("RAW_URL_SENTINEL /private/path RAW_SQL_SENTINEL")

        try:
            result = await runner.run("session", ("session_db", "session_schema"), fail)
            assert result == (
                ReadinessCheck("session_db", False, "probe failed (ValueError)"),
                ReadinessCheck("session_schema", False, "not checked: connectivity probe failed"),
            )
            assert "secret" not in repr(result)
        finally:
            runner.close()


class TestReadinessCache:
    @staticmethod
    def _report(value: str = "ok") -> ReadinessReport:
        return ReadinessReport(True, (ReadinessCheck("auth_mode", True, value),))

    @pytest.mark.asyncio
    async def test_fresh_hit_and_recompute_after_two_second_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = ReadinessCache()
        now = 100.0
        monkeypatch.setattr(cache, "_now", lambda: now)
        calls = 0

        async def compute() -> ReadinessReport:
            nonlocal calls
            calls += 1
            return self._report(str(calls))

        assert (await cache.get(compute)).checks[0].detail == "1"
        now = 101.999
        assert (await cache.get(compute)).checks[0].detail == "1"
        now = 102.001
        assert (await cache.get(compute)).checks[0].detail == "2"
        assert calls == 2

    @pytest.mark.asyncio
    async def test_fifty_callers_collapse_to_one_shared_task(self) -> None:
        cache = ReadinessCache()
        release = asyncio.Event()
        calls = 0

        async def compute() -> ReadinessReport:
            nonlocal calls
            calls += 1
            await release.wait()
            return self._report()

        tasks = [asyncio.create_task(cache.get(compute)) for _ in range(50)]
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(*tasks)
        assert calls == 1
        assert all(result == self._report() for result in results)

    @pytest.mark.asyncio
    async def test_cancelled_leader_does_not_cancel_shared_compute(self) -> None:
        cache = ReadinessCache()
        release = asyncio.Event()
        calls = 0

        async def compute() -> ReadinessReport:
            nonlocal calls
            calls += 1
            await release.wait()
            return self._report()

        leader = asyncio.create_task(cache.get(compute))
        await asyncio.sleep(0)
        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader

        follower = asyncio.create_task(cache.get(compute))
        await asyncio.sleep(0)
        assert calls == 1
        release.set()
        assert await follower == self._report()

    @pytest.mark.asyncio
    async def test_completed_task_is_harvested_without_original_waiter(self) -> None:
        cache = ReadinessCache()
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def compute() -> ReadinessReport:
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return self._report()

        leader = asyncio.create_task(cache.get(compute))
        await started.wait()
        leader.cancel()
        await asyncio.gather(leader, return_exceptions=True)
        release.set()
        await asyncio.sleep(0.01)

        assert await cache.get(compute) == self._report()
        assert calls == 1

    @pytest.mark.asyncio
    async def test_compute_exception_recovers_without_poisoning_prior_report(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = ReadinessCache()
        now = 1.0
        monkeypatch.setattr(cache, "_now", lambda: now)

        async def good() -> ReadinessReport:
            return self._report("prior")

        assert await cache.get(good) == self._report("prior")
        now = 4.0

        async def fail() -> ReadinessReport:
            raise RuntimeError("credential=secret")

        with pytest.raises(RuntimeError):
            await cache.get(fail)

        async def recovered() -> ReadinessReport:
            return self._report("recovered")

        assert await cache.get(recovered) == self._report("recovered")

    @pytest.mark.asyncio
    async def test_follower_does_not_wait_for_two_sequential_computations(self) -> None:
        cache = ReadinessCache()
        release = asyncio.Event()
        calls = 0

        async def fail() -> ReadinessReport:
            nonlocal calls
            calls += 1
            await release.wait()
            raise RuntimeError("failed")

        leader = asyncio.create_task(cache.get(fail))
        follower = asyncio.create_task(cache.get(fail))
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(leader, follower, return_exceptions=True)
        assert calls == 1
        assert all(isinstance(result, RuntimeError) for result in results)
