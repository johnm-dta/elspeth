from __future__ import annotations

import asyncio
import gc
import os
import stat
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from structlog.testing import capture_logs

import elspeth.web.readiness as readiness
from elspeth.web.readiness import (
    ReadinessCache,
    ReadinessCheck,
    ReadinessProbeRunner,
    ReadinessReport,
    _check_auth_mode,
    _check_blob_dir,
    _check_data_dir,
    _check_landscape_database,
    _check_payload_store,
    _check_session_database,
    overall_timeout_report,
    readiness_report,
)
from elspeth.web.schema_probe import SchemaState
from elspeth.web.sessions.engine import create_session_engine


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
        results = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
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

    @pytest.mark.asyncio
    async def test_close_promptly_cancels_running_wrapper_but_keeps_source_registered_and_drained(self) -> None:
        runner = ReadinessProbeRunner()
        entered = threading.Event()
        release = threading.Event()
        observed: list[dict[str, object]] = []
        loop = asyncio.get_running_loop()
        prior_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: observed.append(context))

        def blocked_failure() -> tuple[ReadinessCheck, ...]:
            entered.set()
            release.wait()
            raise RuntimeError("LATE_CLOSE_SENTINEL /private/close-path")

        task = asyncio.create_task(runner.run("session", ("session_db", "session_schema"), blocked_failure))
        try:
            assert await asyncio.to_thread(entered.wait, 1.0)
            runner.close()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), timeout=0.25)
            assert "session" in runner._futures

            release.set()
            for _ in range(50):
                if "session" not in runner._futures:
                    break
                await asyncio.sleep(0.01)
            gc.collect()
            await asyncio.sleep(0)
            assert "session" not in runner._futures
            assert observed == []
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            runner.close()
            loop.set_exception_handler(prior_handler)

    @pytest.mark.asyncio
    async def test_close_cancels_queued_source_and_wrapper_without_executing_probe(self) -> None:
        runner = ReadinessProbeRunner()
        release_workers = threading.Event()
        blockers = [runner._executor.submit(release_workers.wait) for _ in range(5)]
        executed = False

        def queued_probe() -> tuple[ReadinessCheck, ...]:
            nonlocal executed
            executed = True
            return _ok("data_dir")

        task = asyncio.create_task(runner.run("data_dir", ("data_dir",), queued_probe))
        try:
            for _ in range(50):
                if "data_dir" in runner._futures:
                    break
                await asyncio.sleep(0.01)
            assert "data_dir" in runner._futures
            runner.close()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), timeout=0.25)
            assert "data_dir" not in runner._futures
            assert executed is False
        finally:
            release_workers.set()
            await asyncio.gather(task, return_exceptions=True)
            for blocker in blockers:
                blocker.result(timeout=1.0)
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


class _FakeConnection:
    def __init__(self, *, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.statements: list[str] = []

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, statement: object) -> object:
        self.statements.append(str(statement))
        if self.failure is not None:
            raise self.failure
        return object()


class _FakeEngine:
    def __init__(
        self,
        *,
        dialect: str = "postgresql",
        connection: _FakeConnection | None = None,
        url: str = "postgresql+psycopg://redacted.invalid/db",
    ) -> None:
        self.dialect = SimpleNamespace(name=dialect)
        self.url = url
        self.connection = connection or _FakeConnection()
        self.connect_calls = 0
        self.disposed = False

    def connect(self) -> _FakeConnection:
        self.connect_calls += 1
        return self.connection

    def dispose(self) -> None:
        self.disposed = True


def _settings_stub(tmp_path: Path, **overrides: object) -> Any:
    data_dir = tmp_path / "data"
    payload_dir = tmp_path / "payloads"
    blob_dir = data_dir / "blobs"
    data_dir.mkdir(exist_ok=True)
    payload_dir.mkdir(mode=0o700, exist_ok=True)
    blob_dir.mkdir(exist_ok=True)
    values: dict[str, object] = {
        "deployment_target": "default",
        "auth_provider": "local",
        "session_db_url": None,
        "landscape_url": None,
        "data_dir": data_dir,
        "payload_store_path": payload_dir,
        "oidc_issuer": None,
        "oidc_audience": None,
        "oidc_client_id": None,
        "entra_tenant_id": None,
    }
    values.update(overrides)
    settings = SimpleNamespace(**values)
    settings.get_session_db_url = lambda: str(values.get("session_db_url") or f"sqlite:///{data_dir / 'sessions.db'}")
    settings.get_landscape_url = lambda: str(values.get("landscape_url") or f"sqlite:///{data_dir / 'audit.db'}")
    settings.get_payload_store_path = lambda: Path(values.get("payload_store_path") or payload_dir)
    return settings


class TestReadinessDatabaseChecks:
    def test_aws_session_uses_raw_url_isolated_factory_and_same_connection(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        raw_url = "postgresql+psycopg://runtime@db.internal/session"
        settings = _settings_stub(tmp_path, deployment_target="aws-ecs", session_db_url=raw_url)
        settings.get_session_db_url = lambda: pytest.fail("fallback getter must not run in AWS mode")
        live_engine = _FakeEngine()
        owned = _FakeEngine()
        captured: dict[str, object] = {}

        def factory(url: str, **kwargs: object) -> _FakeEngine:
            captured.update(url=url, kwargs=kwargs)
            return owned

        probe_connections: list[object] = []
        monkeypatch.setattr(readiness, "create_session_engine", factory)
        monkeypatch.setattr(
            readiness,
            "probe_session_schema",
            lambda conn: probe_connections.append(conn) or SchemaState.CURRENT,
        )

        checks = _check_session_database(settings, live_engine)

        assert captured == {
            "url": raw_url,
            "kwargs": {
                "pool_size": 1,
                "max_overflow": 0,
                "pool_pre_ping": True,
                "pool_timeout": 0.5,
                "connect_args": {"connect_timeout": 1},
            },
        }
        assert live_engine.connect_calls == 0
        assert probe_connections == [owned.connection]
        assert owned.connection.statements == ["SET LOCAL statement_timeout = '1000ms'", "SELECT 1"]
        assert owned.disposed is True
        assert checks == (
            ReadinessCheck("session_db", True, "connected"),
            ReadinessCheck("session_schema", True, "schema state: CURRENT"),
        )

    def test_aws_landscape_uses_raw_url_and_landscape_factory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        raw_url = "postgresql+psycopg://runtime@db.internal/landscape"
        settings = _settings_stub(tmp_path, deployment_target="aws-ecs", landscape_url=raw_url)
        settings.get_landscape_url = lambda: pytest.fail("fallback getter must not run in AWS mode")
        owned = _FakeEngine()
        captured: dict[str, object] = {}

        def factory(url: str, **kwargs: object) -> _FakeEngine:
            captured.update(url=url, kwargs=kwargs)
            return owned

        monkeypatch.setattr(readiness, "create_engine", factory)
        monkeypatch.setattr(readiness, "probe_landscape_schema", lambda conn: SchemaState.CURRENT)

        checks = _check_landscape_database(settings)

        assert captured["url"] == raw_url
        assert captured["kwargs"] == {
            "pool_size": 1,
            "max_overflow": 0,
            "pool_pre_ping": True,
            "pool_timeout": 0.5,
            "connect_args": {"connect_timeout": 1},
        }
        assert owned.disposed is True
        assert checks[1] == ReadinessCheck("landscape_schema", True, "schema state: CURRENT")

    def test_default_file_sqlite_reuses_live_session_engine(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _settings_stub(tmp_path)
        engine = create_session_engine(f"sqlite:///{tmp_path / 'session.db'}")
        monkeypatch.setattr(readiness, "create_session_engine", lambda *_a, **_kw: pytest.fail("must reuse live SQLite"))
        monkeypatch.setattr(readiness, "probe_session_schema", lambda conn: SchemaState.CURRENT)
        try:
            checks = _check_session_database(settings, engine)
        finally:
            engine.dispose()
        assert checks[0].ok is True
        assert checks[1].detail == "schema state: CURRENT"

    def test_memory_sqlite_is_rejected_without_checkout(self, tmp_path: Path) -> None:
        settings = _settings_stub(tmp_path, session_db_url="sqlite:///:memory:")
        engine = _FakeEngine(dialect="sqlite", url="sqlite:///:memory:")
        checks = _check_session_database(settings, engine)
        remedy = "in-memory SQLite is not readiness-probeable; use a file-backed session database"
        assert checks == (
            ReadinessCheck("session_db", False, remedy),
            ReadinessCheck("session_schema", False, remedy),
        )
        assert engine.connect_calls == 0

    @pytest.mark.parametrize(
        ("state", "ok"),
        [(SchemaState.CURRENT, True), (SchemaState.MISSING, False), (SchemaState.PARTIAL, False), (SchemaState.STALE, False)],
    )
    def test_every_schema_state_is_rendered(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: SchemaState, ok: bool) -> None:
        settings = _settings_stub(tmp_path)
        engine = create_session_engine(f"sqlite:///{tmp_path / 'session.db'}")
        monkeypatch.setattr(readiness, "probe_session_schema", lambda conn: state)
        try:
            checks = _check_session_database(settings, engine)
        finally:
            engine.dispose()
        assert checks[1] == ReadinessCheck("session_schema", ok, f"schema state: {state.name}")

    @pytest.mark.parametrize("failure", [RuntimeError("RAW_DB_SENTINEL"), KeyboardInterrupt("RAW_DB_SENTINEL")])
    def test_owned_engine_disposed_for_exception_and_base_exception(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failure: BaseException,
    ) -> None:
        settings = _settings_stub(
            tmp_path,
            deployment_target="aws-ecs",
            landscape_url="postgresql+psycopg://runtime@db.invalid/landscape",
        )
        owned = _FakeEngine(connection=_FakeConnection(failure=failure))
        monkeypatch.setattr(readiness, "create_engine", lambda *_a, **_kw: owned)
        with pytest.raises(type(failure)):
            _check_landscape_database(settings)
        assert owned.disposed is True

    @pytest.mark.asyncio
    async def test_database_failure_response_and_log_are_class_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = _settings_stub(
            tmp_path,
            deployment_target="aws-ecs",
            session_db_url="postgresql+psycopg://runtime@db.invalid/session",
            landscape_url="postgresql+psycopg://runtime@db.invalid/landscape",
        )

        def fail(*_args: object, **_kwargs: object) -> _FakeEngine:
            raise RuntimeError("RAW_URL_SENTINEL RAW_SQL_SENTINEL RAW_DRIVER_SENTINEL")

        monkeypatch.setattr(readiness, "create_session_engine", fail)
        monkeypatch.setattr(readiness, "create_engine", fail)
        runner = ReadinessProbeRunner()
        try:
            with capture_logs() as logs:
                report = await readiness_report(settings, _FakeEngine(), runner)
        finally:
            runner.close()
        rendered = repr((report, logs))
        assert report.ready is False
        assert "RuntimeError" in rendered
        assert "RAW_URL_SENTINEL" not in rendered
        assert "RAW_SQL_SENTINEL" not in rendered
        assert "RAW_DRIVER_SENTINEL" not in rendered


class TestReadinessFilesystemChecks:
    @pytest.mark.parametrize(
        "function,name", [(_check_data_dir, "data_dir"), (_check_payload_store, "payload_store"), (_check_blob_dir, "blob_dir")]
    )
    def test_existing_directories_pass_without_named_residue(
        self,
        tmp_path: Path,
        function: Any,
        name: str,
    ) -> None:
        settings = _settings_stub(tmp_path)
        assert function(settings) == (ReadinessCheck(name, True, "directory is writable"),)
        assert list(tmp_path.rglob(".readiness-probe-*")) == []

    def test_payload_rejects_symlink_and_unsafe_mode(self, tmp_path: Path) -> None:
        settings = _settings_stub(tmp_path)
        payload = Path(settings.payload_store_path)
        target = tmp_path / "target"
        target.mkdir()
        payload.rmdir()
        payload.symlink_to(target, target_is_directory=True)
        assert _check_payload_store(settings)[0].detail == "payload_store directory must not be a symlink"
        payload.unlink()
        payload.mkdir(mode=0o700)
        payload.chmod(payload.stat().st_mode | stat.S_IWGRP)
        assert _check_payload_store(settings)[0].detail == "payload_store group/world-writable directory is not allowed"

    def test_probe_uses_exclusive_create_immediate_unlink_and_same_fd(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = _settings_stub(tmp_path)
        calls: list[tuple[str, Path]] = []
        real_mkstemp = readiness.tempfile.mkstemp
        real_unlink = readiness.os.unlink

        def mkstemp(*, prefix: str, dir: Path) -> tuple[int, str]:
            calls.append((prefix, dir))
            return real_mkstemp(prefix=prefix, dir=dir)

        def unlink(path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> None:
            assert Path(path).exists()
            real_unlink(path)

        monkeypatch.setattr(readiness.tempfile, "mkstemp", mkstemp)
        monkeypatch.setattr(readiness.os, "unlink", unlink)
        assert _check_data_dir(settings)[0].ok is True
        assert calls == [(".readiness-probe-", Path(settings.data_dir).resolve(strict=True))]
        assert list(tmp_path.rglob(".readiness-probe-*")) == []

    def test_fsync_and_cleanup_failures_are_static_and_leave_no_named_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = _settings_stub(tmp_path)
        monkeypatch.setattr(readiness.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("RAW_PATH_SENTINEL")))
        check = _check_data_dir(settings)[0]
        assert check == ReadinessCheck("data_dir", False, "directory probe failed (OSError)")
        assert "RAW_PATH_SENTINEL" not in check.detail
        assert list(tmp_path.rglob(".readiness-probe-*")) == []

    def test_concurrent_probes_do_not_collide(self, tmp_path: Path) -> None:
        settings = _settings_stub(tmp_path)
        with readiness.concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: _check_data_dir(settings), range(20)))
        assert all(result[0].ok for result in results)
        assert list(tmp_path.rglob(".readiness-probe-*")) == []


class TestReadinessAuthAndReport:
    @pytest.mark.parametrize(
        ("provider", "fields", "ok"),
        [
            ("local", {}, True),
            ("oidc", {"oidc_issuer": "https://issuer.invalid", "oidc_audience": "aud", "oidc_client_id": "client"}, True),
            ("oidc", {"oidc_audience": "aud", "oidc_client_id": "client"}, False),
            ("oidc", {"oidc_issuer": "https://issuer.invalid", "oidc_client_id": "client"}, False),
            ("oidc", {"oidc_issuer": "https://issuer.invalid", "oidc_audience": "aud"}, False),
            ("entra", {"entra_tenant_id": "tenant", "oidc_audience": "aud", "oidc_client_id": "client"}, True),
            ("entra", {"oidc_audience": "aud", "oidc_client_id": "client"}, False),
            ("entra", {"entra_tenant_id": "tenant", "oidc_client_id": "client"}, False),
            ("entra", {"entra_tenant_id": "tenant", "oidc_audience": "aud"}, False),
            ("unknown", {}, False),
        ],
    )
    def test_auth_mode_is_total(self, tmp_path: Path, provider: str, fields: dict[str, object], ok: bool) -> None:
        settings = _settings_stub(tmp_path, auth_provider=provider, **fields)
        check = _check_auth_mode(settings)
        assert check.name == "auth_mode"
        assert check.ok is ok

    @pytest.mark.asyncio
    async def test_report_has_exact_order_and_unique_names(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _settings_stub(tmp_path)
        session_engine = create_session_engine(f"sqlite:///{tmp_path / 'session.db'}")
        monkeypatch.setattr(readiness, "probe_session_schema", lambda conn: SchemaState.CURRENT)
        monkeypatch.setattr(readiness, "probe_landscape_schema", lambda conn: SchemaState.CURRENT)
        runner = ReadinessProbeRunner()
        try:
            report = await readiness_report(settings, session_engine, runner)
        finally:
            runner.close()
            session_engine.dispose()
        assert [check.name for check in report.checks] == list(readiness.READINESS_CHECK_NAMES)
        assert len({check.name for check in report.checks}) == 8
        assert report.ready is True

    @pytest.mark.asyncio
    async def test_unexpected_gather_error_becomes_static_eight_check_report(self, tmp_path: Path) -> None:
        settings = _settings_stub(tmp_path)

        class BrokenRunner:
            async def run(self, *_args: object, **_kwargs: object) -> tuple[ReadinessCheck, ...]:
                raise RuntimeError("RAW_URL_SENTINEL /private/path")

        with capture_logs() as logs:
            report = await readiness_report(settings, _FakeEngine(), BrokenRunner())
        assert [check.name for check in report.checks] == list(readiness.READINESS_CHECK_NAMES)
        assert all(not check.ok and check.detail == "readiness evaluation failed (RuntimeError)" for check in report.checks)
        assert "RAW_URL_SENTINEL" not in repr((report, logs))
        assert "/private/path" not in repr((report, logs))

    def test_overall_timeout_report_is_exact_and_class_free(self) -> None:
        with capture_logs() as logs:
            report = overall_timeout_report()
        assert report.ready is False
        assert [check.name for check in report.checks] == list(readiness.READINESS_CHECK_NAMES)
        assert all(check.detail == "readiness request timed out" for check in report.checks)
        assert len(logs) == 8
        assert all(set(log) >= {"check", "detail"} for log in logs)
