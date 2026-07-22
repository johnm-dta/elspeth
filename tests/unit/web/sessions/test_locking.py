"""Cross-process exclusion tests for shared SQLite session locking."""

from __future__ import annotations

import multiprocessing
import queue
import threading
from pathlib import Path

from elspeth.web.sessions.engine import create_session_engine


def _hold_sqlite_session_lock(database_url: str, entered: object, release: object) -> None:
    from elspeth.web.sessions.locking import sqlite_process_session_lock

    engine = create_session_engine(database_url)
    try:
        with sqlite_process_session_lock(engine, "shared-session"):
            entered.put("entered")  # type: ignore[attr-defined]
            if not release.wait(timeout=15):  # type: ignore[attr-defined]
                raise RuntimeError("release barrier timed out")
    finally:
        engine.dispose()


def test_sqlite_session_lock_excludes_separate_processes(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'locking.sqlite3'}"
    context = multiprocessing.get_context("spawn")
    entered = context.Queue()
    first_release = context.Event()
    second_release = context.Event()
    first = context.Process(target=_hold_sqlite_session_lock, args=(database_url, entered, first_release))
    second = context.Process(target=_hold_sqlite_session_lock, args=(database_url, entered, second_release))

    first.start()
    assert entered.get(timeout=10) == "entered"
    second.start()
    try:
        try:
            entered.get(timeout=0.5)
        except queue.Empty:
            pass
        else:
            raise AssertionError("second process entered the same-session critical section")

        first_release.set()
        first.join(timeout=10)
        assert first.exitcode == 0
        assert entered.get(timeout=10) == "entered"
    finally:
        first_release.set()
        second_release.set()
        first.join(timeout=10)
        second.join(timeout=10)
    assert second.exitcode == 0


def test_file_backed_sqlite_session_lock_is_same_thread_reentrant(tmp_path: Path) -> None:
    from elspeth.web.sessions.locking import sqlite_process_session_lock

    engine = create_session_engine(f"sqlite:///{tmp_path / 'reentrant.sqlite3'}")
    completed = threading.Event()

    def _nest_lock() -> None:
        with sqlite_process_session_lock(engine, "shared-session"):  # noqa: SIM117 - nesting is the behavior under test
            with sqlite_process_session_lock(engine, "shared-session"):
                completed.set()

    thread = threading.Thread(target=_nest_lock, daemon=True)
    thread.start()
    thread.join(timeout=5)
    try:
        assert completed.is_set(), "nested same-thread flock self-blocked"
        assert not thread.is_alive()
    finally:
        engine.dispose()
