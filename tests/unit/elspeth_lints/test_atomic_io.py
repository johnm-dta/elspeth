"""Tests for ``elspeth_lints.core.atomic_io`` (C4-1 closure).

Covers the Tier-1 atomic-write contract:

* happy-path: content lands, no temp/lock orphans;
* parent-dir fsync called before AND after rename;
* file fsync called before the rename;
* cross-process lock contention raises ``AtomicWriteConflictError``;
* concurrent writers serialised at filesystem level;
* mid-write crash (SIGKILL after temp created, before replace) leaves
  the original file byte-identical and the temp file orphaned (we
  assert the orphan is the *only* residue — original integrity is the
  load-bearing invariant);
* O_EXCL refuses to clobber a pre-planted temp inode;
* write through a non-existent parent directory raises ``FileNotFoundError``.

The fsync tests use a wrapping spy (not ``MagicMock``) so the
underlying durability behaviour is preserved while we count calls.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from elspeth_lints.core.atomic_io import (
    AtomicWriteConflictError,
    _lock_path_for,
    _temp_path_for,
    atomic_write_text,
)


def test_happy_path_writes_content_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "allowlist.yaml"
    atomic_write_text(target, "allow_hits: []\n")
    assert target.read_text() == "allow_hits: []\n"
    # Only the target and the persistent lockfile may remain.
    residue = sorted(p.name for p in tmp_path.iterdir())
    assert residue == [".allowlist.yaml.lock", "allowlist.yaml"]


def test_overwrites_existing_file_atomically(tmp_path: Path) -> None:
    target = tmp_path / "allowlist.yaml"
    target.write_text("OLD\n")
    atomic_write_text(target, "NEW\n")
    assert target.read_text() == "NEW\n"


def test_parent_dir_fsync_called_before_and_after_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parent-dir fsync is load-bearing for POSIX rename durability.

    Spy-wraps ``os.fsync`` and ``os.replace`` to observe order:
    file fsync → dir fsync → replace → dir fsync. Without the
    pre-rename dir fsync, a crash between rename and post-rename
    sync can lose the temp inode's directory entry.
    """
    target = tmp_path / "allowlist.yaml"
    events: list[tuple[str, Any]] = []

    real_fsync = os.fsync
    real_replace = os.replace

    def fsync_spy(fd: int) -> None:
        # Distinguish dir fd from file fd by stat'ing the fd; dir
        # fds report ``S_ISDIR``.
        import stat

        st = os.fstat(fd)
        kind = "dir" if stat.S_ISDIR(st.st_mode) else "file"
        events.append(("fsync", kind))
        real_fsync(fd)

    def replace_spy(src: str, dst: str) -> None:
        events.append(("replace", (src, dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync_spy)
    monkeypatch.setattr(os, "replace", replace_spy)

    atomic_write_text(target, "payload\n")

    # Expected sequence: file fsync, dir fsync, replace, dir fsync.
    kinds = [(name, payload) for name, payload in events]
    assert kinds[0] == ("fsync", "file"), kinds
    assert kinds[1] == ("fsync", "dir"), kinds
    assert kinds[2][0] == "replace", kinds
    assert kinds[3] == ("fsync", "dir"), kinds


def test_o_excl_refuses_to_clobber_preplanted_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hostile/racing process planting our exact temp path is rejected.

    We force ``_temp_path_for`` to a deterministic path, plant a
    file there, and assert the second write raises ``FileExistsError``
    (the ``O_EXCL`` defence).
    """
    target = tmp_path / "allowlist.yaml"
    planted = tmp_path / ".allowlist.yaml.tmp-FAKE"
    planted.write_text("ATTACKER\n")

    monkeypatch.setattr(
        "elspeth_lints.core.atomic_io._temp_path_for",
        lambda path: planted,
    )

    with pytest.raises(FileExistsError):
        atomic_write_text(target, "good\n")
    # Planted file is left for forensics; target was never created.
    assert planted.read_text() == "ATTACKER\n"
    assert not target.exists()


def test_concurrent_writer_raises_conflict_error(tmp_path: Path) -> None:
    """Two concurrent ``atomic_write_text`` calls: second raises.

    Cross-process via ``subprocess.Popen``: a slow child holds the
    flock; the parent's call sees ``BlockingIOError`` from
    ``LOCK_NB`` and translates to ``AtomicWriteConflictError``.
    """
    target = tmp_path / "allowlist.yaml"
    ready_marker = tmp_path / ".child_ready"

    # The child takes the lock, signals ready via a marker file,
    # then sleeps long enough for the parent's flock attempt.
    child_src = textwrap.dedent(
        f"""
        import fcntl, os, time, sys
        from pathlib import Path
        lock_path = Path({str(_lock_path_for(target))!r})
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        Path({str(ready_marker)!r}).write_text("ready")
        time.sleep(5)
        """
    )
    child = subprocess.Popen([sys.executable, "-c", child_src])
    try:
        # Wait for child to take the lock.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not ready_marker.exists():
            time.sleep(0.05)
        assert ready_marker.exists(), "child failed to take lock in time"

        with pytest.raises(AtomicWriteConflictError):
            atomic_write_text(target, "payload\n")
    finally:
        child.terminate()
        child.wait(timeout=2)


def test_lock_false_skips_locking(tmp_path: Path) -> None:
    """``lock=False`` exists for tests that exercise durability paths."""
    target = tmp_path / "allowlist.yaml"
    atomic_write_text(target, "payload\n", lock=False)
    assert target.read_text() == "payload\n"
    # No lockfile created on this path.
    assert not _lock_path_for(target).exists()


def test_missing_parent_directory_raises(tmp_path: Path) -> None:
    target = tmp_path / "does_not_exist" / "allowlist.yaml"
    with pytest.raises(FileNotFoundError):
        atomic_write_text(target, "payload\n")


def test_mid_write_kill_leaves_original_intact(tmp_path: Path) -> None:
    """SIGKILL between temp-create and ``os.replace``: original unchanged.

    Child opens its own ``atomic_write_text`` machinery but stalls
    inside ``os.replace`` via a monkeypatched sleep. Parent SIGKILLs
    after the temp file exists. Assert: original file's byte
    content is preserved; ``.tmp-*`` orphan is the only residue
    (we accept orphans on crash — the load-bearing invariant is
    original-file integrity, not zero residue).
    """
    target = tmp_path / "allowlist.yaml"
    target.write_text("ORIGINAL\n")

    # Child writes payload to its temp, then blocks before replace.
    child_src = textwrap.dedent(
        f"""
        import os, time, sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[3] / "elspeth-lints" / "src")!r})
        sys.path.insert(0, {str(Path(__file__).resolve().parents[3] / "src")!r})
        from elspeth_lints.core import atomic_io
        # Make os.replace block forever so SIGKILL lands between
        # temp-create and rename.
        atomic_io.os.replace = lambda src, dst: time.sleep(60)
        atomic_io.atomic_write_text({str(target)!r}, "NEW\\n")
        """
    )
    child = subprocess.Popen([sys.executable, "-c", child_src])
    try:
        # Poll for the temp file to appear, then SIGKILL.
        deadline = time.monotonic() + 5.0
        temp_seen = False
        while time.monotonic() < deadline:
            temps = list(tmp_path.glob(".allowlist.yaml.tmp-*"))
            if temps:
                temp_seen = True
                break
            time.sleep(0.05)
        assert temp_seen, "child failed to create temp file in time"
        child.kill()
        child.wait(timeout=2)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)

    # The Tier-1 invariant: original file is byte-identical to
    # pre-attempt state. ``ORIGINAL\n`` survives the crash.
    assert target.read_text() == "ORIGINAL\n"
    # Temp orphan(s) are acceptable post-crash (the finally block
    # only runs on graceful exit). Forensics: at most one per
    # killed process.
    orphans = list(tmp_path.glob(".allowlist.yaml.tmp-*"))
    assert len(orphans) <= 1


def test_lockfile_persists_across_calls(tmp_path: Path) -> None:
    """The ``.lock`` sibling is reused; not deleted between calls."""
    target = tmp_path / "allowlist.yaml"
    atomic_write_text(target, "one\n")
    lock_after_first = _lock_path_for(target)
    assert lock_after_first.exists()
    first_inode = lock_after_first.stat().st_ino
    atomic_write_text(target, "two\n")
    assert _lock_path_for(target).stat().st_ino == first_inode


def test_temp_path_uniqueness(tmp_path: Path) -> None:
    """``_temp_path_for`` produces distinct paths across rapid calls."""
    target = tmp_path / "allowlist.yaml"
    paths = {_temp_path_for(target) for _ in range(100)}
    # monotonic_ns advances; 100 calls produce 100 distinct names.
    assert len(paths) == 100


def test_oserror_after_temp_create_cleans_up_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If fsync fails after temp create, the orphan is scrubbed."""
    target = tmp_path / "allowlist.yaml"
    target.write_text("ORIGINAL\n")

    call_count = {"n": 0}
    real_fsync = os.fsync

    def fsync_fails_on_second_call(fd: int) -> None:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise OSError("simulated fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync_fails_on_second_call)

    with pytest.raises(OSError, match="simulated fsync failure"):
        atomic_write_text(target, "NEW\n")

    # Original preserved.
    assert target.read_text() == "ORIGINAL\n"
    # Temp scrubbed in the except handler.
    orphans = list(tmp_path.glob(".allowlist.yaml.tmp-*"))
    assert orphans == []
