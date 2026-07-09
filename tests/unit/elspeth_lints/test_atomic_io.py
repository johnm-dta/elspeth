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

import concurrent.futures
import os
import stat
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from elspeth_lints.core.atomic_io import (
    AtomicWriteConflictError,
    AtomicWriteShortWriteError,
    _lock_path_for,
    _temp_path_for,
    atomic_update_text,
    atomic_write_text,
)


def test_happy_path_writes_content_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "allowlist.yaml"
    atomic_write_text(target, "allow_hits: []\n")
    assert target.read_text() == "allow_hits: []\n"
    # Only the target and the persistent lockfile may remain.
    residue = sorted(p.name for p in tmp_path.iterdir())
    assert residue == [".allowlist.yaml.lock", "allowlist.yaml"]


def test_atomic_write_text_uses_owner_only_target_permissions(tmp_path: Path) -> None:
    target = tmp_path / "allowlist.yaml"
    atomic_write_text(target, "allow_hits: []\n")

    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode & 0o077 == 0


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


def test_atomic_update_text_serializes_read_modify_write(tmp_path: Path) -> None:
    """Compound updates see the latest on-disk text while holding the lock."""
    target = tmp_path / "allowlist.yaml"
    target.write_text("", encoding="utf-8")
    labels = ("first", "second")
    start_barrier = threading.Barrier(len(labels))
    observed_inputs: list[str | None] = []
    observed_lock = threading.Lock()

    def append(label: str) -> None:
        start_barrier.wait(timeout=5)

        def update(current: str | None) -> str:
            with observed_lock:
                observed_inputs.append(current)
            time.sleep(0.05)
            return (current or "") + f"{label}\n"

        atomic_update_text(target, update)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(labels)) as executor:
        futures = [executor.submit(append, label) for label in labels]
        for future in futures:
            future.result(timeout=10)

    written = target.read_text(encoding="utf-8")
    assert set(written.splitlines()) == set(labels)
    assert observed_inputs.count("") == 1
    assert any(value in {"first\n", "second\n"} for value in observed_inputs)


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


def test_short_write_raises_typed_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """POSIX-legal short write surfaces as ``AtomicWriteShortWriteError``.

    Pre-T9-cleanup ``os.write`` was called without checking the
    returned byte count, so a partial write (POSIX-legal under signal
    interruption or certain FS conditions) would silently fsync+rename
    a truncated YAML over the live allowlist. The Tier-1 audit-trail
    consequence is the same as the disk-full case ``OSError`` already
    catches: corruption.

    This test monkeypatches ``os.write`` to return a short count and
    verifies the typed exception fires. The temp file is cleaned up
    by the existing ``except BaseException`` branch, so no
    ``.tmp-*`` orphan survives.
    """
    target = tmp_path / "allowlist.yaml"
    target.write_text("ORIGINAL\n")

    real_write = os.write
    call_count = {"n": 0}

    def short_write_on_first_call(fd: int, payload: bytes) -> int:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Pretend we only managed to flush half of the payload.
            # Real ``os.write`` actually wrote some bytes to the fd
            # (we use ``real_write`` for the partial flush) so the
            # tmp file on disk is genuinely truncated — mirrors the
            # POSIX failure shape we're guarding against.
            half = max(1, len(payload) // 2)
            real_write(fd, payload[:half])
            return half
        return real_write(fd, payload)

    monkeypatch.setattr(os, "write", short_write_on_first_call)

    with pytest.raises(AtomicWriteShortWriteError, match=r"short write to"):
        atomic_write_text(target, "NEW_CONTENT_THAT_IS_LONG_ENOUGH_TO_HALF\n")

    # Original preserved — the rename never happened.
    assert target.read_text() == "ORIGINAL\n"
    # Temp scrubbed.
    orphans = list(tmp_path.glob(".allowlist.yaml.tmp-*"))
    assert orphans == []


def test_lockfile_path_is_gitignored() -> None:
    """T9 MINOR: ``.<name>.lock`` siblings inside ``config/cicd/**`` are gitignored.

    ``atomic_write_text`` creates persistent ``.<name>.lock`` files
    next to managed allowlist YAMLs (flock idiom — unlinking inside
    the lock races on re-acquire). After the first ``justify`` /
    ``rotate`` invocation operators see them as untracked in
    ``git status``; without an ignore rule they may commit them by
    accident. Pin the rule so a future ``.gitignore`` refactor
    cannot regress.
    """
    repo_root = Path(__file__).resolve().parents[3]
    target = repo_root / "config" / "cicd" / "enforce_tier_model" / ".web.yaml.lock"
    result = subprocess.run(
        ["git", "check-ignore", "-v", str(target)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    # ``git check-ignore`` exits 0 when the path IS ignored, 1 when
    # not. Tier-1: assert the exact exit code rather than a softer
    # "stdout contains the rule" check.
    assert result.returncode == 0, (
        f"expected {target} to be gitignored by .gitignore but "
        f"git check-ignore returned {result.returncode}. stdout={result.stdout!r} "
        f"stderr={result.stderr!r}. The atomic_write_text lock file is "
        f"operator-local state and must never be committed."
    )
    # The matching rule should be the config/cicd/**/.*.lock pattern
    # so a regression that drops the pattern fires here.
    assert "config/cicd/**/.*.lock" in result.stdout, (
        f"unexpected ignore rule matched {target}: {result.stdout!r}. The intended rule is ``config/cicd/**/.*.lock``."
    )


def test_short_write_typed_exception_subclasses_runtime_error() -> None:
    """``AtomicWriteShortWriteError`` mirrors the existing exception taxonomy.

    Callers that broadly catch ``RuntimeError`` (the rotate/justify
    error-handling convention in ``apply_plan``) MUST still catch a
    short-write failure — it's an atomic-write failure, same class as
    ``AtomicWriteConflictError``. Pinning the inheritance prevents a
    future refactor from accidentally narrowing the taxonomy.
    """
    assert issubclass(AtomicWriteShortWriteError, RuntimeError)
    assert issubclass(AtomicWriteConflictError, RuntimeError)
