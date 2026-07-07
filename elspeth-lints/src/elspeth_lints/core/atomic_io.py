"""Atomic, durable file replacement for Tier-1 allowlist YAML writes.

Closes C4-1 (elspeth-5c9ac11ec6) from the 2026-05-24 cicd-judge-cli
review: the rotate and justify code paths previously called
``path.write_text(...)`` directly on per-module allowlist YAMLs, which
truncates and rewrites the file in place with no rollback story. A
SIGTERM, disk-full, or concurrent writer mid-write left the
file half-written (loader crashes on YAML parse error) or pre-flush
(kernel buffer dropped on SIGKILL, the rotated state lost). Both
outcomes are Tier-1 audit-trail corruption.

This module implements the POSIX-canonical atomic-write recipe:

* acquire an exclusive ``fcntl.flock`` on a sibling lock file (so two
  concurrent writers serialise at filesystem level rather than racing
  through ``os.replace``);
* create a private ``.<name>.tmp-<pid>-<mono>`` next to the target
  with ``O_CREAT | O_WRONLY | O_EXCL`` (refuse to clobber an in-flight
  temp from another process);
* write content, ``flush()``, ``os.fsync(fd)`` — flushes user-space
  and kernel buffers to disk;
* ``os.fsync`` the *parent directory* — the directory entry for the
  temp inode must be durable before the rename, otherwise a system
  crash between ``os.replace`` and the directory sync can lose the
  rename;
* ``os.replace(temp, path)`` — POSIX-atomic rename;
* ``os.fsync`` the parent directory again — the rename itself must be
  durable.

On any error after the temp file is created we ``os.unlink`` it in a
``finally`` block so we never leave ``.tmp-*`` orphans. The lock file
itself persists across calls (standard ``flock`` idiom; unlinking it
inside the lock races on re-acquire) — operators will see a
``.<name>.lock`` next to each managed YAML and should leave it alone.

Tier-1 discipline: any failure here (lock contention, disk full,
permission error, fsync error) raises a typed exception. There is no
silent fallback path. The audit-trail is the legal record; we crash
rather than emit a half-written allowlist.
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Callable
from pathlib import Path


class AtomicWriteConflictError(RuntimeError):
    """Another writer holds the per-file lock for this allowlist YAML.

    Raised when ``atomic_write_text`` cannot acquire the sibling
    ``.<name>.lock`` non-blocking. The operator's recourse is to
    identify the other writer (another ``rotate`` / ``justify`` /
    ``apply`` invocation, a stuck process, a stale CI worker) and
    retry once the lock is released. We refuse rather than block
    indefinitely because Tier-1 writes happen inside short-lived CLI
    invocations; an indefinite hang at the audit boundary is worse
    than a fast, audit-honest failure.
    """


class AtomicWriteShortWriteError(RuntimeError):
    """``os.write`` returned fewer bytes than requested.

    POSIX permits ``write(2)`` to write fewer bytes than the requested
    count for a regular file. The kernel/FS combinations on which this
    surfaces are narrow (signal interruption, certain network FSes,
    out-of-quota near the boundary) and we do not see it in practice
    on the ext4/xfs/btrfs hosts ELSPETH runs on. Tier-1 doctrine,
    however, says "crash on any anomaly": silently shipping a
    truncated YAML to ``os.replace`` would corrupt the audit trail
    just as effectively as the disk-full case ``OSError`` already
    catches, and we cannot detect it post-facto because ``os.fsync``
    happily syncs the truncated state.

    Raised with the byte counts so the operator can correlate the
    underlying cause (signal? FS quota? short-write on the kernel
    side?). Uses a typed exception rather than ``assert`` because
    ``python -O`` strips asserts and the project's Tier-1 guarantees
    must survive optimisation.
    """


def _temp_path_for(path: Path) -> Path:
    """Compute a per-invocation private temp path next to ``path``.

    ``pid`` + monotonic-ns guarantees uniqueness across concurrent
    invocations on the same machine (each gets its own temp before
    the ``flock`` arbitration even decides who proceeds), so the
    ``O_EXCL`` open never collides with our own siblings.
    """
    suffix = f".{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    return path.parent / suffix


def _lock_path_for(path: Path) -> Path:
    return path.parent / f".{path.name}.lock"


def _fsync_directory(directory: Path) -> None:
    """``fsync`` ``directory`` so any rename/create just performed is durable.

    POSIX guarantees ``os.replace`` is *atomic* (no observer sees a
    partial state) but *not* durable until the directory entry is
    fsync'd. Without this, a system crash between the rename and the
    next sync can revive the old file or strand the temp inode.
    Operators who run on filesystems where directory fsync is a no-op
    (e.g. some networked FSes) inherit the underlying FS guarantees;
    we still call it because on ext4/xfs/btrfs it is load-bearing.
    """
    dir_fd = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _acquire_lock_fd(path: Path, *, wait: bool) -> int:
    """Acquire the sibling lock file for ``path`` and return its fd."""
    lock_path = _lock_path_for(path)
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    flags = fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB
    try:
        fcntl.flock(lock_fd, flags)
    except BlockingIOError as exc:
        os.close(lock_fd)
        raise AtomicWriteConflictError(
            f"atomic_write_text: another writer holds {lock_path}; refusing to wait. Identify the concurrent writer and retry."
        ) from exc
    return lock_fd


def _release_lock_fd(lock_fd: int) -> None:
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)


def atomic_write_text(
    path: Path,
    content: str,
    *,
    encoding: str = "utf-8",
    lock: bool = True,
) -> None:
    """Atomically replace ``path`` with ``content``, durable across crash.

    See module docstring for the full recipe and rationale. The
    ``lock`` flag exists for tests that exercise the durability path
    without involving cross-process serialisation; production callers
    must take the default (``True``).

    Raises:
        AtomicWriteConflictError: another writer holds the per-file
            lock. Tier-1: do not retry silently; surface to the
            operator.
        OSError: filesystem failure (disk full, permission denied,
            fsync error, parent directory missing). Propagated
            verbatim — Tier-1 demands a loud crash rather than a
            half-written audit file.
    """
    path = Path(path)
    parent = path.parent
    if not parent.exists():
        raise FileNotFoundError(
            f"atomic_write_text: parent directory does not exist: {parent}. "
            "Refusing to create it implicitly; the caller is responsible "
            "for allowlist directory provisioning."
        )

    lock_fd: int | None = None
    if lock:
        # ``O_CREAT`` so the lock file is created on first use; the
        # file persists between calls and the same inode is reused
        # for every subsequent ``flock``.
        lock_fd = _acquire_lock_fd(path, wait=False)

    temp_path = _temp_path_for(path)
    temp_created = False
    try:
        # ``O_EXCL`` defends against a hostile or racing process that
        # might have planted the exact same temp filename. Combined
        # with the lock, this is belt-and-braces. Until ``os.open``
        # returns success we have not created the inode, so the
        # cleanup branch below does not touch a pre-existing path
        # belonging to someone else.
        fd = os.open(
            str(temp_path),
            os.O_CREAT | os.O_WRONLY | os.O_EXCL,
            0o600,
        )
        temp_created = True
        try:
            payload = content.encode(encoding)
            written = os.write(fd, payload)
            if written != len(payload):
                # POSIX-legal short write: detect at the boundary
                # rather than silently shipping a truncated file
                # through to ``os.replace``. See
                # ``AtomicWriteShortWriteError`` for the rationale.
                raise AtomicWriteShortWriteError(
                    f"atomic_write_text: short write to {temp_path}: "
                    f"os.write returned {written} of {len(payload)} bytes. "
                    "Refusing to fsync+rename a truncated allowlist YAML; "
                    "Tier-1 doctrine treats this as an audit-trail anomaly."
                )
            os.fsync(fd)
        finally:
            os.close(fd)

        # Directory fsync BEFORE rename: ensures the temp inode and
        # its directory entry are durable. Without this, a crash
        # between the rename and the post-rename dir fsync can
        # observe the rename in the page cache but the temp inode
        # has not yet been linked on disk → corruption.
        _fsync_directory(parent)

        os.replace(str(temp_path), str(path))

        # Directory fsync AFTER rename: the rename itself must be
        # durable. POSIX ``rename`` is atomic w.r.t. observers but
        # not durable until the directory entry is sync'd to stable
        # storage.
        _fsync_directory(parent)
    except BaseException:
        # Only scrub temps we successfully created. ``O_EXCL`` may
        # have failed against a pre-existing inode (operator
        # forensics) — we must not silently delete a stranger's
        # file. ``missing_ok`` because ``os.replace`` may have
        # already consumed our temp inode before a post-rename
        # failure (e.g. post-rename dir-fsync error).
        if temp_created:
            try:
                Path(temp_path).unlink(missing_ok=True)
            finally:
                raise
        raise
    finally:
        if lock_fd is not None:
            # ``LOCK_UN`` and close releases the advisory lock; the
            # ``.lock`` file remains on disk for the next invocation.
            _release_lock_fd(lock_fd)


def atomic_update_text(
    path: Path,
    update: Callable[[str | None], str],
    *,
    encoding: str = "utf-8",
    create_parent: bool = False,
) -> None:
    """Atomically read, mutate, and replace ``path`` under one file lock.

    ``atomic_write_text`` protects only the replacement step. Callers that
    compute new content from the current file contents must use this helper
    instead, otherwise two processes can both read the same old bytes and
    then serialize two individually-atomic writes where the later replacement
    drops the earlier mutation.

    The ``update`` callable runs while the sibling ``.<name>.lock`` is held
    and receives the current text, or ``None`` if the file does not exist.
    The returned text is written with the same durable temp+fsync+replace
    recipe as :func:`atomic_write_text`, with the lock intentionally not
    reacquired inside the replacement step.
    """
    path = Path(path)
    parent = path.parent
    if create_parent:
        parent.mkdir(parents=True, exist_ok=True)
    elif not parent.exists():
        raise FileNotFoundError(
            f"atomic_update_text: parent directory does not exist: {parent}. "
            "Refusing to create it implicitly; pass create_parent=True for "
            "callers that own allowlist directory provisioning."
        )

    lock_fd = _acquire_lock_fd(path, wait=True)
    try:
        current = path.read_text(encoding=encoding) if path.exists() else None
        updated = update(current)
        atomic_write_text(path, updated, encoding=encoding, lock=False)
    finally:
        _release_lock_fd(lock_fd)
