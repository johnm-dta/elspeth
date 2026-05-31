"""Sidecar persistence for reaudit sweeps — crash-recovery primitive (T6b).

Closes the second half of elspeth-9a4e54cc01 (P0). T6 (commit 6b33ee5b3)
gave the renderer an "INCOMPLETE SWEEP" banner that fires when a
partial :class:`ReauditReport` is constructed with ``entries_dispatched
< total_entries``. That banner only renders, however, if a partial
report exists — and the prior orchestrator built the report at the END
of the loop, so any out-of-band kill (SIGKILL, ``KeyboardInterrupt``,
``RuntimeError`` propagated from a per-entry component, machine reboot)
left no report and no banner. The N silently-dropped entries vanished.

T6b closes the gap with **append-only sidecar JSONL**. Every reaudit
invocation writes a header line, one outcome line per entry, and a
trailer line on clean exit. ``flush + fsync`` after every line means a
SIGKILL within microseconds preserves the entries already written.
Absence of a trailer line on disk == sweep was killed mid-process.

Two recovery surfaces consume the sidecar:

* ``elspeth-lints reaudit --resume <run_id>`` reads the existing
  sidecar, skips the already-classified entries, continues from the
  first un-classified one, appends new outcomes to the SAME sidecar,
  and writes the trailer when the loop completes.
* ``elspeth-lints reaudit --render-incomplete <run_id>`` reads the
  sidecar, reconstructs a partial :class:`ReauditReport` (with
  ``entries_dispatched < total_entries`` taken from header), and
  hands it to the existing renderer — the T6 "INCOMPLETE SWEEP"
  banner fires automatically.

The sidecar's data model is audit data (Tier-1): malformed JSONL,
missing required fields, type mismatches, integrity hash drift all
crash the loader. No defensive coercion. The directory lives at
``<allowlist_dir>/.reaudit-state/`` — siting it inside the allowlist
config tree keeps the run-state co-located with the configuration it
operates against (and the matching ``.gitignore`` rule is the
operator's responsibility; surfaced in the commit message).

Concurrency control: ``fcntl.flock(LOCK_EX | LOCK_NB)`` on the sidecar
file blocks a second process from opening the same ``run_id`` for
write. Two concurrent reauditors writing the same JSONL would
interleave at line boundaries (Python writes are atomic for buffered
text under POSIX) but a non-trailer-marked sidecar with two interleaved
sequences of outcomes is not reconstructable in any meaningful way —
the lock is the simpler, correct answer.

Resume TOCTOU (T6c): ``--resume`` validation (load + trailer check +
header drift check) MUST happen inside the flock window — splitting
validate-then-acquire-lock allows two concurrent ``--resume`` processes
to both pass validation, then serialise on the flock, then both append
outcomes (the second past the first's trailer). The writer therefore
accepts an optional ``resume_validate`` callback that runs after the
exclusive lock is held; the CLI no longer touches the sidecar file
before entering the writer context.

Tail-truncation recovery (T6c, CRITICAL): a SIGKILL between the writer's
``write()`` and the next ``flush()``+``fsync()`` can leave the sidecar
with N complete lines + 1 partial final line (no trailing newline). The
loader distinguishes:

* **Tail truncation** — last line fails JSON parse AND the file does
  not end with ``\\n``. Treat as "process killed mid-write", log a
  structured warning to stderr naming the byte offset, return the prior
  complete outcomes. ``--render-incomplete`` then surfaces the recovered
  view; ``--resume`` continues from the next entry. The dead in-flight
  entry will be re-classified on resume.
* **Mid-file corruption** — a non-last line fails to parse, or the
  trailing-newline-test indicates the partial line is followed by more
  bytes. Tier-1 corruption: crash with ``SidecarCorruptError`` carrying
  line number + byte offset. Hand-editing, kernel page-cache
  inconsistency, or filesystem damage — operator-actionable, not
  recoverable.

This is the deliberate exception to "crash on Tier-1 corruption" carved
out by the sidecar's whole purpose. It is bounded to ``JSONDecodeError``
on the LAST line of a no-trailing-newline file; lines that parse JSON
but fail structured validation (``_outcome_from_dict`` raising) stay
sweep-fatal even on the last line — evidence corruption looking like a
valid prefix is still corruption.

Retention (T6c): sidecars with a trailer (completed sweeps) are deleted
lazily by ``SidecarWriter.__enter__`` when older than
:data:`COMPLETED_SIDECAR_RETENTION_DAYS`. Sidecars WITHOUT a trailer are
recoverable Tier-1 data and stay forever — they survive until the
operator either ``--resume``\\ s them to completion or starts a fresh
sweep. The cleanup acquires LOCK_NB before touching each candidate so an
in-progress resume on a stale-mtime sidecar is never raced.

Write-side partial-line truncation (T6d, CRITICAL): the T6c read-side
correctly detected the partial last line on load, but ``SidecarWriter``
opened in ``mode="a"`` re-corrupted the file on ``--resume``. POSIX
append mode positions the write head at EOF; when EOF is mid-partial-
line (no trailing newline), the first appended write glues the new
JSON onto the truncated tail, producing an unparseable line that
``load_sidecar``'s structural-validation path then raises
:class:`SidecarCorruptError` on (the file ends with a newline again, so
the tail-truncation heuristic no longer fires). The operator's natural
recovery action destroys the data the read-side fix preserved.

``SidecarWriter.__enter__`` therefore truncates any partial last line
INSIDE the flock-held window, AFTER the resume-validate callback runs
and BEFORE the first append. Detection: the on-disk bytes do not end
with ``\\n``. Truncation point: ``rfind(b"\\n") + 1`` (keep the final
newline; drop the partial bytes after it). Persistence: ``os.ftruncate``
on the open fd, ``os.fsync`` of the fd, then ``os.fsync`` of the parent
directory to make the inode-size change durable across crashes.

The partial last line is unrecoverable data — that is the deliberate
Tier-1 trade-off: losing the one in-flight outcome (which the killed
sweep had not durably persisted anyway) beats losing the entire
sidecar (the prior outcomes plus everything appended after the glued
line). The dropped entry is re-classified on resume because its key
never made it into ``classified_keys``.

Cleanly-terminated sidecars (final byte is ``\\n``) are not modified.
``--render-incomplete`` reads sidecars via :func:`load_sidecar` directly
and never enters the writer; render is read-only and never truncates.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast

from elspeth_lints.core.allowlist import AllowlistEntry, JudgeVerdict
from elspeth_lints.core.reaudit import (
    ReauditCause,
    ReauditDivergence,
    ReauditError,
    ReauditOutcome,
    ReauditReport,
)

# Bump on any breaking change to the JSONL schema. The loader refuses
# to read a sidecar whose header advertises a different version — the
# operator's mid-sweep state is bound to a specific schema, and an
# upgrade that silently changes line shapes would corrupt
# reconstruction.
SIDECAR_SCHEMA_VERSION = 5  # v5: outcome records carry the reaudit cause axis

SIDECAR_DIRNAME = ".reaudit-state"

# Lazy-cleanup horizon for COMPLETED sweeps (trailer present). Sidecars
# without a trailer are recoverable Tier-1 data and stay forever — only
# the operator (via --resume → completion, or by deleting the run_id)
# can retire them. 30 days mirrors the project's observation-expiry
# window: long enough to outlast a post-incident audit cycle, short
# enough that a routine reaudit cadence keeps the directory bounded.
COMPLETED_SIDECAR_RETENTION_DAYS = 30


# =========================================================================
# Public surface
# =========================================================================


def generate_run_id() -> str:
    """Return a fresh ``run_id`` for a new sweep.

    UUID4 hex (32 chars, no hyphens) — short enough to type at the
    ``--resume`` prompt, long enough that operator concurrent invocations
    can't collide by accident.
    """
    return uuid.uuid4().hex


def sidecar_path_for(allowlist_dir: Path, run_id: str) -> Path:
    """Resolve the JSONL path for a given allowlist directory + run_id.

    The sidecar directory is created lazily by :class:`SidecarWriter`'s
    ``__enter__`` so callers don't need to pre-create anything.
    """
    return allowlist_dir / SIDECAR_DIRNAME / f"{run_id}.jsonl"


def compute_allowlist_hash(allowlist_dir: Path) -> str:
    """SHA-256 over (filename, content_bytes) of every YAML file in the dir.

    The hash binds a sidecar to the exact allowlist state the sweep
    observed at start. A ``--resume`` whose recomputed hash diverges
    crashes — the operator must either revert the allowlist edit or
    start a fresh sweep.

    Files are walked in sorted order (lexicographic on relative path)
    so the hash is deterministic across filesystems. Both ``*.yaml`` and
    ``*.yml`` are included — YAML's permitted suffixes per the YAML 1.2
    media-type registration, and operator-authored allowlist files in
    the wild do appear under both. The sidecar's own ``.reaudit-state/``
    directory is excluded (its contents would otherwise self-modify the
    hash mid-sweep).
    """
    hasher = hashlib.sha256()
    candidates = list(allowlist_dir.rglob("*.yaml")) + list(allowlist_dir.rglob("*.yml"))
    for path in sorted(candidates, key=lambda p: p.relative_to(allowlist_dir).as_posix()):
        if SIDECAR_DIRNAME in path.parts:
            continue
        rel = path.relative_to(allowlist_dir).as_posix()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


# =========================================================================
# Header / outcome / trailer line shapes
# =========================================================================


@dataclass(frozen=True, slots=True)
class SidecarHeader:
    """First line of every sidecar — sweep-identifying metadata.

    ``allowlist_hash`` is the integrity check that lets ``--resume``
    detect "the allowlist was edited between the original sweep and the
    resume" (and crash rather than silently produce a corrupt report).

    Filter fields (``rule_filter``, ``since_iso``, ``limit``,
    ``include_pre_judge``) bind the sweep to the exact filtered entry
    list. A ``--resume`` whose filters differ from the header crashes:
    re-deriving ``filtered`` with different filters would produce a
    different entry order and the "skip already-classified" logic
    becomes meaningless.
    """

    run_id: str
    started_at: datetime
    total_entries: int
    allowlist_path: str
    allowlist_hash: str
    rule_filter: str
    since_iso: str | None
    limit: int | None
    include_pre_judge: bool
    schema_version: int = SIDECAR_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class SidecarTrailer:
    """Final line of a cleanly-completed sweep.

    Presence of a trailer == sweep ran to completion (every dispatched
    entry got an outcome). Absence == sweep was killed mid-process or
    crashed above the per-entry boundary.
    """

    run_id: str
    finished_at: datetime
    outcomes_written: int


# =========================================================================
# Writer (append-only, fsync-per-line, flock-guarded)
# =========================================================================


class SidecarWriter:
    """Context manager that owns the sidecar file for one sweep.

    Use as ``with SidecarWriter(path, header) as writer:`` — the header
    is written on enter, ``write_outcome()`` appends one line per
    entry, and ``commit_trailer()`` writes the trailer line on clean
    exit. If the block exits via exception, the trailer is NOT written:
    the sidecar persists without a trailer, marking the sweep as
    incomplete-on-disk.

    The file is opened with ``flock(LOCK_EX | LOCK_NB)`` so a second
    process targeting the same ``run_id`` fails immediately with a
    clear error. The lock is released on context exit (either path).

    Resume mode (T6c TOCTOU fix): when ``append=True``, the caller may
    supply ``on_resume_locked`` — a callback that runs *inside* the
    flock-held window, after the lock is acquired but before any append
    occurs. The callback receives the :class:`LoadedSidecar` parsed
    from the file under the lock. This eliminates the validate-then-
    acquire-lock race in which two concurrent ``--resume`` processes
    both pass validation, serialise on the lock, then both append past
    each other's trailers. With the callback running under the lock,
    only one resume can validate at a time.

    Partial-last-line truncation (T6d): if the sidecar's final byte is
    not ``\\n``, the writer truncates the file to the last newline
    boundary BEFORE the first append, with ``os.ftruncate`` on the
    locked fd plus an ``os.fsync`` of the file and the parent
    directory. The truncation is the deliberate Tier-1 trade-off
    documented at module scope: the partial last line (a SIGKILL'd
    in-flight write, never durably persisted) is LOST, but everything
    written before it is preserved, and the new outcomes append cleanly
    without gluing onto a truncated tail. Without this truncation,
    POSIX append-mode writes start at the mid-line byte offset and
    produce an unparseable glued line that the loader's structural
    validation rejects with :class:`SidecarCorruptError` — destroying
    the very data the T6c read-side recovery preserved. The
    truncation runs AFTER ``on_resume_locked`` (so a rejected resume
    leaves the file untouched) and INSIDE the flock window (so a
    second process cannot append between read-and-truncate).
    """

    def __init__(
        self,
        sidecar_path: Path,
        header: SidecarHeader,
        *,
        append: bool = False,
        on_resume_locked: Callable[[LoadedSidecar], None] | None = None,
    ) -> None:
        if on_resume_locked is not None and not append:
            raise ValueError("on_resume_locked is only meaningful with append=True; a fresh sweep has no prior sidecar to validate.")
        self._sidecar_path = sidecar_path
        self._header = header
        self._append = append
        self._on_resume_locked = on_resume_locked
        self._file: Any = None  # set in __enter__
        self._outcomes_written = 0

    def __enter__(self) -> Self:
        self._sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        # Retention runs OUTSIDE the per-run flock — it acquires its own
        # LOCK_NB on each candidate so an in-progress resume on a stale-
        # mtime sidecar is never raced. Best-effort: failures don't
        # block the sweep.
        _prune_expired_completed_sidecars(self._sidecar_path.parent)
        mode = "a" if self._append else "x"
        try:
            self._file = self._sidecar_path.open(mode, encoding="utf-8")
        except FileExistsError as exc:
            raise SidecarConflictError(
                f"sidecar already exists at {self._sidecar_path} (run_id={self._header.run_id!r}); "
                "this run_id has been used by a prior sweep. Pick a different run_id "
                "or use --resume <run_id> to continue the prior sweep."
            ) from exc
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._file.close()
            self._file = None
            raise SidecarConflictError(
                f"sidecar {self._sidecar_path} is locked by another process; "
                "concurrent reaudit invocations on the same run_id are not "
                "supported because their outcome lines would interleave into "
                "an unreconstructable JSONL stream."
            ) from exc
        if self._on_resume_locked is not None:
            try:
                loaded = load_sidecar(self._sidecar_path)
                self._on_resume_locked(loaded)
            except BaseException:
                # Validation rejected the resume (or load found
                # corruption). Release the lock + fd before re-raising so
                # the operator's retry sees a clean state.
                try:
                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
                finally:
                    self._file.close()
                    self._file = None
                raise
        if self._append:
            # T6d: truncate any partial last line BEFORE the first
            # append. POSIX append mode positions writes at EOF; if EOF
            # sits mid-partial-line (no trailing newline because a
            # SIGKILL caught the writer between write() and the next
            # flush()+fsync()), the first append would glue its JSON
            # onto the truncated tail and produce an unparseable line.
            # The next load_sidecar would then crash with
            # SidecarCorruptError because the file once again ends with
            # "\n", so the T6c tail-truncation heuristic no longer
            # fires. Truncating to the last newline boundary preserves
            # every durable prior outcome at the cost of the
            # never-durable partial one — the Tier-1 trade-off
            # documented in the module docstring.
            try:
                self._truncate_partial_tail()
            except BaseException:
                try:
                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
                finally:
                    self._file.close()
                    self._file = None
                raise
        if not self._append:
            self._write_line(_header_to_dict(self._header))
        return self

    def _truncate_partial_tail(self) -> None:
        """Drop any partial last line; no-op when the file ends with ``\\n``.

        Reads the on-disk bytes (we hold the exclusive flock, so no
        other process can have appended since the fd was opened),
        locates the last newline, and ``ftruncate``s to one past it.
        Then ``fsync`` the fd so the inode-size shrink is durable, and
        ``fsync`` the parent directory so the new size survives a
        crash before the next file-content write.

        Failure modes propagate as :class:`OSError` (disk full,
        permission, EIO from the underlying device) — those are
        operator-actionable and the writer's caller surfaces them
        through the same ReauditError exit path the rest of the module
        uses; wrapping them here would obscure the cause.

        Idempotent: a clean newline-terminated file is left
        byte-identical (no ftruncate, no fsync). An empty file is also
        a no-op (the file ends with no bytes at all; nothing to
        truncate).
        """
        raw_bytes = self._sidecar_path.read_bytes()
        if not raw_bytes:
            return
        if raw_bytes.endswith(b"\n"):
            return
        last_newline = raw_bytes.rfind(b"\n")
        # last_newline == -1 means the file is one partial line with no
        # newline anywhere — truncation point is 0, the file becomes
        # empty. In practice this is unreachable on the resume path
        # because on_resume_locked has already called load_sidecar
        # which would have raised on "no header" or recovered the
        # partial line at the offset and continued. Defending against
        # it costs nothing and keeps this helper self-contained.
        truncate_to = last_newline + 1
        os.ftruncate(self._file.fileno(), truncate_to)
        os.fsync(self._file.fileno())
        # Parent-directory fsync makes the inode-size change durable
        # across a crash — without it, the kernel may have queued the
        # metadata update separately from the file content writes,
        # and a crash between truncate and the next outcome write
        # could resurrect the dropped bytes via the page cache.
        dir_fd = os.open(str(self._sidecar_path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._file is not None:
            try:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            finally:
                self._file.close()
                self._file = None

    def write_outcome(self, outcome: ReauditOutcome) -> None:
        """Append one outcome line, flushing and fsyncing immediately.

        The fsync is the recovery guarantee: a SIGKILL one instruction
        after this method returns must leave the outcome durable on
        disk. Without fsync, the kernel's page cache could swallow the
        line on power loss / abrupt termination.
        """
        if self._file is None:
            raise RuntimeError("SidecarWriter not entered; use 'with SidecarWriter(...)'")
        self._write_line(_outcome_to_dict(outcome))
        self._outcomes_written += 1

    def commit_trailer(self) -> None:
        """Append the trailer line, marking the sweep complete on disk."""
        if self._file is None:
            raise RuntimeError("SidecarWriter not entered; use 'with SidecarWriter(...)'")
        trailer = SidecarTrailer(
            run_id=self._header.run_id,
            finished_at=datetime.now(UTC),
            outcomes_written=self._outcomes_written,
        )
        self._write_line(_trailer_to_dict(trailer))

    def _write_line(self, payload: dict[str, Any]) -> None:
        if self._file is None:
            raise RuntimeError("SidecarWriter not entered; use 'with SidecarWriter(...)'")
        line = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())


class SidecarConflictError(ReauditError):
    """A sidecar with this ``run_id`` already exists or is locked.

    Subclass of :class:`ReauditError` so the CLI's existing exit-2
    branch surfaces it as an operator-actionable error.
    """


class SidecarCorruptError(ReauditError):
    """A sidecar exists but cannot be loaded.

    Used for malformed JSONL, missing required fields, schema-version
    mismatch, header integrity drift. The sidecar is Tier-1 audit data;
    "I can't read it" is a crash, not a recovery path.
    """


# =========================================================================
# Loader (used by --resume and --render-incomplete)
# =========================================================================


@dataclass(frozen=True, slots=True)
class LoadedSidecar:
    """In-memory reconstruction of a sidecar's contents."""

    header: SidecarHeader
    outcomes: tuple[ReauditOutcome, ...]
    trailer: SidecarTrailer | None
    classified_keys: frozenset[str]


def load_sidecar(sidecar_path: Path) -> LoadedSidecar:
    """Read a sidecar JSONL into structured form.

    Crashes on:
    * file missing
    * empty file (missing header)
    * malformed JSON on any non-final line, OR a final line followed by
      bytes (impossible in append-only design but defended against)
    * line missing ``type`` discriminant
    * structured validation failure on any outcome / header / trailer
      payload (even if the JSON parses)
    * header schema_version != SIDECAR_SCHEMA_VERSION
    * trailer ``run_id`` mismatch with header
    * more than one header or trailer

    RECOVERS from (T6c CRITICAL): the LAST line failing JSON parse when
    the file does NOT end with a newline. Treats this as "process killed
    between write() and the next flush()+fsync()" and:

    * Logs a structured warning to stderr naming the byte offset of the
      truncated line and the recovered-outcome count
    * Returns ``LoadedSidecar`` with the prior complete outcomes; no
      trailer, no in-flight outcome
    * The in-flight entry whose write was killed is dropped — resume
      will re-classify it (its key was never added to
      ``classified_keys``); ``--render-incomplete`` will report it via
      ``entries_dispatched < total_entries``

    The recovery path is bounded to ``JSONDecodeError`` on the final line
    of a file missing its terminating newline. Lines that JSON-parse but
    fail structured validation stay sweep-fatal even on the last line —
    evidence corruption that looks like a valid prefix is still
    corruption.
    """
    if not sidecar_path.exists():
        raise SidecarCorruptError(f"sidecar {sidecar_path} does not exist")
    raw_bytes = sidecar_path.read_bytes()
    if not raw_bytes:
        raise SidecarCorruptError(f"sidecar {sidecar_path} is empty (no header line)")
    text = raw_bytes.decode("utf-8")
    # The writer's discipline (line + "\n" → flush → fsync, all in
    # _write_line) guarantees every COMPLETE line ends in "\n". Absence
    # of a trailing newline therefore implies the final line was being
    # written when the process died.
    ends_with_newline = text.endswith("\n")
    # keepends=True so we can compute per-line byte offsets for the
    # truncation warning. Empty / whitespace-only entries are skipped at
    # the consumption site below.
    raw_lines_with_eol = text.splitlines(keepends=True)
    if not raw_lines_with_eol:
        raise SidecarCorruptError(f"sidecar {sidecar_path} is empty (no header line)")

    header: SidecarHeader | None = None
    trailer: SidecarTrailer | None = None
    outcomes: list[ReauditOutcome] = []
    classified_keys: set[str] = set()

    last_index = len(raw_lines_with_eol) - 1
    running_offset = 0
    for line_index, raw_with_eol in enumerate(raw_lines_with_eol):
        line_no = line_index + 1
        line_offset = running_offset
        running_offset += len(raw_with_eol.encode("utf-8"))
        raw = raw_with_eol.rstrip("\n")
        if not raw.strip():
            continue
        is_last_line = line_index == last_index
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            if is_last_line and not ends_with_newline:
                # T6c recovery — partial last line from SIGKILL between
                # write() and flush()+fsync(). The in-flight outcome is
                # lost; everything written before it is durable. Log to
                # stderr (operational visibility for the operator-driven
                # --resume / --render-incomplete decision) and return the
                # prior complete state.
                sys.stderr.write(
                    f"reaudit sidecar {sidecar_path}: partial final line "
                    f"detected at byte offset {line_offset} "
                    f"(no trailing newline; JSON parse failed: {exc}). "
                    "Treating as 'process killed mid-write'; the in-flight "
                    "outcome is dropped and will be re-classified on "
                    f"--resume. {len(outcomes)} prior complete outcome(s) "
                    "recovered.\n"
                )
                break
            raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no} (byte offset {line_offset}): malformed JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: expected JSON object, got {type(payload).__name__}")
        line_type = payload.get("type")
        if line_type == "header":
            if header is not None:
                raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: duplicate header (only one allowed)")
            header = _header_from_dict(payload, sidecar_path=sidecar_path, line_no=line_no)
        elif line_type == "outcome":
            if header is None:
                raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: outcome line precedes header")
            outcome = _outcome_from_dict(payload, sidecar_path=sidecar_path, line_no=line_no)
            outcomes.append(outcome)
            classified_keys.add(outcome.entry.key)
        elif line_type == "trailer":
            if header is None:
                raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: trailer line precedes header")
            if trailer is not None:
                raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: duplicate trailer (only one allowed)")
            trailer = _trailer_from_dict(payload, sidecar_path=sidecar_path, line_no=line_no)
            if trailer.run_id != header.run_id:
                raise SidecarCorruptError(
                    f"sidecar {sidecar_path} line {line_no}: trailer run_id {trailer.run_id!r} "
                    f"does not match header run_id {header.run_id!r}"
                )
        else:
            raise SidecarCorruptError(
                f"sidecar {sidecar_path} line {line_no}: unknown line type {line_type!r}; expected one of header/outcome/trailer"
            )

    if header is None:
        raise SidecarCorruptError(f"sidecar {sidecar_path} has no header line")

    return LoadedSidecar(
        header=header,
        outcomes=tuple(outcomes),
        trailer=trailer,
        classified_keys=frozenset(classified_keys),
    )


def _prune_expired_completed_sidecars(sidecar_dir: Path) -> None:
    """Delete completed sidecars older than ``COMPLETED_SIDECAR_RETENTION_DAYS``.

    Called by :class:`SidecarWriter.__enter__` for lazy cleanup. Only
    sidecars with a trailer (completed sweeps) are eligible — incomplete
    sidecars are recoverable Tier-1 data and stay until the operator
    finishes or abandons them.

    Each candidate is gated by ``LOCK_NB`` on its own fd so an
    in-progress resume on a stale-mtime sidecar is never raced. Best-
    effort: any error on a single candidate is silently skipped (the
    sweep itself must not be blocked by cleanup of an unrelated file).
    """
    if not sidecar_dir.exists():
        return
    horizon_seconds = COMPLETED_SIDECAR_RETENTION_DAYS * 86400
    now = time.time()
    for candidate in sidecar_dir.glob("*.jsonl"):
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if now - mtime <= horizon_seconds:
            continue
        try:
            with candidate.open("r", encoding="utf-8") as fp:
                try:
                    fcntl.flock(fp.fileno(), fcntl.LOCK_NB | fcntl.LOCK_EX)
                except BlockingIOError:
                    continue
                try:
                    loaded = load_sidecar(candidate)
                finally:
                    fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        except (OSError, SidecarCorruptError):
            # A corrupt sidecar past the retention horizon is also
            # operator-actionable, but cleanup is the wrong surface to
            # crash on. Skip; the next direct load will surface it.
            continue
        if loaded.trailer is not None:
            try:
                candidate.unlink()
            except OSError:
                continue


def report_from_loaded_sidecar(loaded: LoadedSidecar) -> ReauditReport:
    """Build a :class:`ReauditReport` from a sidecar's recorded outcomes.

    Used by ``--render-incomplete`` to surface a killed-sweep's partial
    state to the operator. ``entries_dispatched`` = number of outcome
    lines (each represents a successfully-dispatched entry). The
    renderer's T6 banner fires when this is less than the header's
    ``total_entries``.
    """
    return ReauditReport.from_outcomes(
        loaded.outcomes,
        entries_dispatched=len(loaded.outcomes),
        total_entries=loaded.header.total_entries,
    )


# =========================================================================
# JSON encoding / decoding for sidecar line shapes
# =========================================================================


def _header_to_dict(header: SidecarHeader) -> dict[str, Any]:
    return {
        "type": "header",
        "schema_version": header.schema_version,
        "run_id": header.run_id,
        "started_at": header.started_at.isoformat(),
        "total_entries": header.total_entries,
        "allowlist_path": header.allowlist_path,
        "allowlist_hash": header.allowlist_hash,
        "rule_filter": header.rule_filter,
        "since_iso": header.since_iso,
        "limit": header.limit,
        "include_pre_judge": header.include_pre_judge,
    }


def _header_from_dict(payload: dict[str, Any], *, sidecar_path: Path, line_no: int) -> SidecarHeader:
    schema_version = _required(payload, "schema_version", int, sidecar_path, line_no)
    if schema_version != SIDECAR_SCHEMA_VERSION:
        raise SidecarCorruptError(
            f"sidecar {sidecar_path} line {line_no}: schema_version={schema_version} "
            f"is incompatible with this build (expected {SIDECAR_SCHEMA_VERSION}). "
            "A sidecar's lifetime is bound to one schema; pick a different run_id "
            "and start a fresh sweep."
        )
    started_at_str = _required(payload, "started_at", str, sidecar_path, line_no)
    started_at = _parse_iso_datetime(started_at_str, sidecar_path=sidecar_path, line_no=line_no, field="started_at")
    since_iso = payload.get("since_iso")
    if since_iso is not None and not isinstance(since_iso, str):
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: since_iso must be str or null; got {type(since_iso).__name__}")
    limit = payload.get("limit")
    if limit is not None and not isinstance(limit, int):
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: limit must be int or null; got {type(limit).__name__}")
    return SidecarHeader(
        run_id=_required(payload, "run_id", str, sidecar_path, line_no),
        started_at=started_at,
        total_entries=_required(payload, "total_entries", int, sidecar_path, line_no),
        allowlist_path=_required(payload, "allowlist_path", str, sidecar_path, line_no),
        allowlist_hash=_required(payload, "allowlist_hash", str, sidecar_path, line_no),
        rule_filter=_required(payload, "rule_filter", str, sidecar_path, line_no),
        since_iso=since_iso,
        limit=limit,
        include_pre_judge=_required(payload, "include_pre_judge", bool, sidecar_path, line_no),
        schema_version=schema_version,
    )


def _trailer_to_dict(trailer: SidecarTrailer) -> dict[str, Any]:
    return {
        "type": "trailer",
        "run_id": trailer.run_id,
        "finished_at": trailer.finished_at.isoformat(),
        "outcomes_written": trailer.outcomes_written,
    }


def _trailer_from_dict(payload: dict[str, Any], *, sidecar_path: Path, line_no: int) -> SidecarTrailer:
    finished_at_str = _required(payload, "finished_at", str, sidecar_path, line_no)
    return SidecarTrailer(
        run_id=_required(payload, "run_id", str, sidecar_path, line_no),
        finished_at=_parse_iso_datetime(finished_at_str, sidecar_path=sidecar_path, line_no=line_no, field="finished_at"),
        outcomes_written=_required(payload, "outcomes_written", int, sidecar_path, line_no),
    )


def _outcome_to_dict(outcome: ReauditOutcome) -> dict[str, Any]:
    return {
        "type": "outcome",
        "appended_at": datetime.now(UTC).isoformat(),
        "entry": _entry_to_dict(outcome.entry),
        "original_verdict": _verdict_value(outcome.original_verdict),
        "original_model_verdict": _verdict_value(outcome.original_model_verdict),
        "fresh_verdict": _verdict_value(outcome.fresh_verdict),
        "fresh_model_id": outcome.fresh_model_id,
        "fresh_rationale": outcome.fresh_rationale,
        "fresh_recorded_at": outcome.fresh_recorded_at.isoformat() if outcome.fresh_recorded_at is not None else None,
        "judge_call_attempted": outcome.judge_call_attempted,
        "fresh_prompt_tokens_total": outcome.fresh_prompt_tokens_total,
        "fresh_prompt_tokens_cached": outcome.fresh_prompt_tokens_cached,
        "divergence": outcome.divergence.value,
        "cause": outcome.cause.value,
        "code_snapshot": outcome.code_snapshot,
        # Secrets-scrubber audit record (closes elspeth-ebb2b88753 /
        # C2-2 on the sweep path). Each redaction is captured as a
        # dict so the JSONL is self-describing without needing the
        # RedactionRecord dataclass at read time. Empty list when the
        # scrubber ran clean.
        "excerpt_redactions": [
            {
                "pattern_name": r.pattern_name,
                "byte_count": r.byte_count,
                "redacted_hash": r.redacted_hash,
            }
            for r in outcome.excerpt_redactions
        ],
    }


def _outcome_from_dict(payload: dict[str, Any], *, sidecar_path: Path, line_no: int) -> ReauditOutcome:
    entry_payload = payload.get("entry")
    if not isinstance(entry_payload, dict):
        raise SidecarCorruptError(
            f"sidecar {sidecar_path} line {line_no}: outcome.entry must be JSON object; got {type(entry_payload).__name__}"
        )
    entry = _entry_from_dict(entry_payload, sidecar_path=sidecar_path, line_no=line_no)
    divergence_str = _required(payload, "divergence", str, sidecar_path, line_no)
    try:
        divergence = ReauditDivergence(divergence_str)
    except ValueError as exc:
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: unknown divergence {divergence_str!r}") from exc
    cause_str = _required(payload, "cause", str, sidecar_path, line_no)
    try:
        cause = ReauditCause(cause_str)
    except ValueError as exc:
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: unknown cause {cause_str!r}") from exc
    fresh_recorded_at_raw = payload.get("fresh_recorded_at")
    if fresh_recorded_at_raw is not None:
        if not isinstance(fresh_recorded_at_raw, str):
            raise SidecarCorruptError(
                f"sidecar {sidecar_path} line {line_no}: fresh_recorded_at must be str or null; got {type(fresh_recorded_at_raw).__name__}"
            )
        fresh_recorded_at = _parse_iso_datetime(
            fresh_recorded_at_raw, sidecar_path=sidecar_path, line_no=line_no, field="fresh_recorded_at"
        )
    else:
        fresh_recorded_at = None
    fresh_rationale = payload.get("fresh_rationale")
    if fresh_rationale is not None and not isinstance(fresh_rationale, str):
        raise SidecarCorruptError(
            f"sidecar {sidecar_path} line {line_no}: fresh_rationale must be str or null; got {type(fresh_rationale).__name__}"
        )
    if "fresh_model_id" not in payload:
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: missing required field 'fresh_model_id'")
    fresh_model_id = payload["fresh_model_id"]
    if fresh_model_id is not None and not isinstance(fresh_model_id, str):
        raise SidecarCorruptError(
            f"sidecar {sidecar_path} line {line_no}: fresh_model_id must be str or null; got {type(fresh_model_id).__name__}"
        )
    judge_call_attempted = _required(payload, "judge_call_attempted", bool, sidecar_path, line_no)
    fresh_prompt_tokens_total = _optional_int(payload, "fresh_prompt_tokens_total", sidecar_path, line_no)
    fresh_prompt_tokens_cached = _optional_int(payload, "fresh_prompt_tokens_cached", sidecar_path, line_no)
    # excerpt_redactions is REQUIRED on every outcome line written
    # by this version of the writer (sidecar schema v3). Per the
    # project's Tier-1 doctrine + the No Legacy Code Policy, a v1
    # sidecar (without the field) crashes the load via the
    # schema_version check on the header; that is the right
    # operator-actionable signal ("pick a new run_id and restart").
    # An empty list is the meaningful "scrubber ran clean" value —
    # distinct from absence, which would be evidence corruption.
    raw_redactions = _required(payload, "excerpt_redactions", list, sidecar_path, line_no)
    redactions: list[Any] = []
    from elspeth_lints.core.source_excerpt import RedactionRecord

    for r_index, raw_r in enumerate(raw_redactions):
        if not isinstance(raw_r, dict):
            raise SidecarCorruptError(
                f"sidecar {sidecar_path} line {line_no}: excerpt_redactions[{r_index}] must be a JSON object; got {type(raw_r).__name__}"
            )
        redactions.append(
            RedactionRecord(
                pattern_name=_required(raw_r, "pattern_name", str, sidecar_path, line_no),
                byte_count=_required(raw_r, "byte_count", int, sidecar_path, line_no),
                redacted_hash=_required(raw_r, "redacted_hash", str, sidecar_path, line_no),
            )
        )
    return ReauditOutcome(
        entry=entry,
        original_verdict=_verdict_from_value(payload.get("original_verdict"), sidecar_path, line_no, "original_verdict"),
        original_model_verdict=_verdict_from_value(payload.get("original_model_verdict"), sidecar_path, line_no, "original_model_verdict"),
        fresh_verdict=_verdict_from_value(payload.get("fresh_verdict"), sidecar_path, line_no, "fresh_verdict"),
        fresh_rationale=fresh_rationale,
        fresh_recorded_at=fresh_recorded_at,
        fresh_model_id=fresh_model_id,
        judge_call_attempted=judge_call_attempted,
        fresh_prompt_tokens_total=fresh_prompt_tokens_total,
        fresh_prompt_tokens_cached=fresh_prompt_tokens_cached,
        divergence=divergence,
        cause=cause,
        code_snapshot=_required(payload, "code_snapshot", str, sidecar_path, line_no),
        excerpt_redactions=tuple(redactions),
    )


def _entry_to_dict(entry: AllowlistEntry) -> dict[str, Any]:
    return {
        "key": entry.key,
        "owner": entry.owner,
        "reason": entry.reason,
        "safety": entry.safety,
        "expires": entry.expires.isoformat() if entry.expires is not None else None,
        "file_fingerprint": entry.file_fingerprint,
        "scope_fingerprint": entry.scope_fingerprint,
        "judge_signature_version": entry.judge_signature_version,
        "judge_transport": entry.judge_transport,
        "ast_path": entry.ast_path,
        "pattern": entry.pattern,
        "source_file": entry.source_file,
        "judge_verdict": _verdict_value(entry.judge_verdict),
        "judge_recorded_at": entry.judge_recorded_at.isoformat() if entry.judge_recorded_at is not None else None,
        "judge_model": entry.judge_model,
        "judge_rationale": entry.judge_rationale,
        "judge_confidence": entry.judge_confidence,
        "judge_model_verdict": _verdict_value(entry.judge_model_verdict),
        "judge_policy_hash": entry.judge_policy_hash,
        "judge_metadata_signature": entry.judge_metadata_signature,
    }


def _entry_from_dict(payload: dict[str, Any], *, sidecar_path: Path, line_no: int) -> AllowlistEntry:
    expires_raw = payload.get("expires")
    if expires_raw is not None:
        if not isinstance(expires_raw, str):
            raise SidecarCorruptError(
                f"sidecar {sidecar_path} line {line_no}: entry.expires must be str or null; got {type(expires_raw).__name__}"
            )
        try:
            expires: date | None = date.fromisoformat(expires_raw)
        except ValueError as exc:
            raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: entry.expires not ISO-8601 date: {expires_raw!r}") from exc
    else:
        expires = None
    judge_recorded_raw = payload.get("judge_recorded_at")
    if judge_recorded_raw is not None:
        if not isinstance(judge_recorded_raw, str):
            raise SidecarCorruptError(
                f"sidecar {sidecar_path} line {line_no}: entry.judge_recorded_at must be str or null; got {type(judge_recorded_raw).__name__}"
            )
        judge_recorded_at = _parse_iso_datetime(
            judge_recorded_raw, sidecar_path=sidecar_path, line_no=line_no, field="entry.judge_recorded_at"
        )
    else:
        judge_recorded_at = None
    return AllowlistEntry(
        key=_required(payload, "key", str, sidecar_path, line_no),
        owner=_required(payload, "owner", str, sidecar_path, line_no),
        reason=_required(payload, "reason", str, sidecar_path, line_no),
        safety=_required(payload, "safety", str, sidecar_path, line_no),
        expires=expires,
        file_fingerprint=_optional_str(payload, "file_fingerprint", sidecar_path, line_no),
        scope_fingerprint=_optional_str(payload, "scope_fingerprint", sidecar_path, line_no),
        judge_signature_version=_optional_int(payload, "judge_signature_version", sidecar_path, line_no),
        judge_transport=_optional_str(payload, "judge_transport", sidecar_path, line_no),
        ast_path=_optional_str(payload, "ast_path", sidecar_path, line_no),
        pattern=_optional_str(payload, "pattern", sidecar_path, line_no),
        source_file=_required(payload, "source_file", str, sidecar_path, line_no),
        judge_verdict=_verdict_from_value(payload.get("judge_verdict"), sidecar_path, line_no, "entry.judge_verdict"),
        judge_recorded_at=judge_recorded_at,
        judge_model=_optional_str(payload, "judge_model", sidecar_path, line_no),
        judge_rationale=_optional_str(payload, "judge_rationale", sidecar_path, line_no),
        judge_confidence=_optional_confidence(payload, "judge_confidence", sidecar_path, line_no),
        judge_model_verdict=_verdict_from_value(payload.get("judge_model_verdict"), sidecar_path, line_no, "entry.judge_model_verdict"),
        judge_policy_hash=_optional_str(payload, "judge_policy_hash", sidecar_path, line_no),
        judge_metadata_signature=_optional_str(payload, "judge_metadata_signature", sidecar_path, line_no),
    )


def _verdict_value(verdict: JudgeVerdict | None) -> str | None:
    return verdict.value if verdict is not None else None


def _verdict_from_value(
    raw: Any,
    sidecar_path: Path,
    line_no: int,
    field: str,
) -> JudgeVerdict | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: {field} must be str or null; got {type(raw).__name__}")
    try:
        return JudgeVerdict(raw)
    except ValueError as exc:
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: {field} has unknown verdict {raw!r}") from exc


def _required(payload: dict[str, Any], field: str, expected_type: type, sidecar_path: Path, line_no: int) -> Any:
    if field not in payload:
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: missing required field {field!r}")
    value = payload[field]
    # bool is a subclass of int in Python; check bool first so a stray
    # True doesn't satisfy an int field (or vice versa).
    if expected_type is int and isinstance(value, bool):
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: {field} must be int; got bool")
    if expected_type is bool and not isinstance(value, bool):
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: {field} must be bool; got {type(value).__name__}")
    if not isinstance(value, expected_type):
        raise SidecarCorruptError(
            f"sidecar {sidecar_path} line {line_no}: {field} must be {expected_type.__name__}; got {type(value).__name__}"
        )
    return value


def _optional_str(payload: dict[str, Any], field: str, sidecar_path: Path, line_no: int) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: {field} must be str or null; got {type(value).__name__}")
    return value


def _optional_int(payload: dict[str, Any], field: str, sidecar_path: Path, line_no: int) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: {field} must be int or null; got {type(value).__name__}")
    if value < 0:
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: {field} must be non-negative")
    return cast(int, value)


def _optional_confidence(payload: dict[str, Any], field: str, sidecar_path: Path, line_no: int) -> float | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SidecarCorruptError(
            f"sidecar {sidecar_path} line {line_no}: {field} must be a number from 0.0 to 1.0 or null; got {type(value).__name__}"
        )
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: {field} must be between 0.0 and 1.0")
    return confidence


def _parse_iso_datetime(value: str, *, sidecar_path: Path, line_no: int, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SidecarCorruptError(f"sidecar {sidecar_path} line {line_no}: {field} not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise SidecarCorruptError(
            f"sidecar {sidecar_path} line {line_no}: {field} is timezone-naive ({value!r}); sidecar timestamps must be timezone-aware"
        )
    return parsed


# =========================================================================
# Header-integrity validation (--resume)
# =========================================================================


def validate_header_for_resume(
    *,
    header: SidecarHeader,
    allowlist_dir: Path,
    rule_filter: str,
    since_iso: str | None,
    limit: int | None,
    include_pre_judge: bool,
) -> None:
    """Crash if the on-disk header is incompatible with the current resume request.

    Three classes of mismatch:

    1. ``allowlist_hash`` drift — the YAML files were edited between
       the original sweep and the resume. Re-deriving the filtered
       entry list would produce a different order or different entries
       and the "skip already-classified" logic becomes meaningless.
    2. Filter argument drift — ``--rule`` / ``--since`` / ``--limit`` /
       ``--include-pre-judge`` differ from the header. Different
       filters produce a different filtered list; resume becomes
       reconstruction of a sweep that never existed.
    3. ``allowlist_path`` drift — the operator pointed ``--allowlist-dir``
       at a different directory than the one the sweep ran against.
    """
    expected_hash = compute_allowlist_hash(allowlist_dir)
    if expected_hash != header.allowlist_hash:
        raise SidecarCorruptError(
            f"--resume {header.run_id}: allowlist hash drift detected. "
            f"Sidecar recorded hash={header.allowlist_hash} but current "
            f"allowlist dir {allowlist_dir} hashes to {expected_hash}. "
            "The allowlist was edited between the original sweep and this "
            "resume. Either revert the allowlist edit or start a fresh "
            "sweep (the resumed entries would no longer correspond to the "
            "originally-judged findings)."
        )
    header_path = Path(header.allowlist_path)
    if header_path.resolve() != allowlist_dir.resolve():
        raise SidecarCorruptError(
            f"--resume {header.run_id}: --allowlist-dir {allowlist_dir} "
            f"does not match the directory the sweep ran against "
            f"({header.allowlist_path})."
        )
    if rule_filter != header.rule_filter:
        raise SidecarCorruptError(
            f"--resume {header.run_id}: --rule={rule_filter!r} does not match "
            f"the original sweep's rule={header.rule_filter!r}. Different "
            "filters produce different entry lists; resume requires identical "
            "filter arguments."
        )
    if since_iso != header.since_iso:
        raise SidecarCorruptError(
            f"--resume {header.run_id}: --since={since_iso!r} does not match the original sweep's --since={header.since_iso!r}."
        )
    if limit != header.limit:
        raise SidecarCorruptError(
            f"--resume {header.run_id}: --limit={limit!r} does not match the original sweep's --limit={header.limit!r}."
        )
    if include_pre_judge != header.include_pre_judge:
        raise SidecarCorruptError(
            f"--resume {header.run_id}: --include-pre-judge={include_pre_judge!r} "
            f"does not match the original sweep's value={header.include_pre_judge!r}."
        )


def filter_already_classified(
    entries: Sequence[AllowlistEntry],
    classified_keys: Iterable[str],
) -> list[AllowlistEntry]:
    """Return ``entries`` with already-classified keys removed, order preserved.

    Resume semantics: the original sweep classified some prefix of the
    filtered entry list. The remaining suffix is what needs to run.
    Filtering on key (rather than positional index) is robust to YAML
    iteration-order quirks across reloads — though if the allowlist
    hash matches, the order is also guaranteed identical.

    If an entry key appears multiple times in ``entries`` and ``classified_keys``,
    each occurrence in ``classified_keys`` consumes one match in
    ``entries``. This handles the legitimate case where the YAML has
    multiple ``allow_hits`` entries with the same canonical key (same
    finding, different owners) and the sweep classified some of them.
    """
    remaining_counts: dict[str, int] = {}
    for key in classified_keys:
        remaining_counts[key] = remaining_counts.get(key, 0) + 1
    result: list[AllowlistEntry] = []
    for entry in entries:
        if remaining_counts.get(entry.key, 0) > 0:
            remaining_counts[entry.key] -= 1
            continue
        result.append(entry)
    return result
