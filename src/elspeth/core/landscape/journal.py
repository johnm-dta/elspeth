"""Landscape JSONL change journal for emergency backups.

Multi-worker ordering doctrine (ADR-030 §C.4 row 13, design line 284 / 501)
---------------------------------------------------------------------------
At N=1 the journal is a reliable backup stream. Each transaction persists a
batch in ``sidecar_journal_outbox`` before DBAPI commit. Only a successful
commit makes that batch recoverable; publication fsyncs the JSONL file before
acknowledging the outbox row. Durable batch IDs make recovery idempotent if a
process stops between append and acknowledgement. Each batch is bound to a
canonical sidecar owner; drains for that owner are serialized across database
connections and never claim another sidecar's backlog.

At N>1 (leader + followers) each worker writes its own
``db.journal.{worker_hex}.jsonl`` file (derived from the uuid4 hex tail of
its ``worker_id``; see :meth:`LandscapeDB._derive_journal_path`).  Per-worker
files fix the file-corruption half of the shared-file problem but NOT the
ordering half.

**The per-worker journal is FORENSIC-ONLY at N>1.** Records carry
per-statement timestamps (:class:`JournalRecord` key ``"timestamp"``) plus
transaction-local batch identity, but two workers writing to two files still
produce no shared total order.

The **authoritative replay order** is ``run_coordination_events.seq``
(``AUTOINCREMENT`` — design G line 409): every coordination write is fenced
inside a ``BEGIN IMMEDIATE`` transaction and gets an incrementing sequence
number that reflects true WAL commit order across all processes.

Restore tooling **must gate** on single-worker provenance
(``worker_count == 1`` from ``run_workers``) before treating journal records
as an ordered replay log.  A true in-transaction ``journal_seq`` total order
across multiple workers is deferred to a future release.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, BinaryIO, NotRequired, Protocol, TextIO, TypedDict, cast
from uuid import uuid4

import structlog
from sqlalchemy import event
from sqlalchemy.engine import Connection, Engine

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.payload_store import IntegrityError, PayloadNotFoundError
from elspeth.core.landscape._helpers import now
from elspeth.core.landscape.schema import sidecar_journal_outbox_table
from elspeth.core.landscape.serialization import serialize_datetime
from elspeth.core.payload_store import FilesystemPayloadStore

logger = structlog.get_logger(__name__)

_BUFFER_STACK_KEY = "landscape_journal_buffer_stack"
_OUTBOX_WRITE_KEY = "landscape_journal_outbox_write"
_PENDING_BATCH_IDS_KEY = "landscape_journal_pending_batch_ids"
_JOURNAL_DIRECTORY_MODE = 0o700
_JOURNAL_FILE_MODE = 0o600
_NON_OWNER_PERMISSION_BITS = stat.S_IRWXG | stat.S_IRWXO
_JOURNAL_OPEN_SECURITY_FLAGS = 0 if os.name == "nt" else os.O_CLOEXEC | os.O_NOFOLLOW


class _NamedSqlTable(Protocol):
    name: str


class _TableBearingStatement(Protocol):
    table: _NamedSqlTable


class _JournalCompiledContext(Protocol):
    statement: object
    positiontup: Sequence[object] | None
    params: Mapping[str, object]


class _JournalExecutionContext(Protocol):
    compiled: _JournalCompiledContext | None


class PayloadInfo(TypedDict, total=False):
    """Payload enrichment data for calls table inserts."""

    request_ref: str | None
    request_payload: str | None
    response_ref: str | None
    response_payload: str | None
    request_payload_error: str
    response_payload_error: str


class JournalRecord(TypedDict):
    """A journal record capturing a SQL write operation and its parameters."""

    timestamp: str
    statement: str
    parameters: object
    executemany: bool
    # Payload enrichment (when include_payloads is enabled)
    payloads: NotRequired[list[PayloadInfo]]
    request_ref: NotRequired[str | None]
    request_payload: NotRequired[str | None]
    response_ref: NotRequired[str | None]
    response_payload: NotRequired[str | None]
    request_payload_error: NotRequired[str]
    response_payload_error: NotRequired[str]
    _payload_ref_columns: NotRequired[list[str]]
    journal_batch_id: NotRequired[str]
    journal_batch_ordinal: NotRequired[int]
    journal_batch_size: NotRequired[int]


class LandscapeJournal:
    """Append-only JSONL journal of committed database writes.

    Captured statements are inserted into ``sidecar_journal_outbox`` inside
    the transaction being committed. After DBAPI commit succeeds, committed
    batches are fsynced to this file and acknowledged in a new transaction.
    Startup recovery drains any committed-but-unacknowledged batches. This is
    an emergency backup stream, not the canonical audit record.

    At N=1 (single worker) the journal is a reliable ordered stream.
    At N>1 each worker writes its own per-worker file
    (``db.journal.{worker_hex}.jsonl``).  Per-worker files are FORENSIC-ONLY
    because statement timestamps are not cross-process WAL commit order.
    The authoritative replay order is ``run_coordination_events.seq``;
    restore tooling must gate on single-worker provenance.
    See the module docstring for the full multi-worker ordering doctrine.
    """

    def __init__(
        self,
        path: str,
        *,
        fail_on_error: bool,
        include_payloads: bool = False,
        payload_base_path: str | None = None,
    ) -> None:
        self._path = Path(path).resolve(strict=False)
        canonical_path = os.path.normcase(str(self._path))
        self._owner_key = sha256(canonical_path.encode("utf-8")).hexdigest()
        self._owner_lock_key = int.from_bytes(bytes.fromhex(self._owner_key)[:8], byteorder="big", signed=True)
        self._fail_on_error = fail_on_error
        self._include_payloads = include_payloads
        self._payload_store: FilesystemPayloadStore | None = None
        if include_payloads:
            if payload_base_path is None:
                raise ValueError("payload_base_path is required when include_payloads is enabled")
            self._payload_store = FilesystemPayloadStore(Path(payload_base_path))
        self._lock = Lock()
        self._disabled = False
        self._consecutive_failures = 0
        self._total_dropped = 0
        self._engine: Engine | None = None
        self._dialect_do_commit: Callable[[Any], None] | None = None
        self._dialect_name: str | None = None

        self._ensure_owner_only_parent(self._path)

    def attach(self, engine: Engine) -> None:
        """Attach journal listeners to a SQLAlchemy engine.

        The engine's schema must already define ``sidecar_journal_outbox``
        before any captured write commits. ``LandscapeDB`` provisions the
        table through its ordinary schema lifecycle; direct test/tool callers
        must provision the same metadata explicitly. ``attach`` never creates
        schema on an arbitrary engine.

        Listens to savepoint events in addition to commit/rollback so that
        writes inside rolled-back savepoints are discarded from the buffer
        rather than included in the outer transaction's outbox batch.
        """
        if self._engine is not None:
            raise RuntimeError("LandscapeJournal is already attached to an engine")
        self._engine = engine
        self._dialect_name = engine.dialect.name
        original_do_commit = engine.dialect.do_commit
        self._dialect_do_commit = original_do_commit

        def committed_do_commit(dbapi_connection: Any) -> None:
            try:
                original_do_commit(dbapi_connection)
            except BaseException:
                dbapi_connection.info.pop(_PENDING_BATCH_IDS_KEY, None)
                raise
            pending_batch_ids = tuple(dbapi_connection.info.pop(_PENDING_BATCH_IDS_KEY, ()))
            if pending_batch_ids:
                self._drain_committed_outbox(dbapi_connection, original_do_commit)

        engine.dialect.do_commit = committed_do_commit  # type: ignore[method-assign]
        event.listen(engine, "after_cursor_execute", self._after_cursor_execute)
        event.listen(engine, "commit", self._before_commit)
        event.listen(engine, "rollback", self._after_rollback)
        event.listen(engine, "savepoint", self._after_savepoint)
        event.listen(engine, "rollback_savepoint", self._after_rollback_savepoint)
        event.listen(engine, "release_savepoint", self._after_release_savepoint)

    def _ensure_buffer_stack(self, conn: Connection) -> list[list[JournalRecord]]:
        """Return the buffer stack for a connection, creating if needed.

        The stack always has at least one buffer (the root). Savepoint
        events push/pop additional buffers on top.
        """
        if _BUFFER_STACK_KEY not in conn.info:
            conn.info[_BUFFER_STACK_KEY] = [[]]
        stack: list[list[JournalRecord]] = conn.info[_BUFFER_STACK_KEY]
        return stack

    def _after_cursor_execute(
        self,
        conn: Connection,
        cursor: object,
        statement: str,
        parameters: Any,
        context: _JournalExecutionContext,
        executemany: bool,
    ) -> None:
        if conn.info.get(_OUTBOX_WRITE_KEY, False):
            return
        if not self._is_write_statement(statement):
            return

        record: JournalRecord = {
            "timestamp": now().isoformat(),
            "statement": statement,
            "parameters": self._normalize_parameters(parameters),
            "executemany": executemany,
        }
        if self._include_payloads:
            payload_ref_columns = self._payload_ref_columns_from_context(context, parameters)
            if payload_ref_columns is not None:
                record["_payload_ref_columns"] = payload_ref_columns

        stack = self._ensure_buffer_stack(conn)
        stack[-1].append(record)

    def _after_savepoint(self, conn: Connection, name: str) -> None:
        """Push a new buffer for the savepoint's scope."""
        stack = self._ensure_buffer_stack(conn)
        stack.append([])

    def _after_rollback_savepoint(self, conn: Connection, name: str, context: None) -> None:
        """Discard writes from the rolled-back savepoint."""
        stack = self._ensure_buffer_stack(conn)
        if len(stack) > 1:
            stack.pop()

    def _after_release_savepoint(self, conn: Connection, name: str, context: None) -> None:
        """Merge committed savepoint writes into the parent buffer."""
        stack = self._ensure_buffer_stack(conn)
        if len(stack) > 1:
            released = stack.pop()
            stack[-1].extend(released)

    def _take_buffered_records(self, conn: Connection) -> list[JournalRecord]:
        if _BUFFER_STACK_KEY not in conn.info:
            return []

        stack: list[list[JournalRecord]] = conn.info[_BUFFER_STACK_KEY]
        all_records: list[JournalRecord] = []
        for buffer in stack:
            all_records.extend(buffer)
        stack.clear()
        stack.append([])
        return all_records

    def _before_commit(self, conn: Connection) -> None:
        """Persist one journal batch inside the transaction being committed."""
        all_records = self._take_buffered_records(conn)
        if not all_records:
            return
        if self._include_payloads:
            self._enrich_committed_records(all_records)

        batch_id = uuid4().hex
        batch_size = len(all_records)
        for ordinal, record in enumerate(all_records):
            record["journal_batch_id"] = batch_id
            record["journal_batch_ordinal"] = ordinal
            record["journal_batch_size"] = batch_size
        records_json = "[" + ",".join(self._serialize_record(record) for record in all_records) + "]"

        conn.info[_OUTBOX_WRITE_KEY] = True
        try:
            conn.execute(
                sidecar_journal_outbox_table.insert().values(
                    batch_id=batch_id,
                    journal_owner=self._owner_key,
                    created_at=now(),
                    records_json=records_json,
                )
            )
        finally:
            conn.info.pop(_OUTBOX_WRITE_KEY, None)
        pending: list[str] = conn.info.setdefault(_PENDING_BATCH_IDS_KEY, [])
        pending.append(batch_id)

    def _enrich_committed_records(self, records: list[JournalRecord]) -> None:
        for record in records:
            self._enrich_with_payloads(record)
            record.pop("_payload_ref_columns", None)

    def _after_rollback(self, conn: Connection) -> None:
        if _BUFFER_STACK_KEY in conn.info:
            stack: list[list[JournalRecord]] = conn.info[_BUFFER_STACK_KEY]
            stack.clear()
            stack.append([])  # Reset to single root buffer
        conn.info.pop(_PENDING_BATCH_IDS_KEY, None)

    # After this many consecutive failures, disable until next success
    _MAX_CONSECUTIVE_FAILURES = 5

    def recover_pending(self, engine: Engine) -> None:
        """Publish committed outbox batches left by a prior failed drain."""
        if engine is not self._engine or self._dialect_do_commit is None:
            raise RuntimeError("LandscapeJournal must be attached to this engine before recovery")
        dbapi_connection = engine.raw_connection()
        try:
            self._drain_committed_outbox(
                dbapi_connection,
                self._dialect_do_commit,
                raise_on_publish_error=self._fail_on_error,
            )
        finally:
            dbapi_connection.close()

    def _drain_committed_outbox(
        self,
        dbapi_connection: Any,
        do_commit: Callable[[Any], None],
        *,
        raise_on_publish_error: bool = False,
    ) -> None:
        cursor = dbapi_connection.cursor()
        try:
            if self._dialect_name == "sqlite":
                # Acquire write ownership before the read snapshot. A deferred
                # SELECT followed by DELETE can otherwise fail with
                # SQLITE_BUSY_SNAPSHOT when another drain commits first.
                cursor.execute("BEGIN IMMEDIATE")
                placeholder = "?"
            elif self._dialect_name == "postgresql":
                # Serialize the complete select -> fsync -> acknowledge cycle
                # for one durable sidecar destination. SKIP LOCKED is not
                # sufficient: two drains could claim different batches and
                # append to the same file concurrently.
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", (self._owner_lock_key,))
                placeholder = "%s"
            else:
                raise RuntimeError(f"Unsupported journal outbox dialect: {self._dialect_name!r}")

            cursor.execute(
                f"SELECT sequence, batch_id, records_json FROM sidecar_journal_outbox "
                f"WHERE journal_owner = {placeholder} ORDER BY sequence",
                (self._owner_key,),
            )
            rows = cursor.fetchall()
            acknowledged: list[int] = []
            for row_index, (raw_sequence, raw_batch_id, raw_records_json) in enumerate(rows):
                sequence = int(raw_sequence)
                batch_id = str(raw_batch_id)
                records = self._deserialize_outbox_records(batch_id, str(raw_records_json))
                try:
                    published = self._append_committed_batch(
                        batch_id,
                        records,
                        allow_later_torn_tail=row_index < len(rows) - 1,
                    )
                except OSError as exc:
                    if raise_on_publish_error:
                        raise
                    logger.error(
                        "journal_outbox_publish_deferred",
                        event_type="journal_outbox_publish_deferred",
                        batch_id=batch_id,
                        error=str(exc),
                    )
                    break
                if not published:
                    break
                acknowledged.append(sequence)

            for sequence in acknowledged:
                cursor.execute(
                    f"DELETE FROM sidecar_journal_outbox WHERE sequence = {placeholder} AND journal_owner = {placeholder}",
                    (sequence, self._owner_key),
                )
                if cursor.rowcount != 1:
                    raise AuditIntegrityError(f"sidecar journal outbox batch sequence {sequence} lost drain ownership")
            try:
                do_commit(dbapi_connection)
            except BaseException as exc:
                dbapi_connection.rollback()
                logger.error(
                    "journal_outbox_ack_deferred",
                    event_type="journal_outbox_ack_deferred",
                    batch_count=len(acknowledged),
                    error=str(exc),
                )
                if raise_on_publish_error:
                    raise
        except BaseException:
            dbapi_connection.rollback()
            raise
        finally:
            cursor.close()

    @staticmethod
    def _deserialize_outbox_records(batch_id: str, records_json: str) -> list[JournalRecord]:
        try:
            raw_records = json.loads(records_json)
        except json.JSONDecodeError as exc:
            raise AuditIntegrityError(f"sidecar journal outbox batch {batch_id!r} is corrupt") from exc
        if type(raw_records) is not list or not raw_records:
            raise AuditIntegrityError(f"sidecar journal outbox batch {batch_id!r} has invalid records")
        records: list[JournalRecord] = []
        expected_size = len(raw_records)
        for ordinal, raw_record in enumerate(raw_records):
            if type(raw_record) is not dict:
                raise AuditIntegrityError(f"sidecar journal outbox batch {batch_id!r} has an invalid record")
            if (
                raw_record.get("journal_batch_id") != batch_id
                or raw_record.get("journal_batch_ordinal") != ordinal
                or raw_record.get("journal_batch_size") != expected_size
            ):
                raise AuditIntegrityError(f"sidecar journal outbox batch {batch_id!r} metadata is inconsistent")
            records.append(cast(JournalRecord, raw_record))
        return records

    def _append_committed_batch(
        self,
        batch_id: str,
        records: list[JournalRecord],
        *,
        allow_later_torn_tail: bool = False,
    ) -> bool:
        payload = "\n".join(self._serialize_record(record) for record in records) + "\n"
        with self._lock:
            if self._batch_is_fully_published(batch_id, records, allow_later_torn_tail=allow_later_torn_tail):
                # A previous attempt may have fsynced the file but failed while
                # publishing its directory entry. Retry that durability step
                # before the outbox row can be acknowledged.
                self._fsync_parent_directory()
                return True
            return self._append_payload_locked(payload, len(records))

    def _batch_is_fully_published(
        self,
        batch_id: str,
        records: list[JournalRecord],
        *,
        allow_later_torn_tail: bool = False,
    ) -> bool:
        """Return true for a complete batch, repairing only its torn tail.

        A recoverable tear must be an exact prefix of the expected outbox
        payload at EOF. Corrupt JSON elsewhere, changed record contents, and
        incomplete batches followed by other data remain integrity failures.
        """
        if not self._path.exists():
            return False

        expected_lines = [(self._serialize_record(record) + "\n").encode("utf-8") for record in records]
        expected_size = len(expected_lines)
        with self._open_owner_only_read_write() as handle:
            data = handle.read()
            offset = 0
            batch_start: int | None = None
            next_ordinal = 0
            batch_complete = False
            tail: tuple[int, bytes] | None = None
            for line_number, line in enumerate(data.splitlines(keepends=True), start=1):
                if not line.endswith(b"\n"):
                    tail = (offset, line)
                    break
                content = line[:-1]
                if not content.strip():
                    offset += len(line)
                    continue
                try:
                    record = json.loads(content)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise AuditIntegrityError(f"Landscape journal contains corrupt JSON at line {line_number}") from exc
                if type(record) is not dict or record.get("journal_batch_id") != batch_id:
                    if batch_start is not None and not batch_complete:
                        raise AuditIntegrityError(f"Landscape journal batch {batch_id!r} is partial before end of file")
                    offset += len(line)
                    continue
                ordinal = record.get("journal_batch_ordinal")
                if record.get("journal_batch_size") != expected_size or type(ordinal) is not int:
                    raise AuditIntegrityError(f"Landscape journal batch {batch_id!r} has inconsistent metadata")
                if batch_complete or ordinal != next_ordinal or ordinal >= expected_size or line != expected_lines[ordinal]:
                    raise AuditIntegrityError(f"Landscape journal batch {batch_id!r} has inconsistent content")
                if batch_start is None:
                    batch_start = offset
                next_ordinal += 1
                batch_complete = next_ordinal == expected_size
                offset += len(line)

            if batch_complete:
                if tail is not None and not allow_later_torn_tail:
                    raise AuditIntegrityError("Landscape journal contains corrupt JSON at final line")
                return True

            if batch_start is None:
                if tail is None:
                    return False
                tail_offset, tail_content = tail
                first_record = expected_lines[0][:-1]
                if not tail_content or not first_record.startswith(tail_content):
                    raise AuditIntegrityError("Landscape journal contains corrupt JSON at final line")
                batch_start = tail_offset
            elif tail is not None:
                _, tail_content = tail
                expected_tail = expected_lines[next_ordinal][:-1]
                if not tail_content or not expected_tail.startswith(tail_content):
                    raise AuditIntegrityError(f"Landscape journal batch {batch_id!r} has inconsistent torn tail")

            # The only incomplete content is an exact prefix of this batch at
            # EOF. Remove the whole attempted batch so replay can append it
            # atomically from the durable outbox payload.
            handle.seek(batch_start)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
            return False

    def _append_records(self, records: list[JournalRecord]) -> bool:
        payload = "\n".join(self._serialize_record(record) for record in records) + "\n"
        with self._lock:
            return self._append_payload_locked(payload, len(records))

    def _append_payload_locked(self, payload: str, record_count: int) -> bool:
        """Append one prepared payload while ``self._lock`` is held."""
        if self._disabled:
            self._total_dropped += record_count
            attempt_recovery = self._total_dropped % 100 == 0
            logger.warning(
                "journal_recovery_attempt" if attempt_recovery else "journal_records_dropped",
                event_type="journal_records_dropped",
                consecutive_failures=self._consecutive_failures,
                total_dropped=self._total_dropped,
                batch_size=record_count,
            )
            if attempt_recovery:
                self._disabled = False
            else:
                return False

        try:
            with self._open_owner_only_append() as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            self._fsync_parent_directory()
            if self._consecutive_failures > 0:
                logger.info(
                    "journal_recovered",
                    consecutive_failures=self._consecutive_failures,
                    total_dropped=self._total_dropped,
                )
            self._consecutive_failures = 0
            return True
        except OSError as exc:
            self._consecutive_failures += 1
            self._total_dropped += record_count
            logger.error(
                "journal_write_failed",
                event_type="journal_records_dropped",
                consecutive_failures=self._consecutive_failures,
                max_failures=self._MAX_CONSECUTIVE_FAILURES,
                records_dropped=record_count,
                total_dropped=self._total_dropped,
                error=str(exc),
            )
            if self._fail_on_error:
                raise
            if self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                logger.error(
                    "journal_disabled",
                    event_type="journal_records_dropped",
                    consecutive_failures=self._consecutive_failures,
                    total_dropped=self._total_dropped,
                )
                self._disabled = True
            return False

    def _fsync_parent_directory(self) -> None:
        """Durably publish a newly-created journal directory entry on POSIX."""
        if os.name == "nt":
            return
        directory_fd = os.open(self._path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _ensure_owner_only_parent(path: Path) -> None:
        if os.name == "nt":
            path.parent.mkdir(parents=True, exist_ok=True)
            return

        missing: list[Path] = []
        current = path.parent
        while not current.exists():
            missing.append(current)
            parent = current.parent
            if parent == current:
                break
            current = parent

        for directory in reversed(missing):
            try:
                directory.mkdir(mode=_JOURNAL_DIRECTORY_MODE)
            except FileExistsError:
                # Created concurrently between the exists() probe and mkdir.
                # Fall through: the owner-only invariant is verified below on
                # the create and race paths alike — a race-created ancestor
                # with lax permissions would let another user unlink/rename
                # the open journal file.
                pass
            else:
                directory.chmod(_JOURNAL_DIRECTORY_MODE)
            LandscapeJournal._verify_owner_only_dir(directory)

    def _open_owner_only_append(self) -> TextIO:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | _JOURNAL_OPEN_SECURITY_FLAGS

        fd = os.open(self._path, flags, _JOURNAL_FILE_MODE)
        try:
            self._verify_owner_only_file(fd)
            return os.fdopen(fd, "a", encoding="utf-8")
        except Exception:
            os.close(fd)
            raise

    def _open_owner_only_read(self) -> TextIO:
        flags = os.O_RDONLY | _JOURNAL_OPEN_SECURITY_FLAGS
        fd = os.open(self._path, flags)
        try:
            self._verify_owner_only_file(fd)
            return os.fdopen(fd, "r", encoding="utf-8")
        except Exception:
            os.close(fd)
            raise

    def _open_owner_only_read_write(self) -> BinaryIO:
        flags = os.O_RDWR | _JOURNAL_OPEN_SECURITY_FLAGS
        try:
            fd = os.open(self._path, flags)
        except IsADirectoryError as exc:
            raise OSError("Landscape journal path must be a regular file") from exc
        try:
            self._verify_owner_only_file(fd)
            return os.fdopen(fd, "r+b")
        except Exception:
            os.close(fd)
            raise

    @staticmethod
    def _verify_owner_only_file(fd: int) -> None:
        if os.name == "nt":
            return

        info = os.fstat(fd)
        mode = stat.S_IMODE(info.st_mode)
        if not stat.S_ISREG(info.st_mode):
            raise OSError("Landscape journal path must be a regular file")
        if info.st_uid != os.getuid():
            raise PermissionError("Landscape journal file must be owned by the current user")
        if mode & _NON_OWNER_PERMISSION_BITS:
            raise PermissionError("Landscape journal file must be owner-only before appending")

    @staticmethod
    def _verify_owner_only_dir(directory: Path) -> None:
        if os.name == "nt":
            return

        # lstat, not stat: a symlink planted at an ancestor position in the
        # mkdir race window must be rejected outright, never followed.
        info = os.lstat(directory)
        if not stat.S_ISDIR(info.st_mode):
            raise OSError("Landscape journal parent must be a directory")
        if info.st_uid != os.getuid():
            raise PermissionError("Landscape journal parent directory must be owned by the current user")
        if stat.S_IMODE(info.st_mode) & _NON_OWNER_PERMISSION_BITS:
            raise PermissionError("Landscape journal parent directory must be owner-only")

    @staticmethod
    def _serialize_record(record: JournalRecord) -> str:
        public_record = {key: value for key, value in record.items() if not key.startswith("_")}
        safe = serialize_datetime(public_record)
        try:
            return json.dumps(safe, allow_nan=False)
        except TypeError as exc:
            raise AuditIntegrityError(
                f"Journal record failed to serialize — non-JSON-serializable type in "
                f"SQL parameters (Tier 1 violation). Statement: "
                f"{record['statement']!r}. Error: {exc}"
            ) from exc

    @staticmethod
    def _normalize_parameters(parameters: Any) -> Any:
        if isinstance(parameters, list):
            return [LandscapeJournal._normalize_parameters(item) for item in parameters]
        if isinstance(parameters, tuple):
            return [LandscapeJournal._normalize_parameters(item) for item in parameters]
        if isinstance(parameters, dict):
            return {key: LandscapeJournal._normalize_parameters(value) for key, value in parameters.items()}
        return serialize_datetime(parameters)

    @staticmethod
    def _is_write_statement(statement: str) -> bool:
        sql = statement.lstrip().upper()
        return sql.startswith("INSERT") or sql.startswith("UPDATE") or sql.startswith("DELETE") or sql.startswith("REPLACE")

    def _payload_ref_columns_from_context(self, context: _JournalExecutionContext, parameters: Any) -> list[str] | None:
        try:
            compiled = context.compiled
        except AttributeError as exc:
            raise AuditIntegrityError("Landscape journal SQLAlchemy execution context is missing compiled metadata") from exc
        if compiled is None:
            return None
        statement = compiled.statement
        try:
            table = cast("_TableBearingStatement", statement).table
        except AttributeError:
            return None
        if table.name != "calls":
            return None
        positiontup = compiled.positiontup
        if positiontup:
            return [str(column) for column in positiontup]
        if isinstance(parameters, Mapping):
            return [str(column) for column in parameters]
        return [str(column) for column in compiled.params]

    def _enrich_with_payloads(self, record: JournalRecord) -> None:
        columns = record.get("_payload_ref_columns")
        if columns is None or self._payload_store is None:
            return
        if "request_ref" not in columns and "response_ref" not in columns:
            return

        if record["executemany"]:
            enrichments: list[PayloadInfo] = []
            for param_set in cast("list[object]", record["parameters"]):
                enrichments.append(self._payloads_for_params(columns, param_set))
            record["payloads"] = enrichments
        else:
            payload_dict = self._payloads_for_params(columns, record["parameters"])
            if "request_ref" in payload_dict:
                record["request_ref"] = payload_dict["request_ref"]
            if "request_payload" in payload_dict:
                record["request_payload"] = payload_dict["request_payload"]
            if "request_payload_error" in payload_dict:
                record["request_payload_error"] = payload_dict["request_payload_error"]
            if "response_ref" in payload_dict:
                record["response_ref"] = payload_dict["response_ref"]
            if "response_payload" in payload_dict:
                record["response_payload"] = payload_dict["response_payload"]
            if "response_payload_error" in payload_dict:
                record["response_payload_error"] = payload_dict["response_payload_error"]

    def _payloads_for_params(self, columns: list[str], params: Any) -> PayloadInfo:
        values = self._columns_to_values(columns, params)
        return self._payloads_for_values(values)

    def _payloads_for_values(self, values: Mapping[str, object]) -> PayloadInfo:
        result: PayloadInfo = {}

        if "request_ref" in values:
            request_ref = cast("str | None", values["request_ref"])
            request_payload, request_error = self._load_payload(request_ref)
            result["request_ref"] = request_ref
            result["request_payload"] = request_payload
            if request_error is not None:
                result["request_payload_error"] = request_error

        if "response_ref" in values:
            response_ref = cast("str | None", values["response_ref"])
            response_payload, response_error = self._load_payload(response_ref)
            result["response_ref"] = response_ref
            result["response_payload"] = response_payload
            if response_error is not None:
                result["response_payload_error"] = response_error

        return result

    def _load_payload(self, ref: str | None) -> tuple[str | None, str | None]:
        if ref is None:
            return None, None
        if self._payload_store is None:
            return None, "payload_store_not_configured"
        try:
            content = self._payload_store.retrieve(ref)
        except IntegrityError as exc:
            # Hash mismatch = corruption or tampering — Tier 1 violation.
            # Always crash regardless of _fail_on_error: payload integrity
            # failures are not operational issues, they are audit violations.
            raise AuditIntegrityError(
                f"Payload integrity check failed for ref={ref!r}: {exc}. This indicates data corruption or tampering in the payload store."
            ) from exc
        except (OSError, PayloadNotFoundError) as exc:
            logger.error("journal_payload_read_failed", error=str(exc), ref=ref)
            if self._fail_on_error:
                raise
            return None, f"payload_read_failed: {exc}"
        try:
            return content.decode("utf-8"), None
        except UnicodeDecodeError as exc:
            logger.error("journal_payload_decode_failed", error=str(exc), ref=ref)
            if self._fail_on_error:
                raise
            return None, f"payload_decode_failed: {exc}"

    @staticmethod
    def _columns_to_values(columns: list[str], params: Any) -> dict[str, object]:
        if isinstance(params, dict):
            return {col: params[col] for col in columns}
        return dict(zip(columns, params, strict=True))
