"""Landscape JSONL change journal for emergency backups.

Multi-worker ordering doctrine (ADR-030 §C.4 row 13, design line 284 / 501)
---------------------------------------------------------------------------
At N=1 the journal is a reliable backup stream: a single writer means
statement-time and WAL commit order are the same.

At N>1 (leader + followers) each worker writes its own
``db.journal.{worker_hex}.jsonl`` file (derived from the uuid4 hex tail of
its ``worker_id``; see :meth:`LandscapeDB._derive_journal_path`).  Per-worker
files fix the file-corruption half of the shared-file problem but NOT the
ordering half.

**The per-worker journal is FORENSIC-ONLY at N>1.**  Records carry
per-statement timestamps (:class:`JournalRecord` key ``"timestamp"``, buffered
to commit in :meth:`LandscapeJournal._after_commit`) which reflect
*statement-time*, not cross-process WAL commit order.  Two workers writing to
two files produce no shared total order.

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import Any, NotRequired, Protocol, TextIO, TypedDict, cast

import structlog
from sqlalchemy import event
from sqlalchemy.engine import Connection, Engine

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.payload_store import IntegrityError, PayloadNotFoundError
from elspeth.core.landscape._helpers import now
from elspeth.core.landscape.serialization import serialize_datetime
from elspeth.core.payload_store import FilesystemPayloadStore

logger = structlog.get_logger(__name__)

_BUFFER_STACK_KEY = "landscape_journal_buffer_stack"
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


class LandscapeJournal:
    """Append-only JSONL journal of committed database writes.

    Records SQL statements and parameters after a transaction commits.
    This is an emergency backup stream, not the canonical audit record.

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
        self._path = Path(path)
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

        self._ensure_owner_only_parent(self._path)

    def attach(self, engine: Engine) -> None:
        """Attach journal listeners to a SQLAlchemy engine.

        Listens to savepoint events in addition to commit/rollback so that
        writes inside rolled-back savepoints are discarded from the buffer
        rather than flushed on the outer commit.
        """
        event.listen(engine, "after_cursor_execute", self._after_cursor_execute)
        event.listen(engine, "commit", self._after_commit)
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

    def _after_commit(self, conn: Connection) -> None:
        if _BUFFER_STACK_KEY not in conn.info:
            return

        stack: list[list[JournalRecord]] = conn.info[_BUFFER_STACK_KEY]
        # Flatten the entire stack (shouldn't have depth > 1 at commit,
        # but be safe) and flush all committed records.
        all_records: list[JournalRecord] = []
        for buffer in stack:
            all_records.extend(buffer)
        stack.clear()
        stack.append([])  # Reset to single root buffer

        if all_records:
            if self._include_payloads:
                self._enrich_committed_records(all_records)
            self._append_records(all_records)

    def _enrich_committed_records(self, records: list[JournalRecord]) -> None:
        for record in records:
            self._enrich_with_payloads(record)
            record.pop("_payload_ref_columns", None)

    def _after_rollback(self, conn: Connection) -> None:
        if _BUFFER_STACK_KEY in conn.info:
            stack: list[list[JournalRecord]] = conn.info[_BUFFER_STACK_KEY]
            stack.clear()
            stack.append([])  # Reset to single root buffer

    # After this many consecutive failures, disable until next success
    _MAX_CONSECUTIVE_FAILURES = 5

    def _append_records(self, records: list[JournalRecord]) -> None:
        payload = "\n".join(self._serialize_record(record) for record in records) + "\n"
        with self._lock:
            if self._disabled:
                self._total_dropped += len(records)
                attempt_recovery = self._total_dropped % 100 == 0
                logger.warning(
                    "journal_recovery_attempt" if attempt_recovery else "journal_records_dropped",
                    event_type="journal_records_dropped",
                    consecutive_failures=self._consecutive_failures,
                    total_dropped=self._total_dropped,
                    batch_size=len(records),
                )
                if attempt_recovery:
                    self._disabled = False
                else:
                    return

            try:
                with self._open_owner_only_append() as handle:
                    handle.write(payload)
                if self._consecutive_failures > 0:
                    logger.info(
                        "journal_recovered",
                        consecutive_failures=self._consecutive_failures,
                        total_dropped=self._total_dropped,
                    )
                self._consecutive_failures = 0
            except OSError as exc:
                self._consecutive_failures += 1
                self._total_dropped += len(records)
                logger.error(
                    "journal_write_failed",
                    event_type="journal_records_dropped",
                    consecutive_failures=self._consecutive_failures,
                    max_failures=self._MAX_CONSECUTIVE_FAILURES,
                    records_dropped=len(records),
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
                continue
            directory.chmod(_JOURNAL_DIRECTORY_MODE)

    def _open_owner_only_append(self) -> TextIO:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | _JOURNAL_OPEN_SECURITY_FLAGS

        fd = os.open(self._path, flags, _JOURNAL_FILE_MODE)
        try:
            self._verify_owner_only_file(fd)
            return os.fdopen(fd, "a", encoding="utf-8")
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
