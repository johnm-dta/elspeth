"""Landscape JSONL change journal for emergency backups."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, NotRequired, TypedDict, cast

import structlog
from sqlalchemy import event
from sqlalchemy.engine import Connection, Engine

from elspeth.contracts.payload_store import IntegrityError, PayloadNotFoundError
from elspeth.core.landscape._helpers import now
from elspeth.core.landscape.formatters import serialize_datetime
from elspeth.core.payload_store import FilesystemPayloadStore

logger = structlog.get_logger(__name__)

_BUFFER_STACK_KEY = "landscape_journal_buffer_stack"


class PayloadInfo(TypedDict):
    """Payload enrichment data for calls table inserts."""

    request_ref: str | None
    request_payload: str | None
    response_ref: str | None
    response_payload: str | None
    request_payload_error: NotRequired[str]
    response_payload_error: NotRequired[str]


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


class LandscapeJournal:
    """Append-only JSONL journal of committed database writes.

    Records SQL statements and parameters after a transaction commits.
    This is an emergency backup stream, not the canonical audit record.
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

        self._path.parent.mkdir(parents=True, exist_ok=True)

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
        context: object,
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
            self._enrich_with_payloads(record, statement, parameters, executemany)

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
            self._append_records(all_records)

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
                with self._path.open("a", encoding="utf-8") as handle:
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
    def _serialize_record(record: JournalRecord) -> str:
        safe = serialize_datetime(record)
        try:
            return json.dumps(safe, allow_nan=False)
        except TypeError as exc:
            from elspeth.contracts.errors import AuditIntegrityError

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

    def _enrich_with_payloads(self, record: JournalRecord, statement: str, parameters: Any, executemany: bool) -> None:
        table, columns = self._parse_insert_statement(statement)
        base_table = table.split(".")[-1] if table else table
        if base_table != "calls" or columns is None or self._payload_store is None:
            return

        if executemany:
            enrichments: list[PayloadInfo] = []
            for param_set in parameters:
                enrichments.append(self._payloads_for_params(columns, param_set))
            record["payloads"] = enrichments
        else:
            payload_dict = self._payloads_for_params(columns, parameters)
            record["request_ref"] = payload_dict["request_ref"]
            record["request_payload"] = payload_dict["request_payload"]
            record["response_ref"] = payload_dict["response_ref"]
            record["response_payload"] = payload_dict["response_payload"]
            if "request_payload_error" in payload_dict:
                record["request_payload_error"] = payload_dict["request_payload_error"]
            if "response_payload_error" in payload_dict:
                record["response_payload_error"] = payload_dict["response_payload_error"]

    def _payloads_for_params(self, columns: list[str], params: Any) -> PayloadInfo:
        values = self._columns_to_values(columns, params)
        request_ref = cast("str | None", values["request_ref"])
        response_ref = cast("str | None", values["response_ref"])

        request_payload, request_error = self._load_payload(request_ref)
        response_payload, response_error = self._load_payload(response_ref)

        result: PayloadInfo = {
            "request_ref": request_ref,
            "request_payload": request_payload,
            "response_ref": response_ref,
            "response_payload": response_payload,
        }

        if request_error is not None:
            result["request_payload_error"] = request_error
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
            from elspeth.contracts.errors import AuditIntegrityError

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
    def _parse_insert_statement(statement: str) -> tuple[str | None, list[str] | None]:
        sql = statement.strip()
        upper = sql.upper()
        if not upper.startswith("INSERT INTO "):
            return None, None
        after_into = sql[len("INSERT INTO ") :]
        paren_index = after_into.find("(")
        if paren_index == -1:
            return None, None
        table = after_into[:paren_index].strip().strip('"').strip("'").lower()
        end_paren = after_into.find(")", paren_index)
        if end_paren == -1:
            return table, None
        columns_part = after_into[paren_index + 1 : end_paren]
        columns = [col.strip().strip('"').strip("'") for col in columns_part.split(",")]
        return table, columns

    @staticmethod
    def _columns_to_values(columns: list[str], params: Any) -> dict[str, object]:
        if isinstance(params, dict):
            return {col: params[col] for col in columns}
        return dict(zip(columns, params, strict=True))
