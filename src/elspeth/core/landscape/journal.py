"""Landscape JSONL change journal for emergency backups."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Connection, Engine

from elspeth.core.landscape._helpers import now
from elspeth.core.landscape.formatters import serialize_datetime
from elspeth.core.payload_store import FilesystemPayloadStore

logger = logging.getLogger(__name__)

_BUFFER_KEY = "landscape_journal_buffer"


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

        self._path.parent.mkdir(parents=True, exist_ok=True)

    def attach(self, engine: Engine) -> None:
        """Attach journal listeners to a SQLAlchemy engine."""
        event.listen(engine, "after_cursor_execute", self._after_cursor_execute)
        event.listen(engine, "commit", self._after_commit)
        event.listen(engine, "rollback", self._after_rollback)

    def _after_cursor_execute(
        self,
        conn: Connection,
        cursor: object,
        statement: str,
        parameters: Any,
        context: object,
        executemany: bool,
    ) -> None:
        if self._disabled:
            return
        if not self._is_write_statement(statement):
            return

        record = {
            "timestamp": now().isoformat(),
            "statement": statement,
            "parameters": self._normalize_parameters(parameters),
            "executemany": executemany,
        }
        if self._include_payloads:
            self._enrich_with_payloads(record, statement, parameters, executemany)

        if _BUFFER_KEY in conn.info:
            buffer = conn.info[_BUFFER_KEY]
        else:
            buffer = []
            conn.info[_BUFFER_KEY] = buffer

        buffer.append(record)

    def _after_commit(self, conn: Connection) -> None:
        if self._disabled:
            return
        if _BUFFER_KEY not in conn.info:
            return

        buffer = conn.info[_BUFFER_KEY]
        if not buffer:
            return

        self._append_records(buffer)
        buffer.clear()

    def _after_rollback(self, conn: Connection) -> None:
        if _BUFFER_KEY in conn.info:
            conn.info[_BUFFER_KEY].clear()

    def _append_records(self, records: list[dict[str, Any]]) -> None:
        payload = "\n".join(self._serialize_record(record) for record in records) + "\n"
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(payload)
            except Exception as exc:
                logger.error("Landscape journal write failed: %s", exc)
                if self._fail_on_error:
                    raise
                self._disabled = True

    @staticmethod
    def _serialize_record(record: dict[str, Any]) -> str:
        safe = serialize_datetime(record)
        return json.dumps(safe, default=str)

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

    def _enrich_with_payloads(self, record: dict[str, Any], statement: str, parameters: Any, executemany: bool) -> None:
        table, columns = self._parse_insert_statement(statement)
        if table != "calls" or columns is None or self._payload_store is None:
            return

        if executemany:
            payloads: list[dict[str, Any]] = []
            for param_set in parameters:
                payloads.append(self._payloads_for_params(columns, param_set))
            record["payloads"] = payloads
        else:
            payload_dict = self._payloads_for_params(columns, parameters)
            record.update(payload_dict)

    def _payloads_for_params(self, columns: list[str], params: Any) -> dict[str, Any]:
        values = self._columns_to_values(columns, params)
        request_ref = values["request_ref"]
        response_ref = values["response_ref"]

        request_payload, request_error = self._load_payload(request_ref)
        response_payload, response_error = self._load_payload(response_ref)

        payloads: dict[str, Any] = {
            "request_ref": request_ref,
            "request_payload": request_payload,
            "response_ref": response_ref,
            "response_payload": response_payload,
        }

        if request_error is not None:
            payloads["request_payload_error"] = request_error
        if response_error is not None:
            payloads["response_payload_error"] = response_error

        return payloads

    def _load_payload(self, ref: str | None) -> tuple[str | None, str | None]:
        if ref is None:
            return None, None
        if self._payload_store is None:
            return None, "payload_store_not_configured"
        try:
            content = self._payload_store.retrieve(ref)
        except Exception as exc:
            logger.error("Landscape journal payload read failed: %s", exc)
            if self._fail_on_error:
                raise
            return None, f"payload_read_failed: {exc}"
        try:
            return content.decode("utf-8"), None
        except UnicodeDecodeError as exc:
            logger.error("Landscape journal payload decode failed: %s", exc)
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
    def _columns_to_values(columns: list[str], params: Any) -> dict[str, Any]:
        if isinstance(params, dict):
            return {col: params[col] for col in columns}
        return dict(zip(columns, params, strict=True))
