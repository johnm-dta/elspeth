"""Tests for LandscapeJournal — append-only JSONL change journal.

Tests cover:
- Statement classification (_is_write_statement)
- Parameter normalization (_normalize_parameters)
- Record serialization (_serialize_record)
- Column-to-values mapping (_columns_to_values)
- SQLAlchemy event lifecycle (buffer → transactional outbox → commit/rollback)
- Failure circuit breaker with periodic recovery
- Payload enrichment for calls table
"""

from __future__ import annotations

import json
import os
import stat
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Lock, Thread
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, Table, create_engine, event, insert, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError as SQLAlchemyIntegrityError

from elspeth.contracts import CallStatus, CallType, NodeType
from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.journal import JournalRecord, LandscapeJournal
from elspeth.core.landscape.schema import sidecar_journal_outbox_table
from elspeth.core.payload_store import FilesystemPayloadStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_journal(
    tmp_path: Path,
    *,
    fail_on_error: bool = False,
    include_payloads: bool = False,
    payload_base_path: str | None = None,
) -> LandscapeJournal:
    """Create a journal pointed at a temp directory."""
    path = str(tmp_path / "journal.jsonl")
    return LandscapeJournal(
        path,
        fail_on_error=fail_on_error,
        include_payloads=include_payloads,
        payload_base_path=payload_base_path,
    )


class _ConnectionDouble:
    """Small SQLAlchemy connection double for journal event handlers."""

    def __init__(self, buffer: list[Any] | None = None) -> None:
        self.info: dict[str, Any] = {}
        if buffer is not None:
            self.info["landscape_journal_buffer_stack"] = [buffer]


class _PayloadStoreDouble:
    """Configurable payload store double for journal enrichment tests."""

    def __init__(self, *, content: bytes | None = None, error: BaseException | None = None) -> None:
        self.content = content
        self.error = error
        self.refs: list[str] = []

    def retrieve(self, ref: str) -> bytes:
        self.refs.append(ref)
        if self.error is not None:
            raise self.error
        if self.content is None:
            raise AssertionError("PayloadStoreDouble content was not configured")
        return self.content


class _EngineSentinel:
    pass


def _make_conn(buffer: list[Any] | None = None) -> _ConnectionDouble:
    """Create a SQLAlchemy Connection double with an info dict.

    When buffer is provided, it becomes the root buffer in the stack.
    """
    return _ConnectionDouble(buffer)


def _outbox_records(batch_id: str, *, size: int = 1) -> list[JournalRecord]:
    return [
        cast(
            JournalRecord,
            {
                "timestamp": "2026-01-15T12:00:00+00:00",
                "statement": "INSERT INTO rows (id) VALUES (?)",
                "parameters": [f"row-{ordinal}"],
                "executemany": False,
                "journal_batch_id": batch_id,
                "journal_batch_ordinal": ordinal,
                "journal_batch_size": size,
            },
        )
        for ordinal in range(size)
    ]


def _insert_outbox_batch(engine: Engine, journal: LandscapeJournal, batch_id: str, records: list[JournalRecord]) -> None:
    with engine.begin() as connection:
        connection.execute(
            sidecar_journal_outbox_table.insert().values(
                batch_id=batch_id,
                journal_owner=journal._owner_key,
                created_at=datetime.now(UTC),
                records_json=json.dumps(records),
            )
        )


class _ConcurrentProbeJournal(LandscapeJournal):
    """Force unfenced drains past their publication checks together."""

    def __init__(self, path: str, barrier: Barrier) -> None:
        super().__init__(path, fail_on_error=True)
        self._barrier = barrier
        self._probe_lock = Lock()
        self._probe_used = False

    def _append_payload_locked(self, payload: str, record_count: int) -> bool:
        with self._probe_lock:
            probe = not self._probe_used
            self._probe_used = True
        if probe:
            with suppress(BrokenBarrierError):
                self._barrier.wait(timeout=1)
        return super()._append_payload_locked(payload, record_count)


# ===========================================================================
# Statement classification
# ===========================================================================


class TestIsWriteStatement:
    """Tests for _is_write_statement — filters non-mutating SQL."""

    def test_insert_recognized(self) -> None:
        assert LandscapeJournal._is_write_statement("INSERT INTO rows (id) VALUES (?)")

    def test_update_recognized(self) -> None:
        assert LandscapeJournal._is_write_statement("UPDATE runs SET status = ?")

    def test_delete_recognized(self) -> None:
        assert LandscapeJournal._is_write_statement("DELETE FROM rows WHERE id = ?")

    def test_replace_recognized(self) -> None:
        assert LandscapeJournal._is_write_statement("REPLACE INTO rows (id) VALUES (?)")

    def test_select_rejected(self) -> None:
        assert not LandscapeJournal._is_write_statement("SELECT * FROM rows")

    def test_create_table_rejected(self) -> None:
        assert not LandscapeJournal._is_write_statement("CREATE TABLE foo (id INT)")

    def test_leading_whitespace_handled(self) -> None:
        assert LandscapeJournal._is_write_statement("   INSERT INTO rows (id) VALUES (?)")

    def test_case_insensitive(self) -> None:
        assert LandscapeJournal._is_write_statement("insert into rows (id) VALUES (?)")


# ===========================================================================
# Parameter normalization
# ===========================================================================


class TestNormalizeParameters:
    """Tests for _normalize_parameters — recursive type normalization."""

    def test_dict_params_normalized(self) -> None:
        result = LandscapeJournal._normalize_parameters({"a": 1, "b": "hello"})
        assert result == {"a": 1, "b": "hello"}

    def test_list_params_normalized(self) -> None:
        result = LandscapeJournal._normalize_parameters([1, "two", 3])
        assert result == [1, "two", 3]

    def test_tuple_params_converted_to_list(self) -> None:
        result = LandscapeJournal._normalize_parameters((1, "two", 3))
        assert result == [1, "two", 3]

    def test_datetime_serialized(self) -> None:
        dt = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        result = LandscapeJournal._normalize_parameters(dt)
        assert result == "2026-01-15T12:00:00+00:00"

    def test_nested_dict_in_list(self) -> None:
        dt = datetime(2026, 1, 15, tzinfo=UTC)
        result = LandscapeJournal._normalize_parameters([{"ts": dt}])
        assert result == [{"ts": "2026-01-15T00:00:00+00:00"}]

    def test_scalar_passes_through(self) -> None:
        assert LandscapeJournal._normalize_parameters(42) == 42
        assert LandscapeJournal._normalize_parameters("hello") == "hello"
        assert LandscapeJournal._normalize_parameters(None) is None


# ===========================================================================
# Record serialization
# ===========================================================================


class TestSerializeRecord:
    """Tests for _serialize_record — JSON serialization with datetime handling."""

    def test_produces_valid_json(self) -> None:
        record = cast(JournalRecord, {"timestamp": "2026-01-15T12:00:00", "statement": "INSERT", "parameters": {}, "executemany": False})
        result = LandscapeJournal._serialize_record(record)
        parsed = json.loads(result)
        assert parsed["statement"] == "INSERT"

    def test_datetime_values_serialized(self) -> None:
        dt = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        record = cast(JournalRecord, {"timestamp": dt, "statement": "INSERT", "parameters": {}, "executemany": False})
        result = LandscapeJournal._serialize_record(record)
        parsed = json.loads(result)
        assert parsed["timestamp"] == "2026-01-15T12:00:00+00:00"

    def test_non_serializable_type_raises(self) -> None:
        """Non-JSON types must crash with AuditIntegrityError, not silently convert via str()."""
        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {b"bytes": "value"}, "executemany": False})
        with pytest.raises(AuditIntegrityError, match="Tier 1 violation"):
            LandscapeJournal._serialize_record(record)

    def test_nan_rejected(self) -> None:
        """NaN in journal data must be rejected — audit integrity."""
        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {"val": float("nan")}, "executemany": False})
        with pytest.raises(AuditIntegrityError, match="NaN"):
            LandscapeJournal._serialize_record(record)

    def test_set_type_raises(self) -> None:
        """Sets are not JSON-serializable — must crash with AuditIntegrityError."""
        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {"ids": {1, 2, 3}}, "executemany": False})
        with pytest.raises(AuditIntegrityError, match="Tier 1 violation"):
            LandscapeJournal._serialize_record(record)

    def test_internal_payload_ref_columns_not_serialized(self) -> None:
        record = cast(
            JournalRecord,
            {
                "timestamp": "t",
                "statement": "INSERT",
                "parameters": {},
                "executemany": False,
                "_payload_ref_columns": ["request_ref"],
            },
        )
        parsed = json.loads(LandscapeJournal._serialize_record(record))
        assert "_payload_ref_columns" not in parsed


# ===========================================================================
# Columns to values mapping
# ===========================================================================


class TestColumnsToValues:
    """Tests for _columns_to_values — maps column names to parameter values."""

    def test_dict_params(self) -> None:
        result = LandscapeJournal._columns_to_values(["call_id", "state_id"], {"call_id": "c1", "state_id": "s1", "extra": "ignored"})
        assert result == {"call_id": "c1", "state_id": "s1"}

    def test_positional_params(self) -> None:
        result = LandscapeJournal._columns_to_values(["call_id", "state_id"], ("c1", "s1"))
        assert result == {"call_id": "c1", "state_id": "s1"}

    def test_list_params(self) -> None:
        result = LandscapeJournal._columns_to_values(["a", "b"], ["v1", "v2"])
        assert result == {"a": "v1", "b": "v2"}


# ===========================================================================
# Constructor
# ===========================================================================


class TestConstructor:
    """Tests for journal initialization."""

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        LandscapeJournal(str(nested / "journal.jsonl"), fail_on_error=False)
        assert nested.exists()

    def test_creates_missing_parent_directories_owner_only(self, tmp_path: Path) -> None:
        old_umask = os.umask(0)
        try:
            nested = tmp_path / "deep" / "nested"
            LandscapeJournal(str(nested / "journal.jsonl"), fail_on_error=False)
        finally:
            os.umask(old_umask)

        created_dirs = [tmp_path / "deep", tmp_path / "deep" / "nested"]
        for created_dir in created_dirs:
            mode = stat.S_IMODE(created_dir.stat().st_mode)
            assert mode & 0o077 == 0

    def test_include_payloads_requires_base_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="payload_base_path is required"):
            LandscapeJournal(
                str(tmp_path / "journal.jsonl"),
                fail_on_error=False,
                include_payloads=True,
                payload_base_path=None,
            )

    def test_include_payloads_with_base_path_creates_store(self, tmp_path: Path) -> None:
        journal = LandscapeJournal(
            str(tmp_path / "journal.jsonl"),
            fail_on_error=False,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        assert journal._payload_store is not None


# ===========================================================================
# SQLAlchemy event lifecycle
# ===========================================================================


class TestAfterCursorExecute:
    """Tests for _after_cursor_execute — buffers write statements."""

    def test_write_statement_buffered(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        conn = _make_conn()

        journal._after_cursor_execute(
            conn,
            cursor=None,
            statement="INSERT INTO rows (id) VALUES (?)",
            parameters={"id": "r1"},
            context=None,
            executemany=False,
        )

        stack = conn.info["landscape_journal_buffer_stack"]
        assert len(stack) == 1  # single root buffer
        assert len(stack[0]) == 1
        assert stack[0][0]["statement"] == "INSERT INTO rows (id) VALUES (?)"

    def test_select_not_buffered(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        conn = _make_conn()

        journal._after_cursor_execute(
            conn,
            cursor=None,
            statement="SELECT * FROM rows",
            parameters={},
            context=None,
            executemany=False,
        )

        assert "landscape_journal_buffer_stack" not in conn.info

    def test_disabled_journal_still_buffers_for_recovery(self, tmp_path: Path) -> None:
        """When disabled, records still buffer so _append_records can attempt recovery."""
        journal = _make_journal(tmp_path)
        journal._disabled = True
        conn = _make_conn()

        journal._after_cursor_execute(
            conn,
            cursor=None,
            statement="INSERT INTO rows (id) VALUES (?)",
            parameters={},
            context=None,
            executemany=False,
        )

        assert "landscape_journal_buffer_stack" in conn.info
        assert len(conn.info["landscape_journal_buffer_stack"][0]) == 1

    def test_appends_to_existing_buffer(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        existing_buffer: list[Any] = []
        conn = _make_conn(buffer=existing_buffer)

        journal._after_cursor_execute(
            conn,
            cursor=None,
            statement="INSERT INTO rows (id) VALUES (?)",
            parameters={"id": "r1"},
            context=None,
            executemany=False,
        )

        assert len(existing_buffer) == 1

    def test_payload_hydration_deferred_until_commit(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        payload_store = _PayloadStoreDouble(content=b"payload content")
        journal._payload_store = payload_store
        engine = create_engine("sqlite:///:memory:")
        metadata = MetaData()
        calls = Table(
            "calls",
            metadata,
            Column("call_id", String),
            Column("request_ref", String),
            Column("response_ref", String),
        )
        metadata.create_all(engine)
        sidecar_journal_outbox_table.create(engine)
        journal.attach(engine)

        with engine.connect() as conn:
            transaction = conn.begin()
            conn.execute(insert(calls).values(call_id="c1"))
            conn.execute(update(calls).where(calls.c.call_id == "c1").values(request_ref="req-ref", response_ref="resp-ref"))

            assert payload_store.refs == []

            transaction.commit()

        assert payload_store.refs == ["req-ref", "resp-ref"]
        records = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
        call_updates = [record for record in records if record["statement"].lstrip().upper().startswith("UPDATE CALLS SET")]
        assert call_updates[0]["request_payload"] == "payload content"

    def test_payload_enrichment_uses_structured_sqlalchemy_context_not_sql_parser(self, tmp_path: Path) -> None:
        assert not hasattr(LandscapeJournal, "_parse_insert_statement")
        assert not hasattr(LandscapeJournal, "_parse_update_statement")

        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = _PayloadStoreDouble(content=b"payload content")
        engine = create_engine("sqlite:///:memory:")
        metadata = MetaData()
        calls = Table(
            "calls",
            metadata,
            Column("call_id", String),
            Column("request_ref", String),
            Column("response_ref", String),
        )
        metadata.create_all(engine)
        sidecar_journal_outbox_table.create(engine)

        journal.attach(engine)
        with engine.begin() as conn:
            conn.execute(insert(calls).values(call_id="c1"))
            conn.execute(update(calls).where(calls.c.call_id == "c1").values(request_ref="req-ref", response_ref="resp-ref"))

        records = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
        call_updates = [record for record in records if record["statement"].lstrip().upper().startswith("UPDATE CALLS SET")]
        assert call_updates[0]["request_ref"] == "req-ref"
        assert call_updates[0]["request_payload"] == "payload content"


class TestBeforeCommitOutbox:
    """Tests for the transaction-owned batch prepared before DBAPI commit."""

    def test_persists_buffer_as_outbox_batch(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        record = {
            "timestamp": "2026-01-15T12:00:00",
            "statement": "INSERT INTO rows (id) VALUES (?)",
            "parameters": {"id": "r1"},
            "executemany": False,
        }
        engine = create_engine("sqlite:///:memory:")
        sidecar_journal_outbox_table.create(engine)
        with engine.begin() as conn:
            conn.info["landscape_journal_buffer_stack"] = [[record]]
            journal._before_commit(conn)
            row = conn.execute(select(sidecar_journal_outbox_table)).one()
            persisted = json.loads(row.records_json)
            assert persisted[0]["statement"] == "INSERT INTO rows (id) VALUES (?)"
            assert persisted[0]["journal_batch_id"] == row.batch_id
            assert persisted[0]["journal_batch_ordinal"] == 0
            assert persisted[0]["journal_batch_size"] == 1
        engine.dispose()

    def test_clears_buffer_after_outbox_prepare(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        buffer: list[Any] = [{"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False}]
        engine = create_engine("sqlite:///:memory:")
        sidecar_journal_outbox_table.create(engine)
        with engine.begin() as conn:
            conn.info["landscape_journal_buffer_stack"] = [buffer]
            journal._before_commit(conn)
            assert conn.info["landscape_journal_buffer_stack"] == [[]]
        engine.dispose()

    def test_no_buffer_is_noop(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        conn = _make_conn()

        journal._before_commit(cast(Any, conn))

        journal_path = tmp_path / "journal.jsonl"
        assert not journal_path.exists()

    def test_empty_buffer_is_noop(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        conn = _make_conn(buffer=[])

        journal._before_commit(cast(Any, conn))

        journal_path = tmp_path / "journal.jsonl"
        assert not journal_path.exists()

    def test_disabled_sidecar_still_persists_transactional_outbox(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        journal._disabled = True
        buffer = [{"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False}]
        engine = create_engine("sqlite:///:memory:")
        sidecar_journal_outbox_table.create(engine)
        with engine.begin() as conn:
            conn.info["landscape_journal_buffer_stack"] = [buffer]
            journal._before_commit(conn)
            assert len(conn.execute(select(sidecar_journal_outbox_table)).all()) == 1
        assert journal._total_dropped == 0
        engine.dispose()


class TestAfterRollback:
    """Tests for _after_rollback — discards buffered writes."""

    def test_clears_buffer(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        buffer: list[Any] = [{"statement": "INSERT"}]
        conn = _make_conn(buffer=buffer)

        journal._after_rollback(conn)

        # After rollback the stack is reset to a single empty root buffer
        stack = conn.info["landscape_journal_buffer_stack"]
        assert stack == [[]]

    def test_no_buffer_is_noop(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        conn = _make_conn()

        journal._after_rollback(conn)  # Should not raise


# ===========================================================================
# Failure circuit breaker
# ===========================================================================


class TestAppendRecordsFailureHandling:
    """Tests for _append_records — circuit breaker after consecutive failures."""

    def test_owner_only_open_does_not_probe_os_flags_with_hasattr(self) -> None:
        import inspect

        assert "hasattr(" not in inspect.getsource(LandscapeJournal._open_owner_only_append)
        assert "hasattr(" not in inspect.getsource(LandscapeJournal._verify_owner_only_file)

    def test_creates_journal_file_owner_only(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path, fail_on_error=True)
        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False})

        old_umask = os.umask(0)
        try:
            journal._append_records([record])
        finally:
            os.umask(old_umask)

        journal_path = tmp_path / "journal.jsonl"
        mode = stat.S_IMODE(journal_path.stat().st_mode)
        assert mode & 0o077 == 0

    @pytest.mark.skipif(os.name == "nt", reason="directory fsync is a POSIX durability boundary")
    def test_append_fsyncs_file_and_parent_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        journal = _make_journal(tmp_path, fail_on_error=True)
        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False})
        real_fsync = os.fsync
        fsync_targets: list[str] = []

        def recording_fsync(fd: int) -> None:
            mode = os.fstat(fd).st_mode
            fsync_targets.append("directory" if stat.S_ISDIR(mode) else "file")
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", recording_fsync)

        journal._append_records([record])

        assert fsync_targets == ["file", "directory"]

    def test_rejects_existing_group_readable_journal_file(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        journal_path.write_text("", encoding="utf-8")
        journal_path.chmod(0o640)
        journal = _make_journal(tmp_path, fail_on_error=True)
        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False})

        with pytest.raises(PermissionError, match="owner-only"):
            journal._append_records([record])
        assert journal_path.read_text(encoding="utf-8") == ""

    def test_fail_on_error_raises(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path, fail_on_error=True)
        # Make path a directory to cause write failure
        journal_path = tmp_path / "journal.jsonl"
        journal_path.mkdir()

        with pytest.raises(IsADirectoryError):
            journal._append_records(
                [cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False})]
            )

    def test_consecutive_failures_disable_journal(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        # Make path a directory to cause write failure
        journal_path = tmp_path / "journal.jsonl"
        journal_path.mkdir()

        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False})
        for _ in range(5):
            journal._append_records([record])

        assert journal._disabled is True
        assert journal._consecutive_failures == 5

    def test_recovery_after_100_dropped_records(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        journal._disabled = True
        journal._consecutive_failures = 5
        journal._total_dropped = 99  # Next drop will be 100th

        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False})
        # This call should trigger recovery attempt (total_dropped hits 100)
        journal._append_records([record])

        # Recovery succeeded (path is writable now) so disabled should be False
        assert journal._disabled is False
        assert journal._consecutive_failures == 0

    def test_successful_write_resets_failure_count(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        journal._consecutive_failures = 3

        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False})
        journal._append_records([record])

        assert journal._consecutive_failures == 0

    def test_programming_error_not_caught(self, tmp_path: Path) -> None:
        """AuditIntegrityError from serialization must crash, not be silently swallowed."""
        journal = _make_journal(tmp_path)
        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False})

        with (
            patch.object(journal, "_serialize_record", side_effect=AuditIntegrityError("bad serialize")),
            pytest.raises(AuditIntegrityError, match="bad serialize"),
        ):
            journal._append_records([record])

    def test_disabled_drop_always_logs(self, tmp_path: Path) -> None:
        """Every drop in disabled state must be logged, not just every 100th."""
        journal = _make_journal(tmp_path)
        journal._disabled = True
        journal._consecutive_failures = 5
        journal._total_dropped = 5  # Not a multiple of 100

        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False})
        with patch("elspeth.core.landscape.journal.logger") as mock_logger:
            journal._append_records([record])

        mock_logger.warning.assert_called_once()
        call_kwargs = mock_logger.warning.call_args
        assert "journal_records_dropped" in str(call_kwargs)


# ===========================================================================
# Attach
# ===========================================================================


class TestAttach:
    """Tests for attach — registers SQLAlchemy event listeners."""

    def test_registers_six_listeners(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        engine = create_engine("sqlite:///:memory:")

        with patch("elspeth.core.landscape.journal.event") as mock_event:
            journal.attach(engine)

            assert mock_event.listen.call_count == 6
            calls = [c.args for c in mock_event.listen.call_args_list]
            event_names = {c[1] for c in calls}
            assert event_names == {
                "after_cursor_execute",
                "commit",
                "rollback",
                "savepoint",
                "rollback_savepoint",
                "release_savepoint",
            }
        engine.dispose()


# ===========================================================================
# Payload enrichment
# ===========================================================================


class TestLoadPayload:
    """Tests for _load_payload — reads payloads from the store."""

    def test_none_ref_returns_none(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        content, error = journal._load_payload(None)
        assert content is None
        assert error is None

    def test_no_payload_store_returns_error(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        journal._payload_store = None
        content, error = journal._load_payload("some-ref")
        assert content is None
        assert error == "payload_store_not_configured"

    def test_successful_read(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = _PayloadStoreDouble(content=b'{"key": "value"}')

        content, error = journal._load_payload("some-ref")
        assert content == '{"key": "value"}'
        assert error is None

    def test_read_failure_missing_blob_returns_error(self, tmp_path: Path) -> None:
        from elspeth.contracts.payload_store import PayloadNotFoundError

        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = _PayloadStoreDouble(error=PayloadNotFoundError("deadbeef" * 8))

        content, error = journal._load_payload("some-ref")
        assert content is None
        assert error is not None
        assert "payload_read_failed" in error

    def test_read_failure_os_error_returns_error(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = _PayloadStoreDouble(error=OSError("disk failure"))

        content, error = journal._load_payload("some-ref")
        assert content is None
        assert error is not None
        assert "payload_read_failed" in error

    def test_read_failure_missing_blob_with_fail_on_error_raises(self, tmp_path: Path) -> None:
        from elspeth.contracts.payload_store import PayloadNotFoundError

        journal = _make_journal(
            tmp_path,
            fail_on_error=True,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = _PayloadStoreDouble(error=PayloadNotFoundError("deadbeef" * 8))

        with pytest.raises(PayloadNotFoundError):
            journal._load_payload("some-ref")

    def test_read_failure_os_error_with_fail_on_error_raises(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            fail_on_error=True,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = _PayloadStoreDouble(error=OSError("disk failure"))

        with pytest.raises(OSError):
            journal._load_payload("some-ref")

    def test_programming_error_in_retrieve_not_caught(self, tmp_path: Path) -> None:
        """TypeError/AttributeError in payload store must crash, not be silently swallowed."""
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = _PayloadStoreDouble(error=TypeError("bad type in store"))

        with pytest.raises(TypeError, match="bad type in store"):
            journal._load_payload("some-ref")

    def test_integrity_error_always_crashes_as_audit_violation(self, tmp_path: Path) -> None:
        """IntegrityError (hash mismatch) must always crash as AuditIntegrityError.

        Payload integrity failures indicate corruption or tampering — Tier 1
        violations that must never be silently swallowed, regardless of
        _fail_on_error setting. A pipeline that continues past a hash
        mismatch would be operating on potentially tampered data.
        """
        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.contracts.payload_store import IntegrityError

        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = _PayloadStoreDouble(error=IntegrityError("expected abc123, got def456"))

        with pytest.raises(AuditIntegrityError, match="corruption or tampering"):
            journal._load_payload("some-ref")

    def test_integrity_error_with_fail_on_error_also_crashes(self, tmp_path: Path) -> None:
        """IntegrityError crashes as AuditIntegrityError even with fail_on_error=True.

        _fail_on_error only controls OSError/PayloadNotFoundError behavior.
        IntegrityError always crashes regardless.
        """
        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.contracts.payload_store import IntegrityError

        journal = _make_journal(
            tmp_path,
            fail_on_error=True,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = _PayloadStoreDouble(error=IntegrityError("expected abc123, got def456"))

        with pytest.raises(AuditIntegrityError, match="corruption or tampering"):
            journal._load_payload("some-ref")

    def test_decode_failure_returns_error(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = _PayloadStoreDouble(content=b"\x80\x81\x82")  # Invalid UTF-8

        content, error = journal._load_payload("some-ref")
        assert content is None
        assert error is not None
        assert "payload_decode_failed" in error


class TestEnrichWithPayloads:
    """Tests for _enrich_with_payloads — adds payload data to call records."""

    def test_malformed_sqlalchemy_context_crashes_when_payload_columns_requested(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )

        with pytest.raises(AuditIntegrityError, match="compiled metadata"):
            journal._payload_ref_columns_from_context(object(), {})

    def test_raw_sql_compiled_context_without_structured_table_is_skipped(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        compiled = SimpleNamespace(statement=object(), positiontup=("request_ref",), params={"request_ref": "req"})
        context = SimpleNamespace(compiled=compiled)

        assert journal._payload_ref_columns_from_context(context, {}) is None

    def test_records_without_structured_payload_columns_skipped(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False})
        journal._enrich_with_payloads(record)
        # No payload keys should be added
        assert "request_ref" not in record
        assert "payloads" not in record

    def test_single_call_enriched(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = _PayloadStoreDouble(content=b"payload content")

        record = cast(
            JournalRecord,
            {
                "timestamp": "t",
                "statement": "INSERT",
                "parameters": {"call_id": "c1", "request_ref": "req-ref", "response_ref": "resp-ref"},
                "executemany": False,
                "_payload_ref_columns": ["call_id", "request_ref", "response_ref"],
            },
        )
        journal._enrich_with_payloads(record)

        assert record["request_ref"] == "req-ref"
        assert record["request_payload"] == "payload content"
        assert record["response_ref"] == "resp-ref"

    def test_executemany_enriched_as_list(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = _PayloadStoreDouble(content=b"payload")

        record = cast(
            JournalRecord,
            {
                "timestamp": "t",
                "statement": "INSERT",
                "parameters": [
                    {"call_id": "c1", "request_ref": "r1", "response_ref": "r2"},
                    {"call_id": "c2", "request_ref": "r3", "response_ref": None},
                ],
                "executemany": True,
                "_payload_ref_columns": ["call_id", "request_ref", "response_ref"],
            },
        )
        journal._enrich_with_payloads(record)

        assert "payloads" in record
        assert len(record["payloads"]) == 2


class TestPayloadEnrichmentProductionPath:
    """Regression tests for payload inlining through recorder-owned writes."""

    def test_record_call_auto_persist_update_records_inline_payloads(self, tmp_path: Path) -> None:
        """Auto-persisted call refs are written by UPDATE and must be journal-enriched."""
        journal_path = tmp_path / "journal.jsonl"
        payload_dir = tmp_path / "payloads"
        db_path = tmp_path / "audit.db"
        db = LandscapeDB.from_url(
            f"sqlite:///{db_path}",
            dump_to_jsonl=True,
            dump_to_jsonl_path=str(journal_path),
            dump_to_jsonl_include_payloads=True,
            dump_to_jsonl_payload_base_path=str(payload_dir),
        )
        payload_store = FilesystemPayloadStore(payload_dir)
        factory = RecorderFactory(db, payload_store=payload_store)

        schema = SchemaConfig.from_dict({"mode": "observed"})
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="llm_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"input": "test"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={"input": "test"},
        )

        factory.execution.record_call(
            state_id=state.state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data=RawCallPayload({"model": "gpt-4", "prompt": "Hi"}),
            response_data=RawCallPayload({"content": "Hello!"}),
        )

        records = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
        call_updates = [record for record in records if record["statement"].lstrip().upper().startswith("UPDATE CALLS SET")]

        assert any(record.get("request_payload") == '{"model":"gpt-4","prompt":"Hi"}' for record in call_updates)
        assert any(record.get("response_payload") == '{"content":"Hello!"}' for record in call_updates)


# ===========================================================================
# End-to-end: cursor → commit → file
# ===========================================================================


class TestEndToEnd:
    """Integration-style tests for the full event lifecycle."""

    def test_cursor_then_commit_writes_file(self, tmp_path: Path) -> None:
        """Full flow: cursor → transactional outbox → DBAPI commit → JSONL."""
        journal = _make_journal(tmp_path)
        engine = create_engine("sqlite:///:memory:")
        metadata = MetaData()
        rows = Table("rows", metadata, Column("id", String, primary_key=True))
        metadata.create_all(engine)
        sidecar_journal_outbox_table.create(engine)
        journal.attach(engine)
        with engine.begin() as conn:
            conn.execute(rows.insert().values(id="row-1"))

        journal_path = tmp_path / "journal.jsonl"
        lines = journal_path.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["parameters"] == ["row-1"]
        assert parsed["journal_batch_ordinal"] == 0
        assert parsed["journal_batch_size"] == 1
        with engine.connect() as conn:
            assert conn.execute(select(sidecar_journal_outbox_table)).all() == []
        engine.dispose()

    def test_cursor_then_rollback_discards(self, tmp_path: Path) -> None:
        """Rollback discards buffered records."""
        journal = _make_journal(tmp_path)
        conn = _make_conn()

        journal._after_cursor_execute(
            conn,
            cursor=None,
            statement="INSERT INTO rows (id) VALUES (?)",
            parameters={"id": "row-1"},
            context=None,
            executemany=False,
        )
        journal._after_rollback(conn)

        journal_path = tmp_path / "journal.jsonl"
        assert not journal_path.exists()

    def test_multiple_statements_single_commit(self, tmp_path: Path) -> None:
        """Multiple buffered statements flush as separate JSONL lines."""
        journal = _make_journal(tmp_path)
        engine = create_engine("sqlite:///:memory:")
        metadata = MetaData()
        rows = Table("rows", metadata, Column("id", String, primary_key=True))
        metadata.create_all(engine)
        sidecar_journal_outbox_table.create(engine)
        journal.attach(engine)
        with engine.begin() as conn:
            for i in range(3):
                conn.execute(rows.insert().values(id=f"row-{i}"))

        journal_path = tmp_path / "journal.jsonl"
        lines = journal_path.read_text().strip().split("\n")
        assert len(lines) == 3
        batch_ids = {json.loads(line)["journal_batch_id"] for line in lines}
        assert len(batch_ids) == 1
        engine.dispose()


class TestCommitFailureDurability:
    """The recovery-visible sidecar must never get ahead of the database."""

    def test_sidecar_owner_key_is_stable_for_equivalent_paths(self, tmp_path: Path) -> None:
        canonical = LandscapeJournal(str(tmp_path / "journal.jsonl"), fail_on_error=True)
        equivalent = LandscapeJournal(str(tmp_path / "missing" / ".." / "journal.jsonl"), fail_on_error=True)

        assert canonical._owner_key == equivalent._owner_key

    def test_sqlite_deferred_constraint_commit_failure_publishes_no_records(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        engine = create_engine(f"sqlite:///{tmp_path / 'commit-failure.db'}")

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection: Any, _connection_record: object) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.close()

        metadata = MetaData()
        parent = Table("journal_parent", metadata, Column("id", Integer, primary_key=True))
        child = Table(
            "journal_child",
            metadata,
            Column("id", Integer, primary_key=True),
            Column(
                "parent_id",
                Integer,
                ForeignKey(parent.c.id, deferrable=True, initially="DEFERRED"),
                nullable=False,
            ),
        )
        metadata.create_all(engine)
        sidecar_journal_outbox_table.create(engine)
        journal = LandscapeJournal(str(journal_path), fail_on_error=True)
        journal.attach(engine)

        with engine.connect() as connection:
            transaction = connection.begin()
            connection.execute(child.insert().values(id=1, parent_id=999))
            with pytest.raises(SQLAlchemyIntegrityError):
                transaction.commit()
            connection.rollback()

        assert not journal_path.exists() or journal_path.read_text(encoding="utf-8") == ""
        with engine.connect() as connection:
            assert connection.scalar(select(child.c.id)) is None
        engine.dispose()

    def test_committed_batch_survives_sidecar_failure_and_is_published_on_reopen(self, tmp_path: Path) -> None:
        db_path = tmp_path / "recoverable.db"
        journal_path = tmp_path / "journal.jsonl"
        journal_path.mkdir()

        db = LandscapeDB.from_url(
            f"sqlite:///{db_path}",
            dump_to_jsonl=True,
            dump_to_jsonl_path=str(journal_path),
        )
        try:
            RecorderFactory(db).run_lifecycle.begin_run(config={"recoverable": True}, canonical_version="v1")
        finally:
            db.close()

        journal_path.rmdir()
        reopened = LandscapeDB.from_url(
            f"sqlite:///{db_path}",
            create_tables=False,
            dump_to_jsonl=True,
            dump_to_jsonl_path=str(journal_path),
        )
        reopened.close()

        records = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
        assert any(record["statement"].lstrip().upper().startswith("INSERT INTO RUNS") for record in records)

    def test_recovery_does_not_duplicate_batch_published_before_outbox_ack(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        engine = create_engine("sqlite:///:memory:")
        sidecar_journal_outbox_table.create(engine)
        journal = LandscapeJournal(str(journal_path), fail_on_error=True)
        batch_id = "a" * 32
        record = cast(
            JournalRecord,
            {
                "timestamp": "2026-01-15T12:00:00+00:00",
                "statement": "INSERT INTO rows (id) VALUES (?)",
                "parameters": ["row-1"],
                "executemany": False,
                "journal_batch_id": batch_id,
                "journal_batch_ordinal": 0,
                "journal_batch_size": 1,
            },
        )
        with engine.begin() as connection:
            connection.execute(
                sidecar_journal_outbox_table.insert().values(
                    batch_id=batch_id,
                    journal_owner=journal._owner_key,
                    created_at=datetime.now(UTC),
                    records_json=json.dumps([record]),
                )
            )

        journal._append_records([record])
        journal.attach(engine)
        journal.recover_pending(engine)

        lines = journal_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        with engine.connect() as connection:
            assert connection.execute(select(sidecar_journal_outbox_table)).all() == []
        engine.dispose()

    def test_recovery_retries_parent_directory_fsync_before_acknowledging_published_batch(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        engine = create_engine("sqlite:///:memory:")
        sidecar_journal_outbox_table.create(engine)
        journal = LandscapeJournal(str(journal_path), fail_on_error=True)
        batch_id = "9" * 32
        records = _outbox_records(batch_id)
        _insert_outbox_batch(engine, journal, batch_id, records)
        journal.attach(engine)

        with patch.object(journal, "_fsync_parent_directory", side_effect=[OSError("dir fsync failed"), None]) as fsync_parent:
            with pytest.raises(OSError, match="dir fsync failed"):
                journal.recover_pending(engine)
            with engine.connect() as connection:
                assert connection.scalar(select(sidecar_journal_outbox_table.c.batch_id)) == batch_id

            journal.recover_pending(engine)

        assert fsync_parent.call_count == 2
        assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 1
        with engine.connect() as connection:
            assert connection.execute(select(sidecar_journal_outbox_table)).all() == []
        engine.dispose()

    def test_recovery_only_publishes_and_acknowledges_its_sidecar_owner(self, tmp_path: Path) -> None:
        db_path = tmp_path / "owned-outbox.db"
        engine_a = create_engine(f"sqlite:///{db_path}")
        engine_b = create_engine(f"sqlite:///{db_path}")
        sidecar_journal_outbox_table.create(engine_a)
        journal_a = LandscapeJournal(str(tmp_path / "worker-a.jsonl"), fail_on_error=True)
        journal_b = LandscapeJournal(str(tmp_path / "worker-b.jsonl"), fail_on_error=True)
        records = _outbox_records("a" * 32)
        _insert_outbox_batch(engine_a, journal_a, "a" * 32, records)
        journal_a.attach(engine_a)
        journal_b.attach(engine_b)

        journal_b.recover_pending(engine_b)

        assert not (tmp_path / "worker-b.jsonl").exists()
        with engine_b.connect() as connection:
            assert connection.scalar(select(sidecar_journal_outbox_table.c.batch_id)) == "a" * 32

        journal_a.recover_pending(engine_a)

        assert len((tmp_path / "worker-a.jsonl").read_text(encoding="utf-8").splitlines()) == 1
        with engine_a.connect() as connection:
            assert connection.execute(select(sidecar_journal_outbox_table)).all() == []
        engine_a.dispose()
        engine_b.dispose()

    def test_sqlite_concurrent_same_owner_recovery_serializes_before_snapshot(self, tmp_path: Path) -> None:
        db_path = tmp_path / "concurrent-outbox.db"
        journal_path = tmp_path / "shared.jsonl"
        setup_engine = create_engine(f"sqlite:///{db_path}")
        sidecar_journal_outbox_table.create(setup_engine)
        barrier = Barrier(2)
        journal_a = _ConcurrentProbeJournal(str(journal_path), barrier)
        records = _outbox_records("c" * 32)
        _insert_outbox_batch(setup_engine, journal_a, "c" * 32, records)
        setup_engine.dispose()

        engine_a = create_engine(f"sqlite:///{db_path}")
        engine_b = create_engine(f"sqlite:///{db_path}")
        journal_b = _ConcurrentProbeJournal(str(journal_path), barrier)
        journal_a.attach(engine_a)
        journal_b.attach(engine_b)
        errors: list[BaseException] = []

        def recover(journal: LandscapeJournal, engine: Engine) -> None:
            try:
                journal.recover_pending(engine)
            except BaseException as exc:
                errors.append(exc)

        threads = [
            Thread(target=recover, args=(journal_a, engine_a)),
            Thread(target=recover, args=(journal_b, engine_b)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 1
        with engine_a.connect() as connection:
            assert connection.execute(select(sidecar_journal_outbox_table)).all() == []
        engine_a.dispose()
        engine_b.dispose()

    def test_recovery_repairs_recognized_torn_batch_tail(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        engine = create_engine("sqlite:///:memory:")
        sidecar_journal_outbox_table.create(engine)
        journal = LandscapeJournal(str(journal_path), fail_on_error=True)
        batch_id = "d" * 32
        records = _outbox_records(batch_id, size=2)
        _insert_outbox_batch(engine, journal, batch_id, records)
        first_line = journal._serialize_record(records[0])
        owner_marker_start = first_line.index(batch_id)
        journal_path.write_text(first_line[: max(1, owner_marker_start // 2)], encoding="utf-8")
        journal_path.chmod(0o600)
        journal.attach(engine)

        journal.recover_pending(engine)

        assert journal_path.read_text(encoding="utf-8").splitlines() == [journal._serialize_record(record) for record in records]
        with engine.connect() as connection:
            assert connection.execute(select(sidecar_journal_outbox_table)).all() == []
        engine.dispose()

    def test_recovery_repairs_later_torn_batch_after_complete_pending_batch(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        engine = create_engine("sqlite:///:memory:")
        sidecar_journal_outbox_table.create(engine)
        journal = LandscapeJournal(str(journal_path), fail_on_error=True)
        first_batch_id = "f" * 32
        second_batch_id = "a" * 32
        first_records = _outbox_records(first_batch_id)
        second_records = _outbox_records(second_batch_id, size=2)
        _insert_outbox_batch(engine, journal, first_batch_id, first_records)
        _insert_outbox_batch(engine, journal, second_batch_id, second_records)
        second_line = journal._serialize_record(second_records[0])
        journal_path.write_text(
            journal._serialize_record(first_records[0]) + "\n" + second_line[: len(second_line) // 2],
            encoding="utf-8",
        )
        journal_path.chmod(0o600)
        journal.attach(engine)

        journal.recover_pending(engine)

        expected = [journal._serialize_record(record) for record in first_records + second_records]
        assert journal_path.read_text(encoding="utf-8").splitlines() == expected
        with engine.connect() as connection:
            assert connection.execute(select(sidecar_journal_outbox_table)).all() == []
        engine.dispose()

    def test_recovery_rejects_unrelated_mid_file_corruption(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        engine = create_engine("sqlite:///:memory:")
        sidecar_journal_outbox_table.create(engine)
        journal = LandscapeJournal(str(journal_path), fail_on_error=True)
        batch_id = "e" * 32
        records = _outbox_records(batch_id)
        _insert_outbox_batch(engine, journal, batch_id, records)
        journal_path.write_text('{"unrelated":\n' + journal._serialize_record(records[0]) + "\n", encoding="utf-8")
        journal_path.chmod(0o600)
        journal.attach(engine)

        with pytest.raises(AuditIntegrityError, match="corrupt JSON"):
            journal.recover_pending(engine)

        with engine.connect() as connection:
            assert connection.scalar(select(sidecar_journal_outbox_table.c.batch_id)) == batch_id
        engine.dispose()

    def test_explicit_fail_on_error_makes_startup_recovery_fail_closed(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        journal_path.mkdir()
        engine = create_engine("sqlite:///:memory:")
        sidecar_journal_outbox_table.create(engine)
        journal = LandscapeJournal(str(journal_path), fail_on_error=True)
        batch_id = "b" * 32
        record = {
            "timestamp": "2026-01-15T12:00:00+00:00",
            "statement": "INSERT INTO rows (id) VALUES (?)",
            "parameters": ["row-1"],
            "executemany": False,
            "journal_batch_id": batch_id,
            "journal_batch_ordinal": 0,
            "journal_batch_size": 1,
        }
        with engine.begin() as connection:
            connection.execute(
                sidecar_journal_outbox_table.insert().values(
                    batch_id=batch_id,
                    journal_owner=journal._owner_key,
                    created_at=datetime.now(UTC),
                    records_json=json.dumps([record]),
                )
            )

        journal.attach(engine)
        with pytest.raises(OSError, match="regular file"):
            journal.recover_pending(engine)
        engine.dispose()

    def test_live_strict_sidecar_failure_preserves_committed_db_and_outbox(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        journal_path.mkdir()
        engine = create_engine("sqlite:///:memory:")
        metadata = MetaData()
        rows = Table("journal_rows", metadata, Column("id", Integer, primary_key=True))
        metadata.create_all(engine)
        sidecar_journal_outbox_table.create(engine)
        journal = LandscapeJournal(str(journal_path), fail_on_error=True)
        journal.attach(engine)

        with engine.begin() as connection:
            connection.execute(rows.insert().values(id=1))

        with engine.connect() as connection:
            assert connection.scalar(select(rows.c.id)) == 1
            assert len(connection.execute(select(sidecar_journal_outbox_table)).all()) == 1
        engine.dispose()


# ===========================================================================
# Per-worker journal path derivation (ADR-030 §C.4 row 13, design line 284)
# ===========================================================================


class TestDeriveJournalPath:
    """Tests for LandscapeDB._derive_journal_path per-worker path derivation.

    N=1 invariant: the N=1 (no worker_suffix) path must be byte-for-byte
    identical to the old behaviour so existing consumers and tests are
    unaffected.

    N>1 invariant: each worker hex suffix produces a distinct filename
    alongside the N=1 file, preventing file-corruption from concurrent
    appends on the same host.
    """

    def test_derive_journal_path_default_unchanged(self, tmp_path: Path) -> None:
        """Regression pin: N=1 leader path equals db.journal.jsonl (no suffix)."""
        db_path = tmp_path / "landscape.db"
        url = f"sqlite:///{db_path}"
        path = LandscapeDB._derive_journal_path(url)
        assert path == str(tmp_path / "landscape.journal.jsonl")

    def test_derive_journal_path_per_worker(self, tmp_path: Path) -> None:
        """Per-worker path embeds the hex suffix before the .jsonl extension."""
        db_path = tmp_path / "landscape.db"
        url = f"sqlite:///{db_path}"
        suffix = "abc123"
        path = LandscapeDB._derive_journal_path(url, suffix)
        assert path == str(tmp_path / "landscape.journal.abc123.jsonl")

    def test_derive_journal_path_per_worker_none_same_as_default(self, tmp_path: Path) -> None:
        """Passing worker_suffix=None explicitly equals the no-arg call."""
        db_path = tmp_path / "landscape.db"
        url = f"sqlite:///{db_path}"
        assert LandscapeDB._derive_journal_path(url, None) == LandscapeDB._derive_journal_path(url)

    @pytest.mark.parametrize("suffix", ["../escape", "abc/123", "ABC123", "", "abc.123"])
    def test_derive_journal_path_rejects_non_lowercase_hex_worker_suffix(self, tmp_path: Path, suffix: str) -> None:
        """Worker suffixes are filename components, so only lowercase hex is accepted."""
        db_path = tmp_path / "landscape.db"
        url = f"sqlite:///{db_path}"

        with pytest.raises(ValueError, match="dump_to_jsonl_worker_suffix"):
            LandscapeDB._derive_journal_path(url, suffix)

    def test_two_workers_write_distinct_journal_files(self, tmp_path: Path) -> None:
        """Two follower instances with distinct hex suffixes write separate files.

        Simulates two followers sharing a WAL DB on the same host.  Each must
        write to a distinct file; no cross-contamination.
        """
        db_path = tmp_path / "landscape.db"
        url = f"sqlite:///{db_path}"

        suffix_a = "aaaa1111"
        suffix_b = "bbbb2222"

        path_a = LandscapeDB._derive_journal_path(url, suffix_a)
        path_b = LandscapeDB._derive_journal_path(url, suffix_b)

        # Paths must be distinct.
        assert path_a != path_b

        # Write something to each path.
        journal_a = LandscapeJournal(path_a, fail_on_error=True)
        journal_b = LandscapeJournal(path_b, fail_on_error=True)

        journal_a._append_records(
            [
                cast(
                    JournalRecord,
                    {"timestamp": "t", "statement": "INSERT", "parameters": {"id": "row-from-a"}, "executemany": False},
                )
            ]
        )
        journal_b._append_records(
            [
                cast(
                    JournalRecord,
                    {"timestamp": "t", "statement": "INSERT", "parameters": {"id": "row-from-b"}, "executemany": False},
                )
            ]
        )

        # Both files exist and contain only their own record.
        lines_a = Path(path_a).read_text().strip().split("\n")
        lines_b = Path(path_b).read_text().strip().split("\n")
        assert len(lines_a) == 1
        assert len(lines_b) == 1
        assert json.loads(lines_a[0])["parameters"] == {"id": "row-from-a"}
        assert json.loads(lines_b[0])["parameters"] == {"id": "row-from-b"}
