"""Tests for LandscapeJournal — append-only JSONL change journal.

Tests cover:
- Statement classification (_is_write_statement)
- Parameter normalization (_normalize_parameters)
- Record serialization (_serialize_record)
- INSERT statement parsing (_parse_insert_statement)
- Column-to-values mapping (_columns_to_values)
- SQLAlchemy event lifecycle (buffer → commit/rollback)
- Failure circuit breaker with periodic recovery
- Payload enrichment for calls table
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

import pytest

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.journal import JournalRecord, LandscapeJournal

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


def _make_conn(buffer: list[Any] | None = None) -> MagicMock:
    """Create a mock SQLAlchemy Connection with info dict.

    When buffer is provided, it becomes the root buffer in the stack.
    """
    conn = MagicMock()
    conn.info = {}
    if buffer is not None:
        conn.info["landscape_journal_buffer_stack"] = [buffer]
    return conn


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


# ===========================================================================
# INSERT statement parsing
# ===========================================================================


class TestParseInsertStatement:
    """Tests for _parse_insert_statement — extracts table name and columns."""

    def test_basic_insert(self) -> None:
        table, cols = LandscapeJournal._parse_insert_statement("INSERT INTO calls (call_id, state_id) VALUES (?, ?)")
        assert table == "calls"
        assert cols == ["call_id", "state_id"]

    def test_quoted_columns(self) -> None:
        table, cols = LandscapeJournal._parse_insert_statement('INSERT INTO "calls" ("call_id", "state_id") VALUES (?, ?)')
        assert table == "calls"
        assert cols == ["call_id", "state_id"]

    def test_non_insert_returns_none(self) -> None:
        table, cols = LandscapeJournal._parse_insert_statement("UPDATE calls SET status = ?")
        assert table is None
        assert cols is None

    def test_no_column_list_parses_values_as_columns(self) -> None:
        table, cols = LandscapeJournal._parse_insert_statement("INSERT INTO calls VALUES (1, 2)")
        # Parser finds the first '(' which is the VALUES paren, so table name
        # absorbs "VALUES" and the values parens become the "column list"
        assert table == "calls values"
        assert cols == ["1", "2"]

    def test_missing_close_paren_returns_none_columns(self) -> None:
        table, cols = LandscapeJournal._parse_insert_statement("INSERT INTO calls (col1, col2")
        assert table == "calls"
        assert cols is None


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


class TestAfterCommit:
    """Tests for _after_commit — flushes buffer to disk."""

    def test_flushes_buffer_to_file(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        record = {
            "timestamp": "2026-01-15T12:00:00",
            "statement": "INSERT INTO rows (id) VALUES (?)",
            "parameters": {"id": "r1"},
            "executemany": False,
        }
        conn = _make_conn(buffer=[record])

        journal._after_commit(conn)

        journal_path = tmp_path / "journal.jsonl"
        assert journal_path.exists()
        lines = journal_path.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["statement"] == "INSERT INTO rows (id) VALUES (?)"

    def test_clears_buffer_after_flush(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        buffer: list[Any] = [{"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False}]
        conn = _make_conn(buffer=buffer)

        journal._after_commit(conn)

        # After commit the stack is reset to a single empty root buffer
        stack = conn.info["landscape_journal_buffer_stack"]
        assert stack == [[]]

    def test_no_buffer_is_noop(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        conn = _make_conn()

        journal._after_commit(conn)  # Should not raise

        journal_path = tmp_path / "journal.jsonl"
        assert not journal_path.exists()

    def test_empty_buffer_is_noop(self, tmp_path: Path) -> None:
        journal = _make_journal(tmp_path)
        conn = _make_conn(buffer=[])

        journal._after_commit(conn)

        journal_path = tmp_path / "journal.jsonl"
        assert not journal_path.exists()

    def test_disabled_journal_routes_through_append_records(self, tmp_path: Path) -> None:
        """When disabled, _after_commit routes records through _append_records for recovery tracking."""
        journal = _make_journal(tmp_path)
        journal._disabled = True
        buffer = [{"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False}]
        conn = _make_conn(buffer=buffer)

        journal._after_commit(conn)

        # Records are counted as dropped (recovery attempts every 100 drops)
        assert journal._total_dropped == 1
        # Stack is reset after commit processes it
        stack = conn.info["landscape_journal_buffer_stack"]
        assert stack == [[]]


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
        engine = Mock()

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
        journal._payload_store = Mock()
        journal._payload_store.retrieve.return_value = b'{"key": "value"}'

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
        journal._payload_store = Mock()
        journal._payload_store.retrieve.side_effect = PayloadNotFoundError("deadbeef" * 8)

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
        journal._payload_store = Mock()
        journal._payload_store.retrieve.side_effect = OSError("disk failure")

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
        journal._payload_store = Mock()
        journal._payload_store.retrieve.side_effect = PayloadNotFoundError("deadbeef" * 8)

        with pytest.raises(PayloadNotFoundError):
            journal._load_payload("some-ref")

    def test_read_failure_os_error_with_fail_on_error_raises(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            fail_on_error=True,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = Mock()
        journal._payload_store.retrieve.side_effect = OSError("disk failure")

        with pytest.raises(OSError):
            journal._load_payload("some-ref")

    def test_programming_error_in_retrieve_not_caught(self, tmp_path: Path) -> None:
        """TypeError/AttributeError in payload store must crash, not be silently swallowed."""
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = Mock()
        journal._payload_store.retrieve.side_effect = TypeError("bad type in store")

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
        journal._payload_store = Mock()
        journal._payload_store.retrieve.side_effect = IntegrityError("expected abc123, got def456")

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
        journal._payload_store = Mock()
        journal._payload_store.retrieve.side_effect = IntegrityError("expected abc123, got def456")

        with pytest.raises(AuditIntegrityError, match="corruption or tampering"):
            journal._load_payload("some-ref")

    def test_decode_failure_returns_error(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = Mock()
        journal._payload_store.retrieve.return_value = b"\x80\x81\x82"  # Invalid UTF-8

        content, error = journal._load_payload("some-ref")
        assert content is None
        assert error is not None
        assert "payload_decode_failed" in error


class TestEnrichWithPayloads:
    """Tests for _enrich_with_payloads — adds payload data to call records."""

    def test_non_calls_table_skipped(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False})
        journal._enrich_with_payloads(
            record,
            "INSERT INTO rows (id) VALUES (?)",
            {"id": "r1"},
            executemany=False,
        )
        # No payload keys should be added
        assert "request_ref" not in record
        assert "payloads" not in record

    def test_schema_qualified_calls_table_enriched(self, tmp_path: Path) -> None:
        """Regression: schema-qualified table names like public.calls must match."""
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = Mock()
        journal._payload_store.retrieve.return_value = b"payload content"

        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False})
        journal._enrich_with_payloads(
            record,
            "INSERT INTO public.calls (call_id, request_ref, response_ref) VALUES (?, ?, ?)",
            {"call_id": "c1", "request_ref": "req-ref", "response_ref": "resp-ref"},
            executemany=False,
        )
        assert record["request_ref"] == "req-ref"
        assert record["request_payload"] == "payload content"

    def test_single_call_enriched(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = Mock()
        journal._payload_store.retrieve.return_value = b"payload content"

        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": False})
        journal._enrich_with_payloads(
            record,
            "INSERT INTO calls (call_id, request_ref, response_ref) VALUES (?, ?, ?)",
            {"call_id": "c1", "request_ref": "req-ref", "response_ref": "resp-ref"},
            executemany=False,
        )

        assert record["request_ref"] == "req-ref"
        assert record["request_payload"] == "payload content"
        assert record["response_ref"] == "resp-ref"

    def test_executemany_enriched_as_list(self, tmp_path: Path) -> None:
        journal = _make_journal(
            tmp_path,
            include_payloads=True,
            payload_base_path=str(tmp_path / "payloads"),
        )
        journal._payload_store = Mock()
        journal._payload_store.retrieve.return_value = b"payload"

        record = cast(JournalRecord, {"timestamp": "t", "statement": "INSERT", "parameters": {}, "executemany": True})
        journal._enrich_with_payloads(
            record,
            "INSERT INTO calls (call_id, request_ref, response_ref) VALUES (?, ?, ?)",
            [
                {"call_id": "c1", "request_ref": "r1", "response_ref": "r2"},
                {"call_id": "c2", "request_ref": "r3", "response_ref": None},
            ],
            executemany=True,
        )

        assert "payloads" in record
        assert len(record["payloads"]) == 2


# ===========================================================================
# End-to-end: cursor → commit → file
# ===========================================================================


class TestEndToEnd:
    """Integration-style tests for the full event lifecycle."""

    def test_cursor_then_commit_writes_file(self, tmp_path: Path) -> None:
        """Full flow: cursor_execute → buffer → commit → JSONL file."""
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
        journal._after_commit(conn)

        journal_path = tmp_path / "journal.jsonl"
        lines = journal_path.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["parameters"] == {"id": "row-1"}

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
        conn = _make_conn()

        for i in range(3):
            journal._after_cursor_execute(
                conn,
                cursor=None,
                statement="INSERT INTO rows (id) VALUES (?)",
                parameters={"id": f"row-{i}"},
                context=None,
                executemany=False,
            )
        journal._after_commit(conn)

        journal_path = tmp_path / "journal.jsonl"
        lines = journal_path.read_text().strip().split("\n")
        assert len(lines) == 3
