"""Regression tests for P1 sink plugin bugs.

Bug 1: Azure blob sink misses required field validation (P2-2026-02-14)
Bug 2: CSVSink partial batch writes (P1-2026-02-14)
Bug 3: DatabaseSink accepts schema-invalid rows (P2-2026-02-14)
Bug 4: Schema type 'any' mapped to SQL TEXT without serialization (P1-2026-02-14)
Bug 5: JSONSink append mode lacks schema validation (P2-2026-02-14)
"""

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from elspeth.contracts.plugin_context import PluginContext
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_operation_context

# === Shared fixtures and schemas ===

FIXED_SCHEMA = {"mode": "fixed", "fields": ["id: int", "name: str"]}
FLEXIBLE_SCHEMA = {"mode": "flexible", "fields": ["id: int", "name: str"]}
OBSERVED_SCHEMA = {"mode": "observed"}

# Azure Blob Sink test constants
TEST_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key"
TEST_CONTAINER = "output-container"
TEST_BLOB_PATH = "results/output.csv"


class _UploadRecorder:
    def __init__(self) -> None:
        self.call_count = 0

    @property
    def called(self) -> bool:
        return self.call_count > 0

    def __call__(self, *_args: Any, **_kwargs: Any) -> None:
        self.call_count += 1

    def assert_called_once(self) -> None:
        assert self.call_count == 1


class _BlobClientFake:
    def __init__(self) -> None:
        self.upload_blob = _UploadRecorder()


class _ContainerClientFake:
    def __init__(self, blob_client: _BlobClientFake) -> None:
        self._blob_client = blob_client

    def get_blob_client(self, *_args: Any, **_kwargs: Any) -> _BlobClientFake:
        return self._blob_client


@pytest.fixture
def ctx() -> PluginContext:
    """Create a plugin context with real landscape and operation records."""
    return make_operation_context(
        node_id="sink",
        plugin_name="database_sink",
        node_type="SINK",
        operation_type="sink_write",
    )


# =============================================================================
# Bug 2: CSVSink partial batch writes
# =============================================================================


class TestCSVSinkAtomicBatchWrite:
    """Regression tests for P1-2026-02-14: CSVSink partial batch writes.

    CSVSink.write() must never leave a partial batch on disk on an *I/O* failure:
    if the real file write/flush fails part-way, the batch is rolled back so no
    bytes exist without a matching audit record. That batch-integrity guarantee
    still holds.

    Per-row *encode* faults (a row with a field outside the column lock — a
    ValueError from DictWriter, attributable to one row) are NOT batch failures.
    Per audit finding #6 they are diverted (recorded + routed via
    on_write_failure) so the surrounding good rows are written, rather than
    aborting the batch. The P1-2026-02-14 invariant — never a row on disk that
    audit marks FAILED — is preserved: good rows are written AND audited as
    written; the bad row is excluded from disk AND recorded as a diversion.
    """

    def test_extra_field_in_batch_diverts_only_bad_row(self, tmp_path: Path, ctx: PluginContext) -> None:
        """A row with extra fields is diverted; the surrounding good rows write.

        Previously this pinned crash-the-batch behavior: a single row with a
        field outside the column lock raised ValueError and dropped the whole
        batch (no rows written). That pinned audit finding #6. The good rows now
        write and the bad row is recorded as a diversion (row on disk <-> audit
        stays consistent: written rows are audited written, the diverted row is
        recorded diverted, never written-but-FAILED).
        """
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(CSVSink({"path": str(output_file), "schema": FIXED_SCHEMA}))

        # First batch succeeds (establishes file)
        sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.flush()

        # Second batch: row 0 is valid, row 1 has an extra field outside the lock.
        result = sink.write(
            [
                {"id": 2, "name": "bob"},
                {"id": 3, "name": "carol", "c": "extra"},
            ],
            ctx,
        )

        # The bad row is recorded as a diversion (not written, not lost).
        assert len(result.diversions) == 1
        assert result.diversions[0].row_index == 1
        assert result.diversions[0].row_data == {"id": 3, "name": "carol", "c": "extra"}

        sink.close()

        # The good rows from both batches are on disk; the diverted row is absent.
        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert [r["id"] for r in rows] == ["1", "2"]

    def test_valid_batch_still_writes_correctly(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Valid batches still write correctly after the staging change."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(CSVSink({"path": str(output_file), "schema": FIXED_SCHEMA}))

        sink.write(
            [
                {"id": 1, "name": "alice"},
                {"id": 2, "name": "bob"},
                {"id": 3, "name": "carol"},
            ],
            ctx,
        )
        sink.flush()
        sink.close()

        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[2]["name"] == "carol"

    def test_hash_is_correct_after_staged_write(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Content hash matches actual file content after staged write."""
        import hashlib

        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(CSVSink({"path": str(output_file), "schema": FIXED_SCHEMA}))

        artifact = sink.write([{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}], ctx)
        sink.flush()
        sink.close()

        expected_hash = hashlib.sha256(output_file.read_bytes()).hexdigest()
        assert artifact.artifact.content_hash == expected_hash


# =============================================================================
# Bug 6: Write-mode first-batch failure cleanup
# =============================================================================


class TestCSVSinkWriteModeRollback:
    """Regression tests for write-mode first-batch failure cleanup.

    When the first batch fails during staging (e.g., extra fields), the file
    must never be created. When staging succeeds but the post-open write/flush/
    hash/stat phase fails, the newly-created file must be cleaned up.
    """

    def test_first_batch_extra_field_diverts_writes_header_only(self, tmp_path: Path, ctx: PluginContext) -> None:
        """A first-batch row with extra fields is diverted; a header-only file results.

        Previously this pinned crash-the-batch behavior: an extra-field row in the
        first batch raised ValueError and the file was never created. Per audit
        finding #6 the bad row is now diverted (recorded + routed). In fixed mode
        the columns come from the schema (not the row), so the file is created with
        the header and the diverted row is excluded — header-only output, with the
        fault recorded as a diversion rather than dropped.
        """
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(CSVSink({"path": str(output_file), "schema": FIXED_SCHEMA}))

        # Row has an extra field that the fixed schema column lock rejects.
        result = sink.write([{"id": 1, "name": "alice", "c": "extra"}], ctx)

        assert len(result.diversions) == 1
        assert result.diversions[0].row_index == 0
        assert result.diversions[0].row_data == {"id": 1, "name": "alice", "c": "extra"}

        sink.close()

        # Header-only file created; no data rows reached disk.
        assert output_file.exists()
        with open(output_file) as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames or []) == ["id", "name"]
            assert list(reader) == []

    def test_write_failure_after_open_removes_file(self, tmp_path: Path, ctx: PluginContext) -> None:
        """If file.write() fails after open, the newly-created file is removed.

        This tests the rollback guard around the post-open mutation phase.
        """
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(CSVSink({"path": str(output_file), "schema": FIXED_SCHEMA}))

        # Patch file.write to fail AFTER the file is opened (staging succeeds
        # because it uses a separate StringIO buffer, so we target the real file)
        original_open = open

        call_count = 0

        def patched_open(*args, **kwargs):
            nonlocal call_count
            f = original_open(*args, **kwargs)
            # The first open call with "w" mode is _open_file_write_mode.
            # We want file.write to fail on the staged_content write (second write
            # to the real file, after the header). Wrap the file to fail on the
            # second call to write().
            if len(args) >= 2 and args[1] == "w":
                original_write = f.write
                write_call_count = 0

                def failing_write(data):
                    nonlocal write_call_count
                    write_call_count += 1
                    # Let header writes through (writeheader), fail on staged content
                    if write_call_count > 1:
                        raise OSError("Simulated disk full")
                    return original_write(data)

                f.write = failing_write
            return f

        with patch("builtins.open", side_effect=patched_open), pytest.raises(IOError, match="Simulated disk full"):
            sink.write([{"id": 1, "name": "alice"}], ctx)

        # File must have been cleaned up by the rollback guard
        assert not output_file.exists()
        assert sink._file is None
        assert sink._writer is None

    def test_append_mode_rollback_failure_chains_exceptions(self, tmp_path: Path, ctx: PluginContext) -> None:
        """If append-mode rollback itself fails, exceptions are chained.

        The original write exception must not be silently replaced by the
        rollback failure.
        """
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(CSVSink({"path": str(output_file), "schema": FIXED_SCHEMA, "mode": "append"}))

        # First batch succeeds (establishes file)
        sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.flush()

        # Strategy: make the hash read fail (triggers rollback), then make
        # seek fail (rollback itself fails). We patch builtins.open so that
        # the binary-mode hash read raises, while leaving the file handle
        # methods untouched (avoids tell/flush interactions).
        original_file = sink._file
        assert original_file is not None
        real_open = open

        def open_that_fails_on_binary_read(*args, **kwargs):
            # Fail when the hash computation tries to open in "rb" mode
            if len(args) >= 2 and args[1] == "rb":
                raise OSError("simulated read failure")
            if len(args) >= 1 and kwargs.get("mode") == "rb":
                raise OSError("simulated read failure")
            return real_open(*args, **kwargs)

        # Also patch seek so rollback fails
        original_seek = original_file.seek

        def failing_seek(pos):
            raise OSError("seek failed during rollback")

        original_file.seek = failing_seek  # type: ignore[method-assign,assignment]  # intentional monkey-patch for test

        with (
            patch("builtins.open", side_effect=open_that_fails_on_binary_read),
            pytest.raises(RuntimeError, match="rollback also failed") as exc_info,
        ):
            sink.write([{"id": 2, "name": "bob"}], ctx)

        # Verify exception chain: RuntimeError caused by OSError
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, OSError)

        # Restore original methods for cleanup
        original_file.seek = original_seek  # type: ignore[method-assign]  # restoring original method
        sink.close()


# =============================================================================
# Bug 3: DatabaseSink accepts schema-invalid rows
# =============================================================================


class TestDatabaseSinkSchemaEnforcement:
    """Regression tests for P2-2026-02-14: DatabaseSink silently accepts
    schema-invalid rows.

    SQLAlchemy silently drops extra keys on INSERT. DatabaseSink must reject
    rows with fields not in the table schema.
    """

    @pytest.fixture
    def db_url(self, tmp_path: Path) -> str:
        return f"sqlite:///{tmp_path / 'test.db'}"

    def test_extra_fields_rejected_after_table_creation(self, db_url: str, ctx: PluginContext) -> None:
        """Rows with extra fields are rejected after the table is created.

        Before the fix, SQLAlchemy silently dropped unknown keys, hiding
        upstream bugs.
        """
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": FIXED_SCHEMA}))

        # First write creates the table with columns [id, name]
        sink.write([{"id": 1, "name": "alice"}], ctx)

        # Second write with extra field should be rejected
        with pytest.raises(ValueError, match="not in table schema"):
            sink.write([{"id": 2, "name": "bob", "extra": "value"}], ctx)

        sink.close()

    def test_required_fields_enforced_by_nullable_columns(self, db_url: str, ctx: PluginContext) -> None:
        """Required fields create NOT NULL columns in the database.

        Before the fix, all columns were nullable by default, allowing
        NULL to silently replace missing required fields.
        """
        from sqlalchemy import create_engine, inspect

        from elspeth.plugins.sinks.database_sink import DatabaseSink

        required_schema = {"mode": "fixed", "fields": ["id: int", "name: str"]}
        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": required_schema}))

        sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.close()

        # Inspect column nullability
        engine = create_engine(db_url)
        inspector = inspect(engine)
        columns = {col["name"]: col for col in inspector.get_columns("output")}

        # Both fields are required (no '?' suffix) so nullable should be False
        assert columns["id"]["nullable"] is False
        assert columns["name"]["nullable"] is False

        engine.dispose()

    def test_optional_fields_are_nullable(self, db_url: str, ctx: PluginContext) -> None:
        """Optional fields create nullable columns."""
        from sqlalchemy import create_engine, inspect

        from elspeth.plugins.sinks.database_sink import DatabaseSink

        optional_schema = {"mode": "fixed", "fields": ["id: int", "name: str?"]}
        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": optional_schema}))

        sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.close()

        engine = create_engine(db_url)
        inspector = inspect(engine)
        columns = {col["name"]: col for col in inspector.get_columns("output")}

        assert columns["id"]["nullable"] is False  # Required
        assert columns["name"]["nullable"] is True  # Optional

        engine.dispose()


# =============================================================================
# Bug 4: Schema type 'any' mapped to SQL TEXT without serialization
# =============================================================================


class TestDatabaseSinkAnyTypeSerialization:
    """Regression tests for P1-2026-02-14: Schema type 'any' is mapped to
    SQL TEXT without serialization.

    Valid 'any' values like dicts and lists must be serialized to JSON strings
    before INSERT into TEXT columns, not crash with driver errors.
    """

    @pytest.fixture
    def db_url(self, tmp_path: Path) -> str:
        return f"sqlite:///{tmp_path / 'test.db'}"

    def test_dict_value_in_any_field_is_serialized(self, db_url: str, ctx: PluginContext) -> None:
        """Dict values in 'any'-typed fields are serialized to JSON strings.

        Before the fix, inserting {"payload": {"k": 1}} into a TEXT column
        crashed with "type 'dict' is not supported".
        """
        from sqlalchemy import MetaData, Table, create_engine, select

        from elspeth.plugins.sinks.database_sink import DatabaseSink

        any_schema = {"mode": "fixed", "fields": ["id: int", "payload: any"]}
        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": any_schema}))

        sink.write([{"id": 1, "payload": {"key": "value", "nested": [1, 2, 3]}}], ctx)
        sink.close()

        # Verify the JSON was stored as a string
        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table("output", metadata, autoload_with=engine)
        with engine.connect() as conn:
            rows = list(conn.execute(select(table)))
        engine.dispose()

        assert len(rows) == 1
        stored_payload = rows[0][1]  # payload column
        assert isinstance(stored_payload, str)
        parsed = json.loads(stored_payload)
        assert parsed == {"key": "value", "nested": [1, 2, 3]}

    def test_list_value_in_any_field_is_serialized(self, db_url: str, ctx: PluginContext) -> None:
        """List values in 'any'-typed fields are serialized to JSON strings."""
        from sqlalchemy import MetaData, Table, create_engine, select

        from elspeth.plugins.sinks.database_sink import DatabaseSink

        any_schema = {"mode": "fixed", "fields": ["id: int", "items: any"]}
        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": any_schema}))

        sink.write([{"id": 1, "items": [1, 2, 3]}], ctx)
        sink.close()

        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table("output", metadata, autoload_with=engine)
        with engine.connect() as conn:
            rows = list(conn.execute(select(table)))
        engine.dispose()

        assert len(rows) == 1
        stored_items = rows[0][1]
        assert isinstance(stored_items, str)
        assert json.loads(stored_items) == [1, 2, 3]

    def test_scalar_any_values_are_not_modified(self, db_url: str, ctx: PluginContext) -> None:
        """Scalar values (str, int, None) in 'any' fields are stored as-is."""
        from sqlalchemy import MetaData, Table, create_engine, select

        from elspeth.plugins.sinks.database_sink import DatabaseSink

        # Use optional 'any' field (with '?') so None is allowed as NULL
        any_schema = {"mode": "fixed", "fields": ["id: int", "data: any?"]}
        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": any_schema}))

        sink.write(
            [
                {"id": 1, "data": "hello"},
                {"id": 2, "data": None},
            ],
            ctx,
        )
        sink.close()

        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table("output", metadata, autoload_with=engine)
        with engine.connect() as conn:
            rows = list(conn.execute(select(table)))
        engine.dispose()

        assert len(rows) == 2
        assert rows[0][1] == "hello"  # str stored as-is
        assert rows[1][1] is None  # None stored as NULL

    def test_observed_mode_serializes_complex_values(self, db_url: str, ctx: PluginContext) -> None:
        """Observed mode also serializes dict/list values to JSON strings."""
        from sqlalchemy import MetaData, Table, create_engine, select

        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "output", "schema": OBSERVED_SCHEMA}))

        sink.write([{"id": 1, "meta": {"source": "test"}}], ctx)
        sink.close()

        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table("output", metadata, autoload_with=engine)
        with engine.connect() as conn:
            rows = list(conn.execute(select(table)))
        engine.dispose()

        assert len(rows) == 1
        stored_meta = rows[0][1]
        assert isinstance(stored_meta, str)
        assert json.loads(stored_meta) == {"source": "test"}


# =============================================================================
# Bug 5: JSONSink append mode lacks schema validation
# =============================================================================


class TestJSONSinkAppendSchemaValidation:
    """Regression tests for P2-2026-02-14: JSONSink append mode lacks
    schema validation.

    When appending to an existing JSONL file with an explicit schema,
    the sink must validate that the existing data is compatible.
    """

    def test_append_to_incompatible_fixed_schema_raises(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Appending to a JSONL file with incompatible fixed schema raises ValueError.

        Before the fix, incompatible rows were silently appended.
        """
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        # Write existing data with different fields
        output_file.write_text(json.dumps({"x": 1, "y": 2}) + "\n")

        # Try to append with a fixed schema that requires id and name
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "format": "jsonl",
                    "mode": "append",
                    "schema": FIXED_SCHEMA,
                }
            )
        )

        with pytest.raises(ValueError, match="JSONL schema mismatch"):
            sink.write([{"id": 1, "name": "alice"}], ctx)

        sink.close()

    def test_append_to_compatible_fixed_schema_succeeds(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Appending to a JSONL file with compatible fixed schema succeeds."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        # Existing data has same fields
        output_file.write_text(json.dumps({"id": 1, "name": "alice"}) + "\n")

        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "format": "jsonl",
                    "mode": "append",
                    "schema": FIXED_SCHEMA,
                }
            )
        )

        # Should succeed -- schema matches
        sink.write([{"id": 2, "name": "bob"}], ctx)
        sink.flush()
        sink.close()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[1]) == {"id": 2, "name": "bob"}

    def test_append_to_missing_flexible_schema_field_raises(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Appending to JSONL missing a required flexible field raises ValueError."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        # Existing data is missing 'name' field required by flexible schema
        output_file.write_text(json.dumps({"id": 1}) + "\n")

        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "format": "jsonl",
                    "mode": "append",
                    "schema": FLEXIBLE_SCHEMA,
                }
            )
        )

        with pytest.raises(ValueError, match="JSONL schema mismatch"):
            sink.write([{"id": 2, "name": "bob"}], ctx)

        sink.close()

    def test_append_to_nonexistent_file_succeeds(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Appending to a nonexistent JSONL file creates it successfully."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        assert not output_file.exists()

        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "format": "jsonl",
                    "mode": "append",
                    "schema": FIXED_SCHEMA,
                }
            )
        )

        sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.flush()
        sink.close()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"id": 1, "name": "alice"}

    def test_append_observed_mode_skips_validation(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Append with observed schema skips validation (dynamic adapts)."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        # Existing data has completely different fields
        output_file.write_text(json.dumps({"x": 1, "y": 2}) + "\n")

        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "format": "jsonl",
                    "mode": "append",
                    "schema": OBSERVED_SCHEMA,
                }
            )
        )

        # Should succeed -- observed mode doesn't validate
        sink.write([{"a": 1, "b": 2}], ctx)
        sink.flush()
        sink.close()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2


class TestJSONLAppendRollbackPreservesExistingContent:
    """Regression: JSONL append-mode rollback must not destroy pre-existing content.

    Before the fix, ``pre_write_pos`` was captured before the file was opened
    on the first call. In append mode with existing content, this defaulted to 0,
    so rollback would truncate the entire file — destroying pre-existing lines.
    """

    def test_rollback_preserves_existing_lines(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Post-write failure in append mode must leave existing content intact."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"

        # Create pre-existing content via a first sink lifecycle
        sink1 = inject_write_failure(JSONSink({"path": str(output_file), "format": "jsonl", "schema": OBSERVED_SCHEMA}))
        sink1.write([{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}], ctx)
        sink1.flush()
        sink1.close()

        original_content = output_file.read_text()
        assert original_content.count("\n") == 2  # two lines

        # Second sink in append mode — force a failure after write
        sink2 = inject_write_failure(JSONSink({"path": str(output_file), "format": "jsonl", "mode": "append", "schema": OBSERVED_SCHEMA}))

        # Patch _compute_file_hash to fail AFTER content is written,
        # triggering the rollback path
        with (
            patch.object(sink2, "_compute_file_hash", side_effect=OSError("simulated hash failure")),
            pytest.raises(OSError, match="simulated hash failure"),
        ):
            sink2.write([{"id": 3, "name": "charlie"}], ctx)

        # Pre-existing content must be intact after rollback
        surviving_content = output_file.read_text()
        assert surviving_content == original_content

        sink2.close()


# =============================================================================
# Bug 1: Azure blob sink misses required field validation
# =============================================================================
