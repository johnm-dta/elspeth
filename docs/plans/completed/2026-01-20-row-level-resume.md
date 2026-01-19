# Row-Level Resume Implementation Plan (Revised)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the row-level resume functionality so failed pipeline runs can continue from the last checkpoint.

**Architecture:** Resume uses existing checkpoint infrastructure to identify unprocessed rows, reload their data from the payload store, and continue processing through RowProcessor with restored aggregation state. A NullSource satisfies PipelineConfig typing while actual data comes from payloads.

**Tech Stack:** SQLAlchemy Core (queries), existing RecoveryManager/CheckpointManager, PayloadStore for row data retrieval

---

## Revision Notes

This plan addresses issues identified in the fitness assessment:
- ✅ **PipelineConfig.source type** - Create NullSource class (Option B)
- ✅ **PayloadStore.store() signature** - Remove content_type parameter
- ✅ **CSVSink append mode** - Read existing headers from file, extract to method
- ✅ **CLI resume command** - Keep existing --database, add --execute flag
- ✅ **Test fixtures** - Use correct PayloadStore.store() signature

**2026-01-20 Corrections (post-codebase assessment):**
- ✅ **Orchestrator TODO line numbers** - Changed from 897-912 to 931-935 (actual location)
- ✅ **ExecutionGraph API** - Changed from non-existent `add_source()`/`add_transform()`/`add_sink()` convenience methods to actual `add_node(node_id, node_type=, plugin_name=, config=)` API
- ✅ **Task 4 test** - Updated `_create_graph_for_resume()` to use `add_node()`
- ✅ **Task 5 helper** - Updated `_build_resume_graph_from_db()` to use `add_node()`
- ✅ **Task 6 integration test** - Updated graph construction to use `add_node()`

---

## Task 1: Create NullSource for Resume

**Files:**
- Create: `src/elspeth/plugins/sources/null_source.py`
- Modify: `src/elspeth/plugins/sources/__init__.py`
- Test: `tests/plugins/sources/test_null_source.py`

**Step 1: Write the failing test**

```python
# tests/plugins/sources/test_null_source.py
"""Tests for NullSource - a source that yields nothing for resume operations."""

import pytest

from elspeth.plugins.sources.null_source import NullSource
from elspeth.plugins.context import PluginContext


class TestNullSource:
    """Tests for NullSource."""

    def test_null_source_yields_nothing(self) -> None:
        """NullSource.load() yields no rows."""
        source = NullSource({})
        ctx = PluginContext(run_id="test", config={}, landscape=None)

        rows = list(source.load(ctx))

        assert rows == []

    def test_null_source_has_name(self) -> None:
        """NullSource has 'null' as its name."""
        source = NullSource({})
        assert source.name == "null"

    def test_null_source_satisfies_protocol(self) -> None:
        """NullSource satisfies SourceProtocol."""
        from elspeth.plugins.protocols import SourceProtocol

        source = NullSource({})
        # This should not raise - source satisfies protocol
        assert isinstance(source, SourceProtocol)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/sources/test_null_source.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'elspeth.plugins.sources.null_source'"

**Step 3: Write minimal implementation**

```python
# src/elspeth/plugins/sources/null_source.py
"""NullSource - a source that yields nothing.

Used by resume operations where row data comes from the payload store,
not from the original source. Satisfies PipelineConfig.source typing
while actual row data is retrieved separately.
"""

from collections.abc import Iterator
from typing import Any

from elspeth.contracts import SourceRow
from elspeth.plugins.base import BaseSource
from elspeth.plugins.context import PluginContext


class NullSource(BaseSource):
    """A source that yields no rows.

    Used during resume when row data comes from the payload store.
    The source slot in PipelineConfig must be filled, but the source
    is never actually called during resume - rows are retrieved from
    stored payloads instead.
    """

    name = "null"
    plugin_version = "1.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def load(self, ctx: PluginContext) -> Iterator[SourceRow]:
        """Yield no rows.

        Resume operations retrieve row data from the payload store,
        not from this source.
        """
        return iter([])

    def close(self) -> None:
        """No resources to close."""
        pass
```

**Step 4: Update sources __init__.py**

```python
# src/elspeth/plugins/sources/__init__.py
# Add to exports:
from elspeth.plugins.sources.null_source import NullSource

__all__ = [..., "NullSource"]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/plugins/sources/test_null_source.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/plugins/sources/null_source.py src/elspeth/plugins/sources/__init__.py tests/plugins/sources/test_null_source.py
git commit -m "$(cat <<'EOF'
feat(sources): add NullSource for resume operations

NullSource yields no rows - used by resume when row data comes from
payload store rather than the original source. Satisfies
PipelineConfig.source typing requirement.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add Row Data Retrieval to RecoveryManager

**Files:**
- Modify: `src/elspeth/core/checkpoint/recovery.py`
- Create: `tests/core/checkpoint/conftest.py`
- Test: `tests/core/checkpoint/test_recovery_row_data.py`

**Step 1: Create conftest with fixtures using correct PayloadStore signature**

```python
# tests/core/checkpoint/conftest.py
"""Shared fixtures for checkpoint tests."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    runs_table,
    tokens_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore


@pytest.fixture
def db(tmp_path: Path) -> LandscapeDB:
    """Create test database."""
    return LandscapeDB(f"sqlite:///{tmp_path}/test.db")


@pytest.fixture
def payload_store(tmp_path: Path) -> FilesystemPayloadStore:
    """Create test payload store."""
    return FilesystemPayloadStore(tmp_path / "payloads")


@pytest.fixture
def checkpoint_manager(db: LandscapeDB) -> CheckpointManager:
    """Create checkpoint manager."""
    return CheckpointManager(db)


@pytest.fixture
def recovery_manager(
    db: LandscapeDB, checkpoint_manager: CheckpointManager
) -> RecoveryManager:
    """Create recovery manager."""
    return RecoveryManager(db, checkpoint_manager)


@pytest.fixture
def run_with_checkpoint_and_payloads(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    payload_store: FilesystemPayloadStore,
) -> str:
    """Create a failed run with checkpoint and payload data.

    Creates 5 rows (0-4), checkpoint at row 2, so rows 3-4 are unprocessed.
    All rows have payload data stored.
    """
    run_id = "test-run-resume"
    now = datetime.now(UTC)

    with db.engine.connect() as conn:
        # Create run (failed status)
        conn.execute(
            runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                config_hash="test",
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status="failed",
            )
        )

        # Create source node
        conn.execute(
            nodes_table.insert().values(
                node_id="source-node",
                run_id=run_id,
                plugin_name="csv",
                node_type="source",
                plugin_version="1.0",
                determinism="io_read",
                config_hash="x",
                config_json="{}",
                registered_at=now,
            )
        )

        # Create rows with payload data
        for i in range(5):
            row_id = f"row-{i:03d}"
            row_data = {"id": i, "value": f"data-{i}"}

            # Store payload - CORRECT SIGNATURE: store(content) returns hash
            payload_bytes = json.dumps(row_data).encode("utf-8")
            payload_ref = payload_store.store(payload_bytes)

            conn.execute(
                rows_table.insert().values(
                    row_id=row_id,
                    run_id=run_id,
                    source_node_id="source-node",
                    row_index=i,
                    source_data_hash=f"hash{i}",
                    source_data_ref=payload_ref,  # Reference to stored payload
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id=f"tok-{i:03d}",
                    row_id=row_id,
                    created_at=now,
                )
            )

        conn.commit()

    # Create checkpoint at row 2 (rows 3-4 are unprocessed)
    checkpoint_manager.create_checkpoint(
        run_id=run_id,
        token_id="tok-002",
        node_id="source-node",
        sequence_number=2,
    )

    return run_id
```

**Step 2: Write the failing test**

```python
# tests/core/checkpoint/test_recovery_row_data.py
"""Tests for RecoveryManager row data retrieval."""

import pytest

from elspeth.core.checkpoint import RecoveryManager
from elspeth.core.payload_store import FilesystemPayloadStore


class TestRecoveryManagerRowData:
    """Tests for get_unprocessed_row_data()."""

    def test_get_unprocessed_row_data_returns_row_dicts(
        self,
        recovery_manager: RecoveryManager,
        payload_store: FilesystemPayloadStore,
        run_with_checkpoint_and_payloads: str,
    ) -> None:
        """get_unprocessed_row_data returns actual row data, not just IDs."""
        run_id = run_with_checkpoint_and_payloads

        # Get row data for unprocessed rows
        row_data_list = recovery_manager.get_unprocessed_row_data(
            run_id=run_id,
            payload_store=payload_store,
        )

        # Should return list of (row_index, row_data) tuples
        assert len(row_data_list) == 2  # rows 3 and 4
        assert all(isinstance(item, tuple) for item in row_data_list)
        assert all(len(item) == 2 for item in row_data_list)

        # Verify row indices are correct and in order
        indices = [item[0] for item in row_data_list]
        assert indices == [3, 4]

        # Verify row data is correct
        for row_index, row_data in row_data_list:
            assert isinstance(row_data, dict)
            assert row_data["id"] == row_index
            assert row_data["value"] == f"data-{row_index}"

    def test_get_unprocessed_row_data_empty_when_all_processed(
        self,
        recovery_manager: RecoveryManager,
        payload_store: FilesystemPayloadStore,
        db,
        checkpoint_manager,
    ) -> None:
        """Returns empty list when all rows are processed."""
        # Create run where checkpoint is at last row
        from datetime import UTC, datetime
        from elspeth.core.landscape.schema import (
            nodes_table, rows_table, runs_table, tokens_table
        )
        import json

        run_id = "test-all-processed"
        now = datetime.now(UTC)

        with db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id=run_id, started_at=now, config_hash="x",
                settings_json="{}", canonical_version="v1", status="failed",
            ))
            conn.execute(nodes_table.insert().values(
                node_id="node", run_id=run_id, plugin_name="csv",
                node_type="source", plugin_version="1.0",
                determinism="io_read", config_hash="x",
                config_json="{}", registered_at=now,
            ))

            # Single row
            payload_ref = payload_store.store(json.dumps({"id": 0}).encode())
            conn.execute(rows_table.insert().values(
                row_id="row-0", run_id=run_id, source_node_id="node",
                row_index=0, source_data_hash="h", source_data_ref=payload_ref,
                created_at=now,
            ))
            conn.execute(tokens_table.insert().values(
                token_id="tok-0", row_id="row-0", created_at=now,
            ))
            conn.commit()

        # Checkpoint at last row
        checkpoint_manager.create_checkpoint(
            run_id=run_id, token_id="tok-0", node_id="node", sequence_number=0,
        )

        result = recovery_manager.get_unprocessed_row_data(run_id, payload_store)
        assert result == []

    def test_get_unprocessed_row_data_raises_on_missing_payload(
        self,
        recovery_manager: RecoveryManager,
        payload_store: FilesystemPayloadStore,
        db,
        checkpoint_manager,
    ) -> None:
        """Raises ValueError when payload cannot be retrieved."""
        from datetime import UTC, datetime
        from elspeth.core.landscape.schema import (
            nodes_table, rows_table, runs_table, tokens_table
        )

        run_id = "test-missing-payload"
        now = datetime.now(UTC)

        with db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id=run_id, started_at=now, config_hash="x",
                settings_json="{}", canonical_version="v1", status="failed",
            ))
            conn.execute(nodes_table.insert().values(
                node_id="node", run_id=run_id, plugin_name="csv",
                node_type="source", plugin_version="1.0",
                determinism="io_read", config_hash="x",
                config_json="{}", registered_at=now,
            ))

            # Row with invalid payload ref (not stored)
            conn.execute(rows_table.insert().values(
                row_id="row-0", run_id=run_id, source_node_id="node",
                row_index=0, source_data_hash="h",
                source_data_ref="nonexistent_hash",  # Not in payload store
                created_at=now,
            ))
            conn.execute(tokens_table.insert().values(
                token_id="tok-0", row_id="row-0", created_at=now,
            ))

            # Second row (will be unprocessed)
            conn.execute(rows_table.insert().values(
                row_id="row-1", run_id=run_id, source_node_id="node",
                row_index=1, source_data_hash="h",
                source_data_ref="also_nonexistent",
                created_at=now,
            ))
            conn.execute(tokens_table.insert().values(
                token_id="tok-1", row_id="row-1", created_at=now,
            ))
            conn.commit()

        # Checkpoint at row 0
        checkpoint_manager.create_checkpoint(
            run_id=run_id, token_id="tok-0", node_id="node", sequence_number=0,
        )

        with pytest.raises(ValueError, match="payload has been purged"):
            recovery_manager.get_unprocessed_row_data(run_id, payload_store)
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/core/checkpoint/test_recovery_row_data.py -v`
Expected: FAIL with "AttributeError: 'RecoveryManager' object has no attribute 'get_unprocessed_row_data'"

**Step 4: Implement get_unprocessed_row_data**

```python
# src/elspeth/core/checkpoint/recovery.py
# Add this method to RecoveryManager class:

def get_unprocessed_row_data(
    self,
    run_id: str,
    payload_store: Any,
) -> list[tuple[int, dict[str, Any]]]:
    """Get row data for unprocessed rows.

    Retrieves actual row data (not just IDs) for rows that need
    processing during resume. Returns tuples of (row_index, row_data)
    ordered by row_index for deterministic processing.

    Args:
        run_id: The run to get unprocessed rows for
        payload_store: PayloadStore for retrieving row data

    Returns:
        List of (row_index, row_data) tuples, ordered by row_index.
        Empty list if run cannot be resumed or all rows were processed.

    Raises:
        ValueError: If row data cannot be retrieved (payload purged or missing)
    """
    import json

    row_ids = self.get_unprocessed_rows(run_id)
    if not row_ids:
        return []

    result: list[tuple[int, dict[str, Any]]] = []

    with self._db.engine.connect() as conn:
        for row_id in row_ids:
            # Get row metadata
            row_result = conn.execute(
                select(rows_table.c.row_index, rows_table.c.source_data_ref)
                .where(rows_table.c.row_id == row_id)
            ).fetchone()

            if row_result is None:
                raise ValueError(f"Row {row_id} not found in database")

            row_index = row_result.row_index
            source_data_ref = row_result.source_data_ref

            if source_data_ref is None:
                raise ValueError(
                    f"Row {row_id} has no source_data_ref - cannot resume without payload"
                )

            # Retrieve from payload store
            try:
                payload_bytes = payload_store.retrieve(source_data_ref)
                row_data = json.loads(payload_bytes.decode("utf-8"))
            except KeyError:
                raise ValueError(
                    f"Row {row_id} payload has been purged - cannot resume"
                ) from None

            result.append((row_index, row_data))

    return result
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/core/checkpoint/test_recovery_row_data.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/core/checkpoint/recovery.py tests/core/checkpoint/conftest.py tests/core/checkpoint/test_recovery_row_data.py
git commit -m "$(cat <<'EOF'
feat(checkpoint): add get_unprocessed_row_data to RecoveryManager

Retrieves actual row data (not just IDs) for rows that need
processing during resume. Returns (row_index, row_data) tuples
ordered by row_index for deterministic replay.

Raises ValueError if payload has been purged.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add Append Mode to CSVSink

**Files:**
- Modify: `src/elspeth/plugins/sinks/csv_sink.py`
- Test: `tests/plugins/sinks/test_csv_sink_append.py`

**Step 1: Write the failing test**

```python
# tests/plugins/sinks/test_csv_sink_append.py
"""Tests for CSVSink append mode."""

from pathlib import Path
from typing import Any

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.sinks.csv_sink import CSVSink


@pytest.fixture
def ctx() -> PluginContext:
    """Create test context."""
    return PluginContext(run_id="test", config={}, landscape=None)


class TestCSVSinkAppendMode:
    """Tests for CSVSink append mode."""

    def test_append_mode_adds_to_existing_file(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Append mode should add rows without rewriting header."""
        output_path = tmp_path / "output.csv"

        # First write in normal mode
        sink1 = CSVSink({
            "path": str(output_path),
            "schema": {"fields": "dynamic"},
        })
        sink1.write([{"id": 1, "value": "a"}], ctx)
        sink1.flush()
        sink1.close()

        # Verify first write
        content1 = output_path.read_text()
        lines1 = content1.strip().split("\n")
        assert len(lines1) == 2  # header + 1 row
        assert "id,value" in lines1[0]

        # Second write in append mode
        sink2 = CSVSink({
            "path": str(output_path),
            "schema": {"fields": "dynamic"},
            "mode": "append",
        })
        sink2.write([{"id": 2, "value": "b"}], ctx)
        sink2.flush()
        sink2.close()

        # Should have both rows, one header
        content2 = output_path.read_text()
        lines2 = content2.strip().split("\n")
        assert len(lines2) == 3  # header + 2 rows
        assert lines2[0] == "id,value"
        assert "1,a" in lines2[1]
        assert "2,b" in lines2[2]

    def test_append_mode_reads_headers_from_existing_file(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Append mode should use headers from existing file."""
        output_path = tmp_path / "output.csv"

        # First write with specific column order
        sink1 = CSVSink({
            "path": str(output_path),
            "schema": {"fields": "dynamic"},
        })
        sink1.write([{"name": "Alice", "age": 30}], ctx)
        sink1.flush()
        sink1.close()

        # Append with same fields (order might differ in dict)
        sink2 = CSVSink({
            "path": str(output_path),
            "schema": {"fields": "dynamic"},
            "mode": "append",
        })
        sink2.write([{"age": 25, "name": "Bob"}], ctx)  # Different order
        sink2.flush()
        sink2.close()

        # Columns should match original order
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "name,age"  # Original order preserved

    def test_append_mode_creates_file_if_not_exists(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Append mode should create file with header if it doesn't exist."""
        output_path = tmp_path / "new_file.csv"
        assert not output_path.exists()

        sink = CSVSink({
            "path": str(output_path),
            "schema": {"fields": "dynamic"},
            "mode": "append",
        })
        sink.write([{"id": 1}], ctx)
        sink.flush()
        sink.close()

        # Should create file with header
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + row
        assert "id" in lines[0]

    def test_default_mode_is_write(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Default mode should be 'write' (truncate)."""
        output_path = tmp_path / "output.csv"

        # First write
        sink1 = CSVSink({
            "path": str(output_path),
            "schema": {"fields": "dynamic"},
        })
        sink1.write([{"id": 1}], ctx)
        sink1.flush()
        sink1.close()

        # Second write without mode (should truncate)
        sink2 = CSVSink({
            "path": str(output_path),
            "schema": {"fields": "dynamic"},
        })
        sink2.write([{"id": 2}], ctx)
        sink2.flush()
        sink2.close()

        # Should only have second row
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert "2" in lines[1]
        assert "1" not in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/sinks/test_csv_sink_append.py -v`
Expected: FAIL (no mode parameter support)

**Step 3: Implement append mode in CSVSink**

```python
# src/elspeth/plugins/sinks/csv_sink.py

# Update CSVSinkConfig to add mode:
class CSVSinkConfig(PathConfig):
    """Configuration for CSV sink plugin."""

    delimiter: str = ","
    encoding: str = "utf-8"
    validate_input: bool = False
    mode: str = "write"  # "write" (truncate) or "append"


# Update CSVSink.__init__ to capture mode:
def __init__(self, config: dict[str, Any]) -> None:
    super().__init__(config)
    cfg = CSVSinkConfig.from_dict(config)

    self._path = cfg.resolved_path()
    self._delimiter = cfg.delimiter
    self._encoding = cfg.encoding
    self._validate_input = cfg.validate_input
    self._mode = cfg.mode  # NEW: capture mode

    # ... rest unchanged ...


# Update the write() method - replace lines 114-126 with:
def write(
    self, rows: list[dict[str, Any]], ctx: PluginContext
) -> ArtifactDescriptor:
    """Write a batch of rows to the CSV file."""
    if not rows:
        return ArtifactDescriptor.for_file(
            path=str(self._path),
            content_hash=hashlib.sha256(b"").hexdigest(),
            size_bytes=0,
        )

    # Optional input validation
    if self._validate_input and not self._schema_config.is_dynamic:
        for row in rows:
            self._schema_class.model_validate(row)

    # Lazy initialization - open file on first write
    if self._file is None:
        self._open_file(rows)

    # Write all rows
    writer = self._writer
    if writer is None:
        raise RuntimeError("CSVSink writer not initialized - this is a bug")
    for row in rows:
        writer.writerow(row)

    # Flush and compute hash
    self._file.flush()
    content_hash = self._compute_file_hash()
    size_bytes = self._path.stat().st_size

    return ArtifactDescriptor.for_file(
        path=str(self._path),
        content_hash=content_hash,
        size_bytes=size_bytes,
    )


def _open_file(self, rows: list[dict[str, Any]]) -> None:
    """Open file for writing, handling append mode.

    In append mode:
    - If file exists: read headers from it, open in append mode
    - If file doesn't exist: create with headers (like write mode)

    In write mode:
    - Always truncate and write headers

    Args:
        rows: First batch of rows (used for fieldnames if creating new file)
    """
    if self._mode == "append" and self._path.exists():
        # Read existing headers from file
        with open(self._path, "r", encoding=self._encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=self._delimiter)
            existing_fieldnames = reader.fieldnames
            if existing_fieldnames is None:
                # Empty file - treat like write mode
                self._fieldnames = list(rows[0].keys())
                self._file = open(
                    self._path, "w", encoding=self._encoding, newline=""
                )
                self._writer = csv.DictWriter(
                    self._file,
                    fieldnames=self._fieldnames,
                    delimiter=self._delimiter,
                )
                self._writer.writeheader()
            else:
                # Use existing headers, append mode
                self._fieldnames = list(existing_fieldnames)
                self._file = open(
                    self._path, "a", encoding=self._encoding, newline=""
                )
                self._writer = csv.DictWriter(
                    self._file,
                    fieldnames=self._fieldnames,
                    delimiter=self._delimiter,
                )
                # No header write - already exists
    else:
        # Write mode or append to non-existent file
        self._fieldnames = list(rows[0].keys())
        self._file = open(
            self._path, "w", encoding=self._encoding, newline=""
        )
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fieldnames,
            delimiter=self._delimiter,
        )
        self._writer.writeheader()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/sinks/test_csv_sink_append.py -v`
Expected: PASS

**Step 5: Run existing CSV tests to ensure no regression**

Run: `pytest tests/plugins/sinks/test_csv_sink.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/plugins/sinks/csv_sink.py tests/plugins/sinks/test_csv_sink_append.py
git commit -m "$(cat <<'EOF'
feat(sinks): add append mode to CSVSink for resume support

CSVSink now supports mode="append" config option:
- Reads existing headers from file (preserves column order)
- Appends rows without rewriting header
- Creates file with header if it doesn't exist

Default mode remains "write" (truncate).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement Resume Row Processing in Orchestrator

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`
- Test: `tests/engine/test_orchestrator_resume.py`

**Step 1: Write the failing test**

```python
# tests/engine/test_orchestrator_resume.py
"""Tests for Orchestrator.resume() row-level processing."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts import RunStatus
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.config import CheckpointSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    edges_table,
    nodes_table,
    rows_table,
    runs_table,
    tokens_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sources.null_source import NullSource
from elspeth.plugins.transforms.passthrough import PassThrough


class TestOrchestratorResume:
    """Tests for resume functionality."""

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up test environment."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        checkpoint_settings = CheckpointSettings(frequency="every_row")

        return {
            "db": db,
            "payload_store": payload_store,
            "checkpoint_manager": checkpoint_mgr,
            "recovery_manager": recovery_mgr,
            "checkpoint_settings": checkpoint_settings,
            "tmp_path": tmp_path,
        }

    def test_resume_processes_unprocessed_rows(
        self, test_env: dict[str, Any]
    ) -> None:
        """Resume should process rows after checkpoint."""
        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        payload_store = test_env["payload_store"]
        checkpoint_settings = test_env["checkpoint_settings"]
        tmp_path = test_env["tmp_path"]

        # Set up failed run with checkpoint at row 2 (of 5 total)
        run_id, output_path = self._setup_failed_run(
            db, checkpoint_mgr, payload_store, tmp_path
        )

        # Verify unprocessed rows
        unprocessed = recovery_mgr.get_unprocessed_rows(run_id)
        assert len(unprocessed) == 2, "Should have 2 unprocessed rows"

        # Get resume point
        resume_point = recovery_mgr.get_resume_point(run_id)
        assert resume_point is not None

        # Create orchestrator
        orchestrator = Orchestrator(
            db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_settings=checkpoint_settings,
        )

        # Create pipeline config with NullSource (data from payload store)
        config = PipelineConfig(
            source=NullSource({}),
            transforms=[PassThrough({})],
            sinks={"default": CSVSink({
                "path": str(output_path),
                "schema": {"fields": "dynamic"},
                "mode": "append",  # Append to existing output
            })},
        )

        # Create graph matching original run's structure
        graph = self._create_graph_for_resume(run_id, db)

        # Resume the run
        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

        # Should have processed the 2 unprocessed rows
        assert result.rows_processed == 2
        assert result.rows_succeeded == 2
        assert result.status == RunStatus.COMPLETED

    def _setup_failed_run(
        self,
        db: LandscapeDB,
        checkpoint_mgr: CheckpointManager,
        payload_store: FilesystemPayloadStore,
        tmp_path: Path,
    ) -> tuple[str, Path]:
        """Create a failed run with 5 rows, checkpoint at row 2.

        Returns (run_id, output_path) where output already has rows 0-2.
        """
        run_id = "test-resume-run"
        output_path = tmp_path / "output.csv"
        now = datetime.now(UTC)

        with db.engine.connect() as conn:
            # Create run
            conn.execute(runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                config_hash="test",
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status="failed",
            ))

            # Create nodes
            conn.execute(nodes_table.insert().values(
                node_id="source-node", run_id=run_id, plugin_name="csv",
                node_type="source", plugin_version="1.0",
                determinism="io_read", config_hash="x",
                config_json="{}", registered_at=now,
            ))
            conn.execute(nodes_table.insert().values(
                node_id="transform-node", run_id=run_id, plugin_name="passthrough",
                node_type="transform", plugin_version="1.0",
                determinism="deterministic", config_hash="x",
                config_json="{}", registered_at=now,
            ))
            conn.execute(nodes_table.insert().values(
                node_id="sink-node", run_id=run_id, plugin_name="csv",
                node_type="sink", plugin_version="1.0",
                determinism="io_write", config_hash="x",
                config_json="{}", registered_at=now,
            ))

            # Create edges
            conn.execute(edges_table.insert().values(
                edge_id="edge-1", run_id=run_id,
                from_node_id="source-node", to_node_id="transform-node",
                label="continue", registered_at=now,
            ))
            conn.execute(edges_table.insert().values(
                edge_id="edge-2", run_id=run_id,
                from_node_id="transform-node", to_node_id="sink-node",
                label="continue", registered_at=now,
            ))

            # Create 5 rows with payloads
            for i in range(5):
                row_id = f"row-{i:03d}"
                row_data = {"id": i, "value": f"data-{i}"}
                payload_ref = payload_store.store(
                    json.dumps(row_data).encode("utf-8")
                )

                conn.execute(rows_table.insert().values(
                    row_id=row_id, run_id=run_id, source_node_id="source-node",
                    row_index=i, source_data_hash=f"hash{i}",
                    source_data_ref=payload_ref, created_at=now,
                ))
                conn.execute(tokens_table.insert().values(
                    token_id=f"tok-{i:03d}", row_id=row_id, created_at=now,
                ))

            conn.commit()

        # Write "already processed" rows 0-2 to output
        with open(output_path, "w") as f:
            f.write("id,value\n")
            for i in range(3):
                f.write(f"{i},data-{i}\n")

        # Create checkpoint at row 2
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="tok-002",
            node_id="transform-node",
            sequence_number=2,
        )

        return run_id, output_path

    def _create_graph_for_resume(
        self, run_id: str, db: LandscapeDB
    ) -> ExecutionGraph:
        """Create execution graph matching the original run's structure.

        Note: ExecutionGraph uses add_node() with keyword args, not convenience methods.
        """
        graph = ExecutionGraph()
        graph.add_node("source-node", node_type="source", plugin_name="csv", config={})
        graph.add_node("transform-node", node_type="transform", plugin_name="passthrough", config={})
        graph.add_node("sink-node", node_type="sink", plugin_name="csv", config={})
        graph.add_edge("source-node", "transform-node", "continue")
        graph.add_edge("transform-node", "sink-node", "continue")
        return graph
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_orchestrator_resume.py::TestOrchestratorResume::test_resume_processes_unprocessed_rows -v`
Expected: FAIL (resume returns rows_processed=0)

**Step 3: Implement resume row processing**

Replace the TODO section in `orchestrator.py:931-935` and add helper method:

```python
# src/elspeth/engine/orchestrator.py

def resume(
    self,
    resume_point: "ResumePoint",
    config: PipelineConfig,
    graph: ExecutionGraph,
    *,
    payload_store: Any = None,
    settings: "ElspethSettings | None" = None,
) -> RunResult:
    """Resume a failed run from checkpoint.

    Continues processing from where the run left off:
    1. Handles incomplete batches
    2. Restores aggregation state
    3. Retrieves unprocessed row data from payload store
    4. Processes unprocessed rows through RowProcessor
    5. Writes results to sinks (append mode)

    Args:
        resume_point: Checkpoint data from RecoveryManager
        config: Pipeline configuration (source is NullSource - data from payloads)
        graph: Execution graph matching original run structure
        payload_store: PayloadStore for retrieving unprocessed row data (required)
        settings: Optional settings for retry config

    Returns:
        RunResult with counts from resumed processing

    Raises:
        ValueError: If payload_store not provided
    """
    from elspeth.core.checkpoint import RecoveryManager

    if payload_store is None:
        raise ValueError("payload_store required for resume - cannot retrieve row data")

    run_id = resume_point.checkpoint.run_id

    # Create fresh recorder
    recorder = LandscapeRecorder(self._db)

    # 1. Handle incomplete batches
    self._handle_incomplete_batches(recorder, run_id)

    # 2. Update run status to running
    self._update_run_status(recorder, run_id, RunStatus.RUNNING)

    # 3. Build restored aggregation state map
    restored_state: dict[str, dict[str, Any]] = {}
    if resume_point.aggregation_state is not None:
        restored_state[resume_point.node_id] = resume_point.aggregation_state

    # 4. Get unprocessed row data from payload store
    recovery_mgr = RecoveryManager(self._db, self._checkpoint_manager)
    unprocessed_rows = recovery_mgr.get_unprocessed_row_data(run_id, payload_store)

    if not unprocessed_rows:
        # All rows were processed - just complete the run
        recorder.complete_run(run_id, RunStatus.COMPLETED)
        self._delete_checkpoints(run_id)
        return RunResult(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            rows_processed=0,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed=0,
        )

    # 5. Process unprocessed rows
    result = self._process_resumed_rows(
        recorder=recorder,
        run_id=run_id,
        config=config,
        graph=graph,
        unprocessed_rows=unprocessed_rows,
        restored_aggregation_state=restored_state,
        settings=settings,
    )

    # 6. Complete run and cleanup
    final_status = RunStatus.COMPLETED if result.rows_failed == 0 else RunStatus.FAILED
    recorder.complete_run(run_id, final_status)
    self._delete_checkpoints(run_id)

    return RunResult(
        run_id=run_id,
        status=final_status,
        rows_processed=result.rows_processed,
        rows_succeeded=result.rows_succeeded,
        rows_failed=result.rows_failed,
        rows_routed=result.rows_routed,
        rows_quarantined=result.rows_quarantined,
        rows_forked=result.rows_forked,
    )


def _process_resumed_rows(
    self,
    recorder: LandscapeRecorder,
    run_id: str,
    config: PipelineConfig,
    graph: ExecutionGraph,
    unprocessed_rows: list[tuple[int, dict[str, Any]]],
    restored_aggregation_state: dict[str, dict[str, Any]],
    settings: "ElspethSettings | None" = None,
) -> RunResult:
    """Process rows during resume.

    Similar to _execute_run but:
    - Rows come from unprocessed_rows list, not source
    - Aggregation state is restored from checkpoint
    - Nodes/edges already registered (don't re-register)
    """
    from elspeth.contracts import TokenInfo
    from elspeth.engine.executors import SinkExecutor
    from elspeth.engine.retry import RetryConfig, RetryManager

    # Get node IDs from graph
    source_id = graph.get_source()
    if source_id is None:
        raise ValueError("Graph has no source node")

    transform_id_map = graph.get_transform_id_map()
    sink_id_map = graph.get_sink_id_map()
    config_gate_id_map = graph.get_config_gate_id_map()
    output_sink_name = graph.get_output_sink()

    # Build edge_map from graph edges
    edge_map: dict[tuple[str, str], str] = {}
    for edge_info in graph.get_edges():
        edge_map[(edge_info.from_node, edge_info.label)] = f"edge-{edge_info.from_node}-{edge_info.label}"

    route_resolution_map = graph.get_route_resolution_map()

    # Assign node_ids to plugins (source won't be used but satisfies protocol)
    self._assign_plugin_node_ids(
        source=config.source,
        transforms=config.transforms,
        sinks=config.sinks,
        source_id=source_id,
        transform_id_map=transform_id_map,
        sink_id_map=sink_id_map,
    )

    # Create context
    ctx = PluginContext(run_id=run_id, config=config.config, landscape=recorder)

    # Call on_start for transforms and sinks (not source)
    for transform in config.transforms:
        transform.on_start(ctx)
    for sink in config.sinks.values():
        sink.on_start(ctx)

    # Create retry manager
    retry_manager: RetryManager | None = None
    if settings is not None:
        retry_manager = RetryManager(RetryConfig.from_settings(settings.retry))

    # Create processor with restored aggregation state
    processor = RowProcessor(
        recorder=recorder,
        span_factory=self._span_factory,
        run_id=run_id,
        source_node_id=source_id,
        edge_map=edge_map,
        route_resolution_map=route_resolution_map,
        config_gates=config.gates,
        config_gate_id_map=config_gate_id_map,
        aggregation_settings=config.aggregation_settings,
        retry_manager=retry_manager,
        restored_aggregation_state=restored_aggregation_state,
    )

    # Process unprocessed rows
    rows_processed = 0
    rows_succeeded = 0
    rows_failed = 0
    rows_routed = 0
    rows_quarantined = 0
    rows_forked = 0
    pending_tokens: dict[str, list[TokenInfo]] = {name: [] for name in config.sinks}

    try:
        for row_index, row_data in unprocessed_rows:
            rows_processed += 1

            results = processor.process_row(
                row_index=row_index,
                row_data=row_data,
                transforms=config.transforms,
                ctx=ctx,
            )

            # Handle results
            for result in results:
                last_node_id = self._get_last_node_id(
                    config, config_gate_id_map, source_id
                )

                if result.outcome == RowOutcome.COMPLETED:
                    rows_succeeded += 1
                    sink_name = output_sink_name
                    if (
                        result.token.branch_name is not None
                        and result.token.branch_name in config.sinks
                    ):
                        sink_name = result.token.branch_name
                    pending_tokens[sink_name].append(result.token)
                    self._maybe_checkpoint(run_id, result.token.token_id, last_node_id)
                elif result.outcome == RowOutcome.ROUTED:
                    rows_routed += 1
                    assert result.sink_name is not None
                    pending_tokens[result.sink_name].append(result.token)
                    self._maybe_checkpoint(run_id, result.token.token_id, last_node_id)
                elif result.outcome == RowOutcome.FAILED:
                    rows_failed += 1
                elif result.outcome == RowOutcome.QUARANTINED:
                    rows_quarantined += 1
                elif result.outcome == RowOutcome.FORKED:
                    rows_forked += 1

        # Write to sinks
        sink_executor = SinkExecutor(recorder, self._span_factory, run_id)
        step = len(config.transforms) + len(config.gates) + 1

        for sink_name, tokens in pending_tokens.items():
            if tokens and sink_name in config.sinks:
                sink = config.sinks[sink_name]
                sink_executor.write(
                    sink=sink,
                    tokens=tokens,
                    ctx=ctx,
                    step_in_pipeline=step,
                )

    finally:
        # Cleanup
        for transform in config.transforms:
            with suppress(Exception):
                transform.on_complete(ctx)
        for sink in config.sinks.values():
            with suppress(Exception):
                sink.on_complete(ctx)
            sink.close()

    return RunResult(
        run_id=run_id,
        status=RunStatus.RUNNING,
        rows_processed=rows_processed,
        rows_succeeded=rows_succeeded,
        rows_failed=rows_failed,
        rows_routed=rows_routed,
        rows_quarantined=rows_quarantined,
        rows_forked=rows_forked,
    )


def _get_last_node_id(
    self,
    config: PipelineConfig,
    config_gate_id_map: dict[str, str],
    source_id: str,
) -> str:
    """Get the last node ID for checkpoint purposes."""
    if config.gates:
        last_gate_name = config.gates[-1].name
        return config_gate_id_map[last_gate_name]
    elif config.transforms:
        transform_node_id = config.transforms[-1].node_id
        assert transform_node_id is not None
        return transform_node_id
    return source_id
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/engine/test_orchestrator_resume.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator_resume.py
git commit -m "$(cat <<'EOF'
feat(engine): implement row-level resume in Orchestrator

Completes the TODO at orchestrator.py:931-935. Resume now:
1. Retrieves unprocessed row data from payload store
2. Processes rows through RowProcessor with restored aggregation state
3. Writes results to sinks (append mode expected)
4. Updates RunResult with actual processing counts

Uses NullSource to satisfy PipelineConfig.source typing.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update CLI Resume Command

**Files:**
- Modify: `src/elspeth/cli.py`
- Test: Manual verification

**Step 1: Add --execute flag to existing resume command**

```python
# src/elspeth/cli.py
# Update the resume command (lines 600-697):

@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run ID to resume"),
    database: str | None = typer.Option(
        None,
        "--database",
        "-d",
        help="Path to Landscape database file (SQLite).",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        "-x",
        help="Actually execute the resume (default is dry-run).",
    ),
) -> None:
    """Resume a failed run from its last checkpoint.

    By default, shows what WOULD happen (dry run). Use --execute to
    actually resume processing.

    Examples:

        # Dry run - show resume info
        elspeth resume run-abc123

        # Actually resume processing
        elspeth resume run-abc123 --execute
    """
    from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.payload_store import FilesystemPayloadStore
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

    # Resolve database URL (existing logic)
    db_url: str | None = None
    settings_config = None

    if database:
        db_path = Path(database)
        db_url = f"sqlite:///{db_path.resolve()}"
    else:
        settings_path = Path("settings.yaml")
        if settings_path.exists():
            try:
                settings_config = load_settings(settings_path)
                db_url = settings_config.landscape.url
                typer.echo(f"Using database from settings.yaml: {db_url}")
            except Exception as e:
                typer.echo(f"Error loading settings.yaml: {e}", err=True)
                raise typer.Exit(1) from None
        else:
            typer.echo("Error: No settings.yaml found and --database not provided.", err=True)
            raise typer.Exit(1) from None

    # Initialize database
    try:
        db = LandscapeDB.from_url(db_url)
    except Exception as e:
        typer.echo(f"Error connecting to database: {e}", err=True)
        raise typer.Exit(1) from None

    try:
        checkpoint_manager = CheckpointManager(db)
        recovery_manager = RecoveryManager(db, checkpoint_manager)

        # Check if run can be resumed
        check = recovery_manager.can_resume(run_id)
        if not check.can_resume:
            typer.echo(f"Cannot resume run {run_id}: {check.reason}", err=True)
            raise typer.Exit(1)

        # Get resume point
        resume_point = recovery_manager.get_resume_point(run_id)
        if resume_point is None:
            typer.echo(f"Error: Could not get resume point for run {run_id}", err=True)
            raise typer.Exit(1)

        # Get unprocessed rows count
        unprocessed_rows = recovery_manager.get_unprocessed_rows(run_id)

        # Display resume info
        typer.echo(f"Run {run_id} can be resumed.")
        typer.echo(f"\nResume point:")
        typer.echo(f"  Token ID: {resume_point.token_id}")
        typer.echo(f"  Node ID: {resume_point.node_id}")
        typer.echo(f"  Sequence number: {resume_point.sequence_number}")
        typer.echo(f"  Has aggregation state: {'Yes' if resume_point.aggregation_state else 'No'}")
        typer.echo(f"\nUnprocessed rows: {len(unprocessed_rows)}")

        if not execute:
            typer.echo("\nDry run - use --execute to actually resume processing.")
            return

        # Execute resume
        if settings_config is None:
            typer.echo("Error: --execute requires settings.yaml for pipeline config", err=True)
            raise typer.Exit(1)

        # Get payload store path from settings
        payload_dir = Path(settings_config.landscape.payload_dir)
        if not payload_dir.exists():
            typer.echo(f"Error: Payload directory not found: {payload_dir}", err=True)
            raise typer.Exit(1)

        payload_store = FilesystemPayloadStore(payload_dir)

        typer.echo(f"\nResuming run {run_id}...")

        # Build pipeline config from settings
        # (This requires reconstructing the pipeline - simplified here)
        from elspeth.plugins.sources.null_source import NullSource
        from elspeth.plugins.transforms.passthrough import PassThrough

        # For now, use minimal config - full implementation would rebuild from settings
        config = _build_resume_pipeline_config(settings_config, payload_dir)
        graph = _build_resume_graph_from_db(db, run_id)

        # Create orchestrator and resume
        orchestrator = Orchestrator(
            db,
            checkpoint_manager=checkpoint_manager,
            checkpoint_settings=settings_config.checkpoint if hasattr(settings_config, 'checkpoint') else None,
        )

        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=graph,
            payload_store=payload_store,
            settings=settings_config,
        )

        typer.echo(f"\nResume complete:")
        typer.echo(f"  Rows processed: {result.rows_processed}")
        typer.echo(f"  Rows succeeded: {result.rows_succeeded}")
        typer.echo(f"  Rows failed: {result.rows_failed}")
        typer.echo(f"  Status: {result.status.value}")

    finally:
        db.close()
```

**Step 2: Add helper functions for pipeline reconstruction**

```python
# Add to cli.py (before the resume command):

def _build_resume_pipeline_config(
    settings: Any,
    payload_dir: Path,
) -> "PipelineConfig":
    """Build PipelineConfig for resume from settings.

    This is a simplified version - full implementation would
    reconstruct the exact pipeline from the original run.
    """
    from elspeth.engine.orchestrator import PipelineConfig
    from elspeth.plugins.sources.null_source import NullSource

    # For resume, source is NullSource (data from payloads)
    source = NullSource({})

    # Rebuild transforms and sinks from settings
    transforms = _build_transforms_from_settings(settings)
    sinks = _build_sinks_from_settings(settings, mode="append")

    return PipelineConfig(
        source=source,
        transforms=transforms,
        sinks=sinks,
        config=settings.dict() if hasattr(settings, 'dict') else {},
    )


def _build_resume_graph_from_db(
    db: "LandscapeDB",
    run_id: str,
) -> "ExecutionGraph":
    """Reconstruct ExecutionGraph from registered nodes/edges in database.

    Note: ExecutionGraph uses add_node() with keyword args, not convenience methods.
    """
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.landscape.schema import edges_table, nodes_table
    from sqlalchemy import select

    graph = ExecutionGraph()

    with db.engine.connect() as conn:
        # Get nodes
        nodes = conn.execute(
            select(nodes_table).where(nodes_table.c.run_id == run_id)
        ).fetchall()

        # Get edges
        edges = conn.execute(
            select(edges_table).where(edges_table.c.run_id == run_id)
        ).fetchall()

    # Add nodes to graph using add_node() API
    for node in nodes:
        graph.add_node(
            node.node_id,
            node_type=node.node_type,
            plugin_name=node.plugin_name,
            config={},
        )

    # Add edges
    for edge in edges:
        graph.add_edge(edge.from_node_id, edge.to_node_id, edge.label)

    return graph
```

**Step 3: Manual verification**

```bash
# Dry run
elspeth resume <run_id>

# Execute
elspeth resume <run_id> --execute
```

**Step 4: Commit**

```bash
git add src/elspeth/cli.py
git commit -m "$(cat <<'EOF'
feat(cli): add --execute flag to resume command

Resume command now:
- Default: dry run showing resume point and unprocessed row count
- With --execute: actually resumes processing
- Reconstructs pipeline from database and settings

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Integration Test for Full Resume Cycle

**Files:**
- Modify: `tests/integration/test_checkpoint_recovery.py`

**Step 1: Add full resume integration test**

```python
# Add to tests/integration/test_checkpoint_recovery.py:

def test_full_resume_processes_remaining_rows(self, test_env: dict[str, Any]) -> None:
    """Complete cycle: run -> crash simulation -> resume -> all rows processed."""
    import json
    from elspeth.core.landscape.schema import (
        edges_table, nodes_table, rows_table, runs_table, tokens_table
    )
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
    from elspeth.plugins.sources.null_source import NullSource
    from elspeth.plugins.transforms.passthrough import PassThrough
    from elspeth.plugins.sinks.csv_sink import CSVSink
    from elspeth.core.dag import ExecutionGraph

    db = test_env["db"]
    checkpoint_mgr = test_env["checkpoint_manager"]
    recovery_mgr = test_env["recovery_manager"]
    payload_store = test_env["payload_store"]
    checkpoint_settings = test_env["checkpoint_settings"]

    # 1. Set up failed run with 5 rows, checkpoint at row 2
    run_id = "integration-resume-test"
    output_path = Path(test_env.get("tmp_path", "/tmp")) / "resume_output.csv"
    now = datetime.now(UTC)

    with db.engine.connect() as conn:
        conn.execute(runs_table.insert().values(
            run_id=run_id, started_at=now, config_hash="x",
            settings_json="{}", canonical_version="v1", status="failed",
        ))
        conn.execute(nodes_table.insert().values(
            node_id="src", run_id=run_id, plugin_name="csv",
            node_type="source", plugin_version="1.0",
            determinism="io_read", config_hash="x",
            config_json="{}", registered_at=now,
        ))
        conn.execute(nodes_table.insert().values(
            node_id="xform", run_id=run_id, plugin_name="passthrough",
            node_type="transform", plugin_version="1.0",
            determinism="deterministic", config_hash="x",
            config_json="{}", registered_at=now,
        ))
        conn.execute(nodes_table.insert().values(
            node_id="sink", run_id=run_id, plugin_name="csv",
            node_type="sink", plugin_version="1.0",
            determinism="io_write", config_hash="x",
            config_json="{}", registered_at=now,
        ))
        conn.execute(edges_table.insert().values(
            edge_id="e1", run_id=run_id,
            from_node_id="src", to_node_id="xform",
            label="continue", registered_at=now,
        ))
        conn.execute(edges_table.insert().values(
            edge_id="e2", run_id=run_id,
            from_node_id="xform", to_node_id="sink",
            label="continue", registered_at=now,
        ))

        for i in range(5):
            row_data = {"id": i, "name": f"row-{i}"}
            ref = payload_store.store(json.dumps(row_data).encode())
            conn.execute(rows_table.insert().values(
                row_id=f"r{i}", run_id=run_id, source_node_id="src",
                row_index=i, source_data_hash=f"h{i}",
                source_data_ref=ref, created_at=now,
            ))
            conn.execute(tokens_table.insert().values(
                token_id=f"t{i}", row_id=f"r{i}", created_at=now,
            ))
        conn.commit()

    # Simulate partial output (rows 0-2 already written)
    with open(output_path, "w") as f:
        f.write("id,name\n")
        f.write("0,row-0\n")
        f.write("1,row-1\n")
        f.write("2,row-2\n")

    # Checkpoint at row 2
    checkpoint_mgr.create_checkpoint(
        run_id=run_id, token_id="t2", node_id="xform", sequence_number=2,
    )

    # 2. Verify can resume
    assert recovery_mgr.can_resume(run_id).can_resume
    resume_point = recovery_mgr.get_resume_point(run_id)
    assert resume_point is not None

    # 3. Resume
    orchestrator = Orchestrator(
        db,
        checkpoint_manager=checkpoint_mgr,
        checkpoint_settings=checkpoint_settings,
    )

    config = PipelineConfig(
        source=NullSource({}),
        transforms=[PassThrough({})],
        sinks={"default": CSVSink({
            "path": str(output_path),
            "schema": {"fields": "dynamic"},
            "mode": "append",
        })},
    )

    # Build graph using add_node() API (ExecutionGraph doesn't have convenience methods)
    graph = ExecutionGraph()
    graph.add_node("src", node_type="source", plugin_name="csv", config={})
    graph.add_node("xform", node_type="transform", plugin_name="passthrough", config={})
    graph.add_node("sink", node_type="sink", plugin_name="csv", config={})
    graph.add_edge("src", "xform", "continue")
    graph.add_edge("xform", "sink", "continue")

    result = orchestrator.resume(
        resume_point=resume_point,
        config=config,
        graph=graph,
        payload_store=payload_store,
    )

    # 4. Verify
    assert result.rows_processed == 2
    assert result.rows_succeeded == 2
    assert result.status == RunStatus.COMPLETED

    # Check output file has all 5 rows
    lines = output_path.read_text().strip().split("\n")
    assert len(lines) == 6  # header + 5 rows
    assert "0,row-0" in lines[1]
    assert "4,row-4" in lines[5]
```

**Step 2: Run integration test**

Run: `pytest tests/integration/test_checkpoint_recovery.py::TestCheckpointRecoveryIntegration::test_full_resume_processes_remaining_rows -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_checkpoint_recovery.py
git commit -m "$(cat <<'EOF'
test(integration): add full resume cycle integration test

End-to-end test verifying:
- Failed run with checkpoint creates resumable state
- Resume processes exactly the unprocessed rows
- Output file contains all rows (original + resumed)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Run Full Test Suite

**Step 1: Run all tests**

```bash
.venv/bin/python -m pytest tests/ -v
```
Expected: All tests pass

**Step 2: Run type checking**

```bash
.venv/bin/python -m mypy src/elspeth/
```
Expected: No errors

**Step 3: Run linting**

```bash
.venv/bin/python -m ruff check src/elspeth/
```
Expected: No errors

**Step 4: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: complete row-level resume implementation

All tests pass. Row-level resume is now fully functional:
- NullSource for resume operations
- RecoveryManager.get_unprocessed_row_data() retrieves row data
- CSVSink append mode for resume output
- Orchestrator.resume() processes unprocessed rows
- CLI --execute flag for actual resume

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

This revised plan implements row-level resume in 7 tasks:

| Task | What It Does | Addresses Issue |
|------|--------------|-----------------|
| 1 | Create NullSource for resume | PipelineConfig.source type |
| 2 | Add get_unprocessed_row_data() | PayloadStore.store() signature |
| 3 | Add CSVSink append mode | Header handling, refactoring |
| 4 | Implement Orchestrator.resume() | Core resume logic |
| 5 | Update CLI with --execute | Existing CLI structure |
| 6 | Integration test | Full cycle verification |
| 7 | Full test suite | Quality gate |

Each task follows TDD: write failing test → implement → verify → commit.
